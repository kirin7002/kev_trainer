"""Controller independently validates immutable worker artifacts before promotion."""
import hashlib
import json
from pathlib import Path

from datasets import DataError, exposure_keys, parse_json
from evaluation import comparison, evaluate
from kev_contract import CONTRACT, inspect_checkpoint
from model_artifacts import canonical_sha, file_sha, inspect_bundle
from phase_one import system_one, timestamp
from pipeline_validation import TRAIN_STAGES, convert_benchmark, training_metrics, calibration_rows,validate_image_provenance
from final_evaluation import FINAL_STAGES


def read(path):
    return parse_json(Path(path).read_bytes())


def normalized_report(directory, stage, cases, objects, sha):
    root = Path(directory)/'reports'/('benchmark-'+stage)
    lines = [parse_json(line) for line in (root/'predictions.jsonl').read_bytes().splitlines() if line.strip()]
    return convert_benchmark(lines, read(root/'report.json'), cases, objects, sha)


def validate_training_bundle(directory, manifest, trial, inputs, allow_fixture=False):
    real = manifest.get('execution_backend') == 'cuda'
    if not real and not (allow_fixture and manifest.get('execution_backend') == 'software_contract_only'):
        raise DataError('不接受未验证的 GPU 执行后端。')
    if (manifest.get('kind') != 'kev_model_bundle' or tuple(manifest.get('stages', [])) != TRAIN_STAGES
            or manifest.get('training_verified') is not real or manifest.get('loader_verified') is not real):
        raise DataError('训练阶段或验证声明不完整。')
    root = Path(directory)
    config = read(root/'policy/config.json')
    if config != trial['config'] or canonical_sha(config) != trial['config_sha256']:
        raise DataError('产物配置与不可变 Trial 不一致。')
    if read(root/'policy/source-contract.json') != CONTRACT:
        raise DataError('模型包的源码契约不一致。')
    validation = read(root/'reports/VALIDATING.json')
    runtime = validation.get('runtime', {})
    validate_image_provenance(runtime.get('image_provenance'))
    if manifest.get('image_provenance')!=runtime.get('image_provenance'): raise DataError('模型包镜像来源与 runtime 不一致。')
    if (runtime.get('source_commit') != CONTRACT['commit'] or
            runtime.get('parent_archive_sha256') != CONTRACT['released_checkpoint']['sha256'] or
            validation.get('counts') != {role: len(rows) for role, rows in inputs['partitions'].items()} or
            not validation.get('tokenizer_verified') or validation.get('rejected_records') != 0 or validation.get('truncated_records') != 0):
        raise DataError('运行时、源码、父模型或 tokenizer 准入未验证。')
    if real and (runtime.get('runtime_verified') is not True or runtime.get('execution_backend') != 'cuda'):
        raise DataError('GPU runtime 验证失败。')
    inspect_checkpoint(root/'checkpoint', config['model'])
    loaded = read(root/'reports/LOADING.json')
    if (not loaded.get('loader_verified') or loaded.get('objects_verified') != 50
            or loaded.get('head_sha256') != file_sha(root/'checkpoint/head.pt')
            or loaded.get('weights_sha256') != file_sha(root/'checkpoint/adapter_model.safetensors')):
        raise DataError('模型加载回执与当前字节不一致。')
    expanded = len(inputs['partitions']['train'])*10
    for stage in ('SMOKE', 'TRAINING'):
        receipt = read(root/'reports'/(stage+'.json'))
        loss_file = Path(receipt['loss_file'])
        if loss_file.is_absolute() or loss_file.parts[:1] != ('logs',) or '..' in loss_file.parts:
            raise DataError('Loss 文件路径无效。')
        losses = [parse_json(line) for line in (root/loss_file).read_bytes().splitlines() if line.strip()]
        training_metrics(receipt['metrics'], losses, expanded*(1 if stage == 'SMOKE' else config['parameters']['epochs']), stage == 'SMOKE')
    objects = [obj['id'] for obj in inputs['objects']]
    for target, stage, benchmark_stage in (('baseline', 'BASELINE_CALIBRATING', 'BASELINE_CALIBRATION'),
                                           ('candidate', 'CALIBRATING', 'CANDIDATE_CALIBRATION')):
        policy = read(root/'reports'/(stage+'.json'))
        if policy != read(root/'policy'/(target+'-calibration.json')):
            raise DataError('实际导出策略与已验证校准回执不一致。')
        for role in ('train', 'calibration', 'validation'):
            raw = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in system_one(inputs['partitions'][role], {'objects': inputs['objects']}, role)).encode()
            key = 'data_sha256' if role == 'calibration' else role+'_sha256'
            if policy.get(key) != hashlib.sha256(raw).hexdigest():
                raise DataError('校准来源登记与冻结分区不符。')
        if (policy.get('fit_partition') != 'calibration' or not policy.get('groups_disjoint')
                or policy.get('method') != 'pinned_kev_metrics_registered_custom_calibration_v1'
                or policy.get('threshold_strategy') != 'fixed_0_5_no_validation_or_test_tuning'
                or type(policy.get('temperature')) not in (int, float) or not .25 <= policy['temperature'] <= 4
                or policy.get('thresholds') != {obj: .5 for obj in objects}
                or policy.get('rows_sha256') != file_sha(root/'reports'/('benchmark-'+benchmark_stage)/'rows.json')):
            raise DataError('校准行来源/阈值或文件哈希不匹配。')
        calibration_rows(read(root/'reports'/('benchmark-'+benchmark_stage)/'rows.json'),
                         inputs['partitions']['calibration'], objects)
        normalized_report(root, benchmark_stage, inputs['partitions']['calibration'], objects, inputs['partition_sha256']['calibration'])
    predictions = normalized_report(root, 'EVALUATING', inputs['partitions']['validation'], objects, inputs['partition_sha256']['validation'])
    baseline = normalized_report(root, 'BASELINE_SYSTEM', inputs['partitions']['validation'], objects, inputs['partition_sha256']['validation'])
    raw_baseline = normalized_report(root, 'BASELINE_RAW', inputs['partitions']['validation'], objects, inputs['partition_sha256']['validation'])
    raw_candidate = normalized_report(root, 'CANDIDATE_RAW', inputs['partitions']['validation'], objects, inputs['partition_sha256']['validation'])
    return {'real': real, 'runtime':runtime,'predictions': predictions, 'baseline': baseline,
            'model_effect': comparison(inputs['partitions']['validation'], raw_baseline, raw_candidate),
            'system_effect': comparison(inputs['partitions']['validation'], baseline, predictions)}


def commit_training(store, phase, job, artifact):
    trial_dir = store.path('trials', job['trial_id'])
    trial = store.read(trial_dir/'meta.json')
    verified = artifact['verified']
    source = 'authenticated_gpu_worker' if verified['real'] else 'software_contract_fixture'
    validation = {'source': source, 'training_verified': verified['real'],
        'dataset_sha256': verified['predictions']['dataset_sha256'], 'trial_id': trial['id'],
        'metrics': evaluate(phase.worker_inputs(trial['id'])['partitions']['validation'], verified['predictions']),
        'predictions_sha256': canonical_sha(verified['predictions']), 'model_artifact_sha256': artifact['sha256']}
    attempt_dir = trial_dir/'attempts'/job['attempt_id']
    store.write(attempt_dir/'validation-predictions.json', verified['predictions'])
    store.write(attempt_dir/'validation-report.json', validation)
    store.write(attempt_dir/'worker-comparison.json', {'source': source, 'training_verified': verified['real'],
        'model_effect': verified['model_effect'], 'system_effect': verified['system_effect']})
    run_dir = store.path('runs', trial['experiment_id'])
    run = store.read(run_dir/'meta.json')
    if run.get('selected_trial_id') == trial['id']:
        run.update(selected_trial_id=None, selected_validation_sha256=None)
        store.write(run_dir/'meta.json', run)
    # Promotion pointer is the last write. A failed/partial attempt never
    # replaces the previously accepted model pointer.
    trial.update(status='COMPLETED', training_verified=verified['real'], blockers=[],
        active_model={'job_id': job['id'], 'attempt_id': job['attempt_id'], 'sha256': artifact['sha256'],
                      'source': source, 'training_verified': verified['real'], 'loader_verified': verified['real'],
                      'created_at': timestamp(), 'validation_sha256': validation['predictions_sha256']})
    store.write(trial_dir/'meta.json', trial)
    return {'source': source, 'training_verified': verified['real'], 'artifact_sha256': artifact['sha256'],
            'loader_verified': verified['real'], 'stage_count': len(TRAIN_STAGES)}


def validate_evaluation_bundle(directory, manifest, trial, inputs, allow_fixture=False):
    root = Path(directory)
    real = manifest.get('execution_backend') == 'cuda'
    if not real and not (allow_fixture and manifest.get('execution_backend') == 'software_contract_only'):
        raise DataError('最终评估后端无效。')
    if (manifest.get('kind') != 'kev_evaluation_bundle' or tuple(manifest.get('stages', [])) != FINAL_STAGES
            or manifest.get('model_artifact_sha256') != inputs['model_artifact_sha256']
            or manifest.get('selected_validation_sha256') != inputs['selected_validation_sha256']
            or manifest.get('training_verified') is not real):
        raise DataError('最终评估未绑定冻结候选和验证选择。')
    validation = read(root/'reports/FINAL_VALIDATING.json')
    runtime = validation.get('runtime', {})
    validate_image_provenance(runtime.get('image_provenance'))
    if manifest.get('image_provenance')!=runtime.get('image_provenance'): raise DataError('最终评测镜像来源与 runtime 不一致。')
    if (validation.get('test_records') != len(inputs['partitions']['test']) or not validation.get('tokenizer_verified')
            or validation.get('rejected_records') != 0 or validation.get('truncated_records') != 0
            or runtime.get('source_commit') != CONTRACT['commit']
            or runtime.get('parent_archive_sha256') != CONTRACT['released_checkpoint']['sha256']):
        raise DataError('最终评估运行与数据准入不完整。')
    if real and (not runtime.get('runtime_verified') or runtime.get('execution_backend') != 'cuda'):
        raise DataError('最终评估真实 runtime 未验证。')
    restore = read(root/'reports/FINAL_RESTORING.json')
    if not restore.get('restored_without_fitting'): raise DataError('禁止在 test 上拟合校准。')
    for role in ('baseline', 'candidate'):
        path = root/'policy'/(role+'-calibration.json')
        if (restore.get('policy_sha256', {}).get(role) != file_sha(path)
                or file_sha(path) != inputs['frozen_files']['policy/'+role+'-calibration.json']['sha256']):
            raise DataError('最终评估校准策略哈希不符。')
    objects = [obj['id'] for obj in inputs['objects']]
    cases, data_sha = inputs['partitions']['test'], inputs['partition_sha256']['test']
    predictions = {stage: normalized_report(root, stage, cases, objects, data_sha)
                   for stage in ('FINAL_BASELINE_RAW', 'FINAL_CANDIDATE_RAW', 'FINAL_BASELINE_SYSTEM', 'FINAL_CANDIDATE_SYSTEM')}
    loaded = read(root/'reports/FINAL_LOADING.json')
    if (not loaded.get('loader_verified') or loaded.get('objects_verified') != 50
            or loaded.get('head_sha256') != inputs['frozen_files']['checkpoint/head.pt']['sha256']
            or loaded.get('weights_sha256') != inputs['frozen_files']['checkpoint/adapter_model.safetensors']['sha256']):
        raise DataError('最终评估加载器回执不完整。')
    return {'real': real, 'runtime':runtime,'baseline': predictions['FINAL_BASELINE_SYSTEM'], 'candidate': predictions['FINAL_CANDIDATE_SYSTEM'],
            'model_effect': comparison(cases, predictions['FINAL_BASELINE_RAW'], predictions['FINAL_CANDIDATE_RAW']),
            'system_effect': comparison(cases, predictions['FINAL_BASELINE_SYSTEM'], predictions['FINAL_CANDIDATE_SYSTEM'])}


def commit_evaluation(store, phase, job, artifact):
    run_dir = store.path('runs', job['experiment_id'])
    run = store.read(run_dir/'meta.json')
    if run.get('final_evaluation_job_id') != job['id'] or run.get('selected_trial_id') != job['trial_id']:
        raise DataError('最终评估归属或冻结选择已改变。')
    verified = artifact['verified']
    for kind in ('baseline', 'candidate'):
        store.write(run_dir/(kind+'-predictions.json'), verified[kind])
    source = 'authenticated_final_evaluator' if verified['real'] else 'software_contract_fixture'
    report = {**verified['system_effect'], 'source': source, 'training_verified': verified['real'],
              'model_effect': verified['model_effect'], 'system_effect': verified['system_effect'],
              'selected_trial_id': job['trial_id'], 'selected_validation_sha256': job['selected_validation_sha256'],
              'model_artifact_sha256': job['model_artifact_sha256'], 'test_fitted': False}
    store.write(run_dir/'report.json', report)
    run.update(status='completed', training_verified=verified['real'], report_revision=job['attempt_id'],
        evaluation_receipt={'job_id': job['id'], 'artifact_sha256': artifact['sha256'], 'source': source,
            'training_verified': verified['real'], 'selected_trial_id': job['trial_id'],
            'selected_validation_sha256': job['selected_validation_sha256'],
            'baseline_sha256': canonical_sha(verified['baseline']), 'candidate_sha256': canonical_sha(verified['candidate'])})
    store.write(run_dir/'meta.json', run)
    return {'source': source, 'training_verified': verified['real'], 'artifact_sha256': artifact['sha256'],
            'stage_count': len(FINAL_STAGES)}
