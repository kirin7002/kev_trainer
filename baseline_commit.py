"""Controller checks and publishes evaluation-only baseline receipts."""
import hashlib
import json
from pathlib import Path

from datasets import DataError
from evaluation import evaluate
from baseline_execution import BASELINE_STAGES
from kev_contract import CONTRACT
from model_artifacts import canonical_sha,file_sha
from phase_one import system_one,timestamp
from pipeline_commit import read,normalized_report
from pipeline_validation import calibration_rows,validate_image_provenance


def validate_baseline_bundle(directory,manifest,job,inputs,allow_fixture=False):
    root=Path(directory);real=manifest.get('execution_backend')=='cuda'
    if not real and not (allow_fixture and manifest.get('execution_backend')=='software_contract_only'):
        raise DataError('基线后端未验证。')
    if (manifest.get('kind')!='kev_baseline_bundle' or tuple(manifest.get('stages',[]))!=BASELINE_STAGES
            or manifest.get('training_verified') is not False or manifest.get('baseline_verified') is not real
            or manifest.get('optimizer_steps')!=0 or manifest.get('resume_supported') is not False
            or manifest.get('experiment_id')!=job['experiment_id']):
        raise DataError('baseline-only 阶段/零训练声明不匹配。')
    if read(root/'policy/config.json')!=inputs['config'] or read(root/'policy/source-contract.json')!=CONTRACT:
        raise DataError('基线配置/源码契约不一致。')
    registry=read(root/'policy/data-registry.json')
    if registry!={'partition_sha256':inputs['partition_sha256'],'excluded_partitions':['train','test','regression']}:
        raise DataError('基线分区登记不匹配。')
    receipt=read(root/'reports/BASELINE_VALIDATING.json');runtime=receipt.get('runtime',{})
    validate_image_provenance(runtime.get('image_provenance'))
    if (not receipt.get('tokenizer_verified') or receipt.get('counts')!={r:len(v) for r,v in inputs['partitions'].items()}
            or receipt.get('rejected_records')!=0 or receipt.get('truncated_records')!=0 or receipt.get('optimizer_steps')!=0
            or runtime.get('source_commit')!=CONTRACT['commit']
            or runtime.get('parent_archive_sha256')!=CONTRACT['released_checkpoint']['sha256']
            or manifest.get('image_provenance')!=runtime.get('image_provenance')):
        raise DataError('基线 runtime/tokenizer/零训练检查失败。')
    if real and (runtime.get('runtime_verified') is not True or runtime.get('execution_backend')!='cuda'):
        raise DataError('基线 CUDA 未验证。')
    loader=read(root/'reports/BASELINE_LOADING.json')
    if (not loader.get('loader_verified') or loader.get('objects_verified')!=50
            or loader.get('head_sha256')!=file_sha(root/'checkpoint/head.pt')
            or loader.get('weights_sha256')!=file_sha(root/'checkpoint/adapter_model.safetensors')):
        raise DataError('基线 loader/模型字节不一致。')
    adapter=read(root/'checkpoint/adapter_config.json');model=inputs['config']['model']
    if adapter.get('r')!=model['lora_rank'] or adapter.get('base_model_name_or_path')!=model['base']:
        raise DataError('基线 adapter 基座/rank 不一致。')
    policy=read(root/'policy/baseline-calibration.json')
    objects=[o['id'] for o in inputs['objects']]
    if (policy!=read(root/'reports/BASELINE_CALIBRATING.json') or not policy.get('baseline_only')
            or policy.get('train_sha256') is not None or not policy.get('groups_disjoint')
            or policy.get('fit_partition')!='calibration' or policy.get('method')!='pinned_kev_metrics_registered_custom_calibration_v1'
            or policy.get('threshold_strategy')!='fixed_0_5_no_validation_or_test_tuning'
            or policy.get('thresholds')!={o:.5 for o in objects}
            or type(policy.get('temperature')) not in (int,float) or not .25<=policy['temperature']<=4
            or policy.get('rows_sha256')!=file_sha(root/'reports/benchmark-BASELINE_CALIBRATION/rows.json')):
        raise DataError('基线校准策略不完整或来源不一致。')
    for role,key in (('calibration','data_sha256'),('validation','validation_sha256')):
        raw=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in system_one(inputs['partitions'][role],{'objects':inputs['objects']},role)).encode()
        if policy.get(key)!=hashlib.sha256(raw).hexdigest(): raise DataError('基线校准分区 SHA 不匹配。')
    calibration_rows(read(root/'reports/benchmark-BASELINE_CALIBRATION/rows.json'),inputs['partitions']['calibration'],objects)
    normalized_report(root,'BASELINE_CALIBRATION',inputs['partitions']['calibration'],objects,inputs['partition_sha256']['calibration'])
    cases=inputs['partitions']['validation'];sha=inputs['partition_sha256']['validation']
    raw=normalized_report(root,'BASELINE_RAW',cases,objects,sha)
    served=normalized_report(root,'BASELINE_SYSTEM',cases,objects,sha)
    return {'real':real,'training_verified':False,'raw':raw,'baseline':served,
            'raw_metrics':evaluate(cases,raw),'metrics':evaluate(cases,served),
            'runtime':runtime,'calibration_sha256':canonical_sha(policy)}


def commit_baseline(store,phase,job,artifact):
    run=phase.checked_run(job['experiment_id'])
    if run['manifest_sha256']!=job['config']['data_manifest_sha256']: raise DataError('基线数据清单已改变。')
    verified=artifact['verified']
    result={'job_id':job['id'],'artifact_sha256':artifact['sha256'],'config_sha256':job['config_sha256'],
            'source':'authenticated_baseline_worker' if verified['real'] else 'software_contract_fixture',
            'baseline_verified':verified['real'],'training_verified':False,'optimizer_steps':0,
            'validation_dataset_sha256':verified['baseline']['dataset_sha256'],
            'predictions_sha256':canonical_sha(verified['baseline']),'calibration_sha256':verified['calibration_sha256'],
            'metrics':verified['metrics'],'raw_metrics':verified['raw_metrics'],'runtime':verified['runtime'],'at':timestamp()}
    store.write(store.path('runs',run['id'])/'baselines'/job['id']/'report.json',result)
    run['baseline_evaluation']={key:value for key,value in result.items() if key not in ('metrics','raw_metrics')}
    store.write(store.path('runs',run['id'])/'meta.json',run)
    return {key:result[key] for key in ('source','baseline_verified','training_verified','artifact_sha256')}
