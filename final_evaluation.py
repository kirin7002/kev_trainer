"""Separate frozen-candidate evaluator: test-only inputs, no train/calibration ops."""
import json
from pathlib import Path
import shutil

from datasets import DataError, fingerprint
from evaluation import comparison
from execution import Deadline, Supervisor
from kev_contract import CONTRACT
from model_artifacts import build_bundle, canonical_sha, file_sha, inspect_bundle
from phase_one import system_one
from pipeline_validation import convert_benchmark, inference_summary
from trial_execution import read, write

FINAL_STAGES = ('FINAL_VALIDATING', 'FINAL_RESTORING', 'FINAL_BASELINE_RAW', 'FINAL_CANDIDATE_RAW',
                'FINAL_BASELINE_SYSTEM', 'FINAL_CANDIDATE_SYSTEM', 'FINAL_LOADING', 'FINAL_COMPARING')


class FinalEvaluationPipeline:
    def __init__(self, backend, directory, heartbeat, seconds, interval=2):
        self.backend, self.directory = backend, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.supervisor = Supervisor(self.directory, Deadline(seconds), heartbeat, interval=interval)
        self.completed = []

    def stage(self, name, operation, payload, **extra):
        if operation in ('train', 'calibrate', 'validate'): raise DataError('最终评估禁止训练和校准。')
        self.supervisor.check(name)
        result = self.backend.perform(name, operation, {**payload, **extra, 'stage': name}, self.supervisor)
        self.supervisor.check(name)
        write(self.directory/'reports'/(name+'.json'), result)
        self.completed.append(name)
        return result

    def run(self, inputs, job, model_bundle):
        work = self.directory
        if self.backend.mode == 'cuda': self.backend.verify_platform(inputs['config'])
        if set(inputs['partitions']) != {'test'}:
            raise DataError('最终评估 Worker 只接收冻结 test 分区。')
        if canonical_sha(inputs['config']) != inputs['config_sha256']:
            raise DataError('最终评估配置 SHA 不一致。')
        manifest = inspect_bundle(model_bundle, {'kind': 'kev_model_bundle',
            'trial_id': job['trial_id'], 'config_sha256': inputs['config_sha256']}, work/'received')
        if file_sha(model_bundle) != inputs['model_artifact_sha256']:
            raise DataError('冻结模型包 SHA 不匹配。')
        if manifest.get('execution_backend') != self.backend.mode:
            raise DataError('最终评估后端与冻结模型来源不匹配。')
        shutil.copytree(work/'received/checkpoint', work/'candidate')
        for role in ('baseline', 'candidate'):
            policy = work/'received/policy'/(role+'-calibration.json')
            (work/'policy').mkdir(exist_ok=True)
            shutil.copyfile(policy, work/'policy'/policy.name)
        cases = inputs['partitions']['test']
        if fingerprint(cases) != inputs['partition_sha256']['test']:
            raise DataError('最终测试 SHA 不匹配。')
        data = work/'data/test.jsonl'
        data.parent.mkdir(exist_ok=True)
        data.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in system_one(
            cases, {'objects': inputs['objects'], **inputs['config'].get('task_policy', {})}, 'test')))
        payload = {'mode': 'evaluate', 'work': str(work.resolve()), 'config': inputs['config'], 'objects': inputs['objects']}
        validation = self.stage('FINAL_VALIDATING', 'validate-evaluation', payload)
        runtime = validation['runtime']
        real = self.backend.mode == 'cuda'
        if (not validation.get('tokenizer_verified') or validation.get('test_records') != len(cases)
                or validation.get('rejected_records') != 0 or validation.get('truncated_records') != 0
                or runtime.get('source_commit') != CONTRACT['commit']
                or runtime.get('parent_archive_sha256') != CONTRACT['released_checkpoint']['sha256']):
            raise DataError('最终评估 runtime/tokenizer/数据来源未验证。')
        if real and (not runtime.get('runtime_verified') or runtime.get('execution_backend') != 'cuda'):
            raise DataError('最终评估 GPU runtime 未验证。')
        restored = self.stage('FINAL_RESTORING', 'restore', payload)
        if not restored.get('restored_without_fitting'): raise DataError('禁止用 test 拟合校准。')
        objects = [obj['id'] for obj in inputs['objects']]
        def benchmark(stage, role, raw):
            receipt = self.stage(stage, 'benchmark', payload, model_role=role, data_role='test', raw=raw)
            directory = work/receipt['directory']
            if directory.resolve().parent != (work/'benchmarks').resolve(): raise DataError('评估输出路径越界。')
            lines = [read_line for line in (directory/'predictions.jsonl').read_bytes().splitlines() if line.strip()
                     for read_line in [json.loads(line)]]
            result = convert_benchmark(lines, read(directory/'report.json'), cases, objects, inputs['partition_sha256']['test'])
            write(work/'reports'/(stage+'-predictions.json'), result)
            return result
        a_raw = benchmark('FINAL_BASELINE_RAW', 'baseline', True)
        b_raw = benchmark('FINAL_CANDIDATE_RAW', 'candidate', True)
        baseline = benchmark('FINAL_BASELINE_SYSTEM', 'baseline', False)
        candidate = benchmark('FINAL_CANDIDATE_SYSTEM', 'candidate', False)
        loaded = self.stage('FINAL_LOADING', 'load', payload)
        if (not loaded.get('loader_verified') or loaded.get('weights_sha256') != file_sha(work/'candidate/adapter_model.safetensors')
                or loaded.get('head_sha256') != file_sha(work/'candidate/head.pt') or loaded.get('objects_verified') != len(objects)):
            raise DataError('最终评估当前模型加载器验证失败。')
        self.supervisor.check('FINAL_COMPARING')
        self.completed.append('FINAL_COMPARING')
        report = {'source': 'authenticated_final_evaluator' if real else 'software_contract_fixture',
                  'training_verified': real and manifest.get('training_verified') is True,
                  'model_effect': comparison(cases, a_raw, b_raw), 'system_effect': comparison(cases, baseline, candidate),
                  'candidate_inference': inference_summary(candidate, cases), 'runtime': runtime,
                  'selected_trial_id': job['trial_id'], 'model_artifact_sha256': inputs['model_artifact_sha256'],
                  'selected_validation_sha256': inputs['selected_validation_sha256'], 'test_fitted': False}
        write(work/'reports/FINAL_COMPARING.json', report)
        if tuple(self.completed) != FINAL_STAGES: raise DataError('最终评估阶段未串联。')
        package = work/'package'
        package.mkdir()
        for name in ('reports', 'logs', 'policy'):
            if (work/name).exists(): shutil.copytree(work/name, package/name)
        for benchmark in (work/'benchmarks').iterdir():
            shutil.copytree(benchmark, package/'reports'/('benchmark-'+benchmark.name))
        bundle = work/'evaluation.zip'
        build_bundle(package, bundle, {'kind': 'kev_evaluation_bundle', 'job_id': job['id'], 'trial_id': job['trial_id'],
            'attempt_id': job['attempt_id'], 'config_sha256': inputs['config_sha256'],
            'model_artifact_sha256': inputs['model_artifact_sha256'],
            'selected_validation_sha256': inputs['selected_validation_sha256'],
            'execution_backend': self.backend.mode, 'training_verified': report['training_verified'], 'stages': self.completed,
            'image_provenance':runtime.get('image_provenance')})
        self.supervisor.check('FINAL_COMPARING')
        return {'kind': 'gpu_evaluation_v1', 'artifact_path': str(bundle), 'artifact_sha256': file_sha(bundle),
                'training_verified': report['training_verified']}
