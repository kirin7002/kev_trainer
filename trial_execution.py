"""The complete first-stage training pipeline, with one bounded owned process per stage.

Production uses RuntimeBackend; tests inject a clearly labelled software-only
backend. Training workers never receive final test/regression partitions.
"""
import json
import os
from pathlib import Path
import shutil
import sys

from datasets import DataError, fingerprint, parse_json
from evaluation import comparison
from execution import Deadline, Supervisor
from kev_contract import CONTRACT, inspect_checkpoint
from model_artifacts import build_bundle, canonical_sha, file_sha
from phase_one import system_one
from pipeline_validation import TRAIN_STAGES, convert_benchmark, inference_summary, training_metrics

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def read(path):
    return parse_json(Path(path).read_bytes())


def runtime_environment():
    allowed = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
               'HF_HOME', 'HF_HUB_CACHE', 'XDG_CACHE_HOME', 'TMPDIR', 'SYSTEMROOT')
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                       WANDB_DISABLED='true', TOKENIZERS_PARALLELISM='false')
    return environment


class RuntimeBackend:
    mode = 'cuda'

    def __init__(self, source, python, checkpoint_archive):
        self.source, self.python, self.checkpoint_archive = str(Path(source).resolve()), str(Path(python).absolute()), str(Path(checkpoint_archive).resolve())
        self.platform_sha = self.fingerprint()

    @staticmethod
    def fingerprint():
        return {path.name: file_sha(path) for path in list(ROOT.glob('*.py'))+[ROOT/'models.json', ROOT/'kev-source-contract.json'] if path.is_file()}

    def verify_platform(self, config):
        if self.platform_sha != self.fingerprint() or self.platform_sha != config['code_sha256']:
            raise DataError('Worker 平台代码与创建 Trial 的启动指纹不一致。')

    def probe(self, directory, role):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        supervisor = Supervisor(directory, Deadline(120), lambda stage: {'cancel_requested': False})
        return self.perform('VALIDATING', 'probe', {'work': str(directory.resolve()),
            'mode': 'evaluate' if role == 'evaluation' else 'train'}, supervisor)

    def perform(self, stage, operation, payload, supervisor):
        work = Path(payload['work'])
        source, output = work/'requests'/(stage+'.json'), work/'results'/(stage+'.json')
        write(source, {**payload, 'checkpoint_archive': self.checkpoint_archive})
        output.parent.mkdir(exist_ok=True)
        argv = [self.python, '-I', str(ROOT/'kev_runtime.py'), '--source', self.source,
                '--operation', operation, '--input', str(source), '--output', str(output),
                '--seconds', str(supervisor.deadline.remaining())]
        supervisor.run(stage, argv, cwd=work, environment=runtime_environment())
        return read(output)


class TrialPipeline:
    def __init__(self, backend, directory, heartbeat, seconds, interval=2):
        self.backend, self.directory = backend, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.supervisor = Supervisor(self.directory, Deadline(seconds), heartbeat, interval=interval)
        self.completed = []

    def stage(self, name, operation, payload, **extra):
        self.supervisor.check(name)
        result = self.backend.perform(name, operation, {**payload, **extra, 'stage': name}, self.supervisor)
        self.supervisor.check(name)
        write(self.directory/'reports'/(name+'.json'), result)
        self.completed.append(name)
        return result

    def run(self, inputs, job):
        work = self.directory
        if set(inputs['partitions']) != {'train', 'calibration', 'validation'}:
            raise DataError('训练 Worker 禁止接收最终 test/regression。')
        config = inputs['config']
        if self.backend.mode == 'cuda': self.backend.verify_platform(config)
        if canonical_sha(config) != inputs['config_sha256']:
            raise DataError('不可变 Trial 配置哈希不匹配。')
        if config.get('execution_protocol_version') != 1:
            raise DataError('旧 Trial 不包含训练协议，需创建新 Trial。')
        for role, rows in inputs['partitions'].items():
            if fingerprint(rows) != inputs['partition_sha256'][role]:
                raise DataError('输入分区哈希不匹配。')
            path = work/'data'/(role+'.jsonl')
            path.parent.mkdir(exist_ok=True)
            path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in system_one(rows, {'objects': inputs['objects']}, role)))
        write(work/'registry'/'inputs.json', {'partition_sha256': inputs['partition_sha256'],
              'config_sha256': inputs['config_sha256'], 'task_sha256': canonical_sha(inputs['objects']),
              'excluded_partitions': ['test', 'regression']})
        payload = {'work': str(work.resolve()), 'mode': 'train', 'config': config, 'objects': inputs['objects']}
        validation = self.stage('VALIDATING', 'validate', payload)
        runtime = validation['runtime']
        if self.backend.mode not in ('cuda', 'software_contract_only'):
            raise DataError('未知执行后端。')
        real = self.backend.mode == 'cuda'
        if real and (not runtime.get('runtime_verified') or runtime.get('execution_backend') != 'cuda'
                     or runtime.get('source_commit') != CONTRACT['commit']
                     or runtime.get('parent_archive_sha256') != CONTRACT['released_checkpoint']['sha256']):
            raise DataError('真实运行环境、源码或父 checkpoint 验证不通过。')
        if not validation.get('tokenizer_verified') or validation.get('rejected_records') != 0 or validation.get('truncated_records') != 0:
            raise DataError('Tokenizer 准入失败。')
        expected_counts = {role: len(rows) for role, rows in inputs['partitions'].items()}
        if validation.get('counts') != expected_counts or validation.get('training_records_expanded') != expected_counts['train']*10:
            raise DataError('完整对象训练展开覆盖不匹配。')
        objects = [obj['id'] for obj in inputs['objects']]
        cases, data_sha = inputs['partitions']['validation'], inputs['partition_sha256']['validation']

        def benchmark(stage, model, role, raw):
            receipt = self.stage(stage, 'benchmark', payload, model_role=model, data_role=role, raw=raw)
            directory = work / receipt['directory']
            if directory.resolve().parent != (work/'benchmarks').resolve():
                raise DataError('Benchmark 输出路径越界。')
            records = [parse_json(line) for line in (directory/'predictions.jsonl').read_bytes().splitlines() if line.strip()]
            normalized = convert_benchmark(records, read(directory/'report.json'), inputs['partitions'][role],
                                            objects, inputs['partition_sha256'][role])
            write(work/'reports'/(stage+'-predictions.json'), normalized)
            return normalized

        baseline_raw = benchmark('BASELINE_RAW', 'baseline', 'validation', True)
        benchmark('BASELINE_CALIBRATION', 'baseline', 'calibration', True)
        baseline_policy = self.stage('BASELINE_CALIBRATING', 'calibrate', payload, model_role='baseline')
        self.check_calibration(baseline_policy)
        baseline_system = benchmark('BASELINE_SYSTEM', 'baseline', 'validation', False)
        smoke = self.stage('SMOKE', 'train', payload)
        self.check_training(smoke, validation['training_records_expanded'], True)
        trained = self.stage('TRAINING', 'train', payload)
        train_receipt = self.check_training(trained, validation['training_records_expanded']*config['parameters']['epochs'], False)
        candidate_raw = benchmark('CANDIDATE_RAW', 'candidate', 'validation', True)
        benchmark('CANDIDATE_CALIBRATION', 'candidate', 'calibration', True)
        policy = self.stage('CALIBRATING', 'calibrate', payload, model_role='candidate')
        self.check_calibration(policy)
        candidate_system = benchmark('EVALUATING', 'candidate', 'validation', False)
        loaded = self.stage('LOADING', 'load', payload)
        if (not loaded.get('loader_verified') or loaded.get('objects_verified') != len(objects)
                or loaded.get('head_sha256') != file_sha(work/'candidate/head.pt')
                or loaded.get('weights_sha256') != file_sha(work/'candidate/adapter_model.safetensors')):
            raise DataError('当前推理加载器未验证完整产物，不能晋升。')
        structural = inspect_checkpoint(work/'candidate', config['model'])
        model_effect = comparison(cases, baseline_raw, candidate_raw)
        system_effect = comparison(cases, baseline_system, candidate_system)
        result = {'source': 'authenticated_gpu_worker' if real else 'software_contract_fixture',
                  'training_verified': real, 'validation_dataset_sha256': data_sha,
                  'model_effect_fixed_template_temperature_1_threshold_0_5': model_effect,
                  'system_effect_same_calibration_opportunity': system_effect,
                  'candidate_inference': inference_summary(candidate_system, cases), 'runtime': runtime,
                  'training': train_receipt, 'loader': {**loaded, 'loader_verified': real, 'loader_contract_passed': True}, 'calibration': policy,
                  'threshold_strategy': 'fixed_0_5', 'training_question_group_size': 5,
                  'budget_wall_seconds': self.supervisor.deadline.clock()-self.supervisor.deadline.started}
        self.supervisor.check('COMPARING')
        self.completed.append('COMPARING')
        write(work/'reports/COMPARING.json', result)
        if tuple(self.completed) != TRAIN_STAGES:
            raise DataError('阶段未真正完整串联。')
        package = work/'package'
        package.mkdir()
        shutil.copytree(work/'candidate', package/'checkpoint')
        for name in ('reports', 'policy', 'logs'):
            if (work/name).exists(): shutil.copytree(work/name, package/name)
        for benchmark in (work/'benchmarks').iterdir():
            shutil.copytree(benchmark, package/'reports'/('benchmark-'+benchmark.name))
        write(package/'policy/config.json', config)
        write(package/'policy/data-registry.json', read(work/'registry/inputs.json'))
        write(package/'policy/source-contract.json', CONTRACT)
        metadata = {'kind': 'kev_model_bundle', 'trial_id': job['trial_id'], 'job_id': job['id'],
                    'attempt_id': job['attempt_id'], 'config_sha256': inputs['config_sha256'],
                    'training_verified': real, 'execution_backend': self.backend.mode, 'loader_verified': real,
                    'loader_contract_passed': True,
                    'stages': self.completed, 'parent_archive_sha256': runtime['parent_archive_sha256'],
                    'base': config['model']['base'], 'base_revision': config['model']['base_revision'],
                    'base_weights_included': False, 'requires_pinned_cached_base': True,
                    'resume_supported': False, 'budget_policy': 'whole_job_wall_time_conservative_gpu_ceiling'}
        metadata['image_provenance']=runtime.get('image_provenance')
        bundle = work/'model.zip'
        build_bundle(package, bundle, metadata)
        self.supervisor.check('COMPARING')
        return {'kind': 'gpu_training_v1', 'state': 'SUCCEEDED', 'training_verified': real,
                'artifact_sha256': file_sha(bundle), 'artifact_path': str(bundle),
                'execution_backend': self.backend.mode, 'completed_stages': self.completed}

    def check_training(self, receipt, expected, smoke):
        path = (self.directory/receipt['loss_file']).resolve()
        if not path.is_relative_to((self.directory/'logs').resolve()):
            raise DataError('Loss 观测路径越界。')
        observations = [parse_json(line) for line in path.read_bytes().splitlines() if line.strip()]
        return training_metrics(receipt['metrics'], observations, expected, smoke)

    @staticmethod
    def check_calibration(policy):
        if (policy.get('fit_partition') != 'calibration' or not policy.get('groups_disjoint')
                or policy.get('method') != 'pinned_kev_metrics_registered_custom_calibration_v1'
                or policy.get('threshold_strategy') != 'fixed_0_5_no_validation_or_test_tuning'):
            raise DataError('校准没有绑定独立 calibration 来源。')
