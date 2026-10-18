"""Immutable phase-one preparation; no training CLI or GPU calls are invented here."""
from datetime import datetime, timezone
from collections import Counter
import hashlib
import io
import itertools
import json
import math
import random
from pathlib import Path
import uuid
import zipfile

from datasets import DataError, assert_disjoint, exposure_keys, fingerprint, grouped_rows, question_ids
from evaluation import evaluate, validate_predictions
from kev_contract import CONTRACT

PARTITIONS = ('train', 'calibration', 'validation', 'test', 'regression')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def text(value, name, maximum=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DataError(f'{name} 必须是 1～{maximum} 字符的非空字符串。')
    try:
        value.encode('utf-8')
    except UnicodeError:
        raise DataError(f'{name} 包含无效 Unicode。')
    return value.strip()


def fields(value, allowed, required):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise DataError('字段缺失或包含不允许的字段。')


def number(value, name, lower, upper, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise DataError(f'{name} 类型无效。')
    try:
        valid = math.isfinite(value) and lower <= value <= upper
    except OverflowError:
        valid = False
    if not valid:
        raise DataError(f'{name} 必须在 {lower}～{upper} 之间。')
    return value


def assert_unexposed(store, rows):
    path = store.root / 'evaluation-usage.json'
    if path.is_file() and exposure_keys(rows) & set(store.read(path).get('_exposed_groups', [])):
        raise DataError('样本/关联组已参与最终测试，不能通过改 ID 重用为训练、校准、验证或新的独立 test；请作为 regression。')


def task_policy(task):
    return {key: task[key] for key in ('business_scope', 'label_mode') if key in task}


def scoped_rows(rows, task):
    scope = task.get('business_scope', '')
    if scope and any(row.get('scenario', scope) != scope for row in rows):
        raise DataError('样本的业务场景与固定任务不一致，请分别建立任务和实验。')
    return [{**row, 'scenario': scope} for row in rows] if scope else rows


def system_one(rows, task, role='train'):
    rows = scoped_rows(rows, task)
    objects = {item['id']: item for item in task['objects']}
    estimated = len(rows) * sum(len(json.dumps(obj, ensure_ascii=False).encode()) + 100 for obj in objects.values())
    estimated += sum(len(row['text'].encode()) + 500 for row in rows)
    if estimated > 128 * 1024 * 1024:
        raise DataError('问题转换预计超过 128 MB，请减小数据版本或精简任务说明。')
    output = []
    groups = {row['id']: sha(sorted(item['id'] for item in group)) for group in grouped_rows(rows) for row in group}
    for row in rows:
        if set(row['intents']) - objects.keys():
            raise DataError('数据包含任务目录之外的未知对象。')
        original = row.get('questions')
        if task.get('label_mode', 'complete') == 'explicit_questions':
            if not isinstance(original, dict) or not original or set(original) - objects.keys():
                raise DataError('按明确问题标注的任务需要非空 Noul questions，且问题必须属于固定目录。')
        elif original is not None and set(original) != objects.keys():
            raise DataError('完整标注模式必须显式标注全部对象；缺失对象不能补为 false。')
        if original is not None:
            if any(not isinstance(q, dict) or q.get('type') != 'noul' or type(q.get('label')) is not bool
                   for q in original.values()):
                raise DataError('每个 Noul 问题必须提供明确布尔标签。')
            if sorted(key for key, q in original.items() if q['label']) != sorted(row['intents']):
                raise DataError('显式 Noul 标签与记录的 intents 不一致。')
        questions = {}
        asked = set(question_ids(row, objects))
        for name, obj in objects.items():
            if name not in asked:
                continue
            questions[name] = {'type': 'noul', 'instructions': obj['instructions'],
                               'criteria': obj['criteria'], 'label': name in row['intents']}
            # Converted hard-label data uses the frozen task's question schema.
            # Original instructions/criteria/extra fields remain in original.bin
            # and records.json; soft targets cannot silently override bool labels.
        state = {'scenario': row['scenario'], 'text': row['text']} if 'scenario' in row else row['text']
        output.append({'id': row['id'], 'state': state, 'questions': questions,
                       '_meta': {'id': row['id'], 'group_id': groups[row['id']],
                                 'source': row.get('source_id', row.get('source_metadata', {}).get('source', 'user')),
                                 'split': role, 'variant': 'clean'},
                       **{key: value for key, value in row.items() if key.endswith('_id') or key == 'near_duplicate_group'}})
    return output


class PhaseOne:
    def __init__(self, store, models, code_root):
        self.store, self.models, self.code_root = store, models, Path(code_root)
        self.code_sha256 = self.code_fingerprint()
        for name in ('task-specs', 'trials'):
            (store.root / name).mkdir(exist_ok=True)

    def code_fingerprint(self):
        paths = list(self.code_root.glob('*.py')) + [self.code_root / name for name in ('models.json', 'kev-source-contract.json')]
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}

    def task(self, item_id):
        value = self.store.read(self.store.path('task-specs', item_id) / 'meta.json')
        definition = {key: value[key] for key in ('name', 'objects', 'schema_version', 'task_type')}
        definition.update(task_policy(value))
        if sha(definition) != value['sha256']:
            raise DataError('不可变任务目录哈希不匹配。')
        return value

    def create_task(self, body):
        fields(body, ('name', 'objects', 'business_scope', 'label_mode'), ('name', 'objects'))
        name = text(body['name'], '任务名称', 100)
        objects = body['objects']
        if not isinstance(objects, list) or not 1 <= len(objects) <= 100:
            raise DataError('任务需要显式提供 1～100 个问题；平台不会猜测目录。')
        mode = body.get('label_mode', 'complete')
        if mode not in ('complete', 'explicit_questions'):
            raise DataError('label_mode 必须为 complete 或 explicit_questions。')
        scope = body.get('business_scope', '')
        if not isinstance(scope, str) or len(scope) > 100:
            raise DataError('business_scope 必须为不超过 100 字符的业务场景名称。')
        scope = scope.strip()
        seen, validated = set(), []
        for obj in objects:
            fields(obj, ('id', 'instructions', 'criteria'), ('id', 'instructions', 'criteria'))
            item = {'id': text(obj['id'], 'id', 100), 'instructions': text(obj['instructions'], 'instructions', 4000)}
            criteria = obj['criteria']
            if not isinstance(criteria, dict) or set(criteria) - {'true', 'false'}:
                raise DataError('Noul criteria 必须为只含 true/false 描述的对象，可使用 null；不能是字符串。')
            item['criteria'] = {key: text(value, 'criteria', 4000) if value is not None else None for key, value in criteria.items()}
            if item['id'].casefold() in seen or item['id'].casefold() == 'unknown':
                raise DataError('对象 ID 重复或使用保留的 unknown。')
            seen.add(item['id'].casefold())
            validated.append(item)
        value = {'name': name, 'objects': validated, 'schema_version': 1, 'task_type': 'noul-multilabel',
                 'business_scope': scope, 'label_mode': mode}
        value.update(id=str(uuid.uuid4()), created_at=timestamp(), sha256=sha(value))
        self.store.write(self.store.path('task-specs', value['id']) / 'meta.json', value)
        return value

    def validate_dataset(self, rows, task_id, role='train'):
        if any(row.get('source_metadata', {}).get('split') in ('test', 'regression') and
               row['source_metadata']['split'] != role for row in rows):
            raise DataError('原始元数据标明 test/regression 的样本不能改作训练/校准/验证。')
        if role != 'regression':
            self.assert_not_exposed(rows)
        return system_one(rows, self.task(task_id), role)

    def assert_not_exposed(self, rows):
        assert_unexposed(self.store, rows)

    def split_training_pool(self, dataset_id, body):
        fields(body, ('calibration_fraction', 'validation_fraction', 'seed'),
               ('calibration_fraction', 'validation_fraction', 'seed'))
        calibration = number(body['calibration_fraction'], '校准比例', .01, .49)
        validation = number(body['validation_fraction'], '验证比例', .01, .49)
        if calibration + validation >= 1:
            raise DataError('训练分区必须保留非空比例。')
        seed = number(body['seed'], 'seed', 0, 2**32-1, True)
        directory = self.store.path('datasets', dataset_id)
        meta = self.store.read(directory / 'meta.json')
        if meta['role'] != 'train' or not meta.get('task_spec_id'):
            raise DataError('只能从已绑定任务的训练池派生分区；test/regression 不参与切分。')
        rows = self.store.read(directory / 'records.json')
        if fingerprint(rows) != meta['sha256']:
            raise DataError('训练池版本哈希不匹配。')
        groups = grouped_rows(rows)
        if len(groups) < 3:
            raise DataError('至少需要三个独立组，才能形成训练/校准/验证分区。')
        random.Random(seed).shuffle(groups)
        n_cal = min(len(groups)-2, max(1, round(len(groups)*calibration)))
        n_val = min(len(groups)-n_cal-1, max(1, round(len(groups)*validation)))
        return {'calibration': [row for group in groups[:n_cal] for row in group],
                'validation': [row for group in groups[n_cal:n_cal+n_val] for row in group],
                'train': [row for group in groups[n_cal+n_val:] for row in group]}, meta

    def create_experiment(self, body):
        fields(body, ('name', 'task_spec_id', 'partitions', 'model_id', 'max_trials', 'max_training_minutes'),
               ('name', 'task_spec_id', 'partitions', 'max_trials', 'max_training_minutes'))
        name = text(body['name'], '实验名称', 100)
        task = self.task(body['task_spec_id'])
        model_id = body.get('model_id', 'kev-0.8b-v1.0')
        if model_id != 'kev-0.8b-v1.0':
            raise DataError('第一阶段手动执行仅准备 Kev 0.8B；4B 属于后续阶段。')
        partitions = body['partitions']
        fields(partitions, PARTITIONS, PARTITIONS[:4])
        if any(not isinstance(item_id, str) for item_id in partitions.values()):
            raise DataError('分区版本 ID 必须是 UUID 字符串。')
        if len(set(partitions.values())) != len(partitions):
            raise DataError('五类分区必须引用独立的数据版本。')
        records, versions = {}, {}
        for role, item_id in partitions.items():
            directory = self.store.path('datasets', item_id)
            meta = self.store.read(directory / 'meta.json')
            rows = self.store.read(directory / 'records.json')
            if meta['role'] != role or meta.get('task_spec_id') != task['id']:
                raise DataError('分区角色或任务版本不一致；请按对应角色重新导入原文件形成新版本。')
            if not meta.get('original_sha256') or not (directory / 'original.bin').is_file():
                raise DataError('第一阶段需要保留原文件的数据版本，请重新导入。')
            if (hashlib.sha256((directory / 'original.bin').read_bytes()).hexdigest() != meta['original_sha256']
                    or fingerprint(rows) != meta['sha256']):
                raise DataError('数据版本内容或原文件已改变，不能创建可复现实验。')
            self.validate_dataset(rows, task['id'], role)
            records[role], versions[role] = rows, meta
        for a, b in itertools.combinations(records, 2):
            assert_disjoint(records[a], records[b])
        usage_file = self.store.root / 'evaluation-usage.json'
        if usage_file.is_file() and self.store.read(usage_file).get(versions['test']['sha256']):
            raise DataError('此测试版本已经参与最终比较，应归入 regression；请提供新的独立 test。')
        run_id = str(uuid.uuid4())
        manifest = {'schema_version': 1, 'task_spec': {'id': task['id'], 'sha256': task['sha256']},
                    'partitions': {role: {'id': meta['id'], 'sha256': meta['sha256'],
                        'original_sha256': meta['original_sha256'], 'count': len(records[role])}
                        for role, meta in versions.items()}, 'seed': 42,
                    'normalization_version': 'noul-v3-business-questions', 'near_duplicate_check': 'explicit-groups+unicode-surface',
                    'token_length_verified': False}
        limits = {'max_trials': number(body['max_trials'], '候选数量', 1, 20, True),
                  'max_training_minutes': number(body['max_training_minutes'], '训练分钟', 5, 1440, True)}
        now = timestamp()
        value = {'id': run_id, 'name': name, 'workflow_version': 2, 'model': self.models[model_id],
                 'task_spec_id': task['id'], 'train_dataset': versions['train'], 'test_dataset': versions['test'],
                 'split': {role: len(rows) for role, rows in records.items()}, 'manifest': manifest,
                 'manifest_sha256': sha(manifest), 'limits': limits, 'goal': '手动 0.8B Noul 微调', 'notes': '',
                 'max_training_minutes': limits['max_training_minutes'], 'claude_budget_usd': 0,
                 'status': 'draft', 'created_at': now, 'updated_at': now, 'events': [], 'error': None,
                 'report_revision': None, 'selected_trial_id': None, 'training_verified': False}
        value.update(task_policy(task))
        value['scenario_counts'] = {role: dict(Counter(row.get('scenario', '未指定') for row in rows))
                                   for role, rows in records.items()}
        scenarios = set().union(*(counts.keys() for counts in value['scenario_counts'].values()))
        value['coverage_warnings'] = [f'{name} 在 {role} 中没有样本，不能宣称该业务已完成独立验收。'
            for name in sorted(scenarios) for role in ('train', 'calibration', 'validation', 'test')
            if not value['scenario_counts'][role].get(name)]
        directory = self.store.path('runs', run_id)
        with self.store.lock:
            # Training workers/Agents receive only these three partitions.
            for role in ('train', 'calibration', 'validation'):
                self.store.write(directory / (('training' if role == 'train' else role) + '.json'), records[role])
            self.store.write(directory / 'data-manifest.json', manifest)
            self.store.write(directory / 'meta.json', value)
        return value

    def trials(self, run_id):
        self.store.read(self.store.path('runs', run_id) / 'meta.json')
        output = []
        for item in self.store.list('trials'):
            if item['experiment_id'] == run_id:
                report_path = self.validation_path(item)
                metrics = self.store.read(report_path)['metrics'] if report_path.is_file() else {}
                item['validation_summary'] = metrics.get('overall')
                item['validation_by_scenario'] = metrics.get('by_scenario', {})
                item['validation_annotation_scope'] = metrics.get('annotation_scope', 'complete')
                output.append(item)
        return output

    def validation_path(self, trial):
        root = self.store.path('trials', trial['id'])
        active = trial.get('active_model')
        if active:
            return root/'attempts'/active['attempt_id']/'validation-report.json'
        return root/'validation-report.json'

    def checked_run(self, run_id):
        directory = self.store.path('runs', run_id)
        run = self.store.read(directory/'meta.json')
        if (sha(run['manifest']) != run['manifest_sha256']
                or self.store.read(directory/'data-manifest.json') != run['manifest']
                or self.task(run['task_spec_id'])['sha256'] != run['manifest']['task_spec']['sha256']):
            raise DataError('不可变数据/任务清单已改变。')
        return run

    def create_trial(self, run_id, body, *, identity=None):
        keys = ('learning_rate', 'epochs', 'batch_size', 'gradient_accumulation', 'seed', 'precision')
        fields(body, keys, keys)
        config = {name: number(body[name], name, lo, hi, integer) for name, lo, hi, integer in (
            ('learning_rate', 1e-7, .001, False), ('epochs', 1, 20, True), ('batch_size', 1, 64, True),
            ('gradient_accumulation', 1, 256, True), ('seed', 0, 2**32-1, True))}
        if body['precision'] not in ('bf16', 'fp32'):
            raise DataError('固定 Kev 版本的 --dtype 仅支持 bf16/fp32，不接受 fp16。')
        config['precision'] = body['precision']
        with self.store.lock:
            run = self.store.read(self.store.path('runs', run_id) / 'meta.json')
            if identity and (self.store.path('trials', identity['id'])/'meta.json').is_file():
                item = self.store.read(self.store.path('trials', identity['id'])/'meta.json')
                if (item['experiment_id'] != run_id or item['config']['parameters'] != config
                        or any(item.get(key) != value for key, value in identity.items())):
                    raise DataError('自动候选重放身份或参数不一致。')
                return item
            if run.get('workflow_version') != 2:
                raise DataError('手动 Trial 需要完整五类分区方案的新实验。')
            if len(self.trials(run_id)) >= run['limits']['max_trials']:
                raise DataError('已达到用户设置的候选数量上限。')
            code_hashes = self.code_fingerprint()
            if code_hashes != self.code_sha256:
                raise DataError('服务启动后的代码/模型目录契约已改变，请重启服务后创建 Trial，避免记录错误运行版本。')
            frozen = {'schema_version': 1, 'experiment_id': run_id, 'parameters': config,
                      'data_manifest_sha256': run['manifest_sha256'], 'model': run['model'],
                      'code_sha256': code_hashes, 'limits': run['limits'],
                      'parent_checkpoint': run['model']['checkpoint'], 'resume_from': None,
                      'lora_rank': run['model']['lora_rank'], 'head_dim': run['model']['head_dim'],
                      'lora_targets': None, 'image_digest': None, 'calibration': None,
                      'kev_source_commit': CONTRACT['commit'], 'cli_contract_verification': 'source_read_only_not_runtime',
                      'lora_resume_supported': False, 'execution_protocol_version': 1,
                      'evaluation_protocol': {'dtype': 'fp32', 'backend': 'torch', 'merge': False,
                                              'thresholds': 'fixed_0_5', 'training_question_group_size': 5}}
            frozen['task_policy'] = task_policy(self.task(run['task_spec_id']))
            trial_id = identity['id'] if identity else str(uuid.uuid4())
            item = {'id': trial_id, 'experiment_id': run_id, 'created_at': timestamp(),
                    'status': 'WAITING_FOR_USER', 'config_sha256': sha(frozen), 'config': frozen,
                    'training_verified': False, 'blockers': ['kev_runtime_unverified',
                    'checkpoint_and_lora_targets_unverified', 'tokenizer_length_unverified', 'gpu_unverified'],
                    'attempts': []}
            if identity: item.update(identity)
            directory = self.store.path('trials', trial_id)
            self.store.write(directory / 'config.json', frozen)
            self.store.write(directory / 'meta.json', item)
            return item

    def worker_inputs(self, trial_id):
        trial = self.store.read(self.store.path('trials', trial_id) / 'meta.json')
        frozen = self.store.read(self.store.path('trials', trial_id) / 'config.json')
        if sha(frozen) != trial['config_sha256'] or frozen != trial['config']:
            raise DataError('Trial 不可变配置哈希不匹配。')
        run_dir = self.store.path('runs', trial['experiment_id'])
        run = self.checked_run(trial['experiment_id'])
        if (frozen['data_manifest_sha256'] != run['manifest_sha256'] or frozen['model'] != run['model']
                or frozen['limits'] != run['limits']):
            raise DataError('冻结 Trial 与当前数据/模型/预算清单不一致。')
        task = self.task(run['task_spec_id'])
        if frozen.get('task_policy', {}) != task_policy(task):
            raise DataError('冻结的业务场景或标注模式与任务目录不一致。')
        partitions = {role: self.store.read(run_dir / (('training' if role == 'train' else role) + '.json'))
                      for role in ('train', 'calibration', 'validation')}
        for role, rows in partitions.items():
            if fingerprint(rows) != run['manifest']['partitions'][role]['sha256']:
                raise DataError('实验分区快照被改变。')
            self.assert_not_exposed(rows)
        return {'schema_version': 1, 'trial_id': trial_id, 'config_sha256': trial['config_sha256'],
                'config': frozen,
                # Do not include the experiment/full manifest: it contains test IDs/hashes.
                'parameters': trial['config']['parameters'], 'model': trial['config']['model'],
                'objects': task['objects'], 'partitions': partitions,
                'partition_sha256': {role: fingerprint(rows) for role, rows in partitions.items()},
                'training_verified': False}

    def preparation_bundle(self, trial_id):
        with self.store.lock:
            inputs = self.worker_inputs(trial_id)
            trial = self.store.read(self.store.path('trials', trial_id) / 'meta.json')
            files = {'config.json': json.dumps(trial['config'], ensure_ascii=False, sort_keys=True).encode(),
                     'kev-source-contract.json': json.dumps(CONTRACT, sort_keys=True).encode(),
                     'worker-inputs.json': json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()}
            for role, rows in inputs['partitions'].items():
                converted = system_one(rows, {'objects': inputs['objects'], **inputs['config'].get('task_policy', {})}, role)
                files[role + '.jsonl'] = ('\n'.join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                                                    for row in converted) + '\n').encode()
            manifest = {'schema_version': 1, 'kind': 'preparation_bundle', 'is_model': False,
                        'training_verified': False, 'trial_id': trial_id, 'config_sha256': trial['config_sha256'],
                        'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
                        'excluded_partitions': ['test', 'regression'],
                        'blockers': trial['blockers']}
            files['manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
            target = io.BytesIO()
            with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name in sorted(files):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED)
            return target.getvalue()

    def import_validation(self, trial_id, payload):
        with self.store.lock:
            trial_dir = self.store.path('trials', trial_id)
            trial = self.store.read(trial_dir / 'meta.json')
            if trial.get('active_model'):
                raise DataError('已由执行器生成验证结果，不能外部覆盖；请使用独立 Trial。')
            run_dir = self.store.path('runs', trial['experiment_id'])
            run = self.store.read(run_dir / 'meta.json')
            if (run.get('selected_trial_id') == trial_id and
                    all((run_dir / (side + '-predictions.json')).exists() for side in ('baseline', 'candidate'))):
                raise DataError('最终比较已记录，不能重写选定候选的验证结果；后续应使用新独立 test。')
            rows = self.store.read(run_dir / 'validation.json')
            self.assert_not_exposed(rows)
            digest = run['manifest']['partitions']['validation']['sha256']
            objects = [obj['id'] for obj in self.task(run['task_spec_id'])['objects']]
            predictions = validate_predictions(payload, rows, digest, objects)
            result = {'source': 'user_imported_predictions', 'training_verified': False,
                      'dataset_sha256': digest, 'trial_id': trial_id, 'metrics': evaluate(rows, predictions),
                      'predictions_sha256': sha(predictions)}
            if run.get('selected_trial_id') == trial_id:
                run.update(selected_trial_id=None, selected_validation_sha256=None)
                self.store.write(run_dir / 'meta.json', run)
            self.store.write(trial_dir / 'validation-predictions.json', predictions)
            self.store.write(trial_dir / 'validation-report.json', result)
            return result

    def select_trial(self, run_id, trial_id):
        with self.store.lock:
            run_dir = self.store.path('runs', run_id)
            run = self.store.read(run_dir / 'meta.json')
            if run.get('final_evaluation_job_id'):
                raise DataError('最终评估已冻结候选，不能重新选择。')
            trial = self.store.read(self.store.path('trials', trial_id) / 'meta.json')
            if trial['experiment_id'] != run_id:
                raise DataError('Trial 不属于此实验。')
            if all((run_dir / (side + '-predictions.json')).exists() for side in ('baseline', 'candidate')):
                raise DataError('最终比较已记录，不能根据 test 重新选择候选；请另建独立 test。')
            validation = self.store.read(self.validation_path(trial))
            # Only a validation report is consulted here. Final test results
            # are neither an input nor a ranking criterion.
            run.update(selected_trial_id=trial_id, selected_validation_sha256=validation['predictions_sha256'],
                       selected_at=timestamp(), selection_source='manual_from_worker_validation' if trial.get('active_model') else 'manual_from_imported_validation')
            self.store.write(run_dir / 'meta.json', run)
            return {'selected_trial_id': trial_id, 'validation_sha256': validation['predictions_sha256'],
                    'source': run['selection_source'], 'training_verified': False}
