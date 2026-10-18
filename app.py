from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import claude_runner
from datasets import DataError, assert_disjoint, fingerprint, normalize, parse_upload, profile, split_validation
from evaluation import comparison, evaluate, validate_predictions
from experiment_protocol import (baseline_cache_key, build_protocol, digest, native_records,
    prediction_file, report_file, training_manifest, verify_protocol, verify_rows)
from machine import get_machine_client
from storage import Store

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 10 * 1024 * 1024
MODELS = json.loads((ROOT / 'models.json').read_text())
MODEL_MAP = {model['id']: model for model in MODELS}


def now():
    return datetime.now(timezone.utc).isoformat()


class ExperimentInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    train_dataset_id: str
    test_dataset_id: str
    model_id: str
    goal: str = Field(default='优先提高多对象整句准确率', min_length=1, max_length=1000)
    notes: str = Field(default='', max_length=3000)
    max_training_minutes: int = Field(default=60, ge=5, le=1440)
    claude_budget_usd: float = Field(default=1, ge=0.1, le=20, allow_inf_nan=False)
    label_definitions: dict[str, str] = Field(default_factory=dict, max_length=100)
    decision_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    business_scope: str = Field(default='', max_length=100)


class Experiments:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.tasks = {}
        for meta in store.list('runs'):
            if meta['status'] in ('planning', 'cancelling', 'running'):
                meta.update(status='interrupted', error='服务重启，上次任务已中断，请重试。', updated_at=now())
                self.save(meta)

    def directory(self, item_id):
        return self.store.path('runs', item_id)

    def load(self, item_id):
        return self.store.read(self.directory(item_id) / 'meta.json')

    def save(self, meta):
        self.store.write(self.directory(meta['id']) / 'meta.json', meta)

    def log(self, item_id, message):
        with self.lock:
            meta = self.load(item_id)
            meta['events'] = (meta.get('events', []) + [{'time': now(), 'message': claude_runner.redact(str(message))[:2000]}])[-300:]
            meta['updated_at'] = now()
            self.save(meta)

    def update(self, item_id, **changes):
        with self.lock:
            meta = self.load(item_id)
            meta.update(changes, updated_at=now())
            self.save(meta)

    def plan(self, item_id):
        with self.lock:
            meta = self.load(item_id)
            if meta.get('final_candidate'):
                raise HTTPException(409, '最终候选已锁定，不能继续优化；调整方案请建立新实验。')
            if any(task[0].is_alive() for task in self.tasks.values()):
                raise HTTPException(409, '已有方案任务正在运行，请等待或取消后重试。')
            if not claude_runner.binary_path():
                raise HTTPException(409, '未找到 Claude Code，请安装并登录，或设置 CLAUDE_BIN。')
            cancel = threading.Event()
            self.update(item_id, status='planning', error=None)
            thread = threading.Thread(target=self._plan, args=(item_id, cancel), daemon=True)
            self.tasks[item_id] = (thread, cancel)
            thread.start()

    def _plan(self, item_id, cancel):
        try:
            meta = self.load(item_id)
            directory = self.directory(item_id)
            training = self.store.read(directory / 'training.json')
            validation = self.store.read(directory / 'validation.json')
            if meta.get('protocol_sha256'):
                protocol = self.store.read(directory / 'protocol.json')
                if protocol.get('sha256') != meta['protocol_sha256']:
                    raise DataError('实验协议与元数据不一致。')
                verify_rows(protocol, 'training', training)
                verify_rows(protocol, 'validation', validation)
            # 上下文只包含训练/验证资料及其概况；最终测试集不进入 Claude 上下文。
            context = {'task': '根据输入资料提出 Kev 微调方案；不执行训练。',
                'model': meta['model'], 'goal': meta['goal'], 'notes': meta['notes'],
                'limits': {'max_training_minutes': meta['max_training_minutes'], 'machine_connected': False},
                'training_profile': profile(training), 'validation_profile': profile(validation),
                'training_examples': training[:30], 'validation_examples': validation[:20],
                'labeling_rule': '只识别涉及的对象。否定、疑问中明确提及对象也计入；不解析动作。'}
            if meta.get('protocol_sha256'):
                context.update(business_scope=protocol['business_scope'], questions=protocol['questions'],
                    labeling_rule='遵守固定问题定义和样本中的业务场景，不得修改评测协议。')
            self.store.write(directory / 'planning-context.json', context)
            workdir = self.store.root / 'agent-work' / item_id
            workdir.mkdir(parents=True, exist_ok=True)
            self.log(item_id, '正在调用 Claude Code；本轮仅传入训练和验证资料。')
            result = claude_runner.generate_plan(context, workdir, meta['claude_budget_usd'], cancel,
                lambda text: self.log(item_id, text))
            with self.lock:
                if cancel.is_set():
                    raise claude_runner.PlanningCancelled('方案生成已取消。')
                self.store.write(directory / 'plan.json', result)
                self.update(item_id, status='plan_ready', error=None)
                self.log(item_id, '方案已生成并通过校验，接入 GPU 机器后可执行训练。')
        except claude_runner.PlanningCancelled:
            self.update(item_id, status='cancelled', error=None)
            self.log(item_id, '本次方案生成已取消。')
        except Exception as error:
            message = claude_runner.redact(str(error))[:2000]
            self.update(item_id, status='failed', error=message)
            self.log(item_id, '方案生成失败：' + message)

    def cancel(self, item_id):
        with self.lock:
            meta = self.load(item_id)
            task = self.tasks.get(item_id)
            if task and task[0].is_alive():
                task[1].set()
                if meta['status'] in ('planning', 'running'):
                    self.update(item_id, status='cancelling')
            elif meta['status'] in ('planning', 'cancelling', 'running'):
                # 线程已结束但状态未落盘（例如服务重启前后），直接标记取消。
                self.update(item_id, status='cancelled')

    def shutdown(self):
        for thread, cancel in list(self.tasks.values()):
            cancel.set()
        for thread, cancel in list(self.tasks.values()):
            thread.join(timeout=7)


def create_app(data_dir=None):
    store = Store(data_dir or os.environ.get('KEV_TRAINER_DATA', ROOT / '.state'))
    experiments = Experiments(store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        experiments.shutdown()

    app = FastAPI(title='Kev Trainer', lifespan=lifespan)
    app.state.store = store
    app.state.experiments = experiments

    def evaluation_data(meta, split):
        if split not in ('validation', 'test'):
            raise DataError('评测阶段必须为 validation 或 test。')
        directory = experiments.directory(meta['id'])
        if not meta.get('protocol_sha256'):
            if split != 'test':
                raise HTTPException(409, '旧实验没有固定验证协议，请新建实验。')
            return None, store.read(store.path('datasets', meta['test_dataset']['id']) / 'records.json')
        protocol = store.read(directory / 'protocol.json')
        verify_protocol(protocol)
        if protocol['sha256'] != meta['protocol_sha256']:
            raise DataError('实验协议与元数据不一致。')
        cases = store.read(directory / (split + '.json'))
        verify_rows(protocol, split, cases)
        return protocol, cases

    def load_report(meta, split):
        protocol, _ = evaluation_data(meta, split)
        directory = experiments.directory(meta['id'])
        result = store.read(directory / report_file(split))
        if protocol:
            hashes = {kind: digest(store.read(directory / prediction_file(split, kind)))
                      for kind in ('baseline', 'candidate')}
            if result.get('prediction_sha256') != hashes or result.get('protocol_sha256') != protocol['sha256']:
                raise HTTPException(409, '报告与当前预测不一致，请先完成同一评测集的对比。')
        return result

    @app.middleware('http')
    async def local_access(request, call_next):
        host = request.url.hostname
        if host not in ('127.0.0.1', 'localhost', '::1', 'testserver'):
            return JSONResponse({'detail': '第一版仅提供本机访问，请使用 SSH 隧道。'}, status_code=403)
        origin = request.headers.get('origin')
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': '请求来源不匹配。'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(DataError)
    async def data_error(request, error):
        return JSONResponse({'detail': str(error)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({'detail': '记录不存在。'}, status_code=404)

    @app.get('/')
    def index():
        return FileResponse(ROOT / 'templates' / 'index.html', media_type='text/html')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'machine': asdict(get_machine_client().check_connection()),
                'claude': {'installed': bool(claude_runner.binary_path()), 'authentication': 'verified_when_called'},
                'training_execution': 'not_connected'}

    @app.get('/api/models')
    def models():
        return MODELS

    def save_dataset(rows, role, filename):
        item_id = str(uuid.uuid4())
        directory = store.path('datasets', item_id)
        meta = {'id': item_id, 'role': role, 'filename': Path(filename).name[:200],
            'created_at': now(), 'sha256': fingerprint(rows), 'profile': profile(rows)}
        store.write(directory / 'records.json', rows)
        store.write(directory / 'meta.json', meta)
        return meta

    @app.get('/api/datasets')
    def datasets():
        return store.list('datasets')

    @app.post('/api/datasets', status_code=201)
    async def upload_dataset(role: str = Form(...), file: UploadFile = File(...)):
        if role not in ('train', 'test'):
            raise DataError('role 只能为 train 或 test。')
        if not file.filename or Path(file.filename).suffix.lower() not in ('.json', '.jsonl', '.csv'):
            raise DataError('请上传 .json、.jsonl 或 .csv 文件。')
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, '文件不能超过 10 MB。')
        return save_dataset(parse_upload(raw, file.filename), role, file.filename)

    @app.get('/api/datasets/{item_id}')
    def dataset(item_id: str):
        directory = store.path('datasets', item_id)
        return {**store.read(directory / 'meta.json'), 'preview': store.read(directory / 'records.json')[:10]}

    @app.post('/api/examples', status_code=201)
    def examples():
        output = []
        for role in ('train', 'test'):
            filename = role + '.json'
            output.append(save_dataset(normalize(json.loads((ROOT / 'examples' / filename).read_text())),
                role, 'example-' + filename))
        return output

    @app.get('/api/examples/{role}')
    def example_file(role: str):
        if role not in ('train', 'test'):
            raise HTTPException(404, '示例不存在。')
        return FileResponse(ROOT / 'examples' / (role + '.json'), filename=role + '.json')

    @app.post('/api/experiments', status_code=201)
    def create_experiment(body: ExperimentInput):
        name = body.name.strip()
        if not name:
            raise DataError('实验名称不能为空。')
        if body.model_id not in MODEL_MAP:
            raise DataError('未知的基础模型版本。')
        if body.train_dataset_id == body.test_dataset_id:
            raise DataError('训练集和测试集不能是同一个数据集。')
        train_dir, test_dir = store.path('datasets', body.train_dataset_id), store.path('datasets', body.test_dataset_id)
        train_meta, test_meta = store.read(train_dir / 'meta.json'), store.read(test_dir / 'meta.json')
        if train_meta['role'] != 'train' or test_meta['role'] != 'test':
            raise DataError('请分别选择训练集和测试集。')
        training, test = store.read(train_dir / 'records.json'), store.read(test_dir / 'records.json')
        scope = body.business_scope.strip()
        if scope:
            if any(row.get('scenario', scope) != scope for row in training + test):
                raise DataError('本实验的业务场景与样本不一致；不同业务请分别建立实验。')
            training = [{**row, 'scenario': scope} for row in training]
            test = [{**row, 'scenario': scope} for row in test]
        assert_disjoint(training, test)
        train, validation = split_validation(training)
        protocol = build_protocol(MODEL_MAP[body.model_id], training,
            {'training': train, 'validation': validation, 'test': test},
            body.label_definitions, body.decision_threshold, scope)
        item_id = str(uuid.uuid4())
        directory = experiments.directory(item_id)
        meta = {'id': item_id, 'name': name, 'model': MODEL_MAP[body.model_id],
            'train_dataset': train_meta, 'test_dataset': test_meta,
            'split': {'train': len(train), 'validation': len(validation), 'test': len(test), 'seed': 42},
            'goal': body.goal, 'notes': body.notes, 'max_training_minutes': body.max_training_minutes,
            'claude_budget_usd': body.claude_budget_usd, 'status': 'draft',
            'created_at': now(), 'updated_at': now(), 'events': [], 'error': None, 'report_revision': None,
            'protocol_sha256': protocol['sha256'], 'final_candidate': None, 'business_scope': scope}
        store.write(directory / 'training.json', train)
        store.write(directory / 'validation.json', validation)
        store.write(directory / 'test.json', test)
        store.write(directory / 'protocol.json', protocol)
        experiments.save(meta)
        experiments.log(item_id, '实验已建立。最终测试集与训练/验证数据隔离。')
        return experiments.load(item_id)

    @app.get('/api/experiments')
    def list_experiments():
        return [{k: v for k, v in meta.items() if k != 'events'} for meta in store.list('runs')]

    @app.get('/api/experiments/{item_id}')
    def experiment(item_id: str):
        meta = experiments.load(item_id)
        directory = experiments.directory(item_id)
        meta['plan'] = store.read(directory / 'plan.json') if (directory / 'plan.json').exists() else None
        meta['predictions'] = {name: (directory / (name + '-predictions.json')).exists() for name in ('baseline', 'candidate')}
        if meta.get('protocol_sha256'):
            protocol, _ = evaluation_data(meta, 'validation')
            meta['evaluation_protocol'] = protocol
            meta['evaluations'] = {split: {
                **protocol['splits'][split],
                'baseline_cache_key': baseline_cache_key(protocol, split),
                **{kind: (directory / prediction_file(split, kind)).exists() for kind in ('baseline', 'candidate')},
                'report_ready': (directory / report_file(split)).exists(),
                'locked': split == 'test' and not bool(meta.get('final_candidate'))}
                for split in ('validation', 'test')}
            for split, info in meta['evaluations'].items():
                if info['baseline']:
                    _, cases = evaluation_data(meta, split)
                    info['baseline_summary'] = evaluate(cases,
                        store.read(directory / prediction_file(split, 'baseline')))['overall']
        return meta

    @app.post('/api/experiments/{item_id}/finalize')
    def finalize(item_id: str):
        with experiments.lock:
            meta = experiments.load(item_id)
            if not meta.get('protocol_sha256'):
                raise HTTPException(409, '旧实验没有候选锁定流程，请新建实验。')
            if meta['status'] in ('planning', 'cancelling', 'running'):
                raise HTTPException(409, '请先结束当前任务。')
            protocol, _ = evaluation_data(meta, 'validation')
            directory = experiments.directory(item_id)
            if meta.get('final_candidate'):
                return meta['final_candidate']
            if not (directory / report_file('validation')).exists():
                raise HTTPException(409, '请先完成固定验证集上的基线与候选对比。')
            load_report(meta, 'validation')
            candidate = store.read(directory / prediction_file('validation', 'candidate'))
            selected = {'model': candidate['model'], 'validation_predictions_sha256': digest(candidate),
                'protocol_sha256': protocol['sha256'], 'selected_at': now()}
            experiments.update(item_id, final_candidate=selected)
            experiments.log(item_id, '最终候选已锁定，开放保留测试集；本实验不再接受验证集候选更新。')
            return selected

    @app.get('/api/experiments/{item_id}/evaluations/{split}')
    def evaluation_input(item_id: str, split: str):
        meta = experiments.load(item_id)
        protocol, cases = evaluation_data(meta, split)
        if protocol is None:
            raise HTTPException(409, '旧实验没有固定评测输入，请新建实验。')
        if split == 'test' and not meta.get('final_candidate'):
            raise HTTPException(409, '请先在验证集选定并锁定最终候选。')
        return {'split': split, 'dataset_sha256': protocol['splits'][split]['sha256'],
            'protocol_sha256': protocol['sha256'], 'baseline_model': protocol['baseline']['checkpoint'],
            'decision': protocol['decision'], 'records': native_records(cases, protocol['questions'], False)}

    @app.get('/api/experiments/{item_id}/native-training/{name}')
    def native_training(item_id: str, name: str):
        meta = experiments.load(item_id)
        protocol, _ = evaluation_data(meta, 'validation')
        if protocol is None:
            raise HTTPException(409, '旧实验没有固定问题定义，请新建实验。')
        if name == 'manifest.json':
            return training_manifest(protocol)
        if name not in ('training.jsonl', 'validation.jsonl'):
            raise HTTPException(404, '训练包仅包含训练与验证资料。')
        split = name.removesuffix('.jsonl')
        rows = store.read(experiments.directory(item_id) / (split + '.json'))
        verify_rows(protocol, split, rows)
        content = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                          for row in native_records(rows, protocol['questions']))
        return Response(content, media_type='application/x-ndjson',
                        headers={'Content-Disposition': f'attachment; filename="{name}"'})

    @app.post('/api/experiments/{item_id}/plan', status_code=202)
    def plan(item_id: str):
        experiments.plan(item_id)
        return {'status': 'planning'}

    @app.post('/api/experiments/{item_id}/cancel')
    def cancel(item_id: str):
        experiments.cancel(item_id)
        return {'status': experiments.load(item_id)['status']}

    @app.post('/api/experiments/{item_id}/train')
    def train(item_id: str):
        meta = experiments.load(item_id)
        if meta['status'] in ('planning', 'cancelling'):
            raise HTTPException(409, '请等待方案任务结束。')
        if not meta.get('protocol_sha256') and not (experiments.directory(item_id) / 'plan.json').exists():
            raise HTTPException(409, '请先生成并校验训练方案。')
        if meta.get('protocol_sha256'):
            if meta.get('final_candidate'):
                raise HTTPException(409, '最终候选已锁定，不能继续训练，请建立新实验。')
            evaluation_data(meta, 'validation')
            directory = experiments.directory(item_id)
            if not (directory / prediction_file('validation', 'baseline')).exists():
                raise HTTPException(409, '请先完成固定验证集上的基线评测。')
        machine = get_machine_client().check_connection()
        if not machine.connected:
            experiments.update(item_id, status='waiting_for_machine')
            experiments.log(item_id, machine.message)
            raise HTTPException(409, machine.message)
        raise HTTPException(501, '远程训练执行器尚未实现。')

    @app.post('/api/experiments/{item_id}/predictions/{kind}')
    async def predictions(item_id: str, kind: str, split: str = 'test', file: UploadFile = File(...)):
        if kind not in ('baseline', 'candidate'):
            raise DataError('预测类型必须为 baseline 或 candidate。')
        meta = experiments.load(item_id)
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, '预测文件不能超过 10 MB。')
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise DataError('预测文件必须为有效 JSON。')
        with experiments.lock:
            meta = experiments.load(item_id)
            directory = experiments.directory(item_id)
            protocol, cases = evaluation_data(meta, split)
            sha = protocol['splits'][split]['sha256'] if protocol else meta['test_dataset']['sha256']
            validated = validate_predictions(value, cases, sha)
            path = directory / prediction_file(split, kind)
            if protocol:
                if not isinstance(value, dict) or value.get('protocol_sha256') != protocol['sha256'] or value.get('split') != split:
                    raise DataError('预测文件必须携带本实验的 protocol_sha256 和评测 split。')
                model = value.get('model')
                if not isinstance(model, str) or not model.strip() or len(model) > 200:
                    raise DataError('必须提供不超过 200 字符的模型检查点标识。')
                if kind == 'baseline' and model != protocol['baseline']['checkpoint']:
                    raise DataError('基线模型必须与实验选定的检查点版本一致。')
                if any(set(row['intents']) - protocol['questions'].keys() for row in validated['predictions'].values()):
                    raise DataError('预测包含实验协议之外的标签。')
                for case in cases:
                    if 'questions' in case and set(validated['predictions'][case['id']]['intents']) - case['questions'].keys():
                        raise DataError('预测包含当前样本未询问或未标注的问题。')
                selected = meta.get('final_candidate')
                if split == 'test' and not selected:
                    raise HTTPException(409, '请先在验证集选定并锁定最终候选，再执行最终测试。')
                if split == 'validation' and selected:
                    raise HTTPException(409, '最终候选已锁定，验证结果不能再更新。')
                if kind == 'candidate' and not (directory / prediction_file(split, 'baseline')).exists():
                    raise HTTPException(409, '请先导入同一评测集上的基线结果。')
                if kind == 'candidate' and split == 'test' and model != selected['model']:
                    raise DataError('最终测试必须使用已锁定的候选检查点。')
                validated.update(protocol_sha256=protocol['sha256'], split=split,
                    baseline_cache_key=baseline_cache_key(protocol, split) if kind == 'baseline' else None)
                if path.exists() and (kind == 'baseline' or split == 'test') and store.read(path) != validated:
                    raise HTTPException(409, '基线和最终测试结果不可替换；条件变化请创建新实验。')
            store.write(path, validated)
            if all((directory / prediction_file(split, name)).exists() for name in ('baseline', 'candidate')):
                report = comparison(cases, store.read(directory / prediction_file(split, 'baseline')),
                    store.read(directory / prediction_file(split, 'candidate')))
                if protocol:
                    report.update(split=split, protocol_sha256=protocol['sha256'],
                        business_scope=protocol['business_scope'],
                        decision=protocol['decision'], baseline_cache_key=baseline_cache_key(protocol, split),
                        inference_configuration_verified=False,
                        prediction_sha256={name: digest(store.read(directory / prediction_file(split, name)))
                                           for name in ('baseline', 'candidate')})
                store.write(directory / report_file(split), report)
                experiments.update(item_id, report_revision=uuid.uuid4().hex)
            experiments.log(item_id, ('原模型' if kind == 'baseline' else '候选模型') + '的外部预测已导入。此操作不代表平台已完成训练。')
        return {'imported': len(cases), 'report_ready': (directory / report_file(split)).exists()}

    @app.get('/api/experiments/{item_id}/report')
    def report(item_id: str, split: str = 'test'):
        meta = experiments.load(item_id)
        return load_report(meta, split)

    @app.get('/api/experiments/{item_id}/artifacts/{name}')
    def artifact(item_id: str, name: str):
        meta = experiments.load(item_id)
        if name not in ('plan.json', 'report.json', 'meta.json', 'baseline-predictions.json', 'candidate-predictions.json',
                        'protocol.json', 'validation-report.json', 'validation-baseline-predictions.json',
                        'validation-candidate-predictions.json'):
            raise HTTPException(404, '文件不存在。GPU 未接入时不会生成模型包。')
        if meta.get('protocol_sha256') and name.endswith('report.json'):
            load_report(meta, 'validation' if name.startswith('validation-') else 'test')
        path = experiments.directory(item_id) / name
        if not path.is_file():
            raise HTTPException(404, '文件尚未生成。')
        return FileResponse(path, filename=name, media_type='application/json')

    return app


app = create_app()
