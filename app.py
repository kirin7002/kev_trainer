from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import threading
import uuid
import secrets
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

import claude_runner
from datasets import DataError, assert_disjoint, exposure_keys, fingerprint, normalize, parse_json, parse_upload, profile, split_validation
from evaluation import comparison, validate_predictions
from machine import get_machine_client
from storage import Store
from starlette.concurrency import run_in_threadpool
from model_artifacts import MAX_COMPRESSED, file_sha
from phase_one import PARTITIONS, PhaseOne, assert_unexposed, sha
from worker_queue import LeaseConflict, WorkerQueue
from optimization import Optimization

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

    @field_validator('*')
    @classmethod
    def valid_unicode(cls, value):
        if isinstance(value, str):
            try:
                value.encode('utf-8')
            except UnicodeError:
                raise ValueError('文本包含无效 Unicode。')
        return value


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
            if meta.get('workflow_version') == 2:
                raise HTTPException(409, '第一阶段手动 Trial 不依赖 Claude 方案；自动方案与搜索属于后续阶段。')
            if any(task[0].is_alive() for task in self.tasks.values()):
                raise HTTPException(409, '已有方案任务正在运行，请等待或取消后重试。')
            if not claude_runner.binary_path():
                raise HTTPException(409, '未找到 Claude Code，请安装并登录，或设置 CLAUDE_BIN。')
            cancel = threading.Event()
            # An accepted retry replaces the old attempt. Interrupted/failed
            # retries must never expose the previous plan as their own result.
            (self.directory(item_id) / 'plan.json').unlink(missing_ok=True)
            self.update(item_id, status='planning', error=None)
            self.tasks = {key: task for key, task in self.tasks.items() if task[0].is_alive()}
            thread = threading.Thread(target=self._plan, args=(item_id, cancel), daemon=True)
            self.tasks[item_id] = (thread, cancel)
            thread.start()

    def _plan(self, item_id, cancel):
        try:
            meta = self.load(item_id)
            directory = self.directory(item_id)
            training = self.store.read(directory / 'training.json')
            validation = self.store.read(directory / 'validation.json')
            assert_unexposed(self.store, training + validation)
            # 上下文只包含训练/验证资料及其概况；最终测试集不进入 Claude 上下文。
            context = {'task': '根据输入资料提出 Kev 微调方案；不执行训练。',
                'model': meta['model'], 'goal': meta['goal'], 'notes': meta['notes'],
                'limits': {'max_training_minutes': meta['max_training_minutes'], 'machine_connected': False},
                'training_profile': profile(training), 'validation_profile': profile(validation),
                'training_examples': training[:30], 'validation_examples': validation[:20],
                'labeling_rule': '只识别涉及的对象。否定、疑问中明确提及对象也计入；不解析动作。'}
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
                self.log(item_id, '方案已生成并通过校验；GPU 机器与训练执行器尚待接入，当前不能训练。')
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
    phase = PhaseOne(store, MODEL_MAP, ROOT)
    queue = WorkerQueue(store, phase)
    optimization = Optimization(store, phase, queue)

    @asynccontextmanager
    async def lifespan(app):
        optimization.launch()
        yield
        optimization.shutdown()
        experiments.shutdown()

    app = FastAPI(title='Kev Trainer', lifespan=lifespan)
    app.state.store = store
    app.state.experiments = experiments
    app.state.phase_one = phase
    app.state.worker_queue = queue
    app.state.optimization = optimization

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

    @app.exception_handler(LeaseConflict)
    async def lease_conflict(request, error):
        return JSONResponse({'detail': str(error)}, status_code=409)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Invalid input can contain NaN or lone surrogates, which cannot be
        # echoed in a JSON response. Field locations and messages suffice.
        details = [{key: item[key] for key in ('loc', 'msg', 'type')} for item in error.errors()]
        return JSONResponse({'detail': details}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({'detail': '记录不存在。'}, status_code=404)

    @app.get('/')
    def index():
        return FileResponse(ROOT / 'templates' / 'index.html', media_type='text/html')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'machine': queue.connection_status(),
                'claude': {'installed': bool(claude_runner.binary_path()), 'authentication': 'verified_when_called'},
                'training_execution': 'worker_pull' if queue.execution_enabled() else 'not_connected'}

    @app.get('/api/models')
    def models():
        return MODELS

    def save_dataset(rows, role, filename, raw=None, task_spec_id=None, source_rule=''):
        if role != 'regression':
            phase.assert_not_exposed(rows)
        converted = phase.validate_dataset(rows, task_spec_id, role) if task_spec_id else None
        item_id = str(uuid.uuid4())
        directory = store.path('datasets', item_id)
        if raw is None:
            raw = json.dumps(rows, ensure_ascii=False, allow_nan=False).encode()
        meta = {'id': item_id, 'role': role, 'filename': Path(filename).name[:200],
            'created_at': now(), 'sha256': fingerprint(rows), 'profile': profile(rows),
            'version_id': item_id, 'schema_version': 2, 'normalization_version': 'noul-v2',
            'original_sha256': hashlib.sha256(raw).hexdigest(), 'task_spec_id': task_spec_id,
            'source_rule': source_rule[:1000], 'token_length_verified': False,
            'near_duplicate_check': 'explicit-groups+unicode-surface'}
        store.write_bytes(directory / 'original.bin', raw)
        store.write(directory / 'records.json', rows)
        if converted is not None:
            store.write(directory / 'system-one.json', converted)
        meta['files'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in directory.iterdir() if path.is_file()}
        store.write(directory / 'meta.json', meta)
        return meta

    @app.get('/api/datasets')
    def datasets():
        return store.list('datasets')

    @app.post('/api/datasets', status_code=201)
    async def upload_dataset(role: str = Form(...), file: UploadFile = File(...),
                             task_spec_id: str | None = Form(default=None), source_rule: str = Form(default='')):
        if role not in PARTITIONS:
            raise DataError('role 必须为 train/calibration/validation/test/regression。')
        if not file.filename or Path(file.filename).suffix.lower() not in ('.json', '.jsonl', '.csv'):
            raise DataError('请上传 .json、.jsonl 或 .csv 文件。')
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, '文件不能超过 10 MB。')
        return save_dataset(parse_upload(raw, file.filename), role, file.filename, raw, task_spec_id, source_rule)

    @app.get('/api/datasets/{item_id}')
    def dataset(item_id: str):
        directory = store.path('datasets', item_id)
        meta = store.read(directory / 'meta.json')
        if meta['role'] == 'test' and meta.get('task_spec_id'):
            return {**meta, 'preview': [], 'preview_policy': 'withheld_final_test',
                    'preview_message': '最终测试仅展示质量概况；逐条内容与标签在冻结候选后的最终报告中查看，不用于调参。'}
        return {**meta, 'preview': store.read(directory / 'records.json')[:10]}

    @app.post('/api/examples', status_code=201)
    def examples():
        output = []
        for role in ('train', 'test'):
            filename = role + '.json'
            raw = (ROOT / 'examples' / filename).read_bytes()
            output.append(save_dataset(normalize(json.loads(raw)), role, 'example-' + filename, raw))
        return output

    @app.get('/api/examples/{role}')
    def example_file(role: str):
        if role not in ('train', 'test'):
            raise HTTPException(404, '示例不存在。')
        return FileResponse(ROOT / 'examples' / (role + '.json'), filename=role + '.json')

    async def config_body(request):
        raw = await request.body()
        if len(raw) > 65536:
            raise HTTPException(413, '配置请求不能超过 64 KB。')
        value = parse_json(raw)
        if not isinstance(value, dict):
            raise DataError('请求必须是 JSON 对象。')
        return value

    @app.post('/api/task-specs', status_code=201)
    async def create_task_spec(request: Request):
        return phase.create_task(await config_body(request))

    @app.get('/api/task-specs')
    def task_specs():
        return store.list('task-specs')

    @app.get('/api/task-specs/{item_id}')
    def task_spec(item_id: str):
        return phase.task(item_id)

    @app.post('/api/datasets/{item_id}/split', status_code=201)
    async def split_training_pool(item_id: str, request: Request):
        body = await config_body(request)
        with store.lock:
            partitions, parent = phase.split_training_pool(item_id, body)
            raw = (store.path('datasets', item_id) / 'original.bin').read_bytes()
            if hashlib.sha256(raw).hexdigest() != parent['original_sha256']:
                raise DataError('训练池原文件哈希不匹配。')
            versions = {}
            for role, rows in partitions.items():
                meta = save_dataset(rows, role, parent['filename'], raw, parent['task_spec_id'], parent['source_rule'])
                meta.update(parent_dataset_id=item_id, split_config=body, split_method='connected_groups_seeded',
                            split_note='比例按独立组近似，不保证按样本数或标签分层均衡；原文件保留父训练池。')
                store.write(store.path('datasets', meta['id']) / 'meta.json', meta)
                versions[role] = meta
            return versions

    @app.post('/api/manual-experiments', status_code=201)
    async def manual_experiment(request: Request):
        return phase.create_experiment(await config_body(request))

    @app.get('/api/experiments/{item_id}/optimizations')
    def optimization_list(item_id: str):
        experiments.load(item_id)
        return [s for s in store.list('optimizations') if s['experiment_id'] == item_id]

    @app.post('/api/experiments/{item_id}/optimizations', status_code=201)
    async def optimization_create(item_id: str, request: Request):
        return optimization.create(item_id, await config_body(request))

    @app.post('/api/experiments/{item_id}/baseline',status_code=202)
    def queue_baseline(item_id:str):
        if not queue.execution_enabled(): raise HTTPException(409,'GPU 执行未启用；没有创建基线作业。')
        return queue.enqueue_baseline(item_id)

    @app.get('/api/experiments/{item_id}/baseline')
    def baseline_report(item_id:str):
        run=phase.checked_run(item_id)
        current=run.get('baseline_evaluation')
        if not current: raise HTTPException(409,'尚无完整验证的 baseline-only 结果。')
        job=queue.read('jobs',current['job_id'])
        queue.accepted_artifact(job)
        return store.read(store.path('runs',item_id)/'baselines'/job['id']/'report.json')

    @app.get('/api/optimizations/{session_id}')
    def optimization_detail(session_id: str):
        session = optimization.load(session_id)
        if session.get('plan_sha256'): session['plan'] = optimization.plan(session)
        return session

    @app.get('/api/optimizations/{session_id}/inputs')
    def optimization_inputs(session_id: str):
        return optimization.inputs(session_id)

    @app.get('/api/optimizations/{session_id}/plan.md')
    def optimization_markdown(session_id: str):
        from plan_validation import markdown
        session = optimization.load(session_id)
        return Response(markdown(optimization.plan(session)), media_type='text/markdown',
                        headers={'Content-Disposition':'attachment; filename="plan.md"'})

    @app.post('/api/optimizations/{session_id}/{operation}')
    async def optimization_action(session_id: str, operation: str, request: Request):
        if operation == 'import-plan':
            body = await config_body(request)
            if not isinstance(body, dict) or set(body) != {'plan'}: raise DataError('只接受 plan 字段。')
            return optimization.import_plan(session_id, body['plan'])
        if operation == 'review': return optimization.review(session_id, await config_body(request))
        if operation == 'generate': return optimization.generate(session_id, await config_body(request))
        if operation == 'start': return optimization.start(session_id)
        if operation == 'advance': return optimization.advance(session_id)
        if operation == 'cancel': return optimization.cancel(session_id)
        raise HTTPException(404, '自动方案操作不存在。')

    @app.get('/api/experiments/{item_id}/trials')
    def trials(item_id: str):
        return phase.trials(item_id)

    @app.post('/api/experiments/{item_id}/trials', status_code=201)
    async def trial(item_id: str, request: Request):
        return phase.create_trial(item_id, await config_body(request))

    @app.get('/api/trials/{trial_id}')
    def trial_detail(trial_id: str):
        return queue.read('trials', trial_id)

    @app.post('/api/trials/{trial_id}/preflight', status_code=202)
    def queue_preflight(trial_id: str):
        return queue.enqueue(trial_id)

    @app.get('/api/trials/{trial_id}/preparation.zip')
    def preparation_bundle(trial_id: str):
        return Response(phase.preparation_bundle(trial_id), media_type='application/zip',
                        headers={'Content-Disposition': 'attachment; filename="preparation.zip"'})

    @app.post('/api/trials/{trial_id}/train')
    def train_trial(trial_id: str):
        trial = queue.read('trials', trial_id)
        if queue.execution_enabled():
            return JSONResponse(queue.enqueue(trial_id, 'gpu_training_v1'), status_code=202)
        return JSONResponse({'detail': '手动 Trial 已保存；固定版本 Kev CLI、checkpoint/targets、tokenizer 与 GPU 尚未验证，未执行训练。',
                             'blockers': trial['blockers'], 'training_verified': False}, status_code=409)

    @app.post('/api/trials/{trial_id}/validation-predictions')
    async def validation_predictions(trial_id: str, file: UploadFile = File(...)):
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, '预测文件不能超过 10 MB。')
        return phase.import_validation(trial_id, parse_json(raw))

    @app.post('/api/experiments/{item_id}/select-trial')
    async def select_trial(item_id: str, request: Request):
        body = await config_body(request)
        return phase.select_trial(item_id, body.get('trial_id'))

    @app.post('/api/experiments/{item_id}/evaluate', status_code=202)
    def final_evaluation(item_id: str):
        if len(os.environ.get('KEV_EVALUATOR_TOKEN', '')) < 32 and not queue.allow_fixture:
            raise HTTPException(409, '请先在进程环境配置独立的最终评估令牌；训练 Worker 不获得 test。')
        return queue.enqueue_evaluation(item_id)

    @app.get('/api/preparation')
    def preparation():
        with store.lock:
            queue.expire()
            queue.reconcile_publish()
            jobs = [{key: value for key, value in job.items() if key != 'lease_id'} for job in queue.jobs()]
            return {'workers': store.list('workers'), 'jobs': jobs,
                    'budgets': {run['id']: queue.budget(run['id']) for run in store.list('runs') if run.get('workflow_version') == 2},
                    'training_execution': 'worker_pull' if queue.execution_enabled() else 'disabled', 'training_verified': False,
                    'execution_enabled': queue.execution_enabled(),
                    'worker_auth_configured': len(os.environ.get('KEV_WORKER_TOKEN', '')) >= 32}

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        return queue.cancel(job_id)

    def worker_auth(request):
        expected = os.environ.get('KEV_WORKER_TOKEN', '')
        evaluator = os.environ.get('KEV_EVALUATOR_TOKEN', '')
        if expected and expected == evaluator:
            raise HTTPException(503, '训练与最终评估的进程令牌必须不同。')
        if len(expected) < 32 and len(evaluator) < 32:
            raise HTTPException(503, 'Worker 认证未配置：需在进程环境配置至少 32 字符的 KEV_WORKER_TOKEN，不写入数据或代码。')
        supplied = request.headers.get('authorization', '')
        for role, token in (('training', expected), ('evaluation', evaluator)):
            if len(token) >= 32 and secrets.compare_digest(supplied.encode(), ('Bearer ' + token).encode()): return role
        raise HTTPException(401, 'Worker 认证失败。')

    def worker_role(worker_id, role):
        if queue.read('workers', worker_id).get('auth_role', 'training') != role:
            raise HTTPException(403, 'Worker 令牌角色不匹配。')

    @app.post('/api/worker/register')
    async def register_worker(request: Request):
        role = worker_auth(request)
        return queue.register(await config_body(request), role)

    @app.post('/api/worker/claim')
    async def claim_job(request: Request):
        role = worker_auth(request)
        body = await config_body(request)
        worker_role(body.get('worker_id'), role)
        return {'job': queue.claim(body.get('worker_id'))}

    @app.post('/api/worker/resume')
    async def resume_worker(request: Request):
        role = worker_auth(request)
        body = await config_body(request)
        worker_role(body.get('worker_id'), role)
        return queue.read('workers', body['worker_id'])

    @app.post('/api/worker/jobs/{job_id}/{operation}')
    async def worker_job(job_id: str, operation: str, request: Request):
        role = worker_auth(request)
        if operation == 'artifacts':
            worker_id, lease_id = request.headers.get('x-worker-id'), request.headers.get('x-lease-id')
            worker_role(worker_id, role)
            with store.lock: queue.owned(job_id, worker_id, lease_id)
            incoming = store.root/'incoming-artifacts'
            incoming.mkdir(exist_ok=True)
            target = incoming/(str(uuid.uuid4())+'.zip')
            try:
                size = 0
                with target.open('xb') as output:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_COMPRESSED: raise HTTPException(413, '模型包超过 512 MB。')
                        output.write(chunk)
                return await run_in_threadpool(queue.accept_artifact, job_id, worker_id, lease_id, target)
            finally: target.unlink(missing_ok=True)
        body = await config_body(request)
        worker_id, lease_id = body.pop('worker_id', None), body.pop('lease_id', None)
        worker_role(worker_id, role)
        if operation == 'heartbeat':
            return queue.heartbeat(job_id, worker_id, lease_id, body.get('stage'))
        if operation == 'recovery-status':
            return queue.recovery_status(job_id, worker_id, lease_id)
        if operation == 'inputs':
            return queue.inputs(job_id, worker_id, lease_id)
        if operation == 'model':
            path = queue.evaluation_model(job_id, worker_id, lease_id)
            return FileResponse(path, filename='frozen-model.zip', media_type='application/zip')
        if operation == 'finish':
            return queue.finish(job_id, worker_id, lease_id, body)
        if operation == 'stopped':
            return queue.stopped(job_id, worker_id, lease_id, body.get('process_stopped'))
        raise HTTPException(404, 'Worker 操作不存在。')

    @app.get('/api/trials/{trial_id}/model.zip')
    def model_download(trial_id: str):
        trial = queue.read('trials', trial_id)
        active = trial.get('active_model')
        if not active or not active.get('training_verified') or not active.get('loader_verified'):
            raise HTTPException(409, '没有通过真实运行与当前加载器验证的完整模型包。')
        job = queue.read('jobs', active['job_id'])
        if job['state'] != 'SUCCEEDED' or job.get('promotion_pending'):
            raise HTTPException(409, '产物尚未完成原子晋升。')
        path = queue.artifact_root(job['id'], active['sha256'])/'bundle.zip'
        if file_sha(path) != active['sha256']: raise DataError('模型包内容哈希不匹配。')
        return FileResponse(path, filename='kev-model.zip', media_type='application/zip')

    @app.post('/api/experiments', status_code=201)
    async def create_experiment(body: ExperimentInput, request: Request):
        parse_json(await request.body())
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
        assert_disjoint(training, test)
        train, validation = split_validation(training)
        item_id = str(uuid.uuid4())
        directory = experiments.directory(item_id)
        meta = {'id': item_id, 'name': name, 'model': MODEL_MAP[body.model_id],
            'train_dataset': train_meta, 'test_dataset': test_meta,
            'split': {'train': len(train), 'validation': len(validation), 'test': len(test), 'seed': 42},
            'goal': body.goal, 'notes': body.notes, 'max_training_minutes': body.max_training_minutes,
            'claude_budget_usd': body.claude_budget_usd, 'status': 'draft',
            'created_at': now(), 'updated_at': now(), 'events': [], 'error': None, 'report_revision': None}
        store.write(directory / 'training.json', train)
        store.write(directory / 'validation.json', validation)
        experiments.save(meta)
        experiments.log(item_id, '实验已建立。最终测试集与训练/验证数据隔离。')
        return experiments.load(item_id)

    @app.get('/api/experiments')
    def list_experiments():
        return [{k: v for k, v in meta.items() if k != 'events'} for meta in store.list('runs')]

    @app.get('/api/experiments/{item_id}')
    def experiment(item_id: str):
        with experiments.lock:
            meta = experiments.load(item_id)
            directory = experiments.directory(item_id)
            meta['plan'] = store.read(directory / 'plan.json') if (directory / 'plan.json').exists() else None
            meta['predictions'] = {name: (directory / (name + '-predictions.json')).exists() for name in ('baseline', 'candidate')}
            return meta

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
        # Check and transition together with plan/cancel operations. Otherwise
        # a slow connection check can overwrite a newly accepted plan's state.
        with experiments.lock:
            meta = experiments.load(item_id)
            if meta.get('workflow_version') == 2:
                raise HTTPException(409, '此实验使用手动 Trial，不依赖 Claude 方案；请在 Trial 查看训练接入阻塞。')
            if meta['status'] in ('planning', 'cancelling'):
                raise HTTPException(409, '请等待方案任务结束。')
            if not (experiments.directory(item_id) / 'plan.json').exists():
                raise HTTPException(409, '请先生成并校验训练方案。')
            machine = get_machine_client().check_connection()
            if not machine.connected:
                experiments.update(item_id, status='waiting_for_machine')
                experiments.log(item_id, machine.message)
                raise HTTPException(409, machine.message)
            raise HTTPException(501, '远程训练执行器尚未实现。')

    @app.post('/api/experiments/{item_id}/predictions/{kind}')
    async def predictions(item_id: str, kind: str, file: UploadFile = File(...)):
        if kind not in ('baseline', 'candidate'):
            raise DataError('预测类型必须为 baseline 或 candidate。')
        meta = experiments.load(item_id)
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, '预测文件不能超过 10 MB。')
        value = parse_json(raw)
        if meta.get('workflow_version') == 2:
            if meta.get('final_evaluation_job_id'):
                raise HTTPException(409, '已冻结专用最终评估，不能替换测试预测。')
            if not meta.get('selected_trial_id'):
                raise HTTPException(409, '先导入 validation 预测并选定 Trial；不能用最终测试结果选择候选。')
            if kind == 'candidate' and value.get('trial_id') != meta['selected_trial_id']:
                raise DataError('候选预测的 trial_id 必须匹配 validation 选定的 Trial。')
        cases = store.read(store.path('datasets', meta['test_dataset']['id']) / 'records.json')
        objects = [obj['id'] for obj in phase.task(meta['task_spec_id'])['objects']] if meta.get('task_spec_id') else None
        validated = validate_predictions(value, cases, meta['test_dataset']['sha256'], objects)
        with experiments.lock, store.lock:
            directory = experiments.directory(item_id)
            latest = experiments.load(item_id)
            if meta.get('workflow_version') == 2 and any(latest.get(key) != meta.get(key)
                    for key in ('selected_trial_id', 'selected_validation_sha256')):
                raise HTTPException(409, '候选选择已变化，请刷新后重新导入预测。')
            meta = latest
            # A crash during replacement must not leave a report that claims
            # to describe the new predictions. Reads rebuild from the pair.
            (directory / 'report.json').unlink(missing_ok=True)
            store.write(directory / (kind + '-predictions.json'), validated)
            if all((directory / (name + '-predictions.json')).exists() for name in ('baseline', 'candidate')):
                report = comparison(cases, store.read(directory / 'baseline-predictions.json'),
                    store.read(directory / 'candidate-predictions.json'))
                with store.lock:
                    path = store.root / 'evaluation-usage.json'
                    usage = store.read(path) if path.is_file() else {}
                    entries = usage.setdefault(meta['test_dataset']['sha256'], [])
                    pair_sha = sha([store.read(directory / 'baseline-predictions.json'), store.read(directory / 'candidate-predictions.json')])
                    if not any(entry['pair_sha256'] == pair_sha and entry['experiment_id'] == item_id for entry in entries):
                        entries.append({'experiment_id': item_id, 'pair_sha256': pair_sha, 'at': now()})
                    usage['_exposed_groups'] = sorted(set(usage.get('_exposed_groups', [])) | exposure_keys(cases))
                    store.write(path, usage)
                report = report_provenance(meta, report)
                store.write(directory / 'report.json', report)
                experiments.update(item_id, report_revision=uuid.uuid4().hex)
            experiments.log(item_id, ('原模型' if kind == 'baseline' else '候选模型') + '的外部预测已导入。此操作不代表平台已完成训练。')
        return {'imported': len(cases), 'report_ready': (directory / 'report.json').exists()}

    @app.get('/api/experiments/{item_id}/report')
    def report(item_id: str):
        with experiments.lock:
            meta = experiments.load(item_id)
            directory = experiments.directory(item_id)
            cases = store.read(store.path('datasets', meta['test_dataset']['id']) / 'records.json')
            # Prediction files are the source of truth; a derived cache can
            # be absent/stale after a crash or older-version interruption.
            return report_provenance(meta, comparison(cases, store.read(directory / 'baseline-predictions.json'),
                              store.read(directory / 'candidate-predictions.json')))

    def report_provenance(meta, value):
        path = store.root / 'evaluation-usage.json'
        usage = store.read(path) if path.is_file() else {}
        count = len(usage.get(meta['test_dataset']['sha256'], []))
        value['test_usage'] = {'paired_evaluations': count, 'effective_role': 'regression' if count > 1 else 'test',
                               'single_recorded_pair': count == 1, 'independent_final_acceptance': False,
                               'prior_external_exposure': 'unknown',
                               'warning': '测试已重复用于不同预测比较，应转回归集并另建独立 test。' if count > 1 else None}
        receipt = meta.get('evaluation_receipt')
        if receipt:
            directory = experiments.directory(meta['id'])
            if (sha(store.read(directory/'baseline-predictions.json')) != receipt['baseline_sha256']
                    or sha(store.read(directory/'candidate-predictions.json')) != receipt['candidate_sha256']
                    or meta.get('selected_trial_id') != receipt['selected_trial_id']
                    or meta.get('selected_validation_sha256') != receipt['selected_validation_sha256']):
                raise DataError('最终评估回执与冻结模型/预测不一致。')
            job = queue.read('jobs', receipt['job_id'])
            artifact = store.read(queue.artifact_root(job['id'], receipt['artifact_sha256'])/'verified.json')
            verified = artifact['verified']
            value.update(source=receipt['source'], training_verified=receipt['training_verified'],
                model_effect=verified['model_effect'], system_effect=verified['system_effect'], test_fitted=False,
                selected_trial_id=receipt['selected_trial_id'], selected_validation_sha256=receipt['selected_validation_sha256'],
                protocol_final_acceptance_verified=receipt['training_verified'])
        if meta.get('workflow_version') == 2:
            value['experiment_manifest_sha256'] = meta['manifest_sha256']
            value['selected_trial_id'] = meta.get('selected_trial_id')
            value['selected_validation_sha256'] = meta.get('selected_validation_sha256')
            value['gain_attribution'] = 'fixed_protocol_and_equal_calibration_arms' if receipt else 'unverified_imported_predictions'
            if not receipt:
                value['performance_comparable'] = False
                value['comparability_issues'].append('外部导入尚未验证同目录/模板/校准/硬件负载及冷暖协议，不宣称模型收益归因或真实性能。')
        return value

    @app.get('/api/experiments/{item_id}/artifacts/{name}')
    def artifact(item_id: str, name: str):
        experiments.load(item_id)
        if name not in ('plan.json', 'report.json', 'meta.json', 'baseline-predictions.json', 'candidate-predictions.json'):
            raise HTTPException(404, '文件不存在。GPU 未接入时不会生成模型包。')
        if name == 'report.json':
            return JSONResponse(report(item_id), headers={'Content-Disposition': 'attachment; filename="report.json"'})
        with experiments.lock:
            path = experiments.directory(item_id) / name
            if not path.is_file():
                raise HTTPException(404, '文件尚未生成。')
            # Capture a complete revision before a retry/re-import replaces it.
            return JSONResponse(store.read(path), headers={'Content-Disposition': f'attachment; filename="{name}"'})

    return app


app = create_app()
