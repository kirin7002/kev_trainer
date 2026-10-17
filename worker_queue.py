"""Single-process durable pull/lease contract with opt-in CUDA execution.

Expired jobs are fenced and held until their owner reports its process stopped.
Runtime proof and independent artifact validation precede model promotion.
"""
import time
import uuid
import os
from pathlib import Path
import shutil
import tempfile

from datasets import DataError
from claude_runner import redact
from phase_one import fields, text, timestamp
from kev_contract import CONTRACT
from model_artifacts import canonical_sha, file_sha, inspect_bundle
from pipeline_commit import commit_training, commit_evaluation, validate_training_bundle, validate_evaluation_bundle
from pipeline_validation import TRAIN_STAGES,validate_image_provenance
from final_evaluation import FINAL_STAGES
from baseline_execution import BASELINE_STAGES
from baseline_commit import validate_baseline_bundle,commit_baseline
from datasets import exposure_keys, fingerprint


class LeaseConflict(DataError):
    pass


class WorkerQueue:
    def __init__(self, store, phase, clock=time.time, lease_seconds=30):
        self.store, self.phase, self.clock, self.ttl = store, phase, clock, lease_seconds
        for name in ('workers', 'jobs'):
            (store.root / name).mkdir(exist_ok=True)
        self.allow_fixture = False  # only dependency-injected isolated tests set this
        self.execution_enabled = lambda: os.environ.get('KEV_ENABLE_GPU_EXECUTION') == '1'
        self.ledger_path = store.root / 'worker-queue.json'
        if not self.ledger_path.exists():
            store.write(self.ledger_path, {'schema_version': 1, 'jobs': {item['id']: item for item in store.list('jobs')}, 'budgets': {}})

    def ledger(self):
        return self.store.read(self.ledger_path)

    def jobs(self):
        return sorted(self.ledger()['jobs'].values(), key=lambda item: item['created_at'], reverse=True)

    def commit(self, ledger):
        # Ownership, job state and experiment budget are one atomic JSON write.
        self.store.write(self.ledger_path, ledger)

    def budget(self, experiment_id):
        run = self.store.read(self.store.path('runs', experiment_id)/'meta.json')
        value = self.ledger()['budgets'].get(experiment_id, {'charged_seconds': 0, 'reserved_job': None})
        return {**value, 'limit_seconds': run['limits']['max_training_minutes']*60}

    def connection_status(self):
        active = [worker for worker in self.store.list('workers') if self.execution_enabled()
                  and 'gpu_training_v1' in worker.get('capabilities', [])
                  and worker.get('runtime', {}).get('execution_backend') == 'cuda'
                  and worker.get('runtime', {}).get('runtime_verified') is True
                  and self.clock()-worker['last_seen'] <= self.ttl]
        return {'connected': bool(active), 'message': '已收到固定运行时 CUDA Worker 最近心跳；实际模型验收以作业回执为准。' if active
                else 'GPU Worker 尚未接入或心跳已过期；CPU 准备可用，未验证真实训练。'}

    def read(self, kind, item_id):
        if kind == 'jobs':
            self.store.path(kind, item_id)
            try: return self.ledger()['jobs'][item_id]
            except KeyError: raise KeyError('作业不存在。')
        return self.store.read(self.store.path(kind, item_id) / 'meta.json')

    def save(self, kind, value):
        if kind == 'jobs':
            ledger = self.ledger()
            ledger['jobs'][value['id']] = value
            self.commit(ledger)
            return
        self.store.write(self.store.path(kind, value['id']) / 'meta.json', value)

    def register(self, body, auth_role='training'):
        with self.store.lock:
            return self._register(body, auth_role)

    def _register(self, body, auth_role):
        fields(body, ('name', 'platform', 'python_version', 'capabilities', 'runtime', 'worker_id'), ('name', 'platform', 'python_version'))
        item = {key: text(body[key], key, 200) for key in ('name', 'platform', 'python_version')}
        capabilities = body.get('capabilities', ['cpu_preflight_v1'])
        allowed = {'gpu_evaluation_v1'} if auth_role == 'evaluation' else {'cpu_preflight_v1', 'gpu_training_v1','gpu_baseline_v1'}
        if not isinstance(capabilities, list) or not capabilities or any(not isinstance(cap, str) or cap not in allowed for cap in capabilities):
            raise DataError('Worker 角色与能力不匹配。')
        gpu = any(cap.startswith('gpu_') for cap in capabilities)
        runtime = body.get('runtime')
        if gpu:
            if not self.execution_enabled(): raise DataError('GPU 执行未由运行者显式启用。')
            if not isinstance(runtime, dict): raise DataError('缺少运行时验证。')
            validate_image_provenance(runtime.get('image_provenance'))
            real = runtime.get('execution_backend') == 'cuda' and runtime.get('runtime_verified') is True
            fixture = self.allow_fixture and runtime.get('execution_backend') == 'software_contract_only'
            if (not (real or fixture) or runtime.get('source_commit') != CONTRACT['commit']
                    or runtime.get('contract_sha256') != canonical_sha(CONTRACT)
                    or runtime.get('parent_archive_sha256') != CONTRACT['released_checkpoint']['sha256']):
                raise DataError('运行时/源码/父 checkpoint 验证不匹配。')
            if real:
                dependencies = runtime.get('dependencies')
                if (not isinstance(dependencies, dict) or any(not isinstance(dependencies.get(name), str)
                        or dependencies[name].split('+')[0] not in versions
                        for name, versions in CONTRACT['runtime_dependency_pins'].items())):
                    raise DataError('运行时依赖不符合固定 uv.lock。')
        previous = self.read('workers', body['worker_id']) if body.get('worker_id') else None
        if previous:
            if previous.get('auth_role', 'training') != auth_role: raise LeaseConflict('Worker 角色不能改变。')
            self.expire()
            if any(job['worker_id'] == previous['id'] and job['state'] in ('RUNNING', 'CANCELLING', 'WAITING_FOR_STOP_CONFIRMATION') for job in self.jobs()):
                raise LeaseConflict('先恢复并停止旧作业，再刷新运行环境。')
        item.update(id=previous['id'] if previous else str(uuid.uuid4()), created_at=previous['created_at'] if previous else timestamp(), last_seen=self.clock(),
                    capabilities=capabilities, training_capable='gpu_training_v1' in capabilities,
                    gpu_capable=gpu, runtime=runtime, auth_role=auth_role)
        self.save('workers', item)
        return item

    def recovery_status(self, job_id, worker_id, lease_id):
        with self.store.lock:
            self.expire()
            job = self.read('jobs', job_id)
            if job['worker_id'] != worker_id or job['lease_id'] != lease_id:
                raise LeaseConflict('恢复请求归属不匹配。')
            return {'state': job['state'], 'terminal': job['state'] in ('SUCCEEDED', 'FAILED', 'CANCELLED', 'EXPIRED')}

    def expire(self):
        for job in self.jobs():
            if job['state'] in ('RUNNING', 'CANCELLING') and job['lease_expires_at'] <= self.clock():
                job['state'] = 'WAITING_FOR_STOP_CONFIRMATION'
                job['stop_required'] = True
                self.save('jobs', job)

    def enqueue(self, trial_id, kind='cpu_preflight_v1', *, job_id=None, session_id=None):
        with self.store.lock:
            trial = self.read('trials', trial_id)
            if job_id and job_id in self.ledger()['jobs']:
                previous = self.read('jobs', job_id)
                if previous['trial_id'] != trial_id or previous['kind'] != kind: raise LeaseConflict('幂等作业归属不一致。')
                return previous
            if kind not in ('cpu_preflight_v1', 'gpu_training_v1'):
                raise DataError('作业类型不受支持。')
            if kind == 'gpu_training_v1':
                if trial.get('automation_session_id') and session_id != trial['automation_session_id']:
                    raise LeaseConflict('已审批自动候选由编排器提交，不能另开手动尝试。')
                if not self.execution_enabled(): raise DataError('GPU 执行未启用；未创建训练任务。')
                self.phase.worker_inputs(trial_id)
                if trial['config'].get('execution_protocol_version') != 1: raise DataError('旧 Trial 需重新创建。')
                run = self.store.read(self.store.path('runs', trial['experiment_id'])/'meta.json')
                if run.get('final_evaluation_job_id') or any((self.store.path('runs', trial['experiment_id'])/(side+'-predictions.json')).exists() for side in ('baseline','candidate')):
                    raise DataError('最终验收已冻结或外部 test 预测已导入，不能继续训练。')
                if self.budget(trial['experiment_id'])['charged_seconds'] >= run['limits']['max_training_minutes']*60:
                    raise DataError('实验统一时间预算已用尽。')
            if any(job['trial_id'] == trial_id and (job['state'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED', 'EXPIRED') or job.get('promotion_pending'))
                   for job in self.jobs()):
                raise LeaseConflict('此 Trial 仍有作业或旧进程尚未确认停止。')
            job_id, attempt_id = job_id or str(uuid.uuid4()), str(uuid.uuid4())
            if session_id: attempt_id = str(uuid.uuid5(uuid.UUID(job_id), 'attempt-1'))
            job = {'id': job_id, 'trial_id': trial_id, 'experiment_id': trial['experiment_id'],
                   'kind': kind, 'state': 'QUEUED', 'created_at': timestamp(),
                   'attempt_id': attempt_id, 'worker_id': None, 'lease_id': None,
                   'lease_expires_at': None, 'stop_required': False, 'cancel_requested': False,
                   'result': None, 'training_verified': False}
            if not any(item['job_id'] == job_id for item in trial['attempts']):
                trial['attempts'].append({'id': attempt_id, 'job_id': job_id, 'kind': kind})
            if kind == 'gpu_training_v1': trial['status'] = 'QUEUED'
            self.save('trials', trial)
            self.save('jobs', job)
            return job

    def enqueue_baseline(self,experiment_id):
        with self.store.lock:
            if not self.execution_enabled(): raise DataError('GPU 执行未启用；没有创建基线作业。')
            run=self.phase.checked_run(experiment_id)
            if run['model']['id']!='kev-0.8b-v1.0': raise DataError('仅支持固定 0.8B 基线。')
            if run.get('final_evaluation_job_id') or any((self.store.path('runs',experiment_id)/(side+'-predictions.json')).exists() for side in ('baseline','candidate')):
                raise DataError('最终 test 已冻结/导入，不能新建基线优化输入。')
            if self.phase.code_fingerprint()!=self.phase.code_sha256: raise DataError('代码已改变，请重启服务后创建基线。')
            if any(j['experiment_id']==experiment_id and (j['state'] not in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED') or j.get('promotion_pending')) for j in self.jobs()):
                raise LeaseConflict('实验中仍有作业或未确认停止/晋升的结果。')
            budget=self.budget(experiment_id)
            if budget['charged_seconds']>=budget['limit_seconds']: raise DataError('实验统一预算已耗尽。')
            config={'purpose':'baseline_only','execution_protocol_version':1,'model':run['model'],
                'code_sha256':self.phase.code_sha256,'data_manifest_sha256':run['manifest_sha256'],'limits':run['limits'],
                'resume_from':None,'evaluation_protocol':{'dtype':'fp32','backend':'torch','merge':False,'thresholds':'fixed_0_5'}}
            job={'id':str(uuid.uuid4()),'attempt_id':str(uuid.uuid4()),'experiment_id':experiment_id,'trial_id':None,
                'kind':'gpu_baseline_v1','state':'QUEUED','created_at':timestamp(),'worker_id':None,'lease_id':None,
                'lease_expires_at':None,'stop_required':False,'cancel_requested':False,'result':None,'training_verified':False,
                'config':config,'config_sha256':canonical_sha(config)}
            self.baseline_inputs(job)
            self.save('jobs',job)
            return job

    def baseline_inputs(self,job):
        run=self.phase.checked_run(job['experiment_id']);config=job['config']
        if (job['kind']!='gpu_baseline_v1' or canonical_sha(config)!=job['config_sha256']
                or config['data_manifest_sha256']!=run['manifest_sha256'] or config['limits']!=run['limits']
                or config['model']!=run['model']): raise DataError('冻结基线配置/数据清单不匹配。')
        rows={role:self.store.read(self.store.path('runs',run['id'])/(role+'.json')) for role in ('calibration','validation')}
        for role,cases in rows.items():
            if fingerprint(cases)!=run['manifest']['partitions'][role]['sha256']: raise DataError('基线分区已改变。')
            self.phase.assert_not_exposed(cases)
        return {'config':config,'config_sha256':job['config_sha256'],'objects':self.phase.task(run['task_spec_id'])['objects'],
                'partitions':rows,'partition_sha256':{r:fingerprint(v) for r,v in rows.items()}}

    def claim(self, worker_id):
        with self.store.lock:
            worker = self.read('workers', worker_id)
            worker['last_seen'] = self.clock()
            self.save('workers', worker)
            self.expire()
            jobs = self.jobs()
            # Single worker/process slot, including unconfirmed expired jobs.
            if any(job['state'] in ('RUNNING', 'CANCELLING', 'WAITING_FOR_STOP_CONFIRMATION') or job.get('promotion_pending') for job in jobs):
                return None
            pending = [job for job in reversed(jobs) if job['state'] == 'QUEUED' and job['kind'] in worker['capabilities']]
            if not pending:
                return None
            job = pending[0]
            job.update(state='RUNNING', worker_id=worker_id, lease_id=str(uuid.uuid4()),
                       lease_expires_at=self.clock() + self.ttl)
            ledger = self.ledger()
            if job['kind'].startswith('gpu_'):
                budget = self.budget(job['experiment_id'])
                remaining = max(0, budget['limit_seconds']-budget['charged_seconds'])
                if remaining <= 0:
                    job.update(state='FAILED', error='实验统一预算已用尽。')
                    self.save('jobs', job)
                    return None
                job.update(started_at=self.clock(), budget_seconds=remaining,
                           last_accounted_at=self.clock(),
                           execution_deadline=self.clock()+remaining, stage='VALIDATING')
                if job['kind']=='gpu_baseline_v1': job['stage']='BASELINE_VALIDATING'
                ledger['budgets'][job['experiment_id']] = {'charged_seconds': budget['charged_seconds'], 'reserved_job': job['id']}
                if job['kind'] == 'gpu_evaluation_v1':
                    job['stage'] = 'FINAL_VALIDATING'
                    # Reserve exposure before any test input leaves the controller.
                    run = self.store.read(self.store.path('runs', job['experiment_id'])/'meta.json')
                    cases = self.store.read(self.store.path('datasets', run['test_dataset']['id'])/'records.json')
                    path = self.store.root/'evaluation-usage.json'
                    usage = self.store.read(path) if path.is_file() else {}
                    usage['_exposed_groups'] = sorted(set(usage.get('_exposed_groups', [])) | exposure_keys(cases))
                    usage.setdefault(run['test_dataset']['sha256'], []).append({'experiment_id': run['id'],
                        'pair_sha256': job['id'], 'at': timestamp(), 'kind': 'frozen_final_evaluation'})
                    self.store.write(path, usage)
            ledger['jobs'][job['id']] = job
            self.commit(ledger)
            return job

    def owned(self, job_id, worker_id, lease_id, permit_expired=False):
        self.expire()
        job = self.read('jobs', job_id)
        if job['worker_id'] != worker_id or not isinstance(lease_id, str) or job['lease_id'] != lease_id:
            raise LeaseConflict('作业归属或租约不匹配。')
        allowed = ('RUNNING', 'CANCELLING') + (('WAITING_FOR_STOP_CONFIRMATION',) if permit_expired else ())
        if job['state'] not in allowed:
            raise LeaseConflict('租约已过期或作业已结束；必须先停止旧进程，不能续租或提交结果。')
        return job

    def heartbeat(self, job_id, worker_id, lease_id, stage=None):
        with self.store.lock:
            job = self.owned(job_id, worker_id, lease_id)
            if stage is not None:
                allowed = BASELINE_STAGES if job['kind']=='gpu_baseline_v1' else FINAL_STAGES if job['kind'] == 'gpu_evaluation_v1' else TRAIN_STAGES
                if not isinstance(stage, str) or stage not in allowed:
                    raise DataError('阶段不在执行协议内。')
                job['stage'] = stage
                if job['kind'] == 'gpu_training_v1':
                    trial = self.read('trials', job['trial_id'])
                    trial['status'] = stage
                    self.save('trials', trial)
            remaining = max(0, job.get('execution_deadline', self.clock()+86400)-self.clock())
            if job['kind'].startswith('gpu_') and self.clock() < job.get('last_accounted_at', job['started_at']):
                job['clock_rollback'] = True
                remaining = 0
            if job['kind'].startswith('gpu_'): job['last_accounted_at'] = max(self.clock(), job.get('last_accounted_at', job['started_at']))
            if job['kind'].startswith('gpu_') and remaining <= 0:
                job.update(cancel_requested=True, state='CANCELLING', error='统一时间预算已耗尽。', cancel_reason='budget_exhausted')
            job['lease_expires_at'] = self.clock() + self.ttl
            self.save('jobs', job)
            worker = self.read('workers', worker_id)
            worker['last_seen'] = self.clock()
            self.save('workers', worker)
            return {'cancel_requested': job['cancel_requested'], 'lease_seconds': self.ttl, 'remaining_seconds': remaining}

    def inputs(self, job_id, worker_id, lease_id):
        with self.store.lock:
            job = self.owned(job_id, worker_id, lease_id)
            if job['kind']=='gpu_baseline_v1': return self.baseline_inputs(job)
            if job['kind'] == 'gpu_evaluation_v1': return self.evaluation_inputs(job)
            return self.phase.worker_inputs(job['trial_id'])

    def enqueue_evaluation(self, experiment_id):
        with self.store.lock:
            if not self.execution_enabled(): raise DataError('GPU 执行未启用。')
            run_dir = self.store.path('runs', experiment_id)
            run = self.store.read(run_dir/'meta.json')
            if run.get('final_evaluation_job_id'): raise DataError('此实验已冻结一次最终评估，不能据 test 反复重选。')
            if any(job['experiment_id'] == experiment_id and (job['state'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED', 'EXPIRED') or job.get('promotion_pending'))
                   for job in self.jobs()): raise LeaseConflict('先等待实验中的训练/准备任务结束。')
            trial = self.read('trials', run.get('selected_trial_id'))
            active = trial.get('active_model')
            validation = self.store.read(self.phase.validation_path(trial))
            if (not active or (not active.get('training_verified') and not self.allow_fixture)
                    or validation.get('model_artifact_sha256') != active['sha256']
                    or validation.get('predictions_sha256') != run.get('selected_validation_sha256')):
                raise DataError('需要选定由执行器验证且有绑定 validation 的模型。')
            self.phase.assert_not_exposed(self.store.read(self.store.path('datasets', run['test_dataset']['id'])/'records.json'))
            budget = self.budget(experiment_id)
            if budget['charged_seconds'] >= budget['limit_seconds']: raise DataError('最终评估没有剩余统一预算。')
            job = {'id': str(uuid.uuid4()), 'trial_id': trial['id'], 'experiment_id': experiment_id,
                'kind': 'gpu_evaluation_v1', 'state': 'QUEUED', 'created_at': timestamp(), 'attempt_id': str(uuid.uuid4()),
                'worker_id': None, 'lease_id': None, 'lease_expires_at': None, 'stop_required': False,
                'cancel_requested': False, 'result': None, 'training_verified': False,
                'model_job_id': active['job_id'], 'model_artifact_sha256': active['sha256'],
                'selected_validation_sha256': run['selected_validation_sha256']}
            self.save('jobs', job)
            run['final_evaluation_job_id'] = job['id']
            self.store.write(run_dir/'meta.json', run)
            return job

    def evaluation_inputs(self, job):
        run = self.phase.checked_run(job['experiment_id'])
        trial = self.read('trials', job['trial_id'])
        if run['manifest_sha256'] != trial['config']['data_manifest_sha256']:
            raise DataError('最终评测数据清单与冻结模型配置不一致。')
        if (run.get('selected_trial_id') != trial['id'] or run.get('final_evaluation_job_id') != job['id']
                or run.get('selected_validation_sha256') != job['selected_validation_sha256']
                or trial.get('active_model', {}).get('sha256') != job['model_artifact_sha256']):
            raise LeaseConflict('最终评估选择或模型已经改变。')
        cases = self.store.read(self.store.path('datasets', run['test_dataset']['id'])/'records.json')
        if fingerprint(cases) != run['test_dataset']['sha256']: raise DataError('最终 test 内容哈希已改变。')
        frozen = self.store.read(self.artifact_root(job['model_job_id'], job['model_artifact_sha256'])/'verified.json')['manifest']['files']
        return {'config': trial['config'], 'config_sha256': trial['config_sha256'],
                'frozen_files': {name: frozen[name] for name in ('checkpoint/head.pt', 'checkpoint/adapter_model.safetensors',
                    'policy/baseline-calibration.json', 'policy/candidate-calibration.json')},
                'objects': self.phase.task(run['task_spec_id'])['objects'],
                'partitions': {'test': cases}, 'partition_sha256': {'test': run['test_dataset']['sha256']},
                'model_artifact_sha256': job['model_artifact_sha256'],
                'selected_validation_sha256': job['selected_validation_sha256']}

    def evaluation_model(self, job_id, worker_id, lease_id):
        with self.store.lock:
            job = self.owned(job_id, worker_id, lease_id)
            if job['kind'] != 'gpu_evaluation_v1': raise DataError('训练 Worker 不能请求最终评估模型/数据通道。')
            self.evaluation_inputs(job)
            path = self.artifact_root(job['model_job_id'], job['model_artifact_sha256'])/'bundle.zip'
            if file_sha(path) != job['model_artifact_sha256']: raise DataError('冻结模型包损坏。')
            return path

    def cancel(self, job_id):
        with self.store.lock:
            self.expire()
            job = self.read('jobs', job_id)
            if job['state'] == 'QUEUED':
                job.update(state='CANCELLED', cancel_requested=True)
                if job['kind'] == 'gpu_evaluation_v1':
                    run_dir = self.store.path('runs', job['experiment_id'])
                    run = self.store.read(run_dir/'meta.json')
                    if run.get('final_evaluation_job_id') == job_id:
                        run['final_evaluation_job_id'] = None
                        self.store.write(run_dir/'meta.json', run)
            elif job['state'] in ('RUNNING', 'CANCELLING', 'WAITING_FOR_STOP_CONFIRMATION'):
                job['cancel_requested'] = True
                job['cancel_reason'] = 'user_request'
                if job['state'] != 'WAITING_FOR_STOP_CONFIRMATION':
                    job['state'] = 'CANCELLING'
            self.save('jobs', job)
            return job

    def finish(self, job_id, worker_id, lease_id, body):
        fields(body, ('state', 'process_stopped', 'result'), ('state', 'process_stopped', 'result'))
        if body['process_stopped'] is not True:
            raise LeaseConflict('必须先确认本次进程已停止。')
        with self.store.lock:
            existing = self.read('jobs', job_id)
            request_sha = canonical_sha(body)
            if (existing.get('commit_request_sha256') == request_sha and existing['worker_id'] == worker_id
                    and existing['lease_id'] == lease_id and existing['state'] in ('SUCCEEDED', 'FAILED', 'CANCELLED')):
                self.reconcile_publish()
                return self.read('jobs', job_id)
            job = self.owned(job_id, worker_id, lease_id)
            if body['state'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
                raise DataError('无效的终态。')
            if job['cancel_requested'] and body['state'] != 'CANCELLED' and not (job.get('cancel_reason') == 'budget_exhausted' and body['state'] == 'FAILED'):
                raise LeaseConflict('已请求取消；不能报告成功。')
            if job['kind'].startswith('gpu_'):
                return self.finish_gpu(job, body, request_sha)
            if not isinstance(body['result'], dict) or body['result'].get('kind') != 'cpu_preflight_v1':
                raise DataError('这里只接收 CPU 准备检查结果，不接收训练成功或模型效果声明。')
            allowed = ('kind', 'counts', 'hashes_verified', 'full_object_labels_verified', 'blockers', 'log_tail', 'error')
            result = {**{key: value for key, value in body['result'].items() if key in allowed},
                      'training_verified': False, 'gpu_verified': False,
                      'token_length_verified': False, 'training_ready': False,
                      'source': 'worker_reported_preflight'}
            for name in ('log_tail', 'error'):
                if name in result:
                    result[name] = redact(str(result[name]))[-8000:]
            self.store.write(self.store.path('trials', job['trial_id']) / 'attempts' / job['attempt_id'] / 'result.json', result)
            job.update(state=body['state'], result=result, stopped_at=self.clock(), training_verified=False,
                       commit_request_sha256=request_sha)
            self.save('jobs', job)
            return job

    def artifact_root(self, job_id, sha):
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise DataError('产物 SHA 无效。')
        return self.store.path('jobs', job_id)/'artifacts'/sha

    def accept_artifact(self, job_id, worker_id, lease_id, uploaded):
        with self.store.lock:
            job = self.owned(job_id, worker_id, lease_id)
            if job['kind'] not in ('gpu_training_v1', 'gpu_evaluation_v1','gpu_baseline_v1'): raise DataError('此作业不接受产物包。')
            trial = job if job['kind']=='gpu_baseline_v1' else self.read('trials', job['trial_id'])
            inputs = self.baseline_inputs(job) if job['kind']=='gpu_baseline_v1' else self.evaluation_inputs(job) if job['kind'] == 'gpu_evaluation_v1' else self.phase.worker_inputs(trial['id'])
        sha = file_sha(uploaded)
        destination = self.artifact_root(job_id, sha)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.incoming-', dir=destination.parent))
        try:
            manifest = inspect_bundle(uploaded, {'job_id': job_id, 'trial_id': job['trial_id'],
                'attempt_id': job['attempt_id'], 'config_sha256': trial['config_sha256']}, temporary/'contents')
            validator = validate_baseline_bundle if job['kind']=='gpu_baseline_v1' else validate_evaluation_bundle if job['kind'] == 'gpu_evaluation_v1' else validate_training_bundle
            verified = validator(temporary/'contents', manifest, trial, inputs, self.allow_fixture)
            shutil.copyfile(uploaded, temporary/'bundle.zip')
            self.store.write(temporary/'verified.json', {'sha256': sha, 'manifest': manifest, 'verified': verified})
            with self.store.lock:
                latest = self.owned(job_id, worker_id, lease_id)
                if latest.get('artifact_sha256') not in (None, sha):
                    raise LeaseConflict('同一 attempt 不允许替换不同产物。')
                if destination.exists():
                    if file_sha(destination/'bundle.zip') != sha: raise DataError('既有产物损坏。')
                    if file_sha(destination/'verified.json') != file_sha(temporary/'verified.json'):
                        raise DataError('重复上传不能重签已改变的校验缓存。')
                else: os.replace(temporary, destination)
                latest['artifact_sha256'] = sha
                latest['verified_receipt_sha256'] = file_sha(destination/'verified.json')
                self.save('jobs', latest)
            return {'sha256': sha, 'verified': True, 'training_verified': verified['real']}
        finally:
            if temporary.exists(): shutil.rmtree(temporary)

    def charge(self, ledger, job):
        budget = ledger['budgets'].get(job['experiment_id'])
        if not budget or budget.get('reserved_job') != job['id']:
            raise LeaseConflict('GPU 预算预留不匹配。')
        elapsed = self.clock()-job['started_at']
        # A controller clock rollback must not make a repeated attempt free.
        charged = min(job['budget_seconds'], elapsed) if elapsed >= 0 and not job.get('clock_rollback') and self.clock() >= job.get('last_accounted_at', job['started_at']) else job['budget_seconds']
        budget['charged_seconds'] += charged
        budget['reserved_job'] = None
        job['charged_seconds'] = charged
        job['budget_policy'] = 'whole_job_wall_time_conservative_gpu_ceiling'

    def finish_gpu(self, job, body, request_sha):
        result = body['result']
        if not isinstance(result, dict) or result.get('kind') != job['kind']:
            raise DataError('执行结果 kind 不匹配。')
        real = False
        if body['state'] == 'SUCCEEDED':
            if self.clock() > job['execution_deadline']:
                raise LeaseConflict('完成请求超过统一预算；不能提交成功。')
            sha = result.get('artifact_sha256')
            if sha != job.get('artifact_sha256'): raise DataError('缺少已完整校验的产物。')
            artifact = self.accepted_artifact(job)
            if file_sha(self.artifact_root(job['id'], sha)/'bundle.zip') != sha:
                raise DataError('产物字节已改变。')
            real = artifact['verified']['real']
            registered = self.read('workers', job['worker_id'])['runtime']['execution_backend']
            if registered != artifact['manifest']['execution_backend']:
                raise DataError('产物后端与登记运行时不匹配。')
            expected_image=self.read('workers',job['worker_id'])['runtime'].get('image_provenance')
            if expected_image!=artifact['verified'].get('runtime',{}).get('image_provenance'):
                raise DataError('镜像来源在注册和作业执行之间发生变化。')
        ledger = self.ledger()
        self.charge(ledger, job)
        trained=real and job['kind']!='gpu_baseline_v1'
        job.update(state=body['state'], stopped_at=self.clock(), training_verified=trained,baseline_verified=real if job['kind']=='gpu_baseline_v1' else False,
                   commit_request_sha256=request_sha, promotion_pending=body['state'] == 'SUCCEEDED',
                   result={'kind': job['kind'], 'training_verified': trained,
                           'baseline_verified':real if job['kind']=='gpu_baseline_v1' else False,
                           'artifact_sha256': job.get('artifact_sha256'), 'error': redact(str(result.get('error', '')))[:2000]})
        ledger['jobs'][job['id']] = job
        self.commit(ledger)
        if body['state'] == 'SUCCEEDED':
            self.reconcile_publish()
        elif job['trial_id'] is not None:
            trial = self.read('trials', job['trial_id'])
            trial['status'] = body['state']
            # Keep any previously verified active_model pointer unchanged.
            self.save('trials', trial)
        return self.read('jobs', job['id'])

    def accepted_artifact(self, job):
        root = self.artifact_root(job['id'], job['artifact_sha256'])
        if file_sha(root/'verified.json') != job.get('verified_receipt_sha256'):
            raise DataError('已接受的校验回执已改变，禁止晋升。')
        artifact = self.store.read(root/'verified.json')
        if file_sha(root/'bundle.zip') != job['artifact_sha256']:
            raise DataError('已接受的模型包字节已改变。')
        if inspect_bundle(root/'bundle.zip') != artifact['manifest']:
            raise DataError('模型清单与已接受回执不一致。')
        return artifact

    def reconcile_publish(self):
        for job in reversed(self.jobs()):
            if job['state'] == 'SUCCEEDED' and job.get('promotion_pending'):
                try:
                    artifact = self.accepted_artifact(job)
                    result = (commit_baseline if job['kind']=='gpu_baseline_v1' else commit_evaluation if job['kind'] == 'gpu_evaluation_v1' else commit_training)(self.store, self.phase, job, artifact)
                    job.update(promotion_pending=False, result={**job['result'], **result})
                    self.save('jobs', job)
                except (OSError, DataError, KeyError) as error:
                    job['publication_error'] = redact(str(error))[:1000]
                    self.save('jobs', job)

    def stopped(self, job_id, worker_id, lease_id, process_stopped):
        if process_stopped is not True:
            raise LeaseConflict('未确认旧进程停止。')
        with self.store.lock:
            job = self.owned(job_id, worker_id, lease_id, permit_expired=True)
            if job['state'] != 'WAITING_FOR_STOP_CONFIRMATION':
                raise LeaseConflict('作业尚未处于租约过期的停止确认状态。')
            job.update(state='CANCELLED' if job['cancel_requested'] else 'EXPIRED',
                       stop_required=False, stopped_at=self.clock(), stopped_by=worker_id)
            if job['kind'].startswith('gpu_'):
                ledger = self.ledger()
                self.charge(ledger, job)
                ledger['jobs'][job['id']] = job
                self.commit(ledger)
            else: self.save('jobs', job)
            return job
