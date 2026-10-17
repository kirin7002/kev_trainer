"""Pull-worker GPU orchestration. No GPU code is imported in this process."""
import json
from pathlib import Path
import threading
import time
import uuid

from claude_runner import redact
from execution import Deadline, ExecutionCancelled, ExecutionTimeout, process_identity, stop_recorded_group
import os
from trial_execution import TrialPipeline
from final_evaluation import FinalEvaluationPipeline
from baseline_execution import BaselinePipeline


class LeaseKeeper:
    def __init__(self, client, prefix, auth, deadline, interval):
        self.client, self.prefix, self.auth = client, prefix, auth
        self.deadline, self.interval = deadline, interval
        self.stage, self.cancelled, self.lost = 'VALIDATING', False, False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def beat(self):
        value = self.client.call(self.prefix+'/heartbeat', {**self.auth, 'stage': self.stage})
        self.cancelled = self.cancelled or value['cancel_requested']
        self.deadline.ends = min(self.deadline.ends, self.deadline.clock()+value.get('remaining_seconds', self.deadline.remaining()))

    def run(self):
        while not self.stop_event.is_set():
            try: self.beat()
            except Exception:
                self.lost = True
                return
            self.stop_event.wait(self.interval)

    def check(self, stage):
        if self.lost: raise RuntimeError('Worker 已失去租约连接。')
        if self.stage != stage:
            self.stage = stage
            self.beat()  # Persist every transition and check cancel before launching its process.
        return {'cancel_requested': self.cancelled}

    def start(self):
        self.beat()  # Never launch a process before a valid lease heartbeat.
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None: self.thread.join(timeout=6)


def run_gpu_job(client, worker_id, job, backend, work_root, interval=2, claim_elapsed=0):
    auth = {'worker_id': worker_id, 'lease_id': job['lease_id']}
    prefix = '/api/worker/jobs/'+job['id']
    deadline = Deadline(max(0, job['budget_seconds']-claim_elapsed))
    keeper = LeaseKeeper(client, prefix, auth, deadline, interval)
    if job['kind'] == 'gpu_evaluation_v1': keeper.stage = 'FINAL_VALIDATING'
    if job['kind'] == 'gpu_baseline_v1': keeper.stage = 'BASELINE_VALIDATING'
    state, result = 'FAILED', {'kind': job['kind'], 'training_verified': False}
    directory = None
    try:
        if backend is None or work_root is None:
            raise RuntimeError('GPU Worker 未配置固定 runtime 与独立 work root。')
        for value in (job['id'], job['attempt_id']):
            if str(uuid.UUID(value)) != value: raise RuntimeError('作业 ID 无效。')
        directory = Path(work_root)/('attempt-'+job['id']+'-'+job['attempt_id'])
        if directory.exists():
            return {'state': 'HELD_FOR_PROCESS_CONFIRMATION', 'training_verified': False,
                    'attempt_directory': str(directory)}
        keeper.start()
        inputs = client.call(prefix+'/inputs', auth)
        deadline.check()
        pipeline_type = BaselinePipeline if job['kind']=='gpu_baseline_v1' else FinalEvaluationPipeline if job['kind'] == 'gpu_evaluation_v1' else TrialPipeline
        pipeline = pipeline_type(backend, directory, keeper.check, deadline.remaining(), interval=interval)
        pipeline.supervisor.deadline = deadline  # Same deadline covers download, all stages, ZIP and upload.
        (directory/'lease.json').write_text(json.dumps({**auth, 'job_id': job['id'], 'kind': job['kind'],
            'owner_pid': os.getpid(), 'owner_identity': process_identity(os.getpid())}))
        if job['kind'] == 'gpu_evaluation_v1':
            model = directory/'frozen-model.zip'
            client.download(prefix+'/model', model, auth, deadline)
            result = pipeline.run(inputs, job, model)
        else: result = pipeline.run(inputs, job)
        deadline.check()
        uploaded = client.upload(prefix+'/artifacts', result['artifact_path'], auth, timeout=deadline.remaining())
        if uploaded['sha256'] != result['artifact_sha256']: raise RuntimeError('上传产物哈希不一致。')
        deadline.check()
        result.pop('artifact_path', None)
        state = 'SUCCEEDED'
    except ExecutionCancelled as error:
        state, result = 'CANCELLED', {'kind': job['kind'], 'error': str(error), 'training_verified': False}
    except Exception as error:
        result = {'kind': job['kind'], 'error': redact(type(error).__name__+': '+str(error))[:2000], 'training_verified': False}
        if keeper.cancelled and deadline.remaining() > 0: state = 'CANCELLED'
    finally:
        keeper.stop()
    if directory is not None and not stop_recorded_group(directory):
        return {'state': 'HELD_FOR_PROCESS_CONFIRMATION', 'training_verified': False,
                'attempt_directory': str(directory)}
    # Supervisor always stops its child/group before this acknowledgement.
    body = {**auth, 'state': state, 'process_stopped': True, 'result': result}
    try:
        response = client.call(prefix+'/finish', body)
    except Exception:
        try:
            heartbeat = client.call(prefix+'/heartbeat', auth)
            body['state'] = 'CANCELLED' if heartbeat['cancel_requested'] and deadline.remaining() > 0 else 'FAILED'
            response = client.call(prefix+'/finish', body)
        except Exception:
            try: response = client.call(prefix+'/stopped', {**auth, 'process_stopped': True})
            except Exception:
                return {'state': 'STOPPED_WITH_UNACKNOWLEDGED_RESULT', 'training_verified': False,
                        'attempt_directory': str(directory) if directory else None}
    if directory is not None: (directory/'lease.json').unlink(missing_ok=True)
    return response


def recover_attempts(client, worker_id, work_root):
    """No optimizer resume. Stop owned orphan sessions before releasing a lease."""
    held = []
    for directory in Path(work_root).glob('attempt-*'):
        lease = directory/'lease.json'
        if not lease.is_file(): continue
        try:
            value = json.loads(lease.read_text())
            if value['worker_id'] != worker_id: continue
            if process_identity(value['owner_pid']) == value['owner_identity']:
                held.append(directory.name); continue
            if not stop_recorded_group(directory): held.append(directory.name); continue
            auth = {key: value[key] for key in ('worker_id', 'lease_id')}
            prefix = '/api/worker/jobs/'+value['job_id']
            if client.call(prefix+'/recovery-status', auth)['terminal']:
                lease.unlink()
                continue
            try:
                heartbeat = client.call(prefix+'/heartbeat', auth)
                client.call(prefix+'/finish', {**auth, 'process_stopped': True,
                    'state': 'CANCELLED' if heartbeat['cancel_requested'] else 'FAILED',
                    'result': {'kind': value['kind'], 'error': 'Worker 重启，旧进程已停止；LoRA 不做 optimizer resume。'}})
            except Exception:
                client.call(prefix+'/stopped', {**auth, 'process_stopped': True})
            lease.unlink()
        except Exception: held.append(directory.name)
    return held
