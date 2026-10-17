"""Pull worker: CPU preparation and explicitly configured isolated CUDA runtime."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from claude_runner import redact, stop_process
from execution import process_identity, stop_recorded_group
from trial_execution import runtime_environment


class WorkerConnectionError(RuntimeError):
    pass


class Client:
    def __init__(self, server, token):
        url = urlsplit(server)
        if not url.hostname or url.username or url.password or url.query or url.fragment or url.scheme not in ('http', 'https'):
            raise ValueError('Worker 服务地址无效。')
        if url.scheme == 'http' and url.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('非本机 Worker 地址必须使用 HTTPS。')
        self.server, self.token = server.rstrip('/'), token
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.opener = urllib.request.build_opener(NoRedirect)

    def call(self, path, body):
        request = urllib.request.Request(self.server + path, data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=5) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise WorkerConnectionError(f'Worker 请求失败 HTTP {error.code}；未续租时会停止本地进程。') from None
        except (urllib.error.URLError, TimeoutError):
            raise WorkerConnectionError('Worker 连接失败；未续租时会停止本地进程。') from None

    def upload(self, path, file, auth, timeout=30):
        file = Path(file)
        def chunks():
            with file.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''): yield block
        request = urllib.request.Request(self.server+path, data=chunks(), method='POST', headers={
            'Authorization': 'Bearer '+self.token, 'Content-Type': 'application/zip',
            'Content-Length': str(file.stat().st_size), 'X-Worker-Id': auth['worker_id'], 'X-Lease-Id': auth['lease_id']})
        try:
            with self.opener.open(request, timeout=max(.1, timeout)) as response: return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            raise WorkerConnectionError('模型产物上传失败；保留本地 attempt。') from None

    def download(self, path, target, auth, deadline):
        from model_artifacts import MAX_COMPRESSED
        request = urllib.request.Request(self.server+path, data=json.dumps(auth).encode(), headers={
            'Authorization': 'Bearer '+self.token, 'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=max(.1, min(5, deadline.remaining()))) as response, Path(target).open('xb') as output:
                size = 0
                while True:
                    deadline.check()
                    block = getattr(response, 'read1', response.read)(65536)
                    if not block: break
                    size += len(block)
                    if size > MAX_COMPRESSED: raise WorkerConnectionError('冻结模型包过大。')
                    output.write(block)
        except (urllib.error.URLError, TimeoutError):
            raise WorkerConnectionError('冻结模型下载失败。') from None


def run_cpu_job(client, worker_id, job, interval=2):
    auth = {'worker_id': worker_id, 'lease_id': job['lease_id']}
    prefix = '/api/worker/jobs/' + job['id']
    with tempfile.TemporaryDirectory(prefix='kev-worker-') as temporary:
        directory = Path(temporary)
        process = None
        state, result = 'FAILED', {'kind': 'cpu_preflight_v1', 'training_verified': False}
        lease_lost = False
        try:
            value = client.call(prefix + '/inputs', auth)
            source, output = directory / 'inputs.json', directory / 'result.json'
            source.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
            with (directory / 'process.log').open('wb') as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('worker_tasks.py')),
                    '--input', str(source), '--output', str(output)], cwd=directory,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=runtime_environment())
                (directory/'active-process.json').write_text(json.dumps({'pid':process.pid,
                    'identity':process_identity(process.pid),'uid':os.getuid()}))
                next_heartbeat, deadline = 0, time.monotonic() + 120
                cancelled = False
                while process.poll() is None:
                    if time.monotonic() >= next_heartbeat:
                        try:
                            heartbeat = client.call(prefix + '/heartbeat', auth)
                        except Exception:
                            lease_lost = True
                            raise
                        if heartbeat['cancel_requested']:
                            cancelled = True
                            break
                        next_heartbeat = time.monotonic() + interval
                    if time.monotonic() > deadline:
                        raise RuntimeError('CPU 准备检查超时。')
                    time.sleep(.02)
                stop_process(process)
                # Check cancellation even if the short-lived process exited
                # between heartbeats; a cancelled job cannot report success.
                try:
                    heartbeat = client.call(prefix + '/heartbeat', auth)
                except Exception:
                    lease_lost = True
                    raise
                cancelled = cancelled or heartbeat['cancel_requested']
                if cancelled:
                    state = 'CANCELLED'
                elif process.returncode == 0 and output.is_file():
                    result, state = json.loads(output.read_text()), 'SUCCEEDED'
        except Exception as error:
            lease_lost = lease_lost or isinstance(error, WorkerConnectionError)
            result['error'] = type(error).__name__ + ': Worker 准备检查失败。'
        finally:
            if process is not None:
                stop_process(process)
                if not stop_recorded_group(directory):
                    return {'state':'HELD_FOR_PROCESS_CONFIRMATION','training_verified':False}
            log_path = directory / 'process.log'
            if log_path.is_file():
                with log_path.open('rb') as handle:
                    handle.seek(max(0, log_path.stat().st_size - 24000))
                    result['log_tail'] = redact(handle.read().decode('utf-8', errors='replace'))[-8000:]
        if lease_lost:
            try:
                client.call(prefix + '/stopped', {**auth, 'process_stopped': True})
            except Exception:
                pass  # durable queue remains held; never silently re-dispatch
            return {'state': 'STOPPED_AFTER_LEASE_LOSS', 'training_verified': False}
        try:
            return client.call(prefix + '/finish', {**auth, 'state': state, 'process_stopped': True, 'result': result})
        except WorkerConnectionError:
            # Cancellation may arrive after the last heartbeat but before the
            # terminal commit. The process is already stopped; acknowledge the
            # cancellation instead of leaving a successful-but-rejected result.
            try:
                heartbeat = client.call(prefix + '/heartbeat', auth)
                if heartbeat['cancel_requested']:
                    return client.call(prefix + '/finish', {**auth, 'state': 'CANCELLED',
                        'process_stopped': True, 'result': result})
            except WorkerConnectionError:
                try:
                    client.call(prefix + '/stopped', {**auth, 'process_stopped': True})
                except WorkerConnectionError:
                    pass
            return {'state': 'STOPPED_WITH_UNACKNOWLEDGED_RESULT', 'training_verified': False}


def run_once(client, worker_id, interval=2, backend=None, work_root=None):
    started = time.monotonic()
    job = client.call('/api/worker/claim', {'worker_id': worker_id})['job']
    if not job: return None
    if job['kind'] == 'cpu_preflight_v1': return run_cpu_job(client, worker_id, job, interval)
    from worker_execution import run_gpu_job
    return run_gpu_job(client, worker_id, job, backend, work_root, interval,
                       claim_elapsed=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description='仅 CPU 准备检查，不执行训练')
    parser.add_argument('--server', default='http://127.0.0.1:8091')
    parser.add_argument('--name', default='local-preflight')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--role', choices=('cpu', 'training', 'evaluation'), default='cpu')
    parser.add_argument('--kev-source')
    parser.add_argument('--kev-python')
    parser.add_argument('--checkpoint-archive')
    parser.add_argument('--work-root')
    args = parser.parse_args()
    token = os.environ.get('KEV_EVALUATOR_TOKEN' if args.role == 'evaluation' else 'KEV_WORKER_TOKEN')
    if not token:
        raise SystemExit('请在 API 和 Worker 进程环境中配置同一 KEV_WORKER_TOKEN；本工具不创建或保存凭据。')
    client = Client(args.server, token)
    registration = {'name': args.name, 'platform': platform.system(), 'python_version': platform.python_version()}
    backend = None
    identity_file = None
    if args.role != 'cpu':
        if not all((args.kev_source, args.kev_python, args.checkpoint_archive, args.work_root)):
            raise SystemExit('GPU Worker 需要 --kev-source/--kev-python/--checkpoint-archive/--work-root；不自动安装或下载。')
        from trial_execution import RuntimeBackend
        from worker_execution import recover_attempts
        import uuid, hashlib
        root = Path(args.work_root)
        root.mkdir(parents=True, exist_ok=True)
        identity_file = root/('worker-'+hashlib.sha256((client.server+args.role).encode()).hexdigest()[:16]+'.json')
        if identity_file.is_file():
            identity = json.loads(identity_file.read_text())
            previous = client.call('/api/worker/resume', {'worker_id': identity['worker_id']})
            held = recover_attempts(client, previous['id'], root)
            if held: raise SystemExit('旧进程归属尚未确认，未领取新作业：'+', '.join(held))
            registration['worker_id'] = previous['id']
        backend = RuntimeBackend(args.kev_source, args.kev_python, args.checkpoint_archive)
        registration['runtime'] = backend.probe(root/('probe-'+str(uuid.uuid4())), args.role)
        registration['capabilities'] = ['gpu_evaluation_v1'] if args.role == 'evaluation' else ['cpu_preflight_v1', 'gpu_training_v1','gpu_baseline_v1']
    worker = client.call('/api/worker/register', registration)
    if identity_file: identity_file.write_text(json.dumps({'worker_id': worker['id'], 'server': client.server, 'role': args.role}))
    while True:
        result = run_once(client, worker['id'], backend=backend, work_root=args.work_root)
        print(json.dumps(result or {'job': None, 'training_verified': False}), flush=True)
        if args.once:
            return
        time.sleep(2)


if __name__ == '__main__':
    main()
