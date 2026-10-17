"""Software-only execution acceptance; synthetic predictions are not GPU evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest

from datasets import DataError
from execution import Deadline, Supervisor, ExecutionTimeout, ExecutionCancelled, StageFailed, process_identity, stop_recorded_group
from fake_gpu_backend import FakeBackend
from model_artifacts import build_bundle, inspect_bundle, file_sha
from pipeline_validation import TRAIN_STAGES, training_metrics
from final_evaluation import FINAL_STAGES
from test_phase_one import task50, partitioned, manual, new_trial
from worker import run_once, WorkerConnectionError
from worker_execution import recover_attempts
from worker_queue import LeaseConflict
from pipeline_validation import calibration_rows, convert_benchmark
from trial_execution import TrialPipeline
from pipeline_commit import validate_training_bundle, validate_evaluation_bundle


class API:
    def __init__(self, client, token): self.client, self.token = client, token
    def checked(self, response):
        if response.status_code >= 400: raise WorkerConnectionError(str(response.status_code)+' '+response.text)
        return response
    def call(self, path, body):
        return self.checked(self.client.post(path, json=body, headers={'Authorization': 'Bearer '+self.token})).json()
    def upload(self, path, file, auth, timeout=None):
        return self.checked(self.client.post(path, content=Path(file).read_bytes(), headers={
            'Authorization': 'Bearer '+self.token, 'X-Worker-Id': auth['worker_id'], 'X-Lease-Id': auth['lease_id']})).json()
    def download(self, path, file, auth, deadline):
        response = self.checked(self.client.post(path, json=auth, headers={'Authorization': 'Bearer '+self.token}))
        Path(file).write_bytes(response.content)


@pytest.fixture()
def gpu(client, app_instance, monkeypatch):
    monkeypatch.setenv('KEV_WORKER_TOKEN', 't'*40)
    monkeypatch.setenv('KEV_EVALUATOR_TOKEN', 'e'*40)
    queue = app_instance.state.worker_queue
    queue.allow_fixture, queue.execution_enabled = True, lambda: True
    clients = [API(client, token*40) for token in ('t', 'e')]
    workers = [api.call('/api/worker/register', {'name': 'synthetic-'+role, 'platform': 'fixture',
        'python_version': '3.12', 'runtime': FakeBackend.runtime(), 'capabilities': [cap]})
        for api, role, cap in zip(clients, ('training', 'evaluation'), ('gpu_training_v1', 'gpu_evaluation_v1'))]
    return queue, clients, workers


def test_all_training_and_final_stages_auth_isolation_and_provenance(client, manual, app_instance, gpu, tmp_path):
    queue, (trainer, evaluator), (tw, ew) = gpu
    trial = new_trial(client, manual)
    response = client.post('/api/trials/'+trial['id']+'/train')
    assert response.status_code == 202, response.text
    backend = FakeBackend()
    result = run_once(trainer, tw['id'], backend=backend, work_root=tmp_path/'worker', interval=.1)
    assert result['state'] == 'SUCCEEDED', result
    assert result['training_verified'] is False
    assert [c[0] for c in backend.calls] == list(TRAIN_STAGES[:-1])
    active = queue.read('trials', trial['id'])['active_model']
    assert active['training_verified'] is False and not active['loader_verified']
    assert client.get('/api/trials/'+trial['id']+'/model.zip').status_code == 409
    assert queue.budget(manual['id'])['charged_seconds'] > 0
    assert client.post('/api/experiments/'+manual['id']+'/select-trial', json={'trial_id': trial['id']}).status_code == 200
    response = client.post('/api/experiments/'+manual['id']+'/evaluate')
    assert response.status_code == 202, response.text
    assert trainer.call('/api/worker/claim', {'worker_id': tw['id']})['job'] is None
    with pytest.raises(WorkerConnectionError, match='403'):
        trainer.call('/api/worker/claim', {'worker_id': ew['id']})
    final_backend = FakeBackend()
    result = run_once(evaluator, ew['id'], backend=final_backend, work_root=tmp_path/'evaluator', interval=.1)
    assert result['state'] == 'SUCCEEDED', result
    assert [c[0] for c in final_backend.calls] == list(FINAL_STAGES[:-1])
    assert all(c[1] not in ('train', 'calibrate', 'validate') for c in final_backend.calls)
    report = client.get('/api/experiments/'+manual['id']+'/report').json()
    assert report['source'] == 'software_contract_fixture' and report['training_verified'] is False
    assert report['test_fitted'] is False
    assert client.post('/api/experiments/'+manual['id']+'/evaluate').status_code == 400
    assert client.post('/api/trials/'+trial['id']+'/train').status_code == 400


@pytest.mark.parametrize('failure', ['SMOKE', 'TRAINING', 'EVALUATING', 'LOADING'])
def test_stage_failure_stops_and_never_promotes(client, manual, gpu, tmp_path, failure):
    queue, (trainer, _), (tw, _) = gpu
    trial = new_trial(client, manual)
    queue.enqueue(trial['id'], 'gpu_training_v1')
    backend = FakeBackend(fail=failure)
    result = run_once(trainer, tw['id'], backend=backend, work_root=tmp_path/'worker', interval=.1)
    assert result['state'] == 'FAILED', result
    assert backend.calls[-1][0] == failure
    assert not queue.read('trials', trial['id']).get('active_model')
    assert queue.budget(manual['id'])['charged_seconds'] > 0


def test_existing_model_survives_failed_attempt_and_budget_accumulates(client, manual, gpu, tmp_path):
    queue, (trainer, _), (tw, _) = gpu
    trial = new_trial(client, manual)
    queue.enqueue(trial['id'], 'gpu_training_v1')
    assert run_once(trainer, tw['id'], backend=FakeBackend(), work_root=tmp_path/'worker', interval=.1)['state'] == 'SUCCEEDED'
    previous = queue.read('trials', trial['id'])['active_model']
    charged = queue.budget(manual['id'])['charged_seconds']
    queue.enqueue(trial['id'], 'gpu_training_v1')
    assert run_once(trainer, tw['id'], backend=FakeBackend(fail='SMOKE'), work_root=tmp_path/'worker', interval=.1)['state'] == 'FAILED'
    assert queue.read('trials', trial['id'])['active_model'] == previous
    assert queue.budget(manual['id'])['charged_seconds'] > charged


def test_final_bundle_cannot_replace_frozen_policy_or_loader_hash(client, manual, gpu, tmp_path):
    from final_evaluation import FinalEvaluationPipeline
    from trial_execution import read, write
    queue, (trainer, evaluator), (tw, ew) = gpu
    trial = new_trial(client, manual)
    queue.enqueue(trial['id'], 'gpu_training_v1')
    assert run_once(trainer, tw['id'], backend=FakeBackend(), work_root=tmp_path/'worker', interval=.1)['state'] == 'SUCCEEDED'
    queue.phase.select_trial(manual['id'], trial['id'])
    queue.enqueue_evaluation(manual['id']); job = queue.claim(ew['id'])
    inputs = queue.inputs(job['id'], ew['id'], job['lease_id'])
    assert set(inputs['partitions']) == {'test'}
    bundle = queue.evaluation_model(job['id'], ew['id'], job['lease_id'])
    result = FinalEvaluationPipeline(FakeBackend(), tmp_path/'final', lambda _: {}, 30, interval=.01).run(inputs, job, bundle)
    manifest = inspect_bundle(result['artifact_path'], destination=tmp_path/'extracted')
    root = tmp_path/'extracted'
    assert validate_evaluation_bundle(root, manifest, trial, inputs, True)['real'] is False
    path = root/'reports/FINAL_LOADING.json'; receipt = read(path)
    receipt['head_sha256'] = 'f'*64; write(path, receipt)
    with pytest.raises(DataError): validate_evaluation_bundle(root, manifest, trial, inputs, True)
    receipt['head_sha256'] = inputs['frozen_files']['checkpoint/head.pt']['sha256']; write(path, receipt)
    policy = root/'policy/candidate-calibration.json'; value = read(policy); value['temperature'] = 2.; write(policy, value)
    restore = root/'reports/FINAL_RESTORING.json'; value = read(restore)
    value['policy_sha256']['candidate'] = file_sha(policy); write(restore, value)
    with pytest.raises(DataError): validate_evaluation_bundle(root, manifest, trial, inputs, True)


def test_token_observation_and_runtime_environment_exclude_credentials():
    from kev_runtime import token_observation
    from trial_execution import runtime_environment
    observation = token_observation({'_meta':{'id':'example'}, 'questions':{'a':{},'b':{}}},
        {'seg':[0,0,1,1,1,2,2], 'ids':list(range(7)), 'state_tokens':2, 'state_truncated':False})
    assert observation == {'id':'example', 'state_tokens':2, 'state_truncated':False, 'packed_tokens':7,
                           'question_branch_tokens':{'a':3,'b':2}}
    assert not {'KEV_WORKER_TOKEN', 'KEV_EVALUATOR_TOKEN', 'ANTHROPIC_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'HF_TOKEN'} & runtime_environment().keys()


def test_shared_deadline_cancels_owned_child_and_rejects_path_escape(tmp_path):
    deadline = Deadline(.3)
    supervisor = Supervisor(tmp_path, deadline, lambda _: {}, interval=.01)
    supervisor.run('FIRST', [sys.executable, '-c', 'import time;time.sleep(.12)'], cwd=tmp_path, environment={})
    with pytest.raises(ExecutionTimeout):
        supervisor.run('SECOND', [sys.executable, '-c', 'import time;time.sleep(60)'], cwd=tmp_path, environment={})
    assert deadline.remaining() == 0 and not (tmp_path/'active-process.json').exists()
    with pytest.raises(StageFailed): supervisor.run('../unsafe', ['true'], cwd=tmp_path, environment={})


def test_real_process_group_cancel_kills_descendant(tmp_path):
    childfile = tmp_path/'child'
    def heartbeat(_): return {'cancel_requested': childfile.exists()}
    supervisor = Supervisor(tmp_path, Deadline(10), heartbeat, interval=.01)
    command = 'import subprocess,time,pathlib; p=subprocess.Popen(["'+sys.executable+'","-c","import time;time.sleep(60)"]);pathlib.Path("child").write_text(str(p.pid));time.sleep(60)'
    with pytest.raises(ExecutionCancelled): supervisor.run('CANCEL', [sys.executable, '-c', command], cwd=tmp_path, environment={})
    pid = int(childfile.read_text())
    stat = Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'


@pytest.mark.parametrize('name', ['../outside.json', '/outside.json', 'checkpoint/../bad.json', 'checkpoint/.secret.json', 'checkpoint/a.exe'])
def test_unsafe_artifact_paths_never_extract(tmp_path, name):
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr(name, '{}'); output.writestr('manifest.json', '{}')
    with pytest.raises(DataError): inspect_bundle(archive, destination=tmp_path/'extracted')
    assert not (tmp_path/'extracted').exists()


def test_bundle_hash_tampering_duplicate_entries_and_immutability(tmp_path):
    root = tmp_path/'package'; (root/'reports').mkdir(parents=True)
    (root/'reports/test.json').write_text('{}')
    (root/'checkpoint').mkdir()
    (root/'checkpoint/chat_template.jinja').write_text('fixture tokenizer template')
    output = tmp_path/'bundle.zip'
    manifest = build_bundle(root, output, {'kind': 'fixture'})
    assert inspect_bundle(output) == manifest
    with pytest.raises(DataError): build_bundle(root, output, {})
    for duplicate in (False, True):
        bad = tmp_path/('bad'+str(duplicate)+'.zip')
        with zipfile.ZipFile(bad, 'w') as archive:
            archive.writestr('reports/test.json', 'tampered')
            archive.writestr('manifest.json', json.dumps(manifest))
            if duplicate:
                with pytest.warns(UserWarning): archive.writestr('reports/test.json', '{}')
        with pytest.raises(DataError): inspect_bundle(bad)


def test_recovery_never_kills_reused_or_uncertain_pid(tmp_path):
    journal = tmp_path/'active-process.json'
    journal.write_text(json.dumps({'pid': os.getpid(), 'identity': 'other-process', 'uid': os.getuid()}))
    assert stop_recorded_group(tmp_path) is False
    journal.write_text('{"state":"launching","pid":null}')
    assert stop_recorded_group(tmp_path) is False


def test_gpu_execution_is_explicit_and_fixture_registration_disabled_by_default(client, manual, app_instance, monkeypatch):
    trial = new_trial(client, manual)
    assert client.post('/api/trials/'+trial['id']+'/train').status_code == 409
    monkeypatch.setenv('KEV_ENABLE_GPU_EXECUTION', '1')
    with pytest.raises(DataError):
        app_instance.state.worker_queue.register({'name':'fixture', 'platform':'linux', 'python_version':'3.12',
            'capabilities':['gpu_training_v1'], 'runtime': FakeBackend.runtime()})


def test_gpu_wire_stream_upload_and_worker_reconnect(client, manual, gpu, app_instance, tmp_path):
    import socket
    import threading
    import uvicorn
    from worker import Client
    queue, _, (tw, _) = gpu
    trial = new_trial(client, manual)
    queue.enqueue(trial['id'], 'gpu_training_v1')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); sock.listen()
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app_instance, log_level='error', lifespan='off'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        thread.start()
        try:
            until = time.monotonic()+5
            while not server.started and time.monotonic() < until: time.sleep(.01)
            assert server.started
            wire = Client('http://127.0.0.1:'+str(port), 't'*40)
            assert wire.call('/api/worker/resume', {'worker_id':tw['id']})['id'] == tw['id']
            result = run_once(wire, tw['id'], backend=FakeBackend(), work_root=tmp_path/'wire', interval=.1)
            assert result['state'] == 'SUCCEEDED', result
            refreshed = wire.call('/api/worker/register', {'worker_id':tw['id'], 'name':'reconnected', 'platform':'fixture',
                'python_version':'3.12', 'capabilities':['gpu_training_v1'], 'runtime':FakeBackend.runtime()})
            assert refreshed['id'] == tw['id']
        finally:
            server.should_exit = True; thread.join(5)
            assert not thread.is_alive()


@pytest.mark.parametrize('mutation', ['missing-loss', 'bad-label', 'bad-threshold', 'bad-loader', 'partial-coverage'])
def test_controller_rechecks_complete_bundle_independently(client, manual, gpu, tmp_path, mutation):
    from trial_execution import read, write
    queue, _, (tw, _) = gpu
    trial = new_trial(client, manual)
    job = queue.enqueue(trial['id'], 'gpu_training_v1'); job = queue.claim(tw['id'])
    inputs = queue.inputs(job['id'], tw['id'], job['lease_id'])
    directory = tmp_path/'pipeline'
    result = TrialPipeline(FakeBackend(), directory, lambda _: {}, 30, interval=.01).run(inputs, job)
    manifest = inspect_bundle(result['artifact_path'], destination=tmp_path/'extracted')
    root = tmp_path/'extracted'
    if mutation == 'missing-loss': (root/'logs/TRAINING-loss.jsonl').write_text('')
    elif mutation == 'bad-label':
        path = root/'reports/benchmark-CANDIDATE_CALIBRATION/rows.json'
        rows = read(path); rows[0]['label'] = 1-rows[0]['label']; write(path, rows)
        path = root/'reports/CALIBRATING.json'; policy = read(path)
        policy['rows_sha256'] = file_sha(root/'reports/benchmark-CANDIDATE_CALIBRATION/rows.json'); write(path, policy)
    elif mutation == 'bad-threshold':
        path = root/'reports/CALIBRATING.json'; policy = read(path); policy['thresholds']['object-00'] = .3; write(path, policy)
    elif mutation == 'bad-loader':
        path = root/'reports/LOADING.json'; receipt = read(path); receipt['head_sha256'] = '0'*64; write(path, receipt)
    else:
        path = root/'reports/benchmark-EVALUATING/report.json'; report = read(path); report['coverage']['evaluated_questions'] -= 1; write(path, report)
    with pytest.raises(DataError): validate_training_bundle(root, manifest, trial, inputs, True)
    assert not queue.read('trials', trial['id']).get('active_model')


def test_expired_gpu_budget_held_until_stop_and_stale_artifact_rejected(client, manual, gpu, tmp_path):
    queue, _, (tw, _) = gpu
    now = [100.]; queue.clock = lambda: now[0]
    trial = new_trial(client, manual); queue.enqueue(trial['id'], 'gpu_training_v1')
    job = queue.claim(tw['id'])
    assert job['budget_seconds'] == 1800
    with pytest.raises(LeaseConflict):
        queue.register({'worker_id':tw['id'], 'name':'refresh', 'platform':'fixture', 'python_version':'3.12'})
    now[0] += 31
    assert queue.claim(tw['id']) is None
    with pytest.raises(LeaseConflict): queue.accept_artifact(job['id'], tw['id'], job['lease_id'], tmp_path/'absent')
    queue.stopped(job['id'], tw['id'], job['lease_id'], True)
    assert queue.budget(manual['id'])['charged_seconds'] == 31
    queue.enqueue(trial['id'], 'gpu_training_v1')
    later = queue.claim(tw['id'])
    assert later['budget_seconds'] == 1769
    with pytest.raises(LeaseConflict): queue.inputs(job['id'], tw['id'], job['lease_id'])
    now[0] += 1769
    queue.stopped(later['id'], tw['id'], later['lease_id'], True)
    assert queue.budget(manual['id'])['charged_seconds'] == 1800
    with pytest.raises(DataError): queue.enqueue(trial['id'], 'gpu_training_v1')


def test_cancel_between_stages_never_publishes_and_recovery_terminal_is_idempotent(client, manual, gpu, tmp_path):
    queue, (trainer, _), (tw, _) = gpu
    trial = new_trial(client, manual); queued = queue.enqueue(trial['id'], 'gpu_training_v1')
    class CancelBackend(FakeBackend):
        def perform(self, stage, operation, payload, supervisor):
            result = super().perform(stage, operation, payload, supervisor)
            if stage == 'SMOKE': queue.cancel(queued['id']); time.sleep(.06)
            return result
    result = run_once(trainer, tw['id'], backend=CancelBackend(), work_root=tmp_path/'worker', interval=.01)
    assert result['state'] == 'CANCELLED', result
    assert not queue.read('trials', trial['id']).get('active_model')
    directory = next((tmp_path/'worker').glob('attempt-*'))
    job = queue.read('jobs', queued['id'])
    (directory/'lease.json').write_text(json.dumps({'worker_id': tw['id'], 'lease_id':job['lease_id'],
        'job_id':job['id'], 'kind':job['kind'], 'owner_pid':99999999, 'owner_identity':'dead'}))
    assert recover_attempts(trainer, tw['id'], tmp_path/'worker') == []
    assert not (directory/'lease.json').exists()


def test_runtime_loss_observer_preserves_tensor_and_rejects_dropped_records(tmp_path, monkeypatch):
    import types
    import kev_runtime
    work = tmp_path
    (work/'logs').mkdir(); (work/'data').mkdir(); (work/'data/train-expanded.jsonl').write_text('{}\n')
    class Loss:
        def detach(self): return self
        def cpu(self): return self
        def item(self): return 1.
    loss = Loss()
    original_loss = lambda *a, **k: (loss, {'ce':1.})
    original_requests = lambda *a, **k: [{}]
    train = types.SimpleNamespace(batch_loss=original_loss, training_requests=original_requests)
    def main():
        assert train.batch_loss()[0] is loss
        assert len(train.training_requests()) == 1
        (work/'smoke').mkdir(); (work/'smoke/training_metrics.json').write_text('{}')
    train.main = main
    monkeypatch.setitem(sys.modules, 'kev', types.SimpleNamespace(train=train))
    model = {'base':'fixture', 'base_revision':'fixed', 'lora_rank':16, 'head_dim':256}
    params = {'learning_rate':.0001, 'epochs':1, 'batch_size':1, 'gradient_accumulation':1,'seed':42,'precision':'fp32'}
    receipt = kev_runtime.observed_train({'work':str(work), 'config':{'model':model,'parameters':params},'stage':'SMOKE'})
    assert json.loads((work/receipt['loss_file']).read_text())['loss'] == 1.
    assert train.batch_loss is original_loss and train.training_requests is original_requests
    train.training_requests = lambda *a, **k: []
    with pytest.raises(ValueError, match='dropped'):
        kev_runtime.observed_train({'work':str(work), 'config':{'model':model,'parameters':params},'stage':'TRAINING'})


def test_runtime_watchdog_independently_stops_private_session(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = 'import sys,time;sys.path.insert(0,'+repr(str(root))+');from kev_runtime import watchdog;watchdog(.15);time.sleep(60)'
    process = subprocess.Popen([sys.executable, '-I', '-c', code], start_new_session=True)
    try: assert process.wait(timeout=5) == -15
    finally:
        if process.poll() is None: process.kill(); process.wait()
