"""Original-design phase-one contracts. Synthetic 50-object fixtures, never real training."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from datasets import DataError, assert_disjoint, normalize, split_validation
from worker import run_once, Client, WorkerConnectionError
from worker_queue import LeaseConflict, WorkerQueue


@pytest.fixture()
def task50(client):
    body = {'name': '合成 50 对象契约测试，非实际车载目录', 'objects': [
        {'id': f'object-{i:02}', 'instructions': f'合成对象 {i}', 'criteria': {'true': '合成正例', 'false': '合成负例'}} for i in range(50)]}
    response = client.post('/api/task-specs', json=body)
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def partitioned(client, task50):
    versions = {}
    raw_files = {}
    for role in ('train', 'calibration', 'validation', 'test', 'regression'):
        rows = [{'id': role + str(i), 'text': role + ' unique phrase ' + str(i),
                 'intents': ['object-00'] if i == 0 else [], 'conversation_id': role + '-session-' + str(i)}
                for i in range(2)]
        raw = json.dumps(rows, ensure_ascii=False, indent=4).encode()
        response = client.post('/api/datasets', data={'role': role, 'task_spec_id': task50['id'],
            'source_rule': 'Synthetic fixture, not production data'}, files={'file': (role + '.json', raw)})
        assert response.status_code == 201, response.text
        versions[role], raw_files[role] = response.json(), raw
    return versions, raw_files


@pytest.fixture()
def manual(client, task50, partitioned):
    versions, _ = partitioned
    body = {'name': '手动准备测试', 'task_spec_id': task50['id'], 'max_trials': 2,
            'max_training_minutes': 30, 'partitions': {role: meta['id'] for role, meta in versions.items()}}
    response = client.post('/api/manual-experiments', json=body)
    assert response.status_code == 201, response.text
    return response.json()


PARAMETERS = {'learning_rate': 0.0001, 'epochs': 2, 'batch_size': 1,
              'gradient_accumulation': 1, 'precision': 'bf16', 'seed': 42}


def new_trial(client, manual, params=None):
    response = client.post('/api/experiments/' + manual['id'] + '/trials', json=params or PARAMETERS)
    assert response.status_code == 201, response.text
    return response.json()


def test_original_files_and_full_system_one_are_versioned(client, app_instance, task50, partitioned):
    versions, raw_files = partitioned
    for role, meta in versions.items():
        directory = app_instance.state.store.path('datasets', meta['id'])
        assert (directory / 'original.bin').read_bytes() == raw_files[role]
        assert meta['original_sha256'] == hashlib.sha256(raw_files[role]).hexdigest()
        converted = json.loads((directory / 'system-one.json').read_text())
        assert len(converted[0]['questions']) == 50
        assert converted[0]['questions']['object-00']['label'] is True
        assert converted[1]['questions']['object-00']['label'] is False
        assert converted[0]['conversation_id'] == role + '-session-0'
        assert meta['profile']['by_object_count'] == {'0': 1, '1': 1, '2': 0, '3': 0, '4+': 0}
        for name, digest in meta['files'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('objects', [[], [{'id': 'x', 'instructions': 'x', 'criteria': {}}] * 50])
def test_task_requires_explicit_unique_complete_catalog(client, objects):
    assert client.post('/api/task-specs', json={'name': 'bad', 'objects': objects}).status_code == 400


@pytest.mark.parametrize('row', [
    {'text': 'unknown object', 'intents': ['not-in-task']},
    {'text': 'missing explicit false', 'questions': {'object-00': {'type': 'noul', 'label': True}}},
])
def test_task_dataset_rejects_unknown_and_partial_noul_labels(client, task50, row):
    response = client.post('/api/datasets', data={'role': 'train', 'task_spec_id': task50['id']},
                           files={'file': ('bad.json', json.dumps([row]))})
    assert response.status_code == 400
    assert client.get('/api/datasets').json() == []


def test_near_surface_duplicates_and_all_group_fields_do_not_cross_partitions():
    train = normalize([{'text': '打开车窗', 'intents': ['window'], 'conversation_id': 'session-1'}])
    for test in ([{'text': '打开车窗！', 'intents': ['window']}],
                 [{'text': '完全另一句', 'intents': [], 'conversation_id': 'session-1'}]):
        with pytest.raises(DataError):
            assert_disjoint(train, normalize(test))
    rows = normalize([{'text': f'text{i}', 'intents': [], 'template_id': 'a' if i < 3 else 'b'} for i in range(6)])
    a, b = split_validation(rows)
    assert_disjoint(a, b)
    assert len(a) == len(b) == 3


def test_manual_requires_all_four_partitions_and_rejects_cross_partition_near_duplicates(client, task50, partitioned):
    versions, _ = partitioned
    body = {'name': 'manual', 'task_spec_id': task50['id'], 'max_trials': 1, 'max_training_minutes': 30,
            'partitions': {role: meta['id'] for role, meta in versions.items()}}
    incomplete = {**body, 'partitions': {'train': versions['train']['id'], 'test': versions['test']['id']}}
    assert client.post('/api/manual-experiments', json=incomplete).status_code == 400
    bad = client.post('/api/datasets', data={'role': 'validation', 'task_spec_id': task50['id']},
        files={'file': ('near.json', json.dumps([{'text': 'train unique phrase 0！', 'intents': ['object-00']}]))}).json()
    body['partitions']['validation'] = bad['id']
    assert client.post('/api/manual-experiments', json=body).status_code == 400


def test_trial_is_immutable_budgeted_and_no_rank_command_or_resume_override(client, manual, app_instance):
    trial = new_trial(client, manual)
    assert trial['status'] == 'WAITING_FOR_USER'
    assert trial['training_verified'] is False
    assert trial['config']['parent_checkpoint'] == manual['model']['checkpoint']
    assert trial['config']['resume_from'] is None
    directory = app_instance.state.store.path('trials', trial['id'])
    before = (directory / 'config.json').read_bytes()
    for extra in ({'lora_rank': 32}, {'command': ['sh', '-c', 'echo unsafe']}, {'resume_from': 'adapter.bin'}):
        assert client.post('/api/experiments/' + manual['id'] + '/trials', json={**PARAMETERS, **extra}).status_code == 400
    new_trial(client, manual, {**PARAMETERS, 'seed': 43})
    assert client.post('/api/experiments/' + manual['id'] + '/trials', json=PARAMETERS).status_code == 400
    assert (directory / 'config.json').read_bytes() == before


def test_worker_inputs_exclude_test_and_regression_and_detect_tampering(client, manual, app_instance, partitioned):
    trial = new_trial(client, manual)
    phase = app_instance.state.phase_one
    value = phase.worker_inputs(trial['id'])
    assert set(value['partitions']) == {'train', 'calibration', 'validation'}
    encoded = json.dumps(value)
    for role in ('test', 'regression'):
        assert partitioned[0][role]['id'] not in encoded
        assert partitioned[0][role]['sha256'] not in encoded
        assert role + ' unique phrase' not in encoded
    config_path = app_instance.state.store.path('trials', trial['id']) / 'config.json'
    config_path.write_text('{}')
    with pytest.raises(DataError):
        phase.worker_inputs(trial['id'])


def registered(queue, name='worker'):
    return queue.register({'name': name, 'platform': 'fixture-linux', 'python_version': '3.12'})['id']


def test_pull_queue_expiry_requires_old_process_stop_and_fences_results(client, manual, app_instance):
    queue = app_instance.state.worker_queue
    clock = [100.0]
    queue.clock = lambda: clock[0]
    trial1, trial2 = new_trial(client, manual), new_trial(client, manual)
    job1 = queue.enqueue(trial1['id'])
    queue.enqueue(trial2['id'])
    w1, w2 = registered(queue), registered(queue, 'other')
    claimed = queue.claim(w1)
    assert claimed['id'] == job1['id']
    assert queue.claim(w2) is None
    clock[0] += 31
    assert queue.claim(w2) is None
    assert queue.read('jobs', job1['id'])['state'] == 'WAITING_FOR_STOP_CONFIRMATION'
    with pytest.raises(LeaseConflict):
        queue.heartbeat(job1['id'], w1, claimed['lease_id'])
    with pytest.raises(LeaseConflict):
        queue.enqueue(trial1['id'])
    with pytest.raises(LeaseConflict):
        queue.stopped(job1['id'], w2, claimed['lease_id'], True)
    queue.stopped(job1['id'], w1, claimed['lease_id'], True)
    assert queue.claim(w2)['trial_id'] == trial2['id']
    retry = queue.enqueue(trial1['id'])
    assert retry['attempt_id'] != job1['attempt_id']
    with pytest.raises(LeaseConflict):
        queue.finish(job1['id'], w1, claimed['lease_id'], {'state': 'SUCCEEDED',
            'process_stopped': True, 'result': {'kind': 'cpu_preflight_v1'}})


def test_cancel_is_not_terminal_before_process_stop_ack(client, manual, app_instance):
    queue = app_instance.state.worker_queue
    job = queue.enqueue(new_trial(client, manual)['id'])
    worker = registered(queue)
    lease = queue.claim(worker)['lease_id']
    assert queue.cancel(job['id'])['state'] == 'CANCELLING'
    assert queue.heartbeat(job['id'], worker, lease)['cancel_requested'] is True
    with pytest.raises(LeaseConflict):
        queue.finish(job['id'], worker, lease, {'state': 'CANCELLED', 'process_stopped': False,
                                               'result': {'kind': 'cpu_preflight_v1'}})
    assert queue.finish(job['id'], worker, lease, {'state': 'CANCELLED', 'process_stopped': True,
        'result': {'kind': 'cpu_preflight_v1', 'training_verified': True}})['training_verified'] is False


def test_worker_auth_is_independent_and_not_stored(client, app_instance, monkeypatch):
    body = {'name': 'worker', 'platform': 'linux', 'python_version': '3.12'}
    monkeypatch.delenv('KEV_WORKER_TOKEN', raising=False)
    assert client.post('/api/worker/register', json=body).status_code == 503
    token = 'temporary-fixture-not-a-real-key-' + 'x' * 32
    monkeypatch.setenv('KEV_WORKER_TOKEN', token)
    assert client.post('/api/worker/register', json=body).status_code == 401
    assert client.post('/api/worker/register', json=body, headers={'Authorization': 'Bearer ' + token}).status_code == 200
    for path in app_instance.state.store.root.rglob('*.json'):
        assert token not in path.read_text()


class ApiClient:
    def __init__(self, client, token):
        self.client, self.headers = client, {'Authorization': 'Bearer ' + token}

    def call(self, path, body):
        response = self.client.post(path, json=body, headers=self.headers)
        assert response.status_code == 200, response.text
        return response.json()


def test_reference_pull_worker_runs_real_cpu_process_without_training(client, manual, app_instance, monkeypatch):
    token = 'fixture-worker-token-' + 'x' * 32
    monkeypatch.setenv('KEV_WORKER_TOKEN', token)
    api = ApiClient(client, token)
    worker = api.call('/api/worker/register', {'name': 'cpu-contract', 'platform': 'linux', 'python_version': '3.12'})
    trial = new_trial(client, manual)
    job = client.post('/api/trials/' + trial['id'] + '/preflight').json()
    result = run_once(api, worker['id'], interval=.01)
    assert result['id'] == job['id'] and result['state'] == 'SUCCEEDED'
    assert result['result']['hashes_verified'] is True
    assert result['result']['training_ready'] is False
    assert result['training_verified'] is False
    assert not list(app_instance.state.store.root.rglob('*.safetensors'))
    assert client.get('/api/health').json()['machine']['connected'] is False


@pytest.mark.parametrize('address', ['http://worker.example', 'http://user:secret@127.0.0.1', 'file:///tmp/a'])
def test_worker_rejects_insecure_or_embedded_credentials_address(address):
    with pytest.raises(ValueError):
        Client(address, 'fixture-only-token')


def test_preparation_zip_is_deterministic_hashed_and_never_a_model_or_test_bundle(client, manual, partitioned):
    trial = new_trial(client, manual)
    first = client.get('/api/trials/' + trial['id'] + '/preparation.zip')
    second = client.get('/api/trials/' + trial['id'] + '/preparation.zip')
    assert first.status_code == 200 and first.content == second.content
    with zipfile.ZipFile(io.BytesIO(first.content)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['is_model'] is manifest['training_verified'] is False
        assert set(archive.namelist()) == {'config.json', 'kev-source-contract.json', 'worker-inputs.json', 'train.jsonl',
                                         'calibration.jsonl', 'validation.jsonl', 'manifest.json'}
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert b'test unique phrase' not in archive.read(name)
            assert partitioned[0]['test']['sha256'].encode() not in archive.read(name)
        row = json.loads(archive.read('train.jsonl').splitlines()[0])
        assert len(row['questions']) == 50
    assert client.post('/api/trials/' + trial['id'] + '/train').status_code == 409
    assert client.post('/api/experiments/' + manual['id'] + '/plan').status_code == 409


def validation_payload(manual, trial, role='validation'):
    return {'dataset_sha256': manual['manifest']['partitions'][role]['sha256'], 'trial_id': trial['id'],
            'predictions': [{'id': role + str(i), 'intents': ['object-00'] if i == 0 else [],
                            'probabilities': {f'object-{obj:02}': 1.0 if i == obj == 0 else 0.0 for obj in range(50)}}
                            for i in range(2)]}


def test_validation_selection_precedes_final_test_and_test_reuse_is_regression(client, manual):
    trial = new_trial(client, manual)
    base = '/api/experiments/' + manual['id']
    test = validation_payload(manual, trial, 'test')
    assert client.post(base + '/predictions/candidate', files={'file': ('p.json', json.dumps(test))}).status_code == 409
    imported = client.post('/api/trials/' + trial['id'] + '/validation-predictions',
                           files={'file': ('v.json', json.dumps(validation_payload(manual, trial)))})
    assert imported.status_code == 200, imported.text
    metrics = imported.json()['metrics']['overall']['probability_calibration']
    assert metrics['brier_score'] == metrics['ece'] == 0
    assert len(imported.json()['metrics']['overall']['per_object']) == 50
    assert client.post(base + '/select-trial', json={'trial_id': trial['id']}).status_code == 200
    for side in ('baseline', 'candidate'):
        response = client.post(base + '/predictions/' + side, files={'file': ('p.json', json.dumps(test))})
        assert response.status_code == 200, response.text
    first = client.get(base + '/report').json()
    assert first['selected_trial_id'] == trial['id']
    assert first['test_usage']['effective_role'] == 'test'
    assert first['test_usage']['independent_final_acceptance'] is False
    assert first['paired_accuracy_uncertainty']['uncertain'] is True
    assert client.post(base + '/select-trial', json={'trial_id': trial['id']}).status_code == 400
    assert client.post('/api/trials/' + trial['id'] + '/validation-predictions',
                      files={'file': ('v.json', json.dumps(validation_payload(manual, trial)))}).status_code == 400
    test['predictions'][0]['intents'] = []
    assert client.post(base + '/predictions/candidate', files={'file': ('p.json', json.dumps(test))}).status_code == 200
    second = client.get(base + '/report').json()
    assert second['test_usage']['effective_role'] == 'regression'
    assert second['test_usage']['paired_evaluations'] == 2
    # Renaming IDs does not turn an exposed final test into fresh training.
    renamed = [{'id': 'renamed-' + str(i), 'text': 'test unique phrase ' + str(i), 'intents': []} for i in range(2)]
    task_id = manual['task_spec_id']
    for role in ('train', 'calibration', 'validation', 'test'):
        assert client.post('/api/datasets', data={'role': role, 'task_spec_id': task_id},
                           files={'file': ('renamed.json', json.dumps(renamed))}).status_code == 400
    assert client.post('/api/datasets', data={'role': 'regression', 'task_spec_id': task_id},
                       files={'file': ('regression.json', json.dumps(renamed))}).status_code == 201


@pytest.mark.parametrize('mode', ['cancel', 'expired'])
def test_reference_worker_stops_real_process_on_cancel_or_lost_lease(client, manual, app_instance, monkeypatch, mode):
    import worker as worker_module
    token = 'temporary-fixture-worker-' + 'x' * 32
    monkeypatch.setenv('KEV_WORKER_TOKEN', token)
    queue = app_instance.state.worker_queue
    clock = [100.0]
    queue.clock = lambda: clock[0]
    job = queue.enqueue(new_trial(client, manual)['id'])
    processes = []
    original_spawn = subprocess.Popen

    def slow_cpu_process(argv, **kwargs):
        if 'worker_tasks.py' not in ' '.join(map(str,argv)):
            return original_spawn(argv, **kwargs)
        process = original_spawn([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(worker_module.subprocess, 'Popen', slow_cpu_process)

    class CancellingClient(ApiClient):
        def call(self, path, body):
            if path.endswith('/heartbeat'):
                if mode == 'cancel':
                    queue.cancel(job['id'])
                else:
                    clock[0] += 31
            return super().call(path, body)

    api = CancellingClient(client, token)
    worker = api.call('/api/worker/register', {'name': 'stop-fixture', 'platform': 'linux', 'python_version': '3.12'})
    result = run_once(api, worker['id'], interval=.01)
    assert result['state'] == ('CANCELLED' if mode == 'cancel' else 'STOPPED_AFTER_LEASE_LOSS')
    assert processes and all(process.poll() is not None for process in processes)
    assert queue.read('jobs', job['id'])['state'] == ('CANCELLED' if mode == 'cancel' else 'EXPIRED')


def test_upstream_metadata_identity_is_preserved_and_test_cannot_be_relabelled_train(client, task50):
    row = {'state': 'fixture heldout', 'intents': ['object-00'],
           '_meta': {'id': 'stable-upstream-id', 'group_id': 'stable-group', 'source': 'fixture', 'split': 'test'}}
    normalized = normalize([row])[0]
    assert normalized['id'] == 'stable-upstream-id' and normalized['group_id'] == 'stable-group'
    assert normalized['source_metadata'] == row['_meta']
    response = client.post('/api/datasets', data={'role': 'train', 'task_spec_id': task50['id']},
                           files={'file': ('bad.json', json.dumps([row]))})
    assert response.status_code == 400


@pytest.mark.parametrize('change', ['missing-object', 'out-of-range', 'partial-coverage'])
def test_full_probability_metrics_require_all_objects_and_all_samples(client, manual, change):
    trial = new_trial(client, manual)
    payload = validation_payload(manual, trial)
    if change == 'missing-object': payload['predictions'][0]['probabilities'].pop('object-49')
    if change == 'out-of-range': payload['predictions'][0]['probabilities']['object-49'] = 1.1
    if change == 'partial-coverage': payload['predictions'][1].pop('probabilities')
    response = client.post('/api/trials/' + trial['id'] + '/validation-predictions',
                           files={'file': ('v.json', json.dumps(payload))})
    if change == 'partial-coverage':
        assert response.status_code == 200
        metrics = response.json()['metrics']['overall']['probability_calibration']
        assert metrics['coverage'] == 1 and metrics['brier_score'] is None and metrics['ece'] is None
    else:
        assert response.status_code == 400


def test_queue_restart_preserves_unconfirmed_owner_and_public_state_hides_nonce(client, manual, app_instance):
    queue = app_instance.state.worker_queue
    queue.clock = lambda: 100
    job = queue.enqueue(new_trial(client, manual)['id'])
    worker = registered(queue)
    claimed = queue.claim(worker)
    restarted = WorkerQueue(queue.store, queue.phase, clock=lambda: 131)
    assert restarted.claim(worker) is None
    assert restarted.read('jobs', job['id'])['state'] == 'WAITING_FOR_STOP_CONFIRMATION'
    assert claimed['lease_id'] not in json.dumps(client.get('/api/preparation').json())


def test_worker_acknowledges_cancel_racing_with_terminal_result(client, manual, app_instance, monkeypatch):
    token = 'race-fixture-worker-' + 'x' * 32
    monkeypatch.setenv('KEV_WORKER_TOKEN', token)
    queue = app_instance.state.worker_queue
    job = queue.enqueue(new_trial(client, manual)['id'])

    class RacingClient(ApiClient):
        def call(self, path, body):
            if path.endswith('/finish') and body['state'] == 'SUCCEEDED':
                queue.cancel(job['id'])
                response = self.client.post(path, json=body, headers=self.headers)
                assert response.status_code == 409
                raise WorkerConnectionError('fixture cancellation race')
            return super().call(path, body)

    api = RacingClient(client, token)
    worker = api.call('/api/worker/register', {'name': 'race', 'platform': 'linux', 'python_version': '3.12'})
    assert run_once(api, worker['id'], interval=.01)['state'] == 'CANCELLED'
    assert queue.read('jobs', job['id'])['state'] == 'CANCELLED'


@pytest.mark.parametrize('value', [[], {}, None, True, 123])
def test_manual_partition_ids_reject_bad_types_without_500(client, task50, partitioned, value):
    partitions = {role: item['id'] for role, item in partitioned[0].items()}
    partitions['train'] = value
    response = client.post('/api/manual-experiments', json={'name': 'invalid', 'task_spec_id': task50['id'],
        'partitions': partitions, 'max_trials': 1, 'max_training_minutes': 5})
    assert response.status_code == 400


def test_frozen_noul_conversion_does_not_inherit_soft_target_override(client, task50, app_instance):
    questions = {obj['id']: {'type': 'noul', 'label': False, 'target': {'true': 1, 'false': 0},
                             'instructions': 'unreviewed override'} for obj in task50['objects']}
    response = client.post('/api/datasets', data={'role': 'train', 'task_spec_id': task50['id']},
        files={'file': ('original.json', json.dumps([{'state': 'fixture', 'questions': questions}]))})
    assert response.status_code == 201
    directory = app_instance.state.store.path('datasets', response.json()['id'])
    converted = json.loads((directory / 'system-one.json').read_text())[0]
    assert 'target' not in converted['questions']['object-00']
    assert converted['questions']['object-00']['label'] is False
    assert converted['questions']['object-00']['instructions'] == task50['objects'][0]['instructions']
    assert json.loads((directory / 'original.bin').read_text())[0]['questions'] == questions


def test_training_pool_split_is_seeded_group_safe_and_retains_parent_original(client, task50, app_instance):
    rows = [{'id': str(i), 'text': f'pool example {i}', 'intents': ['object-00'],
             'conversation_id': 'conversation-' + str(i//2)} for i in range(12)]
    raw = json.dumps(rows).encode()
    parent = client.post('/api/datasets', data={'role': 'train', 'task_spec_id': task50['id']},
                         files={'file': ('pool.json', raw)}).json()
    body = {'calibration_fraction': .2, 'validation_fraction': .2, 'seed': 42}
    first = client.post('/api/datasets/' + parent['id'] + '/split', json=body)
    second = client.post('/api/datasets/' + parent['id'] + '/split', json=body)
    assert first.status_code == second.status_code == 201
    assert {role: item['sha256'] for role, item in first.json().items()} == {role: item['sha256'] for role, item in second.json().items()}
    outputs = {}
    for role, item in first.json().items():
        directory = app_instance.state.store.path('datasets', item['id'])
        outputs[role] = json.loads((directory / 'records.json').read_text())
        assert item['parent_dataset_id'] == parent['id']
        assert (directory / 'original.bin').read_bytes() == raw
        assert item['original_sha256'] == parent['original_sha256']
    assert sum(map(len, outputs.values())) == len(rows)
    for a, b in (('train', 'calibration'), ('train', 'validation'), ('calibration', 'validation')):
        assert_disjoint(outputs[a], outputs[b])


@pytest.mark.parametrize('role', ['test', 'regression', 'calibration'])
def test_heldout_or_calibration_versions_cannot_be_split_into_training(client, task50, partitioned, role):
    item_id = partitioned[0][role]['id']
    response = client.post('/api/datasets/' + item_id + '/split', json={
        'calibration_fraction': .2, 'validation_fraction': .2, 'seed': 42})
    assert response.status_code == 400


def test_strict_final_test_preview_withholds_cases_before_frozen_acceptance(client, partitioned):
    response = client.get('/api/datasets/' + partitioned[0]['test']['id'])
    assert response.status_code == 200
    assert response.json()['preview'] == []
    assert response.json()['preview_policy'] == 'withheld_final_test'
    assert 'test unique phrase' not in response.text
    assert response.json()['profile']['total'] == 2


def test_trial_creation_refuses_source_changed_since_service_start(client, manual, app_instance, tmp_path):
    from phase_one import PhaseOne
    from app import MODEL_MAP
    code = tmp_path / 'isolated-code'
    code.mkdir()
    (code / 'app.py').write_text('# startup fixture\n')
    phase = PhaseOne(app_instance.state.store, MODEL_MAP, code)
    first = phase.create_trial(manual['id'], PARAMETERS)
    assert 'app.py' in first['config']['code_sha256']
    (code / 'app.py').write_text('# changed after startup\n')
    with pytest.raises(DataError):
        phase.create_trial(manual['id'], PARAMETERS)
