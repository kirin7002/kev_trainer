"""Business and variable-question contracts on the existing execution pipeline.

All GPU paths below use the explicit software-only fixture. No model is loaded.
"""
import io
import json
from contextlib import nullcontext
import sys
import types
import zipfile

import pytest

from datasets import DataError, expanded_training_count, normalize
from evaluation import evaluate, validate_predictions
from phase_one import sha, system_one
from pipeline_validation import calibration_rows, convert_benchmark
from test_phase_one import new_trial
from test_gpu_execution import gpu
from test_baseline_and_image import baseline_gpu
from fake_gpu_backend import FakeBackend
from worker import run_once


def task_body(count=3, mode='complete', scope='car'):
    return {'name': 'synthetic business task', 'business_scope': scope, 'label_mode': mode,
        'objects': [{'id': f'q{i}', 'instructions': f'Business rule {i}',
                     'criteria': {'true': 'yes', 'false': 'no'}} for i in range(count)]}


@pytest.fixture()
def business_factory(client):
    def create(count=3, mode='complete', scope='car', mixed=False):
        response = client.post('/api/task-specs', json=task_body(count, mode, scope))
        assert response.status_code == 201, response.text
        task = response.json()
        versions, rows_by_role = {}, {}
        for role in ('train', 'calibration', 'validation', 'test'):
            rows = []
            for i in range(2):
                row = {'id': role + str(i), 'text': f'{task["id"]} {role} utterance {i}'}
                if mixed:
                    row['scenario'] = 'car' if i == 0 else 'app'
                if mode == 'explicit_questions':
                    keys = ['q0'] if i == 0 else [f'q{count-1}']
                    row['questions'] = {key: {'type': 'noul', 'label': i == 0} for key in keys}
                else:
                    row['intents'] = ['q0'] if i == 0 else []
                rows.append(row)
            response = client.post('/api/datasets', data={'role': role, 'task_spec_id': task['id']},
                files={'file': (role + '.json', json.dumps(rows), 'application/json')})
            assert response.status_code == 201, response.text
            versions[role], rows_by_role[role] = response.json(), rows
        response = client.post('/api/manual-experiments', json={'name': 'business experiment',
            'task_spec_id': task['id'], 'partitions': {key: value['id'] for key, value in versions.items()},
            'max_trials': 2, 'max_training_minutes': 30})
        assert response.status_code == 201, response.text
        return task, response.json(), versions, rows_by_role
    return create


@pytest.mark.parametrize('count', [1, 3, 7, 50, 51, 100])
def test_catalog_size_and_export_use_actual_questions(client, business_factory, count):
    task, run, _, _ = business_factory(count)
    trial = new_trial(client, run)
    response = client.get('/api/trials/' + trial['id'] + '/preparation.zip')
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        row = json.loads(bundle.read('train.jsonl').splitlines()[0])
        assert len(row['questions']) == count
        assert row['state']['scenario'] == 'car'
        assert b'test utterance' not in bundle.read('worker-inputs.json')
    assert trial['config']['task_policy'] == {'business_scope': 'car', 'label_mode': 'complete'}
    assert run['business_scope'] == 'car'


@pytest.mark.parametrize('count', [0, 101])
def test_invalid_catalog_counts_are_rejected(client, count):
    assert client.post('/api/task-specs', json=task_body(count)).status_code == 400


@pytest.mark.parametrize('mode', ['partial', None, {}, 123])
def test_unknown_annotation_modes_are_rejected(client, mode):
    assert client.post('/api/task-specs', json=task_body(mode=mode)).status_code == 400


def test_scope_and_mode_are_immutable_and_do_not_reuse_other_business_data(client, app_instance):
    car = client.post('/api/task-specs', json=task_body()).json()
    app = client.post('/api/task-specs', json=task_body(scope='app')).json()
    assert car['sha256'] != app['sha256']
    response = client.post('/api/datasets', data={'role': 'train', 'task_spec_id': car['id']},
        files={'file': ('bad.json', json.dumps([{'text': 'open window', 'scenario': 'app', 'intents': ['q0']}]))})
    assert response.status_code == 400
    store = app_instance.state.store
    path = store.path('task-specs', car['id']) / 'meta.json'
    changed = store.read(path)
    changed['label_mode'] = 'explicit_questions'
    store.write(path, changed)
    assert client.get('/api/task-specs/' + car['id']).status_code == 400


def test_task_overrides_keep_strict_json_validation(client):
    response = client.post('/api/task-specs?business_scope=app&label_mode=explicit_questions', json=task_body())
    assert response.status_code == 201
    assert response.json()['business_scope'] == 'app'
    assert response.json()['label_mode'] == 'explicit_questions'
    response = client.post('/api/task-specs?business_scope=app',
        content='{"name":"duplicate","name":"bad","objects":[]}', headers={'Content-Type': 'application/json'})
    assert response.status_code == 400


def test_sparse_noul_is_opt_in_and_never_fills_missing_negatives(client, business_factory, app_instance):
    task, run, _, _ = business_factory(7, 'explicit_questions')
    trial = new_trial(client, run)
    inputs = app_instance.state.phase_one.worker_inputs(trial['id'])
    assert expanded_training_count(inputs['partitions']['train'], [o['id'] for o in task['objects']]) == 2
    with zipfile.ZipFile(io.BytesIO(client.get('/api/trials/' + trial['id'] + '/preparation.zip').content)) as bundle:
        rows = [json.loads(line) for line in bundle.read('train.jsonl').splitlines()]
        assert set(rows[0]['questions']) == {'q0'}
        assert rows[0]['questions']['q0']['label'] is True
        assert set(rows[1]['questions']) == {'q6'}
        assert rows[1]['questions']['q6']['label'] is False
    complete = client.post('/api/task-specs', json=task_body(7)).json()
    for rows in ([{'text': 'array is ambiguous', 'intents': []}],
                 [{'text': 'missing boolean', 'questions': {'q0': {'type': 'noul'}}}],
                 [{'text': 'unknown', 'questions': {'unknown-q': {'type': 'noul', 'label': False}}}]):
        response = client.post('/api/datasets', data={'role': 'train', 'task_spec_id': task['id']},
            files={'file': ('bad.json', json.dumps(rows))})
        assert response.status_code == 400
    response = client.post('/api/datasets', data={'role': 'train', 'task_spec_id': complete['id']},
        files={'file': ('bad.json', json.dumps([{'text': 'partial is not implicit',
            'questions': {'q0': {'type': 'noul', 'label': False}}}]))})
    assert response.status_code == 400


def sparse_cases():
    return normalize([
        {'id': 'a', 'text': 'car request', 'scenario': 'car',
         'questions': {'q0': {'type': 'noul', 'label': True}}},
        {'id': 'b', 'text': 'app request', 'scenario': 'app',
         'questions': {'q2': {'type': 'noul', 'label': False}}},
    ])


def test_sparse_probability_metrics_only_count_answered_questions():
    cases = sparse_cases()
    value = {'dataset_sha256': 'sha', 'predictions': [
        {'id': 'a', 'intents': ['q0'], 'probabilities': {'q0': 1.}},
        {'id': 'b', 'intents': [], 'probabilities': {'q2': 0.}},
    ]}
    predictions = validate_predictions(value, cases, 'sha', ['q0', 'q1', 'q2'])
    result = evaluate(cases, predictions)
    assert result['overall']['accuracy'] == 1
    assert result['overall']['annotated_questions'] == 2
    assert result['overall']['probability_calibration']['brier_score'] == 0
    assert result['annotation_scope'] == 'explicit_questions_only'
    assert result['by_scenario']['car']['total'] == 1
    assert result['by_scenario']['app']['annotated_questions'] == 1
    assert result['cases'][0]['asked_questions'] == ['q0']
    for wrong in ({'q0': 1., 'q1': 0.}, {}):
        value['predictions'][0]['probabilities'] = wrong
        with pytest.raises(DataError):
            validate_predictions(value, cases, 'sha', ['q0', 'q1', 'q2'])
    value['predictions'][0]['probabilities'] = {'q0': 1.}
    value['predictions'][0]['intents'] = ['q1']
    with pytest.raises(DataError):
        validate_predictions(value, cases, 'sha', ['q0', 'q1', 'q2'])


def test_sparse_benchmark_and_calibration_require_exact_question_coverage():
    cases = sparse_cases()
    lines, rows = [], []
    for case in cases:
        key = next(iter(case['questions']))
        label = int(case['questions'][key]['label'])
        lines.append({'id': case['id'], 'prediction': {'probabilities': {key: {'false': 1-label, 'true': label}}}})
        rows.append({'id': case['id'], 'question': key, 'label': label, 'type': 'noul', 'keys': ['false', 'true'],
            'variant': 'clean', 'source': 'fixture', 'inference_temperature': 1., 'logits': [0., 1.]})
    report = {'coverage': {'requested_records': 2, 'evaluated_records': 2, 'requested_questions': 2,
        'evaluated_questions': 2, 'rejected_records': 0, 'truncated_records': 0}}
    assert len(convert_benchmark(lines, report, cases, ['q0', 'q1', 'q2'], 'sha')['predictions']) == 2
    calibration_rows(rows, cases, ['q0', 'q1', 'q2'])
    with pytest.raises(DataError):
        calibration_rows(rows + [{**rows[0], 'question': 'q1', 'label': 0}], cases, ['q0', 'q1', 'q2'])
    with pytest.raises(DataError):
        calibration_rows(rows[:1], cases, ['q0', 'q1', 'q2'])
    report['coverage']['requested_questions'] = 6
    with pytest.raises(DataError):
        convert_benchmark(lines, report, cases, ['q0', 'q1', 'q2'], 'sha')


@pytest.mark.parametrize('count,mode', [(1, 'complete'), (7, 'explicit_questions'), (51, 'complete'), (100, 'explicit_questions')])
def test_existing_training_and_final_pipeline_accept_variable_task_contracts(client, business_factory, gpu, tmp_path, count, mode):
    task, run, _, _ = business_factory(count, mode)
    queue, (trainer, evaluator), (tw, ew) = gpu
    trial = new_trial(client, run)
    queue.enqueue(trial['id'], 'gpu_training_v1')
    result = run_once(trainer, tw['id'], backend=FakeBackend(), work_root=tmp_path/'train', interval=.01)
    assert result['state'] == 'SUCCEEDED', result
    assert result['training_verified'] is False
    assert client.get('/api/trials/' + trial['id'] + '/model.zip').status_code == 409
    assert client.post('/api/experiments/' + run['id'] + '/select-trial', json={'trial_id': trial['id']}).status_code == 200
    response = client.post('/api/experiments/' + run['id'] + '/evaluate')
    assert response.status_code == 202, response.text
    result = run_once(evaluator, ew['id'], backend=FakeBackend(), work_root=tmp_path/'evaluate', interval=.01)
    assert result['state'] == 'SUCCEEDED', result
    report = client.get('/api/experiments/' + run['id'] + '/report').json()
    assert report['training_verified'] is False
    assert report['candidate']['by_scenario']['car']['total'] == 2
    assert report['candidate']['overall']['annotated_questions'] == (2 if mode == 'explicit_questions' else 2*count)


def test_baseline_only_uses_same_sparse_business_protocol(client, business_factory, baseline_gpu, tmp_path):
    _, run, _, _ = business_factory(7, 'explicit_questions')
    queue, (trainer, _), (worker, _) = baseline_gpu
    queue.enqueue_baseline(run['id'])
    result = run_once(trainer, worker['id'], backend=FakeBackend(), work_root=tmp_path/'baseline', interval=.01)
    assert result['state'] == 'SUCCEEDED', result
    report = client.get('/api/experiments/' + run['id'] + '/baseline').json()
    assert report['training_verified'] is False and not report['baseline_verified']
    assert report['metrics']['overall']['annotated_questions'] == 2
    assert report['metrics']['by_scenario']['car']['total'] == 2
    assert queue.phase.trials(run['id']) == []


def test_old_task_hashes_still_validate(client, app_instance):
    task = client.post('/api/task-specs', json=task_body()).json()
    task.pop('business_scope')
    task.pop('label_mode')
    task['sha256'] = sha({key: task[key] for key in ('name', 'objects', 'schema_version', 'task_type')})
    store = app_instance.state.store
    store.write(store.path('task-specs', task['id']) / 'meta.json', task)
    assert client.get('/api/task-specs/' + task['id']).json() == task


def test_scene_changes_reach_input_and_conflicting_rules_cannot_be_merged():
    rows = normalize([{'id': 'car', 'text': 'open window', 'scenario': 'car', 'intents': ['q0']},
                      {'id': 'app', 'text': 'open window', 'scenario': 'app', 'intents': []}])
    task = task_body(scope='')
    converted = system_one(rows, task)
    assert converted[0]['state'] == {'scenario': 'car', 'text': 'open window'}
    assert converted[1]['state'] == {'scenario': 'app', 'text': 'open window'}
    with pytest.raises(DataError):
        system_one(rows, task_body(scope='car'))
    with pytest.raises(DataError):
        normalize([{'text': 'invalid', 'scenario': None, 'intents': []}])


def test_business_report_does_not_hide_regression_behind_equal_total_score():
    cases = normalize([{'id': 'car', 'text': 'car words', 'scenario': 'car', 'intents': ['q0']},
                       {'id': 'app', 'text': 'app words', 'scenario': 'app', 'intents': ['q0']}])
    results = []
    for correct in ('car', 'app'):
        value = validate_predictions({'dataset_sha256': 'sha', 'predictions': [
            {'id': row['id'], 'intents': ['q0'] if row['id'] == correct else []} for row in cases]}, cases, 'sha', ['q0'])
        results.append(evaluate(cases, value))
    assert results[0]['overall']['accuracy'] == results[1]['overall']['accuracy']
    assert results[0]['by_scenario']['car']['accuracy'] == 1
    assert results[1]['by_scenario']['car']['accuracy'] == 0


@pytest.mark.parametrize('returned_questions', [7, 1])
def test_native_loader_canary_checks_full_schema_even_for_sparse_first_record(tmp_path, monkeypatch, returned_questions):
    import kev_runtime
    captured = {}
    class Model:
        def encode(self, tokenizer, record, **kwargs):
            captured.update(record)
            return record
        def forward(self, encoded):
            return [object() for _ in range(returned_questions)]
    checkpoint = types.ModuleType('kev.checkpoint')
    checkpoint.Checkpoint = lambda path: types.SimpleNamespace(load=lambda *args, **kwargs: (object(), Model()))
    checkpoint.LoadOptions = lambda **kwargs: kwargs
    data = types.ModuleType('kev.data')
    data.materialize = lambda row: row
    model = types.ModuleType('kev.model')
    model.training_context = lambda: {}
    torch = types.ModuleType('torch')
    torch.no_grad = nullcontext
    torch.float32 = 'fixture-fp32'
    torch.isfinite = lambda value: types.SimpleNamespace(all=lambda: types.SimpleNamespace(item=lambda: True))
    for name, module in {'kev': types.ModuleType('kev'), 'kev.checkpoint': checkpoint,
                         'kev.data': data, 'kev.model': model, 'torch': torch}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(kev_runtime, 'compatibility', lambda *args: {})
    monkeypatch.setattr(kev_runtime, 'digest', lambda *args: 'fixture-hash')
    monkeypatch.setattr(kev_runtime, 'read', lambda *args: {'evaluation_context': {'max_state': 384, 'max_branch': 640}})
    monkeypatch.setattr(kev_runtime, 'records', lambda *args: [{
        'state': {'scenario': 'car', 'text': 'fixture'}, 'questions': {'q0': {'type': 'noul', 'label': True}}}])
    payload = {'mode': 'train', 'work': str(tmp_path), 'config': {'model': {}}, 'objects': task_body(7)['objects']}
    if returned_questions == 7:
        assert kev_runtime.load_model(payload)['objects_verified'] == 7
    else:
        with pytest.raises(ValueError, match='canary'):
            kev_runtime.load_model(payload)
    assert len(captured['questions']) == 7
    assert captured['state']['scenario'] == 'car'
