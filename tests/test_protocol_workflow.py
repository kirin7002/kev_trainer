"""Experiment contracts and native exports, with isolated data and no model execution."""
import json

import pytest
from fastapi.testclient import TestClient

from app import create_app
from conftest import TEST_ROWS, TRAIN_ROWS
from datasets import DataError, normalize
from experiment_protocol import build_protocol, digest, native_records


def payload(client, experiment, split='validation', kind='baseline', model=None, correct=False):
    meta = experiment['meta']
    protocol = client.get(f"/api/experiments/{meta['id']}").json()['evaluation_protocol']
    inputs = client.get(f"/api/experiments/{meta['id']}/evaluations/{split}").json()
    assert 'records' in inputs, inputs
    targets = {row['id']: row['intents'] for row in TRAIN_ROWS + TEST_ROWS}
    return {'split': split, 'protocol_sha256': protocol['sha256'],
        'dataset_sha256': protocol['splits'][split]['sha256'],
        'model': model or (meta['model']['checkpoint'] if kind == 'baseline' else 'local/candidate@sha256:abc'),
        'runtime': {'device': 'test-device', 'backend': 'test-only', 'dtype': 'fp32'},
        'predictions': [{'id': row['id'], 'intents': targets[row['id']] if correct else [], 'latency_ms': 1}
                        for row in inputs['records']]}


def submit(client, experiment, value, kind='baseline', split='validation'):
    return client.post(f"/api/experiments/{experiment['meta']['id']}/predictions/{kind}?split={split}",
        files={'file': ('predictions.json', json.dumps(value), 'application/json')})


def compare_validation(client, experiment):
    assert submit(client, experiment, payload(client, experiment)).status_code == 200
    candidate = payload(client, experiment, kind='candidate', correct=True)
    assert submit(client, experiment, candidate, 'candidate').status_code == 200
    return candidate


def test_native_export_locks_versions_explicit_negatives_and_excludes_test(client, experiment):
    root = f"/api/experiments/{experiment['meta']['id']}"
    manifest = client.get(root + '/native-training/manifest.json').json()
    assert manifest['executed'] is False and manifest['training_verified'] is False
    assert manifest['kev_commit'] == experiment['meta']['model']['kev_commit']
    argv = manifest['argv']
    assert argv[argv.index('--init_from') + 1] == experiment['meta']['model']['checkpoint']
    assert argv[argv.index('--base_revision') + 1] == experiment['meta']['model']['base_revision']
    assert set(manifest['inputs']) == {'training.jsonl', 'validation.jsonl'}
    protocol = client.get(root).json()['evaluation_protocol']
    exported = []
    for split in ('training', 'validation'):
        response = client.get(root + f'/native-training/{split}.jsonl')
        assert response.status_code == 200
        rows = [json.loads(line) for line in response.text.splitlines()]
        assert len(rows) == protocol['splits'][split]['count']
        exported += rows
    assert {row['id'] for row in exported} == {row['id'] for row in TRAIN_ROWS}
    assert not {row['state'] for row in exported} & {row['text'] for row in TEST_ROWS}
    for row in exported:
        target = next(case['intents'] for case in TRAIN_ROWS if case['id'] == row['id'])
        assert set(row['questions']) == set(protocol['questions'])
        for key, question in row['questions'].items():
            assert question['type'] == 'noul'
            assert question['label'] is (key in target)
    assert client.get(root + '/native-training/test.jsonl').status_code == 404
    assert client.get(root + '/evaluations/test').status_code == 409
    inputs = client.get(root + '/evaluations/validation').json()
    assert all('label' not in q for r in inputs['records'] for q in r['questions'].values())


def test_baseline_precedes_candidate_and_baseline_metrics_available(client, experiment):
    candidate = payload(client, experiment, kind='candidate', correct=True)
    assert submit(client, experiment, candidate, 'candidate').status_code == 409
    value = payload(client, experiment)
    assert submit(client, experiment, value).json()['report_ready'] is False
    root = f"/api/experiments/{experiment['meta']['id']}"
    detail = client.get(root).json()
    info = detail['evaluations']['validation']
    assert info['baseline_summary']['total'] == detail['split']['validation']
    assert info['baseline'] and not info['candidate']
    assert client.get(root + '/report?split=validation').status_code == 404
    assert submit(client, experiment, candidate, 'candidate').json()['report_ready'] is True
    report = client.get(root + '/report?split=validation').json()
    assert report['candidate']['overall']['accuracy'] == 1
    assert report['accuracy_delta'] >= 0
    assert report['source'] == 'user_imported_predictions' and report['training_verified'] is False
    assert report['protocol_sha256'] == detail['protocol_sha256']
    assert report['baseline_cache_key'] == info['baseline_cache_key']


@pytest.mark.parametrize('change', [
    {'dataset_sha256': 'other'}, {'protocol_sha256': 'other'}, {'split': 'test'},
    {'model': 'a-different-baseline'}, {'model': ''}, {'model': 123},
])
def test_rejects_mismatched_baseline_identity(client, experiment, change):
    value = payload(client, experiment)
    value.update(change)
    assert submit(client, experiment, value).status_code == 400
    detail = client.get(f"/api/experiments/{experiment['meta']['id']}").json()
    assert not detail['evaluations']['validation']['baseline']


def test_baseline_is_idempotent_and_cannot_be_replaced(client, experiment):
    value = payload(client, experiment)
    assert submit(client, experiment, value).status_code == 200
    assert submit(client, experiment, value).status_code == 200
    changed = json.loads(json.dumps(value))
    changed['predictions'][0]['latency_ms'] = 999
    assert submit(client, experiment, changed).status_code == 409
    root = f"/api/experiments/{experiment['meta']['id']}"
    assert client.get(root + '/artifacts/validation-baseline-predictions.json').json()['predictions'][
        value['predictions'][0]['id']]['latency_ms'] == 1


def test_final_candidate_selection_and_final_test_are_frozen(client, experiment):
    root = f"/api/experiments/{experiment['meta']['id']}"
    assert client.post(root + '/finalize').status_code == 409
    candidate = compare_validation(client, experiment)
    selected = client.post(root + '/finalize').json()
    assert selected['model'] == candidate['model']
    assert client.post(root + '/finalize').json() == selected
    assert submit(client, experiment, candidate, 'candidate').status_code == 409
    assert client.post(root + '/train').status_code == 409
    assert client.post(root + '/plan').status_code == 409
    baseline = payload(client, experiment, split='test')
    final = payload(client, experiment, split='test', kind='candidate', correct=True)
    assert submit(client, experiment, final, 'candidate', 'test').status_code == 409
    assert submit(client, experiment, baseline, split='test').status_code == 200
    assert submit(client, experiment, {**final, 'model': 'unselected'}, 'candidate', 'test').status_code == 400
    assert submit(client, experiment, final, 'candidate', 'test').status_code == 200
    report = client.get(root + '/report').json()
    assert report['split'] == 'test'
    assert report['baseline']['overall']['accuracy'] == pytest.approx(1 / 3)
    assert report['candidate']['overall']['accuracy'] == 1
    assert report['accuracy_delta'] == pytest.approx(2 / 3)
    assert report['improved_ids'] == ['te-1', 'te-2']
    assert report['performance_comparable'] is True
    assert submit(client, experiment, {**final, 'runtime': None}, 'candidate', 'test').status_code == 409
    assert submit(client, experiment, final, 'candidate', 'test').status_code == 200


def test_data_snapshot_is_independent_of_uploaded_dataset(client, experiment, app_instance):
    store = app_instance.state.store
    root = f"/api/experiments/{experiment['meta']['id']}"
    store.write(store.path('datasets', experiment['train']['id']) / 'records.json', [])
    store.write(store.path('datasets', experiment['test']['id']) / 'records.json', [])
    compare_validation(client, experiment)
    assert client.post(root + '/finalize').status_code == 200
    inputs = client.get(root + '/evaluations/test').json()
    assert {row['id'] for row in inputs['records']} == {row['id'] for row in TEST_ROWS}


@pytest.mark.parametrize('target', ['protocol', 'validation', 'training'])
def test_detects_modified_protocol_or_snapshots(client, experiment, app_instance, target):
    store = app_instance.state.store
    directory = store.path('runs', experiment['meta']['id'])
    path = directory / (target + '.json')
    value = store.read(path)
    if target == 'protocol':
        value['decision']['threshold'] = .9
    else:
        value[0]['text'] = 'modified after baseline'
    store.write(path, value)
    url = f"/api/experiments/{experiment['meta']['id']}/native-training/{'training.jsonl' if target == 'training' else 'validation.jsonl'}"
    assert client.get(url).status_code == 400


def test_unknown_labels_in_predictions_are_rejected(client, experiment):
    value = payload(client, experiment)
    value['predictions'][0]['intents'] = ['undefined-label']
    assert submit(client, experiment, value).status_code == 400


def test_protocol_and_baseline_survive_restart(client, experiment, data_dir):
    compare_validation(client, experiment)
    root = f"/api/experiments/{experiment['meta']['id']}"
    before = client.get(root).json()
    with TestClient(create_app(data_dir)) as restarted:
        after = restarted.get(root).json()
        assert after['evaluation_protocol'] == before['evaluation_protocol']
        assert after['evaluations'] == before['evaluations']
        assert restarted.post(root + '/finalize').status_code == 200


def test_stale_report_cannot_finalize_candidate_after_partial_write(client, experiment, app_instance):
    compare_validation(client, experiment)
    store = app_instance.state.store
    root = f"/api/experiments/{experiment['meta']['id']}"
    path = store.path('runs', experiment['meta']['id']) / 'validation-candidate-predictions.json'
    changed = store.read(path)
    changed['model'] = 'another-checkpoint'
    store.write(path, changed)
    assert client.get(root + '/report?split=validation').status_code == 409
    assert client.get(root + '/artifacts/validation-report.json').status_code == 409
    assert client.post(root + '/finalize').status_code == 409


def test_mixed_business_questions_keep_per_sample_scope(client, upload):
    training = [
        {'id': 'a', 'text': 'a car instruction', 'questions': {
            'window': {'type': 'noul', 'instructions': 'Window?', 'label': True}}},
        {'id': 'b', 'text': 'a support ticket', 'questions': {
            'billing': {'type': 'noul', 'instructions': 'Billing?', 'label': False},
            'refund': {'type': 'noul', 'instructions': 'Refund?', 'label': True}}},
        {'id': 'c', 'text': 'another car instruction', 'questions': {
            'window': {'type': 'noul', 'instructions': 'Window?', 'label': False}}},
    ]
    test = [{'id': 'd', 'text': 'held out support request', 'questions': {
        'billing': {'type': 'noul', 'instructions': 'Billing?', 'label': True}}}]
    train_meta, test_meta = upload('train', training).json(), upload('test', test).json()
    response = client.post('/api/experiments', json={'name': 'mixed', 'model_id': 'kev-0.8b-v1.0',
        'train_dataset_id': train_meta['id'], 'test_dataset_id': test_meta['id']})
    assert response.status_code == 201, response.text
    root = '/api/experiments/' + response.json()['id']
    protocol = client.get(root).json()['evaluation_protocol']
    assert set(protocol['questions']) == {'window', 'billing', 'refund'}
    output = []
    for split in ('training', 'validation'):
        output.extend(json.loads(line) for line in client.get(root + f'/native-training/{split}.jsonl').text.splitlines())
    expected = {row['id']: row['questions'] for row in training}
    for row in output:
        assert row['questions'] == expected[row['id']]
    inputs = client.get(root + '/evaluations/validation').json()
    record = inputs['records'][0]
    not_asked = next(key for key in protocol['questions'] if key not in record['questions'])
    value = {'split': 'validation', 'protocol_sha256': protocol['sha256'],
        'dataset_sha256': inputs['dataset_sha256'], 'model': inputs['baseline_model'],
        'predictions': [{'id': r['id'], 'intents': [not_asked] if r['id'] == record['id'] else []}
                        for r in inputs['records']]}
    result = client.post(root + '/predictions/baseline?split=validation',
        files={'file': ('p.json', json.dumps(value), 'application/json')})
    assert result.status_code == 400


def test_cache_identity_changes_with_conditions(client, experiment, upload):
    base = {'name': 'changed', 'train_dataset_id': experiment['train']['id'],
        'test_dataset_id': experiment['test']['id'], 'model_id': experiment['meta']['model']['id']}
    def create(changes):
        response = client.post('/api/experiments', json={**base, **changes})
        assert response.status_code == 201, response.text
        return client.get('/api/experiments/' + response.json()['id']).json()
    original = create({})
    again = create({})
    assert original['protocol_sha256'] == again['protocol_sha256']
    keys = {original['evaluations']['validation']['baseline_cache_key']}
    for changes in ({'decision_threshold': .7}, {'model_id': 'kev-4b-v1.0'},
                    {'label_definitions': {'window': 'Is a car window mentioned?'}}):
        changed = create(changes)
        key = changed['evaluations']['validation']['baseline_cache_key']
        assert key not in keys
        keys.add(key)
    different_test = upload('test', [{'id': 'new', 'text': 'another unseen text', 'intents': []}]).json()
    changed = create({'test_dataset_id': different_test['id']})
    assert changed['protocol_sha256'] != original['protocol_sha256']


def test_no_implicit_negative_for_partially_labeled_noul():
    questions = {'window': {'type': 'noul', 'instructions': 'Window?'},
                 'door': {'type': 'noul', 'instructions': 'Door?'}}
    row = {'id': 'x', 'text': 'window', 'intents': ['window'],
           'questions': {'window': {'type': 'noul', 'instructions': 'Window?', 'label': True}}}
    result = native_records([row], questions)[0]
    assert set(result['questions']) == {'window'}
    assert 'door' not in result['questions']
    row['questions']['window'].pop('label')
    with pytest.raises(DataError, match='布尔'):
        native_records([row], questions)
    row['questions']['window']['label'] = True
    row['questions']['door'] = {**questions['door'], 'label': False}
    result = native_records([row], questions)[0]
    assert result['questions']['door']['label'] is False
    assert result['questions']['window']['label'] is True


def test_question_rules_are_fixed_and_not_inferred_from_test():
    model = {'checkpoint': 'fixed'}
    train = [{'id': 'a', 'text': 'first', 'intents': ['known']}]
    test = [{'id': 'b', 'text': 'second', 'intents': ['test-only']}]
    with pytest.raises(DataError, match='未定义'):
        build_protocol(model, train, {'training': train, 'validation': train, 'test': test}, {}, .5)
    protocol = build_protocol(model, train, {'training': train, 'validation': train, 'test': test},
                              {'test-only': 'An explicitly defined label'}, .5)
    assert protocol['questions']['test-only']['instructions'] == 'An explicitly defined label'


def test_scenario_metadata_and_missing_split_coverage():
    rows = normalize([{'text': 'business one', 'intents': ['a'], 'scenario': 'one'},
                      {'text': 'business two', 'intents': [], 'scenario': 'two'}])
    protocol = build_protocol({'checkpoint': 'fixed'}, rows,
        {'training': rows[:1], 'validation': rows[1:], 'test': rows}, {}, .5)
    assert protocol['scenario_counts']['test'] == {'one': 1, 'two': 1}
    assert len(protocol['coverage_warnings']) == 2
    assert native_records(rows, protocol['questions'])[0]['state'] == {'scenario': 'one', 'text': 'business one'}
    with pytest.raises(DataError, match='scenario'):
        normalize([{'text': 'invalid metadata', 'intents': [], 'scenario': 12}])


def test_legacy_experiment_cannot_claim_frozen_validation(client, legacy_experiment):
    root = f"/api/experiments/{legacy_experiment['meta']['id']}"
    assert client.get(root + '/evaluations/validation').status_code == 409
    assert client.get(root + '/native-training/manifest.json').status_code == 409
    assert client.post(root + '/finalize').status_code == 409


def test_separate_business_experiments_keep_inputs_and_baselines_distinct(client, experiment):
    base = {'name': 'scoped', 'train_dataset_id': experiment['train']['id'],
        'test_dataset_id': experiment['test']['id'], 'model_id': 'kev-0.8b-v1.0'}
    protocols = []
    for scope in ('car', 'app'):
        response = client.post('/api/experiments', json={**base, 'business_scope': scope})
        assert response.status_code == 201
        root = '/api/experiments/' + response.json()['id']
        detail = client.get(root).json()
        assert detail['business_scope'] == scope
        protocols.append(detail['protocol_sha256'])
        inputs = client.get(root + '/evaluations/validation').json()
        assert all(row['state']['scenario'] == scope for row in inputs['records'])
        manifest = client.get(root + '/native-training/manifest.json').json()
        assert manifest['business_scope'] == scope
        assert manifest['executed'] is False
    assert protocols[0] != protocols[1]


def test_scope_rejects_other_business_data(client, upload):
    training = [{**row, 'scenario': 'car'} for row in TRAIN_ROWS]
    train = upload('train', training).json()
    test = upload('test', TEST_ROWS).json()
    response = client.post('/api/experiments', json={'name': 'wrong scope', 'business_scope': 'app',
        'model_id': 'kev-0.8b-v1.0', 'train_dataset_id': train['id'], 'test_dataset_id': test['id']})
    assert response.status_code == 400
