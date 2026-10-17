"""HTTP surface: uploads, queries, examples, experiments, predictions import, artifacts, Host/Origin."""
import json

import pytest

from conftest import TEST_ROWS, TRAIN_ROWS


# ---------- health / models ----------

def test_health(client):
    body = client.get('/api/health').json()
    assert body['status'] == 'ok'
    assert body['machine']['connected'] is False
    assert body['training_execution'] == 'not_connected'
    assert body['claude']['installed'] is False  # CLAUDE_BIN points at a nonexistent file in tests


def test_models_list(client):
    models = client.get('/api/models').json()
    assert [m['id'] for m in models] == ['kev-0.8b-v1.0', 'kev-4b-v1.0']
    assert models[0]['checkpoint']


# ---------- dataset upload / query ----------

def test_upload_train_and_test_and_list(client, upload):
    train = upload('train', TRAIN_ROWS, 'train.json')
    test = upload('test', TEST_ROWS, 'test.json')
    assert train.status_code == 201 and test.status_code == 201
    assert train.json()['role'] == 'train' and test.json()['role'] == 'test'
    assert train.json()['filename'] == 'train.json'
    listed = client.get('/api/datasets').json()
    assert {d['id'] for d in listed} == {train.json()['id'], test.json()['id']}


def test_upload_rejects_bad_role(client, upload):
    response = upload('invalid-role', TRAIN_ROWS)
    assert response.status_code == 400
    assert 'train' in response.json()['detail']


def test_upload_rejects_bad_extension(client, upload):
    assert upload('train', TRAIN_ROWS, filename='data.txt').status_code == 400
    assert upload('train', TRAIN_ROWS, filename='data.exe').status_code == 400


def test_upload_rejects_malformed_content(client, upload):
    response = upload('train', filename='bad.json', raw=b'{oops')
    assert response.status_code == 400
    assert 'detail' in response.json()


def test_upload_rejects_oversize_file(client, upload):
    raw = b'[' + b'{"text":"aaaaaaaa","intents":[]},' * 400000 + b'{"text":"end","intents":[]}]'
    assert len(raw) > 10 * 1024 * 1024
    assert upload('train', filename='big.json', raw=raw).status_code == 413


def test_dataset_detail_and_preview(client, upload):
    many = [{'id': f'r{i}', 'text': f'句子 {i}', 'intents': []} for i in range(25)]
    created = upload('train', many)
    detail = client.get(f"/api/datasets/{created.json()['id']}").json()
    assert len(detail['preview']) == 10
    assert detail['profile']['total'] == 25
    assert detail['sha256'] and len(detail['sha256']) == 64


def test_dataset_query_invalid_id_is_404(client):
    assert client.get('/api/datasets/not-a-uuid').status_code == 404
    assert client.get('/api/datasets/00000000-0000-0000-0000-000000000000').status_code == 404


# ---------- examples ----------

def test_create_examples_datasets(client):
    response = client.post('/api/examples')
    assert response.status_code == 201
    created = response.json()
    assert [c['role'] for c in created] == ['train', 'test']
    assert all(c['filename'].startswith('example-') for c in created)
    # they are queryable like normal datasets
    for c in created:
        assert client.get(f"/api/datasets/{c['id']}").status_code == 200


def test_download_example_files(client):
    from app import ROOT
    for role in ('train', 'test'):
        response = client.get(f'/api/examples/{role}')
        assert response.status_code == 200
        assert response.json() == json.loads((ROOT / 'examples' / f'{role}.json').read_text())
        assert f'{role}.json' in response.headers['content-disposition']
    assert client.get('/api/examples/eval').status_code == 404


# ---------- experiment creation ----------

def test_create_experiment_isolates_test_set(client, experiment, data_dir):
    meta = experiment['meta']
    assert meta['status'] == 'draft'
    assert meta['model']['id'] == 'kev-0.8b-v1.0'
    split = meta['split']
    assert split['test'] == len(TEST_ROWS)
    assert split['train'] + split['validation'] == len(TRAIN_ROWS)
    assert split['seed'] == 42
    # on-disk training/validation must not contain any test text or id
    run_dir = data_dir / 'runs' / meta['id']
    training = json.loads((run_dir / 'training.json').read_text())
    validation = json.loads((run_dir / 'validation.json').read_text())
    test_texts = {r['text'] for r in TEST_ROWS}
    test_ids = {r['id'] for r in TEST_ROWS}
    for rows in (training, validation):
        assert not ({r['text'] for r in rows} & test_texts)
        assert not ({r['id'] for r in rows} & test_ids)
    assert not ({r['id'] for r in training} & {r['id'] for r in validation})
    assert any('隔离' in e['message'] for e in meta['events'])


@pytest.mark.parametrize('changes,expected_status', [
    ({'model_id': 'kev-99b-v9'}, 400),
    ({'name': '   '}, 400),
])
def test_create_experiment_validation(client, upload, changes, expected_status):
    train = upload('train', TRAIN_ROWS).json()
    test = upload('test', TEST_ROWS).json()
    body = {'name': 'x', 'train_dataset_id': train['id'], 'test_dataset_id': test['id'],
            'model_id': 'kev-0.8b-v1.0'}
    body.update(changes)
    assert client.post('/api/experiments', json=body).status_code == expected_status


def test_create_experiment_rejects_same_dataset_and_wrong_roles(client, upload):
    train = upload('train', TRAIN_ROWS).json()
    test = upload('test', TEST_ROWS).json()
    same = client.post('/api/experiments', json={'name': 'x', 'train_dataset_id': train['id'],
        'test_dataset_id': train['id'], 'model_id': 'kev-0.8b-v1.0'})
    assert same.status_code == 400
    swapped = client.post('/api/experiments', json={'name': 'x', 'train_dataset_id': test['id'],
        'test_dataset_id': train['id'], 'model_id': 'kev-0.8b-v1.0'})
    assert swapped.status_code == 400


def test_create_experiment_rejects_train_test_overlap(client, upload):
    train = upload('train', TRAIN_ROWS).json()
    # same text as a training row (modulo whitespace) -> overlap must be rejected
    overlapping = [{'id': 'te-x', 'text': TRAIN_ROWS[0]['text'].replace('车窗', '车窗 '), 'intents': ['window']},
                   {'id': 'te-9', 'text': '完全独立的句子', 'intents': []}]
    test = upload('test', overlapping).json()
    response = client.post('/api/experiments', json={'name': 'x', 'train_dataset_id': train['id'],
        'test_dataset_id': test['id'], 'model_id': 'kev-0.8b-v1.0'})
    assert response.status_code == 400
    assert '重叠' in response.json()['detail']


def test_create_experiment_rejects_group_overlap(client, upload):
    train_rows = [dict(r) for r in TRAIN_ROWS]
    train_rows[0]['group_id'] = 'shared-group'
    test_rows = [dict(r) for r in TEST_ROWS]
    test_rows[0]['group_id'] = 'shared-group'
    train = upload('train', train_rows).json()
    test = upload('test', test_rows).json()
    response = client.post('/api/experiments', json={'name': 'x', 'train_dataset_id': train['id'],
        'test_dataset_id': test['id'], 'model_id': 'kev-0.8b-v1.0'})
    assert response.status_code == 400


def test_create_experiment_unknown_datasets_404(client):
    response = client.post('/api/experiments', json={'name': 'x',
        'train_dataset_id': '00000000-0000-0000-0000-000000000001',
        'test_dataset_id': '00000000-0000-0000-0000-000000000002', 'model_id': 'kev-0.8b-v1.0'})
    assert response.status_code == 404


def test_experiment_list_and_detail(client, experiment):
    listed = client.get('/api/experiments').json()
    assert len(listed) == 1
    assert 'events' not in listed[0]  # list view strips events
    detail = client.get(f"/api/experiments/{experiment['meta']['id']}").json()
    assert detail['plan'] is None
    assert detail['predictions'] == {'baseline': False, 'candidate': False}
    assert client.get('/api/experiments/not-a-uuid').status_code == 404


# ---------- training endpoint with GPU offline ----------

def test_train_without_plan_409(client, experiment):
    response = client.post(f"/api/experiments/{experiment['meta']['id']}/train")
    assert response.status_code == 409
    assert '方案' in response.json()['detail']


def test_train_with_plan_offline_sets_waiting_for_machine(client, app_instance, experiment):
    exp_id = experiment['meta']['id']
    store = app_instance.state.store
    store.write(store.path('runs', exp_id) / 'plan.json', {'plan': {}})
    response = client.post(f'/api/experiments/{exp_id}/train')
    assert response.status_code == 409
    assert 'GPU' in response.json()['detail'] or '机器' in response.json()['detail']
    meta = client.get(f'/api/experiments/{exp_id}').json()
    assert meta['status'] == 'waiting_for_machine'
    # no fabricated job/report/predictions artifacts
    run_dir = store.path('runs', exp_id)
    assert not (run_dir / 'report.json').exists()
    assert not (run_dir / 'baseline-predictions.json').exists()
    assert not (run_dir / 'candidate-predictions.json').exists()


# ---------- predictions import over HTTP ----------

def import_payload(sha, preds, model='m', runtime=None):
    records = [{'id': i, 'intents': v} for i, v in preds.items()]
    body = {'dataset_sha256': sha, 'model': model, 'predictions': records}
    if runtime is not None:
        body['runtime'] = runtime
    return json.dumps(body).encode()


def test_predictions_import_flow_and_report(client, experiment):
    exp_id = experiment['meta']['id']
    sha = experiment['test']['sha256']
    baseline = {'te-1': [], 'te-2': ['window'], 'te-3': []}   # accuracy 1/3
    candidate = {'te-1': ['window'], 'te-2': ['window'], 'te-3': ['x']}  # accuracy 1/3
    r1 = client.post(f'/api/experiments/{exp_id}/predictions/baseline',
        files={'file': ('b.json', import_payload(sha, baseline), 'application/json')})
    assert r1.status_code == 200 and r1.json() == {'imported': 3, 'report_ready': False}
    r2 = client.post(f'/api/experiments/{exp_id}/predictions/candidate',
        files={'file': ('c.json', import_payload(sha, candidate), 'application/json')})
    assert r2.status_code == 200 and r2.json() == {'imported': 3, 'report_ready': True}
    report = client.get(f'/api/experiments/{exp_id}/report').json()
    # hand-computed: baseline correct {te-3}; candidate correct {te-1}
    assert report['baseline']['overall']['accuracy'] == pytest.approx(1 / 3)
    assert report['candidate']['overall']['accuracy'] == pytest.approx(1 / 3)
    assert report['accuracy_delta'] == pytest.approx(0.0)
    assert report['improved_ids'] == ['te-1']
    assert report['regressed_ids'] == ['te-3']
    assert report['source'] == 'user_imported_predictions'
    assert report['training_verified'] is False
    detail = client.get(f'/api/experiments/{exp_id}').json()
    assert detail['predictions'] == {'baseline': True, 'candidate': True}
    assert detail['report_revision']
    assert any('外部预测' in e['message'] for e in detail['events'])


def test_predictions_import_rejections(client, experiment):
    exp_id = experiment['meta']['id']
    sha = experiment['test']['sha256']
    url = f'/api/experiments/{exp_id}/predictions/baseline'

    def post(raw, kind='baseline'):
        return client.post(f'/api/experiments/{exp_id}/predictions/{kind}',
            files={'file': ('p.json', raw, 'application/json')})

    assert post(import_payload('wrong-sha', {'te-1': [], 'te-2': [], 'te-3': []})).status_code == 400
    assert post(import_payload(sha, {'te-1': [], 'te-2': []})).status_code == 400  # incomplete
    assert post(import_payload(sha, {'te-1': [], 'te-2': [], 'te-3': [], 'zzz': []})).status_code == 400
    assert post(b'not json at all').status_code == 400
    assert post(import_payload(sha, {'te-1': [], 'te-2': [], 'te-3': []}), kind='model').status_code == 400
    # nothing was stored by the failed attempts
    detail = client.get(f'/api/experiments/{exp_id}').json()
    assert detail['predictions'] == {'baseline': False, 'candidate': False}


def test_report_missing_is_404(client, experiment):
    assert client.get(f"/api/experiments/{experiment['meta']['id']}/report").status_code == 404


# ---------- artifacts ----------

def test_artifacts(client, experiment):
    exp_id = experiment['meta']['id']
    response = client.get(f'/api/experiments/{exp_id}/artifacts/meta.json')
    assert response.status_code == 200
    assert response.json()['id'] == exp_id
    assert client.get(f'/api/experiments/{exp_id}/artifacts/plan.json').status_code == 404
    assert client.get(f'/api/experiments/{exp_id}/artifacts/report.json').status_code == 404
    assert client.get(f'/api/experiments/{exp_id}/artifacts/model.safetensors').status_code == 404
    assert client.get(f'/api/experiments/{exp_id}/artifacts/..%2fmeta.json').status_code in (400, 404)
    assert client.get(f'/api/experiments/{exp_id}/artifacts/../datasets').status_code in (400, 404)


# ---------- Host / Origin boundary ----------

def test_foreign_host_rejected(client):
    for host in ('192.168.1.50:8091', 'kev.example.com', '0.0.0.0:8091', '127.0.0.2'):
        response = client.get('/api/health', headers={'Host': host})
        assert response.status_code == 403, host
        assert '本机' in response.json()['detail']


def test_local_hosts_allowed(client):
    for host in ('127.0.0.1:8091', 'localhost:8091', '[::1]:8091'):
        assert client.get('/api/health', headers={'Host': host}).status_code == 200, host


def test_cross_origin_mutation_rejected(client):
    response = client.post('/api/examples', headers={'Origin': 'http://evil.example'})
    assert response.status_code == 403
    assert '来源' in response.json()['detail']


def test_same_origin_mutation_allowed(client):
    assert client.post('/api/examples', headers={'Origin': 'http://testserver'}).status_code == 201


def test_get_with_foreign_origin_still_allowed_but_headers_set(client):
    response = client.get('/api/health', headers={'Origin': 'http://evil.example'})
    assert response.status_code == 200
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert response.headers['Referrer-Policy'] == 'no-referrer'
    assert response.headers['Cache-Control'] == 'no-store'
