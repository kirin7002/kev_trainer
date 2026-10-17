"""Cloud review regressions. Only temporary files and disposable fake CLIs."""
import json
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import signal
import threading
import time

import pytest

import claude_runner
from datasets import DataError, parse_upload
from evaluation import summarize
from machine import ConnectionStatus
from test_planning import fake_cli, start_planning, wait_status
from test_review_regressions import make_fake_cli


BAD_JSON = [
    b'[{"text":"a","intents":[],"intents":["x"]}]',
    b'[{"text":"a","intents":[],"extra":NaN}]',
    b'[{"text":"a","intents":[],"extra":1e400}]',
    b'[{"text":"\\ud800","intents":[]}]',
    b'[{"text":"a","intents":[],"extra":' + b'9' * 5000 + b'}]',
    b'[' * 1200 + b'0' + b']' * 1200,
    b'[{"text":"a","questions":{"x":{"type":"noul","label":true,"criteria":'
    + b'[' * 80 + b'0' + b']' * 80 + b'}}}]',
]


@pytest.mark.parametrize('raw', BAD_JSON)
@pytest.mark.parametrize('filename', ['bad.json', 'bad.jsonl'])
def test_invalid_json_upload_is_400_and_leaves_no_dataset(client, raw, filename):
    response = client.post('/api/datasets', data={'role': 'train'},
                           files={'file': (filename, raw, 'application/json')})
    assert response.status_code == 400, response.text
    assert client.get('/api/datasets').json() == []


def test_nested_noul_metadata_rejects_non_finite_numbers():
    with pytest.raises(DataError):
        parse_upload(b'[{"text":"a","questions":{"x":{"type":"noul","label":true,"criteria":NaN}}}]', 'x.json')


@pytest.mark.parametrize('filename,raw', [
    ('data.csv', b'text,intents\na,"[""\\ud800""]"\n'),
    ('data.json', json.dumps([{'text': 'a', 'intents': '["\\ud800"]'}]).encode()),
    ('data.csv', b'text,intents\na,"[not valid JSON]"\n'),
])
def test_embedded_json_labels_are_validated_before_storage(client, filename, raw):
    response = client.post('/api/datasets', data={'role': 'train'}, files={'file': (filename, raw)})
    assert response.status_code == 400
    assert client.get('/api/datasets').json() == []


@pytest.mark.parametrize('field,value', [('claude_budget_usd', float('nan')),
    ('claude_budget_usd', float('inf')), ('name', '\ud800'), ('goal', '\udfff'), ('notes', '\ud800')])
def test_invalid_experiment_input_is_422_without_echoing_unserializable_values(client, experiment, field, value):
    body = {'name': 'valid', 'train_dataset_id': experiment['train']['id'],
            'test_dataset_id': experiment['test']['id'], 'model_id': 'kev-0.8b-v1.0', field: value}
    response = client.post('/api/experiments', content=json.dumps(body), headers={'Content-Type': 'application/json'})
    assert response.status_code == 422
    assert response.json()['detail'][0]['loc'][-1] == field
    assert len(client.get('/api/experiments').json()) == 1


def test_duplicate_experiment_fields_are_rejected_before_creating_run(client, experiment):
    body = {'name': 'valid', 'train_dataset_id': experiment['train']['id'],
            'test_dataset_id': experiment['test']['id'], 'model_id': 'kev-0.8b-v1.0'}
    raw = json.dumps(body)[:-1] + ', "name": "overwritten"}'
    response = client.post('/api/experiments', content=raw, headers={'Content-Type': 'application/json'})
    assert response.status_code == 400
    assert len(client.get('/api/experiments').json()) == 1


@pytest.mark.parametrize('bad', ['NaN', '1e400', '9' * 400])
def test_invalid_prediction_latency_does_not_write(client, experiment, bad):
    raw = json.dumps({'dataset_sha256': experiment['test']['sha256'], 'predictions': [
        {'id': row_id, 'intents': [], 'latency_ms': 'PLACEHOLDER'} for row_id in ('te-1', 'te-2', 'te-3')
    ]}).replace('"PLACEHOLDER"', bad).encode()
    exp_id = experiment['meta']['id']
    response = client.post(f'/api/experiments/{exp_id}/predictions/baseline',
                           files={'file': ('p.json', raw, 'application/json')})
    assert response.status_code == 400
    assert client.get(f'/api/experiments/{exp_id}').json()['predictions']['baseline'] is False


def test_large_finite_latencies_have_finite_mean():
    rows = [{'expected': {'x'}, 'actual': {'x'}, 'passed': True, 'latency_ms': 1e308} for _ in range(3)]
    result = summarize(rows)
    assert result['latency']['mean_ms'] == 1e308
    assert math.isfinite(result['latency']['p95_ms'])
    assert result['per_object'] == [{'object': 'x', 'support': 3, 'precision': 1, 'recall': 1, 'f1': 1}]


@pytest.mark.parametrize('cache', [None, {'source': 'stale', 'training_verified': True}])
def test_reports_recover_from_missing_or_stale_cache(client, experiment, app_instance, cache):
    exp_id = experiment['meta']['id']
    body = {'dataset_sha256': experiment['test']['sha256'], 'predictions': [
        {'id': row_id, 'intents': []} for row_id in ('te-1', 'te-2', 'te-3')
    ]}
    for kind in ('baseline', 'candidate'):
        assert client.post(f'/api/experiments/{exp_id}/predictions/{kind}',
                           files={'file': ('p.json', json.dumps(body), 'application/json')}).status_code == 200
    path = app_instance.state.experiments.directory(exp_id) / 'report.json'
    if cache is None:
        path.unlink()
    else:
        app_instance.state.store.write(path, cache)
    report = client.get(f'/api/experiments/{exp_id}/report')
    download = client.get(f'/api/experiments/{exp_id}/artifacts/report.json')
    assert report.status_code == download.status_code == 200
    assert report.json() == download.json()
    assert report.json()['training_verified'] is False
    assert report.json()['baseline']['overall']['accuracy'] == pytest.approx(1 / 3)
    assert 'attachment' in download.headers['content-disposition']


def test_successful_plan_then_failed_retry_never_serves_old_plan(client, experiment, fake_cli):
    exp_id = experiment['meta']['id']
    fake_cli('ok')
    start_planning(client, exp_id)
    wait_status(client, exp_id, {'plan_ready'})
    assert client.get(f'/api/experiments/{exp_id}/artifacts/plan.json').status_code == 200
    fake_cli('bad_schema')
    start_planning(client, exp_id)
    meta = wait_status(client, exp_id, {'failed'})
    assert meta['plan'] is None
    assert client.get(f'/api/experiments/{exp_id}/artifacts/plan.json').status_code == 404
    assert '方案' in client.post(f'/api/experiments/{exp_id}/train').json()['detail']


@pytest.mark.parametrize('output', ['[]', '{"type":"assistant","message":null}',
                                     '{"type":"assistant","message":{"content":[null]}}', 'x' * 256_001],
                         ids=['array', 'null-message', 'null-content', 'oversize-line'])
def test_malformed_or_unbounded_cli_output_is_controlled(tmp_path, monkeypatch, output):
    cli = make_fake_cli(tmp_path, 'import sys\nsys.stdin.read()\nprint(' + repr(output) + ', flush=True)\n')
    monkeypatch.setenv('CLAUDE_BIN', str(cli))
    with pytest.raises(claude_runner.PlanningError) as caught:
        claude_runner.generate_plan({}, tmp_path, 1, threading.Event(), lambda text: None)
    assert '协议' in str(caught.value) or '超过限制' in str(caught.value)


@pytest.mark.skipif(os.name != 'posix', reason='Private process groups require POSIX')
def test_cli_descendant_holding_stdout_is_killed_after_leader_exit(tmp_path, monkeypatch):
    pid_path = tmp_path / 'child-pid'
    cli = make_fake_cli(tmp_path, f'''
import subprocess, sys
sys.stdin.read()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
with open({str(pid_path)!r}, 'w') as handle:
    handle.write(str(child.pid))
''')
    monkeypatch.setenv('CLAUDE_BIN', str(cli))
    cancel = threading.Event()
    timer = threading.Timer(.5, cancel.set)
    timer.start()
    before = set(threading.enumerate())
    started = time.monotonic()
    try:
        with pytest.raises(claude_runner.PlanningCancelled):
            claude_runner.generate_plan({}, tmp_path, 1, cancel, lambda text: None)
        assert time.monotonic() - started < 5
        # On Linux the orphan can remain as a zombie until PID 1 reaps it;
        # it must no longer be executing or keeping our pipe open.
        pid = int(pid_path.read_text())
        proc_stat = Path(f'/proc/{pid}/stat')
        if proc_stat.exists():
            assert proc_stat.read_text().split(') ')[1].startswith('Z')
        assert not [t for t in threading.enumerate() if t not in before and t.is_alive()]
    finally:
        timer.cancel()
        timer.join()
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cancel_remains_responsive_after_cli_closes_stdout(tmp_path, monkeypatch):
    cli = make_fake_cli(tmp_path, '''
import os, sys, time
sys.stdin.read()
os.close(1)
os.close(2)
time.sleep(60)
''')
    monkeypatch.setenv('CLAUDE_BIN', str(cli))
    cancel = threading.Event()
    timer = threading.Timer(.5, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(claude_runner.PlanningCancelled):
            claude_runner.generate_plan({}, tmp_path, 1, cancel, lambda text: None)
        assert time.monotonic() - started < 3
    finally:
        timer.cancel()
        timer.join()


def test_train_connection_check_cannot_overwrite_new_plan_state(client, app_instance, experiment, fake_cli, monkeypatch):
    import app as app_module
    from conftest import VALID_PLAN

    fake_cli('hang')
    exp_id = experiment['meta']['id']
    experiments = app_instance.state.experiments
    app_instance.state.store.write(experiments.directory(exp_id) / 'plan.json', {'plan': VALID_PLAN})
    experiments.update(exp_id, status='plan_ready')
    checking = threading.Event()
    plan_requested = threading.Event()
    plan_finished = threading.Event()
    original_plan = experiments.plan

    def observed_plan(item_id):
        plan_requested.set()
        try:
            return original_plan(item_id)
        finally:
            plan_finished.set()

    class SlowOfflineMachine:
        def check_connection(self):
            checking.set()
            assert plan_requested.wait(3), 'No concurrent plan request arrived'
            # Before the fix, plan is accepted while this check is in flight;
            # its planning state is then overwritten by waiting_for_machine.
            # With a serialized transition, it is accepted after train returns.
            plan_finished.wait(.5)
            return ConnectionStatus(False, 'GPU fixture offline')

    monkeypatch.setattr(experiments, 'plan', observed_plan)
    monkeypatch.setattr(app_module, 'get_machine_client', SlowOfflineMachine)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            training = pool.submit(client.post, f'/api/experiments/{exp_id}/train')
            assert checking.wait(3)
            planning = pool.submit(client.post, f'/api/experiments/{exp_id}/plan')
            assert training.result(timeout=5).status_code == 409
            assert planning.result(timeout=5).status_code == 202
        assert client.get(f'/api/experiments/{exp_id}').json()['status'] == 'planning'
    finally:
        client.post(f'/api/experiments/{exp_id}/cancel')
        wait_status(client, exp_id, {'cancelled'})
