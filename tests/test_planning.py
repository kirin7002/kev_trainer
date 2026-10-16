"""Background Claude planning: fake CLI subprocess speaking the real stdin/stream-json protocol.

No real Claude model, network or GPU is used. CLAUDE_BIN points at a generated Python
script that emulates the CLI: reads the context JSON from stdin, emits stream-json
events on stdout, exits with the configured behaviour.
"""
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from conftest import TEST_ROWS, VALID_PLAN

FAKE_CLI = '''#!{executable}
import json, os, sys, time
MODE = {mode!r}
PLAN = json.loads({plan_json!r})
data = sys.stdin.read()
with open(os.environ['FAKE_STDIN_DUMP'], 'w', encoding='utf-8') as f:
    f.write(data)
with open(os.environ['FAKE_ARGS_DUMP'], 'w', encoding='utf-8') as f:
    f.write(json.dumps(sys.argv[1:], ensure_ascii=False))
pid_file = os.environ.get('FAKE_PID_FILE')
if pid_file:
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)

if MODE == 'hang':
    while True:
        time.sleep(0.2)
elif MODE == 'error_exit':
    print('fatal: authentication failed (fake)', flush=True)
    sys.exit(3)
elif MODE == 'bad_schema':
    PLAN['training']['learning_rate'] = 0.5
    PLAN['steps'] = []
    emit({{'type': 'system', 'subtype': 'init', 'model': 'fake-model'}})
    emit({{'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'done',
           'structured_output': PLAN, 'session_id': 'sess-bad', 'total_cost_usd': 0.01}})
elif MODE == 'error_result':
    emit({{'type': 'result', 'subtype': 'error_during_execution', 'is_error': True,
           'result': 'API Error: 401 invalid credentials (fake)'}})
    sys.exit(0)
else:
    emit({{'type': 'system', 'subtype': 'init', 'model': 'fake-model'}})
    emit({{'type': 'assistant', 'message': {{'content': [{{'type': 'text', 'text': '正在分析训练数据概况…'}}]}}}})
    emit({{'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'done',
           'structured_output': PLAN, 'session_id': 'sess-ok', 'total_cost_usd': 0.0123}})
'''


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    counter = {'n': 0}

    def make(mode='ok', plan=None):
        counter['n'] += 1
        script = tmp_path / f'fake-claude-{counter["n"]}-{mode}'
        script.write_text(FAKE_CLI.format(executable=sys.executable, mode=mode,
            plan_json=json.dumps(plan or VALID_PLAN, ensure_ascii=False)))
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        monkeypatch.setenv('CLAUDE_BIN', str(script))
        dump = tmp_path / f'stdin-{counter["n"]}.json'
        args_dump = tmp_path / f'args-{counter["n"]}.json'
        pid_file = tmp_path / f'pid-{counter["n"]}'
        monkeypatch.setenv('FAKE_STDIN_DUMP', str(dump))
        monkeypatch.setenv('FAKE_ARGS_DUMP', str(args_dump))
        monkeypatch.setenv('FAKE_PID_FILE', str(pid_file))
        return {'script': script, 'stdin_dump': dump, 'args_dump': args_dump, 'pid_file': pid_file}

    return make


def wait_status(client, exp_id, statuses, timeout=30):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.get(f'/api/experiments/{exp_id}').json()
        if last['status'] in statuses:
            return last
        time.sleep(0.05)
    pytest.fail(f'超时等待状态 {statuses}，最后状态：{last["status"]}，error={last.get("error")}')


def wait_file(path, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).exists():
            return
        time.sleep(0.05)
    pytest.fail(f'超时等待文件 {path}')


def start_planning(client, exp_id):
    response = client.post(f'/api/experiments/{exp_id}/plan')
    assert response.status_code == 202
    assert response.json() == {'status': 'planning'}


def wait_dead(pid_file, timeout=15):
    deadline = time.monotonic() + timeout
    pid = int(Path(pid_file).read_text())
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        time.sleep(0.05)
    pytest.fail(f'假 CLI 进程 {pid} 未被停止')


# ---------- happy path ----------

def test_plan_success_protocol_and_no_test_set_leak(client, experiment, fake_cli, data_dir):
    fake = fake_cli('ok')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    meta = wait_status(client, exp_id, {'plan_ready', 'failed'})
    assert meta['status'] == 'plan_ready', meta.get('error')
    assert meta['error'] is None

    # stored plan wraps the structured output
    plan_file = data_dir / 'runs' / exp_id / 'plan.json'
    stored = json.loads(plan_file.read_text())
    assert stored['plan'] == VALID_PLAN
    assert stored['session_id'] == 'sess-ok'
    assert stored['estimated_cost_usd'] == pytest.approx(0.0123)
    assert stored['source'] == 'claude_code'

    # the fake CLI really received the context over stdin, with schema + hardened flags
    wait_file(fake['stdin_dump'])
    context = json.loads(fake['stdin_dump'].read_text())
    args = json.loads(fake['args_dump'].read_text())
    assert '--output-format' in args and 'stream-json' in args
    assert '--permission-mode' in args and 'dontAsk' in args
    assert '--tools' in args and args[args.index('--tools') + 1] == ''
    assert '--strict-mcp-config' in args
    assert '--max-budget-usd' in args
    assert context['model']['id'] == 'kev-0.8b-v1.0'
    assert context['limits']['machine_connected'] is False
    assert context['training_examples'] and len(context['training_examples']) <= 30
    assert context['validation_examples'] is not None
    assert 'training_profile' in context and 'validation_profile' in context

    # FINAL TEST SET MUST NEVER REACH THE MODEL CONTEXT
    raw_stdin = fake['stdin_dump'].read_text()
    for row in TEST_ROWS:
        assert row['text'] not in raw_stdin
        assert row['id'] not in raw_stdin
    assert experiment['test']['sha256'] not in raw_stdin
    assert 'test_dataset' not in context

    # training/validation examples come only from the train dataset
    for example in context['training_examples'] + context['validation_examples']:
        assert example['id'].startswith('tr-')

    # progress events were streamed into the experiment log
    messages = ' '.join(e['message'] for e in meta['events'])
    assert 'Claude Code 已启动' in messages
    assert '正在分析训练数据概况' in messages
    assert '方案已生成' in messages

    # artifact endpoint serves the plan
    artifact = client.get(f'/api/experiments/{exp_id}/artifacts/plan.json')
    assert artifact.status_code == 200 and artifact.json() == stored


def test_planning_context_file_excludes_test_set(client, experiment, fake_cli, data_dir):
    fake_cli('ok')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    wait_status(client, exp_id, {'plan_ready', 'failed'}, timeout=30)
    context = json.loads((data_dir / 'runs' / exp_id / 'planning-context.json').read_text())
    blob = json.dumps(context, ensure_ascii=False)
    for row in TEST_ROWS:
        assert row['text'] not in blob


# ---------- failure paths ----------

def test_plan_cli_nonzero_exit_marks_failed(client, experiment, fake_cli):
    fake_cli('error_exit')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    meta = wait_status(client, exp_id, {'failed', 'plan_ready'})
    assert meta['status'] == 'failed'
    assert 'authentication failed' in meta['error']
    assert any('方案生成失败' in e['message'] for e in meta['events'])
    assert client.get(f'/api/experiments/{exp_id}/artifacts/plan.json').status_code == 404


def test_plan_error_result_marks_failed(client, experiment, fake_cli):
    fake_cli('error_result')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    meta = wait_status(client, exp_id, {'failed', 'plan_ready'})
    assert meta['status'] == 'failed'
    assert '401' in meta['error']


def test_plan_schema_violation_marks_failed(client, experiment, fake_cli):
    fake_cli('bad_schema')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    meta = wait_status(client, exp_id, {'failed', 'plan_ready'})
    assert meta['status'] == 'failed'
    assert '校验' in meta['error']


def test_retry_after_failure_succeeds(client, experiment, fake_cli):
    fake_cli('bad_schema')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    wait_status(client, exp_id, {'failed'})
    # second attempt with a healthy CLI must be accepted and finish
    fake_cli('ok')
    start_planning(client, exp_id)
    meta = wait_status(client, exp_id, {'plan_ready', 'failed'})
    assert meta['status'] == 'plan_ready', meta.get('error')


def test_plan_without_claude_binary_409(client, experiment):
    # conftest pins CLAUDE_BIN to a nonexistent path
    response = client.post(f"/api/experiments/{experiment['meta']['id']}/plan")
    assert response.status_code == 409
    assert 'Claude Code' in response.json()['detail']
    # experiment stays in draft, no thread started
    assert client.get(f"/api/experiments/{experiment['meta']['id']}").json()['status'] == 'draft'


# ---------- concurrency ----------

def test_concurrent_plan_rejected(client, experiment, fake_cli):
    fake_cli('hang')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    wait_status(client, exp_id, {'planning'})
    second = client.post(f'/api/experiments/{exp_id}/plan')
    assert second.status_code == 409
    assert '正在运行' in second.json()['detail']
    # train is also blocked while planning
    assert client.post(f'/api/experiments/{exp_id}/train').status_code == 409
    # clean up: cancel the hanging task
    client.post(f'/api/experiments/{exp_id}/cancel')
    wait_status(client, exp_id, {'cancelled'}, timeout=15)


# ---------- cancel ----------

def test_cancel_kills_cli_and_marks_cancelled(client, experiment, fake_cli):
    fake = fake_cli('hang')
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    wait_status(client, exp_id, {'planning'})
    wait_file(fake['pid_file'])
    response = client.post(f'/api/experiments/{exp_id}/cancel')
    assert response.status_code == 200
    assert response.json()['status'] in ('cancelling', 'cancelled')
    meta = wait_status(client, exp_id, {'cancelled'}, timeout=15)
    assert meta['error'] is None
    assert any('取消' in e['message'] for e in meta['events'])
    wait_dead(fake['pid_file'])
    assert client.get(f'/api/experiments/{exp_id}/artifacts/plan.json').status_code == 404


def test_cancel_idle_experiment_is_noop(client, experiment):
    exp_id = experiment['meta']['id']
    response = client.post(f'/api/experiments/{exp_id}/cancel')
    assert response.status_code == 200
    assert response.json()['status'] == 'draft'


# ---------- timeout ----------

def test_plan_timeout_kills_process_and_marks_failed(client, experiment, fake_cli, monkeypatch):
    fake = fake_cli('hang')
    monkeypatch.setenv('CLAUDE_TIMEOUT', '1')  # clamped to the 10 s floor by claude_runner
    exp_id = experiment['meta']['id']
    start_planning(client, exp_id)
    meta = wait_status(client, exp_id, {'failed', 'plan_ready'}, timeout=40)
    assert meta['status'] == 'failed'
    assert '超时' in meta['error']
    wait_file(fake['pid_file'])
    wait_dead(fake['pid_file'])


# ---------- restart interruption ----------

def test_restart_marks_in_flight_tasks_interrupted(client, app_instance, experiment, data_dir):
    exp_id = experiment['meta']['id']
    # simulate a crash while planning: status on disk is 'planning', no live thread
    app_instance.state.experiments.update(exp_id, status='planning')
    assert json.loads((data_dir / 'runs' / exp_id / 'meta.json').read_text())['status'] == 'planning'

    from app import create_app
    restarted = create_app(data_dir)  # Experiments.__init__ runs the recovery sweep
    meta = restarted.state.experiments.load(exp_id)
    assert meta['status'] == 'interrupted'
    assert '中断' in meta['error']
    # the API surface reflects the interrupted state
    with_client = client.get(f'/api/experiments/{exp_id}').json()
    assert with_client['status'] == 'interrupted'


# ---------- claude_runner helpers ----------

def test_redact_masks_keys_and_env_secrets(monkeypatch):
    import claude_runner
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'super-secret-value-xyz')
    text = claude_runner.redact('used key super-secret-value-xyz and sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV')
    assert 'super-secret-value-xyz' not in text
    assert 'sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV' not in text
    assert text.count('[REDACTED]') == 2


def test_binary_path_resolution(tmp_path, monkeypatch):
    import claude_runner
    fake = tmp_path / 'claude'
    fake.write_text('#!/bin/sh\nexit 0\n')
    fake.chmod(0o755)
    monkeypatch.setenv('CLAUDE_BIN', str(fake))
    assert claude_runner.binary_path() == str(fake)
    monkeypatch.setenv('CLAUDE_BIN', str(tmp_path / 'missing'))
    assert claude_runner.binary_path() is None  # explicit override never falls back
    monkeypatch.delenv('CLAUDE_BIN')
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    assert claude_runner.binary_path() is None
