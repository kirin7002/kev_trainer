"""最终审查发现的回归测试：仅使用 tmp 目录与 fake CLI，不调用真实模型。"""
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import claude_runner
import datasets
import evaluation
from datasets import DataError


def make_fake_cli(tmp_path, script_body, name='fake-claude'):
    """生成一个可执行 fake CLI（shell wrapper -> python 脚本），返回其路径。"""
    script = tmp_path / (name + '.py')
    script.write_text(script_body, encoding='utf-8')
    wrapper = tmp_path / name
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding='utf-8')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


# ---------------------------------------------------------------- 问题 1：stdin 取消/超时

SLEEP_CLI = '''
import time
time.sleep(60)
'''


def _run_cancel_case(tmp_path, monkeypatch, cancel_after, env=None):
    cli = make_fake_cli(tmp_path, SLEEP_CLI)
    monkeypatch.setenv('CLAUDE_BIN', str(cli))
    monkeypatch.delenv('CLAUDE_MODEL', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    # 60 万字符大输入；fake CLI 完全不读 stdin，写满管道缓冲区后 write 会阻塞。
    context = {'blob': 'x' * 600_000}
    cancel = threading.Event()
    timer = threading.Timer(cancel_after, cancel.set)
    timer.daemon = True  # 非 daemon 的 3600s Timer 会在测试全部结束后卡住解释器退出
    timer.start()
    threads_before = threading.active_count()
    start = time.monotonic()
    result = pytest.raises(claude_runner.PlanningError,
        lambda: claude_runner.generate_plan(context, tmp_path, 1.0, cancel, lambda text: None))
    elapsed = time.monotonic() - start
    # 不留阻塞线程：reader/writer 都应在进程终止后退出。
    deadline = time.monotonic() + 6
    while threading.active_count() > threads_before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= threads_before, '取消后仍有阻塞的后台线程'
    return result.value, elapsed


def test_generate_plan_cancel_with_huge_stdin_returns_promptly(tmp_path, monkeypatch):
    error, elapsed = _run_cancel_case(tmp_path, monkeypatch, cancel_after=0.1)
    assert isinstance(error, claude_runner.PlanningCancelled)
    assert elapsed < 5, f'取消耗时 {elapsed:.1f}s，stdin 发送未受取消约束'


def test_generate_plan_timeout_with_huge_stdin_unread(tmp_path, monkeypatch):
    # deadline 下限被钳制为 10s；进程只 sleep 不读 stdin，也必须超时返回而非永久卡住。
    error, elapsed = _run_cancel_case(tmp_path, monkeypatch, cancel_after=3600,
        env={'CLAUDE_TIMEOUT': '1'})
    assert not isinstance(error, claude_runner.PlanningCancelled)
    assert '超时' in str(error)
    assert elapsed < 20, f'超时耗时 {elapsed:.1f}s'


# ---------------------------------------------------------------- 问题 2：Noul questions 校验

def test_normalize_rejects_null_question_with_data_error():
    with pytest.raises(DataError):
        datasets.normalize([{'text': '开窗', 'questions': {'window': None}}])


def test_normalize_rejects_malformed_questions():
    for bad in (
        {'text': 't', 'questions': {'window': 'yes'}},                    # 非 dict
        {'text': 't', 'questions': {'window': {'type': 'other', 'label': True}}},  # type 非 noul
        {'text': 't', 'questions': {'window': {'type': 'noul'}}},          # 缺 label
        {'text': 't', 'questions': {'window': {'type': 'noul', 'label': 'true'}}},  # 字符串 label
        {'text': 't', 'questions': {'window': {'type': 'noul', 'label': 1}}},       # 非布尔
        {'text': 't', 'questions': {'': {'type': 'noul', 'label': True}}},  # 空对象名
    ):
        with pytest.raises(DataError):
            datasets.normalize([bad])


def test_normalize_rejects_bad_ids():
    base = {'questions': {'window': {'type': 'noul', 'label': True}}}
    for bad_id in (123, '', '   ', None, 'x' * 129):
        with pytest.raises(DataError):
            datasets.normalize([{'text': 't', 'id': bad_id, **base}])
    with pytest.raises(DataError):  # 重复 ID
        datasets.normalize([{'text': 't1', 'id': 'dup', **base}, {'text': 't2', 'id': 'dup', **base}])


def test_normalize_preserves_original_questions():
    questions = {
        'window': {'type': 'noul', 'label': True, 'instructions': '判断是否涉及车窗', 'criteria': 'c1'},
        'door': {'type': 'noul', 'label': False, 'instructions': '判断是否涉及车门'},
    }
    rows = [{'id': 'c1', 'text': '打开车窗', 'questions': questions}]
    out = datasets.normalize(rows)
    assert out[0]['intents'] == ['window']          # 显式 false 不当作正例
    assert out[0]['questions'] == questions          # instructions/criteria/false 标签均保留
    # 其余字段结构保持兼容
    assert set(out[0]) == {'id', 'text', 'intents', 'group_id', 'questions'}
    # 非 Noul 行不新增 questions 键
    plain = datasets.normalize([{'text': '开灯', 'intents': ['light']}])
    assert 'questions' not in plain[0]


def test_upload_bad_questions_returns_400_not_500(tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    application = app_module.create_app(tmp_path / 'state')
    payload = json.dumps([{'text': '开窗', 'questions': {'window': None}}]).encode()
    with TestClient(application) as client:
        response = client.post('/api/datasets', data={'role': 'train'},
            files={'file': ('bad.json', payload, 'application/json')})
    assert response.status_code == 400
    assert isinstance(response.json().get('detail'), str)


# ---------------------------------------------------------------- 问题 3：performance_comparable

CASES = [
    {'id': 'a', 'text': 't1', 'intents': ['x'], 'group_id': 'g1'},
    {'id': 'b', 'text': 't2', 'intents': [], 'group_id': 'g2'},
]
SHA = 'deadbeef'
RUNTIME = {'device': 'nvidia-gpu', 'backend': 'pytorch', 'dtype': 'bf16'}


def make_predictions(latencies, model='m', runtime=RUNTIME):
    return {'dataset_sha256': SHA, 'model': model, 'runtime': runtime,
        'source': 'user_imported_predictions',
        'predictions': {'a': {'intents': ['x'], 'latency_ms': latencies[0]},
                        'b': {'intents': [], 'latency_ms': latencies[1]}}}


def test_comparison_requires_full_latency_coverage_and_matching_runtime():
    baseline = make_predictions([10.0, None], model='base')   # 仅 1/2 条计时
    candidate = make_predictions([8.0, 9.0], model='cand')    # 完整 2/2 条
    report = evaluation.comparison(CASES, baseline, candidate)
    assert report['performance_comparable'] is False
    assert report['comparability_issues'], '缺少覆盖时必须给出说明'
    assert report['latency_coverage'] == {'total': 2, 'baseline': 1, 'candidate': 2}
    # 指标依然显示，不被隐藏
    assert report['baseline']['overall']['latency'] == {'count': 1, 'mean_ms': 10.0,
        'p50_ms': 10.0, 'p95_ms': 10.0}
    assert report['candidate']['overall']['latency']['count'] == 2
    assert report['accuracy_delta'] == 0.0

    full = evaluation.comparison(CASES, make_predictions([10.0, 12.0]), make_predictions([8.0, 9.0]))
    assert full['performance_comparable'] is True
    assert full['comparability_issues'] == []

    mismatch = evaluation.comparison(CASES, make_predictions([10.0, 12.0]),
        make_predictions([8.0, 9.0], runtime={'device': 'cpu', 'backend': 'pytorch', 'dtype': 'bf16'}))
    assert mismatch['performance_comparable'] is False
    assert mismatch['comparability_issues']

    no_runtime = evaluation.comparison(CASES, make_predictions([10.0, 12.0], runtime=None),
        make_predictions([8.0, 9.0], runtime=None))
    assert no_runtime['performance_comparable'] is False


# ---------------------------------------------------------------- 问题 4：子进程 flags/env

PLAN = {'summary': 'ok', 'diagnosis': [], 'steps': [{'title': 't', 'description': 'd', 'stage': 'prepare'}],
    'training': {'learning_rate': 1e-5, 'epochs': 1, 'batch_size': 1, 'gradient_accumulation': 1},
    'evaluation': {'metrics': [], 'safeguards': []}, 'notes': []}

ECHO_CLI = '''
import json, os, sys
plan = json.loads(os.environ['FAKE_PLAN'])
record = os.environ['FAKE_RECORD']
info = {'args': sys.argv[1:],
        'env': {k: os.environ.get(k) for k in
            ('CLAUDE_CODE_DISABLE_CLAUDE_MDS', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY')}}
with open(record, 'w', encoding='utf-8') as f:
    json.dump(info, f)
sys.stdin.read()
print(json.dumps({'type': 'system', 'subtype': 'init', 'model': 'fake-model'}), flush=True)
print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
    'result': json.dumps(plan), 'structured_output': plan,
    'session_id': 's1', 'total_cost_usd': 0.0}), flush=True)
'''


def test_planning_subprocess_disables_hooks_memory_claude_md_keeps_oauth(tmp_path, monkeypatch):
    cli = make_fake_cli(tmp_path, ECHO_CLI, name='fake-echo')
    record = tmp_path / 'argv.json'
    monkeypatch.setenv('CLAUDE_BIN', str(cli))
    monkeypatch.setenv('FAKE_PLAN', json.dumps(PLAN))
    monkeypatch.setenv('FAKE_RECORD', str(record))
    monkeypatch.delenv('CLAUDE_MODEL', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)  # 订阅 OAuth 路径，不走 --bare
    monkeypatch.setenv('CLAUDE_TIMEOUT', '60')

    result = claude_runner.generate_plan({'goal': 'g'}, tmp_path, 1.0, threading.Event(), lambda t: None)
    assert result['plan'] == PLAN and result['source'] == 'claude_code'

    info = json.loads(record.read_text())
    args, env = info['args'], info['env']
    # 保留 user 设置源（本机 OAuth 认证）
    assert args[args.index('--setting-sources') + 1] == 'user'
    # 显式禁用 hooks 与自动记忆
    settings = json.loads(args[args.index('--settings') + 1])
    assert settings.get('disableAllHooks') is True
    assert settings.get('autoMemoryEnabled') is False
    # CLAUDE.md 自动发现与自动记忆通过官方环境变量禁用，仅作用于子进程
    assert env['CLAUDE_CODE_DISABLE_CLAUDE_MDS'] == '1'
    assert env['CLAUDE_CODE_DISABLE_AUTO_MEMORY'] == '1'
    # 工具为空、OAuth 场景不启用 bare（bare 会禁用 OAuth/keychain）
    assert args[args.index('--tools') + 1] == ''
    assert '--bare' not in args
    # 输出不含任何 token 形态字符串
    assert 'sk-ant-' not in json.dumps(info)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
