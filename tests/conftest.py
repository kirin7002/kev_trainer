"""Shared fixtures. All state lives in tmp_path; no real Claude, server or GPU is used.

Imported before any test module: forces KEV_TRAINER_DATA to a throwaway directory so
`import app` (which builds a module-level app) never touches the repo's .state, and
forces CLAUDE_BIN to a nonexistent path so no test can accidentally spawn the real CLI.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_IMPORT_STATE = tempfile.mkdtemp(prefix='kev-import-state-')
os.environ['KEV_TRAINER_DATA'] = _IMPORT_STATE
os.environ['CLAUDE_BIN'] = str(ROOT / 'tests' / 'definitely-not-a-claude-binary')
os.environ.pop('CLAUDE_MODEL', None)

import pytest
from fastapi.testclient import TestClient

TRAIN_ROWS = [
    {'id': 'tr-1', 'text': '把前排车窗打开一点', 'intents': ['window']},
    {'id': 'tr-2', 'text': '车窗关上，座椅加热也关掉', 'intents': ['window', 'seat_heating']},
    {'id': 'tr-3', 'text': '打开空调', 'intents': ['air_conditioner']},
    {'id': 'tr-4', 'text': '音量调小一点', 'intents': ['volume']},
    {'id': 'tr-5', 'text': '今天天气不错', 'intents': []},
    {'id': 'tr-6', 'text': '打开天窗透透气', 'intents': ['sunroof']},
]

TEST_ROWS = [
    {'id': 'te-1', 'text': '帮我升起右侧的车窗', 'intents': ['window']},
    {'id': 'te-2', 'text': '空调和车窗都关上', 'intents': ['air_conditioner', 'window']},
    {'id': 'te-3', 'text': '谢谢你啦', 'intents': []},
]

VALID_PLAN = {
    'summary': '基于训练/验证资料提出的 LoRA 微调测试方案。',
    'diagnosis': ['单对象样本占多数', '存在 unknown 样本'],
    'steps': [
        {'title': '环境探测', 'description': '检查 GPU 与显存', 'stage': 'prepare'},
        {'title': '基线评估', 'description': '在验证集上评估原模型', 'stage': 'baseline'},
        {'title': 'LoRA 训练', 'description': '小步数试跑后正式训练', 'stage': 'train'},
        {'title': '验证', 'description': '验证集指标对比', 'stage': 'evaluate'},
        {'title': '最终对比', 'description': '独立测试集对比', 'stage': 'compare'},
    ],
    'training': {'learning_rate': 0.0001, 'epochs': 2, 'batch_size': 8, 'gradient_accumulation': 1},
    'evaluation': {'metrics': ['accuracy', 'macro_f1'], 'safeguards': ['测试集不进入训练上下文']},
    'notes': ['显存不足时降低 batch_size'],
}


@pytest.fixture()
def data_dir(tmp_path):
    directory = tmp_path / 'state'
    directory.mkdir()
    return directory


@pytest.fixture()
def app_instance(data_dir):
    from app import create_app
    return create_app(data_dir)


@pytest.fixture()
def client(app_instance):
    with TestClient(app_instance) as test_client:
        yield test_client


@pytest.fixture()
def upload(client):
    def uploader(role, rows=None, filename='data.json', raw=None):
        if raw is None:
            import json
            raw = json.dumps(rows, ensure_ascii=False).encode('utf-8')
        return client.post('/api/datasets', data={'role': role},
            files={'file': (filename, raw, 'application/octet-stream')})
    return uploader


@pytest.fixture()
def experiment(client, upload):
    """Uploaded train/test datasets plus a created draft experiment."""
    train = upload('train', TRAIN_ROWS, 'train.json')
    test = upload('test', TEST_ROWS, 'test.json')
    assert train.status_code == 201 and test.status_code == 201
    train_meta, test_meta = train.json(), test.json()
    created = client.post('/api/experiments', json={
        'name': '测试实验', 'train_dataset_id': train_meta['id'], 'test_dataset_id': test_meta['id'],
        'model_id': 'kev-0.8b-v1.0'})
    assert created.status_code == 201, created.text
    return {'train': train_meta, 'test': test_meta, 'meta': created.json()}


@pytest.fixture()
def legacy_experiment(experiment, app_instance):
    """Existing pre-protocol experiments keep their test-import workflow."""
    meta = dict(experiment['meta'])
    meta.pop('protocol_sha256')
    meta.pop('final_candidate')
    app_instance.state.experiments.save(meta)
    return {**experiment, 'meta': meta}
