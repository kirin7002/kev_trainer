"""Real HTTP + launcher acceptance, with fixture predictions and no model/GPU calls."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import httpx
import pytest


def test_real_service_dataset_report_restart_and_stop(tmp_path):
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / 'project'
    project.mkdir()
    for source in list(root.glob('*.py')) + [root/'models.json', root/'kev-source-contract.json']:
        shutil.copyfile(source, project / source.name)
    for name in ('templates', 'examples', 'scripts'):
        shutil.copytree(root / name, project / name)
    (project / '.venv').symlink_to(Path(sys.prefix), target_is_directory=True)
    environment = dict(os.environ, KEV_TRAINER_DATA=str(tmp_path / 'state'),
                       CLAUDE_BIN=str(tmp_path / 'absent-claude'),
                       KEV_WORKER_TOKEN='http-acceptance-ephemeral-fixture-' + 'x' * 32)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base_url = f'http://127.0.0.1:{port}'

    def start(output):
        process = subprocess.Popen(['bash', str(project / 'scripts/start.sh'), '--port', str(port)],
                                   cwd=project, env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                assert process.poll() is None, (tmp_path / 'service.log').read_text()
                try:
                    response = http.get('/api/health')
                    assert response.status_code == 200
                    return process
                except httpx.TransportError:
                    time.sleep(.05)
            pytest.fail('Real service did not become ready')
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise

    def stop(process):
        try:
            stopped = subprocess.run(['bash', str(project / 'scripts/stop.sh')], env=environment,
                                     capture_output=True, text=True, timeout=20)
            assert stopped.returncode == 0, stopped.stdout + stopped.stderr
            process.wait(timeout=10)
            assert not (project / '.run/server.json').exists()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)

    with (tmp_path / 'service.log').open('w') as output, httpx.Client(
            base_url=base_url, trust_env=False, timeout=2) as http:
        process = start(output)
        try:
            health = http.get('/api/health').json()
            assert health['machine']['connected'] is False
            assert health['claude']['installed'] is False
            assert 'Kev Trainer' in http.get('/').text
            datasets = http.post('/api/examples').json()
            created = http.post('/api/experiments', json={
                'name': 'HTTP 验收（夹具预测，非训练）', 'model_id': 'kev-0.8b-v1.0',
                'train_dataset_id': datasets[0]['id'], 'test_dataset_id': datasets[1]['id']})
            assert created.status_code == 201
            meta = created.json()
            exp_id = meta['id']
            assert meta['split'] == {'train': 8, 'validation': 2, 'test': 5, 'seed': 42}
            assert http.post(f'/api/experiments/{exp_id}/plan').status_code == 409
            assert http.post(f'/api/experiments/{exp_id}/train').status_code == 409
            cases = json.loads((project / 'examples/test.json').read_text())
            for kind in ('baseline', 'candidate'):
                payload = {'dataset_sha256': datasets[1]['sha256'], 'model': 'fixture-' + kind,
                           'predictions': [{'id': row['id'], 'intents': row['intents'] if kind == 'candidate' else []}
                                           for row in cases]}
                imported = http.post(f'/api/experiments/{exp_id}/predictions/{kind}',
                                     files={'file': ('fixture.json', json.dumps(payload), 'application/json')})
                assert imported.status_code == 200
            report = http.get(f'/api/experiments/{exp_id}/report').json()
            assert report['baseline']['overall']['accuracy'] == .2
            assert report['candidate']['overall']['accuracy'] == 1
            assert report['accuracy_delta'] == .8
            assert report['training_verified'] is False
            assert report['performance_comparable'] is False
            assert len(report['improved_ids']) == 4
            assert http.get(f'/api/experiments/{exp_id}/artifacts/report.json').json() == report
            # Exercise the new manual preparation path through the real HTTP
            # server and the actual reference Worker CLI (CPU process only).
            task = http.post('/api/task-specs', json={'name': 'HTTP synthetic task', 'objects': [
                {'id': f'object-{i:02}', 'instructions': f'Synthetic object {i}',
                 'criteria': {'true': 'yes fixture', 'false': 'no fixture'}} for i in range(50)]}).json()
            partitions = {}
            for role in ('train', 'calibration', 'validation', 'test'):
                fixture = [{'id': 'http-' + role + str(i), 'text': 'http manual ' + role + str(i),
                            'intents': ['object-00'] if i == 0 else []} for i in range(2)]
                uploaded = http.post('/api/datasets', data={'role': role, 'task_spec_id': task['id']},
                                     files={'file': (role + '.json', json.dumps(fixture))})
                assert uploaded.status_code == 201, uploaded.text
                partitions[role] = uploaded.json()['id']
            manual = http.post('/api/manual-experiments', json={'name': 'CPU preparation acceptance',
                'task_spec_id': task['id'], 'partitions': partitions, 'max_trials': 1, 'max_training_minutes': 5})
            assert manual.status_code == 201, manual.text
            trial = http.post('/api/experiments/' + manual.json()['id'] + '/trials', json={
                'learning_rate': .0001, 'epochs': 1, 'batch_size': 1, 'gradient_accumulation': 1,
                'seed': 5, 'precision': 'fp32'})
            assert trial.status_code == 201, trial.text
            trial_id = trial.json()['id']
            queued = http.post('/api/trials/' + trial_id + '/preflight')
            assert queued.status_code == 202
            worker = subprocess.run([sys.executable, str(project / 'worker.py'), '--server', base_url, '--once'],
                cwd=project, env=environment, capture_output=True, text=True, timeout=15)
            assert worker.returncode == 0, worker.stdout + worker.stderr
            worker_result = json.loads(worker.stdout.strip().splitlines()[-1])
            assert worker_result['state'] == 'SUCCEEDED'
            assert worker_result['result']['training_ready'] is False
            assert worker_result['training_verified'] is False
            assert environment['KEV_WORKER_TOKEN'] not in worker.stdout + worker.stderr
            assert http.get('/api/trials/' + trial_id + '/preparation.zip').status_code == 200
            assert http.post('/api/trials/' + trial_id + '/train').status_code == 409
        finally:
            stop(process)

        # Derived report loss must not lose/relabel the persisted external predictions.
        (tmp_path / 'state/runs' / exp_id / 'report.json').unlink()
        process = start(output)
        try:
            assert http.get(f'/api/experiments/{exp_id}/report').json() == report
            assert http.get(f'/api/experiments/{exp_id}').json()['status'] == 'draft'
        finally:
            stop(process)
