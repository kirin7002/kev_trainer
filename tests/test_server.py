"""Exercise the real launcher on an isolated temporary project and socket."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import pytest

ROOT = Path(__file__).resolve().parents[1]


def isolated_launcher(tmp_path):
    script = tmp_path / 'server.py'
    shutil.copyfile(ROOT / 'server.py', script)
    (tmp_path / 'app.py').write_text(
        'from fastapi import FastAPI\napp=FastAPI()\n'
        '@app.get("/api/health")\ndef health(): return {"status":"ok"}\n')
    return script


@pytest.mark.parametrize('columns', ['70', '4096'])
def test_launch_duplicate_guard_stop_and_pid_cleanup(tmp_path, monkeypatch, columns):
    monkeypatch.setenv('COLUMNS', columns)
    script = isolated_launcher(tmp_path)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with (tmp_path / 'output.log').open('w') as output:
        process = subprocess.Popen([sys.executable, str(script), '--port', str(port)],
                                   cwd=tmp_path, stdout=output, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                assert process.poll() is None, (tmp_path / 'output.log').read_text()
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=.3) as response:
                        assert json.load(response) == {'status': 'ok'}
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise AssertionError('Launcher did not become ready')
            duplicate = subprocess.run([sys.executable, str(script), '--port', str(port)],
                                       capture_output=True, text=True, timeout=5)
            assert duplicate.returncode != 0
            assert '已运行' in duplicate.stdout + duplicate.stderr
            stopped = subprocess.run([sys.executable, str(script), '--stop'],
                                     capture_output=True, text=True, timeout=20)
            assert stopped.returncode == 0, stopped.stdout + stopped.stderr
            process.wait(timeout=10)
            assert not (tmp_path / '.run/server.json').exists()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def test_stop_refuses_an_unrelated_process_even_if_lock_is_held(tmp_path):
    script = isolated_launcher(tmp_path)
    run = tmp_path / '.run'
    run.mkdir()
    pid = os.getpid()
    identity = subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'args='], text=True).strip()
    (run / 'server.json').write_text(json.dumps({'pid': pid, 'identity': identity, 'port': 0}))
    with (run / 'server.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, str(script), '--stop'],
                                capture_output=True, text=True, timeout=5)
        assert result.returncode != 0
        assert '未停止任何服务' in result.stdout + result.stderr
        os.kill(pid, 0)
