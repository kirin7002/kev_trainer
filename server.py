"""Single-process local server launcher for Linux, WSL2 and macOS."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / '.run'
PID = RUN / 'server.json'
LOCK = RUN / 'server.lock'


def process_identity(pid):
    # CI and terminals can set a narrow COLUMNS value. A truncated argv cannot
    # prove this launcher's path; request the full command before comparing it.
    info = subprocess.run(['ps', '-ww', '-p', str(pid), '-o', 'lstart=', '-o', 'args='],
                          capture_output=True, text=True, check=False)
    return info.stdout.strip()


def stop():
    RUN.mkdir(exist_ok=True)
    with LOCK.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            PID.unlink(missing_ok=True)
            print('Kev Trainer 未运行。')
            return
        try:
            data = json.loads(PID.read_text())
            pid = data['pid']
            if type(pid) is not int or pid <= 1 or not data.get('identity'):
                raise ValueError('Invalid PID record')
            if data['identity'] != process_identity(pid) or str(ROOT / 'server.py') not in data['identity']:
                raise ValueError('PID belongs to another process')
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit('无法确认进程归属，未停止任何服务。请检查 .run/server.json。') from error
        os.kill(pid, signal.SIGTERM)
        for _ in range(150):
            if process_identity(pid) != data['identity']:
                # Uvicorn may re-raise SIGTERM before the launcher's finally block.
                # Clean only the record for the process we just stopped.
                try:
                    if json.loads(PID.read_text()) == data:
                        PID.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
                print('Kev Trainer 已停止。')
                return
            time.sleep(.1)
        raise SystemExit('停止尚未完成，请检查 .run/server.log；没有强制终止其他进程。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--port', type=int, default=int(os.environ.get('KEV_TRAINER_PORT', '8091')))
    args = parser.parse_args()
    if args.stop:
        return stop()
    if not 1 <= args.port <= 65535:
        raise SystemExit('Invalid port')
    RUN.mkdir(exist_ok=True)
    with LOCK.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Kev Trainer 已运行，请先执行 scripts/stop.sh。')
        pid = os.getpid()
        PID.write_text(json.dumps({'pid': pid, 'identity': process_identity(pid), 'port': args.port}))
        try:
            import uvicorn
            print(f'Kev Trainer: http://127.0.0.1:{args.port}', flush=True)
            uvicorn.run('app:app', host='127.0.0.1', port=args.port, workers=1)
        finally:
            try:
                if json.loads(PID.read_text()).get('pid') == pid:
                    PID.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass


if __name__ == '__main__':
    main()
