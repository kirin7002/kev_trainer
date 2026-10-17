"""Owned subprocesses with one shared deadline, bounded logs and real cancellation."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
import signal

from claude_runner import stop_process


class ExecutionCancelled(RuntimeError):
    pass


class ExecutionTimeout(RuntimeError):
    pass


class StageFailed(RuntimeError):
    pass


def process_identity(pid):
    try:
        if Path('/proc').is_dir():
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+fields[19]
        return subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'args='], text=True, timeout=2).strip() or None
    except (OSError, subprocess.SubprocessError): return None


def stop_recorded_group(directory):
    """Recover only a journaled private session; a reused/unconfirmed PID is held."""
    journal = Path(directory)/'active-process.json'
    if not journal.exists(): return True
    try:
        record = json.loads(journal.read_text())
        pid = record.get('pid')
        if type(pid) is not int or pid <= 1 or not record.get('identity'): return False
        current = process_identity(pid)
        if current is not None and current != record['identity']: return False
        def members():
            output = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,sid=,uid=,stat='], text=True, timeout=2)
            active = []
            for line in output.splitlines():
                fields = line.split()
                if len(fields) == 5 and fields[1] == str(pid) and not fields[4].startswith('Z'):
                    if fields[2] != str(pid) or fields[3] != str(record['uid']): raise RuntimeError('session ownership mismatch')
                    active.append(int(fields[0]))
            return active
        if not members(): journal.unlink(missing_ok=True); return True
        for signum, seconds in ((signal.SIGTERM, 1), (signal.SIGKILL, 2)):
            try: os.killpg(pid, signum)
            except ProcessLookupError: pass
            until = time.monotonic()+seconds
            while time.monotonic() < until:
                if not members(): journal.unlink(missing_ok=True); return True
                time.sleep(.02)
        return False
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError): return False


@dataclass
class Deadline:
    seconds: float
    clock: object = time.monotonic

    def __post_init__(self):
        self.started = self.clock()
        self.ends = self.started + self.seconds

    def remaining(self):
        return max(0, self.ends - self.clock())

    def check(self):
        if self.remaining() <= 0:
            raise ExecutionTimeout('实验总时间预算已耗尽；本次进程已停止。')


class Supervisor:
    def __init__(self, directory, deadline, heartbeat, interval=2, log_limit=32*1024*1024):
        self.directory, self.deadline = Path(directory), deadline
        self.heartbeat, self.interval, self.log_limit = heartbeat, interval, log_limit
        self.records = []

    def check(self, stage):
        self.deadline.check()
        if self.heartbeat(stage).get('cancel_requested'):
            raise ExecutionCancelled('任务已取消，进程已停止。')

    def run(self, stage, argv, *, cwd, environment):
        if not isinstance(stage, str) or not stage or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for c in stage):
            raise StageFailed('阶段名称无效。')
        self.check(stage)
        logs = self.directory / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (stage + '.log')
        if log.exists():
            raise StageFailed('拒绝覆盖已有阶段日志。')
        started = time.monotonic()
        process = None
        try:
            with log.open('xb') as output:
                journal = self.directory / 'active-process.json'
                journal.write_text(json.dumps({'state': 'launching', 'stage': stage, 'pid': None}))
                process = subprocess.Popen(list(argv), cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                journal.write_text(json.dumps({'pid': process.pid, 'stage': stage, 'argv': list(argv),
                    'identity': process_identity(process.pid), 'uid': os.getuid(),
                    'started_at': time.time(), 'deadline_remaining': self.deadline.remaining()}))
                next_beat = 0
                while process.poll() is None:
                    self.deadline.check()
                    if log.stat().st_size > self.log_limit:
                        raise StageFailed('阶段日志超过上限。')
                    if time.monotonic() >= next_beat:
                        self.check(stage)
                        next_beat = time.monotonic() + self.interval
                    time.sleep(.02)
                self.check(stage)
                if process.returncode:
                    raise StageFailed(f'{stage} 进程失败（exit={process.returncode}），未晋升模型。')
                if log.stat().st_size > self.log_limit:
                    raise StageFailed('阶段日志超过上限。')
        finally:
            if process is not None:
                stop_process(process)
                if not stop_recorded_group(self.directory):
                    raise StageFailed('无法确认阶段进程组已停止；保留日志并保持租约占槽。')
            else:
                (self.directory / 'active-process.json').unlink(missing_ok=True)
            self.records.append({'stage': stage, 'wall_seconds': time.monotonic()-started,
                                 'exit_code': process.returncode if process is not None else None})
