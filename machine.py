"""GPU machine contract. No SSH connection or remote training is implemented yet."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


class MachineUnavailable(RuntimeError):
    """Raised before execution when no GPU machine is connected."""


@dataclass(frozen=True)
class ConnectionStatus:
    connected: bool
    message: str


@dataclass(frozen=True)
class Command:
    """Argument vector; cwd is relative to the experiment's remote workspace."""

    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: int = 7200


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown"]
    exit_code: int | None = None


@dataclass(frozen=True)
class LogChunk:
    text: str
    next_offset: int
    eof: bool


class MachineClient(Protocol):
    """One client is scoped to one experiment workspace.

    Future adapters must enforce that scope for files and commands, retain jobs
    across caller disconnects, and return real process state and byte offsets.
    Cancellation completes only after the remote process has stopped.
    """

    def check_connection(self) -> ConnectionStatus: ...
    def upload(self, local_path: Path, remote_path: str) -> None: ...
    def download(self, remote_path: str, local_path: Path) -> None: ...
    def start(self, command: Command) -> JobStatus: ...
    def status(self, job_id: str) -> JobStatus: ...
    def logs(self, job_id: str, offset: int = 0) -> LogChunk: ...
    def cancel(self, job_id: str) -> None: ...


class UnconfiguredMachineClient:
    """Offline placeholder: never invent jobs, logs, metrics or model files."""

    message = "旧方案执行入口未接入 GPU；请在手动 Trial 中使用显式配置的拉取 Worker。当前可准备数据、方案与外部预测评测。"

    def check_connection(self) -> ConnectionStatus:
        return ConnectionStatus(connected=False, message=self.message)

    def upload(self, local_path: Path, remote_path: str) -> None:
        raise MachineUnavailable(self.message)

    def download(self, remote_path: str, local_path: Path) -> None:
        raise MachineUnavailable(self.message)

    def start(self, command: Command) -> JobStatus:
        raise MachineUnavailable(self.message)

    def status(self, job_id: str) -> JobStatus:
        raise MachineUnavailable(self.message)

    def logs(self, job_id: str, offset: int = 0) -> LogChunk:
        raise MachineUnavailable(self.message)

    def cancel(self, job_id: str) -> None:
        raise MachineUnavailable(self.message)


def get_machine_client() -> MachineClient:
    """Replace with the real adapter after a GPU host becomes reachable."""
    return UnconfiguredMachineClient()
