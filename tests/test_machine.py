"""Offline GPU contract: no connection, no fabricated jobs, logs or metrics."""
import pytest

from machine import (Command, ConnectionStatus, JobStatus, LogChunk, MachineUnavailable,
                     UnconfiguredMachineClient, get_machine_client)


def test_get_machine_client_is_unconfigured():
    client = get_machine_client()
    assert isinstance(client, UnconfiguredMachineClient)
    status = client.check_connection()
    assert isinstance(status, ConnectionStatus)
    assert status.connected is False
    assert 'GPU' in status.message


def test_all_operations_raise_machine_unavailable(tmp_path):
    client = get_machine_client()
    with pytest.raises(MachineUnavailable):
        client.upload(tmp_path / 'x', 'remote/x')
    with pytest.raises(MachineUnavailable):
        client.download('remote/x', tmp_path / 'x')
    with pytest.raises(MachineUnavailable):
        client.start(Command(argv=('python', 'train.py')))
    with pytest.raises(MachineUnavailable):
        client.status('job-1')
    with pytest.raises(MachineUnavailable):
        client.logs('job-1', 0)
    with pytest.raises(MachineUnavailable):
        client.cancel('job-1')
    # and it never writes anything locally while refusing
    assert not (tmp_path / 'x').exists()


def test_start_returns_no_job_object():
    # an offline client must not invent job ids/states
    client = get_machine_client()
    with pytest.raises(MachineUnavailable) as info:
        client.start(Command(argv=('true',)))
    assert not isinstance(info.value, JobStatus)
    assert info.value.args  # carries the human-readable message


def test_contract_dataclasses_frozen():
    status = ConnectionStatus(connected=True, message='x')
    with pytest.raises(Exception):
        status.connected = False
    job = JobStatus(job_id='j', state='running')
    assert job.exit_code is None
    chunk = LogChunk(text='t', next_offset=1, eof=False)
    assert chunk.next_offset == 1
