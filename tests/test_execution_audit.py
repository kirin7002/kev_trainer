"""Regressions from the independent second-pass execution review."""
import json

import pytest

from datasets import DataError
from model_artifacts import inspect_bundle, file_sha
from pipeline_commit import validate_training_bundle
from test_phase_one import task50, partitioned, manual, new_trial
from test_gpu_execution import gpu
from trial_execution import TrialPipeline, read, write
from fake_gpu_backend import FakeBackend
from worker import run_once


def prepared(client, manual, gpu, tmp_path):
    queue, _, (tw,_) = gpu
    trial = new_trial(client, manual)
    queue.enqueue(trial['id'],'gpu_training_v1'); job = queue.claim(tw['id'])
    inputs = queue.inputs(job['id'],tw['id'],job['lease_id'])
    result = TrialPipeline(FakeBackend(),tmp_path/'pipeline',lambda _: {},30,interval=.01).run(inputs,job)
    return queue, trial, job, inputs, result


def test_exported_policy_must_equal_verified_calibration_receipt(client, manual, gpu, tmp_path):
    queue,trial,job,inputs,result = prepared(client,manual,gpu,tmp_path)
    manifest = inspect_bundle(result['artifact_path'],destination=tmp_path/'extracted')
    policy = tmp_path/'extracted/policy/candidate-calibration.json'
    value = read(policy); value['temperature'] = 2.; write(policy,value)
    with pytest.raises(DataError,match='实际导出策略'):
        validate_training_bundle(tmp_path/'extracted',manifest,trial,inputs,True)


@pytest.mark.parametrize('tamper', ['bundle','cached_verification'])
def test_pending_promotion_rechecks_immutable_bytes(client, manual, gpu, tmp_path, monkeypatch, tamper):
    queue,trial,job,inputs,result = prepared(client,manual,gpu,tmp_path)
    queue.accept_artifact(job['id'],job['worker_id'],job['lease_id'],result['artifact_path'])
    original = queue.reconcile_publish
    monkeypatch.setattr(queue,'reconcile_publish',lambda: None)
    queue.finish(job['id'],job['worker_id'],job['lease_id'],{'state':'SUCCEEDED','process_stopped':True,'result':result})
    stored = queue.read('jobs',job['id'])
    root = queue.artifact_root(job['id'],result['artifact_sha256'])
    path = root/('bundle.zip' if tamper == 'bundle' else 'verified.json')
    path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(queue,'reconcile_publish',original); queue.reconcile_publish()
    assert queue.read('jobs',job['id'])['promotion_pending'] is True
    assert queue.read('jobs',job['id']).get('publication_error')
    assert not queue.read('trials',trial['id']).get('active_model')
    assert queue.claim(job['worker_id']) is None


def test_duplicate_upload_cannot_reauthorize_changed_cache(client, manual, gpu, tmp_path):
    queue,trial,job,inputs,result = prepared(client,manual,gpu,tmp_path)
    queue.accept_artifact(job['id'],job['worker_id'],job['lease_id'],result['artifact_path'])
    root = queue.artifact_root(job['id'],result['artifact_sha256'])
    path = root/'verified.json'; path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(DataError,match='重复上传'):
        queue.accept_artifact(job['id'],job['worker_id'],job['lease_id'],result['artifact_path'])


def test_no_stop_ack_when_process_group_confirmation_fails(client, manual, gpu, tmp_path, monkeypatch):
    import worker_execution
    queue,(trainer,_),(tw,_) = gpu
    trial = new_trial(client,manual); queue.enqueue(trial['id'],'gpu_training_v1')
    monkeypatch.setattr(worker_execution,'stop_recorded_group',lambda directory:False)
    result = run_once(trainer,tw['id'],backend=FakeBackend(fail='SMOKE'),work_root=tmp_path/'worker',interval=.05)
    assert result['state'] == 'HELD_FOR_PROCESS_CONFIRMATION'
    assert queue.jobs()[0]['state'] == 'RUNNING' and queue.claim(tw['id']) is None
    assert list((tmp_path/'worker').glob('attempt-*/lease.json'))


def test_controller_clock_rollback_cannot_make_retries_free(client, manual, gpu):
    queue,_,(tw,_) = gpu
    clock = [100.]; queue.clock = lambda:clock[0]
    trial = new_trial(client,manual); queue.enqueue(trial['id'],'gpu_training_v1'); job = queue.claim(tw['id'])
    clock[0] = 105.; queue.heartbeat(job['id'],tw['id'],job['lease_id'])
    clock[0] = 103.
    beat = queue.heartbeat(job['id'],tw['id'],job['lease_id'])
    assert beat['cancel_requested'] and beat['remaining_seconds'] == 0
    queue.finish(job['id'],tw['id'],job['lease_id'],{'state':'FAILED','process_stopped':True,'result':{'kind':job['kind']}})
    assert queue.budget(manual['id'])['charged_seconds'] == 1800
    with pytest.raises(DataError): queue.enqueue(trial['id'],'gpu_training_v1')


def test_trial_rejects_coordinated_run_manifest_replacement(client, manual, gpu):
    from model_artifacts import canonical_sha
    queue,_,_ = gpu
    trial = new_trial(client,manual)
    directory = queue.store.path('runs',manual['id'])
    run = queue.store.read(directory/'meta.json')
    run['manifest']['seed'] = 999
    run['manifest_sha256'] = canonical_sha(run['manifest'])
    queue.store.write(directory/'data-manifest.json',run['manifest']);queue.store.write(directory/'meta.json',run)
    with pytest.raises(DataError,match='冻结 Trial'):
        queue.phase.worker_inputs(trial['id'])
