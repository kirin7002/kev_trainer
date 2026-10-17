"""Baseline-only and image evidence contracts. No real model, image engine or GPU."""
import json
import os
from pathlib import Path

import pytest

from datasets import DataError
from kev_runtime import image_provenance
from baseline_execution import BASELINE_STAGES,BaselinePipeline
from baseline_commit import validate_baseline_bundle
from model_artifacts import inspect_bundle
from test_phase_one import task50,partitioned,manual
from test_gpu_execution import gpu
from fake_gpu_backend import FakeBackend
from worker import run_once
from optimization import Optimization
from trial_execution import read,write


@pytest.fixture()
def baseline_gpu(gpu):
    queue,clients,workers=gpu
    worker=queue.register({'worker_id':workers[0]['id'],'name':'baseline fixture','platform':'fixture','python_version':'3.12',
        'capabilities':['gpu_training_v1','gpu_baseline_v1'],'runtime':FakeBackend.runtime()})
    return queue,clients,(worker,workers[1])


def test_baseline_only_full_chain_before_any_trial_and_agent_safe_inputs(client,app_instance,manual,baseline_gpu,tmp_path):
    queue,(trainer,_),(worker,_)=baseline_gpu
    response=client.post('/api/experiments/'+manual['id']+'/baseline')
    assert response.status_code==202,response.text
    assert response.json()['trial_id'] is None
    assert queue.phase.trials(manual['id'])==[]
    backend=FakeBackend()
    result=run_once(trainer,worker['id'],backend=backend,work_root=tmp_path/'worker',interval=.05)
    assert result['state']=='SUCCEEDED',result
    assert result['training_verified'] is False and result['baseline_verified'] is False
    assert [v[0] for v in backend.calls]==list(BASELINE_STAGES[:-1])
    assert all(v[1]!='train' for v in backend.calls)
    assert queue.phase.trials(manual['id'])==[] and queue.budget(manual['id'])['charged_seconds']>0
    report=client.get('/api/experiments/'+manual['id']+'/baseline').json()
    assert report['source']=='software_contract_fixture' and report['optimizer_steps']==0
    assert report['training_verified'] is False
    assert report['runtime']['image_provenance']['status']=='unknown'
    optimizer=app_instance.state.optimization
    session=optimizer.create(manual['id'],{'claude_budget_usd':0})
    inputs=optimizer.inputs(session['id'])
    assert inputs['baseline_metrics']['available'] is True
    assert inputs['baseline_metrics']['training_verified'] is False
    assert inputs['budget']['remaining_trials']==2
    assert 'test unique phrase' not in json.dumps(inputs)
    assert 'train unique phrase' not in json.dumps(queue.baseline_inputs(queue.jobs()[0]))
    assert not (queue.store.root/'evaluation-usage.json').exists()


@pytest.mark.parametrize('failure',['BASELINE_CALIBRATION','BASELINE_CALIBRATING','BASELINE_LOADING'])
def test_failed_baseline_does_not_replace_accepted_baseline(client,manual,baseline_gpu,tmp_path,failure):
    queue,(trainer,_),(worker,_)=baseline_gpu
    queue.enqueue_baseline(manual['id'])
    assert run_once(trainer,worker['id'],backend=FakeBackend(),work_root=tmp_path/'worker',interval=.05)['state']=='SUCCEEDED'
    path=queue.store.path('runs',manual['id'])/'meta.json';previous=queue.store.read(path)['baseline_evaluation']
    charge=queue.budget(manual['id'])['charged_seconds']
    queue.enqueue_baseline(manual['id'])
    assert run_once(trainer,worker['id'],backend=FakeBackend(fail=failure),work_root=tmp_path/'worker',interval=.05)['state']=='FAILED'
    assert queue.store.read(path)['baseline_evaluation']==previous
    assert queue.budget(manual['id'])['charged_seconds']>charge


def test_baseline_inputs_stages_and_cancel_are_isolated(client,manual,baseline_gpu):
    queue,_,(worker,evaluator)=baseline_gpu
    queued=queue.enqueue_baseline(manual['id']);assert queue.claim(evaluator['id']) is None
    job=queue.claim(worker['id']);inputs=queue.inputs(job['id'],worker['id'],job['lease_id'])
    assert set(inputs['partitions'])=={'calibration','validation'} and 'parameters' not in inputs['config']
    with pytest.raises(DataError):queue.heartbeat(job['id'],worker['id'],job['lease_id'],'TRAINING')
    queue.cancel(job['id']);assert queue.claim(worker['id']) is None
    queue.finish(job['id'],worker['id'],job['lease_id'],{'state':'CANCELLED','process_stopped':True,'result':{'kind':job['kind']}})
    assert queue.read('jobs',job['id'])['state']=='CANCELLED'


def test_baseline_controller_rejects_training_claim_and_policy_changes(client,manual,baseline_gpu,tmp_path):
    queue,_,(worker,_)=baseline_gpu
    queue.enqueue_baseline(manual['id']);job=queue.claim(worker['id']);inputs=queue.inputs(job['id'],worker['id'],job['lease_id'])
    result=BaselinePipeline(FakeBackend(),tmp_path/'baseline',lambda _: {},30,interval=.01).run(inputs,job)
    manifest=inspect_bundle(result['artifact_path'],destination=tmp_path/'contents')
    changed=dict(manifest,training_verified=True)
    with pytest.raises(DataError):validate_baseline_bundle(tmp_path/'contents',changed,job,inputs,True)
    path=tmp_path/'contents/policy/baseline-calibration.json';policy=read(path);policy['train_sha256']='a'*64;write(path,policy)
    with pytest.raises(DataError):validate_baseline_bundle(tmp_path/'contents',manifest,job,inputs,True)


def test_missing_image_metadata_stays_unknown_and_env_is_not_proof(tmp_path,monkeypatch):
    monkeypatch.setenv('KEV_IMAGE_DIGEST','sha256:'+'a'*64)
    result=image_provenance(tmp_path/'missing')
    assert result['status']=='unknown' and result['manifest_digest'] is None and result['config_digest'] is None
    assert not result['independently_attested']


def test_runtime_config_digest_is_not_manifest_digest(tmp_path):
    path=tmp_path/'containerenv';path.write_text('engine="podman-5.0"\nimageid="'+('a'*64)+'"\nimage="registry.example/kev:tag"\n');path.chmod(0o600)
    result=image_provenance(path)
    assert result['config_digest']=='sha256:'+'a'*64 and result['manifest_digest'] is None
    assert result['source']=='podman_containerenv' and not result['independently_attested']
    path.write_text('engine="podman-5.0"\nimageid="'+('a'*64)+'"\nimage="registry.example/kev@sha256:'+('b'*64)+'"\n')
    result=image_provenance(path);assert result['manifest_digest']=='sha256:'+'b'*64


@pytest.mark.parametrize('kind',['symlink','writable','oversized','bad-digest','wrong-engine'])
def test_image_evidence_rejects_untrusted_metadata(tmp_path,kind):
    path=tmp_path/'containerenv';text='engine="podman-5"\nimageid="'+('a'*64)+'"\n'
    if kind=='symlink':
        target=tmp_path/'target';target.write_text(text);path.symlink_to(target)
    else:
        path.write_text('x'*20000 if kind=='oversized' else text.replace('a'*64,'invalid') if kind=='bad-digest' else text.replace('podman-5','fake') if kind=='wrong-engine' else text)
        path.chmod(0o666 if kind=='writable' else 0o600)
    result=image_provenance(path)
    assert result['status']=='unknown' and result['manifest_digest'] is None


def test_lora_resume_refuses_before_loading_any_training_library(tmp_path):
    from kev_runtime import observed_train
    with pytest.raises(ValueError,match='no optimizer resume'):
        observed_train({'work':str(tmp_path),'config':{'resume_from':'adapter-without-optimizer'},'stage':'TRAINING'})


def test_baseline_expired_lease_keeps_slot_and_charges_shared_budget(client,manual,baseline_gpu,tmp_path):
    from worker_queue import LeaseConflict
    queue,_,(worker,_)=baseline_gpu;clock=[100.];queue.clock=lambda:clock[0]
    queue.enqueue_baseline(manual['id']);job=queue.claim(worker['id']);clock[0]+=31
    assert queue.claim(worker['id']) is None
    with pytest.raises(LeaseConflict):queue.accept_artifact(job['id'],worker['id'],job['lease_id'],tmp_path/'absent.zip')
    queue.stopped(job['id'],worker['id'],job['lease_id'],True)
    assert queue.budget(manual['id'])['charged_seconds']==31
    assert queue.phase.trials(manual['id'])==[]


def test_baseline_rejects_image_provenance_change_before_promotion(client,manual,baseline_gpu,tmp_path):
    queue,(trainer,_),(worker,_)=baseline_gpu
    class ChangedRuntime(FakeBackend):
        def perform(self,stage,operation,payload,supervisor):
            value=super().perform(stage,operation,payload,supervisor)
            if operation=='validate-baseline': value['runtime']['image_provenance']['reason']='changed fixture identity'
            return value
    queue.enqueue_baseline(manual['id'])
    result=run_once(trainer,worker['id'],backend=ChangedRuntime(),work_root=tmp_path/'worker',interval=.05)
    assert result['state']=='FAILED',result
    run=queue.store.read(queue.store.path('runs',manual['id'])/'meta.json')
    assert not run.get('baseline_evaluation')


def test_baseline_cannot_run_after_external_test_import(client,manual,baseline_gpu):
    queue,_,_=baseline_gpu
    queue.store.write(queue.store.path('runs',manual['id'])/'baseline-predictions.json',{'fixture':'final test exposed'})
    with pytest.raises(DataError,match='test'):queue.enqueue_baseline(manual['id'])


def test_offline_baseline_returns_conflict_and_creates_no_job(client,manual,app_instance):
    response=client.post('/api/experiments/'+manual['id']+'/baseline')
    assert response.status_code==409
    assert app_instance.state.worker_queue.jobs()==[]
