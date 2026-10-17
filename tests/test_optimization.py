"""Phase-two offline acceptance. Fake CLI and fixture backend; no paid model/GPU."""
import copy
import json
import sys
import time

import pytest

from datasets import DataError
from model_artifacts import canonical_sha
from optimization import Optimization
from test_phase_one import task50, partitioned, manual, PARAMETERS
from test_gpu_execution import gpu
from worker import run_once
from fake_gpu_backend import FakeBackend


def proposal(count=2, **stop):
    return {'schema_version':2, 'objective':'Synthetic validation-only search', 'hypotheses':['Fixture contract, not promised model quality'],
        'trials':[{'id':'candidate-'+str(i),'parameters':{**PARAMETERS,'seed':42+i},'rationale':'Fixture '+str(i)} for i in range(count)],
        'stop_conditions':{'metric':'validation_accuracy','target_accuracy':1.,'min_improvement':0.,'patience':3,'max_baseline_regression':.1,**stop}}


@pytest.fixture()
def optimizer(client, app_instance):
    obj = app_instance.state.optimization
    obj.stop.set(); obj.thread.join(timeout=2)
    return obj


def imported(optimizer, manual, value=None, budget=0):
    session = optimizer.create(manual['id'], {'claude_budget_usd':budget})
    return optimizer.import_plan(session['id'], value or proposal())


def approve(optimizer, session):
    return optimizer.review(session['id'], {'plan_sha256':session['plan_sha256'],'decision':'approve',
        'reviewer':'fixture local reviewer', 'notes':'Reviewed exact parameters, budget and stopping rules; synthetic test'})


def test_six_inputs_no_final_data_and_explicit_approval(client, optimizer, manual, partitioned):
    session = imported(optimizer, manual)
    inputs = client.get('/api/optimizations/'+session['id']+'/inputs').json()
    assert set(inputs) == {'task_spec','data_profile','baseline_metrics','validation_errors','allowed_search_space','budget'}
    encoded = json.dumps(inputs)
    for role in ('test','regression'):
        assert partitioned[0][role]['id'] not in encoded
        assert partitioned[0][role]['sha256'] not in encoded
        assert role+' unique phrase' not in encoded
    assert client.post('/api/optimizations/'+session['id']+'/start').status_code == 400
    assert not optimizer.phase.trials(manual['id'])
    with pytest.raises(DataError): optimizer.review(session['id'], {'plan_sha256':'0'*64,'decision':'approve','reviewer':'x','notes':'x'})
    approved = approve(optimizer, session)
    assert approved['state'] == 'APPROVED' and len(approved['trial_ids']) == 2
    assert optimizer.queue.jobs() == []
    assert optimizer.start(session['id'])['state'] == 'WAITING_RESOURCE'
    assert optimizer.queue.jobs() == []
    assert client.get('/api/optimizations/'+session['id']+'/plan.md').status_code == 200


@pytest.mark.parametrize('mutation', ['command','rank','resume','preprocessing','seed_float','duplicate_id','duplicate_parameters','test_metric','over_count','nonfinite'])
def test_independent_plan_rejects_unsafe_or_incompatible_fields(optimizer, manual, mutation):
    value = proposal()
    if mutation in ('command','rank','resume'): value['trials'][0]['parameters'][mutation] = 'forbidden'
    elif mutation == 'preprocessing': value['preprocessing.patch'] = 'forbidden'
    elif mutation == 'seed_float': value['trials'][0]['parameters']['seed'] = 42.
    elif mutation == 'duplicate_id': value['trials'][1]['id'] = value['trials'][0]['id']
    elif mutation == 'duplicate_parameters': value['trials'][1]['parameters'] = dict(value['trials'][0]['parameters'])
    elif mutation == 'test_metric': value['stop_conditions']['metric'] = 'test_accuracy'
    elif mutation == 'over_count': value = proposal(3)
    else: value['stop_conditions']['min_improvement'] = float('nan')
    session = optimizer.create(manual['id'], {'claude_budget_usd':0})
    with pytest.raises((DataError, ValueError)): optimizer.import_plan(session['id'], value)
    assert not optimizer.phase.trials(manual['id'])


def test_serial_two_candidates_choose_validation_and_never_final_test(optimizer, manual, gpu, tmp_path):
    queue, (trainer,_), (tw,_) = gpu
    session = approve(optimizer, imported(optimizer, manual))
    current = optimizer.start(session['id'])
    assert len(queue.jobs()) == 1
    for i in range(2):
        assert run_once(trainer, tw['id'], backend=FakeBackend(), work_root=tmp_path/'worker', interval=.05)['state'] == 'SUCCEEDED'
        current = optimizer.advance(session['id'])
        assert len(queue.jobs()) == 2
    assert current['state'] == 'AWAITING_FINAL_REVIEW' and current['stop_reason'] == 'candidates_exhausted'
    assert len(current['history']) == 2 and not current['training_verified']
    assert current['best_trial_id'] == session['trial_ids'][0]
    run = optimizer.store.read(optimizer.store.path('runs', manual['id'])/'meta.json')
    assert run['selected_trial_id'] == current['best_trial_id']
    assert not run.get('final_evaluation_job_id')
    assert not (optimizer.store.root/'evaluation-usage.json').exists()
    assert all(j['kind'] == 'gpu_training_v1' for j in queue.jobs())
    with pytest.raises(DataError): queue.enqueue(session['trial_ids'][0], 'gpu_training_v1')


@pytest.mark.parametrize('stop,reason', [({'target_accuracy':.4},'target_reached'), ({'patience':1},'no_improvement')])
def test_stopping_rules_do_not_launch_remaining_candidate(optimizer, manual, gpu, tmp_path, stop, reason):
    queue, (trainer,_), (tw,_) = gpu
    session = approve(optimizer, imported(optimizer, manual, proposal(**stop)))
    optimizer.start(session['id'])
    run_once(trainer, tw['id'], backend=FakeBackend(), work_root=tmp_path/'worker', interval=.05)
    result = optimizer.advance(session['id'])
    assert result['stop_reason'] == reason and len(queue.jobs()) == 1


def test_failed_candidate_stops_and_cancel_waits_for_process_confirmation(optimizer, manual, gpu, tmp_path):
    queue, (trainer,_), (tw,_) = gpu
    session = approve(optimizer, imported(optimizer, manual))
    optimizer.start(session['id'])
    result = run_once(trainer, tw['id'], backend=FakeBackend(fail='SMOKE'), work_root=tmp_path/'worker', interval=.05)
    assert result['state'] == 'FAILED'
    final = optimizer.advance(session['id'])
    assert final['state'] == 'STOPPED' and final['stop_reason'] == 'candidate_failed'
    assert len(queue.jobs()) == 1 and not final['best_trial_id']


def test_cancel_live_lease_does_not_advance_or_release_slot(optimizer, manual, gpu):
    queue, _, (tw,_) = gpu
    session = approve(optimizer, imported(optimizer, manual))
    optimizer.start(session['id']); job = queue.claim(tw['id'])
    assert optimizer.cancel(session['id'])['state'] == 'CANCELLING'
    assert queue.claim(tw['id']) is None
    assert optimizer.advance(session['id'])['state'] == 'CANCELLING'
    queue.finish(job['id'], tw['id'], job['lease_id'], {'state':'CANCELLED','process_stopped':True,'result':{'kind':job['kind']}})
    assert optimizer.advance(session['id'])['state'] == 'CANCELLED'
    assert len(queue.jobs()) == 1


def test_crash_replay_reuses_candidate_job_and_budget(optimizer, manual, gpu, monkeypatch):
    queue, _, _ = gpu
    session = approve(optimizer, imported(optimizer, manual))
    original = optimizer.save
    def fail_after_enqueue(value):
        if value.get('active_job_id'): raise OSError('fixture crash after atomic enqueue')
        original(value)
    monkeypatch.setattr(optimizer, 'save', fail_after_enqueue)
    with pytest.raises(OSError): optimizer.start(session['id'])
    assert len(queue.jobs()) == 1
    monkeypatch.setattr(optimizer, 'save', original)
    restarted = Optimization(optimizer.store, optimizer.phase, queue)
    result = restarted.advance(session['id'])
    assert result['active_job_id'] == queue.jobs()[0]['id'] and len(queue.jobs()) == 1
    assert len(queue.read('trials', session['trial_ids'][0])['attempts']) == 1
    assert queue.budget(manual['id'])['charged_seconds'] == 0


def test_plan_tampering_cannot_start_and_external_test_freezes_optimization(optimizer, manual):
    session = approve(optimizer, imported(optimizer, manual))
    path = optimizer.directory(session['id'])/'proposals'/session['proposal_id']/'plan.json'
    value = optimizer.store.read(path); value['objective'] = 'changed'; optimizer.store.write(path,value)
    with pytest.raises(DataError): optimizer.start(session['id'])
    assert optimizer.queue.jobs() == []
    optimizer.store.write(optimizer.store.path('runs',manual['id'])/'baseline-predictions.json', {'fixture':'test exposed'})
    with pytest.raises(DataError): optimizer.inputs(session['id'])


def fake_cli(tmp_path, monkeypatch, plan, cost=.01, version='Fixture Claude 0', help_ok=True):
    script = tmp_path/'fake-claude'
    context = tmp_path/'captured-context.json'; args = tmp_path/'captured-args.json'
    script.write_text('#!'+sys.executable+'\nimport json,sys,pathlib\n'
        +'if "--version" in sys.argv: print('+repr(version)+');sys.exit()\n'
        +'if "--help" in sys.argv: print('+repr('--bare --tools --json-schema --max-budget-usd --strict-mcp-config --no-session-persistence' if help_ok else 'none')+');sys.exit()\n'
        +'pathlib.Path('+repr(str(context))+').write_text(sys.stdin.read())\n'
        +'pathlib.Path('+repr(str(args))+').write_text(json.dumps(sys.argv))\n'
        +'print(json.dumps('+repr({'type':'result','subtype':'success','is_error':False,'structured_output':plan,'total_cost_usd':cost})+'),flush=True)\n')
    script.chmod(0o700)
    monkeypatch.setenv('CLAUDE_BIN',str(script)); monkeypatch.setenv('KEV_ENABLE_CLAUDE_PLANNING','1')
    monkeypatch.setenv('KEV_CLAUDE_EXPECTED_VERSION','Fixture Claude 0'); monkeypatch.setenv('ANTHROPIC_API_KEY','ephemeral-fake-no-account')
    return context,args


def await_plan(optimizer, session_id):
    until = time.monotonic()+5
    while time.monotonic() < until:
        session = optimizer.load(session_id)
        if session['state'] != 'PLANNING': return session
        time.sleep(.02)
    raise AssertionError('fixture planning did not finish')


def test_fake_bare_cli_generation_awaits_human_review_and_charges_reservation(optimizer, manual, tmp_path, monkeypatch):
    context,args = fake_cli(tmp_path, monkeypatch, proposal())
    session = optimizer.create(manual['id'], {'claude_budget_usd':.1})
    with pytest.raises(DataError): optimizer.generate(session['id'], {'allow_paid_call':False,'max_cost_usd':.1})
    optimizer.generate(session['id'], {'allow_paid_call':True,'max_cost_usd':.1})
    result = await_plan(optimizer, session['id'])
    assert result['state'] == 'WAITING_REVIEW', result
    assert result['approval'] is None and not optimizer.phase.trials(manual['id'])
    assert '--bare' in json.loads(args.read_text())
    assert 'test unique phrase' not in context.read_text()
    with pytest.raises(DataError): optimizer.generate(session['id'], {'allow_paid_call':True,'max_cost_usd':.000001})
    assert result['claude_reserved_units'] == 100000
    assert result['planning_attempts'][0]['reported_cost_usd'] == .01


@pytest.mark.parametrize('mode', ['version','help','cost'])
def test_cli_profile_or_accounting_failure_never_approves_and_does_not_refund(optimizer, manual, tmp_path, monkeypatch, mode):
    fake_cli(tmp_path, monkeypatch, proposal(), version='wrong' if mode == 'version' else 'Fixture Claude 0',
             help_ok=mode != 'help', cost=.2 if mode == 'cost' else .01)
    session = optimizer.create(manual['id'], {'claude_budget_usd':.1})
    optimizer.generate(session['id'], {'allow_paid_call':True,'max_cost_usd':.1})
    result = await_plan(optimizer, session['id'])
    assert result['state'] == 'PLANNING_FAILED' and result['approval'] is None
    assert result['claude_reserved_units'] == 100000
    if mode != 'cost': assert not (tmp_path/'captured-context.json').exists()
    with pytest.raises(DataError): optimizer.generate(session['id'], {'allow_paid_call':True,'max_cost_usd':.000001})


def test_session_retry_cannot_reset_experiment_claude_budget(optimizer, manual):
    session = imported(optimizer, manual, budget=.1)
    optimizer.review(session['id'], {'plan_sha256':session['plan_sha256'],'decision':'reject','reviewer':'fixture','notes':'new proposal needed'})
    with pytest.raises(DataError): optimizer.create(manual['id'], {'claude_budget_usd':.2})
    second = optimizer.create(manual['id'], {'claude_budget_usd':.1})
    assert second['claude_limit_units'] == 100000


def test_spent_claude_reservation_survives_rejected_session_and_restart(optimizer, manual, tmp_path, monkeypatch):
    fake_cli(tmp_path, monkeypatch, proposal())
    session = optimizer.create(manual['id'], {'claude_budget_usd':.1})
    optimizer.generate(session['id'], {'allow_paid_call':True,'max_cost_usd':.1})
    session = await_plan(optimizer,session['id'])
    optimizer.review(session['id'], {'plan_sha256':session['plan_sha256'],'decision':'reject','reviewer':'fixture','notes':'another proposal'})
    restarted = Optimization(optimizer.store,optimizer.phase,optimizer.queue)
    new_session = restarted.create(manual['id'], {'claude_budget_usd':.1})
    with pytest.raises(DataError,match='剩余额度'):
        restarted.generate(new_session['id'],{'allow_paid_call':True,'max_cost_usd':.000001})


def test_gpu_budget_stop_prevents_any_enqueue(optimizer, manual, gpu):
    queue,_,_ = gpu
    session = approve(optimizer,imported(optimizer,manual))
    ledger = queue.ledger(); ledger['budgets'][manual['id']] = {'charged_seconds':1800,'reserved_job':None}; queue.commit(ledger)
    result = optimizer.start(session['id'])
    assert result['stop_reason'] == 'gpu_budget_exhausted' and queue.jobs() == []


def test_baseline_regression_stops_without_selecting_bad_candidate(optimizer, manual, gpu, tmp_path):
    from trial_execution import read,write
    queue,(trainer,_),(tw,_) = gpu
    class BadCandidate(FakeBackend):
        def perform(self,stage,operation,payload,supervisor):
            result = super().perform(stage,operation,payload,supervisor)
            if stage == 'EVALUATING':
                path = tmp_path/'unused'
                from pathlib import Path
                path = Path(payload['work'])/result['directory']/'predictions.jsonl'
                lines = [json.loads(line) for line in path.read_text().splitlines()]
                for line in lines:
                    for pair in line['prediction']['probabilities'].values(): pair.update(false=.25,true=.75)
                path.write_text(''.join(json.dumps(line)+'\n' for line in lines))
            return result
    session = approve(optimizer,imported(optimizer,manual))
    optimizer.start(session['id'])
    assert run_once(trainer,tw['id'],backend=BadCandidate(),work_root=tmp_path/'worker',interval=.05)['state'] == 'SUCCEEDED'
    result = optimizer.advance(session['id'])
    assert result['stop_reason'] == 'baseline_regression' and result['best_trial_id'] is None
    assert len(queue.jobs()) == 1


def test_optional_baseline_is_pinned_validation_not_final_report(client, optimizer, manual, gpu, tmp_path):
    from test_phase_one import new_trial
    queue,(trainer,_),(tw,_) = gpu
    trial = new_trial(client,manual)
    queue.enqueue(trial['id'],'gpu_training_v1')
    assert run_once(trainer,tw['id'],backend=FakeBackend(),work_root=tmp_path/'worker',interval=.05)['state'] == 'SUCCEEDED'
    session = optimizer.create(manual['id'],{'claude_budget_usd':0,'baseline_trial_id':trial['id']})
    inputs = optimizer.inputs(session['id'])
    assert inputs['baseline_metrics']['available'] is True
    assert inputs['baseline_metrics']['source'] == 'software_contract_fixture'
    assert not inputs['baseline_metrics']['training_verified']
    assert inputs['validation_errors']['partition'] == 'validation'
    assert all(row['id'].startswith('validation') for row in inputs['validation_errors']['examples'])
    session = optimizer.import_plan(session['id'],proposal(1))
    assert approve(optimizer,session)['state'] == 'APPROVED'
