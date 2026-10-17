"""Synthetic process/contract fixture. Never loads tensors or measures model quality."""
import json
from pathlib import Path
import sys

from kev_contract import CONTRACT
from model_artifacts import canonical_sha, file_sha
from trial_execution import write, read


class FakeBackend:
    mode = 'software_contract_only'

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    @staticmethod
    def runtime():
        return {'execution_backend': 'software_contract_only', 'runtime_verified': False,
                'image_provenance':{'status':'unknown','manifest_digest':None,'config_digest':None,'source':None,'independently_attested':False,'reason':'Synthetic fixture has no image evidence.'},
                'source_commit': CONTRACT['commit'], 'contract_sha256': canonical_sha(CONTRACT),
                'parent_archive_sha256': CONTRACT['released_checkpoint']['sha256']}

    def perform(self, stage, operation, payload, supervisor):
        self.calls.append((stage, operation, payload.get('data_role')))
        work = Path(payload['work'])
        supervisor.run(stage, [sys.executable, '-c', 'print("software contract fixture; no GPU")' if stage != self.fail else 'raise RuntimeError("injected failure")'],
                       cwd=work, environment={})
        model = payload['config']['model']
        objects = [obj['id'] for obj in payload['objects']]
        def data(role):
            return [json.loads(line) for line in (work/'data'/(role+'.jsonl')).read_text().splitlines()]
        if operation in ('validate', 'validate-evaluation'):
            if operation == 'validate-evaluation':
                return {'runtime': self.runtime(), 'tokenizer_verified': True, 'test_records': len(data('test')),
                        'rejected_records': 0, 'truncated_records': 0}
            return {'runtime': self.runtime(), 'tokenizer_verified': True, 'rejected_records': 0, 'truncated_records': 0,
                    'counts': {role: len(data(role)) for role in ('train', 'calibration', 'validation')},
                    'training_records_expanded': len(data('train'))*10}
        if operation=='validate-baseline':
            output=work/'baseline';output.mkdir()
            (output/'head.pt').write_bytes(b'synthetic baseline head; not a model')
            (output/'adapter_model.safetensors').write_bytes(b'synthetic baseline adapter; not a model')
            write(output/'adapter_config.json',{'r':16,'base_model_name_or_path':model['base']})
            return {'runtime':self.runtime(),'tokenizer_verified':True,'counts':{r:len(data(r)) for r in ('calibration','validation')},
                    'rejected_records':0,'truncated_records':0,'optimizer_steps':0,'training_verified':False}
        if operation == 'train':
            output = work/('smoke' if stage == 'SMOKE' else 'candidate')
            output.mkdir()
            metrics = {'device': 'cuda', 'rejected_records': 0, 'truncated_records': 0, 'optimizer_steps': 1,
                       'records_seen': len(data('train'))*10*payload['config']['parameters']['epochs'],
                       'requested_records': len(data('train'))*10*payload['config']['parameters']['epochs'],
                       'grad_norm': [{'mean': 1., 'max': 1.}], 'peak_device_bytes': 1}
            write(output/'adapter_config.json', {'r': 16, 'base_model_name_or_path': model['base']})
            write(output/'training_config.json', {'base_revision': model['base_revision']})
            write(output/'training_metrics.json', metrics)
            for name in ('head.pt', 'adapter_model.safetensors', 'tokenizer_config.json', 'tokenizer.json'):
                (output/name).write_bytes(b'synthetic fixture, not loadable model')
            losses = work/'logs'/(stage+'-loss.jsonl')
            losses.write_text('{"loss": 1.0}\n')
            return {'metrics': metrics, 'loss_file': str(losses.relative_to(work))}
        if operation == 'benchmark':
            rows, lines = [], []
            for row in data(payload['data_role']):
                probabilities = {qid: {'false': .75, 'true': .25} for qid in objects}
                lines.append({'id': row['_meta']['id'], 'prediction': {'probabilities': probabilities, 'latency_ms': 1.}})
                for qid, question in row['questions'].items():
                    rows.append({'id': row['_meta']['id'], 'question': qid, 'type': 'noul', 'label': int(question['label']),
                                 'keys': ['false', 'true'], 'logits': [1., 0.], 'inference_temperature': 1.,
                                 'variant': 'clean', 'source': 'fixture', 'group': row['_meta']['group_id']})
            directory = work/'benchmarks'/stage
            directory.mkdir(parents=True)
            (directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in lines))
            write(directory/'rows.json', rows)
            coverage = {'requested_records': len(lines), 'evaluated_records': len(lines),
                        'requested_questions': len(rows), 'evaluated_questions': len(rows),
                        'rejected_records': 0, 'truncated_records': 0}
            write(directory/'report.json', {'coverage': coverage, 'run': 'synthetic fixture'})
            return {'directory': str(directory.relative_to(work)), 'coverage': coverage}
        if operation == 'calibrate':
            role = payload['model_role']
            benchmark_stage = 'BASELINE_CALIBRATION' if role == 'baseline' else 'CANDIDATE_CALIBRATION'
            policy = {'fit_partition': 'calibration', 'groups_disjoint': True,
                      'method': 'pinned_kev_metrics_registered_custom_calibration_v1',
                      'threshold_strategy': 'fixed_0_5_no_validation_or_test_tuning', 'temperature': 1.,
                      'thresholds': {qid: .5 for qid in objects},
                      'data_sha256': file_sha(work/'data/calibration.jsonl'),
                      'train_sha256': file_sha(work/'data/train.jsonl') if payload['mode']=='train' else None,
                      'validation_sha256': file_sha(work/'data/validation.jsonl'),
                      'rows_sha256': file_sha(work/'benchmarks'/benchmark_stage/'rows.json')}
            if payload['mode']=='baseline': policy['baseline_only']=True
            write(work/'policy'/(role+'-calibration.json'), policy)
            return policy
        if operation == 'restore':
            return {'restored_without_fitting': True,
                    'policy_sha256': {role: file_sha(work/'policy'/(role+'-calibration.json')) for role in ('baseline', 'candidate')}}
        if operation == 'load':
            role='baseline' if payload['mode']=='baseline' else 'candidate'
            return {'loader_verified': True, 'objects_verified': 50, 'head_sha256': file_sha(work/role/'head.pt'),
                    'weights_sha256': file_sha(work/role/'adapter_model.safetensors')}
        raise AssertionError(operation)
