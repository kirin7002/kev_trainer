"""Standalone pinned-Kev subprocess. Run with python -I; no platform imports.

No operation runs on import. Real use requires operator-provided source checkout,
GPU Python and the hash-pinned release archive. Development uses a fake backend.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import tomllib

PLATFORM = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024*1024), b''):
            result.update(data)
    return result.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def watchdog(seconds):
    if not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise ValueError('invalid remaining deadline')
    if os.name != 'posix' or os.getsid(0) != os.getpid():
        raise RuntimeError('runtime must own a private process session')
    def expired():
        os.write(2, b'Kev Trainer shared deadline expired\n')
        os.killpg(os.getpgrp(), signal.SIGTERM)
    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    return timer


def source_check(source):
    contract = read(PLATFORM / 'kev-source-contract.json')
    source = Path(source).resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != contract['commit']:
        raise ValueError('Kev checkout commit mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('Kev checkout must be clean (including untracked files)')
    for name, meta in contract['files'].items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source) or digest(path) != meta['sha256']:
            raise ValueError('Kev source hash mismatch: ' + name)
    locked = tomllib.loads((source / 'uv.lock').read_text())
    dependencies = {}
    for name in ('accelerate', 'datasets', 'numpy', 'peft', 'pydantic', 'scikit-learn', 'torch', 'transformers'):
        installed = importlib.metadata.version(name)
        versions = {item.get('version') for item in locked['package'] if item['name'].replace('_', '-') == name}
        if installed.split('+')[0] not in versions:
            raise ValueError('dependency differs from pinned uv.lock: ' + name)
        dependencies[name] = installed
    sys.path.insert(0, str(source))
    return contract, dependencies


def image_provenance(path=Path('/run/.containerenv')):
    """Observe bounded runtime-generated Podman metadata; never query a daemon/socket.

    An image config digest is not an OCI manifest digest. Neither a container ID,
    mount-path hash nor an operator environment string proves either digest.
    """
    unknown = {'status':'unknown', 'manifest_digest':None, 'config_digest':None,
               'source':None, 'independently_attested':False,
               'reason':'No supported runtime-generated image digest metadata.'}
    try:
        path = Path(path)
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size > 16384 or info.st_uid not in (0,os.getuid())
                or info.st_mode & 0o022): return {**unknown,'reason':'Runtime metadata is not a bounded owned regular file.'}
        raw = path.read_bytes()
        if len(raw)>16384: return unknown
        values = dict(re.findall(r'^(imageid|image|engine)="([^"\n]{1,1000})"$',raw.decode('utf-8'),re.M))
        if not values.get('engine','').startswith('podman'): return unknown
        config = re.fullmatch(r'(?:sha256:)?([a-f0-9]{64})',values.get('imageid',''))
        manifest = re.fullmatch(r'[A-Za-z0-9._:/-]+@sha256:([a-f0-9]{64})',values.get('image',''))
        if not config and not manifest: return unknown
        return {'status':'observed_runtime_metadata','manifest_digest':'sha256:'+manifest[1] if manifest else None,
                'config_digest':'sha256:'+config[1] if config else None,'source':'podman_containerenv',
                'metadata_sha256':hashlib.sha256(raw).hexdigest(),'independently_attested':False,
                'reason':None if manifest else 'Only image config digest observed; OCI manifest digest unknown.'}
    except (OSError,UnicodeError): return unknown


def runtime_probe(payload):
    if sys.platform != 'linux' or sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise ValueError('GPU runtime requires Linux/WSL2 and Python 3.12/3.13')
    contract, versions = source_check(payload['source'])
    archive = Path(payload['checkpoint_archive'])
    if digest(archive) != contract['released_checkpoint']['sha256']:
        raise ValueError('released parent checkpoint archive SHA mismatch')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; CPU fallback is forbidden')
    probe = torch.ones((2, 2), device='cuda')
    if not torch.isfinite(probe @ probe).all().item():
        raise ValueError('CUDA probe failed')
    del probe
    torch.cuda.empty_cache()
    return {'runtime_verified': True, 'execution_backend': 'cuda', 'source_commit': contract['commit'],
            'contract_sha256': object_sha(contract), 'parent_archive_sha256': digest(archive),
            'python': sys.version.split()[0], 'dependencies': versions, 'device': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda, 'image_provenance':image_provenance(), 'training_verified': False}


def extract_parent(archive, target):
    target = Path(target)
    if target.exists():
        raise ValueError('refusing to overwrite parent snapshot')
    target.mkdir()
    with tarfile.open(archive, 'r:gz') as handle:
        entries = [item for item in handle.getmembers() if not item.isdir()]
        if len(entries) > 200 or sum(item.size for item in entries) > 512*1024*1024:
            raise ValueError('parent archive limits exceeded')
        normalized = []
        for item in entries:
            name = item.name.removeprefix('./')
            p = PurePosixPath(name)
            if (not item.isfile() or p.is_absolute() or '\\' in name or ':' in name or
                    any(part in ('', '.', '..') for part in name.split('/'))):
                raise ValueError('unsafe parent archive member')
            normalized.append((item, p))
        prefix = ()
        if not any(p.as_posix() == 'head.pt' for _, p in normalized):
            roots = {p.parts[0] for _, p in normalized}
            if len(roots) != 1: raise ValueError('unknown release layout')
            prefix = (next(iter(roots)),)
        names = set()
        for item, p in normalized:
            parts = p.parts[len(prefix):]
            if len(parts) != 1 or parts[0] in names:
                raise ValueError('unexpected nested or duplicate release member')
            names.add(parts[0])
            with handle.extractfile(item) as source, (target / parts[0]).open('xb') as output:
                shutil.copyfileobj(source, output)
    if not {'head.pt', 'adapter_config.json', 'adapter_model.safetensors'} <= names:
        raise ValueError('release archive incomplete')


def records(work, role):
    rows = [json.loads(line) for line in (work / 'data' / (role + '.jsonl')).read_text().splitlines() if line.strip()]
    for row in rows:
        for question in row['questions'].values(): question.setdefault('src', role + '_noul')
    return rows


def checkpoint(work, role):
    if role not in ('parent', 'baseline', 'candidate', 'smoke'):
        raise ValueError('checkpoint role forbidden')
    return work / role


def compatibility(path, model):
    import torch
    torch.load(path / 'head.pt', map_location='cpu', weights_only=True)
    from kev.checkpoint import Checkpoint
    ck = Checkpoint(str(path))
    expected = {'base': model['base'], 'base_revision': model['base_revision'], 'lora': model['lora_rank'],
                'head_dim': model['head_dim'], 'weights': 'lora', 'option_isolation': False, 'special_embeddings': False}
    if any(getattr(ck.meta, key) != value for key, value in expected.items()):
        raise ValueError('checkpoint/base/revision/rank/head mismatch')
    # Matching targets are checked against actual tensors by warm_start at smoke.
    adapter = ck.adapter_config()
    if adapter.get('r') != model['lora_rank'] or not adapter.get('target_modules'):
        raise ValueError('invalid adapter config')
    return {**expected, 'lora_targets': 'all', 'target_modules': sorted(adapter['target_modules']),
            'weights_sha256': ck.weights_sha256(), 'head_sha256': digest(path / 'head.pt')}


def token_observation(row, encoded):
    from collections import Counter
    segments = Counter(encoded['seg'])
    return {'id': row['_meta']['id'], 'state_tokens': encoded['state_tokens'],
            'state_truncated': encoded['state_truncated'], 'packed_tokens': len(encoded['ids']),
            'question_branch_tokens': {qid: segments[i] for i, qid in enumerate(row['questions'], 1)}}


def validate_inputs(payload):
    runtime = runtime_probe(payload)
    work, config = Path(payload['work']), payload['config']
    extract_parent(payload['checkpoint_archive'], work / 'parent')
    proof = compatibility(work / 'parent', config['model'])
    shutil.copytree(work / 'parent', work / 'baseline')
    from kev.data import materialize
    from kev.model import load_tokenizer, encode, training_context, fits
    tokenizer = load_tokenizer(config['model']['base'], revision=config['model']['base_revision'])
    context, maximum = training_context(), 0
    counts, expanded, observations = {}, [], {}
    for role in ('train', 'calibration', 'validation'):
        admitted = records(work, role)
        counts[role] = len(admitted)
        observations[role] = []
        for row in admitted:
            encoded = encode(tokenizer, materialize(row), max_state=context['max_state'], max_branch=context['max_branch'], strict=True)
            maximum = max(maximum, len(encoded['ids']))
            observations[role].append(token_observation(row, encoded))
            if role == 'train':
                objects = list(row['questions'])
                for offset in range(0, len(objects), 5):
                    keys = objects[offset:offset+5]
                    part = {**row, 'questions': {key: row['questions'][key] for key in keys},
                            '_meta': {**row['_meta'], 'id': row['_meta']['id'] + '/qgroup-' + str(offset//5),
                                      'parent_id': row['_meta']['id']}}
                    if not fits(materialize(part), tokenizer, **context):
                        raise ValueError('training question group exceeds exact tokenizer context')
                    expanded.append(part)
    (work / 'data' / 'train-expanded.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in expanded))
    # Full 50-object evaluation remains intact; only the packed total is lifted.
    eval_context = {**context, 'max_packed': max(context['max_packed'], maximum)}
    result = {'runtime': runtime, 'compatibility': proof, 'tokenizer_verified': True, 'counts': counts,
              'training_records_expanded': len(expanded), 'question_group_size': 5,
              'token_counts': observations,
              'evaluation_context': eval_context, 'rejected_records': 0, 'truncated_records': 0}
    write(work / 'registry' / 'validation.json', result)
    return result


def validate_evaluation(payload):
    runtime = runtime_probe(payload)
    work = Path(payload['work'])
    extract_parent(payload['checkpoint_archive'], work/'parent')
    compatibility(work/'candidate', payload['config']['model'])
    compatibility(work/'parent', payload['config']['model'])
    shutil.copytree(work/'parent', work/'baseline')
    from kev.data import materialize
    from kev.model import load_tokenizer, encode, training_context
    model = payload['config']['model']
    tokenizer = load_tokenizer(model['base'], revision=model['base_revision'])
    context = training_context()
    admitted = records(work, 'test')
    maximum, observations = 0, []
    for row in admitted:
        encoded = encode(tokenizer, materialize(row), max_state=context['max_state'], max_branch=context['max_branch'], strict=True)
        maximum = max(maximum, len(encoded['ids']))
        observations.append(token_observation(row, encoded))
    result = {'runtime': runtime, 'tokenizer_verified': True, 'test_records': len(admitted),
              'token_counts': {'test': observations},
              'evaluation_context': {**context, 'max_packed': max(context['max_packed'], maximum)},
              'rejected_records': 0, 'truncated_records': 0}
    write(work/'registry/validation.json', result)
    return result


def validate_baseline(payload):
    if payload['mode'] != 'baseline': raise ValueError('baseline-only operation requires baseline mode')
    runtime = runtime_probe(payload)
    work, config = Path(payload['work']), payload['config']
    extract_parent(payload['checkpoint_archive'],work/'parent')
    proof = compatibility(work/'parent',config['model'])
    shutil.copytree(work/'parent',work/'baseline')
    from kev.data import materialize
    from kev.model import load_tokenizer,encode,training_context
    tokenizer = load_tokenizer(config['model']['base'],revision=config['model']['base_revision'])
    context,maximum = training_context(),0
    counts,observations = {},{}
    for role in ('calibration','validation'):
        admitted = records(work,role); counts[role] = len(admitted); observations[role] = []
        for row in admitted:
            encoded = encode(tokenizer,materialize(row),max_state=context['max_state'],max_branch=context['max_branch'],strict=True)
            maximum = max(maximum,len(encoded['ids']));observations[role].append(token_observation(row,encoded))
    result = {'runtime':runtime,'compatibility':proof,'tokenizer_verified':True,'counts':counts,
              'token_counts':observations,'evaluation_context':{**context,'max_packed':max(context['max_packed'],maximum)},
              'rejected_records':0,'truncated_records':0,'optimizer_steps':0,'training_verified':False}
    write(work/'registry/validation.json',result)
    return result


def restore_frozen_policy(payload):
    if payload['mode'] != 'evaluate': raise ValueError('restore operation reserved for final evaluator')
    work = Path(payload['work'])
    from kev.checkpoint import read_meta, write_meta
    for role in ('baseline', 'candidate'):
        policy = read(work/'policy'/(role+'-calibration.json'))
        if (policy.get('fit_partition') != 'calibration' or policy.get('threshold_strategy') != 'fixed_0_5_no_validation_or_test_tuning'
                or policy.get('thresholds') != {obj['id']: .5 for obj in payload['objects']}):
            raise ValueError('final evaluator refuses unfrozen/tuned policy')
        meta = read_meta(checkpoint(work, role))
        if role == 'candidate' and abs(meta.temperature-policy['temperature']) > 1e-12:
            raise ValueError('candidate calibration differs from accepted model')
        if role == 'baseline':
            meta.temperature = policy['temperature']
            meta.extra['temperature_fit'] = policy
            write_meta(checkpoint(work, role), meta)
    return {'restored_without_fitting': True, 'thresholds': 'fixed_0_5',
            'policy_sha256': {role: digest(work/'policy'/(role+'-calibration.json')) for role in ('baseline', 'candidate')}}


def observed_train(payload):
    work, config = Path(payload['work']), payload['config']
    if config.get('resume_from') is not None:
        raise ValueError('Pinned LoRA path has no optimizer resume; create a fresh attempt from the fixed parent.')
    from kev import train
    stage = payload['stage']
    output = checkpoint(work, 'smoke' if stage == 'SMOKE' else 'candidate')
    if output.exists(): raise ValueError('refusing to overwrite previous training output')
    params, model = config['parameters'], config['model']
    original_loss, original_requests = train.batch_loss, train.training_requests
    losses = work / 'logs' / (stage + '-loss.jsonl')
    count = sum(1 for line in (work/'data/train-expanded.jsonl').read_text().splitlines() if line.strip())
    with losses.open('x') as log:
        def observer(*args, **kwargs):
            loss, terms = original_loss(*args, **kwargs)
            value = float(loss.detach().cpu().item())
            if not math.isfinite(value): raise ValueError('non-finite observed loss')
            log.write(json.dumps({'loss': value, 'ce': float(terms.get('ce', value))}, allow_nan=False)+'\n')
            log.flush()
            return loss, terms
        def admitted(*args, **kwargs):
            value = original_requests(*args, **kwargs)
            if len(value) != count: raise ValueError('trainer dropped records despite admission')
            return value
        train.batch_loss, train.training_requests = observer, admitted
        previous = sys.argv
        sys.argv = ['kev.train', '--data', str(work/'data/train-expanded.jsonl'), '--base', model['base'],
            '--base_revision', model['base_revision'], '--init_from', str(work/'parent'),
            '--lora', str(model['lora_rank']), '--head_dim', str(model['head_dim']), '--lora_targets', 'all',
            '--lr', str(params['learning_rate']), '--epochs', '1' if stage == 'SMOKE' else str(params['epochs']),
            '--batch', str(params['batch_size']), '--accum', str(params['gradient_accumulation']),
            '--seed', str(params['seed']), '--dtype', params['precision'], '--device', 'cuda', '--checkpointing', '1',
            '--out', str(output)]
        if stage == 'SMOKE': sys.argv += ['--max_steps', '1']
        try: train.main()
        finally: train.batch_loss, train.training_requests, sys.argv = original_loss, original_requests, previous
    return {'metrics': read(output/'training_metrics.json'), 'loss_file': str(losses.relative_to(work)),
            'instrumentation': 'batch_loss_scalar_observer_v1_no_gradient_change',
            'expected_records': count * (1 if stage == 'SMOKE' else params['epochs'])}


def benchmark(payload):
    work = Path(payload['work'])
    import torch
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.data import load_records
    role, model_role = payload['data_role'], payload['model_role']
    if payload['mode']=='baseline' and model_role!='baseline': raise ValueError('baseline-only benchmark cannot use a candidate')
    if role not in (('test',) if payload['mode'] == 'evaluate' else ('calibration', 'validation')):
        raise ValueError('benchmark partition forbidden for this job kind')
    admitted = load_records(str(work/'data'/ (role+'.jsonl')))
    context = read(work/'registry/validation.json')['evaluation_context']
    opts = LoadOptions(dtype=torch.float32, merge=False, temperature=1.0 if payload.get('raw') else None,
                       backend='torch', cuda_graphs=False, fused=False)
    predictor = LocalPredictor(str(checkpoint(work, model_role)), 'cuda', opts, context=context)
    directory = work/'benchmarks'/payload['stage']
    torch.cuda.reset_peak_memory_stats()
    report, _ = evaluate_records(admitted, predictor, directory, skip_overlong=False)
    report.update(run=model_role, environment={**predictor.environment, 'device_name': torch.cuda.get_device_name(0)},
                  peak_device_bytes=torch.cuda.max_memory_allocated(), dtype='fp32', backend='torch',
                  dataset_role=role, context=context, raw_logits=bool(payload.get('raw')))
    write(directory/'report.json', report)
    return {'directory': str(directory.relative_to(work)), 'coverage': report['coverage'],
            'peak_device_bytes': report['peak_device_bytes']}


def registered_calibration(payload):
    """Use the pinned fitting API after independently proving the platform split registry.

    Unlike the upstream suite-only CLI this accepts registered custom partitions.
    It never passes --allow-in-distribution, never fits validation/test and preserves
    the exact calibration data/rows/source hashes in the exported checkpoint.
    """
    work = Path(payload['work'])
    if payload['mode'] not in ('train','baseline'): raise ValueError('evaluator cannot fit calibration')
    target = payload['model_role']
    if payload['mode']=='baseline' and target!='baseline': raise ValueError('baseline-only job cannot calibrate candidate')
    if target not in ('baseline', 'candidate'): raise ValueError('calibration target forbidden')
    stage = 'BASELINE_CALIBRATION' if target == 'baseline' else 'CANDIDATE_CALIBRATION'
    path = work/'benchmarks'/stage/'rows.json'
    rows = read(path)
    calibration = records(work, 'calibration')
    training = records(work, 'train') if payload['mode']=='train' else []
    validation = records(work, 'validation')
    expected = {(row['_meta']['id'], qid): int(question['label']) for row in calibration for qid, question in row['questions'].items()}
    groups = {row['_meta']['group_id'] for row in calibration}
    if groups & {row['_meta']['group_id'] for row in training+validation}:
        raise ValueError('calibration groups overlap training/validation')
    seen = set()
    for row in rows:
        key = (row.get('id'), row.get('question'))
        if (key not in expected or key in seen or row.get('label') != expected[key] or row.get('type') != 'noul'
                or row.get('variant') != 'clean' or row.get('source') == 'unknowable'
                or row.get('keys') != ['false', 'true'] or row.get('inference_temperature') != 1.0
                or not isinstance(row.get('logits'), list) or len(row['logits']) != 2
                or not all(type(v) in (int, float) and math.isfinite(v) for v in row['logits'])):
            raise ValueError('calibration rows/labels/raw logits do not match registered calibration partition')
        seen.add(key)
    if seen != set(expected): raise ValueError('calibration rows incomplete')
    from kev.metrics import fit_temperature, TEMPERATURE_FIT
    from kev.checkpoint import read_meta, write_meta
    temperature = fit_temperature(rows, **TEMPERATURE_FIT)
    if not math.isfinite(temperature) or temperature <= 0: raise ValueError('invalid fitted temperature')
    registry = {'schema_version': 1, 'method': 'pinned_kev_metrics_registered_custom_calibration_v1',
                'fit_partition': 'calibration', 'data_sha256': digest(work/'data/calibration.jsonl'),
                'train_sha256': digest(work/'data/train.jsonl') if payload['mode']=='train' else None,
                'validation_sha256': digest(work/'data/validation.jsonl'),
                'rows_sha256': digest(path), 'groups_disjoint': True, 'rows': len(rows),
                'parent_pretraining_overlap': 'not_independently_known', 'fit': TEMPERATURE_FIT,
                'temperature': temperature, 'thresholds': {obj['id']: .5 for obj in payload['objects']},
                'threshold_strategy': 'fixed_0_5_no_validation_or_test_tuning'}
    if payload['mode']=='baseline': registry['baseline_only'] = True
    meta = read_meta(checkpoint(work, target))
    meta.temperature = temperature
    meta.extra['temperature_fit'] = registry
    write_meta(checkpoint(work, target), meta)
    write(work/'policy'/(target+'-calibration.json'), registry)
    return registry


def load_model(payload):
    work = Path(payload['work'])
    model_role = 'baseline' if payload['mode']=='baseline' else 'candidate'
    path = checkpoint(work, model_role)
    proof = compatibility(path, payload['config']['model'])
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    tokenizer, model = Checkpoint(str(path)).load('cuda', LoadOptions(dtype=torch.float32, merge=False,
        backend='torch', cuda_graphs=False, fused=False))
    from kev.data import materialize
    from kev.model import training_context
    record = records(work, 'validation' if payload['mode'] in ('train','baseline') else 'test')[0]
    context = read(work/'registry/validation.json')['evaluation_context']
    encoded = model.encode(tokenizer, materialize(record), max_state=context['max_state'], max_branch=context['max_branch'], strict=True)
    with torch.no_grad(): outputs = model.forward(encoded)
    if len(outputs) != len(payload['objects']) or any(not torch.isfinite(item).all().item() for item in outputs):
        raise ValueError('current loader canary output invalid')
    return {'loader_verified': True, 'loader': 'kev.checkpoint.Checkpoint.load+DecisionModel.forward',
            'objects_verified': len(outputs), 'compatibility': proof, 'head_sha256': digest(path/'head.pt'),
            'weights_sha256': digest(path/'adapter_model.safetensors')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--operation', choices=('probe', 'validate', 'validate-baseline', 'validate-evaluation', 'restore', 'train', 'benchmark', 'calibrate', 'load'), required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', required=True, type=float)
    args = parser.parse_args()
    timer = watchdog(args.seconds)
    try:
        payload = read(args.input)
        payload['source'] = args.source
        if payload.get('mode') not in ('train', 'evaluate','baseline'): raise ValueError('job mode missing')
        if payload['mode']=='baseline' and args.operation not in ('probe','validate-baseline','benchmark','calibrate','load'):
            raise ValueError('baseline-only job cannot train, resume or restore candidate')
        if payload['mode'] == 'evaluate' and args.operation in ('train', 'calibrate', 'validate'):
            raise ValueError('evaluation worker cannot train or calibrate')
        if args.operation == 'probe': result = runtime_probe(payload)
        elif args.operation == 'validate': result = validate_inputs(payload)
        elif args.operation == 'validate-baseline': result = validate_baseline(payload)
        elif args.operation == 'validate-evaluation': result = validate_evaluation(payload)
        else:
            source_check(args.source)
            result = {'train': observed_train, 'benchmark': benchmark,
                      'calibrate': registered_calibration, 'load': load_model, 'restore': restore_frozen_policy}[args.operation](payload)
        write(args.output, result)
    finally:
        timer.cancel()


if __name__ == '__main__': main()
