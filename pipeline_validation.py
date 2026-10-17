"""Independent controller-side checks for complete benchmark and training receipts."""
import json
import math
import re
from pathlib import Path

from datasets import DataError
from evaluation import validate_predictions


TRAIN_STAGES = ('VALIDATING', 'BASELINE_RAW', 'BASELINE_CALIBRATION', 'BASELINE_CALIBRATING',
                'BASELINE_SYSTEM', 'SMOKE', 'TRAINING', 'CANDIDATE_RAW', 'CANDIDATE_CALIBRATION',
                'CALIBRATING', 'EVALUATING', 'LOADING', 'COMPARING')


def validate_image_provenance(value):
    if not isinstance(value,dict) or value.get('independently_attested') is not False:
        raise DataError('镜像来源必须明确记录未知或运行时观察，不能冒充独立证明。')
    digests=[value.get('manifest_digest'),value.get('config_digest')]
    if any(v is not None and (not isinstance(v,str) or not re.fullmatch(r'sha256:[a-f0-9]{64}',v)) for v in digests):
        raise DataError('镜像 digest 格式无效。')
    if value.get('status')=='unknown':
        if any(v is not None for v in digests) or value.get('source') is not None: raise DataError('未知镜像不得填入推测 digest。')
    elif value.get('status')=='observed_runtime_metadata':
        if (not any(digests) or value.get('source')!='podman_containerenv'
                or not re.fullmatch(r'[a-f0-9]{64}',str(value.get('metadata_sha256','')))):
            raise DataError('镜像观察缺少运行时元数据来源。')
    else: raise DataError('未知镜像证明级别。')


def calibration_rows(rows, cases, object_ids):
    expected = {(case['id'], obj): int(obj in case['intents']) for case in cases for obj in object_ids}
    seen = set()
    if not isinstance(rows, list): raise DataError('校准 rows 必须为数组。')
    for row in rows:
        key = (row.get('id'), row.get('question'))
        if (key not in expected or key in seen or row.get('label') != expected[key]
                or row.get('type') != 'noul' or row.get('keys') != ['false', 'true']
                or row.get('variant') != 'clean' or row.get('source') == 'unknowable'
                or row.get('inference_temperature') != 1.0
                or not isinstance(row.get('logits'), list) or len(row['logits']) != 2
                or not all(type(v) in (int, float) and math.isfinite(v) for v in row['logits'])):
            raise DataError('校准 rows 必须完整绑定原始 calibration 标签和有限 logits。')
        seen.add(key)
    if seen != set(expected): raise DataError('校准 rows 覆盖不完整。')


def training_metrics(value, observations, expected_records, smoke=False):
    if not isinstance(value, dict) or value.get('device') != 'cuda':
        raise DataError('训练结果未记录 CUDA。')
    if value.get('rejected_records') != 0 or value.get('truncated_records') != 0:
        raise DataError('训练发生样本丢弃或截断。')
    if type(value.get('optimizer_steps')) is not int or value['optimizer_steps'] < 1:
        raise DataError('训练没有完成优化器 step。')
    if not smoke and (value.get('requested_records') != expected_records or value.get('records_seen') != expected_records):
        raise DataError('训练记录覆盖不完整。')
    norms = value.get('grad_norm')
    if not isinstance(norms, list) or not norms:
        raise DataError('缺少梯度观测。')
    for epoch in norms:
        for key in ('mean', 'max'):
            if type(epoch.get(key)) not in (int, float) or not math.isfinite(epoch[key]) or epoch[key] < 0:
                raise DataError('梯度观测无效。')
    if not isinstance(observations, list) or not observations:
        raise DataError('缺少实际 loss 观测，不能放行 smoke/训练。')
    for row in observations:
        if type(row.get('loss')) not in (int, float) or not math.isfinite(row['loss']) or row['loss'] < 0:
            raise DataError('Loss 观测不是有限非负数值。')
    return {'optimizer_steps': value['optimizer_steps'], 'loss_observations': len(observations),
            'loss_first': observations[0]['loss'], 'loss_last': observations[-1]['loss'],
            'peak_device_bytes': value.get('peak_device_bytes'), 'records_seen': value.get('records_seen')}


def convert_benchmark(lines, report, cases, object_ids, dataset_sha, thresholds=None):
    coverage = report.get('coverage', {})
    n = len(cases)
    if (coverage.get('requested_records') != n or coverage.get('evaluated_records') != n or
            coverage.get('requested_questions') != n*len(object_ids) or coverage.get('evaluated_questions') != n*len(object_ids) or
            coverage.get('rejected_records') != 0 or coverage.get('truncated_records') != 0):
        raise DataError('Benchmark coverage 不完整；不能跳过失败或过长样本。')
    expected = {case['id']: case for case in cases}
    result, seen = [], set()
    for item in lines:
        case_id = item.get('id')
        if not isinstance(case_id, str) or case_id not in expected or case_id in seen:
            raise DataError('Benchmark 样本 ID 无效或重复。')
        seen.add(case_id)
        prediction = item.get('prediction', {})
        raw = prediction.get('probabilities')
        if not isinstance(raw, dict) or set(raw) != set(object_ids):
            raise DataError('Benchmark 没有完整 50 对象输出。')
        probabilities = {}
        for obj in object_ids:
            pair = raw[obj]
            if not isinstance(pair, dict) or set(pair) != {'false', 'true'}:
                raise DataError('Noul 概率键必须完整。')
            if any(type(p) not in (int, float) or not 0 <= p <= 1 for p in pair.values()) or abs(sum(pair.values())-1) > 1e-5:
                raise DataError('Noul 概率不是有限归一化分布。')
            probabilities[obj] = pair['true']
        cutoff = thresholds or {obj: .5 for obj in object_ids}
        if set(cutoff) != set(object_ids) or any(type(p) not in (int, float) or not 0 <= p <= 1 for p in cutoff.values()):
            raise DataError('阈值目录不完整或无效。')
        result.append({'id': case_id, 'intents': sorted(obj for obj, p in probabilities.items() if p >= cutoff[obj]),
                       'probabilities': probabilities, 'latency_ms': prediction.get('latency_ms')})
    if seen != set(expected):
        raise DataError('Benchmark 样本覆盖不完整。')
    environment = report.get('environment') or {}
    runtime = {'device': str(environment.get('device_name', environment.get('device', 'cuda'))),
               'backend': 'torch', 'dtype': 'fp32'}
    return validate_predictions({'dataset_sha256': dataset_sha, 'model': str(report.get('run', 'worker-model')),
                                 'runtime': runtime, 'predictions': result}, cases, dataset_sha, object_ids)


def inference_summary(predictions, cases):
    rows = list(predictions['predictions'].values())
    times = [row['latency_ms'] for row in rows if row['latency_ms'] is not None]
    return {'cold_first_request_ms': times[0] if times else None,
            'warm_remaining_count': max(0, len(times)-1),
            'warm_remaining_mean_ms': sum(times[1:])/len(times[1:]) if len(times) > 1 else None,
            'serial_model_requests_per_second': len(times)/(sum(times)/1000) if times and sum(times) else None,
            'low_confidence_policy': 'any_object_max_binary_probability_below_0.6',
            'low_confidence_count': sum(any(max(p, 1-p) < .6 for p in row['probabilities'].values()) for row in rows),
            'unknown_prediction_count': sum(not row['intents'] for row in rows),
            'scope': 'sequential_model_calls_excludes_loading_not_service_qps'}
