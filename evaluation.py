"""Set-based metrics from explicitly imported predictions; never fabricate results."""
import math
from datasets import DataError, labels


def divide(a, b):
    return a / b if b else None


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def validate_predictions(payload, cases, sha):
    if not isinstance(payload, dict) or payload.get('dataset_sha256') != sha:
        raise DataError('dataset_sha256 必须与当前实验的测试集一致。')
    records = payload.get('predictions')
    if not isinstance(records, list):
        raise DataError('缺少 predictions 数组。')
    expected = {r['id']: r for r in cases}
    result = {}
    for row in records:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                or row['id'] not in expected or row['id'] in result):
            raise DataError('预测 ID 缺失、重复或不属于当前测试集。')
        if 'text' in row and row['text'] != expected[row['id']]['text']:
            raise DataError('预测文本与测试样本不一致。')
        if 'intents' not in row:
            raise DataError('预测结果缺少 intents。')
        latency = row.get('latency_ms')
        if latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
            raise DataError('latency_ms 必须是非负有限数值或 null。')
        result[row['id']] = {'intents': labels(row['intents']), 'latency_ms': latency}
    if set(result) != set(expected):
        raise DataError('预测必须覆盖全部测试样本，不能跳过错误或失败的样本。')
    runtime = payload.get('runtime')
    if runtime is not None and (not isinstance(runtime, dict) or
            any(not isinstance(runtime.get(k), str) or not runtime[k] for k in ('device', 'backend', 'dtype'))):
        raise DataError('runtime 需要包含 device、backend、dtype 字符串。')
    return {'dataset_sha256': sha, 'model': str(payload.get('model', 'user-imported'))[:200],
            'runtime': runtime, 'predictions': result, 'source': 'user_imported_predictions'}


def summarize(rows):
    tp = sum(len(r['expected'] & r['actual']) for r in rows)
    fp = sum(len(r['actual'] - r['expected']) for r in rows)
    fn = sum(len(r['expected'] - r['actual']) for r in rows)
    times = [r['latency_ms'] for r in rows if r['latency_ms'] is not None]
    objects = sorted(set().union(*(r['expected'] | r['actual'] for r in rows))) if rows else []
    per_object = []
    for obj in objects:
        a = sum(obj in r['expected'] and obj in r['actual'] for r in rows)
        b = sum(obj not in r['expected'] and obj in r['actual'] for r in rows)
        c = sum(obj in r['expected'] and obj not in r['actual'] for r in rows)
        per_object.append({'object': obj, 'support': a+c, 'precision': divide(a, a+b),
            'recall': divide(a, a+c), 'f1': divide(2*a, 2*a+b+c)})
    f1_values = [r['f1'] for r in per_object if r['f1'] is not None]
    return {'total': len(rows), 'correct': sum(r['passed'] for r in rows),
        'accuracy': divide(sum(r['passed'] for r in rows), len(rows)),
        'precision': divide(tp, tp+fp), 'recall': divide(tp, tp+fn), 'f1': divide(2*tp, 2*tp+fp+fn),
        'macro_f1': sum(f1_values)/len(f1_values) if f1_values else None,
        'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
        'latency': {'count': len(times), 'mean_ms': sum(times)/len(times) if times else None,
                    'p50_ms': percentile(times, .5), 'p95_ms': percentile(times, .95)},
        'per_object': per_object}


def evaluate(cases, predictions):
    rows = []
    for case in cases:
        actual = predictions['predictions'][case['id']]
        expected_set, actual_set = set(case['intents']), set(actual['intents'])
        rows.append({'id': case['id'], 'text': case['text'], 'expected': expected_set, 'actual': actual_set,
            'passed': expected_set == actual_set, 'latency_ms': actual['latency_ms'],
            'group': 'unknown' if not expected_set else 'single' if len(expected_set) == 1 else 'multi'})
    result = {'overall': summarize(rows),
        'groups': {name: summarize([r for r in rows if r['group'] == name]) for name in ('single', 'multi', 'unknown')},
        'by_object_count': {str(n): summarize([r for r in rows if len(r['expected']) == n])
                           for n in sorted({len(r['expected']) for r in rows})},
        'cases': [{**r, 'expected': sorted(r['expected']), 'actual': sorted(r['actual']),
                   'missing': sorted(r['expected']-r['actual']), 'extra': sorted(r['actual']-r['expected'])} for r in rows]}
    return result


def comparison(cases, baseline, candidate):
    a, b = evaluate(cases, baseline), evaluate(cases, candidate)
    # 速度可比性：两侧必须覆盖完全相同的样本计时（无缺失）且 runtime 一致，
    # 否则仍展示各自指标，但不宣称性能/速度可比或提升。
    ids = [case['id'] for case in cases]
    missing = {name: [cid for cid in ids if side['predictions'][cid]['latency_ms'] is None]
               for name, side in (('baseline', baseline), ('candidate', candidate))}
    issues = []
    for name in ('baseline', 'candidate'):
        if missing[name]:
            issues.append(f'{name} 缺少 {len(missing[name])} 条样本的 latency_ms 计时，速度不可比。')
    runtimes_match = bool(baseline['runtime']) and baseline['runtime'] == candidate['runtime']
    if not runtimes_match:
        issues.append('两侧 runtime（device/backend/dtype）缺失或不一致，速度不可比。')
    return {'source': 'user_imported_predictions', 'training_verified': False,
        'dataset_sha256': baseline['dataset_sha256'],
        'baseline_model': baseline['model'], 'candidate_model': candidate['model'],
        'performance_comparable': runtimes_match and not missing['baseline'] and not missing['candidate'],
        'comparability_issues': issues,
        'latency_coverage': {'total': len(ids),
            'baseline': len(ids) - len(missing['baseline']),
            'candidate': len(ids) - len(missing['candidate'])},
        'baseline': a, 'candidate': b,
        'accuracy_delta': b['overall']['accuracy'] - a['overall']['accuracy'],
        'improved_ids': [x['id'] for x, y in zip(a['cases'], b['cases']) if not x['passed'] and y['passed']],
        'regressed_ids': [x['id'] for x, y in zip(a['cases'], b['cases']) if x['passed'] and not y['passed']]}
