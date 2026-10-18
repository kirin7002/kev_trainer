"""Set-based metrics from explicitly imported predictions; never fabricate results."""
import math
from collections import Counter
from statistics import mean
from datasets import DataError, grouped_rows, labels, question_ids


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


def validate_predictions(payload, cases, sha, object_ids=None):
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
        if latency is not None:
            try:
                valid_latency = type(latency) in (int, float) and math.isfinite(latency) and latency >= 0
            except OverflowError:
                valid_latency = False
            if not valid_latency:
                raise DataError('latency_ms 必须是非负有限数值或 null。')
        intents = labels(row['intents'])
        if object_ids is not None and set(intents) - set(object_ids):
            raise DataError('预测包含任务目录之外的未知对象。')
        asked = question_ids(expected[row['id']], object_ids) if object_ids is not None else None
        if asked is not None and set(intents) - set(asked):
            raise DataError('预测包含当前样本未明确标注的问题。')
        result[row['id']] = {'intents': intents, 'latency_ms': latency}
        if 'probabilities' in row:
            probabilities = row['probabilities']
            if object_ids is None or not isinstance(probabilities, dict) or set(probabilities) != set(asked):
                raise DataError('原始概率需要绑定任务目录并完整覆盖该样本的明确问题。')
            if any(type(p) not in (int, float) or not 0 <= p <= 1 for p in probabilities.values()):
                raise DataError('对象概率必须是 0～1 的有限数值。')
            result[row['id']]['probabilities'] = probabilities
    if set(result) != set(expected):
        raise DataError('预测必须覆盖全部测试样本，不能跳过错误或失败的样本。')
    runtime = payload.get('runtime')
    if runtime is not None and (not isinstance(runtime, dict) or
            any(not isinstance(runtime.get(k), str) or not runtime[k] for k in ('device', 'backend', 'dtype'))):
        raise DataError('runtime 需要包含 device、backend、dtype 字符串。')
    return {'dataset_sha256': sha, 'model': str(payload.get('model', 'user-imported'))[:200],
            **({'object_ids': list(object_ids)} if object_ids is not None else {}),
            'runtime': runtime, 'predictions': result, 'source': 'user_imported_predictions'}


def summarize(rows, object_ids=None):
    tp = sum(len(r['expected'] & r['actual']) for r in rows)
    fp = sum(len(r['actual'] - r['expected']) for r in rows)
    fn = sum(len(r['expected'] - r['actual']) for r in rows)
    times = [r['latency_ms'] for r in rows if r['latency_ms'] is not None]
    support, matched, extra, missed = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        support.update(row['expected'])
        matched.update(row['expected'] & row['actual'])
        extra.update(row['actual'] - row['expected'])
        missed.update(row['expected'] - row['actual'])
    objects = sorted(support.keys() | extra.keys() | set(object_ids or []))
    per_object = []
    for obj in objects:
        a, b, c = matched[obj], extra[obj], missed[obj]
        per_object.append({'object': obj, 'support': a+c, 'precision': divide(a, a+b),
            'recall': divide(a, a+c), 'f1': divide(2*a, 2*a+b+c)})
    f1_values = [r['f1'] for r in per_object if r['f1'] is not None]
    probability_rows = [row for row in rows if row.get('probabilities')]
    calibration = {'coverage': len(probability_rows), 'total': len(rows), 'brier_score': None, 'ece': None,
                   'ece_bins': 10, 'source': 'user_imported_probabilities'}
    if probability_rows and len(probability_rows) == len(rows):
        bins = [[0, 0, 0] for _ in range(10)]
        squared, count = 0, 0
        for row in probability_rows:
            for obj, p in row['probabilities'].items():
                y = int(obj in row['expected'])
                squared += (p-y)**2
                count += 1
                bucket = bins[min(9, int(p*10))]
                bucket[0] += 1
                bucket[1] += p
                bucket[2] += y
        calibration['brier_score'] = squared / count
        calibration['ece'] = sum(abs(bucket[1]-bucket[2]) for bucket in bins) / count
    return {'total': len(rows), 'correct': sum(r['passed'] for r in rows),
        'annotated_questions': sum(len(row.get('asked_questions', [])) for row in rows) if object_ids is not None else None,
        'accuracy': divide(sum(r['passed'] for r in rows), len(rows)),
        'precision': divide(tp, tp+fp), 'recall': divide(tp, tp+fn), 'f1': divide(2*tp, 2*tp+fp+fn),
        'macro_f1': sum(f1_values)/len(f1_values) if f1_values else None,
        'probability_calibration': calibration,
        'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
        'latency': {'count': len(times), 'mean_ms': mean(times) if times else None,
                    'p50_ms': percentile(times, .5), 'p95_ms': percentile(times, .95)},
        'per_object': per_object}


def evaluate(cases, predictions):
    objects = predictions.get('object_ids')
    rows = []
    for case in cases:
        actual = predictions['predictions'][case['id']]
        expected_set, actual_set = set(case['intents']), set(actual['intents'])
        rows.append({'id': case['id'], 'text': case['text'], 'expected': expected_set, 'actual': actual_set,
            'scenario': case.get('scenario', '未指定'),
            'asked_questions': question_ids(case, objects) if objects is not None else sorted(expected_set | actual_set),
            'passed': expected_set == actual_set, 'latency_ms': actual['latency_ms'],
            **({'probabilities': actual['probabilities']} if 'probabilities' in actual else {}),
            'group': 'unknown' if not expected_set else 'single' if len(expected_set) == 1 else 'multi'})
    scenarios = {}
    for row in rows:
        scenarios.setdefault(row['scenario'], []).append(row)
    result = {'overall': summarize(rows, objects),
        'annotation_scope': 'explicit_questions_only' if objects is not None and any(
            len(row['asked_questions']) != len(objects) for row in rows) else 'complete',
        'by_scenario': {name: summarize(group, objects) for name, group in sorted(scenarios.items())},
        'groups': {name: summarize([r for r in rows if r['group'] == name], objects) for name in ('single', 'multi', 'unknown')},
        'by_object_count': {str(n): summarize([r for r in rows if len(r['expected']) == n], objects)
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
        'paired_accuracy_uncertainty': paired_uncertainty(cases, a, b),
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


def paired_uncertainty(cases, baseline, candidate):
    deltas = {a['id']: int(b['passed'])-int(a['passed']) for a, b in zip(baseline['cases'], candidate['cases'])}
    groups = grouped_rows(cases)
    n, g = len(cases), len(groups)
    value = {'confidence_level': .95, 'method': 'cluster_robust_normal_approximation',
             'independent_groups': g, 'interval': None, 'uncertain': True,
             'reason': 'too_few_independent_groups' if g < 30 else 'zero_observed_variation'}
    if g >= 30 and n:
        delta = sum(deltas.values()) / n
        variance = g/(g-1) * sum((sum(deltas[row['id']] for row in group)-len(group)*delta)**2 for group in groups) / n**2
        if variance > 0:
            half = 1.96 * math.sqrt(variance)
            value.update(interval=[max(-1, delta-half), min(1, delta+half)], uncertain=False,
                         reason='approximation_requires_independent_groups')
    return value
