"""Evaluation: prediction validation, hand-computed metrics, latency, comparison."""
import math

import pytest

from datasets import DataError, fingerprint, normalize
from evaluation import comparison, divide, evaluate, percentile, summarize, validate_predictions

SHA = 'deadbeef' * 8

CASES = normalize([
    {'id': 'c1', 'text': '开车窗', 'intents': ['window']},
    {'id': 'c2', 'text': '开车窗和座椅加热', 'intents': ['window', 'seat_heating']},
    {'id': 'c3', 'text': '今天几号', 'intents': []},
    {'id': 'c4', 'text': '开天窗', 'intents': ['sunroof']},
])

# Predictions:
#   c1 correct; c2 misses seat_heating; c3 hallucinates volume; c4 correct.
PREDS = {
    'c1': (['window'], 10),
    'c2': (['window'], 20),
    'c3': (['volume'], 30),
    'c4': (['sunroof'], 40),
}


def payload(preds=None, sha=SHA, model='m1', runtime=None):
    records = []
    for case in CASES:
        intents, latency = (preds or PREDS)[case['id']]
        records.append({'id': case['id'], 'intents': intents, 'latency_ms': latency})
    out = {'dataset_sha256': sha, 'predictions': records}
    if model is not None:
        out['model'] = model
    if runtime is not None:
        out['runtime'] = runtime
    return out


# ---------- validate_predictions ----------

def test_valid_payload_marks_external_source():
    result = validate_predictions(payload(), CASES, SHA)
    assert result['source'] == 'user_imported_predictions'
    assert result['dataset_sha256'] == SHA
    assert result['model'] == 'm1'
    assert set(result['predictions']) == {'c1', 'c2', 'c3', 'c4'}
    assert result['predictions']['c2'] == {'intents': ['window'], 'latency_ms': 20}


def test_model_defaults_to_user_imported():
    result = validate_predictions(payload(model=None), CASES, SHA)
    assert result['model'] == 'user-imported'


def test_wrong_sha_rejected():
    with pytest.raises(DataError, match='dataset_sha256'):
        validate_predictions(payload(sha='f' * 64), CASES, SHA)
    with pytest.raises(DataError):
        validate_predictions([1, 2], CASES, SHA)  # not a dict at all


def test_unknown_or_duplicate_or_nonstring_id_rejected():
    p = payload()
    p['predictions'][0]['id'] = 'nope'
    with pytest.raises(DataError, match='ID'):
        validate_predictions(p, CASES, SHA)
    p = payload()
    p['predictions'].append({'id': 'c1', 'intents': []})
    with pytest.raises(DataError, match='ID'):
        validate_predictions(p, CASES, SHA)
    p = payload()
    p['predictions'][0]['id'] = 5
    with pytest.raises(DataError):
        validate_predictions(p, CASES, SHA)


def test_incomplete_coverage_rejected():
    p = payload()
    p['predictions'] = p['predictions'][:3]  # drops c4
    with pytest.raises(DataError, match='覆盖全部'):
        validate_predictions(p, CASES, SHA)


def test_missing_predictions_array_rejected():
    with pytest.raises(DataError, match='predictions'):
        validate_predictions({'dataset_sha256': SHA}, CASES, SHA)


def test_mismatched_text_rejected():
    p = payload()
    p['predictions'][0]['text'] = '不是原来的文本'
    with pytest.raises(DataError, match='文本'):
        validate_predictions(p, CASES, SHA)
    p = payload()
    p['predictions'][0]['text'] = CASES[0]['text']  # exact text is allowed
    validate_predictions(p, CASES, SHA)


def test_missing_intents_rejected():
    p = payload()
    del p['predictions'][0]['intents']
    with pytest.raises(DataError, match='intents'):
        validate_predictions(p, CASES, SHA)


@pytest.mark.parametrize('bad', ['12', -1, -0.5, float('nan'), float('inf'), True, [], {}])
def test_invalid_latency_rejected(bad):
    p = payload()
    p['predictions'][0]['latency_ms'] = bad
    with pytest.raises(DataError, match='latency'):
        validate_predictions(p, CASES, SHA)


def test_null_latency_and_missing_latency_allowed():
    p = payload()
    p['predictions'][0]['latency_ms'] = None
    p['predictions'][1].pop('latency_ms')
    result = validate_predictions(p, CASES, SHA)
    assert result['predictions']['c1']['latency_ms'] is None
    assert result['predictions']['c2']['latency_ms'] is None


def test_runtime_validation():
    good = {'device': 'nvidia-a100', 'backend': 'kev-cli', 'dtype': 'bf16'}
    assert validate_predictions(payload(runtime=good), CASES, SHA)['runtime'] == good
    assert validate_predictions(payload(runtime=None), CASES, SHA)['runtime'] is None
    for bad in ({'device': 'x'}, {'device': '', 'backend': 'b', 'dtype': 'd'},
                {'device': 1, 'backend': 'b', 'dtype': 'd'}, 'a100'):
        with pytest.raises(DataError, match='runtime'):
            validate_predictions(payload(runtime=bad), CASES, SHA)


# ---------- hand-computed metrics ----------
# tp = |{window}|(c1) + |{window}|(c2) + 0(c3) + |{sunroof}|(c4) = 3
# fp = |{volume}|(c3) = 1 ; fn = |{seat_heating}|(c2) = 1
# exact-match correct: c1, c4 -> accuracy 2/4 = 0.5
# precision = 3/4, recall = 3/4, f1 = 6/8 = 0.75
# per object: window a=2,b=0,c=0 -> P/R/F1 = 1 ; sunroof -> 1 ;
#             seat_heating a=0,b=0,c=1 -> P=None, R=0, F1=0 ; volume a=0,b=1,c=0 -> P=0, R=None, F1=0
# macro_f1 = (1 + 0 + 0 + 1)/4 = 0.5
# latency [10,20,30,40]: mean 25 ; p50 = 20+(30-20)*0.5 = 25 ; p95 = 30+(40-30)*0.85 = 38.5

def test_overall_metrics_hand_computed():
    result = evaluate(CASES, validate_predictions(payload(), CASES, SHA))
    o = result['overall']
    assert o['total'] == 4 and o['correct'] == 2
    assert o['accuracy'] == pytest.approx(0.5)
    assert o['precision'] == pytest.approx(0.75)
    assert o['recall'] == pytest.approx(0.75)
    assert o['f1'] == pytest.approx(0.75)
    assert o['macro_f1'] == pytest.approx(0.5)
    assert (o['true_positive'], o['false_positive'], o['false_negative']) == (3, 1, 1)
    assert o['latency'] == {'count': 4, 'mean_ms': pytest.approx(25.0),
                            'p50_ms': pytest.approx(25.0), 'p95_ms': pytest.approx(38.5)}


def test_per_object_metrics_hand_computed():
    result = evaluate(CASES, validate_predictions(payload(), CASES, SHA))
    per = {p['object']: p for p in result['overall']['per_object']}
    assert per['window'] == {'object': 'window', 'support': 2, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert per['sunroof']['support'] == 1 and per['sunroof']['f1'] == 1.0
    assert per['seat_heating']['support'] == 1
    assert per['seat_heating']['precision'] is None
    assert per['seat_heating']['recall'] == 0.0 and per['seat_heating']['f1'] == 0.0
    assert per['volume']['support'] == 0
    assert per['volume']['precision'] == 0.0 and per['volume']['recall'] is None
    assert per['volume']['f1'] == 0.0


def test_group_breakdown_hand_computed():
    result = evaluate(CASES, validate_predictions(payload(), CASES, SHA))
    g = result['groups']
    # single = c1 (correct), c4 (correct)
    assert g['single']['total'] == 2 and g['single']['accuracy'] == 1.0
    # multi = c2: expected {window, seat_heating}, actual {window} -> tp=1 fn=1
    assert g['multi']['total'] == 1 and g['multi']['accuracy'] == 0.0
    assert g['multi']['precision'] == 1.0 and g['multi']['recall'] == pytest.approx(0.5)
    assert g['multi']['f1'] == pytest.approx(2 / 3)
    # unknown = c3: actual {volume} -> fp=1
    assert g['unknown']['total'] == 1 and g['unknown']['accuracy'] == 0.0
    assert g['unknown']['precision'] == 0.0 and g['unknown']['recall'] is None
    assert g['unknown']['f1'] == 0.0


def test_by_object_count():
    result = evaluate(CASES, validate_predictions(payload(), CASES, SHA))
    counts = result['by_object_count']
    assert set(counts) == {'0', '1', '2'}
    assert counts['0']['total'] == 1 and counts['1']['total'] == 2 and counts['2']['total'] == 1
    assert counts['1']['accuracy'] == 1.0


def test_case_details_missing_extra():
    result = evaluate(CASES, validate_predictions(payload(), CASES, SHA))
    by_id = {c['id']: c for c in result['cases']}
    assert by_id['c2']['missing'] == ['seat_heating'] and by_id['c2']['extra'] == []
    assert by_id['c3']['missing'] == [] and by_id['c3']['extra'] == ['volume']
    assert by_id['c1']['passed'] is True and by_id['c2']['passed'] is False


def test_partial_latency_coverage():
    preds = {k: (v[0], None) for k, v in PREDS.items()}
    preds['c2'] = (['window'], 33)  # only one sample measured
    result = evaluate(CASES, validate_predictions(payload(preds), CASES, SHA))
    lat = result['overall']['latency']
    assert lat == {'count': 1, 'mean_ms': pytest.approx(33.0),
                   'p50_ms': pytest.approx(33.0), 'p95_ms': pytest.approx(33.0)}


# ---------- percentile / divide primitives ----------

def test_percentile_hand_computed():
    assert percentile([], 0.95) is None
    assert percentile([7], 0.95) == 7
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    # index = (4-1)*0.95 = 2.85 -> 3 + (4-3)*0.85
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert percentile([5, 1, 3], 0.5) == 3  # unsorted input
    assert not math.isnan(percentile([1, 2], 0.95))


def test_divide_zero_denominator_is_none():
    assert divide(3, 0) is None
    assert divide(0, 0) is None
    assert divide(3, 6) == 0.5


def test_summarize_empty_rows():
    s = summarize([])
    assert s['total'] == 0 and s['accuracy'] is None and s['macro_f1'] is None
    assert s['latency']['count'] == 0 and s['latency']['mean_ms'] is None
    assert s['per_object'] == []


# ---------- comparison ----------

def test_comparison_hand_computed():
    # candidate fixes c2 -> accuracy 3/4, delta +0.25, improved [c2]
    fixed = dict(PREDS)
    fixed['c2'] = (['window', 'seat_heating'], 20)
    runtime = {'device': 'a100', 'backend': 'kev', 'dtype': 'bf16'}
    baseline = validate_predictions(payload(runtime=runtime, model='base'), CASES, SHA)
    candidate = validate_predictions(payload(fixed, runtime=runtime, model='cand'), CASES, SHA)
    report = comparison(CASES, baseline, candidate)
    assert report['source'] == 'user_imported_predictions'
    assert report['training_verified'] is False
    assert report['dataset_sha256'] == SHA
    assert report['baseline_model'] == 'base' and report['candidate_model'] == 'cand'
    assert report['performance_comparable'] is True
    assert report['accuracy_delta'] == pytest.approx(0.25)
    assert report['improved_ids'] == ['c2']
    assert report['regressed_ids'] == []
    assert report['candidate']['overall']['accuracy'] == pytest.approx(0.75)


def test_comparison_regression_and_non_comparable_runtime():
    broken = dict(PREDS)
    broken['c1'] = ([], 10)  # c1 was correct in baseline -> regression
    baseline = validate_predictions(payload(runtime={'device': 'a100', 'backend': 'kev', 'dtype': 'bf16'}), CASES, SHA)
    candidate = validate_predictions(payload(broken, runtime={'device': 'cpu', 'backend': 'other', 'dtype': 'fp32'}), CASES, SHA)
    report = comparison(CASES, baseline, candidate)
    assert report['performance_comparable'] is False
    assert any('runtime' in issue for issue in report['comparability_issues'])
    assert report['improved_ids'] == []
    assert report['regressed_ids'] == ['c1']
    assert report['accuracy_delta'] == pytest.approx(-0.25)


def test_comparison_latency_coverage():
    runtime = {'device': 'a100', 'backend': 'kev', 'dtype': 'bf16'}
    partial = dict(PREDS)
    partial['c2'] = (['window'], None)  # candidate did not time c2
    baseline = validate_predictions(payload(runtime=runtime), CASES, SHA)
    candidate = validate_predictions(payload(partial, runtime=runtime), CASES, SHA)
    report = comparison(CASES, baseline, candidate)
    # accuracy identical on both sides, but speed is not comparable
    assert report['performance_comparable'] is False
    assert report['latency_coverage'] == {'total': 4, 'baseline': 4, 'candidate': 3}
    assert any('candidate' in issue and 'latency_ms' in issue for issue in report['comparability_issues'])
    full = validate_predictions(payload(runtime=runtime), CASES, SHA)
    ok = comparison(CASES, baseline, full)
    assert ok['performance_comparable'] is True
    assert ok['comparability_issues'] == []
    assert ok['latency_coverage'] == {'total': 4, 'baseline': 4, 'candidate': 4}


def test_fingerprint_of_cases_matches_payload_sha_flow():
    sha = fingerprint(CASES)
    validate_predictions(payload(sha=sha), CASES, sha)  # accepted
    with pytest.raises(DataError):
        validate_predictions(payload(sha=sha), CASES, 'other')
