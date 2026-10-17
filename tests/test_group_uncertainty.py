"""Paired uncertainty uses connected groups, not artificially independent rows."""
from datasets import normalize
from evaluation import comparison, validate_predictions


def report(group_count):
    cases = normalize([{'id': str(i), 'text': f'uncertainty fixture {i}', 'intents': ['x'],
                        'conversation_id': str(i//2)} for i in range(group_count * 2)])
    def predictions(candidate):
        return validate_predictions({'dataset_sha256': 'fixture', 'predictions': [
            {'id': row['id'], 'intents': ['x'] if (int(row['id'])//2 % 2 == int(candidate)) else []}
            for row in cases]}, cases, 'fixture')
    return comparison(cases, predictions(False), predictions(True))


def test_too_few_groups_are_explicitly_uncertain_even_with_many_rows():
    value = report(29)['paired_accuracy_uncertainty']
    assert value['independent_groups'] == 29
    assert value['interval'] is None and value['uncertain'] is True


def test_cluster_interval_contains_zero_for_balanced_paired_differences():
    value = report(30)['paired_accuracy_uncertainty']
    assert value['independent_groups'] == 30
    assert value['uncertain'] is False
    assert value['interval'][0] < 0 < value['interval'][1]
    assert value['method'] == 'cluster_robust_normal_approximation'
