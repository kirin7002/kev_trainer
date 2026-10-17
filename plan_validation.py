"""Independent, strict phase-two plan validation. Plans contain data, never commands."""
import copy
from jsonschema import Draft202012Validator

from datasets import DataError
from phase_one import number

PARAMETERS = {'type': 'object', 'additionalProperties': False,
    'required': ['learning_rate', 'epochs', 'batch_size', 'gradient_accumulation', 'seed', 'precision'],
    'properties': {
        'learning_rate': {'type': 'number', 'minimum': 1e-7, 'maximum': .001},
        'epochs': {'type': 'integer', 'minimum': 1, 'maximum': 20},
        'batch_size': {'type': 'integer', 'minimum': 1, 'maximum': 64},
        'gradient_accumulation': {'type': 'integer', 'minimum': 1, 'maximum': 256},
        'seed': {'type': 'integer', 'minimum': 0, 'maximum': 2**32-1},
        'precision': {'enum': ['bf16', 'fp32']}}}
TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 2000}
PLAN_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['schema_version', 'objective', 'hypotheses', 'trials', 'stop_conditions'],
    'properties': {'schema_version': {'const': 2}, 'objective': TEXT,
        'hypotheses': {'type': 'array', 'minItems': 1, 'maxItems': 10, 'items': TEXT},
        'trials': {'type': 'array', 'minItems': 1, 'maxItems': 20, 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['id', 'parameters', 'rationale'],
            'properties': {'id': {'type': 'string', 'pattern': '^[a-zA-Z0-9_-]{1,40}$'},
                           'parameters': PARAMETERS, 'rationale': TEXT}}},
        'stop_conditions': {'type': 'object', 'additionalProperties': False,
            'required': ['metric', 'target_accuracy', 'min_improvement', 'patience', 'max_baseline_regression'],
            'properties': {'metric': {'const': 'validation_accuracy'},
                'target_accuracy': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'min_improvement': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'patience': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                'max_baseline_regression': {'type': 'number', 'minimum': 0, 'maximum': 1}}}}}


def validate_plan(value, available_trials):
    if list(Draft202012Validator(PLAN_SCHEMA).iter_errors(value)):
        raise DataError('方案不符合 v2 schema/固定参数白名单。')
    if len(value['trials']) > available_trials:
        raise DataError('方案候选数超过实验剩余上限。')
    ids, configs = set(), set()
    import json
    for trial in value['trials']:
        if trial['id'] in ids: raise DataError('方案候选 ID 重复。')
        ids.add(trial['id'])
        encoded = json.dumps(trial['parameters'], sort_keys=True, allow_nan=False)
        if encoded in configs: raise DataError('候选配置完全重复，不消耗预算重复搜索。')
        configs.add(encoded)
        for name, spec in PARAMETERS['properties'].items():
            if name != 'precision': number(trial['parameters'][name], name, spec['minimum'], spec['maximum'], spec['type'] == 'integer')
    stop = value['stop_conditions']
    for name in ('target_accuracy', 'min_improvement', 'max_baseline_regression'):
        number(stop[name], name, 0, 1)
    if type(value['schema_version']) is not int or type(stop['patience']) is not int:
        raise DataError('版本与 patience 必须为整数。')
    return copy.deepcopy(value)


def markdown(plan):
    lines = ['# 待人工审查的训练方案', '', plan['objective'], '', '## 假设', '']
    lines += ['- '+value for value in plan['hypotheses']]
    lines += ['', '## 固定候选顺序', '']
    for trial in plan['trials']:
        lines += [f"- {trial['id']}: {trial['rationale']}", '  '+str(trial['parameters'])]
    lines += ['', '## 停止条件', '', str(plan['stop_conditions']), '',
              '只用 validation 排名；不读取 test，不自动最终评测。资源兼容仍需真实 runtime 准入。']
    return '\n'.join(lines)+'\n'
