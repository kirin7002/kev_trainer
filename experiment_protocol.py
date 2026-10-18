"""Frozen questions, data identities and native Kev inputs for one experiment."""
from copy import deepcopy
from collections import Counter
import hashlib
import json

from datasets import DataError, fingerprint


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_protocol(model, source_training, splits, definitions, threshold, business_scope=''):
    questions = {}
    for key, instructions in sorted(definitions.items()):
        if not key.strip() or key != key.strip() or not instructions.strip():
            raise DataError('标签名与判断规则不能为空，标签名不能含首尾空格。')
        questions[key] = {'type': 'noul', 'instructions': instructions}
    for row in source_training:
        for key, question in row.get('questions', {}).items():
            template = {k: deepcopy(v) for k, v in question.items() if k in ('type', 'instructions', 'criteria')}
            template.setdefault('instructions', f'文本是否涉及对象「{key}」？明确提及即是，否定和疑问不改变涉及关系。')
            if not isinstance(template['instructions'], str) or not template['instructions'].strip():
                raise DataError('Noul 判断规则必须是非空字符串。')
            if key in questions and questions[key] != template:
                raise DataError(f'标签 {key} 的判断规则不一致，请使用统一的问题定义。')
            questions[key] = template
    for row in source_training:
        for key in row['intents']:
            questions.setdefault(key, {'type': 'noul',
                'instructions': f'文本是否涉及对象「{key}」？明确提及即是，否定和疑问不改变涉及关系。'})
    if not questions or len(questions) > 100:
        raise DataError('请从训练数据或标签定义提供 1～100 个标签，不能从最终测试集推断标签。')
    questions = dict(sorted(questions.items()))
    for rows in splits.values():
        native_records(rows, questions)
    protocol = {'version': 2, 'baseline': deepcopy(model), 'questions': questions,
        'business_scope': business_scope,
        'decision': {'threshold': threshold, 'operator': 'gte', 'labels_are_complete': True},
        'metrics_version': 'set-metrics-v1', 'split_seed': 42,
        'scenario_counts': {name: dict(Counter(row.get('scenario', '未指定') for row in rows))
                            for name, rows in splits.items()},
        'splits': {name: {'sha256': fingerprint(rows), 'count': len(rows)} for name, rows in splits.items()}}
    scenarios = set().union(*(counts.keys() for counts in protocol['scenario_counts'].values()))
    protocol['coverage_warnings'] = [f'{scenario} 在 {split} 中没有样本，不能据此确认该场景的训练与泛化效果。'
        for scenario in sorted(scenarios) for split, counts in protocol['scenario_counts'].items()
        if not counts.get(scenario)]
    protocol['sha256'] = digest(protocol)
    return protocol


def verify_protocol(protocol):
    if digest({k: v for k, v in protocol.items() if k != 'sha256'}) != protocol.get('sha256'):
        raise DataError('实验评测协议已变化，请创建新实验并重新建立基线。')


def verify_rows(protocol, split, rows):
    verify_protocol(protocol)
    if fingerprint(rows) != protocol['splits'][split]['sha256']:
        raise DataError('实验数据快照已变化，不能继续复用原基线。')


def native_records(rows, questions, include_labels=True):
    records = []
    for row in rows:
        if set(row['intents']) - questions.keys():
            raise DataError('样本含未定义的标签，请在新实验中补充标签定义。')
        explicit = row.get('questions')
        if explicit is not None:
            if not explicit or set(explicit) - questions.keys():
                raise DataError('Noul 样本的问题必须属于实验定义，且至少有一个明确标注。')
            for key, question in explicit.items():
                if type(question.get('label')) is not bool:
                    raise DataError('Noul label 必须为布尔值。')
                for field in ('instructions', 'criteria'):
                    if field in question and question[field] != questions[key].get(field):
                        raise DataError('样本的问题定义与实验协议不一致。')
        output = {}
        for key, template in questions.items():
            if explicit is not None and key not in explicit:
                continue
            output[key] = deepcopy(template)
            if include_labels:
                output[key]['label'] = explicit[key]['label'] if explicit is not None else key in row['intents']
        state = {'scenario': row['scenario'], 'text': row['text']} if 'scenario' in row else row['text']
        records.append({'id': row['id'], 'state': state, 'questions': output})
    return records


def prediction_file(split, kind):
    return ('' if split == 'test' else 'validation-') + kind + '-predictions.json'


def report_file(split):
    return ('' if split == 'test' else 'validation-') + 'report.json'


def baseline_cache_key(protocol, split):
    return digest({'protocol_sha256': protocol['sha256'], 'split': split,
                   'dataset_sha256': protocol['splits'][split]['sha256'], 'model': protocol['baseline']})


def training_manifest(protocol):
    model = protocol['baseline']
    return {'executor': 'kev.train', 'executed': False, 'training_verified': False,
        'business_scope': protocol['business_scope'],
        'kev_commit': model['kev_commit'], 'protocol_sha256': protocol['sha256'],
        'inputs': {name + '.jsonl': protocol['splits'][name] for name in ('training', 'validation')},
        'working_directory': 'an isolated per-experiment directory in the pinned Kev checkout',
        'argv': ['python', '-m', 'kev.train', '--data', 'training.jsonl',
            '--base', model['base'], '--base_revision', model['base_revision'],
            '--init_from', model['checkpoint'], '--lora', str(model['lora_rank']),
            '--head_dim', str(model['head_dim']), '--epochs', '2', '--lr', '2e-5',
            '--batch', '1', '--accum', '8', '--dtype', 'bf16', '--checkpointing', '1',
            '--device', 'cuda', '--out', 'checkpoint'],
        'prerequisites': ['GPU adapter and pinned Kev environment must be configured separately',
            'Evaluate the baseline on frozen validation data before training',
            'Validate GPU memory, tokenizer length filtering, sample coverage and hyperparameters on the target machine',
            'Never include the final test set in training or optimization context']}
