"""Normalize user datasets without changing their labels."""
import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter


class DataError(ValueError):
    pass


def identity(text):
    return re.sub(r"\s+", "", text).casefold()


def labels(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = re.split(r"[|,;，；]", value)
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise DataError('intents 必须是字符串数组，CSV 也可使用 | 分隔。')
    result = sorted(set(v.strip() for v in value))
    if 'unknown' in result:
        if len(result) > 1:
            raise DataError('unknown 不能与其他对象同时出现。')
        result = []
    if len(result) > 100:
        raise DataError('单条样本的对象数过多。')
    return result


def normalize(rows):
    if not isinstance(rows, list) or not rows:
        raise DataError('数据必须包含非空样本数组。')
    if len(rows) > 20000:
        raise DataError('第一版每个数据集最多支持 20,000 条样本。')
    result, ids, texts = [], set(), {}
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise DataError(f'第 {i} 条样本必须是对象。')
        text = row.get('text', row.get('state'))
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise DataError(f'第 {i} 条 text/state 必须是 1～20,000 字符的文本。')
        questions = None
        if 'expected_result' in row:
            expected = row['expected_result']
            if not isinstance(expected, dict) or 'intents' not in expected:
                raise DataError(f'第 {i} 条 expected_result 缺少 intents。')
            target = labels(expected['intents'])
        elif 'intents' in row or 'labels' in row:
            target = labels(row.get('intents', row.get('labels')))
        elif isinstance(row.get('questions'), dict) and row['questions']:
            questions = row['questions']
            for name, q in questions.items():
                if not isinstance(name, str) or not name.strip():
                    raise DataError(f'第 {i} 条 questions 的对象名必须是非空字符串。')
                if not isinstance(q, dict):
                    raise DataError(f'第 {i} 条 questions["{name}"] 必须是对象（含 type 与 label）。')
                if q.get('type') != 'noul':
                    raise DataError(f'第 {i} 条 questions["{name}"].type 必须是 "noul"。')
                if type(q.get('label')) is not bool:
                    raise DataError(f'第 {i} 条 questions["{name}"].label 缺失或不是布尔值；'
                        '第一版支持含明确布尔 label 的 Noul JSONL，缺失标签不会视为 false。')
            target = sorted(name for name, q in questions.items() if q['label'])
        else:
            raise DataError(f'第 {i} 条缺少标签：请提供 expected_result.intents 或 intents。')
        case_id = row.get('id', f'case-{i:05d}')
        if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 128 or case_id in ids:
            raise DataError(f'第 {i} 条 ID 无效或重复。')
        ids.add(case_id)
        key = identity(text)
        if key in texts and texts[key] != target:
            raise DataError(f'第 {i} 条与其他相同文本的标签冲突。')
        texts[key] = target
        group_id = row.get('group_id', key)
        if not isinstance(group_id, str) or not group_id:
            raise DataError('group_id 必须是非空字符串。')
        record = {'id': case_id, 'text': text.strip(), 'intents': target, 'group_id': group_id}
        if questions is not None:
            # 保留原始 questions（含 instructions/criteria 与显式 false 标签），供后续真实训练使用。
            record['questions'] = questions
        result.append(record)
    return result


def parse_upload(raw, filename):
    try:
        text = raw.decode('utf-8-sig')
        if filename.lower().endswith('.csv'):
            rows = list(csv.DictReader(io.StringIO(text)))
        elif filename.lower().endswith('.jsonl'):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            obj = json.loads(text)
            rows = obj.get('cases', obj.get('records')) if isinstance(obj, dict) else obj
        return normalize(rows)
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as error:
        raise DataError('文件格式无效，请上传 UTF-8 编码的 JSON、JSONL 或 CSV。') from error


def profile(rows):
    distribution = Counter(label for row in rows for label in row['intents'])
    counts = Counter('unknown' if not row['intents'] else 'single' if len(row['intents']) == 1 else 'multi' for row in rows)
    return {'total': len(rows), 'single': counts['single'], 'multi': counts['multi'], 'unknown': counts['unknown'],
            'object_count': len(distribution), 'labels': dict(sorted(distribution.items())),
            'duplicate_texts': len(rows) - len({identity(row['text']) for row in rows}),
            'warnings': (['样本量较少，验证结果可能有较大波动。'] if len(rows) < 100 else [])}


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def split_validation(rows, fraction=0.2, seed=42):
    # Union explicit groups and identical texts before splitting, including duplicates
    # whose authors accidentally supplied different group IDs.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in (('group', row['group_id']), ('text', identity(row['text']))):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    keys = list(groups)
    if len(keys) < 2:
        raise DataError('训练集至少需要两个独立文本组，才能留出验证集。')
    random.Random(seed).shuffle(keys)
    size = min(len(keys) - 1, max(1, math.ceil(len(keys) * fraction)))
    validation = [row for key in keys[:size] for row in groups[key]]
    training = [row for key in keys[size:] for row in groups[key]]
    return training, validation


def assert_disjoint(training, test):
    text_overlap = {identity(r['text']) for r in training} & {identity(r['text']) for r in test}
    groups_overlap = {r['group_id'] for r in training} & {r['group_id'] for r in test}
    if text_overlap or groups_overlap:
        raise DataError('训练集与测试集存在相同文本或分组，请移除重叠后重试。')
