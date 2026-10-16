"""Dataset normalization: JSON/JSONL/CSV/Noul formats, labels, IDs, splits, disjointness."""
import json

import pytest

from datasets import DataError, assert_disjoint, fingerprint, identity, normalize, parse_upload, profile, split_validation


def rows(*texts_labels):
    return [{'id': f'r{i}', 'text': t, 'intents': list(l)} for i, (t, l) in enumerate(texts_labels, 1)]


# ---------- JSON formats ----------

def test_parse_json_array():
    raw = json.dumps([{'id': 'a', 'text': '打开车窗', 'intents': ['window']}]).encode()
    result = parse_upload(raw, 'train.json')
    assert result == [{'id': 'a', 'text': '打开车窗', 'intents': ['window'], 'group_id': identity('打开车窗')}]


@pytest.mark.parametrize('key', ['cases', 'records'])
def test_parse_json_wrapped_object(key):
    raw = json.dumps({key: [{'text': '开空调', 'expected_result': {'intents': ['ac']}}]}).encode()
    result = parse_upload(raw, 'f.json')
    assert result[0]['intents'] == ['ac']
    assert result[0]['id'] == 'case-00001'  # auto ID when absent


def test_parse_jsonl_with_blank_lines():
    raw = '\n'.join([
        json.dumps({'text': '开车窗', 'intents': ['window']}, ensure_ascii=False),
        '',
        '   ',
        json.dumps({'text': '开空调', 'intents': ['ac']}, ensure_ascii=False),
    ]).encode()
    result = parse_upload(raw, 'f.jsonl')
    assert len(result) == 2


def test_parse_csv_pipe_and_json_labels():
    csv_text = 'id,text,intents\nc1,开车窗和空调,"window|ac"\nc2,只开车窗,"[""window""]"\nc3,闲聊,[]\nc4,无法归类,unknown'
    result = parse_upload(csv_text.encode(), 'f.csv')
    assert [r['intents'] for r in result] == [['ac', 'window'], ['window'], [], []]


def test_parse_csv_empty_label_cell_rejected():
    # 记录当前行为：CSV 空 intents 单元格不会被当作 []，需要显式写 [] 或 unknown。
    # 已作为可用性疑问上报 Codex（空单元格是 CSV 用户表达“无对象”的自然写法）。
    csv_text = 'id,text,intents\nc1,闲聊,'
    with pytest.raises(DataError, match='intents'):
        parse_upload(csv_text.encode(), 'f.csv')


def test_parse_csv_utf8_bom():
    raw = '﻿' + 'text,intents\n开车窗,window\n'
    result = parse_upload(raw.encode('utf-8'), 'f.csv')
    assert result[0]['text'] == '开车窗'


def test_parse_json_bom():
    raw = '﻿'.encode('utf-8') + json.dumps([{'text': 'x 打开', 'intents': []}]).encode()
    result = parse_upload(raw, 'f.json')
    assert result[0]['intents'] == []


# ---------- Noul boolean format ----------

def test_noul_boolean_questions():
    row = {'id': 'n1', 'text': '车窗和座椅', 'questions': {
        'window': {'type': 'noul', 'label': True},
        'seat_heating': {'type': 'noul', 'label': False},
        'sunroof': {'type': 'noul', 'label': True}}}
    result = normalize([row])
    assert result[0]['intents'] == ['sunroof', 'window']  # sorted, only True labels


def test_noul_missing_label_rejected():
    row = {'text': '车窗', 'questions': {'window': {'type': 'noul'}}}
    with pytest.raises(DataError, match='布尔'):
        normalize([row])


def test_noul_non_bool_label_rejected():
    # 'true'/1 must not silently pass as truthy; only real booleans count.
    for bad in ('true', 1, None):
        row = {'text': '车窗', 'questions': {'window': {'type': 'noul', 'label': bad}}}
        with pytest.raises(DataError):
            normalize([row])


def test_noul_wrong_type_rejected():
    row = {'text': '车窗', 'questions': {'window': {'type': 'multi', 'label': True}}}
    with pytest.raises(DataError):
        normalize([row])


# ---------- labels / unknown handling ----------

def test_unknown_alone_becomes_empty():
    assert normalize([{'text': '随便聊聊', 'intents': ['unknown']}])[0]['intents'] == []


def test_unknown_conflicts_with_other_labels():
    with pytest.raises(DataError, match='unknown'):
        normalize([{'text': '开车窗', 'intents': ['unknown', 'window']}])


def test_labels_deduplicated_and_sorted():
    result = normalize([{'text': '开车窗关车窗', 'intents': ['window', 'window', 'ac']}])
    assert result[0]['intents'] == ['ac', 'window']


def test_labels_must_be_nonempty_strings():
    for bad in ([1], [''], ['  '], None, 3, {'a': 1}):
        with pytest.raises(DataError):
            normalize([{'text': '开车窗', 'intents': bad}])


def test_labels_field_alias():
    assert normalize([{'text': '开车窗', 'labels': ['window']}])[0]['intents'] == ['window']


def test_too_many_labels_per_row():
    with pytest.raises(DataError, match='过多'):
        normalize([{'text': '开车窗', 'intents': [f'o{i}' for i in range(101)]}])


# ---------- missing / conflicting labels, malformed rows ----------

def test_missing_label_rejected():
    with pytest.raises(DataError, match='缺少标签'):
        normalize([{'text': '开车窗'}])


def test_expected_result_without_intents_rejected():
    with pytest.raises(DataError):
        normalize([{'text': '开车窗', 'expected_result': {}}])
    with pytest.raises(DataError):
        normalize([{'text': '开车窗', 'expected_result': 'window'}])


def test_conflicting_labels_for_identical_text_rejected():
    data = rows(('打开车窗', ['window']), ('打开车窗', ['sunroof']))
    with pytest.raises(DataError, match='冲突'):
        normalize(data)


def test_identical_text_same_labels_allowed():
    data = rows(('打开车窗', ['window']), ('打开车窗', ['window']))
    assert len(normalize(data)) == 2


def test_text_identity_ignores_whitespace_and_case():
    assert identity(' Open  Window ') == identity('openwindow')
    data = rows(('打开 车窗', ['window']), ('打开车窗', ['sunroof']))
    with pytest.raises(DataError, match='冲突'):
        normalize(data)


def test_non_dict_row_rejected():
    with pytest.raises(DataError, match='必须是对象'):
        normalize(['开车窗'])


def test_missing_or_bad_text_rejected():
    for row in ({'intents': []}, {'text': 42, 'intents': []}, {'text': '   ', 'intents': []},
                {'text': 'x' * 20001, 'intents': []}):
        with pytest.raises(DataError):
            normalize([row])


def test_empty_and_non_list_datasets_rejected():
    with pytest.raises(DataError):
        normalize([])
    with pytest.raises(DataError):
        normalize({'text': 'x'})


def test_row_limit():
    data = [{'id': f'r{i}', 'text': f'文本 {i}', 'intents': []} for i in range(20001)]
    with pytest.raises(DataError, match='20,000'):
        normalize(data)


# ---------- IDs ----------

def test_duplicate_id_rejected():
    data = [{'id': 'dup', 'text': '开车窗', 'intents': []}, {'id': 'dup', 'text': '开空调', 'intents': []}]
    with pytest.raises(DataError, match='重复'):
        normalize(data)


@pytest.mark.parametrize('bad_id', [42, '', '   ', None, 'x' * 129])
def test_invalid_id_rejected(bad_id):
    with pytest.raises(DataError):
        normalize([{'id': bad_id, 'text': '开车窗', 'intents': []}])


def test_state_field_alias_and_text_stripped():
    result = normalize([{'state': '  开车窗  ', 'intents': ['window']}])
    assert result[0]['text'] == '开车窗'


def test_group_id_must_be_nonempty_string():
    with pytest.raises(DataError):
        normalize([{'text': '开车窗', 'intents': [], 'group_id': ''}])
    with pytest.raises(DataError):
        normalize([{'text': '开车窗', 'intents': [], 'group_id': 7}])


# ---------- malformed uploads ----------

def test_parse_upload_malformed_json():
    with pytest.raises(DataError, match='文件格式无效'):
        parse_upload(b'{not json', 'f.json')


def test_parse_upload_malformed_jsonl_line():
    with pytest.raises(DataError):
        parse_upload(b'{"text": "a", "intents": []}\n{oops\n', 'f.jsonl')


def test_parse_upload_non_utf8():
    with pytest.raises(DataError, match='UTF-8'):
        parse_upload('开车窗'.encode('gbk'), 'f.json')


def test_parse_upload_wrapped_object_without_rows():
    with pytest.raises(DataError):
        parse_upload(json.dumps({'data': []}).encode(), 'f.json')


# ---------- profile / fingerprint ----------

def test_profile_counts():
    data = normalize(rows(('a 开车窗', ['window']), ('b 开车窗', ['window']),
                          ('c 车窗空调', ['window', 'ac']), ('d 闲聊', [])))
    p = profile(data)
    assert p['total'] == 4 and p['single'] == 2 and p['multi'] == 1 and p['unknown'] == 1
    assert p['object_count'] == 2
    assert p['labels'] == {'ac': 1, 'window': 3}
    assert p['duplicate_texts'] == 0
    assert p['warnings']  # fewer than 100 rows


def test_fingerprint_stable_and_order_of_keys_irrelevant():
    a = [{'id': 'x', 'text': 't', 'intents': [], 'group_id': 'g'}]
    b = [{'group_id': 'g', 'intents': [], 'text': 't', 'id': 'x'}]
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint([{'id': 'y', 'text': 't', 'intents': [], 'group_id': 'g'}])


# ---------- split_validation: same text / same group never crosses ----------

def test_split_keeps_groups_and_identical_texts_together():
    # r2 shares text with r1 (different explicit group_id) and r3 shares group_id with r1:
    # all three must land on the same side of the split.
    data = normalize([
        {'id': 'r1', 'text': '打开车窗', 'intents': ['window'], 'group_id': 'g1'},
        {'id': 'r2', 'text': '打开 车窗', 'intents': ['window'], 'group_id': 'g2'},
        {'id': 'r3', 'text': '关闭车窗', 'intents': ['window'], 'group_id': 'g1'},
    ] + [{'id': f'r{i}', 'text': f'独立句子 {i}', 'intents': ['ac'], 'group_id': f'g{i}'} for i in range(4, 12)])
    train, validation = split_validation(data)
    assert train and validation
    train_ids = {r['id'] for r in train}
    val_ids = {r['id'] for r in validation}
    assert not (train_ids & val_ids)
    for cluster in ({'r1', 'r2', 'r3'},):
        assert cluster <= train_ids or cluster <= val_ids
    # no identical normalized text crosses the split
    train_keys = {identity(r['text']) for r in train}
    val_keys = {identity(r['text']) for r in validation}
    assert not (train_keys & val_keys)
    assert not ({r['group_id'] for r in train} & {r['group_id'] for r in validation})


def test_split_deterministic_with_seed():
    data = normalize([{'id': f'r{i}', 'text': f'句子 {i}', 'intents': []} for i in range(20)])
    assert [r['id'] for r in split_validation(data)[1]] == [r['id'] for r in split_validation(data)[1]]


def test_split_validation_size_roughly_20_percent():
    data = normalize([{'id': f'r{i}', 'text': f'句子 {i}', 'intents': []} for i in range(10)])
    train, validation = split_validation(data)
    assert len(validation) == 2 and len(train) == 8


def test_split_requires_two_groups():
    data = normalize([{'id': 'r1', 'text': '同一句', 'intents': []},
                      {'id': 'r2', 'text': '同一句', 'intents': []}])
    with pytest.raises(DataError, match='两个独立文本组'):
        split_validation(data)


# ---------- train/test disjointness ----------

def test_assert_disjoint_rejects_same_text_modulo_whitespace_case():
    train = normalize([{'id': 'a', 'text': 'Open Window', 'intents': ['window']}])
    test = normalize([{'id': 'b', 'text': 'open  window', 'intents': ['window']}])
    with pytest.raises(DataError, match='重叠'):
        assert_disjoint(train, test)


def test_assert_disjoint_rejects_same_group_id():
    train = normalize([{'id': 'a', 'text': '开车窗', 'intents': [], 'group_id': 'shared'}])
    test = normalize([{'id': 'b', 'text': '完全不同的句子', 'intents': [], 'group_id': 'shared'}])
    with pytest.raises(DataError, match='重叠'):
        assert_disjoint(train, test)


def test_assert_disjoint_passes_when_separate():
    train = normalize([{'id': 'a', 'text': '开车窗', 'intents': []}])
    test = normalize([{'id': 'b', 'text': '开空调', 'intents': []}])
    assert_disjoint(train, test)  # no raise
