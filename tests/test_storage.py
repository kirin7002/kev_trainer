"""Store: path validation, atomic JSON writes, listing, restart recovery data."""
import json
import uuid

import pytest

from storage import Store


def test_store_creates_layout(tmp_path):
    store = Store(tmp_path / 'state')
    for name in ('datasets', 'runs', 'agent-work'):
        assert (tmp_path / 'state' / name).is_dir()


@pytest.mark.parametrize('bad', ['../datasets', 'abc', '', str(uuid.uuid4()) + 'x', '../../etc/passwd'])
def test_path_rejects_non_uuid_ids(tmp_path, bad):
    store = Store(tmp_path / 'state')
    with pytest.raises(KeyError):
        store.path('datasets', bad)


def test_path_none_id_should_be_keyerror(tmp_path):
    store = Store(tmp_path / 'state')
    with pytest.raises(KeyError):
        store.path('datasets', None)


def test_path_accepts_valid_uuid(tmp_path):
    store = Store(tmp_path / 'state')
    good = str(uuid.uuid4())
    assert store.path('datasets', good) == tmp_path / 'state' / 'datasets' / good


def test_write_read_roundtrip_unicode(tmp_path):
    store = Store(tmp_path / 'state')
    path = tmp_path / 'state' / 'runs' / 'x' / 'meta.json'
    value = {'名称': '实验一', 'n': 1, 'nested': [1, 2, {'a': None}]}
    store.write(path, value)
    assert store.read(path) == value
    assert '实验一' in path.read_text()  # not \uXXXX-escaped


def test_write_rejects_nan(tmp_path):
    store = Store(tmp_path / 'state')
    with pytest.raises(ValueError):
        store.write(tmp_path / 'state' / 'bad.json', {'x': float('nan')})
    assert not list((tmp_path / 'state').glob('.*.tmp'))


def test_write_leaves_no_temp_files_and_overwrites(tmp_path):
    store = Store(tmp_path / 'state')
    path = tmp_path / 'state' / 'runs' / 'a.json'
    store.write(path, {'v': 1})
    store.write(path, {'v': 2})
    assert store.read(path) == {'v': 2}
    assert [p.name for p in path.parent.iterdir()] == ['a.json']


def test_read_missing_raises_keyerror(tmp_path):
    store = Store(tmp_path / 'state')
    with pytest.raises(KeyError):
        store.read(tmp_path / 'state' / 'missing.json')


def test_list_sorted_by_created_at_desc(tmp_path):
    store = Store(tmp_path / 'state')
    for i, name in enumerate(('old', 'new')):
        item_id = str(uuid.uuid4())
        store.write(store.path('datasets', item_id) / 'meta.json',
            {'id': item_id, 'name': name, 'created_at': f'2026-01-0{i + 1}T00:00:00+00:00'})
    listed = store.list('datasets')
    assert [m['name'] for m in listed] == ['new', 'old']
    # directories without meta.json are ignored
    (tmp_path / 'state' / 'datasets' / 'junk').mkdir()
    assert len(store.list('datasets')) == 2
