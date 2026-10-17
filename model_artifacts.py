"""Bounded, immutable, hash-checked model bundles. Never unpickles a checkpoint."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from datasets import DataError, parse_json

MAX_COMPRESSED = 512 * 1024 * 1024
MAX_EXPANDED = 1024 * 1024 * 1024
MAX_FILES = 500


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or '\x00' in name or path.is_absolute() or
            any(part in ('', '.', '..') for part in name.split('/')) or ':' in name):
        raise DataError('产物包含不安全路径。')
    if name != 'manifest.json' and path.parts[0] not in ('checkpoint', 'reports', 'logs', 'policy'):
        raise DataError('产物目录不在白名单。')
    if path.suffix not in ('.json', '.jsonl', '.log', '.txt', '.pt', '.safetensors', '.model', '.md', '.jinja'):
        raise DataError('产物文件类型不在白名单。')
    if any(part.startswith('.') for part in path.parts):
        raise DataError('产物不允许隐藏文件。')
    return path


def build_bundle(root, output, metadata):
    root, output = Path(root), Path(output)
    if output.exists():
        raise DataError('拒绝覆盖已有模型包。')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise DataError('模型包不允许符号链接。')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            safe_name(name)
            files[name] = {'sha256': file_sha(path), 'bytes': path.stat().st_size}
    if len(files) > MAX_FILES or sum(item['bytes'] for item in files.values()) > MAX_EXPANDED:
        raise DataError('模型包超过文件数或展开大小上限。')
    manifest = {**metadata, 'schema_version': 1, 'files': files}
    temporary = output.with_name('.' + output.name + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name in sorted(files):
                archive.write(root / name, name)
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False))
        if temporary.stat().st_size > MAX_COMPRESSED:
            raise DataError('模型包超过上传大小上限。')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def inspect_bundle(path, expected=None, destination=None):
    """Validate every entry before extraction. destination must not exist."""
    path = Path(path)
    if path.stat().st_size > MAX_COMPRESSED:
        raise DataError('模型包过大。')
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [item.filename for item in entries]
            if len(entries) > MAX_FILES + 1 or len(names) != len(set(names)) or names.count('manifest.json') != 1:
                raise DataError('模型包文件数、重复路径或 manifest 无效。')
            if sum(item.file_size for item in entries) > MAX_EXPANDED:
                raise DataError('模型包展开大小超限。')
            for item in entries:
                safe_name(item.filename)
                if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
                    raise DataError('模型包不允许目录条目、符号链接或加密条目。')
            manifest_info = archive.getinfo('manifest.json')
            if manifest_info.file_size > 1024 * 1024:
                raise DataError('manifest 过大。')
            manifest = parse_json(archive.read('manifest.json'))
            if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
                raise DataError('模型包 manifest 格式无效。')
            if expected and any(manifest.get(key) != value for key, value in expected.items()):
                raise DataError('模型包归属、attempt 或配置哈希不匹配。')
            if set(manifest['files']) != set(names) - {'manifest.json'}:
                raise DataError('模型包清单与实际文件不一致。')
            for name, meta in manifest['files'].items():
                if not isinstance(meta, dict) or meta.get('bytes') != archive.getinfo(name).file_size:
                    raise DataError('文件长度与清单不符。')
                digest = hashlib.sha256()
                with archive.open(name) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != meta.get('sha256'):
                    raise DataError('模型包文件哈希不匹配。')
            if destination is not None:
                destination = Path(destination)
                if destination.exists():
                    raise DataError('拒绝覆盖已有产物目录。')
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix='.bundle-', dir=destination.parent))
                try:
                    for name in names:
                        target = temporary.joinpath(*safe_name(name).parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(name) as source, target.open('xb') as output:
                            shutil.copyfileobj(source, output, 1024 * 1024)
                    os.replace(temporary, destination)
                finally:
                    if temporary.exists(): shutil.rmtree(temporary)
            return manifest
    except (zipfile.BadZipFile, RuntimeError, OSError, KeyError) as error:
        raise DataError('模型包损坏或不可读取。') from error
