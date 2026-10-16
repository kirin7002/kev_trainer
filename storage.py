import json
import os
from pathlib import Path
import threading
import uuid


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        for name in ('datasets', 'runs', 'agent-work'):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def path(self, kind, item_id):
        try:
            if str(uuid.UUID(item_id)) != item_id:
                raise ValueError()
        except (ValueError, AttributeError, TypeError):
            raise KeyError('记录不存在。')
        return self.root / kind / item_id

    def write(self, path, value):
        with self.lock:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('x', encoding='utf-8') as handle:
                    json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def read(self, path):
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            raise KeyError('记录不存在。')

    def list(self, kind):
        with self.lock:
            return sorted((self.read(p / 'meta.json') for p in (self.root / kind).iterdir()
                if p.is_dir() and (p / 'meta.json').is_file()), key=lambda r: r['created_at'], reverse=True)
