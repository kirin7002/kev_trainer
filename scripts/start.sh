#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo '请先创建 .venv 并安装 requirements-lock.txt；步骤见 README。' >&2
  exit 1
fi
cd "$ROOT"
exec "$PYTHON" "$ROOT/server.py" "$@"
