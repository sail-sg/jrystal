#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

mapfile -t TARGETS < <(
  find . \
    \( -path "./.git" -o -path "./.venv" -o -path "./build" -o -path "./out" \
       -o -path "./_cache" -o -path "./docs/_build" -o -path "./__pycache__" \) -prune \
    -o -type f -name "*.py" -print | sort
)

echo "[lint] Using Python: ${PYTHON}"
echo "[lint] Running ruff"
"${PYTHON}" -m ruff check "${TARGETS[@]}"

echo "[lint] Running isort"
"${PYTHON}" -m isort --check "${TARGETS[@]}"

echo "[lint] Running yapf"
"${PYTHON}" -m yapf -r -d "${TARGETS[@]}"

echo "[lint] All checks passed."
