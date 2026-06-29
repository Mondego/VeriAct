#!/usr/bin/env bash
# Resolve a Python interpreter that can `import javalang`; bootstrap a local
# venv on first use if none is available. Sets and exports $PYTHON.
# Sourced by the task-dir wrapper scripts (verify.sh, analyze.sh, ...).
set -euo pipefail

# Cap OpenJML's JVM heap. Without this the JVM defaults its max heap to ~1/4 of
# physical RAM (huge on big machines), and N parallel verifies multiply it.
export OPENJML_JVM="${OPENJML_JVM:--Xmx8g}"

_HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${VERIACT_PYTHON:-}" ]; then
  PYTHON="$VERIACT_PYTHON"
elif python -c 'import javalang' >/dev/null 2>&1; then
  PYTHON="python"
elif python3 -c 'import javalang' >/dev/null 2>&1; then
  PYTHON="python3"
elif [ -x "$_HARNESS_DIR/.venv/bin/python" ]; then
  PYTHON="$_HARNESS_DIR/.venv/bin/python"
else
  echo "[agent_harness] bootstrapping venv at $_HARNESS_DIR/.venv ..." >&2
  if command -v uv >/dev/null 2>&1; then
    uv venv "$_HARNESS_DIR/.venv" >&2
    uv pip install --python "$_HARNESS_DIR/.venv/bin/python" \
      -r "$_HARNESS_DIR/requirements.txt" >&2
  else
    python3 -m venv "$_HARNESS_DIR/.venv" >&2
    "$_HARNESS_DIR/.venv/bin/python" -m pip install -q --upgrade pip >&2
    "$_HARNESS_DIR/.venv/bin/python" -m pip install -q \
      -r "$_HARNESS_DIR/requirements.txt" >&2
  fi
  PYTHON="$_HARNESS_DIR/.venv/bin/python"
fi

export PYTHON
