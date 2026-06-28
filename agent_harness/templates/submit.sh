#!/usr/bin/env bash
# submit — record the final submission (writes submission.json).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/harness/_env.sh"
exec "$PYTHON" "$SCRIPT_DIR/harness/cli.py" submit "$@"
