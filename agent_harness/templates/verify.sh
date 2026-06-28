#!/usr/bin/env bash
# verify_with_openjml — run OpenJML ESC on Solution.java.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/harness/_env.sh"
exec "$PYTHON" "$SCRIPT_DIR/harness/cli.py" verify "$@"
