#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' || {
  echo "ShadeForge requires Python 3.10 or newer." >&2
  exit 1
}
"$PYTHON" -m pip install -e '.[preview]'
echo
echo "Installed. Next: shadeforge doctor"
echo "Load the Claude Code plugin with:  claude --plugin-dir \"$(pwd)/claude-plugin\""
