#!/usr/bin/env python3
"""Run ShadeForge from the repo checkout, whether or not `pip install -e .` was run."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "shadeforge" / "cli.py").exists():
    sys.path.insert(0, str(REPO_ROOT))

from shadeforge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
