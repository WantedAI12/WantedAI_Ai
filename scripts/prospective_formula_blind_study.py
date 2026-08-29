#!/usr/bin/env python3
"""Source-tree wrapper for the packaged prospective study CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fragrance_ai.research.prospective_formula_study_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
