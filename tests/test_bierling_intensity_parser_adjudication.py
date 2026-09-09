from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.adjudicate_bierling_intensity_pilot_parser import (
    COLUMN_ALIASES,
    HEADER_SHA256,
    PARENT_SHA256,
    EXPECTED_ZERO_RATINGS,
    EXPECTED_REPEATED_ANCHOR_GROUPS,
    EXPECTED_REPEATED_ANCHOR_ROWS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_intensity_parser_adjudication_is_one_alias_and_parent_is_frozen():
    assert COLUMN_ALIASES == {"intensity": "intensive"}
    assert len(HEADER_SHA256) == 64
    parent = ROOT / "scripts" / "blind_bierling_intensity_pilot_benchmark.py"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA256
    assert EXPECTED_ZERO_RATINGS == 12
    assert EXPECTED_REPEATED_ANCHOR_ROWS == 4
    assert EXPECTED_REPEATED_ANCHOR_GROUPS == 2
