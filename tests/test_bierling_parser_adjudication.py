from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.adjudicate_bierling_human_olfaction_parser import (
    COLUMN_ALIASES,
    OUTCOME_HEADER_SHA256,
    PARENT_SCRIPT_SHA256,
    UNSCORED_ZERO_ROW_TARGET,
    _validate_frozen_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parser_adjudication_is_narrow_and_parent_is_immutable():
    assert COLUMN_ALIASES == {
        "fruit": "fruity",
        "ammonia/urinous": "ammonia/urinuos",
    }
    parent = ROOT / "scripts" / "blind_bierling_human_olfaction_benchmark.py"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SCRIPT_SHA256
    assert UNSCORED_ZERO_ROW_TARGET == "4Isoprop"


def test_parser_adjudication_rejects_unrecognized_header_before_reading_rows(
    tmp_path: Path,
):
    outcome = tmp_path / "data.csv"
    outcome.write_text("inclusion;fruit;different\n1;1;0\n", encoding="utf-8")
    dummy = tmp_path / "dummy"
    dummy.write_text("x", encoding="utf-8")
    args = SimpleNamespace(
        predictions=dummy,
        seal=dummy,
        receipt=dummy,
        outcome=outcome,
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _validate_frozen_inputs(args)
    assert len(OUTCOME_HEADER_SHA256) == 64
