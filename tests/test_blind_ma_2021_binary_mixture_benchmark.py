from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.blind_ma_2021_binary_mixture_benchmark import (
    ALL_PAIR_COUNT,
    BOOTSTRAP_DRAWS,
    METADATA_BYTES,
    METADATA_MD5,
    OUTCOME_BYTES,
    OUTCOME_FILE,
    TARGET_ODOR_COUNT,
    _dominance_weighted_pleasantness,
    _fechner_pool,
    _pair_id,
    _parse_metadata,
    assert_target_outcome_absent,
    normalize_name,
)


ROOT = Path(__file__).resolve().parents[1]


def test_ma_public_source_and_blind_contract_are_frozen():
    assert TARGET_ODOR_COUNT == 72
    assert ALL_PAIR_COUNT == 2556
    assert METADATA_BYTES == 7013
    assert METADATA_MD5 == "9c6fa6328950a9ddfd1c372a77aaaa0f"
    assert OUTCOME_FILE == "data in brief V2.xlsx"
    assert OUTCOME_BYTES == 487926
    assert BOOTSTRAP_DRAWS == 5000


def test_name_and_pair_normalization_are_symmetric():
    assert normalize_name(" γ-undecalactone\u00a0") == "gammaundecalactone"
    assert normalize_name("β-phenethyl acetate") == "betaphenethylacetate"
    assert _pair_id("100-52-7", "4180-23-8") == _pair_id(
        "4180-23-8", "100-52-7"
    )


def test_fechner_pool_is_symmetric_bounded_and_reduces_to_max():
    assert _fechner_pool(3.0, 7.0, 0.2) == pytest.approx(
        _fechner_pool(7.0, 3.0, 0.2)
    )
    assert _fechner_pool(3.0, 7.0, 0.2, independent_channel_fraction=0.0) == 7.0
    assert 7.0 < _fechner_pool(7.0, 7.0, 0.2) < 7.2
    assert _fechner_pool(11.0, 11.0, 0.2) == 11.0
    with pytest.raises(ValueError):
        _fechner_pool(3.0, 7.0, 0.0)
    with pytest.raises(ValueError):
        _fechner_pool(3.0, 7.0, 0.2, independent_channel_fraction=1.1)


def test_pleasantness_pool_is_symmetric_and_dominance_weighted():
    first = _dominance_weighted_pleasantness(2.0, 8.0, 9.0, 1.0)
    second = _dominance_weighted_pleasantness(8.0, 2.0, 1.0, 9.0)
    assert first == pytest.approx(second)
    assert 2.0 <= first < 3.0


def test_metadata_parser_is_fail_closed_and_keeps_trial_numbers():
    header = "CAS.\tOdorant\tOdor\tCons.(mg/mL)\tSolvent\tPurity\tTrialnumber\n"
    rows = (
        '100-52-7\tbenzaldehyde\talmond\t3.82\t1,2-propanediol\t0.99\t"33, 135"\n'
        "4180-23-8\ttrans-anethol\tanise\t4.4\tmineral oil\t>=99%\t35\n"
    )
    parsed = _parse_metadata((header + rows).encode(), expected_count=2)
    assert parsed[0]["metadata_trial_numbers"] == [33, 135]
    duplicate = header + rows.splitlines(keepends=True)[0] * 2
    with pytest.raises(RuntimeError):
        _parse_metadata(duplicate.encode(), expected_count=2)


def test_outcome_absence_check_covers_original_and_legacy_names(tmp_path):
    target = tmp_path / OUTCOME_FILE
    assert_target_outcome_absent(target)
    (tmp_path / "data in brief.xlsx").write_bytes(b"outcome")
    with pytest.raises(RuntimeError):
        assert_target_outcome_absent(target)


def test_published_ma_report_never_claims_human_90_percent():
    report = ROOT / "benchmarks" / "ma_2021_binary_mixture_blind_benchmark_v1.json"
    if not report.is_file():
        return
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["human_olfactory_90_percent_certified"] is False
    assert value["complex_perfume_recipe_validated"] is False
    assert value["blind_integrity"]["all_2556_pair_predictions_preceded_outcome"] is True
