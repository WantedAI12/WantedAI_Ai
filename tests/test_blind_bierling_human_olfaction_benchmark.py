from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.blind_bierling_human_olfaction_benchmark import (
    BOOTSTRAP_DRAWS,
    CANDIDATES,
    ENDPOINTS,
    OUTCOME_BYTES,
    OUTCOME_FILE,
    OUTCOME_MD5,
    OUTCOME_URL,
    QUALITATIVE_ENDPOINTS,
    SCHEMA_VERSION,
    KELLER_SOURCE_CONTRACT,
    TARGET_METADATA_SHA256,
    VARIABLE_DICTIONARY_SHA256,
    _parse_binary_series,
    assert_target_outcomes_absent,
    canonical_json_sha256,
    spearman,
    verify_prediction_seal,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bierling_endpoint_and_model_selection_contract_is_frozen():
    assert len(ENDPOINTS) == 22
    assert len(set(ENDPOINTS)) == 22
    assert len(QUALITATIVE_ENDPOINTS) == 16
    assert set(QUALITATIVE_ENDPOINTS).issubset(ENDPOINTS)
    assert len({row["name"] for row in CANDIDATES}) == len(CANDIDATES)
    assert {row["algorithm"] for row in CANDIDATES} == {
        "ridge",
        "extra_trees",
        "similarity_knn",
    }
    assert BOOTSTRAP_DRAWS == 1000
    assert OUTCOME_FILE == "data.csv"
    assert OUTCOME_BYTES == 4_898_146
    assert OUTCOME_MD5 == "2b0591617a6e806ee619c2c2466a03ed"
    assert OUTCOME_URL.endswith("/15657278/files/data.csv/content")
    assert len(TARGET_METADATA_SHA256) == 64
    assert len(VARIABLE_DICTIONARY_SHA256) == 64
    assert set(KELLER_SOURCE_CONTRACT) == {
        "molecules.csv",
        "stimuli.csv",
        "behavior.csv",
    }


def test_rank_metric_handles_ties_and_rejects_invalid_inputs():
    assert spearman([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2])
    with pytest.raises(ValueError):
        spearman([1, np.nan, 3], [1, 2, 3])


def test_binary_parser_is_explicit_and_fail_closed():
    parsed = _parse_binary_series([1, 0, True, False, "yes", "no", "", None])
    assert parsed[:6].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert np.isnan(parsed[6:]).all()
    with pytest.raises(RuntimeError):
        _parse_binary_series(["maybe"])


def test_target_outcome_absence_check_covers_csv_and_xlsx(tmp_path):
    target = tmp_path / "data.csv"
    assert_target_outcomes_absent(target)
    (tmp_path / "data.xlsx").write_bytes(b"opened-outcome")
    with pytest.raises(RuntimeError, match="data.xlsx"):
        assert_target_outcomes_absent(target)


def test_prediction_seal_detects_prediction_and_script_mismatch(tmp_path):
    script_hash = hashlib.sha256(
        (ROOT / "scripts" / "blind_bierling_human_olfaction_benchmark.py").read_bytes()
    ).hexdigest()
    rows = [{"molcode": "A", "prediction": 1.0}]
    prediction = {
        "implementation": {"script_sha256": script_hash},
        "predictions": rows,
    }
    prediction_path = tmp_path / "prediction.json"
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "prediction_file_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "prediction_file_bytes": prediction_path.stat().st_size,
        "prediction_rows_sha256": canonical_json_sha256(rows),
        "benchmark_script_sha256": script_hash,
        "target_outcome": {
            "filename": OUTCOME_FILE,
            "url": OUTCOME_URL,
            "expected_bytes": OUTCOME_BYTES,
            "expected_md5": OUTCOME_MD5,
            "path": str(tmp_path / OUTCOME_FILE),
            "present_before_seal": False,
        },
    }
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    verified = verify_prediction_seal(prediction_path, seal_path)
    assert verified["seal"]["prediction_rows_sha256"] == canonical_json_sha256(rows)

    prediction["predictions"][0]["prediction"] = 2.0
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    with pytest.raises(RuntimeError, match="prediction file hash mismatch"):
        verify_prediction_seal(prediction_path, seal_path)


def test_published_bierling_result_remains_bound_when_present():
    prediction = ROOT / "benchmarks" / "bierling_2025_blind_predictions_v1.json"
    seal = ROOT / "benchmarks" / "bierling_2025_blind_prediction_seal_v1.json"
    report = ROOT / "benchmarks" / "bierling_2025_human_blind_benchmark_v1.json"
    if not report.is_file():
        if not prediction.exists() and not seal.exists():
            return
        assert prediction.is_file() and seal.is_file()
        verified = verify_prediction_seal(prediction, seal)
        assert len(verified["predictions"]["predictions"]) == 74
        return
    verified = verify_prediction_seal(prediction, seal)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert len(verified["predictions"]["predictions"]) == 74
    assert result["blind_integrity"]["prediction_sha256"] == hashlib.sha256(
        prediction.read_bytes()
    ).hexdigest()
    assert result["human_olfactory_90_percent_certified"] is False
    assert result["generated_recipe_similarity_validated"] is False
    assert result["dataset"]["population"]["participants"] == 1119
    assert result["dataset"]["population"]["odors"] == 73
    assert result["improvement_gate"]["passed"] is False
    assert result["measured_odor_improvement_gate"]["passed"] is True
    assert result["parser_adjudication"]["developed_after_target_file_opened"] is True
    assert (
        result["parser_adjudication"]["changes"]["unscored_zero_row_target"]
        == "4Isoprop"
    )
    primary = result["results"]["primary_target_exact_label_excluded"]
    baseline = result["results"]["fixed_rdkit_baseline"]
    assert primary["macro_endpoint_spearman"] == pytest.approx(0.34675334654216905)
    assert baseline["macro_endpoint_spearman"] == pytest.approx(0.24241133497812523)
    assert primary["positive_endpoint_count"] == 22
    assert result["two_way_bootstrap"]["primary_minus_baseline_95_interval"][0] > 0
