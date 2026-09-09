import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.blind_human_olfaction_benchmark import (
    _benjamini_hochberg,
    _classification_metrics,
    _partition_stimuli,
    sha256_file,
    verify_prediction_seal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_partition_is_structural_and_reserves_twenty_percent_per_stratum():
    stimuli = [
        {
            "stimulus_id": str(index),
            "components_per_mixture": 10,
            "declared_overlap_percent": 60.0,
        }
        for index in range(1, 11)
    ]
    stimuli.append(
        {
            "stimulus_id": "control",
            "components_per_mixture": 1,
            "declared_overlap_percent": 0.0,
        }
    )

    _partition_stimuli(stimuli)

    assert sum(row["evaluation_partition"] == "calibration" for row in stimuli) == 2
    assert sum(row["evaluation_partition"] == "final_test" for row in stimuli) == 8
    assert stimuli[-1]["evaluation_partition"] == "control"


def test_benjamini_hochberg_and_fixed_classification_metrics():
    rejected = _benjamini_hochberg(
        np.asarray([0.001, 0.01, 0.04, 0.20], dtype=float), alpha=0.05
    )
    assert rejected.tolist() == [True, True, False, False]

    metrics = _classification_metrics(
        np.asarray([True, True, False, False]),
        np.asarray([True, False, True, False]),
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["confusion"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }


def test_prediction_seal_rejects_post_seal_mutation(tmp_path):
    predictions = [{"stimulus_id": "1", "r2_similarity": 0.4}]
    rows_hash = hashlib.sha256(
        json.dumps(
            predictions,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    artifact = {
        "status": "blind_predictions_sealed_before_human_outcomes",
        "blind_contract": {
            "target_behavior_file_read": False,
            "human_outcomes_used_for_partitioning": False,
            "human_outcomes_used_for_model_or_threshold_selection": False,
            "prediction_rows_sha256": rows_hash,
        },
        "predictions": predictions,
    }
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(artifact), encoding="utf-8")
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(
        json.dumps(
            {
                "prediction_file": prediction_path.name,
                "prediction_file_sha256": sha256_file(prediction_path),
                "prediction_file_bytes": prediction_path.stat().st_size,
                "prediction_rows_sha256": rows_hash,
            }
        ),
        encoding="utf-8",
    )

    verify_prediction_seal(prediction_path, seal_path)
    prediction_path.write_text(json.dumps({**artifact, "tampered": True}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_prediction_seal(prediction_path, seal_path)


def test_published_bushdid_run_is_blind_sealed_and_fails_the_90_percent_gate():
    prediction_path = PROJECT_ROOT / "benchmarks" / "bushdid_blind_predictions_v1.json"
    seal_path = PROJECT_ROOT / "benchmarks" / "bushdid_blind_prediction_seal_v1.json"
    report_path = PROJECT_ROOT / "benchmarks" / "bushdid_human_blind_benchmark_v1.json"
    prediction, seal = verify_prediction_seal(prediction_path, seal_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    script = PROJECT_ROOT / "scripts" / "blind_human_olfaction_benchmark.py"

    assert prediction["implementation"]["script_sha256"] == sha256_file(script)
    assert report["implementation"]["script_sha256"] == sha256_file(script)
    assert prediction["dataset"]["partition_counts"] == {
        "calibration": 52,
        "final_test": 208,
        "control": 4,
    }
    assert report["dataset"]["subjects"] == 26
    assert report["blind_integrity"]["prediction_seal_verified_before_behavior_open"]
    assert datetime.fromisoformat(seal["sealed_at"]) < datetime.fromisoformat(
        report["blind_integrity"]["behavior_opened_at"]
    )
    results = report["final_test_results"]
    continuous = results["continuous_human_correct_rate"]
    assert continuous["r2_spearman"] == pytest.approx(0.2916991002354614)
    assert continuous["component_overlap_spearman"] == pytest.approx(
        0.6266899385255676
    )
    assert continuous["paired_difference_95_interval"][1] < 0.0
    assert results["fdr_discriminability"]["r2_roc_auc"] == pytest.approx(
        0.6394289471212549
    )
    gate = results["human_ceiling_90_percent_gate"]
    assert gate["point_estimate"] == pytest.approx(0.3409985907345352)
    assert gate["passed"] is False
    assert report["model_scope"]["molecule_disjoint"] is False
    assert report["model_scope"]["scaffold_disjoint"] is False
