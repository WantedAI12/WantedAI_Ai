import json
from pathlib import Path

from scripts.benchmark_huggingface_olfaction import (
    _paired_unique_record_inference,
    compare_language_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parser_versions_do_not_claim_a_confidence_delta_across_changed_scales(
    tmp_path,
):
    baseline = {
        "claim_boundary": "parser coverage only",
        "tracks": {
            name: {
                "inputs": inputs,
                "parse_coverage_percent": 40.0,
                "mean_semantic_confidence": 0.05,
            }
            for name, inputs in (
                ("odor2ms_test", 303),
                ("huggingface_50_label_prompts", 50),
            )
        },
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current = {
        name: {
            "inputs": inputs,
            "formula_ready_parse_coverage_percent": 80.0,
            "recognition_coverage_percent": 100.0,
            "mean_semantic_confidence": 0.95,
        }
        for name, inputs in (
            ("odor2ms_test", 303),
            ("huggingface_50_label_prompts", 50),
        )
    }

    result = compare_language_coverage(current, baseline_path)

    for comparison in result["tracks"].values():
        assert (
            comparison["mean_semantic_confidence_comparable_across_versions"] is False
        )
        assert "mean_semantic_confidence_delta" not in comparison
        assert (
            comparison["current_formula_ready_minus_legacy_parse_delta_points"] == 40.0
        )


def test_unique_record_inference_removes_repeat_duplication_and_is_paired():
    targets = [float(index) for index in range(8)]
    record_ids = [f"record-{index}" for index in range(8)]
    r2_rows = []
    hf_rows = []
    for repeat in range(2):
        r2_predictions = [value + repeat * 0.01 for value in targets]
        hf_predictions = list(reversed(targets))
        r2_rows.append(
            {
                "predictions": r2_predictions,
                "targets": targets,
            }
        )
        hf_rows.append(
            {
                "record_ids": record_ids,
                "targets": targets,
                "algorithm_predictions": {"adapter": hf_predictions},
            }
        )

    result = _paired_unique_record_inference(
        r2_rows,
        hf_rows,
        algorithm="adapter",
        seed=7,
        bootstrap_draws=2_000,
        randomization_draws=2_000,
    )

    assert result["unique_records"] == 8
    assert result["repeated_split_appearances"] == 16
    assert result["spearman_delta"] == 2.0
    assert result["paired_record_bootstrap_95_interval"][0] > 0
    assert result["paired_randomization_two_sided_p"] < 0.05


def test_published_molecule_disjoint_advantage_has_positive_lower_bounds():
    report = json.loads(
        (PROJECT_ROOT / "benchmarks" / "huggingface_olfaction_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    mixture = report["mixture_similarity"]
    molecule = mixture["final_comparison"]["molecule_disjoint"]

    assert mixture["molecule_disjoint_statistical_gate"]["passed"] is True
    assert molecule["r2_minus_molformer"][
        "paired_repeat_bootstrap_95_interval"
    ][0] > 0
    assert molecule["unique_record_inference"][
        "paired_record_bootstrap_95_interval"
    ][0] > 0
    assert (
        molecule["unique_record_inference"][
            "paired_randomization_two_sided_p"
        ]
        < 0.05
    )
    assert mixture["superseded_single_seed_reference"]["molecule_disjoint"][
        "r2_minus_molformer"
    ]["paired_repeat_bootstrap_95_interval"][0] < 0
