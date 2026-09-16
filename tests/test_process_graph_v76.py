from fragrance_ai.recommender.formulation_process_graph import process_state
import numpy as np
import pytest


def test_independent_lotion_premixes_can_be_prepared_in_either_order():
    values = {
        "mixer_kind": "high_shear",
        "method_code": 1,
        "batch_mass_g": 400,
        "measured_ph": 5.5,
    }
    prefix = ["brief", "material_review", "weigh"]
    for first, second in [("phase_a", "phase_b"), ("phase_b", "phase_a")]:
        result = process_state("cold_lotion", prefix + [first, second], values)
        assert not result["checks"][0]
        assert "fragrance_add" in result["allowed"]


def test_revision_opens_a_new_perfume_trial_without_reusing_quality_approval():
    prefix = [
        "brief",
        "material_review",
        "weigh",
        "basic_accord",
        "blend",
        "dilute",
        "compare",
        "revise",
    ]
    result = process_state("perfume", prefix + ["blend", "dilute", "compare"], {})
    assert not result["checks"][0] and "revise" in result["allowed"]
    assert (
        "fill_record" not in result["allowed"] and not result["manufacturing_approved"]
    )
    full = [
        "brief",
        "material_review",
        "weigh",
        "basic_accord",
        "modify",
        "blend",
        "fix",
        "dilute",
        "compare",
        "revise",
    ]
    for repeat in (["modify", "blend"], ["weigh", "basic_accord", "blend"]):
        assert not process_state("perfume", full + repeat, {})["checks"][0]


def test_dependency_conflict_cannot_be_waived_by_a_permutation():
    result = process_state(
        "hot_lotion",
        ["brief", "material_review", "weigh", "phase_a", "phase_b", "emulsify"],
        {"peak_temperature_c": 72.5},
    )
    assert result["allowed"] == ["hold"]
    assert result["checks"][0]


def test_source_numeric_conditions_are_still_checked():
    result = process_state(
        "hot_lotion", [], {"peak_temperature_c": 90.0, "method_code": 2}
    )
    assert result["allowed"] == ["hold"]
    assert result["checks"][4]


def test_runtime_uses_legal_set_without_hiding_unconstrained_prediction():
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.recommender.formulation_process import ACTIONS
    from fragrance_ai.recommender.formulation_process_graph import VERSION

    core = object.__new__(FormulationCore)
    core.feature_width = 1599
    core.actions = tuple(ACTIONS)
    core.sha256 = "a" * 64
    core.manifest = {
        "process_graph_training": {
            "schema": VERSION,
            "source_legal_holdout_passed": True,
        }
    }
    logits = np.zeros((1, len(ACTIONS)))
    logits[0, ACTIONS.index("fill_record")] = 10
    logits[0, ACTIONS.index("phase_b")] = 2
    core.forward = lambda *args: {"action": logits, "check": np.zeros((1, 8))}
    history = [{"action": a} for a in ("brief", "material_review", "weigh")]
    result = core.procedure("cold_lotion", history, values={"mixer_kind": "high_shear"})
    assert set(result["allowed_next_actions"]) == {"phase_a", "phase_b"}
    assert result["next_action"] == "phase_b"
    assert result["unconstrained_next_action"] == "fill_record"
    assert result["unconstrained_action_source_consistent"] is False
    assert result["model_probability"] < result["conditional_source_probability"]


def test_unverified_graph_cannot_mask_model_action():
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.recommender.formulation_process import ACTIONS

    core = object.__new__(FormulationCore)
    core.feature_width = 1599
    core.actions = tuple(ACTIONS)
    core.manifest = {"process_graph_training": {"schema": "untrained"}}
    core.forward = lambda *args: {
        "action": np.zeros((1, len(ACTIONS))),
        "check": np.zeros((1, 8)),
    }
    with pytest.raises(ValueError, match="verified source dependency"):
        core.procedure("perfume")
