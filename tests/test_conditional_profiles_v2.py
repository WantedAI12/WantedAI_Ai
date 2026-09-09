"""Contract tests, separate from the actual human-data development metrics."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

from fragrance_ai.research import conditional_profiles as cp
from scripts import train_conditional_profiles_v2 as runner


def training_fixture():
    keys = [(str(i + 1), "0.01", "pg") for i in range(10)]
    keys += [("1", "1", "pg"), ("-1", "1", "nt")]
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 4))
    x[-1] = np.nan
    y = np.abs(rng.normal(size=(12, 3)))
    return keys, x, y


def test_exact_observation_replayed_without_creating_graph_for_solvent():
    keys, x, y = training_fixture()
    model = cp.fit_conditional(x, y, keys, 10, anchored=True)
    prediction, status = cp.predict_conditional(model, x, keys)
    assert np.array_equal(prediction, y)
    assert status[-1]["basis"] == "measured_solvent_control"
    unknown, unknown_status = cp.predict_conditional(model, x[-1:], [("-1", "0.5", "nt")])
    assert np.isnan(unknown).all()
    assert unknown_status[0]["basis"] == "unsupported_structure"


def test_measured_log_dose_interpolation_uses_same_molecule_and_solvent():
    keys, x, y = training_fixture()
    model = cp.fit_conditional(x, y, keys, 10, anchored=True)
    prediction, status = cp.predict_conditional(model, x[:1], [("1", "0.1", "pg")])
    assert np.allclose(prediction[0], (y[0] + y[10]) / 2)
    assert status[0]["basis"] == "same_molecule_solvent_log_dose_interpolation"
    wrong_solvent, status = cp.predict_conditional(model, x[:1], [("1", "0.1", "dep")])
    assert status[0]["basis"] == "molecular_prediction"
    plain = cp.fit_conditional(x, y, keys, 10, anchored=False)
    expected, _ = cp.predict_conditional(plain, x[:1], [("1", "0.1", "dep")])
    assert np.array_equal(wrong_solvent, expected)


def test_extrapolation_is_explicitly_a_proxy_not_a_measurement():
    keys, x, y = training_fixture()
    model = cp.fit_conditional(x, y, keys, 10, anchored=True)
    prediction, status = cp.predict_conditional(model, x[:1], [("1", "0.001", "pg")])
    assert np.isfinite(prediction).all()
    assert status[0]["basis"] == "same_molecule_solvent_extrapolation_proxy"
    assert not status[0]["measured_at_requested_stock"]
    assert status[0]["anchor_distance_decades"] == 1


def test_held_out_molecule_cannot_use_its_observation_as_an_anchor():
    keys, x, y = training_fixture()
    training = np.asarray([key[0] != "1" for key in keys])
    model = cp.fit_conditional(x[training], y[training], [key for key, keep in zip(keys, training) if keep], 10, anchored=True)
    _, status = cp.predict_conditional(model, x[:1], [keys[0]])
    assert status[0]["basis"] == "molecular_prediction"
    assert all(row["key"][0] != "1" for row in model["anchors"])


def test_plain_regressor_does_not_replay_training_labels():
    keys, x, y = training_fixture()
    model = cp.fit_conditional(x, y, keys, 10, anchored=False)
    prediction, status = cp.predict_conditional(model, x, keys)
    assert not np.allclose(prediction[:-1], y[:-1])
    assert all(not row["measured_at_requested_stock"] for row in status)


def test_portable_conditional_model_replays_exactly():
    keys, x, y = training_fixture()
    model = cp.fit_conditional(x, y, keys, 10, anchored=True)
    restored = json.loads(json.dumps(model, allow_nan=False))
    a, sa = cp.predict_conditional(model, x, keys)
    b, sb = cp.predict_conditional(restored, x, keys)
    assert np.array_equal(a, b)
    assert sa == sb


def test_duplicate_conditions_and_invalid_human_values_fail():
    keys, x, y = training_fixture()
    keys[1] = keys[0]
    with pytest.raises(ValueError, match="duplicate stock"):
        cp.fit_conditional(x, y, keys, 10, anchored=True)
    y[0, 0] = -1
    with pytest.raises(ValueError):
        cp.fit_conditional(x, y, keys, 10, anchored=True)


def test_artifact_cannot_enable_a_human_accuracy_claim():
    keys, x, y = training_fixture()
    model = cp.fit_conditional(x, y, keys, 10, anchored=True)
    model["actual_human_accuracy_90_authorized"] = True
    with pytest.raises(ValueError, match="cannot authorize"):
        cp.predict_conditional(model, x, keys)


def test_chirality_is_kept_in_structure_features():
    first = cp.molecule_features("C[C@H](O)F")
    second = cp.molecule_features("C[C@@H](O)F")
    assert first["canonical_smiles"] != second["canonical_smiles"]
    assert first["fingerprint_bits"] != second["fingerprint_bits"]
    assert len(first["physical"]) == len(cp.PHYSICAL)


def test_graph_prediction_extends_native_coverage_without_claiming_identity():
    molecule = cp.molecule_features("CCO", native=None)
    bank = {"123": molecule}
    key = ("123", ".1", "pg")
    assert cp.features_for(key, bank, "native") is None
    assert np.isfinite(cp.features_for(key, bank, "molecular")).all()
    assert cp.features_for(("-1", "1", "nt"), bank, "molecular") is None


def test_carrier_features_remain_distinct_and_unknown_is_marked():
    bank = {"1": cp.molecule_features("CCO")}
    features = [cp.features_for(("1", ".1", solvent), bank) for solvent in cp.CARRIERS]
    assert len({tuple(value[-len(cp.CARRIERS):]) for value in features}) == len(cp.CARRIERS)


def test_nested_selection_uses_inner_groups_only():
    keys, x, y = training_fixture()
    _, audit = cp.choose_conditional(x, y, keys, [key[0] for key in keys], anchored=True)
    assert audit["selection_scope"] == "inner_training_groups_only"
    assert {row["alpha"] for row in audit["candidates"]} == set(cp.ALPHAS)


def test_guard_blocks_target_outcomes_and_unapproved_source_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    report = tmp_path / "HUMAN_MIXTURE_PROFILES_V1" / "REPORT.JSON"
    report.parent.mkdir()
    paths = [source / "TASK2_Leaderboard_ActualValue.csv", source / "unapproved.csv", tmp_path / "TASK2_Test_ActualValue.csv", report]
    for path in paths:
        path.write_text("not read", encoding="utf-8")
    code = f"""
from pathlib import Path
from scripts.train_conditional_profiles_v2 import install_training_only_guard
attempts=install_training_only_guard(Path({str(source)!r}))
for name in {list(map(str, paths))!r}:
    try:
        Path(name).read_text()
    except PermissionError:
        pass
    else:
        raise RuntimeError('forbidden file was read')
assert len(attempts)==4
"""
    subprocess.run([sys.executable, "-c", code], cwd=runner.ROOT, check=True, capture_output=True, text=True)


def test_outcome_files_are_not_in_the_allowed_source_contract():
    assert "target_outcomes" not in runner.INPUTS
    assert all("ActualValue" not in path for path, _ in runner.INPUTS.values())
    assert runner.INPUTS["task1_training"][0].endswith("TASK1_training_All.csv")


def test_blend_portable_replay_keeps_unsupported_rows_missing():
    rng = np.random.default_rng(9)
    x, y, mean = rng.normal(size=(30, 6)), np.abs(rng.normal(size=(30, 3))), np.abs(rng.normal(size=(30, 3)))
    x[-1] = mean[-1] = np.nan
    fitted, _ = runner.fit_blend(x, y, mean, [str(i // 2) for i in range(30)])
    a = runner.predict_blend(fitted, x, mean)
    b = runner.predict_blend(json.loads(json.dumps(fitted)), x, mean)
    assert np.allclose(a, b, equal_nan=True, atol=0, rtol=0)
    assert np.isnan(a[-1]).all()


def test_v1_hashes_remain_bound_and_target_scoring_absent():
    for path, digest in runner.FROZEN_V1_CODE.items():
        assert runner.v1.sha(runner.ROOT / path) == digest
    source = inspect.getsource(runner.run)
    assert '"target_outcomes_scored": False' in source
    assert "v1.score(" not in source


def test_task1_join_deduplicates_same_stimulus_and_keeps_new_dose(monkeypatch, tmp_path):
    observed = {"stimulus": "a", "components": "1", "molecule": "1", "dilution": ".01",
                **{name: "1" for name in runner.v1.ENDPOINTS}}
    source_rows = {
        runner.INPUTS["single"][0]: [observed],
        runner.INPUTS["task1_stimuli"][0]: [
            {"stimulus": "a", "molecule": "1", "dilution": ".01", "solvent": "PG"},
            {"stimulus": "b", "molecule": "1", "dilution": ".1", "solvent": "PG"}],
        runner.INPUTS["task1_training"][0]: [observed, {"stimulus": "b", **{name: "2" for name in runner.v1.ENDPOINTS}}],
    }
    monkeypatch.setattr(runner.v1, "load_design", lambda source: ({}, {}, {}, {}))
    monkeypatch.setattr(runner.v1, "rows", lambda path: source_rows[path.relative_to(tmp_path).as_posix()])
    keys, y, _, audit = runner.extended_observations(tmp_path)
    assert len(keys) == 2
    assert audit["cross_file_duplicate_stimuli_removed"] == 1
    assert audit["observations_after_stimulus_deduplication"] == 2
    assert np.array_equal(y[:, 0], [1, 2])
    source_rows[runner.INPUTS["task1_training"][0]][0] = {**observed, "Woody": "2"}
    with pytest.raises(ValueError, match="conflicting duplicate"):
        runner.extended_observations(tmp_path)
