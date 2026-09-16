"""Independent zero-dose, carrier balance, evidence and derivative contracts."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_dose_refinement import materials
from fragrance_ai.recommender.dose_refinement import DoseModel
from fragrance_ai.recommender.science import TemporalMixtureSimulator
from fragrance_ai.recommender.mixture_physics import (
    carrier_moles_per_gram,
    hill_response,
)
from fragrance_ai.recommender.physical_evidence_v76 import load


def forward(items, weights, concentration, draws=64):
    engine = TemporalMixtureSimulator()
    lines = [
        SimpleNamespace(
            ingredient_id=i.ingredient_id,
            finished_product_percent=w * concentration,
            active_strength_percent=i.active_strength_percent,
        )
        for i, w in zip(items, weights)
    ]
    prepared = engine._prepare(lines, {i.ingredient_id: i for i in items}, {})
    _, signal, profile = engine._sampled_temporal_arrays(
        prepared,
        engine._interaction_matrix(prepared),
        [(np.ones(19) / 19, [], [])] * 5,
        draws,
        np.random.default_rng(0),
        coupled=True,
    )
    return profile.mean(0), signal.mean(0)


def test_zero_dose_and_padding_have_no_odor():
    first = materials()[0]
    absent = [replace(materials()[1], ingredient_id=f"zero-{i}") for i in range(64)]
    a = DoseModel([first], {}, 15, draws=64).evaluate([1.0])
    b = DoseModel([first, *absent], {}, 15, draws=64).evaluate([1.0, *([0.0] * 64)])
    np.testing.assert_allclose(a.temporal, b.temporal, atol=0, rtol=1e-14)
    np.testing.assert_allclose(a.signal, b.signal, atol=0, rtol=1e-14)
    z = DoseModel([first, *absent], {}, 15, draws=64).evaluate(np.zeros(65))
    assert not z.temporal.any() and not z.signal.any()


@pytest.mark.parametrize("concentration", [1.0, 15.0, 100.0])
@pytest.mark.parametrize("carrier", ["ethanol", "water", "dpg", "triethyl citrate"])
def test_carrier_mass_and_forward_search_share_exact_physics(concentration, carrier):
    items = materials()
    items[0] = replace(items[0], active_strength_percent=1.0, carrier=carrier)
    x = np.array([0.25, 0.35, 0.4])
    state = DoseModel(items, {}, concentration, draws=64).evaluate(x)
    profile, signal = forward(items, x, concentration)
    np.testing.assert_allclose(state.temporal, profile, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(state.signal, signal, atol=2e-14, rtol=2e-14)
    model = DoseModel(items, {}, concentration, draws=64)
    derivative = np.stack(
        [
            (
                model.evaluate(x + d * 1e-6).temporal
                - model.evaluate(x - d * 1e-6).temporal
            )
            / 2e-6
            for d in np.eye(3)
        ],
        -1,
    )
    np.testing.assert_allclose(state.temporal_jac, derivative, atol=1e-8, rtol=1e-5)


def test_neat_product_dilution_does_not_cancel_itself():
    a = materials()[0]
    neat = DoseModel([a], {}, 100, draws=64).evaluate([1.0])
    dilute = DoseModel(
        [replace(a, active_strength_percent=1.0, carrier="dpg")], {}, 100, draws=64
    ).evaluate([1.0])
    assert np.all(dilute.signal < neat.signal)
    assert carrier_moles_per_gram(
        [replace(a, active_strength_percent=1.0, carrier="dpg")]
    )[0] == pytest.approx(0.99 / 134.174)


def test_duplicate_ids_rejected_before_partition_and_saturation():
    a = materials()[0]
    with pytest.raises(ValueError, match="unique"):
        DoseModel([a, a], {}, 15)
    with pytest.raises(ValueError, match="unique"):
        forward([a, a], [0.5, 0.5], 15)


def test_unknown_carrier_and_negative_activity_are_not_silent():
    with pytest.raises(ValueError, match="carrier"):
        carrier_moles_per_gram(
            [
                replace(
                    materials()[0],
                    active_strength_percent=10.0,
                    carrier="mystery solvent",
                )
            ]
        )
    for x in ([-1.0], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            hill_response(x)


def test_physical_index_rejects_hash_unit_and_temperature_drift(tmp_path):
    entry = {
        "value": 12.0,
        "unit": "Pa",
        "source_ref": "test-source",
        "observations": [{"value": 12.0}],
        "reference_temperature_k": 298.15,
    }
    payload = {
        "schema": "physical-evidence-index/v2",
        "human_recipe_validation": False,
        "imputer_predictions_imported": False,
        "by_inchikey": {
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N": {
                "graph": "CCO",
                "molecular_weight": 46.069,
                "properties": {"vapor_pressure_pa_25c": entry},
            }
        },
    }
    path = tmp_path / "evidence.json"

    def read():
        path.write_text(json.dumps(payload), encoding="utf8")
        raw = path.read_bytes()
        stat = path.stat()
        return load(
            str(path), hashlib.sha256(raw).hexdigest(), stat.st_size, stat.st_mtime_ns
        )

    assert read()
    entry["reference_temperature_k"] = 303.15
    with pytest.raises(ValueError, match="298.15"):
        read()
    entry["reference_temperature_k"] = 298.15
    entry["unit"] = "mmHg"
    with pytest.raises(ValueError, match="unit"):
        read()
