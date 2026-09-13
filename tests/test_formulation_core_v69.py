import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fragrance_ai.research.formulation_network import FormulationNetwork, export_arrays  # noqa: E402 - optional Torch guard above
from fragrance_ai.recommender.formulation_core import (  # noqa: E402
    FormulationCore,
    VERSION,
    forward_arrays,
)
from fragrance_ai.recommender.formulation_process import (  # noqa: E402
    ACTIONS,
    TEMPLATES,
    process_history,
    process_context,
    procedure_target,
)
from fragrance_ai.recommender.formulation_views import shared_views  # noqa: E402


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared_formulation")
    torch.manual_seed(19)
    net = FormulationNetwork(1599, 450, 292, len(ACTIONS)).eval()
    arrays = export_arrays(net)
    arrays.update(
        feature_mean=np.zeros(1599, np.float32),
        feature_scale=np.ones(1599, np.float32),
        quantitative_scale=np.ones(292, np.float32),
        transport_mean=np.zeros(8, np.float32),
        transport_scale=np.ones(8, np.float32),
    )
    np.savez_compressed(root / "weights.npz", **arrays)
    manifest = {
        "schema": VERSION,
        "training_executed": True,
        "single_shared_checkpoint": True,
        "accepted_for_local_inference": True,
        "architecture": {"molecule_features": 1599},
        "weights": {
            "path": "weights.npz",
            "sha256": hashlib.sha256((root / "weights.npz").read_bytes()).hexdigest(),
        },
        "actions": list(ACTIONS),
        "quantitative_endpoints": [f"q{i}" for i in range(146)],
        "fine_endpoints": [f"f{i}" for i in range(450)],
        "fine_features": {
            "vocabulary": [f"x{i}" for i in range(536)],
            "by_structure": {},
        },
        "native_profiles": {},
        "source_annotations": {},
        "structures": {},
        "parameter_count": sum(p.numel() for p in net.parameters()),
        "training_sources": {"kind": "random_test_fixture_not_a_trained_model"},
        "evaluation": {
            "scope": "synthetic_unit_fixture",
            "gates": dict.fromkeys(
                (
                    "quantitative_fidelity",
                    "fine_fidelity",
                    "transport_fidelity",
                    "procedure_fidelity",
                    "mixture_proxy_fidelity",
                    "revision_deficit",
                    "cpu_export",
                    "measured_emulsion_vs_mean",
                ),
                True,
            ),
        },
        "process_context_schema": "source_relative_context/v2",
        "teacher_bindings": {"unified_product": {"sha256": "0" * 64}},
    }
    (root / "model.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((root / "model.json").read_bytes()).hexdigest()
    return root / "model.json", digest, net, arrays


def inputs():
    rng = np.random.default_rng(44)
    x = rng.normal(0, 0.1, (3, 5, 1599)).astype(np.float32)
    mass = rng.uniform(0.1, 1, (3, 5)).astype(np.float32)
    mass[:, -1] = 0.0
    context = rng.normal(0, 0.1, (3, 64)).astype(np.float32)
    ids = np.array([[1, 2, 0], [3, 4, 5], [6, 0, 0]], np.int64)
    values = rng.normal(0, 0.1, (3, 3, 12)).astype(np.float32)
    return x, mass, context, ids, values


def test_independent_numpy_torch_export(fixture):
    _, _, net, arrays = fixture
    data = inputs()
    with torch.no_grad():
        expected = net(*(torch.from_numpy(v) for v in data))
    actual = forward_arrays(arrays, *data)
    assert set(actual) == {
        "fine",
        "quantitative",
        "transport",
        "action",
        "check",
        "revision",
        "emulsion",
    }
    for name in actual:
        np.testing.assert_allclose(
            actual[name], expected[name].numpy(), atol=2e-6, rtol=1e-5
        )


def test_ingredient_order_and_zero_mass_invariance(fixture):
    model = FormulationCore(*fixture[:2])
    x, w, c, ids, v = inputs()
    baseline = model.forward(x, w, c, ids, v)
    order = [3, 1, 4, 0, 2]
    reordered = model.forward(x[:, order], w[:, order], c, ids, v)
    x[:, -1] = 1000
    padded_changed = model.forward(x, w, c, ids, v)
    for key in baseline:
        np.testing.assert_allclose(baseline[key], reordered[key], atol=2e-6)
        np.testing.assert_allclose(baseline[key], padded_changed[key], atol=2e-6)


def test_duplicate_component_coalescing(fixture):
    model = FormulationCore(*fixture[:2])
    x, w, c, ids, v = inputs()
    baseline = model.forward(x, w, c, ids, v)
    xx = np.concatenate([x, x[:, :1]], axis=1)
    ww = np.concatenate([w.copy(), w[:, :1] / 2], axis=1)
    ww[:, 0] /= 2
    split = model.forward(xx, ww, c, ids, v)
    for key in baseline:
        np.testing.assert_allclose(baseline[key], split[key], atol=2e-6)


def test_process_order_matters_but_padding_does_not(fixture):
    model = FormulationCore(*fixture[:2])
    x, w, c, ids, v = inputs()
    baseline = model.forward(x, w, c, ids, v)
    reversed_history = model.forward(x, w, c, ids[:, ::-1], v[:, ::-1])
    assert not np.allclose(baseline["action"], reversed_history["action"])
    padded_ids = np.pad(ids, ((0, 0), (0, 2)))
    padded_values = np.pad(v, ((0, 0), (0, 2), (0, 0)), constant_values=50)
    padded = model.forward(x, w, c, padded_ids, padded_values)
    for key in baseline:
        np.testing.assert_allclose(baseline[key], padded[key], atol=2e-6)


@pytest.mark.parametrize(
    "invalid", ["negative_mass", "nan", "float_step", "unknown_step", "wrong_context"]
)
def test_invalid_network_inputs_fail(fixture, invalid):
    model = FormulationCore(*fixture[:2])
    x, w, c, ids, v = inputs()
    if invalid == "negative_mass":
        w[0, 0] = -1
    if invalid == "nan":
        x[0, 0, 0] = np.nan
    if invalid == "float_step":
        ids = ids.astype(float)
    if invalid == "unknown_step":
        ids[0, 0] = 999
    if invalid == "wrong_context":
        c = c[:, :20]
    with pytest.raises(ValueError):
        model.forward(x, w, c, ids, v)


def test_one_weight_owner_without_legacy_model_load(fixture, monkeypatch):
    from fragrance_ai.recommender import (
        local_runtime,
        fine_odor_model,
        unified_transport,
    )

    path, digest = fixture[:2]
    monkeypatch.setattr(
        local_runtime,
        "local_profile",
        lambda: {"formulation_core": (str(path), digest)},
    )

    def fail(*args, **kwargs):
        raise AssertionError("old model loader called")

    monkeypatch.setattr(local_runtime, "_atlas", fail)
    monkeypatch.setattr(fine_odor_model, "_load", fail)
    monkeypatch.setattr(unified_transport, "_load", fail)
    q = local_runtime.local_odor_backbone_provider()
    f = fine_odor_model.configured_fine_odor()
    t = unified_transport.configured_unified_transport()
    assert q.core is f.core is t.core
    assert len({q.sha256, f.sha256, t.sha256}) == 1


def test_physics_nonnegative_conservative_and_structural_zeros(fixture):
    model = FormulationCore(*fixture[:2])
    raw = np.array(
        [
            [1, 0.2, 0, 0.1, 0.5, 0.1, 0.7, 0.1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [3, 1, 2, 0.5, 1, 0, 0.2, 0.2],
        ]
    )
    result = model.kernel(raw)
    assert np.min(result) >= 0
    np.testing.assert_allclose(result.sum(-1), 1, atol=2e-12)
    np.testing.assert_array_equal(result[1], [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    assert (result[0, :, 4] == 0).all()


def test_manifest_hash_rejected(fixture):
    with pytest.raises(ValueError, match="hash"):
        FormulationCore(fixture[0], "0" * 64)


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_full_source_sequence_and_wrong_order(template):
    steps = TEMPLATES[template]
    for i in range(len(steps) + 1):
        target, checks = procedure_target(template, steps[:i], {})
        assert ACTIONS[target] == steps[min(i, len(steps) - 1)]
        assert not checks[0]
    target, checks = procedure_target(template, [steps[1], steps[0]], {})
    assert ACTIONS[target] == "hold" and checks[0]


def test_component_ph_not_universal_rule():
    _, unchecked = procedure_target("cold_lotion", [], {"measured_ph": 2})
    _, checked = procedure_target(
        "cold_lotion", [], {"measured_ph": 2, "pe9010_present": True}
    )
    assert not unchecked[3] and checked[3]


@pytest.mark.parametrize(
    "values",
    [
        {"measured_ph": float("nan")},
        {"method_code": 3},
        {"peak_temperature_c": True},
        {"bogus": 1},
    ],
)
def test_process_context_rejects_unsupported_fields(values):
    with pytest.raises(ValueError):
        process_context("cold_lotion", values)


def test_empty_history_and_unknown_action():
    ids, numbers = process_history([])
    assert ids.shape == (0,) and numbers.shape == (0, 12)
    with pytest.raises(ValueError):
        process_history(["invented_stage"])


def test_shared_recipe_products_own_same_core(fixture, monkeypatch):
    from fragrance_ai.recommender import local_runtime, perception_runtime

    monkeypatch.setattr(
        local_runtime,
        "local_profile",
        lambda: {"formulation_core": (str(fixture[0]), fixture[1])},
    )

    def fail(*a, **k):
        raise AssertionError("legacy perception loader called")

    monkeypatch.setattr(perception_runtime, "_load", fail)
    # A real endpoint vocabulary is required by the reference-shape adapters.
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION

    core = configured_formulation_core()
    endpoints = [name for names in ATLAS_PROJECTION.values() for name in names]
    original = core.quantitative_endpoints
    try:
        core.quantitative_endpoints = tuple(
            endpoints + [f"other{i}" for i in range(146 - len(endpoints))]
        )
        if hasattr(core, "_views"):
            del core._views
        perfume = perception_runtime.configured_perception("perfume")
        lotion = perception_runtime.configured_perception("body_lotion")
        assert perfume is not lotion and perfume.core is lotion.core is core
        assert (
            perfume.runtime_product == "perfume"
            and lotion.runtime_product == "body_lotion"
        )
    finally:
        core.quantitative_endpoints = original
        if hasattr(core, "_views"):
            del core._views


@pytest.mark.parametrize("mass", [[], [0.0], [float("nan")], [-1.0]])
def test_mixture_requires_positive_mass_before_feature_calculation(fixture, mass):
    model = FormulationCore(*fixture[:2])
    with pytest.raises(ValueError, match="positive-mass"):
        model.mixture(["CCO"], mass)


def test_source_relative_features_are_scoped():
    hot = process_context(
        "hot_lotion",
        {"peak_temperature_c": 75, "measured_ph": 12, "pe9010_present": True},
    )
    cold = process_context("cold_lotion", {"peak_temperature_c": 75, "measured_ph": 12})
    assert hot[25] == 1 and hot[26] == 1 and hot[31] == 1 and hot[32] == 1
    assert cold[25] == cold[26] == cold[31] == cold[32] == 0


def test_emulsion_unit_conversion_and_domain():
    from fragrance_ai.recommender.emulsion_science import (
        features,
        validate_raw,
        REFERENCE,
    )

    raw = np.array([[500, 0.000488, 750, 0.00104, 997, 6.07, 0.01]])
    z = features(raw)[0]
    n = 500 / 60
    d = REFERENCE["impeller_diameter_m"]
    assert np.exp(z[0]) == pytest.approx(997 * n * d * d / 0.00104)
    assert np.exp(z[1]) == pytest.approx(997 * n * n * d**3 / 0.00607)
    assert np.exp(z[5]) == pytest.approx(np.sqrt(np.exp(z[1])) / np.exp(z[0]))
    for changed in (np.zeros((1, 7)), np.full((1, 7), np.nan), np.ones((1, 7))):
        with pytest.raises(ValueError):
            validate_raw(changed)


def test_emulsion_request_requires_reference_domain():
    from fragrance_ai.platform.emulsion_inputs import EmulsionRequest

    values = {
        "speed_rpm": 500,
        "dispersed_viscosity_pa_s": 0.000488,
        "dispersed_density_kg_m3": 750,
        "continuous_viscosity_pa_s": 0.00104,
        "continuous_density_kg_m3": 997,
        "interfacial_tension_mn_m": 6.07,
        "dispersed_volume_fraction": 0.01,
    }
    with pytest.raises(ValueError):
        EmulsionRequest(conditions=[values])
    request = EmulsionRequest(
        reference_domain="rodgers_2025_stirred_tank_24h_silicone_emulsion",
        conditions=[values],
    )
    assert request.raw_rows() == [[500, 0.000488, 750, 0.00104, 997, 6.07, 0.01]]


def test_neural_to_optimizer_boundary_uses_double_precision(fixture, monkeypatch):
    model = FormulationCore(*fixture[:2])
    raw = np.full((2, 146), 0.3, np.float32)
    monkeypatch.setattr(
        model, "molecular", lambda *a, **k: {"applicability": raw, "use": raw}
    )
    q = shared_views(model)[0].predict(["CCO", "CCC"])
    assert q["applicability"].dtype == np.float64 and q["use"].dtype == np.float64
    np.testing.assert_array_equal(q["applicability"], raw)
    normalized = q["applicability"] / q["applicability"].sum(-1, keepdims=True)
    np.testing.assert_allclose(normalized.sum(-1), 1, atol=1e-15)
