"""Independent optimality/PSD/serialization checks for our molecular learner."""

import json

import numpy as np
import pytest

from fragrance_ai.research.structured_kernel import fit_structured_kernel
from fragrance_ai.research.atlas_profiles import (
    atlas_features,
    atlas_kernel,
    fit_atlas_structured,
    predict_atlas,
)


def arrays():
    rng = np.random.default_rng(166)
    x = rng.normal(size=(12, 7))
    k = x @ x.T / 7
    y = rng.normal(size=(12, 5))
    return k, y


def test_exact_intercept_matches_independent_augmented_linear_system():
    k, y = arrays()
    alpha = 0.3
    actual = fit_structured_kernel(k, y, alpha=alpha)
    full = np.block(
        [
            [k + alpha * np.eye(len(k)), np.ones((len(k), 1))],
            [np.ones((1, len(k))), np.zeros((1, 1))],
        ]
    )
    expected = np.linalg.solve(full, np.vstack([y, np.zeros((1, y.shape[1]))]))
    np.testing.assert_allclose(actual["coefficients"], expected[:-1], atol=2e-13)
    np.testing.assert_allclose(actual["intercept"], expected[-1], atol=2e-13)
    assert actual["diagnostics"]["normal_equation_relative_residual"] < 1e-12


def test_output_coupling_matches_independent_kronecker_solve():
    k, y = arrays()
    alpha, rho = 0.3, 0.7
    n, d = y.shape
    h = np.eye(n) - np.ones((n, n)) / n
    kc, yc = h @ k @ h, h @ y
    cov = yc.T @ yc / (n - 1)
    b = (1 - rho) * np.eye(d) + rho * d * cov / np.trace(cov)
    # Small independent reference only. Production never builds this n*d matrix.
    expected = np.linalg.solve(
        np.kron(np.eye(d), kc) + alpha * np.kron(np.linalg.inv(b), np.eye(n)),
        yc.reshape(-1, order="F"),
    ).reshape((n, d), order="F")
    actual = fit_structured_kernel(k, y, alpha=alpha, output_coupling=rho)
    np.testing.assert_allclose(actual["coefficients"], expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual["coefficients"].sum(axis=0), 0.0, atol=1e-13)


def test_training_is_permutation_equivariant_and_intercept_shift_equivariant():
    k, y = arrays()
    permutation = np.arange(len(k))[::-1]
    a = fit_structured_kernel(k, y, alpha=0.1, output_coupling=0.5)
    b = fit_structured_kernel(
        k[np.ix_(permutation, permutation)],
        y[permutation] + 3,
        alpha=0.1,
        output_coupling=0.5,
    )
    np.testing.assert_allclose(
        a["coefficients"], b["coefficients"][permutation], atol=1e-12
    )
    np.testing.assert_allclose(a["intercept"] + 3, b["intercept"], atol=1e-12)


@pytest.mark.parametrize(
    "defect",
    [
        "asymmetric",
        "indefinite",
        "negative_constant",
        "nan",
        "negative_alpha",
        "singular_output_metric",
    ],
)
def test_invalid_operator_cannot_silently_be_fitted(defect):
    k, y = arrays()
    kwargs = dict(alpha=0.1)
    if defect == "asymmetric":
        k[0, 1] += 1
    if defect == "indefinite":
        k = -np.eye(len(k))
    if defect == "negative_constant":
        k = -np.ones_like(k)
    if defect == "nan":
        y[0, 0] = np.nan
    if defect == "negative_alpha":
        kwargs["alpha"] = -1
    if defect == "singular_output_metric":
        kwargs["output_coupling"] = 1
    with pytest.raises(ValueError):
        fit_structured_kernel(k, y, **kwargs)


def test_constant_outputs_and_rank_deficient_kernel_are_well_defined():
    k, y = np.ones((5, 5)), np.tile([2.0, 0.0, 3.0], (5, 1))
    value = fit_structured_kernel(k, y, alpha=1e-4, output_coupling=0.9)
    np.testing.assert_allclose(value["coefficients"], 0.0, atol=1e-12)
    np.testing.assert_allclose(value["intercept"], y[0], atol=1e-12)


def features():
    graphs = ["CCO", "CCCO", "CCCCO", "CC(C)O", "CC=O", "CC(=O)C"]
    fine = {"vocabulary": ["fruit", "wood"], "by_structure": {"CCO": [0], "CCCO": [1]}}
    return atlas_features(graphs, ["high"] * len(graphs), {}, fine)


def test_unit_diagonal_psd_without_turning_missing_terms_into_odor_observations():
    x = features()
    scale = np.ones(36)
    k = atlas_kernel(x, x, scale, 0.5, normalize=True)
    np.testing.assert_allclose(np.diag(k), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(k).min() > -1e-12
    assert np.all(x[2:, 1060:-3] == 0)


@pytest.mark.parametrize("transform", ["identity", "sqrt", "log1p"])
def test_new_backbone_round_trip_keeps_all_outputs_and_legacy_features(transform):
    x = features()
    y = np.arange(len(x) * 4, dtype=float).reshape(len(x), 4) / 10
    model = fit_atlas_structured(
        x, y, 0.1, 0.5, target_transform=transform, output_coupling=0.5
    )
    restored = json.loads(json.dumps(model, allow_nan=False))
    expected = predict_atlas(model, x)
    np.testing.assert_array_equal(predict_atlas(restored, x), expected)
    assert (
        expected.shape == y.shape
        and np.isfinite(expected).all()
        and (expected >= 0).all()
    )
    restored["kernel_normalization"] = "none"
    with pytest.raises(ValueError, match="normalization"):
        predict_atlas(restored, x)


@pytest.mark.parametrize("kind", ["legacy", "structured"])
def test_compiled_heads_match_reference_and_share_kernel(monkeypatch, kind):
    from fragrance_ai.research import atlas_inference
    from fragrance_ai.research.atlas_profiles import fit_atlas

    x = features()
    y = np.arange(len(x) * 4, dtype=float).reshape(len(x), 4) / 10
    fit = fit_atlas if kind == "legacy" else fit_atlas_structured
    models = {
        head: fit(x, y + offset, 0.1, 0.5) for head, offset in (("a", 0), ("b", 0.2))
    }
    compiled = atlas_inference.CompiledAtlasHeads(models)
    query = np.tile(x, (90, 1))  # crosses the bounded-memory batch boundary
    calls = []
    original = atlas_inference.atlas_kernel

    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(atlas_inference, "atlas_kernel", counted)
    actual = compiled.predict(query)
    assert calls == [256, 256, 28]  # one, not two, kernels per batch
    for head, model in models.items():
        np.testing.assert_allclose(
            actual[head], predict_atlas(model, query), rtol=1e-12, atol=1e-12
        )
    models["a"]["weights"][0][0] = 10000  # caller mutation cannot corrupt compilation
    np.testing.assert_array_equal(compiled.predict(query)["a"], actual["a"])
    with pytest.raises(TypeError):
        compiled.models["a"]["fine_weight"] = 0
    with pytest.raises(ValueError):
        compiled.models["a"]["weights"][0, 0] = 0
    with pytest.raises(ValueError):
        compiled.predict(np.full_like(x, np.nan))


def test_compiled_different_kernels_and_transforms_do_not_share_wrong_features():
    from fragrance_ai.research.atlas_inference import CompiledAtlasHeads

    x = features()
    y = np.arange(len(x) * 4, dtype=float).reshape(len(x), 4) / 10
    models = {
        t: fit_atlas_structured(x, y, 0.1, w, target_transform=t, output_coupling=0.5)
        for t, w in (("sqrt", 0.5), ("log1p", 0.25))
    }
    compiled = CompiledAtlasHeads(models)
    assert len(compiled.groups) == 2
    actual = compiled.predict(x)
    for head, model in models.items():
        np.testing.assert_allclose(actual[head], predict_atlas(model, x), atol=1e-12)


@pytest.fixture
def runtime_models(tmp_path, monkeypatch):
    import hashlib
    from fragrance_ai.research.atlas_profiles import fit_atlas
    from fragrance_ai.research.fine_odor_features import SCHEMA
    from fragrance_ai.recommender import local_runtime
    from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION

    x = features()
    endpoints = sorted(
        {name for values in ATLAS_PROJECTION.values() for name in values}
    )
    y = (
        np.arange(len(x) * len(endpoints), dtype=float).reshape(len(x), len(endpoints))
        / 10
    )
    fine = {
        "schema": SCHEMA,
        "vocabulary": ["fruit", "wood"],
        "by_structure": {"CCO": [0], "CCCO": [1]},
        "not_observed_does_not_mean_sensory_absence": True,
    }
    fine["content_sha256"] = hashlib.sha256(
        json.dumps(fine, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    base = {
        "schema": "atlas-profile-candidate/v1",
        "runtime_promotion_allowed": False,
        "data_redistribution_authorized": False,
        "endpoints": endpoints,
        "fine_features": fine,
        "native_profiles": {},
        "source": {"fixture": True},
    }
    parent = {
        **base,
        "models": {
            head: fit_atlas(x, y, 0.1, 0.5) for head in ("applicability", "use")
        },
    }

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    parent_binding = write("parent.json", parent)
    candidate = {
        **base,
        "parent_checkpoint_sha256": parent_binding["sha256"],
        "quantitative_target_model_version": "structured-multioutput-atlas/v66",
        "development_nonregression_passed": True,
        "models": {
            head: fit_atlas_structured(x, y, 0.1, 0.5)
            for head in ("applicability", "use")
        },
    }
    value = {
        "schema": "perfumery-local-runtime/v1",
        "scope": "local_research",
        "atlas": parent_binding,
        "odor_backbone": write("candidate.json", candidate),
    }
    for name in ("catalog", "perfume", "body_lotion", "stock_mixture"):
        value[name] = {"path": name + ".json", "sha256": "a" * 64}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv(local_runtime.PROFILE_ENV, str(path))
    monkeypatch.delenv("PERFUMERY_AI_ENV", raising=False)
    return value, candidate, path, write


def test_explicit_runtime_upgrade_keeps_old_stock_parent(runtime_models):
    from fragrance_ai.recommender import local_runtime

    value, _, _, _ = runtime_models
    parent = local_runtime.local_atlas_provider()
    candidate = local_runtime.local_odor_backbone_provider()
    assert parent.sha256 == value["atlas"]["sha256"]
    assert candidate.sha256 == value["odor_backbone"]["sha256"] != parent.sha256
    actual = candidate.predict(["CCO"])
    assert set(actual) == {"applicability", "use"} and actual["use"].shape == (
        1,
        len(candidate.endpoints),
    )
    bridge = local_runtime.local_lotion_provider(
        type("Component", (), {"structures": {}})()
    )
    assert bridge.component_model_sha256 == candidate.sha256


@pytest.mark.parametrize(
    "defect", ["parent", "source", "endpoints", "acceptance", "hash"]
)
def test_odor_only_upgrade_fails_closed_without_its_declared_lineage(
    runtime_models, defect
):
    from fragrance_ai.recommender import local_runtime

    value, candidate, path, write = runtime_models
    if defect == "parent":
        candidate["parent_checkpoint_sha256"] = "b" * 64
    if defect == "source":
        candidate["source"] = {"different": True}
    if defect == "endpoints":
        candidate["endpoints"] = ["wrong"] + candidate["endpoints"][1:]
    if defect == "acceptance":
        candidate["development_nonregression_passed"] = False
    value["odor_backbone"] = write("candidate.json", candidate)
    if defect == "hash":
        value["odor_backbone"]["sha256"] = "f" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        local_runtime.local_odor_backbone_provider()
