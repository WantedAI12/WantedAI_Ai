"""Independent algebra/geometry tests; these fixtures are not sensory evidence."""

from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.optimize import minimize

from fragrance_ai.research.atlas_profiles import fit_atlas_scientific, predict_atlas
from fragrance_ai.research.atlas_inference import CompiledAtlasHeads
from fragrance_ai.research.scientific_kernel import (
    MolecularGeometry,
    ScientificKernel,
    fit_alignment,
    fit_geometry,
    kernel_prior,
    project_simplex,
)
from tests.test_main_backbone_v66 import features, runtime_models as runtime_models


def sample():
    x = features()
    x[0, 1040:1043], x[0, 1059] = [1, 2, 3], 1
    x[1, 1040:1043], x[1, 1059] = [3, 2, 1], 1
    x[1, -2:] = [0, 1]
    y = np.arange(len(x) * 5, dtype=float).reshape(len(x), 5) / 10
    return x, y


def test_simplex_matches_independent_constrained_optimizer():
    vector = np.array([0.5, -2.0, 0.3, 4.0])
    expected = minimize(
        lambda w: 0.5 * np.sum((w - vector) ** 2),
        np.full(4, 0.25),
        jac=lambda w: w - vector,
        bounds=[(0.0, None)] * 4,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: w.sum() - 1,
                "jac": lambda w: np.ones_like(w),
            }
        ],
        method="SLSQP",
        options={"ftol": 1e-13},
    )
    assert expected.success
    np.testing.assert_allclose(project_simplex(vector), expected.x, atol=1e-12)


def test_alignment_stationary_point_matches_independent_slsqp():
    x, y = sample()
    geometry = MolecularGeometry.from_dict(fit_geometry(x))
    blocks = geometry.blocks(x, x)
    result = fit_alignment(blocks, y, regularization=0.3, prior=kernel_prior())
    h = np.eye(len(x)) - np.ones((len(x), len(x))) / len(x)
    centered = np.stack([h @ block @ h for block in blocks])
    normalized = centered / np.asarray(result["centered_norms"])[:, None, None]
    target = (h @ y) @ (h @ y).T
    target /= np.linalg.norm(target)
    prior = kernel_prior()

    def objective(weights):
        error = np.einsum("i,ijk->jk", weights, normalized) - target
        return 0.5 * np.sum(error**2) + 0.15 * np.sum((weights - prior) ** 2)

    expected = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(prior),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-13, "maxiter": 2000},
    )
    assert expected.success
    actual = (np.array(result["weights"]) - 0.01 * prior) / 0.99
    np.testing.assert_allclose(objective(actual), objective(expected.x), atol=1e-10)
    assert result["diagnostics"]["projected_kkt_residual"] < 2e-10


def test_masked_psd_blocks_and_unit_diagonal_do_not_invent_annotations():
    x, y = sample()
    geometry = MolecularGeometry.from_dict(fit_geometry(x))
    blocks = geometry.blocks(x, x)
    assert np.all(blocks[4, 2:, :] == 0)  # both unknown is NOT a native match
    assert np.all(blocks[5, 2:, :] == 0)
    for block in blocks:
        assert np.linalg.eigvalsh(block).min() > -1e-12
    model = fit_atlas_scientific(x, y, 0.1, 0.3)
    kernel = ScientificKernel(model["kernel_specification"])(x, x)
    assert np.linalg.eigvalsh(kernel).min() > -1e-12
    np.testing.assert_allclose(np.diag(kernel), 1, atol=1e-13)


def test_train_geometry_is_unit_change_invariant_and_query_does_not_refit():
    x, y = sample()
    scaled = x.copy()
    multipliers = np.linspace(0.02, 100.0, 16)
    offsets = np.linspace(-10.0, 20.0, 16)
    scaled[:, 1024:1040] = x[:, 1024:1040] * multipliers + offsets
    a = fit_atlas_scientific(x, y, 0.1, 0.3)
    b = fit_atlas_scientific(scaled, y, 0.1, 0.3)
    np.testing.assert_allclose(
        predict_atlas(a, x), predict_atlas(b, scaled), atol=1e-10
    )
    snapshot = json.dumps(a, sort_keys=True)
    query = x.copy()
    query[:, 1024:1040] += 1e8
    results, diagnostic = CompiledAtlasHeads({"a": a}).predict_with_diagnostics(query)
    assert np.isfinite(results["a"]).all()
    assert all(row["outside_training_radius"] for row in diagnostic["a"])
    assert json.dumps(a, sort_keys=True) == snapshot


@pytest.mark.parametrize("transform", ["identity", "sqrt", "log1p"])
def test_compiled_portable_roundtrip_batch_parity_and_immutable_geometry(transform):
    x, y = sample()
    model = fit_atlas_scientific(x, y, 0.1, 0.3, target_transform=transform)
    restored = json.loads(json.dumps(model, allow_nan=False))
    query = np.tile(x, (87, 1))
    compiled = CompiledAtlasHeads({"a": restored, "b": restored})
    assert len(compiled.groups) == 1
    actual, diagnostics = compiled.predict_with_diagnostics(query)
    expected = predict_atlas(model, query)
    np.testing.assert_allclose(actual["a"], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        predict_atlas(compiled.models["a"], query), expected, atol=1e-12
    )
    assert len(diagnostics["a"]) == len(query)
    assert diagnostics["a"][0]["maximum_training_similarity"] == pytest.approx(1)
    assert "confidence_percent" not in diagnostics["a"][0]
    with pytest.raises(TypeError):
        compiled.models["a"]["kernel_specification"]["geometry"]["physical_mean"][0] = (
            100
        )
    restored["kernel_specification"]["geometry"]["physical_mean"][0] = 100
    np.testing.assert_array_equal(compiled.predict(query)["a"], actual["a"])


@pytest.mark.parametrize(
    "defect",
    ["mask", "negative_native", "negative_kernel_weight", "singular_whitener", "nan"],
)
def test_invalid_geometry_cannot_silently_enter_runtime(defect):
    x, y = sample()
    model = fit_atlas_scientific(x, y, 0.1, 0.3)
    changed = deepcopy(model)
    geometry = changed["kernel_specification"]["geometry"]
    if defect == "mask":
        x[0, 1059] = 0
    elif defect == "negative_native":
        x[0, 1040] = -1
    elif defect == "negative_kernel_weight":
        changed["kernel_specification"]["alignment"]["weights"][0] = -1
    elif defect == "singular_whitener":
        geometry["physical_whitener"] = np.zeros((16, 16)).tolist()
    else:
        geometry["physical_mean"][0] = float("nan")
    with pytest.raises(ValueError):
        predict_atlas(changed, x)


def test_training_is_row_permutation_equivariant():
    x, y = sample()
    permutation = np.array([3, 1, 4, 0, 5, 2])
    a = fit_atlas_scientific(x, y, 0.1, 0.3)
    b = fit_atlas_scientific(x[permutation], y[permutation], 0.1, 0.3)
    np.testing.assert_allclose(predict_atlas(a, x), predict_atlas(b, x), atol=1e-10)


def test_distinct_learned_weights_share_geometric_feature_work(monkeypatch):
    x, y = sample()
    a = fit_atlas_scientific(x, y, 0.1, 0.03)
    b = fit_atlas_scientific(x, y, 0.1, 3.0)
    compiled = CompiledAtlasHeads({"a": a, "b": b})
    assert len(compiled.groups) == 2
    original, calls = MolecularGeometry.blocks, []

    def counted(self, left, right):
        calls.append(len(left))
        return original(self, left, right)

    monkeypatch.setattr(MolecularGeometry, "blocks", counted)
    output = compiled.predict(np.tile(x, (87, 1)))
    assert calls == [256, 256, 10]
    assert len(output["a"]) == 522


def test_local_api_modules_do_not_import_cloud_deployment_sdk():
    import subprocess
    import sys
    from pathlib import Path

    code = """import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'modal' or name.startswith('modal.') or name == 'deploy.modal_app':
        raise AssertionError('local API imported cloud deployment setup')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from deploy.web_app import create_web_app
from scripts.serve_product_runtime_v42 import create_app
assert callable(create_web_app) and callable(create_app)
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert result.returncode == 0, result.stderr


def test_v68_is_selected_in_runtime_without_rebinding_stock_training(runtime_models):
    from fragrance_ai.recommender import local_runtime

    profile, candidate, path, write = runtime_models
    old = candidate['models']['applicability']
    x = np.array(old['support'])
    y = predict_atlas(old, x)
    candidate['models'] = {head:fit_atlas_scientific(x, y, .1, .3) for head in ('applicability','use')}
    candidate['quantitative_target_model_version'] = 'scientific-molecular-atlas/v68'
    profile['odor_backbone'] = write('candidate-v68.json', candidate)
    path.write_text(json.dumps(profile), encoding='utf-8')
    selected = local_runtime.local_odor_backbone_provider()
    assert selected.sha256 == profile['odor_backbone']['sha256']
    assert local_runtime.local_atlas_provider().sha256 == profile['atlas']['sha256']
    outputs, diagnostics = selected.predict_with_diagnostics(['CCO'])
    assert all(value.shape == (1,len(candidate['endpoints'])) for value in outputs.values())
    assert all(rows[0]['basis'] == 'training_geometry_not_calibrated_predictive_uncertainty' for rows in diagnostics.values())


def test_v68_label_cannot_hide_an_old_unupgraded_head(runtime_models):
    from fragrance_ai.recommender import local_runtime

    profile, candidate, path, write = runtime_models
    candidate['quantitative_target_model_version'] = 'scientific-molecular-atlas/v68'
    profile['odor_backbone'] = write('wrong-v68.json', candidate)
    path.write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(ValueError, match='every measurement head'):
        local_runtime.local_odor_backbone_provider()
