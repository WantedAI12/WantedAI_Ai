from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fragrance_ai.platform.ai_extensions import register_ai_extensions
from fragrance_ai.platform.lotion_inputs import LotionSimulationRequest
from fragrance_ai.platform.lotion_reference import lotion_reference
from fragrance_ai.recommender.catalog import IngredientCatalog
from tests.test_ai_extensions import Formula
from tests.test_lotion import data, rebind, run
from tests.test_lotion_v21 import fixture, optimize


def split_data(value=None):
    value = deepcopy(value or data())
    value["water_loss_per_min"] = .03
    value["phase_components"][1] = {
        "name": "test oil phase", "source_kind": "simulated", "source_reference": "synthetic split fixture only",
        "portions": [{"compartment": "lipid", "mass_percent": 60., "density_g_ml": .8},
                     {"compartment": "aqueous_nonvolatile", "mass_percent": 10., "density_g_ml": 1.2},
                     {"compartment": "water", "mass_percent": 30., "density_g_ml": 1.}]}
    return value


def test_reference_matches_saved_context_without_inventing_coefficients():
    reference = lotion_reference()
    saved = Path(__file__).resolve().parents[1] / "benchmarks/reference_lotion/lc_cct_ow_01/application_context.json"
    assert reference["application_context"] == json.loads(saved.read_text(encoding="utf-8"))
    assert sum(row["mass_percent"] for row in reference["application_context"]["base_components"]) == 100
    assert not reference["ready_to_simulate"] and not reference["ready_to_optimize"]
    assert reference["application_context"]["release_series"] is None
    assert not reference["measured_release_data_available"]
    reference["application_context"]["base_components"][0]["mass_percent"] = 1
    assert lotion_reference()["application_context"]["base_components"][0]["mass_percent"] == 84


def test_reference_api_is_rate_limited_and_never_runs_inference():
    app, rates = FastAPI(), []
    def unexpected(*args, **kwargs):
        pytest.fail("reference lookup must not run inference")
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), unexpected, lambda: rates.append(1))
    with TestClient(app) as client:
        detail = client.get("/v1/applications/body-lotion/references/LC-CCT-OW-01")
        assert detail.status_code == 200
        assert client.get("/v1/applications/body-lotion/references").json()["references"] == [detail.json()]
        assert client.get("/v1/applications/body-lotion/references/unknown").status_code == 404
        assert len(rates) == 3
        validation = client.post("/v1/applications/validate", json=detail.json()["application_context"])
        assert validation.json()["status"] == "needs_input"
        # A known reference is not a license to silently invent missing parameters.
        incomplete = data()
        incomplete["application_context"] = detail.json()["application_context"]
        assert client.post("/v1/applications/body-lotion/simulate", json=incomplete).status_code == 422


@pytest.mark.parametrize("mode", ["open_sink", "bidirectional_air"])
def test_split_blend_equals_explicitly_expanded_base(mode):
    value = split_data()
    value["transport_mode"] = mode
    actual = run(value)
    expanded = deepcopy(value)
    component = expanded["application_context"]["base_components"].pop()
    row = expanded["phase_components"].pop()
    for part in row["portions"]:
        name = "expanded " + part["compartment"]
        role = {"lipid": "oil", "water": "water", "aqueous_nonvolatile": "humectant"}[part["compartment"]]
        expanded["application_context"]["base_components"].append({"name": name,
            "mass_percent": component["mass_percent"] * part["mass_percent"] / 100, "role": role})
        expanded["phase_components"].append({"name": name, "density_g_ml": part["density_g_ml"],
            "phase": "lipid" if role == "oil" else "aqueous"})
    rebind(expanded)
    expected = run(expanded)
    for a, b in zip(actual["temporal_profile"], expected["temporal_profile"]):
        assert a["aqueous_volume_ml_cm2"] == pytest.approx(b["aqueous_volume_ml_cm2"], rel=1e-12)
        assert a["lipid_volume_ml_cm2"] == pytest.approx(b["lipid_volume_ml_cm2"], rel=1e-12)
        for key in ("remaining_mg_cm2", "headspace_mg_cm2", "skin_sink_mg_cm2", "ventilated_mg_cm2"):
            assert a["materials"][0][key] == pytest.approx(b["materials"][0][key], rel=1e-12, abs=1e-14)
    assert actual["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12
    assert actual["base_phase_assignments"][1]["source_kind"] == "simulated"
    assert not actual["base_phase_assignment_source_verified"]


@pytest.mark.parametrize("case", ["total", "duplicate", "ambiguous", "missing_source", "missing_kind", "boolean", "nan", "water"])
def test_bad_split_data_rejected(case):
    value = split_data()
    row = value["phase_components"][1]
    if case == "total": row["portions"][0]["mass_percent"] = 50
    if case == "duplicate": row["portions"][1]["compartment"] = "lipid"
    if case == "ambiguous": row.update(phase="lipid", density_g_ml=1.)
    if case == "missing_source": row["source_reference"] = " "
    if case == "missing_kind": row.pop("source_kind")
    if case == "boolean": row["portions"][0]["mass_percent"] = True
    if case == "nan": row["portions"][0]["density_g_ml"] = float("nan")
    if case == "water":
        value["application_context"]["base_components"][1]["role"] = "water"
        rebind(value)
    with pytest.raises(ValueError):
        LotionSimulationRequest.model_validate(value)


def test_inverse_design_accepts_split_components_and_keeps_mass_balance():
    value, catalog = fixture()
    value["simulation"] = split_data(value["simulation"])
    result = optimize(value, catalog)
    assert np.isfinite(result["score"])
    assert result["simulation"]["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-12
    assert not result["simulation"]["base_phase_assignment_source_verified"]


def test_modal_app_exposes_reference_without_changing_old_paths():
    pytest.importorskip("modal")
    from deploy.modal_app import REGISTRY, create_web_app
    with TestClient(create_web_app(str(REGISTRY))) as client:
        response = client.get("/v1/applications/body-lotion/references/LC-CCT-OW-01")
        assert response.status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/formulas" in paths and "/v1/applications/body-lotion/optimize" in paths
        assert not response.json()["ready_to_optimize"]


def test_prepared_v23_preserves_catalog_evidence_and_packages_source():
    import gzip
    import zipfile
    pytest.importorskip("modal")
    from deploy.modal_app import ROOT, RUNTIME_CATALOG, WHEEL, WHEEL_SHA256
    old = json.loads(gzip.decompress((ROOT / "dist/body-lotion-v22/runtime/runtime_catalog_v3.json.gz").read_bytes()))
    new = json.loads(gzip.decompress(RUNTIME_CATALOG.read_bytes()))
    old.pop("wheel_sha256")
    new.pop("wheel_sha256")
    if old == new:
        assert old == new
    else:
        # V30 intentionally repairs only fingerprinted identity-linked rows;
        # retain a strict no-unrelated-data-loss contract, not an unconditional
        # equality to a catalog now known to omit source records.
        from scripts.audit_odor_identity_repair import audit
        # Verify the original identity repair independently of the later
        # versioned projection extension. Neither audit may waive source,
        # identity, price, availability or policy preservation.
        repaired_base = json.loads(gzip.decompress((ROOT / 'dist/body-lotion-v30/identity-repair/runtime_catalog_v3.json.gz').read_bytes()))
        repaired = audit(old, repaired_base)
        assert repaired['changed_materials'] == 195
        assert repaired['new_active'] == 55 and repaired['newly_blocked'] == 13
        assert repaired['unchanged_materials'] == 29064
        if repaired_base['ingredients'] != new['ingredients']:
            from scripts.audit_odor_projection_catalog import audit as projection_audit
            projected = projection_audit(repaired_base, new)
            assert projected['new_active'] == 54 and projected['newly_blocked'] == 24
            assert projected['profile_changed'] == 90
            assert projected['assertions_refs_identity_price_availability_risk_caps_preserved']
    with zipfile.ZipFile(WHEEL) as wheel:
        # This module audits an immutable historical release, not whichever
        # newer source happens to be checked out. Validate the trusted archived
        # wheel and its recorded members; current source/wheel parity is checked
        # by the candidate preparation and release tests.
        import hashlib
        import ast
        assert hashlib.sha256(WHEEL.read_bytes()).hexdigest() == WHEEL_SHA256
        for name in ("fragrance_ai/platform/lotion_reference.py", "fragrance_ai/platform/lotion_inputs.py",
                     "fragrance_ai/platform/ai_extensions.py", "fragrance_ai/recommender/lotion.py"):
            ast.parse(wheel.read(name), filename=name)
            assert wheel.getinfo(name).file_size > 0
