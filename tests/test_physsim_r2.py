from __future__ import annotations

import hashlib
import json
from importlib import resources

import numpy as np
import pytest


def test_r2_architecture_contract_and_soft_core_are_finite():
    torch = pytest.importorskip("torch")
    from fragrance_ai.research.r2_physsim import (
        EXPECTED_PARAMETER_COUNT,
        R2PhysSimCore,
    )

    model = R2PhysSimCore()
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT
    descriptors = torch.zeros((2, 3, 217), dtype=torch.float32)
    mask = torch.ones((2, 3), dtype=torch.float32)
    output = model(descriptors, mask, descriptors, mask)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_frozen_r2_artifacts_have_matching_hashes_and_descriptor_contracts():
    data = resources.files("fragrance_ai").joinpath("data")
    checkpoint = data.joinpath("physsim_r2_checkpoint.pt").read_bytes()
    manifest = json.loads(data.joinpath("physsim_r2_manifest.json").read_text(encoding="utf-8"))
    components = data.joinpath("r2_ingredient_components.npz").read_bytes()
    component_manifest = json.loads(
        data.joinpath("r2_ingredient_components_manifest.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256(checkpoint).hexdigest() == manifest["checkpoint_sha256"]
    assert hashlib.sha256(components).hexdigest() == component_manifest["artifact_sha256"]
    assert manifest["descriptor_contract_sha256"] == component_manifest["descriptor_contract_sha256"]
    assert component_manifest["covered_ingredient_count"] == 34
    assert component_manifest["descriptor_count"] == 217
    assert manifest["ensemble_calibration"]["method"] == "centered_residual_on_primary_score"
    assert 0.0 < manifest["ensemble_calibration"]["neutral_similarity_percent"] < 100.0
    assert manifest["schema_version"] == "2.0"
    assert manifest["release_gate"]["passed"] is True
    assert manifest["release_gate"]["approved_primary_score_weight"] == 0.10
    assert all(manifest["release_gate"]["checks"].values())
    assert manifest["transfer_learning"]["ontology_labels"] == 113
    assert manifest["transfer_learning"]["descriptor_molecules_after_filter"] == 5526

    ensemble = json.loads(
        data.joinpath("physsim_r2_ensemble_manifest.json").read_text(encoding="utf-8")
    )
    assert ensemble["release_gate"]["passed"] is True
    assert all(ensemble["release_gate"]["checks"].values())
    assert ensemble["descriptor_contract_sha256"] == manifest["descriptor_contract_sha256"]
    assert sum(member["weight"] for member in ensemble["members"]) == pytest.approx(1.0)
    assert [member["weight"] for member in ensemble["members"]] == pytest.approx(
        [0.3, 0.7]
    )
    assert ensemble["selection"]["final_labels_used_for_selection"] is False
    assert ensemble["selection"]["selected_seed_20260715_weight"] == pytest.approx(
        0.3
    )
    for member in ensemble["members"]:
        payload = data.joinpath(member["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == member["sha256"]
    assert ensemble["uncertainty"]["maximum_member_disagreement"] > 0
    assert ensemble["uncertainty"]["absolute_error_q95"] > 0


def test_natural_gc_ms_and_threshold_lineage_are_explicitly_non_lot_specific():
    data = resources.files("fragrance_ai").joinpath("data")
    natural = json.loads(
        data.joinpath("natural_material_compositions.json").read_text(encoding="utf-8")
    )
    thresholds = json.loads(
        data.joinpath("odor_threshold_registry.json").read_text(encoding="utf-8")
    )
    assert len(natural["materials"]) == 4
    assert "not a GC-MS measurement" in natural["claim_boundary"]
    assert all(material["source"]["url"].startswith("https://") for material in natural["materials"])
    assert thresholds["source"]["unit"] == "ppmv"
    assert thresholds["catalog_match_count"] == 3
    assert thresholds["natural_component_match_count"] == 9
    assert "excluded" in thresholds["claim_boundary"]


def test_component_descriptor_registry_is_finite_and_complete():
    data = resources.files("fragrance_ai").joinpath("data")
    with np.load(data.joinpath("r2_ingredient_components.npz"), allow_pickle=False) as registry:
        assert registry["descriptors"].shape == (72, 217)
        assert np.isfinite(registry["descriptors"]).all()
        assert len(set(registry["ingredient_ids"].tolist())) == 34
