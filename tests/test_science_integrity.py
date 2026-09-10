"""Regression tests for fail-closed scientific-model contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_positive_unlabeled_loss_accepts_only_source_backed_positive_assertions():
    torch = pytest.importorskip("torch")
    from fragrance_ai.research.r2_physsim import positive_unlabeled_descriptor_loss

    logits = torch.tensor([[0.2, -0.4], [0.7, 0.1], [-0.2, 0.6]])
    # The zero entries are absent from a public descriptor catalogue.  They
    # are deliberately passed as unlabeled, not as BCE target negatives.
    positive_assertions = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    class_prior = torch.tensor([1.0 / 3.0, 1.0 / 3.0])
    loss = positive_unlabeled_descriptor_loss(
        logits, positive_assertions, class_prior
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    with pytest.raises(ValueError, match=r"equal \[N, L\] shape"):
        positive_unlabeled_descriptor_loss(logits, positive_assertions[:2], class_prior)


def test_external_source_disjointness_audit_detects_molecule_and_scaffold_overlap():
    pytest.importorskip("rdkit")
    from fragrance_ai.research.r2_physsim import audit_external_source_disjointness

    audit = audit_external_source_disjointness(
        ["c1ccccc1O"],
        {
            "descriptor_pretraining": ["c1ccccc1O"],
            "normalizer_fit": ["c1ccccc1Cl"],
        },
    )
    assert audit["passed"] is False
    pretraining = audit["populations"]["descriptor_pretraining"]
    assert pretraining["molecule_overlap_count"] == 1
    assert pretraining["molecule_disjoint"] is False
    normalizer = audit["populations"]["normalizer_fit"]
    assert normalizer["scaffold_overlap_count"] == 1
    assert normalizer["scaffold_disjoint"] is False


def test_legacy_checkpoint_is_loaded_but_scientific_weight_fails_closed():
    pytest.importorskip("torch")
    from fragrance_ai.recommender.physsim_checkpoint import FrozenR2PhysSim

    adapter = FrozenR2PhysSim()
    adapter._load()  # Deliberately verify the packaged legacy manifest path.
    assert adapter._loaded is True
    assert adapter._validation_contract_passed is False
    assert adapter._direct_formulation_capability_authorized is False
    assert adapter._approved_weight == 0.0


def test_release_trainer_rejects_pre_contract_strict_reports():
    from scripts.train_physsim_r2_transfer_release import _strict_summary

    legacy_report = json.loads(
        (PROJECT_ROOT / "benchmarks" / "physsim_r2_transfer_final_strict.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(RuntimeError, match="all-components-held-out"):
        _strict_summary(legacy_report, "molecule_disjoint")
