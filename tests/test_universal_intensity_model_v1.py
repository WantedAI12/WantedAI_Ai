from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_universal_intensity_model_v1 import (
    CANDIDATES,
    PHYSICAL_DESCRIPTOR_NAMES,
    _balanced_folds,
    _candidate_contract,
)


def test_universal_intensity_candidates_are_unique_and_identity_free():
    candidates = _candidate_contract()
    names = [row["name"] for row in candidates]
    assert candidates == CANDIDATES
    assert len(names) == len(set(names))
    assert all("component" not in name and "ingredient" not in name for name in names)
    assert any(row["portable"] for row in candidates)


def test_universal_physical_descriptor_contract_has_transport_features():
    names = set(PHYSICAL_DESCRIPTOR_NAMES)
    assert {"MolWt", "MolLogP", "TPSA", "LabuteASA", "MolMR"} <= names
    assert not any(name.startswith("component::") for name in names)


def test_balanced_molecule_folds_keep_identical_molecules_together():
    values = ["A", "A", "B", "C", "C", "D", "E"]
    folds = _balanced_folds(values, folds=3, salt="unit-test")
    assert folds.shape == (len(values),)
    assert folds[0] == folds[1]
    assert folds[3] == folds[4]
    assert set(np.unique(folds)) == {0, 1, 2}


def test_published_universal_v1_remains_rejected_and_weight_zero():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "universal_intensity_model_v1.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["retrospective_external_gate"]["passed"] is False
    assert value["final_model"]["runtime_primary_score_weight"] == 0.0
    assert value["human_olfactory_90_percent_certified"] is False
