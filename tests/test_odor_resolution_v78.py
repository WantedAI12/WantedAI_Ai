"""Typed source estimates and a non-bypassable missing-reference deployment hold."""

from copy import deepcopy
import hashlib
import json

import pytest

from fragrance_ai.recommender.odor_resolution import (
    validate_resolution,
    reference_evidence,
    deployment_readiness,
)
from scripts.prepare_external_odor_evidence_v78 import labels
from scripts.resolve_odor_space_v78 import synsets


def fixture():
    source = {"citrus": [[0.7, 0.3], [0.6, 0.4]]}
    metadata = {
        "reference_kind": "source_annotation_conditioned_model_inference",
        "measured_profile": False,
        "reference_is_model_estimate": True,
        "distinct_identity_groups": 3,
        "source_model_sha256": "core",
        "candidate_catalog_used": False,
        "recipe_outcomes_used": False,
    }
    return {
        "parent_atlas_sha256": "core",
        "profiles": {**source, "inferred": [[0.4, 0.6], [0.3, 0.7]]},
        "concept_metadata": {"citrus": {}, "inferred": metadata},
        "resolution_extension": {
            "schema": "odor-source-resolution/v78",
            "original_missing_count": 3595,
            "every_original_entry_audited": True,
            "model_inferred_references_are_not_measured": True,
            "candidate_catalog_used": False,
            "recipe_outcomes_used": False,
            "source_profile_keys": ["citrus"],
            "source_profiles_sha256": hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest(),
            "new_reference_metadata": {"inferred": deepcopy(metadata)},
        },
    }


def test_estimates_stay_estimates():
    value = fixture()
    assert validate_resolution(value)
    report = reference_evidence(value["concept_metadata"]["inferred"])
    assert report["model_estimated"] and not report["measured_reference"]
    assert report["shared_predictor_self_consistency_only"]


@pytest.mark.parametrize(
    "change", ["observed", "parent", "candidate", "count", "old_vector", "unlisted"]
)
def test_false_evidence_or_changed_parent_rejected(change):
    value = fixture()
    metadata = value["concept_metadata"]["inferred"]
    if change == "observed":
        metadata["measured_profile"] = True
    elif change == "parent":
        metadata["source_model_sha256"] = "other"
    elif change == "candidate":
        metadata["candidate_catalog_used"] = True
    elif change == "count":
        metadata["distinct_identity_groups"] = 2
    elif change == "old_vector":
        value["profiles"]["citrus"][0] = [0.2, 0.8]
    else:
        value["profiles"]["extra"] = [[0.5, 0.5], [0.5, 0.5]]
    value["resolution_extension"]["new_reference_metadata"]["inferred"] = deepcopy(
        metadata
    )
    with pytest.raises(ValueError):
        validate_resolution(value)


def test_deployment_does_not_pass_by_deleting_missing_rows():
    required = [str(i) for i in range(134)]
    space = {
        "resolution_extension": {"required_reference_ids_before_deployment": required},
        "reference_bindings": {k: "ref" for k in required[:-1]},
    }
    status = deployment_readiness(space, {"ref": []})
    assert not status["ready"] and status["missing_reference_ids"] == ["133"]
    space["resolution_extension"]["required_reference_ids_before_deployment"] = (
        required[:-1]
    )
    assert not deployment_readiness(space, {"ref": []})["ready"]


def test_all_required_bindings_are_checked():
    required = [str(i) for i in range(134)]
    space = {
        "resolution_extension": {"required_reference_ids_before_deployment": required},
        "reference_bindings": {k: "ref" for k in required},
    }
    assert deployment_readiness(space, {"ref": []})["ready"]
    assert not deployment_readiness(space, {})["ready"]


def test_opaque_language_ids_cannot_be_cross_joined():
    assert synsets({"synset": "s12345"}) == ()
    assert synsets({"synset": "rose.n.01"}) == ("rose.n.01",)
    assert synsets({"synset": "['rose.n.01']"}) == ("rose.n.01",)


def test_public_binary_annotations_are_not_intensity_labels():
    assert labels({"Stimulus": "1", "Descriptors": "rose;woody"}) == ["rose", "woody"]
    assert labels({"Stimulus": "1", "rose": "1", "woody": "0"}) == ["rose"]
    assert labels({"Stimulus": "1", "descriptors": "['rose', 'woody']"}) == [
        "rose",
        "woody",
    ]
