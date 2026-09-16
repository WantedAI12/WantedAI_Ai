"""Evidence-type validation for additive V78 references and vocabulary roles."""

import hashlib
import json

VERSION = "odor-source-resolution/v78"
REFERENCE_KINDS = frozenset(
    {
        "source_annotation_conditioned_model_inference",
        "source_taxonomy_child_composition",
        "named_molecule_observed_profile",
        "named_molecule_model_inference",
        "limited_source_model_reference",
        "named_structure_alternatives_model_reference",
        "explicit_lexical_reference_composition",
        "limited_observed_endpoint_reference",
        "published_native_attribute_projection",
    }
)


def validate_resolution(value):
    extension = value.get("resolution_extension")
    if extension is None:
        return None
    if (
        extension.get("schema") != VERSION
        or extension.get("original_missing_count") != 3595
        or extension.get("every_original_entry_audited") is not True
        or extension.get("model_inferred_references_are_not_measured") is not True
        or extension.get("candidate_catalog_used") is not False
        or extension.get("recipe_outcomes_used") is not False
    ):
        raise ValueError("invalid odor resolution scope")
    protected = {k: value["profiles"][k] for k in extension["source_profile_keys"]}
    digest = hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest()
    if digest != extension["source_profiles_sha256"]:
        raise ValueError("existing observed reference vectors changed")
    if extension.get("validation_revision") == 2:
        metadata = {
            k: value["concept_metadata"][k] for k in extension["source_profile_keys"]
        }
        if hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest() != extension.get("source_metadata_sha256"):
            raise ValueError("existing source evidence metadata changed")
    additions = extension["new_reference_metadata"]
    if set(value["profiles"]) != set(protected) | set(additions):
        raise ValueError("unclassified quantitative reference")
    for key, metadata in additions.items():
        if (
            value["concept_metadata"].get(key) != metadata
            or metadata.get("reference_kind") not in REFERENCE_KINDS
        ):
            raise ValueError("reference evidence metadata mismatch")
        if (
            metadata.get("candidate_catalog_used") is not False
            or metadata.get("recipe_outcomes_used") is not False
        ):
            raise ValueError("reference cannot be fitted to candidates")
        if metadata["reference_kind"] in (
            "source_annotation_conditioned_model_inference",
            "named_molecule_model_inference",
        ):
            if (
                metadata.get("measured_profile") is not False
                or metadata.get("reference_is_model_estimate") is not True
                or metadata.get("distinct_identity_groups", 0)
                < (
                    1
                    if metadata["reference_kind"] == "named_molecule_model_inference"
                    else 3
                )
                or metadata.get("source_model_sha256") != value["parent_atlas_sha256"]
            ):
                raise ValueError(
                    "model reference falsely labelled observed or wrong parent"
                )
        if metadata['reference_kind'] in ('limited_source_model_reference',
                                         'named_structure_alternatives_model_reference'):
            graphs = metadata.get('source_graphs_used', [])
            if (metadata.get('measured_profile') is not False or
                    metadata.get('reference_is_model_estimate') is not True or
                    metadata.get('source_model_sha256') != value['parent_atlas_sha256'] or
                    not graphs or len(set(graphs)) != len(graphs) or
                    metadata.get('distinct_identity_groups') != len(graphs) or
                    metadata.get('population_validated') is not False or
                    metadata.get('human_error_interval') is not None):
                raise ValueError('invalid limited-support reference provenance')
            if len(graphs) == 1 and metadata.get('between_identity_dispersion') is not None:
                raise ValueError('one exemplar cannot estimate between-identity dispersion')
            if metadata['reference_kind'] == 'named_structure_alternatives_model_reference':
                if (len(graphs) < 2 or metadata.get('same_connectivity') is not True or
                        metadata.get('identity_assumption') != 'equal_source_alternatives_not_physical_mixture'):
                    raise ValueError('unresolved structural alternatives cannot be silently merged')
        if metadata['reference_kind'] == 'limited_observed_endpoint_reference':
            import math
            weights = metadata.get('positive_observation_weights', [])
            effective = metadata.get('effective_source_observations', 0)
            if (metadata.get('measured_profile') is not True or metadata.get('reference_is_model_estimate') is not False
                    or not weights or any(not math.isfinite(w) or w<=0 for w in weights)
                    or abs(sum(weights)-1)>1e-8 or not math.isfinite(effective) or effective<1-1e-8
                    or len(metadata.get('source_observation_ids', [])) != len(weights)
                    or metadata.get('positive_source_observations') != len(weights)
                    or metadata.get('population_validated') is not False):
                raise ValueError('invalid limited observed reference provenance')
        if metadata['reference_kind'] == 'published_native_attribute_projection':
            if (metadata.get('measured_profile') is not False or metadata.get('reference_is_model_estimate') is not True
                    or metadata.get('project_authored_projection_not_measured_full_profile') is not True
                    or not metadata.get('source', {}).get('sha256')
                    or not metadata.get('explicit_attribute_mapping')
                    or metadata.get('complete_source_attribute_coverage') != (not metadata.get('unmapped_source_attributes'))):
                raise ValueError('invalid published native attribute projection')
    return extension


def reference_evidence(metadata):
    kind = metadata.get("reference_kind", "observed_conditional_profile")
    estimated = bool(metadata.get("reference_is_model_estimate"))
    return {
        "kind": kind,
        "model_estimated": estimated,
        "measured_reference": not estimated and metadata.get("measured_profile", True),
        "independent_human_recipe_validation": False,
        "shared_predictor_self_consistency_only": estimated,
        "distinct_source_identities": metadata.get("distinct_identity_groups"),
        "support_policy": metadata.get('support_policy'),
        "population_validated": metadata.get('population_validated', False),
        "identity_assumption": metadata.get('identity_assumption'),
        "identity_alternatives": metadata.get('identity_alternatives', []),
        "human_error_interval": None,
        "limited_support": bool(metadata.get('limited_support')) or metadata.get('reference_kind') == 'limited_source_model_reference',
        "reference_context": metadata.get('reference_context'),
        "unmapped_source_attributes": metadata.get('unmapped_source_attributes', []),
        "complete_source_attribute_coverage": metadata.get('complete_source_attribute_coverage', True),
    }


def deployment_readiness(space, profiles):
    """The user requires ALL 134 formerly missing targets before deployment."""
    extension = space.get("resolution_extension") or {}
    required = extension.get("required_reference_ids_before_deployment")
    if (
        not isinstance(required, list)
        or len(required) != 134
        or len(set(required)) != 134
    ):
        return {
            "ready": False,
            "blockers": ["frozen_134_reference_requirement_missing"],
            "missing_reference_ids": [],
        }
    bindings = space.get("reference_bindings", {})
    missing = [
        k for k in required if not bindings.get(k) or bindings[k] not in profiles
    ]
    return {
        "ready": not missing,
        "required_reference_count": 134,
        "missing_reference_ids": missing,
        "blockers": ["all_required_quantitative_references_not_available"]
        if missing
        else [],
        "requirement_not_satisfied_by_relabelling_or_dropping_terms": True,
    }
