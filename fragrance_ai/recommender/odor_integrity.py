"""Do not turn molecular names, appearance, taste or hashes into odor data."""

from __future__ import annotations

from functools import lru_cache
import copy
import hashlib
import json
import math

from .models import SCENT_DIMENSIONS, normalize_profile
from .odor_descriptors import load_builtin_odor_descriptor_lexicon


REGISTRY_SOURCE_PREFIX = "industrial-registry-public-descriptor-conditional-"
REGISTRY_ODOR_SOURCE = REGISTRY_SOURCE_PREFIX + "v3"
ODOR_INTEGRITY_VERSION = "explicit-odor-assertions-1"
LEGACY_ODOR_PROJECTION = "explicit-odor-projection-1"
EXPANDED_ODOR_PROJECTION = "explicit-odor-projection-2"
CONCEPT_ODOR_PROJECTION = "explicit-odor-projection-3"
# Exact odor-field terms only. These are coarse semantic categories, NOT
# measured intensity, invented observations, molecular-name inference or a
# target-dependent embedding. Existing detailed projections take precedence.
COARSE_ODOR_ALIASES = {
    'citrus': ('lemon', 'orange', 'grapefruit', 'bergamot', 'lime', 'mandarin', 'citrusy', 'yuzu'),
    'fruity': ('apple', 'pear', 'berry', 'berries', 'peach', 'apricot', 'plum', 'cherry',
               'grape', 'strawberry', 'raspberry', 'blackcurrant', 'pineapple'),
    'floral': ('flower', 'flowers', 'flowery', 'bouquet'),
    'rose': ('rosy', 'roses'),
    'whitefloral': ('gardenia', 'muguet', 'lily', 'tuberose', 'orangeflower'),
    'green': ('grass', 'grassy', 'leaf', 'leaves', 'foliage', 'galbanum'),
    'herbal': ('herb', 'herbs', 'lavender', 'rosemary', 'basil', 'fennel'),
    'woody': ('wood', 'woods', 'cedar', 'sandalwood'),
    'spicy': ('spice', 'spices', 'pepper', 'peppery', 'cinnamon', 'clove', 'cloves'),
    'musky': ('musklike',),
    'clean': ('soapy',),
    'gourmand': ('caramel', 'chocolate'),
    'smoky': ('smoke',),
    'earthy': ('earth', 'soil'),
}


def resolved_projection_version(version):
    if version in ('', LEGACY_ODOR_PROJECTION):
        return LEGACY_ODOR_PROJECTION
    if version in (EXPANDED_ODOR_PROJECTION, CONCEPT_ODOR_PROJECTION):
        return version
    raise ValueError('unsupported odor projection version')
POSITIVE_ODOR_STATUS = "supported_public_odor_descriptors"
# The old registry mixed FlavorDB's Odor and Flavor Percepts columns. Until
# source-column lineage is rebuilt, its positive terms are not odor evidence.
ODOR_SOURCES = frozenset({"leffingwell", "goodscents", "flavornet", "aromadb", "ifra_2019", "flavordb_odor"})
NEGATIVE_TERMS = frozenset({"odorless", "odourless", "noodor", "noodour", "nosmell", "odorfree", "odourfree", "withoutodor", "withoutodour"})
AMBIGUOUS_TERMS = frozenset({"powder", "crystal", "crystals", "crystalline", "white", "colorless", "colourless", "solid", "liquid", "granular", "bland", "tasteless", "characteristic"})
DIRECT = {
    **{name.replace("_", ""): {name: 1.} for name in SCENT_DIMENSIONS},
    "aldehydic": {"clean": .65, "fresh": .35}, "balsam": {"amber": .65, "woody": .35},
    "fruit": {"fruity": 1.}, "herbaceous": {"aromatic": .65, "green": .35},
    "herbal": {"aromatic": .65, "green": .35}, "jasmin": {"white_floral": .8, "floral": .2},
    "jasmine": {"white_floral": .8, "floral": .2}, "leather": {"leathery": .8, "smoky": .2},
    "marine": {"aquatic": .8, "fresh": .2}, "musk": {"musky": 1.},
    "oceanic": {"aquatic": .8, "fresh": .2}, "ozonic": {"aquatic": .55, "fresh": .45},
    "sweet": {"gourmand": .85, "amber": .15}, "tobacco": {"leathery": .45, "smoky": .30, "woody": .25},
    "vanilla": {"gourmand": .75, "powdery": .25},
}


def normalize_term(value):
    return "".join(c for c in value.casefold() if c.isalnum())


@lru_cache(maxsize=1)
def projection_maps():
    supported, unsupported = {}, set()
    for row in load_builtin_odor_descriptor_lexicon().descriptors:
        for alias in row.aliases:
            key = normalize_term(alias)
            if row.formula_supported:
                supported[key] = (row.projection_confidence, row.profile)
            else:
                unsupported.add(key)
    for key, profile in DIRECT.items():
        supported.setdefault(key, (.75, profile))
    for key in NEGATIVE_TERMS | AMBIGUOUS_TERMS:
        supported.pop(key, None)
    return supported, unsupported


@lru_cache(maxsize=1)
def canonical_projection_terms():
    """Deduplicate declared model concepts, never merely equal coarse vectors."""
    terms = {}
    for row in load_builtin_odor_descriptor_lexicon().descriptors:
        for alias in row.aliases:
            terms[normalize_term(alias)] = normalize_term(row.descriptor)
    for alias, canonical in {
        'fruit': 'fruity', 'herbaceous': 'herbal', 'jasmin': 'jasmine',
        'oceanic': 'marine', 'musk': 'musky',
    }.items():
        terms.setdefault(alias, canonical)
    return terms


def odor_assertion_representation(assertions):
    """Explain source-bound support, not measured intensity or sensory absence.

    Unknown terms remain visible. Source count never multiplies a concept's
    contribution. Different detailed concepts and their raw evidence survive.
    """
    assertions = tuple(assertions)
    supported, unsupported = projection_maps()
    canonical = canonical_projection_terms()
    aliases = {alias: key for key, words in COARSE_ODOR_ALIASES.items() for alias in words}
    groups, unknown, ignored = {}, set(), set()
    for assertion in assertions:
        if not isinstance(assertion, str) or assertion.count(':') != 1:
            raise ValueError('invalid odor assertion')
        source, raw = assertion.split(':', 1)
        term = normalize_term(raw)
        if source not in ODOR_SOURCES:
            ignored.add(assertion)
            continue
        projected = term if term in supported or term in unsupported else aliases.get(term, term)
        if projected not in supported:
            unknown.add(assertion)
            continue
        concept = canonical.get(projected, projected)
        row = groups.setdefault(concept, {'terms': set(), 'sources': set()})
        row['terms'].add(raw)
        row['sources'].add(source)
    status, items = assess_odor_assertions(assertions, CONCEPT_ODOR_PROJECTION)
    return {
        'version': CONCEPT_ODOR_PROJECTION, 'status': status,
        'concepts': {key: {'terms': sorted(row['terms']), 'sources': sorted(row['sources']),
                          'projection_confidence': supported[key][0]}
                     for key, row in sorted(groups.items())},
        'unprojected_assertions': sorted(unknown), 'ignored_nonodor_assertions': sorted(ignored),
        'coarse_profile': dict(items), 'intensity_measured': False,
        'conditions_status': 'not_resolved_by_descriptor_projection',
        'claim_boundary': 'semantic_descriptor_support_not_measured_intensity_or_sensory_absence',
    }


@lru_cache(maxsize=32768)
def assess_odor_assertions(assertions: tuple[str, ...], projection_version=LEGACY_ODOR_PROJECTION):
    """Return a status and immutable profile from explicit source-tagged terms."""
    supported, unsupported = projection_maps()
    version = resolved_projection_version(projection_version)
    expanded = version in (EXPANDED_ODOR_PROJECTION, CONCEPT_ODOR_PROJECTION)
    canonical = canonical_projection_terms() if version == CONCEPT_ODOR_PROJECTION else {}
    aliases = {alias: canonical for canonical, values in COARSE_ODOR_ALIASES.items() for alias in values}
    positive, negative, unmodeled = set(), set(), set()
    for assertion in assertions:
        if not isinstance(assertion, str) or assertion.count(":") != 1:
            return "invalid_odor_assertions", ()
        source, raw = assertion.split(":", 1)
        term = normalize_term(raw)
        if term in NEGATIVE_TERMS or (expanded and term == 'noaroma'):
            negative.add(term)
        if source not in ODOR_SOURCES:
            continue
        if term in unsupported and term not in NEGATIVE_TERMS:
            unmodeled.add(term)
        if expanded and term not in supported and term not in unsupported:
            term = aliases.get(term, term)
        if term in supported:
            positive.add(canonical.get(term, term))
    if negative:
        return ("conflicting_odor_reports" if positive else "reported_odorless"), ()
    if unmodeled:
        return "unmodeled_odor_descriptors", ()
    if not positive:
        return "no_positive_odor_evidence", ()
    values = dict.fromkeys(SCENT_DIMENSIONS, 0.)
    for term in sorted(positive):
        confidence, profile = supported[term]
        for dimension, value in profile.items():
            values[dimension] += confidence * value
    return POSITIVE_ODOR_STATUS, tuple(normalize_profile(values).items())


def is_registry_material(ingredient):
    # Verified promotions have a different, verifier-enforced `industrial_`
    # identity. A source label must never exempt an unverified `registry_` ID.
    return ingredient.ingredient_id.startswith("registry_") or (
        isinstance(ingredient.data_source, str) and ingredient.data_source.startswith(REGISTRY_SOURCE_PREFIX)
    )


@lru_cache(maxsize=32768)
def _valid_lineage(assertions, refs, registry_sha256):
    if not isinstance(registry_sha256, str) or len(registry_sha256) != 64 or any(c not in "0123456789abcdef" for c in registry_sha256):
        return False
    covered = set()
    try:
        for encoded in refs:
            ref = json.loads(encoded)
            digest = ref["source_file_sha256"]
            source, source_field = ref["source_tag"], ref["source_field"]
            allowed_fields = {
                "flavordb_odor": {"Odor Percepts"}, "goodscents": {"Descriptors"},
                "flavornet": {"Descriptors"}, "aromadb": {"Filtered Descriptors", "Raw Descriptors"},
                "ifra_2019": {"Descriptor 1", "Descriptor 2", "Descriptor 3"},
                "leffingwell": {ref["descriptor"]},
            }
            if (ref["semantic_role"] != "odor" or ref["normalization_version"] != ODOR_INTEGRITY_VERSION
                    or source not in allowed_fields or source_field not in allowed_fields[source]
                    or not isinstance(digest, str)
                    or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                    or not ref["source_record_id"] or not ref["source_field"] or not ref["conditions_status"]):
                continue
            covered.add(ref["source_tag"] + ":" + normalize_term(ref["descriptor"]))
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
    return all(value in covered for value in assertions if value.split(":", 1)[0] in ODOR_SOURCES)


def registry_odor_rejection(ingredient):
    """An experimental safety override cannot manufacture positive odor data."""
    if not is_registry_material(ingredient):
        return None
    observed = isinstance(ingredient.data_source, str) and ingredient.data_source.startswith("odor-observed:")
    if (ingredient.data_source != REGISTRY_ODOR_SOURCE and not observed) or ingredient.odor_integrity_version != ODOR_INTEGRITY_VERSION:
        return "legacy_registry_odor_profile_unverified"
    if ingredient.odor_evidence_status != POSITIVE_ODOR_STATUS:
        return "registry_" + (ingredient.odor_evidence_status or "missing_odor_evidence")
    if ingredient.registry_structural_alerts:
        return "registry_structural_review_required"
    if not ingredient.structure_smiles or ingredient.ingredient_id != "registry_" + hashlib.sha256(ingredient.structure_smiles.encode()).hexdigest()[:24]:
        return "registry_structure_identity_mismatch"
    values = ingredient.structure_properties
    expected_fields = {"molecular_weight", "xlogp", "tpsa", "hbond_donors", "hbond_acceptors", "rotatable_bonds"}
    if (not isinstance(values, dict) or set(values) != expected_fields
            or not isinstance(ingredient.structure_properties_version, str)
            or not ingredient.structure_properties_version.startswith("rdkit-")
            or not ingredient.structure_properties_version.endswith("-calculated-not-measured")
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values())
            or not 0 < values["molecular_weight"] <= 5000
            or not -20 <= values["xlogp"] <= 30 or values["tpsa"] < 0
            or any(values[key] < 0 or int(values[key]) != values[key] for key in ("hbond_donors", "hbond_acceptors", "rotatable_bonds"))):
        return "registry_calculated_structure_properties_invalid"
    if not all(isinstance(value, str) for value in (*ingredient.odor_assertions, *ingredient.odor_evidence_refs)):
        return "registry_odor_lineage_invalid"
    if not _valid_lineage(tuple(ingredient.odor_assertions), tuple(ingredient.odor_evidence_refs), ingredient.odor_registry_sha256):
        return "registry_odor_lineage_missing"
    try:
        status, expected_items = assess_odor_assertions(tuple(ingredient.odor_assertions), ingredient.odor_projection_version)
    except ValueError:
        return 'registry_unsupported_odor_projection_version'
    if status != POSITIVE_ODOR_STATUS:
        return "registry_" + status
    expected = dict(expected_items)
    if (not isinstance(ingredient.profile, dict)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in ingredient.profile.values())):
        return "registry_odor_profile_not_bound_to_assertions"
    if observed:
        return None if (set(ingredient.profile) <= set(SCENT_DIMENSIONS)
                        and all(math.isfinite(v) and 0 <= v <= 1 for v in ingredient.profile.values())
                        and math.isclose(sum(ingredient.profile.values()), 1., abs_tol=1e-9)) else "invalid_observed_odor_profile"
    if set(ingredient.profile) - set(SCENT_DIMENSIONS) or any(
        not math.isfinite(value := ingredient.profile.get(key, 0.)) or not math.isclose(value, expected[key], abs_tol=1e-9, rel_tol=0.)
        for key in SCENT_DIMENSIONS
    ):
        return "registry_odor_profile_not_bound_to_assertions"
    return None


def legacy_registry_line_ids(payload):
    """Detect the untraceable registry provenance in a historical formula."""
    if not isinstance(payload, dict):
        return []
    found = set()
    for field in ("recipe", "closest_candidate"):
        for line in payload.get(field, []) or []:
            if not isinstance(line, dict):
                continue
            source = str(line.get("data_source", ""))
            identifier = str(line.get("ingredient_id", ""))
            registry = identifier.startswith("registry_") or source.startswith(REGISTRY_SOURCE_PREFIX)
            current_observation = source.startswith("odor-observed:") and line.get("odor_integrity_version") == ODOR_INTEGRITY_VERSION
            current_registry = source == REGISTRY_ODOR_SOURCE and line.get("odor_integrity_version") == ODOR_INTEGRITY_VERSION
            if registry:
                try:
                    resolved_projection_version(line.get('odor_projection_version', ''))
                except ValueError:
                    current_registry = current_observation = False
            if registry and not current_registry and not current_observation:
                found.add(identifier)
    return sorted(found)


def quarantine_legacy_payload(payload):
    """A derived API view; never overwrite the immutable stored payload."""
    if not isinstance(payload, dict):
        return payload
    result = copy.deepcopy(payload)
    for key in ("result", "payload", "latest_version", "version", "formula", "formula_version"):
        if isinstance(result.get(key), dict):
            result[key] = quarantine_legacy_payload(result[key])
            if key == "payload" and result[key].get("data_integrity_status") == "legacy_registry_profile_quarantined" and "content_sha256" in result:
                result.update(payload_view_transformed=True, content_sha256_scope="immutable_stored_historical_payload")
    identifiers = legacy_registry_line_ids(payload)
    if not identifiers:
        return result
    result["historical_odor_snapshot"] = copy.deepcopy(payload)
    result.update(status="requires_odor_data_regeneration", recipe=[], closest_candidate=[],
                  calculated_profile_similarity=None, full_profile_target_met=False,
                  similarity_score=0., raw_similarity_score=0., simulation_only_approved=False,
                  human_similarity_90_claim_authorized=False,
                  data_integrity_status="legacy_registry_profile_quarantined",
                  invalid_odor_ingredient_ids=identifiers,
                  message="Legacy registry profiles were not bound to odor evidence. Regenerate from the original brief with the corrected catalog.",
                  full_profile_assessment={"status": "invalidated_odor_input_data", "score": None, "target_met": False})
    for key in ("actual_olfactory_similarity_score", "sensory_similarity_score", "simulated_similarity_score", "temporal_similarity_score"):
        if key in result:
            result[key] = None
    for key, value in list(result.items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and any(term in key for term in ("similarity", "p05", "p95", "human_discrimination")):
            result[key] = None
    result.update(similarity_score=0., raw_similarity_score=0., legacy_preference_score=None,
                  confidence="invalidated_odor_input_data", simulation_status="invalidated_odor_input_data",
                  temporal_profile=[], ingredient_temporal_profile=[], physsim_temporal_profile=[], achieved_profile={},
                  scientific_model_domain_passed=False, scientific_twin_status="invalidated_odor_input_data",
                  scientific_uncertainty_kind="invalidated_odor_input_data", physsim_comparison_authorized=False,
                  perceptual_prediction_status="invalidated_odor_input_data", sensory_validation_status="invalidated_odor_input_data",
                  olfactory_validation_status="invalidated_odor_input_data", release_scope_verified=False,
                  external_regulatory_signoff_valid=False, release_evidence_status="invalidated_odor_input_data")
    for key in list(result):
        if key.endswith(("coverage_percent", "applicability_percent", "applied_weight", "approved_weight")):
            result[key] = 0.
    contract = result.get("score_contract")
    result["score_contract"] = {**(contract if isinstance(contract, dict) else {}), "assessment_valid": False, "status": "invalidated_odor_input_data"}
    if isinstance(result.get("safety"), dict):
        result["safety"].update(internal_gate_passed=False, manufacturing_ready=False, regulatory_data_complete=False,
                                 status="odor_integrity_blocked")
    if isinstance(result.get("manufacturing_plan"), dict):
        result["manufacturing_plan"].update(ready_for_lab_trial=False, ready_for_manufacture=False)
    return result
