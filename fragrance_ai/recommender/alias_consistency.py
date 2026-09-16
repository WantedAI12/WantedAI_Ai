"""Explicit cross-language equivalences, independent of recipes or scores.

Keep the narrow spicy-aroma source identity available, but route the general
Korean spice descriptors through the same identity as their English wording.
Pepper, ginger and cardamom are not interchangeable with the general family.
"""

VERSION = "bilingual-odor-alias-consistency/v87"
EQUIVALENTS = {"스파이시": "spicy", "향신료": "spice"}


def consistent_aliases(aliases, rows):
    resolved = dict(aliases)
    changes = []
    for localized, canonical in EQUIVALENTS.items():
        previous, current = aliases.get(localized), aliases.get(canonical)
        if previous is None or current is None or previous == current:
            continue
        before, after = rows[previous], rows[current]
        if (before.get("kind") != "odor" or after.get("kind") != "odor"
                or before.get("coarse_projection") != after.get("coarse_projection")
                or not after.get("reference_key")):
            continue
        resolved[localized] = current
        changes.append({"alias": localized, "canonical_wording": canonical,
                        "previous_concept": previous, "resolved_concept": current,
                        "reason": "explicit_translation_equivalence_not_score_selection"})
    return resolved, {"version": VERSION, "changes": changes,
                      "reference_vectors_modified": False, "recipe_scores_used": False}
