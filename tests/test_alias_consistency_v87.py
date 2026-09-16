from copy import deepcopy

from fragrance_ai.recommender.alias_consistency import consistent_aliases


def test_same_general_spice_meaning_uses_same_reference_without_merging_subtypes():
    rows = {key: {"kind": "odor", "coarse_projection": {"spicy": 1.}, "reference_key": key}
            for key in ("spicy", "spices", "pepper", "ginger", "cardamom")}
    aliases = {"spicy": "spices", "spice": "spices", "spicyaroma": "spicy",
               "스파이시": "spicy", "향신료": "spicy", "후추": "pepper", "생강": "ginger", "카다멈": "cardamom"}
    original = deepcopy((aliases, rows))
    actual, report = consistent_aliases(aliases, rows)
    assert actual["스파이시"] == actual["spicy"] == "spices"
    assert actual["향신료"] == actual["spice"]
    assert all(actual[k] == aliases[k] for k in ("spicyaroma", "후추", "생강", "카다멈"))
    assert (aliases, rows) == original
    assert len(report["changes"]) == 2 and not report["recipe_scores_used"]
    assert not report["reference_vectors_modified"]


def test_missing_or_incompatible_reference_is_not_silently_substituted():
    aliases = {"spicy": "spices", "스파이시": "spicy"}
    rows = {"spicy": {"kind": "odor", "coarse_projection": {"spicy": 1.}, "reference_key": "spicy"},
            "spices": {"kind": "odor", "coarse_projection": {"fruity": 1.}, "reference_key": "spices"}}
    assert consistent_aliases(aliases, rows)[0] == aliases
    assert consistent_aliases({"스파이시": "spicy"}, rows)[0] == {"스파이시": "spicy"}
