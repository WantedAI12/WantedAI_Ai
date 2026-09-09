import pytest

from fragrance_ai.recommender.brief_parser import (
    NaturalLanguageBriefParser,
    UnsupportedOdorDescriptorError,
)
from fragrance_ai.recommender.catalog import IngredientCatalog, normalize_name
from fragrance_ai.recommender.odor_descriptors import (
    load_builtin_odor_descriptor_lexicon,
)


def _parser() -> NaturalLanguageBriefParser:
    return NaturalLanguageBriefParser(IngredientCatalog.load_builtin())


def test_descriptor_projection_asset_is_validated_and_fail_closed():
    lexicon = load_builtin_odor_descriptor_lexicon()
    supported = [item for item in lexicon.descriptors if item.formula_supported]
    unsupported = [item for item in lexicon.descriptors if not item.formula_supported]

    assert lexicon.version == "odor-descriptor-projection-1.1.0"
    assert len(lexicon.descriptors) == 51
    assert len(supported) == 34
    assert {item.descriptor for item in unsupported} == {
        "alliaceous",
        "ammoniacal",
        "cheesy",
        "creosote",
        "fermented",
        "fishy",
        "hydrocarbon",
        "meaty",
        "odorless",
        "oily",
        "onion",
        "pyridinic",
        "rancid",
        "savory",
        "sour_acrid",
        "sulfurous",
        "sweaty",
    }


def test_latin_boundaries_prevent_pear_and_peach_from_matching_pea_material():
    brief = _parser().parse("a pear and peach fragrance")

    assert brief.desired_dimensions == ["fruity"]
    assert "Phenethyl Alcohol" not in brief.requested_ingredients
    assert "rose" not in brief.desired_dimensions


def test_dairy_no_longer_matches_airy_substring():
    brief = _parser().parse("a dairy fragrance")

    assert {"gourmand", "musky"}.issubset(brief.desired_dimensions)
    assert "fresh" not in brief.desired_dimensions
    assert brief.recognized_descriptors == ["dairy"]
    assert brief.semantic_confidence == pytest.approx(0.78)


def test_supported_fine_descriptors_create_transparent_coarse_profile():
    brief = _parser().parse("minty metallic pineapple with a honey nuance")

    assert {"fresh", "aromatic", "green", "clean", "aquatic"}.issubset(
        brief.desired_dimensions
    )
    assert {"fruity", "gourmand", "floral", "amber"}.issubset(brief.desired_dimensions)
    assert brief.recognized_descriptors == [
        "honey",
        "metallic",
        "minty",
        "pineapple",
    ]
    assert brief.descriptor_projection_version == "odor-descriptor-projection-1.1.0"
    assert "not measured sensory equivalence" in (
        brief.descriptor_projection_claim_boundary
    )


def test_positive_unrepresentable_descriptors_fail_closed():
    with pytest.raises(UnsupportedOdorDescriptorError) as captured:
        _parser().parse("a sulfurous onion fragrance")

    assert captured.value.descriptors == ("onion", "sulfurous")


def test_english_word_ending_in_no_cannot_fake_descriptor_negation():
    with pytest.raises(UnsupportedOdorDescriptorError) as captured:
        _parser().parse("a fresh amino sulfurous fragrance")

    assert captured.value.descriptors == ("sulfurous",)


def test_negated_unrepresentable_descriptors_are_recognized_without_generation_block():
    brief = _parser().parse("fresh aldehydic fragrance without sulfurous or onion")

    assert {"fresh", "clean"}.issubset(brief.desired_dimensions)
    assert set(brief.avoided_descriptors) == {"onion", "sulfurous"}
    assert {"aldehydic", "onion", "sulfurous"}.issubset(brief.recognized_descriptors)


def test_any_explicit_negative_material_mention_dominates_positive_reference():
    brief = _parser().parse("Lilial 같은 맑은 플로럴이지만 Lilial은 빼고")

    assert "Lilial" in brief.excluded_ingredients
    assert "Lilial" not in brief.requested_ingredients
    assert normalize_name("Lilial") in brief.constraints.explicit_bans


def test_exact_short_material_alias_still_matches():
    brief = _parser().parse("PEA rose fragrance")

    assert "Phenethyl Alcohol" in brief.requested_ingredients
    assert {"rose", "floral"}.issubset(brief.desired_dimensions)


def test_exact_lexical_evidence_controls_confidence_and_blocks_hash_noise():
    brief = _parser().parse("a fresh fragrance")

    assert brief.desired_dimensions == ["fresh"]
    assert brief.semantic_confidence == pytest.approx(0.96)
    assert brief.semantic_backend.startswith("hybrid_lexical+")


@pytest.mark.parametrize(
    ("text", "descriptor", "dimensions"),
    [
        ("a caraway fragrance", "caraway", {"aromatic", "spicy", "green"}),
        ("a banana fragrance", "banana", {"fruity", "gourmand"}),
        ("a coconut fragrance", "coconut", {"fruity", "gourmand"}),
        ("a greenfruity fragrance", "green_fruity", {"fruity", "green"}),
        ("a dusty rooty fragrance", "dusty", {"powdery", "earthy"}),
        ("a malty fragrance", "malty", {"gourmand", "earthy"}),
    ],
)
def test_development_vocabulary_projects_only_to_declared_axes(
    text, descriptor, dimensions
):
    brief = _parser().parse(text)

    assert descriptor in brief.recognized_descriptors
    assert dimensions.issubset(brief.desired_dimensions)


@pytest.mark.parametrize(
    ("text", "descriptor"),
    [
        ("a sweaty fragrance", "sweaty"),
        ("an ammonia fragrance", "ammoniacal"),
        ("a fishy fragrance", "fishy"),
        ("a rancid fragrance", "rancid"),
        ("a sour fragrance", "sour_acrid"),
        ("a gasoline fragrance", "hydrocarbon"),
        ("an odorless fragrance", "odorless"),
        ("a pyridine fragrance", "pyridinic"),
        ("a creosote fragrance", "creosote"),
    ],
)
def test_development_vocabulary_without_a_safe_axis_fails_closed(text, descriptor):
    with pytest.raises(UnsupportedOdorDescriptorError) as captured:
        _parser().parse(text)

    assert captured.value.descriptors == (descriptor,)
