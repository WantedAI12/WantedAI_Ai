from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.semantic_ontology import ScentSemanticOntology


def _parse(text: str):
    return NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(text)


def test_bundled_semantic_encoder_is_offline_and_versioned():
    result = ScentSemanticOntology(model_path=None).infer("햇빛에 말린 흰 셔츠")
    assert result.backend == "bundled_signed_hash_ngram"
    assert result.ontology_version == "scent-ontology-2.0.0"
    assert "clean" in result.concepts


def test_metaphorical_laundry_brief_maps_to_clean_fresh():
    brief = _parse("햇빛에 말린 흰 셔츠처럼 보송하고 투명한 향")
    assert {"clean", "fresh"}.issubset(brief.desired_dimensions)
    assert brief.semantic_confidence > 0


def test_wet_forest_metaphor_maps_without_gourmand():
    brief = _parse("비 온 뒤 젖은 숲바닥의 흙과 이끼처럼, 단맛은 없이")
    assert {"green", "earthy", "aquatic"}.issubset(brief.desired_dimensions)
    assert "gourmand" in brief.avoided_dimensions
    assert "gourmand" not in brief.desired_dimensions


def test_english_metaphor_maps_to_smoke_and_wood():
    brief = _parse("the last smoke of a dying campfire over charred timber")
    assert {"smoky", "woody"}.issubset(brief.desired_dimensions)


def test_cold_transparent_sea_breeze_maps_to_aquatic_fresh():
    brief = _parse("차가운 바닷바람과 젖은 돌, 아주 투명하고 가볍게")
    assert {"aquatic", "fresh"}.issubset(brief.desired_dimensions)
