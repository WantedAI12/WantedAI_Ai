import pytest

from fragrance_ai.recommender.brief_parser import _perceptual_intent
from fragrance_ai.recommender.lotion_evaluation import refinement_score_floors
from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest


def test_spicy_does_not_request_icy_cooling_and_phase_is_not_texture():
    assert _perceptual_intent("spicy scent", "medium")[3] == {}
    assert _perceptual_intent("icy scent", "medium")[3] == {"cooling": 1.}
    assert "dry" not in _perceptual_intent("opening citrus, drydown woody", "medium")[2]
    assert "dry" not in _perceptual_intent("드라이다운 우디", "medium")[2]
    assert _perceptual_intent("dry woody scent", "medium")[2]["dry"] == 1.
    assert _perceptual_intent("드라이한 우디 향", "medium")[2]["dry"] == 1.


def test_explicit_90_is_valid_but_default_95_and_guard_semantics_remain():
    assert LotionEstimateRequest(brief="rose scent", target_similarity=90.).target_similarity == 90.
    assert LotionEstimateRequest(brief="rose scent").target_similarity == 95.
    with pytest.raises(ValueError):
        LotionEstimateRequest(brief="rose scent", target_similarity=89.9)
    assert refinement_score_floors([89., 93.], 90.).tolist() == [89., 90.]


def test_lotion_prepare_does_not_silently_raise_90_to_95():
    from tests.test_lotion_v21 import fixture
    from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
    from fragrance_ai.recommender.lotion_optimizer import prepare_lotion_optimization
    value, catalog = fixture()
    value["target_similarity"] = 90.
    result, _, _ = prepare_lotion_optimization(LotionOptimizationRequest.model_validate(value), catalog)
    assert result["effective_target"] == 90.
