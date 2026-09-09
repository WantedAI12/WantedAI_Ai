import json
import numpy as np
import pytest

from fragrance_ai.research.kernel_profiles import fit_kernel, predict_component_regressor
from fragrance_ai.recommender.compact_language import AssistantRequest, assistant_reply, validate_proposal, completion_payload
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog


def test_kernel_portable_nonlinear_and_invalid_contract():
    rng = np.random.default_rng(33)
    x = np.c_[rng.integers(0, 2, (12, 1024)), rng.normal(size=(12, 5))]
    y = rng.random((12, 4))
    model = fit_kernel(x, y, .1)
    a = predict_component_regressor(model, x)
    assert np.allclose(a, predict_component_regressor(json.loads(json.dumps(model)), x))
    assert np.isfinite(a).all() and (a >= 0).all()
    with pytest.raises(ValueError):
        predict_component_regressor(model, np.zeros((1, 2)))
    model['scale'][0] = 0
    with pytest.raises(ValueError):
        predict_component_regressor(model, x)


def test_assistant_is_proposal_only_and_not_generator():
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    result = assistant_reply(AssistantRequest(message='달지 않은 우디 향수'), parser,
        lambda _: {'desired': ['woody'], 'avoided': ['gourmand'], 'product': 'perfume', 'clarification': 'none'})
    assert result['requires_confirmation'] and not result['formula_generated']
    assert result['intent_proposal']['avoided'] == ['gourmand']
    assert '우디' in result['message']


def test_malicious_model_fields_rejected_and_fallback_does_not_retry():
    calls = []
    def backend(text):
        calls.append(text)
        return {'recipe': [{'name': 'invented', 'percent': 100}], 'accuracy': 100}
    result = assistant_reply(AssistantRequest(message='시트러스 향수'), NaturalLanguageBriefParser(IngredientCatalog.load_builtin()), backend)
    assert len(calls) == 1 and result['source'] == 'deterministic_fallback_after_model_failure'
    assert 'recipe' not in result and 'accuracy' not in result


def test_conflict_and_unknown_axes_fail_closed():
    value = {'desired': ['woody'], 'avoided': ['woody'], 'product': 'perfume', 'clarification': 'none'}
    assert validate_proposal(value).clarification == 'conflict'
    value['desired'] = ['invented']
    with pytest.raises(ValueError):
        validate_proposal(value)
    assert completion_payload('hello')['max_tokens'] == 160


def test_assistant_api_uses_same_rate_limiter_and_never_generates():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    class Formula(BaseModel):
        brief: str
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('language endpoint must not invoke generation')
    app = FastAPI()
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), forbidden, lambda: calls.append('rate'),
        language_backend=lambda _: {'desired': ['woody'], 'avoided': [], 'product': 'perfume', 'clarification': 'none'})
    with TestClient(app) as client:
        r = client.post('/v1/ai/assistant', json={'message': '우디 향수'})
        assert r.status_code == 200 and r.json()['source'] == 'quantized_language_model'
        assert r.json()['requires_confirmation'] and calls == ['rate']
        assert client.post('/v1/ai/assistant', json={'message': 'a'*1201}).status_code == 422
