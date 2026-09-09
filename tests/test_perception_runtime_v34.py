import hashlib
from pathlib import Path
from types import SimpleNamespace
from datetime import date

import pytest

from fragrance_ai.recommender.perception_runtime import PATH_ENV, HASH_ENV, configured_perception, environment_snapshot
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.runtime import RuntimeAIFactory
from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.fixed_assessment import assess_fixed_formula
from tests.test_dose_refinement import materials

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'.benchmarks/perception_runtime_v34/manifest.json'
DIGEST = 'a0be5c314fbac46e175d718a5e3ed8ed920e07b0e8ef8500ac8a1f1f810fbea1'


def test_missing_or_bad_configuration_never_silently_falls_back(monkeypatch, tmp_path):
    monkeypatch.delenv(PATH_ENV, raising=False)
    monkeypatch.delenv(HASH_ENV, raising=False)
    assert configured_perception() is None
    monkeypatch.setenv(PATH_ENV, str(tmp_path/'missing'))
    with pytest.raises(ValueError, match='together'):
        configured_perception()
    monkeypatch.setenv(HASH_ENV, '0'*64)
    path = tmp_path/'invalid.json'
    path.write_text('{}')
    monkeypatch.setenv(PATH_ENV, str(path))
    with pytest.raises(ValueError, match='hash mismatch'):
        configured_perception()


def test_unpromoted_production_rejected_before_reading_models(monkeypatch):
    monkeypatch.setenv(PATH_ENV, 'missing')
    monkeypatch.setenv(HASH_ENV, '0'*64)
    monkeypatch.setenv('PERFUMERY_AI_ENV', 'production')
    with pytest.raises(ValueError, match='production'):
        configured_perception()


def test_fixed_formula_calls_model_without_changing_weights():
    calls = []
    class Session:
        def evaluate_lines(self, lines, by_id):
            calls.append({line.ingredient_id: line.concentrate_percent for line in lines})
            return {'score': 60}
        def report(self, baseline, selected, **kwargs):
            return {'selected': selected, 'recipe_changed': False}
    provider = SimpleNamespace(begin=lambda brief: Session())
    weights = {'top': 25., 'heart': 40., 'base': 35.}
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog(materials()), perception_guidance=provider) as ai:
        r = assess_fixed_formula(ai, 'citrus floral woody', RecipeConstraints(), weights, as_of=date(2026, 9, 5))
    assert calls == [weights]
    assert r.perception_guidance['operation'] == 'fixed_formula_reassessment'
    assert not r.perception_guidance['weights_modified'] and not r.recipe


@pytest.mark.skipif(not MANIFEST.exists(), reason='local research artifacts not distributed')
def test_real_model_shared_by_sdk_factory_and_drift_guard(monkeypatch):
    monkeypatch.setenv('PERFUMERY_AI_ENV', 'development')
    monkeypatch.setenv(PATH_ENV, str(MANIFEST))
    monkeypatch.setenv(HASH_ENV, DIGEST)
    provider = configured_perception()
    assert configured_perception() is provider
    factory = RuntimeAIFactory()
    with factory() as worker, NaturalLanguagePerfumeryAI() as sdk:
        assert worker.perception_guidance is sdk.perception_guidance is provider
        assert worker.runtime_contract['perception_model']['component_model_sha256'] == provider.component_model_sha256
    monkeypatch.delenv(PATH_ENV)
    monkeypatch.delenv(HASH_ENV)
    with pytest.raises(ValueError, match='policy changed'):
        factory()


def test_product_and_unmapped_axes_are_not_misreported_as_full_application():
    from tests.test_perception_guidance import provider, brief
    b = brief({'woody': 1.})
    b.constraints.product_category = 'body_lotion'
    session = provider().begin(b)
    assert not session.enabled
    report = session.report(None, None, changed=False, variants=0)
    assert not report['model_application']['product_supported_as_research_prior']
    assert not report['model_application']['full_model_application']
    b = brief({'woody': 1.}, avoided=['musky'])
    report = provider().begin(b).report(None, None, changed=False, variants=0)
    assert report['unsupported_avoided_dimensions'] == ['musky']


def test_manual_edit_discards_previous_learned_prediction(tmp_path):
    from fragrance_ai.platform.workspace import FormulaWorkspaceService
    from fragrance_ai.platform.store import SqliteWorkspaceStore
    with_store = SqliteWorkspaceStore(tmp_path/'edits.db')
    try:
        workspace = FormulaWorkspaceService(store=with_store, ai_factory=lambda: None, catalog=IngredientCatalog(materials()))
        base = {'brief': {'target_profile': {'woody': 1.}, 'constraints': {'max_risk_tier': 2}},
                'perception_guidance': {'selected': {'score': 99}, 'model_application': {'full_model_application': True}}}
        lines = [{'ingredient_id': k, 'concentrate_percent': v} for k,v in {'top':25., 'heart':40., 'base':35.}.items()]
        result = workspace._manual_payload(base, lines)
        assert result['perception_guidance']['selected'] is None
        assert not result['perception_guidance']['model_application']['full_model_application']
        assert base['perception_guidance']['selected']['score'] == 99
    finally:
        with_store.close()
