from copy import copy
from pathlib import Path

import numpy as np
import pytest

from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from fragrance_ai.recommender.lotion_perception import LotionShapePredictor, REFERENCE_DILUTIONS
from tests.test_lotion_learned_search_v36 import setup
from tests.test_perception_guidance import material, provider


def test_fractional_step_reaches_known_optimum_not_coarse_grid(monkeypatch):
    request, catalog, p, _ = setup(monkeypatch)
    result = optimize_lotion(request, catalog, perception_guidance=p)
    report = result['learned_optimization']
    assert report['selected_affinity'] == pytest.approx(100., abs=1e-7)
    assert report['selected_affinity'] > report['precision_refinement']['previous_affinity']+.1
    assert report['precision_refinement']['accepted_steps'] > 0
    assert result['score'] == pytest.approx(100.)


def test_precision_timeout_preserves_previous_solver_candidate(monkeypatch):
    from types import SimpleNamespace
    from fragrance_ai.recommender import lotion_learned_search
    request, catalog, p, _ = setup(monkeypatch)
    monkeypatch.setattr(lotion_learned_search, 'conditioned_linprog', lambda *a, **kw: SimpleNamespace(status=1))
    result = optimize_lotion(request, catalog, perception_guidance=p)
    report = result['learned_optimization']
    assert report['selected_affinity'] == pytest.approx(report['precision_refinement']['previous_affinity'])
    assert report['recipe_changed'] and report['solver_incomplete']


def test_prefetch_chunks_all_materials_and_reuses_request_cache():
    items = [material(str(i)) for i in range(257)]
    p = provider({item.ingredient_id: ('CCO', None) for item in items})
    predictor = LotionShapePredictor(p)
    sizes = []
    def batch(values, doses):
        sizes.append(len(values))
        predictor.session.stock_batch_forward_calls = getattr(predictor.session, 'stock_batch_forward_calls', 0)+1
        return [(np.ones((3, len(p.endpoints))), [{}]*3) for _ in values]
    predictor.session.predict_stock_batch = batch
    predictor.prefetch(items)
    assert sizes == [128, 128, 1]
    assert predictor.calls == 257 and predictor.forward_batches == 3
    predictor.prefetch(items)
    for item in items:
        assert predictor.shape(item).sum(axis=1) == pytest.approx(np.ones(3))
    assert sizes == [128, 128, 1]


def test_shared_cache_is_bound_to_checkpoint_structure_and_native_features(monkeypatch):
    from collections import OrderedDict
    from dataclasses import replace
    import threading
    _, catalog, p, calls = setup(monkeypatch)
    p.lotion_shape_cache = OrderedDict()
    p.lotion_shape_cache_lock = threading.RLock()
    item = catalog.ingredients[0]
    first = LotionShapePredictor(p)
    expected = first.shape(item).copy()
    second = LotionShapePredictor(p)
    np.testing.assert_array_equal(second.shape(item), expected)
    assert len(calls) == 1 and second.shared_cache_hits == 1
    with pytest.raises(ValueError):
        second.shape(item)[0, 0] = 100
    second.basis[item.ingredient_id]['reference_predictions'][0]['basis'] = 'changed by caller'
    third = LotionShapePredictor(p)
    third.shape(item)
    assert third.basis[item.ingredient_id]['reference_predictions'][0]['basis'] != 'changed by caller'
    LotionShapePredictor(p).shape(replace(item, profile={'woody': 1.}))
    assert len(calls) == 2
    p.component_model_sha256 = 'another-checkpoint'
    LotionShapePredictor(p).shape(item)
    assert len(calls) == 3


@pytest.mark.skipif(not Path('.benchmarks/perception_runtime_v34/manifest.json').exists(), reason='local checkpoint unavailable')
def test_actual_checkpoint_batch_matches_scalar_predictions_and_anchor_provenance(monkeypatch, tmp_path):
    import hashlib
    import json
    from fragrance_ai.recommender.perception_runtime import LOTION_PATH_ENV, LOTION_HASH_ENV, configured_perception
    from fragrance_ai.recommender.catalog import IngredientCatalog
    monkeypatch.setenv('PERFUMERY_AI_ENV', 'development')
    source = Path('.benchmarks/perception_runtime_v34/manifest.json').resolve()
    document = json.loads(source.read_text(encoding='utf-8'))
    document['product'] = 'body_lotion'
    for name in ('base_model', 'component_model', 'registry'):
        document[name]['path'] = str((source.parent / document[name]['path']).resolve())
    path = tmp_path/'lotion-v3.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    monkeypatch.setenv(LOTION_PATH_ENV, str(path))
    monkeypatch.setenv(LOTION_HASH_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    p = copy(configured_perception('body_lotion'))
    predictor = LotionShapePredictor(p)
    items = [item for item in IngredientCatalog.load_builtin().ingredients if predictor.session.supports(item)]
    # Include an out-of-bank graph as well as measured-anchor core structures.
    for identifier, (smiles, _) in p.structures.items():
        if identifier.startswith('registry_') and '.' not in smiles and smiles not in p.by_structure:
            items.append(material(identifier))
            break
    assert any(predictor.session._prepare(item)[0] is None for item in items)
    assert any(predictor.session._prepare(item)[0] is not None for item in items)
    expected = [predictor.session._predict(item, np.asarray(REFERENCE_DILUTIONS)) for item in items]
    actual = predictor.session.predict_stock_batch(items, REFERENCE_DILUTIONS)
    for (a, basis_a), (b, basis_b) in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-11)
        assert basis_a == basis_b
