"""Exact-solve reuse and bounded language caching; no relaxed quality gates."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog

from fragrance_ai.recommender import linear_program_cache as cache


def problem():
    return dict(A_ub=sparse.csr_matrix([[1., 2.]]), b_ub=np.array([3.]),
                A_eq=np.array([[1., 1.]]), b_eq=np.array([1.]),
                bounds=[(0., 1.), (0., None)], method='highs', options={'time_limit': 2.})


def test_lp_reuse_matches_solver_and_returned_arrays_are_owned():
    calls = []

    def solve(*a, **kw):
        calls.append(1)
        return linprog(*a, **kw)

    @cache.linear_program_request
    def request():
        first = cache.cached_linprog(solve, np.array([1., 0.]), **problem())
        expected = first.x.copy()
        first.x[:] = 99.
        second = cache.cached_linprog(solve, np.array([1., 0.]), **problem())
        np.testing.assert_array_equal(second.x, expected)
        second.ineqlin.residual[:] = -99.
        third = cache.cached_linprog(solve, np.array([1., 0.]), **problem())
        assert np.all(third.ineqlin.residual >= 0)

    request()
    assert len(calls) == 1
    request()
    assert len(calls) == 2  # no cross-request/model-snapshot state


@pytest.mark.parametrize('field', ['A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'method', 'options'])
def test_every_numeric_constraint_and_solver_policy_changes_key(field):
    before = problem()
    after = problem()
    changes = {'A_ub': sparse.csr_matrix([[2., 2.]]), 'b_ub': np.array([4.]),
               'A_eq': np.array([[2., 1.]]), 'b_eq': np.array([2.]),
               'bounds': [(0., .5), (0., None)], 'method': 'highs-ipm',
               'options': {'time_limit': 1.}}
    after[field] = changes[field]
    assert cache._problem_key(np.array([1., 0.]), before) != cache._problem_key(np.array([1., 0.]), after)
    assert cache._problem_key(np.array([0., 1.]), before) != cache._problem_key(np.array([1., 0.]), before)


@pytest.mark.parametrize('status', [1, 2, 3, 4])
def test_unsuccessful_solves_are_not_cached(status):
    calls = []

    def solve(*a, **kw):
        calls.append(1)
        return OptimizeResult(status=status, success=False, x=None)

    @cache.linear_program_request
    def request():
        for _ in range(2):
            cache.cached_linprog(solve, [1., 0.], **problem())

    request()
    assert len(calls) == 2


def test_unknown_solver_features_bypass_and_exception_resets_scope():
    assert cache._problem_key([1., 0.], {**problem(), 'callback': object()}) is None

    @cache.linear_program_request
    def failed():
        raise ValueError('test')

    with pytest.raises(ValueError, match='test'):
        failed()
    assert cache._REQUEST.get() is None


def test_parallel_requests_do_not_share_solver_state():
    barrier = Barrier(2)
    calls = []

    def solve(*a, **kw):
        calls.append(1)
        return OptimizeResult(status=0, success=True, x=np.array([.5, .5]))

    @cache.linear_program_request
    def request(_):
        barrier.wait(timeout=5)
        for _ in range(2):
            cache.cached_linprog(solve, [1., 0.], **problem())

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(request, range(2)))
    assert len(calls) == 2


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(cache, '_MAX_ENTRIES', 2)
    calls = []

    def solve(*a, **kw):
        calls.append(1)
        return OptimizeResult(status=0, success=True, x=np.ones(2))

    @cache.linear_program_request
    def request():
        for value in (1., 2., 3., 1.):
            cache.cached_linprog(solve, [value, 0.], **problem())
            assert len(cache._REQUEST.get()['entries']) <= 2

    request()
    assert len(calls) == 4


def test_actual_full_pool_reuses_solve_without_restricting_pool(monkeypatch):
    from fragrance_ai.recommender import global_profile_search as search
    from tests.test_full_pool_search import material, brief
    items = [material('a', 'top'), material('b', 'heart'), material('c', 'base'),
             material('unused', 'heart', profile={'floral': 1.})]
    calls = []
    original = search.linprog

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(search, 'linprog', counted)

    @cache.linear_program_request
    def request():
        first = search.optimize_full_pool(items, brief())
        second = search.optimize_full_pool(items, brief())
        assert first == second and first.pool_size == 4
        assert first.weights_percent == pytest.approx({'a': 25., 'b': 40., 'c': 35.})

    request()
    assert len(calls) == 1


def test_realism_sourcing_uses_formula_only_and_preserves_unique_id_semantics():
    from fragrance_ai.recommender.realism import assess_realism
    from fragrance_ai.recommender.catalog import HistoricalReferenceCorpus
    from tests.test_dose_refinement import setup_case
    items, brief, lines, _ = setup_case()

    class NoCatalogScan(dict):
        def values(self):
            raise AssertionError('full catalog iteration is unnecessary')

    ingredients = NoCatalogScan((item.ingredient_id, replace(item, availability=.71 + .07 * i))
                               for i, item in enumerate(items))
    for i in range(100):
        ingredients[f'unused-{i}'] = replace(items[0], ingredient_id=f'unused-{i}', availability=.1)
    result = assess_realism(lines, ingredients, brief, HistoricalReferenceCorpus())
    assert result.components['sourcing_plausibility'] == 78.
    duplicate = assess_realism([*lines, lines[0]], ingredients, brief, HistoricalReferenceCorpus())
    assert duplicate.components['sourcing_plausibility'] == 58.5


def language_app(backend, guard=None):
    from fastapi import FastAPI
    from pydantic import BaseModel
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    from fragrance_ai.recommender.catalog import IngredientCatalog

    class Formula(BaseModel):
        brief: str

    app = FastAPI()
    rate_calls = []
    register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(),
        lambda *a, **kw: pytest.fail('assistant invoked recipe generation'),
        lambda: rate_calls.append(1), language_backend=backend, runtime_guard=guard)
    return app, rate_calls


def test_language_repeat_uses_zero_inference_calls_same_response_and_model_identity():
    from fastapi.testclient import TestClient
    calls, version = [], [1]

    def backend(message):
        calls.append(message)
        return {'desired': ['woody'], 'avoided': [], 'product': 'perfume', 'clarification': 'none'}

    backend.contract = lambda: {'version': version[0]}
    app, rates = language_app(backend)
    with TestClient(app) as client:
        first = client.post('/v1/ai/assistant', json={'message': '우디 향수'})
        repeat = client.post('/v1/ai/assistant', json={'message': '우디 향수'})
        assert first.status_code == repeat.status_code == 200
        assert first.json() == repeat.json()
        assert first.headers['X-Perfumery-LLM-Calls'] == '1'
        assert repeat.headers['X-Perfumery-LLM-Calls'] == '0'
        assert repeat.headers['X-Perfumery-Language-Cache'] == 'hit'
        assert len(calls) == 1 and len(rates) == 2
        version[0] = 2
        changed = client.post('/v1/ai/assistant', json={'message': '우디 향수'})
        assert changed.headers['X-Perfumery-LLM-Calls'] == '1'
        assert len(calls) == 2
        client.post('/v1/ai/assistant', json={'message': '우디 향수를 원해요'})
        assert len(calls) == 3  # only exact requests may share


def test_language_invalid_output_is_not_cached_or_retried():
    from fastapi.testclient import TestClient
    calls = []

    def backend(message):
        calls.append(message)
        return {'invented_recipe': 100}

    app, _ = language_app(backend)
    with TestClient(app) as client:
        for _ in range(2):
            value = client.post('/v1/ai/assistant', json={'message': '우디 향수'})
            assert value.json()['source'] == 'deterministic_fallback_after_model_failure'
            assert value.headers['X-Perfumery-LLM-Calls'] == '1'
    assert len(calls) == 2 and app.state.language_cache.stats()['entries'] == 0


def test_language_guard_is_checked_even_on_hits():
    from fastapi.testclient import TestClient
    current = [True]

    def guard():
        if not current[0]:
            raise ValueError('changed snapshot')

    app, _ = language_app(lambda _: {'desired': ['woody'], 'avoided': [],
        'product': 'perfume', 'clarification': 'none'}, guard)
    with TestClient(app) as client:
        assert client.post('/v1/ai/assistant', json={'message': '우디 향수'}).status_code == 200
        current[0] = False
        with pytest.raises(ValueError, match='changed snapshot'):
            client.post('/v1/ai/assistant', json={'message': '우디 향수'})
