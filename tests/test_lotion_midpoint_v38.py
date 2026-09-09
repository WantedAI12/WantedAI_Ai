from fragrance_ai.recommender import lotion_estimation as module
from tests.test_lotion_base_design_v28 import result


def test_midpoint_can_recover_target_without_replacing_reference(monkeypatch):
    calls = []
    def run(*a, **kw):
        oil = kw.get('oil_base_percent', 10.)
        calls.append(oil)
        return result(96. if oil == 25 else 94. if oil == 30 else 93.)
    monkeypatch.setattr(module, '_estimate_single_base', run)
    r = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='citrus woody', base_design={}), None)
    assert calls == [10, 20, 30, 25]
    assert r['profile_target_met'] and r['base_design']['selected_oil_base_percent'] == 25
    assert not r['base_design']['fixed_reference_passed95']
    assert r['base_design']['fixed_reference_score'] == 93


def test_explicit_zero_restores_three_variant_search(monkeypatch):
    calls = []
    def run(*a, **kw):
        calls.append(kw.get('oil_base_percent', 10.))
        return result(94.)
    monkeypatch.setattr(module, '_estimate_single_base', run)
    r = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='citrus woody',
        base_design={'adaptive_oil_refinement_steps': 0}), None)
    assert calls == [10, 20, 30]
    assert not r['profile_target_met']


def test_midpoint_keeps_within_custom_endpoint_range(monkeypatch):
    calls = []
    def run(*a, **kw):
        calls.append(kw.get('oil_base_percent', 10.))
        return result(90.)
    monkeypatch.setattr(module, '_estimate_single_base', run)
    module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='citrus woody',
        base_design={'additional_oil_base_percents': [12.], 'adaptive_oil_refinement_steps': 3}), None)
    assert len(calls) == 5 and all(10 <= oil <= 12 for oil in calls)


def test_proven_profile_ceiling_skips_unhelpful_base_search(monkeypatch):
    calls = []
    def run(*a, **kw):
        calls.append(kw.get('oil_base_percent', 10.))
        r = result(40.)
        r['preparation'] = {'evaluation_targets': [{'target_profile': {'clean': 1.}}]}
        return r
    monkeypatch.setattr(module, '_estimate_single_base', run)
    monkeypatch.setattr(module, 'lotion_profile_coverage', lambda *a: {'target_excluded': True,
        'optimistic_profile_upper_percent': 43.})
    r = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief='clean scent', base_design={}), None)
    assert calls == [10]
    assert not r['profile_target_met'] and r['score'] == 40.
    assert r['base_design']['adaptive_oil_trials'] == []
