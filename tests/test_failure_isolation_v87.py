from copy import deepcopy
from types import SimpleNamespace

import pytest

from fragrance_ai.recommender.failure_recovery import (
    failed_only_recovery, recovery_active, recovery_context, dose_correction_active, dose_correction_seed,
)


@pytest.mark.parametrize('product', ['perfume', 'body_lotion'])
def test_success_is_returned_verbatim_without_a_second_call_or_metadata(product):
    original = {'recipe':[{'ingredient_id':'kept','concentrate_percent':100.}],
                'profile_target_met':True,'score':93.,'confidence':None,'nested':{'version':'original'}}
    snapshot = deepcopy(original)
    calls = []
    @failed_only_recovery(product)
    def run():
        calls.append(recovery_active())
        return original
    actual = run()
    assert actual is original and actual == snapshot
    assert calls == [False] and not recovery_active()


def test_successful_recipe_object_is_not_copied_or_rewritten():
    original = SimpleNamespace(recipe=[1], closest_candidate=[], full_profile_target_met=True,
                               calculated_profile_similarity=93.,score_contract={'unchanged':True})
    @failed_only_recovery('perfume')
    def run():
        assert not recovery_active()
        return original
    assert run() is original and original.score_contract == {'unchanged':True}


def test_success_bypasses_even_recovery_eligibility_planning():
    original = {'recipe':[1], 'score':92.}
    def should_not_run(*args, **kwargs):
        raise AssertionError('successful response entered recovery planning')
    @failed_only_recovery('perfume', eligibility=should_not_run)
    def run():
        return original
    assert run() is original


def test_only_scored_failure_with_candidate_retries_and_accepts_an_improvement():
    baseline = {'recipe':[],'closest_candidate':[1], 'score':88.,'profile_target_met':False}
    candidate = {'recipe':[2],'closest_candidate':[], 'score':91.,'profile_target_met':True}
    flags = []
    @failed_only_recovery('perfume')
    def run():
        flags.append(recovery_active())
        return candidate if recovery_active() else baseline
    result = run()
    assert flags == [False, True] and result['recipe'] == [2]
    assert result['failure_recovery']['recovery_selected']
    assert 'failure_recovery' not in baseline and 'failure_recovery' not in candidate


def test_worse_repair_does_not_replace_the_original_score_or_recipe():
    baseline = {'recipe':[], 'closest_candidate':['old'], 'score':88., 'profile_target_met':False}
    @failed_only_recovery('perfume')
    def run():
        return {'recipe':[], 'closest_candidate':['worse'], 'score':85., 'profile_target_met':False} if recovery_active() else baseline
    result = run()
    assert result['score'] == 88. and result['closest_candidate'] == ['old']
    assert not result['failure_recovery']['recovery_selected']


def test_missing_reference_failure_does_not_trigger_unbounded_reinterpretation():
    baseline = {'recipe':[], 'closest_candidate':[], 'score':None, 'profile_target_met':False}
    @failed_only_recovery('perfume')
    def run():
        assert not recovery_active()
        return baseline
    assert run() is baseline


def test_context_is_restored_after_a_repair_exception():
    @failed_only_recovery('perfume')
    def run():
        if recovery_active():
            raise RuntimeError('visible repair failure')
        return {'recipe':[], 'closest_candidate':[1], 'score':88.}
    with pytest.raises(RuntimeError, match='visible repair failure'):
        run()
    assert not recovery_active()


def test_alias_recovery_view_never_mutates_the_shared_baseline_registry():
    from fragrance_ai.recommender.odor_expression import _recovery_view
    rows = {key:{'kind':'odor','coarse_projection':{'spicy':1.},'reference_key':key} for key in ('spicy','spices')}
    original = {'rows':rows,'aliases':{'스파이시':'spicy','spicy':'spices'}}
    snapshot = deepcopy(original)
    assert _recovery_view(original) is original
    with recovery_context():
        modified = _recovery_view(original)
        assert modified['aliases']['스파이시'] == 'spices'
    assert original == snapshot and _recovery_view(original) is original


def test_lotion_without_the_changed_language_path_is_not_recomputed():
    original = {'recipe':[], 'closest_candidate':[1], 'score':88., 'profile_target_met':False}
    @failed_only_recovery('body_lotion')
    def run(request):
        assert not recovery_active()
        return original
    assert run(SimpleNamespace(brief='aquatic scent')) is original


def test_observed_lotion_failure_supplies_actual_recipe_to_dose_correction():
    original = {'recipe':[], 'closest_candidate':[{'ingredient_id':'x', 'concentrate_percent':100.}],
                'score':88., 'profile_target_met':False,
                'product_model':{'evaluation_version':'lotion-observed-exposure/v2'}}
    calls = []
    @failed_only_recovery('body_lotion')
    def run(request, *, _incumbent_recipe=None):
        calls.append((recovery_active(), _incumbent_recipe))
        return ({'recipe':[{'ingredient_id':'x', 'concentrate_percent':99.},
                           {'ingredient_id':'y', 'concentrate_percent':1.}],
                 'score':90., 'profile_target_met':True} if recovery_active() else original)
    result = run(SimpleNamespace(brief='aquatic scent'))
    assert calls == [(False, None), (True, original['closest_candidate'])]
    assert result['recipe'][0]['concentrate_percent'] == 99.
    assert original['closest_candidate'][0]['concentrate_percent'] == 100.


def test_progress_observer_can_cancel_a_failed_formula_repair():
    calls = []
    def observer(stage):
        calls.append(stage)
        if len(calls) == 2:
            raise RuntimeError('caller cancelled')
    @failed_only_recovery('perfume')
    def run(*, progress_callback):
        progress_callback('INGREDIENT_SCREENING')
        return {'recipe':[], 'closest_candidate':[1], 'score':88.}
    with pytest.raises(RuntimeError, match='caller cancelled'):
        run(progress_callback=observer)
    assert len(calls) == 2 and not recovery_active()


def test_v87_recovered_success_bypasses_v88_calibration_entirely():
    states = []
    successful = {'recipe':[1], 'score':91., 'profile_target_met':True}
    @failed_only_recovery('perfume')
    def run():
        states.append((recovery_active(), dose_correction_active()))
        assert not dose_correction_active()
        return successful if recovery_active() else {'recipe':[], 'closest_candidate':[2], 'score':88.}
    result = run()
    assert result['recipe'] == [1]
    assert states == [(False, False), (True, False)]
    assert result['failure_recovery']['version'] == 'failed-only-recovery/v87'


def test_v88_corrects_only_after_complete_v87_failure_and_uses_its_latest_recipe():
    states = []
    @failed_only_recovery('perfume')
    def run():
        states.append((recovery_active(), dose_correction_active()))
        if dose_correction_active():
            assert dose_correction_seed() == ['v87-candidate']
            return {'recipe':['corrected-dose'], 'score':90.2}
        return {'recipe':[], 'closest_candidate':['v87-candidate' if recovery_active() else 'old'],
                'score':89. if recovery_active() else 88.}
    result = run()
    assert states == [(False, False), (True, False), (True, True)]
    assert result['recipe'] == ['corrected-dose']
    assert result['failure_recovery']['core_evaluations'] == 3
    assert dose_correction_seed() == ()
    assert not dose_correction_active()


def test_lotion_language_recovery_applies_only_after_failure():
    flags = []
    @failed_only_recovery('body_lotion')
    def run(request):
        flags.append(recovery_active())
        return ({'recipe':[1], 'score':91., 'profile_target_met':True} if recovery_active() else
                {'recipe':[], 'closest_candidate':[1], 'score':88., 'profile_target_met':False})
    assert run(SimpleNamespace(brief='스파이시 향'))['recipe'] == [1]
    assert flags == [False, True]


def test_recovery_context_cannot_spill_into_another_request_thread():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered, release = Event(), Event()
    def repairing():
        with recovery_context():
            entered.set()
            assert release.wait(5)
            return recovery_active()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(repairing)
        assert entered.wait(5)
        try:
            assert pool.submit(recovery_active).result(timeout=5) is False
            assert not recovery_active()
        finally:
            release.set()
        assert first.result(timeout=5) is True
    assert not recovery_active()
