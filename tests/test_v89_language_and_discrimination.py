from types import SimpleNamespace
import numpy as np
import pytest
from fragrance_ai.recommender import reference_discrimination as discrimination
from fragrance_ai.recommender.odor_expression import parse_expression, registry
from fragrance_ai.recommender.odor_language_v89 import SCENES, resolve_atoms, dictionary


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv('PERFUMERY_AI_LOCAL_PROFILE','disabled')


def test_every_authored_scene_has_resolved_atoms():
    source=registry()
    assert len(SCENES)>=60
    assert all(resolve_atoms(words,source) for _,_,words in SCENES)


def test_composition_retains_explicit_exclusion():
    result=parse_expression('sun dried linen, without musk')
    assert result['scene_interpretations']
    assert result['avoided']
    assert not (set(result['wanted']) & set(result['avoided']))


def test_excluded_material_does_not_create_a_scene():
    text='sun dried linen'
    result=parse_expression(text,excluded_material_spans=[(0,len(text))])
    assert result['scene_interpretations']==[]
    assert not any(m.get('scene_id') for m in result['matches'])


def test_negated_scene_is_not_requested():
    result=parse_expression('without old library, rose scent')
    scene=next(s for s in result['scene_interpretations'] if s['scene_id']=='old_library')
    assert scene['polarity']=='avoid'
    assert set(scene['atoms']) <= set(result['avoided'])


def test_dictionary_pagination_is_stable():
    source=registry()
    whole=dictionary(source,limit=10)
    first=dictionary(source,limit=5)
    second=dictionary(source,offset=first['next_offset'],limit=5)
    assert first['items']+second['items']==whole['items']


def test_family_discriminator_keeps_background_rejected_and_scope_local():
    target=np.array([.8,.15,.05]); background=np.array([.1,.3,.6])
    bank=SimpleNamespace(profiles={'clean':np.stack([target,target])},background=np.stack([background,background]))
    old=discrimination.contrast_direction(target,background)
    with discrimination.reference_context(bank):
        changed=discrimination.contrast_direction(target,background)
        assert not np.allclose(old,changed)
        assert discrimination.assess(target,target,background,clean_reference=target)['passed']
        assert not discrimination.assess(target,background,background,clean_reference=target)['passed']
        other=np.array([.2,.5,.3])
        np.testing.assert_allclose(discrimination.contrast_direction(other,background),
                                   discrimination.contrast_direction(other,background,method='cosine'))
    np.testing.assert_array_equal(discrimination.contrast_direction(target,background),old)


def test_identical_target_and_background_cannot_pass():
    target=np.array([.2,.8])
    result=discrimination.assess(target,target,target,clean_reference=target)
    assert not result['passed'] and not result['identifiable']


def test_exhausted_search_does_not_start_new_linear_solve(monkeypatch):
    from fragrance_ai.recommender import lotion_reference_search as search
    monkeypatch.setattr(search,'exhausted',lambda:True)
    def forbidden(*a,**k):
        raise AssertionError('new solve after deadline')
    monkeypatch.setattr(search,'linprog',forbidden)
    result=search._solve_with_time_recovery(np.ones(2),{}, {})
    assert not result.success and result.status==1


def test_nested_solver_budget_respects_outer_deadline():
    from time import monotonic
    from fragrance_ai.recommender import failure_recovery,search_budget
    budgets=[]
    @search_budget.governed('body_lotion')
    def run():
        budgets.append(search_budget.ACTIVE.get().total_seconds)
        return {'score':80.}
    token=failure_recovery._DEADLINE.set(monotonic()+20.)
    try:
        run()
        run()
    finally:
        failure_recovery._DEADLINE.reset(token)
    run()
    assert 0<budgets[1]<=budgets[0]<=20.
    assert budgets[2]==180.


def test_first_recovery_reuses_real_baseline_without_cross_request_leakage():
    from fragrance_ai.recommender.failure_recovery import recovery_context,dose_correction_seed,recovery_seed_score
    lines=[{'ingredient_id':'test-only','concentrate_percent':100.}]
    with recovery_context({'score':88.5,'closest_candidate':lines}):
        assert dose_correction_seed() is lines
        assert recovery_seed_score()==88.5
        with recovery_context():
            assert dose_correction_seed()==()
        assert dose_correction_seed() is lines
    assert dose_correction_seed()==() and recovery_seed_score() is None


def test_proposal_portfolio_never_promotes_a_worse_exact_blend(monkeypatch):
    from fragrance_ai.recommender import physical_blend_correction as physical
    from fragrance_ai.recommender import blend_correction,failure_inverse_v87
    class Engine:
        def predict(self,profiles,weights):
            return np.broadcast_to(weights,(2,2,2)).copy(),None
        def signal_value_gradient(self,weights):
            return np.ones(1),np.zeros((1,2))
        def __call__(self,*a,**k):
            return 0.,np.zeros((1,2)),None
    monkeypatch.setattr(physical,'release_factors',lambda *a:(np.ones((2,2)),np.ones((1,2)),None))
    monkeypatch.setattr(blend_correction,'correct_blend',lambda **k:(np.array([.1,.9]),{'surrogate_score':99.}))
    monkeypatch.setattr(physical,'subset_engine',lambda e,i:e)
    monkeypatch.setattr(failure_inverse_v87,'expanded_reference_support',lambda *a,**k:np.array([0,1]))
    monkeypatch.setattr(failure_inverse_v87,'polish_prepared',lambda *a,**k:(np.array([.7,.3]),{}))
    target=np.broadcast_to([1.,0.],(2,2,2))
    original=np.array([.8,.2])
    selected,report=physical.correct_physical_blend(Engine(),np.ones((2,2,2)),target,[0.,1.],
        np.zeros_like(target),original,np.zeros(2),np.ones(2),np.ones(2),2.,target_score=90.)
    np.testing.assert_array_equal(selected,original)
    assert report['selected_score']==pytest.approx(80.)
    assert report['score_offset']==0.
    assert {r['proposal_strategy'] for r in report['rounds']}=={'linearized_release','exact_final_objective'}
