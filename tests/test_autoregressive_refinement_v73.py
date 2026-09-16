import numpy as np
import pytest

from fragrance_ai.recommender.autoregressive_refinement import ResidualFeedback, perfume_residual, perfume_feedback_weights


def test_feedback_uses_previous_predictions_and_bounded_mass_steps():
    history=ResidualFeedback(target=95.)
    history.observe([.8,.2],[.5,-.5],50.)
    history.observe([.6,.4],[.3,-.3],70.)
    proposal=history.proposal()
    assert proposal is not None
    assert 0 <= proposal[0] < .6
    assert proposal.sum()==pytest.approx(1.)
    assert np.abs(proposal-[.6,.4]).sum() <= .15+1e-10
    assert not history.observe([.9,.1],[.8,-.8],20.)
    np.testing.assert_array_equal(history.history[-1].weights,[.6,.4])
    assert history.report()['network_weights_updated'] is False


def test_feedback_never_overwrites_qualified_formula_with_nonspecific_score():
    history=ResidualFeedback(target=95.)
    history.observe([.5,.5],[.01,-.01],99.)
    assert history.observe([.47,.53],[.03,-.03],97.,qualified=True)
    assert not history.observe([.5,.5],[0.,0.],100.)
    assert history.proposal() is None


def test_history_is_independent_bounded_and_rejects_contract_drift():
    history=ResidualFeedback(target=99.,history_size=3)
    weights=np.array([.8,.2])
    for i in range(6):
        history.observe(weights-[.01*i,-.01*i],[.5-.01*i,-.5+.01*i],50.+i)
    assert len(history.history)==3
    weights[:]=0
    assert history.history[-1].weights.sum()==pytest.approx(1.)
    with pytest.raises(ValueError):
        history.observe([.5,.2],[.1,-.1],60.)
    with pytest.raises(ValueError):
        history.observe([.5,.5],[.1,-.1],90.,qualified=True)


def test_singular_feedback_has_no_fabricated_direction():
    history=ResidualFeedback(target=99.)
    history.observe([.8,.2],[.5,-.5],50.)
    history.observe([.6,.4],[.5,-.5],51.)
    assert history.proposal() is None


def test_perfume_feedback_keeps_extra_axes_and_phase_weights():
    row={'target_profile':{'a':1.},'predicted_profile':{'a':.75,'b':.25}}
    result=perfume_residual({'nominal':row,'temporal':[{**row,'weight':.25}]},('a','b'))
    np.testing.assert_array_equal(result,[.25,-.25,.125,-.125])


def test_feedback_accepts_existing_render_precision_without_mutating_recipe():
    original=np.array([33.3333,33.3333,33.3335])
    preserved=original.copy()
    values=perfume_feedback_weights(original)
    np.testing.assert_array_equal(original,preserved)
    assert values.sum()==pytest.approx(100.)
    with pytest.raises(ValueError):
        perfume_feedback_weights([50.,51.])


def test_actual_neural_proposal_path_accepts_array_valued_temporal_targets(monkeypatch):
    from types import SimpleNamespace
    from tests.test_dose_refinement import setup_case
    from fragrance_ai.recommender import dose_refinement,fractional_transfers
    from fragrance_ai.recommender.profile_match import assess_recipe_profiles
    items,brief,_,policy=setup_case('floral')
    calls=[]
    def learned(candidates,profiles,targets,responses,weights,**kwargs):
        assert targets.shape==(1,6,19)
        np.testing.assert_allclose(targets.sum(-1),1.)
        calls.append(kwargs['product'])
        return np.array([.2,.6,.2]),{'trained_recurrent_decoder_used':True}
    monkeypatch.setattr(dose_refinement,'_one_pass_dose_refinement_proposals',lambda *a,**k:iter(()))
    monkeypatch.setattr(fractional_transfers,'fractional_transfer_seeds',lambda *a,**k:([],{}))
    lines=[SimpleNamespace(ingredient_id=item.ingredient_id,concentrate_percent=weight) for item,weight in zip(items,[35.,30.,35.])]
    assessment=assess_recipe_profiles(brief,{'floral':.7,'woody':.3},[],[])
    guide=SimpleNamespace(provider=SimpleNamespace(core=SimpleNamespace(autoregressive_proposal=learned)),enabled=False)
    records=list(dose_refinement.dose_refinement_proposals(items,brief,{},lines,policy,[],{},
        current_lines=lambda:lines,current_score=lambda:assessment['score'],current_feedback=lambda:assessment,
        guidance=guide))
    assert calls==['perfume']
    assert any(row['replacement_mode']=='trained_autoregressive_decoder' for row in records)


def test_low_cap_neural_receiver_keeps_incumbent_mass_balance_support():
    from types import SimpleNamespace
    from fragrance_ai.recommender.autoregressive_refinement import neural_mass_seeds
    items=[SimpleNamespace(ingredient_id=str(i)) for i in range(100)]
    predicted=np.zeros(100)
    predicted[80]=1.
    original={'0':30.,'1':30.,'2':40.}
    proposals,report=neural_mass_seeds(items,predicted,original,{},12)
    assert report['full_pool_scored']==100 and not report['permanent_candidate_shortlist']
    for _,seed in proposals:
        assert set(original)<=set(seed) and '80' in seed
        assert sum(seed.values())==pytest.approx(100.)
    damped=proposals[0][1]
    assert damped['80']==pytest.approx(7.5)
    assert sum(abs(damped.get(key,0.)-original.get(key,0.)) for key in set(damped)|set(original))<=15.+1e-8
