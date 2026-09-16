"""Global hierarchy invariants; synthetic fixtures never become source data."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.recommender.models import ScentBrief, RecipeConstraints
from fragrance_ai.recommender.odor_space import OdorSpace, compile_targets, target_report, unresolved_named_odors, contextual_spans


@pytest.fixture
def bank(tmp_path):
    rows=[]
    for name,family,ref in [('citrus','citrus','citrus'),('lemon','citrus','lemon'),
        ('orange','citrus','orange'),('rose','floral','rose'),('woody','woody','woody'),
        ('newflower','floral',None),('sweet','gourmand','sweet')]:
        rows.append({'id':name,'label_en':name,'aliases':[name],'kind':'odor','coarse_projection':{family:1.},
            'reference_key':ref,'source_annotation_model_connected':bool(ref)})
    value={'schema':'hierarchical-odor-space/v77','concepts':rows,'aliases':{r['id']:r['id'] for r in rows},
        'reference_bindings':{r['id']:r['reference_key'] for r in rows},
        'endpoint_routes':{'sweet':['SWEET']},'recipe_outcomes_used':False}
    path=tmp_path/'space.json';path.write_text(json.dumps(value))
    space=OdorSpace(path,hashlib.sha256(path.read_bytes()).hexdigest())
    profiles={k:np.tile(p,(2,1)) for k,p in {'citrus':[.7,.2,.1],'lemon':[.9,.08,.02],
        'orange':[.5,.1,.4],'rose':[.1,.8,.1],'woody':[.1,.2,.7],'sweet':[.02,.08,.9]}.items()}
    return SimpleNamespace(odor_space=space,profiles=profiles,metadata={},endpoints=('CITRUS','FLORAL','SWEET'),sha256='bank')


def brief(wanted,avoided=None):
    return ScentBrief('',{'citrus':1.},['citrus'],[],[],[],'medium',{},RecipeConstraints(),
        expression_targets=wanted,expression_avoided=avoided or {})


def rows(b):
    return [{'phase':'overall','target_profile':b.target_profile,'avoided':b.avoided_dimensions}]


def test_companion_facets_are_preserved(bank):
    b=brief({'citrus':1.})
    targets,missing=compile_targets(bank,b,rows(b))
    assert not missing
    np.testing.assert_allclose(targets[0]['profiles'],bank.profiles['citrus'])
    assert targets[0]['profiles'][0,1]>0


def test_specific_detail_replaces_inferred_family_without_double_counting(bank):
    b=brief({'lemon':.9,'orange':.1})
    t,m=compile_targets(bank,b,rows(b))
    assert not m and set(t[0]['concepts'])=={'lemon','orange'}
    np.testing.assert_allclose(t[0]['profiles'],.9*bank.profiles['lemon']+.1*bank.profiles['orange'])


def test_unknown_detail_never_falls_back_to_supported_parent(bank):
    b=replace(brief({'newflower':1}),target_profile={'floral':1})
    t,m=compile_targets(bank,b,rows(b))
    assert t==[None] and 'reference_missing:newflower' in m


def test_explicit_exclusion_alone_changes_background(bank):
    b=brief({'lemon':1.},{'sweet':1.})
    t,m=compile_targets(bank,b,rows(b))
    assert not m and t[0]['profiles'][0,2]==0
    assert bank.profiles['lemon'][0,2]>.0


def test_phase_scopes_do_not_spread_details(bank):
    b=replace(brief({}),phase_expressions={'opening':{'wanted':{'lemon':1}},'drydown':{'wanted':{'woody':1}}})
    t,m=compile_targets(bank,b,[{'phase':'opening','target_profile':{'citrus':1},'avoided':[]},
        {'phase':'drydown','target_profile':{'woody':1},'avoided':[]}])
    assert not m and set(t[0]['concepts'])=={'lemon'} and set(t[1]['concepts'])=={'woody'}


def test_intent_is_product_and_candidate_independent(bank):
    b=brief({'citrus':1.})
    a=target_report(bank,b,rows(b))
    c=replace(b,constraints=replace(b.constraints,product_category='body_lotion',max_ingredients=2))
    assert a['sha256']==target_report(bank,c,rows(c))['sha256']


def test_conflicting_and_unmeasurable_exclusions_not_erased(bank):
    b=brief({'lemon':1.},{'lemon':1.})
    _,missing=compile_targets(bank,b,rows(b))
    assert 'conflicting_concept:lemon' in missing
    assert 'exclusion_endpoint_missing:lemon' in missing


def test_arbitrary_new_name_is_visible():
    assert unresolved_named_odors('quasifloral scent',[])[0]['text']=='quasifloral'
    assert not unresolved_named_odors('lemon scent',[{'start':0,'end':5}])
    assert not contextual_spans('make a fresh citrus perfume','make')
    assert contextual_spans('absinthe scent','absinthe')==[(0,8)]


def test_hash_mismatch_rejected(tmp_path):
    p=tmp_path/'space.json';p.write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        OdorSpace(p,'0'*64)


def test_subtype_cannot_survive_banning_its_whole_family(bank):
    b=replace(brief({'lemon':1.}),avoided_dimensions=['citrus'])
    _,m=compile_targets(bank,b,rows(b))
    assert 'excluded_required_family:citrus' in m


def test_nonfinite_intent_is_not_silently_dropped(bank):
    b=brief({'lemon':float('nan')})
    with pytest.raises(ValueError,match='finite'):
        compile_targets(bank,b,rows(b))


def test_overall_phase_summary_keeps_specific_details(bank):
    b=replace(brief({}),target_profile={'citrus':.5,'woody':.5},
        temporal_emphasis={'opening':.5,'drydown':.5},
        phase_expressions={'opening':{'wanted':{'lemon':1}},'drydown':{'wanted':{'woody':1}}})
    t,m=compile_targets(bank,b,rows(b))
    assert not m and t[0]['reference_weights']=={'lemon':.5,'woody':.5}


def test_prohibition_only_phase_inherits_named_positive_not_a_new_family(bank):
    b=replace(brief({}),phase_expressions={'opening':{'avoided':{'sweet':1}},'drydown':{'wanted':{'woody':1}}})
    t,m=compile_targets(bank,b,[{'phase':'opening','target_profile':{},'avoided':[]}])
    assert not m and t[0]['reference_weights']=={'woody':1.}
    assert t[0]['positive_intent_inherited_for_prohibition_only_phase']


def test_physical_guidance_and_final_readout_use_identical_objective(bank):
    from collections import OrderedDict
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.hierarchical_perfume import PhysicalReferenceSession, evaluate_reference, target_rows
    from fragrance_ai.recommender.science import TemporalMixtureSimulator
    items=IngredientCatalog.load_builtin().ingredients[:3]
    shapes={item.ingredient_id:bank.profiles[name] for item,name in zip(items,('lemon','orange','woody'))}
    shape_session=SimpleNamespace(prefetch=lambda _:None,shape=lambda item:shapes[item.ingredient_id])
    provider=SimpleNamespace(core=SimpleNamespace(sha256='synthetic'),complete_reference_bank=bank,
        begin_reference_shapes=lambda:shape_session)
    b=brief({'citrus':1.})
    b.constraints.simulation_draws=64
    bank.parent_sha256='synthetic'
    intent=target_report(bank,b,target_rows(b))
    # Isolate the mathematical session from unrelated molecular encoders.
    session=object.__new__(PhysicalReferenceSession)
    session.enabled=True;session.properties={};session.functions=OrderedDict();session.shapes=shape_session
    session.q=np.stack([r['profiles'] for r in intent['targets']],axis=1)
    session.mask=np.zeros((6,3));session.tw=np.r_[0.,TemporalMixtureSimulator.time_weights(b)]
    session.brief=b;session.provider=provider;session.bank=bank;session.intent=intent
    session.missing_ids=set();session.exact_calls=0;session.grid_calls=0
    w=np.array([40.,35.,25.])
    guided=session.evaluate(w,items)
    lines=[SimpleNamespace(ingredient_id=i.ingredient_id,concentrate_percent=float(v)) for i,v in zip(items,w)]
    actual=evaluate_reference(provider,b,lines,{i.ingredient_id:i for i in items},{},64)
    assert actual['score']==pytest.approx(guided['score'],abs=1e-12)
    assert actual['intent']['sha256']==guided['target_sha256']
    score,gradient=session.gradient_function(items)(w/100)
    h=1e-7;d=np.array([1.,-1.,0.])
    f=session.gradient_function(items)
    finite=(f(w/100+h*d)[0]-f(w/100-h*d)[0])/(2*h)
    assert gradient@d==pytest.approx(finite,rel=1e-5,abs=1e-5)


def test_final_primary_score_keeps_legacy_score_separate(bank):
    from fragrance_ai.recommender.profile_match import assess_recipe_profiles
    from fragrance_ai.recommender.hierarchical_perfume import target_rows
    b=brief({'citrus':1.});intent=target_report(bank,b,target_rows(b))
    prediction=np.stack([t['profiles'] for t in intent['targets']],axis=1)
    metadata={'version':'hierarchical-perfume-reference/v77','intent':intent,'score':100.,'target_met':True,
        'endpoints':list(bank.endpoints),'predicted_profiles':prediction.tolist(),'aggregation':'synthetic_fixture'}
    points=[{'minutes':t,'phase':p,'scent_profile':{'woody':1.},'full_reference_prediction':metadata if j==0 else None}
        for j,(t,p) in enumerate(zip((0,15,60,240,480),('opening','opening','heart','drydown','drydown')))]
    value=assess_recipe_profiles(b,{'woody':1.},points,[.1,.1,.3,.25,.25])
    assert value['score']==100 and value['legacy_19_axis_assessment']['score']==0
    assert not value['legacy_scores_directly_comparable']
