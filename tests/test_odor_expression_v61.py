"""V61 language/CPU/search contracts; fixtures are NOT sensory benchmarks."""
from dataclasses import replace
import hashlib
import json
from types import MappingProxyType

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from pydantic import BaseModel
import pytest

from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.odor_expression import registry, expression_contract, parse_expression, expression_utility
from fragrance_ai.recommender.fine_odor_model import FineOdorModel, VERSION, structure_features
from fragrance_ai.recommender.global_profile_search import optimize_full_pool
from fragrance_ai.recommender.optimizer import ConstrainedFormulaOptimizer


@pytest.fixture
def parser():
    return NaturalLanguageBriefParser(IngredientCatalog.load_builtin())


def test_registry_counts_synonyms_and_qualities_separately():
    c = expression_contract()
    assert c['canonical_concepts']>=600
    assert c['odor_concepts']>=500 and c['quality_concepts']>=50
    assert c['annotated_concepts']==450 and c['quantitative_backbone_axes']==146
    assert not c['quantitative_axes_changed']
    r = registry()
    assert len(r['aliases'])==sum(len(row['aliases']) for row in r['rows'].values())
    assert 'MIT License' in r['payload']['upstream_license']
    assert r['rows']['lychee']['source_terms']==[]


@pytest.mark.parametrize('text,keys',[
    ('피오니와 청사과 그리고 녹차',{'peony','greenapple','greentea'}),
    ('목련과 무화과',{'magnolia','fig'}),
    ('볶은 아몬드와 생강',{'roastedalmond','ginger'}),
    ('티로즈와 오리스',{'tearose','orris'}),
    ('샌달우드와 베티버',{'sandalwood','vetiver'}),
    ('자몽 껍질과 블랙커런트 새싹',{'grapefruitpeel','blackcurrantbud'}),
    ('편백과 갓 벤 건초',{'cypress','newmownhay'}),
    ('cooked onion and green tea',{'cookedonion','greentea'}),
    ('green apple and roasted almond',{'greenapple','roastedalmond'}),
    ('tea rose and pear skin',{'tearose','pearskin'}),
])
def test_detailed_identity_and_compound_spans(text,keys):
    x = parse_expression(text)
    assert set(x['wanted'])==keys


def test_scopes_relative_weights_and_korean_projection(parser):
    b = parser.parse('피오니와 청사과, 코코넛은 제외')
    assert set(b.expression_targets)=={'peony','greenapple'}
    assert set(b.expression_avoided)=={'coconut'}
    assert b.target_profile['floral']>0 and b.target_profile['fruity']>0
    assert not b.target_profile['gourmand']
    x = parse_expression('90% green apple and 10% peach')
    assert x['wanted']['greenapple']/x['wanted']['peach']==pytest.approx(9.)


def test_phase_exclusions_are_not_global(parser):
    b = parser.parse('opening rose; drydown sandalwood without rose')
    assert not b.expression_avoided and not b.expression_targets
    assert 'rose' in b.phase_expressions['opening']['wanted']
    assert 'rose' in b.phase_expressions['drydown']['avoided']


def test_negative_details_do_not_ban_the_entire_parent(parser):
    b = parser.parse('peach without green apple')
    assert b.expression_targets=={'peach':1.}
    assert b.expression_avoided=={'greenapple':1.}
    assert 'fruity' not in b.avoided_dimensions


def test_unknown_is_not_fabricated_into_a_supported_concept():
    x = parse_expression('zzyxxyz 향')
    assert not x['wanted'] and not x['avoided']
    assert x['unknown_text_is_not_a_zero_odor']
    assert parse_expression('lychee')['matches'][0]['support']=='vocabulary_only'


@pytest.fixture
def tiny_model(tmp_path):
    arrays = {'mean':np.zeros(16),'scale':np.ones(16),'prior':np.array([.1,.2]),
        'w0':np.zeros((1040,4)),'b0':np.zeros(4),
        'w1':np.zeros((4,3)),'b1':np.zeros(3),
        'w2':np.zeros((3,2)),'b2':np.array([-2.,-1.])}
    np.savez(tmp_path/'weights.npz',**arrays)
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    value={'schema':VERSION,'registry_sha256':registry()['sha256'],
        'label_kind':'public_descriptor_annotations_not_measured_absence_or_intensity',
        'annotation_inputs_used':False,'endpoints':['greenapple','peach'],'neural_blend':.75,
        'weights':{'path':'weights.npz','sha256':sha(tmp_path/'weights.npz')},
        'source_annotations':{'CCO':[0],'CCC':[1]},'evaluation_summary':{'fixture':True}}
    path=tmp_path/'model.json'; path.write_text(json.dumps(value))
    return FineOdorModel(path,sha(path))


def test_real_feature_shape_and_stereo_graphs():
    x=structure_features(['CCO','CC[C@H](O)C','CC[C@@H](O)C'])
    assert x.shape==(3,1040) and np.isfinite(x).all()
    assert set(np.unique(x[:,:1024]))<={0.,1.}
    assert np.any(x[1,:1024]!=x[2,:1024])


def test_prediction_never_uses_target_annotations(tiny_model):
    before=tiny_model.predict(['CCO'])
    tiny_model.annotations['CCO']=[1]
    np.testing.assert_array_equal(tiny_model.predict(['CCO']),before)
    assert not tiny_model.contract()['annotation_inputs_used']


def test_cpu_model_hash_guard(tiny_model):
    tiny_model.weights_path.write_bytes(b'drift')
    with pytest.raises(ValueError,match='changed'):
        tiny_model.assert_current()


def test_curated_missing_graph_uses_exact_id_and_cas_binding(tiny_model):
    item=IngredientCatalog.load_builtin().ingredients[0]
    item=replace(item,structure_smiles='')
    tiny_model.structures=MappingProxyType({item.ingredient_id:('CCO',item.cas_number)})
    values,evidence=tiny_model.materials([item])
    assert values[0,0]==1. and evidence[0]['source_positive_labels']==1
    with pytest.raises(ValueError,match='CAS'):
        tiny_model.materials([replace(item,cas_number='invalid')])


def test_disconnected_graph_is_not_treated_as_a_molecule():
    with pytest.raises(ValueError,match='one explicit'):
        structure_features(['CCO.CCC'])


@pytest.fixture
def fine_search(monkeypatch,tiny_model,parser):
    monkeypatch.setattr('fragrance_ai.recommender.fine_odor_model.configured_fine_odor',lambda:tiny_model)
    seed=IngredientCatalog.load_builtin().ingredients[0]
    # Identical coarse profiles, price and note. Only fine source identity
    # differs; no safety or cardinality constraint is weakened.
    a=replace(seed,ingredient_id='a',name='Fixture A',aliases=(),structure_smiles='CCO',
        profile={'fruity':1.},pyramid='heart',max_concentrate_percent=100.,price_per_kg=1.)
    b=replace(a,ingredient_id='b',name='Fixture B',structure_smiles='CCC')
    return [a,b],parser


@pytest.mark.parametrize('text,expected',[('green apple scent','a'),('peach scent','b')])
def test_full_pool_recipe_changes_with_fine_identity_without_losing_coarse_score(fine_search,text,expected):
    items,p=fine_search
    brief=p.parse(text)
    brief=replace(brief,pyramid_ratios={'top':0.,'heart':100.,'base':0.})
    result=optimize_full_pool(items,brief)
    assert result.relaxed_overlap_score==pytest.approx(100.,abs=1e-5)
    assert result.weights_percent.get(expected,0.)>99.


def test_missing_structure_does_not_win_an_avoidance_request(fine_search):
    items,p=fine_search
    missing=replace(items[0],ingredient_id='unknown',structure_smiles='')
    values,report=expression_utility([items[0],missing],p.parse('rose without green apple'))
    assert values[1]<=values[0]
    assert report['missing_identity_avoidance_policy'].startswith('pessimistic')


def test_api_vocabulary_and_interpretation_are_available_without_model():
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    class Formula(BaseModel):
        brief:str
    app=FastAPI()
    register_ai_extensions(app,Formula,IngredientCatalog.load_builtin(),lambda *a:None,lambda:None)
    with TestClient(app) as client:
        result=client.get('/v1/odor-expressions',params={'q':'목련'}).json()
        assert result['total']==1 and result['items'][0]['id']=='magnolia'
        x=client.post('/v1/odor-expressions/interpret',json={'text':'피오니와 청사과'}).json()
        assert set(x['wanted'])=={'peony','greenapple'} and not x['recipe_generated']
        assert client.get('/v1/odor-expressions',params={'limit':10000}).status_code==422
        assert client.post('/v1/odor-expressions/predict',json={'ingredient_ids':['a']}).status_code==503
