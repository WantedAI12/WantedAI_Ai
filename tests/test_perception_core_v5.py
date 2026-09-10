from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from fragrance_ai.research.fine_odor_features import catalog_features, append_features, validate_features
from fragrance_ai.research.kernel_profiles import fit_kernel_v5, predict_component_regressor, _kernel_v5
from tests.test_perception_core_v4 import arrays


def fine_fixture():
    from fragrance_ai.recommender.odor_integrity import ODOR_INTEGRITY_VERSION
    ref = json.dumps({'source_tag':'goodscents','source_field':'Descriptors','descriptor':'violet',
        'source_file_sha256':'a'*64,'source_record_id':'1','conditions_status':'unknown',
        'semantic_role':'odor','normalization_version':ODOR_INTEGRITY_VERSION})
    def material(graph, assertions, digest='b'*64):
        return SimpleNamespace(structure_smiles=graph,odor_assertions=assertions,odor_evidence_refs=(ref,),odor_registry_sha256=digest)
    return SimpleNamespace(ingredients=[material('CCO',('goodscents:violet',)),
        material('CCC',('goodscents:violet','goodscents:violet')),material('C',('goodscents:violet',),'unverified')])


def test_features_preserve_fine_evidence_without_synthetic_intensity():
    data = catalog_features(fine_fixture(),minimum_support=2)
    validate_features(data)
    assert data['vocabulary'] == ['violet']
    assert data['by_structure'] == {'CCC':[0],'CCO':[0]}
    assert data['unverified_lineage_rows_omitted'] == 1
    x,_ = arrays()
    extended = append_features(x[:3],['CCC','CCO','missing'],data)
    np.testing.assert_array_equal(extended[:,:1105],x[:3])
    np.testing.assert_array_equal(extended[:,1105:],[[1,1],[1,1],[0,0]])
    changed = deepcopy(data)
    changed['by_structure']['CCO'] = []
    with pytest.raises(ValueError,match='content mismatch'):
        validate_features(changed)


def test_kernel_psd_context_and_scalar_batch_roundtrip():
    x,y = arrays()
    features = catalog_features(fine_fixture(),minimum_support=2)
    x = append_features(x,['CCO','CCC','missing']*4,features)
    model = json.loads(json.dumps(fit_kernel_v5(x,y,.1,fine_weight=.5)))
    kernel = _kernel_v5(x,x,np.asarray(model['scale']),model['kernel_parameters'])
    np.testing.assert_allclose(kernel,kernel.T,atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10
    batch = predict_component_regressor(model,x)
    scalar = np.vstack([predict_component_regressor(model,r[None,:]) for r in x])
    np.testing.assert_allclose(batch,scalar,atol=1e-12)
    assert np.isfinite(batch).all() and np.all(batch >= 0)


@pytest.mark.parametrize('weight',[float('nan'),-.1,1.1])
def test_invalid_v5_weight_fails_closed(weight):
    x,y = arrays()
    x = append_features(x,['CCO']*len(x),catalog_features(fine_fixture(),minimum_support=2))
    with pytest.raises(ValueError):
        fit_kernel_v5(x,y,.1,fine_weight=weight)


def test_actual_v5_runtime_scalar_batch_and_checkpoint_features(monkeypatch):
    from fragrance_ai.recommender.perception_runtime import configured_perception,PATH_ENV,HASH_ENV
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from scripts.verify_odor_concepts_v39 import read_catalog
    root = Path(__file__).resolve().parents[1]
    path = root/'.benchmarks/product_runtime_v45/perfume.json'
    monkeypatch.setenv('PERFUMERY_AI_ENV','development')
    monkeypatch.setenv(PATH_ENV,str(path))
    monkeypatch.setenv(HASH_ENV,hashlib.sha256(path.read_bytes()).hexdigest())
    provider = configured_perception()
    assert provider.component_model_version == 'v5'
    assert provider.model['feature_width'] == 1642
    catalog = read_catalog(root/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json')
    session = provider.begin(NaturalLanguageBriefParser(catalog).parse('citrus woody'))
    materials = [i for i in catalog.ingredients if i.formulation_ready and not i.blocked and session.supports(i)][:5]
    assert len(materials) == 5
    doses = np.array([.001,.01,.1])
    batch = session.predict_stock_batch(materials,doses)
    for item,(prediction,statuses) in zip(materials,batch):
        one,detail = session._predict(item,doses)
        np.testing.assert_allclose(prediction,one,atol=1e-10)
        assert detail == statuses


def test_v5_reports_identical_held_out_groups_without_reading_sealed_outcomes():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/'.benchmarks/perception_core_v5/run-01/report.json').read_text(encoding='utf-8'))
    for fold in range(5):
        train = {g for g,f in zip(report['outer_group_ids'],report['outer_fold_ids']) if f != fold}
        held = {g for g,f in zip(report['outer_group_ids'],report['outer_fold_ids']) if f == fold}
        assert not train & held
    assert report['comparison']['paired_profiles'] == report['predicted_conditions'] == 333
    assert report['all_conditions'] == 335 and report['outcome_read_attempts'] == []
    assert not report['recipe_400_pass_rate_measured']
