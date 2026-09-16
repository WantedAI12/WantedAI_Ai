import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.recommender.reference_observations import ComponentReferenceObservations,VERSION
from fragrance_ai.recommender.lotion_atlas import AtlasLotionShapes
from tests.test_dose_refinement import materials


def make_store(tmp_path,**changes):
    profile=np.zeros((2,146))
    profile[:,0]=.8
    profile[:,1]=.2
    value={'schema':VERSION,'source_ordinal_level':'high','endpoints':[str(i) for i in range(146)],
        'recipe_outcomes_used':False,'product_sensory_calibration':False,
        'records':[{'canonical_smiles':'CCO','profiles':profile.tolist(),'source_stimulus_ids':['source-1']}],**changes}
    path=Path(tmp_path)/'observations.json'
    path.write_text(json.dumps(value),encoding='utf-8')
    return ComponentReferenceObservations(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_source_values_are_preserved_and_cannot_be_mutated(tmp_path):
    store=make_store(tmp_path)
    shape,ids=store.lookup('CCO')
    np.testing.assert_array_equal(shape[:,:2],[[.8,.2]]*2)
    assert ids==('source-1',) and store.lookup('COC') is None
    with pytest.raises(ValueError):
        shape[0,0]=1.
    store.path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        store.assert_current()


@pytest.mark.parametrize('change',[{'source_ordinal_level':'low'},{'recipe_outcomes_used':True},{'product_sensory_calibration':True}])
def test_product_or_recipe_claims_cannot_be_smuggled_into_component_data(tmp_path,change):
    with pytest.raises(ValueError,match='scope'):
        make_store(tmp_path,**change)


def test_exact_observation_precedes_prediction_without_polluting_model_cache(tmp_path):
    store=make_store(tmp_path)
    item=materials()[0]
    shapes=AtlasLotionShapes.__new__(AtlasLotionShapes)
    shapes.observations=store
    shapes.shapes,shapes.basis,shapes.graphs,shapes.missing={},{},{item.ingredient_id:'CCO'},set()
    shapes._graph=lambda value:'CCO'
    shapes.provider=SimpleNamespace(model=SimpleNamespace(native={}),endpoints=store.endpoints)
    # There is deliberately no model cache or predictor. Exact source data
    # must be usable without inventing a prediction or overwriting that cache.
    shapes.prefetch([item])
    np.testing.assert_array_equal(shapes.shape(item),store.lookup('CCO')[0])
    assert shapes.basis[item.ingredient_id]['shape_predicted_by_model'] is False
    report=shapes.observation_summary()
    assert report['observed_materials_used']==1 and report['predicted_materials_used']==0
