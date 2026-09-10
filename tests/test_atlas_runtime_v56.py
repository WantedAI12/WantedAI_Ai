"""Portable old/new checkpoint inference must agree with the training equations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor, predict_atlas, SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def test_actual_transformed_checkpoint_uses_inverse_scale_once():
    path = ROOT/'.benchmarks/quantitative_profiles_v54/run-01/model.json'
    artifact = json.loads(path.read_text(encoding='utf-8'))
    provider = AtlasProfilePredictor(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), experimental=True)
    for name, model in artifact['models'].items():
        inputs = np.asarray(model['support'][:3])
        # Independent reconstruction of the previous training-only prediction.
        raw_model = {**model, 'kind': SCHEMA}
        raw_model.pop('target_transform', None)
        latent = predict_atlas(raw_model, inputs)
        transform = model['target_transform']
        expected = latent**2 if transform == 'sqrt' else np.expm1(latent)
        np.testing.assert_allclose(predict_atlas(provider.models[name], inputs), expected, rtol=0, atol=1e-12)


def test_original_checkpoint_remains_unchanged():
    path = ROOT/'.benchmarks/atlas_profiles_v50/run-01/model.json'
    artifact = json.loads(path.read_text(encoding='utf-8'))
    provider = AtlasProfilePredictor(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), experimental=True)
    for name, model in artifact['models'].items():
        inputs = np.asarray(model['support'][:2])
        np.testing.assert_array_equal(predict_atlas(provider.models[name], inputs), predict_atlas(model, inputs))


@pytest.mark.parametrize('kind,transform', [(SCHEMA, 'sqrt'), ('unknown', 'identity'), ('atlas-quantitative-profiles/v2', 'unknown')])
def test_invalid_transform_is_rejected_before_numerical_prediction(kind, transform):
    with pytest.raises(ValueError, match='transform'):
        predict_atlas({'kind': kind, 'target_transform': transform}, np.ones((1, 1)))
