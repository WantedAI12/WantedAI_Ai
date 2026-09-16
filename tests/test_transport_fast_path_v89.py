"""The optimized inference path must reproduce the unchanged full network."""
import numpy as np
import pytest

from fragrance_ai.recommender.formulation_core import (
    FormulationCore, forward_arrays, transport_forward_arrays, transport_distribution)
from fragrance_ai.recommender.unified_transport import features, baseline_kernel


@pytest.fixture(scope='module')
def arrays():
    rng = np.random.default_rng(571)
    a = {}
    def linear(name, source, target):
        a[name+'.weight'] = rng.normal(0., .05, (target, source)).astype(np.float32)
        a[name+'.bias'] = rng.normal(0., .01, target).astype(np.float32)
    linear('molecule_in', 17, 64)
    linear('molecule_out', 64, 128)
    linear('context_in', 64, 96)
    linear('attention_key', 128, 64)
    linear('attention_query', 96, 64)
    linear('process_input', 36, 96)
    linear('process_state', 96, 96)
    a['step_embedding.weight'] = rng.normal(size=(4,24)).astype(np.float32)
    linear('fusion', 3*128+96+96, 192)
    linear('shared_in', 192, 128)
    for index in range(2):
        a[f'shared.{index}.norm.weight'] = np.ones(128, np.float32)
        a[f'shared.{index}.norm.bias'] = np.zeros(128, np.float32)
        linear(f'shared.{index}.up', 128, 256)
        linear(f'shared.{index}.down', 256, 128)
    for name, width in {'fine':450,'quantitative':292,'transport':10,'action':3,
                        'check':3,'revision':19,'emulsion':4,'blend':19,'aqueous':2}.items():
        linear(name+'_head', 128, width)
    a['transport_mean'], a['transport_scale'] = np.zeros(8, np.float32), np.ones(8, np.float32)
    a['feature_mean'], a['feature_scale'] = np.zeros(17, np.float32), np.ones(17, np.float32)
    return a


def full_transport(a, context, molecules=None):
    b = len(context)
    molecules = np.zeros((b,1,17), np.float32) if molecules is None else molecules
    return forward_arrays(a, molecules, np.zeros(molecules.shape[:2], np.float32), context,
                          np.zeros((b,0), np.int64), np.zeros((b,0,12), np.float32))['transport']


@pytest.mark.parametrize('batch', [0,1,7,128,1049])
def test_transport_logits_are_bitwise_equal_to_full_forward(arrays, batch):
    c = np.random.default_rng(batch).normal(size=(batch,64)).astype(np.float32)
    c[:,12] = 1.
    np.testing.assert_array_equal(transport_forward_arrays(arrays,c), full_transport(arrays,c))


def test_molecular_features_cannot_affect_zero_mass_transport(arrays):
    rng = np.random.default_rng(47)
    c = rng.normal(size=(8,64)).astype(np.float32)
    c[:,12] = 1.
    molecules = rng.normal(size=(8,5,17)).astype(np.float32)
    np.testing.assert_array_equal(transport_forward_arrays(arrays,c), full_transport(arrays,c,molecules))


def core(a):
    model = object.__new__(FormulationCore)
    model.arrays, model.feature_width = a, 17
    model.actions = []
    return model


@pytest.mark.parametrize('converted', [('attention_query.weight',), ('feature_mean','feature_scale'),
                                     ('context_in.weight',), ('all',)])
def test_mixed_precision_checkpoints_keep_original_kernel_values(arrays, converted):
    a = {key: value.astype(np.float64) if converted == ('all',) or key in converted else value
         for key, value in arrays.items()}
    model = core(a)
    model.assert_current = lambda: None
    raw = np.array([[.1,.01,.002,.2,1.,.1,.7,.1], [.3,.1,.002,.1,2.,.4,.6,.2]])
    c = np.zeros((len(raw),64), np.float32)
    c[:,4:12], c[:,12] = features(raw), 1.
    expected = transport_distribution(model.forward(np.zeros((len(raw),1,17)), np.zeros((len(raw),1)), c)['transport'],raw)
    np.testing.assert_array_equal(model.kernel(raw),expected)


def test_kernel_keeps_distribution_and_integrity_checks_without_full_forward(arrays):
    raw = np.array([[.1,.01,.002,.2,1.,.1,.7,.1], [.3,.1,0.,0.,0.,0.,.6,.2]])
    c = np.zeros((len(raw),64), np.float32)
    c[:,4:12], c[:,12] = features(raw), 1.
    expected = transport_distribution(full_transport(arrays,c),raw)
    model, checks = core(arrays), []
    model.assert_current = lambda: checks.append('checked')
    model.forward = lambda *a, **k: pytest.fail('unused full-network forward was executed')
    actual = model.kernel(raw)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[1:],baseline_kernel(raw[1:]))
    assert checks == ['checked','checked']
    assert np.all(actual >= 0)
    np.testing.assert_allclose(actual.sum(-1),1.,atol=1e-15)


@pytest.mark.parametrize('invalid', ['shape','nan','other_head'])
def test_fast_path_rejects_other_contexts(arrays,invalid):
    c = np.zeros((2,64), np.float32)
    c[:,12] = 1.
    if invalid == 'shape': c = c[:,:20]
    elif invalid == 'nan': c[0,4] = np.nan
    else: c[0,12] = 2.
    with pytest.raises(ValueError):
        transport_forward_arrays(arrays,c)


def test_original_memory_budget_and_changed_model_guard_remain(arrays):
    model = core(arrays)
    model.assert_current = lambda: None
    model.feature_width = 32_000_001
    with pytest.raises(ValueError, match='memory work budget'):
        model.kernel(np.array([[.1,.01,.002,.2,1.,.1,.7,.1]]))
    model.feature_width = 17
    def changed():
        raise ValueError('checkpoint changed')
    model.assert_current = changed
    with pytest.raises(ValueError, match='checkpoint changed'):
        model.kernel(np.array([[.1,.01,.002,.2,1.,.1,.7,.1]]))
