"""MX invariants, unit boundaries and portable replay (no sensory proof)."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fragrance_ai.research.physsim_mx import (
    MODES, mx_features, fit_mx, predict_mx, ReleaseEnvironment, release_trajectory)


P = np.array([[1.,.1,0.], [0.,1.,.2]])


@pytest.mark.parametrize('mode', MODES)
def test_order_and_split_identity_do_not_create_interactions(mode):
    expected = mx_features(P, ['a','b'], [.1,.01], ['pg','dep'], [.4,.6], mode=mode)
    permuted = mx_features(P[::-1], ['b','a'], [.01,.1], ['dep','pg'], [.6,.4], mode=mode)
    split = mx_features(P[[0,0,1]], ['a','a','b'], [.1,.1,.01], ['pg','pg','dep'], [.1,.3,.6], mode=mode)
    np.testing.assert_allclose(expected, permuted, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected, split, rtol=0, atol=1e-12)


@pytest.mark.parametrize('mode', MODES)
def test_zero_weight_and_zero_total_create_no_scent(mode):
    expected = mx_features(P[:1], ['a'], [.1], ['pg'], mode=mode)
    with_zero = mx_features(P, ['a','b'], [.1,.01], ['pg','dep'], [1.,0.], mode=mode)
    np.testing.assert_allclose(expected, with_zero, rtol=0, atol=1e-12)
    assert not mx_features(P, ['a','b'], [0.,0.], ['pg','dep'], mode=mode).any()


def test_absolute_dilution_retained_not_replaced_by_relative_proportions():
    a = mx_features(P, ['a','b'], [.01,.001], ['pg','pg'], mode='concentration')
    b = mx_features(P, ['a','b'], [.1,.01], ['pg','pg'], mode='concentration')
    np.testing.assert_array_equal(a[:3], b[:3])
    assert not np.allclose(a[3:], b[3:])


def test_identical_graph_with_inconsistent_semantics_rejected():
    with pytest.raises(ValueError, match='conflicting'):
        mx_features(P, ['a','a'], [.1,.01], ['pg','pg'])


@pytest.mark.parametrize('dilutions', [[float('nan'),.1], [-.1,.1], [2.,.1]])
def test_invalid_dilution_rejected(dilutions):
    with pytest.raises(ValueError, match='input'):
        mx_features(P, ['a','b'], dilutions, ['pg','pg'])


def test_portable_head_replay_blank_and_gas_unit_rejection():
    x = np.array([mx_features(P, ['a','b'], [.01*k,.001*k], ['pg','pg']) for k in (1,2,3,4,5)])
    y = np.array([[.1*k,.2*k] for k in (1,2,3,4,5)])
    model = fit_mx(x, y, 1., 1., mode='interaction')
    restored = json.loads(json.dumps(model))
    np.testing.assert_allclose(predict_mx(model,x), predict_mx(restored,x), rtol=0, atol=1e-12)
    assert not predict_mx(model, np.zeros((1,x.shape[1]))).any()
    with pytest.raises(ValueError, match='domain'):
        predict_mx(model,x,input_domain='air_mol_per_m3')


def test_release_has_nonnegative_mass_and_exhaust_accounting():
    env = ReleaseEnvironment(1e-6,.001,1e-6,1e-5)
    result = release_trajectory([1e-6,2e-6],[.01,.001],[0,1,15,60,240],env)
    n = np.asarray(result['moles'])
    np.testing.assert_allclose(n.sum(2), np.tile([1e-6,2e-6], (5,1)), rtol=0, atol=1e-14)
    assert np.min(n) >= 0 and n[-1,:,2].sum() > 0
    assert result['mass_balance_relative_error'] < 1e-10
    assert result['release_calibrated'] is False


def test_closed_headspace_has_known_analytic_equilibrium():
    env = ReleaseEnvironment(1.,2.,1.,0.)
    result = release_trajectory([3.],[.5],[0.,1000.],env)
    # At equilibrium c_air=.5*c_product, so n_air/n_product = 1.
    np.testing.assert_allclose(result['moles'][-1], [[1.5,1.5,0.]], rtol=0, atol=1e-11)


def test_zero_transfer_and_zero_dose_remain_zero_in_air():
    env = ReleaseEnvironment(1.,2.,0.,.1)
    result = release_trajectory([3.,0.],[.5,1.],[0.,1.,100.],env)
    np.testing.assert_array_equal(np.asarray(result['air_mol_per_m3']), np.zeros((3,2)))


def test_formulation_specific_partition_changes_trajectory():
    env = ReleaseEnvironment(1e-6,.001,1e-6,1e-5)
    a = release_trajectory([1e-6],[.01],[0,1,15,60],env)
    b = release_trajectory([1e-6],[.001],[0,1,15,60],env)
    assert a['air_mol_per_m3'] != b['air_mol_per_m3']


def test_rate_correction_cannot_create_mass_and_zero_matches_prior():
    env = ReleaseEnvironment(1e-6,.001,1e-6,1e-5)
    prior = release_trajectory([1e-6],[.01],[0,1,15],env)
    zero = release_trajectory([1e-6],[.01],[0,1,15],env,log_rate_correction=[0.])
    altered = release_trajectory([1e-6],[.01],[0,1,15],env,log_rate_correction=[1.])
    np.testing.assert_array_equal(prior['moles'],zero['moles'])
    assert altered['mass_balance_relative_error'] < 1e-10


@pytest.mark.parametrize('change', [{'air_volume_m3': 0.}, {'conductance_m3_per_minute': -.1}])
def test_invalid_release_geometry_rejected(change):
    env = replace(ReleaseEnvironment(1.,2.,1.,0.), **change)
    with pytest.raises(ValueError, match='scenario'):
        release_trajectory([1.],[.5],[0.,1.],env)


def test_real_experimental_checkpoint_inference_and_permutation():
    from fragrance_ai.research.mx_runtime import V54MXPredictor
    path = Path(__file__).resolve().parents[1]/'.benchmarks/v54_mx/run-02/model.json'
    predictor = V54MXPredictor(path,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),experimental=True)
    stocks = [{'smiles':'CCO','dilution':.01,'carrier':'pg','relative_aliquot':.4},
              {'smiles':'CC(=O)OCC','dilution':.001,'carrier':'pg','relative_aliquot':.6}]
    a, b = predictor.predict(stocks), predictor.predict(stocks[::-1])
    np.testing.assert_allclose(list(a['predicted_rata'].values()),list(b['predicted_rata'].values()),atol=1e-12,rtol=0)
    assert len(a['predicted_rata']) == 51 and a['mode'] == 'concentration'
    assert a['runtime_promoted'] is False
    with pytest.raises(ValueError,match='units'):
        predictor.predict([{**stocks[0], 'air_mol_per_m3': .01}])
    with pytest.raises(ValueError,match='fragmented'):
        predictor.predict([{**stocks[0], 'smiles': '[Na+].[Cl-]'}])
