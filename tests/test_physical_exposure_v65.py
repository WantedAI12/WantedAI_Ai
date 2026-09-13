"""Independent analytic/ODE checks, not measurements of human similarity."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from fragrance_ai.platform.lotion_inputs import LotionSimulationRequest, LotionOptimizationRequest
from fragrance_ai.recommender.lotion import _simulate_lotion_transport, EXPOSURE_INTEGRATION_VERSION
from fragrance_ai.recommender.lotion_transport import bidirectional_step
from fragrance_ai.recommender.lotion_basis_cache import reuse_basis
from fragrance_ai.recommender.runtime_cache import InferenceCache
from fragrance_ai.recommender.lotion_optimizer import _optimize_lotion_transport
from fragrance_ai.recommender.unified_transport import trajectory, baseline_kernel, validate_raw
from tests.test_lotion import data
from tests.test_lotion_v21 import fixture
from tests.test_lotion_reference_objective_v59 import bank as bank, fixture_predictor
from tests.test_unified_product_v60 import model as model, make_predictor, request_payload
from fragrance_ai.platform.unified_product_inputs import UnifiedProductRequest
from fragrance_ai.recommender.catalog import IngredientCatalog


def simulate(value, windows, catalog=None):
    return _simulate_lotion_transport(LotionSimulationRequest.model_validate(value),
        catalog or IngredientCatalog.load_builtin(), exposure_windows=windows)


@pytest.mark.parametrize('rates', [(0., 0., 0., 0.), (1., 0., 0., 1.),
    (.3, .1, .4, 0.), (1e4, 1e-4, .1, .001), (1e-10, 0., 1e-10, 1e-10)])
def test_augmented_integrals_match_independent_matrix_exponential(rates):
    e, u, r, v = rates
    # Last two states integrate film and air directly, independent of sinks.
    matrix = np.array([[-e-u,r,0,0,0,0], [e,-r-v,0,0,0,0],
        [u,0,0,0,0,0], [0,v,0,0,0,0], [1,0,0,0,0,0], [0,1,0,0,0,0]])
    expected = expm(matrix*.7) @ [.02,.003,0.,0.,0.,0.]
    actual = np.array([x[0] for x in bidirectional_step(np.array([.02]), np.array([.003]),
        *[np.array([x]) for x in rates], .7, return_integrals=True)])
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-14)
    assert np.all(actual >= 0)
    assert actual[:4].sum() == pytest.approx(.023, abs=2e-14)


def rapid_fixture():
    value, catalog = fixture()
    s = value['simulation']
    s.update(transport_mode='open_sink', times_minutes=[0.,15.,60.,240.,480.],
             air_exchange_per_min=1.)
    capacity = .00198 * (0.8 + 10*.2)
    for m, e in zip(s['materials'], [1., .01]):
        m.update(concentrate_percent=50., skin_permeability_cm_min=0.,
                 air_water_partition=e*capacity/m['gas_transfer_cm_min'])
    return s, catalog


def test_early_evaporation_not_lost_between_display_times():
    s, catalog = rapid_fixture()
    result = simulate(s, [(0.,480.)], catalog)
    integral = np.array([r['air_exposure_mg_min_m3'] for r in result['exposure_windows'][0]['materials']])
    # Analytic sequential F -> A -> exhaust, v=1/min, initial F=.01.
    t = 480.
    film = .01*np.exp(-np.array([1.,.01])*t)
    air = np.array([.01*t*np.exp(-t), .01*.01*(np.exp(-.01*t)-np.exp(-t))/.99])
    expected = (.01-film-air)*1e6
    np.testing.assert_allclose(integral, expected, rtol=2e-12)
    sparse = np.trapezoid([[r['air_concentration_mg_m3'] for r in p['materials']]
        for p in result['temporal_profile']], s['times_minutes'], axis=0)
    assert sparse[0]/integral[0] < .0002
    assert integral[0]/integral.sum() == pytest.approx(.5020869, abs=1e-7)
    assert result['diagnostics']['mass_balance_max_abs_error_mg_cm2'] < 1e-14


@pytest.mark.parametrize('mode', ['open_sink', 'bidirectional_air'])
def test_observation_grid_and_horizon_do_not_change_existing_states(mode):
    s = data()
    s.update(transport_mode=mode, water_loss_per_min=.07,
        air_exchange_per_min=.02, times_minutes=[0.,3.07,60.,79.17,120.])
    windows = [(3.07,79.17), (60.,120.)]
    a = simulate(s, windows)
    s['times_minutes'] = [0.,.031,3.07,4.113,16.009,60.,60.003,79.17,100.231,120.]
    b = simulate(s, windows)
    by_time = {p['minutes']:p for p in b['temporal_profile']}
    for point in a['temporal_profile']:
        assert point['materials'] == by_time[point['minutes']]['materials']
    assert a['exposure_windows'] == b['exposure_windows']
    s['times_minutes'] = [0.,3.07,79.17]
    short = simulate(s, [(3.07,79.17)])
    assert short['temporal_profile'][-1]['materials'] == by_time[79.17]['materials']
    assert short['exposure_windows'][0] == a['exposure_windows'][0]


def test_drying_reaction_back_transfer_integral_converges_against_independent_ode():
    s = data()
    s.update(transport_mode='bidirectional_air', water_loss_per_min=.03,
             air_exchange_per_min=.02, times_minutes=[0.,15.,60.,120.])
    s['materials'][0].update(initial_parent_fraction=.7, aqueous_hydrolysis_per_min=.02,
                             reaction_source_reference='synthetic independent ODE fixture')
    def rhs(t, y):
        aqueous = .00198*.8*(.1+.9*np.exp(-.03*t))
        capacity = aqueous+.00198*10*.2
        e = u = .00001/capacity
        h = .02*aqueous/capacity
        flux = e*y[0]-.01*y[1]
        return [-flux-(u+h)*y[0], flux-.02*y[1], u*y[0], .02*y[1], h*y[0], y[1]]
    ode = solve_ivp(rhs, [0.,120.], [.014,0.,0.,0.,.006,0.],
                    rtol=2e-12, atol=1e-15, dense_output=True)
    expected_integral = (ode.sol(79.17)[5]-ode.sol(3.07)[5])*1e6
    errors = []
    for h in [.5,.25,.125]:
        s['integration_step_minutes'] = h
        result = simulate(s, [(3.07,79.17)])
        last = result['temporal_profile'][-1]['materials'][0]
        actual = [last[k] for k in ('remaining_mg_cm2','headspace_mg_cm2','skin_sink_mg_cm2',
                                    'ventilated_mg_cm2','degraded_parent_equivalent_mg_cm2')]
        integral = result['exposure_windows'][0]['materials'][0]['air_exposure_mg_min_m3']
        errors.append((abs(integral-expected_integral), np.max(np.abs(actual-ode.y[:5,-1]))))
        assert np.all(np.array(actual) >= 0)
        assert sum(actual) == pytest.approx(.02, abs=5e-15)
        assert result['diagnostics']['mass_balance_max_abs_error_mg_cm2'] < 5e-15
    assert errors[1][0] < errors[0][0]/3.8 and errors[2][0] < errors[1][0]/3.8
    assert errors[2][1] < 1e-9


def test_late_window_keeps_small_exposure_after_lifetime_sink_has_rounded_off():
    s, catalog = rapid_fixture()
    # Equal first material e=v=1. Exhaust is ~initial by 60 min, but air>0.
    result = simulate(s, [(60.,61.)], catalog)
    actual = result['exposure_windows'][0]['materials'][0]['air_exposure_mg_min_m3']
    expected = .01*((60.+1)*np.exp(-60.)-(61.+1)*np.exp(-61.))*1e6
    assert actual > 0
    assert actual == pytest.approx(expected, rel=2e-12)
    points = simulate(dict(s, times_minutes=[0.,60.,61.]), [], catalog)['temporal_profile']
    assert points[1]['materials'][0]['ventilated_mg_cm2'] == points[2]['materials'][0]['ventilated_mg_cm2']


@pytest.mark.parametrize('windows', [[(0,0)], [(-1,3)], [(3,121)], [(0,float('nan'))], [(0,1)]*4])
def test_invalid_exposure_windows_rejected(windows):
    with pytest.raises(ValueError, match='exposure windows'):
        simulate(data(), windows)


@pytest.mark.parametrize('defect', ['negative', 'nonfinite', 'mass_leak'])
def test_invalid_forward_result_cannot_be_used_as_a_scored_simulation(monkeypatch, defect):
    def broken(*args, **kwargs):
        values = list(bidirectional_step(*args, **kwargs))
        if defect == 'negative':
            values[1] = np.full_like(values[1], -.01)
        elif defect == 'nonfinite':
            values[1] = np.full_like(values[1], np.nan)
        else:
            values[0] = values[0]*.99
        return tuple(values)
    monkeypatch.setattr('fragrance_ai.recommender.lotion.bidirectional_step', broken)
    with pytest.raises(ValueError, match='transport failed'):
        simulate(dict(data(), times_minutes=[0.,.25]), [(0.,.25)])


def test_basis_cache_keys_include_windows_not_only_display_times(monkeypatch):
    monkeypatch.setattr('fragrance_ai.recommender.lotion_basis_cache._CACHE', InferenceCache())
    request = LotionSimulationRequest.model_validate(data())
    catalog = IngredientCatalog.load_builtin()
    calls = []
    def compute():
        calls.append(1)
        return {'value':len(calls)}
    _, a = reuse_basis([request], catalog.ingredients[:1], compute, exposure_windows=[(0.,15.)])
    _, b = reuse_basis([request], catalog.ingredients[:1], compute, exposure_windows=[(0.,15.)])
    _, c = reuse_basis([request], catalog.ingredients[:1], compute, exposure_windows=[(1.,15.)])
    assert (a,b,c) == ('miss','hit','miss') and len(calls) == 2


def test_full_mixture_exposure_equals_linear_basis_with_chemistry():
    value, catalog = fixture()
    s = value['simulation']
    s.update(water_loss_per_min=.03, air_exchange_per_min=.02)
    s['materials'][1].update(initial_parent_fraction=.8, aqueous_hydrolysis_per_min=.02,
                            reaction_source_reference='synthetic kinetic fixture')
    windows = [(3.07,15.13),(15.13,60.)]
    baseline = simulate(s, windows, catalog)
    old_weights = np.array([m['concentrate_percent']/100 for m in s['materials']])
    new_weights = np.array([.23,.77])
    for m,w in zip(s['materials'],new_weights):
        m['concentrate_percent'] = 100*w
    actual = simulate(s, windows, catalog)
    for a,b in zip(actual['exposure_windows'], baseline['exposure_windows']):
        expected = np.array([m['air_exposure_mg_min_m3'] for m in b['materials']])/old_weights*new_weights
        np.testing.assert_allclose([m['air_exposure_mg_min_m3'] for m in a['materials']], expected, rtol=2e-13)


def test_observed_optimizer_and_final_formula_use_same_analytic_exposure(bank, monkeypatch):
    value, catalog = fixture()
    value.update(brief='citrus woody scent', evaluation_mode='observed_reference')
    for m in value['simulation']['materials']:
        m['odor_threshold_mg_m3'] = 1e-8
    value['simulation']['materials'][1]['lipid_water_partition'] = 100.
    monkeypatch.setattr('fragrance_ai.recommender.lotion_reference_objective.configured_reference', lambda r,p: bank)
    kwargs = dict(_shape_predictor=fixture_predictor(bank,catalog), reuse_transport_basis=True)
    first = _optimize_lotion_transport(LotionOptimizationRequest.model_validate(value), catalog, **kwargs)
    second = _optimize_lotion_transport(LotionOptimizationRequest.model_validate(value), catalog, **kwargs)
    assert first['profile_target_met'] and first['score'] >= 95
    assert second['score'] == first['score'] and second['recipe'] == first['recipe']
    assert second['basis_cache_status'] == 'hit'
    assert first['transport_simulation_calls'] == 2 and second['transport_simulation_calls'] == 1
    assert first['perceptual_evaluation']['exposure_integration_version'] == EXPOSURE_INTEGRATION_VERSION
    assert all(not t['integration_weights_used'] for t in first['perceptual_evaluation']['targets'])
    values = np.array([m['mean_odor_activity_proxy'] for m in first['simulation']['exposure_windows'][0]['materials']])
    assert first['timepoint_assessments'][0]['total_odor_activity_proxy'] == pytest.approx(values.sum())
    assert first['human_similarity_percent'] is None


def test_extreme_drying_stays_inside_transition_domain_without_coefficient_clipping():
    rates, fractions = np.array([[1.,0.,0.,.5,1.,10.]]), np.array([[.9,.1]])
    seen = []
    def checked(raw):
        seen.append(validate_raw(raw))
        return baseline_kernel(raw)
    a, _ = trajectory(rates, fractions, [0.,15.,480.], operator=checked)
    b, _ = trajectory(rates, fractions, [0.,.07,15.,71.,480.], operator=checked)
    np.testing.assert_allclose(a,b[[0,2,4]],atol=1e-14)
    np.testing.assert_allclose(a.sum(axis=2),1.,atol=1e-13)
    assert np.all(a >= 0) and np.all(np.diff(a[:,:,2:],axis=0) >= 0)
    assert max(raw[:,5].max() for raw in seen) <= 200.


def test_domain_refinement_rejects_unbounded_work_not_fake_acceptance():
    with pytest.raises(ValueError, match='work budget'):
        trajectory([[1.,0.,0.,.5,1.,1e10]], [[.9,.1]], [0.,480.])


def test_domain_refinement_budget_counts_materials_not_only_time_cells():
    with pytest.raises(ValueError, match='work budget'):
        trajectory(np.tile([1.,0.,0.,.5,1.,200.], (1500,1)),
                   np.tile([.9,.1], (1500,1)), [0.,1440.])


def test_shared_budget_is_consumed_across_multiple_stages():
    budget = {'remaining_material_transitions': 60}
    rates, fractions = [[.1,0.,0.,.1,.1,.01]], [[.5,.2]]
    trajectory(rates, fractions, [0.,1.], work_budget=budget)
    assert budget['remaining_material_transitions'] == 26
    with pytest.raises(ValueError, match='multi-stage'):
        trajectory(rates, fractions, [0.,1.], work_budget=budget)
    assert budget['remaining_material_transitions'] == 26


def test_product_pipeline_shares_one_budget_for_all_stages(model, monkeypatch):
    from fragrance_ai.recommender.exposure_transport import exposure_trajectory
    seen = []
    def counted(*args, **kwargs):
        budget = kwargs['work_budget']
        seen.append((id(budget), budget['remaining_material_transitions']))
        return exposure_trajectory(*args, **kwargs)
    monkeypatch.setattr('fragrance_ai.recommender.unified_product.exposure_trajectory', counted)
    result = make_predictor(model).predict(UnifiedProductRequest(**request_payload()))
    assert len(seen) == 2 and seen[0][0] == seen[1][0]
    assert seen[1][1] < seen[0][1]
    assert result['diagnostics']['transport_material_transitions'] > 0
