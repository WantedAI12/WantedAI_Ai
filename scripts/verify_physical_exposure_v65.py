"""Numerical verification of in-house transport; not sensory validation."""
import hashlib
import json
from pathlib import Path
import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from tests.test_physical_exposure_v65 import rapid_fixture, simulate
    from tests.test_lotion import data
    from fragrance_ai.recommender.lotion import EXPOSURE_INTEGRATION_VERSION
    s, catalog = rapid_fixture()
    result = simulate(s, [(0.,480.)], catalog)
    integral = np.array([r['air_exposure_mg_min_m3'] for r in result['exposure_windows'][0]['materials']])
    sparse = np.trapezoid([[r['air_concentration_mg_m3'] for r in p['materials']]
        for p in result['temporal_profile']], s['times_minutes'], axis=0)
    expected = np.array([1., 1.-np.exp(-4.8)/.99])*10000.
    np.testing.assert_allclose(integral, expected, rtol=2e-12)
    report = {'scope': 'synthetic_coefficient_numerical_verification_not_human_accuracy',
        'integration_version': EXPOSURE_INTEGRATION_VERSION,
        'rapid_evaporation': {'times_minutes': s['times_minutes'], 'evaporation_per_min': [1.,.01],
            'ventilation_per_min': 1., 'initial_mass_mg_cm2': [.01,.01],
            'old_display_trapezoid_mg_min_m3': sparse.tolist(),
            'analytic_air_integral_mg_min_m3': integral.tolist(),
            'independent_exact_mg_min_m3': expected.tolist(),
            'relative_error_max': float(np.max(np.abs(integral-expected)/expected)),
            'fast_material_old_fraction': float(sparse[0]/sparse.sum()),
            'fast_material_analytic_fraction': float(integral[0]/integral.sum())},
        'human_similarity_percent': None, 'new_empirical_measurements': 0}
    s = data()
    s.update(transport_mode='bidirectional_air', water_loss_per_min=.03,
             air_exchange_per_min=.02, times_minutes=[0.,15.,60.,120.])
    s['materials'][0].update(initial_parent_fraction=.7, aqueous_hydrolysis_per_min=.02,
                             reaction_source_reference='synthetic independent ODE fixture')
    def rhs(t,y):
        aqueous = .00198*.8*(.1+.9*np.exp(-.03*t))
        capacity = aqueous+.00198*10*.2
        e = u = .00001/capacity
        h = .02*aqueous/capacity
        flux = e*y[0]-.01*y[1]
        return [-flux-(u+h)*y[0], flux-.02*y[1], u*y[0], .02*y[1], h*y[0], y[1]]
    ode = solve_ivp(rhs,[0.,120.],[.014,0.,0.,0.,.006,0.],rtol=2e-12,atol=1e-15,dense_output=True)
    expected = (ode.sol(79.17)[5]-ode.sol(3.07)[5])*1e6
    rows = []
    for h in [.5,.25,.125]:
        s['integration_step_minutes'] = h
        result = simulate(s,[(3.07,79.17)])
        integral = result['exposure_windows'][0]['materials'][0]['air_exposure_mg_min_m3']
        last = result['temporal_profile'][-1]['materials'][0]
        state = [last[k] for k in ('remaining_mg_cm2','headspace_mg_cm2','skin_sink_mg_cm2',
                                  'ventilated_mg_cm2','degraded_parent_equivalent_mg_cm2')]
        rows.append({'step_minutes':h, 'air_exposure_abs_error_mg_min_m3':abs(integral-expected),
            'final_state_max_abs_error_mg_cm2':float(np.max(np.abs(state-ode.y[:5,-1]))),
            'mass_balance_max_abs_error_mg_cm2':result['diagnostics']['mass_balance_max_abs_error_mg_cm2']})
    report['drying_hydrolysis_ode_convergence'] = {'window_minutes':[3.07,79.17],
        'reference_solver':'solve_ivp_RK45_rtol_2e-12_atol_1e-15',
        'reference_air_exposure_mg_min_m3':float(expected), 'rows':rows,
        'observed_orders':[float(np.log2(a['air_exposure_abs_error_mg_min_m3']/b['air_exposure_abs_error_mg_min_m3']))
            for a,b in zip(rows,rows[1:])]}
    report['source_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('fragrance_ai/recommender/lotion.py','fragrance_ai/recommender/lotion_transport.py',
                     'fragrance_ai/recommender/unified_transport.py','fragrance_ai/recommender/unified_product.py',
                     'scripts/verify_physical_exposure_v65.py')}
    (args.output/'numerical-report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
