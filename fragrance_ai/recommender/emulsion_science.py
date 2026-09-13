"""Dimensionless emulsion inputs with explicit experimental applicability.

Auxiliary measured domain: Rodgers 2025, doi:10.48420/30178552 (CC BY 4.0).
These stirred-tank, steady-state silicone-oil dispersions are not a validated
cosmetic lotion, production-scale shear process, or fragrance release assay.
"""
from __future__ import annotations

import numpy as np

REFERENCE = {
    'doi': '10.48420/30178552', 'creator': 'Thomas Rodgers',
    'url': 'https://research.manchester.ac.uk/en/datasets/emulsification-drop-size-distribution-dataset/',
    'license': 'CC BY 4.0', 'impeller_diameter_m': .0483,
    'vessel_diameter_m': .137, 'baffle_width_m': .0137,
    'emulsification_time_h': 24.,
    'geometry': 'six_45_degree_pitched_blades_four_baffles_clearance_T_over_3',
    'material_domain': 'silicone_oil_in_SLES_or_glucose_SLES_or_water_ethanol',
}
FIELDS = ('speed_rpm', 'dispersed_viscosity_pa_s', 'dispersed_density_kg_m3',
          'continuous_viscosity_pa_s', 'continuous_density_kg_m3',
          'interfacial_tension_mn_m', 'dispersed_volume_fraction')


def validate_raw(raw):
    x = np.asarray(raw, np.float64)
    if (x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all()
            or np.any(x <= 0) or np.any(x[:, 6] >= 1)):
        raise ValueError('positive SI-related emulsion inputs and 0 < volume fraction < 1 required')
    return x


def features(raw, *, impeller_diameter_m=REFERENCE['impeller_diameter_m']):
    x = validate_raw(raw)
    if not np.isfinite(impeller_diameter_m) or impeller_diameter_m <= 0:
        raise ValueError('positive impeller diameter required')
    n, mud, rhod, muc, rhoc, sigma, phi = x.T
    n, sigma, d = n / 60., sigma / 1000., impeller_diameter_m
    re = rhoc*n*d*d/muc
    we = rhoc*n*n*d*d*d/sigma
    # Oh is derived, not an additional independent physical variable.
    oh = muc / np.sqrt(rhoc*sigma*d)
    return np.column_stack((np.log(re), np.log(we), np.log(mud/muc),
                            np.log(rhod/rhoc), np.log(phi/(1-phi)), np.log(oh)))


def normal_log_bins(log_diameters, mean, sigma):
    sigma = np.maximum(np.asarray(sigma), .08)
    logp = -.5*((np.asarray(log_diameters)[None] - np.asarray(mean)[:, None])/sigma[:, None])**2
    logp -= logp.max(1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(1, keepdims=True))
    return logp


def baseline(raw, specification):
    x = features(raw)
    z = (x - specification['feature_mean']) / specification['feature_scale']
    design = np.column_stack((np.ones(len(z)), z))
    estimates = design @ np.asarray(specification['lognormal_coefficients'])
    return normal_log_bins(np.log(specification['diameter_bins_um']), estimates[:, 0], np.exp(estimates[:, 1]))


def context(raw, specification):
    result = np.zeros((len(raw), 64), np.float32)
    result[:, 1] = 1.
    result[:, 11] = 1.  # measured-emulsion auxiliary task, never odor/transport
    result[:, 33:39] = (features(raw) - specification['feature_mean']) / specification['feature_scale']
    return result


def report(raw, probabilities, specification):
    raw = validate_raw(raw)
    d = np.asarray(specification['diameter_bins_um'])
    p = np.asarray(probabilities)
    if p.shape != (len(raw), len(d)) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('finite nonnegative drop volume distribution required')
    if not np.allclose(p.sum(-1), 1., atol=1e-6):
        raise ValueError('drop volume distribution must sum to one')
    lo, hi = np.asarray(specification['raw_min']), np.asarray(specification['raw_max'])
    return [{'volume_mean_diameter_um': float(q @ d),
             'geometric_mean_diameter_um': float(np.exp(q @ np.log(d))),
             'volume_fractions': q.tolist(), 'diameter_bins_um': d.tolist(),
             'input_within_training_box': bool(np.all((row >= lo) & (row <= hi))),
             'domain': REFERENCE, 'lotion_product_validation': False,
             'source': 'measured_emulsification_auxiliary_not_fragrance_assay'} for row,q in zip(raw,p)]
