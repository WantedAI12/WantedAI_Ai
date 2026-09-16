"""Lotion-owned target interpretation and evaluation policy.

Descriptor overlap is a shared mathematical primitive, not a product model.
Lotion predictions must come from the O/W film transport path; no perfume
evaporation model, phase scheduler or perfume learned projection is imported.
The initial vocabulary and phase boundaries preserve the previous policy,
not a claim that these boundaries or additive shapes were measured in lotion.
"""
from dataclasses import replace
import numpy as np
from .profile_match import compare_profiles as _compare_profiles

LOTION_EVALUATION_VERSION = 'body-lotion-transport-profile/v2'
LOTION_PROJECTION = {
    'citrus': ('Citrus',), 'green': ('Green',), 'floral': ('Floral',),
    'fruity': ('Fruity', 'Tropical', 'Berry', 'Peach'), 'spicy': ('BrownSpice',),
    'aromatic': ('Herbal', 'Mint', 'Pine'), 'woody': ('Woody',),
    'gourmand': ('Sweet', 'Caramellic', 'Vanilla'), 'powdery': ('Powdery',),
    'smoky': ('Smoky',), 'earthy': ('Earthy',),
}


def phase_for_time(minutes, schedule=None):
    opening = 15. if schedule is None else schedule.opening_until_minutes
    heart = 240. if schedule is None else schedule.heart_until_minutes
    return 'opening' if minutes <= opening else 'heart' if minutes <= heart else 'drydown'


def compare_lotion_profiles(target, predicted, *, avoided=()):
    """Compare the lotion transport output, never a perfume recipe prediction."""
    return replace(_compare_profiles(target, predicted, avoided=avoided),
                   version=LOTION_EVALUATION_VERSION,
                   score_kind='lotion_airborne_profile_agreement_not_human_similarity')


def refinement_score_floors(scores, target_score=95.):
    """Preserve the worst score, every failed point and every attained target.

    Above-target headroom at non-worst timepoints may be used for a better
    learned profile. This is a search guard, not a different acceptance score.
    For [94, 99] at target 95 the floors are [94, 95]; for [96, 99], [96, 96].
    """
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 100+1e-7)
            or isinstance(target_score, bool) or not np.isfinite(target_score)
            or not 90 <= target_score <= 100):
        raise ValueError('finite profile scores and explicit 90..100 target required')
    return np.minimum(values, max(float(values.min()), target_score))


def evaluation_contract():
    return {
        'product': 'body_lotion', 'evaluation_version': LOTION_EVALUATION_VERSION,
        'prediction_model': 'finite_dose_ow_film_transport',
        'target_basis': 'time_resolved_post_application_airborne_profile',
        'aggregation': 'minimum_over_requested_timepoints_and_transport_scenarios',
        'profile_representation': 'shared_catalog_19_axis_additive_prior',
        'phase_schedule_basis': 'explicit_request_or_lotion_owned_legacy_defaults_not_calibration',
        'matrix_calibrated': False, 'human_similarity_percent': None,
        'cross_product_scores_comparable': False,
    }
