"""Project the common regulatory contract onto every lotion response path.

Only the fragrance component is screened. Base excipients, microbial quality,
preservative challenge testing, and finished-product approval are not inferred.
"""
import math

from .regulatory_status import regulatory_summary


def attach_lotion_regulatory(payload, request_data, catalog, operation):
    result = dict(payload)
    lines = payload.get('recipe') or payload.get('closest_candidate') or []
    subject = 'returned_recipe' if payload.get('recipe') else 'closest_candidate_only' if lines else 'no_formula'
    simulation = request_data.get('simulation', request_data)
    context = simulation.get('application_context', {})
    concentration = request_data.get('fragrance_concentration_percent', context.get('fragrance_concentration_percent'))
    if not lines and operation in ('simulate', 'trained-release'):
        lines = simulation.get('materials') or []
        subject = 'input_formula' if lines else 'no_formula'
    if any(not math.isfinite(float(row['concentrate_percent'])) or not 0 < float(row['concentrate_percent']) <= 100 for row in lines):
        raise ValueError('invalid lotion regulatory formula amount')
    if lines and all('finished_product_percent' in row for row in lines):
        doses = [float(row['finished_product_percent']) * 100. / float(row['concentrate_percent']) for row in lines]
        if not all(math.isfinite(value) and value > 0 for value in doses) or max(doses)-min(doses) > 1e-7:
            raise ValueError('lotion output contains inconsistent finished-product dose')
        # Dose trials may select a different dose from the original request.
        concentration = sum(doses) / len(doses)
    if lines and (not isinstance(concentration, (int, float)) or not math.isfinite(concentration) or not 0 < concentration <= 100):
        raise ValueError('lotion regulatory screen has no valid finished-product dose')
    by_id = {item.ingredient_id: item for item in catalog.ingredients}
    normalized = []
    for row in lines:
        identifier = row['ingredient_id']
        if identifier not in by_id:
            raise ValueError('unknown lotion regulatory material identity')
        amount = float(row['concentrate_percent'])
        if not math.isfinite(amount) or not 0 < amount <= 100:
            raise ValueError('invalid lotion regulatory formula amount')
        normalized.append({'ingredient_id': identifier, 'cas_number': by_id[identifier].cas_number,
                           'concentrate_percent': amount})
    if normalized and (len({r['ingredient_id'] for r in normalized}) != len(normalized)
                       or abs(sum(r['concentrate_percent'] for r in normalized)-100.) > .001):
        raise ValueError('lotion regulatory formula must be unique and sum to 100 percent')
    shell = {'brief': {'constraints': {'product_category': 'body_lotion',
             'target_region': request_data.get('target_region') or 'unspecified',
             'product_concentration_percent': concentration}},
        'recipe': normalized if subject in ('returned_recipe', 'input_formula') else [],
        'closest_candidate': normalized if subject == 'closest_candidate_only' else [],
        'safety': {'internal_gate_passed': False, 'violations': [],
            'missing_documents': ['lotion_base_and_finished_product_safety_review_not_performed_by_this_fragrance_screen']}}
    regulatory = regulatory_summary(shell)
    regulatory.update(subject=subject, scope='lotion_fragrance_component_only_not_complete_base_or_product_review',
        dose_basis='returned_formula_finished_mass_percent' if lines and all('finished_product_percent' in r for r in lines) else 'explicit_application_context',
        base_formulation_screened=False, numerical_prediction_changed=False)
    result['regulatory'] = regulatory
    return result
