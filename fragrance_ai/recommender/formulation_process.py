"""Explicit source-scoped process states and units for v69 joint learning.

These are our structured summaries, not copied source prose or measured process
outcomes. A workflow ending is never a manufacturing approval. References do not
establish a universal heat, shear, ageing, fragrance-addition or preservation law.
"""
from __future__ import annotations

import math
import numpy as np

ACTIONS = ('brief', 'material_review', 'weigh', 'basic_accord', 'modify', 'blend',
           'fix', 'dilute', 'compare', 'revise', 'batch_record', 'phase_a', 'phase_b',
           'heat', 'emulsify', 'cool', 'fragrance_add', 'quality_review', 'fill_record',
           'done', 'hold')
TEMPLATES = {
    'perfume': ('brief', 'material_review', 'weigh', 'basic_accord', 'modify', 'blend',
                'fix', 'dilute', 'compare', 'revise', 'batch_record', 'quality_review',
                'fill_record', 'done'),
    'cold_lotion': ('brief', 'material_review', 'weigh', 'phase_a', 'phase_b',
                   'fragrance_add', 'emulsify', 'batch_record', 'quality_review', 'fill_record', 'done'),
    'hot_lotion': ('brief', 'material_review', 'weigh', 'phase_a', 'phase_b', 'heat',
                  'emulsify', 'cool', 'fragrance_add', 'batch_record', 'quality_review', 'fill_record', 'done'),
}
SOURCES = {
    'functional_composition': 'https://www.perfumersworld.com/formulating-by-function.php',
    'cold_reference': 'https://lotioncrafter.com/blogs/skin-care/basic-lotion-with-simulgel-eg',
    'hot_reference': 'https://lotioncrafter.com/blogs/skin-care/light-sprayable-lotion',
    'batch_quality_records': 'https://www.fda.gov/cosmetics/cosmetics-guidance-documents/good-manufacturing-practice-gmp-guidelinesinspection-checklist-cosmetics',
}
CHECKS = ('invalid_order', 'method_conflict', 'mixer_conflict', 'ph_conflict',
          'temperature_conflict', 'fragrance_addition_unverified', 'scale_up_unverified', 'unknown_process')
NUMERIC = ('peak_temperature_c', 'mixing_minutes', 'measured_ph',
           'fragrance_addition_temperature_c', 'batch_mass_g', 'method_code')


def process_context(template, values):
    if template not in TEMPLATES:
        raise ValueError('no source-scoped workflow for this process template')
    if not isinstance(values, dict) or set(values) - set(NUMERIC) - {'mixer_kind', 'pe9010_present'}:
        raise ValueError('unknown process context field')
    result = np.zeros(64, np.float32)
    result[0 if template == 'perfume' else 1] = 1.
    result[7 if template == 'cold_lotion' else 8 if template == 'hot_lotion' else 9] = 1.
    if type(values.get('pe9010_present', False)) is not bool:
        raise ValueError('preservative identity flag must be boolean')
    result[10] = float(values.get('pe9010_present', False))
    mixer = values.get('mixer_kind')
    if mixer not in (None, 'high_shear', 'overhead', 'manual'):
        raise ValueError('unknown mixer type')
    if mixer:
        result[4 + ('high_shear', 'overhead', 'manual').index(mixer)] = 1.
    for i, key in enumerate(NUMERIC):
        value = values.get(key)
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('finite numeric process values required')
        if value < 0 or (key == 'measured_ph' and value > 14):
            raise ValueError('process values outside input domain')
        if key == 'method_code' and value not in (0, 1, 2):
            raise ValueError('unknown process method code')
        result[13 + i] = (math.log1p(value) / 10 if key in ('mixing_minutes', 'batch_mass_g')
                         else value / 100 if 'temperature' in key else value / 14 if key == 'measured_ph' else value / 2)
        result[19 + i] = 1.
    # Source-relative coordinates resolve narrow operating windows without
    # feeding the decision label or the next action into the network. These
    # limits belong to the selected reference, not to arbitrary emulsions.
    temperature = values.get('peak_temperature_c')
    if template == 'hot_lotion' and temperature is not None:
        result[25], result[31] = (temperature - 72.5) / 2.5, 1.
    ph = values.get('measured_ph')
    if values.get('pe9010_present') and ph is not None:
        result[26], result[32] = (ph - 7.5) / 4.5, 1.
    mass = values.get('batch_mass_g')
    if template != 'perfume' and mass is not None:
        reference = 400 if template == 'cold_lotion' else 500
        result[27] = math.log1p(mass) - math.log1p(reference)
        result[28] = abs(mass - reference) / reference
    result[29] = float(values.get('method_code') == 1)
    result[30] = float(values.get('method_code') == 2)
    return result


def process_history(completed, actions=ACTIONS):
    ids, numbers = [], []
    for item in completed:
        record = {'action': item} if isinstance(item, str) else dict(item)
        action = record.pop('action', None)
        if action not in actions or set(record) - set(NUMERIC):
            raise ValueError('invalid or unknown manufacturing history event')
        values = process_context('perfume', record)
        ids.append(actions.index(action) + 1)
        numbers.append(values[13:25])
    return np.asarray(ids, np.int64), np.asarray(numbers, np.float32).reshape(-1, 12)


def procedure_target(template, completed, values):
    """Teacher for documented procedure selection, not a sensory-label oracle."""
    process_context(template, values)
    sequence = TEMPLATES[template]
    completed = tuple(completed)
    checks = np.zeros(len(CHECKS), np.float32)
    checks[0] = completed != sequence[:len(completed)] or len(completed) > len(sequence)
    method = values.get('method_code')
    expected = 0 if template == 'perfume' else 1 if template == 'cold_lotion' else 2
    checks[1] = method is not None and method not in (0, expected)
    checks[2] = template == 'cold_lotion' and values.get('mixer_kind') not in (None, 'high_shear')
    ph = values.get('measured_ph')
    checks[3] = values.get('pe9010_present', False) and ph is not None and not 3 <= ph <= 12
    temp = values.get('peak_temperature_c')
    checks[4] = template == 'hot_lotion' and temp is not None and not 70 <= temp <= 75
    # The hot reference omits a proven fragrance addition temperature; a user
    # entering a number does not turn the reference into a measurement.
    checks[5] = template == 'hot_lotion'
    reference_mass = 400 if template == 'cold_lotion' else 500
    checks[6] = template != 'perfume' and values.get('batch_mass_g') != reference_mass
    next_action = 'hold' if checks[:5].any() else sequence[min(len(completed), len(sequence) - 1)]
    return ACTIONS.index(next_action), checks


def learned_workflow(request, core, completed=()):
    """Neural next-step output plus the source predicate it cannot override."""
    from .formulation_workflow import formulation_workflow
    plan = formulation_workflow(request, include_learned=False)
    if plan['product_type'] == 'perfume':
        template = 'perfume'
    elif plan['selected_method'] in ('cold', 'hot'):
        template = plan['selected_method'] + '_lotion'
    else:
        return {'status': 'unsupported_reference_process', 'checkpoint_sha256': core.sha256}
    process = plan['process']
    values = {key: process.get(key) for key in NUMERIC if key != 'method_code'}
    values.update(mixer_kind=process.get('mixer_kind'), method_code={'auto': 0, 'cold': 1, 'hot': 2}[process['method']])
    values['pe9010_present'] = any(row['id'] == 'lc_pe9010' for row in plan['sources'])
    predicted = core.procedure(template, completed, values=values)
    teacher, checks = procedure_target(template, completed, values)
    return {**predicted, 'status': 'source_consistent' if predicted['next_action'] == ACTIONS[teacher] else 'model_source_conflict',
            'template': template, 'completed_step_ids': list(completed),
            'source_next_action': ACTIONS[teacher], 'source_checks': dict(zip(CHECKS, checks.astype(bool).tolist())),
            'workflow_steps': list(TEMPLATES[template]), 'sources': SOURCES,
            'full_source_scoped_sequence_covered': True,
            'manufacturing_approved': False, 'physical_outcome_labels_available': False}
