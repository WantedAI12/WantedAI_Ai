"""Publish accepted input units, never invent measured product coefficients."""
from .unified_product_inputs import StageKinetics


def coefficient_contract():
    schema = StageKinetics.model_json_schema()
    rates = ('evaporation_per_min', 'uptake_per_min', 'hydrolysis_per_min',
             'air_return_per_min', 'ventilation_per_min', 'capacity_decay_per_min')
    fractions = ('evaporating_capacity_fraction', 'nonreactive_capacity_fraction')
    return {'schema_version':'unified-coefficient-contract-1',
        'coefficient_sources_verified':False, 'measured_reference_table_available':False,
        'server_supplied_default_coefficients':False,
        'fields':{key:{**schema['properties'][key], 'unit':'1/min' if key in rates else 'fraction_0_1',
                      'required':True, 'default_value':None} for key in (*rates,*fractions)},
        'source_kind_values':schema['properties']['source_kind']['enum'],
        'source_reference_required':True, 'measured_label_is_independent_verification':False,
        'coverage':'every_component_in_every_stage',
        'context_binding':{'prepare_endpoint':'/v1/applications/unified/context',
                           'required_field':'parameter_context_id'},
        'body_wash_rinse':{'required':True,'field':'rinse_retained_film_fractions',
                           'unit':'fraction_0_1','coverage':'every_component',
                           'source_field':'rinse_source_reference'},
        'missing_values_policy':'reject_not_fill_with_zero_or_synthetic_measurement'}
