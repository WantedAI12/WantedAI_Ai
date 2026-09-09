"""Research bridge: stock-reference odor shapes weighted by lotion transport.

Gas concentration is NEVER passed as a stock dilution. Three fixed surrogate
stock conditions yield normalized component shapes, not lotion intensities.
Transport/OAV weights enter once. This factorization is uncalibrated and does
not replace the strict catalog-profile optimizer or its acceptance score.
"""
import numpy as np
from copy import deepcopy

from .models import RecipeConstraints, ScentBrief, SCENT_DIMENSIONS
from .lotion_evaluation import LOTION_PROJECTION as PROJECTION
from .perception_runtime import assert_provider_current, assert_provider_product, configured_perception, model_contract

REFERENCE_DILUTIONS = (0.001, 0.01, 0.1)
BRIDGE_VERSION = 'lotion-stock-shape-transport/v1'


def resolve_provider(provider):
    if provider is None:
        from .local_runtime import local_lotion_provider
        provider = local_lotion_provider(configured_perception('body_lotion'))
    assert_provider_product(provider, 'body_lotion')
    return provider


def make_lotion_shape_predictor(provider):
    """Keep reference semantics with their provider, never fabricate stock doses."""
    assert_provider_product(provider, 'body_lotion')
    factory = getattr(provider, 'begin_lotion_shapes', None)
    return factory() if factory is not None else LotionShapePredictor(provider)


class LotionShapePredictor:
    """Request-local shared stock-shape cache for search and final reporting."""
    def __init__(self, provider):
        assert_provider_product(provider, 'body_lotion')
        self.provider = provider
        self.projection = PROJECTION
        self.reference_scenarios = tuple({'stock_dilution': dose} for dose in REFERENCE_DILUTIONS)
        self.reference_count = len(self.reference_scenarios)
        self.nominal_reference_index = 1
        self.bridge_version = BRIDGE_VERSION
        brief = ScentBrief('', {}, [], [], [], [], 'medium', {}, RecipeConstraints(product_category='body_lotion'))
        self.session = provider.begin(brief)
        self.shapes, self.basis, self.missing = {}, {}, set()
        self.calls = 0
        self.forward_batches = 0
        self.shared_cache_hits = 0
        self._cache_keys = {}

    def _cache_key(self, item):
        return ('body_lotion', BRIDGE_VERSION, getattr(self.provider, 'component_model_sha256', None), self.provider.solvent,
                REFERENCE_DILUTIONS, item.ingredient_id, item.cas_number,
                self.provider.structures.get(item.ingredient_id), float(item.odor_impact),
                tuple(float(item.profile.get(axis, 0.)) for axis in SCENT_DIMENSIONS))

    def _restore(self, item):
        cache = getattr(self.provider, 'lotion_shape_cache', None)
        if cache is None:
            return False
        key = self._cache_key(item)
        self._cache_keys[item.ingredient_id] = key
        with self.provider.lotion_shape_cache_lock:
            if key not in cache:
                return False
            shape, basis = cache[key]
            cache.move_to_end(key)
            self.shapes[item.ingredient_id] = shape
            self.basis[item.ingredient_id] = deepcopy(basis)
            self.shared_cache_hits += 1
            return True

    def prefetch(self, items):
        pending = [item for item in items if item.ingredient_id not in self.shapes and not self._restore(item)]
        # Tiny formulas already amortize well; bound temporary feature/kernel
        # matrices without limiting which candidate materials are searched.
        if len(pending) < 64:
            return
        supported = []
        for item in pending:
            if self.session.supports(item):
                supported.append(item)
            else:
                self.shapes[item.ingredient_id] = None
                self.missing.add(item.ingredient_id)
        start = getattr(self.session, 'stock_batch_forward_calls', 0)
        for offset in range(0, len(supported), 128):
            batch = supported[offset:offset+128]
            for item, (prediction, basis) in zip(batch, self.session.predict_stock_batch(batch, REFERENCE_DILUTIONS)):
                self.calls += 1
                self._store(item.ingredient_id, prediction, basis)
        self.forward_batches += getattr(self.session, 'stock_batch_forward_calls', 0)-start

    def shape(self, item):
        identifier = item.ingredient_id
        if identifier in self.shapes:
            return self.shapes[identifier]
        if self._restore(item):
            return self.shapes[identifier]
        if not self.session.supports(item):
            self.missing.add(identifier)
            self.shapes[identifier] = None
            return None
        prediction, basis = self.session._predict(item, np.asarray(REFERENCE_DILUTIONS))
        self.calls += 1
        self.forward_batches += 1
        return self._store(identifier, prediction, basis)

    def _store(self, identifier, prediction, basis):
        prediction = np.asarray(prediction, dtype=float)
        if (prediction.shape != (3, len(self.provider.endpoints)) or not np.isfinite(prediction).all()
                or np.any(prediction < 0)):
            raise ValueError('invalid learned lotion component prediction')
        totals = prediction.sum(axis=1)
        if np.any(totals <= 0):
            self.missing.add(identifier)
            self.shapes[identifier] = None
            return None
        self.shapes[identifier] = prediction / totals[:, None]
        self.shapes[identifier].setflags(write=False)
        self.basis[identifier] = {'ingredient_id': identifier, 'reference_predictions': [
            {'stock_dilution': dose, **status} for dose, status in zip(REFERENCE_DILUTIONS, basis)]}
        cache = getattr(self.provider, 'lotion_shape_cache', None)
        if cache is not None and identifier in self._cache_keys:
            with self.provider.lotion_shape_cache_lock:
                key = self._cache_keys[identifier]
                cache[key] = (self.shapes[identifier], deepcopy(self.basis[identifier]))
                cache.move_to_end(key)
                while len(cache) > 3840:
                    cache.popitem(last=False)
        return self.shapes[identifier]


def attach_lotion_perception(result, catalog, provider, *, _shape_predictor=None):
    """Return additive fields only; preserve scores, recipes and transport.

    Cache component inference across final transport scenarios in this request.
    LP basis evaluations never call this bridge.
    """
    if provider is None:
        return result
    assert_provider_current(provider)
    predictor = _shape_predictor or make_lotion_shape_predictor(provider)
    if predictor.provider is not provider:
        raise ValueError('lotion predictor checkpoint mismatch')
    known = {item.ingredient_id: item for item in catalog.ingredients}
    shapes, component_basis, missing = {}, [], set()
    projection = predictor.projection
    references = predictor.reference_scenarios
    nominal = predictor.nominal_reference_index
    stock_references = all('stock_dilution' in row for row in references)
    mapped = sorted({provider.endpoints.index(name) for names in projection.values() for name in names})
    evaluated_points, partial_points = 0, 0

    def shape(identifier):
        if identifier in shapes:
            return shapes[identifier]
        item = known.get(identifier)
        if item is None:
            missing.add(identifier)
            shapes[identifier] = None
            return None
        prediction = predictor.shape(item)
        if prediction is None:
            missing.add(identifier)
            shapes[identifier] = None
            return None
        shapes[identifier] = prediction
        component_basis.append(predictor.basis[identifier])
        return shapes[identifier]

    def simulation(value):
        nonlocal evaluated_points, partial_points
        points = []
        for point in value['temporal_profile']:
            use_oav = point['profile_basis'] == 'linear_odor_activity_proxy'
            key = 'odor_activity_proxy' if use_oav else 'air_concentration_mg_m3'
            rows = [(row, float(row[key])) for row in point['materials']]
            weights = np.asarray([weight for _, weight in rows])
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError('invalid lotion transport weights for learned profile')
            total = float(weights.sum())
            output = {'status': 'no_airborne_material', 'profile_basis': predictor.bridge_version,
                      'transport_weighting': 'odor_activity_proxy' if use_oav else 'air_mass',
                      'covered_transport_weight_percent': None, 'predicted_endpoint_fraction': None,
                      'supported_endpoint_contribution': None, 'unknown_transport_weight_fraction': None,
                      'endpoint_fraction_bounds': None,
                      'stock_reference_sensitivity': [], 'reference_profile_sensitivity': [],
                      'unmapped_endpoint_fraction': None}
            if value['status'] == 'outside_open_sink_assumption':
                output['status'] = 'abstained_invalid_transport_assumption'
            elif total > 0:
                mixture = np.zeros((predictor.reference_count, len(provider.endpoints)))
                covered, incomplete = 0., False
                for row, weight in rows:
                    if weight <= 0:
                        continue
                    prediction = shape(row['ingredient_id'])
                    if prediction is None:
                        incomplete = True
                        continue
                    # Shape is normalized at a fixed reference, independent of
                    # the formula dose. Transport already includes formula mass.
                    mixture += prediction * (weight / total)
                    covered += weight
                output['covered_transport_weight_percent'] = min(100., 100.*covered/total)
                unknown = float(np.clip(1.-covered/total, 0., 1.))
                output['unknown_transport_weight_fraction'] = unknown
                if covered > 0:
                    # These contributions sum to COVERAGE, never to one when
                    # a molecular identity is missing. Bounds allow arbitrary
                    # missing component shape; they are not confidence limits.
                    output['supported_endpoint_contribution'] = dict(zip(provider.endpoints, mixture[nominal].tolist()))
                    output['endpoint_fraction_bounds'] = {name: {
                        'lower': float(mixture[:, i].min()),
                        'upper': float(min(1., mixture[:, i].max()+unknown))}
                        for i, name in enumerate(provider.endpoints)}
                    if incomplete:
                        partial_points += 1
                output['status'] = 'abstained_incomplete_component_coverage' if incomplete else 'research_shape_evaluated'
                if not incomplete:
                    evaluated_points += 1
                    output['predicted_endpoint_fraction'] = dict(zip(provider.endpoints, mixture[nominal].tolist()))
                    output['unmapped_endpoint_fraction'] = float(np.clip(1.-mixture[nominal, mapped].sum(), 0., 1.))
                    sensitivity = [
                        {**reference, 'predicted_endpoint_fraction': dict(zip(provider.endpoints, vector.tolist()))}
                        for reference, vector in zip(references, mixture)]
                    output['reference_profile_sensitivity'] = sensitivity
                    if stock_references:
                        output['stock_reference_sensitivity'] = sensitivity
            points.append({**point, 'learned_perception': output})
        return {**value, 'temporal_profile': points}

    if 'temporal_profile' in result:
        output = simulation(result)
    else:
        output = dict(result)
        if result.get('simulation'):
            output['simulation'] = simulation(result['simulation'])
        if 'scenario_simulations' in result:
            output['scenario_simulations'] = [simulation(value) for value in result['scenario_simulations']]
    output['perception_model'] = {
        **model_contract(provider), 'operation': 'lotion', 'bridge_version': predictor.bridge_version,
        'status': 'research_transport_shape_partial' if partial_points else 'research_transport_shape_evaluated'
                  if evaluated_points else 'abstained_no_supported_temporal_profile',
        'applied': evaluated_points + partial_points > 0,
        'applied_to': 'temporal_profiles_and_candidate_search' if result.get('learned_optimization') else 'auxiliary_temporal_profiles_only',
        'optimization_guidance_evaluated': bool(result.get('learned_optimization')),
        'optimization_mode': 'secondary_constrained_affinity_not_score_blending' if result.get('learned_optimization') else None,
        'partial_target_guidance': bool(result.get('learned_optimization', {}).get('partial_target_guidance')),
        'modeled_target_mass_fractions': result.get('learned_optimization', {}).get('modeled_target_mass_fractions'),
        'unmodeled_target_axes': result.get('learned_optimization', {}).get('unsupported_target_axes', []),
        'full_note_balance_refinement': (result.get('learned_optimization') or {}).get('profile_balance'),
        'search_weight': 0., 'optimization_guidance_applied': bool(result.get('learned_optimization', {}).get('recipe_changed')),
        'full_model_application': False,
        'strict_score_modified': False, 'recipe_weights_modified': bool(result.get('learned_optimization', {}).get('recipe_changed')),
        'stock_reference_dilutions': [row['stock_dilution'] for row in references] if stock_references else [],
        'nominal_stock_reference_dilution': references[nominal].get('stock_dilution'),
        'reference_profile_scenarios': list(references),
        'nominal_reference_profile': dict(references[nominal]),
        'stock_reference_is_lotion_concentration': False, 'gas_to_stock_conversion_used': False,
        'lotion_matrix_validated': False, 'absolute_intensity_validated': False,
        'human_similarity_percent': None, 'evaluated_timepoint_count': evaluated_points,
        'partial_timepoint_count': partial_points,
        'component_model_calls': predictor.calls, 'unmapped_ingredient_ids': sorted(missing),
        'model_forward_batches': predictor.forward_batches,
        'shared_component_cache_hits': predictor.shared_cache_hits,
        'component_basis': component_basis,
        'supported_projection_dimensions': list(projection),
        'limitations': [
            'Reference stock dilutions and configured solvent are surrogate scenarios, not the lotion phase or measured headspace conditions.',
            'Each predicted endpoint vector is normalized; learned absolute intensity is discarded.',
            'Additive transport-weighted shapes do not model mixture masking, synergy or concentration-dependent shape in lotion.',
            'Three stock references form a deterministic sensitivity check, not a confidence interval.',
            'Partial endpoint contributions sum to supported transport weight, not one; bounds permit arbitrary missing component shapes and are not simultaneous probability intervals.',
            'Transport estimates and available sourced thresholds retain their separate provenance; neither is lotion sensory validation.']}
    if result.get('perceptual_evaluation'):
        output['perception_model'].update(
            applied_to='primary_observed_reference_objective_and_final_transport',
            optimization_mode='full_146_endpoint_observed_reference',
            learned_profiles_drive_primary_objective=True,
            legacy_score_used_for_acceptance=False,
            primary_score_kind=result.get('score_kind'))
    if not stock_references:
        output['perception_model']['limitations'] = [
            'Atlas applicability and descriptor-use measurements are distinct, separately normalized reference shapes, not absolute intensities.',
            'Ordinal source high/low labels are not numeric stock dilutions, gas concentrations or lotion doses.',
            'Transport-weighted reference shapes are an uncalibrated lotion prior; masking, synergy and matrix-dependent descriptor changes are not inferred.',
            'Reference-head bounds are deterministic sensitivity bounds, not statistical confidence intervals.',
            'Unmapped endpoint mass and missing molecular identities retain their full mass; neither is renormalized away.',
            'Transport estimates and available sourced thresholds retain their separate provenance; neither is lotion sensory validation.']
    if output.get('preparation', {}).get('intent', {}).get('representation'):
        prep = output['preparation']
        intent = prep['intent']
        requested = set(intent['avoided_dimensions']) | {k for k,v in intent['target_profile'].items() if v > 0}
        requested.update(k for profile in intent.get('phase_target_profiles', {}).values() for k,v in profile.items() if v > 0)
        requested.update(k for values in intent.get('phase_avoided_dimensions', {}).values() for k in values)
        # Runtime coverage belongs to the predictor report. Attaching a model
        # must not rewrite the already interpreted user's target.
        output['perception_model']['intent_coverage'] = {
            'learned_supported_axes': sorted(projection),
            'learned_unmodeled_requested_axes': sorted(requested-set(projection))}
    assert_provider_current(provider)
    return output
