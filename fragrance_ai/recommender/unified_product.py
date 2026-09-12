"""Shared V54 odor backbone + V60 trained transport for finished-product input.

This is one conditional forward path, not three copies of model weights. The
stock-aliquot V54 mixture residual is intentionally not applied to a finished
product outside its assay domain. Existing recipe generators remain compatible.
"""
import hashlib
import json

import numpy as np

from .unified_transport import trajectory
from .lotion_atlas import AtlasReferenceGuidance, ATLAS_PROJECTION
from .lotion_surrogate import MASS_FIELDS
from .models import RecipeConstraints, SCENT_DIMENSIONS


class UnifiedProductPredictor:
    def __init__(self, transport, atlas, structures, catalog, *, reference_bank=None, fine_model=None,
                 odor_backbone_sha256=None):
        # Legacy callers keep the exact original parent contract. A newer
        # odor-only backbone must be selected explicitly by its runtime pin.
        expected = odor_backbone_sha256 or transport.manifest['parent_models']['atlas']['sha256']
        if expected != atlas.sha256:
            raise ValueError('unified transport / molecular backbone binding mismatch')
        if reference_bank is not None and (reference_bank.parent_sha256 != atlas.sha256
                                           or tuple(reference_bank.endpoints) != tuple(atlas.endpoints)):
            raise ValueError('unified target references use a different molecular backbone')
        self.transport, self.atlas, self.catalog, self.references = transport, atlas, catalog, reference_bank
        self.fine_model = fine_model
        self.backbone = AtlasReferenceGuidance(atlas, structures, experimental=True)
        self.known = {i.ingredient_id:i for i in catalog.ingredients}
        from .brief_parser import NaturalLanguageBriefParser
        self.parser = NaturalLanguageBriefParser(catalog)

    def assert_current(self):
        self.transport.assert_current()
        self.backbone.assert_current()
        if self.references is not None:
            self.references.assert_current()
        if self.fine_model is not None:
            self.fine_model.assert_current()

    def contract(self):
        self.assert_current()
        return {**self.transport.contract(), 'shared_odor_backbone_sha256': self.atlas.sha256,
            'odor_backbone_version': getattr(self.atlas,'artifact_version',None),
            'transport_training_atlas_sha256': self.transport.manifest['parent_models']['atlas']['sha256'],
            'target_reference_sha256': self.references.sha256 if self.references is not None else None,
            'fine_odor_expression':self.fine_model.contract() if self.fine_model is not None else None,
            'odor_endpoints': len(self.atlas.endpoints), 'learned_stock_assay_head_applied': False,
            'prediction_scope': 'conditional_product_transport_and_ordinal_odor_reference_proxy',
            'coefficient_sources_verified': False, 'cross_product_human_scores_comparable': False}

    def predict(self, request):
        self.assert_current()
        ids = [r.ingredient_id for r in request.components]
        if set(ids)-set(self.known):
            raise ValueError('unknown unified product ingredient IDs')
        items = [self.known[key] for key in ids]
        if any(i.blocked or abs(i.active_strength_percent-100.) > 1e-8 for i in items):
            raise ValueError('unified transport requires admissible undiluted materials')
        from .odor_expression import brief_expression, summarize_expression
        category = 'eau_de_parfum' if request.context.product_type == 'perfume' else request.context.product_type
        parsed_brief = self.parser.parse(request.brief,RecipeConstraints(product_category=category)) if request.brief else None
        fine_intent = brief_expression(parsed_brief) if parsed_brief is not None else None
        fine_matrix, fine_evidence = (None,[])
        if self.fine_model is not None:
            fine_matrix,fine_evidence = self.fine_model.materials(items)
            if any(row['status']=='missing_or_multicomponent_structure' for row in fine_evidence):
                fine_matrix = None
        shapes = self.backbone.begin_reference_shapes()
        shapes.prefetch(items)
        missing = [i.ingredient_id for i in items if shapes.shape(i) is None]
        # Missing profiles are not zeros and cannot improve a recipe's score.
        learned_profiles = None if missing else np.stack([shapes.shape(i) for i in items], axis=1)
        context = request.context
        mass = (context.application_mass_mg_cm2*context.fragrance_concentration_percent/100.
                *np.array([c.concentrate_percent/100. for c in request.components]))
        parent = np.array([c.initial_parent_fraction for c in request.components])
        state = np.zeros((len(ids), 6))
        state[:, 0] = mass*parent
        state[:, 4] = mass*(1-parent)
        times = np.asarray(request.times_minutes)
        # Reference evaluation uses fixed integration samples, not user-chosen
        # display points (which could otherwise hide an unwanted scent phase).
        canonical, offset = [0.], 0.
        for stage in context.stages:
            canonical.extend((offset+np.linspace(0., 1., 33)[1:]**2*stage.duration_minutes).tolist())
            offset += stage.duration_minutes
        canonical = np.asarray(canonical)
        model_times = np.unique(np.r_[times, canonical])
        observations, event_rows, start = {0.: state.copy()}, [], 0.
        work_budget = {'remaining_material_transitions': 2_000_000}
        for stage_context, stage in zip(context.stages, request.stages):
            duration = stage_context.duration_minutes
            end = start+duration
            by_id = {r.ingredient_id:r for r in stage.coefficients}
            coefficients = [by_id[key] for key in ids]
            rates = np.array([[r.evaporation_per_min, r.uptake_per_min, r.hydrolysis_per_min,
                r.air_return_per_min, r.ventilation_per_min, r.capacity_decay_per_min] for r in coefficients])
            fractions = np.array([[r.evaporating_capacity_fraction, r.nonreactive_capacity_fraction] for r in coefficients])
            requested = model_times[(model_times > start)&(model_times <= end)]
            local_times = np.unique(np.r_[requested-start, duration])
            rows, state = trajectory(rates, fractions, local_times, duration=duration,
                initial=state, operator=self.transport.stable_kernel, work_budget=work_budget)
            for t, index in zip(requested, np.searchsorted(local_times, requested-start)):
                observations[float(t)] = rows[index].copy()
            if stage.rinse_retained_film_fractions is not None:
                retention = np.array([stage.rinse_retained_film_fractions[key] for key in ids])
                removed = state[:, 0]*(1-retention)
                state[:, 0] *= retention
                state[:, 5] += removed
                event_rows.append({'minutes': end, 'stage_id': stage.stage_id, 'kind': 'rinse',
                    'washed_off_mg_cm2': float(removed.sum()), 'retention_basis': 'caller_supplied_not_learned_deposition',
                    'source_reference': stage.rinse_source_reference})
                # At a jump the public sample is explicitly right-continuous.
                if end in observations:
                    observations[end] = state.copy()
            start = end
        masses = np.stack([observations[float(t)] for t in times])
        all_thresholds = all(c.odor_threshold_mg_m3 is not None for c in request.components)
        use_oav = request.profile_weighting != 'air_mass' and all_thresholds
        thresholds = np.array([c.odor_threshold_mg_m3 or 1. for c in request.components])
        air = masses[:, :, 1]/context.headspace_height_cm*1e6
        weights = air/thresholds[None, :] if use_oav else air
        projection = np.zeros((len(self.atlas.endpoints), len(SCENT_DIMENSIONS)))
        for j, axis in enumerate(SCENT_DIMENSIONS):
            for label in ATLAS_PROJECTION.get(axis, ()):
                projection[list(self.atlas.endpoints).index(label), j] = 1.
        temporal = []
        for i, (t, values) in enumerate(zip(times, masses)):
            total = float(weights[i].sum())
            full = (np.einsum('n,hnk->hk', weights[i], learned_profiles)/total
                    if total > 0 and learned_profiles is not None else None)
            coarse = dict(zip(SCENT_DIMENSIONS, (full[0]@projection).tolist())) if full is not None else None
            fine_profile = weights[i]@fine_matrix/total if fine_matrix is not None and total>0 else None
            temporal.append({'minutes': float(t), 'total_air_concentration_mg_m3': float(air[i].sum()),
                'total_odor_activity_proxy': total if use_oav else None, 'scent_profile': coarse,
                'full_odor_reference_profiles': None if full is None else {'applicability': full[0].tolist(), 'use': full[1].tolist()},
                'fine_odor_expression':summarize_expression(fine_profile,self.fine_model.endpoints if self.fine_model else (),
                    requested=set(fine_intent['wanted'])|set(fine_intent['avoided']) if fine_intent else ()),
                'materials': [{'ingredient_id': key, 'initial_mg_cm2': float(mass[j]),
                    **{field: float(values[j, k]) for k, field in enumerate((*MASS_FIELDS, 'washed_off_mg_cm2'))},
                    'air_concentration_mg_m3': float(air[i, j])} for j, key in enumerate(ids)]})
        canonical_air = np.stack([observations[float(t)][:, 1] for t in canonical])/context.headspace_height_cm*1e6
        canonical_weights = canonical_air/thresholds[None, :] if use_oav else canonical_air
        assessment = self._assess(request, canonical, canonical_weights, learned_profiles,parsed_brief=parsed_brief)
        if fine_intent is not None:
            assessment['fine_expression_target_met'] = None
            assessment['fine_expression_similarity_calibrated'] = False
        request_sha = hashlib.sha256(json.dumps(request.model_dump(mode='json'), sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {'schema_version': 'unified-product-prediction-1', 'request_id': request_sha,
            'product_type': context.product_type, 'model': self.contract(), 'parameter_context_id': request.parameter_context_id,
            'status': 'research_prediction' if not missing else 'transport_only_missing_odor_profiles',
            'temporal_profile': temporal, 'process_events': event_rows, 'reference_assessment': assessment,
            'fine_expression_intent':fine_intent,'fine_expression_material_evidence':fine_evidence,
            'odor_endpoints': list(self.atlas.endpoints), 'missing_odor_profile_ids': missing,
            'profile_weighting': 'odor_activity_proxy' if use_oav else 'air_mass_proxy',
            'diagnostics': {'mass_balance_max_abs_error_mg_cm2': float(np.max(np.abs(masses.sum(axis=2)-mass))),
                'nonnegative': bool(np.all(masses >= 0)),
                'cumulative_sinks_monotone': bool(np.all(np.diff(masses[:, :, 2:], axis=0) >= -1e-14)),
                'sampling_at_process_jumps': 'right_continuous', 'shared_checkpoint_used': True,
                'transport_material_transitions': 2_000_000-work_budget['remaining_material_transitions'],
                'transport_material_transition_budget': 2_000_000,
                'molecular_forward_batches': shapes.forward_batches, 'molecular_cache_hits': shapes.shared_cache_hits},
            'caller_declared_coefficient_sources': [{'stage_id': s.stage_id,
                'materials': [{'ingredient_id': r.ingredient_id, 'kind': r.source_kind, 'reference': r.source_reference}
                              for r in s.coefficients]} for s in request.stages],
            'human_similarity_percent': None, 'manufacturing_approved': False,
            'limitations': ['Transport checkpoint is trained on numerical transition labels, not measured finished-product release.',
                'Supplied stage coefficients must describe the actual carrier, dilution, temperature and phase capacity.',
                'No inferred micellar phase transitions, ethanol activity coefficients or skin deposition without context data.',
                'The shared 146-axis reference shape and its linear airborne mixture are not calibrated human similarity.',
                'Existing recipe acceptance gates and the stock-aliquot assay head have not been replaced by this product proxy.']}

    def _assess(self, request, times, weights, learned_profiles, *, parsed_brief=None):
        if request.brief is None:
            return {'status': 'not_requested'}
        if self.references is None or learned_profiles is None:
            return {'status': 'unavailable', 'score': None, 'reason': 'complete_backbone_and_reference_bank_required'}
        category = 'eau_de_parfum' if request.context.product_type == 'perfume' else request.context.product_type
        brief = parsed_brief or self.parser.parse(request.brief, RecipeConstraints(product_category=category))
        if brief.phase_target_profiles or brief.phase_avoided_dimensions:
            return {'status': 'unsupported', 'score': None, 'reason': 'explicit_phase_evaluation_not_implemented_in_unified_v60'}
        targets, unsupported = self.references.targets(brief, [{'phase': 'overall', 'target_profile': brief.target_profile,
            'avoided': brief.avoided_dimensions}])
        if unsupported:
            return {'status': 'unsupported', 'score': None, 'unsupported_requirements': unsupported}
        exposure = np.sum((weights[:-1]+weights[1:])*np.diff(times)[:, None]/2., axis=0)/times[-1]
        if exposure.sum() <= 0:
            return {'status': 'unavailable', 'score': None, 'reason': 'no_modeled_airborne_exposure'}
        predicted = np.einsum('n,hnk->hk', exposure/exposure.sum(), learned_profiles)
        assessments = [self.references.compare(targets[0], predicted[head], head) for head in range(2)]
        # Comparison remains diagnostic. No 95-percent manufacturing/user claim.
        return {'status': 'diagnostic', 'measurement_heads': assessments,
            'aggregation': 'time_integrated_airborne_reference_profile',
            'score_kind': 'observed_reference_profile_agreement_not_user_similarity',
            'human_similarity_percent': None}


def configured_unified_product(catalog, *, component_provider=None):
    from .unified_transport import configured_unified_transport
    from .local_runtime import local_odor_backbone_provider, local_profile
    from .perception_runtime import configured_perception
    from .lotion_reference_objective import load_configured_reference_bank
    from .fine_odor_model import configured_fine_odor
    model = configured_unified_transport()
    if model is None:
        return None
    provider = component_provider or configured_perception()
    if provider is None:
        raise ValueError('unified model requires a bound material-identity registry')
    profile = local_profile()
    expected = profile.get('odor_backbone',profile['atlas'])[1]
    return UnifiedProductPredictor(model, local_odor_backbone_provider(), provider.structures, catalog,
        reference_bank=load_configured_reference_bank(),fine_model=configured_fine_odor(),
        odor_backbone_sha256=expected)
