"""Recipe search views of the single shared network, with no legacy weights.

Ordinal odor-shape learning does not identify absolute intensity or a solvent
response curve. Product physics supplies dose/time weights separately.
"""

from functools import lru_cache
import numpy as np

from .formulation_views import shared_views
from .lotion_atlas import AtlasReferenceGuidance, AtlasLotionGuidance, ATLAS_PROJECTION
from .models import SCENT_DIMENSIONS


class SharedPerfumeGuidance(AtlasReferenceGuidance):
    coarse_dimensions = SCENT_DIMENSIONS
    def __init__(self, core, *, reference_bank=None):
        super().__init__(
            shared_views(core)[0], core.manifest["structures"], experimental=True
        )
        self.core = core
        self.runtime_product = "perfume"
        self.runtime_manifest_sha256 = core.sha256
        self.weight = 0.25
        self.model_scope = (
            "single_shared_neural_ordinal_shape_not_calibrated_absolute_intensity"
        )
        bind_complete_reference_support(self,core,reference_bank)

    def begin(self, brief):
        self.assert_current()
        if brief.constraints.validation_level != "prototype":
            raise ValueError("unpromoted shared model requires prototype evaluation")
        if self.core.manifest.get('complete_reference_guidance') is True:
            from .hierarchical_perfume import active, PhysicalReferenceSession
            if active(self, brief):
                return PhysicalReferenceSession(self, brief)
            from .full_reference_guidance import FullReferenceSession
            return FullReferenceSession(self, brief)
        return SharedRecipeSession(self, brief)


class SharedRecipeSession:
    def __init__(self, provider, brief):
        self.provider, self.brief = provider, brief
        self.shapes = provider.begin_reference_shapes()
        self.supported_intent = [
            a for a in brief.desired_dimensions if a in ATLAS_PROJECTION
        ]
        self.unsupported_intent = [
            a for a in brief.desired_dimensions if a not in ATLAS_PROJECTION
        ]
        self.product_supported = brief.constraints.product_category in {
            "eau_de_parfum",
            "eau_de_toilette",
            "eau_de_cologne",
        }
        self.enabled = (
            self.product_supported
            and bool(self.supported_intent)
            and not brief.constraints.reference_target_id
        )
        self.weight, self.grid_calls, self.exact_calls = provider.weight, 0, 0
        self.target = np.array(
            [brief.target_profile.get(a, 0.0) for a in SCENT_DIMENSIONS]
        )
        self.target /= max(self.target.sum(), 1e-12)
        self.projection = np.zeros((len(provider.endpoints), 19))
        for j, a in enumerate(SCENT_DIMENSIONS):
            for name in ATLAS_PROJECTION.get(a, ()):
                self.projection[provider.endpoints.index(name), j] = 1.0
        self.avoided = [
            SCENT_DIMENSIONS.index(a)
            for a in brief.avoided_dimensions
            if a in SCENT_DIMENSIONS
        ]
        self.missing_ids = set()

    def supports(self, ingredient):
        return self.shapes._graph(ingredient) is not None

    def evaluate(self, weights, ingredients, *, exact=False):
        if not self.enabled:
            return None
        weights = np.asarray(weights, float)
        if (
            weights.shape != (len(ingredients),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or abs(weights.sum() - 100) > 0.002
        ):
            raise ValueError("finite formula percentages summing to 100 required")
        active = [(w, i) for w, i in zip(weights, ingredients) if w > 1e-8]
        self.shapes.prefetch([i for _, i in active])
        missing = [i.ingredient_id for _, i in active if self.shapes.shape(i) is None]
        if missing:
            self.missing_ids.update(missing)
            return None
        masses = np.array([w * i.active_strength_percent / 100 for w, i in active])
        if masses.sum() <= 0:
            return None
        masses /= masses.sum()
        # Each molecular reference already sums to one. Apply active mass ONCE.
        q = np.einsum(
            "i,ird->rd", masses, np.stack([self.shapes.shape(i) for _, i in active])
        )
        projected = q @ self.projection
        cosine = (
            projected
            @ self.target
            / np.maximum(
                np.linalg.norm(projected, axis=-1) * np.linalg.norm(self.target), 1e-12
            )
        )
        explained = projected.sum(-1) / np.maximum(q.sum(-1), 1e-12)
        avoided = projected[:, self.avoided].sum(-1) / np.maximum(
            projected.sum(-1), 1e-12
        )
        scores = np.clip(100 * (cosine * explained - avoided), 0, 100)
        if exact:
            self.exact_calls += 1
        else:
            self.grid_calls += 1
        return {
            "score": float(scores.min()),
            "nominal_score": float(scores.mean()),
            "reference_scenario_scores": scores.tolist(),
            "predicted_ordinal_reference_profile": dict(
                zip(self.provider.endpoints, q.mean(0).tolist())
            )
            if exact
            else {},
            "unexplained_reference_mass_fraction": float(1 - explained.mean()),
            "exact_forward": exact,
            "checkpoint_sha256": self.provider.core.sha256,
            "physical_concentration_model_applied_here": False,
            "prediction_kind": ("observed_and_predicted_component_shapes_with_additive_active_mass_proxy"
                                if getattr(self.shapes,'observations',None) is not None
                                else "shared_neural_component_shapes_with_additive_active_mass_proxy"),
        }

    def score(self, weights, ingredients):
        value = self.evaluate(weights, ingredients)
        return None if value is None else value["score"]

    def gradient_function(self, ingredients):
        """Exact derivative of the unchanged neural score on a fixed support.

        The dose solver previously proposed solely against the structural
        objective and rejected neural regressions afterward. Binding this
        derivative lets it seek improvements inside that same admissible set.
        """
        if not self.enabled:
            return None
        self.shapes.prefetch(ingredients)
        rows = [self.shapes.shape(item) for item in ingredients]
        if any(row is None for row in rows):
            return None
        shapes = np.stack(rows) @ self.projection  # material, head, axis
        strengths = np.asarray([item.active_strength_percent/100 for item in ingredients])
        return projected_guidance_gradient(shapes, strengths, self.target, self.avoided)

    def evaluate_lines(self, lines, ingredients):
        return self.evaluate(
            [line.concentrate_percent for line in lines],
            [ingredients[line.ingredient_id] for line in lines],
            exact=True,
        )

    def replacement_scores(self, original, ingredients, donor_id, replacements):
        """Exact shared-network scores for every same-mass substitution.

        The source mixture is computed once. No candidate is shortlisted by
        structural scent first and no extra network is loaded for each swap.
        Missing identities remain NaN and cannot be used as a favorable score.
        """
        if not self.enabled:
            return None
        pool = {item.ingredient_id: item for item in ingredients}
        source = [pool[key] for key in original]
        if donor_id not in original:
            raise ValueError('replacement donor is absent from source formula')
        self.shapes.prefetch([*source, *replacements])
        source_shapes = [self.shapes.shape(item) for item in source]
        if any(shape is None for shape in source_shapes):
            return np.full(len(replacements), np.nan)
        masses = np.array([original[item.ingredient_id]*item.active_strength_percent/100 for item in source])
        projected = np.stack(source_shapes)@self.projection
        total = np.einsum('n,nhd->hd', masses, projected)
        donor = next(i for i, item in enumerate(source) if item.ingredient_id == donor_id)
        rest, rest_mass = total-masses[donor]*projected[donor], masses.sum()-masses[donor]
        scores = np.full(len(replacements), np.nan)
        covered = [(i, item, self.shapes.shape(item)) for i, item in enumerate(replacements)]
        covered = [(i, item, shape) for i, item, shape in covered if shape is not None]
        if not covered:
            return scores
        new_mass = np.array([original[donor_id]*item.active_strength_percent/100 for _, item, _ in covered])
        profiles = rest[None]+new_mass[:, None, None]*(np.stack([s for _, _, s in covered])@self.projection)
        total_mass = rest_mass+new_mass
        sums = profiles.sum(-1)
        cosine = (profiles@self.target)/np.maximum(np.linalg.norm(profiles, axis=-1)*np.linalg.norm(self.target), 1e-12)
        explained = sums/np.maximum(total_mass[:, None], 1e-12)
        avoidance = profiles[:, :, self.avoided].sum(-1)/np.maximum(sums, 1e-12)
        values = np.clip(100*(cosine*explained-avoidance), 0., 100.).min(-1)
        scores[[i for i, _, _ in covered]] = values
        return scores

    def transfer_scores(self, original, ingredients, donor_id, replacements, amounts_percent):
        """Exact neural scores for batched PARTIAL donor-to-receiver transfers."""
        if not self.enabled:
            return None
        amounts = np.asarray(amounts_percent, float)
        if (donor_id not in original or amounts.shape != (len(replacements),)
                or not np.isfinite(amounts).all() or np.any(amounts < 0)
                or np.any(amounts > original[donor_id]+1e-10)):
            raise ValueError('invalid partial neural transfer')
        lookup = {item.ingredient_id:item for item in ingredients}
        source = [lookup[key] for key in original]
        self.shapes.prefetch([*source,*replacements])
        measured = [self.shapes.shape(item) for item in source]
        if any(row is None for row in measured):
            return np.full(len(replacements), np.nan)
        source_mass = np.asarray([original[item.ingredient_id]*item.active_strength_percent/100 for item in source])
        projected = np.stack(measured)@self.projection
        total = np.einsum('n,nhd->hd',source_mass,projected)
        donor = next(i for i,item in enumerate(source) if item.ingredient_id == donor_id)
        output = np.full(len(replacements), np.nan)
        covered = [(i,item,self.shapes.shape(item)) for i,item in enumerate(replacements)]
        covered = [(i,item,value) for i,item,value in covered if value is not None]
        if not covered:
            return output
        index = np.asarray([i for i,_,_ in covered])
        removed = amounts[index]*source[donor].active_strength_percent/100
        added = amounts[index]*np.asarray([item.active_strength_percent/100 for _,item,_ in covered])
        profiles = total[None]-removed[:,None,None]*projected[donor]+added[:,None,None]*(np.stack([row for _,_,row in covered])@self.projection)
        mass = source_mass.sum()-removed+added
        sums = profiles.sum(-1)
        cosine = profiles@self.target/np.maximum(np.linalg.norm(profiles,axis=-1)*np.linalg.norm(self.target),1e-12)
        explained = sums/np.maximum(mass[:,None],1e-12)
        avoidance = profiles[:,:,self.avoided].sum(-1)/np.maximum(sums,1e-12)
        output[index] = np.clip(100*(cosine*explained-avoidance),0.,100.).min(-1)
        return output

    def report(self, baseline, selected, *, changed, variants):
        self.shapes.assert_observations_current()
        return {
            "status": "shared_neural_guidance_evaluated"
            if selected
            else "abstained_no_supported_formula_or_intent",
            "model_sha256": self.provider.core.sha256,
            "component_model_sha256": self.provider.core.sha256,
            "single_shared_checkpoint": True,
            "baseline": baseline,
            "selected": selected,
            "recipe_changed": bool(changed and selected),
            "guidance_variants_added": variants,
            "grid_objective_calls": self.grid_calls,
            "exact_formula_evaluations": self.exact_calls,
            "supported_intent_dimensions": self.supported_intent,
            "unsupported_intent_dimensions": self.unsupported_intent,
            "unmapped_ingredient_ids": sorted(self.missing_ids),
            "applied_search_weight": self.weight,
            "score_kind": "ordinal_reference_proxy_not_human_similarity",
            "manufacturing_approval": False,
            "absolute_intensity_validated": False,
            "actual_human_accuracy_90_authorized": False,
            "component_reference_observations":self.shapes.observation_summary(),
        }


def projected_guidance_gradient(shapes, strengths, target, avoided):
    """Return score and its derivative with respect to supplied mass fractions."""
    shapes, strengths, target = (np.asarray(x, float) for x in (shapes, strengths, target))
    basis = shapes*strengths[:, None, None]
    target_norm = np.linalg.norm(target)
    if target_norm <= 0 or not np.isfinite(basis).all() or np.any(basis < 0):
        raise ValueError('finite nonnegative guidance support and positive target required')
    sums = basis.sum(axis=2)
    banned = basis[:, :, avoided].sum(axis=2)

    def evaluate(weights):
        weights = np.asarray(weights, float)
        if weights.shape != strengths.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('invalid guidance mass fractions')
        mass = float(weights@strengths)
        profile = np.einsum('n,nhd->hd', weights, basis)
        norm, total = np.linalg.norm(profile, axis=1), profile.sum(axis=1)
        if mass <= 0 or np.any(norm <= 0) or np.any(total <= 0):
            return 0., np.zeros_like(weights)
        cosine = (profile@target)/(norm*target_norm)
        derivative_cosine = target[None, :]/(norm[:, None]*target_norm)-cosine[:, None]*profile/(norm[:, None]**2)
        cosine_jac = np.einsum('hd,nhd->hn', derivative_cosine, basis)
        explained = total/mass
        explained_jac = (sums.T*mass-total[:, None]*strengths[None, :])/(mass**2)
        avoided_mass = weights@banned
        avoidance_jac = (banned.T*total[:, None]-avoided_mass[:, None]*sums.T)/(total[:, None]**2)
        raw = 100*(cosine*explained-avoided_mass/total)
        scores = np.clip(raw, 0., 100.)
        index = int(np.argmin(scores))
        jac = 100*(cosine_jac[index]*explained[index]+cosine[index]*explained_jac[index]-avoidance_jac[index])
        if raw[index] <= 0 or raw[index] >= 100:
            jac = np.zeros_like(weights)
        return float(scores[index]), jac
    return evaluate


@lru_cache(maxsize=2)
def product_guidance(core, product):
    if product == "perfume":
        return SharedPerfumeGuidance(core)
    if product != "body_lotion":
        raise ValueError("unknown recipe product")
    value = AtlasLotionGuidance(
        shared_views(core)[0], core.manifest["structures"], experimental=True
    )
    value.core = core
    value.runtime_manifest_sha256 = core.sha256
    bind_complete_reference_support(value,core)
    return value


def bind_complete_reference_support(provider,core,bank=None):
    if core.manifest.get('complete_reference_guidance') is not True:
        return
    from .lotion_reference_objective import load_configured_reference_bank
    bank=load_configured_reference_bank() if bank is None else bank
    if bank is None or bank.parent_sha256!=core.sha256:
        raise ValueError('complete reference support requires a bound source bank')
    provider.complete_reference_dimensions=tuple(x for x in SCENT_DIMENSIONS if x in bank.profiles)
    provider.complete_reference_sha256=bank.sha256
    provider.complete_reference_bank=bank
