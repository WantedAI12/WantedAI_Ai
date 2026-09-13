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
    def __init__(self, core):
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

    def begin(self, brief):
        self.assert_current()
        if brief.constraints.validation_level != "prototype":
            raise ValueError("unpromoted shared model requires prototype evaluation")
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
            "prediction_kind": "shared_neural_component_shapes_with_additive_active_mass_proxy",
        }

    def score(self, weights, ingredients):
        value = self.evaluate(weights, ingredients)
        return None if value is None else value["score"]

    def evaluate_lines(self, lines, ingredients):
        return self.evaluate(
            [line.concentrate_percent for line in lines],
            [ingredients[line.ingredient_id] for line in lines],
            exact=True,
        )

    def report(self, baseline, selected, *, changed, variants):
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
        }


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
    return value
