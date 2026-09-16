"""Complete source-reference guidance without the lossy 146-to-15 routing.

This ordinal regularizer is separate from the dose/temporal physical objective.
Targets are frozen source profiles, never fitted to generated recipes.
"""

import numpy as np

from .formulation_guidance import SharedRecipeSession
from .lotion_reference_objective import load_configured_reference_bank
from .science import TemporalMixtureSimulator


def agreement_gradient(basis, strengths, targets, avoided):
    """Exact gradient for the minimum cosine, overlap and avoidance score."""
    basis = np.asarray(basis, float)
    strengths = np.asarray(strengths, float)
    targets = np.asarray(targets, float)
    masks = np.asarray(avoided, float)
    if (
        basis.ndim != 3
        or strengths.shape != (len(basis),)
        or targets.shape != basis.shape[1:]
        or masks.shape != targets.shape
        or any(
            not np.isfinite(v).all() or np.any(v < 0)
            for v in (basis, strengths, targets, masks)
        )
        or np.any(masks > 1)
        or np.any(targets.sum(-1) <= 0)
        or not np.allclose(targets.sum(-1), 1.0, atol=1e-6)
    ):
        raise ValueError(
            "aligned finite source basis and normalized positive targets required"
        )
    basis = basis * strengths[:, None, None]

    def evaluate(weights):
        w = np.asarray(weights, float)
        if w.shape != strengths.shape or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("finite nonnegative guidance mass fractions required")
        raw = np.einsum("n,nhd->hd", w, basis)
        totals = raw.sum(-1)
        if np.any(totals <= 0):
            return 0.0, np.zeros_like(w)
        prediction = raw / totals[:, None]
        norms = np.linalg.norm(prediction, axis=-1)
        qnorm = np.linalg.norm(targets, axis=-1)
        cosine = (prediction * targets).sum(-1) / (norms * qnorm)
        components = np.stack(
            (
                cosine,
                np.minimum(prediction, targets).sum(-1),
                1.0 - (prediction * masks).sum(-1),
            ),
            -1,
        )
        h, component = np.unravel_index(components.argmin(), components.shape)
        if component == 0:
            partial = (
                targets[h] / (norms[h] * qnorm[h])
                - cosine[h] * prediction[h] / norms[h] ** 2
            )
        elif component == 1:
            partial = (prediction[h] < targets[h]).astype(float)
        else:
            partial = -masks[h]
        gradient = (
            basis[:, h] @ partial - (prediction[h] @ partial) * basis[:, h].sum(-1)
        ) / totals[h]
        return float(100 * components[h, component]), 100 * gradient

    return evaluate


class FullReferenceSession(SharedRecipeSession):
    def __init__(self, provider, brief, bank=None):
        super().__init__(provider, brief)
        self.bank = (
            (
                getattr(provider, "complete_reference_bank", None)
                or load_configured_reference_bank()
            )
            if bank is None
            else bank
        )
        if (
            self.bank is None
            or tuple(provider.endpoints) != self.bank.endpoints
            or self.bank.parent_sha256 != provider.core.sha256
        ):
            raise ValueError(
                "full reference guidance requires its exact shared checkpoint binding"
            )
        rows = [
            {
                "phase": TemporalMixtureSimulator._phase_for_time(t),
                "target_profile": dict(
                    zip(provider.coarse_dimensions, target.tolist())
                ),
                "avoided": avoided,
            }
            for t, (target, _, avoided) in zip(
                (0, 15, 60, 240, 480), TemporalMixtureSimulator.targets_by_time(brief)
            )
        ]
        targets, unsupported = self.bank.targets(brief, rows)
        # Absolute intensity and persistence are evaluated by product physics,
        # not falsely interpreted as a missing ordinal odor descriptor.
        excluded = {
            "requested_absolute_intensity_not_calibrated",
            "requested_persistence_not_calibrated",
            "explicit_19_axis_profile_requires_legacy_profile_mode",
        }
        self.unsupported_intent = [x for x in unsupported if x not in excluded]
        self.supported_intent = [
            x for x in brief.desired_dimensions if x in self.bank.profiles
        ]
        self.enabled = (
            self.product_supported
            and not self.unsupported_intent
            and all(x is not None for x in targets)
        )
        self.reference_targets = targets
        self.target_matrix = (
            np.concatenate([row["profiles"] for row in targets], axis=0)
            if self.enabled
            else None
        )
        self.avoidance = np.zeros_like(self.target_matrix) if self.enabled else None
        if self.enabled:
            for t, row in enumerate(targets):
                ids = [self.bank.endpoints.index(x) for x in row["avoided"]]
                self.avoidance[2 * t : 2 * t + 2, ids] = 1.0

    def _basis(self, ingredients):
        self.shapes.prefetch(ingredients)
        rows = [self.shapes.shape(item) for item in ingredients]
        if any(row is None for row in rows):
            self.missing_ids.update(
                item.ingredient_id
                for item, row in zip(ingredients, rows)
                if row is None
            )
            return None
        return np.tile(np.stack(rows), (1, 5, 1))

    def gradient_function(self, ingredients):
        if not self.enabled:
            return None
        basis = self._basis(ingredients)
        if basis is None:
            return None
        strengths = np.array([i.active_strength_percent / 100 for i in ingredients])
        return agreement_gradient(basis, strengths, self.target_matrix, self.avoidance)

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
        positive = weights > 0
        selected = [item for item, ok in zip(ingredients, positive) if ok]
        function = self.gradient_function(selected)
        if function is None:
            return None
        score, _ = function(weights[positive] / 100)
        self.exact_calls += int(exact)
        self.grid_calls += int(not exact)
        return {
            "score": score,
            "nominal_score": score,
            "checkpoint_sha256": self.provider.core.sha256,
            "reference_sha256": self.bank.sha256,
            "exact_forward": exact,
            "descriptor_count": len(self.bank.endpoints),
            "prediction_kind": "full_source_ordinal_component_regularizer_not_physical_sensory_accuracy",
            "target_fit_to_candidate_catalogue": False,
            "unmentioned_endpoints_forced_to_zero": False,
        }

    def transfer_scores(
        self, original, ingredients, donor_id, replacements, amounts_percent
    ):
        if not self.enabled:
            return None
        amounts = np.asarray(amounts_percent, float)
        if (
            donor_id not in original
            or amounts.shape != (len(replacements),)
            or not np.isfinite(amounts).all()
            or np.any((amounts < 0) | (amounts > original[donor_id] + 1e-10))
        ):
            raise ValueError("invalid reference transfer")
        lookup = {i.ingredient_id: i for i in ingredients}
        source = [lookup[key] for key in original]
        source_basis = self._basis(source)
        if source_basis is None:
            return np.full(len(replacements), np.nan)
        self.shapes.prefetch(replacements)
        result = np.full(len(replacements), np.nan)
        masses = np.array(list(original.values())) / 100
        donor = list(original).index(donor_id)
        strengths = np.array([i.active_strength_percent / 100 for i in source])
        total = np.einsum("n,nhd->hd", masses * strengths, source_basis)
        # Bounded vectorized scoring; all eligible receivers are retained.
        for start in range(0, len(replacements), 128):
            covered = [
                (j, item, self.shapes.shape(item))
                for j, item in enumerate(replacements[start : start + 128], start)
            ]
            covered = [row for row in covered if row[2] is not None]
            if not covered:
                continue
            ids = np.array([row[0] for row in covered])
            shapes = np.tile(np.stack([row[2] for row in covered]), (1, 5, 1))
            removed = amounts[ids] / 100 * strengths[donor]
            added = (
                amounts[ids]
                / 100
                * np.array([row[1].active_strength_percent / 100 for row in covered])
            )
            raw = (
                total[None]
                - removed[:, None, None] * source_basis[donor]
                + added[:, None, None] * shapes
            )
            if raw.min() < -1e-10:
                raise ValueError("negative mass in source reference transfer")
            raw = np.maximum(raw, 0.0)
            prediction = raw / np.maximum(raw.sum(-1, keepdims=True), 1e-30)
            cosine = (prediction * self.target_matrix).sum(-1) / np.maximum(
                np.linalg.norm(prediction, axis=-1)
                * np.linalg.norm(self.target_matrix, axis=-1),
                1e-30,
            )
            overlap = np.minimum(prediction, self.target_matrix).sum(-1)
            avoidance = 1.0 - (prediction * self.avoidance).sum(-1)
            result[ids] = 100 * np.minimum(np.minimum(cosine, overlap), avoidance).min(
                -1
            )
        return result

    def replacement_scores(self, original, ingredients, donor_id, replacements):
        return self.transfer_scores(
            original,
            ingredients,
            donor_id,
            replacements,
            np.full(len(replacements), original[donor_id]),
        )

    def report(self, *args, **kwargs):
        value = super().report(*args, **kwargs)
        value.update(
            guidance_version="complete-source-reference-guidance/v76",
            target_reference_sha256=self.bank.sha256,
            full_descriptor_count=len(self.bank.endpoints),
            lossy_coarse_projection_used=False,
            physical_objective_separate=True,
        )
        return value
