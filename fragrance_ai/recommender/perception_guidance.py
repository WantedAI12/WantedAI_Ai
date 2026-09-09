"""Experimental connection from frozen human-profile models to recipe search.

This is NOT the equal-volume DREAM mixture regressor: arbitrary perfume mass
fractions do not satisfy that experiment. We predict each component at its
finished active mass fraction, then average its RATA vector without multiplying
the dose a second time. This additive product-context extension is an unvalidated
search prior. It never supplies an actual-human score or a release approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from collections import OrderedDict
import threading

import numpy as np

from .models import Ingredient, RecipeResult, ScentBrief, SCENT_DIMENSIONS

MODEL_SHA256 = "5776e528cd0244b5a9aa99f2c5b44f0ae5014bdcba798658afc5af83bbd3913e"
REGISTRY_SHA256 = "d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765"
PROJECTION = {
    "citrus": ("Citrus",), "green": ("Green",), "floral": ("Floral",),
    "fruity": ("Fruity", "Tropical", "Berry", "Peach"), "spicy": ("BrownSpice",),
    "aromatic": ("Herbal", "Mint", "Pine"), "woody": ("Woody",),
    "gourmand": ("Sweet", "Caramellic", "Vanilla"), "powdery": ("Powdery",),
    "smoky": ("Smoky",), "earthy": ("Earthy",),
}
LOG_DOSES = np.linspace(-9.0, 0.0, 49)
DOSE_FACTORS = (0.90, 1.0, 1.10)


@dataclass
class PerceptionGuidedRecipeResult(RecipeResult):
    perception_guidance: dict = field(default_factory=dict)


def attach_guidance(result: RecipeResult, report: dict) -> RecipeResult:
    from .profile_match import FullProfileRecipeResult
    if isinstance(result, FullProfileRecipeResult):
        return replace(result, perception_guidance=report)
    return PerceptionGuidedRecipeResult(
        **{item.name: getattr(result, item.name) for item in fields(RecipeResult)},
        perception_guidance=report,
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PerceptionGuidance:
    """Explicit local research configuration; no model paths are accepted via HTTP."""

    def __init__(self, model_path: str | Path, registry_path: str | Path, *,
                 solvent: str, experimental: bool = False, weight: float = 0.25,
                 component_model_path: str | Path | None = None, component_model_sha256: str | None = None):
        from fragrance_ai.research.conditional_profiles import CARRIERS

        if not experimental:
            raise ValueError("human-profile recipe guidance requires explicit experimental opt-in")
        if solvent not in CARRIERS or solvent == "other":
            raise ValueError("an explicit supported solvent scenario is required")
        if not math.isfinite(weight) or not 0 < weight <= 0.5:
            raise ValueError("guidance weight must be in (0, 0.5]")
        model_path, registry_path = Path(model_path), Path(registry_path)
        if _sha(model_path) != MODEL_SHA256 or _sha(registry_path) != REGISTRY_SHA256:
            raise ValueError("perception model/registry hash mismatch")
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if (model.get("schema") != "conditional-profile-training-only/v2"
                or model.get("runtime_promotion_allowed") is not False
                or model.get("data_redistribution_authorized") is not False):
            raise ValueError("unexpected research model provenance contract")
        self.model = model["component_models"]["anchored"]
        self.component_model_sha256 = MODEL_SHA256
        self.component_model_version = 'v2'
        self.fine_odor_features = None
        if (component_model_path is None) != (component_model_sha256 is None):
            raise ValueError("component model requires a trusted hash and path together")
        if component_model_path is not None:
            path = Path(component_model_path)
            if _sha(path) != component_model_sha256:
                raise ValueError("component model hash mismatch")
            candidate = json.loads(path.read_text(encoding='utf-8'))
            if (candidate.get('schema') not in ('perception-core-candidate/v3', 'perception-core-candidate/v4', 'perception-core-candidate/v5')
                    or candidate.get('runtime_promotion_allowed') is not False
                    or candidate.get('data_redistribution_authorized') is not False
                    or candidate.get('profile_dimensions') != model['profile_dimensions']):
                raise ValueError('component candidate provenance mismatch')
            self.model = candidate['component_model']
            self.component_model_sha256 = component_model_sha256
            self.component_model_version = candidate['schema'].rsplit('/', 1)[1]
            if self.component_model_version == 'v5':
                from fragrance_ai.research.fine_odor_features import validate_features
                self.fine_odor_features = candidate.get('fine_odor_features')
                validate_features(self.fine_odor_features)
                if (self.model.get('feature_family') != 'source_bound_fine_v1'
                        or self.model['feature_width'] != 1106+len(self.fine_odor_features['vocabulary'])
                        or self.model['regressor'].get('kind') != 'molecular-kernel-v5'):
                    raise ValueError('V5 component feature contract mismatch')
        # Frozen routines accept arrays, avoiding repeated conversion of 56k
        # coefficients at every coordinate-search proposal.
        array_keys = ('support', 'scale', 'weights', 'intercept') if self.model['regressor'].get('kind') in ('molecular-kernel-v3', 'molecular-kernel-v4', 'molecular-kernel-v5') else ('center', 'scale', 'intercept', 'coefficients')
        for key in array_keys:
            value = np.asarray(self.model["regressor"][key], dtype=float)
            value.setflags(write=False)
            self.model["regressor"][key] = value
        self.bank = model["molecular_bank"]
        self.by_structure = {value["canonical_smiles"]: cid for cid, value in self.bank.items()}
        self.endpoints = tuple(model["profile_dimensions"])
        self.solvent, self.weight = solvent, weight
        self.structures = {}
        # Stock-reference shapes depend on checkpoint/material features, not
        # the user's brief. Bounded process-local reuse across lotion requests.
        self.lotion_shape_cache = OrderedDict()
        self.lotion_shape_cache_lock = threading.RLock()
        con = sqlite3.connect(registry_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for identifier, smiles in con.execute("SELECT registry_id, canonical_smiles FROM ingredients"):
                self.structures["registry_" + identifier.split(":", 1)[-1]] = (smiles, None)
            linked = con.execute("""
                SELECT f.ingredient_id, i.canonical_smiles, f.cas_number
                FROM formulation_materials f JOIN ingredients i ON i.registry_id=f.linked_registry_id
                WHERE f.cas_number IS NOT NULL AND EXISTS (
                    SELECT 1 FROM ingredient_identifiers x WHERE x.registry_id=i.registry_id
                    AND lower(x.identifier_type)='cas' AND x.identifier_value=f.cas_number)
            """).fetchall()
            for identifier, smiles, cas in linked:
                self.structures[identifier] = (smiles, cas)
        finally:
            con.close()

    def begin(self, brief: ScentBrief) -> "PerceptionSearchSession":
        if brief.constraints.validation_level != "prototype":
            raise ValueError("experimental perception guidance cannot be used for qualified/commercial release")
        return PerceptionSearchSession(self, brief)


class PerceptionSearchSession:
    """All mutable prediction caches and request data are per recipe request."""

    def __init__(self, provider: PerceptionGuidance, brief: ScentBrief):
        self.provider, self.brief = provider, brief
        self.dimensions = tuple(PROJECTION)
        self.supported_intent = [name for name in brief.desired_dimensions if name in PROJECTION]
        self.unsupported_intent = [name for name in brief.desired_dimensions if name not in PROJECTION]
        self.target = np.asarray([brief.target_profile.get(name, 0.0) for name in self.dimensions])
        self.target_coverage = float(self.target.sum())
        self.target /= max(float(self.target.sum()), 1e-12)
        self.target_norm = float(np.linalg.norm(self.target))
        self.avoid_indices = [self.dimensions.index(name) for name in brief.avoided_dimensions if name in self.dimensions]
        self.projection = np.zeros((len(provider.endpoints), len(self.dimensions) + 2))
        for index, dimension in enumerate(self.dimensions):
            for name in PROJECTION[dimension]:
                row = provider.endpoints.index(name)
                self.projection[row, index] = 1.0 / len(PROJECTION[dimension])
                self.projection[row, -1] = 1.0
        self.projection[:, -2] = 1.0  # total RATA mass, including unmapped notes
        self.product_supported = brief.constraints.product_category in {'eau_de_parfum', 'eau_de_toilette', 'eau_de_cologne'}
        self.enabled = self.product_supported and bool(self.supported_intent) and self.target_coverage > 0 and not brief.constraints.reference_target_id
        self.weight = provider.weight * min(1.0, self.target_coverage)
        self.prepared, self.tables = {}, {}
        self.grid_calls, self.exact_calls = 0, 0
        self.missing_ids = set()
        self.exact_cache = {}

    def supports(self, ingredient: Ingredient) -> bool:
        row = self.provider.structures.get(ingredient.ingredient_id)
        if row is None or (row[1] is not None and row[1] != ingredient.cas_number):
            return False
        return "." not in row[0]  # no unknown-ratio mixture/salt decomposition

    def _prepare(self, ingredient: Ingredient) -> tuple[str | None, dict] | None:
        from fragrance_ai.research.conditional_profiles import molecule_features

        key = ingredient.ingredient_id
        if key in self.prepared:
            return self.prepared[key]
        if not self.supports(ingredient):
            self.missing_ids.add(key)
            self.prepared[key] = None
            return None
        # Registry internal/negative representative IDs are not PubChem IDs.
        # Resolve anchors by exact stereochemical graph identity instead.
        smiles = self.provider.structures[key][0]
        value = molecule_features(smiles, {"profile": [float(ingredient.profile.get(name, 0.0)) for name in SCENT_DIMENSIONS],
                                            "odor_impact": ingredient.odor_impact,
                                            "ingredient_id": key})
        cid = self.provider.by_structure.get(value["canonical_smiles"])
        if cid is not None:
            value = self.provider.bank[cid]
        self.prepared[key] = (cid, value)
        return self.prepared[key]

    def _predict(self, ingredient: Ingredient, doses: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        from fragrance_ai.research.conditional_profiles import features_for, predict_conditional
        from fragrance_ai.research.kernel_profiles import predict_component_regressor

        prepared = self._prepare(ingredient)
        if prepared is None:
            raise ValueError("unsupported ingredient must not be treated as a zero profile")
        cid, molecule = prepared
        # '1' is a private feature-row index ONLY. No invented PubChem identity
        # is emitted and unknown graphs never enter the measured-anchor lookup.
        row_key = cid if cid is not None else "1"
        keys = [(row_key, str(Decimal(format(float(dose), ".12g")).normalize()), self.provider.solvent) for dose in doses]
        x = np.asarray([features_for(key, {row_key: molecule}) for key in keys])
        if getattr(self.provider, 'fine_odor_features', None) is not None:
            from fragrance_ai.research.fine_odor_features import append_features
            x = append_features(x, [molecule['canonical_smiles']]*len(keys), self.provider.fine_odor_features)
        if cid is None:
            prediction = predict_component_regressor(self.provider.model["regressor"], x)
            return prediction, [{"basis": "new_registry_graph_prediction", "anchor_distance_decades": None} for _ in keys]
        return predict_conditional(self.provider.model, x, keys)

    def _scores(self, projected: np.ndarray) -> np.ndarray:
        profile = projected[:, :len(self.dimensions)]
        denominator = np.maximum(np.linalg.norm(profile, axis=1) * self.target_norm, 1e-12)
        cosine = (profile @ self.target) / denominator
        # Never discard fishy/fecal/chemical or other unrepresented RATA mass
        # and then call the tiny remaining woody/citrus part a perfect match.
        explained_mass = projected[:, -1] / np.maximum(projected[:, -2], 1e-12)
        avoided = profile[:, self.avoid_indices].sum(axis=1) / np.maximum(profile.sum(axis=1), 1e-12)
        return np.clip(100.0 * cosine * explained_mass - 100.0 * avoided, 0.0, 100.0)

    def predict_stock_batch(self, ingredients, doses):
        """Bounded exact stock predictions; unknown graphs never get anchors."""
        from fragrance_ai.research.conditional_profiles import features_for, predict_conditional
        from fragrance_ai.research.kernel_profiles import predict_component_regressor
        doses = np.asarray(doses, dtype=float)
        if (len(ingredients) > 128 or doses.ndim != 1 or not 0 < len(doses) <= 49
                or not np.isfinite(doses).all() or np.any(doses <= 0) or np.any(doses > 1)):
            raise ValueError('invalid bounded stock batch')
        groups = {True: [], False: []}
        for i, item in enumerate(ingredients):
            prepared = self._prepare(item)
            if prepared is None:
                raise ValueError('unsupported ingredient cannot enter stock batch')
            cid, molecule = prepared
            row_key = cid if cid is not None else '1'
            for dose in doses:
                key = (row_key, str(Decimal(format(float(dose), '.12g')).normalize()), self.provider.solvent)
                groups[cid is not None].append((i, key, features_for(key, {row_key: molecule}), molecule['canonical_smiles']))
        predictions = [[] for _ in ingredients]
        statuses = [[] for _ in ingredients]
        for anchored, rows in groups.items():
            if not rows:
                continue
            x = np.asarray([row[2] for row in rows])
            if getattr(self.provider, 'fine_odor_features', None) is not None:
                from fragrance_ai.research.fine_odor_features import append_features
                x = append_features(x, [row[3] for row in rows], self.provider.fine_odor_features)
            if anchored:
                y, details = predict_conditional(self.provider.model, x, [row[1] for row in rows])
            else:
                y = predict_component_regressor(self.provider.model['regressor'], x)
                details = [{'basis': 'new_registry_graph_prediction', 'anchor_distance_decades': None} for _ in rows]
            self.stock_batch_forward_calls = getattr(self, 'stock_batch_forward_calls', 0)+1
            for row, value, detail in zip(rows, y, details):
                predictions[row[0]].append(value)
                statuses[row[0]].append(detail)
        return [(np.asarray(values), details) for values, details in zip(predictions, statuses)]

    def prepare_stock_molecules(self, ingredients):
        """Resolve explicit stocks, including exact known multi-graph anchors.

        A known assay stock is not decomposed into hypothetical ingredients.
        Unknown fragmented materials remain unsupported; perfume/lotion graph
        acceptance is unchanged.
        """
        result = []
        for item in ingredients:
            prepared = self._prepare(item)
            if prepared is None:
                row = self.provider.structures.get(item.ingredient_id)
                if row is not None and (row[1] is None or row[1] == item.cas_number):
                    cid = self.provider.by_structure.get(row[0])
                    if cid is not None:
                        prepared = (cid, self.provider.bank[cid])
            if prepared is None:
                raise ValueError('unsupported stock material cannot be removed or zero-filled')
            result.append(prepared)
        return result

    def predict_stock_conditions(self, ingredients, dilutions, solvents):
        """Paired stock conditions in two bounded forwards, not an N-by-N grid.

        This is assay input, distinct from finished-product mass fractions or
        a lotion gas concentration. Each stock carries its explicit solvent.
        """
        from fragrance_ai.research.conditional_profiles import CARRIERS, features_for, predict_conditional
        from fragrance_ai.research.kernel_profiles import predict_component_regressor
        doses = np.asarray(dilutions, float)
        if (not 0 < len(ingredients) <= 128 or doses.shape != (len(ingredients),)
                or len(solvents) != len(ingredients) or not np.isfinite(doses).all()
                or np.any(doses <= 0) or np.any(doses > 1)
                or any(s not in CARRIERS or s == 'other' for s in solvents)):
            raise ValueError('paired explicit stock conditions required')
        groups = {True: [], False: []}
        identities = [None]*len(ingredients)
        molecules = self.prepare_stock_molecules(ingredients)
        for index,(prepared,dose,solvent) in enumerate(zip(molecules,doses,solvents)):
            cid,molecule = prepared
            dilution = str(Decimal(format(float(dose), '.12g')).normalize())
            key = (cid if cid is not None else '1', dilution, solvent)
            identities[index] = (cid if cid is not None else 'smiles:'+molecule['canonical_smiles'],dilution,solvent)
            groups[cid is not None].append((index,key,features_for(key,{key[0]:molecule}),molecule['canonical_smiles']))
        prediction = np.empty((len(ingredients),len(self.provider.endpoints)))
        details = [None]*len(ingredients)
        for anchored,rows in groups.items():
            if not rows:
                continue
            x = np.asarray([r[2] for r in rows])
            if getattr(self.provider,'fine_odor_features',None) is not None:
                from fragrance_ai.research.fine_odor_features import append_features
                x = append_features(x,[r[3] for r in rows],self.provider.fine_odor_features)
            if anchored:
                values,basis = predict_conditional(self.provider.model,x,[r[1] for r in rows])
            else:
                values = predict_component_regressor(self.provider.model['regressor'],x)
                basis = [{'basis':'new_registry_graph_prediction','anchor_distance_decades':None} for _ in rows]
            self.stock_batch_forward_calls = getattr(self,'stock_batch_forward_calls',0)+1
            for row,value,status in zip(rows,values,basis):
                prediction[row[0]],details[row[0]] = value,status
        return prediction,details,identities

    def evaluate(self, weights, ingredients: list[Ingredient], *, exact: bool = False) -> dict | None:
        if not self.enabled:
            return None
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(ingredients),) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(f"invalid formula weights: shape={weights.shape}, min={weights.min() if weights.size else None}, total={weights.sum()}")
        if abs(float(weights.sum()) - 100.0) > 0.002:
            raise ValueError("formula mass fractions must sum to 100 percent")
        active = [(float(weight), item) for weight, item in zip(weights, ingredients) if weight > 1e-8]
        if not active or any(not self.supports(item) for _, item in active):
            self.missing_ids.update(item.ingredient_id for _, item in active if not self.supports(item))
            return None
        cache_key = tuple((item.ingredient_id, weight, item.active_strength_percent) for weight, item in active)
        if exact and cache_key in self.exact_cache:
            return self.exact_cache[cache_key]
        vectors = np.zeros((3, len(self.provider.endpoints) if exact else self.projection.shape[1]))
        basis = []
        for weight, ingredient in active:
            dose = weight / 100.0 * self.brief.constraints.product_concentration_percent / 100.0 * ingredient.active_strength_percent / 100.0
            if not 1e-9 <= dose <= 1:
                return None
            doses = np.clip(np.asarray(DOSE_FACTORS) * dose, 1e-9, 1.0)
            if exact:
                prediction, statuses = self._predict(ingredient, doses)
                basis.append({"ingredient_id": ingredient.ingredient_id, "active_fraction": dose, **statuses[1]})
            else:
                if ingredient.ingredient_id not in self.tables:
                    self.tables[ingredient.ingredient_id] = self._predict(ingredient, 10 ** LOG_DOSES)[0] @ self.projection
                table = self.tables[ingredient.ingredient_id]
                query = np.log10(doses)
                lower = np.clip(np.searchsorted(LOG_DOSES, query, side="right") - 1, 0, len(LOG_DOSES) - 2)
                fraction = ((query - LOG_DOSES[lower]) / (LOG_DOSES[lower + 1] - LOG_DOSES[lower]))[:, None]
                prediction = table[lower] * (1 - fraction) + table[lower + 1] * fraction
            vectors += prediction
        # Concentration already entered each component prediction; no second
        # multiplication by ingredient fraction or odor_impact is performed.
        vectors /= len(active)
        projected = vectors @ self.projection if exact else vectors
        scores = self._scores(projected)
        result = {"score": float(scores.min()), "nominal_score": float(scores[1]), "dose_sensitivity_scores": scores.tolist(),
                  "predicted_rata_profile": {name: float(value) for name, value in zip(self.provider.endpoints, vectors[1])} if exact else {},
                  "unexplained_rata_mass_fraction": float(1.0 - projected[1, -1] / max(projected[1, -2], 1e-12)),
                  "component_basis": basis, "exact_forward": exact}
        if exact:
            self.exact_calls += 1
            self.exact_cache[cache_key] = result
        else:
            self.grid_calls += 1
        return result

    def score(self, weights, ingredients: list[Ingredient]) -> float | None:
        result = self.evaluate(weights, ingredients)
        return None if result is None else result["score"]

    def evaluate_lines(self, lines, ingredient_map: dict) -> dict | None:
        return self.evaluate([line.concentrate_percent for line in lines], [ingredient_map[line.ingredient_id] for line in lines], exact=True)

    def report(self, baseline: dict | None, selected: dict | None, *, changed: bool, variants: int) -> dict:
        unsupported_avoid = [name for name in self.brief.avoided_dimensions if name not in PROJECTION]
        return {
            "status": "experimental_guidance_applied" if changed and selected is not None else "experimental_guidance_evaluated" if selected else "abstained_no_supported_formula_or_intent",
            "model_sha256": MODEL_SHA256, "registry_sha256": REGISTRY_SHA256,
            "component_model_sha256": getattr(self.provider, 'component_model_sha256', MODEL_SHA256),
            "supported_intent_dimensions": self.supported_intent, "unsupported_intent_dimensions": self.unsupported_intent,
            "unsupported_avoided_dimensions": unsupported_avoid,
            "model_application": {
                "evaluated": selected is not None,
                "product_supported_as_research_prior": self.product_supported,
                "all_requested_axes_supported": not self.unsupported_intent and not unsupported_avoid and bool(self.supported_intent),
                "phase_targets_supported": not bool(self.brief.phase_target_profiles),
                "full_model_application": selected is not None and not self.unsupported_intent and not unsupported_avoid
                    and bool(self.supported_intent) and not self.brief.phase_target_profiles and self.product_supported,
                "scope": "configured_research_prior_only_not_full_product_or_human_validation"},
            "target_mass_covered_percent": 100.0 * self.target_coverage, "applied_search_weight": self.weight,
            "baseline": baseline, "selected": selected, "recipe_changed": changed and selected is not None,
            "guidance_variants_added": variants, "grid_objective_calls": self.grid_calls, "exact_formula_evaluations": self.exact_calls,
            "unmapped_ingredient_ids": sorted(self.missing_ids), "solvent_scenario": self.provider.solvent,
            "solvent_scenario_is_verified_product_base": False,
            "score_kind": "projected_supported_intent_match_not_human_similarity",
            "mixing_basis": "unvalidated_final_active_mass_fraction_additive_proxy",
            "robustness_kind": "deterministic_dose_sensitivity_not_human_confidence_interval",
            "experimental_only": True, "time_evolution_validated": False, "absolute_intensity_validated": False,
            "actual_human_accuracy_90_authorized": False, "manufacturing_approval": False,
        }
