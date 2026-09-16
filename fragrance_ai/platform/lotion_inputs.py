"""Explicit, source-labelled inputs for the finite-dose lotion research model."""
from datetime import date
from typing import Literal
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .application_context import ApplicationContext, assess_application_context


class LotionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def no_boolean_numbers(cls, values):
        if isinstance(values, dict) and any(isinstance(value, bool) for value in values.values()):
            raise ValueError("lotion numeric fields cannot be booleans")
        times = values.get("times_minutes") if isinstance(values, dict) else None
        if isinstance(times, (list, tuple)) and any(isinstance(value, bool) for value in times):
            raise ValueError("simulation times cannot be booleans")
        return values


class LotionTargetInput(LotionInput):
    evaluation_mode: Literal["auto", "legacy_profile", "observed_reference"] = "auto"
    target_profile: dict[str, float] | None = None
    phase_target_profiles: dict[str, dict[str, float]] | None = None

    @model_validator(mode='before')
    @classmethod
    def validate_target_controls(cls, values):
        from ..recommender.intent_controls import normalized_target
        if isinstance(values, dict):
            if values.get('target_profile') is not None:
                normalized_target(values['target_profile'])
            phases = values.get('phase_target_profiles')
            if phases is not None:
                if not isinstance(phases, dict) or not phases or set(phases)-{'opening','heart','drydown'}:
                    raise ValueError('phase targets require opening, heart or drydown')
                for profile in phases.values():
                    normalized_target(profile)
        return values


class PhasePortion(LotionInput):
    compartment: Literal["water", "aqueous_nonvolatile", "lipid"]
    mass_percent: float = Field(gt=0, le=100)
    density_g_ml: float = Field(gt=0.1, le=10)


class PhaseComponent(LotionInput):
    name: str = Field(min_length=1, max_length=200)
    phase: Literal["aqueous", "lipid"] | None = None
    density_g_ml: float | None = Field(default=None, gt=0.1, le=10)
    # Fractions are of THIS supplied component, not of the entire base.
    portions: list[PhasePortion] | None = Field(default=None, min_length=2, max_length=3)
    source_reference: str | None = Field(default=None, min_length=1, max_length=1000)
    source_kind: Literal["measured", "estimated", "simulated"] | None = None

    @model_validator(mode="after")
    def explicit_assignment(self):
        if self.portions is None:
            if self.phase is None or self.density_g_ml is None:
                raise ValueError("provide phase and density, or documented component portions")
        else:
            if self.phase is not None or self.density_g_ml is not None:
                raise ValueError("component portions cannot be combined with a single phase/density")
            names = [part.compartment for part in self.portions]
            if len(set(names)) != len(names) or abs(sum(p.mass_percent for p in self.portions) - 100) > 1e-8:
                raise ValueError("unique component portions must sum to 100 percent")
            if not self.source_reference or self.source_kind is None:
                raise ValueError("split component assignment requires its source and evidence kind")
        return self


class TransportMaterial(LotionInput):
    ingredient_id: str = Field(min_length=1, max_length=300)
    concentrate_percent: float = Field(gt=0, le=100)
    # Concentration ratios in the actual lotion phases, NOT octanol logP.
    lipid_water_partition: float = Field(ge=1e-12, le=1e12)
    air_water_partition: float = Field(ge=1e-12, le=1e6)
    gas_transfer_cm_min: float = Field(ge=1e-12, le=1e4)
    skin_permeability_cm_min: float = Field(ge=0, le=1e3)
    odor_threshold_mg_m3: float | None = Field(default=None, ge=1e-12, le=1e12)
    initial_parent_fraction: float = Field(default=1., ge=0, le=1)
    aqueous_hydrolysis_per_min: float = Field(default=0., ge=0, le=1e6)
    reaction_source_reference: str | None = Field(default=None, min_length=1, max_length=1000)
    source_reference: str = Field(min_length=1, max_length=1000)
    source_date: date
    source_kind: Literal["measured", "estimated", "simulated"]

    @model_validator(mode="after")
    def source_not_future(self):
        if (self.initial_parent_fraction != 1 or self.aqueous_hydrolysis_per_min != 0) and not self.reaction_source_reference:
            raise ValueError("parent loss or hydrolysis requires kinetic provenance")
        if self.source_date > date.today():
            raise ValueError("transport source date cannot be in the future")
        return self


class LotionSimulationRequest(LotionInput):
    application_context: ApplicationContext
    # Bind supplied coefficients to this exact base/application/formula context.
    parameter_context_id: str = Field(min_length=64, max_length=64)
    phase_components: list[PhaseComponent] = Field(min_length=2, max_length=200)
    materials: list[TransportMaterial] = Field(min_length=1, max_length=50000)
    # Effective water loss coefficient at this context's temperature and RH.
    water_loss_per_min: float = Field(ge=0, le=10)
    retained_water_fraction: float = Field(ge=0, le=1)
    water_loss_source_reference: str = Field(min_length=1, max_length=1000)
    headspace_height_cm: float = Field(ge=1e-6, le=1e5)
    air_exchange_per_min: float = Field(ge=1e-9, le=1e5)
    times_minutes: list[float] = Field(default_factory=lambda: [0., 15., 30., 60., 120., 240., 480.], min_length=2, max_length=200)
    integration_step_minutes: float = Field(default=.25, ge=.01, le=1.)
    transport_mode: Literal["open_sink", "bidirectional_air"] = "open_sink"
    coefficient_scope: Literal["fixed_composition", "dilute_fixed_base"] = "fixed_composition"
    coefficient_scope_reference: str | None = Field(default=None, min_length=1, max_length=1000)
    profile_weighting: Literal["auto", "air_mass", "odor_activity"] = "auto"

    @model_validator(mode="after")
    def consistent(self):
        context = self.application_context
        assessment = assess_application_context(context)
        required = set(assessment["missing_fields"]) - {"release_series"}
        if required:
            raise ValueError("missing lotion context fields: " + ", ".join(sorted(required)))
        if context.application_mass_mg_cm2 < 1e-6 or context.fragrance_concentration_percent < 1e-6:
            raise ValueError("dose and concentration are below the numerical model range")
        if self.parameter_context_id != assessment["context_id"]:
            raise ValueError("transport parameters do not match application context; provide coefficients for this context")
        if context.emulsion_type != "oil_in_water":
            raise ValueError("this rapid-partition model currently supports oil-in-water emulsions only")
        names = [row.name.casefold() for row in self.phase_components]
        base = {row.name.strip().casefold(): row for row in context.base_components}
        if len(names) != len(set(names)) or set(names) != set(base):
            raise ValueError("phase assignment must cover every base component exactly once")
        phases = {("lipid" if part.compartment == "lipid" else "aqueous")
                  for row in self.phase_components for part in (row.portions or [])}
        phases.update(row.phase for row in self.phase_components if row.portions is None)
        if phases != {"aqueous", "lipid"}:
            raise ValueError("both aqueous and lipid phases are required")
        for row in self.phase_components:
            if base[row.name.casefold()].role == "water" and (row.phase != "aqueous" or row.portions is not None):
                raise ValueError("water must belong to the aqueous phase")
        ids = [row.ingredient_id for row in self.materials]
        if self.profile_weighting == "odor_activity" and any(row.odor_threshold_mg_m3 is None for row in self.materials):
            raise ValueError("odor-activity weighting requires every material's air odor threshold")
        if len(ids) != len(set(ids)) or abs(sum(row.concentrate_percent for row in self.materials) - 100) > .001:
            raise ValueError("unique concentrate ingredients summing to 100 percent are required")
        if context.substrate == "inert_surface" and any(row.skin_permeability_cm_min != 0 for row in self.materials):
            raise ValueError("inert surface cannot have skin absorption")
        if context.substrate == "fabric":
            raise ValueError("fabric sorption is outside the lotion skin model")
        times = self.times_minutes
        if any(not math.isfinite(t) or t < 0 or t > 1440 for t in times) or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("simulation times must start at zero and increase up to 1440 minutes")
        # Bound computational work, not the catalogue or recipe candidate pool.
        steps = sum(math.ceil((b-a) / self.integration_step_minutes) for a, b in zip(times, times[1:]))
        if (steps + len(times)) * len(self.materials) > 5_000_000:
            raise ValueError("simulation work budget exceeded; reduce duration/output points or use a coarser step")
        if len(times) * len(self.materials) > 100_000:
            raise ValueError("simulation response budget exceeded; request fewer output points")
        return self


class TransitionSchedule(LotionInput):
    opening_until_minutes: float = Field(default=15., gt=0, lt=1440)
    heart_until_minutes: float = Field(default=240., gt=0, lt=1440)

    @model_validator(mode="after")
    def ordered(self):
        if self.opening_until_minutes >= self.heart_until_minutes:
            raise ValueError("opening boundary must precede heart boundary")
        return self


class LotionOptimizationRequest(LotionTargetInput):
    simulation: LotionSimulationRequest
    brief: str = Field(min_length=3, max_length=2000)
    target_similarity: float = Field(default=95., ge=90., le=100.)
    search_goal: Literal["maximize", "reach_target"] = "maximize"
    # An explicit physical floor; NOT a calibrated human detection threshold.
    minimum_air_concentration_mg_m3: float = Field(ge=1e-12, le=1e9)
    max_formula_cost_per_kg: float = Field(default=180., gt=0, le=1e6)
    max_ingredient_price_per_kg: float = Field(default=300., gt=0, le=1e6)
    min_availability: float = Field(default=.75, ge=0, le=1)
    max_risk_tier: int = Field(default=1, strict=True, ge=1, le=2)
    registry_pool: Literal["core", "conditional_research"] = Field(default="conditional_research",
        description="Optional; omission searches the extended registry under the unchanged risk, price and safety constraints. Explicit core preserves the former limited pool.")
    excluded_ingredient_ids: list[str] = Field(default_factory=list, max_length=50000)
    transition_schedule: TransitionSchedule | None = None
    maximum_modeled_uptake_mg_cm2: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def scope_and_work(self):
        simulation = self.simulation
        if simulation.coefficient_scope != "dilute_fixed_base" or not simulation.coefficient_scope_reference:
            raise ValueError("optimization requires an explicit dilute-fixed-base coefficient scope reference")
        if simulation.transport_mode != "bidirectional_air":
            raise ValueError("lotion optimization requires bidirectional air transport")
        if (len(simulation.times_minutes)-1) * len(simulation.materials) * 19 > 2_000_000:
            raise ValueError("lotion optimization matrix budget exceeded; reduce output points")
        return self
