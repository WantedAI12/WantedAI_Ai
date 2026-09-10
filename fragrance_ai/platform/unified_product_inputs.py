"""Explicit product/process conditioning for the shared V60 model."""
import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from .lotion_inputs import LotionInput


class ProductStageContext(LotionInput):
    stage_id: str = Field(min_length=1, max_length=100)
    duration_minutes: float = Field(gt=0, le=1440)
    formulation_reference: str = Field(min_length=1, max_length=1000)


class UnifiedProductContext(LotionInput):
    product_type: Literal['perfume', 'body_lotion', 'body_wash']
    application_mass_mg_cm2: float = Field(ge=1e-6, le=100.)
    fragrance_concentration_percent: float = Field(ge=1e-6, le=30.)
    temperature_c: float = Field(ge=0, le=60)
    relative_humidity_percent: float = Field(ge=0, le=100)
    headspace_height_cm: float = Field(ge=1e-6, le=1e5)
    stages: list[ProductStageContext] = Field(min_length=1, max_length=16)

    @model_validator(mode='after')
    def ordered_stages(self):
        names = [s.stage_id for s in self.stages]
        if len(set(names)) != len(names) or sum(s.duration_minutes for s in self.stages) > 1440:
            raise ValueError('unique ordered stages spanning at most 1440 minutes required')
        return self


def context_id(context):
    return hashlib.sha256(json.dumps(context.model_dump(mode='json'), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


class ProductComponent(LotionInput):
    ingredient_id: str = Field(min_length=1, max_length=300)
    concentrate_percent: float = Field(gt=0, le=100)
    odor_threshold_mg_m3: float | None = Field(default=None, ge=1e-12, le=1e12)
    initial_parent_fraction: float = Field(default=1., ge=0, le=1)
    parent_loss_source_reference: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode='after')
    def parent_source(self):
        if self.initial_parent_fraction != 1. and not self.parent_loss_source_reference:
            raise ValueError('initial parent loss requires a source reference')
        return self


class StageKinetics(LotionInput):
    ingredient_id: str = Field(min_length=1, max_length=300)
    # Initial stage rates are derived from the ACTUAL carrier/phase capacity.
    # Neither octanol logP nor ordinal Atlas levels may stand in for these.
    evaporation_per_min: float = Field(ge=0, le=1e8)
    uptake_per_min: float = Field(ge=0, le=1e8)
    hydrolysis_per_min: float = Field(ge=0, le=1e4)
    air_return_per_min: float = Field(ge=0, le=1e7)
    ventilation_per_min: float = Field(ge=0, le=1e7)
    capacity_decay_per_min: float = Field(ge=0, le=200)
    evaporating_capacity_fraction: float = Field(ge=0, le=.999)
    nonreactive_capacity_fraction: float = Field(ge=0, le=1)
    source_kind: Literal['measured', 'estimated', 'simulated']
    source_reference: str = Field(min_length=1, max_length=1000)

    @model_validator(mode='after')
    def capacity(self):
        if self.evaporating_capacity_fraction+self.nonreactive_capacity_fraction > 1.:
            raise ValueError('capacity fractions cannot exceed one')
        return self


class ProductStage(LotionInput):
    stage_id: str = Field(min_length=1, max_length=100)
    coefficients: list[StageKinetics] = Field(min_length=1, max_length=50000)
    # End-of-stage rinse jump. Retention is material-specific and supplied,
    # not predicted skin deposition and not silently assumed to be 100 percent.
    rinse_retained_film_fractions: dict[str, float] | None = None
    rinse_source_reference: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode='before')
    @classmethod
    def no_boolean_retention(cls, values):
        fractions = values.get('rinse_retained_film_fractions') if isinstance(values, dict) else None
        if isinstance(fractions, dict) and any(isinstance(v, bool) for v in fractions.values()):
            raise ValueError('rinse fractions cannot be booleans')
        return values

    @model_validator(mode='after')
    def rinse(self):
        fractions = self.rinse_retained_film_fractions
        if fractions is not None:
            import math
            if (not fractions or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
                                     for v in fractions.values()) or not self.rinse_source_reference):
                raise ValueError('rinse fractions require finite values in [0,1] and provenance')
        elif self.rinse_source_reference is not None:
            raise ValueError('rinse reference without a rinse event')
        return self


class UnifiedProductRequest(LotionInput):
    context: UnifiedProductContext
    parameter_context_id: str = Field(pattern='^[0-9a-f]{64}$')
    components: list[ProductComponent] = Field(min_length=1, max_length=50000)
    stages: list[ProductStage] = Field(min_length=1, max_length=16)
    times_minutes: list[float] = Field(min_length=2, max_length=200)
    brief: str | None = Field(default=None, min_length=1, max_length=2000)
    profile_weighting: Literal['auto', 'air_mass', 'odor_activity'] = 'auto'

    @model_validator(mode='after')
    def complete_context(self):
        import math
        if self.parameter_context_id != context_id(self.context):
            raise ValueError('kinetics do not match the exact product/process context')
        if [s.stage_id for s in self.stages] != [s.stage_id for s in self.context.stages]:
            raise ValueError('coefficient stages must match the ordered context stages')
        ids = [c.ingredient_id for c in self.components]
        if len(set(ids)) != len(ids) or abs(sum(c.concentrate_percent for c in self.components)-100.) > 1e-6:
            raise ValueError('unique concentrate components must sum to 100 percent')
        for stage in self.stages:
            names = [r.ingredient_id for r in stage.coefficients]
            if len(set(names)) != len(names) or set(names) != set(ids):
                raise ValueError('each stage must cover every material exactly once')
            if stage.rinse_retained_film_fractions is not None and set(stage.rinse_retained_film_fractions) != set(ids):
                raise ValueError('rinse retention must cover every material')
        has_rinse = any(s.rinse_retained_film_fractions is not None for s in self.stages)
        if has_rinse != (self.context.product_type == 'body_wash'):
            raise ValueError('a rinse event is required only for the body_wash product')
        times, duration = self.times_minutes, sum(s.duration_minutes for s in self.context.stages)
        if (times[0] != 0 or times[-1] != duration or any(not math.isfinite(t) for t in times)
                or any(b <= a for a, b in zip(times[:-1], times[1:]))):
            raise ValueError('display times must increase from zero through the final stage end')
        if len(ids)*(32*len(self.stages)+len(times)) > 2_000_000 or len(ids)*len(times) > 100_000:
            raise ValueError('unified simulation work/response budget exceeded')
        if self.profile_weighting == 'odor_activity' and any(c.odor_threshold_mg_m3 is None for c in self.components):
            raise ValueError('odor-activity weighting needs every material threshold')
        return self
