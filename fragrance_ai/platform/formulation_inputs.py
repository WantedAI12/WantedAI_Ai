"""Source-labelled quantitative formulation constraints, not approvals."""
from datetime import date
from typing import Literal
from pydantic import Field, model_validator
from .lotion_inputs import LotionInput


class HydrolysisParameters(LotionInput):
    ingredient_id: str = Field(min_length=1, max_length=300)
    neutral_per_min: float = Field(ge=0, le=1e6)
    acid_m_inv_min: float = Field(ge=0, le=1e12)
    base_m_inv_min: float = Field(ge=0, le=1e12)
    temperature_c: Literal[25] = 25
    minimum_ph: float = Field(ge=0, le=14)
    maximum_ph: float = Field(ge=0, le=14)
    source_reference: str = Field(min_length=1, max_length=1000)
    source_kind: Literal["measured", "estimated", "simulated"]
    source_date: date

    @model_validator(mode="after")
    def domain(self):
        if self.minimum_ph > self.maximum_ph or self.source_date > date.today():
            raise ValueError("invalid kinetic pH domain or future source date")
        return self


class StorageConditions(LotionInput):
    ph: float = Field(ge=0, le=14)
    days: float = Field(ge=0, le=3650)
    temperature_c: Literal[25] = 25
    minimum_parent_retention_percent: float | None = Field(default=None, gt=0, le=100)
    kinetics: list[HydrolysisParameters] = Field(default_factory=list, max_length=50000)

    @model_validator(mode="after")
    def sources(self):
        ids = [row.ingredient_id for row in self.kinetics]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate hydrolysis ingredient ID")
        if any(not row.minimum_ph <= self.ph <= row.maximum_ph for row in self.kinetics):
            raise ValueError("requested pH lies outside a supplied kinetic domain")
        return self


class SkinExposureConditions(LotionInput):
    skin_type: Literal["all", "normal", "dry", "oily", "combination", "sensitive"] = "all"
    area_cm2: float = Field(default=1000., gt=0, le=30000)
    body_mass_kg: float = Field(default=60., gt=0, le=500)
    applications_per_day: float = Field(default=1., gt=0, le=24)
    maximum_modeled_uptake_mg_cm2: float | None = Field(default=None, gt=0, le=100)


class LotionBaseDesignOptions(LotionInput):
    """Opt-in research design space, not a supplier-validated stability range."""
    additional_oil_base_percents: list[float] = Field(default_factory=lambda: [20., 30.], min_length=1, max_length=3)
    adaptive_oil_refinement_steps: int = Field(default=3, strict=True, ge=0, le=6)

    @model_validator(mode="after")
    def candidates(self):
        import math
        values = self.additional_oil_base_percents
        if (any(not math.isfinite(v) or not 5 <= v <= 30 or v == 10 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("unique additional oil fractions in [5,30], excluding the fixed 10 percent baseline, required")
        return self
