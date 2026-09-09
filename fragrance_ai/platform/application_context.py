"""Validate matrix/application data without granting a model qualification."""
from datetime import date
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def reject_numeric_booleans(cls, values):
        numeric = {"mass_percent", "minutes", "signal", "product_density_g_ml", "application_mass_mg_cm2",
                   "temperature_c", "relative_humidity_percent", "fragrance_concentration_percent"}
        if isinstance(values, dict) and any(isinstance(values.get(key), bool) for key in numeric):
            raise ValueError("numeric application fields cannot be booleans")
        return values


class BaseComponent(ContextInput):
    name: str = Field(min_length=1, max_length=200)
    mass_percent: float = Field(gt=0, le=100)
    role: Literal["water", "oil", "emulsifier", "humectant", "preservative", "other"]


class ReleasePoint(ContextInput):
    minutes: float = Field(ge=0, le=43200)
    signal: float = Field(ge=0, le=1e12)


class ReleaseSeries(ContextInput):
    source_reference: str = Field(min_length=1, max_length=1000)
    source_date: date
    source_kind: Literal["measured", "simulated"]
    signal_unit: Literal["ppm", "ppb", "instrument_units"]
    points: list[ReleasePoint] = Field(min_length=2, max_length=2048)

    @model_validator(mode="after")
    def check_time(self):
        if not self.source_reference.strip():
            raise ValueError("release source reference cannot be blank")
        if self.points[0].minutes != 0 or any(b.minutes <= a.minutes for a, b in zip(self.points, self.points[1:])):
            raise ValueError("release series must start at zero and have strictly increasing times")
        if self.source_date > date.today():
            raise ValueError("release source date cannot be in the future")
        return self


class ApplicationContext(ContextInput):
    product_type: Literal["body_lotion"] = "body_lotion"
    emulsion_type: Literal["oil_in_water", "water_in_oil"] | None = None
    # Percentages describe the fragrance-free base, not the finished formula.
    base_components: list[BaseComponent] | None = Field(default=None, min_length=1, max_length=200)
    product_density_g_ml: float | None = Field(default=None, gt=0, le=10)
    application_mass_mg_cm2: float | None = Field(default=None, gt=0, le=1e6)
    temperature_c: float | None = Field(default=None, ge=-20, le=80)
    relative_humidity_percent: float | None = Field(default=None, ge=0, le=100)
    substrate: Literal["skin", "skin_model", "fabric", "inert_surface"] | None = None
    fragrance_concentration_percent: float | None = Field(default=None, gt=0, le=30)
    formula_reference: str | None = Field(default=None, min_length=1, max_length=500)
    release_series: ReleaseSeries | None = None

    @model_validator(mode="after")
    def composition(self):
        if self.formula_reference is not None and not self.formula_reference.strip():
            raise ValueError("formula reference cannot be blank")
        if self.base_components:
            names = [row.name.strip().casefold() for row in self.base_components]
            if "" in names or len(names) != len(set(names)):
                raise ValueError("base component names must be nonempty and unique")
            if abs(sum(row.mass_percent for row in self.base_components) - 100) > .001:
                raise ValueError("fragrance-free base percentages must sum to 100")
        return self


def assess_application_context(context):
    required = ("emulsion_type", "base_components", "product_density_g_ml", "application_mass_mg_cm2", "temperature_c",
                "relative_humidity_percent", "substrate", "fragrance_concentration_percent", "formula_reference", "release_series")
    missing = [key for key in required if getattr(context, key) is None]
    data = context.model_dump(mode="json")
    return {"schema_version": "application-context-1", "context_id": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            "product_type": context.product_type, "status": "needs_input" if missing else "ready_for_model_data_review",
            "missing_fields": missing, "base_composition_basis": "fragrance_free_base_w_w_percent",
            "release_point_count": len(context.release_series.points) if context.release_series else 0,
            "caller_declared_source_kind": context.release_series.source_kind if context.release_series else None,
            "source_verified": False, "body_lotion_prediction_enabled": False,
            "human_detection_calibration_available": False,
            "scope": "schema_and_consistency_checks_only_not_model_training_or_qualification"}
