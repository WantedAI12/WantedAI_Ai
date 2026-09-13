"""Manufacturing conditions are distinct from storage/application conditions."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .application_context import ApplicationContext


class ProcessInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def no_numeric_booleans(cls, values):
        if isinstance(values, dict) and any(isinstance(value, bool) for value in values.values()):
            raise ValueError("process inputs cannot be booleans")
        return values


class ManufacturingProcess(ProcessInput):
    method: Literal["auto", "cold", "hot"] = "auto"
    batch_mass_g: float | None = Field(default=None, gt=0, le=1e7)
    peak_temperature_c: float | None = Field(default=None, ge=0, le=150)
    mixing_minutes: float | None = Field(default=None, gt=0, le=1440)
    mixer_kind: Literal["high_shear", "overhead", "manual"] | None = None
    measured_ph: float | None = Field(default=None, ge=0, le=14)
    fragrance_addition_temperature_c: float | None = Field(default=None, ge=0, le=150)

    @model_validator(mode="after")
    def temperatures(self):
        if (self.peak_temperature_c is not None and self.fragrance_addition_temperature_c is not None
                and self.fragrance_addition_temperature_c > self.peak_temperature_c):
            raise ValueError("fragrance addition temperature exceeds process peak")
        return self


class WorkflowRequest(ProcessInput):
    product_type: Literal["perfume", "body_lotion"]
    application_context: ApplicationContext | None = None
    process: ManufacturingProcess = Field(default_factory=ManufacturingProcess)
    completed_step_ids: list[str] = Field(default_factory=list, max_length=512)

    @model_validator(mode='after')
    def known_completed_steps(self):
        from ..recommender.formulation_process import ACTIONS
        if any(step not in ACTIONS for step in self.completed_step_ids):
            raise ValueError('unknown manufacturing history action')
        return self

    @model_validator(mode="after")
    def product_domain(self):
        if self.product_type == "perfume" and self.application_context is not None:
            raise ValueError("a lotion application context cannot be used for perfume")
        if self.product_type == "perfume" and self.process.method != "auto":
            raise ValueError("cold/hot emulsion methods are only defined for body lotion")
        return self
