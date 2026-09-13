"""An explicit measured-domain request, not inferred lotion manufacturing data."""
from typing import Literal
from pydantic import Field
from .process_inputs import ProcessInput


class EmulsionCondition(ProcessInput):
    speed_rpm: float = Field(gt=0,le=1e6)
    dispersed_viscosity_pa_s: float = Field(gt=0,le=1e6)
    dispersed_density_kg_m3: float = Field(gt=0,le=1e5)
    continuous_viscosity_pa_s: float = Field(gt=0,le=1e6)
    continuous_density_kg_m3: float = Field(gt=0,le=1e5)
    interfacial_tension_mn_m: float = Field(gt=0,le=1e4)
    dispersed_volume_fraction: float = Field(gt=0,lt=1)


class EmulsionRequest(ProcessInput):
    reference_domain: Literal['rodgers_2025_stirred_tank_24h_silicone_emulsion']
    conditions: list[EmulsionCondition] = Field(min_length=1,max_length=128)

    def raw_rows(self):
        from ..recommender.emulsion_science import FIELDS
        return [[getattr(row,key) for key in FIELDS] for row in self.conditions]
