from pydantic import Field, model_validator
from .process_inputs import ProcessInput


class ObservedLiquidRequest(ProcessInput):
    reference_protocol: str
    ingredient_percent: dict[str, float] = Field(min_length=1, max_length=18)

    @model_validator(mode="before")
    @classmethod
    def finite_percentages(cls, value):
        percentages = (
            value.get("ingredient_percent") if isinstance(value, dict) else None
        )
        if isinstance(percentages, dict) and any(
            isinstance(x, bool) for x in percentages.values()
        ):
            raise ValueError("ingredient percentages cannot be booleans")
        return value


class ObservedPairRequest(ProcessInput):
    first_smiles: str = Field(min_length=1, max_length=4096)
    second_smiles: str = Field(min_length=1, max_length=4096)
