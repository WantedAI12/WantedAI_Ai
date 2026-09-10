"""Bounded answer application and explicit support boundaries for UI conditions."""
from datetime import date
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProductPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    skin_type: Literal["all", "normal", "dry", "oily", "combination", "sensitive"] | None = None
    ph_minimum: float | None = Field(default=None, strict=True, ge=0, le=14)
    ph_maximum: float | None = Field(default=None, strict=True, ge=0, le=14)
    note_transition_speed: Literal["slow", "medium", "fast"] | None = None
    skin_residual_required: bool | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def ordered_ph(self):
        if (self.ph_minimum is None) != (self.ph_maximum is None):
            raise ValueError("pH minimum and maximum must be provided together")
        if self.ph_minimum is not None and self.ph_minimum > self.ph_maximum:
            raise ValueError("pH minimum must not exceed maximum")
        return self


def unsupported_preferences(preferences):
    """Receiving a preference does not imply that the engine can enforce it."""
    data = preferences.model_dump(exclude_none=True)
    checks = (
        ("skin_type", preferences.skin_type not in (None, "all"), "skin_compatibility_model_missing", "피부 유형별 적합성 모델은 아직 연결되지 않았습니다."),
        ("ph_minimum", preferences.ph_minimum is not None, "ph_compatibility_model_missing", "pH 범위는 기록되지만 제형 안정성·향료 적합성 계산은 지원하지 않습니다."),
        ("note_transition_speed", preferences.note_transition_speed is not None, "transition_speed_control_missing", "시간대별 목표 향조는 지원하지만 전환 속도의 직접 제어는 지원하지 않습니다."),
        ("skin_residual_required", preferences.skin_residual_required is not None, "skin_residual_model_missing", "피부 잔향의 유무를 보장하는 모델은 연결되지 않았습니다."),
    )
    return [{"code": code, "field": "product_preferences." + field,
             "value": data.get(field), "message": message} for field, active, code, message in checks if active]


def apply_question_answers(original: dict, questions: list, answers: dict[str, Any]):
    """Only current question IDs can update their explicit input fields."""
    if not answers or len(answers) > 32:
        raise ValueError("provide between 1 and 32 answers")
    if len(json.dumps(answers, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 16384:
        raise ValueError("answers exceed 16 KiB")
    allowed = {q["id"]: q for q in questions}
    unknown = set(answers) - set(allowed)
    if unknown:
        raise ValueError("answer IDs must belong to the current questions: " + ", ".join(sorted(unknown)))
    # A JSON round trip also prevents mutation of the caller's nested objects.
    result = json.loads(json.dumps(original, allow_nan=False))
    for key, answer in answers.items():
        kind = allowed[key]["input_type"]
        if kind == "number":
            if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
                raise ValueError(key + " requires a finite number")
        elif kind in ("text", "date", "budget_scope"):
            if not isinstance(answer, str) or not answer.strip() or len(answer) > 2000:
                raise ValueError(key + " requires nonempty text of at most 2000 characters")
            answer = answer.strip()
            if kind == "date":
                if len(answer) != 10 or date.fromisoformat(answer).isoformat() != answer:
                    raise ValueError(key + " requires YYYY-MM-DD")
            if kind == "budget_scope" and answer not in ("fragrance_only", "finished_product_materials"):
                raise ValueError("unknown budget scope")
        elif kind in ("budget", "percent_range"):
            if not isinstance(answer, dict):
                raise ValueError(key + " requires an object")
            numeric = {"minimum", "maximum", "amount", "per_volume_ml", "product_density_g_ml",
                       "base_material_cost_krw", "krw_per_catalog_price_unit"}
            if any(k in numeric and (isinstance(v, bool) or not isinstance(v, (float, int))
                                     or not math.isfinite(v)) for k, v in answer.items()):
                raise ValueError(key + " requires finite numeric values")
        else:
            raise ValueError("unsupported answer type")
        # This allowlist is independent of the question producer: no arbitrary
        # paths or changes to quality/safety policy can enter through answers.
        if key in ("budget", "concentration_range"):
            result[key] = answer
        elif key == "formula.brief":
            result["formula"]["brief"] = answer
        elif key in {"budget." + name for name in (
            "scope", "product_density_g_ml", "krw_per_catalog_price_unit", "price_basis_reference",
            "price_as_of", "base_material_cost_krw")}:
            result["budget"][key.split(".")[1]] = answer
        else:
            raise ValueError("question does not have a supported answer mapping")
    return result
