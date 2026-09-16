"""Typed, stateless R&D answers. Never mark unsubmitted defaults as confirmed."""
import json
import math

from .clarification import apply_question_answers


REQUIRED_FORMULA_FIELDS = {
    "product_category": {"input_type": "choice", "label": "사용할 제품", "unit": None},
    "target_region": {"input_type": "choice", "label": "대상 지역", "unit": None,
                      "options": [{"value": key, "label": key} for key in ("EU", "KR", "US")]},
    "product_concentration_percent": {"input_type": "number", "label": "완제품의 향료 농도",
                                      "unit": "w/w_percent", "exclusive_minimum": 0, "maximum": 30},
    "max_formula_cost_per_kg": {"input_type": "number", "label": "향료 농축액 원가 상한",
                                "unit": "USD/kg_concentrate", "exclusive_minimum": 0},
}


def question_for(field, *, conflict=False, product_codes=()):
    info = dict(REQUIRED_FORMULA_FIELDS[field])
    label = info.pop("label")
    if field == "product_category":
        info["options"] = [{"value": key, "label": key} for key in sorted(product_codes)]
    path = "request.formula." + field
    return {"id": path, "field": path, "required": True, **info,
            "reason_code": "text_and_structured_value_conflict" if conflict else "explicit_value_missing",
            "question": f"{label}를 확인해 주세요." + (" 원문과 입력값이 서로 다릅니다." if conflict else "")}


def answer_rd_questions(original, questions, answers):
    if not answers or len(answers) > 32:
        raise ValueError("provide between 1 and 32 answers")
    if len(json.dumps(answers, ensure_ascii=False, allow_nan=False).encode()) > 16384:
        raise ValueError("answers exceed 16 KiB")
    allowed = {row["id"]: row for row in questions}
    if set(answers) - set(allowed):
        raise ValueError("answer IDs must belong to the current R&D questions")
    output = json.loads(json.dumps(original, allow_nan=False))
    legacy = {}
    for key, answer in answers.items():
        if key.startswith('evidence_policy.'):
            from .rd_policy import POLICY_QUESTIONS, ReviewEvidencePolicy
            field = key.removeprefix('evidence_policy.')
            if field not in POLICY_QUESTIONS:
                raise ValueError('unsupported evidence policy answer')
            parsed = ReviewEvidencePolicy.model_validate({field: answer})
            if getattr(parsed, field) is None:
                raise ValueError('policy answer cannot be null')
            output.setdefault('evidence_policy', {})[field] = getattr(parsed, field)
            continue
        if not key.startswith("request.formula."):
            legacy[key] = answer
            continue
        field = key.removeprefix("request.formula.")
        if field not in REQUIRED_FORMULA_FIELDS:
            raise ValueError("unsupported R&D answer field")
        question = allowed[key]
        if question["input_type"] == "number":
            if (isinstance(answer, bool) or not isinstance(answer, (int, float))
                    or not math.isfinite(answer) or answer <= 0
                    or answer > question.get("maximum", 1e7)):
                raise ValueError(key + " requires a finite positive number within its range")
        else:
            choices = {row["value"] for row in question.get("options", [])}
            if not isinstance(answer, str) or answer not in choices:
                raise ValueError(key + " requires one of the advertised choices")
        output.setdefault("request", {}).setdefault("formula", {})[field] = answer
    if legacy:
        output["request"] = apply_question_answers(output["request"], questions, legacy)
    return output
