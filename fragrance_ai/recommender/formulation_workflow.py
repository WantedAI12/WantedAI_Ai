"""Source-bound procedural reasoning. Never changes a sensory prediction/score.

No network access, legacy harmony scores, or recipe text from a language model.
Source parameters remain reference conditions, not validated scale-up settings.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from ..platform.process_inputs import WorkflowRequest


ASSET = "formulation_process_knowledge_v1.json"
DATA = Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=1)
def _knowledge():
    raw = (DATA / ASSET).read_bytes()
    manifest = json.loads((DATA / "data_manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["assets"][ASSET]["sha256"]:
        raise ValueError("formulation knowledge asset hash mismatch")
    value = json.loads(raw)
    if value["schema_version"] != "formulation-process-knowledge/1":
        raise ValueError("unsupported formulation knowledge version")
    sources = {row["id"] for row in value["sources"]}
    if len(sources) != len(value["sources"]) or any(
            set(step["source_ids"]) - sources for steps in value["steps"].values() for step in steps):
        raise ValueError("invalid formulation knowledge source references")
    return value, digest


def knowledge_contract():
    data, digest = _knowledge()
    return {"schema_version": data["schema_version"], "sha256": digest,
            "source_count": len(data["sources"]), "reviewed_on": data["reviewed_on"],
            "evidence_kind": data["evidence_kind"], "runtime_external_api_calls": 0,
            **deepcopy(data["learning_contract"])}


def _name(value):
    return re.sub(r"[\s™®/\-]+", "", value.casefold())


# Exact aliases only: a claimed formula_reference is not evidence of composition.
_ALIASES = {
    "water": ("Distilled water", "water", "정제수"),
    "glycerin": ("Glycerin", "Glycerol", "글리세린"),
    "cct": ("Caprylic/Capric Triglyceride", "Lotioncrafter CCT", "CCT"),
    "simulgel": ("Simulgel EG", "Simulgel™ EG"),
    "pe9010": ("Lotioncrafter PE 9010", "PE 9010"),
    "hemp": ("Hydrolyzed Hemp",),
    "squalane": ("Olive Squalane", "Neossance Squalane", "Squalane"),
    "ipm": ("Lotioncrafter IPM", "Isopropyl Myristate"),
    "cetearyl": ("Cetearyl Alcohol",),
    "gms": ("Lotioncrafter GMS", "Glyceryl Stearate", "Glyceryl Monostearate"),
    "ceteareth20": ("Ceteareth 20", "Ceteareth-20"),
    "phenonip": ("Phenonip XB",),
}
_IDENTITIES = {_name(alias): key for key, aliases in _ALIASES.items() for alias in aliases}
_COLD_ROLES = {"water": "water", "glycerin": "humectant", "cct": "oil",
               "simulgel": "emulsifier", "pe9010": "preservative"}
_HOT_ROLES = {"water": "water", "glycerin": "humectant", "hemp": "other", "squalane": "oil",
              "ipm": "oil", "cetearyl": "other", "gms": "emulsifier", "ceteareth20": "emulsifier",
              "phenonip": "preservative"}
_HOT_SOURCE_PERCENT = {"water": 84.7, "glycerin": 2., "hemp": 2., "squalane": 5., "ipm": 2.,
                       "cetearyl": 1.5, "gms": .6, "ceteareth20": 1.2, "phenonip": .5}
_ELECTROLYTES = {_name(s) for s in ("Sodium Hyaluronate", "Sodium Lactate", "Sodium PCA",
                                   "소듐하이알루로네이트", "소듐락테이트", "소듐피씨에이")}


def _identify(context):
    rows, unknown, duplicate = {}, [], []
    for row in (context.base_components or ()) if context else ():
        key = _IDENTITIES.get(_name(row.name))
        if key is None:
            unknown.append(row.name)
        elif key in rows:
            duplicate.append(row.name)
        else:
            rows[key] = row
    if context is None or context.emulsion_type != "oil_in_water" or unknown or duplicate:
        return None, rows, unknown, duplicate
    if (set(rows) == set(_COLD_ROLES) and all(rows[k].role == role for k, role in _COLD_ROLES.items())
            and all(abs(rows[k].mass_percent - v) < .001 for k, v in {"glycerin": 3., "simulgel": 2., "pe9010": 1.}.items())
            and 5 <= rows["cct"].mass_percent <= 30
            and abs(rows["water"].mass_percent + rows["cct"].mass_percent - 94) < .001):
        # The project CCT base and its searched oil variants are adaptations.
        # Selecting a process family does not qualify their emulsion stability.
        return "cold", rows, unknown, duplicate
    if set(rows) == set(_HOT_ROLES) and all(rows[k].role == role for k, role in _HOT_ROLES.items()):
        if all(abs(rows[k].mass_percent - v / 99.5 * 100) < .001 for k, v in _HOT_SOURCE_PERCENT.items()):
            return "hot", rows, unknown, duplicate
    return None, rows, unknown, duplicate


def _batch(context, process, method):
    if context is None or not context.base_components or context.fragrance_concentration_percent is None:
        return None
    # ApplicationContext allows tiny sum rounding, not an unexplained mass gain.
    total = sum(row.mass_percent for row in context.base_components)
    fragrance = context.fragrance_concentration_percent
    lines = []
    for row in context.base_components:
        key = _IDENTITIES.get(_name(row.name))
        phase = None
        if method == "cold":
            phase = "A" if key in ("water", "glycerin", "pe9010") else "B"
        elif method == "hot":
            phase = "A" if key in ("water", "glycerin", "hemp") else "B"
        percent = row.mass_percent / total * (100 - fragrance)
        lines.append({"name": row.name, "role": row.role, "phase": phase,
                      "finished_product_percent": percent,
                      "mass_g": process.batch_mass_g * percent / 100 if process.batch_mass_g is not None else None})
    lines.append({"name": "Fragrance concentrate", "role": "fragrance", "phase": "B" if method == "cold" else "C" if method == "hot" else None,
                  "finished_product_percent": fragrance,
                  "mass_g": process.batch_mass_g * fragrance / 100 if process.batch_mass_g is not None else None})
    return {"basis": "finished_product_w_w_percent", "base_input_basis": "fragrance_free_base_w_w_percent",
            "base_input_total_percent": total, "base_roundoff_normalization_factor": 100 / total,
            "batch_mass_g": process.batch_mass_g, "lines": lines,
            "total_percent": sum(row["finished_product_percent"] for row in lines),
            "total_mass_g": sum(row["mass_g"] for row in lines) if process.batch_mass_g is not None else None,
            "scope": "mass_balance_only_not_a_manufacturing_approval"}


def formulation_workflow(request):
    request = WorkflowRequest.model_validate(request)
    data, _ = _knowledge()
    result = {"schema_version": "formulation-workflow/1", "product_type": request.product_type,
              "knowledge": knowledge_contract(), "process": request.process.model_dump(mode="json"),
              "status": "research_plan", "checks": [], "missing_fields": [],
              "selected_method": None, "reference_parameters": None, "batch": None,
              "component_identity_basis": "caller_declared_exact_name_alias_not_lot_verification",
              "manufacturing_approved": False, "human_similarity_percent": None,
              "sensory_score_adjustment": 0., "scale_up_validated": False}
    used = set()

    def check(code, severity, message, source_ids=()):
        result["checks"].append({"code": code, "severity": severity, "message": message, "source_ids": list(source_ids)})
        used.update(source_ids)

    if request.product_type == "perfume":
        steps = deepcopy(data["steps"]["perfume"])
        result["trial_policy"] = {"fixed_universal_note_percentages": None,
            "method": "preserve_incumbent_compare_bounded_accord_changes",
            "numeric_acceptance": "existing_product_model_and_strict_gates_only",
            "required_record_fields": ["brief", "baseline_formula", "candidate_formula", "active_strengths",
                "concentration", "solvent_or_base", "evaluation_times", "observation_kind", "observed_result", "source_reference"]}
        if any(getattr(request.process, key) is not None for key in
               ("peak_temperature_c", "mixing_minutes", "mixer_kind", "measured_ph", "fragrance_addition_temperature_c")):
            check("perfume_process_not_covered", "conflict", "향수의 열·혼합·pH 조건을 판정할 원료별 공정 자료가 연결되지 않았습니다.")
    else:
        context, process = request.application_context, request.process
        method, rows, unknown, duplicate = _identify(context)
        result["selected_method"] = method
        result["unrecognized_components"] = unknown
        if duplicate:
            check("duplicate_component_identity", "conflict", "별칭이 같은 원료가 중복되었습니다: " + ", ".join(duplicate))
        if method is None:
            result["status"] = "needs_matrix_data"
            result["missing_fields"].append("documented_process_for_exact_base")
            check("matrix_process_not_covered", "unresolved", "유화 형식과 실제 성분이 연결된 참조 공정에 맞지 않습니다. 참조명만으로 공정을 선택하지 않습니다.")
        else:
            result["reference_parameters"] = {**deepcopy(data["reference_parameters"][method]),
                "scope": "supplier_example_not_validated_operating_setpoints"}
            used.update(result["reference_parameters"]["source_ids"])
            if method == "cold":
                check("adapted_base_not_qualified", "unresolved", "CCT 베이스와 유상·향료량 변형은 프로젝트 적응형이며 공급사 원배합과 동일하지 않습니다.", ["lc_simulgel"])
            if process.method != "auto" and process.method != method:
                check("process_method_conflict", "conflict", "요청한 냉간·가열 방식이 성분에 연결된 공정과 다릅니다.", ["lc_simulgel" if method == "cold" else "lc_sprayable"])
            if method == "cold" and process.peak_temperature_c is not None:
                check("unvalidated_cold_process_temperature", "unresolved", "이 참조에는 가열 온도가 없습니다. 입력 온도를 검증된 공정값으로 사용하지 않습니다.", ["lc_simulgel"])
            if method == "cold" and process.mixer_kind not in (None, "high_shear"):
                check("mixer_conflict", "conflict", "이 참조 공정은 고전단 혼합이며 수동·일반 교반으로의 대체 조건은 없습니다.", ["lc_simulgel"])
            if method == "hot" and process.peak_temperature_c is not None:
                low, high = data["reference_parameters"]["hot"]["phase_temperature_range_c"]
                if not low <= process.peak_temperature_c <= high:
                    check("temperature_outside_reference", "conflict", "입력 온도가 선택된 공급사 가열 예시 범위를 벗어납니다.", ["lc_sprayable"])
            if method == "hot":
                check("fragrance_addition_condition_missing", "unresolved", "향료 투입 온도·시점이 원문에 없습니다. 입력값이 있어도 원료별 근거는 별도로 필요합니다.", ["lc_sprayable"])
                if context.fragrance_concentration_percent != .5:
                    check("hot_reference_dose_changed", "unresolved", "향료 농도가 공급사 예시와 다르거나 지정되지 않았습니다.", ["lc_sprayable"])
            if process.batch_mass_g != data["reference_parameters"][method]["batch_mass_g"]:
                check("batch_scale_not_qualified", "unresolved", "계량은 중량에 비례하지만 혼합 시간·전단·열 이력은 비례 환산할 수 없습니다.")
            if process.mixing_minutes is not None:
                check("mixing_time_not_qualified", "unresolved", "혼합 시간만으로 장비·용기·전단 조건의 동등성을 확인할 수 없습니다.")
        if "simulgel" in rows and context:
            incompatible = [row.name for row in context.base_components if _name(row.name) in _ELECTROLYTES]
            if incompatible:
                check("simulgel_electrolyte_conflict", "conflict", "참조에서 부적합으로 지목한 첨가제: " + ", ".join(incompatible), ["lc_simulgel"])
        batch = _batch(context, process, method)
        result["batch"] = batch
        if batch is None:
            result["missing_fields"].append("base_components_and_finished_fragrance_concentration")
        if "pe9010" in rows:
            limits = data["reference_parameters"]["pe9010"]
            used.add("lc_pe9010")
            ph = process.measured_ph
            low, high = limits["component_ph_range"]
            if ph is not None and not low <= ph <= high:
                check("pe9010_ph_outside_supplier_range", "conflict", "입력 pH가 PE 9010 공급사 안내 범위를 벗어납니다.", ["lc_pe9010"])
            if process.peak_temperature_c is not None and process.peak_temperature_c > limits["brief_exposure_temperature_c"]:
                check("pe9010_temperature_outside_guidance", "conflict", "입력 온도가 PE 9010의 단시간 내열 안내를 초과합니다.", ["lc_pe9010"])
            if batch:
                percent = next(row["finished_product_percent"] for row in batch["lines"] if _IDENTITIES.get(_name(row["name"])) == "pe9010")
                if percent > limits["supplier_maximum_percent"]:
                    check("pe9010_above_supplier_maximum", "conflict", "완제품 중 PE 9010 양이 공급사 최대 안내량을 초과합니다.", ["lc_pe9010"])
                elif percent < limits["typical_finished_product_percent"][0]:
                    check("pe9010_below_typical_guidance", "unresolved", "완제품 중 PE 9010 양이 통상 안내량 미만입니다. 보존력 자료가 필요합니다.", ["lc_pe9010"])
            check("preservation_not_proven", "unresolved", "방부제의 pH·사용량 범위 충족은 완제품 보존력, pH 안정성 또는 피부 적합성의 증명이 아닙니다.", ["lc_pe9010"])
        steps = deepcopy(data["steps"]["lotion_review"][:1])
        if method:
            steps += deepcopy(data["steps"][method])
        steps += deepcopy(data["steps"]["lotion_review"][1:])
    for step in steps:
        used.update(step["source_ids"])
    result["steps"] = steps
    result["sources"] = [deepcopy(row) for row in data["sources"] if row["id"] in used]
    result["process_constraints_satisfied"] = False if any(row["severity"] == "conflict" for row in result["checks"]) else None
    if result["process_constraints_satisfied"] is False:
        result["status"] = "process_conflict"
    return result


def procedure_answer(message):
    """Conservative deterministic routing; odor orders keep their old LLM path."""
    text = message.casefold().strip()
    # A request to *apply* a manufacturing method to an odor order is still an
    # odor order, not permission to discard its desired/excluded descriptors.
    if re.search(r"만들어\s*줘|만들어\s*주세요|뽑아\s*줘|생성\s*해|generate\s+(?:a\s+)?(?:recipe|formula)", text):
        return None
    question = re.search(r"제조\s*(?:법|방법|과정|공정|순서)|만드는\s*(?:법|방법|과정|순서)|조향\s*(?:방법|과정|순서)|how\s+(?:do\s+(?:i|you)\s+|to\s+)(?:make|formulate)|manufacturing\s+(?:process|steps)", text)
    if not question:
        return None
    lotion = bool(re.search(r"로션|\blotion\b", text))
    perfume = bool(re.search(r"향수|조향|\bperfume\b|\bperfumery\b", text))
    if lotion == perfume:
        return None
    product = "body_lotion" if lotion else "perfume"
    plan = formulation_workflow({"product_type": product})
    # No default lotion recipe is smuggled into an underspecified question.
    answer = ("로션은 유화제와 실제 베이스 조성에 맞춰 냉간·가열 공정을 구분합니다. " if lotion else
              "향수는 주제 어코드에 변조·연결·배경 역할을 더하며 비교합니다. 탑·미들·베이스에 고정 비율을 강제하지 않습니다. ")
    answer += " → ".join(step["action"] for step in plan["steps"])
    if lotion:
        answer += " 베이스 성분과 함량을 주시면 해당 공정의 단계·조건 충돌·배치 계량을 계산할 수 있습니다."
    return {"schema_version": "assistant-intent-1", "message": answer, "original_message": message,
            "intent_proposal": {"desired": [], "avoided": [], "product": product, "clarification": "none"},
            "requires_confirmation": False, "source": "sourced_formulation_knowledge",
            "formula_generated": False, "scientific_score_generated": False, "llm_calls_maximum": 0,
            "answer_kind": "procedure_guidance", "formulation_workflow": plan, "sources": plan["sources"]}
