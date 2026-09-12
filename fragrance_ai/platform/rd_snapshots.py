"""Portable comparison of backend-owned immutable evaluation snapshots.

Checksums detect changed content, not authenticity. These read-only operations
never promote candidates, infer approvals or replace durable backend storage.
"""
import json
import math
import re
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .rd_evidence import Input, FormulaLine, content_id


GATES = {"registered_evidence", "existing_safety", "scientific_domain", "profile_and_persistence"}


def line_map(lines):
    parsed = [FormulaLine.model_validate({key: row[key] for key in ("ingredient_id", "concentrate_percent")})
              for row in lines]
    if not parsed or len(parsed) > 50000 or len({row.ingredient_id for row in parsed}) != len(parsed):
        raise ValueError("snapshot requires nonempty, unique formula lines")
    if abs(math.fsum(row.concentrate_percent for row in parsed) - 100.) > .001 + 1e-9:
        raise ValueError("snapshot concentrate percentages must sum to 100 without normalization")
    return {row.ingredient_id: row.concentrate_percent for row in parsed}


def composition_diff(left, right):
    changes = []
    for key in sorted(left.keys() | right.keys()):
        before, after = left.get(key, 0.), right.get(key, 0.)
        if before != after:
            changes.append({"ingredient_id": key, "before_concentrate_percent": before,
                "after_concentrate_percent": after, "difference_percentage_points": after - before,
                "change": "added" if key not in left else "removed" if key not in right else "changed"})
    return {"unit": "concentrate_w/w_percentage_points", "changes": changes,
            "total_variation": math.fsum(abs(right.get(k, 0.) - left.get(k, 0.))
                                          for k in left.keys() | right.keys()) / 200.,
            "distance_kind": "composition_not_odor_similarity", "input_normalized": False}


class StoredCandidate(Input):
    evaluation: dict[str, JsonValue]
    candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Identifiers are supplied by the authorized backend, not looked up by AI.
    backend_version_id: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def consistent_snapshot(self):
        value = self.evaluation
        try:
            if len(json.dumps(value, allow_nan=False).encode()) > 8 * 1024 * 1024:
                raise ValueError("evaluation snapshot exceeds 8 MiB")
            if value.get("schema_version") != "rd-candidates-2":
                raise ValueError("only rd-candidates-2 evaluation snapshots are supported")
            if not isinstance(value.get("review_id"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["review_id"]):
                raise ValueError("snapshot requires its original review ID")
            if value.get("result_id") != content_id({k: v for k, v in value.items() if k != "result_id"}):
                raise ValueError("evaluation snapshot content hash mismatch")
            if value.get("manufacturing_approval") is not False or value.get("state_changed") is not False:
                raise ValueError("AI snapshot cannot assert a workflow or manufacturing approval")
            context = value.get("input_snapshot")
            if not isinstance(context, dict) or not all(k in context for k in ("request", "scenario", "intent", "evidence_policy")):
                raise ValueError("snapshot lacks input_snapshot; preserve historical data and use a new evaluation for comparable results")
            if not isinstance(context["request"], dict) or not isinstance(context["request"].get("formula"), dict):
                raise ValueError("snapshot requires its original typed request")
            scenario = context["scenario"]
            if not isinstance(scenario, dict) or not all(k in scenario for k in
                    ("product_category", "product_concentration_percent", "max_formula_cost_per_kg")):
                raise ValueError("snapshot requires its selected product/concentration/cost scenario")
            if not isinstance(value.get("contract"), dict) or not all(k in value["contract"] for k in ("runtime", "evidence", "evaluation_date")):
                raise ValueError("snapshot requires its model, evidence and evaluation-date contract")
            seen = set()
            for key, expected in (("candidates", True), ("diagnostic_candidates", False)):
                if not isinstance(value.get(key), list):
                    raise ValueError("snapshot requires recommendation and diagnostic arrays")
                for row in value[key]:
                    identity = row["candidate_id"]
                    if identity in seen:
                        raise ValueError("snapshot candidate IDs must be unique")
                    seen.add(identity)
                    gates = row.get("rd_gates")
                    if not isinstance(gates, dict) or not GATES <= gates.keys() or any(type(gates[k]) is not bool for k in GATES):
                        raise ValueError("snapshot requires explicit Boolean R&D gates")
                    if (row.get("recommendation_allowed") is not expected or
                            row.get("status") != ("ready_for_review" if expected else "abstained") or
                            all(gates[k] for k in GATES) != expected):
                        raise ValueError("snapshot status and R&D gate combination disagree")
                    score = row.get("target_match_score")
                    if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100):
                        raise ValueError("snapshot score must be null or finite model points from 0 to 100")
                    if row.get("target_match_unit") != "model_points_0_100":
                        raise ValueError("unsupported snapshot score unit")
                    payload = row["result"]
                    if (row.get("product_concentration_percent") != scenario["product_concentration_percent"]
                            or row.get("concentration_basis") != "w/w_percent"
                            or row["formula_id"] != payload.get("formula_id")):
                        raise ValueError("candidate identity or concentration disagrees with its source result")
                    lines = payload.get("recipe") or payload.get("closest_candidate") or []
                    if lines:
                        weights = line_map(lines)
                        if (row.get("material_count") != len(weights) or type(row.get("composition_total_percent")) not in (float, int)
                                or abs(row["composition_total_percent"] - math.fsum(weights.values())) > 1e-8):
                            raise ValueError("candidate composition summary disagrees with its formula")
                    if row["candidate_id"] != content_id({"review_id": value["review_id"],
                            "formula_id": row["formula_id"], "assessment": row.get("evidence_assessment")}):
                        raise ValueError("candidate identity does not match its evaluation binding")
            if self.candidate_id not in seen:
                raise ValueError("selected candidate does not belong to the supplied evaluation")
            if value.get("status") != ("ready_for_review" if value["candidates"] else "abstained"):
                raise ValueError("evaluation status and candidate groups disagree")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("incomplete or invalid evaluation snapshot") from error
        return self

    def candidate(self):
        return next(row for key in ("candidates", "diagnostic_candidates") for row in self.evaluation[key]
                    if row["candidate_id"] == self.candidate_id)

    def composition(self):
        result = self.candidate()["result"]
        lines = result.get("recipe") or result.get("closest_candidate") or []
        return line_map(lines) if lines else {}


class SnapshotComparison(Input):
    candidates: list[StoredCandidate] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def unique_versions(self):
        if len({row.backend_version_id for row in self.candidates}) != len(self.candidates):
            raise ValueError("backend version IDs must be unique")
        if len({(row.evaluation["result_id"], row.candidate_id) for row in self.candidates}) != len(self.candidates):
            raise ValueError("comparison requires distinct evaluation candidates")
        return self


class SnapshotRevision(Input):
    source: StoredCandidate
    instruction: str = Field(min_length=1, max_length=1000)


class RevisionSource(Input):
    parent_result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_backend_version_id: str = Field(min_length=1, max_length=200)
    parent_lines: list[FormulaLine] = Field(min_length=1, max_length=50000)
    instruction: str = Field(min_length=1, max_length=1000)
    mode: Literal["regenerate_for_revised_intent"] = "regenerate_for_revised_intent"

    @model_validator(mode="after")
    def valid_composition(self):
        line_map([row.model_dump() for row in self.parent_lines])
        return self


def comparison_basis(snapshot):
    value = snapshot.evaluation
    context = value["input_snapshot"]
    request = context["request"]
    return {"intent": context["intent"], "scenario": context["scenario"],
            "formula_constraints": {k: v for k, v in request["formula"].items() if k != "brief"},
            "structured_edits": request.get("edits"), "product_preferences": request.get("product_preferences"),
            "region": request["formula"].get("target_region"), "budget": request.get("budget"),
            "model_persistence": request.get("model_persistence"),
            "application_context": request.get("application_context"), "process": request.get("process"),
            "evidence_policy": context["evidence_policy"], "runtime": value["contract"]["runtime"],
            "evidence": value["contract"]["evidence"], "evaluation_date": value["contract"]["evaluation_date"]}


def compare_snapshots(request, current_contract):
    rows, pairs = [], []
    bases = [comparison_basis(item) for item in request.candidates]
    compositions = [item.composition() for item in request.candidates]
    for item, basis in zip(request.candidates, bases):
        candidate = item.candidate()
        rows.append({"backend_version_id": item.backend_version_id, "candidate_id": item.candidate_id,
            "result_id": item.evaluation["result_id"], "formula_id": candidate["formula_id"],
            "source_status": candidate["status"], "source_rd_gates": candidate["rd_gates"],
            "source_target_match_score": candidate.get("target_match_score"),
            "score_unit": candidate["target_match_unit"], "cost_krw": candidate.get("cost_krw"),
            "cost_basis": candidate.get("cost_basis"), "cost_per_volume_ml": candidate.get("cost_per_volume_ml"),
            "model_persistence": candidate.get("model_persistence"),
            "temporal_profile": candidate.get("temporal_profile", []),
            "evidence": candidate.get("evidence"), "evidence_assessment": candidate.get("evidence_assessment"),
            "comparison_basis": basis,
            "differs_from_active_contract": item.evaluation["contract"] != current_contract})
    for i, left in enumerate(request.candidates):
        for j in range(i + 1, len(request.candidates)):
            right = request.candidates[j]
            different = [key for key in bases[i] if bases[i][key] != bases[j][key]]
            a, b = left.candidate(), right.candidate()
            scores = [a.get("target_match_score"), b.get("target_match_score")]
            pairs.append({"left_version_id": left.backend_version_id, "right_version_id": right.backend_version_id,
                "comparison_status": "same_basis" if not different else "different_basis",
                "different_basis_fields": different,
                "score_difference_points": (scores[1] - scores[0]) if not different and all(v is not None for v in scores) else None,
                "composition": composition_diff(compositions[i], compositions[j]) if compositions[i] and compositions[j] else None})
    result = {"schema_version": "rd-comparison-2", "candidates": rows, "pairs": pairs,
        "automatic_ranking_performed": False, "recommendation_decision_performed": False,
        "new_inference_count": 0, "state_changed": False,
        "provenance": {"source": "backend_supplied_immutable_evaluation_snapshots",
                       "content_hashes_verified": True, "source_authenticity_verified": False,
                       "requires_authorized_backend_storage": True}}
    result["result_id"] = content_id(result)
    return result
