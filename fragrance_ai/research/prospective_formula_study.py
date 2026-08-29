"""Prospective, outcome-blind validation for generated perfume formulas.

The module deliberately separates three phases:

* ``prepare_study`` generates safe R&D formulas, freezes model predictions,
  randomises vial codes and participant assignments, and seals every byte.
* an independent sensory laboratory manufactures and evaluates the coded
  samples.  This repository never fabricates those observations.
* ``finalize_study`` verifies the seal, RFC3161 timestamp, laboratory Ed25519
  signature, exact assignment coverage, and a one-use evidence ledger before
  computing the preregistered metrics.

The resulting 90% gate is an empirical endpoint.  Proxy scores in the sealed
prediction file are never interpreted as human olfactory accuracy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import secrets
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fragrance_ai.recommender.artifact_trust import (
    EvidenceTrustRoot,
    canonical_json,
    sha256_file,
)
from fragrance_ai.recommender.models import RecipeConstraints, RecipeLine, profile_vector
from fragrance_ai.recommender.optimizer import cosine_similarity_percent
from fragrance_ai.recommender.physsim import ConcentrationAwarePhysSim
from fragrance_ai.recommender.physsim_checkpoint import FrozenR2PhysSim
from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI


STUDY_SCHEMA = "perfumery-prospective-formula-study/v1"
SEAL_SCHEMA = "perfumery-prospective-formula-study-seal/v1"
LAB_ARTIFACT_TYPE = "perfumery-prospective-formula-study-outcomes/v1"
MANUFACTURING_SCHEMA = "perfumery-prospective-formula-manufacturing-evidence/v1"
PREDICTION_VERSION = "formula-human-similarity-preregistered-composite-1.0"
OUTCOME_COLUMNS = (
    "assignment_id",
    "participant_id",
    "pair_id",
    "code_left",
    "code_right",
    "similarity_0_100",
    "confidence_0_100",
    "intensity_left_0_100",
    "intensity_right_0_100",
    "notes",
)
FORMULA_BRIEFS: tuple[str, ...] = (
    "sparkling bergamot citrus with green tea and clean cedar",
    "cool aquatic citrus with airy musk and dry woods",
    "crisp green aromatic herbs with fresh lemon and vetiver",
    "soft clean musk with powdery floral and pale woods",
    "velvety rose with warm amber and dry cedar",
    "creamy white floral with sandalwood and soft musk",
    "juicy peach fruity floral with clean musk",
    "dry cedar vetiver woods with aromatic freshness",
    "fresh lavender aromatic with citrus and clean woods",
    "powdery iris floral with soft musk and sandalwood",
    "warm amber spicy woods with vanilla softness",
    "gourmand vanilla cacao with amber and soft woods",
    "woody amber with spicy aromatic lift and soft musk",
    "airy transparent floral with clean citrus and musk",
    "juicy berry fruity musk with soft floral heart",
    "cool mint citrus with green herbs and dry cedar",
    "tropical fruity creamy floral with soft clean musk",
    "soft sandalwood vanilla with creamy musk",
    "bright orange blossom white floral with citrus",
    "fresh pear fruity green with transparent musk",
    "spicy cardamom amber with dry woods",
    "clean aquatic floral with cooling green notes",
    "rosy fruity floral with powdery musk",
    "citrus aromatic amber with woody drydown",
)

FORMULA_COUNT = 24
ORDINARY_PAIR_COUNT = 100
IDENTICAL_CONTROL_COUNT = 10
CONCENTRATION_CONTROL_COUNT = 10
PAIR_COUNT = 120
PARTICIPANT_COUNT = 80
PAIRS_PER_PARTICIPANT = 30
RATINGS_PER_PAIR = 20
DEFAULT_BOOTSTRAP_DRAWS = 5_000
DEFAULT_RELIABILITY_REPEATS = 512


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload) + b"\n")


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if any(None in row for row in rows):
        raise ValueError(f"CSV contains fields beyond its declared header: {path}")
    return list(reader.fieldnames), rows


def _canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(list(rows))).hexdigest()


def _formula_lines_at_concentration(
    lines: Sequence[RecipeLine], product_concentration_percent: float
) -> list[RecipeLine]:
    if not 0.0 < product_concentration_percent <= 100.0:
        raise ValueError("product concentration must be in (0, 100]")
    scale = product_concentration_percent / 100.0
    return [
        replace(
            line,
            finished_product_percent=line.concentrate_percent * scale,
            volume_ml_for_batch=None,
            mass_g_for_batch=0.0,
            active_mass_g_for_batch=0.0,
        )
        for line in lines
    ]


def _weighted_overlap(left: Sequence[RecipeLine], right: Sequence[RecipeLine]) -> float:
    left_map = {
        line.ingredient_id: max(0.0, float(line.finished_product_percent))
        for line in left
    }
    right_map = {
        line.ingredient_id: max(0.0, float(line.finished_product_percent))
        for line in right
    }
    keys = set(left_map) | set(right_map)
    denominator = sum(max(left_map.get(key, 0.0), right_map.get(key, 0.0)) for key in keys)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(min(left_map.get(key, 0.0), right_map.get(key, 0.0)) for key in keys)
    return 100.0 * numerator / denominator


def _ingredient_jaccard(left: Sequence[RecipeLine], right: Sequence[RecipeLine]) -> float:
    left_ids = {line.ingredient_id for line in left}
    right_ids = {line.ingredient_id for line in right}
    union = left_ids | right_ids
    return 100.0 * len(left_ids & right_ids) / len(union) if union else 0.0


def _formula_fingerprint(lines: Sequence[RecipeLine]) -> str:
    rows = [
        (
            line.ingredient_id,
            round(float(line.concentrate_percent), 8),
            round(float(line.active_strength_percent), 8),
            line.carrier or "",
        )
        for line in sorted(lines, key=lambda item: item.ingredient_id)
    ]
    return hashlib.sha256(canonical_json(rows)).hexdigest()


def _pair_features(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_concentration: float,
    right_concentration: float,
    physsim: ConcentrationAwarePhysSim,
    r2: FrozenR2PhysSim,
    ingredients: Mapping[str, Any],
    scientific_store: Any,
) -> dict[str, Any]:
    left_lines = _formula_lines_at_concentration(
        left["recipe_lines"], left_concentration
    )
    right_lines = _formula_lines_at_concentration(
        right["recipe_lines"], right_concentration
    )
    semantic = cosine_similarity_percent(
        profile_vector(left["achieved_profile"]),
        profile_vector(right["achieved_profile"]),
    )
    deterministic = physsim.compare(
        left_lines,
        right_lines,
        dict(ingredients),
        scientific_store,
    )
    learned = r2.evaluate(left_lines, right_lines)
    weighted_overlap = _weighted_overlap(left_lines, right_lines)
    jaccard = _ingredient_jaccard(left_lines, right_lines)
    # Fixed before outcomes.  The learned R2 branch remains a reported
    # diagnostic because its direct-formulation production weight is zero.
    predicted = (
        0.45 * deterministic.temporal_similarity_mean
        + 0.35 * weighted_overlap
        + 0.20 * semantic
    )
    return {
        "predicted_similarity_0_100": round(float(predicted), 6),
        "fixed_component_overlap_baseline_0_100": round(weighted_overlap, 6),
        "ingredient_jaccard_0_100": round(jaccard, 6),
        "semantic_profile_similarity_0_100": round(float(semantic), 6),
        "headspace_physsim_similarity_0_100": round(
            deterministic.temporal_similarity_mean, 6
        ),
        "headspace_physsim_minimum_timepoint_0_100": round(
            deterministic.minimum_temporal_similarity, 6
        ),
        "headspace_physsim_applicability_0_100": round(
            deterministic.model_applicability_percent, 6
        ),
        "learned_r2_similarity_0_100": (
            None if learned.similarity is None else round(learned.similarity, 6)
        ),
        "learned_r2_status": learned.status,
        "learned_r2_applied_weight": learned.applied_primary_score_weight,
        "learned_r2_applicability_0_100": learned.applicability_percent,
        "prediction_weights": {
            "headspace_physsim": 0.45,
            "component_overlap": 0.35,
            "semantic_profile": 0.20,
            "learned_r2": 0.0,
        },
    }


def _build_formulas() -> tuple[list[dict[str, Any]], NaturalLanguagePerfumeryAI]:
    constraints = RecipeConstraints(
        target_similarity=70.0,
        max_ingredients=10,
        simulation_draws=64,
        physics_search_population=2,
        minimum_realism_score=55.0,
        product_concentration_percent=15.0,
        max_ingredient_price_per_kg=300.0,
        max_formula_cost_per_kg=180.0,
        min_availability=0.75,
        max_risk_tier=1,
        allow_rare=False,
    )
    ai = NaturalLanguagePerfumeryAI()
    formulas: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    try:
        for index, brief in enumerate(FORMULA_BRIEFS, start=1):
            result = ai.create_recipe(brief, constraints)
            if result.status != "prototype_ready" or not result.recipe:
                raise RuntimeError(
                    f"formula F{index:02d} did not pass the safe R&D recipe gate: "
                    f"{result.status}"
                )
            if not result.safety.internal_gate_passed:
                raise RuntimeError(f"formula F{index:02d} failed the internal safety gate")
            if any(line.risk_tier > 1 for line in result.recipe):
                raise RuntimeError(f"formula F{index:02d} contains a disallowed risk tier")
            fingerprint = _formula_fingerprint(result.recipe)
            if fingerprint in fingerprints:
                raise RuntimeError(
                    f"formula F{index:02d} duplicates an earlier quantitative formula"
                )
            fingerprints.add(fingerprint)
            formulas.append(
                {
                    "formula_id": f"F{index:02d}",
                    "brief": brief,
                    "generator_status": result.status,
                    "generator_similarity_kind": result.similarity_kind,
                    "formula_fingerprint_sha256": fingerprint,
                    "achieved_profile": dict(result.achieved_profile),
                    "recipe_lines": list(result.recipe),
                    "estimated_concentrate_cost_per_kg": (
                        result.estimated_concentrate_cost_per_kg
                    ),
                    "internal_safety_audit_id": result.safety.audit_id,
                    "execution_authorized": False,
                }
            )
    except Exception:
        ai.close()
        raise
    if len(formulas) != FORMULA_COUNT:
        ai.close()
        raise RuntimeError("formula library size differs from preregistration")
    return formulas, ai


def _select_ordinary_pairs(
    candidates: Sequence[dict[str, Any]], formula_ids: Sequence[str]
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            float(row["features"]["predicted_similarity_0_100"]),
            row["formula_a"],
            row["formula_b"],
        ),
    )
    strata = np.array_split(np.asarray(ordered, dtype=object), 5)
    usage = Counter({formula_id: 0 for formula_id in formula_ids})
    selected: list[dict[str, Any]] = []
    for stratum_index, stratum in enumerate(strata, start=1):
        remaining = [dict(row) for row in stratum.tolist()]
        for _ in range(ORDINARY_PAIR_COUNT // 5):
            if not remaining:
                raise RuntimeError("pair stratum is too small for the frozen design")
            chosen = min(
                remaining,
                key=lambda row: (
                    max(usage[row["formula_a"]], usage[row["formula_b"]]),
                    usage[row["formula_a"]] + usage[row["formula_b"]],
                    row["formula_a"],
                    row["formula_b"],
                ),
            )
            remaining.remove(chosen)
            chosen["similarity_stratum"] = stratum_index
            selected.append(chosen)
            usage[chosen["formula_a"]] += 1
            usage[chosen["formula_b"]] += 1
    if len(selected) != ORDINARY_PAIR_COUNT:
        raise RuntimeError("ordinary pair count differs from preregistration")
    return selected


def _generate_pairs(
    formulas: Sequence[dict[str, Any]], ai: NaturalLanguagePerfumeryAI
) -> list[dict[str, Any]]:
    by_id = {row["formula_id"]: row for row in formulas}
    ingredients = {item.ingredient_id: item for item in ai.catalog.ingredients}
    physsim = ConcentrationAwarePhysSim()
    r2 = FrozenR2PhysSim()
    candidates: list[dict[str, Any]] = []
    formula_ids = sorted(by_id)
    for left_index, left_id in enumerate(formula_ids):
        for right_id in formula_ids[left_index + 1 :]:
            candidates.append(
                {
                    "pair_type": "ordinary_formula_pair",
                    "formula_a": left_id,
                    "formula_b": right_id,
                    "concentration_a_percent": 15.0,
                    "concentration_b_percent": 15.0,
                    "features": _pair_features(
                        by_id[left_id],
                        by_id[right_id],
                        left_concentration=15.0,
                        right_concentration=15.0,
                        physsim=physsim,
                        r2=r2,
                        ingredients=ingredients,
                        scientific_store=ai.scientific_store,
                    ),
                }
            )
    pairs = _select_ordinary_pairs(candidates, formula_ids)
    usage = Counter()
    for row in pairs:
        usage[row["formula_a"]] += 1
        usage[row["formula_b"]] += 1

    identical_ids = sorted(formula_ids, key=lambda item: (usage[item], item))[
        :IDENTICAL_CONTROL_COUNT
    ]
    for formula_id in identical_ids:
        pairs.append(
            {
                "pair_type": "identical_formula_control",
                "formula_a": formula_id,
                "formula_b": formula_id,
                "concentration_a_percent": 15.0,
                "concentration_b_percent": 15.0,
                "similarity_stratum": "identical_control",
                "features": _pair_features(
                    by_id[formula_id],
                    by_id[formula_id],
                    left_concentration=15.0,
                    right_concentration=15.0,
                    physsim=physsim,
                    r2=r2,
                    ingredients=ingredients,
                    scientific_store=ai.scientific_store,
                ),
            }
        )
        usage[formula_id] += 2

    concentration_ids = sorted(formula_ids, key=lambda item: (usage[item], item))[
        :CONCENTRATION_CONTROL_COUNT
    ]
    for formula_id in concentration_ids:
        pairs.append(
            {
                "pair_type": "same_formula_concentration_control",
                "formula_a": formula_id,
                "formula_b": formula_id,
                "concentration_a_percent": 10.0,
                "concentration_b_percent": 20.0,
                "similarity_stratum": "concentration_control",
                "features": _pair_features(
                    by_id[formula_id],
                    by_id[formula_id],
                    left_concentration=10.0,
                    right_concentration=20.0,
                    physsim=physsim,
                    r2=r2,
                    ingredients=ingredients,
                    scientific_store=ai.scientific_store,
                ),
            }
        )
    if len(pairs) != PAIR_COUNT:
        raise RuntimeError("total pair count differs from preregistration")
    return pairs


def _randomization(
    pairs: Sequence[dict[str, Any]], seed: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    resolved_seed = secrets.randbits(128) if seed is None else int(seed)
    if resolved_seed < 0:
        raise ValueError("randomization seed must be nonnegative")
    rng = random.Random(resolved_seed)
    opaque_pair_ids = [
        f"Q{value:08d}"
        for value in rng.sample(range(10_000_000, 100_000_000), PAIR_COUNT)
    ]
    for pair, pair_id in zip(pairs, opaque_pair_ids):
        pair["pair_id"] = pair_id
    codes = [f"{value:03d}" for value in rng.sample(range(100, 1000), PAIR_COUNT * 2)]
    key_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        key_rows.append(
            {
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "formula_a": pair["formula_a"],
                "concentration_a_percent": pair["concentration_a_percent"],
                "code_a": codes[2 * index],
                "formula_b": pair["formula_b"],
                "concentration_b_percent": pair["concentration_b_percent"],
                "code_b": codes[2 * index + 1],
            }
        )
    key_by_pair = {row["pair_id"]: row for row in key_rows}
    orientation_by_pair: dict[str, list[bool]] = {}
    for pair_id in key_by_pair:
        orientations = [False] * (RATINGS_PER_PAIR // 2) + [True] * (
            RATINGS_PER_PAIR // 2
        )
        rng.shuffle(orientations)
        orientation_by_pair[pair_id] = orientations
    assignments: list[dict[str, Any]] = []
    assignment_number = 0
    pair_ids = [row["pair_id"] for row in pairs]
    for block in range(RATINGS_PER_PAIR):
        shuffled = list(pair_ids)
        rng.shuffle(shuffled)
        for chunk in range(4):
            participant_id = f"S{block * 4 + chunk + 1:03d}"
            for pair_id in shuffled[
                chunk * PAIRS_PER_PARTICIPANT : (chunk + 1) * PAIRS_PER_PARTICIPANT
            ]:
                assignment_number += 1
                key = key_by_pair[pair_id]
                swap = orientation_by_pair[pair_id][block]
                code_left = key["code_b"] if swap else key["code_a"]
                code_right = key["code_a"] if swap else key["code_b"]
                assignments.append(
                    {
                        "assignment_id": f"A{assignment_number:04d}",
                        "participant_id": participant_id,
                        "pair_id": pair_id,
                        "code_left": code_left,
                        "code_right": code_right,
                        "similarity_0_100": "",
                        "confidence_0_100": "",
                        "intensity_left_0_100": "",
                        "intensity_right_0_100": "",
                        "notes": "",
                    }
                )
    participant_counts = Counter(row["participant_id"] for row in assignments)
    pair_counts = Counter(row["pair_id"] for row in assignments)
    if set(participant_counts.values()) != {PAIRS_PER_PARTICIPANT}:
        raise RuntimeError("participant assignment balance failed")
    if set(pair_counts.values()) != {RATINGS_PER_PAIR}:
        raise RuntimeError("pair assignment balance failed")
    randomization = {
        "schema": STUDY_SCHEMA,
        "seed_decimal": str(resolved_seed),
        "generator": "python_random_mt19937_seeded_once",
        "vial_code_space": "unique_3_digit_codes_100_to_999",
        "pair_id_space": "random_unique_Q_plus_8_digits_independent_of_pair_type",
        "left_right_randomized_per_assignment": True,
        "left_right_balance_per_pair": "exactly_10_each_orientation",
        "public_seed_disclosure": False,
    }
    return key_rows, assignments, randomization


def _formula_rows(formulas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for formula in formulas:
        for line_number, line in enumerate(formula["recipe_lines"], start=1):
            rows.append(
                {
                    "formula_id": formula["formula_id"],
                    "brief": formula["brief"],
                    "line_number": line_number,
                    "ingredient_id": line.ingredient_id,
                    "ingredient_name": line.name,
                    "pyramid": line.pyramid,
                    "concentrate_percent": f"{line.concentrate_percent:.8f}",
                    "active_strength_percent": f"{line.active_strength_percent:.8f}",
                    "carrier": line.carrier or "",
                    "risk_tier": line.risk_tier,
                    "availability": f"{line.availability:.8f}",
                    "price_per_kg": f"{line.price_per_kg:.8f}",
                    "supplier_sku": "",
                    "lot_number": "",
                    "coa_sha256": "",
                    "sds_sha256": "",
                    "ifra_certificate_sha256": "",
                    "manufacturing_execution_authorized": "false",
                }
            )
    return rows


def _formula_manifest(formulas: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": STUDY_SCHEMA,
        "formula_count": len(formulas),
        "formulas": [
            {
                "formula_id": formula["formula_id"],
                "brief": formula["brief"],
                "formula_fingerprint_sha256": formula[
                    "formula_fingerprint_sha256"
                ],
                "generator_status": formula["generator_status"],
                "generator_similarity_kind": formula["generator_similarity_kind"],
                "estimated_concentrate_cost_per_kg": formula[
                    "estimated_concentrate_cost_per_kg"
                ],
                "internal_safety_audit_id": formula["internal_safety_audit_id"],
                "human_accuracy_claim": False,
                "manufacturing_execution_authorized": False,
                "lines": [
                    {
                        "ingredient_id": line.ingredient_id,
                        "target_concentrate_percent": round(
                            float(line.concentrate_percent), 8
                        ),
                        "active_strength_percent": round(
                            float(line.active_strength_percent), 8
                        ),
                        "carrier": line.carrier or "",
                    }
                    for line in formula["recipe_lines"]
                ],
            }
            for formula in formulas
        ],
    }


def _manufacturing_template(
    study_id: str,
    formulas: Sequence[dict[str, Any]],
    key_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    for key in key_rows:
        sample_rows.extend(
            (
                {
                    "code": key["code_a"],
                    "formula_id": key["formula_a"],
                    "finished_product_concentration_percent": key[
                        "concentration_a_percent"
                    ],
                    "concentrate_batch_id": "",
                    "concentrate_mass_g": None,
                    "base_mass_g": None,
                    "finished_mass_g": None,
                    "vial_id": "",
                    "prepared_at": "",
                },
                {
                    "code": key["code_b"],
                    "formula_id": key["formula_b"],
                    "finished_product_concentration_percent": key[
                        "concentration_b_percent"
                    ],
                    "concentrate_batch_id": "",
                    "concentrate_mass_g": None,
                    "base_mass_g": None,
                    "finished_mass_g": None,
                    "vial_id": "",
                    "prepared_at": "",
                },
            )
        )
    return {
        "schema": MANUFACTURING_SCHEMA,
        "study_id": study_id,
        "execution_authorized": False,
        "laboratory_id": "",
        "safety_approval": {"approval_id": "", "document": ""},
        "ethics_approval": {"approval_id": "", "document": ""},
        "environment": {
            "sop_id": "",
            "sop_version": "",
            "product_base_id": "",
            "product_base_lot": "",
            "product_base_coa_document": "",
            "product_base_sds_document": "",
            "dilution_solvent": "",
            "maturation_hours": None,
            "test_temperature_c": None,
            "relative_humidity_percent": None,
            "headspace_equilibration_minutes": None,
            "sniff_interval_seconds": None,
            "vial_fill_ml": None,
            "vial_headspace_ml": None,
            "container_lot": "",
        },
        "formula_batches": [
            {
                "formula_id": formula["formula_id"],
                "formula_fingerprint_sha256": formula[
                    "formula_fingerprint_sha256"
                ],
                "concentrate_batch_id": "",
                "weighed_total_g": None,
                "ingredient_lots": [
                    {
                        "ingredient_id": line.ingredient_id,
                        "target_concentrate_percent": round(
                            float(line.concentrate_percent), 8
                        ),
                        "active_strength_percent": round(
                            float(line.active_strength_percent), 8
                        ),
                        "carrier": line.carrier or "",
                        "actual_mass_g": None,
                        "supplier_sku": "",
                        "lot_number": "",
                        "coa_document": "",
                        "sds_document": "",
                        "ifra_document": "",
                    }
                    for line in formula["recipe_lines"]
                ],
            }
            for formula in formulas
        ],
        "sample_vials": sample_rows,
        "document_hashes": {},
        "completion_note": (
            "The independent laboratory must fill every blank, set execution_authorized "
            "true, and sign this JSON, outcomes CSV, and every referenced document."
        ),
    }


def _participant_instructions(study_id: str) -> str:
    return f"""# 독립 블라인드 조향식 유사도 시험

연구 ID: `{study_id}`

각 행의 `code_left`와 `code_right` 시료를 독립 시험기관의 승인된 표준작업절차에
따라 비교하고, 향의 전체적인 유사도를 0(전혀 다름)부터 100(구별 불가)까지
기록하십시오. 강도는 유사도와 별도로 기록합니다. 이전 행으로 돌아가 점수를
수정하지 말고, 시료 코드의 조성이나 모델 예측을 참가자에게 공개하지 마십시오.

이 파일은 사람 노출을 승인하지 않습니다. 시험기관의 안전성 검토, 적용 지역의
윤리/기관 승인, 환기·노출 한도, 임신·천식·알레르기 등 제외 기준, 응급 절차가
먼저 승인되어야 합니다. 시료를 맛보거나 피부에 바르지 마십시오. 부작용이
발생하면 즉시 중단하고 기관 절차를 따르십시오. 개인식별정보는 CSV에 넣지
마십시오.
"""


def prepare_study(
    output_dir: str | Path,
    *,
    study_id: str | None = None,
    randomization_seed: int | None = None,
    openssl: str | Path | None = None,
) -> dict[str, Any]:
    """Create and seal a prospective study before any outcome exists."""
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("study output directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    resolved_study_id = study_id or (
        "PFBS-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,127}", resolved_study_id):
        raise ValueError("study_id must be 6-128 safe identifier characters")
    outcomes_path = root / "external" / "human_outcomes.csv"
    manufacturing_path = root / "external" / "manufacturing_execution.json"
    if outcomes_path.exists() or manufacturing_path.exists():
        raise RuntimeError(
            "human outcomes and manufacturing evidence must not exist before sealing"
        )

    formulas, ai = _build_formulas()
    try:
        pairs = _generate_pairs(formulas, ai)
    finally:
        ai.close()
    key_rows, assignments, randomization = _randomization(
        pairs, randomization_seed
    )
    created_at = _utc_now()
    predictions = {
        "schema": STUDY_SCHEMA,
        "study_id": resolved_study_id,
        "created_at": created_at,
        "prediction_version": PREDICTION_VERSION,
        "outcome_data_accessed": False,
        "human_accuracy_claim": False,
        "primary_endpoint": {
            "name": "pair_mean_absolute_similarity_accuracy",
            "definition": "100 - mean(abs(predicted_similarity - human_pair_mean))",
            "scale": "percent_0_to_100",
            "population": "100 ordinary non-identical formula pairs only",
            "controls_in_primary_metric": False,
            "ninety_percent_gate": "crossed_bootstrap_lower_95_at_least_90",
        },
        "secondary_endpoints": [
            "pair_level_spearman",
            "pair_level_pearson",
            "concordance_correlation_coefficient",
            "human_ceiling_normalized_spearman",
            "fixed_component_overlap_baseline_comparison",
        ],
        "prediction_contract": {
            "fixed_weights": {
                "headspace_physsim": 0.45,
                "component_overlap": 0.35,
                "semantic_profile": 0.20,
                "learned_r2": 0.0,
            },
            "no_post_outcome_refit": True,
            "all_participants_intention_to_test": True,
            "human_similarity_is_not_available_at_prepare_time": True,
        },
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "pair_type": row["pair_type"],
                "similarity_stratum": row["similarity_stratum"],
                **row["features"],
            }
            for row in pairs
        ],
    }
    protocol = {
        "schema": STUDY_SCHEMA,
        "study_id": resolved_study_id,
        "created_at": created_at,
        "design": {
            "formula_count": FORMULA_COUNT,
            "ordinary_formula_pairs": ORDINARY_PAIR_COUNT,
            "identical_formula_controls": IDENTICAL_CONTROL_COUNT,
            "same_formula_concentration_controls": CONCENTRATION_CONTROL_COUNT,
            "pair_count": PAIR_COUNT,
            "participant_count": PARTICIPANT_COUNT,
            "pairs_per_participant": PAIRS_PER_PARTICIPANT,
            "ratings_per_pair": RATINGS_PER_PAIR,
            "total_ratings": PAIR_COUNT * RATINGS_PER_PAIR,
            "ordinary_pair_strata": 5,
            "ordinary_pairs_per_stratum": 20,
            "blinding": "unique coded vials; formula key and predictions restricted",
            "left_right_order": (
                "randomized with exactly 10 observations in each orientation per pair"
            ),
        },
        "preregistered_analysis": {
            "bootstrap": "participant-by-pair crossed cluster resampling",
            "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
            "confidence_interval": "two-sided percentile 95%",
            "reliability": "pairwise split-half with Spearman-Brown correction",
            "rating_exclusions": "none; intention-to-test primary analysis",
            "missing_or_out_of_range_rows": "fail entire study",
            "ninety_percent_gate_checks": [
                "independent_laboratory_signature_valid",
                "prediction_seal_RFC3161_timestamp_valid",
                "exact_80_participants_and_2400_assignments",
                "exact_20_ratings_per_pair",
                "absolute_similarity_accuracy_lower_95_at_least_90",
                "spearman_lower_95_at_least_0.90",
                "pearson_lower_95_at_least_0.90",
                "identical_control_mean_at_least_90",
                "identical_control_bootstrap_lower_95_at_least_85",
            ],
        },
        "safety_and_execution": {
            "human_exposure_authorized": False,
            "manufacturing_authorized": False,
            "required_before_execution": [
                "independent laboratory safety approval",
                "applicable ethics or institutional approval",
                "supplier SKU and lot-specific COA/SDS/IFRA evidence",
                "controlled base, dilution, maturation, temperature and headspace SOP",
                "adverse-event and exclusion procedures",
            ],
            "claim_boundary": (
                "Preparation is a research protocol, not permission to manufacture, "
                "expose humans, certify safety, or claim 90% olfactory accuracy."
            ),
        },
        "expected_external_outcome": {
            "relative_path": "external/human_outcomes.csv",
            "columns": list(OUTCOME_COLUMNS),
            "present_before_seal": False,
            "required_artifact_type": LAB_ARTIFACT_TYPE,
            "required_signer_role": "sensory_laboratory",
        },
        "expected_manufacturing_evidence": {
            "relative_path": "external/manufacturing_execution.json",
            "template_relative_path": (
                "restricted/manufacturing_execution_template.json"
            ),
            "schema": MANUFACTURING_SCHEMA,
            "required_bindings": [
                "24 quantitative formula fingerprints and weighed masses",
                "supplier SKU and lot-specific COA/SDS/IFRA documents",
                "product base, dilution, maturation, temperature and headspace",
                "all 240 blinded vial codes and finished concentrations",
            ],
            "present_before_seal": False,
        },
    }

    formula_columns = (
        "formula_id",
        "brief",
        "line_number",
        "ingredient_id",
        "ingredient_name",
        "pyramid",
        "concentrate_percent",
        "active_strength_percent",
        "carrier",
        "risk_tier",
        "availability",
        "price_per_kg",
        "supplier_sku",
        "lot_number",
        "coa_sha256",
        "sds_sha256",
        "ifra_certificate_sha256",
        "manufacturing_execution_authorized",
    )
    key_columns = (
        "pair_id",
        "pair_type",
        "formula_a",
        "concentration_a_percent",
        "code_a",
        "formula_b",
        "concentration_b_percent",
        "code_b",
    )
    _write_csv(root / "restricted" / "formulas.csv", formula_columns, _formula_rows(formulas))
    _write_json(root / "restricted" / "formula_manifest.json", _formula_manifest(formulas))
    _write_csv(root / "restricted" / "study_key.csv", key_columns, key_rows)
    _write_json(root / "restricted" / "predictions.json", predictions)
    _write_json(root / "restricted" / "randomization.json", randomization)
    _write_json(
        root / "restricted" / "manufacturing_execution_template.json",
        _manufacturing_template(resolved_study_id, formulas, key_rows),
    )
    _write_csv(root / "public" / "outcomes_template.csv", OUTCOME_COLUMNS, assignments)
    (root / "public" / "participant_instructions.md").write_text(
        _participant_instructions(resolved_study_id), encoding="utf-8", newline="\n"
    )
    _write_json(root / "study_protocol.json", protocol)

    sealed_relatives = (
        "study_protocol.json",
        "restricted/formulas.csv",
        "restricted/formula_manifest.json",
        "restricted/study_key.csv",
        "restricted/predictions.json",
        "restricted/randomization.json",
        "restricted/manufacturing_execution_template.json",
        "public/outcomes_template.csv",
        "public/participant_instructions.md",
    )
    files = {
        relative: {
            "sha256": sha256_file(root / relative),
            "bytes": (root / relative).stat().st_size,
        }
        for relative in sealed_relatives
    }
    implementation_path = Path(__file__).resolve(strict=True)
    seal = {
        "schema": SEAL_SCHEMA,
        "study_id": resolved_study_id,
        "sealed_at": _utc_now(),
        "outcome_data_accessed": False,
        "human_outcome_present_before_seal": False,
        "human_accuracy_claim": False,
        "implementation": {
            "module": str(implementation_path),
            "module_sha256": sha256_file(implementation_path),
            "prediction_version": PREDICTION_VERSION,
        },
        "files": files,
        "prediction_rows_sha256": _canonical_rows_sha256(predictions["pairs"]),
        "assignment_rows_sha256": _canonical_rows_sha256(assignments),
        "expected_outcome_relative_path": "external/human_outcomes.csv",
        "expected_outcome_present_before_seal": False,
        "expected_manufacturing_relative_path": (
            "external/manufacturing_execution.json"
        ),
        "expected_manufacturing_present_before_seal": False,
        "expected_report_relative_path": "final-report.json",
        "expected_ledger_filename": "prospective_formula_evidence_ledger.jsonl",
    }
    _write_json(root / "seal.json", seal)
    timestamp_query: str | None = None
    if openssl is not None:
        timestamp_query = str(create_timestamp_query(root, openssl))
    return {
        "study_id": resolved_study_id,
        "study_dir": str(root),
        "seal_sha256": sha256_file(root / "seal.json"),
        "prediction_sha256": files["restricted/predictions.json"]["sha256"],
        "formula_count": FORMULA_COUNT,
        "pair_count": PAIR_COUNT,
        "participant_count": PARTICIPANT_COUNT,
        "total_assignments": len(assignments),
        "timestamp_query": timestamp_query,
        "human_accuracy_claim": False,
        "execution_authorized": False,
    }


def create_timestamp_query(study_dir: str | Path, openssl: str | Path) -> Path:
    root = Path(study_dir).expanduser().resolve(strict=True)
    executable = Path(openssl).expanduser().resolve(strict=True)
    output = root / "timestamp" / "seal.tsq"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("timestamp query already exists")
    command = [
        str(executable),
        "ts",
        "-query",
        "-data",
        str((root / "seal.json").resolve(strict=True)),
        "-sha256",
        "-cert",
        "-out",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(
            "RFC3161 query generation failed: "
            + (completed.stdout + completed.stderr)[-1000:]
        )
    return output


def verify_rfc3161_timestamp(
    *,
    openssl: str | Path,
    seal_path: str | Path,
    response_path: str | Path,
    ca_path: str | Path,
    tsa_path: str | Path,
) -> dict[str, Any]:
    executable = Path(openssl).expanduser().resolve(strict=True)
    seal = Path(seal_path).expanduser().resolve(strict=True)
    response = Path(response_path).expanduser().resolve(strict=True)
    ca = Path(ca_path).expanduser().resolve(strict=True)
    tsa = Path(tsa_path).expanduser().resolve(strict=True)
    verification = subprocess.run(
        [
            str(executable),
            "ts",
            "-verify",
            "-data",
            str(seal),
            "-in",
            str(response),
            "-CAfile",
            str(ca),
            "-untrusted",
            str(tsa),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    verification_text = verification.stdout + verification.stderr
    if verification.returncode != 0 or "Verification: OK" not in verification_text:
        raise RuntimeError("RFC3161 timestamp verification failed: " + verification_text[-1000:])
    reply = subprocess.run(
        [str(executable), "ts", "-reply", "-in", str(response), "-text"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if reply.returncode != 0:
        raise RuntimeError("RFC3161 timestamp reply inspection failed")
    match = re.search(r"^Time stamp:\s*(.+)$", reply.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("RFC3161 timestamp reply has no timestamp")
    raw_timestamp = match.group(1).strip()
    try:
        parsed_timestamp = datetime.strptime(
            raw_timestamp, "%b %d %H:%M:%S %Y GMT"
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise RuntimeError("RFC3161 timestamp has an unsupported time format") from error
    return {
        "verified": True,
        "time_stamp": raw_timestamp,
        "timestamp_utc": parsed_timestamp.isoformat(),
        "response_sha256": sha256_file(response),
        "ca_sha256": sha256_file(ca),
        "tsa_sha256": sha256_file(tsa),
    }


def record_timestamp_verification(
    study_dir: str | Path,
    *,
    openssl: str | Path,
    response_path: str | Path,
    ca_path: str | Path,
    tsa_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a TSA response against the sealed bytes and record its hashes."""
    verified = verify_study_seal(study_dir)
    root = verified["root"]
    timestamp = verify_rfc3161_timestamp(
        openssl=openssl,
        seal_path=root / "seal.json",
        response_path=response_path,
        ca_path=ca_path,
        tsa_path=tsa_path,
    )
    result = {
        "schema": STUDY_SCHEMA,
        "study_id": verified["seal"]["study_id"],
        "seal_sha256": sha256_file(root / "seal.json"),
        "verified_at": _utc_now(),
        "timestamp": timestamp,
        "human_outcome_present_at_verification": (
            root / verified["seal"]["expected_outcome_relative_path"]
        ).exists(),
        "manufacturing_evidence_present_at_verification": (
            root / verified["seal"]["expected_manufacturing_relative_path"]
        ).exists(),
    }
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "timestamp" / "verification.json"
    )
    if target.exists():
        raise FileExistsError("timestamp verification record already exists")
    _write_json(target, result)
    return result


def verify_study_seal(study_dir: str | Path) -> dict[str, Any]:
    root = Path(study_dir).expanduser().resolve(strict=True)
    seal_path = root / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != SEAL_SCHEMA:
        raise ValueError("unsupported prospective study seal schema")
    files = seal.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("study seal has no files")
    for relative, evidence in files.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("study seal contains an unsafe relative path")
        if not isinstance(evidence, Mapping):
            raise ValueError("study seal file evidence must be an object")
        target = (root / relative).resolve(strict=True)
        if root not in target.parents:
            raise ValueError("study seal file escapes the study directory")
        if sha256_file(target) != evidence.get("sha256"):
            raise ValueError(f"sealed file changed: {relative}")
        if target.stat().st_size != int(evidence.get("bytes", -1)):
            raise ValueError(f"sealed file size changed: {relative}")
    implementation = seal.get("implementation", {})
    if implementation.get("prediction_version") != PREDICTION_VERSION:
        raise ValueError("sealed prediction implementation version changed")
    if implementation.get("module_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("prospective study implementation changed after sealing")
    predictions = json.loads(
        (root / "restricted" / "predictions.json").read_text(encoding="utf-8")
    )
    formula_manifest = json.loads(
        (root / "restricted" / "formula_manifest.json").read_text(encoding="utf-8")
    )
    _, study_key = _read_csv(root / "restricted" / "study_key.csv")
    _, assignments = _read_csv(root / "public" / "outcomes_template.csv")
    if seal.get("prediction_rows_sha256") != _canonical_rows_sha256(predictions["pairs"]):
        raise ValueError("sealed prediction rows changed")
    if seal.get("assignment_rows_sha256") != _canonical_rows_sha256(assignments):
        raise ValueError("sealed assignment rows changed")
    if seal.get("human_outcome_present_before_seal") is not False:
        raise ValueError("seal does not prove an outcome-blind preparation")
    if seal.get("expected_manufacturing_present_before_seal") is not False:
        raise ValueError("seal does not prove manufacturing evidence was absent")
    return {
        "root": root,
        "seal": seal,
        "predictions": predictions,
        "formula_manifest": formula_manifest,
        "study_key": study_key,
        "assignments": assignments,
    }


def _required_external_text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} characters")
    return value.strip()


def _required_external_number(
    value: Any, name: str, *, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _validate_external_timestamp(value: Any, name: str) -> str:
    text = _required_external_text(value, name)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def validate_manufacturing_execution(
    verified_study: Mapping[str, Any],
    manufacturing_path: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Bind observed ratings to exact lots, quantitative batches and coded vials."""
    root = Path(verified_study["root"])
    seal = verified_study["seal"]
    manufacturing = Path(manufacturing_path).expanduser().resolve(strict=True)
    expected = (root / seal["expected_manufacturing_relative_path"]).resolve()
    if manufacturing != expected:
        raise ValueError("manufacturing evidence path differs from preregistration")
    payload = json.loads(manufacturing.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != MANUFACTURING_SCHEMA:
        raise ValueError("unsupported manufacturing evidence schema")
    if payload.get("study_id") != seal["study_id"]:
        raise ValueError("manufacturing evidence study ID mismatch")
    if payload.get("execution_authorized") is not True:
        raise ValueError("laboratory did not attest execution authorization")
    laboratory_id = _required_external_text(payload.get("laboratory_id"), "laboratory_id")

    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("manufacturing environment must be an object")
    for name in (
        "sop_id",
        "sop_version",
        "product_base_id",
        "product_base_lot",
        "dilution_solvent",
        "container_lot",
    ):
        _required_external_text(environment.get(name), f"environment.{name}")
    numeric_environment = {
        "maturation_hours": (24.0, 10_000.0),
        "test_temperature_c": (15.0, 30.0),
        "relative_humidity_percent": (20.0, 80.0),
        "headspace_equilibration_minutes": (1.0, 120.0),
        "sniff_interval_seconds": (30.0, 3_600.0),
        "vial_fill_ml": (0.1, 100.0),
        "vial_headspace_ml": (0.1, 100.0),
    }
    for name, bounds in numeric_environment.items():
        _required_external_number(
            environment.get(name),
            f"environment.{name}",
            minimum=bounds[0],
            maximum=bounds[1],
        )

    approval_documents: list[str] = []
    for approval_name in ("safety_approval", "ethics_approval"):
        approval = payload.get(approval_name)
        if not isinstance(approval, dict):
            raise ValueError(f"{approval_name} must be an object")
        _required_external_text(
            approval.get("approval_id"), f"{approval_name}.approval_id"
        )
        approval_documents.append(
            _required_external_text(
                approval.get("document"), f"{approval_name}.document"
            )
        )
    if len(set(approval_documents)) != len(approval_documents):
        raise ValueError("safety and ethics approvals must be distinct documents")
    base_documents = [
        _required_external_text(
            environment.get("product_base_coa_document"),
            "environment.product_base_coa_document",
        ),
        _required_external_text(
            environment.get("product_base_sds_document"),
            "environment.product_base_sds_document",
        ),
    ]
    if len(set(base_documents)) != len(base_documents):
        raise ValueError("product-base COA and SDS must be distinct documents")

    formula_rows = verified_study["formula_manifest"].get("formulas", [])
    expected_formulas = {row["formula_id"]: row for row in formula_rows}
    batches = payload.get("formula_batches")
    if not isinstance(batches, list) or len(batches) != FORMULA_COUNT:
        raise ValueError("manufacturing evidence must contain exactly 24 formula batches")
    batch_ids: dict[str, str] = {}
    referenced_documents: set[str] = set(approval_documents + base_documents)
    document_claims: dict[str, tuple[str, ...]] = {
        approval_documents[0]: ("safety_approval",),
        approval_documents[1]: ("ethics_approval",),
        base_documents[0]: (
            "product_base_coa",
            environment["product_base_id"],
            environment["product_base_lot"],
        ),
        base_documents[1]: (
            "product_base_sds",
            environment["product_base_id"],
            environment["product_base_lot"],
        ),
    }
    if len(document_claims) != 4:
        raise ValueError("approval and product-base evidence documents must be distinct")
    seen_formulas: set[str] = set()
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("formula batch entries must be objects")
        formula_id = _required_external_text(batch.get("formula_id"), "formula_id")
        if formula_id in seen_formulas or formula_id not in expected_formulas:
            raise ValueError("formula batch IDs must exactly match the sealed formulas")
        seen_formulas.add(formula_id)
        expected_formula = expected_formulas[formula_id]
        if batch.get("formula_fingerprint_sha256") != expected_formula.get(
            "formula_fingerprint_sha256"
        ):
            raise ValueError(f"formula fingerprint changed for {formula_id}")
        batch_id = _required_external_text(
            batch.get("concentrate_batch_id"), f"{formula_id}.concentrate_batch_id"
        )
        if batch_id in batch_ids.values():
            raise ValueError("concentrate batch IDs must be unique")
        batch_ids[formula_id] = batch_id
        stated_total = _required_external_number(
            batch.get("weighed_total_g"),
            f"{formula_id}.weighed_total_g",
            minimum=1.0,
            maximum=1_000_000.0,
        )
        expected_lines = {
            row["ingredient_id"]: row for row in expected_formula["lines"]
        }
        lots = batch.get("ingredient_lots")
        if not isinstance(lots, list) or len(lots) != len(expected_lines):
            raise ValueError(f"ingredient lot count changed for {formula_id}")
        observed_masses: dict[str, float] = {}
        for lot in lots:
            if not isinstance(lot, dict):
                raise ValueError("ingredient lot entries must be objects")
            ingredient_id = _required_external_text(
                lot.get("ingredient_id"), f"{formula_id}.ingredient_id"
            )
            if ingredient_id in observed_masses or ingredient_id not in expected_lines:
                raise ValueError(f"ingredient IDs changed for {formula_id}")
            stated_target_percent = _required_external_number(
                lot.get("target_concentrate_percent"),
                f"{formula_id}/{ingredient_id}.target_concentrate_percent",
                minimum=0.0,
                maximum=100.0,
            )
            if not math.isclose(
                stated_target_percent,
                float(expected_lines[ingredient_id]["target_concentrate_percent"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(
                    f"target concentrate percentage changed for {formula_id}/{ingredient_id}"
                )
            stated_active_strength = _required_external_number(
                lot.get("active_strength_percent"),
                f"{formula_id}/{ingredient_id}.active_strength_percent",
                minimum=0.000001,
                maximum=100.0,
            )
            if not math.isclose(
                stated_active_strength,
                float(expected_lines[ingredient_id]["active_strength_percent"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ) or lot.get("carrier", "") != expected_lines[ingredient_id]["carrier"]:
                raise ValueError(
                    f"active strength or carrier changed for {formula_id}/{ingredient_id}"
                )
            observed_masses[ingredient_id] = _required_external_number(
                lot.get("actual_mass_g"),
                f"{formula_id}/{ingredient_id}.actual_mass_g",
                minimum=0.000001,
                maximum=1_000_000.0,
            )
            supplier_sku = _required_external_text(
                lot.get("supplier_sku"),
                f"{formula_id}/{ingredient_id}.supplier_sku",
            )
            lot_number = _required_external_text(
                lot.get("lot_number"),
                f"{formula_id}/{ingredient_id}.lot_number",
            )
            for document_name in ("coa_document", "sds_document", "ifra_document"):
                document_path = _required_external_text(
                    lot.get(document_name),
                    f"{formula_id}/{ingredient_id}.{document_name}",
                )
                referenced_documents.add(document_path)
                identity = (
                    (ingredient_id, supplier_sku, lot_number)
                    if document_name == "coa_document"
                    else (ingredient_id, supplier_sku)
                )
                claim = (document_name, *identity)
                if (
                    document_path in document_claims
                    and document_claims[document_path] != claim
                ):
                    raise ValueError(
                        "one supplier document was reused across incompatible materials or lots"
                    )
                document_claims[document_path] = claim
            if len(
                {
                    lot["coa_document"],
                    lot["sds_document"],
                    lot["ifra_document"],
                }
            ) != 3:
                raise ValueError("COA, SDS and IFRA evidence must be distinct documents")
        actual_total = sum(observed_masses.values())
        mass_balance_error = 100.0 * abs(actual_total - stated_total) / stated_total
        if mass_balance_error > 0.5:
            raise ValueError(f"mass balance exceeds 0.5% for {formula_id}")
        for ingredient_id, mass in observed_masses.items():
            observed_percent = 100.0 * mass / actual_total
            if abs(
                observed_percent
                - float(expected_lines[ingredient_id]["target_concentrate_percent"])
            ) > 0.25:
                raise ValueError(
                    f"weighed formula deviates by more than 0.25 points for "
                    f"{formula_id}/{ingredient_id}"
                )
    if seen_formulas != set(expected_formulas):
        raise ValueError("formula manufacturing evidence is incomplete")

    expected_samples: dict[str, tuple[str, float]] = {}
    for key in verified_study["study_key"]:
        expected_samples[key["code_a"]] = (
            key["formula_a"],
            float(key["concentration_a_percent"]),
        )
        expected_samples[key["code_b"]] = (
            key["formula_b"],
            float(key["concentration_b_percent"]),
        )
    vials = payload.get("sample_vials")
    if not isinstance(vials, list) or len(vials) != PAIR_COUNT * 2:
        raise ValueError("manufacturing evidence must contain exactly 240 coded vials")
    seen_codes: set[str] = set()
    vial_ids: set[str] = set()
    for vial in vials:
        if not isinstance(vial, dict):
            raise ValueError("sample vial entries must be objects")
        code = _required_external_text(vial.get("code"), "sample_vial.code")
        if code in seen_codes or code not in expected_samples:
            raise ValueError("sample vial codes must exactly match the sealed study key")
        seen_codes.add(code)
        formula_id, concentration = expected_samples[code]
        stated_concentration = _required_external_number(
            vial.get("finished_product_concentration_percent"),
            "sample_vial.finished_product_concentration_percent",
            minimum=0.000001,
            maximum=100.0,
        )
        if vial.get("formula_id") != formula_id or not math.isclose(
            stated_concentration, concentration, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"sample vial formula or concentration changed for code {code}")
        if vial.get("concentrate_batch_id") != batch_ids[formula_id]:
            raise ValueError(f"sample vial batch binding changed for code {code}")
        concentrate_mass = _required_external_number(
            vial.get("concentrate_mass_g"),
            "sample_vial.concentrate_mass_g",
            minimum=0.000001,
            maximum=10_000.0,
        )
        base_mass = _required_external_number(
            vial.get("base_mass_g"),
            "sample_vial.base_mass_g",
            minimum=0.000001,
            maximum=10_000.0,
        )
        finished_mass = _required_external_number(
            vial.get("finished_mass_g"),
            "sample_vial.finished_mass_g",
            minimum=0.000001,
            maximum=10_000.0,
        )
        if 100.0 * abs(concentrate_mass + base_mass - finished_mass) / finished_mass > 0.5:
            raise ValueError(f"sample vial mass balance exceeds 0.5% for code {code}")
        actual_concentration = 100.0 * concentrate_mass / (
            concentrate_mass + base_mass
        )
        if abs(actual_concentration - concentration) > 0.25:
            raise ValueError(
                f"sample vial dilution deviates by more than 0.25 points for code {code}"
            )
        vial_id = _required_external_text(vial.get("vial_id"), "sample_vial.vial_id")
        if vial_id in vial_ids:
            raise ValueError("physical vial IDs must be unique")
        vial_ids.add(vial_id)
        _validate_external_timestamp(vial.get("prepared_at"), "sample_vial.prepared_at")
    if seen_codes != set(expected_samples):
        raise ValueError("sample vial evidence is incomplete")

    document_hashes = payload.get("document_hashes")
    if not isinstance(document_hashes, dict) or set(document_hashes) != referenced_documents:
        raise ValueError("document hash manifest must exactly cover referenced evidence")
    if len(document_hashes) > 512:
        raise ValueError("supporting evidence document count exceeds 512")
    evidence_directory = Path(evidence_root).expanduser().resolve(strict=True)
    if not evidence_directory.is_dir():
        raise ValueError("evidence_root must be a directory")
    artifact_paths: dict[str, Path] = {"manufacturing_execution_json": manufacturing}
    verified_documents: dict[str, str] = {}
    resolved_documents: set[Path] = set()
    for relative, claimed_hash in sorted(document_hashes.items()):
        if not isinstance(relative, str):
            raise ValueError("supporting evidence paths must be text")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("supporting evidence path is unsafe")
        document = (evidence_directory / relative_path).resolve(strict=True)
        if evidence_directory not in document.parents or not document.is_file():
            raise ValueError("supporting evidence escapes evidence_root")
        if document in resolved_documents:
            raise ValueError("supporting evidence aliases the same file more than once")
        resolved_documents.add(document)
        if document.stat().st_size <= 0 or document.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("supporting evidence document has an invalid size")
        actual_hash = sha256_file(document)
        if claimed_hash != actual_hash:
            raise ValueError(f"supporting evidence hash mismatch: {relative}")
        normalized = relative_path.as_posix()
        artifact_paths[f"supporting_document::{normalized}"] = document
        verified_documents[normalized] = actual_hash
    return {
        "laboratory_id": laboratory_id,
        "manufacturing_sha256": sha256_file(manufacturing),
        "formula_batch_count": len(batches),
        "sample_vial_count": len(vials),
        "supporting_document_count": len(verified_documents),
        "supporting_document_hashes": verified_documents,
        "artifact_paths": artifact_paths,
    }


def _numeric_rating(value: str, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ValueError(f"{name} must be finite and between 0 and 100")
    return result


def validate_outcomes(
    assignments: Sequence[Mapping[str, str]], outcomes_path: str | Path
) -> list[dict[str, Any]]:
    columns, raw_rows = _read_csv(Path(outcomes_path).expanduser().resolve(strict=True))
    if tuple(columns) != OUTCOME_COLUMNS:
        raise ValueError("human outcome CSV columns differ from the sealed template")
    if len(raw_rows) != PAIR_COUNT * RATINGS_PER_PAIR:
        raise ValueError("human outcome CSV must contain exactly 2400 rows")
    expected = {row["assignment_id"]: dict(row) for row in assignments}
    if len(expected) != len(assignments):
        raise ValueError("sealed template has duplicate assignment IDs")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in raw_rows:
        assignment_id = str(row.get("assignment_id", "")).strip()
        if assignment_id in seen:
            raise ValueError(f"duplicate outcome assignment: {assignment_id}")
        seen.add(assignment_id)
        template = expected.get(assignment_id)
        if template is None:
            raise ValueError(f"unknown outcome assignment: {assignment_id}")
        for field in ("participant_id", "pair_id", "code_left", "code_right"):
            if row.get(field) != template[field]:
                raise ValueError(f"outcome assignment identity changed: {assignment_id}/{field}")
        notes = str(row.get("notes", ""))
        if len(notes) > 1000:
            raise ValueError("outcome notes must be at most 1000 characters")
        validated.append(
            {
                "assignment_id": assignment_id,
                "participant_id": row["participant_id"],
                "pair_id": row["pair_id"],
                "code_left": row["code_left"],
                "code_right": row["code_right"],
                "similarity": _numeric_rating(row["similarity_0_100"], "similarity"),
                "confidence": _numeric_rating(row["confidence_0_100"], "confidence"),
                "intensity_left": _numeric_rating(
                    row["intensity_left_0_100"], "intensity_left"
                ),
                "intensity_right": _numeric_rating(
                    row["intensity_right_0_100"], "intensity_right"
                ),
            }
        )
    if seen != set(expected):
        raise ValueError("human outcome assignments are incomplete")
    participant_counts = Counter(row["participant_id"] for row in validated)
    pair_counts = Counter(row["pair_id"] for row in validated)
    if len(participant_counts) != PARTICIPANT_COUNT or set(participant_counts.values()) != {
        PAIRS_PER_PARTICIPANT
    }:
        raise ValueError("human outcomes violate participant balance")
    if len(pair_counts) != PAIR_COUNT or set(pair_counts.values()) != {RATINGS_PER_PAIR}:
        raise ValueError("human outcomes violate pair balance")
    return validated


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _ccc(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))
    covariance = float(np.mean((left - left_mean) * (right - right_mean)))
    denominator = float(np.var(left) + np.var(right) + (left_mean - right_mean) ** 2)
    return 0.0 if denominator <= 1e-12 else 2.0 * covariance / denominator


def _metrics(prediction: np.ndarray, human: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(prediction - human)))
    return {
        "absolute_similarity_accuracy_percent": 100.0 - mae,
        "mae_0_100": mae,
        "rmse_0_100": float(np.sqrt(np.mean((prediction - human) ** 2))),
        "pearson": _pearson(prediction, human),
        "spearman": _spearman(prediction, human),
        "concordance_correlation_coefficient": _ccc(prediction, human),
    }


def _percentile_interval(values: Sequence[float]) -> list[float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def _split_half_reliability(
    validated: Sequence[Mapping[str, Any]], *, seed: int, repeats: int
) -> dict[str, Any]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in validated:
        by_pair[str(row["pair_id"])].append(float(row["similarity"]))
    rng = np.random.default_rng(seed)
    corrected: list[float] = []
    for _ in range(repeats):
        left_means: list[float] = []
        right_means: list[float] = []
        for pair_id in sorted(by_pair):
            values = np.asarray(by_pair[pair_id], dtype=float)
            order = rng.permutation(len(values))
            midpoint = len(values) // 2
            left_means.append(float(np.mean(values[order[:midpoint]])))
            right_means.append(float(np.mean(values[order[midpoint:]])))
        correlation = _spearman(np.asarray(left_means), np.asarray(right_means))
        corrected_value = (
            0.0
            if correlation <= -1.0 + 1e-12
            else 2.0 * correlation / (1.0 + correlation)
        )
        corrected.append(max(0.0, min(1.0, corrected_value)))
    return {
        "method": "pairwise_random_split_half_spearman_brown",
        "repeats": repeats,
        "median": float(np.median(corrected)),
        "interval_95": _percentile_interval(corrected),
    }


def evaluate_outcomes(
    predictions: Mapping[str, Any],
    validated: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    reliability_repeats: int = DEFAULT_RELIABILITY_REPEATS,
    seed: int = 20260828,
) -> dict[str, Any]:
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    if reliability_repeats < 100:
        raise ValueError("reliability_repeats must be at least 100")
    pair_predictions = {row["pair_id"]: row for row in predictions["pairs"]}
    if len(pair_predictions) != PAIR_COUNT:
        raise ValueError("prediction pair count differs from the sealed design")
    by_pair: dict[str, list[float]] = defaultdict(list)
    participant_ids = sorted({str(row["participant_id"]) for row in validated})
    pair_ids = sorted(pair_predictions)
    participant_index = {value: index for index, value in enumerate(participant_ids)}
    pair_index = {value: index for index, value in enumerate(pair_ids)}
    observation_participants: list[int] = []
    observation_pairs: list[int] = []
    observation_values: list[float] = []
    for row in validated:
        pair_id = str(row["pair_id"])
        by_pair[pair_id].append(float(row["similarity"]))
        observation_participants.append(participant_index[str(row["participant_id"])])
        observation_pairs.append(pair_index[pair_id])
        observation_values.append(float(row["similarity"]))
    human = np.asarray([np.mean(by_pair[pair_id]) for pair_id in pair_ids], dtype=float)
    candidate = np.asarray(
        [pair_predictions[pair_id]["predicted_similarity_0_100"] for pair_id in pair_ids],
        dtype=float,
    )
    baseline = np.asarray(
        [
            pair_predictions[pair_id]["fixed_component_overlap_baseline_0_100"]
            for pair_id in pair_ids
        ],
        dtype=float,
    )
    primary_indices = np.asarray(
        [
            index
            for index, pair_id in enumerate(pair_ids)
            if pair_predictions[pair_id]["pair_type"] == "ordinary_formula_pair"
        ],
        dtype=int,
    )
    if len(primary_indices) != ORDINARY_PAIR_COUNT:
        raise ValueError("primary ordinary-pair population differs from preregistration")
    candidate_metrics = _metrics(candidate[primary_indices], human[primary_indices])
    baseline_metrics = _metrics(baseline[primary_indices], human[primary_indices])
    primary_pair_ids = {pair_ids[index] for index in primary_indices}
    reliability = _split_half_reliability(
        [row for row in validated if str(row["pair_id"]) in primary_pair_ids],
        seed=seed + 1,
        repeats=reliability_repeats,
    )
    attenuation_ceiling = math.sqrt(max(reliability["median"], 1e-12))
    normalized_spearman = max(
        -1.0, min(1.0, candidate_metrics["spearman"] / attenuation_ceiling)
    )

    participant_array = np.asarray(observation_participants, dtype=int)
    pair_array = np.asarray(observation_pairs, dtype=int)
    value_array = np.asarray(observation_values, dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_draws):
        participant_counts = np.bincount(
            rng.integers(0, len(participant_ids), size=len(participant_ids)),
            minlength=len(participant_ids),
        )
        sampled_pair_counts = np.bincount(
            rng.integers(0, len(pair_ids), size=len(pair_ids)), minlength=len(pair_ids)
        )
        pair_means = np.full(len(pair_ids), np.nan, dtype=float)
        observation_weights = participant_counts[participant_array]
        for index in np.flatnonzero(sampled_pair_counts):
            mask = pair_array == index
            weights = observation_weights[mask]
            if weights.sum() > 0:
                pair_means[index] = float(np.average(value_array[mask], weights=weights))
        selected_indices = np.repeat(
            np.flatnonzero(np.isfinite(pair_means) & (sampled_pair_counts > 0)),
            sampled_pair_counts[np.isfinite(pair_means) & (sampled_pair_counts > 0)],
        )
        primary_selected_indices = np.asarray(
            [
                index
                for index in selected_indices
                if pair_predictions[pair_ids[index]]["pair_type"]
                == "ordinary_formula_pair"
            ],
            dtype=int,
        )
        if len(primary_selected_indices) < 25:
            continue
        candidate_draw = _metrics(
            candidate[primary_selected_indices], pair_means[primary_selected_indices]
        )
        baseline_draw = _metrics(
            baseline[primary_selected_indices], pair_means[primary_selected_indices]
        )
        for name, value in candidate_draw.items():
            draws[f"candidate::{name}"].append(value)
        for name, value in baseline_draw.items():
            draws[f"baseline::{name}"].append(value)
        for name in (
            "absolute_similarity_accuracy_percent",
            "pearson",
            "spearman",
            "concordance_correlation_coefficient",
        ):
            draws[f"difference::{name}"].append(
                candidate_draw[name] - baseline_draw[name]
            )
        control_indices = [
            index
            for index in selected_indices
            if pair_predictions[pair_ids[index]]["pair_type"]
            == "identical_formula_control"
        ]
        if control_indices:
            draws["identical_control_mean"].append(
                float(np.mean(pair_means[control_indices]))
            )
    if len(draws["candidate::spearman"]) < int(bootstrap_draws * 0.99):
        raise RuntimeError("too many crossed bootstrap replicates were invalid")
    candidate_intervals = {
        name: _percentile_interval(draws[f"candidate::{name}"])
        for name in candidate_metrics
    }
    baseline_intervals = {
        name: _percentile_interval(draws[f"baseline::{name}"])
        for name in baseline_metrics
    }
    differences = {
        name: {
            "point": candidate_metrics[name] - baseline_metrics[name],
            "interval_95": _percentile_interval(draws[f"difference::{name}"]),
        }
        for name in (
            "absolute_similarity_accuracy_percent",
            "pearson",
            "spearman",
            "concordance_correlation_coefficient",
        )
    }
    identical_indices = np.asarray(
        [
            index
            for index, pair_id in enumerate(pair_ids)
            if pair_predictions[pair_id]["pair_type"] == "identical_formula_control"
        ],
        dtype=int,
    )
    identical_mean = float(np.mean(human[identical_indices]))
    gate_checks = {
        "absolute_similarity_accuracy_lower_95_at_least_90": (
            candidate_intervals["absolute_similarity_accuracy_percent"][0] >= 90.0
        ),
        "spearman_lower_95_at_least_0_90": (
            candidate_intervals["spearman"][0] >= 0.90
        ),
        "pearson_lower_95_at_least_0_90": (
            candidate_intervals["pearson"][0] >= 0.90
        ),
        "identical_control_mean_at_least_90": identical_mean >= 90.0,
        "identical_control_lower_95_at_least_85": (
            _percentile_interval(draws["identical_control_mean"])[0] >= 85.0
        ),
    }
    pair_results = [
        {
            "pair_id": pair_id,
            "pair_type": pair_predictions[pair_id]["pair_type"],
            "human_mean_similarity_0_100": round(float(human[index]), 6),
            "predicted_similarity_0_100": round(float(candidate[index]), 6),
            "fixed_baseline_0_100": round(float(baseline[index]), 6),
            "absolute_error_0_100": round(
                abs(float(candidate[index]) - float(human[index])), 6
            ),
            "rating_count": len(by_pair[pair_id]),
        }
        for index, pair_id in enumerate(pair_ids)
    ]
    return {
        "analysis_contract": {
            "primary_population": (
                "all_80_participants_intention_to_test_on_100_ordinary_formula_pairs"
            ),
            "controls_excluded_from_primary_accuracy_and_correlation": True,
            "bootstrap": "participant_by_pair_crossed_cluster_resampling",
            "bootstrap_draws_requested": bootstrap_draws,
            "bootstrap_draws_valid": len(draws["candidate::spearman"]),
            "random_seed": seed,
            "post_outcome_refit": False,
        },
        "counts": {
            "participants": len(participant_ids),
            "pairs": len(pair_ids),
            "primary_ordinary_pairs": len(primary_indices),
            "ratings": len(validated),
            "ratings_per_pair_min": min(len(values) for values in by_pair.values()),
            "ratings_per_pair_max": max(len(values) for values in by_pair.values()),
        },
        "candidate": {
            "metrics": candidate_metrics,
            "intervals_95": candidate_intervals,
            "human_ceiling_normalized_spearman": normalized_spearman,
        },
        "fixed_component_overlap_baseline": {
            "metrics": baseline_metrics,
            "intervals_95": baseline_intervals,
        },
        "candidate_minus_baseline": differences,
        "human_reliability": reliability,
        "identical_controls": {
            "pair_count": len(identical_indices),
            "mean_similarity_0_100": identical_mean,
            "crossed_bootstrap_interval_95": _percentile_interval(
                draws["identical_control_mean"]
            ),
        },
        "statistical_gate_checks": gate_checks,
        "statistical_ninety_percent_gate_passed": all(gate_checks.values()),
        "pair_results": pair_results,
    }


def _ledger_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    previous = "0" * 64
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"evidence ledger line {line_number} is invalid") from error
        if not isinstance(value, dict):
            raise ValueError("evidence ledger entries must be objects")
        if value.get("previous_entry_sha256") != previous:
            raise ValueError(f"evidence ledger chain broke at line {line_number}")
        unsigned = dict(value)
        claimed = unsigned.pop("entry_sha256", None)
        actual = hashlib.sha256(canonical_json(unsigned)).hexdigest()
        if claimed != actual:
            raise ValueError(f"evidence ledger hash changed at line {line_number}")
        previous = actual
        entries.append(value)
    return entries


def finalize_study(
    study_dir: str | Path,
    *,
    outcomes_path: str | Path,
    manufacturing_evidence_path: str | Path,
    evidence_root: str | Path,
    signature_envelope_path: str | Path,
    trust_root_path: str | Path,
    openssl: str | Path,
    timestamp_response_path: str | Path,
    timestamp_ca_path: str | Path,
    timestamp_tsa_path: str | Path,
    report_path: str | Path,
    ledger_path: str | Path,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    reliability_repeats: int = DEFAULT_RELIABILITY_REPEATS,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Verify independent evidence and consume it exactly once."""
    verified = verify_study_seal(study_dir)
    root = verified["root"]
    seal = verified["seal"]
    study_id = str(seal["study_id"])
    outcome = Path(outcomes_path).expanduser().resolve(strict=True)
    expected_outcome = (root / seal["expected_outcome_relative_path"]).resolve()
    if outcome != expected_outcome:
        raise ValueError("outcome path does not match the preregistered study path")
    timestamp = verify_rfc3161_timestamp(
        openssl=openssl,
        seal_path=root / "seal.json",
        response_path=timestamp_response_path,
        ca_path=timestamp_ca_path,
        tsa_path=timestamp_tsa_path,
    )
    timestamp_record_path = root / "timestamp" / "verification.json"
    timestamp_record = json.loads(
        timestamp_record_path.resolve(strict=True).read_text(encoding="utf-8")
    )
    if (
        timestamp_record.get("study_id") != study_id
        or timestamp_record.get("seal_sha256") != sha256_file(root / "seal.json")
        or timestamp_record.get("human_outcome_present_at_verification") is not False
        or timestamp_record.get("manufacturing_evidence_present_at_verification")
        is not False
        or timestamp_record.get("timestamp") != timestamp
    ):
        raise ValueError("pre-outcome timestamp verification record is invalid")
    protocol_hash = sha256_file(root / "study_protocol.json")
    seal_hash = sha256_file(root / "seal.json")
    manufacturing = validate_manufacturing_execution(
        verified,
        manufacturing_evidence_path,
        evidence_root,
    )
    envelope_path = Path(signature_envelope_path).expanduser().resolve(strict=True)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    trust_root = EvidenceTrustRoot.from_json_file(trust_root_path)
    laboratory = trust_root.verify(
        envelope,
        {
            "human_outcomes_csv": outcome,
            **manufacturing["artifact_paths"],
        },
        expected_artifact_type=LAB_ARTIFACT_TYPE,
        expected_scope={
            "study_id": study_id,
            "protocol_sha256": protocol_hash,
            "seal_sha256": seal_hash,
            "manufacturing_sha256": manufacturing["manufacturing_sha256"],
        },
        allowed_roles={"sensory_laboratory"},
    )
    timestamp_moment = datetime.fromisoformat(timestamp["timestamp_utc"])
    laboratory_issued = datetime.fromisoformat(laboratory.issued_at)
    if timestamp_moment >= laboratory_issued:
        raise ValueError("prediction timestamp must precede laboratory evidence issuance")
    validated = validate_outcomes(verified["assignments"], outcome)
    analysis = evaluate_outcomes(
        verified["predictions"],
        validated,
        bootstrap_draws=bootstrap_draws,
        reliability_repeats=reliability_repeats,
        seed=seed,
    )
    evidence_gate_checks = {
        "prediction_seal_verified": True,
        "rfc3161_timestamp_verified": bool(timestamp["verified"]),
        "independent_laboratory_signature_verified": True,
        "timestamp_precedes_laboratory_evidence": True,
        "exact_lot_formula_base_headspace_and_vial_binding": (
            manufacturing["formula_batch_count"] == FORMULA_COUNT
            and manufacturing["sample_vial_count"] == PAIR_COUNT * 2
            and manufacturing["supporting_document_count"] >= 2
        ),
        "exact_participant_and_assignment_coverage": (
            analysis["counts"]
            == {
                "participants": PARTICIPANT_COUNT,
                "pairs": PAIR_COUNT,
                "primary_ordinary_pairs": ORDINARY_PAIR_COUNT,
                "ratings": PAIR_COUNT * RATINGS_PER_PAIR,
                "ratings_per_pair_min": RATINGS_PER_PAIR,
                "ratings_per_pair_max": RATINGS_PER_PAIR,
            }
        ),
    }
    ninety_pass = all(evidence_gate_checks.values()) and bool(
        analysis["statistical_ninety_percent_gate_passed"]
    )
    report = {
        "schema": STUDY_SCHEMA,
        "study_id": study_id,
        "finalized_at": _utc_now(),
        "outcome_sha256": sha256_file(outcome),
        "signature_envelope_sha256": sha256_file(envelope_path),
        "prediction_seal_sha256": seal_hash,
        "timestamp": timestamp,
        "laboratory_evidence": asdict(laboratory),
        "manufacturing_evidence": {
            key: value
            for key, value in manufacturing.items()
            if key != "artifact_paths"
        },
        "analysis": analysis,
        "evidence_gate_checks": evidence_gate_checks,
        "human_olfactory_similarity_90_gate_passed": ninety_pass,
        "claim_status": (
            "independent_blind_90_percent_gate_passed"
            if ninety_pass
            else "independent_blind_90_percent_gate_failed"
        ),
        "claim_boundary": (
            "This endpoint concerns the exact sealed formulas, concentrations, "
            "participants and laboratory protocol only; it is not universal "
            "text-to-fragrance accuracy or manufacturing authorization."
        ),
    }

    report_target = Path(report_path).expanduser().resolve()
    ledger_target = Path(ledger_path).expanduser().resolve()
    if report_target != (root / seal["expected_report_relative_path"]).resolve():
        raise ValueError("final report path differs from the sealed study contract")
    if ledger_target != (root.parent / seal["expected_ledger_filename"]).resolve():
        raise ValueError("evidence ledger path differs from the sealed study contract")
    lock_path = ledger_target.with_suffix(ledger_target.suffix + ".lock")
    ledger_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("evidence ledger is locked by another finalizer") from error
    try:
        os.close(lock_fd)
        entries = _ledger_entries(ledger_target)
        outcome_hash = report["outcome_sha256"]
        if any(
            entry.get("study_id") == study_id
            or entry.get("outcome_sha256") == outcome_hash
            for entry in entries
        ):
            raise RuntimeError("study or outcome bytes were already consumed")
        report_target.parent.mkdir(parents=True, exist_ok=True)
        if report_target.exists():
            raise FileExistsError("final report path already exists")
        pending_report = report_target.with_name(
            report_target.name + f".pending-{secrets.token_hex(8)}"
        )
        _write_json(pending_report, report)
        ledger_entry = {
            "schema": STUDY_SCHEMA,
            "study_id": study_id,
            "consumed_at": report["finalized_at"],
            "outcome_sha256": outcome_hash,
            "seal_sha256": seal_hash,
            "report_sha256": sha256_file(pending_report),
            "laboratory_artifact_id": laboratory.artifact_id,
            "laboratory_signer_id": laboratory.signer_id,
            "ninety_percent_gate_passed": ninety_pass,
            "previous_entry_sha256": (
                entries[-1]["entry_sha256"] if entries else "0" * 64
            ),
        }
        ledger_entry["entry_sha256"] = hashlib.sha256(
            canonical_json(ledger_entry)
        ).hexdigest()
        ledger_committed = False
        try:
            with ledger_target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(ledger_entry, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            ledger_committed = True
            os.replace(pending_report, report_target)
        finally:
            if not ledger_committed:
                pending_report.unlink(missing_ok=True)
    finally:
        lock_path.unlink(missing_ok=True)
    return report
