"""Deterministic Korean/English natural-language scent brief parser."""

from __future__ import annotations

import re
from dataclasses import replace

from .catalog import IngredientCatalog, find_text_spans, normalize_name
from .models import RecipeConstraints, ScentBrief, normalize_profile
from .odor_descriptors import load_builtin_odor_descriptor_lexicon
from .semantic_ontology import ScentSemanticOntology


class BriefParseError(ValueError):
    pass


class UnsupportedOdorDescriptorError(BriefParseError):
    """A descriptor was recognized but has no honest formulation projection."""

    def __init__(self, descriptors: list[str] | tuple[str, ...]):
        self.descriptors = tuple(sorted(set(descriptors)))
        joined = ", ".join(self.descriptors)
        super().__init__(
            "현재 안전 원료 프로필로 직접 조향할 수 없는 세부 냄새 표현입니다: "
            f"{joined}. 더 넓은 향조로 바꿔 요청하세요."
        )


KEYWORDS: dict[str, tuple[str, ...]] = {
    "citrus": (
        "시트러스",
        "감귤",
        "레몬",
        "오렌지",
        "자몽",
        "베르가못",
        "유자",
        "라임",
        "citrus",
        "citrusy",
        "lemon",
        "orange",
        "bergamot",
        "grapefruit",
        "lime",
        "yuzu",
    ),
    "fresh": (
        "상쾌",
        "산뜻",
        "청량",
        "시원",
        "맑은",
        "맑고",
        "가벼운",
        "투명한",
        "공기 같은",
        "fresh",
        "refreshing",
        "airy",
        "light",
        "crisp",
        "transparent",
    ),
    "clean": (
        "깨끗",
        "클린",
        "비누",
        "세탁",
        "코튼",
        "샴푸",
        "린넨",
        "살결",
        "clean",
        "soapy",
        "laundry",
        "cotton",
        "linen",
    ),
    "green": (
        "그린",
        "풀잎",
        "잔디",
        "잎사귀",
        "줄기",
        "허브잎",
        "이끼",
        "green",
        "leaf",
        "leaves",
        "grass",
        "stem",
        "foliage",
        "moss",
        "mossy",
    ),
    "aquatic": (
        "아쿠아틱",
        "아쿠아",
        "바다",
        "해변",
        "물향",
        "비온뒤",
        "비 온 뒤",
        "수생",
        "오존",
        "aquatic",
        "marine",
        "ocean",
        "rain",
        "ozonic",
    ),
    "floral": (
        "플로럴",
        "꽃향",
        "꽃다발",
        "꽃",
        "floral",
        "flower",
        "flowers",
        "bouquet",
    ),
    "rose": ("장미", "로즈", "rose", "roses", "rosy"),
    "white_floral": (
        "자스민",
        "재스민",
        "치자",
        "은방울꽃",
        "화이트플로럴",
        "화이트 플로럴",
        "jasmine",
        "jasmin",
        "gardenia",
        "white floral",
    ),
    "fruity": (
        "과일",
        "복숭아",
        "배향",
        "사과",
        "베리",
        "열대과일",
        "fruity",
        "fruit",
        "fruits",
        "peach",
        "pear",
        "apple",
        "berry",
        "berries",
    ),
    "spicy": (
        "스파이시",
        "후추",
        "향신료",
        "생강",
        "카다멈",
        "spicy",
        "pepper",
        "ginger",
        "cardamom",
        "spices",
    ),
    "aromatic": (
        "아로마틱",
        "허브",
        "라벤더",
        "로즈마리",
        "aromatic",
        "herbal",
        "herb",
        "herbaceous",
        "basil",
        "fennel",
        "lavender",
        "rosemary",
        "herbs",
    ),
    "woody": (
        "우디",
        "우드",
        "나무",
        "삼나무",
        "샌달",
        "백단",
        "숲",
        "드라이 우드",
        "wood",
        "woods",
        "woody",
        "forest",
        "cedar",
        "sandal",
    ),
    "amber": ("앰버", "따뜻", "포근", "수지", "amber", "warm", "resinous"),
    "musky": ("머스크", "사향", "살냄새", "musky", "musk", "skin scent"),
    "gourmand": (
        "달콤",
        "단맛",
        "단 향",
        "단향",
        "바닐라",
        "카라멜",
        "디저트",
        "초콜릿",
        "구르망",
        "sweet",
        "sweetness",
        "vanilla",
        "caramel",
        "gourmand",
    ),
    "powdery": ("파우더", "보송", "분내", "화장품", "powdery", "cosmetic"),
    "smoky": ("스모키", "연기", "그을린", "인센스", "smoky", "smoke", "incense"),
    "leathery": ("가죽", "레더", "스웨이드", "leather", "suede"),
    "earthy": ("흙", "이끼", "축축한 숲", "땅", "earthy", "mossy", "soil"),
}

BEFORE_NEGATION = re.compile(r"(?:(?:안|덜)|\b(?:not|no|without|exclude|avoid))\s*$")
AFTER_NEGATION = re.compile(
    r"^\s*(?:향|느낌|노트)?(?:은|는|이|가|을|를)?\s*"
    r"(?:빼|제외|없이|없는|싫|금지|말고|제거|낮추|낮춰|줄이|줄여|덜|약하게|하지\s*않|아니|-free)"
)
STRONG_MODIFIERS = ("매우", "강조", "중심", "위주", "dominant", "mainly", "very")
SOFT_MODIFIERS = ("살짝", "조금", "은은", "미세", "subtle", "slight", "hint of")
REVISION_INCREASE = re.compile(
    r"(?:더|높이|높여|올리|올려|늘리|늘려|강화|강하게|강조|increase|more|boost|stronger)"
)
REVISION_DECREASE = re.compile(
    r"(?:덜|낮추|낮춰|줄이|줄여|약하게|reduce|less|lower|soften)"
)
REVISION_REMOVE = re.compile(r"(?:빼|제거|없애|없이|없는|금지|remove|without|exclude)")


def apply_relative_revision_profile(
    base_profile: dict[str, float], instruction: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply bounded relative scent-dimension edits to an existing profile.

    The returned adjustment map contains the multiplier for every dimension
    explicitly changed by the instruction. Unmentioned dimensions retain their
    relative mass, which makes a conversational edit materially different from
    regenerating an unrelated formula from the original brief.
    """

    if not isinstance(instruction, str):
        raise ValueError("revision instruction must be text")
    lowered = instruction.casefold()
    adjustments: dict[str, float] = {}
    mentions: list[tuple[int, int, str]] = []
    for dimension, aliases in KEYWORDS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            for start, end in find_text_spans(lowered, alias):
                mentions.append((start, end, dimension))
    mentions.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    mentions = [
        mention
        for index, mention in enumerate(mentions)
        if not any(
            other[0] <= mention[0] and mention[1] <= other[1]
            for other in mentions[:index]
        )
    ]
    for index, (start, end, dimension) in enumerate(mentions):
        previous_end = mentions[index - 1][1] if index else 0
        next_start = (
            mentions[index + 1][0] if index + 1 < len(mentions) else len(lowered)
        )
        before = lowered[max(previous_end, start - 18) : start]
        after = lowered[end : min(next_start, end + 28)]
        window = before + lowered[start:end] + after
        if re.search(r"(?:유지|그대로|keep|same|unchanged)", after) or re.search(
            r"(?:keep|same|unchanged|그대로)\s*$", before
        ):
            continue
        softer = any(token in window for token in SOFT_MODIFIERS)
        prefix_remove = REVISION_REMOVE.search(before)
        suffix_remove = REVISION_REMOVE.search(after)
        prefix_decrease = REVISION_DECREASE.search(before)
        prefix_increase = REVISION_INCREASE.search(before)
        suffix_decrease = REVISION_DECREASE.search(after)
        suffix_increase = REVISION_INCREASE.search(after)
        if prefix_remove or suffix_remove:
            adjustments[dimension] = 0.0
        elif prefix_decrease or suffix_decrease:
            adjustments[dimension] = 0.70 if softer else 0.50
        elif prefix_increase or suffix_increase:
            adjustments[dimension] = 1.35 if softer else 1.60

    revised = normalize_profile(base_profile)
    if not adjustments:
        return revised, adjustments
    positive_values = [value for value in revised.values() if value > 0]
    introduction_floor = (
        max(0.02, sum(positive_values) / max(len(positive_values), 1) * 0.25)
        if positive_values
        else 0.02
    )
    for dimension, multiplier in adjustments.items():
        current = revised.get(dimension, 0.0)
        if multiplier > 1.0 and current <= 0:
            current = introduction_floor
        revised[dimension] = current * multiplier
    return normalize_profile(revised), adjustments


def _is_negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 14) : start]
    after = text[end : min(len(text), end + 18)]
    coordinated_before = text[max(0, start - 40) : start]
    coordinated = re.search(
        r"\b(?:without|no)\s+[a-z\s]+(?:or|and)\s*$", coordinated_before
    )
    return bool(
        BEFORE_NEGATION.search(before) or AFTER_NEGATION.search(after) or coordinated
    )


def _match_weight(text: str, start: int, end: int) -> float:
    window = text[max(0, start - 10) : min(len(text), end + 10)]
    if any(token in window for token in SOFT_MODIFIERS):
        return 0.65
    if any(token in window for token in STRONG_MODIFIERS):
        return 1.5
    return 1.0


def _apply_numeric_constraints(text: str, constraints: RecipeConstraints) -> None:
    price_patterns = (
        r"(?:원료|재료)?\s*(?:kg|킬로)(?:당)?\s*(?:가격)?\s*(?:은|는)?\s*"
        r"(?:최대|이하|under|max)?\s*\$?\s*(\d+(?:\.\d+)?)",
        r"(?:under|below|max(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/\s*kg|per\s*kg)",
    )
    for pattern in price_patterns:
        match = re.search(pattern, text)
        if match:
            constraints.max_ingredient_price_per_kg = min(
                constraints.max_ingredient_price_per_kg, float(match.group(1))
            )
            break
    batch = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|그램)\s*(?:배치|batch)", text)
    if batch:
        constraints.finished_batch_mass_g = float(batch.group(1))
    volume = re.search(r"(\d+(?:\.\d+)?)\s*ml", text)
    if volume:
        constraints.finished_volume_ml = float(volume.group(1))
    concentration = re.search(r"(?:향료\s*)?농도\s*(\d+(?:\.\d+)?)\s*%", text)
    if concentration:
        constraints.product_concentration_percent = float(concentration.group(1))


def _perceptual_intent(
    text: str,
    intensity_label: str,
) -> tuple[float, float, dict[str, float], dict[str, float], dict[str, float]]:
    """Extract non-odor-family controls without renormalizing them away."""

    absolute_intensity = {"low": 0.30, "medium": 0.55, "high": 0.82}[intensity_label]
    numeric_intensity = re.search(
        r"(?:향\s*)?(?:강도|intensity)\s*(\d+(?:\.\d+)?)\s*%?", text
    )
    if numeric_intensity:
        value = float(numeric_intensity.group(1))
        absolute_intensity = max(0.0, min(1.0, value / 100.0))

    diffusion = 0.50
    if any(
        token in text
        for token in (
            "확산력 강",
            "멀리 퍼",
            "존재감",
            "high projection",
            "projecting",
            "strong sillage",
        )
    ):
        diffusion = 0.82
    elif any(
        token in text
        for token in (
            "살결 가까이",
            "피부 가까이",
            "잔잔하게 퍼",
            "skin scent",
            "low projection",
            "intimate",
        )
    ):
        diffusion = 0.25

    texture_terms = {
        "transparent": ("투명", "맑은", "airy", "transparent"),
        "dense": ("밀도", "농밀", "두꺼운", "dense", "thick"),
        "dry": ("드라이", "건조", "dry"),
        "creamy": ("크리미", "부드러운 크림", "creamy"),
        "soft": ("포근", "부드럽", "보송", "soft", "velvety"),
    }
    texture = {
        name: 1.0
        for name, aliases in texture_terms.items()
        if any(alias in text for alias in aliases)
    }
    trigeminal_terms = {
        "cooling": ("차가운", "쿨링", "얼음", "icy", "cooling"),
        "warming": ("따뜻", "온기", "warming"),
        "tingling": ("톡 쏘", "알싸", "따끔", "tingling", "pungent"),
    }
    trigeminal = {
        name: 1.0
        for name, aliases in trigeminal_terms.items()
        if any(alias in text for alias in aliases)
    }

    temporal = {"opening": 0.25, "heart": 0.40, "drydown": 0.35}
    if any(token in text for token in ("첫 향", "오프닝", "첫인상", "opening")):
        temporal = {"opening": 0.55, "heart": 0.30, "drydown": 0.15}
    elif any(
        token in text
        for token in ("잔향", "드라이다운", "오래 지속", "drydown", "long lasting")
    ):
        temporal = {"opening": 0.15, "heart": 0.30, "drydown": 0.55}
    elif any(token in text for token in ("미들 노트", "중반", "heart note")):
        temporal = {"opening": 0.20, "heart": 0.60, "drydown": 0.20}
    return absolute_intensity, diffusion, texture, trigeminal, temporal


class NaturalLanguageBriefParser:
    def __init__(
        self,
        catalog: IngredientCatalog,
        semantic_ontology: ScentSemanticOntology | None = None,
    ):
        self.catalog = catalog
        self.semantic_ontology = semantic_ontology or ScentSemanticOntology()

    def parse(
        self,
        text: str,
        constraints: RecipeConstraints | None = None,
    ) -> ScentBrief:
        if not isinstance(text, str) or len(text.strip()) < 3:
            raise BriefParseError("향 브리프를 세 글자 이상 입력하세요.")

        # Natural-language numeric overrides belong to this request.  Copy the
        # caller's dataclass so reusing one RecipeConstraints instance across
        # requests cannot leak a parsed value into the next formula.
        if constraints is None:
            constraints = RecipeConstraints()
        else:
            if not isinstance(constraints.explicit_bans, (set, frozenset)) or not all(
                isinstance(item, str) for item in constraints.explicit_bans
            ):
                raise BriefParseError("explicit_bans must be a set of ingredient names")
            constraints = replace(
                constraints,
                explicit_bans=set(constraints.explicit_bans),
            )
        lowered = text.casefold()
        _apply_numeric_constraints(lowered, constraints)
        scores: dict[str, float] = {key: 0.0 for key in KEYWORDS}
        desired: set[str] = set()
        avoided: set[str] = set()
        lexical_confidence = 0.0

        for dimension, aliases in KEYWORDS.items():
            positive_weight = 0.0
            negative_match = False
            occupied: list[tuple[int, int]] = []
            for alias in sorted(aliases, key=len, reverse=True):
                for start, end in find_text_spans(lowered, alias):
                    span = (start, end)
                    if any(
                        span[0] < end and start < span[1] for start, end in occupied
                    ):
                        continue
                    occupied.append(span)
                    if _is_negated(lowered, start, end):
                        negative_match = True
                    else:
                        positive_weight = max(
                            positive_weight,
                            _match_weight(lowered, start, end),
                        )
            if positive_weight:
                scores[dimension] += positive_weight
                desired.add(dimension)
                lexical_confidence = max(lexical_confidence, 0.96)
            if negative_match:
                avoided.add(dimension)

        descriptor_lexicon = load_builtin_odor_descriptor_lexicon()
        recognized_descriptors: set[str] = set()
        avoided_descriptors: set[str] = set()
        unsupported_descriptors: set[str] = set()
        for projection in descriptor_lexicon.descriptors:
            positive_weight = 0.0
            negative_match = False
            occupied: list[tuple[int, int]] = []
            for alias in sorted(projection.aliases, key=len, reverse=True):
                for start, end in find_text_spans(lowered, alias):
                    span = (start, end)
                    if any(
                        span[0] < occupied_end and occupied_start < span[1]
                        for occupied_start, occupied_end in occupied
                    ):
                        continue
                    occupied.append(span)
                    recognized_descriptors.add(projection.descriptor)
                    if _is_negated(lowered, start, end):
                        negative_match = True
                    else:
                        positive_weight = max(
                            positive_weight, _match_weight(lowered, start, end)
                        )
            if negative_match:
                avoided_descriptors.add(projection.descriptor)
                continue
            if not positive_weight:
                continue
            lexical_confidence = max(
                lexical_confidence, projection.projection_confidence
            )
            if not projection.formula_supported:
                unsupported_descriptors.add(projection.descriptor)
                continue
            for dimension, value in projection.profile.items():
                scores[dimension] += value * positive_weight
                if value >= 0.20:
                    desired.add(dimension)

        if unsupported_descriptors:
            raise UnsupportedOdorDescriptorError(sorted(unsupported_descriptors))

        requested_ingredients: list[str] = []
        excluded_ingredients: list[str] = []
        mention_groups: dict[str, list] = {}
        for mention in self.catalog.mentioned_ingredient_spans(text):
            mention_groups.setdefault(mention.ingredient.ingredient_id, []).append(
                mention
            )
        for mentions in mention_groups.values():
            ingredient = mentions[0].ingredient
            negated = False
            for mention in mentions:
                after = lowered[mention.end : mention.end + 10]
                after_without_particle = after.lstrip(" ,./은는이가을를")
                if _is_negated(lowered, mention.start, mention.end) or (
                    after_without_particle.startswith(
                        ("빼", "제외", "없이", "없는", "금지")
                    )
                ):
                    negated = True
                    break
            if negated:
                excluded_ingredients.append(ingredient.name)
                constraints.explicit_bans.add(normalize_name(ingredient.name))
                continue
            requested_ingredients.append(ingredient.name)
            lexical_confidence = max(lexical_confidence, 0.98)
            for dimension, value in ingredient.profile.items():
                scores[dimension] += float(value) * 1.25
                if value >= 0.4:
                    desired.add(dimension)

        # Add ontology-embedding evidence after exact negation and descriptor
        # handling. With exact lexical evidence present, a stricter threshold
        # prevents hash collisions from adding unrelated dimensions.
        semantic = self.semantic_ontology.infer(text)
        lexical_dimensions = {name for name, value in scores.items() if value > 0.0}
        semantic_threshold = 0.30 if lexical_dimensions else 0.16
        if constraints.enable_semantic_ontology:
            for dimension, value in semantic.scores.items():
                if value >= semantic_threshold and dimension not in lexical_dimensions:
                    scores[dimension] += min(0.85, value * 0.85)
                    desired.add(dimension)

        for dimension in avoided:
            scores[dimension] = 0.0

        # Specific flowers also imply a general floral body in sensory briefs.
        if scores["rose"] > 0:
            scores["floral"] += scores["rose"] * 0.7
            desired.add("floral")
        if scores["white_floral"] > 0:
            scores["floral"] += scores["white_floral"] * 0.7
            desired.add("floral")

        if any(
            phrase in lowered
            for phrase in (
                "달지 않",
                "안 달",
                "단 향은 없이",
                "단 향 없이",
                "단향 없이",
                "단맛은 없이",
                "단맛 없이",
                "단맛을 빼고",
                "not sweet",
                "without sweetness",
            )
        ):
            avoided.add("gourmand")
            scores["gourmand"] = 0.0
        if "무겁지 않" in lowered or "가볍" in lowered:
            scores["fresh"] += 0.8
            scores["clean"] += 0.4
            scores["amber"] *= 0.2
            scores["smoky"] *= 0.2
        if (
            "고급" in lowered
            or "우아" in lowered
            or "luxury" in lowered
            or "elegant" in lowered
        ):
            scores["woody"] += 0.4
            scores["amber"] += 0.35
            scores["floral"] += 0.25

        if sum(scores.values()) <= 0:
            raise BriefParseError(
                "향 특성을 찾지 못했습니다. 예: '깨끗하고 시원한 시트러스 우디 향'"
            )

        intensity = "medium"
        if any(
            token in lowered for token in ("은은", "가볍", "약하게", "soft", "subtle")
        ):
            intensity = "low"
        elif any(
            token in lowered
            for token in ("진한", "강한", "강렬", "묵직", "intense", "strong")
        ):
            intensity = "high"

        if (
            intensity == "low"
            or scores["fresh"] + scores["clean"]
            > scores["amber"] + scores["woody"] + 1.0
        ):
            pyramid = {"top": 35.0, "heart": 40.0, "base": 25.0}
        elif (
            intensity == "high"
            or scores["amber"] + scores["woody"] + scores["smoky"] > 2.0
        ):
            pyramid = {"top": 15.0, "heart": 35.0, "base": 50.0}
        else:
            pyramid = {"top": 25.0, "heart": 40.0, "base": 35.0}

        (
            absolute_intensity,
            diffusion,
            texture_profile,
            trigeminal_profile,
            temporal_emphasis,
        ) = _perceptual_intent(lowered, intensity)

        if any(
            token in lowered
            for token in ("저렴", "가성비", "합리적", "affordable", "budget")
        ):
            constraints.max_ingredient_price_per_kg = min(
                constraints.max_ingredient_price_per_kg, 180.0
            )
            constraints.max_formula_cost_per_kg = min(
                constraints.max_formula_cost_per_kg, 120.0
            )

        if any(
            token in lowered
            for token in ("상업 생산", "판매용 제조", "commercial production")
        ):
            constraints.validation_level = "commercial"
        elif any(
            token in lowered
            for token in ("공급사 검증", "관능 검증", "supplier qualified")
        ):
            constraints.validation_level = "qualified"

        region_tokens = {
            "eu": ("eu 판매", "유럽 판매", "유럽연합"),
            "kr": ("한국 판매", "국내 판매"),
            "us": ("미국 판매", "us market"),
            "jp": ("일본 판매", "jp market"),
        }
        for region, tokens in region_tokens.items():
            if any(token in lowered for token in tokens):
                constraints.target_region = region.upper()
                break

        semantic_enabled = constraints.enable_semantic_ontology
        semantic_backend = "disabled_for_ablation"
        semantic_confidence = 0.0
        if semantic_enabled:
            semantic_backend = semantic.backend
            semantic_confidence = round(max(semantic.confidence, lexical_confidence), 6)
            if lexical_confidence > 0:
                semantic_backend = f"hybrid_lexical+{semantic.backend}"

        return ScentBrief(
            original_text=text.strip(),
            target_profile=normalize_profile(scores),
            desired_dimensions=sorted(desired - avoided),
            avoided_dimensions=sorted(avoided),
            requested_ingredients=requested_ingredients,
            excluded_ingredients=excluded_ingredients,
            intensity=intensity,
            pyramid_ratios=pyramid,
            constraints=constraints,
            semantic_backend=semantic_backend,
            semantic_confidence=semantic_confidence,
            ontology_version=semantic.ontology_version,
            ontology_concepts=(list(semantic.concepts) if semantic_enabled else []),
            recognized_descriptors=sorted(recognized_descriptors),
            avoided_descriptors=sorted(avoided_descriptors),
            descriptor_projection_version=descriptor_lexicon.version,
            descriptor_projection_claim_boundary=descriptor_lexicon.claim_boundary,
            absolute_intensity_target=absolute_intensity,
            diffusion_target=diffusion,
            texture_profile=texture_profile,
            trigeminal_profile=trigeminal_profile,
            temporal_emphasis=temporal_emphasis,
        )
