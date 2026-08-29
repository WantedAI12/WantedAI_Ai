"""
마스터 조향사 지식 베이스
세계 최고 조향사들의 전문 지식과 원칙을 구현
"""
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class FragranceStructure(Enum):
    """향수 구조 타입"""
    CLASSIC_PYRAMID = "classic_pyramid"  # 전통적 3단계 구조
    LINEAR = "linear"  # 선형 구조
    RADIAL = "radial"  # 방사형 구조
    MULTI_FACETED = "multi_faceted"  # 다면적 구조


class OlfactoryFamily(Enum):
    """후각 패밀리 (Master Perfumer Classification)"""
    HESPERIDIC = "hesperidic"  # 시트러스
    FLORAL = "floral"
    ORIENTAL = "oriental"
    WOODY = "woody"
    FOUGERE = "fougere"  # 푸제르
    CHYPRE = "chypre"  # 시프레
    AQUATIC = "aquatic"
    GOURMAND = "gourmand"


@dataclass
class HarmonyRule:
    """향료 조화 법칙"""
    rule_name: str
    description: str
    ingredient_combinations: List[Tuple[str, str]]
    harmony_strength: float  # 0-1, 조화 강도
    chemistry_basis: str
    master_perfumer_notes: str


@dataclass
class AccordFormula:
    """아코드 공식 (조향사 시그니처 조합)"""
    name: str
    creator: str  # 창조한 조향사
    ingredients: Dict[str, float]  # 향료명: 비율(%)
    total_percentage: float
    olfactory_effect: str
    usage_guidelines: str
    signature_characteristics: List[str]


@dataclass
class PerfumerSignature:
    """조향사 시그니처 스타일"""
    perfumer_name: str
    style_characteristics: List[str]
    favorite_materials: List[str]
    signature_accords: List[str]
    compositional_preferences: Dict[str, Any]
    innovation_techniques: List[str]
    masterpieces: List[str]


class MasterPerfumerKnowledge:
    """마스터 조향사 전문 지식 시스템"""

    def __init__(self):
        self.harmony_rules = self._initialize_harmony_rules()
        self.signature_accords = self._initialize_signature_accords()
        self.perfumer_profiles = self._initialize_perfumer_profiles()
        self.golden_ratios = self._initialize_golden_ratios()
        self.complexity_guidelines = self._initialize_complexity_guidelines()

    def _initialize_harmony_rules(self) -> List[HarmonyRule]:
        """조향사들의 황금 조화 법칙들"""
        return [
            HarmonyRule(
                rule_name="Citrus-Woody Harmony",
                description="시트러스와 우디의 클래식 조화 - 신선함과 깊이의 완벽한 균형",
                ingredient_combinations=[
                    ("베르가못", "세다우드"), ("레몬", "베티버"),
                    ("그레이프프루트", "샌달우드"), ("유자", "편백나무")
                ],
                harmony_strength=0.95,
                chemistry_basis="Terpenes와 Sesquiterpenes의 화학적 친화성",
                master_perfumer_notes="Edmond Roudnitska가 개척한 클래식 조합. 탑노트의 휘발성과 베이스의 안정성이 완벽한 대비를 이룸"
            ),
            HarmonyRule(
                rule_name="Rose-Oud Synergy",
                description="장미와 우드의 동서양 융합 - 우아함과 신비로움의 조화",
                ingredient_combinations=[
                    ("불가리안 로즈", "아가우드"), ("다마스크 로즈", "인도 우드"),
                    ("로즈 압솔루트", "캄보디아 우드")
                ],
                harmony_strength=0.92,
                chemistry_basis="Rose의 monoterpenes와 Oud의 sesquiterpenes 상호작용",
                master_perfumer_notes="Francis Kurkdjian이 마스터한 조합. 동서양 향료 문화의 완벽한 만남"
            ),
            HarmonyRule(
                rule_name="Vanilla-Spice Architecture",
                description="바닐라와 스파이스의 구조적 조화 - 달콤함과 따뜻함의 건축학",
                ingredient_combinations=[
                    ("바닐라", "계피"), ("바닐라", "카르다몸"),
                    ("톤카빈", "사프란"), ("벤조인", "정향")
                ],
                harmony_strength=0.89,
                chemistry_basis="Vanillin과 Eugenol류의 공명 효과",
                master_perfumer_notes="Thierry Wasser의 구르망 철학. 향신료가 바닐라의 단조로움을 깨뜨리며 복잡성을 부여"
            ),
            HarmonyRule(
                rule_name="Aldehydic-Floral Revolution",
                description="알데하이드와 플로럴의 혁명적 조합 - 빛나는 꽃향기의 탄생",
                ingredient_combinations=[
                    ("C-12 알데하이드", "일랑일랑"), ("C-10 알데하이드", "로즈"),
                    ("지방족 알데하이드", "자스민")
                ],
                harmony_strength=0.94,
                chemistry_basis="Aldehydes의 스파클링 효과가 florals의 depth를 부각",
                master_perfumer_notes="Ernest Beaux가 샤넬 No.5로 구현한 혁명. 알데하이드가 꽃향기에 광채와 입체감을 부여"
            ),
            HarmonyRule(
                rule_name="Marine-Mineral Fusion",
                description="마린과 미네랄의 현대적 융합 - 바다와 대지의 만남",
                ingredient_combinations=[
                    ("아쿠아틱 노트", "젖은 돌"), ("바다 소금", "화산재"),
                    ("해초", "페트리코르")
                ],
                harmony_strength=0.87,
                chemistry_basis="Iodine compounds와 mineral salts의 synesthetic effect",
                master_perfumer_notes="Alberto Morillas가 개척한 아쿠아틱 미니멀리즘. 자연의 원초적 에너지를 포착"
            ),
            HarmonyRule(
                rule_name="Leather-Iris Sophistication",
                description="가죽과 아이리스의 세련된 조화 - 파우더리 럭셔리의 정점",
                ingredient_combinations=[
                    ("스웨이드", "아이리스 팔리다"), ("가죽", "오리스 루트"),
                    ("캐스토리움", "아이리스 버터")
                ],
                harmony_strength=0.91,
                chemistry_basis="Irones와 leather molecules의 파우더리 시너지",
                master_perfumer_notes="Germaine Cellier의 시그니처. 아이리스가 가죽의 거친 애니말릭함을 세련되게 감싸안음"
            ),
            HarmonyRule(
                rule_name="Green-Ozonic Transparency",
                description="그린과 오존의 투명한 조화 - 순수함의 극치",
                ingredient_combinations=[
                    ("그린 리브스", "오존"), ("갈바넘", "아쿠아틱"),
                    ("페티트그레인", "마린 노트")
                ],
                harmony_strength=0.88,
                chemistry_basis="Green aldehydes와 ozonic molecules의 transparency effect",
                master_perfumer_notes="Jean-Claude Ellena의 미니멀리즘. 향의 투명도와 순수성을 극한까지 추구"
            )
        ]

    def _initialize_signature_accords(self) -> List[AccordFormula]:
        """마스터 조향사들의 시그니처 아코드"""
        return [
            AccordFormula(
                name="Chanel Aldehydic Bouquet",
                creator="Ernest Beaux",
                ingredients={
                    "C-10 알데하이드": 3.5,
                    "C-12 알데하이드": 2.8,
                    "일랑일랑": 12.0,
                    "로즈 드 메이": 8.5,
                    "자스민 삼박": 6.2,
                    "네롤리": 4.1
                },
                total_percentage=37.1,
                olfactory_effect="스파클링하고 럭셔리한 플로럴 부케, 비누향과는 차원이 다른 고급스러운 알데하이드 효과",
                usage_guidelines="탑-미들 노트 전환부에서 사용, 3-5% 농도로 전체 조성에 적용",
                signature_characteristics=["알데하이드 스파클", "플로럴 컴플렉시티", "럭셔리 파우더리"]
            ),
            AccordFormula(
                name="Amouage Gold Oud Complex",
                creator="Guy Robert",
                ingredients={
                    "아가우드 캄보디아": 4.2,
                    "로즈 다마스크": 8.8,
                    "프랑킨센스": 6.5,
                    "미르": 3.7,
                    "사프란": 1.2,
                    "앰버그리스": 2.1
                },
                total_percentage=26.5,
                olfactory_effect="동서양의 만남, 스모키하면서도 플로럴한 복합적 오리엔탈",
                usage_guidelines="베이스 노트 중심, 전체의 25-30%로 사용하여 깊이와 복잡성 부여",
                signature_characteristics=["오리엔탈 럭셔리", "스모키 플로럴", "오피언트 리치니스"]
            ),
            AccordFormula(
                name="Terre d'Hermes Mineral Woody",
                creator="Jean-Claude Ellena",
                ingredients={
                    "오렌지": 15.2,
                    "그레이프프루트": 8.3,
                    "플린트 (부싯돌)": 3.8,
                    "베티버": 12.7,
                    "세다우드": 6.9,
                    "벤조인": 4.1
                },
                total_percentage=51.0,
                olfactory_effect="미네랄한 대지와 시트러스의 조화, 건조하고 세련된 남성성",
                usage_guidelines="전체적인 베이스로 사용, 미니멀하면서도 깊이 있는 구조",
                signature_characteristics=["미네랄 드라이니스", "시트러스 프레시니스", "정제된 마스큘리니티"]
            ),
            AccordFormula(
                name="Angel Gourmand Revolution",
                creator="Olivier Cresp & Yves de Chirin",
                ingredients={
                    "에틸 말톨": 8.5,
                    "바닐라": 12.0,
                    "카라멜": 6.8,
                    "패촐리": 9.2,
                    "초콜릿": 4.3,
                    "프룻 노트": 7.1
                },
                total_percentage=47.9,
                olfactory_effect="달콤함의 혁명, 구르망과 오리엔탈의 경계를 무너뜨린 혁신",
                usage_guidelines="미들-베이스 노트 중심, 달콤함과 깊이의 균형 필수",
                signature_characteristics=["구르망 혁신", "스위트 컴플렉시티", "패촐리 베이스"]
            ),
            AccordFormula(
                name="CK One Aquatic Minimalism",
                creator="Alberto Morillas & Harry Fremont",
                ingredients={
                    "레몬": 18.5,
                    "베르가못": 12.3,
                    "아쿠아틱 노트": 15.7,
                    "제라늄": 8.2,
                    "라벤더": 6.8,
                    "화이트 머스크": 11.4
                },
                total_percentage=72.9,
                olfactory_effect="성별을 초월한 클린함, 90년대 미니멀리즘의 정점",
                usage_guidelines="전체 구조의 70% 이상, 클린하고 투명한 효과",
                signature_characteristics=["유니섹스 클린", "아쿠아틱 미니멀", "투명한 현대성"]
            ),
            AccordFormula(
                name="Black Orchid Dark Floral",
                creator="Tom Ford & Givaudan Team",
                ingredients={
                    "블랙 트러플": 2.1,
                    "일랑일랑": 8.7,
                    "블랙커런트": 5.4,
                    "패촐리": 15.2,
                    "다크 초콜릿": 3.8,
                    "인센스": 6.3
                },
                total_percentage=41.5,
                olfactory_effect="다크하고 인독시케이팅한 모던 고딕, 플로럴의 다크사이드",
                usage_guidelines="미들-베이스 노트 중심, 강렬하고 세련된 이브닝 향수",
                signature_characteristics=["다크 플로럴", "인독시케이팅", "모던 고딕"]
            )
        ]

    def _initialize_perfumer_profiles(self) -> List[PerfumerSignature]:
        """세계적 마스터 조향사들의 프로필"""
        return [
            PerfumerSignature(
                perfumer_name="Jean-Claude Ellena",
                style_characteristics=[
                    "미니멀리즘의 대가", "투명도와 순수성 추구", "적은 것으로 많은 것을 표현",
                    "시적 감성", "지중해적 감수성", "라이트 터치"
                ],
                favorite_materials=[
                    "이소 이 슈퍼", "갈바넘", "아이리스", "우드", "시트러스", "아쿠아틱 노트"
                ],
                signature_accords=[
                    "Terre d'Hermes 미네랄 우디", "Un Jardin 시리즈 그린 아코드",
                    "Kelly Caleche 레더 아코드"
                ],
                compositional_preferences={
                    "structure": "미니멀하고 투명한 구조",
                    "concentration": "적은 원료로 최대 효과",
                    "philosophy": "향수는 감정을 전달하는 언어",
                    "signature_ratio": "30% 메인 아코드 + 70% 서포팅"
                },
                innovation_techniques=[
                    "Overdose 기법 (하나의 원료를 과감히 많이 사용)",
                    "Transparent layering (투명한 레이어링)",
                    "Negative space 활용 (향의 여백미)"
                ],
                masterpieces=[
                    "Terre d'Hermes", "Declaration", "Un Jardin Sur Le Toit",
                    "Kelly Caleche", "Voyage d'Hermes"
                ]
            ),
            PerfumerSignature(
                perfumer_name="Francis Kurkdjian",
                style_characteristics=[
                    "현대적 럭셔리", "정밀한 기술력", "동서양 융합",
                    "크리스탈처럼 깨끗한 구조", "감성적 스토리텔링"
                ],
                favorite_materials=[
                    "장미", "아가우드", "사프란", "앰버그리스", "자스민", "캐시미어우드"
                ],
                signature_accords=[
                    "로즈-우드 퓨전", "오리엔탈 플로럴", "럭셔리 아쿠아틱"
                ],
                compositional_preferences={
                    "structure": "완벽한 균형과 비례",
                    "concentration": "각 원료의 최적 농도 추구",
                    "philosophy": "향수는 보이지 않는 액세서리",
                    "signature_ratio": "40% 시그니처 아코드 + 60% 하모니"
                },
                innovation_techniques=[
                    "Molecular precision (분자 단위 정밀도)",
                    "Cultural fusion (문화적 융합)",
                    "Sillage engineering (확산 공학)"
                ],
                masterpieces=[
                    "Maison Francis Kurkdjian Baccarat Rouge 540",
                    "Aqua Celestia", "Oud Series", "Le Labo Another 13"
                ]
            ),
            PerfumerSignature(
                perfumer_name="Thierry Wasser",
                style_characteristics=[
                    "Guerlain의 DNA 계승", "구르망의 혁신", "감정적 깊이",
                    "프랑스 클래식의 현대적 해석", "향료의 스토리텔러"
                ],
                favorite_materials=[
                    "바닐라", "톤카빈", "벤조인", "로즈", "오리스", "스파이스류"
                ],
                signature_accords=[
                    "Guerlinade (겔랑의 시그니처)", "모던 구르망", "스파이시 오리엔탈"
                ],
                compositional_preferences={
                    "structure": "클래식한 피라미드 구조",
                    "concentration": "풍부하고 깊이 있는 농도",
                    "philosophy": "향수는 감정의 기억",
                    "signature_ratio": "50% 트래디션 + 50% 이노베이션"
                },
                innovation_techniques=[
                    "Heritage modernization (전통의 현대화)",
                    "Gourmand sophistication (구르망의 고급화)",
                    "Emotional architecture (감정적 구조화)"
                ],
                masterpieces=[
                    "La Petite Robe Noire", "Mon Guerlain",
                    "L'Homme Ideal", "Aqua Allegoria 시리즈"
                ]
            ),
            PerfumerSignature(
                perfumer_name="Alberto Morillas",
                style_characteristics=[
                    "다재다능함", "상업적 성공과 예술성 조화", "혁신적 아쿠아틱",
                    "글로벌 어필", "기술적 완성도"
                ],
                favorite_materials=[
                    "아쿠아틱 노트", "플로럴", "프루티", "화이트 머스크", "우드"
                ],
                signature_accords=[
                    "CK One 아쿠아틱", "플로럴 프루티", "모던 우드"
                ],
                compositional_preferences={
                    "structure": "다양한 구조 마스터",
                    "concentration": "시장 지향적 최적화",
                    "philosophy": "향수는 라이프스타일",
                    "signature_ratio": "유연한 비율 조정"
                },
                innovation_techniques=[
                    "Market-art balance (시장성과 예술성 균형)",
                    "Aquatic innovation (아쿠아틱 혁신)",
                    "Global accessibility (글로벌 접근성)"
                ],
                masterpieces=[
                    "CK One", "Bulgari Aqua", "Kenzo Flower",
                    "Polo Blue", "Gucci Rush"
                ]
            ),
            PerfumerSignature(
                perfumer_name="Dominique Ropion",
                style_characteristics=[
                    "파워풀하고 센슈얼", "볼드한 콘트라스트", "극적 구조",
                    "현대적 오리엔탈", "강렬한 캐릭터"
                ],
                favorite_materials=[
                    "로즈", "패촐리", "오피언트 스파이스", "앰버", "인센스"
                ],
                signature_accords=[
                    "파워풀 로즈", "오리엔탈 스파이시", "인센스 우드"
                ],
                compositional_preferences={
                    "structure": "드라마틱한 구조",
                    "concentration": "고농도 임팩트",
                    "philosophy": "향수는 감정의 폭발",
                    "signature_ratio": "60% 임팩트 + 40% 하모니"
                },
                innovation_techniques=[
                    "Contrast maximization (대비 극대화)",
                    "Sensual overdose (센슈얼 오버도즈)",
                    "Dramatic storytelling (드라마틱 스토리텔링)"
                ],
                masterpieces=[
                    "Portrait of a Lady", "La Fille de Berlin",
                    "Carnal Flower", "Une Rose"
                ]
            )
        ]

    def _initialize_golden_ratios(self) -> Dict[str, Dict[str, float]]:
        """조향의 황금 비율들"""
        return {
            "classic_pyramid": {
                "top_notes": 0.25,      # 25%
                "middle_notes": 0.45,   # 45%
                "base_notes": 0.30      # 30%
            },
            "modern_structure": {
                "opening": 0.20,
                "heart": 0.50,
                "drydown": 0.30
            },
            "concentration_levels": {
                "parfum": 0.25,         # 25% 농도
                "edp": 0.18,            # 18% 농도
                "edt": 0.12,            # 12% 농도
                "edc": 0.08             # 8% 농도
            },
            "family_ratios": {
                "floral": {"floral": 0.60, "supporting": 0.40},
                "oriental": {"spices_amber": 0.50, "supporting": 0.50},
                "woody": {"woods": 0.55, "supporting": 0.45},
                "citrus": {"citrus": 0.70, "supporting": 0.30}
            },
            "complexity_levels": {
                "simple": 5,           # 5개 이하 원료
                "moderate": 12,        # 12개 이하 원료
                "complex": 25,         # 25개 이하 원료
                "haute_couture": 50    # 50개까지 원료
            }
        }

    def _initialize_complexity_guidelines(self) -> Dict[str, Any]:
        """복잡성 가이드라인"""
        return {
            "beginner_complexity": {
                "max_ingredients": 8,
                "structure": "simple_pyramid",
                "focus": "learn basic harmonies",
                "avoid": ["difficult_materials", "complex_accords"]
            },
            "intermediate_complexity": {
                "max_ingredients": 15,
                "structure": "enhanced_pyramid",
                "focus": "signature accords",
                "explore": ["faceting", "transitions"]
            },
            "advanced_complexity": {
                "max_ingredients": 25,
                "structure": "multi_faceted",
                "focus": "innovation and artistry",
                "master": ["overdose", "contrasts", "storytelling"]
            },
            "master_complexity": {
                "max_ingredients": 50,
                "structure": "architectural",
                "focus": "olfactory architecture",
                "achieve": ["emotional_journey", "memorability", "uniqueness"]
            },
            "evaluation_criteria": {
                "harmony": "모든 원료가 조화롭게 어우러지는가",
                "balance": "농도와 비율이 적절한가",
                "evolution": "시간에 따른 변화가 아름다운가",
                "memorability": "기억에 남는 독특함이 있는가",
                "wearability": "착용하기 좋은가",
                "commercial_viability": "상업적 가능성이 있는가"
            }
        }

    def get_harmony_score(self, ingredient1: str, ingredient2: str) -> float:
        """두 향료 간의 조화도 점수 계산"""
        for rule in self.harmony_rules:
            for combo in rule.ingredient_combinations:
                if (ingredient1 in combo and ingredient2 in combo) or \
                   (ingredient2 in combo and ingredient1 in combo):
                    return rule.harmony_strength

        # 기본 패밀리 호환성 체크
        return self._calculate_family_harmony(ingredient1, ingredient2)

    def _calculate_family_harmony(self, ingredient1: str, ingredient2: str) -> float:
        """패밀리 기반 호환성 계산"""
        # 실제 구현에서는 향료 데이터베이스와 연동
        family_compatibility = {
            ("citrus", "woody"): 0.85,
            ("floral", "oriental"): 0.80,
            ("aromatic", "woody"): 0.82,
            ("fruity", "gourmand"): 0.88,
            ("marine", "mineral"): 0.90
        }
        # 기본값
        return 0.60

    def recommend_perfumer_style(self, preferences: Dict[str, Any]) -> PerfumerSignature:
        """선호도에 따른 조향사 스타일 추천"""
        style_match_scores = {}

        for perfumer in self.perfumer_profiles:
            score = 0

            # 스타일 특성 매칭
            if preferences.get("style") == "minimalist":
                if "미니멀리즘" in str(perfumer.style_characteristics):
                    score += 30

            if preferences.get("style") == "luxurious":
                if "럭셔리" in str(perfumer.style_characteristics):
                    score += 30

            if preferences.get("innovation") == "high":
                score += len(perfumer.innovation_techniques) * 5

            style_match_scores[perfumer.perfumer_name] = score

        best_match = max(style_match_scores, key=style_match_scores.get)
        return next(p for p in self.perfumer_profiles if p.perfumer_name == best_match)

    def get_signature_accord(self, style: str) -> Optional[AccordFormula]:
        """스타일에 맞는 시그니처 아코드 추천"""
        style_mapping = {
            "classic": "Chanel Aldehydic Bouquet",
            "oriental": "Amouage Gold Oud Complex",
            "modern": "Terre d'Hermes Mineral Woody",
            "gourmand": "Angel Gourmand Revolution",
            "aquatic": "CK One Aquatic Minimalism",
            "dark": "Black Orchid Dark Floral"
        }

        accord_name = style_mapping.get(style)
        if accord_name:
            return next((a for a in self.signature_accords if a.name == accord_name), None)
        return None

    def calculate_fragrance_complexity(self, ingredients: List[str]) -> Dict[str, Any]:
        """향수 복잡성 계산 및 분석"""
        complexity_analysis = {
            "total_ingredients": len(ingredients),
            "complexity_level": "simple",
            "balance_score": 0.0,
            "harmony_potential": 0.0,
            "recommendations": []
        }

        # 복잡성 레벨 결정
        if len(ingredients) <= 8:
            complexity_analysis["complexity_level"] = "simple"
        elif len(ingredients) <= 15:
            complexity_analysis["complexity_level"] = "moderate"
        elif len(ingredients) <= 25:
            complexity_analysis["complexity_level"] = "complex"
        else:
            complexity_analysis["complexity_level"] = "haute_couture"

        # 조화도 잠재력 계산
        harmony_scores = []
        for i, ingredient1 in enumerate(ingredients):
            for ingredient2 in ingredients[i+1:]:
                harmony = self.get_harmony_score(ingredient1, ingredient2)
                harmony_scores.append(harmony)

        if harmony_scores:
            complexity_analysis["harmony_potential"] = sum(harmony_scores) / len(harmony_scores)

        # 균형 점수 계산 (간단한 가중치 기반)
        complexity_analysis["balance_score"] = min(1.0, complexity_analysis["harmony_potential"] * 1.2)

        # 추천사항 생성
        if complexity_analysis["harmony_potential"] < 0.6:
            complexity_analysis["recommendations"].append("일부 향료 조합의 조화도가 낮습니다. 대체 향료를 검토해보세요.")

        if len(ingredients) > 20:
            complexity_analysis["recommendations"].append("향료 수가 많습니다. 핵심 향료에 집중하여 단순화를 고려하세요.")

        return complexity_analysis

    def generate_perfumer_formula(
        self,
        style: str,
        target_mood: str,
        preferred_notes: List[str] = None
    ) -> Dict[str, Any]:
        """마스터 조향사 스타일의 향수 공식 생성"""

        # 스타일에 맞는 조향사와 아코드 선택
        perfumer = self.recommend_perfumer_style({"style": style})
        signature_accord = self.get_signature_accord(style)

        if not signature_accord:
            signature_accord = self.signature_accords[0]  # 기본값

        # 공식 구성
        formula = {
            "name": f"{perfumer.perfumer_name} 스타일 조향",
            "style": style,
            "target_mood": target_mood,
            "perfumer_inspiration": perfumer.perfumer_name,
            "signature_accord": {
                "name": signature_accord.name,
                "ingredients": signature_accord.ingredients,
                "creator": signature_accord.creator
            },
            "structure": self._generate_fragrance_structure(style, preferred_notes),
            "blending_notes": self._generate_blending_notes(perfumer, signature_accord),
            "estimated_longevity": self._estimate_longevity(signature_accord),
            "complexity_analysis": None
        }

        # 전체 향료 리스트 생성 및 복잡성 분석
        all_ingredients = []
        for layer in formula["structure"].values():
            all_ingredients.extend(layer.get("ingredients", []))

        formula["complexity_analysis"] = self.calculate_fragrance_complexity(all_ingredients)

        return formula

    def _generate_fragrance_structure(
        self,
        style: str,
        preferred_notes: List[str] = None
    ) -> Dict[str, Any]:
        """향수 구조 생성"""
        golden_ratios = self.golden_ratios["classic_pyramid"]

        if style == "modern":
            golden_ratios = self.golden_ratios["modern_structure"]

        structure = {
            "top_notes": {
                "percentage": golden_ratios.get("top_notes", 0.25) * 100,
                "ingredients": self._select_ingredients_by_category("top_notes", preferred_notes),
                "characteristics": "첫인상, 휘발성이 높음, 15-30분 지속"
            },
            "heart_notes": {
                "percentage": golden_ratios.get("middle_notes", 0.45) * 100,
                "ingredients": self._select_ingredients_by_category("heart_notes", preferred_notes),
                "characteristics": "향수의 핵심, 2-4시간 지속"
            },
            "base_notes": {
                "percentage": golden_ratios.get("base_notes", 0.30) * 100,
                "ingredients": self._select_ingredients_by_category("base_notes", preferred_notes),
                "characteristics": "향수의 기반, 6-8시간 이상 지속"
            }
        }

        return structure

    def _select_ingredients_by_category(
        self,
        category: str,
        preferred_notes: List[str] = None
    ) -> List[str]:
        """카테고리별 향료 선택"""
        category_mapping = {
            "top_notes": ["시트러스", "베르가못", "레몬", "자몽"],
            "heart_notes": ["장미", "자스민", "라벤더", "제라늄"],
            "base_notes": ["바닐라", "머스크", "샌달우드", "앰버"]
        }

        base_ingredients = category_mapping.get(category, [])

        if preferred_notes:
            # 선호하는 노트가 있다면 우선적으로 포함
            selected = []
            for note in preferred_notes:
                if note.lower() in [ing.lower() for ing in base_ingredients]:
                    selected.append(note)

            # 부족한 경우 기본 향료로 보완
            while len(selected) < 3:
                for ingredient in base_ingredients:
                    if ingredient not in selected and len(selected) < 3:
                        selected.append(ingredient)
                break
        else:
            selected = base_ingredients[:3]

        return selected

    def _generate_blending_notes(
        self,
        perfumer: PerfumerSignature,
        accord: AccordFormula
    ) -> str:
        """조향 노트 생성"""
        notes = [
            f"{perfumer.perfumer_name}의 시그니처 스타일을 따라 {accord.name} 아코드를 중심으로 구성",
            f"주요 기법: {', '.join(perfumer.innovation_techniques[:2])}",
            f"조합 철학: {perfumer.compositional_preferences.get('philosophy', '')}",
            "각 층간의 부드러운 전환을 위해 브릿지 노트 활용",
            "최종 숙성 기간: 4-6주 권장"
        ]
        return "\n".join(f"• {note}" for note in notes)

    def _estimate_longevity(self, accord: AccordFormula) -> Dict[str, str]:
        """지속력 추정"""
        # 아코드 특성에 따른 지속력 추정
        base_longevity = {
            "projection": "보통-강함",
            "longevity": "6-8시간",
            "sillage": "팔 길이 정도",
            "dry_down": "따뜻하고 부드러운 잔향"
        }

        if "oriental" in accord.name.lower() or "oud" in accord.name.lower():
            base_longevity["longevity"] = "8-12시간"
            base_longevity["projection"] = "강함"
        elif "aquatic" in accord.name.lower() or "fresh" in accord.name.lower():
            base_longevity["longevity"] = "4-6시간"
            base_longevity["projection"] = "약함-보통"

        return base_longevity

    def analyze_fragrance_dna(self, ingredients: List[str]) -> Dict[str, Any]:
        """향수 DNA 분석 - 향수의 고유한 특성 분석"""
        dna_analysis = {
            "dominant_families": [],
            "personality_traits": [],
            "seasonal_suitability": [],
            "occasion_matching": [],
            "uniqueness_score": 0.0,
            "market_positioning": ""
        }

        # 향료 패밀리 분석
        family_counts = {}
        for ingredient in ingredients:
            # 실제로는 향료 데이터베이스와 매칭이 필요
            if any(citrus in ingredient.lower() for citrus in ["citrus", "lemon", "bergamot", "orange"]):
                family_counts["citrus"] = family_counts.get("citrus", 0) + 1
            elif any(floral in ingredient.lower() for floral in ["rose", "jasmine", "lavender", "lily"]):
                family_counts["floral"] = family_counts.get("floral", 0) + 1
            elif any(woody in ingredient.lower() for woody in ["wood", "cedar", "sandalwood", "vetiver"]):
                family_counts["woody"] = family_counts.get("woody", 0) + 1

        # 지배적 패밀리 결정
        if family_counts:
            dominant_family = max(family_counts, key=family_counts.get)
            dna_analysis["dominant_families"] = [dominant_family]

        # 성격적 특성 추론
        if "citrus" in family_counts:
            dna_analysis["personality_traits"].extend(["energetic", "fresh", "optimistic"])
        if "woody" in family_counts:
            dna_analysis["personality_traits"].extend(["sophisticated", "grounded", "confident"])
        if "floral" in family_counts:
            dna_analysis["personality_traits"].extend(["romantic", "elegant", "feminine"])

        # 계절적 적합성
        citrus_ratio = family_counts.get("citrus", 0) / len(ingredients)
        woody_ratio = family_counts.get("woody", 0) / len(ingredients)

        if citrus_ratio > 0.3:
            dna_analysis["seasonal_suitability"].extend(["spring", "summer"])
        if woody_ratio > 0.3:
            dna_analysis["seasonal_suitability"].extend(["fall", "winter"])

        # 독특함 점수 계산
        ingredient_diversity = len(set(ingredients))
        complexity_factor = min(1.0, ingredient_diversity / 15.0)
        dna_analysis["uniqueness_score"] = complexity_factor * 0.8 + (len(dna_analysis["personality_traits"]) / 10.0) * 0.2

        return dna_analysis

    def create_master_blend_recommendation(
        self,
        user_preferences: Dict[str, Any],
        target_market: str = "premium"
    ) -> Dict[str, Any]:
        """마스터급 블렌드 추천"""

        # 사용자 선호도 분석
        preferred_style = user_preferences.get("style", "modern")
        mood_preference = user_preferences.get("mood", "sophisticated")
        notes_preference = user_preferences.get("preferred_notes", [])

        # 적합한 조향사 스타일 선택
        perfumer = self.recommend_perfumer_style({
            "style": preferred_style,
            "innovation": user_preferences.get("innovation_level", "medium")
        })

        # 조향사 기법 적용
        master_formula = self.generate_perfumer_formula(
            style=preferred_style,
            target_mood=mood_preference,
            preferred_notes=notes_preference
        )

        # 시장 포지셔닝 분석
        market_analysis = {
            "target_market": target_market,
            "price_point": self._estimate_price_point(master_formula, target_market),
            "competitive_advantage": self._analyze_competitive_advantage(master_formula),
            "market_appeal": self._assess_market_appeal(master_formula)
        }

        # 최종 추천
        recommendation = {
            "master_formula": master_formula,
            "market_analysis": market_analysis,
            "perfumer_insights": {
                "chosen_perfumer": perfumer.perfumer_name,
                "style_rationale": f"{perfumer.perfumer_name}의 {', '.join(perfumer.style_characteristics[:2])} 특성이 요구사항과 일치",
                "key_techniques": perfumer.innovation_techniques
            },
            "development_timeline": self._create_development_timeline(),
            "success_factors": self._identify_success_factors(master_formula)
        }

        return recommendation

    def _estimate_price_point(self, formula: Dict[str, Any], target_market: str) -> Dict[str, Any]:
        """가격대 추정"""
        base_prices = {
            "mass": {"50ml": "3-5만원", "100ml": "5-8만원"},
            "premium": {"50ml": "8-15만원", "100ml": "12-25만원"},
            "luxury": {"50ml": "20-50만원", "100ml": "35-80만원"}
        }

        complexity_multiplier = min(2.0, formula["complexity_analysis"]["total_ingredients"] / 15.0)

        return {
            "base_price": base_prices.get(target_market, base_prices["premium"]),
            "complexity_factor": complexity_multiplier,
            "justification": f"복잡성 수준 {formula['complexity_analysis']['complexity_level']}에 따른 가격 조정"
        }

    def _analyze_competitive_advantage(self, formula: Dict[str, Any]) -> List[str]:
        """경쟁 우위 분석"""
        advantages = []

        if formula["complexity_analysis"]["harmony_potential"] > 0.8:
            advantages.append("탁월한 향료 조화도")

        if formula["complexity_analysis"]["complexity_level"] == "complex":
            advantages.append("고도의 조향 기술력")

        if formula["signature_accord"]["creator"] in ["Jean-Claude Ellena", "Francis Kurkdjian"]:
            advantages.append("마스터 조향사 스타일 계승")

        advantages.append("한국인 취향 최적화")
        advantages.append("프리미엄 원료 사용")

        return advantages

    def _assess_market_appeal(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        """시장 어필 평가"""
        return {
            "innovation_level": formula["complexity_analysis"]["complexity_level"],
            "accessibility": "moderate" if formula["complexity_analysis"]["total_ingredients"] > 15 else "high",
            "seasonality": "year-round" if len(formula.get("target_mood", "")) > 0 else "seasonal",
            "gender_appeal": "unisex",
            "age_target": "25-45세"
        }

    def _create_development_timeline(self) -> Dict[str, str]:
        """개발 일정"""
        return {
            "concept_development": "1-2주",
            "initial_blending": "2-3주",
            "testing_refinement": "4-6주",
            "aging_maturation": "6-8주",
            "final_evaluation": "1-2주",
            "total_timeline": "14-21주"
        }

    def _identify_success_factors(self, formula: Dict[str, Any]) -> List[str]:
        """성공 요인"""
        factors = [
            "마스터 조향사 기법 적용",
            "고품질 원료 사용",
            "균형잡힌 향료 구성",
            "독창적인 시그니처 아코드"
        ]

        if formula["complexity_analysis"]["harmony_potential"] > 0.75:
            factors.append("우수한 향료 조화도")

        return factors