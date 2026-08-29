"""
실제 향수 화학 및 조향 도메인 지식
Real Fragrance Chemistry and Perfumery Domain Knowledge
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class FragranceNote:
    """향료 노트 정보"""
    name: str
    category: str  # top, middle, base
    odor_family: str  # floral, woody, citrus, etc
    volatility: float  # 0-1 (높을수록 빨리 증발)
    molecular_weight: float  # 분자량
    odor_intensity: float  # 0-1 (향의 강도)
    stability: float  # 0-1 (안정성)
    cost_per_gram: float  # 원료 가격

# 실제 향료 데이터베이스 (조향사들이 사용하는 실제 데이터)
FRAGRANCE_DATABASE = {
    # Top Notes (가장 휘발성 높음)
    'bergamot': FragranceNote('Bergamot', 'top', 'citrus', 0.9, 136.24, 0.7, 0.6, 45),
    'lemon': FragranceNote('Lemon', 'top', 'citrus', 0.95, 136.24, 0.8, 0.5, 35),
    'orange': FragranceNote('Orange', 'top', 'citrus', 0.92, 136.24, 0.6, 0.5, 30),
    'grapefruit': FragranceNote('Grapefruit', 'top', 'citrus', 0.88, 136.24, 0.65, 0.55, 40),
    'mandarin': FragranceNote('Mandarin', 'top', 'citrus', 0.9, 136.24, 0.5, 0.5, 38),
    'lavender': FragranceNote('Lavender', 'top', 'aromatic', 0.8, 154.25, 0.7, 0.7, 55),
    'eucalyptus': FragranceNote('Eucalyptus', 'top', 'fresh', 0.85, 154.25, 0.8, 0.6, 25),
    'peppermint': FragranceNote('Peppermint', 'top', 'fresh', 0.83, 156.27, 0.9, 0.65, 35),
    'aldehydes': FragranceNote('Aldehydes', 'top', 'metallic', 0.95, 114.19, 0.95, 0.4, 80),
    'pink_pepper': FragranceNote('Pink Pepper', 'top', 'spicy', 0.75, 136.24, 0.6, 0.7, 120),

    # Middle Notes (중간 휘발성)
    'rose': FragranceNote('Rose', 'middle', 'floral', 0.5, 154.25, 0.8, 0.8, 250),
    'jasmine': FragranceNote('Jasmine', 'middle', 'floral', 0.45, 196.29, 0.9, 0.75, 300),
    'ylang_ylang': FragranceNote('Ylang Ylang', 'middle', 'floral', 0.4, 204.35, 0.85, 0.7, 180),
    'geranium': FragranceNote('Geranium', 'middle', 'floral', 0.55, 154.25, 0.7, 0.75, 90),
    'iris': FragranceNote('Iris', 'middle', 'powdery', 0.35, 248.32, 0.6, 0.85, 500),
    'violet': FragranceNote('Violet', 'middle', 'floral', 0.48, 192.26, 0.5, 0.7, 150),
    'cinnamon': FragranceNote('Cinnamon', 'middle', 'spicy', 0.4, 132.16, 0.8, 0.8, 45),
    'cardamom': FragranceNote('Cardamom', 'middle', 'spicy', 0.45, 196.29, 0.7, 0.75, 85),
    'nutmeg': FragranceNote('Nutmeg', 'middle', 'spicy', 0.42, 162.27, 0.6, 0.8, 60),
    'black_pepper': FragranceNote('Black Pepper', 'middle', 'spicy', 0.5, 136.24, 0.75, 0.7, 70),

    # Base Notes (가장 낮은 휘발성, 오래 지속)
    'sandalwood': FragranceNote('Sandalwood', 'base', 'woody', 0.1, 220.35, 0.5, 0.95, 200),
    'cedarwood': FragranceNote('Cedarwood', 'base', 'woody', 0.12, 222.37, 0.6, 0.9, 80),
    'patchouli': FragranceNote('Patchouli', 'base', 'earthy', 0.08, 222.37, 0.8, 0.9, 120),
    'vetiver': FragranceNote('Vetiver', 'base', 'earthy', 0.05, 222.37, 0.7, 0.95, 150),
    'musk': FragranceNote('Musk', 'base', 'animalic', 0.02, 342.0, 0.4, 0.98, 400),
    'amber': FragranceNote('Amber', 'base', 'warm', 0.03, 296.0, 0.6, 0.95, 350),
    'vanilla': FragranceNote('Vanilla', 'base', 'sweet', 0.08, 152.15, 0.7, 0.9, 180),
    'benzoin': FragranceNote('Benzoin', 'base', 'balsamic', 0.06, 212.25, 0.65, 0.92, 90),
    'tonka_bean': FragranceNote('Tonka Bean', 'base', 'sweet', 0.07, 146.14, 0.75, 0.88, 160),
    'oakmoss': FragranceNote('Oakmoss', 'base', 'mossy', 0.04, 404.0, 0.5, 0.96, 280)
}

# 향료 가족 간 조화도 매트릭스 (조향사들의 경험적 지식)
HARMONY_MATRIX = {
    ('citrus', 'citrus'): 0.9,
    ('citrus', 'floral'): 0.85,
    ('citrus', 'woody'): 0.75,
    ('citrus', 'spicy'): 0.7,
    ('citrus', 'fresh'): 0.95,
    ('citrus', 'aromatic'): 0.8,

    ('floral', 'floral'): 0.95,
    ('floral', 'woody'): 0.85,
    ('floral', 'sweet'): 0.9,
    ('floral', 'powdery'): 0.88,
    ('floral', 'spicy'): 0.7,
    ('floral', 'fresh'): 0.75,

    ('woody', 'woody'): 0.9,
    ('woody', 'earthy'): 0.95,
    ('woody', 'balsamic'): 0.85,
    ('woody', 'spicy'): 0.8,
    ('woody', 'mossy'): 0.88,

    ('spicy', 'spicy'): 0.85,
    ('spicy', 'sweet'): 0.8,
    ('spicy', 'warm'): 0.9,

    ('fresh', 'fresh'): 0.9,
    ('fresh', 'aromatic'): 0.85,

    ('earthy', 'earthy'): 0.9,
    ('earthy', 'mossy'): 0.85,

    ('sweet', 'sweet'): 0.95,
    ('sweet', 'balsamic'): 0.9,
    ('sweet', 'warm'): 0.88,

    ('animalic', 'woody'): 0.7,
    ('animalic', 'sweet'): 0.75,
    ('animalic', 'floral'): 0.65,

    # 기본값
    ('default', 'default'): 0.5
}

class FragranceChemistry:
    """실제 향수 화학 계산 클래스"""

    @staticmethod
    def calculate_vapor_pressure(molecular_weight: float, temperature: float = 25) -> float:
        """
        Clausius-Clapeyron 방정식으로 증기압 계산
        실제 향료의 휘발성 예측
        """
        # 간략화된 버전 (실제로는 더 복잡한 계산 필요)
        R = 8.314  # 기체 상수
        T = temperature + 273.15  # 켈빈 온도

        # 분자량이 클수록 증기압이 낮음 (덜 휘발)
        vapor_pressure = np.exp(-molecular_weight / (R * T * 10))
        return vapor_pressure

    @staticmethod
    def calculate_diffusion_coefficient(molecular_weight: float) -> float:
        """
        Graham의 법칙으로 확산 계수 계산
        향의 확산 속도 예측
        """
        # 공기 중 확산 계수 (간략화)
        D0 = 1e-5  # m²/s (기준값)
        D = D0 * np.sqrt(28.97 / molecular_weight)  # 28.97은 공기의 평균 분자량
        return D

    @staticmethod
    def calculate_odor_threshold(intensity: float, molecular_weight: float) -> float:
        """
        후각 역치 계산 (얼마나 적은 양으로도 감지되는지)
        Weber-Fechner 법칙 응용
        """
        # 강도와 분자량을 고려한 역치 계산
        threshold = 0.001 * np.exp(-intensity * 5) * (molecular_weight / 200)
        return threshold

    @staticmethod
    def calculate_longevity(notes: List[Tuple[str, float]]) -> float:
        """
        향수의 지속력 계산
        베이스 노트의 비율과 휘발성을 고려
        """
        total_longevity = 0
        total_weight = 0

        for note_name, concentration in notes:
            if note_name in FRAGRANCE_DATABASE:
                note = FRAGRANCE_DATABASE[note_name]
                # 휘발성이 낮을수록 오래 지속
                longevity_contribution = (1 - note.volatility) * concentration * note.stability
                total_longevity += longevity_contribution
                total_weight += concentration

        if total_weight > 0:
            return total_longevity / total_weight
        return 0

    @staticmethod
    def calculate_sillage(notes: List[Tuple[str, float]]) -> float:
        """
        향수의 확산력(sillage) 계산
        향의 강도와 확산 계수를 고려
        """
        total_sillage = 0
        total_weight = 0

        for note_name, concentration in notes:
            if note_name in FRAGRANCE_DATABASE:
                note = FRAGRANCE_DATABASE[note_name]
                diffusion = FragranceChemistry.calculate_diffusion_coefficient(note.molecular_weight)
                sillage_contribution = note.odor_intensity * concentration * diffusion * 10000
                total_sillage += sillage_contribution
                total_weight += concentration

        if total_weight > 0:
            return min(1.0, total_sillage / total_weight)
        return 0

    @staticmethod
    def calculate_harmony(notes: List[Tuple[str, float]]) -> float:
        """
        향료들 간의 조화도 계산
        조향사의 경험적 지식 기반
        """
        if len(notes) < 2:
            return 1.0

        harmony_score = 0
        pair_count = 0

        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                note1_name, conc1 = notes[i]
                note2_name, conc2 = notes[j]

                if note1_name in FRAGRANCE_DATABASE and note2_name in FRAGRANCE_DATABASE:
                    family1 = FRAGRANCE_DATABASE[note1_name].odor_family
                    family2 = FRAGRANCE_DATABASE[note2_name].odor_family

                    # 조화도 매트릭스에서 점수 찾기
                    key = (family1, family2) if (family1, family2) in HARMONY_MATRIX else (family2, family1)
                    harmony = HARMONY_MATRIX.get(key, HARMONY_MATRIX[('default', 'default')])

                    # 농도를 가중치로 사용
                    weight = conc1 * conc2
                    harmony_score += harmony * weight
                    pair_count += weight

        if pair_count > 0:
            return harmony_score / pair_count
        return 0.5

    @staticmethod
    def calculate_balance(top_notes: List[Tuple[str, float]],
                         middle_notes: List[Tuple[str, float]],
                         base_notes: List[Tuple[str, float]]) -> float:
        """
        향수의 균형감 계산
        전통적인 피라미드 구조 준수 여부
        """
        # 이상적인 비율: Top 20-30%, Middle 30-40%, Base 30-50%
        total_top = sum(c for _, c in top_notes)
        total_middle = sum(c for _, c in middle_notes)
        total_base = sum(c for _, c in base_notes)
        total = total_top + total_middle + total_base

        if total == 0:
            return 0

        # 비율 계산
        ratio_top = total_top / total
        ratio_middle = total_middle / total
        ratio_base = total_base / total

        # 이상적인 비율과의 거리 계산
        ideal_distance = 0
        ideal_distance += min(abs(ratio_top - 0.25), 0.2)  # Top: 25% ± 5%
        ideal_distance += min(abs(ratio_middle - 0.35), 0.2)  # Middle: 35% ± 5%
        ideal_distance += min(abs(ratio_base - 0.40), 0.2)  # Base: 40% ± 10%

        # 0-1 범위로 정규화 (거리가 작을수록 균형이 좋음)
        balance_score = 1 - (ideal_distance / 0.6)
        return max(0, balance_score)

    @staticmethod
    def calculate_cost(notes: List[Tuple[str, float]]) -> float:
        """
        향수 제조 원가 계산
        """
        total_cost = 0

        for note_name, concentration in notes:
            if note_name in FRAGRANCE_DATABASE:
                note = FRAGRANCE_DATABASE[note_name]
                # 농도(%) * 100ml 기준 * 원료 가격
                cost = concentration * 0.01 * 100 * note.cost_per_gram
                total_cost += cost

        return total_cost

    @staticmethod
    def evaluate_fragrance_complete(
        top_notes: List[Tuple[str, float]],
        middle_notes: List[Tuple[str, float]],
        base_notes: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        향수의 종합적인 평가
        """
        all_notes = top_notes + middle_notes + base_notes

        evaluation = {
            'harmony': FragranceChemistry.calculate_harmony(all_notes),
            'longevity': FragranceChemistry.calculate_longevity(all_notes),
            'sillage': FragranceChemistry.calculate_sillage(all_notes),
            'balance': FragranceChemistry.calculate_balance(top_notes, middle_notes, base_notes),
            'cost': FragranceChemistry.calculate_cost(all_notes),
            'complexity': len(all_notes) / 30,  # 노트 개수 기반 복잡도
            'uniqueness': 1.0 - (len(all_notes) / 50),  # 너무 많은 노트는 독특함을 해침
        }

        # 종합 점수 (가중 평균)
        weights = {
            'harmony': 0.25,
            'longevity': 0.20,
            'sillage': 0.15,
            'balance': 0.20,
            'complexity': 0.10,
            'uniqueness': 0.10
        }

        overall_score = sum(evaluation[key] * weights.get(key, 0)
                          for key in evaluation if key != 'cost')
        evaluation['overall'] = overall_score

        return evaluation