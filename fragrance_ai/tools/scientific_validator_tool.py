"""
과학적 향수 조합 검증 도구
화학적 호환성, 안정성, 조화도 검증
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum

class ValidationResult(Enum):
    """검증 결과"""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"

@dataclass
class NotesComposition:
    """향료 조합 정보"""
    note_id: int
    note_name: str
    percentage: float
    pyramid_level: str
    volatility: float
    strength: float
    longevity: float
    is_natural: bool

@dataclass
class ValidationReport:
    """검증 보고서"""
    overall_result: ValidationResult
    harmony_score: float
    stability_score: float
    safety_score: float
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]

class ScientificValidator:
    """과학적 향수 검증기"""

    def __init__(self):
        """초기화"""
        # 화학적 호환성 매트릭스 (실제 데이터 기반)
        self.compatibility_matrix = self._load_compatibility_matrix()

        # 안전성 규칙
        self.safety_rules = self._load_safety_rules()

        # 조화도 계산 가중치
        self.harmony_weights = {
            'volatility': 0.3,
            'strength': 0.25,
            'longevity': 0.25,
            'chemical_compatibility': 0.2
        }

    def _load_compatibility_matrix(self) -> Dict[Tuple[str, str], float]:
        """화학적 호환성 매트릭스 로드"""
        # 실제 향료 화학적 호환성 데이터
        return {
            ('citrus', 'floral'): 0.8,
            ('citrus', 'woody'): 0.7,
            ('citrus', 'oriental'): 0.6,
            ('floral', 'woody'): 0.9,
            ('floral', 'oriental'): 0.8,
            ('woody', 'oriental'): 0.9,
            ('spicy', 'oriental'): 0.8,
            ('spicy', 'woody'): 0.7,
            ('fresh', 'citrus'): 0.9,
            ('fresh', 'aquatic'): 0.8,
        }

    def _load_safety_rules(self) -> List[Dict[str, Any]]:
        """안전성 규칙 로드"""
        return [
            {
                'rule': 'max_natural_percentage',
                'condition': lambda comp: sum(n.percentage for n in comp if n.is_natural) > 80.0,
                'message': '자연 향료 비율이 80%를 초과합니다'
            },
            {
                'rule': 'min_top_notes',
                'condition': lambda comp: sum(n.percentage for n in comp if n.pyramid_level == 'top') < 10.0,
                'message': '탑 노트 비율이 너무 낮습니다 (최소 10% 권장)'
            },
            {
                'rule': 'max_single_note',
                'condition': lambda comp: max(n.percentage for n in comp) > 40.0,
                'message': '단일 향료 비율이 40%를 초과합니다'
            }
        ]

    def validate_composition(self, composition: List[NotesComposition]) -> ValidationReport:
        """
        향수 조합 검증

        Args:
            composition: 향료 조합 리스트

        Returns:
            검증 보고서
        """
        warnings = []
        errors = []
        recommendations = []

        # 기본 검증
        self._validate_percentages(composition, warnings, errors)
        self._validate_pyramid_structure(composition, warnings, recommendations)

        # 화학적 검증
        harmony_score = self._calculate_harmony_score(composition)
        stability_score = self._calculate_stability_score(composition)
        safety_score = self._calculate_safety_score(composition, warnings, errors)

        # 전체 결과 결정
        if errors:
            overall_result = ValidationResult.INVALID
        elif warnings:
            overall_result = ValidationResult.WARNING
        else:
            overall_result = ValidationResult.VALID

        return ValidationReport(
            overall_result=overall_result,
            harmony_score=harmony_score,
            stability_score=stability_score,
            safety_score=safety_score,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )

    def _validate_percentages(self, composition: List[NotesComposition], warnings: List[str], errors: List[str]):
        """비율 검증"""
        total_percentage = sum(note.percentage for note in composition)

        if abs(total_percentage - 100.0) > 0.1:
            errors.append(f"총 비율이 100%가 아닙니다: {total_percentage:.1f}%")

        for note in composition:
            if note.percentage <= 0:
                errors.append(f"{note.note_name}의 비율이 0 이하입니다")
            elif note.percentage > 50:
                warnings.append(f"{note.note_name}의 비율이 너무 높습니다: {note.percentage:.1f}%")

    def _validate_pyramid_structure(self, composition: List[NotesComposition], warnings: List[str], recommendations: List[str]):
        """피라미드 구조 검증"""
        top_notes = [n for n in composition if n.pyramid_level == 'top']
        middle_notes = [n for n in composition if n.pyramid_level == 'middle']
        base_notes = [n for n in composition if n.pyramid_level == 'base']

        top_percentage = sum(n.percentage for n in top_notes)
        middle_percentage = sum(n.percentage for n in middle_notes)
        base_percentage = sum(n.percentage for n in base_notes)

        # 피라미드 구조 권장 비율: Top 15-25%, Middle 30-50%, Base 25-40%
        if top_percentage < 10:
            warnings.append(f"탑 노트 비율이 낮습니다: {top_percentage:.1f}%")
            recommendations.append("탑 노트 비율을 15-25%로 조정하세요")
        elif top_percentage > 30:
            warnings.append(f"탑 노트 비율이 높습니다: {top_percentage:.1f}%")

        if middle_percentage < 20:
            warnings.append(f"미들 노트 비율이 낮습니다: {middle_percentage:.1f}%")
            recommendations.append("미들 노트 비율을 30-50%로 조정하세요")

        if base_percentage < 20:
            warnings.append(f"베이스 노트 비율이 낮습니다: {base_percentage:.1f}%")
            recommendations.append("베이스 노트 비율을 25-40%로 조정하세요")

    def _calculate_harmony_score(self, composition: List[NotesComposition]) -> float:
        """조화도 점수 계산"""
        if not composition:
            return 0.0

        # 휘발성 균형
        volatility_scores = [n.volatility for n in composition]
        volatility_balance = 1.0 - np.std(volatility_scores)

        # 강도 균형
        strength_scores = [n.strength for n in composition]
        strength_balance = 1.0 - np.std(strength_scores)

        # 지속성 균형
        longevity_scores = [n.longevity for n in composition]
        longevity_balance = 1.0 - np.std(longevity_scores)

        # 화학적 호환성
        chemical_compatibility = self._calculate_chemical_compatibility(composition)

        # 가중 평균
        harmony_score = (
            volatility_balance * self.harmony_weights['volatility'] +
            strength_balance * self.harmony_weights['strength'] +
            longevity_balance * self.harmony_weights['longevity'] +
            chemical_compatibility * self.harmony_weights['chemical_compatibility']
        )

        return max(0.0, min(1.0, harmony_score))

    def _calculate_stability_score(self, composition: List[NotesComposition]) -> float:
        """안정성 점수 계산"""
        if not composition:
            return 0.0

        # 자연 향료 비율 (높을수록 안정성 증가)
        natural_ratio = sum(n.percentage for n in composition if n.is_natural) / 100.0

        # 휘발성 분산 (낮을수록 안정성 증가)
        volatility_scores = [n.volatility for n in composition]
        volatility_stability = 1.0 - np.std(volatility_scores)

        # 비율 균형 (균등할수록 안정성 증가)
        percentages = [n.percentage for n in composition]
        percentage_balance = 1.0 - np.std(percentages) / np.mean(percentages)

        stability_score = (
            natural_ratio * 0.4 +
            volatility_stability * 0.3 +
            percentage_balance * 0.3
        )

        return max(0.0, min(1.0, stability_score))

    def _calculate_safety_score(self, composition: List[NotesComposition], warnings: List[str], errors: List[str]) -> float:
        """안전성 점수 계산"""
        safety_score = 1.0

        # 안전성 규칙 검사
        for rule in self.safety_rules:
            if rule['condition'](composition):
                if rule['rule'].startswith('max_') or rule['rule'].startswith('min_'):
                    errors.append(rule['message'])
                    safety_score -= 0.3
                else:
                    warnings.append(rule['message'])
                    safety_score -= 0.1

        return max(0.0, min(1.0, safety_score))

    def _calculate_chemical_compatibility(self, composition: List[NotesComposition]) -> float:
        """화학적 호환성 계산"""
        if len(composition) < 2:
            return 1.0

        compatibility_scores = []

        for i, note1 in enumerate(composition):
            for note2 in composition[i+1:]:
                # 향료 타입 기반 호환성 (실제로는 더 복잡한 화학적 분석 필요)
                compatibility = self.compatibility_matrix.get(
                    (note1.note_name.lower(), note2.note_name.lower()), 0.5
                )
                compatibility_scores.append(compatibility)

        return np.mean(compatibility_scores) if compatibility_scores else 0.5

    def suggest_improvements(self, composition: List[NotesComposition]) -> List[str]:
        """개선 제안"""
        suggestions = []

        # 비율 최적화 제안
        total_percentage = sum(n.percentage for n in composition)
        if total_percentage != 100.0:
            suggestions.append(f"총 비율을 100%로 조정하세요 (현재: {total_percentage:.1f}%)")

        # 피라미드 구조 개선
        top_percentage = sum(n.percentage for n in composition if n.pyramid_level == 'top')
        if top_percentage < 15:
            suggestions.append("탑 노트를 추가하여 향수의 첫인상을 개선하세요")

        # 조화도 개선
        harmony_score = self._calculate_harmony_score(composition)
        if harmony_score < 0.7:
            suggestions.append("향료 간의 조화도를 높이기 위해 호환성 있는 향료를 선택하세요")

        return suggestions


# 전역 인스턴스
_validator = None

def get_validator() -> ScientificValidator:
    """싱글톤 검증기 반환"""
    global _validator
    if _validator is None:
        _validator = ScientificValidator()
    return _validator

def validate_composition(composition: List[NotesComposition]) -> ValidationReport:
    """향수 조합 검증 함수"""
    validator = get_validator()
    return validator.validate_composition(composition)
