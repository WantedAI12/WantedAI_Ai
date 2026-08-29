"""
조향사 지식베이스 도구
향수 제조 기술, 역사, 원료 정보 제공
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class KnowledgeEntry:
    """지식베이스 항목"""
    id: str
    category: str
    title: str
    content: str
    tags: List[str]
    confidence: float
    source: str

class PerfumerKnowledgeBase:
    """조향사 지식베이스"""

    def __init__(self):
        """초기화"""
        self.knowledge_data = self._load_knowledge_data()

    def _load_knowledge_data(self) -> List[KnowledgeEntry]:
        """지식베이스 데이터 로드"""
        # 실제로는 데이터베이스에서 로드하지만, 여기서는 기본 데이터 제공
        return [
            KnowledgeEntry(
                id="pyramid_structure",
                category="technique",
                title="향수 피라미드 구조",
                content="향수는 탑 노트(15-25%), 미들 노트(30-50%), 베이스 노트(25-40%)로 구성됩니다. 탑 노트는 첫인상을, 미들 노트는 향수의 핵심을, 베이스 노트는 지속성을 담당합니다.",
                tags=["구조", "피라미드", "기초"],
                confidence=0.95,
                source="조향사 교과서"
            ),
            KnowledgeEntry(
                id="citrus_notes",
                category="ingredient",
                title="시트러스 노트",
                content="시트러스 노트는 레몬, 오렌지, 베르가못 등으로 구성되며, 높은 휘발성을 가집니다. 주로 탑 노트로 사용되며, 상쾌하고 밝은 느낌을 줍니다.",
                tags=["시트러스", "탑노트", "휘발성"],
                confidence=0.9,
                source="원료 가이드"
            ),
            KnowledgeEntry(
                id="floral_blending",
                category="technique",
                title="플로럴 블렌딩 기법",
                content="플로럴 향료는 서로 잘 어울리며, 로즈와 재스민의 조합이 특히 인기입니다. 비율은 로즈 60%, 재스민 40% 정도가 균형잡힌 조합입니다.",
                tags=["플로럴", "블렌딩", "로즈", "재스민"],
                confidence=0.85,
                source="조향사 경험"
            ),
            KnowledgeEntry(
                id="woody_base",
                category="ingredient",
                title="우디 베이스 노트",
                content="샌달우드, 시더우드, 베티버 등이 베이스 노트로 사용됩니다. 낮은 휘발성과 높은 지속성을 가지며, 향수의 깊이를 더합니다.",
                tags=["우디", "베이스", "지속성"],
                confidence=0.9,
                source="원료 가이드"
            ),
            KnowledgeEntry(
                id="fixative_notes",
                category="technique",
                title="픽사티브 노트",
                content="픽사티브는 향수의 지속성을 높이는 역할을 합니다. 앰버그리스, 머스크, 바닐라 등이 대표적이며, 전체 조합의 5-10% 정도 사용합니다.",
                tags=["픽사티브", "지속성", "앰버그리스"],
                confidence=0.8,
                source="조향사 교과서"
            ),
            KnowledgeEntry(
                id="oriental_perfumes",
                category="style",
                title="오리엔탈 향수",
                content="오리엔탈 향수는 스파이시하고 따뜻한 느낌이 특징입니다. 바닐라, 앰버, 스파이스 노트를 중심으로 구성되며, 겨울에 특히 인기입니다.",
                tags=["오리엔탈", "스파이시", "바닐라"],
                confidence=0.85,
                source="향수 스타일 가이드"
            ),
            KnowledgeEntry(
                id="fresh_aquatic",
                category="style",
                title="프레시 아쿠아틱 향수",
                content="상쾌하고 물의 느낌을 주는 향수입니다. 시트러스 노트와 아쿠아틱 노트를 조합하며, 여름에 인기입니다. 칼론, 아쿠아 디 지오가 대표적입니다.",
                tags=["프레시", "아쿠아틱", "여름"],
                confidence=0.8,
                source="향수 스타일 가이드"
            ),
            KnowledgeEntry(
                id="chypre_structure",
                category="technique",
                title="시프르 구조",
                content="시프르는 오크모스, 패출리, 베르가못의 조합으로 만드는 클래식한 향수 구조입니다. 1917년 코티의 시프르가 시초이며, 현대 향수의 기반이 되었습니다.",
                tags=["시프르", "오크모스", "패출리", "클래식"],
                confidence=0.9,
                source="향수 역사"
            ),
            KnowledgeEntry(
                id="fougere_structure",
                category="technique",
                title="푸제르 구조",
                content="푸제르는 라벤더, 쿠마린, 오크모스의 조합으로 만드는 남성향 구조입니다. 1882년 우비간의 푸제르 로얄이 시초이며, 현대 남성향의 기반이 되었습니다.",
                tags=["푸제르", "라벤더", "쿠마린", "남성향"],
                confidence=0.9,
                source="향수 역사"
            ),
            KnowledgeEntry(
                id="natural_vs_synthetic",
                category="ingredient",
                title="자연향료 vs 합성향료",
                content="자연향료는 식물에서 추출한 순수한 향료이며, 합성향료는 화학적으로 합성한 향료입니다. 자연향료는 복잡한 향을, 합성향료는 일관된 품질을 제공합니다.",
                tags=["자연향료", "합성향료", "품질"],
                confidence=0.85,
                source="원료 가이드"
            )
        ]

    def query(self, question: str, category: Optional[str] = None, limit: int = 5) -> List[KnowledgeEntry]:
        """
        지식베이스 쿼리

        Args:
            question: 질문
            category: 카테고리 필터
            limit: 반환할 결과 수

        Returns:
            관련 지식 항목들
        """
        question_lower = question.lower()
        results = []

        for entry in self.knowledge_data:
            # 카테고리 필터
            if category and entry.category != category:
                continue

            # 키워드 매칭
            score = 0
            if question_lower in entry.title.lower():
                score += 3
            if question_lower in entry.content.lower():
                score += 2
            for tag in entry.tags:
                if question_lower in tag.lower():
                    score += 1

            if score > 0:
                results.append((entry, score))

        # 점수순 정렬
        results.sort(key=lambda x: x[1], reverse=True)

        return [entry for entry, score in results[:limit]]

    def get_by_category(self, category: str) -> List[KnowledgeEntry]:
        """카테고리별 지식 조회"""
        return [entry for entry in self.knowledge_data if entry.category == category]

    def get_categories(self) -> List[str]:
        """사용 가능한 카테고리 목록"""
        return list(set(entry.category for entry in self.knowledge_data))

    def add_knowledge(self, entry: KnowledgeEntry):
        """새 지식 추가"""
        self.knowledge_data.append(entry)

    def search_by_tags(self, tags: List[str]) -> List[KnowledgeEntry]:
        """태그로 검색"""
        results = []
        for entry in self.knowledge_data:
            if any(tag.lower() in [t.lower() for t in entry.tags] for tag in tags):
                results.append(entry)
        return results

# 전역 인스턴스
_knowledge_base = None

def get_knowledge_base() -> PerfumerKnowledgeBase:
    """싱글톤 지식베이스 반환"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = PerfumerKnowledgeBase()
    return _knowledge_base

def query_knowledge_base(question: str, category: Optional[str] = None, limit: int = 5) -> List[KnowledgeEntry]:
    """지식베이스 쿼리 함수"""
    kb = get_knowledge_base()
    return kb.query(question, category, limit)
