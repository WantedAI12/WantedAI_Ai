"""Multilingual scent ontology and deterministic semantic-embedding fallback.

The bundled encoder deliberately has no network or heavyweight model dependency.
It embeds Unicode character/word n-grams into a fixed signed hashing space.  A
locally provisioned SentenceTransformer can be enabled explicitly, but the
runtime never downloads a model or silently changes its semantic contract.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from .models import SCENT_DIMENSIONS


ONTOLOGY_VERSION = "scent-ontology-2.0.0"
HASH_DIMENSIONS = 768


# Phrases include literal notes, perceptual descriptions, and common metaphors.
# They map language into the stable 19-dimensional public scent contract.
ONTOLOGY_PHRASES: dict[str, tuple[str, ...]] = {
    "citrus": (
        "레몬 껍질과 베르가못", "감귤 과즙", "유자 껍질", "상큼한 노란 과일",
        "lemon peel and bergamot", "sparkling citrus zest", "orange grapefruit lime yuzu",
    ),
    "fresh": (
        "이른 새벽의 차가운 공기", "창문을 연 듯 맑은 공기", "가볍고 투명한 바람",
        "청량하고 시원한", "투명하고 가볍게", "차가운 바람", "crisp cold morning air", "airy transparent refreshing breeze",
    ),
    "clean": (
        "햇빛에 말린 흰 셔츠", "갓 세탁한 흰 이불", "깨끗한 비누 거품",
        "보송한 코튼과 린넨", "sun dried white shirt", "fresh laundry cotton linen", "clean soap lather",
    ),
    "green": (
        "손으로 비빈 푸른 잎", "잘린 풀과 줄기", "젖은 이끼 낀 숲", "초록 수액",
        "crushed green leaves", "cut grass and stems", "wet mossy forest foliage",
    ),
    "aquatic": (
        "비 내린 뒤의 물기", "차가운 바닷바람", "젖은 돌과 물안개", "투명한 빗물",
        "after rain water mist", "cool sea breeze", "marine ozone wet stone",
    ),
    "floral": (
        "막 피어난 꽃다발", "부드러운 꽃잎", "정원의 꽃 향기", "fresh flower bouquet", "soft petals garden bloom",
    ),
    "rose": (
        "이슬 맺힌 장미 꽃잎", "붉은 장미 부케", "dewy rose petals", "red rose bouquet",
    ),
    "white_floral": (
        "밤에 핀 재스민과 치자", "하얀 꽃잎", "튜베로즈 부케", "night blooming jasmine gardenia", "creamy white flowers tuberose",
    ),
    "fruity": (
        "잘 익은 복숭아와 배", "과즙 많은 붉은 베리", "아삭한 사과", "ripe peach pear", "juicy berries and apple",
    ),
    "spicy": (
        "코끝이 알싸한 후추", "생강과 카다멈", "따뜻한 향신료", "peppery ginger cardamom", "warm dry spices",
    ),
    "aromatic": (
        "라벤더와 로즈마리 허브", "말린 허브 다발", "서늘한 약초 정원", "lavender rosemary herbs", "cool aromatic herb garden",
    ),
    "woody": (
        "마른 연필심과 삼나무", "매끈한 백단목", "오래된 나무 상자", "cedar pencil shavings", "smooth sandalwood dry timber",
        "charred timber", "dry timber",
    ),
    "amber": (
        "따뜻한 황금빛 수지", "포근한 벤조인과 라브다넘", "해 질 녘의 온기", "warm golden resin", "benzoin labdanum glowing warmth",
    ),
    "musky": (
        "깨끗하고 포근한 살결", "부드러운 피부 냄새", "하얀 머스크 베일", "clean warm skin", "soft skin scent white musk veil",
    ),
    "gourmand": (
        "바닐라 크림과 캐러멜", "달콤한 디저트", "초콜릿과 구운 설탕", "vanilla caramel dessert", "sweet chocolate toasted sugar",
    ),
    "powdery": (
        "보송한 화장분", "부드러운 베이비 파우더", "빈티지 립스틱", "soft cosmetic powder", "baby powder vintage lipstick",
    ),
    "smoky": (
        "꺼져가는 모닥불 연기", "검게 그을린 나무", "향 연기", "dying campfire smoke", "charred wood incense smoke",
    ),
    "leathery": (
        "새 가죽 가방", "부드러운 스웨이드 장갑", "무두질한 가죽", "new leather bag", "soft suede gloves tanned hide",
    ),
    "earthy": (
        "비 온 뒤 젖은 흙", "숲바닥의 부엽토", "축축한 뿌리와 이끼", "wet soil after rain", "forest floor humus damp roots moss",
    ),
}


@dataclass(frozen=True)
class SemanticOntologyResult:
    scores: dict[str, float]
    concepts: tuple[str, ...]
    confidence: float
    backend: str
    ontology_version: str = ONTOLOGY_VERSION


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _features(text: str) -> Iterable[str]:
    normalized = _normalize(text)
    compact = re.sub(r"[^0-9a-z가-힣]+", "", normalized)
    words = re.findall(r"[0-9a-z가-힣]+", normalized)
    for word in words:
        yield f"w:{word}"
    for n in (2, 3, 4, 5):
        for index in range(max(0, len(compact) - n + 1)):
            yield f"c{n}:{compact[index:index+n]}"


def _hash_embedding(text: str) -> np.ndarray:
    vector = np.zeros(HASH_DIMENSIONS, dtype=np.float32)
    for feature in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % HASH_DIMENSIONS
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        # Word features stabilize exact concepts; character features retain
        # Korean particles, inflections, misspellings, and compound words.
        weight = 1.65 if feature.startswith("w:") else 1.0 / math.sqrt(int(feature[1]))
        vector[index] += sign * weight
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


@lru_cache(maxsize=1)
def _hashed_prototypes() -> dict[str, np.ndarray]:
    return {
        dimension: np.vstack([_hash_embedding(phrase) for phrase in phrases])
        for dimension, phrases in ONTOLOGY_PHRASES.items()
    }


class ScentSemanticOntology:
    """Infer scent dimensions from literal and metaphorical multilingual text."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("PERFUMERY_AI_SENTENCE_MODEL")
        self._neural_model = None
        self._neural_prototypes: dict[str, np.ndarray] | None = None
        self._neural_error: str | None = None

    def _load_neural(self) -> bool:
        if not self.model_path or self._neural_error is not None:
            return False
        if self._neural_model is not None:
            return True
        try:
            # The opt-in path must already exist locally. Network downloads are
            # intentionally impossible in the production inference path.
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.model_path,
                local_files_only=True,
                device="cpu",
            )
            self._neural_prototypes = {
                dimension: np.asarray(
                    model.encode(list(phrases), normalize_embeddings=True),
                    dtype=np.float32,
                )
                for dimension, phrases in ONTOLOGY_PHRASES.items()
            }
            self._neural_model = model
            return True
        except Exception as error:  # Optional backend; deterministic fallback.
            self._neural_error = f"{type(error).__name__}:{error}"
            return False

    @staticmethod
    def _calibrate(similarity: float, *, neural: bool) -> float:
        floor = 0.34 if neural else 0.11
        return float(np.clip((similarity - floor) / (1.0 - floor), 0.0, 1.0))

    def infer(self, text: str) -> SemanticOntologyResult:
        if self._load_neural():
            assert self._neural_model is not None and self._neural_prototypes is not None
            query = np.asarray(
                self._neural_model.encode([text], normalize_embeddings=True)[0],
                dtype=np.float32,
            )
            raw = {
                dimension: float(np.max(matrix @ query))
                for dimension, matrix in self._neural_prototypes.items()
            }
            backend = "local_sentence_transformer"
            neural = True
        else:
            query = _hash_embedding(text)
            raw = {
                dimension: float(np.max(matrix @ query))
                for dimension, matrix in _hashed_prototypes().items()
            }
            backend = "bundled_signed_hash_ngram"
            neural = False

        scores = {
            dimension: self._calibrate(raw.get(dimension, 0.0), neural=neural)
            for dimension in SCENT_DIMENSIONS
        }
        ranked = sorted(scores, key=lambda name: (scores[name], name), reverse=True)
        concepts = tuple(name for name in ranked if scores[name] >= 0.16)[:8]
        confidence = max((scores[name] for name in concepts), default=0.0)
        return SemanticOntologyResult(
            scores=scores,
            concepts=concepts,
            confidence=round(confidence, 6),
            backend=backend,
        )
