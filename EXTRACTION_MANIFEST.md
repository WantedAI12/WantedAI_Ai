# Extraction Manifest

## 기준

`Newss`를 더 최신의 통합 구현으로 보고 조향 레시피 생성, 평가, 규정 검사, 최적화, 사용자 피드백 학습에 직접 필요한 모듈만 선별했습니다. `ai project`의 영화 장면 분석과 중복된 구형 조향 모듈은 포함하지 않았습니다.

## 원본에서 복사한 파일

| 추출 파일 | 원본 |
|---|---|
| `fragrance_ai/ai/unified_ai_system.py` | `Newss/fragrance_ai/ai/unified_ai_system.py` |
| `fragrance_ai/domain/fragrance_chemistry.py` | `Newss/fragrance_ai/domain/fragrance_chemistry.py` |
| `fragrance_ai/knowledge/master_perfumer_principles.py` | `Newss/fragrance_ai/knowledge/master_perfumer_principles.py` |
| `fragrance_ai/rules/ifra_rules.py` | `Newss/fragrance_ai/rules/ifra_rules.py` |
| `fragrance_ai/tools/perfumer_knowledge_tool.py` | `Newss/fragrance_ai/tools/perfumer_knowledge_tool.py` |
| `fragrance_ai/tools/scientific_validator_tool.py` | `Newss/fragrance_ai/tools/scientific_validator_tool.py` |
| `fragrance_ai/training/moga_optimizer_stable.py` | `Newss/fragrance_ai/training/moga_optimizer_stable.py` |
| `fragrance_ai/training/rlhf_complete.py` | `Newss/fragrance_ai/training/rlhf_complete.py` |
| `fragrance_ai/utils/units.py` | `Newss/fragrance_ai/utils/units.py` |
| `fragrance_ai/data/moga_ingredients.db` | `Newss/fragrance_ai/data/moga_ingredients.db` |
| `tests/test_ifra.py` | `Newss/tests/test_ifra.py` |

## 추출본에서 정상화한 부분

- 깨진 원본 `fragrance_ai.ai.__init__` 대신 최소 공개 API를 정의했습니다.
- MOGA 데이터 경로를 현재 작업 디렉터리가 아니라 패키지 내부 `data/` 기준으로 변경했습니다.
- 테스트의 운영 로거 의존성을 Python 표준 로거로 교체했습니다.
- 조합 검증기의 농도 임계값을 0~1 비율이 아닌 코드 전체가 사용하는 0~100 퍼센트 기준으로 통일했습니다.
- 소형 CPU 설정의 CLI와 스모크 테스트를 추가했습니다.

## 제외한 항목

- FastAPI 라우터, 인증, 관리자, 결제, 이메일, SMS
- 프론트엔드, Docker/Kubernetes, 모니터링, 배포 스크립트
- 고객지원 LLM, 범용 RAG, 벡터 검색
- 영화 영상·오디오·장면 분석
- 문법 오류가 있는 `fragrance_recipe_generator.py`
- 사용자·주문·운영 데이터베이스
- 약 7GB의 모델 및 체크포인트

대형 모델은 어느 코드 경로에서 어떤 학습 설정으로 생성됐는지 먼저 확정해야 하므로 복사하지 않았습니다. 필요하면 후속 작업에서 검증된 체크포인트 하나만 선택해 로더와 함께 연결할 수 있습니다.

## 자연어 안전 레시피 확장

2026-07-11에 다음 기능을 추출본 위에 추가했습니다.

- `fragrance_ai/recommender/`: 자연어 파싱, 카탈로그, 안전 게이트, 제약 최적화, 서비스, 평가
- `fragrance_ai/data/safe_ingredient_catalog.json`: 제조 화이트리스트와 차단 원료
- `fragrance_ai/data/reference_fragrances.db`: 10,000개 향수/103,419개 노트 참고 코퍼스
- `benchmarks/brief_benchmark.json`: 의미 프로필 게이트 회귀 벤치마크
- `scripts/evaluate_recommender.py`: 벤치마크 실행기
- `REGULATORY_AND_DATA.md`: 공식 데이터와 규제 경계

## 2026-07-12 현실성 개선 확장

- `fragrance_ai/recommender/supplier.py`: 공급사 견적·재고·IFRA·SDS·COA·정량 알레르겐 증빙 게이트
- `fragrance_ai/recommender/odor_profiles.py`: 실측 원료 후각 프로파일 저장·평균·카탈로그 보정
- `fragrance_ai/recommender/realism.py`: 어코드 조화·충격 원료 지배·역사 조합 신호 평가
- `fragrance_ai/recommender/manufacturing.py`: 질량 기준 제조 계획·밀도 누락 차단·안정성 시험 계획
- `fragrance_ai/recommender/sensory.py`, `quality.py`: 블라인드 관능·안정성·파일럿 증빙 저장소
- `benchmarks/holdout_benchmark.json`: 30개 한·영 적대적 회귀셋
- `scripts/import_odor_data.py`, `scripts/audit_data.py`: 실측 관능 데이터 및 패키지 지문 검증
