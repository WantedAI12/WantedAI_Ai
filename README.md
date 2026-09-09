# WantedAI Perfumery AI Backend

자연어 향 의도를 정량 조향식과 시간별 향 예측으로 연결하는 CPU 기반 AI 백엔드입니다. 향수·바디로션·바디워시의 제형 차이를 구분하며, 원료 선택·배합 탐색·후보 비교·입력 보완·제조 절차 안내를 제공합니다.

현재 릴리스는 **V62 / Perfumery AI Core 1.4.0**입니다. 비상업 연구·개발용 서비스이며 제조 또는 판매 승인 서비스가 아닙니다.

## 서비스 연결

- API: [Perfumery AI Core](https://junseong2im--perfumery-ai-core-web.modal.run)
- API 문서: [OpenAPI / Swagger](https://junseong2im--perfumery-ai-core-web.modal.run/docs)
- 저장소 브랜치: dev
- 최신 배포: 2026-09-10, 인증·모델 식별값·세 제품 예측·조향식·입력 도우미의 원격 응답 확인 완료
- 기존 Modal Proxy Token과 API 주소를 유지합니다.

프론트엔드는 팀 백엔드를 호출하고, 팀 백엔드가 AI API를 호출합니다. API 키를 브라우저 코드에 넣지 않습니다. 외부 LLM 서비스의 API 키는 필요하지 않습니다.

```http
Authorization: Bearer <MODAL_TOKEN_ID>.<MODAL_TOKEN_SECRET>
Content-Type: application/json
```

인증 형식은 [Modal 공식 Proxy Token 문서](https://modal.com/docs/guide/webhook-proxy-auth)를 참고하세요. 실제 키는 저장소에 커밋하지 마세요.

## 주요 기능

- 한국어·영어 자연어 향 요청과 원하는 향 / 제외할 향 / 시간대별 조건 해석
- 665개 향 표현 개념과 332개 한국어 별칭, 450개 세부 향 서술어 예측
- 기존 146축 분자 향 표현과 세부 향 표현을 함께 활용하는 원료 탐색
- 안전·원가·가용성·제품 농도·원료별 허용량을 반영한 정량 배합
- 조향식 평가·재평가·대안 생성·후보 비교·대화형 수정
- 향수·로션·바디워시의 제형 조건과 세척 이벤트를 구분하는 시간별 향 방출 예측
- 로션 베이스 준비·배합 탐색·분배·방출 계산 및 조향·제조 절차 안내
- IFRA·EU REACH·K-REACH·FDA 관련 상태와 근거를 구분하는 규제 정보 출력
- CPU 경량 언어 모델을 통한 입력 의도 정리와 보완 질문. 정량 배합은 수치 조향 엔진이 수행

규제 정보는 자동 인증이나 법률 판단이 아닙니다. 로션·바디워시의 상세 방출 예측에는 해당 제형의 계수와 공정 조건이 필요하며, 제품명만으로 실제 물성을 확정하지 않습니다.

## 조향 요청

```http
POST /v1/formulas
```

```json
{
  "brief": "피오니와 청사과 향, 코코넛은 제외",
  "max_risk_tier": 2,
  "enable_registry_trace_candidates": true,
  "target_region": "EU",
  "product_category": "eau_de_parfum",
  "max_ingredient_price_per_kg": 180,
  "require_full_profile_match": true,
  "target_similarity": 95
}
```

원료 레지스트리에는 29,259개 행이 연결돼 있으며, 참조용 항목과 조향 후보는 구분합니다. 모든 레지스트리 항목이 안전하거나 자동 배합 가능한 원료라는 뜻은 아닙니다. 실제 후보군은 원료 프로필과 요청의 안전·가격·가용성 조건에 따라 결정됩니다.

| 응답 필드 | 의미 |
|---|---|
| recipe | 현재 조건과 평가 기준을 충족한 정량 조향식 |
| closest_candidate | 기준 미달 시 반환하는 가장 가까운 연구 후보 |
| calculated_profile_similarity | 계산 모델의 목표 향 프로필 일치 점수 |
| full_profile_target_met | 현재 목표 기준의 충족 여부 |
| score_contract | 점수 정의, 모델 적용 범위와 근거 |
| temporal_profile | 시간별 향 변화 |
| ingredient_temporal_profile | 원료별 잔존 농도·헤드스페이스 기여 |
| safety | 안전·규제 검사 결과와 제한 사항 |

기본 목표는 95점입니다. 목표 미달 또는 평가 불가일 때 recipe가 빈 배열인 것은 정상 응답일 수 있으며, closest_candidate를 승인된 조향식으로 표시하면 안 됩니다. no_safe_match 상태만으로 독성 원료가 발견됐다고 단정하지 말고 점수·제약·사유를 함께 확인하세요.

원격 예제의 최초 조향은 약 108초, 동일 요청의 캐시 응답은 약 0.42초였습니다. 팀 백엔드는 최대 300초 계산 시간을 고려해 AI 호출 timeout과 작업 상태 표시를 설정해야 합니다.

## 주요 API

| 용도 | 경로 |
|---|---|
| 상태·원료·기능 확인 | GET /health, GET /v1/catalog, GET /v1/ai/capabilities |
| 요청 구조화·보완 | POST /v1/briefs/prepare, POST /v1/briefs/clarify |
| 입력 도우미 | POST /v1/ai/assistant |
| 조향식 생성·평가 | POST /v1/formulas, POST /v1/formulas/evaluate, POST /v1/formulas/reassess |
| 대안·비교·수정 | POST /v1/formulas/alternatives, POST /v1/formulas/compare, POST /v1/formulas/revise |
| 향 표현 조회·해석·예측 | GET /v1/odor-expressions, POST /v1/odor-expressions/interpret, POST /v1/odor-expressions/predict |
| 통합 제형 조건·시간 예측 | POST /v1/applications/unified/context, POST /v1/applications/unified/predict |
| 로션 베이스·배합 설계 | POST /v1/applications/body-lotion/prepare, POST /v1/applications/body-lotion/design, POST /v1/applications/body-lotion/optimize |
| 로션 방출 계산 | POST /v1/applications/body-lotion/simulate, POST /v1/applications/body-lotion/predict-release |
| 조향·제조 절차 | POST /v1/formulation-workflows/plan |

향수 조향식 생성과 완성 로션 설계는 서로 다른 입력 계약을 사용합니다. 로션은 전용 경로를 사용하고 전체 스키마는 /docs에서 확인하세요. 기존 필드와 함께 새 응답 필드도 전달하도록 팀 백엔드 DTO를 구성하는 것이 좋습니다.

## 검증 결과

V62의 분자 구조 기반 향 서술어 예측을 기존 분리 평가 원료 714개에서 확인했습니다. 평가 원료의 정답 기록을 입력으로 조회하지 않았습니다.

| 지표 | 결과 |
|---|---:|
| Micro average precision × 100 | 37.4162 |
| Macro average precision × 100 | 25.6012 |
| 상위 5개 서술어 precision × 100 | 40.0280 |
| Binary log loss | 0.03787624 |
| Brier loss | 0.00886431 |

AP는 향 서술어 검색·구별력 지표이지 실제 후각 정확도 백분율이 아닙니다. Macro AP는 기록량 조건을 만족한 183개 서술어로 계산했습니다. 이 평가 집합은 이전 모델에서도 사용한 회귀 평가 집합이며 새로운 블라인드 연구는 아닙니다.

로컬 연결 검증에서는 다음을 확인했습니다.

- 최근 모델·API 관련 테스트 87개 통과. 전체 저장소 테스트 결과와는 별개입니다.
- 향수·로션·바디워시 API 연결 및 반복 요청 결과 일치
- 고정 예제에서 기존 146축 향 값과 시간별 공기 농도 계산 유지
- ‘피오니와 청사과 향’ 조향 후보: 계산 점수 87.4298, 12개 원료, 목표 95점에는 미달

원격 배포 확인 결과는 [배포 검증 기록](benchmarks/modal_v62_release.json)에 정리했습니다. 원격의 동일 예제 점수는 87.1753이었고 역시 목표 미달로 반환했습니다. 입력 도우미는 언어 모델의 제안을 기존 조건 검사기와 함께 확인합니다.

모든 자연어 요청이 95점을 통과한다거나, 생성 조향식이 사람에게 95% 동일하게 느껴진다고 주장하지 않습니다. 과거의 합성 프록시·혼합물 진단·분류 지표를 자연어 조향식의 실제 정확도로 환산하지 않습니다.

## 실행 환경과 소스 구성

API 서버는 CPU 1 core / RAM 1 GiB, GPU 없이 실행합니다. 언어 도우미는 별도의 비공개 CPU 1 core / RAM 2 GiB 워커를 필요할 때 호출합니다. 각 워커는 최소 인스턴스 0, 최대 1이며 외부 유료 LLM API를 호출하지 않습니다. 일반 조향 요청마다 언어 워커를 호출하지 않습니다.

- fragrance_ai/: 향 해석·원료 탐색·수치 모델·API 계약
- deploy/modal_release_v62.py: 현재 연구 서비스 배포 진입점
- deploy/runtime_release_v62.py: 해시가 고정된 모델 의존 파일 묶음 준비
- deploy/compact_language_worker.py: 비공개 CPU 언어 워커
- tests/: 모델·요청·배포·안전 계약 테스트
- scripts/: 학습·평가·패키징·연결 검증 도구
- [MODAL_DEPLOYMENT.md](MODAL_DEPLOYMENT.md): 배포·인증·원격 확인 기록

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[commercial,test]" rdkit==2025.9.4
```

이 저장소는 소스 코드와 공개 가능한 계약·검증 도구를 제공합니다. 해시가 고정된 연구 모델·원본 데이터·로컬 실행 설정은 승인된 내부 아티팩트로 별도 관리하며, 저장소 clone만으로 해당 모델이 자동 다운로드되지는 않습니다. 서비스 이용자는 기존 API 주소와 백엔드 인증 정보만 있으면 됩니다.

API 키, .env, 개인 Modal 설정, 연구 원본 데이터는 커밋하지 않습니다. 라이선스는 LicenseRef-Proprietary이며 세부 범위는 [LICENSE_POLICY.md](LICENSE_POLICY.md)를 확인하세요.
