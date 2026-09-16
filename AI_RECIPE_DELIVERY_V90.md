# 백엔드 전달 계약: 배합 반환과 향 목표 일치도 분리

2026-09-17 업데이트: 이 계약은 V91에 포함해 배포했다. [최신 배포 및 백엔드 호환 내용](AI_BACKEND_ALIGNMENT_V91.md)을 우선한다. 아래는 2026-09-16 V90 로컬 검증 당시의 개발 기록이다. V91에서는 목표 미달 향수의 전달 상태와 원 엔진 상태를 분리하고, 큰 로션 세부 진단에 무손실 전송을 추가했다.

## 달라지는 동작

기존에는 목표가 90점일 때 85.56점으로 계산된 배합을 `closest_candidate`에만 반환하고 `recipe`를 비웠다. 새 계약은 계산 가능한 유효 배합이 있으면 **90점 미달이어도 동일한 원료·배합비를 `recipe`에도 반환**한다.

점수, 목표 달성 판정, 규제 검토 및 운영 승인 상태는 바꾸지 않는다. 기존 `closest_candidate`, `score`, `calculated_profile_similarity`, `status`와 상세 진단도 유지한다. 새 추론이나 추가 LLM/API 호출 없이 응답을 구성하는 단계에서만 처리한다.

전체 점수를 계산할 수 없는 요청, 부분적으로만 해석된 목표, 내부 안전 검사 실패, 물리적 존재 조건 실패, 유효한 배합이 없는 경우까지 레시피를 만들어 넣지는 않는다. 반대로 점수가 80점 아래라는 이유만으로 올리거나 별도의 반환 하한을 추가하지도 않는다.

## 백엔드에서 사용할 필드

| 필드 | 의미와 처리 |
|---|---|
| `recipe` | 반환된 배합 배열. 90점 미달 배합도 들어갈 수 있으므로, 배열이 있다는 이유로 목표 달성이나 승인으로 처리하면 안 됨 |
| `target_match_score` | 해당 배합과 요청 향의 계산된 일치도. 숫자 0~100 또는 null. 향수의 기존 `calculated_profile_similarity`, 로션의 기존 `score`를 그대로 전달 |
| `target_match_unit` | `model_points_0_100`. 모델 평가 점수이며 사람의 정확도 확률이 아님 |
| `target_match_met` | 원래 향수의 `full_profile_target_met`, 로션의 `profile_target_met`와 동일. 목표에 못 미치면 false 유지 |
| `model_accuracy_percent` | 별도 모델 성능 검증 수치용 nullable 숫자 필드. 이번 레시피 평가에는 해당 검증값이 없으므로 null |
| `model_accuracy_status` | 현재는 `separate_benchmark_required`. 개별 일치도를 모델 전체 정확도로 복사하지 않음 |
| `recipe_delivery.contract` | `best-available-recipe/v1` |
| `recipe_delivery.returned` | 이번 응답의 `recipe`에 배합이 실제 들어 있는지 |
| `recipe_delivery.status` | `target_met`, `target_not_met`, `target_unverified`, `blocked`, `unavailable` |
| `recipe_delivery.source` | 기존 `recipe`인지, 기존 `closest_candidate`를 전달한 것인지, 배합이 없는지 구분 |
| `recipe_delivery.approval_inferred` | 항상 false. 데이터 전달로 승인을 새로 추론하지 않음 |
| `recipe_delivery.blockers` | 전체 목표 미평가·내부 안전 검사·물리 조건 등의 이유로 후보를 `recipe`에 옮기지 못한 경우의 코드 |

`confidence`는 기존과 같이 숫자 또는 null이며 설명은 `confidence_kind`다. 이번 변경으로 confidence의 타입이나 값을 일치도 점수로 대체하지 않는다.

화면의 ‘목표 일치도’는 `target_match_score`를 사용하면 된다. 예를 들어 85.5626이면 85.6점으로 표시할 수 있다. 90점이나 80점으로 보정하지 않는다. null은 0점이나 90점이 아니라 ‘미평가’로 처리한다.

## 적용 경로와 기존 판정

- `POST /v1/formulas`: 최종 JSON에 새 필드와 배합을 반환.
- `POST /v1/formulas/stream`: `event: result`에 같은 JSON을 반환. 별도 점수를 다시 만들지 않음.
- `POST /v1/applications/body-lotion/design`, `/optimize`: 같은 배합 전달 계약 사용. `/prepare`, `/simulate`, `/predict-release`의 역할은 바꾸지 않음.
- `/v1/formulas/evaluate`, 고정 배합 평가, 자연어 수정: 후보의 `result` 안에도 적용. 후보의 `status`, `selection_gates`는 원래 평가 결과를 유지.
- `/v2` 운영 평가: 등록 근거·안전·과학적 적용 범위·목표 달성 게이트를 그대로 유지. 근거 미등록 시 422이며, 점수 미달 결과를 승인 후보로 이동시키지 않음.

백엔드는 레시피 표시/저장과 승인 여부를 분리해야 한다. `recipe`가 있다는 이유만으로 `target_match_met`나 `recommendation_allowed`를 true로 바꾸지 않는다. 기존 최상위 `status`는 평가 원문이므로 `no_safe_match` 또는 `research_candidate_only`와 함께 배합이 전달될 수 있다. 배합 표시 여부에는 `recipe_delivery.returned`를 사용하고, 목표 달성과 운영 승인에는 각각 해당 판정 필드를 사용한다.

규제 결과는 기존 `regulatory`를 그대로 읽는다. 저장 응답 재생에 사용한 85.56점 로션에는 IFRA 검토 차단 표시도 있었다. 새 계약은 그 배합 데이터를 반환하지만 **규제 상태를 통과로 변경하지 않는다.** `recipe`는 승인된 제조 지시라는 뜻이 아니다.

연동 시 DTO에 신규 필드를 추가하거나 알 수 없는 응답 필드의 처리 정책을 확인해야 한다. 기존 입력 JSON은 바꿀 필요가 없다. 직접 생성 JSON에는 `X-Perfumery-Recipe-Contract: best-available-recipe/v1` 헤더도 제공한다. SSE에서는 result 본문의 `recipe_delivery.contract`로 확인한다.

## 검증 결과

- 최종 설치 패키지의 관련 테스트 **187개 통과**, 실패·오류·건너뜀 0개.
- 기존 운영에서 저장된 미달 결과 10건과 통과 결과 2건을 새 반환 함수에 재생하여 **12/12건에서 배합 반환**을 확인. 이것은 12건의 품질 통과나 새 모델 평가를 의미하지 않음.
- 전체 원점수와 기존 판정·배합비 보존 확인.
- 실제 로컬 API에 저장된 모델 출력을 주입해 6개 호출 검사 완료: 향수 JSON, 향수 SSE, 로션 JSON, 후보 평가, v2 사전 확인, 근거 없는 운영 평가의 422.
- JSON과 SSE의 result가 동일함을 확인.
- 0점·79점 등 80점 미만 입력도 계약 시험에 포함. 어떠한 점수 하한이나 상향 보정도 하지 않음.
- 원료·학습 가중치·목표 참조 자료·탐색 및 물리 계산 코드는 변경하지 않음.

새 신경망 추론이나 운영 HTTPS 검사를 실행한 것으로 기록하지 않는다. 이번 검사는 반환 계약에 대한 **저장 결과 재생과 로컬 API 검사**다.

실제 응답 원문과 검사 기록:

- [로션 응답](output/backend-recipe-delivery-v90-local-20260916-final/lotion_json.response.json): 원점수 85.56261626811559와 1,096개 원료 배합을 `recipe`에 전달. `profile_target_met=false` 유지.
- [향수 응답](output/backend-recipe-delivery-v90-local-20260916-final/perfume_json.response.json): 원점수 80.24911852729562와 15개 원료 배합을 전달. `full_profile_target_met=false` 유지.
- [조향 SSE 원문](output/backend-recipe-delivery-v90-local-20260916-final/perfume_sse.response.sse)
- [검사 기록](output/backend-recipe-delivery-v90-local-20260916-final/verification.json)
- [회귀 검사](.benchmarks/v90_recipe_delivery/installed-contract-tests-03.xml)

## 최종 패키지

기준은 운영 v24에 사용한 package-13이다. 응답 계약과 연결 코드 4개 파일만 덮어썼고 나머지 `fragrance_ai` Python 모듈 209개는 그대로 보존했다. 원료 29,259행과 모델·목표 참조 해시도 같다. 작업 폴더의 다른 미채택 모델 실험은 포함하지 않았다.

- 최종 준비 기록: `.benchmarks/v90_recipe_delivery/package-03/preparation.json`
- 변경 범위: `.benchmarks/v90_recipe_delivery/package-03/recipe-delivery-provenance.json`
- wheel: `.benchmarks/v90_recipe_delivery/package-03/wheel/perfumery_ai_core-1.4.0-py3-none-any.whl`
- SHA256: `59418cc297de607883ab4be0fcb911dcdc962ef7d72754364324cae3623bd655`

초기 package-01의 재생 검사에서는 규제 검토 상태를 배합 데이터 반환 차단과 혼합한 문제를 발견했다. 배합 반환과 승인 판정을 분리해 수정했고, 해당 초기 기록은 덮어쓰지 않았다. package-01·02는 최종 후보가 아니며, **최종 검증 대상은 package-03**이다.
