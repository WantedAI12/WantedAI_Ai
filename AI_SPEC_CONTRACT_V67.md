# 2026-09-12 PDF 요구사항 대응 — AI 입출력 보완

대상 문서: 백엔드 기술명세서 95쪽, 프론트 기술명세서 89쪽, 디자인 와이어프레임 수정안 32쪽의 2026-09-12 파일.

상태: **로컬 AI 구현·연결 완료. 소스와 계약서를 `dev` 변경에 포함하며 운영 배포는 별개다.** V67은 이번 API 보완 작업의 식별명이다. 분자 백본은 기존 V66이며 모델을 새로 학습하거나 평가 기준을 바꾼 작업이 아니다.

`dev` 통합 점검에서는 향수의 제외 조건 전용 시간대가 다른 시간대의 긍정 향 목표를 상속하던 오류를 추가 수정했다. 로션의 별도 상속 계약은 변경하지 않았다. 관련 회귀 135개, 원료·입력·물성 회귀 89개를 통과했고, 아래의 이전 V67 검증과 중복될 수 있으므로 합산하지 않는다. 기존 테스트의 기대 점수를 낮추지 않고 구조-only/실측 연결 분기와 명시 원료 수 상한을 따로 검증했다.

통합 소스용 로컬 wheel SHA-256은 `0d750d2702ed59186860695397eafe7452d265f71f57ba629b88d3954c720796`, 재연결 카탈로그 manifest SHA-256은 `c36e7f51b485784079498b8e94e3247b29590cb1b0d99cb761fee41d97eb9d98`이다. 모든 원료 데이터 필드와 학습 모델 해시는 보존했다. 이 로컬 번들·프로필 파일 자체는 Git 업로드 대상이 아니다.

세 문서의 본문에서 AI 연동 요구를 추려 실제 AI 코드와 대조했고, 단위·구조화 화면·안전 상태 표의 렌더링도 확인했다. 문서에서 `미구현`으로 표시한 항목은 대부분 팀 백엔드/프론트의 연결 현황이며, AI 코드 전체의 미구현 판정은 아니다. PDF 안의 예시 점수·날짜·규정 상태는 구현의 고정값으로 사용하지 않았다.

## 1. 이번에 보완한 항목

| 문서 요구 | 실제로 부족했던 점 | 이번 결과 |
|---|---|---|
| BE-014~016, FE-032~035, INT-06, FG-06/07 | v2 필수 질문은 입력 유형·단위·답변 처리 경로가 기존 질문과 달랐음 | `POST /v2/briefs/clarify` 추가. 현재 질문만 답할 수 있고 오래된 질문 답변은 409. 부분 답변이 다른 누락값까지 확인한 것으로 바뀌지 않음 |
| BE-023/098, FE-029/030, INT-07 | capabilities에서 제품 목록만 보면 로션 전용 경로와 바디워시 세척 예측 지원을 구분하기 어려움 | 기존 capabilities에 `integration_contract` 추가. 제품별 경로·현재 등록 여부·모델 가용 여부·선행 입력·단위·저장 책임 제공 |
| BE-037/038, FE-051/052, INT-11, FG-08 | 기존 `/v1/formulas/compare`가 같은 프로세스의 최대 30분 캐시에 의존 | `POST /v2/formulas/compare` 추가. 백엔드가 저장한 평가 원본으로 새 프로세스에서도 비교. 비교 자체의 신규 추론은 0회 |
| BE-102, FE-048/050, FG-14 | 자연어 수정 요청에 저장된 원본 후보·평가·배합 버전을 연결하는 AI 계약이 없음 | `POST /v2/briefs/revise` 추가. 부모 평가/후보/백엔드 버전/배합 보존 → 수정 의도 확인 → 기존 v2 evaluate로 실제 생성. 승인 상속 없음 |
| BE-058/101, FE-066/068, INT-15, FG-15/16 | 감사 JSON/SSE에 분류 필터만 있고 기간·특정 버전·백엔드 스냅샷 범위가 없음 | 기존 감사/보고서 경로에 기간·버전·스냅샷 ID 추가. 제외 개수와 부족한 버전 참조 명시. JSON과 SSE가 같은 필터 결과 사용 |

기존 `/v1` URL·입력·응답의 핵심 필드는 유지한다. 추가 선택 필드와 추가 경로를 사용하는 방식이다. 배포 설정·키를 바꾸지 않았으며, 기존 운영 주소에 새 경로가 이미 있다고 가정하면 안 된다.

## 2. 이미 구현되어 있어 유지한 부분

- 자연어 향/제외 원료/시간대별 목표/강도 입력, 명시 농도와 원가 조건, 구조화 입력의 실제 계산 전달.
- 고정 배합 재평가. 원료 중복·합계 오류를 거부하고 입력 배합을 임의 정규화하지 않음.
- 시간별 향·농도·모델 불확실성 결과. 누락된 지속시간을 8시간 등으로 채우지 않음.
- 기본 조향 SSE의 실제 단계·단조 증가 진행률·명시적 오류·완료 이벤트.
- 등록 근거 평가와 원료/공급 변경 영향 비교. 실제 근거가 없는 항목은 통과로 바꾸지 않음.
- 로션을 기본 향수 요청으로 보내면 차단하는 경계. 로션 전용 설계·시뮬레이션·최적화 경로가 별도로 존재함.
- 바디워시의 세척 전후 예측은 제공된 단계별 계수와 세척 잔류 비율을 사용하는 전용 조건부 예측 경로.

제형 조건이 향수로 조용히 대체되는 것으로 처음 의심했으나, 로션 `application_context`는 실제 코드에서 이미 차단되고 있었다. 이 기능을 새로 구현했다고 집계하지 않는다.

## 3. 보완 질문: 한 경로로 답변하기

### 준비

`POST /v2/briefs/prepare`의 입력 계약은 유지한다. 예:

```json
{
  "request": {"formula": {"brief": "시원한 그린 우디 향"}},
  "evidence_policy": {
    "finished_batch_mass_g": 1000,
    "maximum_lead_time_days": 10,
    "maximum_purchase_cost_usd": 100
  }
}
```

제품·지역·완제품 향료 농도·농축액 원가를 명시하지 않았으므로 `needs_input`과 해당 질문을 반환한다. 기본값이 존재해도 사용자가 제출한 값으로 간주하지 않는다.

각 질문은 `id`, `field`, `input_type`, `required`, `reason_code`, `question`을 제공한다. 숫자 질문에는 단위와 범위, 선택 질문에는 선택지를 제공한다.

- `request.formula.product_category`: 제품 코드 선택.
- `request.formula.target_region`: EU/KR/US 선택.
- `request.formula.product_concentration_percent`: 완제품 향료 질량 %, 0 초과 30 이하.
- `request.formula.max_formula_cost_per_kg`: 향료 농축액 USD/kg 상한.
- 기존 `budget.scope`, `budget.product_density_g_ml`, `formula.brief` 같은 질문 ID도 그대로 답할 수 있음.

### 답변

`POST /v2/briefs/clarify`에 **직전에 prepare로 보낸 본문 그대로**, `prepared_result_id`, `answers`를 보낸다. `prepared_result_id`는 직전 준비 응답의 `result_id`이며 `review_id`와 다르다.

```json
{
  "request": {"formula": {"brief": "시원한 그린 우디 향"}},
  "evidence_policy": {
    "finished_batch_mass_g": 1000,
    "maximum_lead_time_days": 10,
    "maximum_purchase_cost_usd": 100
  },
  "prepared_result_id": "직전 응답의 64자리 result_id",
  "answers": {
    "request.formula.product_category": "eau_de_parfum",
    "request.formula.target_region": "EU",
    "request.formula.product_concentration_percent": 15,
    "request.formula.max_formula_cost_per_kg": 180
  }
}
```

문자열 자리표시자는 실제 반환 ID로 교체해야 한다. 조향 계산이나 사용자 확인 기록 저장은 이 호출에서 하지 않는다.

응답의 `request`는 다음 prepare/evaluate에 쓸 전체 본문이고, `prepared`는 새 준비 결과다. 그 안의 `prepared.status == ready`일 때 사용자 확인을 받은 후, 반환된 `request`에 `confirmed_review_id = prepared.review_id`를 추가하여 `/v2/formulas/evaluate`로 보낸다.

부분 답변을 했다면 **응답의 `request`와 `prepared.result_id`**로 다음 clarify를 호출한다. `reviewed_request`는 정규화된 검토 표시용이므로, 아직 누락된 필드를 기본값으로 채운 채 재제출하지 않는다.

원문과 구조화 값이 충돌하면 원문을 임의 수정하지 않는다. 예를 들어 원문이 농도 3%인데 구조화 값이 15%라면, 3%로 확인하거나 원문을 수정한 다음 다시 준비해야 한다. 답변에 안전 비활성화 필드를 끼워 넣거나 현재 질문에 없는 필드를 바꾸면 거부한다.

## 4. 제품과 단위: capabilities 연결

`GET /v1/ai/capabilities`의 새 `integration_contract`를 사용한다.

- `operations`: 실제 경로, HTTP 메서드, 전송 방식, 현재 프로세스 등록 여부, 가용 여부, 선행 조건, OpenAPI 요청 스키마 위치.
- `product_routing`: 향수/바디로션/바디워시에 맞는 경로.
- `units`: 질량%, g, kg당 원가, USD 구매예산, 일, 분, 모델 점수의 의미.
- `conversion`: 완제품 함량 관계와 부피 기준 예산에 필요한 근거 필드.
- `evidence.bundle_configured`: 근거 묶음의 설정 여부. 설정되었다고 개별 요청의 근거가 통과한 것은 아님.
- `workflow`: 프로젝트 저장·작업 취소·승인·PDF 변환의 담당 범위.

| 제품/작업 | 경로 | 구분 |
|---|---|---|
| 향수 기본 생성 | `/v1/formulas`, `/v1/formulas/stream` | 기존 계약 유지 |
| 향수 입력 확인·근거 기반 평가 | `/v2/briefs/prepare`, `/v2/formulas/evaluate` | 확인 ID와 등록 근거 필요 |
| 바디로션 자동 설계 | `/v1/applications/body-lotion/design` | 로션 전용 입력이며 기본 FormulaRequest와 다름 |
| 고정 로션 제형·배합 예측 | `/v1/applications/body-lotion/simulate` | 해당 제형의 계수·조건 필요 |
| 고정 베이스의 로션 배합 최적화 | `/v1/applications/body-lotion/prepare`, `/v1/applications/body-lotion/optimize` | 계수 적용 범위와 질량수지 검증 |
| 향수/로션/바디워시 단계별 조건부 예측 | `/v1/applications/unified/context`, `/v1/applications/unified/predict` | 실제 요청의 단계·원료별 계수. 바디워시는 세척 잔류값 필요 |

기본 `body_wash` 제품 코드로 향료 농축액을 생성할 수 있다는 사실과, 완제품 바디워시의 세척 잔류·향 방출을 계산할 수 있다는 사실은 별개다. 기본 생성만 하고 세척 모델을 적용했다고 표시하면 안 된다.

농축액 25% 원료를 향료 농도 15% 완제품에 넣으면 해당 원료의 완제품 질량 비율은 3.75%이다. 원/100mL 환산에는 밀도·농도·가격 기준일·환산값·비용 범위가 필요하다. 새 코드는 없는 환율·재고·밀도를 만들어 넣지 않는다.

## 5. 저장된 후보 비교

새 평가 `/v2/formulas/evaluate` 및 `/v2/formulas/reassess` 결과에는 `input_snapshot`이 추가된다. 요청·선택 농도 시나리오·목표 향·근거 정책·고정 배합 참조를 평가 원문에 함께 저장한다.

백엔드는 **평가 응답 전체를 변경 없이** 보존한다. 후보 ID와 점수만 추려 저장하면 새 비교에서 원래 조건을 확인할 수 없다.

`POST /v2/formulas/compare`:

```json
{
  "candidates": [
    {"backend_version_id": "BE-version-1", "candidate_id": "실제 후보 ID", "evaluation": "첫 번째 rd-candidates-2 응답 객체 전체"},
    {"backend_version_id": "BE-version-2", "candidate_id": "실제 후보 ID", "evaluation": "두 번째 rd-candidates-2 응답 객체 전체"}
  ]
}
```

위 `evaluation` 문자열 자리는 실제 JSON 객체로 보낸다. 비교 후보는 2~10개이고 기본 화면의 최대 선택 개수와 별개다. 백엔드 버전 ID와 평가 후보가 중복되면 거부한다.

응답:

- 후보별 원래 상태·게이트·점수·비용·시간축·근거·비교 기준.
- 원료 추가/제거/배합비 변화와 조성 차이. 조성 차이는 향 유사도가 아니다.
- `comparison_status`: `same_basis` 또는 `different_basis`.
- `different_basis_fields`: 목표, 제품/농도, 제약, 예산, 모델, 근거, 평가일 등이 다른 경우 해당 항목.
- `score_difference_points`: 비교 기준이 같고 두 점수가 모두 있을 때만 산출. 기준이 다르거나 값이 없으면 null.
- 자동 순위·승인·추천 결정 없음. `new_inference_count = 0`.
- `differs_from_active_contract`: 과거 평가와 현재 모델/근거/평가일의 차이 여부. 과거 내용을 현재 계산으로 덮어쓰지 않음.

원본 `result_id` 체크섬과 후보 식별·합계·상태/게이트 조합을 검증한다. 다만 체크섬은 **전자서명이 아니다**. 임의 내용을 만든 후 체크섬을 다시 계산하는 행위까지 인증하지 못하므로, 권한 있는 백엔드 저장소에서 읽은 원본만 전달해야 한다. AI는 비교 요청으로 승인이나 서비스 상태를 변경하지 않는다.

`input_snapshot`이 없는 이전 평가 원본은 422로 안내한다. 과거 결과를 버리거나 현재 버전 결과로 바꿔 적지 말고, 새 평가가 필요할 때 별도 버전으로 생성한다. 기존 `/v1/formulas/compare`의 캐시 기반 계약은 유지한다.

## 6. 저장된 후보의 자연어 수정

1. `/v2/briefs/revise`에 `source`와 `instruction`을 보낸다. `source`는 위 비교의 후보 항목 하나와 같은 구조이다.
2. AI가 저장된 원래 목표 향을 기준으로 상대적인 수정 의도를 만든다. 현재는 전체 향축의 `우디를 줄여줘` 같은 수정을 지원하며, 수치 배합 편집이나 특정 노트 구간 수정으로 조용히 해석하지 않는다.
3. 응답의 `request.revision`에는 부모 평가 ID·AI 후보 ID·백엔드 버전 ID·원래 배합·수정 지시가 들어간다.
4. 새 `prepared`를 사용자에게 확인받는다. 기존 확인 ID는 사용할 수 없다.
5. 반환된 전체 `request`에 새 `confirmed_review_id`를 붙여 `/v2/formulas/evaluate`로 보낸다. 이때 실제 조향 계산을 수행한다.
6. 새 응답의 `revision`과 후보별 `revision_composition_diff`를 저장한다. 성공 응답 전 예상 개선 점수를 만들지 않는다.

이 흐름은 **수정된 향 의도에 맞춰 재생성**하는 기능이다. 원본 배합 일부만 고정한 국소 편집이라고 주장하지 않는다. 특정 원료 비율을 직접 바꾸는 경우에는 기존 prepare(lines) → 확인 → reassess를 사용한다. 새 결과가 기권이면 원래 후보/승인을 자동 변경하지 않는다.

## 7. 감사 JSON·SSE의 범위 필터

기존 경로 세 개에 동일하게 적용한다.

- `/v1/audit-logs/stream`
- `/v1/reports/audit`
- `/v1/reports/audit/stream`

추가 선택 입력:

- `snapshot_id`: 백엔드가 고정한 이력 조회 묶음 ID.
- `selected_version_ids`: 포함할 버전 ID 목록. 중복·빈 목록은 거부.
- `period_start`, `period_end`: 시간대가 있는 시각. 시작·끝 포함. 시작이 끝보다 늦으면 거부.
- 각 감사 이벤트의 `version_id`: 그 이벤트가 실제로 속한 후보 버전. 기존 `reference_ids`에서 추측하지 않음.

분류·기간·버전 조건은 함께 적용된다. 버전 필터가 있는데 이벤트 버전이 없으면 제외하고 그 사실을 경고한다. 보고서의 `summary.filters`, 제외 이벤트/버전 개수, 공급된 버전 중 부모가 없는 경우의 경고를 그대로 표시한다.

기간 필터는 감사 이벤트에는 `occurred_at`, 버전 이력에는 `created_at`을 적용한다. 분류 필터는 감사 이벤트에만 적용한다. 현재 버전과 생성 시각이 기간 밖인 부모 버전이 있을 수 있으므로, 제외 이유를 표시한다.

같은 입력 묶음과 필터라면 JSON과 SSE의 `report_id`, 이력 내용, 요약은 같다. 보고서 생성 시각은 호출 시각으로 별도 제공한다. SSE는 유한 스냅샷이며 미래 이벤트 구독이나 중단 지점 재개가 아니다. 영구 이벤트 ID는 백엔드가 제공한 `event_id`이고 SSE의 stream/sequence ID와 다르다.

AI는 이 감사 기능에서 조향 계산·LLM 호출·승인 추론을 하지 않는다. PDF 렌더링, 저장 및 다운로드 권한은 팀 백엔드가 맡는다.

## 8. 입력/오류 및 책임 경계

- 잘못된 질문 답변, 손상된 평가 객체, 잘못된 배합, 지원하지 않는 수정은 422.
- 입력/모델/근거/날짜가 바뀐 질문 답변 또는 확인 ID는 409. 다시 준비·확인해야 한다.
- `/v2`의 미등록 근거는 기존과 같이 422로 생성 전 차단한다. 계산 후의 `abstained` 결과와 구분한다.
- 비교/저장 후보 수정 본문은 실제 웹 앱에서 최대 8 MiB, 그 밖의 새 `/v2` 본문은 최대 1 MiB. 초과는 413. 원료 후보 수 제한이 아니라 한 번의 전송/메모리 제한이다.
- NaN/Infinity 입력은 웹 앱에서 JSON 검증 오류로 처리하며 스트림이나 추론을 시작하지 않는다.
- 이 AI의 ID는 로그인 권한·테넌트 권한·영구 업무 ID가 아니다. 프론트에는 AI 비밀키를 주지 않는다.
- 프로젝트 생성/체크리스트/마감일/권한/작업 저장/승인/공개/알림/파일 업로드는 이번에 팀 서비스 코드 대신 구현한 기능이 아니다.
- 실제 규제·공급 증빙, 피부 적합성의 실측 검증, 실제 제형의 미제공 계수는 가짜 값으로 채우지 않았다. 제품별 지원/미지원 안내를 유지한다.

## 9. 검증 및 코드

- 선택 회귀 116개 통과: 기존 입력·시나리오·재평가·SSE·감사와 신규 28개 검사 포함. 전체 저장소 테스트나 원격 CI는 아님.
- 신규 최종 검사 28개 통과. 새 프로세스의 저장 후보 비교, 부분 답변의 기본값 승격 방지, 잘못된 게이트/체크섬/배합 차단, 수정의 새 확인 요구, 기간·버전 필터의 JSON/SSE 일치 확인.
- 새 wheel을 별도 폴더에 설치한 뒤 해당 패키지에서 질문 보완·캐시 없는 비교·수정 확인·capabilities 4개 검사 통과. 저장소 소스를 대신 import하지 않는 것도 확인.
- 테스트의 규제/견적·후보 응답 fixture는 실제 공급·안전·관능 증거가 아니다.
- 실제 로컬 V66 모델·카탈로그 실행 기록: `.benchmarks/spec_contract_v67/api-01/report.json`.
- 실제 모델의 보완 질문 4개 해결 → `ready`, 질문 처리의 추가 조향 추론 0회. 등록 근거가 없는 v2 생성은 422로 기존 차단 유지.
- 동일 향수/로션 요청 2건은 모두 HTTP 200, 반복 JSON 일치 및 캐시 적중. 기존 V66 대비 점수 변화 0: 피오니·청사과 향수 87.4297728279점(`no_safe_match`), 우디 로션 95.2점(`research_profile_target_met`). 새 정확도 달성 시험이 아니라 수치 결과 보존 확인이다.
- 실제 카탈로그는 소스 패키지 연결만 갱신. 원료 프로필·상한·가격·위험·모든 원료 필드 변경 0건.
- source model pin은 V66 그대로, stock/transport의 기존 학습 부모도 그대로 유지.
- 패키지: `dist/spec-contract-v67/candidate-01/wheel/perfumery_ai_core-1.4.0-py3-none-any.whl`.
- wheel SHA256: `b8af2484a74aa8cb0601a0858d64ff6e4d03cb64d71f0122870fce2111b16647`.
- 연결 카탈로그 SHA256: `6da1dc6d812356b16999827bf32e90751b3a53d6c1bfd113b9ddf5fbe9fc70e2`.
- `deploy/audit_api.py`는 웹 실행 파일이므로 wheel과 함께 최신 deploy 소스가 필요하다. wheel만 교체했다고 SSE 보고서 변경까지 적용된 것은 아니다.

주요 코드:

- `fragrance_ai/platform/rd_api.py`: 준비·답변·수정·확인·평가 연결.
- `fragrance_ai/platform/rd_clarification.py`: 질문 계약 및 허용된 답변 적용.
- `fragrance_ai/platform/rd_snapshots.py`: 저장 평가 검증·비교·배합 차이.
- `fragrance_ai/platform/operation_contracts.py`: 제품 경로·단위·필수 조건.
- `fragrance_ai/platform/ai_extensions.py`: 기존 API 및 모델과 연결.
- `deploy/audit_api.py`: 기간·버전 범위, JSON/SSE, 입력 크기 제한.
- `tests/test_spec_contract_v67.py`: 새 회귀 검사.
- `scripts/verify_spec_contract_v67.py`: 실제 로컬 모델과 API 확인.

이번 결과는 문서와 AI 사이의 연결 누락을 보완한 것이다. 전체 서비스 출시 완료, 실제 인간 후각 정확도 향상 또는 운영 배포 완료를 뜻하지 않는다.
