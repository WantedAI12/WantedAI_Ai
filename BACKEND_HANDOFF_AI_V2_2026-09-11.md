# Perfumery AI Core — 백엔드 연동 전달서

작성일: 2026-09-11

대상: 팀 백엔드 개발 담당자
범위: 2026-09-10 기능명세서에 대응해 추가한 AI `/v2` 계약과 백엔드 연결 작업

2026-09-12 로컬 추가: [PDF 요구사항 대응 V67](AI_SPEC_CONTRACT_V67.md). `/v2/briefs/clarify`, 저장된 후보의 `/v2/formulas/compare`·`/v2/briefs/revise`, 제품별 경로/단위 안내, 감사 보고서 기간·버전 필터를 추가했습니다. 기존 질문 ID는 유지하며 이제 새 clarify 경로에서 함께 처리할 수 있습니다. 새 경로의 운영 배포는 아직 하지 않았습니다.

## 1. 전달 요약

AI 쪽에 필수 조건 확인, 입력 확인 ID, 자연어 해석 신뢰도·OOD 정보, 등록된 규제·공급 근거 평가, 고정 처방 재평가, 원료 변경 영향 비교를 추가했습니다. 기존 `/v1`의 동작은 유지됩니다.

**로컬 구현·패키지·검증이 완료됐고 소스와 계약서는 `dev` 변경에 포함합니다. 운영 서버는 이번 Git 변경으로 교체되지 않습니다.** 기존 운영 URL에서 `/v2`가 제공된다고 가정하면 안 됩니다. 서비스 담당자가 새 버전의 적용 상태와 인증 설정을 전달한 후 연결해 주세요.

백엔드의 우선 작업은 다음 네 가지입니다.

1. 향 요청을 `/v2/briefs/prepare`로 보내고, 보완과 사용자 확인을 완료한 뒤 생성 요청을 보냅니다.
2. AI 응답의 상태·게이트를 해석하고 요청·응답·모델·데이터 버전을 영구 저장합니다.
3. 검토 담당자·AI 서비스 운영자와 규제·공급 근거 파일의 등록·버전 변경 절차를 연결합니다.
4. 원료 변경 시 영향받는 후보를 찾아 AI에 비교를 요청하고, 재검토 상태·알림·후속 결정을 관리합니다.

실제 규제 문서·공급사 견적·재고 증빙은 아직 등록되지 않았습니다. 현재 설정에서 생성 요청이 근거 누락으로 차단되는 것은 의도된 동작입니다.

## 2. AI와 백엔드의 책임

| 작업 | AI 서비스에 구현된 내용 | 팀 백엔드에서 구현할 내용 |
|---|---|---|
| 자연어 입력 | 향 의도 구조화, 누락·충돌 표시, 해석 신뢰도·OOD 정보 | 프로젝트 권한 확인, 입력 원문 저장, 보완 질문·답변 전달 |
| 입력 확인 | 입력·모델·근거·날짜에 연결된 `review_id` 계산 | 사용자가 확인한 내용·사용자 ID·시각을 저장하고 생성 요청과 연결 |
| 생성·재평가 | 기존 수치 엔진 호출, 안전·프로필·적용범위·등록 근거 게이트 | 비동기 작업 큐, 상태 조회, 오류·재시도·취소 처리 |
| 후보 관리 | 후보와 진단 결과 반환 | 후보 영구 ID·처방 버전·비교·복제·편집 이력 관리 |
| 근거 평가 | 서버에 등록된 근거의 범위·기한·가격·공급 조건 검사 | 검토자·운영자 간 증빙 검토 및 등록 절차, 문서 보관·접근제어 |
| 변경 영향 | 같은 처방을 이전/현재 근거 버전에서 비교 | 원료→후보→프로젝트 관계 탐색, 재검토 상태 전이, 검토자 알림 |
| 승인·감사 | 분석 결과와 식별값 반환. 사용자 상태를 직접 변경하지 않음 | 승인·반려·보류·기권 결정, 권한 검증, 감사 로그, 관능 결과 연결 |

프론트엔드는 팀 백엔드만 호출하고, AI 서비스 인증 정보는 서버 측에 둡니다. 새 `/v2`는 기존 서비스의 인증 경계를 사용하며 별도 사용자 로그인·테넌트 관리 기능을 추가하지 않습니다.

## 3. 추가된 API

모든 API는 `POST`, 요청 본문은 JSON입니다. 경로 앞에는 **새 버전이 제공되는 것으로 확인된 AI 서비스 주소**를 붙입니다.

| API | 용도 | 핵심 반환값 |
|---|---|---|
| `/v2/briefs/prepare` | 필수 조건·충돌 확인 및 입력 검토 | `status`, `questions`, `reviewed_request`, `reviewed_lines`, `review_id`, `contract` |
| `/v2/formulas/evaluate` | 확인된 입력으로 후보 생성·근거 검사 | `candidates`, `diagnostic_candidates`, `status`, `result_id` |
| `/v2/formulas/reassess` | 확인된 고정 배합을 변경 없이 재평가 | 생성 API와 같은 응답 구조. 기존 고정 처방의 진단 지위 유지 |
| `/v2/formulas/assess-evidence` | 한 처방의 현재 등록 근거·공급 조건만 평가 | `gate_passed`, `blockers`, `materials`, `snapshot_version`, `result_id` |
| `/v2/formulas/change-impact` | 이전 근거 버전과 현재 활성 버전의 영향 비교 | `before`, `after`, `changes`, `review_required`, `contract`, `result_id` |

새 API는 동기 응답을 반환합니다. `/v2` 전용 job 생성·polling·진행률·취소 API는 추가되지 않았습니다. 장시간 호출은 팀 백엔드 작업 큐로 감싸 주세요.

## 4. 생성 요청 연결 순서

### 4.1 입력 준비

`POST /v2/briefs/prepare`

```json
{
  "request": {
    "formula": {
      "brief": "깨끗한 시트러스 우디 향",
      "product_category": "eau_de_parfum",
      "target_region": "EU",
      "product_concentration_percent": 15.0,
      "max_formula_cost_per_kg": 180.0
    },
    "scenario_index": 0
  },
  "evidence_policy": {
    "finished_batch_mass_g": 1000.0,
    "maximum_lead_time_days": 10,
    "maximum_purchase_cost_usd": 100.0
  }
}
```

아래 네 필드는 **요청에 명시적으로 포함**해야 합니다. 기존 `/v1` 기본값이 채워졌다는 이유로 `/v2`에서 확인 완료로 처리하지 않습니다.

| 필드 | 의미·단위 |
|---|---|
| `request.formula.product_category` | 제품군. 위 예시는 향수 `eau_de_parfum` |
| `request.formula.target_region` | 대상 시장: `EU`, `KR`, `US` |
| `request.formula.product_concentration_percent` | 완제품 내 향료 농도, 질량 %, `0 < 값 ≤ 30` |
| `request.formula.max_formula_cost_per_kg` | 향료 농축액 원가 상한. 등록 견적 평가에서는 USD/kg |
| `evidence_policy.finished_batch_mass_g` | 완제품 배치 질량, g |
| `evidence_policy.maximum_lead_time_days` | 허용 최대 납기, 정수 일수 |
| `evidence_policy.maximum_purchase_cost_usd` | MOQ를 포함한 향료 원료 구매 예산, USD |

배치 구매 예산은 완제품 전체 제조비나 베이스·포장·배송비가 아닙니다. 제품군은 기존 엔진의 지원 범위를 따릅니다. 이 API가 향수·로션·바디워시의 전용 모델을 자동으로 같은 방식으로 처리한다고 가정하지 마세요. 제품별 지원 범위는 실제 서비스의 `/v1/ai/capabilities`와 OpenAPI를 기준으로 연결합니다.

### 4.2 준비 응답에 따른 분기

| `status` | 백엔드 처리 |
|---|---|
| `needs_input` | `missing_fields`와 `questions`를 표시하고 필수 값을 받아 다시 prepare |
| `needs_clarification` | `conflicting_fields`, `questions`, `prepared`를 확인해 입력을 수정한 뒤 다시 prepare |
| `unsupported_requirements` | `prepared.unsupported_requirements`를 표시. 요청을 지원 가능한 조건으로 수정 |
| `ready` | 구조화 결과·적용 조건을 사용자에게 보여주고 확인 기록을 저장 |

최상위 `questions`에는 새 필수 조건 질문과 기존 파서의 보완 질문이 함께 포함될 수 있습니다. 질문 ID 경로가 모두 같다고 가정하지 마세요. 새 필수 조건은 원래 요청을 수정해 다시 prepare합니다. 기존 문장·예산 보완은 기존 `/v1/briefs/clarify`를 사용할 수 있으며, 그 결과로 `/v2/briefs/prepare`를 다시 호출해야 합니다.

확인 화면에는 다음을 제공해 주세요.

- `reviewed_request`: 정규화된 입력과 적용 제약
- `prepared.intent`: 목표·제외 향, 시간대별 목표 등 구조화 결과
- `prepared.scenarios`: 농도와 원가 상한 등 계산 조건
- `evidence_policy`: 배치량·납기·구매 예산
- 고정 처방이면 `reviewed_lines`
- `prepared.interpretation`: 해석 신뢰도와 OOD·불확실성의 한계

`ready`는 **입력 검토가 가능한 상태**입니다. 처방 생성 성공, 안전 통과 또는 제조 승인을 뜻하지 않습니다.

### 4.3 사용자 확인 후 생성

동일 요청 본문에 `confirmed_review_id`를 추가해 `/v2/formulas/evaluate`에 전송합니다.

```json
{
  "request": {
    "formula": {
      "brief": "깨끗한 시트러스 우디 향",
      "product_category": "eau_de_parfum",
      "target_region": "EU",
      "product_concentration_percent": 15.0,
      "max_formula_cost_per_kg": 180.0
    },
    "scenario_index": 0
  },
  "evidence_policy": {
    "finished_batch_mass_g": 1000.0,
    "maximum_lead_time_days": 10,
    "maximum_purchase_cost_usd": 100.0
  },
  "confirmed_review_id": "<prepare가 반환한 실제 review_id로 교체>"
}
```

위 확인 ID는 설명용 자리표시자입니다. 실제 값은 AI 응답의 64자리 소문자 16진수 문자열을 그대로 사용해야 합니다. 백엔드에서 자체 계산하거나 임의 생성하지 않습니다.

확인 ID는 입력, 선택 시나리오, 고정 배합, 모델·카탈로그·근거 연결 정보와 AI 프로세스의 평가 날짜에 묶입니다. 수정·버전 변경·날짜 변경으로 불일치하면 `409`가 발생합니다. AI 서버의 `contract.evaluation_date`를 저장하고, 서버 날짜가 한국 시간과 같다고 가정하지 마세요.

**`review_id` 자체는 사용자 인증 토큰이나 승인 서명이 아닙니다.** 해당 내용을 누가 언제 확인했는지는 백엔드가 별도로 검증·저장해야 합니다. 생성 요청 전에 확인한 원본 요청을 불변 버전으로 고정하는 방식을 권장합니다.

### 4.4 결과 판정

| 결과 | 의미·처리 |
|---|---|
| `status = ready_for_review` | 검토 가능한 추천 후보가 있음. 전문가의 실험·제조 승인과는 별도 |
| `status = abstained` | 추천 가능한 후보 없음. 사유와 진단 정보를 표시 |
| `candidates` | 현재 게이트를 통과한 후보 목록 |
| `diagnostic_candidates` | 차단된 후보의 연구 진단. 추천 목록으로 합치지 않음 |
| 후보 `recommendation_allowed` | 추천 허용 여부. 최상위 상태·목록·게이트와 함께 확인 |
| `state_changed = false` | AI가 프로젝트나 후보의 영구 상태를 변경하지 않았음 |
| `manufacturing_approval = false` | 제조 승인을 부여하지 않았음 |

후보의 `rd_gates`에는 `registered_evidence`, `existing_safety`, `scientific_domain`, `profile_and_persistence`가 들어갑니다. 네 조건을 모두 통과해야 추천 후보가 됩니다. `HTTP 200`만으로 승인하거나 작업의 사업적 성공을 판정하지 마세요.

진단 후보의 중첩 `result.recipe`나 `result.closest_candidate`에 배합이 있더라도 최상위 기권 판정을 덮어쓰면 안 됩니다.

## 5. 고정 처방 재평가

기존 또는 수동 편집한 처방을 평가할 때는 prepare 요청의 **최상위**에 `lines`를 추가합니다. 확인이 끝나면 같은 `lines`와 `confirmed_review_id`를 `/v2/formulas/reassess`에 전송합니다.

```json
[
  {"ingredient_id": "dihydromyrcenol", "concentrate_percent": 25.0},
  {"ingredient_id": "hedione", "concentrate_percent": 35.0},
  {"ingredient_id": "linalyl_acetate", "concentrate_percent": 20.0},
  {"ingredient_id": "phenethyl_alcohol", "concentrate_percent": 20.0}
]
```

위 목록은 요청 형식 예시이며 생산 처방 권고가 아닙니다. 원료 ID는 현재 카탈로그 값이어야 하고 배합비는 향료 농축액 기준 질량 %입니다.

- ID 중복은 금지합니다. 합계 허용 오차는 기존 계약의 약 0.001%p이며 임의 정규화하지 않습니다.
- 고정 처방 엔진은 소수점 네 자리까지의 배합비를 검사하며 원료별 한도·제외 조건·안전 게이트를 유지합니다.
- 확인한 배합을 수정하면 다시 prepare하고 확인해야 합니다.
- `lines`가 있는 요청을 `/v2/formulas/evaluate`에 보내면 `422`입니다.
- `/reassess`는 배합 최적화나 승인 기능이 아닙니다. 현재 고정 처방 경로는 `recipe=[]`, `closest_candidate`와 `full_generator_approval=false`를 사용하는 진단 결과를 유지하므로, 계산 완료 후에도 추천 후보로 승격되지 않을 수 있습니다.

## 6. 처방 근거 평가·변경 영향

### 6.1 현재 근거만 평가

`POST /v2/formulas/assess-evidence`

```json
{
  "lines": [
    {"ingredient_id": "dihydromyrcenol", "concentrate_percent": 25.0},
    {"ingredient_id": "hedione", "concentrate_percent": 35.0},
    {"ingredient_id": "linalyl_acetate", "concentrate_percent": 20.0},
    {"ingredient_id": "phenethyl_alcohol", "concentrate_percent": 20.0}
  ],
  "target_region": "EU",
  "product_category": "eau_de_parfum",
  "product_concentration_percent": 15.0,
  "max_formula_cost_per_kg": 180.0,
  "max_ingredient_price_per_kg": 300.0,
  "policy": {
    "finished_batch_mass_g": 1000.0,
    "maximum_lead_time_days": 10,
    "maximum_purchase_cost_usd": 100.0
  }
}
```

이 API에서는 정책 키가 `policy`입니다. 생성·준비 API의 `evidence_policy`와 구분해 주세요. 이 평가 자체에는 사용자 확인 ID가 필요하지 않습니다. 원료 단가 상한은 생략 시 300 USD/kg입니다.

정상 평가 응답은 `status = supported_by_registered_evidence` 또는 `blocked`, `gate_passed`, 원료별 `materials`, `blockers`, `quoted_formula_cost_usd_per_kg`, `purchase_cost_usd`, `snapshot_version`, `evidence_contract`, `evaluated_on`, `input_id`, `result_id`를 포함합니다. 증빙이 없으면 `HTTP 200`의 `blocked` 결과가 나올 수 있습니다. **근거 평가 통과만으로 기존 안전·과학·프로필 게이트 통과를 의미하지 않습니다.**

### 6.2 변경 전후 비교

위 요청에 `previous_evidence_version`을 추가해 `POST /v2/formulas/change-impact`로 보냅니다. 이전 값은 서버에 보존된 근거 스냅샷 버전이어야 합니다. 변경 후 버전은 서버의 현재 활성 버전으로 선택됩니다.

```json
{
  "previous_evidence_version": "<서버에 등록된 이전 근거 버전>"
}
```

위 객체만 보내는 것이 아니라 6.1의 전체 본문에 해당 필드를 추가해야 합니다.

응답에는 이전/현재 평가 `before`, `after`, 원료별 변경 필드와 이전/현재 값 `changes`, `affected_material_count`, `review_required`, `result_id`, `contract`가 들어갑니다. 활성 버전과 같은 이전 버전, 알 수 없는 버전, 미래 버전 등은 거부합니다.

백엔드 처리 순서:

1. 근거 버전 변경 이벤트를 기록합니다.
2. 변경 원료를 포함하는 처방 버전·후보·프로젝트를 DB에서 찾습니다.
3. 후보별 고정 처방과 조건으로 change-impact를 호출합니다.
4. 결과와 버전을 저장하고, `review_required=true`이면 백엔드 정책에 따라 재검토 상태와 알림을 기록합니다.
5. 이후 사용자의 재평가·보류·기권 결정을 권한과 사유를 포함해 저장합니다.

비교는 **같은 처방·같은 조건을 오늘 기준으로 두 근거 버전에서 평가**합니다. 과거 평가 시점의 상태를 재현하는 감사 API가 아닙니다. 현재 근거가 불충분하면 변경 필드가 없어도 재검토가 필요할 수 있습니다. 전체 프로젝트 탐색·알림 발송은 AI가 하지 않습니다.

## 7. 오류·재시도 계약

현재 오류 응답의 `detail`은 문자열, 구조화 객체 또는 FastAPI 검증 오류 배열일 수 있습니다. 단일 문자열로 고정해 파싱하지 마세요.

| HTTP / 응답 | 처리 |
|---|---|
| `200` + prepare의 미완료 상태 | 사용자 보완 후 다시 prepare |
| `200` + `abstained` 또는 근거 평가 `blocked` | 작업 응답을 저장하고 차단 사유 표시. 무조건 재시도하지 않음 |
| `409` | 확인한 입력·모델·근거·날짜 불일치. 새 prepare 및 사용자 재확인 |
| `422` + 필드 오류·충돌·미지원 조건 | 입력을 수정. 생성 재시도로 해결하지 않음 |
| `422` + 근거 누락·만료·해시 오류 | 증빙 등록·운영 설정 확인. 증빙을 우회하거나 기존 API로 자동 전환하지 않음 |
| `429` 또는 `503` | 기존 제한·추론 혼잡 정책에 따라 제한된 재시도. 제공된 `Retry-After`가 있으면 준수 |
| 인증 실패·연결 오류·그 밖의 서버 오류 | 기존 서비스 인증/장애 정책으로 처리하고 요청·시도 이력 저장 |

실제로 확인한 근거 미등록 생성 응답:

```json
{
  "detail": {
    "status": "abstained",
    "reason": "registered_regulatory_and_supply_evidence_missing",
    "candidates": []
  }
}
```

`review_id`와 `result_id`는 백엔드의 멱등성 키나 작업 ID를 대신하지 않습니다. 같은 호출의 중복 계산 방지, 재시도 횟수 제한, 대기 작업 취소와 늦게 도착한 결과의 반영 여부는 백엔드가 관리해야 합니다. HTTP 연결 취소가 이미 시작한 AI 계산의 취소를 보장하지는 않습니다. 호출 타임아웃은 연결 환경의 측정 결과에 맞춰 정하고 작업 상태 조회로 사용자에게 안내해 주세요.

## 8. 백엔드 저장·비교 규칙

다음 항목을 불변 평가 기록으로 묶어 저장하는 것을 권장합니다. 이는 AI가 제공하는 DB 스키마가 아니라 백엔드 저장 설계 제안입니다.

| 기록 | 저장할 내용 |
|---|---|
| 업무 연결 | 조직·프로젝트·사용자 ID, 백엔드 작업 ID, 후보 영구 ID, 처방 버전 |
| 요청 | 입력 원문, 실제 전송 JSON, 선택 시나리오, 배치·가격·공급 정책 |
| 사용자 확인 | `review_id`, 확인한 요청·처방 버전, 확인자와 확인 시각 |
| AI 결과 | 엔드포인트, HTTP 상태, 원문 JSON, `schema_version`, `status`, 오류·기권 사유 |
| 버전·근거 | `result_id`, 존재하는 `request_id`/`input_id`, `contract`, `evidence_contract`, `snapshot_version`, `evaluated_on` |
| 분석·판정 | 후보별 `rd_gates`, 추천 허용 여부, 근거 평가, 불확실성·적용범위, 백엔드 후속 결정 |

엔드포인트별 반환 필드는 다릅니다. 예를 들어 assess-evidence는 모델 추론을 하지 않으며 생성 API와 같은 `contract` 대신 `evidence_contract`를 반환합니다. 없는 버전 값을 임의로 채우지 말고 실제 응답을 보관합니다.

**중요: `/v2` 결과는 기존 `/v1/formulas/compare`의 임시 후보 캐시에 저장되지 않습니다.** 새 `candidate_id`를 그대로 기존 compare에 넘기지 마세요. 백엔드에 저장한 결과로 비교표를 구성하고, 후보의 모델·카탈로그·근거·평가 조건이 서로 비교 가능한지 확인해야 합니다. 서버의 `candidate_id`는 내용 기반 식별값이며 프로젝트 소속이나 접근권한을 증명하지 않습니다.

## 9. 규제·공급 근거 파일 등록

AI 서비스 운영자가 검토된 JSON 파일을 아래 환경 변수 **쌍**으로 지정하고 새 서비스 인스턴스를 시작해야 합니다.

| 환경 변수 | 값 |
|---|---|
| `PERFUMERY_AI_RD_EVIDENCE_PATH` | AI 프로세스에서 읽을 수 있는 근거 JSON 파일 경로 |
| `PERFUMERY_AI_RD_EVIDENCE_SHA256` | 해당 파일 실제 바이트의 SHA-256, 소문자 64자리 |

일반 AI 요청으로 근거 레코드를 업로드하거나 승인 상태를 전달하는 API는 없습니다. 브라우저가 보내는 `approved=true` 같은 값을 규제 근거로 사용하지 않습니다. 원문 문서는 검토자가 확인하고, 운영자가 그 검토 결과와 견적을 등록해야 합니다.

빈 파일의 구조 예시 — 실제 근거를 포함하지 않으며 후보 통과에 사용할 수 없음:

```json
{
  "schema_version": "rd-evidence-1",
  "active_version": "reviewed-2026-09-11-001",
  "snapshots": [
    {
      "version": "reviewed-2026-09-11-001",
      "effective_on": "2026-09-11",
      "records": []
    }
  ]
}
```

각 `records` 항목의 필수 필드:

| 구분 | 필드 |
|---|---|
| 대상 | `ingredient_id`, `cas_number`, `region`, `product_category` |
| 검토 근거 | `source_reference`, `document_sha256`, `reviewer`, `reviewed_on`, `valid_until`, `rule_version` |
| 검토 내용 | `frameworks`, `maximum_finished_product_percent` |
| 공급 품목 | `supplier`, `sku`, `quote_reference`, `quote_document_sha256` |
| 견적·재고 | `quoted_on`, `quote_valid_until`, `currency`, `price_per_kg`, `available_kg`, `minimum_order_kg`, `lead_time_days` |

현재 구현의 규칙:

- 날짜는 `YYYY-MM-DD`, 문서 해시는 소문자 SHA-256, 통화는 `USD`만 지원합니다.
- `frameworks`는 프레임워크 이름을 키로 하고 `supported`, `restricted`, `prohibited`, `unknown` 중 하나를 값으로 갖습니다.
- 필수 검토 키는 EU: `IFRA`+`EU_REACH`, KR: `IFRA`+`K_REACH`, US: `IFRA`+`FDA`입니다. 이는 코드의 검토 범위 구분이며 법적 인증을 의미하지 않습니다.
- 원료 ID·CAS·시장·제품 범위가 일치해야 합니다. `supported/restricted`라도 등록 사용 한도를 넘으면 차단합니다.
- 재고·납기는 등록된 공급사 기록 기준입니다. 구매량은 `max(배치 필요량, MOQ)`이며 이 구매량으로 재고·구매 예산을 검사합니다.
- 파일 최대 크기는 16 MiB입니다. 버전과 원료/시장/제품별 항목의 중복을 금지합니다. 이전 버전은 영향 비교에 사용할 수 있도록 보존해야 합니다.
- 파일 해시는 로딩·평가 시 검사합니다. 실행 중 같은 파일을 덮어쓰면 해시 불일치로 거부될 수 있습니다. 새 파일·해시·서비스 인스턴스로 전환하고 해당 버전을 백엔드에 기록합니다.

문서 해시는 증빙 식별값입니다. AI가 문서 원본을 자동 취득하거나 진위를 검증했다는 뜻은 아닙니다. 검토자가 원료의 실제 공급 형태·희석도·적용 조건과 동등한 증빙인지 확인해야 합니다. 예제·테스트용 가상 근거를 운영 기록으로 등록하지 마세요.

## 10. 해석할 때 지켜야 할 현재 한계

1. `interpretation.confidence.value`는 파서 휴리스틱입니다. `0.96`을 인간 후각 정확도 96%나 문장 해석 성공확률로 표시하지 않습니다.
2. OOD는 알려진 미지원 표현 검사 범위입니다. `no_known_vocabulary_violation`이 전체 문장 이해를 보증하지 않습니다. 보정된 OOD 확률·자연어 오류 신뢰구간은 없으며 null/false로 표시합니다.
3. 농도·시간축 프록시와 그 불확실성은 실제 관능 검증이나 제조·판매 승인과 다릅니다.
4. 등록 견적·재고는 **생성 후보의 통과 판정**에 적용됩니다. 최적화 탐색 자체는 기존 카탈로그 가격을 사용합니다. 새 견적 기준 최적해 탐색·실시간 공급사 재고 조회는 추가되지 않았습니다.
5. 근거 평가 통과·`ready_for_review`·HTTP 성공을 실험 승인이나 제조 승인으로 바꾸지 않습니다. 승인과 상태 전이는 백엔드의 별도 권한·정책을 거칩니다.

## 11. 검증 결과와 인수 기준

기록된 로컬 검증 결과:

- 신규 테스트 18개 통과.
- 기존 기능을 포함한 서로 다른 93개 테스트 중 92개 통과, 기존 실패 1개. 처음 의존성 누락으로 건너뛴 3개는 별도 환경에서 모두 실행했습니다.
- 기존 실패는 `test_actual_api_fixed_formula_round_trip_and_comparison`이 특정 요청에 `target_met`를 기대하는 부분입니다. 이전 V63과 새 코드 모두 동일 처방 ID·배합·68.46028308034408점을 반환해 `candidate_only`였습니다. 기준이나 테스트를 완화하지 않았습니다.
- 실제 로컬 앱에서 새 5개 경로와 OpenAPI 생성, 기존 prepare 호환성, 근거 누락 차단을 확인했습니다. 실제 수치 엔진을 호출하는 고정 배합 재평가도 검증했습니다.
- 원료 29,259개의 모든 필드와 기존 모델 선택을 보존하고 새 소스·패키지 연결만 갱신했습니다.

백엔드 연결 후 필요한 인수 확인:

| 확인 상황 | 기대 동작 |
|---|---|
| 필수 제품·시장·농도·원가 누락 | 보완 표시, 생성 진행 차단 |
| 문장의 조건과 구조화 값 충돌 | 보완 후 새 prepare |
| 확인 후 입력·배합·버전 변경 | 이전 확인 ID 사용 시 거부, 재확인 |
| 근거 미등록·누락·만료 | 추천 실패/기권과 구체적 사유 저장 |
| 안전·프로필·적용범위 게이트 실패 | 진단을 추천 후보와 분리 |
| 기존 처방 편집 | 새 처방 버전 생성 후 고정 배합 재평가 |
| 원료·공급 근거 버전 변경 | 영향 후보 탐색, 비교 결과 저장, 재검토·알림 |
| 작업 실패·재시도·취소 | 시도 이력 저장, 중복 반영 방지, 늦은 응답 처리 |
| 후보 비교·승인 | 저장한 버전·근거와 권한 확인. AI 상태만으로 자동 승인하지 않음 |

위 인수 확인은 전달 대상 백엔드에서 수행할 작업이며, 이번에 해당 백엔드·브라우저 화면의 연결까지 검증했다는 뜻은 아닙니다.

## 12. 전달 파일·코드 위치

다음은 저장소 루트 `WantedAI_Ai-dev-clean` 기준 경로입니다.

| 항목 | 경로 |
|---|---|
| 이 전달 문서 | `BACKEND_HANDOFF_AI_V2_2026-09-11.md` |
| 기존 상세 API 설명 | `RD_API_V2.md` |
| v2 API 구현 | `fragrance_ai/platform/rd_api.py` |
| 근거 스키마·평가 구현 | `fragrance_ai/platform/rd_evidence.py` |
| 기존 API 연결부 | `fragrance_ai/platform/ai_extensions.py` |
| 신규 테스트 | `tests/test_rd_contract.py` |
| 로컬 패키지 | `dist/rd-api-v2/candidate-01/wheel/perfumery_ai_core-1.4.0-py3-none-any.whl` |
| 로컬 카탈로그 연결 | `dist/rd-api-v2/candidate-01/catalog/catalog_manifest.json` |
| 검증 요약 | `dist/rd-api-v2/candidate-01/verification.json` |
| 실제 로컬 앱 응답 | `dist/rd-api-v2/candidate-01/local-api-smoke.json` |
| 전체 작업 결과 | `dist/rd-api-v2/candidate-01/RESULT.md` |

wheel SHA-256:

```text
b00a82f82172c35176054cc77beb73e7899afca4adc46d1387794bd615c49427
```

모델·카탈로그 연결은 로컬 프로필과 별도 고정 아티팩트에 의존합니다. wheel 하나를 설치하는 것만으로 동일 AI 서비스를 재현하거나 외부 배포가 완료되지는 않습니다. 서비스 담당자가 실행 환경·연결 주소·인증 방식·활성 근거 버전을 준비한 뒤 백엔드에 전달해야 합니다.
