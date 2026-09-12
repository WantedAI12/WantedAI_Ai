# 명세서 AI 요구사항 보완 API

2026-09-10. 기존 `/v1` 계약은 유지한다. 명세서의 입력 확인·외부 근거 게이트가 필요한 팀 백엔드는 `/v2`를 사용한다. 기존 서비스의 인증 경계를 그대로 사용하며, AI는 프로젝트 승인·기권 상태를 저장하거나 변경하지 않는다.

## 호출 순서

1. `POST /v2/briefs/prepare`에 아래 요청을 보낸다.
2. `missing_fields`, `conflicting_fields`, `questions`를 해결해 다시 prepare한다. 기존 문장 보완은 `/v1/briefs/clarify`도 사용할 수 있다.
3. 사용자가 `reviewed_request`, `prepared.intent`, `prepared.scenarios`, `evidence_policy`를 확인하면 팀 백엔드가 `review_id`를 확인 기록과 함께 보관한다.
4. 동일 입력에 `confirmed_review_id`를 추가해 `POST /v2/formulas/evaluate`를 호출한다. 입력·선택 시나리오·모델·근거·평가 날짜가 달라지면 새 확인이 필요하다. 확인 ID는 사용자 신원 인증이나 승인 서명이 아니다. 실제 사용자 확인·권한 기록은 팀 백엔드 책임이다.
5. 추천 가능한 처방은 `candidates`에만 반환된다. `diagnostic_candidates`는 차단된 연구 진단으로, 추천·실험 승인에 사용하면 안 된다.

```json
{
  "request": {
    "formula": {
      "brief": "깨끗한 시트러스 우디 향",
      "product_category": "eau_de_parfum",
      "target_region": "EU",
      "product_concentration_percent": 15.0,
      "max_formula_cost_per_kg": 180.0
    }
  },
  "evidence_policy": {
    "finished_batch_mass_g": 1000.0,
    "maximum_lead_time_days": 10,
    "maximum_purchase_cost_usd": 100.0
  }
}
```

제품군·시장·농도·농축액 원가 상한은 반드시 명시한다. 제품군은 기존 수치 엔진의 지원 범위를 따른다. 향수용 계약이 로션·바디워시 전용 모델을 대신하지 않는다. 고정 배합은 prepare에 `lines: [{"ingredient_id": "...", "concentrate_percent": 25.0}, ...]`를 추가하고, 같은 배합과 확인 ID를 `/v2/formulas/reassess`에 보낸다. 배합 합계는 100이어야 하고 확인 후 배합이 바뀌면 거부한다.

`/v1/briefs/prepare`에도 `interpretation`과 `parsed_conditions`가 추가된다. 신뢰도는 기존 파서의 휴리스틱이며 보정된 확률이 아니다. OOD는 알려진 미지원 표현 검사 범위만 설명한다. 전체 문장 이해 검증이나 통계적으로 보정된 OOD 확률·신뢰구간이 없으면 각각 false/null로 표시한다. 기존 과학 프록시의 실제 관능 오차 보정은 이번 변경에 포함되지 않는다.

## 규제·공급 근거 연결

서버 운영자가 검토한 JSON 파일을 다음 환경 변수 **쌍**으로 지정하고 서비스를 새로 시작한다. HTTP 요청은 근거 파일 경로·승인 레코드를 받지 않는다.

```powershell
$env:PERFUMERY_AI_RD_EVIDENCE_PATH = 'C:\approved-data\rd-evidence.json'
$env:PERFUMERY_AI_RD_EVIDENCE_SHA256 = (Get-FileHash -LiteralPath $env:PERFUMERY_AI_RD_EVIDENCE_PATH -Algorithm SHA256).Hash.ToLowerInvariant()
```

파일 구조는 `schema_version: "rd-evidence-1"`, `active_version`, `snapshots`이다. 각 스냅샷은 `version`, `effective_on`, `records`를 가진다. 활성 버전은 반드시 존재해야 하며 미래 버전·중복 버전·중복 원료/시장/제품 항목은 거부한다. 스냅샷 변경은 새 파일·해시와 새 서비스 인스턴스로 반영한다. 실행 중 파일 변경은 오류가 된다.

레코드 필드와 형식은 `fragrance_ai/platform/rd_evidence.py`의 `EvidenceRecord`와 API OpenAPI 스키마의 평가 입력을 따른다.

| 구분 | 필수 레코드 필드 |
|---|---|
| 적용 대상 | `ingredient_id`, `cas_number`, `region`, `product_category` |
| 검토 근거 | `source_reference`, `document_sha256`, `reviewer`, `reviewed_on`, `valid_until`, `rule_version` |
| 검토 판단 | `frameworks`, `maximum_finished_product_percent` |
| 공급 식별 | `supplier`, `sku`, `quote_reference`, `quote_document_sha256` |
| 견적·공급 | `quoted_on`, `quote_valid_until`, `currency`(USD), `price_per_kg`, `available_kg`, `minimum_order_kg`, `lead_time_days` |

`frameworks` 값은 `supported`, `restricted`, `prohibited`, `unknown` 중 하나다. EU는 IFRA+EU_REACH, KR은 IFRA+K_REACH, US는 IFRA+FDA 검토 근거를 요구한다. `supported/restricted`라도 완제품 투입률이 등록된 한도를 넘으면 차단한다. FDA 등 이름은 검토 범위를 식별하며 기관 인증을 의미하지 않는다.

문서 해시는 등록된 검토 근거 식별값이다. 서비스가 원본 문서를 취득하거나 문서 내용·진위를 자동 검증했다는 뜻은 아니다. 실제 원료와 동등한 품목·공급 조건인지 운영자가 검토해야 한다. 별도 실측·공급사 증빙이 없는 테스트 fixture를 운영 근거로 등록하지 않는다.

평가는 등록 근거의 CAS·시장·제품 범위, 검토/견적 날짜, 규제 한도, 원료 단가, 배치 필요량, MOQ, 기록된 재고, 납기와 구매 예산을 검사한다. 구매량은 `max(필요량, MOQ)`이다. 가격은 USD/kg, 배합비는 공급 형태 원료의 질량 기준이다. 기존 안전·프로필·과학 적용범위 게이트도 모두 통과해야 `ready_for_review`가 된다. 고정 처방 재평가는 기존 엔진이 부여하는 진단 지위를 유지한다.

카탈로그 기반 탐색 뒤 등록 견적·공급 조건으로 재검사하는 방식이다. 이번 변경은 새 견적을 최적화 엔진의 원료 가격으로 교체하거나 실시간 공급사 재고를 조회하지 않는다. 새 견적에서 탈락한 후보가 있으면 백엔드/사용자가 제약·후보를 조정해 재요청한다. 근거가 없으면 평가를 완료로 표시하거나 자동 승인하지 않는다.

## 처방 근거 평가와 변경 영향

`POST /v2/formulas/assess-evidence` 입력:

- `lines`: 원료 ID와 `concentrate_percent` 목록
- `target_region`, `product_category`, `product_concentration_percent`
- `max_formula_cost_per_kg`, 선택적인 `max_ingredient_price_per_kg`(기본 300 USD/kg)
- `policy`: 위 `evidence_policy`와 같은 구조

`POST /v2/formulas/change-impact`는 여기에 `previous_evidence_version`을 추가한다. 과거 등록 버전과 현재 활성 버전에서 **동일 처방·조건을 오늘 기준으로** 검사해 `before`, `after`, 변경 필드의 이전/현재 값, 차단 사유, `review_required`를 반환한다. 당시 상태를 재현하는 과거 시점 감사가 아니다. 근거 누락·현재 부적합이면 변경값이 없어도 검토 필요로 반환한다. 원료 삭제, 식별 변경, 견적 변경도 비교 대상이다.

다수 후보·프로젝트 영향 탐색, 영구 감사 저장, 알림, 재검토·보류·기권 상태 전이는 팀 백엔드에서 처리한다. 반환된 `result_id`, `review_id`, 모델·카탈로그·근거 버전과 원문 응답을 함께 저장한다.

## 검증

`tests/test_rd_contract.py`는 명시 입력, 확인 후 변경, 기존 안전/OOD 차단, 기한·범위·CAS·단가·재고·MOQ·납기, 변경 전후 비교, 해시 변경·배합비·요청 스키마 검증을 다룬다. 실제 수치 엔진을 호출하는 고정 처방 경로도 포함한다. 테스트 근거는 가상 fixture이며 실제 공급·규제 적합성 증거가 아니다.
