# 백엔드 전달용 SSE·감사 보고서 계약

상태: 로컬 구현·검증 완료. 소스와 계약서를 `dev` 변경에 포함하며, 기존 Modal 배포에 적용됐다는 의미는 아니다.
기존 API base URL과 Proxy Token 인증은 변경하지 않는다. 운영 배포는 이번 소스 갱신 범위에 포함하지 않는다.

## 경로와 역할

| 메서드·경로 | 입력 | 응답 |
|---|---|---|
| POST /v1/formulas | 기존 조향 요청 JSON | 기존 조향 결과 JSON, 변경 없음 |
| POST /v1/formulas/stream | 위와 동일한 JSON | 조향 진행·결과 SSE |
| POST /v1/audit-logs/stream | 백엔드가 보유한 후보의 감사·버전 이력 JSON | 감사 이력을 레코드별로 전송하는 SSE |
| POST /v1/reports/audit | 동일한 이력 JSON | 다운로드용 감사·버전 이력 보고서 JSON |
| POST /v1/reports/audit/stream | 동일한 이력 JSON | 보고서 항목별 SSE |

SSE 응답은 `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-store, no-transform`, `X-Accel-Buffering: no`를 사용한다. NDJSON 모드는 제공하지 않는다.

백엔드가 기존 인증 헤더와 `Content-Type: application/json`, `Accept: text/event-stream`을 넣어 POST한다. SSE의 `data:`를 JSON으로 파싱한다. HTTP 청크 경계는 이벤트 경계가 아니므로 UTF-8 디코더 상태를 유지하고 빈 줄까지 모아서 처리한다. `:`로 시작하는 heartbeat 주석은 무시한다.

브라우저의 기본 EventSource는 이 POST body/인증 계약에 직접 사용하지 않는다. 팀 백엔드의 streaming HTTP 클라이언트가 AI와 통신하고 필요하면 프론트용 SSE를 중계한다. Modal 키를 프론트에 노출하지 않는다.

## 1. 조향 진행 SSE

기존 `/v1/formulas`와 같은 요청을 `/v1/formulas/stream`으로 전송한다. `result` 이벤트의 data는 기존 조향 응답 전체이며 별도 wrapper나 감사 이력을 추가하지 않는다.

```text
event: progress
data: {"stage":"INGREDIENT_SCREENING","percent":15,"message":"원료 후보 필터링 중"}

event: result
data: { ...기존 FormulaGenerationResponse 전체... }

event: progress
data: {"stage":"DONE","percent":100,"message":"계산 결과 전송 완료"}

event: done
data: {"status":"completed","cache_status":"miss"}

```

위의 `...`는 설명용이며 실제 응답에는 유효한 JSON만 전달한다. 각 프레임에는 `id: <stream_id>:<sequence>`도 포함된다. `X-Perfumery-Stream-ID` 헤더로 호출별 ID를 확인한다.

### 고정 stage enum

| stage | percent | 실제 발생 지점 |
|---|---:|---|
| RECEIVED | 0 | 스트림 작업 접수 |
| INGREDIENT_SCREENING | 15 | 원료 후보 필터 호출 직전 |
| SAFETY_CHECK | 55 | 원료 안전·가격·공급 제약 필터 완료 후 |
| RATIO_OPTIMIZATION | 80 | 후보 선택·배합 최적화 진입 |
| TEMPORAL_PROFILE | 95 | 모든 탐색 후 최종 시간별 프로필 평가 |
| DONE | 100 | 최종 result 이벤트 전송 후 |

percent는 단계 표시값이다. 시간 예측·연산 완료 비율·품질 점수가 아니다. 재탐색 시 단계가 뒤로 가지 않으며 같은 단계는 반복 발행하지 않는다. 실행되지 않은 단계는 생략할 수 있다.

`SAFETY_CHECK`는 최종 안전 통과나 제조 승인이 아니다. 실제 판정은 result의 `safety`를 사용한다. `DONE` 또한 목표 향 점수 달성을 의미하지 않는다. 목표 미달은 정상 result로 반환하며 기존처럼 `recipe: []`, `closest_candidate`, `full_profile_target_met`를 유지한다.

캐시 hit/shared는 실제 재계산을 수행하지 않으므로 보통 RECEIVED → result → DONE → done만 나온다. 캐시 상태는 최종 done에 전달된다. 서로 다른 요청을 같은 캐시 결과로 처리하지 않는다.

### 오류·종료·재시도

스트림 시작 후 계산 오류는 다음 형태다. 오류 상세에 서버 경로·키·예외 원문은 노출하지 않는다.

```text
event: error
data: {"code":"INFERENCE_BUSY","message":"추론 처리 용량이 부족합니다.","retryable":true,"http_status":503}

event: done
data: {"status":"failed"}

```

| code | 의미 | retryable |
|---|---|---|
| INFERENCE_BUSY | 추론 큐·처리 용량 부족 | true |
| RATE_LIMITED | 요청 한도 초과 | true |
| INVALID_REQUEST_OR_RUNTIME | 요청 조건·런타임 오류 | false |
| INFERENCE_TIMEOUT | 계산 대기 제한 시간 초과 | true |
| STREAM_TIMEOUT | 스트림 제한 285초 초과 | true |
| INFERENCE_FAILED | 그 외 계산 실패 | false |

입력 스키마 오류는 시작 전 HTTP 422, 요청 한도는 HTTP 429, 동시 스트림 용량 부족은 HTTP 503으로 반환할 수 있다. 따라서 HTTP 상태와 SSE error 둘 다 처리한다. HTTP 200만으로 성공 처리하지 않는다.

정상 완료는 `result`와 `done.status == completed`를 모두 받은 경우다. done 없이 EOF가 발생하면 연결 중단으로 취급한다. 기존 withRetry는 같은 요청 body로 처음부터 재호출하며, 중간 프레임부터 복구하지 않는다. `Last-Event-ID`를 보내면 HTTP 409로 거부한다. 재시도 횟수·backoff는 백엔드에서 제한하고, retryable=false인 계산 오류는 자동 반복하지 않는다.

SSE는 계산 중 10초마다 `: keep-alive`를 보낸다. 컨테이너 콜드 스타트 전에는 heartbeat가 나올 수 없다. 중계 서버는 응답 버퍼링/압축을 끄고 read timeout을 heartbeat·콜드 스타트·Modal 제한에 맞춘다.

연결을 닫아도 진행 중 CPU 계산을 강제로 종료하지 않는다. 계산은 기존 안전·캐시 경로로 끝나며, 연결과 계산이 모두 끝날 때까지 동시 실행 슬롯을 유지한다. 동시 조향 스트림은 최대 4개다. SSE로 CPU 실행 병렬도를 늘리지 않는다.

## 2. 감사 로그·보고서 요청 JSON

백엔드는 자기 DB에서 해당 후보의 기록을 조회해 아래 구조로 전송한다. 아래 값은 계약 예시이며 실제 이력이 아니다.

```json
{
  "schema_version": "audit-history-request/v1",
  "formula_id": "formula-01",
  "formula_name": "FORMULA 01",
  "category_filter": "all",
  "history_complete": false,
  "events": [
    {
      "event_id": "event-001",
      "occurred_at": "2026-08-14T10:02:00+09:00",
      "category": "formula",
      "event_type": "safety.evaluated",
      "title": "안전 조건 평가 결과 기록",
      "actor": {"actor_id": "system", "display_name": "system", "kind": "system"},
      "reason": "AI 응답의 안전 평가 상태 수신",
      "changes": [{"field": "safety.status", "label": "평가 상태", "before": null, "after": "pending"}],
      "model_version": "실제 모델 버전",
      "data_version": "백엔드가 관리하는 실제 데이터 버전",
      "reference_ids": ["backend-request-001"]
    }
  ],
  "versions": [
    {
      "version_id": "version-001",
      "version_label": "V1",
      "parent_version_id": null,
      "created_at": "2026-08-14T10:02:00+09:00",
      "actor": {"actor_id": "system", "display_name": "system", "kind": "system"},
      "change_reason": "최초 기록",
      "changes": []
    }
  ]
}
```

입력 규칙:

- events는 필수 배열이며 빈 배열은 허용한다. events 최대 500개, versions 최대 200개, 전체 body 최대 1 MiB.
- category/filter는 candidate(후보), formula(조향식), data(데이터), filter에는 all(전체)을 추가로 허용한다.
- actor.kind는 system/user/service. 실제 사용자 이름·ID는 백엔드가 제공한다.
- 시간은 timezone이 있는 ISO 8601이어야 하며 UTC로 정규화한다. 프론트에서 사용자 timezone으로 표시한다.
- event_id/version_id는 각 배열에서 유일해야 한다. 버전 parent 순환은 거부한다. 제공되지 않은 parent는 경고로 남긴다.
- reason, model_version, data_version은 없으면 null로 유지한다. 변경 값 before/after의 null도 원래 값이 미제공됐음을 표시하며 임의로 '미확인 → 통과'를 만들지 않는다.
- model_sha256/data_sha256도 선택적으로 전달할 수 있다. 64자리 소문자 SHA-256 형식만 확인하며, 전달된 해시와 원본 파일의 일치까지 이 경로가 검증하는 것은 아니다.
- history_complete는 백엔드의 전체 이력 제공 주장이다. 서비스가 DB 완전성이나 승인 증거를 재검증한 표시가 아니다.
- 기존 조향 요청/recipe 전체를 이 이력 API에 그대로 넣지 않는다. 이력 DTO로 매핑해야 한다.

이력은 발생 시간·ID 순으로 정렬한다. category_filter는 audit_log에만 적용하고 version_history는 해당 후보의 제공된 버전을 유지한다. 전체 건수·필터 후 건수를 구분한다.

### 감사 SSE

`POST /v1/audit-logs/stream`의 이벤트 순서:

1. audit.started: 후보·report_id·보고서 생성 시각
2. audit.entry: 감사 기록 한 건씩
3. version.entry: 버전 이력 한 건씩
4. audit.summary: 건수·범위·누락 경고·출처
5. done: status=completed

모든 감사/보고서 SSE data는 아래 공통 envelope를 쓴다. 조향 SSE와는 별도 계약이다.

```json
{
  "schema_version": "audit-sse/v1",
  "stream_id": "호출별 ID",
  "sequence": 1,
  "data": {"event_id": "백엔드 원본 이벤트 ID", "title": "감사 이벤트 내용"}
}
```

실제 audit.entry에는 위 요청의 AuditEntry 전체 필드가 들어간다. 재호출할 때 stream_id는 새 값이며 원본 event_id는 유지된다. 백엔드는 `(formula_id, event_id)`로 중복을 방지한다.

이 스트림은 **이번 요청에서 제공한 이력 스냅샷**을 전달하고 끝난다. 다른 사용자의 미래 변경을 구독하는 이벤트 버스나 영구 감사 저장소가 아니다. 새 이력은 백엔드가 DB에 저장한 후 다시 요청/중계한다. 중단 시 전체 JSON을 다시 보내며 Last-Event-ID 복구는 지원하지 않는다.

### 보고서 다운로드 JSON

`POST /v1/reports/audit`는 `Content-Disposition: attachment; filename="audit-report-<id>.json"`과 다음 구조를 반환한다.

| 필드 | 내용 |
|---|---|
| schema_version | audit-report/v1 |
| report_id | 정렬된 요청 내용의 SHA-256 식별값; 전자서명/감사 체인 아님 |
| generated_at | 이번 보고서 생성 시각 |
| formula | 후보 ID·이름 |
| summary | 전체/표시 건수, 카테고리 건수, 제공된 이력의 시간 범위, 완전성 주장 |
| audit_log | 시간순 감사 기록 |
| version_history | 시간순 버전 기록 |
| provenance | backend_supplied_history, not_independently_verified, llm_calls=0 등 |
| warnings | 일부 이력·모델/데이터 식별값 누락·필터 적용 등의 코드 |

보고서 다운로드 버튼 → 팀 백엔드에서 후보 이력 조회 → 이 경로에 POST → 응답을 JSON 파일로 전달한다. PDF/Excel 렌더링은 이번 API에 포함되지 않는다. 보고서는 사용자 입력을 HTML로 해석하지 않으며, 화면에서도 텍스트로 출력한다.

### 보고서 SSE

`POST /v1/reports/audit/stream`은 report.started의 메타데이터에 이어 report.section 이벤트를 보낸다. 각 section.data는 `{"section":"audit_log","value":[...]}` 형태다. summary/audit_log/version_history/provenance/warnings를 조합하고 마지막 done을 확인하면 JSON 보고서와 같은 내용을 구성할 수 있다. 마지막에 전체 보고서를 중복 전송하지 않는다.

보고서·감사 경로는 조향/LLM/외부 API를 호출하지 않고 기록을 정규화한다. 승인 여부를 추론하거나 화면 예시의 승인 이벤트를 만들어 넣지 않는다. 기록 저장·테넌트 권한·사용자 후보 확정·원본 자료 조회는 팀 백엔드 책임이다. AI는 요청 간 이력을 보관하거나 다른 후보의 이력을 섞지 않는다. 기존 30회/분 요청 제한은 이 경로들과 공유한다.

## 검증과 배포 경계

관련 테스트: tests/test_formula_stream.py, tests/test_audit_reporting_api.py, tests/test_modal_deployment.py의 SSE 통합 테스트.
실제 Uvicorn TCP 연결로 최초 진행 프레임이 계산 종료 전에 전달되는지, heartbeat와 연결 종료를 확인한다. 별도로 실제 조향 엔진의 SSE 결과와 기존 JSON 결과를 비교한다.

엔진 진행 콜백이 추가돼 패키지 소스 해시가 변경됐다. 배포 시 새 wheel과 소스에 맞는 카탈로그 manifest를 정상 절차로 다시 묶어야 한다. 기존 V63의 고정 해시를 무시하거나 새 transport만 구버전 wheel과 섞어 배포하지 않는다. 이 구현만으로 현재 운영 URL에 SSE가 활성화되지는 않는다.

프로토콜 참고: [WHATWG SSE 규격](https://html.spec.whatwg.org/multipage/server-sent-events.html), [Modal streaming endpoints](https://modal.com/docs/guide/streaming-endpoints).
