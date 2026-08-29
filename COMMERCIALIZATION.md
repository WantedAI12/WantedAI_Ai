# 상용화 아키텍처와 운영 경계

## 1. 현재 구현 수준

Perfumery AI Core 1.4는 단순 라이브러리를 넘어 조향 R&D 상용 서비스의 기반 계층을 구현합니다.

| 영역 | 구현 |
|---|---|
| 인증·권한 | OIDC/JWKS JWT 검증, RBAC, 선택적 세부 permission claim, production 정적-token 차단 |
| tenant | 인증 tenant 고정, header mismatch 차단, 프로젝트·처방·버전·작업 tenant predicate |
| 조향 업무 | 프로젝트, 자연어 생성, 상대 자연어 수정, accord, 시각 배합 편집, immutable version, optimistic conflict, diff |
| 비동기 실행 | PostgreSQL durable queue, `SKIP LOCKED`, lease heartbeat, retry, 다중 worker |
| 감사 | 순서화 hash/HMAC chain, PostgreSQL advisory lock, UPDATE/DELETE 거부 trigger |
| 관측성 | JSON log, request ID, health/readiness, Prometheus 요청·추론·job·latency 지표 |
| 배포 | non-root Docker image, PostgreSQL/API/worker/Prometheus Compose, replica scale-out |
| 성능 | 실제 추론과 control plane을 분리한 재현 가능한 ASGI 부하 도구와 버전별 기준선 |
| 모델 공급망 | 체크포인트·descriptor·ensemble hash binding, NumPy/JSON 전용 R2·ridge 추론, runtime wheel의 pickle 계열 모델 파일 제거 |

SQLite와 정적 token은 로컬 개발 전용입니다. `PERFUMERY_AI_ENV=production`에서는 OIDC, PostgreSQL과 32바이트 이상 감사 HMAC 키가 모두 없으면 프로세스가 시작되지 않습니다.

## 2. 처방 작업 흐름

1. 사용자는 OIDC principal의 tenant 범위에서 프로젝트를 만듭니다.
2. 자연어 생성·수정·accord 요청은 durable job으로 저장됩니다.
3. 여러 worker가 lease를 획득하고 heartbeat로 소유권을 갱신하며 AI 추론을 실행합니다.
4. 안전·가격·의미 조건을 통과한 결과만 R&D 처방 또는 새 버전으로 저장됩니다. 독립 정량 기준향이 없으면 후각 비교는 기권으로 저장됩니다.
5. 수동 편집은 원료별 상한과 총량 100%를 서버에서 재검증합니다. 저장 즉시 기존 시뮬레이션·PhysSim·안전 승인 상태를 무효화합니다.
6. 모든 변경은 parent version과 content SHA-256을 가진 불변 버전이 되며, stale parent는 409 conflict로 거부됩니다.
7. 요청·완료·실패·버전 생성은 인증 주체·역할·tenant scope와 함께 감사 chain에 기록됩니다.

상대 자연어 수정은 현재 처방의 `achieved_profile`을 기준으로 “조금 높여”, “줄여”, “유지” 같은 한국어·영어 지시를 명시적 profile multiplier로 변환합니다. 변환 전·후 profile과 multiplier는 `revision_context`에 저장되어 재현할 수 있습니다.

## 3. 보안 경계

- OIDC access token은 HTTPS issuer/JWKS, 서명 알고리즘 allowlist, issuer, audience, 시간 claim, 최대 수명, role, tenant를 모두 통과해야 합니다.
- role이 허용한 권한과 token의 permission claim을 교집합으로 적용하므로 permission claim이 권한을 확대할 수 없습니다.
- bearer token 교체로 rate limit을 초기화할 수 없도록 tenant+subject를 hash한 identity를 사용합니다.
- production rate limit은 PostgreSQL의 atomic upsert를 사용해 API replica 사이에서 공유합니다.
- request body 제한은 `Content-Length`가 없는 chunked 요청에도 적용합니다.
- UI는 bearer token을 서버나 localStorage에 보관하지 않고 현재 탭 sessionStorage만 사용합니다. 실제 배포에서는 조직의 OIDC PKCE/BFF 정책에 맞춰 access token 전달 계층을 구성합니다.
- PostgreSQL 감사 chain은 애플리케이션 변경을 탐지하지만 DB superuser까지 신뢰 경계 안에 넣지는 않습니다. 주기적으로 서명된 head와 snapshot을 독립 WORM 저장소에 내보내야 합니다.

## 4. 과학·규제 경계

시뮬레이션 점수는 후보 순위와 fail-closed 판단을 위한 비인간 프록시입니다. 실제 관능 데이터가 없는 상태에서 실제 후각 90%를 인증할 수 없으며 시스템도 그 값을 생성하지 않습니다.

상용 제조·판매 승인을 위해서는 다음 외부 증거가 release specification에 추가로 연결되어야 합니다.

- 실제 공급 SKU·lot의 COA, SDS, IFRA certificate, 정량 allergen, 재고·lead time
- 목표 시장·제품 category에 고정된 최신 전체 규제 rule pack과 전문가 sign-off
- 실제 base·농도·공정·포장의 pilot batch, 안정성, 용해성·변색·용기 적합성 결과
- 조직 정책상 필요한 인간 관능 또는 소비자 평가 결과
- 검증 가능한 승인자 전자서명, 취소·만료 상태와 재승인 규칙

현재 서비스는 이 외부 증거를 조작해 `manufacturing_ready`로 승격하는 endpoint를 제공하지 않습니다.

## 5. 관측과 용량

`/health/live`는 process 생존, `/health/ready`는 workspace storage 연결을 확인합니다. `/metrics`는 `metrics:read` 권한이 필요하며 요청 수·상태·in-flight, 고정 bucket latency, inference와 job 결과를 제공합니다. 로그는 request ID와 route·status·duration을 JSON으로 출력합니다.

v1.4 wheel 대표 기준선은 실제 추론 30회와 control plane 300회를 같은 도구로 다시 측정합니다. 정확한 실행값과 wheel SHA는 배포 폴더의 `software_load_wheel_v1_4.json`을 기준으로 합니다. 이는 장비별 in-process 기준선이므로 배포 환경에서는 외부 OIDC, PostgreSQL, ingress, 실제 replica 수를 포함해 같은 도구로 다시 측정하고 SLO를 고정해야 합니다.

## 6. 배포 승인 조건

배포 후보는 잠긴 의존성으로 clean build한 뒤 전체 `-W error` 회귀, Ruff, compileall, UI syntax·browser QA, 데이터 감사, wheel 격리 설치, SBOM·release policy, 실제 환경 PostgreSQL migration/backup/restore와 OIDC integration test를 통과해야 합니다.

소프트웨어 release gate 통과는 조향 제조 승인이나 실제 후각 정확도 인증이 아닙니다. 해당 경계는 자동 보고서에도 `commercial_formula_release_approved=false`, `human_olfactory_90_percent_certified=false`로 유지됩니다.
