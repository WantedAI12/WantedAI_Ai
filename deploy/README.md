# 수평 확장 배포

이 구성은 PostgreSQL을 프로젝트·버전·작업 큐·감사 체인의 공통 상태로 사용합니다. 각 API 컨테이너는 단일 Uvicorn 프로세스로 실행하고 컨테이너 수를 늘립니다. worker는 `FOR UPDATE SKIP LOCKED` lease로 같은 작업을 중복 실행하지 않습니다.

API와 worker는 시작 시 idempotent schema를 확인합니다. 작업 처리 중에는 lease heartbeat가 소유권을 연장하며, 만료되거나 다른 worker가 소유한 작업은 완료 처리할 수 없습니다. API replica 간 rate limit도 PostgreSQL atomic upsert로 공유됩니다.

개발 확인:

```powershell
cd deploy
docker compose up --build --scale api=3 --scale worker=4
```

Compose 파일의 토큰, 데이터베이스 비밀번호와 감사 HMAC 키는 개발 전용입니다. `PERFUMERY_AI_ENV=production`에서는 정적 토큰이나 SQLite로 시작할 수 없으며 다음 값이 모두 필요합니다.

- `PERFUMERY_AI_DATABASE_URL=postgresql://...`
- `PERFUMERY_AI_AUTH_MODE=oidc`
- `PERFUMERY_AI_OIDC_ISSUER`, `PERFUMERY_AI_OIDC_AUDIENCE`, `PERFUMERY_AI_OIDC_JWKS_URL`
- 32바이트 이상 키를 URL-safe base64로 인코딩한 `PERFUMERY_AI_AUDIT_HMAC_KEY`

TLS는 ingress 또는 reverse proxy에서 종료합니다. `PERFUMERY_AI_TRUST_PROXY_HEADERS=1`은 신뢰할 수 있는 proxy가 전달 헤더를 덮어쓰는 환경에서만 설정합니다.

운영 ingress는 `/health/live`와 `/health/ready`를 별도로 사용하고 `/metrics`에는 `auditor` 또는 `metrics:read` permission을 가진 서비스 access token을 전달해야 합니다. `/ui/`는 동일 origin API만 호출하므로 CSP의 `connect-src 'self'`를 유지할 수 있습니다.

상용 환경에서는 Compose의 개발 password·token·HMAC key를 사용하지 않습니다. secret manager에서 주입하고, PostgreSQL backup/restore와 point-in-time recovery를 실제로 시험하며, 감사 chain head와 snapshot을 DB 관리자와 분리된 WORM 저장소에 주기적으로 내보냅니다.

## 지속개선 controller

`continual` profile은 immutable 후보 inbox를 5분마다 확인합니다. 외부 인간 관능 결과를 쓰는 prospective challenge, 분자·scaffold·출처 cold 분리, 2,000회 이상 source→target 계층 bootstrap, portable JSON 동등성을 모두 통과한 후보만 shadow champion으로 자동 승격합니다. 합성·시뮬레이션·자기 예측 label은 입력 단계에서 차단됩니다. Outcome 취득은 별도 `external_evidence_acquirer` 서명이 필요하며 production signer와 같은 키를 쓸 수 없습니다.

```powershell
docker compose --profile continual up --build continual
```

production 점수 변경에는 별도의 Ed25519 `model_release_approver` 서명이 필요합니다. 운영 API/worker가 새 champion을 읽게 하려면 같은 read-only registry volume과 다음 두 값을 함께 주입한 뒤 rolling restart합니다. 하나만 설정되거나 registry·감사 체인·artifact·서명이 변조되면 시작이 실패합니다.

- `PERFUMERY_AI_CONTINUAL_STATE=/var/lib/perfumery-ai/continuous/registry.json`
- `PERFUMERY_AI_CONTINUAL_TRUST_ROOT=/run/secrets/perfumery_continual_trust_root.json`

어떤 서명 개인키도 controller 컨테이너에 두지 않습니다. Controller는 후보를 생성·평가할 수 있지만 outcome 취득 증명이나 자기 production 승인을 만들 수 없습니다. API와 worker에는 registry volume이 read-only로 연결되고 continual controller만 write mount를 가집니다.
