# Modal CPU 배포

Perfumery AI Core의 현재 백엔드 전용 CPU API는 Modal에 배포되어 있다. Modal
Proxy Token이 없는 요청은 container에 도달하기 전에 거부된다.

- Web/API: `https://junseong2im--perfumery-ai-core-web.modal.run`
- Modal app: `perfumery-ai-core`
- Dashboard: `https://modal.com/apps/junseong2im/main/deployed/perfumery-ai-core`
- 인증: Modal Proxy Token (`wk-...` + `ws-...`)

## 실행 사양

- CPU: 1 physical core
- memory: 1,024 MiB
- GPU: 없음
- minimum containers: 0
- maximum containers: 1
- idle scale-down: 300초
- formula concurrency: container당 1
- formula rate cap: container당 30회/분
- 요청 timeout: 120초

Wheel `f7b79713d4930baee87abdabef1e3a10522e9d1b5c14997068e5f95b65af51d0`와
29,240개 registry `d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765`는
배포 전에 다시 해시 검증되고 immutable image에 복사된다. Python 3.11,
NumPy 2.2.6, FastAPI 0.116.1, cryptography 46.0.3을 고정했다.

## 엔드포인트

- `GET /`: 정적 조향 UI
- `GET /health`: CPU·Wheel·registry identity
- `GET /v1/catalog`: 29,240개 registry와 활성 tier 통계
- `POST /v1/formulas`: 자연어 brief와 제약을 정량 조향식으로 변환
- `GET /docs`: OpenAPI 문서

`POST /v1/formulas`는 알 수 없는 필드를 거부하고 brief 2,000자, risk tier 1~2,
가격·가용성·농도·원료 수, 시장과 제품군을 제한한다. 희귀 원료는 항상 꺼져 있다.
전체 레지스트리 조건부 trace pool은 `max_risk_tier=2`와
`enable_registry_trace_candidates=true`를 함께 보낸 prototype 요청에서만 열린다.
기존 Tier 2 요청은 이 플래그가 없으면 기존 34개 curated pool을 그대로 사용한다.
공개 frontend origin과 로컬 개발 origin만 CORS 허용한다.

백엔드 서버는 다음 중 한 방식으로 인증한다. Secret은 브라우저 JavaScript에 넣지
않고 백엔드 환경변수나 secret manager에 저장한다.

```http
Authorization: Bearer wk-<token-id>.ws-<token-secret>
```

또는:

```http
Modal-Key: wk-<token-id>
Modal-Secret: ws-<token-secret>
```

재배포:

```powershell
$env:PYTHONUTF8='1'
$modalPython = "$HOME\.modal-cli-venv\Scripts\python.exe"
& $modalPython -m modal token info
& $modalPython -m modal deploy deploy/modal_app.py
```

## 실제 원격 검증

2026-09-02 KST에 full-registry activation 배포를 원격 URL에서 확인했다.

- `/health`: HTTP 200, CPU, GPU false, 새 Wheel/registry SHA 일치
- `/v1/catalog`: HTTP 200, registry 29,240 연결, 조건부 trace 637,
  Tier-2 전체 후보 671
- Tier 1 `/v1/formulas`: HTTP 200, `prototype_ready`, 12개 원료,
  registry trace 0개, warm 1.663초
- 명시적 확장 `/v1/formulas`: HTTP 200,
  `experimental_registry_candidate`, 12개 중 registry trace 2개,
  각 최대 0.05%, manufacturing false, warm 1.434초
- `/`: HTTP 200, UI 본문 확인
- 인증 없음: HTTP 401
- Proxy Token 인증: health/catalog/formula HTTP 200
- 원격 검증용 임시 Proxy Token은 검증 직후 삭제

이 검증은 배포·인증·API 동작 증거다. 실제 사람 후각 90%, 실제 공급사 서류 또는 제조
승인을 새로 만들지 않는다.
