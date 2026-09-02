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

Wheel `0cf3beb6d6ae3d8e7b36eda151a029336709c617d8e632f91df1b5f599832c28`와
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
가격·가용성·농도·원료 수, 시장과 제품군을 제한한다. 전체 레지스트리 실험은
`enable_registry_trace_candidates=true`와 `experimental_disable_safety=true`를
함께 보낸 prototype 요청에서 열린다. 후보군은 29,259개 전체이며 개별 상한은
100%다. 최종 레시피 라인은 최대 20개이고 내부 목적 후보 중 최적 1개만 반환한다.
공개 frontend origin과 로컬 개발 origin만 CORS 허용한다.

응답에는 다음 시간 변화 필드가 추가된다. 기존 endpoint와 요청 body는 바뀌지
않았고 모두 additive JSON 필드이므로 응답을 그대로 전달하는 백엔드는 수정할
필요가 없다. 다만 엄격한 response DTO를 사용하는 백엔드는 이 필드를 DTO에
추가해야 화면에서 사용할 수 있다.

- `temporal_timepoints_minutes`: `[0, 15, 60, 240, 480]`
- `temporal_profile`: 시간대, opening/heart/drydown 구간, 오프닝 대비 강도,
  19개 향축 분포와 목표 향 유사도 구간
- `ingredient_temporal_profile`: 원료별 추정 반감기, 시간대별 도포 표면 잔존
  농축액/완제품 농도, 증발량, headspace 기여와 후각 기여
- `temporal_concentration_basis`: 농도 곡선의 계산 기준
- `temporal_model_claim_boundary`: 실측과 시뮬레이션의 경계

농도 곡선은 도포 후 표면에서의 1차 증발 프록시다. 밀폐 향수병의 조성 변화,
GC-MS 실측 또는 사람 관능 결과로 해석하면 안 된다.

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

2026-09-02 KST에 temporal-evolution-v3 배포를 원격 URL에서 확인했다.

- `/health`: HTTP 200, CPU, GPU false, 새 Wheel/registry SHA 일치
- `/v1/catalog`: HTTP 200, registry 29,240 연결, 전체 조향 후보 29,259
- Tier 1 `/v1/formulas`: HTTP 200, `prototype_ready`, 9개 원료
- 시간 변화: 0·15·60·240·480분 5개 점, 점마다 19개 향축
- 원료 변화: 레시피 9개 원료 모두 5개 잔존 농도·headspace·후각 기여점
- 잔존 농도 단조 감소, 각 시간점의 headspace·후각 기여 합 100%
- `/`: HTTP 200, 시간별 향/원료별 잔존 농도 표 확인
- 인증 없음: HTTP 401
- Proxy Token 인증: health/catalog/formula HTTP 200
- 원격 검증용 임시 Proxy Token은 검증 직후 삭제

이 검증은 배포·인증·API 동작 증거다. Bushdid 89.5788% 모델은 paired quantitative
mixture용 회고 진단이고 자연어 요청에는 정량 기준 조성이 없어 적용되지 않는다.
99.9954는 의미 프로필 프록시이며 실제 사람 후각 정확도나 제조 승인이 아니다.
