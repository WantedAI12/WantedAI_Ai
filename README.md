# WantedAI Perfumery AI Backend

자연어 향 요청을 안전·가격·가용성 제약을 만족하는 정량 조향식으로 변환하는
CPU 기반 AI 백엔드입니다. `dev` 브랜치는 현재 Modal에 실제 배포된 Perfumery
AI Core 1.4.0을 기준으로 구성했습니다.

## 현재 배포

- Modal API: `https://junseong2im--perfumery-ai-core-web.modal.run`
- API 문서: `https://junseong2im--perfumery-ai-core-web.modal.run/docs`
- Runtime: CPU 1 core / RAM 1 GiB / GPU 없음
- Scaling: min 0 / max 1 / idle 300초 후 scale-to-zero
- 인증: Modal Proxy Token 필수
- Registry: 29,240 molecules connected / 29,259 experimental formula candidates

무인증 요청은 Modal edge에서 `401`로 거부됩니다. Proxy Token은 저장소나
브라우저에 넣지 않고 백엔드 환경변수 또는 Secret Manager로 전달합니다.

```http
Authorization: Bearer wk-<token-id>.ws-<token-secret>
Content-Type: application/json
```

## 조향 요청

```http
POST /v1/formulas
```

```json
{
  "brief": "깨끗하고 시원한 시트러스 우디 향",
  "max_risk_tier": 1,
  "target_region": "EU",
  "product_category": "eau_de_parfum",
  "max_ingredient_price_per_kg": 180,
  "max_ingredients": 12
}
```

성공 응답은 `status=prototype_ready`, 정량 `recipe`, `safety`, 시뮬레이션,
PhysSim, 시간 변화, 제조 계획, 데이터 적용범위와 증거 상태를 포함합니다. 안전한
해가 없으면 억지 배합 대신 `status=no_safe_match`, `recipe=[]`를 반환합니다.

29,240개 전체 레지스트리를 후보 공간으로 사용하려면 다음 필드를 보냅니다. 이
모드에서는 내부 안전 차단을 사용하지 않고 모든 후보를 100% 개별 상한으로 평가한
뒤 최적 처방 1개만 반환합니다.

```json
{
  "max_risk_tier": 2,
  "enable_registry_trace_candidates": true,
  "experimental_disable_safety": true
}
```

## 주요 경로

- `fragrance_ai/`: 자연어 해석·최적화·안전·과학 시뮬레이션 코어
- `deploy/modal_app.py`: 현재 Modal CPU 배포 정의
- `benchmarks/industrial_ingredient_registry_v1.db`: 29,240개 registry
- `dist/full-registry-activation-v2/`: 현재 배포 Wheel과 release manifest
- `tests/`: 배포·안전·API 중심 회귀 테스트
- `MODAL_DEPLOYMENT.md`: 배포와 인증 연동 절차
- `SIGNED_INGREDIENT_PROMOTIONS.md`: 서명 승인 원료 자동 합류 계약

## 로컬 설치

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[commercial,test]"
.\.venv\Scripts\python.exe -m pytest
```

Modal CLI는 프로젝트 Python과 분리하는 것을 권장합니다.

```powershell
python -m venv "$HOME\.modal-cli-venv"
& "$HOME\.modal-cli-venv\Scripts\python.exe" -m pip install modal==1.5.5
$env:PYTHONUTF8='1'
& "$HOME\.modal-cli-venv\Scripts\python.exe" -m modal deploy deploy/modal_app.py
```

## 검증 상태

- 전체 원본 workspace 회귀: 344 passed / 1 PostgreSQL environment skip
- Modal-enabled 배포 테스트: 2/2
- Modal 인증 계약 테스트: PASS
- 원격 무인증: HTTP 401
- 원격 인증 health/catalog/formula: HTTP 200
- 실제 원격 Tier 1 처방: `prototype_ready`, 12 ingredients, registry trace 0
- 실제 원격 확장 처방: `experimental_registry_candidate`, 12 ingredients 중
  registry 10, 최대 22.6559%, 의미 프로필 근접도 99.9954, manufacturing false

## 주장 경계

`prototype_ready`는 연구개발 후보 상태이며 제조·시장 출시 승인이 아닙니다.
시뮬레이션 점수는 실제 인간 후각 정확도를 의미하지 않습니다. Reference archive
등재나 구조 경고 없음만으로 원료를 formula pool에 넣지 않으며, 신규 원료는 실제
supplier·SDS·COA·IFRA·독성·알레르겐·농도·서명 dossier를 통과해야 자동 합류합니다.
29,212개 미연결 분자는 모두 실험 후보로 열리며, descriptor가 없는 분자는 결정론적
의미 프로필을 사용합니다. 89.58% Bushdid 수치는 회고적 paired-mixture 진단이고
자연어 생성 처방에는 적용되지 않으므로 인간 정확도로 표시하지 않습니다.

License: `LicenseRef-Proprietary`. 자세한 범위는 `LICENSE_POLICY.md`를 확인하세요.
