# 산업 규모 원료 레지스트리 V1.2

Leffingwell, GoodScents, FlavorNet, AromaDB, IFRA 2019 reference archive,
FlavorDB의 고정 원본 13개를 canonical structure로 통합했다. 모든 입력은 기존 R2
source manifest의 SHA-256과 byte size를 다시 검증한다.

## 현재 범위

| 항목 | 수량 |
|---|---:|
| 고유 분자 | **29,240** |
| source membership | 36,327 |
| 이름·별칭 | 60,423 |
| odor descriptor assertion | 71,458 |
| descriptor가 있는 분자 | 28,549 |
| 기존 formulation metadata | 47 |
| 기본 prototype-safe active | 29 |
| Risk-2 조건부 prototype active | 5 |
| 전체 prototype 조향 가능 pool | **34** |
| 분자 구조까지 연결된 formulation material | 28 |
| 전 분자 안전 스크리닝 | **29,240** |
| 증거 승격 큐 | **29,212** |
| 일반 증거보강 대기 | 20,757 |
| 구조 전문검토 추가 필요 | 8,455 |
| 검증된 CAS 연결 | 3,787 |
| checksum 오류 CAS 제외 | 1 |
| IFRA reference 연결 | 1,060 |
| 고우선순위 증거보강 대상 | 856 |

29,240개 모두 구조·분자량·원소·반응성 SMARTS 스크리닝과 증거 요구사항을
가진다. 활성 tier에 연결되지 않은 29,212개는 전부 승격 큐에 들어간다. 공개 archive
등재나 구조 경고 없음은 안전 인증이 아니며, Formula optimizer에 들어가려면 실제
증거 파일과 독립 서명이 검증되어야 한다. 구조 경고 8,455개는 자동 위험 판정이
아니라 별도 전문가 검토가 필요한 fail-closed 분류다.

## 계층

- `reference_only`: 분자·이름·odor descriptor 검색 가능, formula 사용 불가
- `evidence_pending`: 기본 구조 스크리닝 후 필수 증거 대기
- `structural_review_required`: 구조 경고 해소 서명이 추가로 필요한 증거 대기
- `prototype_safe_active`: 현재 prototype 제약에서 사용 가능
- `prototype_conditional_active`: `max_risk_tier>=2`, 원료별 상한과 알레르겐·산화
  검토가 있을 때만 사용 가능
- `formulation_metadata_only`: 내장 조향 메타데이터는 있으나 활성 조건 미충족
- qualified/commercial: 현재 0개, supplier SKU·lot 증거가 들어올 때만 승격

모든 신규 승격에는 identity/CAS/structure, supplier SKU/lot, COA, SDS, IFRA,
시장별 규제 상태와 category rule pack, 독성, 정량 알레르겐, 가격·가용성,
농도 상한, odor profile, 전문가 승인 파일이 필요하다. `prototype_safe_active`에는
`low_risk_signoff`, 구조 경고가 있으면 `structural_review_signoff`가 추가된다.
파일 SHA-256, 정확한 registry DB SHA-256, 분자 구조, 목표 tier, 시장과 제품군을
포함한 dossier 전체는 허용 목록의 toxicologist 또는 regulatory safety officer가
Ed25519로 서명해야 한다.

조건부 활성 5개는 methyl ionone gamma, sweet orange oil, lavender oil, Virginia
cedarwood oil, patchouli oil이다. 기존 카탈로그에 이미 formulation-ready와 사용상한이
있던 고가용 risk-tier-2 원료만 분리했으며, 기본 risk-tier-1 요청에는 들어가지 않는다.

## 사용

```python
from fragrance_ai.recommender.industrial_catalog import IndustrialIngredientRegistry

with IndustrialIngredientRegistry(
    "benchmarks/industrial_ingredient_registry_v1.db"
) as registry:
    print(registry.stats())
    print(registry.search("woody", limit=20))
    print(registry.search("woody", limit=20, formulation_only=True))
    candidate = registry.promotion_candidates(limit=1)[0]
    screen = registry.safety_screening(candidate["registry_id"])
    print(registry.evaluate_safety_promotion(
        candidate["registry_id"], screen.required_evidence
    ))
```

`evaluate_safety_promotion`은 UI/API용 누락 증거 진단이며 원료를 활성화하지 않는다.
실제 승격 결정은 `verify_safety_promotion`에 실제 파일, 시장·제품군 scope,
독립 signer allowlist와 서명 envelope를 넣었을 때만 반환된다. 정책상 차단된 원료는
이 경로로도 승격되지 않는다.

검증 완료 뒤의 조향식 자동 투입도 연결되어 있다. `formulation_spec`을 포함한 승인
패키지를 배포 디렉터리에 넣고 세 환경변수를 지정하면 API와 worker 시작 시 기본
29개 안전+5개 조건부 pool에 자동 병합된다. 사용법과 서명 파일 계약은
[`SIGNED_INGREDIENT_PROMOTIONS.md`](SIGNED_INGREDIENT_PROMOTIONS.md)를 따른다.

재구축:

```powershell
python scripts\build_industrial_ingredient_registry_v1.py
pytest -q tests\test_industrial_catalog.py
```

산출물:

- `benchmarks/industrial_ingredient_registry_v1.db`
- `benchmarks/industrial_ingredient_registry_v1.json`

레지스트리는 workspace research artifact이며 wheel에 포함하지 않는다. Source별
재배포 권리와 실제 공급·규제 증거가 확정된 material만 별도 release scope로
승격한다.
