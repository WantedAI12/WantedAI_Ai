# 서명 승인 원료 자동 투입

안전·공급 증거를 모두 통과한 reference 원료는 소스 코드를 수정하지 않아도
`NaturalLanguagePerfumeryAI`, API 서버, 작업 worker의 조향 후보 pool에 자동으로
합류한다. 서비스 시작 시 승인 패키지를 전부 검증하며 하나라도 변조·누락·만료되면
해당 구성을 조용히 무시하지 않고 시작을 중단한다.

## 실행 설정

다음 세 환경변수를 모두 지정한다. 일부만 지정된 구성은 허용하지 않는다.

```powershell
$env:PERFUMERY_AI_INDUSTRIAL_REGISTRY_DB = "C:\deploy\industrial_ingredient_registry_v1.db"
$env:PERFUMERY_AI_PROMOTION_DIRECTORY = "C:\deploy\approved-ingredients"
$env:PERFUMERY_AI_PROMOTION_TRUST_ROOT = "C:\deploy\promotion-trust-root.json"

perfumery-ai-api --host 0.0.0.0 --port 8000
# 또는
perfumery-ai-worker
```

승인 디렉터리의 각 하위 폴더는 `envelope.json`과 서명에 포함된 실제 증거 파일을
가진다. `envelope.json`의 `artifact_files`는 증거 label과 폴더 내부 상대경로를,
`artifact_hashes`는 같은 label의 SHA-256을 기록한다. 절대경로, 상위폴더 탈출,
symlink, 중복 파일은 거부된다.

```text
approved-ingredients/
  approved-material-001/
    envelope.json
    identity_cas_structure.json
    supplier_sku_lot.pdf
    coa.pdf
    sds.pdf
    ifra_limit_or_certificate.pdf
    regulatory_status.pdf
    market_category_rule_pack.json
    toxicology_assessment.pdf
    quantitative_allergen.json
    price_availability.json
    concentration_cap.json
    odor_profile.json
    formulation_spec.json
    expert_signoff.pdf
    low_risk_signoff.pdf
```

구조 경고가 있는 원료는 `structural_review_signoff`, 안전 tier 승격은
`low_risk_signoff`가 추가로 필요하다. 조건부 tier는 `risk_tier=2`로 활성화되어
사용자가 `max_risk_tier>=2`를 명시해야 처방에 들어간다.

## formulation_spec 계약

`formulation_spec.json`은 다른 증거와 함께 서명되는 기계 판독 입력이다.

```json
{
  "schema": "perfumery-promoted-ingredient-formulation/v1",
  "registry_id": "mol:...",
  "canonical_smiles": "...",
  "target_tier": "prototype_safe_active",
  "market": "EU",
  "product_category": "eau_de_parfum",
  "ingredient_id": "industrial_...",
  "name": "Approved Material",
  "aliases": ["approved material"],
  "pyramid": "heart",
  "profile": {"floral": 1.0, "fresh": 0.3},
  "rarity": "common",
  "odor_impact": 1.2,
  "max_concentrate_percent": 3.0,
  "oxidation_risk": "low",
  "discoloration_risk": "low",
  "shelf_life_months": 24,
  "eu_allergens": ["Linalool"],
  "solubility": ["ethanol"],
  "supplier_material": {
    "supplier": "Approved Supplier",
    "sku": "SKU-001",
    "cas_number": "78-70-6",
    "price_per_kg": 35.0,
    "currency": "USD",
    "moq_kg": 1.0,
    "in_stock": true,
    "lead_time_days": 7,
    "density_g_ml": 0.86,
    "active_strength_percent": 100.0,
    "carrier": null,
    "regions": ["EU"],
    "ifra_amendment": "51",
    "ifra_certificate_valid_until": "2027-12-31",
    "sds_valid_until": "2027-12-31",
    "coa_available": true,
    "allergen_statement_valid_until": "2027-12-31",
    "allergen_fractions": {"Linalool": 0.2},
    "lot_number": "LOT-001"
  }
}
```

자동 활성화는 다음을 재검사한다.

- 정확한 registry DB SHA-256, registry ID, canonical structure
- 허용 signer와 toxicologist/regulatory role의 Ed25519 서명
- 증거 파일 전부의 실제 byte hash, 유효기간과 revocation
- CAS checksum 및 registry CAS 일치
- 시장·제품군 scope와 supplier region
- IFRA 51, SDS, COA, 정량 allergen, 재고, 가격, MOQ, lead time
- 희귀 원료 제외, 향 profile 축, 농도 상한, 안정성·shelf-life
- 기존 원료 ID·이름·별칭과의 충돌

검증된 원료는 시작 시 기본 활성 34개에 병합된다. 이후 자연어 parser, 후보 안전
screen, formula optimizer, 최종 농도·시장·문서 gate에서 일반 원료와 동일하게
사용된다. `catalog_stats.formulation_ready`는 전체 활성 수를,
`catalog_stats.signed_promotions_active`는 자동 추가 수를 반환한다. 승인 폴더를
변경한 뒤에는 모든 API/worker 인스턴스를 재시작해야 새 snapshot이 적용된다.
