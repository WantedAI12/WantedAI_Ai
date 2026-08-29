# 공급사·로트 데이터 입력

기본 `fragrance_ai/data/supplier_registry.json`에는 공급사 레코드가 없다.
이는 예시 데이터를 공급 근거나 규제 문서로 오인하지 않기 위한 fail-closed
설계다. 실제 운영자는 권한이 있는 공급사·로트 자료를 검증한 뒤 별도 registry를
가져와야 한다.

## 입력 형식

개발용 CSV 형식은 `fragrance_ai/data/supplier_materials_template.csv`에 있다.
이 템플릿은 wheel에 배포하지 않는다. 한 행에는 아래 정보를 입력한다.

- `ingredient_id`, `supplier`, `sku`, `cas_number`, `lot_number`
- `price_per_kg`, `currency`, `moq_kg`, `in_stock`, `lead_time_days`, `regions`
- `density_g_ml`, `active_strength_percent`, `carrier`
- `ifra_amendment`, `ifra_certificate_valid_until`, `sds_valid_until`,
  `coa_available`
- `allergen_statement_valid_until`, `allergen_fractions`

`allergen_fractions`는 원료 내 질량분율 JSON이다. 예를 들어
`{"Linalool": 0.012}`는 1.2%를 뜻한다. 알 수 없는 값을 0으로 넣어서는 안 된다.

```powershell
python scripts\import_supplier_data.py completed_supplier.csv --output supplier_registry.json
```

가져오기 과정은 형식과 카탈로그 원료 ID를 검사한다. 문서의 진위, 계약 조건,
규제 해석 또는 공급사 신원은 이 명령이 확인하지 않는다.

## qualified/commercial 게이트

`SupplierRegistry.assess_offer()`는 요청일 기준으로 대상 지역, 재고, 리드타임,
MOQ, 가격, 농도, 캐리어 및 다음 문서를 검사한다.

- IFRA 51차 증명서와 유효기간
- 최신 SDS
- 로트 COA
- 유효한 정량 알레르겐 성명서와 질량분율
- 밀도 명세

조건을 하나라도 만족하지 못하면 qualified/commercial 후보에서 해당 원료를
제외한다. 프로토타입은 공급 증빙이 없는 제한적 후보 탐색일 수 있으나, 이를
구매 가능·안전·상용 가능으로 표시하지 않는다.

## 상용 release에 필요한 추가 결속

상용 release scope는 registry의 요약 필드만 신뢰하지 않는다. 처방이 선택한
정확한 공급사·SKU·로트와 각 문서의 실제 파일을 요구한다. COA, SDS, IFRA
증명서, 알레르겐 성명서는 SHA-256으로 결속되며 승인과 제조 직전에 재검증된다.
또한 완제품 농도, 제품 범주, 시장, 베이스, 포장, 규칙/데이터/모델 버전이 같은
scope에 포함된다.

파일이 없거나, 만료되었거나, 해시가 달라지거나, 공급 농도·캐리어·로트가
처방과 다르면 release는 차단된다. 이 시스템은 공급사 문서가 없는 상태에서
상용 승인을 만들어 내지 않는다.

공급사 자료의 입력은 내장 IFRA 부분 스크린을 완전한 시장 규제 판정으로
바꾸지 않는다. 전체 규칙팩 적용과 외부 책임자 서명은 별도 절차다. 관련 경계는
[규제 및 데이터 경계](REGULATORY_AND_DATA.md)를 참조한다.
