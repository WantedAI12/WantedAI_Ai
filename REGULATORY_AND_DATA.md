# 규제 및 데이터 경계

이 문서는 패키지에 포함된 데이터와 자동 검사 범위를 설명한다. 법률 자문,
시장 출시 승인, IFRA 전체 규칙팩 또는 사람 대상 관능 검증을 대체하지 않는다.

## IFRA 및 시장 규제

`fragrance_ai.rules.ifra_rules`에는 일부 원료·제품 범주만 담긴 **IFRA
Amendment 50 내장 부분 스크린**이 있다. 이 표는 프로토타입 단계에서 명백한
초과를 찾기 위한 엔지니어링 검사일 뿐이다.

- 표에 없는 원료 또는 제품 범주는 `uncovered`(알 수 없음)이다. 무제한 또는
  적합 판정이 아니다.
- `apply_ifra_limits()`는 제한 원료를 일괄 재정규화하지 않는다. 남은 허용
  원료의 여유 범위 안에서만 부족분을 재배분하며, 불가능하면 처방을
  `formula_complete=false`로 남긴다.
- `embedded_limits_compliant`는 이 부분 표를 통과했다는 뜻이다.
  `ifra_compliant`와 `check_compliance().overall_compliant`는 부분 표만으로
  참이 되지 않는다.
- 따라서 프로토타입의 안전 상태는 `prototype_partial_screen`이며, 시장 규제
  적합·판매 승인 상태가 아니다.

공급사 자료의 IFRA 증명서 게이트는 별도다. 현재 코드는 qualified/commercial
요청에서 **IFRA 51차 증명서**를 요구하지만, 그 요구가 내장 Amendment 50 표를
51차 전체 규칙팩으로 바꾸지는 않는다.

공식 참고 자료는 [IFRA Standards 문서](https://ifrafragrance.org/initiatives-positions/safe-use-fragrance-science/ifra-standards/ifra-standards-documentation)와
[IFRA Transparency List](https://ifrafragrance.org/transparency-list)다. 프로젝트는
그 목록의 존재만으로 원료의 공급 가능성, 규제 적합성 또는 권리를 주장하지
않는다.

## 승인 단계와 fail-closed 동작

기본 `supplier_registry.json`은 의도적으로 빈 레지스트리다. 따라서 실제
운영자 자료를 가져오기 전에는 qualified/commercial 후보 풀이 비어 있고 요청은
차단된다.

qualified/commercial에 필요한 공급사 레코드는 최소한 다음을 만족해야 한다.

- 원료 ID, 공급사, SKU, CAS, 가격·통화, MOQ, 재고, 리드타임, 대상 지역
- 공급 농도, 캐리어, 밀도
- 현재 유효한 IFRA 51 증명서와 SDS
- 로트 COA, 정량 알레르겐 성명서와 질량분율
- 선택된 공급사·SKU·로트가 처방의 농도와 캐리어에 정확히 일치함

상용 release scope는 위의 레코드만으로도 충분하지 않다. 각 원료의 COA, SDS,
IFRA 증명서, 알레르겐 성명서의 실제 파일 경로를 읽어 SHA-256을 계산하고,
승인 및 제조 직전에 다시 해시한다. 완제품 농도·제품 범주·시장·베이스·포장과
규칙/데이터/모델 버전도 같은 scope에 묶인다. 누락, 만료, 해시 불일치 또는
로트 불일치가 하나라도 있으면 release는 차단된다.

## 데이터 자산과 패키지 정책

각 번들 자산의 출처, SHA-256, 바이트 수, 허용 용도와 금지 주장은
[`fragrance_ai/data/data_manifest.json`](fragrance_ai/data/data_manifest.json)에
기록된다. 다음 명령은 현재 파일과 manifest의 일치만 검사한다.

```powershell
python scripts\audit_data.py
```

이 감사는 외부 출처의 최신성, 법적 라이선스 적합성 또는 상용 승인을 새로
부여하지 않는다.

소스 작업공간에만 있는 `reference_fragrances.db`는 원 출처와 라이선스가
확립되지 않은 역사적 동시출현 참고 자료다. wheel에는 포함하지 않으며, 설치된
패키지는 파일이 없을 때 빈 `HistoricalReferenceCorpus`로 안전하게 동작한다.
사람 관능·공급사 입력용 CSV 템플릿도 wheel 배포 대상이 아니다. 자세한 정책은
[`LICENSE_POLICY.md`](LICENSE_POLICY.md)를 따른다.

## 과학·비인간 데이터의 용도

- 공개 물성, 공개 조성, 비인간 독성·제품용도 데이터는 물성 기반 후보 순위와
  위해성 스크린의 보조 근거다.
- EPA CompTox 추출은 CAS/DTXSID 연결, 제품용도 및 독성 참고용이다. IFRA 적합,
  공급사 적격성, 자동 안전 승인 또는 사람의 후각 정확도를 뜻하지 않는다.
- 공개 GC-MS 조성과 냄새 역치는 참조 prior이며, 현재 공급 로트의 GC-MS 또는
  생성 처방의 관능 결과가 아니다.
- R2와 농도 반응 자산은 manifest에 기록된 적용 범위 안에서만 사용한다. 실제
  사람의 후각 유사도 90%를 인증하지 않는다.

DREAM 2025 혼합향 challenge 저장소는 공개 접근 가능하지만 현재 확인한 commit의
root에 라이선스 파일이 없다. 따라서 원본 CSV, OpenPOM profile 및 그 자료로 적합한
ridge 계수는 `benchmarks/`의 로컬 회고 연구 증거로만 유지하며 wheel·상용 runtime
자산에 포함하지 않는다. 명시적 재배포·상용 이용 권리가 확인되기 전에는 성능이
개선되어도 production 승격 근거가 될 수 없다.

Odor-pair GNN source에는 MIT 파일이 있으나 저작권자 자리표시자가 완성되지 않았고,
v2가 적합한 DREAM mixture label,
fine-tuned checkpoint, precomputed embedding과 최종 ridge 계수는 라이선스가 미명시된
DREAM release에 결합돼 있다. 따라서 MIT source 권리만으로 결합 runtime의 상용
재배포를 추정하지 않으며, v2 report/runtime도 로컬 `benchmarks/`에만 두고 wheel의
package-data allowlist에는 추가하지 않는다.

관련 범위는 [EPA CompTox 데이터](EPA_COMPTOX_DATA.md),
[비인간 데이터 허브](NONHUMAN_DATA_HUB.md),
[공급사 데이터 입력](SUPPLIER_DATA_GUIDE.md)에 각각 분리해 설명한다.
