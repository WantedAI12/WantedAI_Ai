# V70 공개 규제·공급 자료와 AI 연결

2026-09-14 로컬 수정본 기준입니다. 운영 Modal은 V69/v17이며, 이 수정본의 배포·GitHub push는 아직 하지 않았습니다. API 주소와 인증키를 바꾸지 않는 추가 계약입니다.

## 적용 대상

활성 원료 3,830개 전부를 대상으로 합니다. 자료가 일치하지 않는 원료도 응답과 집계에서 제외하지 않습니다. 향수·로션·바디워시뿐 아니라 현재 지원하는 12개 제품 분류를 검사합니다.

| 자료 | 가져온 범위 | 활성 원료에서 확인된 연결 |
| --- | --- | --- |
| IFRA | 51차 개정판 전체 263개 기준, 726쪽 기준 PDF, 사용 지침, 천연·쉬프베이스 기여 자료 | 직접 기준 또는 성분 기여 연결 226개 |
| EU REACH 및 관련 CLP | 제한 79개 항목, SVHC 후보 253개 항목, 허가 대상 59개 항목, 등록 목록 25,370행, 분류 4,822행과 그룹 구성 물질 목록 | 한 개 이상의 목록에 일치한 원료 860개 |
| K-REACH 관련 정부 공개 목록 | 화학물질 47,520건 전체 | CAS 직접 일치 943개, 구조를 확인한 CAS 별칭 일치 230개 |
| FDA | 금지·제한 원료 안내의 11개 항목 및 용도·표시·불순물 관련 확인 사항 | 현재 활성 원료의 구현된 이름·원소 검사에서 해당 항목 없음. 안전성 확인 완료라는 뜻은 아님 |
| 공급사 | PerfumersWorld 공개 상품 1,212개를 전체 대조하고 일치 SKU 257개의 자료 확인 | 공개 가격 연결 243개. 재고 수량·납기·견적 유효기간은 미확인 |
| 분자 식별 | 기존 출처의 CID 2,937개를 공개 자료와 일괄 대조 | 2,937개에서 입체구조 일치. 원래 CAS와 원료 프로필을 임의 변경하지 않음 |

EU의 여러 목록은 서로 겹칩니다. 43,408개 원본·이력·그룹 행을 서로 다른 화학물질 수로 합산하면 안 됩니다. CLP 그룹 파일의 10,000건 내보내기 제한은 마지막 페이지의 140개 CAS·EC·ECHA 식별자로 보완했습니다. 그 140건의 전체 이름과 그룹 관계 문구까지 복제한 것은 아니며, 공식 상세 링크를 보존합니다. 공개 그룹 관계 자체도 완전한 법적 범위 판정은 아닙니다.

IFRA는 **51차 개정판**으로 명시합니다. 52차 협의 소식을 최종 규칙으로 처리하지 않습니다. 12개 제품에 사용하는 수치 한도 1,887개를 개별 기준 PDF와 대조했으며 차이는 없었습니다. 천연·쉬프베이스 기여 표의 1,046행을 읽었습니다. 원문 수치 `0..3` 한 건은 추정 수정하지 않았고 미확인으로 남겼습니다. 해당 원료 CAS는 현재 활성 목록과 일치하지 않습니다.

## 규제 계산 방식

IFRA 사용량은 향료 원액의 배합비가 아닌 **완제품 중 질량 백분율**로 계산합니다. 원료의 공급 상태 농도도 반영합니다. 같은 제한 성분이 여러 원료에서 들어오면 기준별로 합산합니다.

직접 첨가 성분과 천연·쉬프베이스에서 유래한 성분을 구분합니다. 여러 천연 품목이 같은 CAS에 연결되면 연결된 기여 값의 최댓값을 선별 검사에 사용하고, 실제 배치의 GC-MS 측정값이라고 표시하지 않습니다. 순도·과산화물·불순물·특정 용도·주석 조건은 별도 확인 사항으로 남습니다.

`No Restriction`, 빈칸, `See Notebox`, 특정 성분 기준 한도는 서로 다르게 처리합니다. 빈칸을 0이나 무제한으로 바꾸지 않습니다. 샴푸는 씻어내는 제품, 룸스프레이는 수동 분사, 디퓨저는 수동 취급하는 리드 디퓨저라는 분류 가정을 응답에 명시합니다. 다른 사용 형태는 제품 분류 확인이 필요합니다.

목록에 없다는 사실은 안전·합법·등록 완료를 의미하지 않습니다. EU 등록 목록에 있다는 사실도 해당 사업자의 등록이나 용도가 확인됐다는 뜻은 아닙니다. FDA에는 일반 화장품의 일괄 사전 승인 표시를 만들지 않습니다.

## 백엔드가 받는 추가 내용

### 기존 조향식 응답

기존 `regulatory` 아래의 네 탭에 `public_source_screen`과 `public_registry_check`가 추가됩니다. 기존 원료·배합비·점수·시간별 예측 필드는 유지합니다.

- `findings`: 원료별 공개 목록 또는 기준의 일치 내용과 출처.
- `material_coverage`: 해당 원료를 대조한 결과. 미일치·식별자 부족도 포함.
- IFRA의 `formula_rule_checks.checks`: 기준별 완제품 사용량, 한도, 기여 원료, 초과 여부.
- `source_url`, `source_sha256`, `pdf_pages`: 사용한 출처와 버전 근거.
- `unresolved_scope_rules`: 목록 대조만으로 판단하지 못하는 조건.
- `compliance_verified=false`: 공개 자료 대조가 사업자·품목·배치 적합성 승인이 아니라는 구분.

`public_registry_review_required`는 공개 목록을 대조했지만 추가 검토가 필요하다는 상태입니다. `blocked`는 공개 IFRA 한도 초과 또는 기존 차단 사유가 있다는 뜻입니다. 화면에서 `reference_available`나 `public_registry_review_required`를 통과 표시로 바꾸면 안 됩니다.

### 로션 전용 응답

로션의 `design`, `optimize`, `simulate`, `predict-release`, `prepare`도 같은 `regulatory` 계약을 제공합니다. 규제 부가 정보를 붙이기 위해 배합 탐색이나 신경망 예측을 다시 실행하지 않습니다.

- 설계·최적화: 반환된 레시피 또는 가장 가까운 후보를 검사합니다.
- 농도 비교에서 다른 농도를 선택했다면 최초 요청값이 아니라 반환된 배합의 농도로 검사합니다.
- 시뮬레이션·방출 예측: 입력한 실제 향료 배합을 `input_formula`로 구분합니다.
- 준비 단계 또는 후보 생성 실패: 가상의 레시피를 만들지 않고 `no_formula`로 표시합니다.
- 검사 범위는 로션의 **향료 부분**입니다. 베이스·방부력·미생물·완제품 피부 적합성까지 검사했다고 표시하지 않습니다.

### 자료 연결 상태

`GET /v2/evidence/status`는 공개 자료와 사업자 검토 자료의 연결 상태, 전체 원료 수와 가격 연결 수를 제공합니다.

`GET /v2/evidence/coverage`는 원료별 상태를 페이지로 제공합니다. `offset`은 0부터 시작하고 `limit`은 최대 500입니다. `has_more=false`까지 받아야 전체 3,830개를 확인한 것입니다. 자료가 있는 원료만 골라 전체 커버리지라고 계산하면 안 됩니다.

`download_failures`에는 수집 도중의 과거 직접 다운로드 실패도 보존되어 있습니다. 이후 확보한 대체 공식 자료의 연결 상태는 `public_regulatory_indexes`와 실제 `source_references`로 확인합니다. 실패 이력 한 건을 전체 자료 미연결로 해석하지 않습니다.

### 평가·재평가와 변경 영향

기본 `/v2/formulas/evaluate`, `/v2/formulas/reassess`는 사업자·품목·공급 자료가 없으면 기존처럼 422를 반환합니다. 공개 가격을 확인된 재고나 유효한 견적으로 바꾸어 통과시키지 않습니다.

연구 계산만 필요하면 준비 요청 최상위에 `diagnostic_only=true`를 지정하고 새 `review_id`를 받은 뒤, 동일한 모드로 평가합니다. 응답은 `diagnostic_candidates`에 결과를 두고 `candidates`는 비웁니다. `recommendation_allowed=false`를 보존합니다. 준비 후 모드를 바꾸면 409이며 다시 확인해야 합니다.

`assess-evidence`는 HTTP 200이어도 `gate_passed`가 false일 수 있습니다. 실제 등록 검토 자료가 있어도 공개 IFRA 사용량 검사와 충돌하면 `published_ifra_limit_conflicts_with_recipe`로 차단합니다.

`change-impact`는 실제로 보존한 이전 공개 관측 버전과 비교합니다. 가격이 그대로여도 규제 기준이나 목록의 연결 내용이 바뀌면 원료별 변경 목록에 포함합니다. `regulatory_previous`, `regulatory_current`, `regulatory_sources_changed`를 함께 전달합니다. 이 API가 백엔드의 승인 상태나 재고를 수정하지는 않습니다.

## 검증과 배포 구분

전체 원료·제품·체계 57,450회 검사 및 같은 성분의 전 원료 합산 158건을 확인했습니다. 이 횟수는 소프트웨어 검사 횟수이지 실물 안전 승인 횟수가 아닙니다. API 연결·전체 회귀·전체 요청 품질 결과는 별도 결과 보고서에 구분합니다.

새 공개 자료는 서버에서 한 번 읽는 해시 고정 묶음입니다. 사용자 요청 때 정부·공급사·LLM 서비스를 다시 호출하지 않습니다. 원문·모델·원료 카탈로그·개인 설정은 공개 소스 저장소와 분리합니다.

자료 갱신 시 원문과 이전 스냅샷을 보존하고 새 묶음의 해시를 등록해야 합니다. 임의로 기존 파일만 바꾸면 무결성 검사에서 거부합니다. 공개 자료를 다운로드한 것은 공급사 증명서 발급이나 사업자 등록 신청을 한 것이 아닙니다.

## 공식 출처

- [IFRA 공식 기준 문서](https://ifrafragrance.org/initiatives-positions/safe-use-fragrance-science/ifra-standards/ifra-standards-documentation)
- [ECHA CHEM 제한 목록](https://chem.echa.europa.eu/obligation-lists/restrictionList), [SVHC 후보 목록](https://chem.echa.europa.eu/obligation-lists/candidateList), [허가 대상 목록](https://chem.echa.europa.eu/obligation-lists/authorisationList), [통일 분류 목록](https://chem.echa.europa.eu/obligation-lists/clhList)
- [국내 정부 화학물질 검색·공개 내보내기](https://kreach.mcee.go.kr/repwrt/mttr/kr/mttrList.do)
- [FDA 금지·제한 화장품 원료](https://www.fda.gov/cosmetics/cosmetics-laws-regulations/prohibited-restricted-ingredients-cosmetics)
- [PerfumersWorld 공개 상품 목록](https://www.perfumersworld.com/product-search.php)

ECHA 자료 표시: Source: European Chemicals Agency, https://chem.echa.europa.eu. 원문과 공개 데이터의 의미를 바꾸지 않으며, 정보 목록과 법적 효력을 갖는 원문을 구별합니다.
