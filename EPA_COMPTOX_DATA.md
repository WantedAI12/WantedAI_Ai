# EPA CompTox 데이터 연결

`fragrance_ai/data/epa_comptox_extract.db`는 향료 카탈로그와 직접 식별자를
연결할 수 있는 행만 보존한 읽기 전용 SQLite 추출본이다. 이 데이터는 위해성
스크린과 출처 추적을 돕지만, 규제 승인 엔진이 아니다.

## 현재 번들 추출본

현재 DB의 `source_files`에는 다음 다섯 공개 bulk 원본의 버전·URL·바이트 수·
SHA-256이 기록되어 있다.

- DSSTox: December 2025 / Figshare version 8
- CPDat: 4.0 / Figshare version 5
- ToxRefDB study summary 및 POD: 3.0
- ToxValDB 입력 파일: 9.7.0 / Figshare version 11

현재 추출본의 행 수는 DB를 조회해 확인할 수 있다. 현 스냅샷에는 카탈로그
원료 38개의 DSSTox 연결, CPDat 25,986행, ToxRefDB study 13행·POD 30행,
ToxValDB 2,654행이 있다. 이는 원본 전체의 커버리지나 특정 원료의 안전성을
뜻하지 않는 저장된 행 수다.

```powershell
python -c "from fragrance_ai.recommender.epa_comptox import EPACompToxStore; print(EPACompToxStore().stats())"
```

## 허용 용도

- CAS, DTXSID, 구조 및 명칭의 연결 확인
- CPDat 제품용도·기능 관찰의 참고
- ToxRefDB/ToxValDB의 비인간 독성 관찰 탐색
- 모델 특성 연구와 출처 감사

## 금지된 해석

다음은 이 DB만으로 판정하거나 주장할 수 없다.

- IFRA 또는 지역 법규 적합성
- 상용 처방의 안전 승인·출시 승인
- 공급사, SKU, 재고, 가격, COA, SDS 또는 로트 적격성
- 사람 대상 관능, 냄새 역치의 완전성, 실제 후각 유사도
- 원본 전체 또는 전 세계 화학물질에 대한 포괄성

독성 행의 단위·종·투여 경로·연구 조건은 서로 다르므로, 단일 임계값 또는
최솟값으로 축약하지 않는다. 실제 제품 판단은 제품 범주, 노출, 규칙팩과
검증된 외부 문서를 포함한 별도 절차가 필요하다.

## 재생성과 감사

`scripts/import_epa_comptox.py`는 명시된 공개 bulk 파일을 내려받아 해시와
예상 바이트 수를 확인한 뒤 카탈로그 식별자에 맞는 행만 추출한다. 다운로드는
외부 상태에 따라 실패할 수 있으며, 실패·버전 불일치·해시 불일치를 승인으로
대체하지 않는다.

```powershell
python scripts\import_epa_comptox.py --include-toxval
python scripts\audit_data.py
```

패키지에 든 추출본의 현재 해시와 금지 주장은
[`data_manifest.json`](fragrance_ai/data/data_manifest.json)의
`epa_comptox_extract.db` 항목을 기준으로 한다. EPA의 공개 다운로드 안내는
[EPA CompTox downloadable data](https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data)에서
확인할 수 있다.
