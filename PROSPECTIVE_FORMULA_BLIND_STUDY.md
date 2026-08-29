# 전향적 조향식 인간 블라인드 검증 v1

이 절차는 현재 모델의 자연어 조향 결과가 실제 사람에게 얼마나 비슷하게
느껴지는지를 **결과 미개봉 상태에서** 검증하기 위한 산업형 연구 패키지다. 기존
Bushdid·Ma·Bierling·Minnesota 평가는 공개 분자 또는 공개 혼합물에 대한 외부
검증이었고, 생성된 정량 조향식 자체의 종단 관능시험은 아니었다. 이 v1이 그
빈칸을 대상으로 한다.

## 고정 설계

- 안전·가격·공급성 내부 제약을 통과한 자연어 조향식 24개
- 일반 조향식 비교 100쌍: 예측 유사도 5개 구간에서 각 20쌍
- 같은 조향식을 별도 코드로 담은 동일 대조 10쌍
- 같은 농축액의 완제품 농도 10% 대 20% 대조 10쌍
- 총 120쌍, 익명 참가자 80명, 참가자당 30쌍, 쌍당 20평가, 총 2,400행
- 모든 시료는 쌍별 고유 3자리 코드이며 각 쌍의 좌우 순서는 10:10으로 균형
  무작위화
- 결과 행 제외 없음. 누락, 중복, 범위 오류, 코드 변경은 연구 전체를 실패 처리

사전고정 예측은 농도·시간 헤드스페이스 PhysSim 45%, 정량 성분 겹침 35%, 향
프로필 20%다. R2 체크포인트 값은 진단용으로 저장하지만 직접 배합 능력이 외부
검증되지 않아 가중치는 0이다. 이 가중치는 인간 결과를 연 뒤 다시 맞추지 않는다.

## 준비와 봉인

```powershell
$openssl = 'C:\Program Files\Git\mingw64\bin\openssl.exe'
python scripts\prospective_formula_blind_study.py prepare `
  --output-dir benchmarks\prospective_formula_blind_study_v1 `
  --study-id PFBS-20260828-V1 `
  --openssl $openssl

python scripts\prospective_formula_blind_study.py verify-seal `
  --study-dir benchmarks\prospective_formula_blind_study_v1
```

`restricted`에는 정량 조성, 조성-시료 코드 키, 모델 예측과 무작위화 seed가
들어간다. 참가자나 평정자에게 공개해서는 안 된다. `public`에는 코드와 빈 평정
열만 있는 2,400행 템플릿 및 지침이 들어간다. `seal.json`은 이 파일 전부와 현재
예측 구현의 SHA-256을 결합하며, 결과 파일이 봉인 전에 없었다는 계약을 기록한다.

생성된 `timestamp/seal.tsq`를 독립 RFC3161 TSA에 제출하고 응답, CA 인증서,
TSA 인증서를 보존한다. 최종화는 OpenSSL이 `seal.json` 바이트에 대한 타임스탬프를
직접 검증하지 못하면 중단된다.

```powershell
python scripts\prospective_formula_blind_study.py verify-timestamp `
  --study-dir benchmarks\prospective_formula_blind_study_v1 `
  --openssl $openssl `
  --timestamp-response benchmarks\prospective_formula_blind_study_v1\timestamp\seal.tsr `
  --timestamp-ca benchmarks\prospective_formula_blind_study_v1\timestamp\tsa-ca.pem `
  --timestamp-tsa benchmarks\prospective_formula_blind_study_v1\timestamp\tsa.crt
```

이 검증 기록은 인간 결과와 제조 실행 파일이 아직 없을 때만 생성한다. 최종화는
타임스탬프 시각이 독립기관 결과 서명 발급 시각보다 앞서는지도 다시 확인한다.

## 외부 기관 실행 경계

이 저장소는 사람 노출이나 제조를 승인하지 않는다. 실제 실행 전에 독립 기관이
다음을 채우고 승인해야 한다.

- 원료별 공급사 SKU·lot, COA, SDS, 적용 범주의 IFRA 문서
- 제품 베이스, 희석·숙성 시간, 용기, 온도, headspace 및 시향 간격 SOP
- 적용 지역의 윤리/기관 승인, 포함·제외 기준, 환기·노출 한도, 이상반응 절차
- 실험자와 참가자에 대한 prediction/formula-key 차단
- 개인식별정보를 포함하지 않는 완성 CSV

독립 관능기관은 완성된 `external/human_outcomes.csv`의 SHA-256, 연구 ID,
`external/manufacturing_execution.json`, 여기에 참조된 안전·윤리·lot별 COA/SDS/
IFRA 원문, `study_protocol.json` SHA-256, `seal.json` SHA-256을 하나의 Ed25519
envelope에 서명한다. 제조 실행 JSON은 24개 batch의 실제 칭량(총량 오차 0.5%
이하, 각 조성 편차 0.25 percentage point 이하), 240개 vial 코드·농도·batch,
제품 베이스와 환경 조건을 sealed template에 그대로 결합해야 한다.
신뢰 루트는 기관 공개키와 `sensory_laboratory` 역할 및 이 artifact type을 명시한
운영자 관리 파일이어야 한다. 저장소가 자체 생성한 키나 자체 서명 결과는 독립
증거가 아니다.

## 최종화와 90% 게이트

```powershell
python scripts\prospective_formula_blind_study.py finalize `
  --study-dir benchmarks\prospective_formula_blind_study_v1 `
  --outcomes benchmarks\prospective_formula_blind_study_v1\external\human_outcomes.csv `
  --manufacturing-evidence benchmarks\prospective_formula_blind_study_v1\external\manufacturing_execution.json `
  --evidence-root independent-lab-evidence `
  --signature-envelope lab-outcome-envelope.json `
  --trust-root independent-lab-trust-root.json `
  --openssl 'C:\Program Files\Git\mingw64\bin\openssl.exe' `
  --timestamp-response benchmarks\prospective_formula_blind_study_v1\timestamp\seal.tsr `
  --timestamp-ca benchmarks\prospective_formula_blind_study_v1\timestamp\tsa-ca.pem `
  --timestamp-tsa benchmarks\prospective_formula_blind_study_v1\timestamp\tsa.crt `
  --report benchmarks\prospective_formula_blind_study_v1\final-report.json `
  --ledger benchmarks\prospective_formula_evidence_ledger.jsonl
```

주 분석은 동일·농도 대조를 제외한 일반 조향식 100쌍에서 참가자와 향 쌍을 함께
재표집하는 crossed cluster bootstrap 5,000회다. 대조 20쌍은 주 정확도를 쉽게
올리는 데 쓰지 않고 별도 품질관리 endpoint로만 사용한다.
90% 통과에는 다음 조건이 모두 필요하다.

- seal·RFC3161·독립기관 서명 검증
- 정확히 80명·120쌍·2,400행 및 쌍당 20평가
- `100 - MAE`의 95% 하한이 90 이상
- Pearson과 Spearman의 95% 하한이 각각 0.90 이상
- 동일 조성 대조 평균 90 이상, 그 bootstrap 95% 하한 85 이상

사람 결과가 없을 때는 이 판정 자체가 존재하지 않는다. 통과하더라도 결론은 이
24개 조향식, 농도, 기관, 참가자와 SOP에 한정되며 임의 자연어 요청 전체의 90%,
안전 인증 또는 제조 승인을 뜻하지 않는다. 실패한 결과도 같은 evidence ledger에
한 번만 기록되며 다른 후보 승격에 재사용할 수 없다.
