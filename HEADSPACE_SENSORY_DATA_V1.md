# 농도·헤드스페이스·혼합 관능 데이터 v1

이 연구 자산은 공개 농도·혼합 관능 자료와 EPA 실측 물성을 한 SQLite 허브로
연결한다. 목적은 물리·관능 조건이 없는 분자 평균 표현을 진단하고, 실제
데이터가 없는 곳을 값으로 꾸며 넣지 않는 것이다. 상용 runtime 데이터나 인간
후각 90% 인증 자료가 아니다.

## 고정 원천과 저장 범위

- Pyrfume data commit `8054ea98ed675005ec10e67359902f500e4911b0`
  - Keller 2016: 480분자 × 2농도, paraffin oil, 강도·쾌도·친숙도
  - Ravia 2020: 단일향·혼합향, 3단계까지의 희석, 강도·유사도·판별
  - Ma 2021: 72개 stock 농도와 6,660개 이성분 관능 trial
  - Bushdid 2014: 264개 3AFC trial, 각 trial의 세 혼합물과 희석
  - Snitz 2013: 360개 혼합향 유사도
  - Abraham 2012: `log10(1/ODT[ppmv])` 역치 268개
- EPA OPERA paper data `S1.zip`
  - SHA-256 `84a51d3615f61c6d752a0d0cb1254fa73ff00c9a3103f1830c695864d2ff1b7c`
  - vapor pressure, Henry constant, boiling point, logKoa, logP, water solubility
  - 실측 train/test SDF 27,283행, CC0
- PubChem PUG REST는 Pyrfume 한 개 누락 구조의 식별자 보충에만 사용한다.

생성된 [허브 보고서](benchmarks/headspace_sensory_hub_v1.json)는 1,642개 분자,
2,689개 자극, 21,708개 자극 구성분, 1,473개 Ravia 희석 단계, 109,688개 관능
관측을 기록한다. 원본 파일 29개의 SHA-256과 바이트 수를 먼저 검증한 뒤에만
재구축된다. DB는 [연구 허브](benchmarks/headspace_sensory_hub_v1.db)이며 wheel에
넣지 않는다. Pyrfume repository의 MIT 라이선스가 개별 원 논문의 재배포 권리까지
대신한다고 추정하지 않는다.

## 물리 계산 계약

[headspace.py](fragrance_ai/research/headspace.py)는 DB를 read-only로 열고 보고서의
DB 해시를 다시 확인한다.

- 비수계 이상용액: `p_i = x_i * gamma_i * P_i_sat`
- 기체 농도: `C_i = p_i / (R*T)`, `ppmv_i = p_i/P_total*10^6`
- 묽은 수용액: `p_i = H_i * C_liquid_i`
- 증기압 SDF 단위: `10^LogVP mmHg`를 Pa로 변환
- Henry SDF 단위: `10^LogHL atm*m^3/mol`을 `Pa*m^3/mol`로 변환
- 증기압이 없고 측정 끓는점만 있으면 Trouton-rule fallback으로 별도 표시
- 둘 다 없으면 `missing`으로 남기며 운영용 추정값을 만들지 않음
- Abraham 역치는 `ODT_ppmv = 10^(-log10(1/ODT))`로만 변환

Raoult와 Henry 결과는 25 °C 이상평형 기준이다. 실제 제품 베이스의 활동도,
동적 방출, vial·wick·피부·섬유, 산화, 포장, 공급 로트의 headspace GC-MS를
측정한 결과가 아니다.

### 증기압 결측 보간 감사

EPA OPERA train 2,032분자로 217개 RDKit descriptor Ridge를 만들고, OPERA test
679분자에서 R² `0.9048`, log10(mmHg) RMSE `1.0970`을 얻었다. 그러나 acyclic
분자마다 별도 scaffold를 주는 약한 분리를 금지하고 모든 acyclic 분자를 하나의
그룹으로 묶자, 완전 Bemis–Murcko-disjoint test는 93분자만 남았다. 이 엄격 test의
R²는 `0.7493`, RMSE는 `1.7347`, 절대오차 95% 분위수는 `3.9060` log10(mmHg)다.
사전 gate의 표본 300, R² 0.75, RMSE 1.5, q95 2.5를 모두 충족하지 못했다.

따라서 [보간 보고서](benchmarks/opera_vapor_pressure_imputer_v1.json)와 pickle 없는
[portable runtime](benchmarks/opera_vapor_pressure_runtime_v1.json)은 진단용으로만
보존하며 가중치는 `0`이다. DREAM의 46개 결측은 이 예측으로 채우지 않았다.

## 농도 보정 결과

[농도 보정 보고서](benchmarks/concentration_headspace_calibration_v1.json)는 Keller
480개 분자를 CID hash로 378개 학습·102개 holdout으로 분리했다. 학습 분자의
중앙값만으로 다음 지수를 고정했다.

```text
log1p(I2) = log1p(I1) + 0.1282690033 * ln(C2/C1)
```

Holdout 결과는 다음과 같다.

| 방향 | 후보 MAE | 농도 무시 기준 MAE | paired bootstrap 개선 95% 구간 |
|---|---:|---:|---:|
| 저농도 → 고농도 | 15.8829 | 26.1929 | [6.8934, 13.5554] |
| 고농도 → 저농도 | 9.9761 | 26.1929 | [12.8163, 19.4177] |

이는 동일 단일분자의 농도 이동을 anchor 관능값으로 보정하는 결과다. 미지 분자의
절대 강도, 복합 처방 또는 자연어 레시피 정확도를 검증한 결과가 아니다. 따라서
농도 transfer gate만 통과하고 production weight는 `0`이다.

## DREAM 혼합향 적용 결과

[DREAM headspace v3 보고서](benchmarks/dream_headspace_retrospective_v3.json)는
독립 Keller 지수를 사용해 equal-liquid-component 가정 아래 상대 기상 가중치를
만들었다. DREAM 구성분 204개 중 다음 근거를 사용할 수 있었다.

- EPA 실측 증기압: 120개
- EPA 측정 끓는점 + Trouton fallback: 38개
- 결측: 46개

Nested training-source holdout으로 고른 보수적 residual 후보는 test
Pearson/RMSE를 `0.4304/0.09967`에서 `0.4413/0.09696`으로 개선했다. Validation
Pearson도 `0.6397`에서 `0.6481`로 올랐지만 RMSE/MAE가 `0.11192/0.09162`에서
`0.11339/0.09348`로 악화했고 test Spearman도 소폭 낮아졌다. Pearson·Spearman·
RMSE·MAE를 두 외부 세트에서 모두 개선해야 하는 point-Pareto 후보는 0개였다.

따라서 residual scale `1.0`이 training source에서 선택됐어도 외부 gate에서
자동 탈락했고 runtime weight는 `0`이다. 원인은 물성 행의
부재만이 아니라 DREAM 혼합물의 정량 액상 조성이 없다는 적용범위 불일치다.
물성을 연결했다는 사실만으로 실제 headspace가 알려졌다고 간주하지 않는다.

## 재현 명령

```powershell
python scripts\acquire_headspace_sensory_sources_v1.py

python scripts\build_headspace_sensory_hub_v1.py `
  --pyrfume-root tmp\pyrfume_data_source_20260828 `
  --opera-zip .cache\epa_opera\S1.zip

python scripts\calibrate_concentration_headspace_v1.py

python scripts\train_opera_vapor_pressure_imputer_v1.py

python scripts\benchmark_dream_headspace_v3.py `
  --dream-root tmp\dream_olfactory_mixtures_2025_source `
  --pommix-root tmp\pommix_source_20260828 `
  --pair-source-root tmp\laura_dream_source_20260828

python scripts\audit_data.py
```

공식 원천: [EPA CompTox downloadable data](https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data),
[EPA OPERA paper data](https://epa.figshare.com/articles/dataset/OPERA_Models_for_Predicting_Physicochemical_Properties_and_Environmental_Fate_Endpoints_Data_Associated_with_Publication/6062758),
[Pyrfume data repository](https://github.com/pyrfume/pyrfume-data),
[Abraham et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3278675/).
