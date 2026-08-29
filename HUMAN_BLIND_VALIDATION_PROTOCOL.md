# 인간 후각 데이터 블라인드 검증 기록

기준일: 2026-08-18

대상 모델: frozen R2 PhysSim 2-seed ensemble (`0.3:0.7`)

외부 인간 데이터: Bushdid 2014 odd-one-out 원시 행동 자료

## 판정

이 검증은 실제 인간 행동값을 정답으로 사용했지만, R2 모델은 인간 후각 ceiling의 90%에 도달하지 못했다.

| 봉인된 최종 시험 지표 | R2 | 성분 겹침 기준선 | R2 - 기준선 |
|---|---:|---:|---:|
| 인간 정답률 Spearman | 0.2917 | 0.6267 | -0.3350 |
| FDR 판별가능 자극 ROC-AUC | 0.6394 | 0.8126 | -0.1731 |
| 보정 후 정답률 MAE | 14.003%p | 12.016%p | - |

- R2 Spearman의 참가자×자극 two-way bootstrap 95% 구간: `[0.1150, 0.4007]`
- R2와 기준선의 Spearman 차이 95% 구간: `[-0.4384, -0.1516]`
- R2 AUC 95% 구간: `[0.5628, 0.7129]`
- R2와 기준선의 AUC 차이 95% 구간: `[-0.2480, -0.1014]`
- 인간 신뢰도 ceiling 대비 정규화 점수: `0.3410`, 95% 구간 `[0.1341, 0.4698]`
- 사전 정의한 인간 ceiling 90% 게이트: **FAIL**

R2는 인간 정답률과 0보다 큰 전이 신호를 보였지만, 성분이 얼마나 겹치는지만 보는 기준선보다 유의하게 낮았다. 성분 겹침과 혼합물 크기를 통제한 R2 partial Spearman은 `0.0417`이었다.

## 블라인드 순서와 무결성

검증은 한 실행 경로 안에서 다음 순서를 강제했다.

1. `molecules.csv`와 `stimuli.csv`만 읽어 자극별 예측과 평가 파티션을 생성했다.
2. 예측 파일과 예측 행의 SHA-256을 봉인 파일에 기록했다.
3. 별도 `score` 명령이 봉인을 다시 계산해 일치한 경우에만 `behavior.csv`를 열었다.
4. 인간값은 예측 봉인 뒤에만 사용했다. 보정 파티션은 isotonic 확률 보정에 사용했고, 전체 자극의 인간값은 사전 정의한 전역 FDR 보조 endpoint 계산에 사용했으며, 어느 인간값도 frozen 모델·가중치·threshold 선택에는 사용하지 않았다.

| 사건 | Asia/Seoul 시각 |
|---|---|
| 예측 봉인 | `2026-08-18T20:32:57.746010+09:00` |
| 봉인 재검증 | `2026-08-18T20:33:06.583922+09:00` |
| 인간 행동 파일 읽기 완료 기록 | `2026-08-18T20:33:06.608235+09:00` |

봉인된 예측 파일 SHA-256은 `9e254e9c00c5835f54d1167834445e8e19b9c2d82dfb1f1b7c132776fb254b69`, 예측 행 SHA-256은 `df0771826a9f88ec10187f2d1b3815217d2546fc1c7128cf53d0821e1e0e37d7`이다. 대상 인간 행동 파일 SHA-256은 `04a47b210c5a8972c29ed5dd502bf755985a6f7e0937090420b1d9910fbcfff9`이다.

이 봉인은 파일 변조 감지를 위한 재현성 장치다. 제3자 타임스탬프 공증이나 사전등록을 뜻하지는 않는다.

## 데이터와 사전 고정 평가 계약

- 인간 참가자: 26명
- 전체 자극: 264개
- 혼합물 자극: 260개
- 단분자 대조 자극: 4개
- 최종 시험: 208개
- 사후 보정 전용: 52개
- 구성분 수: 10, 20, 30개 혼합물
- 우연 정답률: `1/3`

혼합물 자극은 인간 행동값과 무관하게 `(구성분 수, 선언된 성분 겹침률)` 층 안에서 고정 salt SHA-256 순서로 20%를 보정, 나머지를 최종 시험에 배정했다. 단분자 자극은 대조군으로 분리했다.

주 평가값은 R2가 예측한 `1 - mixture similarity`와 자극별 인간 평균 정답률의 Spearman 상관이다. 불확실성은 참가자와 자극을 동시에 재표집하는 20,000회 two-way bootstrap으로 계산했다.

보조 이진 평가는 자극마다 26회 응답에 대해 우연 정답률 `1/3`과의 exact binomial 검정을 수행하고, 전체 264개 자극에서 Benjamini-Hochberg FDR `q=0.05`를 적용한 판별가능 여부를 정답으로 사용한다. 이 라벨에 대한 R2 ROC-AUC와 average precision을 측정하고, 층화 대응 bootstrap 20,000회로 기준선과 비교했다.

결과 JSON의 `behavior_opened_at` 필드는 `_read_behavior_matrix`가 반환된 직후 기록되므로 정확히는 읽기 완료 시각이다. 또한 `calibration_labels_used_only_for_post_hoc_probability_mapping`은 모델·threshold 선택 문맥의 표시다. FDR 보조 endpoint 자체는 명시한 대로 전체 264개 인간값으로 정의됐다. 최종 208개만으로 BH를 독립 재계산해도 양성 117개와 각 라벨은 모두 동일했다.

인간 정답률 절대 오차는 사전 분리된 52개 보정 자극에서만 isotonic mapping을 적합한 뒤 208개 최종 자극에서 측정했다. 이 보정은 R2 순위나 체크포인트를 다시 학습하지 않는다.

인간 noise ceiling은 참가자를 무작위 반분한 20,000회 split-half Spearman 상관에 Spearman-Brown 보정을 적용하고, 그 신뢰도의 제곱근으로 근사했다. 보정 신뢰도 중앙값은 `0.7318`, 상관 ceiling은 `0.8554`였다.

## 모델·데이터 독립성 감사

인간 행동 라벨은 R2 학습, ensemble 가중치, neutral threshold, 자극 분할에 사용되지 않았다. 그러나 화학 구조는 완전한 외삽 조건이 아니다.

| 감사 항목 | 결과 |
|---|---:|
| Bushdid 고유 분자 | 128 |
| Snitz 학습 사용 분자와 정확 중복 | 71 |
| descriptor pretraining과 정확 중복 | 125 |
| 전체 선언 모델 원천에서 처음 보는 분자 | 0 |
| Bushdid scaffold | 23 |
| 전체 선언 모델 원천에서 처음 보는 scaffold | 0 |

따라서 올바른 해석은 **실제 인간값을 사용한 behavior-label-blind 전이 평가**다. molecule-disjoint 또는 scaffold-disjoint 외부 검증이라고 부를 수 없다.

## 주장 경계

이 시험은 다음을 증명하지 않는다.

- 자연어 요청으로 생성한 완제품 레시피가 사람에게 90% 유사하게 느껴진다는 것
- 배합비, 원료 순도, 용매, 희석 농도, 숙성, 시간별 headspace가 정확히 재현된다는 것
- 보지 못한 분자 또는 scaffold에 대한 zero-shot 일반화
- 단일 연구의 26명 표본을 전체 인구에 그대로 일반화할 수 있다는 것

특히 원 자료의 희석 조건은 모델 입력에 들어가지 않았고, R2는 구성분 존재 기반 혼합물 표현을 사용한다. 그러므로 이 결과는 frozen R2의 인간 혼합물 판별 전이 성능이며, 생성 레시피의 종단 관능 정확도가 아니다.

## 재현 산출물

- 예측·채점 코드: `scripts/blind_human_olfaction_benchmark.py`
- 회귀·변조 검출 테스트: `tests/test_blind_human_olfaction_benchmark.py`
- 봉인 예측: `benchmarks/bushdid_blind_predictions_v1.json`
- 봉인 메타데이터: `benchmarks/bushdid_blind_prediction_seal_v1.json`
- 전체 채점 결과: `benchmarks/bushdid_human_blind_benchmark_v1.json`
- 사람이 읽는 결과표: `benchmarks/bushdid_human_blind_benchmark_v1.md`

동결 wheel은 `perfumery_ai_core-1.2.1-py3-none-any.whl`이며 SHA-256은 `71d413b281959fe63eff979d2a851db534b136d37277db3569ab00a275b50001`이다. 검증기는 wheel 내부 모델 코드·manifest·두 체크포인트가 source tree와 byte-identical인지 확인했다.

## v1.4 회고적 프로토콜 보정

원 봉인 시험과 산출물은 변경하지 않았다. 후속 v1.4 보정은 사전 지정된 calibration 52개 자극에서만 wrong-vial log 희석 spread의 계수와 monotonic 확률 곡선을 적합한다. 각 조성 수·겹침률 stratum의 네 자극을 네 fold에 하나씩 배치하여 52개 out-of-fold 잔차로 cross-conformal q95를 계산한다.

historical final 208개에서 성분 겹침 순위 상관 `0.626690`은 프로토콜 인지 점수 `0.646013`으로 개선됐고, paired stimulus bootstrap 개선 구간은 `[0.000933, 0.038201]`이다. 확률 MAE는 `12.0004%p`, q95는 `29.4872%p`다. 이 후속 모델은 최종 결과 공개 후 개발됐으므로 prospective blind 재검증이 아니며, 90% 향 유사도 주장도 승인하지 않는다.

- 빌더: `scripts/build_human_mixture_calibration.py`
- 런타임 아티팩트: `fragrance_ai/data/human_mixture_calibration.json`
- 기계 판독 감사: `benchmarks/bushdid_human_protocol_calibration_v3.json`
- 결과표: `benchmarks/bushdid_human_protocol_calibration_v3.md`

## Bierling 2025 공개 인간 outcome-unopened 검증

후속 독립 트랙은 2025년 Zenodo `10.5281/zenodo.15657278`의 74개 단분자 자극 메타데이터만 먼저 읽었다. Keller 2016 인간 라벨 중 목표와 정확히 겹치는 55개 분자를 모두 제외하고, 남은 419개로 RDKit·Morgan·동결 MolFormer 후보를 5-fold molecule-disjoint 교차검증했다. 개발 Macro Spearman은 선택 모델 `0.432783`, 고정 RDKit `0.349191`이었다.

74×22 예측 SHA-256 `e35a5694676c30070b1ae1077a6c23f6e434a46c743d09a62f17fea2784029d8`과 부모 코드 SHA-256 `f4fc960456b2415969cd3543d5c244ff0f364e1deb11f20410391d1fe43f2632`는 FreeTSA RFC 3161 `2026-08-26 02:31:47 UTC`에 봉인됐다. 공식 인간 outcome 다운로드는 그 뒤 `02:31:49 UTC`에 시작했고 SHA-256은 `4e7ec47089cfc43df3e008ed558ffd1ee05d23f51c364e5e2538ce247ef163a4`다.

| 공개 행동 파일에서 측정 가능한 73개 냄새 결과 | 값 |
|---|---:|
| HumanPOM Macro endpoint Spearman | 0.346753 |
| 고정 RDKit 기준선 | 0.242411 |
| 차이 | +0.104342 |
| participant×odor bootstrap 차이 95% 구간 | [0.040303, 0.150424] |
| 양의 endpoint | 22/22 |
| 정성 profile 평균 Spearman | 0.724647 |
| Top-3 정성 descriptor recall | 0.607306 |
| Ring-scaffold sensitivity | 0.252478 |
| Keller 인간 교차집단 기준, 54개 overlap | 0.487827 |

공개 자극표에는 `4Isoprop`이 있지만 행동 파일에는 해당 행이 전혀 없다. 또한 첫 scoring 호출은 통계 계산 전에 세미콜론/BOM과 `fruit`, `ammonia/urinous` 열 이름에서 중단됐다. 결과 개봉 후 별도 parser/availability adjudicator가 이 읽기 계약만 수정했으며 74개 예측, 모델, 모집단, endpoint, 지표와 bootstrap은 변경하지 않았다. 따라서 74개 원 게이트는 실패로, 공개 파일에 실측된 73개 보조 개선 게이트는 통과로 보존한다.

- 예측/부모 scorer: `scripts/blind_bierling_human_olfaction_benchmark.py`
- 사후 parser adjudicator: `scripts/adjudicate_bierling_human_olfaction_parser.py`
- 예측·seal·획득 receipt: `benchmarks/bierling_2025_blind_*_v1.*`
- 기계 판독 결과: `benchmarks/bierling_2025_human_blind_benchmark_v1.json`
- 결과표: `benchmarks/bierling_2025_human_blind_benchmark_v1.md`

이 결과는 단분자 집단 평균 지각 예측이며 생성 향수, 혼합물, 실제 개인 후각 또는 90% 향 유사도 인증이 아니다.

## Bierling intensity pilot 농도 검증

주평가와 별개인 `intensity_piloting.csv`는 예측 시점에 다운로드하지 않았다. Keller 2016의 두 농도 강도 데이터에서 target과 같은 분자를 제외하고 419분자·846조건으로 연속 농도곡선을 적합했다. 선택된 RDKit interaction ridge의 molecule-disjoint 개발 선택점수는 `0.496456`, 전역 농도 ridge는 `0.327555`였다. 74개 곡선은 농도 증가에 비감소하도록 동결했다.

예측/부모 코드 SHA-256은 `65f4a191f9408971f0944a602e25f58d810d1521e2b5504ffca2139ddeb39aaa` / `0eca50312fbe722f03b7dc4ce1f773e67d99e0f63bffd289b6b3d601d50d2561`이며 FreeTSA 시각은 `2026-08-26 03:12:19 UTC`, 공식 pilot 취득 시작은 `03:12:22 UTC`다.

| 동결 blind pilot 결과 | Spearman | MAE |
|---|---:|---:|
| Main-human anchor + Ravia delta | 0.554152 | 21.6465 |
| Target-excluded strict curve | 0.159085 | 24.1703 |
| Frozen Ravia global | -0.118430 | 12.9604 |
| Structure-only | 0.247156 | 13.0463 |

Anchor branch는 순위가 크게 개선됐지만 절대 보정은 Ravia보다 나빴다. Pilot에는 73개 분자·75조건만 있고 다농도 분자가 두 개라 delta Spearman을 추정할 수 없다. 따라서 blind condition-transfer 전체 게이트와 strict 외부 게이트는 모두 실패로 보존한다. 공개 CSV는 변수사전과 다른 `intensity` 열, 12개 0점, participant 18의 두 anchor 반복행을 포함해 사후 parser adjudication을 명시했다.

결과 개봉 뒤 frozen branch만 입력으로 사용한 nested 5×4 molecule-disjoint affine 보정은 Spearman `0.536042`, MAE `10.6853`을 얻었다. 참가자×조건 bootstrap에서 Ravia 대비 Spearman 개선 구간은 `[0.287653, 0.929409]`이지만 MAE 감소 구간은 `[-0.214284, 3.779014]`로 0을 지난다. 따라서 회고 보정 release gate도 실패다. 이 아티팩트는 진단 전용이며 runtime weight `0`, concentration delta·혼합물·레시피·90% 후각 주장은 모두 `false`다.

추가 v2는 main anchor, 다섯 frozen prediction branches, log 농도차와 volume을 portable affine/hinge/Huber/isotonic 후보로 구성하고, inner selection을 가진 5회 반복 outer molecule-disjoint 5-fold로 평가했다. 결과는 Spearman `0.612721`, MAE `8.937125`로 Ravia MAE `12.960352`보다 `31.04%` 낮다. 참가자×조건 bootstrap에서 Spearman 개선 구간 `[0.349304, 0.984766]`, MAE 감소 구간 `[0.790638, 6.044785]`가 모두 양수여서 v2 대폭 개선 게이트는 통과했다.

선택된 Huber hinge 모델과 표준화·계수는 JSON 숫자 배열로 고정했지만 개발 시점이 pilot 개봉 후이므로 runtime weight는 `0`이다. 새 외부 데이터에서 재현되기 전에는 현재 Ravia production 곡선을 교체하지 않으며 delta·혼합물·레시피·90% 주장은 승인하지 않는다.

## Ma 2021 이성분 혼합물 outcome-unopened 검증

공식 Dataverse의 첫 worksheet TSV는 CAS·원료명·농도·용매·순도·trial 번호만 포함하고 IA/IB/IAB 관능값을 포함하지 않는다. 이 표와 PubChem 구조만 읽고 실제 222개 조합을 선택하지 않은 채 가능한 2,556쌍 전부를 먼저 생성했다. Keller 인간 학습에서는 72개 표적 구조를 정확 일치 기준으로 제외했고, R2·Ravia 자산은 기존 동결본만 사용했다. 예측 JSON과 코드·수식·채점 계약은 FreeTSA RFC 3161 `2026-08-26 11:01:35 UTC`에 봉인했으며 원본 Excel 다운로드는 그 뒤 시작했다.

행 단위 결과 파일은 봉인 전 열지 않았지만 관련 논문의 집계 결론은 source selection 중 이미 확인했다. 그러므로 승인된 연구 라벨은 `row-level outcome-unopened, publication-summary-aware external test`이고 fully outcome-naive prospective blind는 `false`다. 이 사후 scope adjudication은 예측·원 평정·지표·게이트를 변경하지 않는다.

| 고유 198 혼합물 | Spearman | MAE (0–10) | RMSE |
|---|---:|---:|---:|
| 사전선정 Ravia Weber–Fechner | 0.729566 | 0.269321 | 0.336481 |
| 고정 strongest-component | 0.726863 | 0.265200 | 0.331461 |

사전선정 연산자는 rank가 아주 조금 높았지만 MAE·RMSE가 모두 나빴고, 참가자×혼합물 bootstrap max-minus-primary MAE 95% 구간 `[-0.009436, 0.002450]`도 0을 지났다. 따라서 원 블라인드 통합 게이트는 실패다. 반면 max의 MAE 자체는 척도의 2.65%였으며, 같은 참가자·같은 혼합물 반복 평정 MAE `1.16559`보다 훨씬 작다. 이는 고유 혼합물의 패널 평균 예측이지 개인 평정 오차가 아니다.

결과 개봉 후 v2는 사전동결 입력과 측정된 IA/IB/PA/PB만 사용해 max 잔차를 적합했다. Repeated nested exact-pair holdout에서 Spearman `0.798020`, MAE `0.217662`, RMSE `0.274896`; MAE 감소율 `17.93%`; bootstrap 개선 구간 `[0.005587, 0.064628]`을 얻었다. 이 구간은 이미 훈련에 등장한 성분들의 새로운 정확한 쌍을 평가한다. 더 엄격하게 각 fold에서 두 성분을 모두 훈련에서 제외하고 후보 선택도 그 fold의 훈련 자료 안에서만 다시 수행한 pooled 379예측은 Spearman `0.708903`, MAE `0.274030`으로 max `0.735636`, `0.264224`보다 나빴다. Component fit·selection leakage는 0이지만 strict cold gate와 전체 gate는 실패다. 최종 `ridge_full_100` JSON 재실행 parity 오차는 `0`이나 runtime primary weight는 `0`이다.

- 부모 예측·봉인·획득·채점: `scripts/blind_ma_2021_binary_mixture_benchmark.py`
- 사후 보정: `scripts/build_ma_2021_mixture_calibration_v2.py`
- 블라인드 결과: `benchmarks/ma_2021_binary_mixture_blind_benchmark_v1.json`
- v2 결과: `benchmarks/ma_2021_mixture_calibration_v2.json`

이 검증은 측정된 단일 성분 강도에서 이성분 전체 강도를 예측하는 제한된 문제다. 생성 향수나 90% 실제 후각 유사도 인증으로 해석하지 않는다.

## Universal intensity와 Minnesota 결과 미개봉 검증

미지 원료 전이를 위해 Keller·Ravia·Bierling을 통합한 ID-free 물성·농도 모델을 만들었다. Ma와 같은 구조를 전부 제거한 raw universal 모델은 Ma rank를 높였지만 절대 강도 bias로 실패했다. 결과 개봉 후 만든 equal-centered hybrid v2는 단일향 MAE `0.069045`와 이성분 Fechner MAE `0.475906`을 기록해 HumanPOM `0.076520`, `0.512650`보다 낮았고 두 MAE bootstrap 하한은 양수였다. 이는 Ma-informed retrospective repair이며 prospective external 또는 runtime 승격이 아니다.

PubChem 공개 물성은 575구조를 연결했지만 증기압은 188개뿐이었고 끓는점 기반 fallback을 포함한 transport v3가 v2를 이기지 못했다. `[ppm]`과 low/high 문맥을 처리하도록 역치 파서를 고쳐 556구조 중 97개 역치를 확보했으나 threshold v4도 v2보다 MAE가 나빴다. 두 가지 모두 실패 아티팩트와 가중치 0을 보존한다.

Minnesota 농도 매칭 시험은 v2 이후 처음 여는 외부 결과로 사용했다. Butyric acid, delta-decalactone, furaneol, methional과 5 ppm acetylpropionyl 표준을 학습에서 제외하고 네 match 농도를 먼저 생성·RFC 3161 봉인했다. Retest readme 360행과 실제 repository 파일 250행 불일치는 사후 acquisition adjudication에 기록했으며 원본 MD5와 실제 1,630행을 보존했다. 부모 pooled score의 hybrid log10 match MAE는 `1.747698`이었다. 봉인된 “four-compound mean”을 정확히 재현한 사후 weighting adjudication에서는 hybrid `1.668654`, 5 ppm 상수 기준선 `1.070744`로 역시 더 나빠 외부 농도 게이트가 실패했다. 예측값·원 평정·participant match interpolation은 바꾸지 않았다. 이 자료에는 전체 혼합 강도 평정이 없으므로 mixture gate도 열 수 없다.

- universal v1/v2/v3/v4: `benchmarks/universal_intensity_*.json`
- PubChem 물성·역치: `benchmarks/universal_intensity_physchem_v1.json`, `benchmarks/universal_odor_thresholds_v2.json`
- Minnesota 예측·봉인·결과·scoring adjudication: `benchmarks/minnesota_intensity_blind_*_v1.json`
- Minnesota 결과표: `benchmarks/minnesota_intensity_blind_benchmark_v1.md`

현재 검증된 결론은 “상대 구조 신호는 일부 개선 가능하지만 공개 일반 물성만으로 절대 농도/후각 강도를 재현할 수 없다”이다. 실제 후각 90% 인증은 계속 `false`다.
