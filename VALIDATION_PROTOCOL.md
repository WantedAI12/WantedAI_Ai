# 검증 프로토콜

## 1. 주장 경계

이 프로젝트의 기본 출력은 **처방 후보를 정렬하기 위한 계산 프록시**다. `simulation_score`, `simulation_p05`, 헤드스페이스 추정치, 물성 coverage, R2 유사도는 모두 계산 결과이며 사람의 후각 유사도를 직접 측정한 값이 아니다.

다음 실제 관능 필드와 상태는 외부의 서명된 인간 관능 기록이 연결되기 전에는 `null`, `false` 또는 기권 상태여야 한다.

- `actual_olfactory_similarity_score`
- `actual_olfactory_lower_bound_95`
- `human_validated_90`

`simulation_only_approved`는 호환성을 위해 남은 폐기 예정 필드이며 현재 런타임에서는 항상 `false`다. 모델 적용 범위 진단은 `scientific_model_domain_passed`로, 독립 정량 기준향 비교 가능 여부는 `physsim_comparison_authorized`로 분리한다. 두 필드 모두 인간 시향 검증이나 실제 후각 90%를 뜻하지 않는다.

## 2. 두 종류의 검증을 분리한다

| 구분 | 허용되는 입력과 출력 | 금지되는 주장 |
|---|---|---|
| 계산 프록시 검증 | 레시피, 구조, 공개 물성, 추정 헤드스페이스, 시간축, 불확실성, OOD, 안전 제약 | 인간 후각 유사도 측정 또는 인증 |
| 실제 관능 검증 | 사전 등록된 프로토콜, 독립 패널, 배치·희석·제품 매트릭스가 고정된 실물 시료, 서명된 원시 결과 | 프록시 점수만으로 사람 점수를 대체 |

실제 관능 필드는 후자의 증거가 정확한 대상 처방과 연결될 때만 채운다. 합성 시향, 모델 출력, 자유 텍스트 평가, 임의 해시값은 실제 관능 증거가 아니다.

## 3. 기준향(reference target) 계약

텍스트에서 내부적으로 만든 목표 레시피는 후보 탐색의 보조 신호일 수 있지만, 실제 기준향 재현 검증의 정답으로 사용할 수 없다. 실제 기준향 평가에는 등록된 `reference_target_id@version`이 필요하며 다음을 모두 검증한다.

1. 정량 조성이 있고 합계가 100%이다.
2. 조성의 출처 문서와 버전이 등록되어 있다.
3. 제품 카테고리, 향료 농도, 제품 베이스/매트릭스가 후보와 일치한다.
4. 원료 식별자와 조성 근거가 누락되지 않았다.
5. 같은 기준향 버전, 동일 배치 또는 배치 간 변동을 명시한 증거를 사용한다.

이 계약이 실패하면 기준향 유사도는 계산하지 않고 fail-closed 한다. 노트 목록, 향명, 또는 자연어 설명만 있는 향수 데이터는 정량 기준향 조성으로 승격하지 않는다.

## 4. 계산 프록시 검증 절차

### 4.1 결정론적 물성·헤드스페이스 가지

후보와 기준 조성에 대해 다음을 기록한다.

- 시간점별 헤드스페이스/odor-activity 프록시와 시간축 유사도
- 직접 측정 물성, 공개 조성 prior, QSPR 추정치의 coverage
- Monte Carlo 평균, 5% 및 95% 분위수
- 제형·구조 applicability와 불확실성 폭
- 금지·위험·가격·가용성·규제 제약 결과

모델 applicability가 설정된 하한 미만이거나 불확실성이 허용 범위를 넘으면 모델-domain 진단을 실패시킨다. Monte Carlo 분위수는 입력 prior 전파 구간이며 인간 관능의 경험적 오차 구간이 아니다.

`evidenced_nonhuman_pass`는 등록된 정량 기준향이 있을 때만 가능하다. 텍스트·시간축 프록시의 5% 하한, 물성 model-domain 게이트와 함께 **기준향 대비 승인된 PhysSim 점수 자체도 요청 임계값 이상**이어야 한다. 기준향 비교가 OOD이거나 그 점수가 임계값 미만이면 다른 프록시 점수가 높아도 실패한다.

### 4.2 농도 반응 가지

농도 반응 모델은 단일 분자 강도 관측과 혼합물·완제품 관능을 구분한다. 구조별 또는 혼합물별 효과가 독립적인 holdout에서 검증되지 않았다면 해당 효과의 production weight는 `0`이다. 전역 농도 보정도 독립적인 혼합물·제품 매트릭스 검증과 신뢰 계약이 없으면 기본 가중치는 `0`이다.

따라서 농도 가지의 숫자가 존재해도 실제 기준향 유사도 또는 혼합물 인간 후각 유사도의 증거가 되지 않는다.

### 4.3 R2 PhysSim 가지

R2는 역사적 mixture-pair similarity용 연구 모델이다. 구조 descriptor와 구성분 존재 기반 입력을 사용하며, 현 레거시 체크포인트는 다음 입력을 직접 인코딩하지 않는다.

- 상대 배합비
- 완제품 향료 농도
- 시간 또는 헤드스페이스 궤적

런타임은 위 세 입력을 직접 다루는 capability contract와 검증 게이트가 모두 있을 때만 R2의 최종 점수 가중치를 허용한다. 현재 패키지의 체크포인트에는 그 계약이 없으므로 `approved_primary_score_weight=0`과 `applied_primary_score_weight=0`이 정상 동작이다. 체크포인트가 로드되었다는 사실은 점수 기여 승인과 다르다. 운영 추론은 원 체크포인트를 빌드 시점에 안전한 NumPy 배열로 내보내고 수치 동등성을 검증한다. 배포 시에는 portable 배열·runtime manifest·ensemble/descriptor 계약의 SHA-256을 검증하며, 역직렬화 가능한 원 `.pt` 체크포인트는 wheel에 넣지 않는다.

R2를 다시 승인하려면 다음을 모두 만족해야 한다.

1. positive-only odor descriptor를 미관측 음성으로 취급하지 않는 PU-safe 학습 계약
2. 모든 구성분 held-out 점수만을 사용하는 molecule/scaffold 분리 계약
3. 외부 Ravia 분자와 scaffold가 supervised pretraining, fine-tuning, normalizer fitting과 모두 분리되었음을 보이는 overlap audit
4. 강한 기준선, 다중 시드, 불확실성 보정, artifact hash 검증
5. 배합비·농도·시간/헤드스페이스를 직접 입력하는 모델 capability 계약

하나라도 빠지거나 거짓이면 가중치는 `0`이다.

### 4.4 OOD와 불확실성

R2 및 계산 가지는 다음 경우 점수 기여를 중단하거나 프록시 승인에서 제외한다.

- 구조·조성 coverage 또는 descriptor domain coverage 부족
- 설정된 applicability 하한 미달
- 앙상블 member disagreement가 개발 보정 상한을 초과
- conformal 구간 또는 시간축 불확실성이 허용 범위를 초과
- 원료 성분·물성·규제 증거의 해시 또는 버전 불일치

OOD 차단은 정확도를 보증하는 장치가 아니라 검증되지 않은 영역에서 과도한 신뢰를 막는 장치다.

## 5. 실제 인간 후각 검증을 위한 별도 프로토콜

실제 유사도 또는 90% 하한을 주장하려면 다음의 외부 실행이 필요하다.

1. 기준향과 후보의 정확한 조성, 배치, 희석, 제품 베이스, 포장을 고정한다.
2. 사전 등록된 similarity 척도·분석 계획·승인 기준을 사용한다.
3. 무작위화·블라인드·반복 제시와 독립 패널을 사용한다.
4. 원시 응답, 제외 규칙, 패널 크기, 통계 모형, 95% 하한을 전자서명된 증거와 함께 보존한다.
5. 학습·모델 선택에 사용되지 않은 기준향으로 재현한다.

이 절차가 완료된 정확한 처방 지문에만 `actual_olfactory_similarity_score`와 `actual_olfactory_lower_bound_95`를 연결한다.

### 5.1 현재 연결된 인간 행동 보정의 범위

Bushdid 2014 odd-one-out 행동값으로 만든 보정 아티팩트는 향수 유사도가 아니라 **3개 중 다른 혼합물을 찾을 정답 확률**을 예측한다. v1.4는 성분 겹침뿐 아니라 원 시험의 wrong-vial log 희석 spread를 nuisance 변수로 사용한다. 계수·확률 곡선은 사전 분리한 calibration 52개 자극만으로 적합했고, 각 조성 수·겹침률 stratum을 나눈 4-fold out-of-fold 잔차 전체로 절대오차 95% 분위수를 정했다. historical final 208개 자극은 적합이나 계수 선택에 사용하지 않았다.

- 적용 가능: 원 보고서·stimulus 표 해시, 성분 수와 `0.25/0.5/1.0` 세 vial 희석 설계가 모두 일치하는 등록 Bushdid supplemental 프로토콜 연구 감사
- 적용 불가: 실제 향수 배합비, 에탄올 희석, 제품 베이스, 천연 원료 로트, 일반 자연어 생성 처방
- 회고적 프로토콜 인지 결과: 순위 Spearman `0.6460`, 절대 정답률 MAE `12.00%p`, 4-fold cross-conformal `q95=29.49%p`, 인간 ceiling 정규화 `0.7552`
- 개발 시점: 원 봉인 최종 라벨 공개 후이므로 새로운 prospective external validation은 아님
- 90% 향 유사도 주장 승인: `false`

일반 처방 행은 분자별 stock 희석·용매와 자극 vial 희석을 표현하지 못한다. 따라서 비율이 같아 보이더라도 확률을 외삽하지 않고 `abstained_formula_endpoint_not_supported`를 반환한다.

### 5.2 Bierling 2025 outcome-unopened 단분자 검증

2025년 공개 laypeople 데이터의 `data.csv`를 받기 전에 `odors.csv`의 74개 구조만 읽고 22개 지각 endpoint를 예측했다. Keller 2016 인간 데이터에서 목표와 정확히 같은 55개 분자는 학습 라벨에서 제외했으며, RDKit·Morgan·동결 MolFormer 후보와 endpoint별 선택은 Keller 내부 5-fold molecule-disjoint 결과로만 결정했다. 예측·부모 scorer·모집단·endpoint·bootstrap 계약은 FreeTSA RFC 3161 시각 `2026-08-26 02:31:47 UTC`에 묶였고 공식 outcome 다운로드는 `02:31:49 UTC`에 시작했다.

- 사전 고정 모집단: `study=main`, `inclusion=1`, `sampling_group in {home,lab}`
- 실제 공개 행동 표본: 1,119명, 73개 냄새. 자극표의 `4Isoprop`은 행동 파일에 행이 없어 미측정으로 유지
- Primary Macro endpoint Spearman: `0.346753`
- 고정 RDKit 기준선: `0.242411`; 차이 `+0.104342`
- participant×odor bootstrap 차이 95% 구간: `[0.040303, 0.150424]`
- 22개 중 양의 endpoint: `22`; 정성 profile 평균 Spearman: `0.724647`
- exact-overlap Keller 인간 교차집단 기준: `0.487827` (54개 냄새)
- ring-scaffold sensitivity: `0.252478`

첫 scoring 호출은 통계 계산 전에 CSV delimiter 자동탐지와 `fruit`/`ammonia/urinous`의 변수사전 표기 차이에서 중단됐다. 결과 개봉 뒤 별도 adjudicator가 UTF-8 BOM, 세미콜론 delimiter와 두 1:1 열 별칭만 수정했고 모델·예측·모집단 필터·endpoint·통계량은 바꾸지 않았다. 따라서 **예측은 outcome-unopened**이지만 scorer 전체가 완전 사전봉인됐다고 표현하지 않는다. 원 74개 게이트는 `false`, 공개 파일에서 측정 가능한 73개 개선 게이트만 `true`다.

이 검증은 단분자 집단 평균 descriptor 예측이다. 혼합물, 배합비, 제품 베이스, 자연어 레시피 또는 실제 후각 90%를 검증하지 않으며 운영 `actual_olfactory_similarity_score`에 입력하지 않는다.

### 5.3 Bierling intensity pilot 농도 조건전이

별도 `intensity_piloting.csv`는 내려받지 않은 상태에서 Keller 2016의 두 농도 강도값으로 target-excluded RDKit×log농도 interaction ridge를 선택하고, 74개 분자의 `10^-6..1` 연속곡선을 비감소 형태로 동결했다. 같은 분자의 이미 검증된 main-study 강도를 기준점으로 두고 Ravia 곡선의 농도 변화만 더한 condition-transfer branch도 별도로 고정했다. 예측 SHA-256 `65f4a191...39aaa`는 RFC 3161 `2026-08-26 03:12:19 UTC`에 봉인됐고 pilot 다운로드는 `03:12:22 UTC`에 시작했다.

- Pilot: 100명, 964개 participant-condition 평가, 73개 분자, 75개 조건
- 동결 condition-transfer Spearman: `0.554152`; MAE `21.6465`
- target-excluded strict curve: Spearman `0.159085`; MAE `24.1703`
- 동결 Ravia: Spearman `-0.118430`; MAE `12.9604`
- condition-transfer 대 Ravia Spearman 차이 bootstrap 95% 구간: `[0.271141, 0.929437]`
- 실제 다농도 분자가 두 개뿐이어서 delta Spearman은 세 점 미만으로 `0`; blind 전체 게이트와 strict 외부 게이트: `false`

첫 scoring은 통계 계산 전에 `intensity`/`intensive` 열 차이에서 중단됐다. 사후 adjudicator는 이 별칭, trailing empty 열, 원 파일의 12개 0점 slider endpoint 허용과 participant 18의 두 anchor 반복평가를 participant-condition 평균으로 합치는 처리만 기록했다. 곡선·원 평가값·지표·bootstrap은 바꾸지 않았다.

Pilot 개봉 뒤 수행한 5×4 nested molecule-disjoint 회고 보정의 점 추정은 Spearman `0.536042`, MAE `10.6853`으로 Ravia보다 좋았다. 참가자×조건 bootstrap Spearman 개선 95% 구간은 `[0.287653, 0.929409]`이지만 MAE 감소 구간은 `[-0.214284, 3.779014]`로 0을 지난다. 따라서 회고 보정 release gate도 `false`이며 portable diagnostic artifact의 runtime primary weight는 `0`, concentration-delta 검증도 `false`다.

v2는 이 실패를 숨기지 않고 feature bank를 main anchor, strict/Keller/Ravia frozen branches, log 농도차, absolute 농도차, participant-weighted volume으로 확장했다. Portable ridge·hinge·Huber·isotonic 49개 후보를 inner fold에서만 선택하고, 분자 전체를 가리는 outer 5-fold를 5회 반복했다.

- Repeated nested Spearman: `0.612721`
- Repeated nested MAE: `8.937125` (`Ravia 12.960352` 대비 `31.04%` 감소)
- 참가자×조건 bootstrap Spearman 개선 95% 구간: `[0.349304, 0.984766]`
- 참가자×조건 bootstrap MAE 감소 95% 구간: `[0.790638, 6.044785]`
- 대폭 개선 gate: `true`
- 선택 모델: portable `Huber(protocol + hinge 20/40/60/80, alpha=0.01, epsilon=1.2)`
- runtime primary weight: `0`

v2 gate 통과는 같은 pilot을 개봉한 뒤 설계한 회고적 nested-CV 결과다. 새로운 외부 target 재현 전에는 운영 농도곡선을 교체하지 않으며 concentration-delta, 혼합물, 레시피, 90% 후각 주장도 여전히 `false`다.

### 5.4 Ma 2021 outcome-unopened 이성분 혼합 강도

Ma et al. 공개 저장소는 72개 원료 정보가 든 첫 worksheet를 비결과 TSV로 별도 제공한다. 원본 3-sheet Excel을 받기 전에 PubChem 구조를 고정하고 72 choose 2인 2,556개 전 조합에 HumanPOM 단일향 예측, R2/Morgan 유사도, strongest-component·Ravia Weber–Fechner·RSS·완전가산 연산을 생성했다. 예측 SHA-256은 `9573050fb603b8630799f0500f775f01c4a0c7559228174817c304b711c1271b`이며 FreeTSA RFC 3161 시각은 `2026-08-26 11:01:35 UTC`다. 원본 취득은 그 뒤 시작했고 SHA-256은 `c540f18ba71b778c36756810fff38bdf177c2af9d593567a6dba57a30503c950`이다.

단, 관련 논문의 strongest-component/partial-addition 집계 결과는 source selection 중 확인됐다. 따라서 행 단위 평정 파일은 outcome-unopened였지만 연구 전체를 fully outcome-naive prospective blind라고 부르지 않는다. 사후 scope adjudication의 승인 표현은 `row-level outcome-unopened, publication-summary-aware external test`이며 예측·평정·지표·게이트는 바꾸지 않았다.

- 평가자 59명(공식 규약대로 subject 47 제외), 개별 평정 6,531개
- trial 222개, 반복을 접은 고유 이성분 혼합물 198개
- 사전선정 Ravia Weber–Fechner: Spearman `0.729566`, MAE `0.269321`, RMSE `0.336481`
- 사전고정 strongest-component: Spearman `0.726863`, MAE `0.265200`, RMSE `0.331461`
- participant×mixture bootstrap의 max-minus-primary MAE 구간: `[-0.009436, 0.002450]`
- 원 블라인드 mixture-operator integration gate: `false`

결과 개봉 뒤 만든 v2는 `IAB - mean(max(IA,IB))` 잔차만 예측하며 `IAmix`, `IBmix`, `PAB`, 사후 유의성 Group, Repeat를 특성에서 금지한다. 5회 반복 outer exact-pair 5-fold와 inner 4-fold 선택 결과 MAE `0.217662`, RMSE `0.274896`, Spearman `0.798020`으로 max 대비 MAE가 `17.93%` 낮았다. 참가자×혼합물 bootstrap MAE 개선 구간은 `[0.005587, 0.064628]`이다. 반면 각 all-components-cold fold의 훈련 부분 안에서 후보까지 다시 선택한 379개 pooled 예측은 Spearman `0.708903`, MAE `0.274030`으로 max의 `0.735636`, `0.264224`보다 나빴다. 따라서 seen-component new-pair gate만 `true`, strict all-components-cold와 전체 회고 개선 gate는 `false`다. 새 외부 원자료 재현도 `false`이므로 final runtime weight는 `0`이다.

이 결과는 측정된 두 단일 성분 강도가 주어진 0–10 이성분 혼합 강도 문제다. 자연어 기준향, 복합 향수, 시간축 헤드스페이스 또는 전체 후각 동일성을 검증하지 않는다.

### 5.5 미지 원료 강도와 Minnesota 결과 미개봉 검증

범용 단일향 v1은 Keller·Ravia·Bierling을 0–1 척도로 정규화하고 같은 분자의 모든 농도를 같은 fold에 배정했다. Ma 72개와 정확히 같은 구조는 적합 전에 제거했으며 원료 ID는 특성에 포함하지 않았다. Source-only 선택 모델 `ridge_physical_1000`은 Ma 단일향 순위를 HumanPOM `0.076067`에서 `0.233037`로 높였지만 MAE는 `0.076520`에서 `0.115946`으로 악화했고 혼합물에 큰 음의 bias를 만들어 v1 gate는 실패했다.

회고 hybrid v2는 `mean(HumanPOM) + 0.5·centered(HumanPOM) + 0.5·centered(universal)`로 절대 평균과 상대 구조 신호를 분리했다. Ma 단일향 MAE `0.069045`, 혼합 Fechner MAE `0.475906`; HumanPOM 대비 MAE 개선 bootstrap 구간은 각각 `[0.000105, 0.015135]`, `[0.005109, 0.067649]`다. 그러나 이 식은 Ma v1 실패를 본 뒤 설계했으므로 retrospective repair gate만 `true`, prospective external gate와 runtime primary weight는 `false/0`이다.

물성 연결은 다음처럼 분리해 시험했다.

- PubChem 575구조: XLogP 572, 증기압 188, 끓는점 513
- 끓는점만 있을 때 기존 Trouton-rule prior를 사용하고 measured/fallback/missing 플래그 분리
- transport v3: Ma component MAE `0.069643`, mixture MAE `0.479501`; v2보다 나빠 gate `false`
- bracketed ppm과 local low/high 문맥을 고친 역치 레지스트리: 556구조 중 97개
- threshold v4: Ma component MAE `0.069722`, mixture MAE `0.484911`; v2보다 나빠 gate `false`

Minnesota `10.13020/D68591`은 네 물질의 여러 농도를 5 ppm acetylpropionyl 표준과 비교한 독립 자료다. 네 대상과 표준 구조를 학습에서 제외하고 예측 SHA `d958be1199aba353c3cf36fdc817aeb08c9b2d4021acb0765f4d5c2906b26cae`를 FreeTSA `2026-08-26 17:54:47 UTC`에 봉인했다. 원시 파일 취득은 그 뒤 시작했다. Repository readme의 retest 360행과 실제 MD5 고정 파일 250행 차이는 additive acquisition adjudication으로 기록했고 원 평정·예측·scoring contract는 바꾸지 않았다.

- uncensored participant-compound match: 118
- 부모 pooled participant score의 hybrid log10 MAE `1.747698`, Spearman `0.360553`
- 봉인 계약대로 네 화합물을 균등 가중한 corrected hybrid MAE `1.668654`, compound-median Spearman `0.2`
- corrected constant 5 ppm baseline log10 MAE `1.070744`, compound-median Spearman `0`
- baseline-minus-primary MAE bootstrap `[−2.139177, 0.920841]`
- external concentration gate `false`; mixture-intensity gate `false`; runtime weight `0`

부모 scorer가 참가자-화합물 행을 pooled 평균해 봉인의 “four-compound mean”과 다르게 가중한 문제는 사후 scoring adjudicator에서 예측·평정·match interpolation을 유지한 채 화합물 균등 가중으로 바로잡았다. Corrected gate도 실패한다. 따라서 공개 일반 물성·희소 역치만으로 실험별 headspace와 절대 강도를 교정했다는 주장은 금지한다. 통과한 것은 Ma를 본 뒤의 진단적 centered hybrid뿐이며 운영 모델은 그대로다.

### 5.6 지속개선 prospective promotion 계약

지속개선 candidate는 `prepare-blind`와 `finalize-blind` 사이에 경계를 둔다. 전자는 historical external-human training CSV와 label이 없는 challenge input만 읽고 candidate·baseline 예측 및 local seal을 출력한다. 후자는 독립 timestamp와 outcome acquisition receipt가 prediction·seal·outcome 원본의 SHA-256을 모두 묶고, prediction seal 시각이 outcome 최초 열람보다 앞선 경우에만 평가한다. 결과 공개 뒤 재학습한 모델은 다음 challenge의 후보가 될 수는 있지만 방금 공개된 결과로 production 승격할 수 없다.

자동 평가 최소 조건은 다음과 같다.

- `evidence_class=prospective_external_human`, `label_origin=external_human_measurement`
- synthetic/model-generated/simulation/self-training/weak-label 행 0
- training/evaluation source·molecule·scaffold 교집합 0
- evaluation source 2개 이상, target 30개 이상, 행 100개 이상
- source→target 계층 bootstrap 2,000회 이상
- MAE 개선 95% 구간 하한 `>0`, Spearman 차이 하한 `>=-0.02`
- candidate와 baseline의 전체 challenge 농도 적용 범위 통과
- 측정 농도 범위 전체에서 농도 증가에 따른 강도 응답 비감소
- pickle 없는 JSON runtime, portable parity 최대 절대 오차 `<=1e-10`

Training/challenge/outcome 원본은 candidate bundle에 포함한다. Controller는 training label origin과 lineage, challenge-prediction 행 대응, 모든 metric과 계층 bootstrap을 다시 계산한다. 외부 취득기의 Ed25519 envelope가 timestamp response와 outcome receipt를 먼저 증명해야 shadow 평가가 가능하며, 한 번 소비한 dataset/outcome/행은 다른 challenger에 재사용할 수 없다.

모든 조건을 만족하면 shadow champion은 자동 갱신된다. Production은 acquisition signer와 다른 별도 allowlist의 `model_release_approver` Ed25519 서명까지 검증한 경우에만 최대 primary weight `0.05`로 승격한다. Registry는 atomic state와 hash-chain audit로 기록하고 런타임이 다시 전 artifact와 두 서명을 검증한다. 이 계약은 실제 향수 혼합물·자연어 레시피·인간 후각 90% 주장을 허가하지 않는다.

### 5.7 DREAM 2025 공개 혼합향 거리 검증

공개 저장소 commit `d4294949...`의 training/test/validation 원본과 공개 예측 파일 11개를 SHA-256으로 고정했다. Upstream README는 507개 training pair를 설명하지만 실제 `TrainingData_mixturedist.csv`에는 중복 없는 730행과 6개 source가 있으므로 실제 파일을 권위로 사용하고 차이를 보고서에 남긴다.

모델 선택은 training source 전체를 하나씩 제외하는 GroupKFold에서만 한다. 선택된 POM+RDKit ridge의 결과는 다음과 같다.

- 과거 숨김 test 46쌍: Pearson `0.4003`, RMSE `0.1006`; 공개 SOTA ensemble `0.5625`, `0.0785`보다 열세
- 별도 validation 50쌍: Pearson `0.6362`, RMSE `0.1121`; 공개 top-6 평균 `0.4736`, `0.1270`보다 개선
- validation test/retest Pearson `0.7415`, Spearman-Brown ceiling `0.9228`
- 인간 ceiling 정규화 Pearson `0.6894`, 참가자×혼합쌍 이중-bootstrap 95% `[0.3874, 0.8981]`; 90% 하한 실패

Validation RMSE 개선의 이중-bootstrap 하한은 0보다 크지만 Pearson 개선 구간은 `[-0.0176, 0.3352]`로 0을 지나 validation 종합 게이트도 실패한다. Test SOTA·분자/scaffold disjoint·내부 outcome-unopened·license gate 역시 실패한다. 따라서 이 결과는 기존 R2의 구조 병목을 확인한 연구 개선이며 production weight는 `0`이다.

### 5.8 DREAM odor-pair GNN 앙상블 v2

후속 진단은 MIT odor-pair source commit `32c25530...`, 21개 Python source tree, config, checkpoint, SMILES, 공개 training embedding을 각각 SHA-256으로 고정한다. OGB `1.3.6` CPU graph를 사용하며 730개 원 training embedding 전체의 재현 최대 오차는 `1.5e-5` 이하여야 한다. Training/test/validation 898개 embedding의 단일 hash를 확인하고, pair 순서를 바꿔도 260개 odor-pair feature가 byte-identical해야 하며, 두 Ridge member의 portable JSON 출력은 46+50행 sklearn 출력과 `1e-12` 이내여야 한다.

- 과거 test 46쌍: Pearson `0.4304`, Spearman `0.5252`, RMSE `0.09967`, MAE `0.08496`
- validation 50쌍: Pearson `0.6397`, Spearman `0.5517`, RMSE `0.11192`, MAE `0.09162`
- 인간 ceiling 정규화 Pearson `0.69325`, 참가자×혼합쌍 95% `[0.40189, 0.89711]`

기존 후보보다 네 지표가 양쪽 세트 모두 점 추정상 좋아야 point-Pareto gate를 통과한다. 하지만 가중치 `0.6/0.4`는 두 outcome을 본 뒤 선택했으며, test pair bootstrap과 validation 참가자×pair bootstrap의 RMSE·Pearson 개선 하한이 모두 0보다 커야 하는 통계 게이트는 실패한다. 이 결과는 새 prospective 증거가 아니며 production/90% gate와 runtime weight는 `false/0`이다.

### 5.9 농도·헤드스페이스·혼합 관능 v1

`headspace_sensory_hub_v1.db`는 원천 29개의 hash와 Pyrfume commit, EPA OPERA
CC0 S1.zip hash를 확인한 뒤 구축한다. SQLite integrity/foreign-key 검사와 다음
고정 count를 모두 요구한다: 분자 1,642, 물성 27,283, 자극 2,689, 구성분 21,708,
Ravia 희석 1,473, 관능 관측 109,688. Abraham 단위는 반드시
`log10_inverse_ppmv`이며 molar 역치로 바꾸지 않는다.

Keller 농도 지수는 CID hash fold 0의 102개 분자를 모델 선택에서 제외하고,
저→고 및 고→저 방향 모두에서 농도 무시 MAE보다 낮으며 paired bootstrap
`baseline-candidate` MAE 개선 구간 하한이 0보다 커야 한다. 이 gate는 통과했다.

DREAM 적용은 증기압 measured, boiling-point fallback, missing을 분리한다. 모든
외부 세트의 Pearson·Spearman은 증가하고 RMSE·MAE는 감소해야 point-Pareto다.
현재 후보 수는 0이고 통계·90%·production gate도 모두 실패한다. 완전 nested
training-source OOF residual은 alpha `100000`, scale `1.0`을 선택했지만 외부
point-Pareto를 통과하지 못한다. Hub, calibration, DREAM 보고서, 세 builder
script의 hash가 `audit_data.py`에서 다시 결합되지 않으면 감사는 실패한다.

OPERA 증기압 보간은 publisher train/test보다 보수적인 scaffold 계약을 우선한다.
고리가 없는 각 분자를 독립 scaffold로 취급할 수 없으며 모두 하나의 acyclic
그룹으로 묶는다. 이때 training과 겹치지 않는 test는 93분자이고 R² `0.7493`,
RMSE `1.7347`, q95 `3.9060`이다. 최소 300분자, R² 0.75, RMSE 1.5, q95 2.5
gate가 실패하므로 portable coefficient가 생성돼도 보간 사용과 production
가중치는 `false/0`이다.

### 5.10 DREAM protocol-conditional v4

Exact 10×10·무중첩 라우터는 test/validation의 Pearson·Spearman 증가와 RMSE·MAE
감소를 모두 요구한다. 현재 점-Pareto gate는 통과하지만 후보군 4,960개와
라우팅 규칙이 공개 결과를 본 뒤 선택됐으므로 모든 bootstrap 구간은 사후
기술통계다. 원 source를 통째로 제외한 exact-10 OOF는 23개뿐이며, Ravia 5/6
87쌍에는 같은 exact-10 프로토콜이 0개다. Validation 참가자×pair bootstrap의
Pearson·RMSE 개선 하한은 0을 넘지 못하고 human-ceiling 정규화 하한도 `0.4258`
이다. 따라서 90%·통계·production gate는 `false`, runtime weight는 `0`이다.

### 5.11 DREAM attention set encoder v5

성분 순열 불변 self-attention 후보는 Pair-GNN·OpenPOM·RDKit component feature와
descriptor reconstruction 보조 손실을 사용한다. 4 config×3 seed의 고정 80 epoch
검색 결과 모든 config가 v4의 test/validation Pearson·Spearman·RMSE·MAE 교집합을
통과하지 못했다. 최선 후보도 8개 비교 조건이 모두 false이므로 즉시 reject하고
체크포인트·portable runtime·운영 가중치를 생성하지 않는다.

### 5.12 생성 조향식 전향적 블라인드 v1

공개 분자·혼합물 회고 자료와 별도로, 자연어에서 생성된 정량 조향식의 종단 인간
유사도를 시험하는 `prospective_formula_blind_study.py`를 둔다. 결과 및 제조 실행
증거 파일이 없는
상태에서 안전 제약 조향식 24개, 120 비교쌍, 고정 예측, formula-key, 참가자 80명
배정 2,400행을 생성하고 모든 파일 및 구현 SHA-256을 `seal.json`에 묶는다.
RFC3161 timestamp는 이 seal 바이트에 직접 검증되어야 한다.

최종 결과는 allowlist의 독립 `sensory_laboratory` Ed25519 서명이 연구 ID,
protocol hash, seal hash, 결과 CSV, 24개 실제 batch 칭량, 240개 vial, base/headspace
조건과 lot별 COA/SDS/IFRA 원문을 함께 증명해야만 읽는다. 행·참가자·쌍·
시료 코드·범위를 sealed template과 전수 대조하고, 참가자×쌍 crossed bootstrap
5,000회로 일반 조향식 100쌍의 `100-MAE`, Pearson, Spearman을 계산한다. 동일·
농도 대조 20쌍은 주 정확도 계산에서 제외하고 별도 품질관리로만 사용한다. 90%
gate는
절대 유사도 정확도 하한 90, 두 상관 하한 0.90, 동일조성 대조 평균 90 및 하한
85를 모두 요구한다. 결과는 one-use ledger에 기록하며 결과 공개 뒤 재적합은 이
연구의 판정에 사용할 수 없다. 사람 결과가 없으면 gate는 미실행이다. 상세 계약은
`PROSPECTIVE_FORMULA_BLIND_STUDY.md`를 따른다.

## 6. 과거 개발 진단의 해석

`benchmarks/physsim_r2_transfer_*`, `physsim_r2_strong_baselines.json`, `physsim_r2_ensemble_validation.json`, `system_ablation_report.json`의 수치는 과거 개발 시점의 역사적 mixture-pair/프록시 진단이다. 이 수치들은 현재의 all-components-held-out, 외부 source-disjoint, formulation-capability 계약 이전에 생성된 자료일 수 있으므로 release 근거나 실제 후각 성능으로 재사용하지 않는다.

`benchmarks/commercial_v1_readiness.json` 역시 자동화 게이트의 당시 상태를 보존한 기록이다. 실제 인간 후각 유사도는 그 파일에서도 측정·추정·인증된 값이 아니다.

## 7. 최소 감사 항목

- 모델·정규화기·원료 구성·규제 자료·기준향 증거의 SHA-256 및 버전
- 계산 프록시와 실제 관능 레코드의 별도 저장소 및 별도 필드
- 정확한 reference target 계약 결과
- 입력 coverage, applicability, OOD, 불확실성, 적용된 가중치
- 외부 검증 source overlap audit 및 split 계약 버전
- 실제 관능이 있을 때만 서명자, 원시 결과, 사전 등록 식별자, 처방 지문

이 항목이 없으면 결과는 연구용 후보이며 상용 관능 성능 주장에 사용할 수 없다.
