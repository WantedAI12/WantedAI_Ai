# 과학 모델과 계산 경계

## 목적

이 시스템은 자연어 요청을 안전·가격·가용성 제약 안의 향료 농축액 후보로 변환하고, 실제 제조 전에 후보를 **계산 프록시로 정렬**한다. 모델은 사람의 전체 후각계, 실제 공급 로트, 완제품의 모든 매트릭스 상호작용을 직접 측정하지 않는다. 따라서 출력은 실제 인간 후각 유사도나 “90% 재현”의 인증이 아니다.

## 입력과 출력

### 입력

- 자연어 향 요청과 명시적 회피 조건
- 처방 가능한 원료 카탈로그와 사용 제한
- 원료별 구조, 공개 물성, 역치, 조성 prior, 공급·규제 증거의 가용 범위
- 등록된 정량 기준향이 있을 때의 기준향 조성·제품 조건

### 출력

- 안전 제약을 통과한 레시피 후보
- 시간축 물성·헤드스페이스 기반 프록시 점수와 5% 하한
- 데이터 coverage, applicability, OOD/불확실성 상태
- 농도 반응 및 R2 가지의 상태와 실제 적용 가중치

기준향 조성 증거가 없으면 결과는 자연어 기반 후보 생성이며, 특정 상용 향수 재현의 점수로 해석하지 않는다.

## 전체 계산 흐름

1. 자연어 파서가 향 차원, 강도, 피라미드, 금지 조건을 추출한다. 다국어 향 온톨로지와 의미 표현은 후보 탐색을 보조하지만 관능 정답 데이터가 아니다.
2. 제약 최적화기가 저가·가용·저위험·제형 가능 원료만으로 복수 후보를 만든다.
3. 각 후보를 조성·희석·시간 조건에서 결정론적 물성/헤드스페이스 프록시로 평가한다.
4. coverage, applicability, OOD, 불확실성을 계산해 모델-domain 상태를 기록한다. 이 상태는 인간 유사도 승인이 아니다.
5. 실제 기준향 평가가 필요하면 등록된 정량 기준향과만 비교한다. 텍스트에서 만든 목표 처방이나 노트 목록은 대체 기준향이 될 수 없다.

## 결정론적 물성·헤드스페이스 가지

이 가지는 조성에 따른 휘발·지속·시간축 변화를 근사한다. 직접 연결된 물성은 출처와 단위를 유지하고, 값이 없는 경우에는 구조 기반 사전분포/추정치를 별도로 표기한다. 추정치는 측정값으로 승격되지 않는다.

주요 계산 요소는 다음과 같다.

- 조성 및 원료 활성 농도
- 증기압, 분배, 극성·친유성 등의 공개 물성 또는 추정치
- 희석과 제품 농도에 따른 헤드스페이스 프록시
- odor activity의 역치 기반 근사
- 초기·중기·후기 시간점의 유사도와 Monte Carlo 불확실성
- 0·15·60·240·480분의 19개 향축 분포와 오프닝 대비 상대 강도
- 원료별 도포 표면 잔존량, 추정 반감기, headspace·후각 기여의 시간 곡선

원료별 잔존 농도는 도포 표면의 1차 증발 프록시다. 밀폐 용기 안의 농도 변화나
GC-MS 측정값이 아니다. 비이상 용액, 제품 베이스, 포장, 산화, 실제 로트 GC-MS,
모든 수용체 상호작용은 완전하게 모델링되지 않는다. 그러므로 이 가지는 물리적으로
그럴듯한 후보의 우선순위 도구이지 관능 측정기가 아니다.

## 농도 반응 가지

농도 반응 자료는 단일 분자의 강도와 혼합물·완제품의 감각 효과를 구분해서 다룬다. 구조별 효과 또는 전역 농도 효과가 독립 혼합물/제품 holdout에서 검증되지 않으면 그 효과는 최종 프록시 점수에 반영하지 않으며 가중치는 `0`이다.

2026-08-27 범용 강도 연구는 이 경계를 더 세분화했다. 모델 구조는 `단일 원료 상대 강도 → cohort scale anchor → 강도 가중 선형 향질 혼합 → 제한적 혼합 잔차`로 분리한다. Ma 회고 자료에서는 centered universal/HumanPOM hybrid가 기존 HumanPOM보다 MAE를 낮췄지만, 결과 미개봉 Minnesota 농도 매칭에서 절대 농도 오차가 기준선보다 컸다. 그러므로 상대 강도·향질 프로필 진단과 절대 농도/headspace 권한은 별개다.

PubChem 증기압·끓는점은 출처가 있는 분자 일반값이며 실제 제품 매트릭스의 headspace가 아니다. 끓는점 fallback은 Trouton-rule prior로 명시하고 measured/fallback/missing을 분리한다. 공개 역치도 방법·매질·패널이 이질적이며 결측이 많다. Transport v3와 threshold v4가 독립 기준을 통과하지 못했으므로 두 가지 모두 운영 점수 가중치가 `0`이다.

이 규칙은 개별 데이터셋의 과거 MAE나 상관계수보다 우선한다. 단일 분자 희석 자료가 있다고 해서 향료 혼합물의 실제 헤드스페이스나 인간 후각 조합 효과를 검증한 것은 아니다.

## 지속개선 champion/challenger

지속개선은 모델 출력으로 모델을 재학습하는 self-training이 아니다. 과거 외부 인간 측정값만 training lake에 들어가며, 다음 외부 challenge의 결과 열은 예측 시점에 존재할 수 없다. Challenger와 현재 baseline의 예측 파일, portable runtime, model manifest를 먼저 고정하고 외부 timestamp를 받은 뒤 outcome을 취득한다.

자동 shadow 승격은 다음 교집합이 모두 빈 집합일 때만 가능하다.

- training source와 evaluation source
- training molecule과 evaluation molecule
- training scaffold와 evaluation scaffold

평가 단위는 최소 2개 외부 source, 30개 target, 100행이다. Source를 재표집한 뒤 각 source 안에서 target을 다시 재표집하는 계층 bootstrap 95% 구간에서 baseline-minus-candidate MAE의 하한이 0보다 커야 하고, candidate-minus-baseline Spearman 하한은 `-0.02` 이상이어야 한다. Portable JSON 경로와 봉인 예측 경로의 최대 절대 오차는 `1e-10` 이하여야 한다. 이 조건은 benchmark 선택 편향과 배포 구현 불일치를 줄이기 위한 최소 계약이지 후각 재현율의 물리적 증명이 아니다.

Outcome acquisition도 controller가 자기 증명할 수 없다. Allowlist의 `external_evidence_acquirer`가 dataset receipt, prediction, seal, outcome과 timestamp response를 Ed25519로 묶는다. Training CSV와 label 없는 challenge input도 최종 candidate에 포함되며 controller가 label origin·evidence class·lineage·행 대응을 원본 bytes에서 다시 계산한다. 한 번 평가에 사용된 dataset ID, outcome hash 또는 source/target/row scope는 다른 challenger 선택에 재사용하지 않는다.

Production 가중치는 controller 자체가 만들 수 없다. Acquisition signer와 다른 독립 `model_release_approver`의 Ed25519 envelope가 candidate manifest, runtime, training/challenge 원본, 평가 보고서, acquisition authorization, dataset receipt, prediction seal, timestamp response, prediction과 outcome 원본을 모두 해시로 묶어야 한다. API/worker는 시작할 때 registry hash chain과 모든 bytes와 두 서명을 다시 확인한다. 어느 하나라도 다르면 bundled diagnostic 모델의 weight `0`으로 조용히 downgrade하지 않고 설정 오류로 시작을 거부한다.

## R2 PhysSim 연구 가지

### DREAM 2025 혼합향 거리 보강

DREAM 공개 release의 730개 실제 training pair와 과거 숨김 46개 test pair, 별도 50개 test/retest validation pair를 회고 benchmark로 연결한다. Candidate는 MIT POMMix 196축 embedding, OpenPOM 138축의 mean/max/noisy-or mixture aggregate, 고정 217개 RDKit descriptor와 Morgan/overlap/size 대칭 feature를 사용한다. 모델·alpha·feature set 선택은 6개 training source를 통째로 제외하는 fold에서만 수행하며 pair 순서를 바꿔도 feature가 byte-identical하다.

선택된 `ridge_pommix_pom_rdkit_30000`은 validation 점 추정 Pearson `0.6362`, RMSE `0.1121`로 frozen R2 `0.1677/0.1285`와 공개 top-6 평균 `0.4736/0.1270`보다 낫다. 참가자×혼합쌍 이중 bootstrap에서 RMSE 개선은 유지되지만 Pearson 개선 95% 구간은 `[-0.0176, 0.3352]`로 0을 지난다. 과거 숨김 test에서도 Pearson `0.4003`, RMSE `0.1006`으로 공개 SOTA ensemble `0.5625/0.0785`보다 낮다. Validation test-retest ceiling으로 정규화한 상관은 `0.6894`, 이중-bootstrap 구간은 `[0.3874, 0.8981]`이다. 따라서 validation·90%·production gate는 모두 실패한다.

Upstream root license가 명시되지 않았고 결과가 이 구현 전에 공개됐으며 training과 test/validation의 구성분·scaffold가 약 99% 겹친다. 최종 candidate ranking 코드는 학습 source 홀드아웃만 사용하지만 표현·후보 개발은 공개 외부 결과를 이미 본 outcome-aware 회고 과정이다. Portable coefficient JSON은 feature 계약 hash와 수치 유효성을 fail-closed로 검사하고 직렬화 후 96개 외부 행에서 sklearn과 `1e-12` 이내 등가성을 확인하지만, 재현 연구용일 뿐 runtime weight는 `0`이고 wheel에 포함하지 않는다.

### Odor-pair GNN 이중 공간 앙상블 v2

MIT `laurahsisson/dream`의 3-GIN+Set-Transformer blend encoder와 DREAM 공개 fine-tuned checkpoint를 고정 commit·source-tree·checkpoint hash로 재현한다. Training 730개 전체에서 원 공개 embedding과 최대 차이는 `1.15e-5`이고, training/test/validation 총 898개 mixture embedding과 260개 대칭 pair feature를 단일 hash로 고정한다. 기존 2,664개 POMMIX/OpenPOM/RDKit feature에 이를 결합한 Ridge `alpha=30000/100000`을 `0.6/0.4`로 평균하며 portable JSON의 96행 sklearn 등가 오차는 `1e-12` 이하다.

이 v2는 두 공개 세트에서 Pearson·Spearman·RMSE·MAE 점 추정이 모두 좋아졌지만 test/validation outcome을 본 뒤 가중치를 선택했다. Test Pearson은 `0.4304`, validation은 `0.6397`, 인간 ceiling 정규화는 `0.6932`이고 이중-bootstrap 구간은 `[0.4019, 0.8971]`이다. 개선 구간은 모두 0을 지나므로 통계·90%·production gate는 실패하며 runtime weight는 `0`이다.

### 농도·헤드스페이스·혼합 관능 v1

EPA OPERA 실측 SDF와 Pyrfume 6개 공개 관능 archive를 별도 연구 허브로 연결했다.
Keller 480분자의 두 농도에서 학습한 log-response 지수 `0.128269`는 CID-hash
holdout 102분자의 저→고농도 MAE를 농도 무시 기준 `26.1929`에서 `15.8829`로,
고→저농도 MAE를 `9.9761`로 낮췄다. 두 paired bootstrap 개선 구간의 하한은
0보다 크다. 이 통과 범위는 anchor 강도가 있는 동일 단일분자의 농도 이동뿐이다.

DREAM 구성분 204개 중 120개는 실측 증기압, 38개는 측정 끓는점 fallback,
46개는 결측이었다. Equal-liquid 가정의 headspace 후보는 test RMSE를 낮췄지만
validation RMSE를 `0.11192`에서 `0.12400`으로 악화했고 point-Pareto 후보가
0개였다. Source-holdout residual 선택도 scale `0`을 선택했다. 정량 조성이 없는
혼합물에 일반 물성을 적용해 실제 headspace라고 부르지 않는다. 완전 nested
source-holdout residual은 scale `1.0`을 골라 test와 validation Pearson을 높였지만
validation RMSE/MAE와 test Spearman을 동시에 개선하지 못했다. 따라서 production
weight는 `0`이다. 전체 계약은
[HEADSPACE_SENSORY_DATA_V1.md](HEADSPACE_SENSORY_DATA_V1.md)에 고정한다.

OPERA 증기압 Ridge는 publisher random test R² `0.9048`이지만, 모든 acyclic
분자를 하나의 보수적 scaffold 그룹으로 묶은 완전 scaffold-disjoint test에서는
93분자, R² `0.7493`, RMSE `1.7347`, q95 `3.9060`이었다. 엄격 보간 gate가
실패하므로 결측 증기압을 측정값처럼 채우지 않으며 이 runtime도 weight `0`이다.

### DREAM protocol-conditional search v4

외부 test는 전부 10×10성분·무중첩이고 validation은 9–11성분·무중첩인 반면,
학습 730쌍은 1–43성분과 다양한 중첩률을 함께 포함한다. 이 적용범위 차이를
이용해 exact 10×10·무중첩일 때만 compact-feature ExtraTrees 잔차를 0.35
결합하고 그 밖에는 Pair-GNN v2를 유지하는 라우터를 회고 탐색했다.

후보는 test Pearson/RMSE `0.4304/0.09967 → 0.5067/0.08875`, validation
`0.6397/0.11192 → 0.6560/0.11191`로 두 세트의 Pearson·Spearman·RMSE·MAE를
모두 점 추정상 개선했다. Source-group OOF의 exact-10 표본 23개에서도 Pearson
`0.2172 → 0.5232`, RMSE `0.1833 → 0.1699`였다. 그러나 4,960개 후보를 이미
공개된 test/validation 결과로 탐색했고, validation 통계 개선 구간은 0을 지난다.
Human-ceiling 정규화 Pearson은 `0.7108`, 사후 95% 구간 `[0.4258, 0.9082]`로
90% gate를 실패한다. 따라서 이 라우터는 연구 진단이며 runtime weight는 `0`이다.

후속 v5는 성분별 Pair-GNN 128축, OpenPOM 138축, RDKit PCA 32축을 순서 없는
self-attention으로 결합하고 4개 구조×3시드를 GPU에서 비교했다. 최선 config도
test/validation Pearson `0.3327/0.5566`, RMSE `0.1014/0.1715`로 v4보다 낮아
8개 비교 지표를 모두 실패했다. Attention 평균집계 대체는 이 데이터 규모에서
과적합되어 탈락했으며 runtime artifact를 만들지 않는다.

R2는 역사적 mixture-pair similarity를 모델링하는 연구용 신경망이다. 설계는 217개 RDKit descriptor, 최대 50개 분자 구성분, 16-step 잠재 동역학, 대칭 pair head를 포함한다. 잠재 변수와 학습된 상수는 질량·전하·위치의 실측값이 아니다.

### 학습 라벨 계약

공개 odor descriptor 아카이브는 대체로 positive assertion 목록이다. 표현이 없다는 사실은 “냄새가 없다”는 관측이 아니므로, 미기재 기술어를 dense BCE의 음성 라벨로 사용하지 않는다. 현재 학습 계약은 다음을 요구한다.

- 원료·라벨별 source-backed positive observation mask
- 라벨별 source lineage
- 미관측값을 음성으로 확정하지 않는 non-negative positive-unlabeled(PU) risk

### 검증 계약

엄격한 분자/scaffold 검증은 validation pair의 **모든 구성분**이 held-out인 경우만 점수와 모델 선택에 사용한다. 구성분 일부만 held-out인 pair는 진단용이며 strict 결과로 표기하지 않는다.

Ravia 같은 외부 source는 분자와 Bemis–Murcko scaffold 모두에 대해 다음 집합과의 overlap audit을 통과해야 한다.

- descriptor pretraining
- mixture fine-tuning
- descriptor normalizer fitting

중첩이 있으면 “완전한 외부 zero-shot 검증”이라고 주장하지 않으며 release gate는 실패한다.

### 현재 런타임 상태

현 R2 체크포인트는 구성분 존재와 구조 기반 입력을 사용한다. 상대 배합비, 완제품 향료 농도, 시간/헤드스페이스 궤적을 직접 인코딩하지 않는다. 따라서 이 세 capability를 명시하고 검증하는 manifest가 없으면 런타임은 R2 최종 가중치를 `0`으로 설정한다. 체크포인트가 로드되더라도 최종 프록시를 조정하지 않는 것이 의도된 fail-closed 동작이다.

운영 경로는 두 원 체크포인트의 tensor를 NPZ로 내보낸 뒤 manifest, descriptor 계약, ensemble 가중치와 원본 SHA-256을 묶는다. exact-erf GELU, LayerNorm epsilon, inference dropout 계약을 고정한 순수 NumPy forward pass를 사용하므로 Torch·pickle 객체 역직렬화가 필요 없다. 이 변경은 배포 안전성과 재현성을 높이지만 모델의 과학적 적용 범위를 넓히지는 않는다.

새 R2 모델이 점수에 기여하려면 PU 계약, all-components-held-out split, 외부 source-disjoint audit, artifact hash, applicability/OOD 보정, 그리고 세 formulation capability가 모두 승인되어야 한다.

## 적용 범위와 불확실성

출력에는 다음 진단을 함께 보존한다.

- 원료 성분 및 구조 coverage
- descriptor domain coverage
- 모델 applicability
- Monte Carlo 분위수와 시간축 불확실성
- 앙상블 member disagreement 및 conformal 구간
- 실제 적용된 concentration/R2 가중치

coverage 부족, OOD, member disagreement 초과, 불확실성 초과, 계약·해시 불일치에서는 해당 가지의 가중치를 `0`으로 하거나 프록시 승인을 차단한다. 이는 모델의 정확성을 보증하는 것이 아니라 알 수 없는 영역에서의 과신을 제한한다.

## 기준향과 실제 관능의 경계

실제 향 재현을 평가하려면 기준향의 정량 조성, 제품 카테고리, 향료 농도, 제품 베이스, 근거 문서와 버전이 모두 등록되어야 한다. 이 조건이 맞지 않으면 시스템은 exact reference similarity를 계산하지 않는다.

정량 기준향이 연결된 비인간 통과 상태도 기준향 PhysSim 점수, 물성 적용범위, 시간축 프록시 하한을 모두 요구한다. 텍스트 목표 점수만 높아서 기준향 재현 게이트를 통과하는 경로는 차단한다.

`actual_olfactory_similarity_score`와 `actual_olfactory_lower_bound_95`는 독립적이고 서명된 인간 관능 결과만 기록할 수 있다. 계산 진단, R2 historical similarity, 내부 기준 레시피, 노트 기반 참조 데이터는 이 필드의 근거가 아니다.

별도로 연결된 Bushdid 인간 행동 보정은 해당 supplemental 자극 프로토콜의 3AFC 판별 확률 endpoint다. 봉인 최종 순위 Spearman은 `0.6267`, 인간 ceiling 정규화 값은 `0.7326`이었고 90% 향 유사도 주장을 승인하지 않았다. 보고서 해시와 연구 프로토콜이 일치하는 감사 경로에서만 값을 내며 일반 처방에는 적용하지 않는다.

## 알려진 한계

- 공급 로트별 GC-MS와 배치 드리프트가 완전하게 연결되지 않았다.
- 비이상 용액, 베이스·포장·산화·보관 조건을 완전하게 예측하지 않는다.
- 공개 역치와 descriptor는 출처별 조건 차이와 누락 편향을 가진다.
- 자연 원료 조성은 실제 공급 로트 분석이 아니라 참고 prior일 수 있다.
- 역사적 mixture-pair 상관은 text-to-odor 또는 사람의 향 동일성 지표가 아니다.

결론적으로, 시스템은 추적 가능한 계산 후보 생성·거부·정렬 도구다. 인간 후각 유사도의 최종 근거는 별도 관능 연구와 정확한 기준향 증거에서만 나온다.
