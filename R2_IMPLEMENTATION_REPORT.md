# R2 PhysSim 구현 및 상태 보고서

기준일: 2026-07-28
대상: `perfumery-ai-core`의 R2 연구 가지

## 현재 결론

R2는 역사적 mixture-pair similarity를 위한 연구 모델로 보존되어 있다. 현재 패키지의 레거시 체크포인트는 배합비, 완제품 농도, 시간/헤드스페이스 궤적을 직접 입력하지 않으므로 최종 프록시 점수에 기여하지 않는다.

런타임 계약은 다음 중 하나라도 없거나 실패하면 `approved_primary_score_weight=0` 및 `applied_primary_score_weight=0`으로 fail-closed 한다.

1. direct formulation capability manifest
2. positive-unlabeled label-supervision 계약
3. all-components-held-out molecule/scaffold split 계약
4. 외부 source의 molecule/scaffold overlap audit
5. artifact hash, applicability, OOD, 불확실성 게이트

따라서 레거시 아티팩트가 로드되는 것, 과거 benchmark 파일이 존재하는 것, 또는 R2가 수치를 반환하는 것은 production 점수 기여 승인이 아니다.

## 아키텍처 범위

레거시 구현은 논문의 체크포인트 호환 구조를 유지한다.

- 파라미터 수: 162,059
- 입력: 217개 RDKit descriptor와 구성분 존재 기반 분자 집합
- 최대 구성분: 50개 분자
- 잠재 동역학: 16 step, soft-core 거리 안정화
- 비교: 순서 교환에 불변인 symmetric pair head

이 구조의 잠재 질량·전하·위치·속도와 학습 상수는 물리 실측량이 아니다. 또한 구성분 존재 기반 입력은 동일 성분 집합에서 비율만 달라진 처방을 직접 구별하지 못한다. 이 한계가 formulation capability contract를 요구하는 이유다.

## 학습 데이터와 positive-unlabeled 계약

공개 단분자 odor archive의 기술어는 대부분 양성 서술의 목록이다. 기술어 부재는 음성 관측이 아니므로 현 구현은 다음을 사용한다.

- canonical molecule별 positive observation mask
- 기술어별 source lineage
- source-backed positive assertion만을 양성으로 사용하는 non-negative PU risk

즉, “라벨이 없음”을 0으로 채운 dense BCE로 학습하지 않는다. 새 학습 산출물은 PU 계약 버전과 objective를 manifest에 기록해야 한다.

## split 및 외부 검증 계약

`strict`라는 명칭은 다음 조건을 뜻한다.

- validation pair의 모든 구성분이 held-out이다.
- molecule protocol은 held-out 분자가 training, pretraining, normalizer에 없다.
- scaffold protocol은 held-out Bemis–Murcko scaffold가 위 집합에 없다.

일부 구성분만 held-out인 pair는 mixed-component 진단용이며 strict 점수·모델 선택에 사용하지 않는다. 레거시 보고서가 이 split contract version을 포함하지 않으면 release trainer는 거절해야 한다.

외부 Ravia 평가는 label을 선택에 쓰지 않는 것만으로 충분하지 않다. 외부 분자와 scaffold는 descriptor pretraining, Snitz fine-tuning, normalizer fitting 각각과 중첩되지 않아야 한다. audit은 원시 목록 대신 overlap 수와 재현 가능한 해시를 보존하며, 중첩이 하나라도 있으면 외부 zero-shot release gate가 실패한다.

## 역사적 혼합물 진단: 보존하되 인간 후각 승인 근거로 사용하지 않음

다음 수치는 고정 Snitz all-components-held-out 분할에서 재현한 **역사적 혼합물 진단**이다. 두 체크포인트의 `0.3:0.7` 가중치는 0.1 간격의 개발 전용 grid에서 분자·스캐폴드 pooled/fold-mean Spearman의 균형 평균으로 선택했다.

| 보고 항목 | 기록된 값 | 해석 제한 |
|---|---:|---|
| ensemble molecule-disjoint Spearman | 0.7373 | 역사적 mixture-pair 상관; 인간 후각 정확도 백분율이 아님 |
| MoLFormer molecule-disjoint Spearman | 0.5719 | 동일 strict pair·개발 선택 adapter 비교값 |
| molecule 반복 차이 95% 구간 | [+0.0182, +0.3448] | 겹치는 5개 분할의 기술적 재표집 구간 |
| molecule 고유 65쌍 차이 95% 구간 | [+0.0145, +0.4149] | 중복 record_id 평균 후 대응 bootstrap; 공유 분자는 독립 cluster가 아님 |
| Morgan/Tanimoto molecule baseline | 0.4375 | 위와 같은 과거 분할·데이터 조건의 비교값 |
| ensemble scaffold-disjoint Spearman | 0.7273 | 실제 향 재현 또는 인간 관능 성능이 아님 |
| MoLFormer scaffold-disjoint Spearman | 0.6487 | 동일 strict pair·개발 선택 adapter 비교값 |
| Morgan/Tanimoto scaffold baseline | 0.6154 | 위와 같은 제한 적용 |
| Ravia transfer 수치 | 과거 보고서 참조 | source-disjoint audit을 통과하지 않은 경우 release 근거가 아님 |
| conformal q95 | 0.39694 | 개발 잔차로 재보정한 역사적 label 불확실성 진단이며 인간 관능 구간이 아님 |

관련 파일은 `benchmarks/physsim_r2_transfer_*`, `benchmarks/physsim_r2_strong_baselines.json`, `benchmarks/physsim_r2_ensemble_validation.json`이다. 이 파일들의 숫자는 재현성 조사에는 사용할 수 있지만, 현재 runtime nonzero weight 또는 실제 인간 후각 성능을 주장하는 데 사용하지 않는다.

## Bushdid 인간 행동값 블라인드 외부 검증

2026-08-18에 frozen R2 ensemble의 예측을 Bushdid 2014 인간 행동 파일을 열기 전에 생성·해시 봉인한 뒤, 26명 원시 odd-one-out 응답으로 채점했다. 인간값과 무관하게 분리한 보정 52개, 최종 208개, 단분자 대조 4개 자극을 사용했다.

| 봉인된 최종 시험 지표 | R2 | 성분 겹침 기준선 | R2 - 기준선 |
|---|---:|---:|---:|
| 인간 정답률 Spearman | 0.2917 | 0.6267 | -0.3350 |
| FDR 판별가능 자극 ROC-AUC | 0.6394 | 0.8126 | -0.1731 |
| 보정 후 인간 정답률 MAE | 14.003%p | 12.016%p | - |

R2 Spearman의 참가자×자극 bootstrap 95% 구간은 `[0.1150, 0.4007]`이므로 0보다 큰 인간 전이 신호는 있었다. 그러나 성분 겹침 기준선과의 차이 구간은 `[-0.4384, -0.1516]`이고, 인간 noise ceiling 대비 점수는 `0.3410` (`95% [0.1341, 0.4698]`)이므로 사전 정의한 90% 게이트는 **FAIL**이다.

이 검증에서 Bushdid 행동 라벨은 학습·선택·분할에 사용되지 않았지만, 128개 분자와 23개 scaffold 모두 선언된 모델 원천 중 적어도 하나에 등장했다. 따라서 이는 실제 인간값 기반 behavior-label-blind 전이 결과이지 molecule/scaffold-disjoint 결과가 아니며, 사람이 생성 레시피를 직접 맡은 종단 시험도 아니다. 세부 프로토콜, 봉인 시각, 해시, 통계법은 `HUMAN_BLIND_VALIDATION_PROTOCOL.md`에 고정했다.

## 농도 반응과 R2의 관계

농도 반응 가지는 R2와 독립적으로 평가된다. 단일 분자 희석 강도 자료가 존재해도 구조별·혼합물별·완제품별 효과는 별도 holdout 증거가 없으면 가중치 `0`이다. R2의 역사적 pair similarity도 그 공백을 메우지 못한다.

두 가지 모두 독립적인 formulation-level 검증을 통과하기 전에는 결정론적 물성·헤드스페이스 프록시를 대체하지 않는다.

## 재승인을 위한 최소 산출물

새 R2 모델을 학습·배포할 때는 다음을 하나의 버전 묶음으로 생성해야 한다.

1. 배합비, 완제품 농도, 시간/헤드스페이스를 직접 입력하는 모델 및 capability manifest
2. PU-safe 학습 레코드와 source/label lineage
3. all-components-held-out molecule/scaffold report
4. Ravia 또는 다른 외부 source에 대한 molecule/scaffold disjoint audit
5. 개발 전용 모델 선택, final holdout, 다중 시드, 강한 기준선, OOD/conformal 보정
6. checkpoint·normalizer·source 파일·report의 SHA-256
7. 실제 향 재현을 주장하려면 별도로 등록된 정량 기준향과 독립·서명된 인간 관능 결과

마지막 항목이 없으면 R2 결과는 historical mixture similarity 연구 결과이며, `actual_olfactory_similarity_score`나 실제 후각 90%의 근거가 될 수 없다.
