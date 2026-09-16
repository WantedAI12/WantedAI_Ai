# V85 재생 기반 MLP 학습 탐색

개선형 MLP의 학습 경로에 기록 재생 기반 자원 배분을 연결했다. 기존 향수/로션 생성 API와 운영 배포는 변경하지 않았다.

## 구현

- `fragrance_ai/research/replay_search.py`: 이미 관측한 단계만 읽는 정책, 기록 재생, 실제 실행, 숫자 정책 개선.
- `fragrance_ai/research/mixture_replay_training.py`: 독립적인 모델·옵티마이저·난수 상태를 가진 실제 학습 경로.
- `scripts/train_replay_mlp_v85.py`: 3개 시드 × 5개 혼합물 분할의 중첩 평가. 정책 학습 자료는 각 외부 분할의 학습 부분 안에만 둔다.
- `scripts/finalize_replay_mlp_v85.py`: 검증 자료에서 선택한 구조·에포크로 전체 학습 자료에 단일 모델을 재적합한다.
- `fragrance_ai/recommender/replay_mixture.py`: 학습된 단일 모델의 CPU 실행. 분자 구조 검증 캐시는 배합비·결과 캐시와 분리한다.

원 논문처럼 외부 LLM이 정책 코드를 수정하는 방식이 아니다. 허용한 숫자 정책 109개를 기록에서 평가하고 선택하는 변형이다. 정책은 평가 점수 정의, 최종 평가 정답, 아직 기록되지 않은 결과에 접근하지 않는다. 존재하지 않는 후속 기록을 성공이나 수렴으로 처리하지 않는다.

## 결과

완료 기록: `.benchmarks/v85_replay_mlp/report-02/comparison.md`와 `report.json`.

| 항목 | 균등 탐색 | 재생 탐색 |
|---|---:|---:|
| 후속 학습 갱신 수 | 21,996 | 14,885 |
| 후속 학습 구간 시간 | 273.02초 | 193.14초 |
| 혼합물 분리 360쌍 MAE, 단일 선택 | 8.976 | 8.957 |
| 전체 재학습 단일 모델의 Ravia 182쌍 MAE | 11.206 | 10.816 |

MAE는 0~100 관능 점수 척도의 오차다. 각 행은 서로 다른 평가 범위다. 정확도가 크게 높아졌다고 결론 내릴 수는 없으며, 가장 확실한 개선은 학습 계산량 감소다. 최초 기록 구축에는 약 601초의 추가 학습이 들었다.

단일 모델은 `set_mlp`, 25에포크, Snitz 360쌍 전체 재학습으로 저장했다. 같은 단계의 균등 탐색 기준 모델은 `covariance_mlp`, 27에포크다. 구조와 에포크는 최종 평가 점수로 고르지 않았다. 가중 결합은 이득이 확인되지 않아 기본 실행 형태로 채택하지 않았다.

## 로컬 사용

모델: `.benchmarks/v85_replay_mlp/bundle-01/replay/model.json`

SHA256: `04b47a2efcd42bd79335750b6402183ffc57f30a26f27cbd33fbada1bec3cb99`

같은 폴더의 `weights.npz`, `encoder.npz`, `features.json`, `policy.json`을 함께 보존한다. 현재 저장소의 `ReplayMixtureModel(path, expected_sha256)`으로 로드한 뒤 `predict(first_graphs, first_amounts, second_graphs, second_amounts)`를 호출한다.

입력은 단일 분자 SMILES 목록과 상대 배합량이다. 중복 성분은 합치고 각 혼합물의 양은 정규화한다. 반환값은 명목 혼합물 유사도이며 측정된 농도·헤드스페이스·제형 방출을 의미하지 않는다. 자연어 생성 서비스가 자동으로 이 모델을 사용하도록 변경하지 않았다.

CPU 검증: Snitz/Ravia 542쌍 전체에서 학습 구현 대비 최대 차이 `2.19e-7`. Torch 없는 CPU 프로세스에서 실행했다. 구조 검증과 캐시 특징을 포함한 40개 원료 p50은 1.806ms이며 전체 레시피 API 응답시간이 아니다. 관련 테스트 47개가 통과했다.

## 재실행

- 새 기록부터: `scripts/train_replay_mlp_v85.py`에 `--core`, `--data-root`, `--output`을 지정한다.
- 호환되는 기록 재사용: `--history-source`를 추가한다. 분할·데이터·학습 코드가 다르면 거부한다.
- 완료된 분할 재개: 같은 설정과 `--resume`을 사용한다. 해시나 설정이 달라지면 재개하지 않는다.
- CPU 전체 검증: `scripts/verify_replay_bundle_v85.py`.
- 결과 독립 재계산: `scripts/report_replay_mlp_v85.py`.

원시 이력·실패한 첫 비교·수정 후 비교·선택된 모델을 각각 보존했다. 이 기록을 실제 제품 정확도나 전체 자연어 요청 95점 달성으로 대체하지 않는다.
