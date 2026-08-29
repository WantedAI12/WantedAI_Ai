# Bushdid 정확도 개선 V4

기존 frozen R2의 인간-ceiling 정규화 순위상관 `0.3410`을 개선하기 위해 Bushdid
protocol 변수와 DREAM odor-pair GNN 임베딩을 결합했다. Control 4개는 제외하고
calibration+final 260개 자극을 mixture size×overlap으로 층화한 5-fold stimulus
교차적합으로 평가한다.

## 고정 후보

- 입력: 성분 겹침, 혼합물 크기, right/wrong dilution 구조, 기존 R2 진단
- odor-pair: commit `32c25530...`, 128차원 mixture embedding
- 최종 feature: 24개 compact scalar
- 모델: CatBoost MAE, depth 3, 600 trees, `l2_leaf_reg=300`
- 직렬화: pickle 없는 deterministic CatBoost JSON

## 결과

| 평가 | 절대 정확도 `100-MAE` | Spearman | ceiling 정규화 |
|---|---:|---:|---:|
| 기존 frozen R2 | 별도 final calibration 기준 | 0.291699 | 0.340999 |
| cross-fit 성분겹침 | 88.6455% | 0.569278 | - |
| cross-fit odor-pair isotonic | 88.8950% | 0.600229 | - |
| compact V4 | **89.5788%** | **0.665137** | **0.777550** |

V4의 절대 정확도 bootstrap 95% 구간은 `[88.5301, 90.5682]`이며 성분겹침 대비
정확도 개선 구간은 `[+0.2884, +1.6012]` percentage point다. 점 추정 90%와
정규화 순위 0.90은 아직 통과하지 못했다.

이 모델은 Bushdid 결과가 공개된 뒤 설계한 outcome-aware 개발 모델이다. 운영
primary score 가중치는 `0`이며 새 데이터 승격 근거로 자동 사용하지 않는다.

## 재현

```powershell
python scripts\experiment_bushdid_accuracy_v4.py
pytest -q tests\test_bushdid_accuracy_v4.py
```

산출물:

- `benchmarks/bushdid_accuracy_v4.json`
- `benchmarks/bushdid_accuracy_v4_catboost.json`
