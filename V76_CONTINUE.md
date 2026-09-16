# V76 계속 작업할 때

이 파일은 V76 작업 당시의 기록이다. 이후 사용자가 모든 향에 세분화 구조 적용을 요청하여 V77 공통 해석·참조 경로를 구현했다. 현재 로컬 설정과 검증 결과는 `ODOR_SPACE_V77.md`를 우선 확인한다. 아래의 V76 전용 재개 명령과 원래 기준 시험은 별도로 보존한다.

## 완료된 로컬 작업

- 자세한 수정/측정 근거는 `SYSTEM_REPAIR_V76.md`.
- V76 단일 모델: `.benchmarks/v76_system_repair/assembled-01/model.json`.
- SHA256: `158b9f82ff9fb784eeec114ef1f7bbfc4d613badeb98f46e0f60435d5af38d2f`.
- 명시적 검증용 프로필: `perfumery.v76-system-01.local.json`.
- 소스/카탈로그/모델/물성 결합: `.benchmarks/v76_system_repair/package-01/preparation.json`.
- 실제 API 예시 검증: `.benchmarks/v76_system_repair/api-verification-02/report.json`.
- 전체 회귀 1,907 통과/1 건너뜀. 이후 학습 브리지 26개, 관측 출력 10개 관련 테스트 통과.
- 배포/push/기본 프로필 교체는 하지 않았다. 기존 `perfumery.local.json`의 소스 결합은 코드 변경 전 것이므로 현재 소스에서는 명시적 V76 검증 프로필을 사용한다. 기본 선택 문제를 해결하기 전 운영 준비 완료라고 보고하지 않는다.

## 실행 중인 기존 기준 전체 검증

- 실행 세션 ID: `42284` (세션이 유지돼 있으면 write_stdin으로 확인).
- 결과: `.benchmarks/v76_system_repair/full400-after-01/summary.json`, `results.jsonl`, `responses/`.
- 6 workers, 향수/로션 각각 동일한 400개 요청, 95점 기준 그대로.
- 실행을 중단시키지 않았다. 이어서 시작하기 전 결과 파일과 실제 프로세스를 확인하여 중복 실행하지 않는다.
- 종료됐지만 미완료라면 기존 `scripts/benchmark_candidate_v72.py`에 동일 인자와 `--resume`을 사용한다. profile/wheel/protocol/workers를 바꾸지 않는다.
- 실행 인자: candidate `.benchmarks/v76_system_repair/package-01/preparation.json`, ablation-protocol `.benchmarks/v75_system_audit/full400-before-01/protocol.json`, output `.benchmarks/v76_system_repair/full400-after-01`, workers `6`.

## 완료 후 비교

- `scripts/compare_full_system_v76.py`: before `.benchmarks/v75_system_audit/full400-before-01`, common-before `.benchmarks/v76_system_repair/common-oracle-before-01`, after `.benchmarks/v76_system_repair/full400-after-01`, 새 output 사용.
- `scripts/analyze_full_failures_v76.py`: 완결된 after run을 전체 분석. `--allow-partial` 없이 사용한다.
- 원래 V75 API 통과는 향수 42/400, 로션 247/400. 새 물성/모델 계산으로 기존 배합만 재평가한 값은 24/400, 0/400이며 API 승인 건수가 아니다. 이 두 비교를 섞지 않는다.

## 사용자 선택이 필요한 추가 핵심 문제

후속 상태: 사용자가 출처 기반 세부 향 표현을 모든 향으로 확장하도록 요청했다. 새 기준은 V77로 버전 구분하고 기존 19축 진단을 함께 남겼다. 이전 800회 시험의 기준은 변경하지 않았다.

주 향수 점수: 19축 키워드 목표. '시트러스'이면 다른 축은 목표 질량 0.
보조 모델: 자연스러운 배경을 가진 고정 출처 참조 프로필(146차원).
두 목표가 충돌하며, 중간 향수 90건 중 69건에서 보조 가이드가 더 높은 주 점수 후보를 거부했다. 이 중간 숫자를 400건 전체 결과로 보고하지 않는다.

사용자에게 보낸 질문: 자연어 요청을 출처 기반 참조 프로필로 통일하고 기존 점수도 따로 보존할지, 기존 19축 채점을 유지할지.
답변 없이 95점 기준을 바꾸거나 보조 가이드를 꺼서 통과율을 올리지 않는다. 새 기준을 택하면 고정 참조/동일 계산으로 이전·이후를 다시 비교하고 기존 시험과 별도 보고한다.

## 중요한 해석

- inverse-training-06은 새 설정 16회가 기존 기본 8회보다 개선됐지만, 같은 16회에서는 새 가중치의 단독 우위가 없었다. 모델에도 이 사실을 기록했다.
- 증기압 282/3,830, 역치 92/3,830만 관측값 연결. 나머지를 측정값으로 만들지 않았다.
- 공개 제형 812건은 특정 세정용 수계 제형 자료이지 임의의 바디로션 또는 피부 안전성 검증이 아니다.
- pair 109 출력에는 비후각 메타데이터 2개가 있어 API에서는 향 라벨 107개와 분리했다.
- 사전학습 지식이 포함된 출처별 분리 검증을 전역 새 분자 블라인드 결과로 주장하지 않는다.
- 평가 기준 변경, 배포, 실제 후각 90% 달성을 완료했다고 보고하지 않는다.
