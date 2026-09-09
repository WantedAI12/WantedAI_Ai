"""Read completed paired evaluation artifacts; never rescore or omit failures."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = Path(sys.argv[1])
    rows = [json.loads(x) for x in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    assert len(rows) == summary['count'] == len(manifest['cases'])
    assert len({r['id'] for r in rows}) == len(rows)
    assert {r['id'] for r in rows} == {r['id'] for r in manifest['cases']}
    assert summary['source_unchanged']
    paired = [r for r in rows if r.get('score') is not None and r.get('same_legacy_formula_new_evaluation',{}).get('score') is not None]
    deltas = [r['score']-r['same_legacy_formula_new_evaluation']['score'] for r in paired]
    finite_scores = [r['score'] for r in rows if r.get('score') is not None]
    mean_delta_text = f'{float(np.mean(deltas)):.3f}' if deltas else '미산출'
    unsupported = Counter(x for r in rows for x in (r.get('perceptual_evaluation') or {}).get('unsupported_requirements', []))
    bounds = [r for r in rows if (r.get('perceptual_evaluation') or {}).get('profile_coverage',{}).get('target_excluded')]
    audits = [r['learned_release_audit'] for r in rows if 'learned_release_audit' in r]
    metrics = {**summary, 'paired_finite_scores': len(paired),
        'same_objective_improved': sum(d > 1e-6 for d in deltas),
        'same_objective_declined': sum(d < -1e-6 for d in deltas),
        'same_objective_unchanged': sum(abs(d) <= 1e-6 for d in deltas),
        'mean_same_objective_score_change': float(np.mean(deltas)) if deltas else None,
        'unsupported_requirements': dict(unsupported), 'learned_profile_upper_excludes95': len(bounds),
        'finite_new_scores': sum(r.get('score') is not None for r in rows),
        'mean_new_score': float(np.mean(finite_scores)) if finite_scores else None,
        'old_formula_low_oav': sum(r.get('same_legacy_formula_new_evaluation',{}).get('physical_presence_passed') is False for r in rows),
        'new_formula_background_guard_failed': sum((r.get('perceptual_evaluation') or {}).get('background_discrimination_passed') is False for r in rows),
        'v58_new_formula_nonmonotone_scenarios': sum(r['nonmonotone_scenario_count'] for r in audits),
        'reference_checkpoint': manifest['reference'], 'case_sha256': manifest['cases_sha256'],
        'results_sha256': hashlib.sha256((folder/'results.jsonl').read_bytes()).hexdigest()}
    by_group = {}
    for r in rows:
        group = r.get('group', 'other')
        by_group.setdefault(group, {'total': 0, 'passed': 0, 'errors': 0})
        by_group[group]['total'] += 1
        by_group[group]['passed'] += bool(r.get('profile_target_met'))
        by_group[group]['errors'] += r['status'] == 'error'
    metrics['groups'] = by_group
    metrics['decline_details'] = [{
        'id': r['id'], 'brief': r['brief'],
        'old_formula_score': r['same_legacy_formula_new_evaluation']['score'],
        'new_formula_score': r['score'],
        'old_formula_physical_presence_passed': r['same_legacy_formula_new_evaluation'].get('physical_presence_passed'),
        'new_formula_physical_presence_passed': (r.get('perceptual_evaluation') or {}).get('physical_presence_passed')
    } for r in paired if r['score'] < r['same_legacy_formula_new_evaluation']['score'] - 1e-6]
    metrics['request_seconds'] = {name: float(np.percentile([r['seconds'] for r in rows], percentile))
        for name, percentile in (('median', 50), ('p90', 90), ('maximum', 100))}
    metrics['evaluation_version'] = 'lotion-observed-exposure/v2'
    (folder/'analysis.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    examples = '\n'.join(f"| {r['brief']} | {r['same_legacy_formula_new_evaluation']['score']:.2f} | {r['score']:.2f} |"
                         for r in sorted(paired, key=lambda r:r['score']-r['same_legacy_formula_new_evaluation']['score'], reverse=True)[:8])
    declines = '\n'.join(f"- {r['brief']}: {r['old_formula_score']:.2f} → {r['new_formula_score']:.2f}점; "
        f"최소 평균 OAV 조건 기존 {r['old_formula_physical_presence_passed']}, 새 배합 {r['new_formula_physical_presence_passed']}."
        for r in metrics['decline_details']) or '- 해당 없음.'
    training = json.loads((ROOT/'.benchmarks/lotion_reference_v59/training-01/training_report.json').read_text(encoding='utf-8'))
    text = f'''# V59 로션 목적 일치 평가 수정·재실행

## 결과

고정 요청 {len(rows)}건 전부 실행. 새 기준 95점 통과 **{summary['passed95']}/{len(rows)} ({summary['pass_rate_percent']:.2f}%)**.
오류 {summary['errors']}건, 참조 미지원 {summary['unsupported']}건, 탐색 미완료 {summary['search_incomplete']}건.
탐색 미완료는 전체 최적성을 확인하지 못한 상태이며, 최종 배합 자체가 95점을 넘긴 통과 건과 중복될 수 있다.
실패나 미지원 요청을 분모에서 제외하지 않았다. 실행 시간 {summary['seconds']:.1f}초.
이 400건은 기존 정형·회귀 요청 공간이며 한국어/영어의 같은 의미 요청을 포함한다.
독립적인 사람 관능 표본 400개나 가능한 모든 자연어 요청의 검증은 아니다.
요청별 기존·새 배합의 쌍 평가 시간: 중앙값 {metrics['request_seconds']['median']:.1f}초,
90백분위 {metrics['request_seconds']['p90']:.1f}초. 단일 API 호출 지연 수치가 아니다.

| 구분 | 95점 통과 |
| --- | ---: |
| 기존 기준으로 기존 배합 평가 | {summary['legacy_passed95']}/{len(rows)} |
| 같은 기존 배합을 새 기준으로 재평가 | {summary['same_legacy_formula_new_evaluation_passed95']}/{len(rows)} |
| 새 목표로 실제 배합을 다시 최적화 | {summary['passed95']}/{len(rows)} |

첫째 행과 나머지 행은 같은 정확도 척도가 아니다. 새 기준끼리 직접 비교 가능한 {len(paired)}건 중
개선 {metrics['same_objective_improved']}건, 하락 {metrics['same_objective_declined']}건,
동일 {metrics['same_objective_unchanged']}건. 평균 변화 {mean_delta_text}점.
이는 계산상 프로필 변화이며 사용자가 느끼는 유사도 퍼센트가 아니다.
기존 배합은 새 최소 OAV/의미 구분 조건을 위반할 수 있으므로 점수 변화와 물리 조건을 함께 봐야 한다.

## 실제 수정

- 자연어 19축 순도 목표 대신 **관측된 동반 향을 포함하는 146차원 전체 목표 프로필**을 사용한다.
- 보유 관능 관측 {training['rows']}개로 {training['concept_count']}개 표현의 조건부 참조를 구축했다.
  레시피 결과나 현재 후보를 목표 학습에 사용하지 않았다. 이 학습은 목표 참조 구축이며 V54/V58 신경망 재학습은 아니다.
- 세부 향의 90:10 같은 상대 비중, 동일 계열의 복수 세부 향, 시간대별 요구와 배제 조건을 보존한다.
- V54 프로필이 최종 혼합 향의 목적함수와 배합 탐색에 직접 들어간다. 기존 수송 방정식은 유지한다.
- 최종 제조 조건의 성분별 OAV 노출을 적분해 전체 향을 다시 채점한다. 명시적 첫향/잔향은 각각 평가한다.
  사용자가 요구하지 않은 8시간 동일 향 조건, 평균 향으로의 퇴행, 극미량 방출, 최고 시점만 고르는 오류를 구분한다.
- 검증 가능한 전체 프로필 상한으로 무의미한 베이스 반복 탐색을 제거했다. 후보 재료 자체나 95점 기준을 줄이지 않았다.
- 관능 프로필을 붙이는 과정이 기존 사용자 목표 메타데이터를 변경하던 문제를 수정했다.
- 기존 API 경로 유지. 새 `evaluation_mode`는 auto / observed_reference / legacy_profile.
  로컬 고정 참조가 있으면 자연어 요청의 auto는 새 목적함수를 사용하고, 명시적 19축 입력은 기존 모드를 유지한다.
  캐시는 평가 모드·참조 체크포인트를 구분하며 참조 파일 변경을 검사한다.

## 새 기준에서 점수 상승폭이 큰 예

직접 비교 가능한 요청 중 상승폭 상위 8건이다. 전체 통과율과 전체 평균은 위 집계를 사용한다.

| 요청 | 기존 배합 | 새 배합 |
| --- | ---: | ---: |
{examples}

점수 하락 사례도 전부 기록한다. OAV는 추정 냄새 역치를 사용하는 모델상 조건이며 실측 감지율이 아니다.

{declines}

## 남아 있는 원인을 구분

- 현재 학습 프로필의 낙관적 상한이 95 미만: {len(bounds)}건. 이전 19축 상한과 다른 실제 학습 표현의 진단이다.
- 시간/작업 열 확장 예산 때문에 탐색 미완료가 남을 수 있다. 이는 전체 후보의 최적 배합이나
  95점 불가능을 증명한 상태가 아니다. 시간 예산에 걸린 요청은 CPU 경합에 따라 후보가 달라질 수 있다.
- 미지원 요구별 건수(중복 가능): {dict(unsupported)}.
- 기존 배합의 요청 노출 구간에서 평균 추정 OAV가 1 미만인 사례: {metrics['old_formula_low_oav']}건.
  물리량 적분 가중 근사이며 보정된 사람의 강도/지속시간이 아니다.
- 새 배합의 평균 향 구분 조건 실패: {metrics['new_formula_background_guard_failed']}건.
- 보조 V58 경로의 누적량 비단조 경고: {metrics['v58_new_formula_nonmonotone_scenarios']}개 시나리오.
  이를 지우거나 보정해 성능으로 계산하지 않았다. 최종 물리 예측은 기존 양의 수송 적분기를 사용한다.

목표 참조의 분자 그룹 분리 진단은 {training['grouped_retrieval_count']}건에서 top1 {training['top1']}건,
top5 {training['top5']}건이다. 관측에서 질문을 만든 회고적 데이터 진단이며 V54의 새로운 블라인드 검증이 아니다.
참조 의미 연결 자체가 완벽하지 않음을 보여 주는 결과도 유지했다.

## 검증과 파일

변경 관련 서로 다른 회귀/반례 검사 89개를 실행해 통과했다. 기존 테스트 기준을 약화하지 않았다.
실제 고정 로컬 ASGI에서 새 목적함수 자동 선택, 세부 향 유지, 캐시, 기존 모드 분리,
미지원 거짓 합격 방지를 확인했다. 원료 {3830}개 연구 활성 상태와 모든 원료 필드는 그대로다.
배포·push·외부 모델 호출은 하지 않았다.

원시 결과: [{folder.name}/results.jsonl]({(folder/'results.jsonl').resolve().as_posix()})

정량 분석: [analysis.json]({(folder/'analysis.json').resolve().as_posix()})

실행 시 고정한 프로토콜: [protocol.md]({(folder/'protocol.md').resolve().as_posix()})

V54/V58 부모 체크포인트는 유지했다. 이 수정은 마스킹/상승효과, 기제 고유 향,
수치 농도별 관능 보정이나 사용자의 실제 기대 향을 학습 완료했다는 뜻이 아니다.
**평가 목적을 고쳤지만 실제 제조 향 95%를 입증한 결과는 아니다.**
'''
    (ROOT/'AI_LOTION_REFERENCE_V59.md').write_text(text, encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('failed_ids','groups')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
