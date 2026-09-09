"""Summarize completed frozen-model evaluations without retraining or rescoring."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.benchmarks/lotion_evaluation_v58'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_diagnosis():
    folder = OUT/'fresh600-r2'
    cases, details = read(folder/'cases.json'), read(folder/'case_results.json')
    with np.load(folder/'predictions.npz', allow_pickle=False) as archive:
        predicted, reference = archive['predicted'], archive['reference']
    counts, worst = np.zeros(3, dtype=int), np.zeros(3)
    reference_violations, worst_case = 0, None
    violations = []
    source = read(ROOT/'.benchmarks/lotion_training_v58/source.json')
    axes = sorted({key for material in source['materials'] for key in material['profile']})
    strong_profile_errors, largest_profile_context = [], None
    for c in cases:
        i, n, length = c['offset'], c['material_count'], c['rows']
        p = predicted[i:i+length].reshape(-1, n, 5)
        r = reference[i:i+length].reshape(-1, n, 5)
        delta = np.diff(p[:, :, 2:], axis=0)
        minima = delta.min(axis=(0, 1))
        counts += minima < -1e-5
        worst = np.maximum(worst, -minima)
        reference_violations += int(np.any(np.diff(r[:, :, 2:], axis=0) < -1e-5))
        value = float(-delta.min())
        if value > 1e-5:
            violations.append(value)
        if worst_case is None or value > worst_case['decrease']:
            loc = np.unravel_index(delta.argmin(), delta.shape)
            worst_case = {'case': c['id'], 'decrease': value,
                'state': ('skin_sink','ventilated','degraded')[loc[2]],
                'time_pair': [c['minutes'][loc[0]], c['minutes'][loc[0]+1]],
                'material_index': c['material_indices'][loc[1]]}
        mass = np.asarray(c['initial_mg_cm2'])*c['initial_parent_fraction']
        materials = [source['materials'][j] for j in c['material_indices']]
        component_profiles = np.array([[m['profile'].get(axis,0.) for axis in axes] for m in materials])
        thresholds = np.array([m['parameters']['odor_threshold_mg_m3'] for m in materials])
        air_pred, air_ref = (p[:, :, 1]*mass/c['headspace_height_cm']*1e6,
                             r[:, :, 1]*mass/c['headspace_height_cm']*1e6)
        a, b = air_pred/thresholds@component_profiles, air_ref/thresholds@component_profiles
        a /= np.maximum(a.sum(axis=1,keepdims=True),1e-300)
        b /= np.maximum(b.sum(axis=1,keepdims=True),1e-300)
        tv = .5*abs(a-b).sum(axis=1)
        oav = (air_ref/thresholds).sum(axis=1)
        strong_profile_errors.extend(tv[oav >= 1.].tolist())
        j = int(tv.argmax())
        if largest_profile_context is None or float(tv[j]) > largest_profile_context['total_variation']:
            largest_profile_context = {'case':c['id'],'minutes':c['minutes'][j], 'total_variation':float(tv[j]),
                'reference_total_air_mg_m3':float(air_ref[j].sum()), 'reference_total_estimated_OAV':float(oav[j]),
                'scope':'engineering_threshold_proxy_not_measured_detectability'}
    figure_path = OUT/'cumulative_sink_counterexample.png'
    if not figure_path.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font = Path('C:/Windows/Fonts/malgun.ttf')
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
        case = next(c for c in cases if c['id'] == worst_case['case'])
        i, length, n = case['offset'],case['rows'],case['material_count']
        material = case['material_indices'].index(worst_case['material_index'])
        state = ('remaining','headspace','skin_sink','ventilated','degraded').index(worst_case['state'])
        a = predicted[i:i+length].reshape(-1,n,5)[:,material,state]*100
        b = reference[i:i+length].reshape(-1,n,5)[:,material,state]*100
        fig, ax = plt.subplots(figsize=(8,4.4),layout='constrained')
        ax.plot(case['minutes'],b,color='#2463ad',marker='o',label='기준 수치 시뮬레이터',linewidth=2)
        ax.plot(case['minutes'],a,color='#c34738',marker='s',label='학습된 V58 방출 모델',linewidth=2)
        ax.axvspan(*worst_case['time_pair'],color='#f0c0ac',alpha=.25)
        ax.set(xlabel='도포 후 시간 (분)',ylabel='누적 피부 이동량 (정규화 질량, %)',
            title=f"시간 일관성 반례: {case['id']}\n210→300분 누적량 {worst_case['decrease']*100:.3f}%p 감소")
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        fig.savefig(figure_path,dpi=160)
        plt.close(fig)
    return {'nonmonotone_by_state': dict(zip(('skin_sink','ventilated','degraded'), counts.tolist())),
        'worst_decrease_by_state': dict(zip(('skin_sink','ventilated','degraded'), worst.tolist())),
        'decrease_fraction_threshold_counts': {str(t): sum(v > t for v in violations) for t in (1e-5,1e-4,1e-3,.01)},
        'reference_monotonicity_violations_at_same_tolerance': reference_violations,
        'worst_case': worst_case, 'largest_profile_errors': sorted(details,key=lambda x:x['max_profile_total_variation'],reverse=True)[:10],
        'largest_profile_error_context':largest_profile_context,
        'posthoc_estimated_OAV_at_least_one_profile_errors':{'count':len(strong_profile_errors),
            'mean_total_variation':float(np.mean(strong_profile_errors)),
            'p95_total_variation':float(np.quantile(strong_profile_errors,.95)),
            'maximum_total_variation':float(np.max(strong_profile_errors)),
            'not_used_to_replace_all_rows_or_quality_gates':True}}


def main():
    fresh = read(OUT/'fresh600-r2/summary.json')
    diagnosis = fresh_diagnosis()
    (OUT/'fresh_diagnosis.json').write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2),encoding='utf-8')
    summary_path = OUT/'full400/summary.json'
    if not summary_path.exists():
        print(json.dumps({'fresh_diagnosis': diagnosis, 'full400_status': 'still_running'}, ensure_ascii=False))
        return
    summary, manifest = read(summary_path), read(OUT/'full400/manifest.json')
    values = rows(OUT/'full400/results.jsonl')
    assert len(values) == 400 and len({x['id'] for x in values}) == 400
    assert {x['id'] for x in values} == {x['id'] for x in manifest['cases']}
    assert summary['source_unchanged'] and summary['count'] == 400
    baseline_manifest = read(ROOT/'.benchmarks/local_repair_v56/full400/manifest.json')
    assert manifest['case_sha256'] == baseline_manifest['case_sha256']
    for key in ('target','auto_base_design','adaptive_oil_refinement_steps','model_mode','pool'):
        assert manifest[key] == baseline_manifest[key], key
    baseline = {x['id']: x for x in rows(ROOT/'.benchmarks/local_repair_v56/full400/results.jsonl')}
    repairs = rows(ROOT/'.benchmarks/local_repair_v56/roundoff_replay/results.jsonl')
    for row in repairs:
        assert baseline[row['id']]['status'] == 'error'
        baseline[row['id']] = row
    defined = [r for r in values if r['score'] is not None]
    categories = {}
    for group in sorted({r['group'] for r in values}):
        chosen = [r for r in values if r['group'] == group]
        categories[group] = {'count': len(chosen), 'passed95': sum(bool(r['profile_target_met']) for r in chosen),
            'errors': sum(r['status'] == 'error' for r in chosen),
            'average_score_defined_only': float(np.mean([r['score'] for r in chosen if r['score'] is not None]))}
    paired = [{'id':r['id'],'before':baseline[r['id']]['score'],'after':r['score'],
        'pass_before':bool(baseline[r['id']]['profile_target_met']),'pass_after':bool(r['profile_target_met'])}
        for r in values]
    score_deltas = [p['after']-p['before'] for p in paired if p['before'] is not None and p['after'] is not None]
    by_id = {r['id']: r for r in values}
    bilingual = []
    for row in values:
        if row['id'].startswith('en-') and row['score'] is not None:
            other = by_id.get('ko-'+row['id'][3:])
            if other is not None and other['score'] is not None:
                bilingual.append({'english_id':row['id'],'korean_id':other['id'],'score_difference':abs(row['score']-other['score']),
                    'pass_agreement':bool(row['profile_target_met']) == bool(other['profile_target_met'])})
    audit = summary['learned_release_audit']
    failing = [r for r in values if not r['profile_target_met']]
    gaps = Counter()
    for row in failing:
        checks = row.get('timepoint_assessments',[])
        if checks:
            worst = min(checks,key=lambda x:x['score'])
            for axis, value in sorted(worst.get('deficit',{}).items(),key=lambda x:x[1],reverse=True)[:2]:
                if value > 1e-6:
                    gaps[axis] += 1
    cases_with_flags = [r['id'] for r in values if r.get('learned_release_audit',{}).get('nonmonotone_scenario_count',0)]
    active_sha = manifest['trained_lotion_release']['checkpoint_sha256']
    assert audit['checkpoint_sha256s'] == [active_sha]
    result = {'schema':'lotion-model-evaluation-combined/v1', 'status':'completed',
        'scope':'frozen_local_model_evaluation_not_training_or_deployment',
        'source_manifest_sha256':sha(OUT/'full400/manifest.json'), 'request_results_sha256':sha(OUT/'full400/results.jsonl'),
        'fresh_evaluation_sha256':sha(OUT/'fresh600-r2/summary.json'),
        'model_sha256':active_sha, 'full400':summary, 'categories':categories,
        'mean_score_defined_only':float(np.mean([r['score'] for r in defined])),
        'median_score_defined_only':float(np.median([r['score'] for r in defined])),
        'requests_above90':sum(r['score'] is not None and r['score']>=90 for r in values),
        'candidate_latency_seconds_concurrent_workers':{'p50':float(np.median([r['seconds'] for r in values])),
            'p95':float(np.quantile([r['seconds'] for r in values],.95)), 'maximum':max(r['seconds'] for r in values)},
        'comparison_to_prior_v56':{'baseline_scope':'initial_400_plus_all_4_targeted_error_replays_not_fresh_final_source_400',
            'baseline_passed95':sum(bool(r['profile_target_met']) for r in baseline.values()),
            'new_pass_ids':[p['id'] for p in paired if p['pass_after'] and not p['pass_before']],
            'lost_pass_ids':[p['id'] for p in paired if p['pass_before'] and not p['pass_after']],
            'defined_score_pairs':len(score_deltas),'mean_score_delta':float(np.mean(score_deltas)),
            'max_abs_score_delta':float(np.max(np.abs(score_deltas)))},
        'bilingual':{'pairs':len(bilingual),'pass_disagreements':sum(not p['pass_agreement'] for p in bilingual),
            'max_score_difference':max(p['score_difference'] for p in bilingual),
            'differences_above_0_01':[p for p in bilingual if p['score_difference']>.01]},
        'dominant_deficit_axes_in_failed_requests':dict(gaps.most_common()),
        'learned_search_statuses':dict(Counter(r.get('learned_optimization_status') for r in values)),
        'requests_with_nonmonotone_neural_output':cases_with_flags,
        'fresh600':fresh, 'fresh_diagnosis':diagnosis,
        'checkpoints_modified':False,'human_similarity_percent':None,'full_model_acceptance':False,
        'reason_full_acceptance_denied':'strict_recipe_95_failures_and_neural_cumulative_sink_monotonicity_failures'}
    (OUT/'combined_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    examples = '\n'.join(f"| {by_id[key]['brief']} | {by_id[key]['score']:.4f} | {(by_id[key].get('profile_coverage') or {}).get('optimistic_profile_upper_percent', float('nan')):.4f} |"
                         for key in ('en-2','en-4','en-12','en-15'))
    category_rows = '\n'.join(f"| {name} | {v['passed95']}/{v['count']} | {100*v['passed95']/v['count']:.2f}% |"
                              for name,v in categories.items())
    f = fresh['trained']
    previous = result['comparison_to_prior_v56']
    diagnosis_md = f"""# V58 로션 모델 종합 평가

2026-09-09. **평가는 완료했고, 전체 품질 합격 판정은 보류한다.** 모델·레시피 점수·95점 기준·원료 정책을 변경하거나 재학습·배포하지 않았다.

## 결과 요약

| 평가 | 결과 |
| --- | --- |
| 기존 고정 자연어 요청 | **{summary['passed95']}/400 통과 ({summary['pass_rate_percent']:.2f}%)**, 미달 {400-summary['passed95']}건 |
| 실행 오류 / 미완료 탐색 | {summary['errors']} / {sum(bool(r.get('search_incomplete')) for r in values)} |
| 이전 V56 + 오류 4건 재검사 결과와 비교 | 기존 {previous['baseline_passed95']}/400; 새 통과 {len(previous['new_pass_ids'])}건, 통과 상실 {len(previous['lost_pass_ids'])}건 |
| 새 방출 모델 연결 | {audit['requests_with_all_scenarios_attached']}/400 요청, {audit['attached_scenarios']}/{audit['final_scenarios']} 최종 시나리오 |
| 400개 요청의 새 모델 시간 일관성 경고 | {len(cases_with_flags)}개 요청, {audit['nonmonotone_scenarios']}개 시나리오 |
| 신규 합성 혼합 조성 | 600개, 학습 제외 분자 553개 전체, 12개 시간점, {fresh['rows']:,}개 성분-시간점 |
| 신규 혼합 조성의 시간 일관성 위반 | **{fresh['nonmonotone_mixtures']}/600 ({100*fresh['nonmonotone_mixtures']/600:.2f}%)** |

400개 요청은 이전과 같은 사례 해시 `{manifest['case_sha256']}`, 95점 기준, 조건부 연구 후보 정책, 오일/물 비율 자동 설계 및 적응 탐색 3회로 실행했다. 실패·오류를 분모에서 제외하지 않았다. 모든 자연어 표현이나 실제 제조 향의 정확도를 포괄하는 시험은 아니다.

## 자연어 요청별 레시피 성능

| 요청군 | 95점 통과 | 통과율 |
| --- | ---: | ---: |
{category_rows}

정의된 {len(defined)}개 점수 평균 {result['mean_score_defined_only']:.4f}, 중앙값 {result['median_score_defined_only']:.4f}. 90점 이상은 {result['requests_above90']}/400이다. 기본 기준 로션에서 {summary['fixed_reference_passed95_in_same_run']}건 통과했고, 오일/물 비율 탐색으로 {summary['additional_passes_from_base_design']}건이 추가 통과했다.

현재 원료 프로필 집합의 낙관적 상한만으로도 95점을 넘지 못하는 요청은 {len(summary['profile_upper_excluded_ids'])}건이다. 이 상한은 현재 고정 프로필·선택 후보의 계산적 진단이며 실제 향 재현의 불가능성을 뜻하지 않는다. 상한 계산은 물리 방출·비용·상한을 일부 완화한 조건으로 계산하므로, 특정 모델 표현의 부족을 찾는 데 사용한다.

| 요청 | 실제 최저 시점 점수 | 현재 프로필 집합의 낙관적 상한 |
| --- | ---: | ---: |
{examples}

방출 계수의 예측을 개선해도 기존 원료의 향 표현 공간 자체가 달라지지는 않는다. 또한 현재 V54 참조 향 표현은 `fresh`, `aquatic`, `amber`, `white_floral` 4개 축을 직접 지원하지 않는다. 이 네 축이 모든 실패의 유일한 원인이라는 뜻은 아니다. 예를 들어 `clean`과 `powdery`도 고정 원료 프로필의 상한에 걸린다.

## 새 방출 신경망의 독립 조건 평가

체크포인트 `{active_sha}`를 고정했다. 원래 학습에서 제외했던 553개 분자만 사용했고, 새 seed 20260909로 새로운 혼합·제형 조건 600개를 구성했다. 시험 결과를 학습·선택에 다시 사용하지 않았다. 새 표본의 정답은 정밀 수송 방정식이 만든 **합성 값**이다.

여기서 분자 분리는 **V58 방출 신경망의 학습 분리**를 뜻한다. 부모 V54나 기존 카탈로그까지 한 번도 접하지 않은 분자라는 주장, 또는 새로운 분자의 실제 냄새를 구조만으로 맞췄다는 주장이 아니다.

| 지표 | 결과 |
| --- | ---: |
| 상태 질량분율 MAE | {f['state_fraction_mae']:.9f} |
| 상태 질량분율 RMSE | {f['state_fraction_rmse']:.9f} |
| 표본별 최대 상태 오차의 95백분위 | {f['p95_max_state_fraction_error']:.9f} |
| 전체 최악 상태 질량분율 오차 | {f['max_state_fraction_error']:.9f} |
| 헤드스페이스 log10 RMSE | {f['headspace_log10_rmse_above_1e_10_fraction']:.9f} dex |
| 질량 보존 최대 오차 | {f['mass_balance_max_error']:.3g} |
| 음의 질량 | {fresh['negative_mass_entries']}건 |
| 범위 내/밖 응답 경계 검사 | {sum(r['passed'] for r in fresh['runtime_boundaries'])}/{len(fresh['runtime_boundaries'])} |

질량분율은 정규화된 상태량의 오차이며 실제 향 정확도 퍼센트가 아니다. 헤드스페이스 로그 지표는 기준 헤드스페이스 질량분율이 1e-10보다 큰 {f['headspace_evaluated_rows']:,}개 값으로 계산했고, 나머지도 전체 상태 오차 지표에는 유지했다. 32개 어려운/분산 표본은 별도 Radau ODE로 확인했으며 기준 생성기 최대 차이는 {read(OUT/'fresh600-r2/independent_ode.json')['maximum_teacher_error']:.3g}였다.

혼합 후 19축 카탈로그/OAV 프로필의 기준 시뮬레이터 대비 총변동 거리는 평균 {fresh['profile_propagation']['mean_total_variation']:.9f}, 95백분위 {fresh['profile_propagation']['p95_total_variation']:.9f}, 최악 {fresh['profile_propagation']['max_total_variation']:.9f}였다. **성분 질량의 작은 오차도 역치 가중과 정규화를 거치면 특정 혼합에서 프로필 차이로 커질 수 있다.** 이는 요청 향 일치도나 인간 후각 유사도가 아니라 같은 계산 모델 사이의 비교다.

다만 최악 프로필 차이 사례는 `{diagnosis['largest_profile_error_context']['case']}`의 {diagnosis['largest_profile_error_context']['minutes']:g}분이며, 기준 총 기체 농도 {diagnosis['largest_profile_error_context']['reference_total_air_mg_m3']:.3g} mg/m³, 추정 OAV 합 {diagnosis['largest_profile_error_context']['reference_total_estimated_OAV']:.3g}인 극미량 조건이었다. 이를 사용자에게 뚜렷하게 느껴지는 향 차이로 해석하면 과장이다. 추가 사후 분석에서 추정 OAV 합 1 이상인 {diagnosis['posthoc_estimated_OAV_at_least_one_profile_errors']['count']}개 시점의 최대 총변동 거리는 {diagnosis['posthoc_estimated_OAV_at_least_one_profile_errors']['maximum_total_variation']:.9f}였다. 이 조건부 분석으로 전체 표본이나 사전 품질 기준을 대체하지 않았다. OAV 자체도 실측 후각이 아니라 현재 추정 역치를 사용한 값이다.

## 확인된 결함

1. **시간축의 누적량 일관성 미충족.** {fresh['nonmonotone_mixtures']}개 신규 혼합에서 정규화 누적 싱크가 허용 오차 1e-5보다 크게 감소했다. 피부 싱크 {diagnosis['nonmonotone_by_state']['skin_sink']}개, 환기 배출 {diagnosis['nonmonotone_by_state']['ventilated']}개, 분해 {diagnosis['nonmonotone_by_state']['degraded']}개이며 중복 포함이다. 같은 기준 시뮬레이터에는 같은 허용 오차의 위반이 {diagnosis['reference_monotonicity_violations_at_same_tolerance']}건이었다.
2. 가장 큰 감소는 `{diagnosis['worst_case']['case']}`의 {diagnosis['worst_case']['time_pair'][0]:g}→{diagnosis['worst_case']['time_pair'][1]:g}분 구간, {diagnosis['worst_case']['state']}에서 질량분율 {diagnosis['worst_case']['decrease']:.9f}였다. 0.001보다 큰 감소가 {diagnosis['decrease_fraction_threshold_counts']['0.001']}개 혼합, 0.01보다 큰 감소도 {diagnosis['decrease_fraction_threshold_counts']['0.01']}개 있다. 전부 기계 오차라고 볼 수 없다.
3. 원인은 각 시간점의 상태를 독립적으로 예측하고 softmax로 **그 시간점의** 양수·질량 합만 맞추는 구조다. 시점 사이의 누적 보존은 보장하지 않는다. 현재 API는 발견된 경우 `research_prediction_flagged`를 내보낸다. 이번 평가에서 임의 보정이나 경고 제거를 하지 않았다.
4. 레시피 실패는 별도 문제다. 새 신경망은 빠른 방출 예측 경로이며 기존 고정 향 프로필, 목표 표현, 최적화 점수를 대체하지 않는다. 따라서 이번 방출 학습을 400개 요청의 레시피 성공률 개선으로 주장하지 않는다.

다음 개선 대상은 시간 간 전이를 보존하는 방출 모델과 실제 레시피 평가에 사용되는 향 표현·목표 모델이다. 이 문서는 **평가 결과**이며 해당 개선을 구현했다는 완료 보고가 아니다.

![시간 일관성 반례](/C:/Users/user/Desktop/Newera/WantedAI_Ai-dev-clean/.benchmarks/lotion_evaluation_v58/cumulative_sink_counterexample.png)

## 실행 범위와 근거

- 400개 평가는 로컬 8개 계산 프로세스로 {summary['seconds']:.1f}초 실행했다. 동시 실행 및 탐색을 포함한 측정이므로 배포 환경 단일 요청 SLA로 해석하지 않는다.
- 이전 결과는 V56 최초 전체 실행과 오류 4건 재검사 결과를 합쳐 비교했다. 새로운 최종 소스 400건 실행과 동일한 형태의 이전 측정이라고 포장하지 않는다.
- 신규 혼합 평가의 집계 단계에서 변수명 충돌이 발생해 보고서 생성을 재개했다. 저장된 raw 예측·기준값·사례가 동일함을 확인하고 재사용했으며, 신경망 재학습이나 정답 교체는 하지 않았다. 최초 `fresh600`은 평가 시작 전 입력 처리 오류, `fresh600-r2`가 실제 raw 결과다.
- 신규 혼합 JSON의 재개 실행 `total_seconds`는 집계 재개 단계만의 시간이다. 전체 새 표본 계산 시간으로 인용하지 않았으며, 저장되지 않은 최초 실행 타이밍을 추정해 채우지 않았다.
- 신경망·부모 모델·로컬 설정은 평가 전후 동일하다. 통과 기준 완화, 모델 교체, 배포, push, 유료 호출 없음.

기계 판독 결과: `.benchmarks/lotion_evaluation_v58/combined_report.json`.

원시 400개 결과: `.benchmarks/lotion_evaluation_v58/full400/results.jsonl`.

신규 혼합 예측: `.benchmarks/lotion_evaluation_v58/fresh600-r2/predictions.npz`.

결함 사례: `.benchmarks/lotion_evaluation_v58/fresh_diagnosis.json`.
"""
    (ROOT/'AI_LOTION_EVALUATION_V58.md').write_text(diagnosis_md,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','categories','comparison_to_prior_v56','dominant_deficit_axes_in_failed_requests')},ensure_ascii=False))


if __name__ == '__main__':
    main()
