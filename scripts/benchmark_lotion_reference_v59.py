"""Paired legacy/new-objective 400-request evaluation; retain every failure."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_lotion_design as old


def score_old_formula(result, bank):
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.models import RecipeConstraints
    from fragrance_ai.recommender.lotion_reference_objective import exposure_groups
    brief = NaturalLanguageBriefParser(old.CATALOG).parse(result['preparation']['intent']['original_text'],
                                                         RecipeConstraints(product_category='body_lotion'))
    rows = result['preparation']['evaluation_targets']
    targets, unsupported = bank.targets(brief, rows)
    if unsupported or any(t is None for t in targets):
        return {'score': None, 'passed95': False, 'unsupported': unsupported}
    groups = exposure_groups(brief, rows)
    checks = []
    presence = True
    for simulation in [result.get('simulation'), *result.get('scenario_simulations', [])]:
        if not simulation:
            return {'score': None, 'passed95': False, 'reason': 'no_final_formula'}
        point_shapes, activity = [], []
        for point in simulation['temporal_profile'][1:]:
            references = point.get('learned_perception', {}).get('reference_profile_sensitivity', [])
            if len(references) != 2:
                return {'score': None, 'passed95': False, 'reason': 'incomplete_learned_coverage'}
            point_shapes.append([[ref['predicted_endpoint_fraction'][e] for e in bank.endpoints] for ref in references])
            activity.append(point['total_odor_activity_proxy'])
        for g in groups:
            weights = g['weights']*np.asarray(activity)
            if weights.sum() <= 0:
                return {'score': None, 'passed95': False, 'reason': 'empty_exposure'}
            mixture = np.einsum('t,thd->hd',weights/weights.sum(),np.asarray(point_shapes))
            for h in range(2):
                checks.append(bank.compare(targets[g['target_index']],mixture[h],h))
            presence &= weights.sum()+1e-8 >= 1.
    score = min(c['score'] for c in checks)
    specific = all(c['reference_more_specific_than_background'] for c in checks)
    return {'score': score, 'passed95': bool(score+1e-8 >= 95 and presence and specific),
        'physical_presence_passed': bool(presence), 'background_discrimination_passed': specific}


def run(case):
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    from fragrance_ai.recommender.lotion_reference_objective import configured_reference
    from fragrance_ai.recommender.lotion_perception import make_lotion_shape_predictor
    start = time.perf_counter()
    output = {**case}
    request = LotionEstimateRequest(brief=case['brief'], evaluation_mode='legacy_profile',
        registry_pool='conditional_research', max_risk_tier=2,
        base_design={'adaptive_oil_refinement_steps': 3})
    try:
        legacy = estimate_lotion_recipe(request, old.CATALOG, perception_guidance=old.PROVIDER,
                                        use_configured_perception=False)
        output['legacy'] = {k: legacy.get(k) for k in ('status','score','profile_target_met','formula_id')}
        new_request = request.model_copy(update={'evaluation_mode': 'observed_reference'})
        bank = configured_reference(new_request, make_lotion_shape_predictor(old.PROVIDER))
        output['same_legacy_formula_new_evaluation'] = score_old_formula(legacy, bank)
        result = estimate_lotion_recipe(new_request, old.CATALOG, perception_guidance=old.PROVIDER,
            use_configured_perception=False, _incumbent_recipe=legacy.get('recipe') or legacy.get('closest_candidate'))
        output.update({k: result.get(k) for k in ('status','score','profile_target_met','formula_id',
            'search_incomplete','solver_calls','base_design','perceptual_evaluation','incumbent_reuse')})
        output['candidate_recipe'] = result.get('candidate_recipe', [])
        output['timepoint_assessments'] = result.get('timepoint_assessments', [])
        output['legacy_score_of_new_formula'] = result.get('legacy_evaluation', {}).get('score')
        output['learned_release_audit'] = old.release_audit(result)
        output['seconds'] = time.perf_counter()-start
    except Exception as error:
        output.update(status='error', score=None, profile_target_met=False,
            error=type(error).__name__+': '+str(error), seconds=time.perf_counter()-start)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8, choices=range(1,9))
    p.add_argument('--case-id', action='append')
    args = p.parse_args()
    if args.output.exists():
        p.error('new output directory required')
    from scripts.evaluate_request_space import request_cases
    from fragrance_ai.recommender.runtime import _source_snapshot, _data_snapshot
    from fragrance_ai.recommender.local_runtime import local_snapshot, local_profile
    cases = request_cases()
    if args.case_id:
        if set(args.case_id)-{c['id'] for c in cases}:
            p.error('unknown case ID')
        cases = [c for c in cases if c['id'] in args.case_id]
    old.initialize(True, None, 3, None, 'local')
    snapshot = local_snapshot()
    hashes = {**_source_snapshot(), **_data_snapshot()}
    manifest = {'cases': cases, 'cases_sha256': hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'target': 95, 'source_sha256': hashes, 'local_profile_sha256': snapshot[0],
        'reference': local_profile()['lotion_target_reference'], 'workers': args.workers,
        'protocol_sha256': hashlib.sha256((ROOT/'AI_LOTION_REFERENCE_V59_PROTOCOL.md').read_bytes()).hexdigest(),
        'same_cases_not_same_metric': True, 'no_human_lotion_accuracy_claim': True,
        'scope': 'paired_legacy_recipe_rescoring_and_new_objective_optimization',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output/'protocol.md').write_text((ROOT/'AI_LOTION_REFERENCE_V59_PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
    start = time.perf_counter()
    records = []
    with (args.output/'results.jsonl').open('w', encoding='utf-8') as f, ProcessPoolExecutor(
            args.workers, initializer=old.initialize, initargs=(True,None,3,None,'local')) as executor:
        for future in as_completed([executor.submit(run, c) for c in cases]):
            record = future.result()
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            f.flush()
            print(json.dumps({'completed': len(records), 'total': len(cases),
                'passed95': sum(bool(r.get('profile_target_met')) for r in records),
                'errors': sum(r['status'] == 'error' for r in records),
                'elapsed_seconds': round(time.perf_counter()-start,1)}), flush=True)
    unchanged = hashes == {**_source_snapshot(), **_data_snapshot()} and local_snapshot() == snapshot
    result = {'count': len(records), 'passed95': sum(bool(r.get('profile_target_met')) for r in records),
        'legacy_passed95': sum(bool(r.get('legacy',{}).get('profile_target_met')) for r in records),
        'same_legacy_formula_new_evaluation_passed95': sum(bool(r.get('same_legacy_formula_new_evaluation',{}).get('passed95')) for r in records),
        'errors': sum(r['status'] == 'error' for r in records),
        'unsupported': sum(r['status'] == 'insufficient_observed_target_coverage' for r in records),
        'search_incomplete': sum(bool(r.get('search_incomplete')) for r in records),
        'source_unchanged': unchanged, 'seconds': time.perf_counter()-start,
        'human_similarity_percent': None,
        'failed_ids': [r['id'] for r in records if not r.get('profile_target_met')]}
    result['pass_rate_percent'] = 100*result['passed95']/len(records)
    (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
