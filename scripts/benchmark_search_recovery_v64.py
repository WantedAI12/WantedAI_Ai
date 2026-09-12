"""Paired solver ablation using identical pinned models, data and final gates.

The optional control loads ONLY the old search module from a hash-verified
local wheel. It is an explicit experiment, never a production model fallback.
Every requested case and failure is retained; results are not human accuracy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEMBER = 'fragrance_ai/recommender/lotion_reference_search.py'


def initialize(control_wheel, control_sha):
    from scripts import benchmark_lotion_design as runner
    runner.initialize(False, None, 3, None, 'local')
    if control_wheel:
        raw = Path(control_wheel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != control_sha:
            raise ValueError('control wheel hash mismatch')
        with zipfile.ZipFile(control_wheel) as archive:
            source = archive.read(MEMBER)
        module = types.ModuleType('fragrance_ai.recommender._frozen_search_control')
        module.__package__ = 'fragrance_ai.recommender'
        exec(compile(source, str(control_wheel)+'!/'+MEMBER, 'exec'), module.__dict__)
        from fragrance_ai.recommender import lotion_reference_search
        lotion_reference_search.optimize_observed_reference = module.optimize_observed_reference


def run(case):
    from scripts import benchmark_lotion_design as runner
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    started = time.perf_counter()
    try:
        value = estimate_lotion_recipe(LotionEstimateRequest(brief=case['brief'],
            registry_pool='conditional_research', max_risk_tier=2,
            evaluation_mode='observed_reference'), runner.CATALOG,
            perception_guidance=runner.PROVIDER, use_configured_perception=False)
        return {**case, 'seconds': time.perf_counter()-started,
            **{key: value.get(key) for key in ('status', 'score', 'profile_target_met',
                'search_incomplete', 'solver_calls', 'material_column_search', 'recipe',
                'closest_candidate', 'perceptual_evaluation', 'timepoint_assessments')},
            'human_similarity_percent': None}
    except Exception as error:
        return {**case, 'seconds': time.perf_counter()-started, 'status': 'error',
            'score': None, 'profile_target_met': False, 'error': type(error).__name__+': '+str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-wheel', type=Path)
    parser.add_argument('--control-sha256')
    parser.add_argument('--case-id', action='append', required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    if bool(args.control_wheel) != bool(args.control_sha256):
        parser.error('control wheel and its trusted SHA256 must be supplied together')
    from scripts.evaluate_request_space import request_cases
    from fragrance_ai.recommender.runtime import _source_snapshot, _data_snapshot
    from fragrance_ai.recommender.local_runtime import local_snapshot
    cases = [c for c in request_cases() if c['id'] in args.case_id]
    if len(cases) != len(set(args.case_id)):
        parser.error('unknown case ID')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = (_source_snapshot(), _data_snapshot(), local_snapshot())
    manifest = {'scope': 'diagnostic_solver_ablation_not_fresh_blind_test', 'cases': cases,
        'control_wheel': str(args.control_wheel) if args.control_wheel else None,
        'control_sha256': args.control_sha256, 'replaced_module_only': MEMBER if args.control_wheel else None,
        'source_data_profile_snapshot': snapshot, 'target': 95, 'workers': args.workers,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    started, rows = time.perf_counter(), []
    with (args.output/'results.jsonl').open('w', encoding='utf-8') as output, ProcessPoolExecutor(
            args.workers, initializer=initialize, initargs=(args.control_wheel, args.control_sha256)) as pool:
        for future in as_completed([pool.submit(run, case) for case in cases]):
            row = future.result()
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            output.flush()
            print(json.dumps({k: row.get(k) for k in ('id', 'status', 'score', 'profile_target_met', 'seconds')}), flush=True)
    report = {'count': len(rows), 'passed95': sum(bool(r['profile_target_met']) for r in rows),
        'errors': sum(r['status'] == 'error' for r in rows),
        'search_incomplete': sum(bool(r.get('search_incomplete')) for r in rows),
        'seconds': time.perf_counter()-started,
        'source_data_profile_unchanged': snapshot == (_source_snapshot(), _data_snapshot(), local_snapshot()),
        'human_similarity_percent': None}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
