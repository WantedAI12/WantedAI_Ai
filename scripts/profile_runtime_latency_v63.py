"""Measure the real pinned API, without changing its model or quality policy."""
import argparse
import cProfile
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time
import warnings
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--brief', default='피오니와 청사과 향')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--lp-threads', type=int, choices=(1,))
    parser.add_argument('--lp-method', choices=('highs-ipm',))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False,
            indent=2, allow_nan=False), encoding='utf-8')

    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI
    from fragrance_ai.recommender.local_runtime import local_profile

    profile = local_profile()
    if profile is None:
        raise ValueError('a real pinned runtime is required')
    profiler = cProfile.Profile()
    original = NaturalLanguagePerfumeryAI.create_recipe
    timings = {}
    lp_calls = []

    def timed(name, function):
        def call(*a, **kw):
            start = time.perf_counter()
            result = None
            try:
                if name == 'fragrance_ai.recommender.global_profile_search.linprog' and args.lp_method:
                    kw['method'] = args.lp_method
                if name.endswith('.linprog') and args.lp_threads:
                    from scipy.optimize import OptimizeWarning
                    kw['options'] = {**kw.get('options', {}), 'threads': args.lp_threads}
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads.*', category=OptimizeWarning)
                        result = function(*a, **kw)
                else:
                    result = function(*a, **kw)
                return result
            finally:
                row = timings.setdefault(name, {'calls': 0, 'seconds': 0.})
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - start
                if name.endswith('.linprog') and result is not None:
                    lp_calls.append({'caller': name, 'seconds': time.perf_counter() - start,
                        'variables': len(a[0]), 'method': kw.get('method'), 'options': kw.get('options'),
                        'status': result.status, 'iterations': result.nit, 'objective': result.fun})
        return call

    def measured(*a, **kw):
        if args.profile:
            return profiler.runcall(original, *a, **kw)
        return original(*a, **kw)

    payload = {'brief': args.brief, 'max_risk_tier': 2,
               'enable_registry_trace_candidates': True}
    save('request.json', payload)
    start = time.perf_counter()
    app = create_app(enable_language=False)
    report = {'scope': 'local_pinned_api_latency_not_human_accuracy',
              'profile_sha256': profile['profile_sha256'],
              'profiled': args.profile, 'startup_seconds': time.perf_counter() - start}
    with ExitStack() as stack:
        from fragrance_ai.recommender import global_profile_search, dose_refinement
        from fragrance_ai.recommender.optimizer import ConstrainedFormulaOptimizer
        from fragrance_ai.recommender.science import TemporalMixtureSimulator
        for owner, name in [(global_profile_search, 'linprog'), (dose_refinement, 'linprog'),
                            (NaturalLanguagePerfumeryAI, '_create_recipe_impl'),
                            (ConstrainedFormulaOptimizer, '_optimize_selected'),
                            (dose_refinement, 'optimize_dose_support'),
                            (TemporalMixtureSimulator, 'evaluate')]:
            stack.enter_context(patch.object(owner, name, timed(owner.__name__ + '.' + name, getattr(owner, name))))
        stack.enter_context(patch.object(NaturalLanguagePerfumeryAI, 'create_recipe', measured))
        with TestClient(app) as client:
            start = time.perf_counter()
            response = client.post('/v1/formulas', json=payload)
            report['first_seconds'] = time.perf_counter() - start
            report['http_status'] = response.status_code
            value = response.json()
            save('response.json', value)
            if response.status_code != 200:
                raise AssertionError((response.status_code, response.text[:1000]))
            start = time.perf_counter()
            repeat = client.post('/v1/formulas', json=payload)
            report['repeat_seconds'] = time.perf_counter() - start
            report['repeat_cache'] = repeat.headers.get('X-Perfumery-Cache')
            report['repeat_equal'] = repeat.status_code == 200 and repeat.json() == value
    report.update(status=value['status'], score=value.get('calculated_profile_similarity'),
                  target_met=value.get('full_profile_target_met'),
                  candidate=value.get('recipe') or value.get('closest_candidate'))
    report['timings'] = timings
    report['lp_threads_override'] = args.lp_threads
    report['lp_method_override'] = args.lp_method
    save('lp-calls.json', lp_calls)
    report['source_sha256'] = hashlib.sha256(Path(NaturalLanguagePerfumeryAI.__module__.replace('.', '/') + '.py').read_bytes()).hexdigest()
    if args.profile:
        profiler.dump_stats(str(args.output / 'recipe.prof'))
        stats = pstats.Stats(profiler)
        rows = []
        for (file, line, function), (primitive, calls, own, cumulative, callers) in stats.stats.items():
            rows.append({'file': file, 'line': line, 'function': function, 'calls': calls,
                         'own_seconds': own, 'cumulative_seconds': cumulative})
        save('profile.json', sorted(rows, key=lambda r: r['cumulative_seconds'], reverse=True))
    save('report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'candidate'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
