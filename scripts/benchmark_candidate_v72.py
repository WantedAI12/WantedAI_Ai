"""Execute all frozen requests against a pinned improvement candidate.

The separated/shared experiment is not interrupted or reused as fresh work.
Its exact requests and threshold define this independent 800-call execution.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_worker(package, profile, digest, output):
    from scripts.benchmark_model_ablation_v72 import init_worker
    if importlib.metadata.version('clarabel') != '0.11.1':
        raise ValueError('the candidate requires its pinned cone solver')
    init_worker(package, profile, digest, output, 'candidate')


def main():
    from scripts.benchmark_model_ablation_v72 import evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--ablation-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    binding = json.loads(args.candidate.read_text(encoding='utf-8'))
    before = json.loads(args.ablation_protocol.read_text(encoding='utf-8'))
    cases, products = before['cases'], before['products']
    if (len(cases) != 400 or len({c['id'] for c in cases}) != 400
            or set(products) != {'perfume', 'body_lotion'} or before['target'] != 95.):
        raise ValueError('the entire fixed 400-per-product protocol is required')
    if sha(binding['wheel']) != binding['wheel_sha256'] or sha(binding['profile']) != binding['profile_sha256']:
        raise ValueError('candidate wheel or profile changed')
    protocol = {'cases': cases, 'case_count': 400, 'products': products,
        'versions': {'candidate': binding}, 'expected_results': 800, 'target': 95.,
        'fresh_executions': True, 'script_sha256': sha(__file__),
        'worker_script_sha256': sha(ROOT/'scripts/benchmark_model_ablation_v72.py'),
        'ablation_protocol_sha256': sha(args.ablation_protocol), 'workers': args.workers,
        'dependency_versions': {name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'rdkit', 'clarabel')},
        'scope': 'same_fixed_cases_and_threshold_full_catalog_improvement_not_human_accuracy'}
    output = args.output.resolve()
    if args.resume:
        if json.loads((output/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('resume protocol drift')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'responses').mkdir()
        (output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    rows_path = output/'results.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()] if rows_path.exists() else []
    seen = {(r['version'], r['product'], r['case_id']) for r in rows}
    expected = {('candidate', product, case['id']) for product in products for case in cases}
    if len(seen) != len(rows) or not seen <= expected:
        raise ValueError('duplicate or out-of-protocol saved results')
    package = output/'candidate-package'
    if not package.exists():
        package.mkdir()
        with zipfile.ZipFile(binding['wheel']) as archive:
            if any(not (package/name).resolve().is_relative_to(package) for name in archive.namelist()):
                raise ValueError('escaping package entry')
            archive.extractall(package)
    with ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=candidate_worker,
            initargs=(str(package), binding['profile'], binding['profile_sha256'], str(output))) as pool:
        futures = [pool.submit(evaluate, case, product) for case in cases for product in products
                   if ('candidate', product, case['id']) not in seen]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with rows_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            summary = {'completed': len(rows), 'expected': 800, 'complete': len(rows) == 800,
                'updated_at': datetime.now(timezone.utc).isoformat(), 'groups': {}}
            for product in products:
                subset = [r for r in rows if r['product'] == product]
                summary['groups'][product] = {'evaluated': len(subset),
                    'passed95': sum(r.get('target_met') is True for r in subset),
                    'errors': sum('error' in r or r.get('http', 500) >= 500 for r in subset),
                    'no_score': sum(r.get('score') is None for r in subset)}
            (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            if len(rows) % 20 == 0 or summary['complete']:
                print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
