"""Fresh installed-package inference and the backend's multi-step API contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads(args.preparation.read_text(encoding='utf8'))
    assert hashlib.sha256(Path(meta['wheel']).read_bytes()).hexdigest() == meta['wheel_sha256']
    sys.path[:0] = [str(ROOT/'tmp/lotion-latency-v89-deps'), meta['installed'], str(ROOT)]
    os.environ.update(PERFUMERY_AI_LOCAL_PROFILE=meta['profile'], PERFUMERY_AI_ENV='research',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import fragrance_ai
    assert Path(fragrance_ai.__file__).resolve().is_relative_to(Path(meta['installed']))
    from deploy.target_runtime_v87 import create_release_app
    from fastapi.testclient import TestClient
    check = runpy.run_path(str(ROOT/'deploy/verify_backend_v91.py'))['run_backend_checks']
    args.output.mkdir(parents=True, exist_ok=False)
    app = create_release_app(registry_path=str(ROOT/'benchmarks/industrial_ingredient_registry_v1.db'))
    with TestClient(app) as client:
        report = check(client, meta['wheel_sha256'])
    for row in report['calls']:
        raw = row.pop('response_body').encode('utf8')
        suffix = 'sse' if 'event-stream' in row['response_headers'].get('content-type', '') else 'json'
        path = args.output/(row['name']+'.response.'+suffix)
        path.write_bytes(raw)
        row['response_file'] = path.name
        if row.get('request') is not None:
            (args.output/(row['name']+'.request.json')).write_text(
                json.dumps(row['request'], ensure_ascii=False, indent=2), encoding='utf8')
    report.update(execution='installed_package_fresh_local_inference_and_ASGI', deployed=False,
        wheel_sha256=meta['wheel_sha256'], model_weights_sha256=meta['model_sha256'])
    (args.output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({key: value for key, value in report.items() if key != 'calls'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
