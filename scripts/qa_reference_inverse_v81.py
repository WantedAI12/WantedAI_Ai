"""Complete public route QA of a frozen local candidate, no deployment writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    a = p.parse_args()
    meta = json.loads(a.preparation.read_text(encoding='utf8'))
    if not a.worker:
        a.output.mkdir(parents=True,exist_ok=False)
        package = a.output/'installed'
        package.mkdir()
        wheel = Path(meta['wheel'])
        if hashlib.sha256(wheel.read_bytes()).hexdigest()!=meta['wheel_sha256']:
            raise ValueError('wheel drift')
        with zipfile.ZipFile(wheel) as z:
            if any(not (package/n).resolve().is_relative_to(package.resolve()) for n in z.namelist()):
                raise ValueError('wheel path escape')
            z.extractall(package)
        env = dict(os.environ,PERFUMERY_AI_LOCAL_PROFILE=meta['profile'],PERFUMERY_AI_ENV='research',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf8')
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],env=env,check=True)
        return
    sys.path.insert(0,str(ROOT))
    from scripts.audit_modal_v69_full import Audit,run
    sys.path.insert(0,str(a.output.resolve()/'installed'))
    import fragrance_ai
    assert Path(fragrance_ai.__file__).resolve().is_relative_to((a.output/'installed').resolve())
    from fastapi.testclient import TestClient
    from deploy.system_runtime_v76 import create_local_app
    report = {'wheel_sha256':meta['wheel_sha256'],'model_sha256':meta['model_sha256'],
        'scope':'local_frozen_wheel_full_public_route_qa','deployed':False,
        'target_reference_sha256':meta['target_reference_sha256'],'external_model_accuracy_verified':False}
    with TestClient(create_local_app(),raise_server_exceptions=False) as client:
        audit = Audit(a.output,client,{},True)
        audit.expected_wheel = meta['wheel_sha256']
        audit.expected_core = meta['model_sha256']
        audit.v76_observed_data = ROOT/'.benchmarks/v76_system_repair/observed-data-02'
        try:
            run(audit)
        except Exception as error:
            report.update(error_type=type(error).__name__,message=str(error)[:500])
        report.update(calls=len(audit.calls),checks=len(audit.checks),
            failed_calls=[r for r in audit.calls if not r['passed']],
            failed_checks=[r for r in audit.checks if not r['passed']])
    report['passed'] = not (report.get('error_type') or report['failed_calls'] or report['failed_checks'])
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
