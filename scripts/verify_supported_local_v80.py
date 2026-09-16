"""Exercise an extracted source-bound wheel without touching production."""
import argparse
import gzip
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
        target = a.output/'installed'
        target.mkdir()
        wheel = Path(meta['wheel'])
        if hashlib.sha256(wheel.read_bytes()).hexdigest()!=meta['wheel_sha256']:
            raise ValueError('wheel drift')
        with zipfile.ZipFile(wheel) as z:
            if any(not (target/n).resolve().is_relative_to(target.resolve()) for n in z.namelist()):
                raise ValueError('package path escape')
            z.extractall(target)
        env = dict(os.environ, PERFUMERY_AI_LOCAL_PROFILE=meta['profile'],PERFUMERY_AI_ENV='research',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf8')
        subprocess.run([sys.executable,__file__,'--preparation',str(a.preparation.resolve()),
            '--output',str(a.output.resolve()),'--worker'],env=env,check=True)
        return
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(a.output/'installed'))
    import fragrance_ai
    assert Path(fragrance_ai.__file__).resolve().is_relative_to((a.output/'installed').resolve())
    from deploy.system_runtime_v76 import create_local_app
    from deploy.verify_supported_api_v80 import run_checks
    from fastapi.testclient import TestClient
    def save(name,value):
        raw = json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        safe = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
        (a.output/(safe+'.json.gz')).write_bytes(gzip.compress(raw,mtime=0))
    with TestClient(create_local_app(),raise_server_exceptions=False) as client:
        report = run_checks(client,meta['wheel_sha256'],save)
    report.update(wheel_sha256=meta['wheel_sha256'],profile=meta['profile'],deployed=False)
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))


if __name__=='__main__':
    main()
