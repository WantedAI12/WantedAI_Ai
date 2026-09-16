"""Verify the source-bound installed wheel, both products and unknown handling."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--v78',action='store_true')
    a=p.parse_args();meta=json.loads(a.preparation.read_text(encoding='utf8'))
    if not a.worker:
        a.output.mkdir(parents=True,exist_ok=False)
        wheel=Path(meta['wheel']);installed=a.output/'installed';installed.mkdir()
        if hashlib.sha256(wheel.read_bytes()).hexdigest()!=meta['wheel_sha256']:raise ValueError('wheel hash')
        with zipfile.ZipFile(wheel) as z:
            if any(not (installed/n).resolve().is_relative_to(installed.resolve()) for n in z.namelist()):raise ValueError('wheel path')
            z.extractall(installed)
        env=dict(os.environ,PERFUMERY_AI_LOCAL_PROFILE=meta['profile'],PERFUMERY_AI_ENV='research',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf8')
        subprocess.run([sys.executable,__file__,'--preparation',str(a.preparation.resolve()),
            '--output',str(a.output.resolve()),'--worker',*(['--v78'] if a.v78 else [])],env=env,check=True)
        return
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(a.output/'installed'))
    import fragrance_ai
    assert Path(fragrance_ai.__file__).resolve().is_relative_to((a.output/'installed').resolve())
    from fastapi.testclient import TestClient
    from deploy.system_runtime_v76 import create_local_app
    checks=[]
    with TestClient(create_local_app(),raise_server_exceptions=False) as c:
        cases=[('capabilities','GET','/v1/ai/capabilities',None),
            ('vocabulary','GET','/v1/odor-expressions?q=lemon',None),
            ('interpret_citrus','POST','/v1/odor-expressions/interpret',{'text':'citrus scent'}),
            ('interpret_composition','POST','/v1/odor-expressions/interpret',{'text':'fruity floral scent'}),
            ('interpret_unknown','POST','/v1/odor-expressions/interpret',{'text':'quasifloral scent'}),
            ('perfume_citrus','POST','/v1/formulas',{'brief':'citrus scent','max_risk_tier':2,
                'enable_registry_trace_candidates':True,'target_similarity':95,'max_ingredients':12}),
            ('lotion_citrus','POST','/v1/applications/body-lotion/design',{'brief':'citrus scent',
                'max_risk_tier':2,'target_similarity':95}),
            ('perfume_unknown','POST','/v1/formulas',{'brief':'absinthe scent','max_risk_tier':2})]
        if a.v78:
            cases.extend([('interpret_inferred','POST','/v1/odor-expressions/interpret',{'text':'bergamot scent'}),
                ('perfume_inferred','POST','/v1/formulas',{'brief':'bergamot scent','max_risk_tier':2,
                    'enable_registry_trace_candidates':True,'target_similarity':95,'max_ingredients':12})])
        for name,method,path,payload in cases:
            start=time.perf_counter()
            r=c.request(method,path,json=payload) if payload is not None else c.request(method,path)
            try:value=r.json()
            except ValueError:value={'text':r.text}
            raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            (a.output/(name+'.json.gz')).write_bytes(gzip.compress(raw,mtime=0))
            row={'name':name,'http':r.status_code,'seconds':time.perf_counter()-start,
                'response_sha256':hashlib.sha256(raw).hexdigest()}
            if name=='perfume_citrus':
                row['score']=value.get('calculated_profile_similarity')
                row['version']=value.get('full_profile_assessment',{}).get('version')
                row['legacy_diagnostic_retained']='legacy_19_axis_assessment' in value.get('full_profile_assessment',{})
            if name=='lotion_citrus':row['score']=value.get('score')
            checks.append(row)
            (a.output/'progress.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
            print(json.dumps(row),flush=True)
            assert r.status_code==200,(name,value)
            if name=='perfume_citrus':
                assert row['version']=='hierarchical-perfume-reference/v77'
                assert row['legacy_diagnostic_retained']
            if name=='perfume_unknown':
                assert not value['recipe'] and not value['full_profile_target_met']
            if name=='interpret_unknown':assert value['unresolved_named_odors']
            if name=='interpret_inferred':
                target=value['representation']['hierarchical_target']
                assert target['status']=='resolved_source_reference'
                assert any(x['model_estimated'] for x in target['targets'][0]['reference_evidence'].values())
            if name=='perfume_inferred':
                target=value['full_profile_assessment']['reference_assessment']['intent']
                assert any(x['model_estimated'] for x in target['targets'][0]['reference_evidence'].values())
            time.sleep(.5)
    (a.output/'report.json').write_text(json.dumps({'checks':checks,'passed':True,
        'wheel_sha256':meta['wheel_sha256'],'profile':meta['profile'],'deployed':False},indent=2),encoding='utf8')


if __name__=='__main__':main()
