"""Load the packaged trained model and exercise both real local API products."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--version',choices=['v73','v75'],default='v73')
    args=parser.parse_args()
    prepared=json.loads(args.preparation.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False)
    package=args.output.resolve()/'installed'
    package.mkdir()
    if hashlib.sha256(Path(prepared['wheel']).read_bytes()).hexdigest()!=prepared['wheel_sha256']:
        raise ValueError('candidate wheel changed')
    with zipfile.ZipFile(prepared['wheel']) as archive:
        if any(not (package/name).resolve().is_relative_to(package) for name in archive.namelist()):
            raise ValueError('escaping wheel member')
        archive.extractall(package)
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(package))
    os.environ.update(PERFUMERY_AI_ENV='research',PERFUMERY_AI_LOCAL_PROFILE=prepared['profile'])
    import numpy as np
    from fragrance_ai.recommender.formulation_core import configured_formulation_core,FormulationCore
    if args.version=='v75':
        from deploy.autoregressive_runtime_v75 import create_local_app
    else:
        from deploy.autoregressive_runtime_v73 import create_local_app
    from fastapi.testclient import TestClient
    import fragrance_ai
    if not Path(fragrance_ai.__file__).resolve().is_relative_to(package):
        raise ValueError('working source substituted for packaged model')
    selected=configured_formulation_core()
    old=FormulationCore(ROOT/'.benchmarks/formulation_core_v69/train-04/model.json',
        '3186e467446abc71f249c052c8248c17c97d0e337976abf3666628af47dd997f')
    protected=old.arrays if not selected.manifest.get('profile_refit') else {
        k:v for k,v in old.arrays.items() if not k.startswith(('molecule_in.','molecule_out.','quantitative_head.'))}
    if any(not np.array_equal(value,selected.arrays[key]) for key,value in protected.items()):
        raise ValueError('original model parameters changed')
    results=[]
    with TestClient(create_local_app(),raise_server_exceptions=False) as client:
        health=client.get('/health')
        assert health.status_code==200 and health.json()['wheel_sha256']==prepared['wheel_sha256']
        for product,brief in [('perfume','floral scent'),('perfume','clean fresh citrus woody'),('perfume','citrus scent'),
                ('body_lotion','green scent'),('body_lotion','fresh scent'),('body_lotion','citrus scent')]:
            time.sleep(3.1)
            request={'brief':brief,'max_risk_tier':2,'target_similarity':95.}
            if product=='perfume':
                request.update(enable_registry_trace_candidates=True,require_full_profile_match=True)
                route='/v1/formulas'
            else:
                request['registry_pool']='conditional_research'
                route='/v1/applications/body-lotion/design'
            started=time.perf_counter()
            response=client.post(route,json=request)
            try:
                value=response.json()
            except ValueError:
                value={'non_json_response':response.text,'http':response.status_code}
            filename=f'{len(results):02d}-{product}.json'
            (args.output/filename).write_text(json.dumps({'request':request,'response':value},ensure_ascii=False,indent=2),encoding='utf-8')
            if product=='perfume':
                feedback=value.get('full_profile_assessment',{}).get('search',{}).get('full_pool_search',{}).get('dose_refinement',{}).get('autoregressive_refinement',{})
                neural=feedback.get('neural_autoregressive')
                usage=feedback.get('neural_usage',{})
                score=value.get('calculated_profile_similarity')
                met=value.get('full_profile_target_met')
            else:
                search=value.get('material_column_search',{})
                feedback=search.get('autoregressive_refinement',{})
                neural=search.get('neural_autoregressive')
                usage=search.get('neural_usage',{})
                score=value.get('score')
                met=value.get('profile_target_met')
            row={'product':product,'brief':brief,'http':response.status_code,'score':score,
                'target_met':met,'feedback':feedback,'neural':neural,'usage':usage,'seconds':time.perf_counter()-started}
            results.append(row)
            (args.output/'progress.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(row,ensure_ascii=False),flush=True)
            if response.status_code!=200:
                raise ValueError('packaged autoregressive API failed; raw response preserved')
            if met and (score is None or score+1e-7<95.):
                raise ValueError('false pass')
    report={'completed':True,'checks':len(results),'model_sha256':selected.sha256,
        'wheel_sha256':prepared['wheel_sha256'],'protected_parameter_arrays_exactly_retained':True,
        'profile_refit_applied':bool(selected.manifest.get('profile_refit')),
        'neural_called_for':sorted({r['product'] for r in results if r['neural']}),
        'scope':'packaged_API_integration_not_whole_400_quality_benchmark',
        'deployment_changed':False,'torch_imported_in_inference_process':'torch' in sys.modules,'results':results}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if set(report['neural_called_for'])!={'perfume','body_lotion'}:
        raise ValueError('both actual product branches must call the trained decoder')


if __name__=='__main__':
    main()
