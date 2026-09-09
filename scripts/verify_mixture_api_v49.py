"""Real local product API and frozen mixture checkpoint; never deploys."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from scripts.verify_odor_concepts_v39 import read_catalog
    from fragrance_ai.recommender.models import ScentBrief,RecipeConstraints
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new evidence file required')
    sources = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (ROOT/'fragrance_ai').rglob('*.py')}
    app = create_app()
    provider = configured_perception()
    catalog = read_catalog(ROOT/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json')
    session = provider.begin(ScentBrief('',{},[],[],[],[],'medium',{},RecipeConstraints()))
    selected,ids = [],set()
    for item in catalog.ingredients:
        if not session.supports(item): continue
        cid,_ = session._prepare(item)
        if cid is not None and cid not in ids:
            selected.append(item); ids.add(cid)
            if len(selected) == 2: break
    assert len(selected) == 2
    # Explicit densities here are controlled numerical inputs for a conversion
    # check, NOT measurements asserted for these two actual materials.
    body = {'basis':'supplied_mass','components':[
        {'ingredient_id':i.ingredient_id,'stock_dilution':dose,'supplied_mass_g':mass,
         'stock_density_g_ml':density,'solvent':'pg'}
        for i,dose,mass,density in zip(selected,(.1,.01),(2.,1.),(2.,1.))]}
    with TestClient(app) as client:
        start = time.perf_counter()
        response = client.post('/v1/formulations/stock-mixture/predict',json=body)
        elapsed = time.perf_counter()-start
        assert response.status_code == 200,response.text
        cached = client.post('/v1/formulations/stock-mixture/predict',json=body)
        assert cached.status_code == 200 and cached.json() == response.json()
        assert cached.headers['X-Perfumery-Stock-Cache'] == 'hit'
        by_volume = {'basis':'relative_volume','components':[
            {k:v for k,v in row.items() if k not in ('supplied_mass_g','stock_density_g_ml')}|{'relative_volume':1.}
            for row in body['components']]}
        volume = client.post('/v1/formulations/stock-mixture/predict',json=by_volume)
        assert volume.status_code == 200,volume.text
        assert volume.json()['predicted_rata_profile'] == response.json()['predicted_rata_profile']
        caps = client.get('/v1/ai/capabilities').json()
        assert caps['features']['explicit_stock_mixture_prediction']
        assert caps['product_models']['body_lotion']['component_model']['product'] == 'body_lotion'
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha for path,sha in sources.items())
    report = {'scope':'actual_local_API_forward_not_generated_recipe_accuracy_or_lotion_validation',
        'density_inputs_are_test_values_not_measurements':True,'seconds_after_model_load':elapsed,
        'request':body,'response':response.json(),'cache_hit_verified':True,'mass_volume_equivalence_verified':True,
        'existing_product_lanes_preserved':True,'source_unchanged':True,'source_sha256':sources,'deployed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('request','response','source_sha256')},ensure_ascii=False))


if __name__ == '__main__': main()
