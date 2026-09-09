"""Actual V5 plus nonlinear-mixture SDK run; no held-out outcome reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from fragrance_ai import PerceptionGuidance,StockAliquot,StockMixturePredictor
    from fragrance_ai.recommender.models import ScentBrief,RecipeConstraints
    from scripts.verify_odor_concepts_v39 import read_catalog
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new evidence file required')
    component = ROOT/'.benchmarks/perception_core_v5/run-01/model.json'
    mixture = ROOT/'.benchmarks/mixture_core_v48/run-03/model.json'
    hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (ROOT/'fragrance_ai').rglob('*.py')}
    provider = PerceptionGuidance(ROOT/'.benchmarks/conditional_profiles_v2/final-01/models.json',
        ROOT/'benchmarks/industrial_ingredient_registry_v1.db',solvent='pg',experimental=True,
        component_model_path=component,component_model_sha256=hashlib.sha256(component.read_bytes()).hexdigest())
    predictor = StockMixturePredictor(provider,mixture,sha256=hashlib.sha256(mixture.read_bytes()).hexdigest(),experimental=True)
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
    aliquots = [StockAliquot(selected[0],.1,1.,'pg'),StockAliquot(selected[1],.01,1.,'pg')]
    start = time.perf_counter(); result = predictor.predict(aliquots); elapsed = time.perf_counter()-start
    split = predictor.predict([aliquots[1],StockAliquot(selected[0],.1,.5,'pg'),StockAliquot(selected[0],.1,.5,'pg')])
    np.testing.assert_allclose(list(result['predicted_rata_profile'].values()),list(split['predicted_rata_profile'].values()),atol=1e-12,rtol=0)
    pure = predictor.predict(aliquots[:1])
    scalar,_ = session._predict(selected[0],np.array([.1]))
    np.testing.assert_allclose(list(pure['predicted_rata_profile'].values()),scalar[0],atol=1e-12,rtol=0)
    # The new paired batch consumes one requested condition per material,
    # matching the existing scalar component path at those exact stocks.
    paired,_,_ = session.predict_stock_conditions(selected,[.1,.01],['pg','pg'])
    expected = np.array([session._predict(item,np.array([dose]))[0][0] for item,dose in zip(selected,[.1,.01])])
    np.testing.assert_allclose(paired,expected,atol=1e-12,rtol=0)
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha for path,sha in hashes.items())
    report = {'scope':'actual_local_stock_assay_SDK_not_lotion_API_or_accuracy_measurement','seconds':elapsed,
        'result':result,'single_stock_identity_verified':True,'aliquot_split_invariance_verified':True,
        'paired_batch_matches_scalar':True,'source_unchanged':True,'source_sha256':hashes,'deployed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','result')},ensure_ascii=False))


if __name__ == '__main__': main()
