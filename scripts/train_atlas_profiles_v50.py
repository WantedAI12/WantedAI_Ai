"""Full 146-endpoint molecule-disjoint development, without changing recipe scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fragrance_ai.research.atlas_profiles import load_atlas,atlas_features,fit_atlas,predict_atlas
from fragrance_ai.research.fine_odor_features import validate_features
from fragrance_ai.research.perception_validation import group_folds,fit_ridge,predict_ridge,profile_errors,paired_profile_comparison,summarize_profiles

OPTIONS = [(alpha,weight) for alpha in (.1,1.,10.) for weight in (0.,.25,.5,.75)]


def choose(x,y,groups,*,coarse=False):
    folds = group_folds(groups,3)
    options = [(a,None) for a in (1.,10.,100.)] if coarse else OPTIONS
    choices = []
    for alpha,weight in options:
        p = np.full(y.shape,np.nan)
        for fold in range(3):
            train,test = folds != fold,folds == fold
            model = fit_ridge(x[train],y[train],alpha) if coarse else fit_atlas(x[train],y[train],alpha,weight)
            p[test] = predict_ridge(model,x[test]) if coarse else predict_atlas(model,x[test])
        errors = profile_errors(p,y)
        choices.append({'alpha':alpha,'fine_weight':weight,
                        'loss':float(np.nan_to_num(errors['cosine_distance'],nan=1.).mean())})
    chosen = min(choices,key=lambda r:r['loss'])
    model = fit_ridge(x,y,chosen['alpha']) if coarse else fit_atlas(x,y,chosen['alpha'],chosen['fine_weight'])
    return model,{'selected':chosen,'options':choices}


def write(path,data):
    with path.open('x',encoding='utf-8') as handle: json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    from rdkit import Chem
    from scripts.verify_odor_concepts_v39 import read_catalog
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--catalog-manifest',type=Path,required=True)
    p.add_argument('--component',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new output directory required')
    rows,endpoints,source = load_atlas(args.source)
    component = json.loads(args.component.read_text(encoding='utf-8'))
    if component.get('schema') != 'perception-core-candidate/v5': raise ValueError('fixed source features required')
    fine = component['fine_odor_features']; validate_features(fine)
    catalog = read_catalog(args.catalog_manifest)
    # Keep every duplicate graph's available qualitative vector equally, rather
    # than selecting the material/profile which scores highest on this data.
    native_rows = {}
    for item in catalog.ingredients:
        if not item.structure_smiles or item.vector().sum() <= 0: continue
        mol = Chem.MolFromSmiles(item.structure_smiles)
        if mol is None or '.' in item.structure_smiles: continue
        graph = Chem.MolToSmiles(mol,isomericSmiles=True)
        native_rows.setdefault(graph,[]).append(item.vector())
    native = {g:np.mean(v,axis=0).tolist() for g,v in native_rows.items()}
    graphs,levels = [r['graph'] for r in rows],[r['level'] for r in rows]
    groups = [r['graph'] or 'unresolved:'+r['id'] for r in rows]
    x = atlas_features(graphs,levels,native,fine)
    coarse = np.c_[x[:,1040:1060],x[:,-2:]]
    present = np.isfinite(x).all(1)
    folds = group_folds(groups,5)
    protocol = {'scope':'full_Atlas_applicability_and_use_development_not_intensity_or_lotion_accuracy',
        'source':source,'endpoints':endpoints,'all_source_stimuli':len(rows),'predictable_stimuli':int(present.sum()),
        'resolved_graph_groups':len({g for g in graphs if g}),
        'unresolved_stimuli':[r['id'] for r in rows if r['graph'] is None],
        'outer_folds':folds.tolist(),'groups':groups,'options':OPTIONS,
        'baseline':'training_only_19_axis_to_146_endpoint_ridge_with_ordinal_condition',
        'numeric_concentration_or_solvent_invented':False,'endpoint_filtering_by_score':False,
        'code_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__),ROOT/'fragrance_ai/research/atlas_profiles.py')},
        'component_sha256':hashlib.sha256(args.component.read_bytes()).hexdigest(),
        'catalog_manifest_sha256':hashlib.sha256(args.catalog_manifest.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True)
    write(args.output/'protocol.json',protocol)
    started = time.perf_counter()
    predictions,models,reports,selections = {},{},{},{}
    for measurement in ('applicability','use'):
        y = np.array([r[measurement] for r in rows])
        predictions[measurement] = {key:np.full(y.shape,np.nan) for key in ('coarse','atlas')}
        selections[measurement] = []
        for fold in range(5):
            train,test = present&(folds != fold),present&(folds == fold)
            assert not set(np.asarray(groups)[train])&set(np.asarray(groups)[test])
            old,old_choice = choose(coarse[train],y[train],np.asarray(groups)[train].tolist(),coarse=True)
            model,choice = choose(x[train],y[train],np.asarray(groups)[train].tolist())
            predictions[measurement]['coarse'][test] = predict_ridge(old,coarse[test])
            predictions[measurement]['atlas'][test] = predict_atlas(model,x[test])
            selections[measurement].append({'fold':fold,'coarse':old_choice,'atlas':choice})
            print(f'Completed {measurement} molecule fold {fold+1}/5',flush=True)
        model,choice = choose(x[present],y[present],np.asarray(groups)[present].tolist())
        models[measurement] = model
        reports[measurement] = {'comparison':paired_profile_comparison(predictions[measurement]['coarse'],predictions[measurement]['atlas'],y,groups),
            'results':{k:summarize_profiles(v,y,groups) for k,v in predictions[measurement].items()},'selected':choice}
    artifact = {'schema':'atlas-profile-candidate/v1','models':models,'endpoints':endpoints,
        'fine_features':fine,'native_profiles':native,'source':source,
        'runtime_promotion_allowed':False,'data_redistribution_authorized':False,
        'numeric_dilution_calibrated':False,'lotion_matrix_calibrated':False}
    write(args.output/'model.json',artifact)
    report = {k:v for k,v in protocol.items() if k not in ('source','outer_folds','groups','code_sha256')}
    report.update(measurements=reports,selections=selections,seconds=time.perf_counter()-started,
        model_sha256=hashlib.sha256((args.output/'model.json').read_bytes()).hexdigest(),
        recipe400_measured=False,actual_human_accuracy_authorized=False)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha for name,sha in protocol['code_sha256'].items())
    write(args.output/'report.json',report)
    write(args.output/'predictions.json',{'ids':[r['id'] for r in rows],**{
        name:{k:np.where(np.isfinite(v),v,-1.).tolist() for k,v in values.items()} for name,values in predictions.items()}})
    print(json.dumps({'stimuli':len(rows),'predicted':int(present.sum()),'endpoints':len(endpoints),
        'seconds':report['seconds'],'comparisons':{k:v['comparison'] for k,v in reports.items()}},ensure_ascii=False))


if __name__ == '__main__': main()
