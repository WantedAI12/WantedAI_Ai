"""V66 molecular backbone training with frozen molecule/scaffold outer folds.

No recipe outcomes, output-axis selection, test-set tuning or external model.
Both measurement heads retain every source endpoint. This is development data,
not a newly collected blind study or a user-similarity accuracy certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fragrance_ai.research.atlas_profiles import load_atlas, atlas_features, fit_atlas_structured, predict_atlas
from fragrance_ai.research.perception_validation import group_folds, profile_errors, paired_profile_comparison, summarize_profiles
from scripts.train_quantitative_profiles_v54 import choose as choose_baseline

ROOT = Path(__file__).resolve().parents[1]

BASELINE_SHA = '4c01417cd4b9cbd37ea2ba88ca0a62784e5ead60bec9638b1e7bbfc7af557591'
OPTIONS = [(a,w,t,rho) for a in (.1,1.,10.) for w in (0.,.25,.5,.75)
           for t in ('identity','sqrt','log1p') for rho in (0.,.5,.9)]


def choose(x,y,groups):
    folds = group_folds(groups,3)
    records = []
    for alpha,weight,transform,coupling in OPTIONS:
        output = np.full(y.shape,np.nan)
        for fold in range(3):
            train,test = folds != fold,folds == fold
            model = fit_atlas_structured(x[train],y[train],alpha,weight,
                target_transform=transform,output_coupling=coupling)
            output[test] = predict_atlas(model,x[test])
        errors = profile_errors(output,y)
        records.append({'alpha':alpha,'fine_weight':weight,'target_transform':transform,
            'output_coupling':coupling,'loss':float(np.nan_to_num(errors['cosine_distance'],nan=1.).mean())})
    selected = min(records,key=lambda r:r['loss'])
    return fit_atlas_structured(x,y,selected['alpha'],selected['fine_weight'],
        target_transform=selected['target_transform'],output_coupling=selected['output_coupling']), {
        'selected':selected, 'options':records}


def write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    from rdkit.Chem.Scaffolds import MurckoScaffold
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    baseline_path = ROOT/'.benchmarks/quantitative_profiles_v54/run-01/model.json'
    raw = baseline_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError('frozen V54 main backbone drift')
    baseline = json.loads(raw)
    rows,endpoints,source = load_atlas(ROOT/'.benchmarks/atlas_profiles_v50/source')
    if endpoints != baseline['endpoints'] or source != baseline['source']:
        raise ValueError('baseline source/endpoint identity mismatch')
    x = atlas_features([r['graph'] for r in rows],[r['level'] for r in rows],
                       baseline['native_profiles'],baseline['fine_features'])
    present = np.isfinite(x).all(axis=1)
    groups = [r['graph'] or 'unresolved:'+r['id'] for r in rows]
    scaffold = [('scaffold:'+MurckoScaffold.MurckoScaffoldSmiles(smiles=r['graph'],includeChirality=False))
                if r['graph'] else 'unresolved:'+r['id'] for r in rows]
    # All acyclic molecules share the empty scaffold group. Do not relabel
    # them by molecule and call the resulting easier split scaffold-disjoint.
    splits = {'molecule':groups,'scaffold':scaffold}
    snapshot = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/'fragrance_ai/research/atlas_profiles.py',ROOT/'fragrance_ai/research/structured_kernel.py')}
    protocol = {'schema':'main-backbone-v66-development/v1','source':source,'baseline_sha256':BASELINE_SHA,
        'stimulus_ids':[r['id'] for r in rows], 'source_stimuli':len(rows),'predictable_stimuli':int(present.sum()),
        'endpoints':endpoints,'options':OPTIONS,'inner_group_folds':3,
        'outer_splits':{name:{'groups':g,'folds':group_folds(g,5).tolist()} for name,g in splits.items()},
        'selection_metric':'complete_profile_cosine_distance','new_blind_evaluation':False,
        'recipe_results_used':False,'source_sha256':snapshot,
        'local_acceptance_rule':'both_heads_raw_MAE_and_cosine_nonregression_on_both_outer_splits_without_coverage_loss',
        'acceptance_is_not_statistical_superiority':True}
    write(args.output/'protocol.json',protocol)
    started, reports, predictions, selections = time.perf_counter(),{},{},{}
    for name,group in splits.items():
        folds = group_folds(group,5)
        reports[name],predictions[name],selections[name] = {},{},{}
        for head in ('applicability','use'):
            y = np.array([r[head] for r in rows])
            outputs = {key:np.full(y.shape,np.nan) for key in ('baseline','candidate')}
            selections[name][head] = []
            for fold in range(5):
                train,test = present&(folds != fold),present&(folds == fold)
                assert not set(np.asarray(group)[train])&set(np.asarray(group)[test])
                old,old_choice = choose_baseline(x[train],y[train],np.asarray(group)[train].tolist())
                new,new_choice = choose(x[train],y[train],np.asarray(group)[train].tolist())
                outputs['baseline'][test],outputs['candidate'][test] = predict_atlas(old,x[test]),predict_atlas(new,x[test])
                selections[name][head].append({'fold':fold,'baseline':old_choice['selected'],
                                              'candidate':new_choice['selected']})
                print(json.dumps({'split':name,'head':head,'fold':fold+1,'candidate':new_choice['selected']}),flush=True)
            reports[name][head] = {'comparison':paired_profile_comparison(outputs['baseline'],outputs['candidate'],y,group),
                **{k:summarize_profiles(v,y,group) for k,v in outputs.items()}}
            predictions[name][head] = {k:np.where(np.isfinite(v),v,-1).tolist() for k,v in outputs.items()}
    models, final_choice = {},{}
    for head in ('applicability','use'):
        y = np.array([r[head] for r in rows])
        models[head],choice = choose(x[present],y[present],np.asarray(groups)[present].tolist())
        final_choice[head] = choice
    acceptable = all(value['comparison']['baseline_mae_minus_candidate_mae'] >= 0
        and value['comparison']['baseline_cosine_distance_minus_candidate'] >= 0
        and all(value['candidate'][metric]['defined_profiles'] >= value['baseline'][metric]['defined_profiles']
                for metric in ('mae','cosine_distance'))
        for split in reports.values() for value in split.values())
    artifact = {**baseline,'models':models,'parent_checkpoint_sha256':BASELINE_SHA,
        'quantitative_target_model_version':'structured-multioutput-atlas/v66',
        'training_executed':True,'development_nonregression_passed':acceptable,
        'training_protocol_sha256':hashlib.sha256((args.output/'protocol.json').read_bytes()).hexdigest()}
    write(args.output/'model.json',artifact)
    report = {'scope':'existing_source_molecule_and_scaffold_development_not_user_accuracy',
        'measurements':reports,'selections':selections,'full_training_selection':final_choice,
        'development_nonregression_passed':acceptable,'runtime_promoted':False,
        'seconds':time.perf_counter()-started,'model_sha256':hashlib.sha256((args.output/'model.json').read_bytes()).hexdigest()}
    for path,digest in snapshot.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError('training source changed during execution')
    write(args.output/'predictions.json',{'ids':[r['id'] for r in rows],**predictions})
    write(args.output/'report.json',report)
    print(json.dumps({'comparison':{s:{h:v['comparison'] for h,v in heads.items()} for s,heads in reports.items()},
        'accepted':acceptable,'seconds':report['seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
