"""Frozen V54 plus V48-compatible stock supervision; nested composition CV."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_conditional_profiles_v2 as old
from scripts.train_mixture_core_v48 import AXES, choose as choose_v48
from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor
from fragrance_ai.research.conditional_profiles import feature_matrix, predict_conditional
from fragrance_ai.research.fine_odor_features import append_features
from fragrance_ai.research.mixture_profiles import mixture_features, predict_mixture_model
from fragrance_ai.research.perception_validation import group_folds, profile_errors, paired_profile_comparison
from fragrance_ai.research.unified_mixture import (
    atlas_mixture_features, prepare_views, view_kernel, residual_targets,
    restore_response, fit_unified, predict_unified,
)

V54 = ROOT/'.benchmarks/quantitative_profiles_v54/run-01'
V48 = ROOT/'.benchmarks/mixture_core_v48/run-03'
COMPONENT = ROOT/'.benchmarks/perception_core_v5/run-01/model.json'
PINS = {V54/'model.json':'4c01417cd4b9cbd37ea2ba88ca0a62784e5ead60bec9638b1e7bbfc7af557591',
        V48/'model.json':'5cdfbded3b40cdc46b046672a1fd3ff75faf2871b8024ca4f22dc8bdc95d3ae9',
        V48/'predictions.json':'af9593d2e9380c56c36c6ece1424145b11be540d18e7698c61ff1b8a516b8148',
        V48/'protocol.json':'14f2e78f705dbbe64a8f97618513113129e1920c1d9bd818bebc57270ac57d0f',
        COMPONENT:'33075f5af60d7579ebb2c3df8aeada146c782db73e576f7d4f41c1b9ef4732d9'}
OPTIONS = [dict(zip(('alpha','bandwidth','scaling','atlas_weight','output_transform'), row))
           for row in itertools.product((.01,.1,1.),(.25,1.,4.),('per_feature','shared_sensory'),
                                         (0.,.15,.5,1.),('identity','sqrt','log1p'))]


def choose(x, a, y, groups):
    folds = group_folds(groups, 3)
    losses = np.zeros(len(OPTIONS))
    for fold in range(3):
        tr, te = folds != fold, folds == fold
        for scaling, bw, weight in itertools.product(('per_feature','shared_sensory'),(.25,1.,4.),(0.,.15,.5,1.)):
            prepared = prepare_views(x[tr], a[tr], scaling)
            k = view_kernel(x[tr],a[tr],x[tr],a[tr],prepared,bw,weight)
            q = view_kernel(x[tr],a[tr],x[te],a[te],prepared,bw,weight)
            eigenvalues, eigenvectors = np.linalg.eigh(k)
            qe = q@eigenvectors
            for transform in ('identity','sqrt','log1p'):
                residual = residual_targets(x[tr], y[tr], transform)
                intercept = residual.mean(0)
                er = eigenvectors.T@(residual-intercept)
                for alpha in (.01,.1,1.):
                    correction = (qe/(eigenvalues+alpha))@er+intercept
                    pred = restore_response(x[te], correction, transform)
                    opt = {'alpha':alpha,'bandwidth':bw,'scaling':scaling,'atlas_weight':weight,'output_transform':transform}
                    idx = OPTIONS.index(opt)
                    errors = profile_errors(pred[:,AXES], y[te][:,AXES])
                    losses[idx] += np.nan_to_num(errors['cosine_distance'], nan=1.).sum()/len(y)
    choices = [{**opt,'loss':float(loss)} for opt,loss in zip(OPTIONS,losses)]
    # A predeclared stock-only ablation separates improved calibration from
    # contributions of the frozen V54 molecular representation.
    selected = min(choices,key=lambda r:r['loss'])
    stock_only = min((r for r in choices if r['atlas_weight'] == 0),key=lambda r:r['loss'])
    def fit(choice):
        return fit_unified(x,a,y,**{k:choice[k] for k in OPTIONS[0]})
    return fit(selected), fit(stock_only), {'selected':selected,'stock_only':stock_only,'choices':choices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.benchmarks'):
        parser.error('new local .benchmarks output required')
    for path, digest in PINS.items():
        if old.v1.sha(path) != digest:
            raise ValueError('frozen parent drift: '+str(path))
    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    _, _, stimuli, _ = old.extended_observations(args.source)
    bank = old.molecular_bank(args.source,ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    protected = {old.v1.formula_key(stimuli[r['stimulus']]) for name in ('target_ids','final_test_ids')
                 for r in old.v1.rows(args.source/old.INPUTS[name][0])}
    raw = old.v1.rows(args.source/old.INPUTS['training'][0])
    rows = [r for r in raw if old.v1.formula_key(stimuli[r['stimulus']]) not in protected]
    ids = [r['stimulus'] for r in rows]
    groups = [old.v1.formula_key(stimuli[i]) for i in ids]
    folds = group_folds(groups,5)
    legacy = json.loads((V48/'predictions.json').read_text(encoding='utf-8'))
    lp = json.loads((V48/'protocol.json').read_text(encoding='utf-8'))
    if ids != legacy['ids'] or groups != lp['composition_groups'] or folds.tolist() != lp['outer_folds'] or bindings != lp['inputs']:
        raise ValueError('same-case V48 comparison failed')
    component = json.loads(COMPONENT.read_text(encoding='utf-8'))
    if component['inputs'] != bindings:
        raise ValueError('stock component data provenance mismatch')
    keys = sorted({k for i in ids for k in stimuli[i]})
    stock_x = append_features(feature_matrix(keys,bank,'molecular'),
        [bank.get(k[0],{}).get('canonical_smiles','') for k in keys],component['fine_odor_features'])
    stock_p, _ = predict_conditional(component['component_model'],stock_x,keys)
    stocks = dict(zip(keys,stock_p))
    atlas = AtlasProfilePredictor(V54/'model.json',sha256=PINS[V54/'model.json'],experimental=True)
    graphs = sorted({bank[k[0]]['canonical_smiles'] for k in keys if k[0] in bank and '.' not in bank[k[0]]['canonical_smiles']})
    hi, lo = atlas.predict(graphs,reference_level='high'),atlas.predict(graphs,reference_level='low')
    embeddings = dict(zip(graphs,np.concatenate([hi['applicability'],hi['use'],lo['applicability'],lo['use']],axis=1)))
    x, a, missing = [], [], []
    for i in ids:
        ks = stimuli[i]
        identities = [bank.get(k[0],{}).get('canonical_smiles','unresolved:'+str(k[0])) for k in ks]
        supported = [s in embeddings for s in identities]
        p = [embeddings.get(s,np.zeros(584)) for s in identities]
        x.append(mixture_features([stocks[k] for k in ks],ks))
        a.append(atlas_mixture_features(p,identities,[float(k[1]) for k in ks],[k[2] for k in ks],supported=supported))
        if not all(supported): missing.append(i)
    x,a,y = np.asarray(x),np.asarray(a),np.asarray([old.v1.vector(r) for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError('all original V48 rows must be retained')
    out.mkdir(parents=True)
    for parent,name in ((V54,'frozen_v54'),(V48,'frozen_v48')):
        (out/name).mkdir()
        shutil.copy2(parent/'model.json',out/name/'model.json')
    code = [Path(__file__),ROOT/'fragrance_ai/research/unified_mixture.py',ROOT/'fragrance_ai/research/mixture_profiles.py',
            ROOT/'fragrance_ai/research/physsim_mx.py',ROOT/'fragrance_ai/research/atlas_profiles.py',
            ROOT/'scripts/train_mixture_core_v48.py',ROOT/'scripts/train_conditional_profiles_v2.py']
    hashes = {p.relative_to(ROOT).as_posix():old.v1.sha(p) for p in code}
    for p in code:
        target = out/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    protocol = {'schema':'v54-integrated-experiment/v1','inputs':bindings,'parent_hashes':{str(p.relative_to(ROOT)):v for p,v in PINS.items()},
        'ids':ids,'composition_groups':groups,'outer_folds':folds.tolist(),'inner_folds':3,'options':OPTIONS,
        'selection':'inner_composition_cv_49_axis_cosine_distance','source_sha256':hashes,
        'development_data_reused':True,'new_blind_evaluation':False,'protected_rows_removed':len(raw)-len(rows),
        'unsupported_atlas_retained_with_coverage_mask':missing,'denominator':len(ids),
        'promotion_requires':'all382 covered; lower MAE and cosine than V48; positive Atlas contribution vs stock-only ablation',
        'recipe95_gate_modified':False,'gas_or_lotion_calibration':False,'human_accuracy_claim':False}
    old.v1.write_new(out/'protocol.json',protocol)
    preds = {name:np.full_like(y,np.nan) for name in ('v48_recomputed','unified','stock_only')}
    selections = []
    start = time.perf_counter()
    for fold in range(5):
        tr,te = folds != fold,folds == fold
        tg = np.asarray(groups)[tr].tolist()
        baseline,bc = choose_v48(x[tr],y[tr],tg)
        preds['v48_recomputed'][te] = predict_mixture_model(baseline,x[te])
        model,stock_model,selection = choose(x[tr],a[tr],y[tr],tg)
        preds['unified'][te] = predict_unified(model,x[te],a[te])
        preds['stock_only'][te] = predict_unified(stock_model,x[te],a[te])
        selections.append({'fold':fold,'v48':bc['selected'],**selection})
        print(json.dumps({'fold':fold+1,'selected':selection['selected'],'seconds':round(time.perf_counter()-start,2)}),flush=True)
    np.testing.assert_allclose(preds['v48_recomputed'],np.asarray(legacy['nonlinear']),atol=1e-9,rtol=0)
    model,_,selection = choose(x,a,y,groups)
    comparisons = {name:paired_profile_comparison(preds[baseline][:,AXES],preds['unified'][:,AXES],y[:,AXES],groups)
                   for name,baseline in (('vs_v48','v48_recomputed'),('vs_stock_only','stock_only'))}
    accepted = all(c['paired_profiles'] == len(y) and c['baseline_mae_minus_candidate_mae'] > 0
                   and c['baseline_cosine_distance_minus_candidate'] > 0 for c in comparisons.values()) and model['atlas_weight'] > 0
    artifact = {'schema':'v54-integrated-candidate/v1','model':model,'endpoints':old.v1.ENDPOINTS,
        'component_sha256':PINS[COMPONENT],'atlas_sha256':PINS[V54/'model.json'],
        'v48_sha256':PINS[V48/'model.json'],'inputs':bindings,
        'application_domain':'relative_aliquots_of_explicit_stock_conditions',
        'local_development_accepted':bool(accepted),'runtime_promotion_allowed':False,
        'data_redistribution_authorized':False,'lotion_headspace_calibrated':False}
    old.v1.write_new(out/'model.json',artifact)
    restored = json.loads((out/'model.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(predict_unified(model,x,a),predict_unified(restored['model'],x,a),atol=1e-12,rtol=0)
    assert all(old.v1.sha(ROOT/p) == sha for p,sha in hashes.items())
    report = {'schema':protocol['schema'],'denominator':len(y),'predicted':len(y),'composition_count':len(set(groups)),
        'comparisons':comparisons,'selected':selection,'fold_selections':selections,'local_development_accepted':bool(accepted),
        'v48_oof_reproduced':True,'atlas_partial_rows_retained':len(missing),'outcome_read_attempts':denied,
        'model_sha256':old.v1.sha(out/'model.json'),'protocol_sha256':old.v1.sha(out/'protocol.json'),
        'seconds':time.perf_counter()-start,'new_blind_evaluation':False,'recipe400_measured':False,
        'bootstrap_is_development_composition_not_rater_uncertainty':True}
    old.v1.write_new(out/'report.json',report)
    old.v1.write_new(out/'predictions.json',{'ids':ids,**{k:v.tolist() for k,v in preds.items()}})
    print(json.dumps({'accepted':bool(accepted),'comparisons':comparisons,'seconds':report['seconds']}),flush=True)


if __name__ == '__main__':
    main()
