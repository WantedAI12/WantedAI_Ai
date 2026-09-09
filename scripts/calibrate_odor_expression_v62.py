"""Fit V61 correction values using only its validation identities.

Selection uses scaffold-group OOF predictions on the 580 validation molecules.
The previously reported V61 test set is a regression holdout, not a new blind
human study. Its labels are opened only after the correction is selected/fixed.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from scipy.optimize import minimize
from fragrance_ai.recommender.fine_odor_model import FineOdorModel,structure_features
from fragrance_ai.recommender.odor_calibration import SCHEMA,LABEL_KIND,probability_logits,apply_correction,neighbor_probabilities
from scripts.train_odor_expression_v61 import metrics as annotation_metrics

PARENT = ROOT/'.benchmarks/odor_expression_v61/run-02/model.json'
PARENT_SHA = '418afec8b8363adb45ea78517b11c3a45de71404cb58b85ffad5c80605c6ce58'
SPLIT_SHA = '65cd756f980214f3763ac491a688acca5491202ef4ab4838af47c1024e4b8007'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def fit_correction(predictions,target,*,label_penalty=None,method='logit',neighbor=None,neighbor_k=None):
    p=np.asarray(predictions,float); y=np.asarray(target,float)
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or np.any((y!=0)&(y!=1)):
        raise ValueError('calibration targets must be binary annotation records')
    if method=='neighbors':
        from scipy.optimize import minimize_scalar
        if neighbor is None or np.shape(neighbor)!=p.shape:
            raise ValueError('training-only neighbor predictions required')
        def loss(alpha):
            q=np.clip((1.-alpha)*p+alpha*neighbor,1e-7,1.-1e-7)
            return float(-np.mean(y*np.log(q)+(1.-y)*np.log1p(-q)))
        result=minimize_scalar(loss,bounds=(0.,.5),method='bounded',options={'xatol':1e-8})
        if not result.success:raise ValueError('neighbor correction fitting failed')
        alpha=min((0.,float(result.x),.5),key=loss)
        return {'method':'neighbors','slope':1.,'negative_log_slope':1.,'intercept':0.,
            'label_offsets':np.zeros(p.shape[1]).tolist(),'label_penalty':None,
            'neighbor_weight':alpha,'neighbor_k':neighbor_k,'neighbor_power':4.,'neighbor_prior_fraction':.1,
            'directly_fitted_label_indices':[],'optimization_iterations':int(result.nfev)}
    logits=probability_logits(p)
    if method not in ('logit','beta','beta_tail'):
        raise ValueError('unknown calibration method')
    tail=-np.log1p(-np.clip(p,1e-7,1.-1e-7))
    if method=='logit':
        features=np.stack([logits,np.ones_like(p)],axis=0)
        initial=np.array([1.,0.]);bounds=[(.25,4.),(-5.,5.)];base=np.zeros_like(p)
    elif method=='beta':
        features=np.stack([logits-tail,tail,np.ones_like(p)],axis=0)
        initial=np.array([1.,1.,0.]);bounds=[(.25,4.),(.25,4.),(-5.,5.)];base=np.zeros_like(p)
    else:
        features=tail[None,:,:]
        initial=np.array([0.]);bounds=[(-.75,3.)];base=logits
    n_global=len(initial)
    positives=y.sum(axis=0)
    active=np.flatnonzero((positives>=5)&((len(y)-positives)>=5)) if label_penalty is not None else np.array([],int)
    # L2 acts like a zero-centered prior on descriptor-specific correction.
    # Rare labels (<5) receive only global temperature/intercept correction.
    denominator=y.size
    def objective(theta):
        offsets=np.zeros(y.shape[1]); offsets[active]=theta[n_global:]
        z=base+np.einsum('k,kij->ij',theta[:n_global],features)+offsets
        q=1./(1.+np.exp(-np.clip(z,-50.,50.)))
        residual=q-y
        loss=np.logaddexp(0.,z).sum()-np.sum(y*z)
        deviation=theta[:n_global]-initial
        loss+=5.*np.dot(deviation,deviation)
        gradient=np.r_[np.einsum('ij,kij->k',residual,features)+10.*deviation,residual.sum(axis=0)[active]]
        if len(active):
            loss+=.5*label_penalty*np.dot(theta[n_global:],theta[n_global:])
            gradient[n_global:]+=label_penalty*theta[n_global:]
        return float(loss/denominator),gradient/denominator
    result=minimize(objective,np.r_[initial,np.zeros(len(active))],jac=True,method='L-BFGS-B',
        bounds=bounds+[(-3.,3.)]*len(active),
        options={'maxiter':300,'ftol':1e-12,'gtol':1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('calibration optimization failed: '+str(result.message))
    offsets=np.zeros(y.shape[1]);offsets[active]=result.x[n_global:]
    a,b,c=(result.x[0],result.x[0],result.x[1]) if method=='logit' else (
        tuple(result.x[:3]) if method=='beta' else (1.,1.+result.x[0],0.))
    return {'slope':float(a),'negative_log_slope':float(b),'intercept':float(c),'method':method,
        'label_offsets':offsets.tolist(),
        'directly_fitted_label_indices':active.tolist(),'label_penalty':label_penalty,
        'minimum_positive_records':5,'optimization_iterations':int(result.nit)}


def apply(p,coefficients,neighbor=None):
    if coefficients.get('method')=='neighbors':
        if neighbor is None or np.shape(neighbor)!=np.shape(p):
            raise ValueError('neighbor correction requires matched structure predictions')
        alpha=coefficients['neighbor_weight']
        p=(1.-alpha)*p+alpha*neighbor
    return apply_correction(p,coefficients['slope'],coefficients['intercept'],coefficients['label_offsets'],
                            coefficients.get('negative_log_slope'))


def metrics(y,p,train_support):
    p=np.asarray(p,float);y=np.asarray(y,float)
    report=annotation_metrics(y,p,train_support)
    report['brier_loss']=float(np.mean((p-y)**2))
    # Equal-width reliability bins; full counts retained so mostly-zero
    # annotation data cannot conceal the class balance or empty bins.
    index=np.minimum(14,(p.ravel()*15).astype(int))
    bins=[];ece=0.
    for i in range(15):
        take=index==i;n=int(take.sum())
        predicted=float(p.ravel()[take].mean()) if n else None
        observed=float(y.ravel()[take].mean()) if n else None
        if n: ece+=n/p.size*abs(predicted-observed)
        bins.append({'lower':i/15,'upper':(i+1)/15,'count':n,'predicted':predicted,'observed':observed})
    report.update(ece_15_equal_width=ece,reliability_bins=bins,positive_annotation_fraction=float(y.mean()))
    return report


def eligible(candidate,baseline):
    return (candidate['source_annotation_binary_log_loss']<baseline['source_annotation_binary_log_loss']-1e-8
        and candidate['brier_loss']<=baseline['brier_loss']+1e-10
        and candidate['micro_average_precision']>=baseline['micro_average_precision']-1e-8
        and candidate['precision_at_5']>=baseline['precision_at_5']-1e-8
        and candidate['macro_average_precision']>=baseline['macro_average_precision']-1e-8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('logit','beta','neighbors'),default='logit')
    args=p.parse_args();out=args.output
    if out.exists():raise ValueError('use a new output directory')
    out.mkdir(parents=True)
    start=time.perf_counter()
    if sha(PARENT)!=PARENT_SHA or sha(PARENT.parent/'split.json')!=SPLIT_SHA:
        raise ValueError('frozen parent/split changed')
    parent=FineOdorModel(PARENT,PARENT_SHA)
    split=json.loads((PARENT.parent/'split.json').read_text(encoding='utf-8'))['records']
    source=parent.annotations
    train=[r for r in split if r['split']==0]
    validation=[r for r in split if r['split']==1]
    test=[r for r in split if r['split']==2]
    for a,b in ((train,validation),(train,test),(validation,test)):
        if {r['group'] for r in a}&{r['group'] for r in b}:raise ValueError('source split leaks scaffold groups')
    def labels(rows):
        y=np.zeros((len(rows),len(parent.endpoints)))
        for i,r in enumerate(rows):y[i,source[r['graph']]]=1.
        return y
    train_support=labels(train).sum(axis=0)
    y=labels(validation)
    predictions=parent.predict([r['graph'] for r in validation])
    fold=np.asarray([int(hashlib.sha256(('620910'+r['group']).encode()).hexdigest()[:8],16)%4 for r in validation])
    if any((fold==f).sum()<20 for f in range(4)):raise ValueError('insufficient calibration crossfit fold')
    baseline=metrics(y,predictions,train_support)
    candidates=[]
    # Search budget and acceptance rules fixed before opening test outcomes.
    plan={'schema':'odor-calibration-experiment/v1','parent_sha256':PARENT_SHA,'split_sha256':SPLIT_SHA,
        'fit_source':'V61 validation only; V61 train/test excluded',
        'prior_model_selection_used_same_validation_set':True,
        'crossfit_folds':4,'fold_counts':dict(Counter(str(v) for v in fold)),
        'candidates':([{'method':'logit','label_penalty':v} for v in [None,10.,50.,200.]] if args.family=='logit'
            else ([{'method':'beta','label_penalty':v} for v in [None,50.,200.]]+[
                {'method':'beta_tail','label_penalty':None}] if args.family=='beta' else [
                {'method':'neighbors','neighbor_k':k,'label_penalty':None} for k in [16,64,128]])),
        'method_source':('https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html'
            if args.family=='neighbors' else ('https://proceedings.mlr.press/v54/kull17a.html'
            if args.family=='beta' else 'https://scikit-learn.org/stable/modules/calibration.html')),
        'selection':'minimum OOF log loss among candidates with nonregressing Brier, micro/macro AP and precision@5',
        'test_role':'previously_reported_regression_holdout_not_new_blind_data',
        'test_used_for_coefficient_or_hyperparameter_fit':False,
        'unchanged_recipe_target':95,'recipe_score_offset':0.}
    write(out/'experiment-plan.json',plan)
    neighbor_bank=None;neighbor_validation={}
    if args.family=='neighbors':
        neighbor_bank=(structure_features([r['graph'] for r in train])[:,:1024],labels(train))
        features=structure_features([r['graph'] for r in validation])
        for spec in plan['candidates']:
            neighbor_validation[spec['neighbor_k']]=neighbor_probabilities(features,*neighbor_bank,k=spec['neighbor_k'])
    for spec in plan['candidates']:
        neighbor=neighbor_validation.get(spec.get('neighbor_k'))
        oof=np.empty_like(predictions);fold_fits=[]
        for f in range(4):
            fit=fit_correction(predictions[fold!=f],y[fold!=f],**spec,
                               neighbor=neighbor[fold!=f] if neighbor is not None else None)
            oof[fold==f]=apply(predictions[fold==f],fit,neighbor[fold==f] if neighbor is not None else None)
            fold_fits.append(fit)
        result=metrics(y,oof,train_support)
        candidates.append({**spec,'metrics':result,'eligible':eligible(result,baseline),'fold_fits':fold_fits})
        print(json.dumps({'phase':'crossfit',**spec,'log_loss':result['source_annotation_binary_log_loss'],
            'micro_ap':result['micro_average_precision'],'eligible':candidates[-1]['eligible']}),flush=True)
    accepted=[r for r in candidates if r['eligible']]
    if not accepted:
        write(out/'report.json',{'status':'not_selected','validation_baseline':baseline,'candidates':candidates,
            'reason':'no OOF candidate passed fixed nonregression guards','test_opened':False})
        print('No eligible correction; original model remains unchanged.',flush=True)
        return
    selected=min(accepted,key=lambda r:r['metrics']['source_annotation_binary_log_loss'])
    selected_neighbor=neighbor_validation.get(selected.get('neighbor_k'))
    coefficients=fit_correction(predictions,y,label_penalty=selected['label_penalty'],method=selected['method'],
                                neighbor=selected_neighbor,neighbor_k=selected.get('neighbor_k'))
    frozen={'coefficients':coefficients,'selected_penalty':selected['label_penalty'],
            'fit_graphs':[r['graph'] for r in validation],'fit_scaffold_groups':sorted({r['group'] for r in validation}),
            'test_outcomes_seen':False,'parent_sha256':PARENT_SHA,'split_sha256':SPLIT_SHA}
    write(out/'frozen-before-test.json',frozen)
    if neighbor_bank is not None:
        np.savez_compressed(out/'training-bank.npz',fingerprints=neighbor_bank[0].astype(np.uint8),
            targets=neighbor_bank[1].astype(np.uint8),graphs=np.asarray([r['graph'] for r in train]))
    # Only now evaluate the existing regression holdout; no retry/retuning on
    # these results. A rejected candidate stays disconnected from the runtime.
    test_y=labels(test)
    before=parent.predict([r['graph'] for r in test])
    neighbor_test=(neighbor_probabilities(structure_features([r['graph'] for r in test]),*neighbor_bank,
                    k=coefficients['neighbor_k']) if selected_neighbor is not None else None)
    after=apply(before,coefficients,neighbor_test)
    base_test=metrics(test_y,before,train_support);new_test=metrics(test_y,after,train_support)
    passed=eligible(new_test,base_test)
    # Paired losses with a scaffold-cluster bootstrap, not 321300 independent
    # descriptor observations. Fixed 500 replicates, diagnostic interval only.
    losses_before=np.mean((before-test_y)**2,axis=1);losses_after=np.mean((after-test_y)**2,axis=1)
    groups=sorted({r['group'] for r in test});lookup={g:i for i,g in enumerate(groups)}
    gidx=np.array([lookup[r['group']] for r in test]);counts=np.bincount(gidx)
    delta=np.bincount(gidx,weights=losses_after-losses_before)
    rng=np.random.default_rng(620911)
    boot=[]
    for _ in range(500):
        ids=rng.integers(0,len(groups),len(groups));boot.append(float(delta[ids].sum()/counts[ids].sum()))
    report={'status':'accepted_local_research' if passed else 'rejected_holdout_nonregression',
        'validation_baseline':baseline,'crossfit_candidates':candidates,'selected_penalty':selected['label_penalty'],
        'selected_method':selected['method'],'selected_neighbor_k':selected.get('neighbor_k'),
        'regression_holdout':{'before':base_test,'after':new_test,'scaffold_groups':len(groups),
            'brier_delta_cluster_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()},
        'frozen_before_test_sha256':sha(out/'frozen-before-test.json'),'experiment_plan_sha256':sha(out/'experiment-plan.json'),
        'source_labels_are_not_measured_sensory_absence':True,'recipe_score_offset':0.,
        'all_recipe_95_pass_not_claimed':True,'elapsed_seconds':time.perf_counter()-start}
    write(out/'report.json',report)
    np.savez_compressed(out/'holdout.npz',graphs=np.asarray([r['graph'] for r in test]),
                        target=test_y,before=before,after=after)
    if passed:
        artifact={'schema':SCHEMA,'scope':'local_research','parent_model_sha256':PARENT_SHA,
            'endpoints':list(parent.endpoints),'label_kind':LABEL_KIND,'coefficients':coefficients,
            'accepted_for_local_research':True,'test_labels_used_for_fitting':False,
            'recipe_score_offset':0,'recipe_threshold_changed':False,'redistribution_authorized':False,
            'evaluation_summary':{'before':{k:v for k,v in base_test.items() if k!='reliability_bins'},
                'after':{k:v for k,v in new_test.items() if k!='reliability_bins'},
                'brier_delta_cluster_bootstrap_95':report['regression_holdout']['brier_delta_cluster_bootstrap_95'],
                'scope':'public_annotation_regression_holdout_not_human_odor_similarity'},
            'report_sha256':sha(out/'report.json'),'fit_manifest_sha256':sha(out/'frozen-before-test.json')}
        if selected_neighbor is not None:
            artifact['training_bank']={'path':'training-bank.npz','sha256':sha(out/'training-bank.npz'),
                'rows':len(train),'source_split':'V61 train only; validation/test identities and groups excluded',
                'split_sha256':SPLIT_SHA}
        write(out/'calibration.json',artifact)
    print(json.dumps({'status':report['status'],'before':{k:v for k,v in base_test.items() if k!='reliability_bins'},
        'after':{k:v for k,v in new_test.items() if k!='reliability_bins'},
        'calibration_sha256':sha(out/'calibration.json') if passed else None},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
