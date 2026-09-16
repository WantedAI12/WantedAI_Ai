"""Prepare full public pair-annotation and measured formulation tasks.

Pairs crossing scaffold partitions are retained as a separate quarantine, not
quietly assigned to training. No unknown blend ratio is made into a measured
50:50 formula. Experimental liquid outcomes retain missingness and source scope.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from rdkit import Chem
from fragrance_ai.recommender.formulation_core import FormulationCore
from scripts.train_odor_expression_v61 import group_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split(key):
    value=int(hashlib.sha256(('observed-formulation-v76:'+key).encode()).hexdigest()[:8],16)%100
    return 0 if value<70 else 1 if value<85 else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--core',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.sources/'manifest.json').read_text())
    for name,record in source['files'].items():
        if sha(a.sources/name)!=record['sha256']:
            raise ValueError('source changed: '+name)
    core=FormulationCore(a.core,sha(a.core))
    pairs=json.loads((a.sources/'odor-pairs.json').read_text())
    labels=sorted({label for row in pairs for label in row['blend_notes']})
    graphs={}
    invalid=[]
    for raw in sorted({row[key] for row in pairs for key in ('mol1','mol2')}):
        molecule=Chem.MolFromInchi(raw) if raw.startswith('InChI=') else Chem.MolFromSmiles(raw)
        if molecule is None or len(Chem.GetMolFrags(molecule))!=1:
            invalid.append(raw)
        else:
            graphs[raw]=Chem.MolToSmiles(molecule,isomericSmiles=True)
    unique=sorted(set(graphs.values()))
    lookup={g:i for i,g in enumerate(unique)}
    features=np.vstack([core.features(unique[i:i+256]) for i in range(0,len(unique),256)])
    groups=[group_key(g) for g in unique]
    assignments=np.array([split(g) for g in groups])
    ids=[];targets=[];splits=[];row_ids=[]
    for j,row in enumerate(pairs):
        if row['mol1'] not in graphs or row['mol2'] not in graphs:
            continue
        i,k=lookup[graphs[row['mol1']]],lookup[graphs[row['mol2']]]
        y=np.zeros(len(labels),np.float32)
        y[[labels.index(x) for x in row['blend_notes']]]=1.
        ids.append([i,k]);targets.append(y);row_ids.append(j)
        splits.append(assignments[i] if assignments[i]==assignments[k] else 3)
    np.savez_compressed(a.output/'odor-pairs.npz',x=features,ids=np.asarray(ids,np.int32),
                        target=np.asarray(targets),split=np.asarray(splits,np.int8),source_rows=np.asarray(row_ids))
    liquid=json.loads((a.sources/'liquid-formulations.json').read_text())
    names=[key for key in liquid[0] if key not in {'Stability_Test','Turbidity_NTU','Turbidity_Error','Viscosity','Rheology_Type','Rheology_Data','ID'}]
    shear=np.geomspace(1.5,700.,16)
    x=np.array([[float(r[k]) for k in names] for r in liquid])
    if len(names)!=18 or np.any(x<0) or not np.isfinite(x).all() or np.any(x.sum(-1)>100):
        raise ValueError('invalid experimental as-supplied percentages')
    group=['|'.join(np.array(names)[row>0]) for row in x]
    sp=np.array([split(g) for g in group],np.int8)
    y=np.zeros((len(x),18));mask=np.zeros_like(y,dtype=bool)
    nonpositive_rheology=[]
    for j,row in enumerate(liquid):
        if type(row['Stability_Test']) is not bool:
            raise ValueError('source stability must be an observed boolean')
        y[j,0]=row['Stability_Test'];mask[j,0]=True
        if isinstance(row['Turbidity_NTU'],(float,int)):
            y[j,1]=np.log1p(row['Turbidity_NTU']);mask[j,1]=True
        flow=row['Rheology_Data']
        if isinstance(flow,list):
            values={k:v for d in flow for k,v in d.items()}
            r,mu=np.asarray(values['shear_rate'],float),np.asarray(values['avg_viscosity'],float)
            if np.any(np.diff(r)<=0) or not np.isfinite(mu).all():
                raise ValueError('invalid experimental rheology')
            nonpositive_rheology.extend({'source_id':row['ID'],'shear_rate':float(rate),'viscosity':float(visc)} for rate,visc in zip(r,mu) if visc<=0)
            for k,rate in enumerate(shear):
                right=int(np.searchsorted(r,rate));left=right-1
                if left<0 or right>=len(r) or mu[left]<=0 or mu[right]<=0:
                    continue
                mask[j,k+2]=True
                y[j,k+2]=np.interp(np.log(rate),np.log(r[left:right+1]),np.log(mu[left:right+1]))
    mean=np.zeros(18);scale=np.ones(18)
    for j in range(1,18):
        data=y[(sp==0)&mask[:,j],j]
        if len(data)<3:
            raise ValueError('insufficient train-only outcome normalization support')
        mean[j]=data.mean();scale[j]=max(data.std(),.1)
    np.savez_compressed(a.output/'liquid.npz',x=x,target=y,mask=mask,split=sp,
                        normalized_target=(y-mean)/scale,source_ids=np.array([r['ID'] for r in liquid]))
    specification={'ingredient_names':names,'shear_rates_s_inverse':shear.tolist(),
        'outcome_mean':mean.tolist(),'outcome_scale':scale.tolist(),
        'viscosity_unit':'mPa*s','composition_unit':'as_supplied_finished_mass_percent',
        'reference_protocol':'chitre_2024_25c_360rpm_35min_ph5.8_36h',
        'source_doi':'10.1038/s41597-024-03573-w','license':'CC0',
        'raw_min':x[sp==0].min(0).tolist(),'raw_max':x[sp==0].max(0).tolist(),
        'training_ingredient_combinations':sorted(set(np.array(group)[sp==0])),
        'missing_regression_targets_are_zero':False,'stability_is_manufacturing_approval':False}
    specification['nonpositive_rheology_preserved_but_not_log_fitted']=nonpositive_rheology
    report={'schema':'observed-formulation-data/v76','core_sha256':core.sha256,
        'source_manifest_sha256':sha(a.sources/'manifest.json'),
        'odor_pairs':{'labels':labels,'graphs':unique,'scaffold_groups':groups,'graph_splits':assignments.tolist(),
                     'source_rows':len(pairs),'prepared_rows':len(ids),'invalid_graphs':invalid,
                     'split_counts':np.bincount(splits,minlength=4).tolist(),
                     'ratios_observed':False,'scope':'pair_annotation_recovery_not_measured_dose_response',
                     'unlisted_notes_are_verified_absent':False},
        'liquid':{**specification,'source_rows':len(liquid),'split_counts':np.bincount(sp,minlength=3).tolist(),
                  'stable_rows':int(y[:,0].sum()),'groups':group},
        'files':{name:sha(a.output/name) for name in ('odor-pairs.npz','liquid.npz')}}
    (a.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pair_splits':report['odor_pairs']['split_counts'],'liquid_splits':report['liquid']['split_counts'],
                      'pair_labels':len(labels),'observed_liquid_rows':len(liquid)}))


if __name__=='__main__':
    main()
