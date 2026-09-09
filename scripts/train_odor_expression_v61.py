"""Structure-only multi-label learning; scaffold-disjoint annotation evaluation.

Zeros are unrecorded annotations, not confirmed absence of a human sensation.
The source-label retrieval used during recipe research is OFF in every metric.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score
from fragrance_ai.recommender.odor_expression import registry
from fragrance_ai.recommender.fine_odor_model import VERSION, structure_features, FineOdorModel
from fragrance_ai.research.fine_odor_features import validate_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def group_key(graph):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(graph)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('fine head requires one molecular graph')
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms():
        return 'murcko:'+Chem.MolToSmiles(scaffold,isomericSmiles=False)
    # Empty Murcko scaffolds do not become a single enormous acyclic group.
    # Generic full topology groups acyclic chemistry/stereoisomer variants.
    generic = MurckoScaffold.MakeScaffoldGeneric(mol)
    return 'acyclic_topology:'+Chem.MolToSmiles(generic,isomericSmiles=False)


def metrics(y,p,train_support):
    support = y.sum(axis=0)
    eligible = (support>=3)&(train_support>=3)&(support<len(y))
    ap = [float(average_precision_score(y[:,i],p[:,i])) for i in np.flatnonzero(eligible)]
    order = np.argsort(-p,axis=1,kind='stable')[:,:5]
    found = np.take_along_axis(y,order,axis=1).sum(axis=1)
    return {'macro_average_precision':float(np.mean(ap)) if ap else None,
        'macro_ap_eligible_labels':int(eligible.sum()),
        'micro_average_precision':float(average_precision_score(y.ravel(),p.ravel())),
        'precision_at_5':float(found.mean()/5),
        'recall_at_5':float(np.mean(found/np.maximum(1,y.sum(axis=1)))),
        'unobserved_test_labels':int((support==0).sum()),
        'source_annotation_binary_log_loss':float(-np.mean(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1)))),
        'rows':len(y), 'retrieval_enabled':False, 'human_similarity_percent':None}


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=160)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError('use a new output directory; retain existing experiments')
    out.mkdir(parents=True)
    start = time.perf_counter()
    r = registry()
    parent = ROOT/'.benchmarks/quantitative_profiles_v54/run-01/model.json'
    if sha(parent)!=r['payload']['source_annotation_parent_sha256']:
        raise ValueError('annotation parent changed')
    fine = json.loads(parent.read_text(encoding='utf-8'))['fine_features']
    validate_features(fine)
    mapping = r['payload']['source_term_to_concept']
    endpoints = sorted(set(mapping.values()))
    index = {s:i for i,s in enumerate(endpoints)}
    all_graphs = sorted(fine['by_structure'])
    groups_by_graph, rejections = {}, []
    for graph in all_graphs:
        try:
            groups_by_graph[graph] = group_key(graph)
        except ValueError as error:
            rejections.append({'graph':graph,'reason':str(error)})
    graphs = sorted(groups_by_graph)
    write(out/'source-coverage.json',{'source_structures':len(all_graphs),
        'single_molecular_graphs':len(graphs),'rejected_records':rejections,
        'no_disconnected_mixture_was_silently_treated_as_a_molecule':True})
    annotations = {g: sorted({index[mapping[fine['vocabulary'][i]]] for i in fine['by_structure'][g]}) for g in graphs}
    x = structure_features(graphs)
    groups = [groups_by_graph[g] for g in graphs]
    folds = {g:int(hashlib.sha256(('610909'+g).encode()).hexdigest()[:8],16)%100 for g in set(groups)}
    split = np.asarray([0 if folds[g]<70 else 1 if folds[g]<85 else 2 for g in groups])
    y = np.zeros((len(graphs),len(endpoints)),np.float32)
    for i,g in enumerate(graphs):
        y[i,annotations[g]]=1.
    for a,b in ((0,1),(0,2),(1,2)):
        if {g for g,s in zip(groups,split) if s==a}&{g for g,s in zip(groups,split) if s==b}:
            raise ValueError('group leakage')
    write(out/'split.json', {'seed':610909,'definition':'Murcko scaffold; acyclic generic full topology; stereo variants grouped',
        'records':[{'graph':g,'group':k,'split':int(s)} for g,k,s in zip(graphs,groups,split)],
        'counts':dict(Counter(str(s) for s in split)), 'structure_count':len(graphs),
        'annotation_source_sha256':sha(parent),'source_feature_payload_sha256':fine['content_sha256']})
    train,valid,test = (split==s for s in (0,1,2))
    if min(train.sum(),valid.sum(),test.sum())<100:
        raise ValueError('insufficient group-disjoint split')
    mean = x[train,1024:].mean(axis=0)
    scale = np.maximum(x[train,1024:].std(axis=0),1e-5)
    z = x.copy(); z[:,1024:]=np.clip((z[:,1024:]-mean)/scale,-8,8)
    prior = (y[train].sum(axis=0)+.5)/(train.sum()+1)
    train_support = y[train].sum(axis=0)
    torch.manual_seed(610909)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model = torch.nn.Sequential(torch.nn.Linear(1040,256),torch.nn.ReLU(),torch.nn.Dropout(.15),
        torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,len(endpoints))).to(device)
    with torch.no_grad():
        model[-1].bias.copy_(torch.tensor(np.log(prior/(1-prior)),device=device))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    tx = torch.tensor(z[train],device=device); ty = torch.tensor(y[train],device=device)
    vx = torch.tensor(z[valid],device=device)
    weights = torch.tensor(np.clip(1/np.sqrt(np.maximum(prior,.01)),1.,8.),device=device)
    best_score,best_epoch,best = -1.,0,None
    history = []
    print(json.dumps({'phase':'training','device':device,'labels':len(endpoints),'shapes':x.shape,
                      'splits':dict(Counter(int(s) for s in split))}),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        order = torch.randperm(len(tx),device=device)
        loss_total=0.
        for ids in order.split(256):
            optimizer.zero_grad()
            logits = model(tx[ids])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits,ty[ids],reduction='none')*weights).mean()
            loss.backward(); optimizer.step()
            loss_total += float(loss.detach())*len(ids)
        if epoch%5==0 or epoch==1:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(vx)).cpu().numpy()
            check = metrics(y[valid],vp,train_support)
            score = check['macro_average_precision']
            history.append({'epoch':epoch,'loss':loss_total/len(tx),**check})
            if score>best_score:
                best_score,best_epoch,best=score,epoch,copy.deepcopy(model.state_dict())
            if epoch%40==0:
                print(json.dumps({'epoch':epoch,'validation_macro_ap':score,'best_epoch':best_epoch}),flush=True)
    model.load_state_dict(best); model.eval()
    arrays = {'mean':mean,'scale':scale,'prior':prior}
    for j,layer in enumerate((model[0],model[3],model[6])):
        arrays['w'+str(j)]=layer.weight.detach().cpu().numpy().T.copy()
        arrays['b'+str(j)]=layer.bias.detach().cpu().numpy().copy()
    np.savez_compressed(out/'weights.npz',**arrays)
    with torch.no_grad():
        vp = torch.sigmoid(model(vx)).cpu().numpy()
        tp = torch.sigmoid(model(torch.tensor(z[test],device=device))).cpu().numpy()
    blends = []
    for alpha in (0.,.25,.5,.75,1.):
        blends.append({'alpha':alpha,**metrics(y[valid],alpha*vp+(1-alpha)*prior,train_support)})
    chosen = max(blends,key=lambda a:(a['macro_average_precision'],a['micro_average_precision']))['alpha']
    prediction = chosen*tp+(1-chosen)*prior
    baseline = metrics(y[test],np.broadcast_to(prior,y[test].shape),train_support)
    learned = metrics(y[test],prediction,train_support)
    evaluation = {'scope':'scaffold_disjoint_public_annotation_recovery_not_new_odor_measurement',
        'baseline':baseline,'learned':learned,'selected_epoch':best_epoch,'selected_neural_blend':chosen,
        'validation_only_selection':True,'validation_blends':blends,'split_sha256':sha(out/'split.json'),
        'source_annotation_retrieval_disabled_for_metrics':True,
        'label_support':{key:{'train':int(y[train,i].sum()),'validation':int(y[valid,i].sum()),'test':int(y[test,i].sum())}
            for i,key in enumerate(endpoints)},'training_history':history,'elapsed_seconds':time.perf_counter()-start}
    write(out/'evaluation.json',evaluation)
    np.savez_compressed(out/'holdout.npz',target=y[test],prediction=prediction,split=split,test_graphs=np.asarray(graphs)[test])
    manifest = {'schema':VERSION,'endpoints':endpoints,'registry_sha256':r['sha256'],
        'parent_source_sha256':sha(parent),'weights':{'path':'weights.npz','sha256':sha(out/'weights.npz')},
        'label_kind':'public_descriptor_annotations_not_measured_absence_or_intensity',
        'annotation_inputs_used':False,'neural_blend':chosen,'source_annotations':annotations,
        'evaluation_summary':{'baseline':baseline,'learned':learned,'split_sha256':sha(out/'split.json')},
        'evaluation_sha256':sha(out/'evaluation.json'),'runtime_scope':'local_research',
        'redistribution_authorized':False,'training_parameter_count':sum(p.numel() for p in model.parameters())}
    write(out/'model.json',manifest)
    runtime = FineOdorModel(out/'model.json',sha(out/'model.json'))
    numpy_prediction = runtime.predict_features(x[test])
    maximum = float(np.max(np.abs(numpy_prediction-prediction)))
    if maximum>1e-5:
        raise ValueError('CPU exported model differs from trained checkpoint')
    write(out/'export-check.json',{'cpu_pytorch_max_abs_difference':maximum,'manifest_sha256':sha(out/'model.json'),
        'weights_bytes':(out/'weights.npz').stat().st_size})
    print(json.dumps({'status':'trained_and_cpu_verified','baseline':baseline,'learned':learned,
        'model_sha256':sha(out/'model.json'),'export_max_abs_difference':maximum}),flush=True)


if __name__=='__main__':
    main()
