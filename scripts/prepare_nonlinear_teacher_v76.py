"""Dose/time-dependent mixture teacher using the actual forward primitives.

Labels are explicitly physical-prior calculations, not observed mixtures. All
known active materials participate in their molecule-disjoint source partition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from rdkit import Chem
from fragrance_ai.recommender.models import Ingredient
from fragrance_ai.recommender.science import ScientificPropertyStore, TemporalMixtureSimulator
from fragrance_ai.recommender.physical_evidence_v76 import enrich


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','bank','core','evidence','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=dict(np.load(a.data/'molecules.npz',allow_pickle=False))
    graphs=json.loads((a.data/'molecule-identities.json').read_text())['graphs']
    graph_ids={g:i for i,g in enumerate(graphs)}
    structures=json.loads(a.core.read_text())['structures']
    items=[Ingredient(**r) for r in json.loads((a.bank/'materials.json').read_text(encoding='utf8'))]
    store=ScientificPropertyStore.load_builtin()
    try:
        props=store.with_catalog_structures(items,store.get_many([i.ingredient_id for i in items]))
    finally:
        store.close()
    props=enrich(items,props,(str(a.evidence),sha(a.evidence)))
    mapped={};missing=[]
    for item in items:
        supplied=item.structure_smiles or (structures.get(item.ingredient_id) or [None])[0]
        molecule=Chem.MolFromSmiles(supplied) if supplied else None
        graph=Chem.MolToSmiles(molecule,isomericSmiles=True) if molecule else None
        if graph not in graph_ids:
            missing.append(item.ingredient_id)
        else:
            mapped.setdefault(graph_ids[graph],item)
    rng=np.random.default_rng(760915)
    engine=TemporalMixtureSimulator()
    result={k:[] for k in ('ids','masses','context','quantitative','fine','revision','split')}
    coverage={}
    for split,count in enumerate((6000,1000,1000)):
        pool=np.array([i for i in mapped if m['split'][i]==split])
        coverage[split]={'available':len(pool),'used':set()}
        if count<len(pool) or len(pool)<8:
            raise ValueError('curriculum must cover every eligible identity in each split')
        order=rng.permutation(pool)
        for j in range(count):
            n=int(rng.integers(2,9))
            first=order[j%len(order)]
            others=rng.choice(pool[pool!=first],n-1,replace=False)
            ids=np.r_[first,others]
            coverage[split]['used'].update(ids.tolist())
            selected=[mapped[i] for i in ids]
            w=rng.dirichlet(np.full(n,.8))
            concentration=float(10.**rng.uniform(0,np.log10(30.)))
            minutes=float(rng.choice([0.,15.,60.,240.,480.]))
            lines=[SimpleNamespace(ingredient_id=i.ingredient_id,finished_product_percent=concentration*f,
                                    active_strength_percent=i.active_strength_percent) for i,f in zip(selected,w)]
            prepared=engine._prepare(lines,{i.ingredient_id:i for i in selected},props)
            oav=np.array([v.mole_fraction*v.activity_coefficient*v.vapor_pressure_pa/101325.*1e6/v.odor_threshold_ppm for v in prepared])
            life=np.array([engine._half_life_minutes(v.ingredient,v.properties,v.vapor_pressure_pa) for v in prepared])
            activation=engine._odor_response(oav*.5**(minutes/life))*np.array([engine._air_to_receptor_transport(v.properties) for v in prepared])
            response=activation/(1+.2*(engine._interaction_matrix(prepared)@activation))
            fractions=response/response.sum()
            q=fractions@m['high'][ids]
            fine=fractions@m['fine'][ids]
            coarse=fractions@np.array([i.vector() for i in selected])
            desired=rng.dirichlet(.15+3*coarse)
            context=np.zeros(64,np.float32);context[0]=1.;context[13]=minutes/480.
            context[25:44]=desired;context[44:63]=coarse;context[63]=np.log10(concentration/100)
            padded=np.zeros(8,np.int64);padded[:n]=ids
            mass=np.zeros(8,np.float32);mass[:n]=w
            for key,value in zip(result,(padded,mass,context,q,fine,desired-coarse,split)):
                result[key].append(value)
    np.savez_compressed(a.output/'mixtures.npz',**{k:np.asarray(v) for k,v in result.items()})
    report={'schema':'nonlinear-mixture-teacher/v76','label_kind':'forward_physics_prior_not_new_measurement',
            'linear_mass_average_used':False,'saturation_suppression_evaporation_in_targets':True,
            'product_reference':'hydroalcoholic_perfume_not_lotion_transfer',
            'coverage':{str(k):{'available':v['available'],'used':len(v['used'])} for k,v in coverage.items()},
            'missing_or_unrepresented_identity':missing,'dataset_sha256':sha(a.output/'mixtures.npz'),
            'source_bindings':{str(p):sha(p) for p in (a.data/'manifest.json',a.bank/'manifest.json',a.core,a.evidence,Path(__file__))}}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
