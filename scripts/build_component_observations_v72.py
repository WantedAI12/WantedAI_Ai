"""Freeze all usable source component observations without recipe outcomes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from fragrance_ai.research.atlas_profiles import load_atlas
    from fragrance_ai.recommender.reference_observations import VERSION,ComponentReferenceObservations
    rows,endpoints,source=load_atlas(ROOT/'.benchmarks/atlas_profiles_v50/source')
    groups=defaultdict(list)
    excluded=[]
    for row in rows:
        if row['level']!='high':
            continue
        if not row['graph']:
            excluded.append({'stimulus_id':row['id'],'reason':'source_molecular_identity_unresolved'})
            continue
        raw=np.stack((row['applicability'],row['use']))
        if np.any(raw.sum(-1)<=0):
            excluded.append({'stimulus_id':row['id'],'reason':'empty_source_profile'})
            continue
        groups[row['graph']].append((row['id'],raw/raw.sum(-1,keepdims=True)))
    records=[{'canonical_smiles':graph,'profiles':np.stack([v for _,v in entries]).mean(0).tolist(),
        'source_stimulus_ids':[key for key,_ in entries],
        'basis':'measured_component_ordinal_reference_not_measured_product_mixture'} for graph,entries in sorted(groups.items())]
    value={'schema':VERSION,'source_ordinal_level':'high','endpoints':endpoints,'records':records,
        'source':source,'excluded':excluded,'recipe_outcomes_used':False,'product_sensory_calibration':False,
        'source_overlap_with_target_reference':True,
        'scope':'known_component_reference_lookup_not_unseen_molecule_or_new_blind_validation',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'observations.json'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    store=ComponentReferenceObservations(path,digest)
    report={'observed_component_identities':len(store.entries),'source_stimuli':sum(len(row['source_stimulus_ids']) for row in records),
        'excluded_high_stimuli':len(excluded),'sha256':digest,'recipe_outcomes_used':False,
        'new_blind_validation_claimed':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    main()
