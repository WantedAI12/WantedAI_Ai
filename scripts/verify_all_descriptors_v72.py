"""Verify descriptor cache value/error equivalence over every active material."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ['PERFUMERY_AI_LOCAL_PROFILE']='disabled'
    from fragrance_ai.recommender.registry_activation import load_runtime_catalog
    from fragrance_ai.research.conditional_profiles import molecule_features,_molecular_descriptor_tuple
    path=ROOT/'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    binding=manifest['runtime_catalog']
    catalog,_,_=load_runtime_catalog(path.parent/binding['path'],expected_sha256=binding['sha256'],
        expected_wheel_sha256=binding['wheel_sha256'],expected_registry_sha256=binding['registry_sha256'])
    items=[item for item in catalog.ingredients if item.formulation_ready and not item.blocked]
    records=[]
    for item in items:
        graph=item.structure_smiles
        try:
            expected=_molecular_descriptor_tuple.__wrapped__(graph)
            error=None
        except Exception as exc:
            expected,error=None,type(exc).__name__
        try:
            actual=molecule_features(graph)
            assert error is None
            observed=(actual['canonical_smiles'],tuple(actual['fingerprint_bits']),tuple(actual['physical']))
            assert expected==observed
            digest=hashlib.sha256(json.dumps(observed).encode()).hexdigest()
        except Exception as exc:
            if type(exc).__name__!=error:
                raise
            digest=None
        records.append({'ingredient_id':item.ingredient_id,'same':True,'error_kind':error,'descriptor_sha256':digest})
    started=time.perf_counter()
    for item in items:
        try:
            molecule_features(item.structure_smiles)
        except (ValueError,TypeError):
            pass
    warm=time.perf_counter()-started
    report={'all_active_materials':len(items),'all_value_and_error_outcomes_identical':True,
        'valid_structure_count':sum(row['error_kind'] is None for row in records),
        'unchanged_unavailable_count':sum(row['error_kind'] is not None for row in records),
        'warm_seconds':warm,'scope':'descriptor_extraction_only_not_total_API_latency',
        'cache':_molecular_descriptor_tuple.cache_info()._asdict(),'records':records}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':
    main()
