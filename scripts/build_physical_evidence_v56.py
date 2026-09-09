"""Index existing local public thresholds and supplied properties by identity.

No network, inferred measurements, database overwrites, or new sensory labels.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from rdkit import Chem


def build(source, database):
    paths = {name: source/name for name in ('behavior.csv', 'molecules.csv')}
    rows = {}
    for name, path in paths.items():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows[name] = list(csv.DictReader(stream))
    behaviors = {r['Stimulus']: r for r in rows['behavior.csv']}
    if len(behaviors) != len(rows['behavior.csv']):
        raise ValueError('duplicate source stimulus identity')
    graph_records, ambiguous = {}, set()
    for molecule in rows['molecules.csv']:
        cid = molecule['CID']
        if cid not in behaviors:
            continue
        raw = molecule['IsomericSMILES']
        mol = Chem.MolFromSmiles(raw)
        if mol is None or '.' in raw:
            continue
        graph = Chem.MolToSmiles(mol, isomericSmiles=True)
        value = 10**(-float(behaviors[cid]['Log (1/ODT)']))
        if not math.isfinite(value) or value <= 0:
            raise ValueError('invalid published gas threshold')
        record = {'odor_threshold_ppm': value, 'pubchem_cid': cid,
            'source_ref': 'doi:10.1093/chemse/bjr094;pyrfume:abraham_2012;CID:'+cid,
            'evidence_class': 'published_gas_detection_threshold_not_formula_similarity',
            'unit': 'ppmv', 'identity_basis': 'exact_isomeric_graph',
            'molecular_weight': float(molecule['MolecularWeight'])}
        if graph in graph_records and graph_records[graph]['odor_threshold_ppm'] != value:
            ambiguous.add(graph)
        graph_records[graph] = record
    for graph in ambiguous:
        graph_records.pop(graph, None)
    cas_records, ambiguous_cas = {}, set()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM molecular_properties'):
            item = dict(row)
            cas = item['cas_number']
            if not cas or item['source_ref'].startswith('composition-derived'):
                continue
            values = {k: item[k] for k in ('molecular_weight','vapor_pressure_pa_25c','boiling_point_c','odor_threshold_ppm')
                if item[k] is not None}
            values.update(source_ref=item['source_ref'], verified_on=item['verified_on'],
                          source_ingredient_id=item['ingredient_id'], identity_basis='exact_CAS_and_molecular_weight')
            if cas in cas_records and any(cas_records[cas].get(k) != v for k,v in values.items() if k not in ('source_ref','source_ingredient_id')):
                ambiguous_cas.add(cas)
            cas_records[cas] = values
    for cas in ambiguous_cas:
        cas_records.pop(cas, None)
    return {'schema': 'physical-evidence-index/v1', 'by_structure': graph_records, 'by_cas': cas_records,
        'source_files': {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                         for name,path in {**paths, 'scientific_properties.db': database}.items()},
        'identity_policy': 'exact_isomeric_graph_or_unambiguous_CAS_with_mass_check;no_name_fuzzy_matching',
        'ambiguous_graphs_excluded': len(ambiguous), 'ambiguous_cas_excluded': len(ambiguous_cas),
        'unit': 'gas_ppmv_at_source_conditions', 'human_recipe_validation': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new evidence output; never overwrite a source snapshot')
    data = build(args.source, args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'graph_thresholds': len(data['by_structure']), 'cas_properties': len(data['by_cas']),
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'network_calls': 0}))


if __name__ == '__main__':
    main()
