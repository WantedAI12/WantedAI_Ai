"""Bulk local evidence joins; an unmatched value stays missing, not measured."""
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent/'data/physical_evidence_v56.json'
FIELDS = ('vapor_pressure_pa_25c', 'boiling_point_c', 'odor_threshold_ppm')


@lru_cache(maxsize=4)
def _read_index(path, size, mtime):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get('schema') != 'physical-evidence-index/v1' or value.get('human_recipe_validation') is not False:
        raise ValueError('invalid physical evidence index')
    for table in ('by_structure', 'by_cas'):
        if not isinstance(value.get(table), dict):
            raise ValueError('invalid physical evidence identities')
        for record in value[table].values():
            if not isinstance(record, dict) or not record.get('source_ref') or record.get('molecular_weight') is None:
                raise ValueError('physical evidence source required')
            if table == 'by_structure' and (record.get('unit') != 'ppmv' or record.get('identity_basis') != 'exact_isomeric_graph'):
                raise ValueError('gas ppmv and exact isomeric graph evidence required')
            if table == 'by_cas' and record.get('identity_basis') != 'exact_CAS_and_molecular_weight':
                raise ValueError('exact CAS evidence required')
            for name in ('molecular_weight', *FIELDS):
                number = record.get(name)
                if number is not None and (isinstance(number, bool) or not isinstance(number, (float,int)) or not math.isfinite(number)
                                          or (name != 'boiling_point_c' and number <= 0)
                                          or (name == 'boiling_point_c' and number <= -273.15)):
                    raise ValueError('invalid physical evidence value')
    value['_sha256'] = hashlib.sha256(raw).hexdigest()
    return value


def evidence_contract(index_path=INDEX):
    from .physical_evidence_v76 import configured_pair, load
    pair = configured_pair()
    if pair:
        selected,digest=pair
        stat=Path(selected).stat()
        index=load(str(selected),digest,stat.st_size,stat.st_mtime_ns)
        return {'available':True,'index_sha256':digest,'index_schema':index['schema'],
                'exact_structure_threshold_records':index['counts']['odor_threshold_ppm'],
                'exact_cas_property_records':len(index.get('legacy_by_cas',{})),
                'source_identity_counts':dict(index['counts']),
                'counts_scope':'qualified_source_identities_not_active_catalog_coverage',
                'join_policy':index['identity_policy'],
                'joined_values_are_not_formula_sensory_validation':True}
    path = Path(index_path)
    if not path.is_file():
        return {'available': False, 'joined_values_are_not_formula_sensory_validation': True}
    stat = path.stat()
    index = _read_index(str(path), stat.st_size, stat.st_mtime_ns)
    return {'available': True, 'index_sha256': index['_sha256'],
            'exact_structure_threshold_records': len(index['by_structure']),
            'exact_cas_property_records': len(index['by_cas']),
            'join_policy': index.get('identity_policy'),
            'joined_values_are_not_formula_sensory_validation': True}


@lru_cache(maxsize=32768)
def _graph(smiles):
    if not smiles or '.' in smiles:
        return None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def enrich_properties(ingredients, properties, *, index_path=INDEX):
    ingredients = tuple(ingredients)
    result = dict(properties)
    path = Path(index_path)
    if not path.is_file():
        from .physical_evidence_v76 import configured_pair, enrich
        pair = configured_pair()
        return enrich(ingredients, result, pair) if pair else result
    stat = path.stat()
    index = _read_index(str(path), stat.st_size, stat.st_mtime_ns)
    for item in ingredients:
        prior = result.get(item.ingredient_id)
        if prior is None or abs(item.active_strength_percent-100.) > 1e-8:
            continue
        records = [index['by_structure'].get(_graph(item.structure_smiles)), index['by_cas'].get(item.cas_number)]
        additions, sources = {}, []
        for record in records:
            if record is None or abs(record['molecular_weight']-prior.molecular_weight) > max(.1, .005*prior.molecular_weight):
                continue
            used = False
            for name in FIELDS:
                if getattr(prior, name) is None and name not in additions and record.get(name) is not None:
                    additions[name] = record[name]
                    used = True
            if used:
                sources.append(record['source_ref'])
        if additions:
            result[item.ingredient_id] = replace(prior, **additions,
                source_ref=prior.source_ref+';identity-joined-evidence:'+';'.join(sources))
    from .physical_evidence_v76 import configured_pair, enrich
    pair = configured_pair()
    return enrich(ingredients, result, pair) if pair else result
