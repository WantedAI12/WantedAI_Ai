"""Whole-active-pool PubChem identity corroboration, not a canonical CAS rewrite.

Existing source record IDs are only lookup candidates. A record is connected
only after exact stereochemical structure equality. CAS synonyms remain aliases
requiring grade confirmation, never permissions to substitute another substance.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import httpx
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(smiles):
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None


def cas_number(value):
    if not re.fullmatch(r'\d{2,7}-\d{2}-\d', value):
        return False
    digits = value.replace('-', '')
    return sum(int(c) * (i + 1) for i, c in enumerate(reversed(digits[:-1]))) % 10 == int(digits[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    (out / 'documents').mkdir(exist_ok=args.resume)
    m = ROOT / 'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest = json.loads(m.read_text(encoding='utf-8'))
    raw = (m.parent / manifest['runtime_catalog']['path']).read_bytes()
    if sha(raw) != manifest['runtime_catalog']['sha256']:
        raise ValueError('catalog snapshot changed')
    active = [r for r in json.loads(gzip.decompress(raw))['ingredients'] if r['formulation_ready'] and not r['blocked']]
    lookups = {}
    for item in active:
        cids = set()
        for ref in item['odor_evidence_refs']:
            data = json.loads(ref)
            token = str(data.get('source_record_id', ''))
            if token.isdecimal() and 0 < int(token) < 10**10:
                cids.add(int(token))
        lookups[item['ingredient_id']] = sorted(cids)
    cids = sorted({cid for values in lookups.values() for cid in values})
    protocol = {'catalog_sha256': manifest['runtime_catalog']['sha256'], 'active_materials': len(active),
                'lookup_cids': cids, 'script_sha256': sha(Path(__file__).read_bytes()),
                'matching': 'exact_isomeric_canonical_smiles_no_desalting_no_stereo_removal'}
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('identity resume protocol changed')
    else:
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    docs = json.loads((out / 'checkpoint.json').read_text()).get('documents', []) if args.resume else []
    by_id = {r['id']: r for r in docs}
    failures = []
    properties, synonyms = {}, {}
    last_call = 0.
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        for offset in range(0, len(cids), 80):
            batch = cids[offset:offset + 80]
            for kind, operation in (('properties', 'property/IsomericSMILES,InChIKey/JSON'), ('synonyms', 'synonyms/JSON')):
                label = f'PUBCHEM_{kind}_{offset // 80:03d}'
                url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + ','.join(map(str, batch)) + '/' + operation
                try:
                    if label in by_id:
                        document = by_id[label]
                        body = (out / document['path']).read_bytes()
                        if sha(body) != document['sha256']:
                            raise ValueError('archived PubChem source changed')
                    else:
                        for attempt in range(3):
                            time.sleep(max(0., .75 - (time.monotonic() - last_call)))
                            last_call = time.monotonic()
                            response = client.get(url)
                            if response.status_code not in (429, 503):
                                break
                            # Respect provider backoff. Never bypass a deny or
                            # create another identity/session to evade throttling.
                            retry = response.headers.get('retry-after', '')
                            if retry.isdigit() and int(retry) > 60:
                                raise ValueError('provider requested long backoff; retry in a later run')
                            time.sleep(max(5. * (attempt + 1), float(retry) if retry.isdigit() else 0.))
                        response.raise_for_status()
                        body = response.content
                        if len(body) > 12 * 1024 * 1024:
                            raise ValueError('PubChem batch exceeds size bound')
                        digest = sha(body)
                        document = {'id': label, 'url': str(response.url), 'sha256': digest,
                                    'path': 'documents/' + digest + '.json', 'bytes': len(body),
                                    'retrieved_at': datetime.now(timezone.utc).isoformat(),
                                    'source_kind': 'public_chemical_identity_not_regulatory_approval'}
                        (out / document['path']).write_bytes(body)
                        by_id[label] = document
                    value = json.loads(body)
                    entries = value['PropertyTable']['Properties'] if kind == 'properties' else value['InformationList']['Information']
                    if any(row['CID'] not in batch for row in entries):
                        raise ValueError('PubChem returned an unexpected CID')
                    dest = properties if kind == 'properties' else synonyms
                    for row in entries:
                        if row['CID'] in dest:
                            raise ValueError('duplicate PubChem CID')
                        dest[row['CID']] = (row, label)
                except (httpx.HTTPError, ValueError, KeyError) as error:
                    failures.append({'id': label, 'lookup_cids': batch, 'error': str(error)[:250]})
            (out / 'checkpoint.json').write_text(json.dumps({'completed_batches': offset // 80 + 1,
                'documents': list(by_id.values()), 'failures': failures}, indent=2), encoding='utf-8')
            print(json.dumps({'completed_lookup_cids': min(offset + 80, len(cids)), 'total_lookup_cids': len(cids),
                              'documents': len(by_id), 'failures': len(failures)}), flush=True)
    RDLogger.DisableLog('rdApp.warning')
    identities = []
    for item in sorted(active, key=lambda r: r['ingredient_id']):
        structure = canonical(item['structure_smiles'])
        matches = []
        for cid in lookups[item['ingredient_id']]:
            if cid not in properties or cid not in synonyms or structure is None:
                continue
            compound, prop_source = properties[cid]
            if canonical(compound.get('SMILES', compound.get('IsomericSMILES'))) != structure:
                continue
            names, synonym_source = synonyms[cid]
            aliases = sorted({value for value in names.get('Synonym', []) if cas_number(value)})
            matches.append({'pubchem_cid': cid, 'isomeric_smiles': compound.get('SMILES', compound.get('IsomericSMILES')),
                            'inchi_key': compound.get('InChIKey'), 'cas_aliases': aliases,
                            'source_ids': [prop_source, synonym_source]})
        identities.append({'ingredient_id': item['ingredient_id'], 'name': item['name'],
            'catalog_cas_number': item.get('cas_number'), 'catalog_structure_smiles': item['structure_smiles'],
            'structure_sha256': sha(item['structure_smiles'].encode()), 'lookup_cids': lookups[item['ingredient_id']],
            'status': 'structure_corroborated_aliases_not_grade_certification' if matches else
                'no_catalog_structure' if structure is None else 'no_exact_structure_corroboration',
            'matches': matches, 'canonical_cas_number_changed': False})
    result = {'schema_version': 'public-material-identities-1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'protocol': protocol, 'documents': list(by_id.values()), 'download_failures': failures,
              'active_materials': len(active), 'status_counts': dict(Counter(r['status'] for r in identities)),
              'materials': identities, 'source_aliases_are_not_batch_identity_or_regulatory_approval': True}
    (out / 'material_identities.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'active_materials': len(active), 'status_counts': result['status_counts'], 'failures': len(failures)}), flush=True)


if __name__ == '__main__':
    main()
