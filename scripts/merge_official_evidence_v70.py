"""Assemble real public sources and whole-population indexes, without promotion."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
FDA_HEADINGS = ('Bithionol', 'Chlorofluorocarbon propellants', 'Chloroform',
    'Halogenated salicylanilides', 'Hexachlorophene', 'Mercury compounds',
    'Methylene chloride', 'Prohibited cattle materials', 'Sunscreens in cosmetics',
    'Vinyl chloride', 'Zirconium-containing complexes')
# CAS identifiers denote named chemicals; they do not encode the rule's use,
# impurity, product category or exemption conditions. Those remain unresolved.
FDA_NAMED_CAS = {'Bithionol': {'97-18-7'}, 'Chloroform': {'67-66-3'},
                 'Hexachlorophene': {'70-30-4'}, 'Methylene chloride': {'75-09-2'},
                 'Vinyl chloride': {'75-01-4'}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fda_matches(material):
    found = set()
    names = {str(name).strip().lower() for name in [material['name'], *material['aliases']]}
    for name, identifiers in FDA_NAMED_CAS.items():
        if material.get('cas_number') in identifiers or name.lower() in names:
            found.add((name, 'named_chemical_match_requires_rule_scope_review'))
    if any(name in names for name in ('dibromsalan', 'tribromsalan', 'metabromsalan', 'tetrachlorosalicylanilide')):
        found.add(('Halogenated salicylanilides', 'named_chemical_match_requires_rule_scope_review'))
    molecule = Chem.MolFromSmiles(material['structure_smiles']) if material['structure_smiles'] else None
    if molecule is not None:
        atoms = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
        if 80 in atoms:
            found.add(('Mercury compounds', 'element_class_match_requires_metal_basis_and_impurity_review'))
        if 40 in atoms:
            found.add(('Zirconium-containing complexes', 'element_class_match_requires_aerosol_and_complex_review'))
        if {6, 9, 17}.issubset(atoms) and atoms.issubset({6, 9, 17}):
            found.add(('Chlorofluorocarbon propellants', 'possible_chemical_class_requires_propellant_use_review'))
    return sorted(found)


def main():
    from bs4 import BeautifulSoup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--official-sources', type=Path, required=True)
    parser.add_argument('--korea-index', type=Path, required=True)
    parser.add_argument('--identities', type=Path, required=True)
    parser.add_argument('--extra-source-manifest', type=Path, action='append', default=[])
    parser.add_argument('--extra-index', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'documents').mkdir()
    previous = json.loads(args.previous.read_text(encoding='utf-8'))
    official = json.loads(args.official_sources.read_text(encoding='utf-8'))
    korea = json.loads(args.korea_index.read_text(encoding='utf-8'))
    identities = json.loads(args.identities.read_text(encoding='utf-8'))
    documents = {}

    def copy_document(document, root):
        source = (root / document['path']).resolve()
        if not source.is_relative_to(root.resolve()) or sha(source) != document['sha256']:
            raise ValueError('raw source changed or escapes its directory')
        target = out / 'documents' / source.name
        if target.exists() and sha(target) != document['sha256']:
            raise ValueError('source filename collision')
        shutil.copy2(source, target)
        return {**document, 'path': 'documents/' + source.name}

    for document in previous['documents']:
        documents[document['id']] = copy_document(document, args.previous.parent)
    for document in official['sources']:
        documents[document['id']] = copy_document(document, args.official_sources.parent)
    for document in identities['documents']:
        documents[document['id']] = copy_document(document, args.identities.parent)
    for source_manifest in args.extra_source_manifest:
        for document in json.loads(source_manifest.read_text(encoding='utf-8'))['sources']:
            documents[document['id']] = copy_document(document, source_manifest.parent)
    history_keys = ('version', 'created_at', 'documents', 'materials', 'regulatory_indexes', 'material_identities')
    history = [*previous.get('previous_snapshots', []), {k: previous[k] for k in history_keys if k in previous}][-12:]
    for snapshot in history:
        snapshot['documents'] = [copy_document(document, args.previous.parent) for document in snapshot['documents']]
    m = ROOT / 'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest = json.loads(m.read_text(encoding='utf-8'))
    raw = (m.parent / manifest['runtime_catalog']['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['runtime_catalog']['sha256']:
        raise ValueError('catalog snapshot changed')
    active = [r for r in json.loads(gzip.decompress(raw))['ingredients'] if r['formulation_ready'] and not r['blocked']]
    if {r['ingredient_id'] for r in active} != {r['ingredient_id'] for r in identities['materials']}:
        raise ValueError('whole-catalog identity denominator changed')
    source = documents['FDA_PROHIBITED_RESTRICTED_COSMETICS']
    soup = BeautifulSoup((out / source['path']).read_bytes(), 'html.parser')
    rules = []
    for name in FDA_HEADINGS:
        matches = [li.get_text(' ', strip=True) for li in soup.select('li')
                   if li.get_text(' ', strip=True).startswith(name + ('.' if name != 'Halogenated salicylanilides' else ' ('))]
        if len(matches) != 1:
            raise ValueError('FDA publication headings changed')
        rules.append({'name': name, 'source_text': matches[0], 'source_id': source['id'],
                      'use_scope_exceptions_automatically_resolved': False})
    coverage, observations = [], []
    for item in sorted(active, key=lambda r: r['ingredient_id']):
        matches = fda_matches(item)
        coverage.append({'ingredient_id': item['ingredient_id'], 'status': 'possible_list_match_review_required' if matches else
                         'no_named_or_element_hit_not_full_safety_clearance', 'matched_rules': [name for name, _ in matches]})
        for name, status in matches:
            observations.append({'ingredient_id': item['ingredient_id'], 'cas_number': item['cas_number'],
                'source_id': source['id'], 'rule_name': name, 'status': status,
                'source_fields': next(rule for rule in rules if rule['name'] == name)})
    fda = {'schema_version': 'public-regulatory-index-1', 'framework': 'FDA',
           'source_ids': [source['id']], 'source_row_count': len(rules), 'source_rules': rules,
           'source_scope': 'complete_FDA_summary_list_not_all_CFR_MoCRA_product_safety_or_label_review',
           'active_material_count': len(active), 'matched_material_count': sum(bool(r['matched_rules']) for r in coverage),
           'coverage': coverage, 'observations': observations, 'manufacturing_approval': False, 'compliance_verified': False,
           'unresolved_scope_rules': ['animal_origin_and_prohibited_cattle_material_traceability',
               'sunscreen_or_therapeutic_label_claims', 'aerosol_propellant_use', 'impurity_and_residual_exceptions',
               'full_named_halogenated_salicylanilide_identity', 'colour_additive_use_and_batch_certification',
               'MoCRA_facility_listing_safety_and_label_obligations']}
    value = {**previous, 'version': 'public-registry-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'),
        'created_at': datetime.now(timezone.utc).isoformat(), 'documents': list(documents.values()),
        'previous_snapshots': history, 'material_identities': identities['materials'],
        'regulatory_indexes': [korea, fda, *(json.loads(path.read_text(encoding='utf-8')) for path in args.extra_index)],
        'download_failures': [*previous.get('download_failures', []), *official['failures']],
        'identity_acquisition_protocol': identities['protocol']}
    path = out / 'public_evidence.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report = {'path': str(path), 'sha256': sha(path), 'documents': len(documents), 'active_materials_examined': len(active),
        'structure_corroborated_materials': sum(bool(r['matches']) for r in identities['materials']),
        'korean_official_source_rows': korea['source_row_count'], 'korean_exact_cas_material_matches': korea['exact_catalog_cas_matched_material_count'],
        'korean_structure_alias_material_matches': korea['structure_alias_matched_material_count'],
        'fda_summary_rules': len(rules), 'fda_possible_material_hits': fda['matched_material_count'],
        'indexed_frameworks': [index['framework'] for index in value['regulatory_indexes']],
        'operating_evidence_complete': False, 'deployed': False}
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
