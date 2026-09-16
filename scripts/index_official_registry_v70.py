"""Read the complete official Korean XLSX and bind exact CAS matches to all active materials.

This is read-only spreadsheet analysis; raw workbooks are never rewritten.
The resulting JSON preserves source wording and does not infer registration,
exemption, product approval, or unrestricted use from an inventory match.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ('NO', 'CAS번호', '영문명', '국문명', '기존', '급성·만성·생태', '사고대비',
           '제한/금지/허가', '중점', '잔류', '유해특성분류 및 혼합물 함량기준(%)',
           '등록대상기존화학물질', '기존물질여부')
FIELDS = ('source_row', 'cas_number', 'english_name', 'korean_name', 'existing_inventory_id',
          'hazard_designation', 'accident_preparedness_designation',
          'restriction_prohibition_authorisation_designation', 'priority_designation',
          'persistent_pollutant_designation', 'classification_and_concentration_source_text',
          'registration_target_designation', 'existing_inventory_indicator')


def valid_cas(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{2,7}-\d{2}-\d', value):
        return False
    digits = value.replace('-', '')
    return sum(int(c) * (i + 1) for i, c in enumerate(reversed(digits[:-1]))) % 10 == int(digits[-1])


def read_registry(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if len(workbook.worksheets) != 1:
        raise ValueError('unexpected official export sheet count')
    sheet = workbook.active
    declared_dimensions = sheet.calculate_dimension()
    # Apache POI's export incorrectly declares A1 as the used range. Trust the
    # streamed rows, not this metadata, or 47,520 public entries disappear.
    sheet.reset_dimensions()
    iterator = sheet.iter_rows(values_only=True)
    if tuple(next(iterator)) != HEADERS:
        raise ValueError('official registry columns changed')
    rows = []
    try:
        for offset, values in enumerate(iterator, 1):
            if len(values) != len(FIELDS) or str(values[0]) != str(offset):
                raise ValueError('registry row sequence or width changed')
            row = dict(zip(FIELDS, (str(v).strip() if v is not None else '' for v in values)))
            row['source_row'] = offset + 1  # one-based workbook row including header
            row['exact_cas_available'] = valid_cas(row['cas_number'])
            rows.append(row)
    finally:
        workbook.close()
    return rows, {'sheet': sheet.title, 'declared_dimensions': declared_dimensions,
                  'actual_data_rows': len(rows), 'all_rows_streamed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identities', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.sources.read_text(encoding='utf-8'))
    source = next(row for row in manifest['sources'] if row['id'] == 'K_REACH_PUBLIC_EXPORT')
    path = args.sources.parent / source['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('official export changed before parsing')
    rows, read_receipt = read_registry(path)
    index = next(row for row in manifest['sources'] if row['id'] == 'K_REACH_PUBLIC_INDEX')
    raw_index = (args.sources.parent / index['path']).read_bytes()
    if hashlib.sha256(raw_index).hexdigest() != index['sha256']:
        raise ValueError('official index changed')
    total = re.search(r'총\s*<[^>]+>\s*([\d,]+)', raw_index.decode('utf-8'))
    if total is None:
        # Keep the exact visible total, allowing intervening span/strong tags.
        text = re.sub('<[^>]+>', ' ', raw_index.decode('utf-8'))
        total = re.search(r'총\s+([\d,]+)\s*건', text)
    if total is None or int(total.group(1).replace(',', '')) != len(rows):
        raise ValueError('workbook population does not match the official index total')
    m = ROOT / 'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    catalog_manifest = json.loads(m.read_text(encoding='utf-8'))
    data = (m.parent / catalog_manifest['runtime_catalog']['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != catalog_manifest['runtime_catalog']['sha256']:
        raise ValueError('active catalog changed')
    active = [item for item in json.loads(gzip.decompress(data))['ingredients']
              if item['formulation_ready'] and not item['blocked']]
    by_cas = {}
    for row in rows:
        if row['exact_cas_available']:
            by_cas.setdefault(row['cas_number'], []).append(row)
    identities = {}
    if args.identities:
        identity_bundle = json.loads(args.identities.read_text(encoding='utf-8'))
        identities = {row['ingredient_id']: row for row in identity_bundle['materials']}
    coverage, observations = [], []
    for item in sorted(active, key=lambda item: item['ingredient_id']):
        cas = item.get('cas_number')
        matches = by_cas.get(cas, []) if valid_cas(cas) else []
        match_mode = 'exact_catalog_cas'
        identity = identities.get(item['ingredient_id'])
        if identity and (identity['catalog_cas_number'] != cas or identity['catalog_structure_smiles'] != item['structure_smiles']):
            raise ValueError('identity index belongs to a different material snapshot')
        if not cas and identity:
            aliases = sorted({alias for match in identity['matches'] for alias in match['cas_aliases']})
            matches = [row for alias in aliases for row in by_cas.get(alias, [])]
            match_mode = 'structure_corroborated_cas_alias_requires_grade_confirmation'
        coverage.append({'ingredient_id': item['ingredient_id'], 'cas_number': cas,
                         'status': ('exact_cas_inventory_match' if match_mode == 'exact_catalog_cas' else 'structure_corroborated_alias_inventory_match') if matches else
                            'no_exact_cas_inventory_match' if valid_cas(cas) else 'material_cas_unavailable_or_invalid',
                         'identity_match_mode': match_mode,
                         'matched_source_rows': [r['source_row'] for r in matches]})
        for row in matches:
            observations.append({'ingredient_id': item['ingredient_id'], 'cas_number': row['cas_number'],
                                 'catalog_cas_number': cas, 'identity_match_mode': match_mode,
                                 'source_id': source['id'], 'source_fields': row,
                                 'status': 'listed_regulatory_designation_requires_use_review' if any(row[k] for k in FIELDS[5:10]) else
                                    'inventory_identity_only_not_safety_or_registration_approval'})
    result = {'schema_version': 'public-regulatory-index-1', 'framework': 'K_REACH',
              'generated_at': datetime.now(timezone.utc).isoformat(),
              'source_ids': [index['id'], source['id']], 'source_sha256': source['sha256'],
              'source_row_count': len(rows), 'source_scope': 'complete_public_chemical_inventory_export',
              'active_material_count': len(active), 'matched_material_count': sum(bool(r['matched_source_rows']) for r in coverage),
              'exact_catalog_cas_matched_material_count': sum(r['status'] == 'exact_cas_inventory_match' for r in coverage),
              'structure_alias_matched_material_count': sum(r['status'] == 'structure_corroborated_alias_inventory_match' for r in coverage),
              'no_exact_cas_row_count_in_source': sum(not r['exact_cas_available'] for r in rows),
              'duplicate_exact_cas_source_rows': sum(n - 1 for n in Counter(r['cas_number'] for r in rows if r['exact_cas_available']).values()),
              'spreadsheet_read_receipt': read_receipt,
              'coverage': coverage, 'observations': observations,
              'group_salt_hydrate_rules_exhaustively_resolved': False,
              'business_registration_verified': False, 'compliance_verified': False,
              'manufacturing_approval': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('do not overwrite an existing source index')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('coverage', 'observations')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
