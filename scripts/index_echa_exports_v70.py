"""Index full official ECHA exports obtained through the authorized public UI."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = (
    ('restriction', 'restriction_list_full-2026-08-24.xlsx', 'https://chem.echa.europa.eu/obligation-lists/restrictionList', 238),
    ('candidate', 'candidate_list_full-2026-08-24.xlsx', 'https://chem.echa.europa.eu/obligation-lists/candidateList', None),
    ('authorisation', 'authorisation_list_full-2025-09-13.xlsx', 'https://chem.echa.europa.eu/obligation-lists/authorisationList', 59),
    ('registrations', 'reach_registrations.xlsx', 'https://chem.echa.europa.eu/api-substance/v1/substance/generated-export?listParticipation=REACH%20REGISTERED', 25370),
    ('harmonised_classification', 'Harmonised_List_2026-09-14 12_43_52.xlsx', 'https://chem.echa.europa.eu/obligation-lists/clhList', 4822),
    ('restriction_scope_members', 'restriction_list_2026-09-14 13_18_30.xlsx', 'https://chem.echa.europa.eu/obligation-lists/restrictionList', 1868),
    ('candidate_scope_members', 'candidate_list_2026-09-14 13_15_45.xlsx', 'https://chem.echa.europa.eu/obligation-lists/candidateList', 507),
    ('authorisation_scope_members', 'authorisation_list_2026-09-14 13_26_14.xlsx', 'https://chem.echa.europa.eu/obligation-lists/authorisationList', 140),
    ('harmonised_scope_members', 'Harmonised_Association_List_2026-09-14 13_12_45.xlsx', 'https://chem.echa.europa.eu/obligation-lists/clhList', 10140),
)


def cas_numbers(value):
    numbers = set(re.findall(r'(?<!\d)\d{2,7}-\d{2}-\d(?!\d)', str(value)))
    def valid(cas):
        digits = cas.replace('-', '')
        return sum(int(c) * (i + 1) for i, c in enumerate(reversed(digits[:-1]))) % 10 == int(digits[-1])
    return sorted(cas for cas in numbers if valid(cas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--identities', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources, entries, datasets = [], [], []
    today = datetime.now(timezone.utc).date()
    for kind, filename, url, expected in EXPORTS:
        path = args.downloads / filename
        raw = path.read_bytes()
        if not raw.startswith(b'PK\x03\x04') or len(raw) > 16 * 1024 * 1024:
            raise ValueError('unexpected ECHA export format or size')
        digest = hashlib.sha256(raw).hexdigest()
        target = out / (digest + '.xlsx')
        shutil.copy2(path, target)
        source = {'id': 'EU_REACH_' + kind.upper(), 'url': url, 'path': target.name, 'sha256': digest,
                  'bytes': len(raw), 'retrieved_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                  'source_kind': 'official_ECHA_public_UI_export',
                  'attribution': 'Source: European Chemicals Agency, https://chem.echa.europa.eu',
                  'terms_accepted_with_explicit_user_authorisation': True,
                  'download_action': 'Download full list' if kind in ('restriction', 'candidate', 'authorisation') else
                     'Export results with Show all substances in scope enabled' if kind.endswith('_scope_members') else
                     'Export results without filters' if kind == 'harmonised_classification' else 'Download all substances: REACH registrations'}
        sources.append(source)
        workbook = openpyxl.load_workbook(target, read_only=True, data_only=True)
        sheet = workbook.active
        sheet.reset_dimensions()
        iterator = sheet.iter_rows(values_only=True)
        headers = tuple(next(iterator))
        first_data_row = 2
        truncated = headers == ('Message',)
        if truncated:
            notice = next(iterator)
            if kind != 'harmonised_scope_members' or 'Only the first 10,000 records' not in str(notice[0]):
                raise ValueError('unexpected ECHA export warning')
            headers = tuple(next(iterator))
            first_data_row = 4
        cas_column = next((h for h in headers if str(h).lower() == 'cas number'), None)
        if not cas_column or not any(str(h).lower() in ('name', 'substance name') for h in headers):
            raise ValueError('ECHA export column contract changed')
        rows = []
        for i, values in enumerate(iterator, first_data_row):
            if len(values) != len(headers):
                raise ValueError('ECHA export has shifted columns')
            fields = dict(zip(headers, values))
            active = True
            if kind == 'harmonised_classification':
                starts = fields['From']
                ends = fields['To']
                active = (not starts or starts == '-' or datetime.strptime(starts, '%d-%b-%Y').date() <= today) and (
                    not ends or ends == '-' or datetime.strptime(ends, '%d-%b-%Y').date() >= today)
            rows.append({'list_kind': kind, 'source_id': source['id'], 'source_row': i,
                         'cas_numbers': cas_numbers(fields[cas_column]), 'source_fields': fields,
                         'classification_applicable_on_snapshot_date': bool(active) if kind == 'harmonised_classification' else None})
        workbook.close()
        if truncated:
            tail_path = ROOT / '.benchmarks/v70_repair/clp_scope_tail_observed.json'
            tail_raw = tail_path.read_bytes()
            tail = json.loads(tail_raw)
            encoded = json.dumps(tail['rows'], separators=(',', ':'))
            checksum = 2166136261
            for char in encoded:
                checksum = ((checksum ^ ord(char)) * 16777619) & 0xffffffff
            if (len(rows) != 10000 or len(tail['rows']) != 140 or checksum != tail['fnv1a32_rows']
                    or len(encoded) != tail['row_json_ascii_characters']
                    or rows[-1]['source_fields'][cas_column] != tail['first_10000_overlap_last_cas']):
                raise ValueError('ECHA export/UI tail overlap or transcription mismatch')
            tail_digest = hashlib.sha256(tail_raw).hexdigest()
            tail_target = out / (tail_digest + '.json')
            shutil.copy2(tail_path, tail_target)
            tail_source = {'id': 'EU_REACH_HARMONISED_SCOPE_TAIL_UI', 'url': tail['source_url'],
                'path': tail_target.name, 'sha256': tail_digest, 'bytes': len(tail_raw),
                'retrieved_at': datetime.fromtimestamp(tail_path.stat().st_mtime, timezone.utc).isoformat(),
                'source_kind': tail['source_kind'], 'scope_toggle': tail['scope_toggle'],
                'attribution': 'Source: European Chemicals Agency, https://chem.echa.europa.eu',
                'scope': 'last_140_identifiers_only_full_names_and_relationships_remain_unparsed'}
            sources.append(tail_source)
            for number, (cas, ec, echa_id, index_number, order) in enumerate(tail['rows'], 10001):
                detail = f'https://chem.echa.europa.eu/{echa_id}/obligations/clhList/details?indexNumber={index_number}&entryOrder={order}'
                rows.append({'list_kind': kind, 'source_id': tail_source['id'], 'source_row': number,
                    'cas_numbers': cas_numbers(cas), 'source_fields': {'CAS number': cas, 'EC number': ec,
                        'ECHA record': echa_id, 'Index number': index_number, 'Entry order': order,
                        'Substance name': None, 'Full source detail URL': detail,
                        'name_and_group_relationship_extraction_status': 'not_copied_use_official_detail'},
                    'classification_applicable_on_snapshot_date': None})
        if expected is not None and len(rows) != expected:
            raise ValueError(f'ECHA {kind} export population mismatch')
        datasets.append({'kind': kind, 'source_id': source['id'], 'source_row_count': len(rows),
            'distinct_entries': len({str(row['source_fields'].get('Entry number')) for row in rows}) if kind == 'restriction' else
                len({tuple(row['source_fields'].get(k) for k in ('Substance name', 'Description', 'EC number', 'CAS number')) for row in rows}) if kind == 'candidate' else len(rows),
            'rows_without_explicit_CAS': sum(not row['cas_numbers'] for row in rows),
            'scope': '10000_export_rows_plus_140_UI_identity_rows_not_full_tail_column_capture' if truncated else 'entire_offered_public_export_including_history_rows'})
        entries.extend(rows)
    m = ROOT / 'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest = json.loads(m.read_text())
    raw = (m.parent / manifest['runtime_catalog']['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['runtime_catalog']['sha256']:
        raise ValueError('catalog changed')
    active = [r for r in json.loads(gzip.decompress(raw))['ingredients'] if r['formulation_ready'] and not r['blocked']]
    identities = {r['ingredient_id']: r for r in json.loads(args.identities.read_text(encoding='utf-8'))['materials']}
    by_cas = {}
    for row in entries:
        for cas in row['cas_numbers']:
            by_cas.setdefault(cas, []).append(row)
    coverage, observations = [], []
    for item in sorted(active, key=lambda r: r['ingredient_id']):
        identity = identities[item['ingredient_id']]
        if identity['catalog_cas_number'] != item['cas_number'] or identity['catalog_structure_smiles'] != item['structure_smiles']:
            raise ValueError('identity snapshot mismatch')
        aliases = [item['cas_number']] if item['cas_number'] else sorted({c for m in identity['matches'] for c in m['cas_aliases']})
        matched = {(r['source_id'], r['source_row']): r for c in aliases for r in by_cas.get(c, [])}
        kinds = sorted({r['list_kind'] for r in matched.values()})
        mode = 'exact_catalog_cas' if item['cas_number'] else 'structure_corroborated_alias_requires_grade_confirmation'
        coverage.append({'ingredient_id': item['ingredient_id'], 'status': 'public_list_match_review_required' if kinds else 'no_explicit_cas_list_hit_not_compliance',
                         'matched_lists': kinds, 'identity_match_mode': mode})
        for row in matched.values():
            observations.append({**row, 'ingredient_id': item['ingredient_id'], 'cas_number': item['cas_number'],
                'identity_match_mode': mode, 'status': 'public_registration_identity_not_operator_registration' if row['list_kind'] == 'registrations' else
                   'public_regulatory_list_match_requires_conditions_review',
                'list_inclusion_is_not_automatic_prohibition': True})
    index = {'schema_version': 'public-regulatory-index-1', 'framework': 'EU_REACH', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_ids': [r['id'] for r in sources], 'source_row_count': len(entries), 'source_datasets': datasets,
        'source_scope': 'ECHA_exports_and_UI_tail_identity_recovery_not_exhaustive_group_or_business_compliance',
        'active_material_count': len(active), 'matched_material_count': sum(bool(r['matched_lists']) for r in coverage),
        'coverage': coverage, 'observations': observations, 'compliance_verified': False, 'manufacturing_approval': False,
        'business_registration_verified': False, 'group_salt_hydrate_rules_exhaustively_resolved': False,
        'unresolved_scope_rules': ['group_entries_without_explicit_CAS', 'all_Annex_XVII_conditions_and_exceptions',
            'CMR_group_classification_and_Annex_appendices', 'hydrated_and_salt_forms',
            'operator_registration_tonnage_uses_and_exemptions', 'latest_Official_Journal_text_overrides_information_exports']}
    (out / 'manifest.json').write_text(json.dumps({'sources': sources, 'failures': []}, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'eu_reach_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'datasets': datasets, 'all_active': len(active), 'matched': index['matched_material_count'],
                      'matched_by_list': dict(Counter(kind for row in coverage for kind in row['matched_lists']))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
