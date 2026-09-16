"""Index all published IFRA 51 standards and contribution tables, not examples."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {'eau_de_parfum': '4', 'eau_de_toilette': '4', 'eau_de_cologne': '4',
              'body_lotion': '5A', 'face_cream': '5B', 'face_toner': '5B', 'mouthwash': '6',
              'shampoo': '9', 'body_wash': '9', 'candle': '12', 'room_spray': '10B', 'diffuser': '10A'}


def cas_numbers(value):
    return sorted(set(re.findall(r'(?<!\d)\d{2,7}-\d{2}-\d(?!\d)', str(value))))


def limit_value(value, rule_type):
    if value is None:
        return {'kind': 'prohibition_direct_addition', 'maximum_finished_product_percent': 0.} if rule_type == 'PROHIBITION' else {
            'kind': 'not_quantified_requires_rule_notes', 'maximum_finished_product_percent': None}
    text = str(value).strip()
    if text.lower() == 'no restriction':
        return {'kind': 'no_numeric_restriction_in_this_category', 'maximum_finished_product_percent': None}
    if re.fullmatch(r'\d+(?:[.,]\d+)?', text):
        number = float(text.replace(',', '.'))
        if not 0 <= number <= 100:
            raise ValueError('IFRA percentage outside [0,100]')
        return {'kind': 'numeric_finished_product_limit', 'maximum_finished_product_percent': number}
    if text == '0.0 (Prohibited)':
        return {'kind': 'prohibition_direct_addition', 'maximum_finished_product_percent': 0.}
    return {'kind': 'qualified_limit_requires_constituent_or_notebox_review', 'maximum_finished_product_percent': None,
            'raw_value': text}


def pdf_sections(pages):
    sections, current = [], None
    for page in pages:
        text = page['text']
        if 'IFRA STANDARD' not in text:
            continue
        if re.search(r'CAS-No\.?\s*:', text):
            if current:
                sections.append(current)
            header = re.split(r'CAS-No\.?\s*:', text, maxsplit=1)[1].split('Synonyms:')[0].split('History:')[0]
            current = {'pages': [], 'cas_numbers': cas_numbers(header), 'text': ''}
        if current:
            current['pages'].append(page['page'])
            current['text'] += '\n' + text
    if current:
        sections.append(current)
    for section in sections:
        values = defaultdict(set)
        for category, amount in re.findall(r'Category\s+(\d+[A-D]?)\s+([\d.,]+)\s*%', section['text']):
            values[category].add(float(amount.replace(',', '.')))
        section['limits'] = {category: next(iter(amounts)) for category, amounts in values.items() if len(amounts) == 1}
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--identities', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding='utf-8'))['sources']
    docs = {row['id']: row for row in sources}
    for source in sources:
        if hashlib.sha256((args.sources.parent / source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('IFRA source changed')
    pdf = json.loads((args.sources.parent / 'IFRA_51_FULL_STANDARDS_pages.json').read_text(encoding='utf-8'))
    if pdf['source_sha256'] != docs['IFRA_51_FULL_STANDARDS']['sha256']:
        raise ValueError('PDF extraction is not bound to source')
    sections = pdf_sections(pdf['pages'])
    workbook = openpyxl.load_workbook(args.sources.parent / docs['IFRA_51_OVERVIEW']['path'], read_only=True, data_only=True)
    rows = list(workbook.active.values)
    headers = rows[2]
    if headers[0] != 'Key' or headers[7] != 'CAS numbers' or headers[22] != 'Category 4 (%)':
        raise ValueError('IFRA overview column contract changed')
    rules, verification = [], []
    for number, values in enumerate(rows[3:], 4):
        source_fields = dict(zip(headers, values))
        if not re.fullmatch(r'IFRA_STD_\d+', str(values[0])):
            raise ValueError('IFRA standard key missing')
        cas = cas_numbers(values[7])
        matches = [section for section in sections if set(cas) & set(section['cas_numbers'])]
        limits = {}
        for product, category in CATEGORIES.items():
            parsed = limit_value(source_fields['Category ' + category + ' (%)'], values[10])
            candidates = {section['limits'][category] for section in matches if category in section['limits']}
            parsed['pdf_numeric_verified'] = False
            if parsed['kind'] == 'numeric_finished_product_limit' and len(candidates) == 1:
                amount = next(iter(candidates))
                parsed['overview_maximum_percent'] = parsed['maximum_finished_product_percent']
                parsed['overview_matches_pdf'] = abs(parsed['maximum_finished_product_percent'] - amount) < 1e-10
                # The individual PDF standard prevails, exactly as the overview
                # specifies. Any disagreement stays visible for source review.
                parsed.update(maximum_finished_product_percent=amount, pdf_numeric_verified=True)
            limits[product] = {**parsed, 'ifra_category': category,
                'category_assumption': {'shampoo': 'rinse_off_not_dry_shampoo',
                    'room_spray': 'manual_spray_not_automated_metered_device',
                    'diffuser': 'reed_diffuser_with_manual_handling_not_automated_device'}.get(product),
                'category_applicability_independently_verified': False}
        rules.append({'rule_id': values[0], 'name': values[6], 'cas_numbers': cas, 'type': values[10],
            'amendment': values[1], 'source_id': 'IFRA_51_OVERVIEW', 'source_row': number,
            'pdf_source_id': 'IFRA_51_FULL_STANDARDS', 'pdf_pages': sorted({p for m in matches for p in m['pages']}),
            'category_limits': limits, 'source_fields': source_fields})
        verification.append({'rule_id': values[0], 'pdf_sections_found': len(matches),
            'numeric_categories': sum(v['kind'] == 'numeric_finished_product_limit' for v in limits.values()),
            'numeric_categories_verified': sum(v['pdf_numeric_verified'] for v in limits.values()),
            'overview_pdf_conflicts': [k for k, v in limits.items() if v.get('overview_matches_pdf') is False]})
    workbook.close()
    if len(rules) != 263 or len({r['rule_id'] for r in rules}) != len(rules):
        raise ValueError('published whole standard population changed')
    annex = openpyxl.load_workbook(args.sources.parent / docs['IFRA_51_OTHER_SOURCE_CONTRIBUTIONS']['path'], read_only=True, data_only=True)
    contributions, unresolved = [], []
    for sheet, skip, parent_fields, constituent_field, amount_field, kind in (
        ('Natural contributions', 7, (3, 4), 7, 9, 'natural_annex_estimate_not_batch_GCMS'),
        ('Schiff bases', 4, (3,), 1, 4, 'schiff_base_annex_stoichiometric_contribution')):
        for number, values in enumerate(list(annex[sheet].values)[skip:], skip + 1):
            parent = sorted({c for index in parent_fields for c in cas_numbers(values[index])})
            constituents = cas_numbers(values[constituent_field])
            amount = values[amount_field]
            if not parent or not constituents or not isinstance(amount, (int, float)) or not 0 <= amount <= 100:
                unresolved.append({'sheet': sheet, 'row': number, 'parent_cas': parent,
                                   'constituent_cas': constituents, 'source_amount': amount})
                continue
            contributions.append({'parent_cas': parent, 'constituent_cas': constituents,
                'coefficient_percent': float(amount), 'kind': kind, 'source_sheet': sheet, 'source_row': number})
    annex.close()
    m = ROOT / 'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest = json.loads(m.read_text())
    raw = (m.parent / manifest['runtime_catalog']['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['runtime_catalog']['sha256']:
        raise ValueError('catalog changed')
    active = [r for r in json.loads(gzip.decompress(raw))['ingredients'] if r['formulation_ready'] and not r['blocked']]
    identities = {r['ingredient_id']: r for r in json.loads(args.identities.read_text(encoding='utf-8'))['materials']}
    by_cas = defaultdict(list)
    for rule in rules:
        for cas in rule['cas_numbers']:
            by_cas[cas].append(rule)
    coverage, observations = [], []
    for item in sorted(active, key=lambda r: r['ingredient_id']):
        identity = identities[item['ingredient_id']]
        aliases = {item['cas_number']} if item['cas_number'] else {c for match in identity['matches'] for c in match['cas_aliases']}
        direct = {rule['rule_id']: rule for cas in aliases for rule in by_cas.get(cas, [])}
        connected = {}
        for contribution in contributions:
            if aliases & set(contribution['parent_cas']):
                for cas in contribution['constituent_cas']:
                    for rule in by_cas.get(cas, []):
                        connected.setdefault(rule['rule_id'], []).append(contribution)
        all_rules = sorted(set(direct) | set(connected))
        coverage.append({'ingredient_id': item['ingredient_id'], 'status': 'published_IFRA_rule_connected' if all_rules else
            'no_exact_rule_hit_group_and_constituent_review_unresolved', 'direct_rule_ids': sorted(direct),
            'contribution_rule_ids': sorted(connected), 'full_conformity_verified': False})
        for key in all_rules:
            rule = next(rule for rule in rules if rule['rule_id'] == key)
            extra = connected.get(key, [])
            coefficient = 100. if key in direct else max(row['coefficient_percent'] for row in extra)
            observations.append({'ingredient_id': item['ingredient_id'], 'cas_number': item['cas_number'],
                'source_id': rule['source_id'], 'rule_id': key, 'rule_name': rule['name'], 'rule_type': rule['type'],
                'category_limits': rule['category_limits'], 'source_fields': rule['source_fields'],
                'pdf_pages': rule['pdf_pages'], 'pdf_source_id': rule['pdf_source_id'],
                'contribution_coefficient_percent': coefficient, 'contribution_records': extra,
                'material_active_strength_percent': item['active_strength_percent'],
                'contribution_source_id': 'IFRA_51_OTHER_SOURCE_CONTRIBUTIONS' if extra else None,
                'contribution_kind': 'direct_added_material' if key in direct else 'annex_maximum_of_matching_source_estimates_not_batch_assay',
                'status': 'published_rule_screen_requires_specification_scope_and_batch_review'})
    index = {'schema_version': 'public-regulatory-index-1', 'framework': 'IFRA', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_ids': list(docs), 'source_row_count': len(rules), 'source_scope': 'whole_published_IFRA_51_overview_and_contributions_with_PDF_crosscheck',
        'active_material_count': len(active), 'matched_material_count': sum(bool(r['direct_rule_ids'] or r['contribution_rule_ids']) for r in coverage),
        'coverage': coverage, 'observations': observations, 'compliance_verified': False, 'manufacturing_approval': False,
        'published_rule_count': len(rules), 'rule_type_counts': dict(Counter(r['type'] for r in rules)),
        'contribution_row_count': len(contributions), 'unresolved_contribution_rows': unresolved,
        'pdf_verification': verification, 'unresolved_scope_rules': ['unmatched_group_CAS_and_names', 'batch_purity_peroxide_and_impurity_specifications',
            'actual_natural_composition_and_grade', 'qualified_or_notebox_limits', 'new_amendment_implementation_scope', 'supplier_mixture_conformity'],
        'group_salt_hydrate_rules_exhaustively_resolved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('preserve prior index')
    args.output.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'rules': len(rules), 'PDF_sections': len(sections), 'all_active': len(active), 'connected': index['matched_material_count'],
        'numeric_categories': sum(r['numeric_categories'] for r in verification),
        'numeric_categories_verified': sum(r['numeric_categories_verified'] for r in verification),
        'overview_pdf_conflicts': [r for r in verification if r['overview_pdf_conflicts']],
        'contributions': len(contributions), 'unresolved_contribution_rows': len(unresolved)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
