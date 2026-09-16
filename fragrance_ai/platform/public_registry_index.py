"""Validated whole-catalog public indexes, separate from operator review records."""
from copy import deepcopy
import hashlib
import math
import re

FRAMEWORKS = ('IFRA', 'EU_REACH', 'K_REACH', 'FDA')


def valid_cas(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{2,7}-\d{2}-\d', value):
        return False
    digits = value.replace('-', '')
    return sum(int(char) * (i + 1) for i, char in enumerate(reversed(digits[:-1]))) % 10 == int(digits[-1])


class PublicRegistryIndex:
    def __init__(self, snapshot, documents):
        self.frameworks = {}
        self.identities = {}
        for row in snapshot.get('material_identities', []):
            identifier = row['ingredient_id']
            if identifier in self.identities or row.get('canonical_cas_number_changed') is not False:
                raise ValueError('duplicate or mutating public material identity')
            if row['structure_sha256'] != hashlib.sha256(row['catalog_structure_smiles'].encode()).hexdigest():
                raise ValueError('unbound public structure identity')
            for match in row['matches']:
                if not match['source_ids'] or any(source not in documents for source in match['source_ids']):
                    raise ValueError('unbound chemical identity publication')
                if any(not valid_cas(cas) for cas in match['cas_aliases']):
                    raise ValueError('invalid public CAS alias')
            self.identities[identifier] = row
        for index in snapshot.get('regulatory_indexes', []):
            framework = index['framework']
            if (framework not in FRAMEWORKS or framework in self.frameworks
                    or index.get('compliance_verified') is not False
                    or index.get('manufacturing_approval') is not False):
                raise ValueError('public index cannot certify compliance')
            if not index['source_ids'] or any(source not in documents for source in index['source_ids']):
                raise ValueError('unbound public regulatory index')
            coverage = {row['ingredient_id']: row for row in index['coverage']}
            if len(coverage) != len(index['coverage']) or len(coverage) != index['active_material_count']:
                raise ValueError('public index has duplicate or missing denominator rows')
            observations = {}
            ifra_keys = set()
            for row in index['observations']:
                if row['ingredient_id'] not in coverage or row['source_id'] not in documents:
                    raise ValueError('regulatory observation outside indexed population or source')
                observations.setdefault(row['ingredient_id'], []).append(row)
                if framework == 'IFRA':
                    key = (row['ingredient_id'], row['rule_id'])
                    if key in ifra_keys:
                        raise ValueError('duplicate IFRA material-standard contribution')
                    ifra_keys.add(key)
                    for key in ('contribution_coefficient_percent', 'material_active_strength_percent'):
                        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or not 0 <= row[key] <= 100:
                            raise ValueError('invalid IFRA material concentration factor')
                    if row.get('pdf_source_id') not in documents:
                        raise ValueError('IFRA limit is not bound to its PDF source')
                    if row.get('contribution_source_id') and row['contribution_source_id'] not in documents:
                        raise ValueError('IFRA contribution is not bound to its source')
            if index['matched_material_count'] != len(observations):
                raise ValueError('public regulatory match denominator is inconsistent')
            self.frameworks[framework] = (index, coverage, observations)

    def contract(self):
        return {'frameworks': {name: {'source_row_count': index['source_row_count'],
                    'active_material_count': index['active_material_count'],
                    'matched_material_count': index['matched_material_count'],
                    'source_scope': index['source_scope'], 'compliance_verified': False}
                for name, (index, _, _) in self.frameworks.items()},
                'material_identity_count': len(self.identities),
                'structure_corroborated_count': sum(bool(row['matches']) for row in self.identities.values()),
                'canonical_material_data_modified': False}

    def coverage_for(self, item):
        identity = self.identities.get(item.ingredient_id)
        if identity and (identity['catalog_cas_number'] != item.cas_number
                         or identity['catalog_structure_smiles'] != item.structure_smiles):
            raise ValueError('public identity belongs to a different catalog material')
        return {'identity_status': identity['status'] if identity else 'no_public_identity_corroboration',
                'cas_aliases': sorted({cas for match in identity['matches'] for cas in match['cas_aliases']}) if identity else [],
                'cas_aliases_require_grade_confirmation': True,
                'regulatory_lists': {name: deepcopy(coverage.get(item.ingredient_id,
                    {'ingredient_id': item.ingredient_id, 'status': 'not_in_indexed_population'}))
                    for name, (_, coverage, _) in self.frameworks.items()}}

    def findings(self, framework, lines, documents, *, category=None, concentration=None):
        item = self.frameworks.get(framework)
        if item is None:
            return None
        index, coverage, observations = item
        findings, material_coverage = [], []
        for line in lines:
            identifier = line['ingredient_id']
            identity = self.identities.get(identifier)
            # Serialized recipes include their current CAS. A reused material
            # identifier cannot silently inherit a different molecule's facts.
            if identity and 'cas_number' in line and line['cas_number'] != identity['catalog_cas_number']:
                raise ValueError('recipe identity differs from public indexed identity')
            material_coverage.append(deepcopy(coverage.get(identifier,
                {'ingredient_id': identifier, 'status': 'not_in_indexed_population'})))
            if float(line.get('concentrate_percent', 0)) <= 0:
                continue
            for row in observations.get(identifier, []):
                document = documents[row['source_id']]
                findings.append({**deepcopy(row), 'source_url': document['url'],
                                 'source_sha256': document['sha256'], 'compliance_verified': False})
        calculated = self._ifra_totals(findings, lines, category, concentration) if framework == 'IFRA' else None
        return {'status': 'published_limit_exceeded_review_required' if calculated and calculated['source_limit_exceeded'] else
                    'listed_material_review_required' if findings else 'source_list_screened_not_cleared',
                'findings': findings, 'material_coverage': material_coverage,
                'source_scope': index['source_scope'], 'source_row_count': index['source_row_count'],
                'unresolved_scope_rules': deepcopy(index.get('unresolved_scope_rules', [])),
                'group_salt_hydrate_rules_exhaustively_resolved': index.get('group_salt_hydrate_rules_exhaustively_resolved', False),
                'compliance_verified': False, 'business_registration_verified': False,
                **({'formula_rule_checks': calculated} if calculated is not None else {})}

    @staticmethod
    def _ifra_totals(findings, lines, category, concentration):
        if not isinstance(concentration, (int, float)) or not math.isfinite(concentration) or not 0 < concentration <= 100:
            raise ValueError('IFRA screening requires finite finished-product concentration')
        weights = {row['ingredient_id']: float(row['concentrate_percent']) for row in lines}
        if len(weights) != len(lines) or any(not math.isfinite(v) or not 0 <= v <= 100 for v in weights.values()):
            raise ValueError('IFRA screening requires unique finite formula lines')
        groups = {}
        for row in findings:
            key = row['rule_id']
            limit = row['category_limits'].get(category)
            if limit is None:
                limit = {'kind': 'product_category_not_mapped', 'maximum_finished_product_percent': None}
            group = groups.setdefault(key, {'rule_id': key, 'rule_name': row['rule_name'],
                'limit': deepcopy(limit), 'finished_product_percent': 0., 'contributions': [],
                'contains_annex_estimates': False, 'source_url': row['source_url'],
                'source_sha256': row['source_sha256'], 'pdf_pages': row['pdf_pages'],
                'batch_composition_verified': False})
            if group['limit'] != limit:
                raise ValueError('conflicting IFRA limits for the same standard')
            contribution = (weights[row['ingredient_id']] * concentration / 100.
                * row['material_active_strength_percent'] / 100.
                * row['contribution_coefficient_percent'] / 100.)
            group['finished_product_percent'] += contribution
            group['contains_annex_estimates'] |= row['contribution_kind'] != 'direct_added_material'
            group['contributions'].append({'ingredient_id': row['ingredient_id'],
                'finished_product_percent': contribution, 'kind': row['contribution_kind'],
                'active_strength_percent': row['material_active_strength_percent'],
                'constituent_fraction_percent': row['contribution_coefficient_percent']})
        for group in groups.values():
            cap = group['limit']['maximum_finished_product_percent']
            group['source_limit_exceeded'] = cap is not None and group['finished_product_percent'] > cap + 1e-10
            group['status'] = 'source_limit_exceeded_review_required' if group['source_limit_exceeded'] else (
                'within_numeric_source_limit_other_conditions_unverified' if cap is not None else
                'no_numeric_limit_other_conditions_unverified' if group['limit']['kind'] == 'no_numeric_restriction_in_this_category' else 'notebox_or_scope_review_required')
        return {'product_category': category, 'product_concentration_percent': concentration,
                'basis': 'finished_product_mass_percent_sum_of_direct_and_annex_estimated_constituents',
                'checks': list(groups.values()), 'source_limit_exceeded': any(g['source_limit_exceeded'] for g in groups.values()),
                'unknown_composition_is_not_zero': True, 'full_IFRA_conformity_verified': False}
