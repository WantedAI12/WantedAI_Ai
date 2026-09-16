"""Hash-pinned public document observations, never operator approval records."""
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re


class PublicEvidenceStore:
    def __init__(self, path, digest):
        self.path = Path(path).resolve()
        if not re.fullmatch('[0-9a-f]{64}', digest or ''):
            raise ValueError('public evidence requires a trusted SHA256')
        self.digest = digest
        raw = self.path.read_bytes()
        if len(raw) > 16*1024*1024 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('public evidence hash or size mismatch')
        self.value = json.loads(raw)
        if (self.value.get('schema_version') != 'public-regulatory-supply-1'
                or self.value.get('scope') != 'public_source_facts_not_operator_reviewed_evidence'
                or self.value.get('manufacturing_approval') is not False):
            raise ValueError('public observations cannot claim operator approval')
        self.documents = {row['id']: row for row in self.value['documents']}
        if len(self.documents) != len(self.value['documents']):
            raise ValueError('duplicate source identifier')
        self.members = [(self.path, digest)]
        history = self.value.get('previous_snapshots', [])
        if len(history) > 12 or len({row['version'] for row in history}) != len(history) or any(row['version'] == self.value['version'] for row in history):
            raise ValueError('invalid public observation history')
        historical_documents = [document for snapshot in history for document in snapshot['documents']]
        for row in [*self.documents.values(), *historical_documents]:
            member = (self.path.parent/row['path']).resolve()
            if not member.is_relative_to(self.path.parent) or member == self.path:
                raise ValueError('public document escapes pinned directory')
            stamp = datetime.fromisoformat(row['retrieved_at'])
            if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
                raise ValueError('source retrieval must be a past timezone-aware time')
            self.members.append((member, row['sha256']))
        members = {}
        for member, checksum in self.members:
            if member in members and members[member] != checksum:
                raise ValueError('conflicting source hashes for the same path')
            members[member] = checksum
        self.members = list(members.items())
        self.by_id = {}
        for item in self.value['materials']:
            if item['source_id'] not in self.documents or item.get('frameworks_verified'):
                raise ValueError('invalid public material provenance or approval claim')
            if not math.isfinite(item['price_per_kg']) or item['price_per_kg'] <= 0:
                raise ValueError('invalid observed price')
            for limit in item.get('supplier_ifra_limits', []):
                if limit['source_id'] not in self.documents or not 0 <= limit['maximum_finished_product_percent'] <= 100:
                    raise ValueError('unbound supplier limit')
            for identifier in item['ingredient_ids']:
                self.by_id.setdefault(identifier, []).append(item)
        from .public_registry_index import PublicRegistryIndex
        self.registry_indexes = {snapshot['version']: PublicRegistryIndex(snapshot,
            {row['id']: row for row in snapshot['documents']}) for snapshot in [self.value, *history]}
        self._metadata = None
        self.assert_current()

    def assert_current(self):
        metadata = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path, _ in self.members)
        if metadata != self._metadata:
            for path, digest in self.members:
                if path.stat().st_size > 16*1024*1024 or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError('public evidence or source document changed')
            self._metadata = metadata

    @classmethod
    def configured(cls):
        path = os.environ.get('PERFUMERY_AI_PUBLIC_EVIDENCE_PATH', '').strip()
        digest = os.environ.get('PERFUMERY_AI_PUBLIC_EVIDENCE_SHA256', '').strip()
        if not path and not digest:
            from ..recommender.local_runtime import local_profile
            profile = local_profile()
            if profile and 'public_evidence' in profile:
                path, digest = profile['public_evidence']
        if bool(path) != bool(digest):
            raise ValueError('public evidence path and digest must be configured together')
        if not path:
            return None
        result = _configured(str(Path(path).resolve()), digest)
        result.assert_current()
        return result

    def contract(self):
        self.assert_current()
        return {'version': self.value['version'], 'sha256': self.digest, 'materials': len(self.by_id),
                'previous_versions': [row['version'] for row in self.value.get('previous_snapshots', [])],
                'documents': len(self.documents), 'scope': self.value['scope'],
                'complete_operational_evidence': False, 'live_inventory_verified': False,
                'missing_operator_fields': list(self.value['missing_operator_fields']),
                'public_regulatory_indexes': self.registry_indexes[self.value['version']].contract(),
                'download_failures': deepcopy(self.value.get('download_failures', []))}

    def coverage(self, catalog):
        """Total-population denominator; unmatched materials are never omitted."""
        self.assert_current()
        active = [item for item in catalog.ingredients if item.formulation_ready and not item.blocked]
        rows = []
        for item in sorted(active, key=lambda item:item.ingredient_id):
            matches = self.by_id.get(item.ingredient_id, [])
            products = sorted({limit['product_category'] for row in matches for limit in row.get('supplier_ifra_limits', [])})
            rows.append({'ingredient_id':item.ingredient_id,'name':item.name,'cas_number':item.cas_number,
                'supplier_skus':[row['sku'] for row in matches], 'public_price_connected':bool(matches),
                'supplier_ifra_products':products, 'quantitative_stock_connected':False,
                'confirmed_lead_time_connected':False, 'operational_evidence_complete':False,
                **self.registry_indexes[self.value['version']].coverage_for(item)})
        return {'active_material_count':len(rows),'public_price_connected_count':sum(row['public_price_connected'] for row in rows),
                'no_public_price_count':sum(not row['public_price_connected'] for row in rows),
                'operationally_complete_count':0,'source_products_examined':self.value.get('supplier_catalog_products_examined'),
                'rows':rows,'scope':'all_active_materials_missing_entries_preserved'}

    def screen(self, lines, *, category, concentration, version=None):
        self.assert_current()
        source = self.value
        if version is not None and version != self.value['version']:
            source = next((row for row in self.value.get('previous_snapshots', []) if row['version'] == version), None)
            if source is None:
                raise ValueError('unknown public observation version')
        documents = {row['id']: row for row in source['documents']}
        by_id = {}
        for item in source['materials']:
            for identifier in item['ingredient_ids']:
                by_id.setdefault(identifier, []).append(item)
        facts, findings = [], []
        for line in lines:
            identifier = line['ingredient_id']
            matches = by_id.get(identifier, [])
            facts.append({'ingredient_id': identifier, 'source_observations': deepcopy(matches),
                          'status': 'source_observed' if matches else 'no_public_material_observation'})
            percent = float(line.get('concentrate_percent', 0))*concentration/100.
            if not math.isfinite(percent) or percent < 0:
                raise ValueError('invalid formula amount for source screening')
            for item in matches:
                for limit in item.get('supplier_ifra_limits', []):
                    if limit['product_category'] == category:
                        document = documents[limit['source_id']]
                        findings.append({'ingredient_id': identifier, 'cas_number': item['cas_number'],
                            'supplier': item['supplier'], 'sku': item['sku'],
                            'finished_product_percent': percent, 'source_maximum_percent': limit['maximum_finished_product_percent'],
                            'source_limit_exceeded': percent > limit['maximum_finished_product_percent']+1e-9,
                            'source_url': document['url'], 'source_sha256': document['sha256'],
                            'scope': 'supplier_specimen_limit_requires_grade_and_batch_confirmation'})
        frameworks = []
        index = self.registry_indexes[source['version']]
        for framework in ('IFRA', 'EU_REACH', 'K_REACH', 'FDA'):
            references = [dict(row) for row in documents.values() if row['id'] == framework or row['id'].startswith(framework+'_')]
            scoped = findings if framework == 'IFRA' else []
            status = 'source_limit_exceeded_review_required' if any(row['source_limit_exceeded'] for row in scoped) else 'partial_source_screen' if scoped else 'reference_available' if references else 'not_assessed'
            registry_check = index.findings(framework, lines, documents, category=category, concentration=concentration)
            if registry_check:
                scoped = [*scoped, *registry_check['findings']]
                if status in ('reference_available', 'not_assessed') or registry_check['status'] == 'published_limit_exceeded_review_required':
                    status = registry_check['status']
            frameworks.append({'id': framework, 'status': status, 'findings': scoped,
                               'source_references': references, 'compliance_verified': False,
                               'business_or_batch_evidence_required': True,
                               **({'public_registry_check': registry_check} if registry_check else {})})
        return {'schema_version': 'public-material-screen-1', 'snapshot_version': source['version'], 'contract': self.contract(),
                'materials': facts, 'frameworks': frameworks,
                'gate_passed': False, 'manufacturing_approval': False,
                'source_screen_only': True, 'live_network_calls': 0}


@lru_cache(maxsize=4)
def _configured(path, digest):
    return PublicEvidenceStore(path, digest)
