"""Join the complete public supplier catalog against EVERY active material.

No hand-picked product list. CAS identity is required; diluted supplier stocks
are kept separately, not silently treated as the same neat ingredient.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CATALOG_URL = 'https://www.perfumersworld.com/product-search.php'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'documents').mkdir()
    old = json.loads(args.previous.read_text(encoding='utf-8'))
    history = [*old.get('previous_snapshots', []), {key: old[key] for key in ('version','created_at','documents','materials')}][-12:]
    for snapshot in history:
        for item in snapshot['documents']:
            source = (args.previous.parent/item['path']).resolve()
            if not source.is_relative_to(args.previous.parent.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('untrusted previous document')
            shutil.copy2(source, out/item['path'])
    documents = {row['id']: row for row in old['documents'] if row['source_kind'] != 'supplier_public_document'}
    failures = list(old.get('download_failures', []))
    m = ROOT/'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest = json.loads(m.read_text(encoding='utf-8'))
    data = (m.parent/manifest['runtime_catalog']['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['runtime_catalog']['sha256']:
        raise ValueError('material snapshot changed')
    materials = json.loads(gzip.decompress(data))['ingredients']
    active = [row for row in materials if row['formulation_ready'] and not row['blocked']]
    by_cas = {}
    for row in active:
        if row.get('cas_number'):
            by_cas.setdefault(row['cas_number'], []).append(row['ingredient_id'])

    def archive(label, url, content):
        digest = hashlib.sha256(content).hexdigest()
        path = 'documents/'+digest+'.html'
        (out/path).write_bytes(content)
        return {'id': label, 'url': url, 'path': path, 'sha256': digest, 'bytes': len(content),
                'retrieved_at': datetime.now(timezone.utc).isoformat(), 'source_kind': 'supplier_public_document'}

    with httpx.Client(timeout=45, follow_redirects=True) as client:
        response = client.get(CATALOG_URL)
        response.raise_for_status()
    if len(response.content) > 16*1024*1024:
        raise ValueError('public catalog too large')
    documents['PWA_FULL_CATALOG'] = archive('PWA_FULL_CATALOG', str(response.url), response.content)
    soup = BeautifulSoup(response.content, 'html.parser')
    entries, selected = [], []
    for card in soup.select('.material-card[data-sku]'):
        sku, name, cas = card.get('data-sku','').strip(), card.get('data-name','').strip(), card.get('data-cas-no','').strip()
        price = card.select_one('form.product-form[data-unit-price]')
        if not re.fullmatch('[A-Za-z0-9]{6,16}', sku) or price is None:
            continue
        amount = float(price['data-unit-price'])
        dilution = bool(re.search(r'\d\s*%|\bin\s+(?:dpg|ipm|tec|ethanol)\b', name, re.I))
        identifiers = by_cas.get(cas, []) if not dilution else []
        entry = {'sku': sku, 'name': name, 'cas_number': cas or None, 'unit_price_usd_per_gram': amount,
                 'is_explicit_dilution': dilution, 'active_ingredient_ids': identifiers,
                 'mapping_status': 'exact_cas_requires_grade_confirmation' if identifiers else
                    'diluted_stock_not_neat_material' if dilution else 'not_matched_to_active_catalog'}
        entries.append(entry)
        if not identifiers or amount <= 0:
            continue
        stats = {}
        for row in card.select('.stat-row'):
            label, value = row.select_one('.stat-label'), row.select_one('.stat-value')
            if label and value:
                stats[label.get_text(' ',strip=True).rstrip(':')] = value.get_text(' ',strip=True)
        limit = stats.get('IFRA #4')
        limits = []
        if limit and re.fullmatch(r'\d+(?:\.\d+)?', limit) and float(limit) <= 100:
            limits.append({'product_category':'eau_de_parfum', 'ifra_category':'4',
                'maximum_finished_product_percent':float(limit), 'source_id':'PWA_FULL_CATALOG',
                'scope':'supplier_public_listing_not_full_rule_pack_or_batch_certificate'})
        selected.append({'cas_number':cas, 'ingredient_ids':identifiers, 'supplier':'PerfumersWorld',
            'sku':sku, 'name':name, 'source_id':'PWA_FULL_CATALOG', 'document_source_id':None,
            'currency':'USD', 'unit_price_usd_per_gram':amount, 'price_per_kg':1000*amount,
            'price_kind':'public_unit_price_not_binding_quote_excludes_bottling_shipping_and_tax',
            'stock_indicator':'public_orderable_listing_not_quantified_stock',
            'available_kg':None, 'lead_time_days':None, 'minimum_order_kg':None, 'quote_valid_until':None,
            'supplier_ifra_limits':limits, 'business_registration_verified':False, 'frameworks_verified':[],
            'reviewer':None, 'identity_scope':'exact_cas_only_grade_purity_and_batch_unverified'})

    def retrieve(item):
        time.sleep(.4)
        url = 'https://www.perfumersworld.com/document-list.php?pro_id='+item['sku']
        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
            if len(response.content) > 8*1024*1024:
                raise ValueError('document too large')
            page = BeautifulSoup(response.content, 'html.parser')
            text = page.get_text(' ', strip=True)
            if item['sku'] not in text or item['cas_number'] not in text:
                raise ValueError('supplier document does not confirm requested SKU and CAS')
            document = archive(item['sku']+'_documents', str(response.url), response.content)
            limits = []
            for row in page.select('tr'):
                cells = [cell.get_text(' ',strip=True) for cell in row.find_all(['td','th'])]
                if len(cells) != 3 or not re.fullmatch(r'\d+(?:\.\d+)?', cells[-1]) or float(cells[-1]) > 100:
                    continue
                product = 'eau_de_parfum' if 'Fine Fragrance unshaved skin' in cells[0] else 'body_lotion' if 'Body Creams/Oil/Lotion' in cells[0] else 'body_wash' if 'Rinse-off toiletries' in cells[0] else None
                if product:
                    limits.append({'product_category':product, 'ifra_category':cells[1],
                        'maximum_finished_product_percent':float(cells[-1]), 'source_id':document['id'],
                        'scope':'supplier_specimen_not_batch_certificate'})
            return document, limits, None
        except (httpx.HTTPError, ValueError) as error:
            return None, [], {'id':item['sku'], 'url':url, 'reason':type(error).__name__, 'message':str(error)[:180]}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(retrieve,item):item for item in selected}
        for completed, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            document, limits, failure = future.result()
            if failure:
                failures.append(failure)
            else:
                documents[document['id']] = document
                item['document_source_id'] = document['id']
                # Preserve distinct listing and document observations, including
                # contradictions; neither is silently promoted to approval.
                item['supplier_ifra_limits'].extend(limits)
            (out/'checkpoint.json').write_text(json.dumps({'completed_documents':completed,
                'total_matching_skus':len(selected),'download_failures':len(failures)}),encoding='utf-8')
            if completed%20 == 0 or completed == len(selected):
                print(json.dumps({'completed_documents':completed,'total_matching_skus':len(selected)}),flush=True)
    value = {**old, 'version':'public-all-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'),
             'created_at':datetime.now(timezone.utc).isoformat(), 'documents':list(documents.values()),
             'materials':selected, 'previous_snapshots':history, 'download_failures':failures,
             'supplier_catalog_products_examined':len(entries), 'active_materials_examined':len(active),
             'selection_scope':'entire_active_catalog_exact_CAS_join_not_example_product_selection'}
    path = out/'public_evidence.json'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'supplier_catalog_entries.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
    by_id = {}
    for item in selected:
        for identifier in item['ingredient_ids']:
            by_id.setdefault(identifier,[]).append(item)
    coverage = []
    for item in active:
        matched = by_id.get(item['ingredient_id'],[])
        coverage.append({'ingredient_id':item['ingredient_id'], 'name':item['name'], 'cas_number':item.get('cas_number'),
            'supplier_skus':[row['sku'] for row in matched], 'public_price_connected':bool(matched),
            'supplier_ifra_products':sorted({limit['product_category'] for row in matched for limit in row['supplier_ifra_limits']}),
            'frameworks':{'IFRA':'supplier_source_screen' if matched else 'no_scoped_source',
                'EU_REACH':'reference_only_requires_business_and_substance_evidence',
                'K_REACH':'reference_only_requires_importer_tonnage_and_registration',
                'FDA':'reference_only_requires_product_safety_and_labeling_review'},
            'quantitative_stock_connected':False,'confirmed_lead_time_connected':False,
            'operational_gate_complete':False,
            'missing_reason':None if matched else 'no_exact_neat_CAS_offer_in_complete_supplier_catalog'})
    (out/'all_active_coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
    report = {'source_products':len(entries), 'active_examined':len(active), 'mapped_skus':len(selected),
              'active_with_public_prices':len(by_id), 'active_without_public_prices':len(active)-len(by_id),
              'documents':len(documents), 'failures':len(failures), 'path':str(path),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'operator_evidence_complete':False}
    (out/'coverage_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
