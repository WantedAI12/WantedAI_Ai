"""Archive public primary-source documents and explicit facts, not approvals.

Public stock indicators are never converted into kilograms or promised lead
times. Supplier specimen certificates remain distinct from batch documents.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ('3RM00358', '3JF00052', '3RL00204', '3MC00233', '3WY00244', '3LL00278')
GENERAL = {
    'IFRA': 'https://ifrafragrance.org/standards-library',
    'IFRA_AMENDMENTS': 'https://ifrafragrance.org/initiatives-positions/safe-use-fragrance-science/ifra-standards',
    'EU_REACH': 'https://echa.europa.eu/candidate-list-package',
    'EU_REACH_COMMISSION': 'https://environment.ec.europa.eu/topics/chemicals/reach-regulation_en',
    'K_REACH': 'https://www.law.go.kr/lsInfoP.do?ancYnChk=0&lsId=011857',
    'FDA': 'https://www.fda.gov/cosmetics/cosmetic-ingredients/fragrances-cosmetics',
}
CAS = re.compile(r'(?<!\d)\d{2,7}-\d{2}-\d(?!\d)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'documents').mkdir()
    documents, failures, materials, standards = [], [], [], []
    source = ROOT/'tmp/modal-runtime-v69/release-02/dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest = json.loads(source.read_text(encoding='utf-8'))
    raw = (source.parent/manifest['runtime_catalog']['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['runtime_catalog']['sha256']
    catalog = json.loads(gzip.decompress(raw))['ingredients']
    by_cas = {}
    for item in catalog:
        if item.get('cas_number'):
            by_cas.setdefault(item['cas_number'], []).append(item['ingredient_id'])
    with httpx.Client(timeout=30, follow_redirects=True, headers={'User-Agent': 'PerfumeryResearchEvidenceCollector/1.0'}) as client:
        def fetch(label, url):
            time.sleep(.3)
            try:
                response = client.get(url)
                response.raise_for_status()
                if len(response.content) > 8*1024*1024 or 'html' not in response.headers.get('content-type', ''):
                    raise ValueError('unexpected source format or size')
                soup = BeautifulSoup(response.content, 'html.parser')
                if any(text in soup.get_text(' ', strip=True).casefold()[:300] for text in ('access denied', 'just a moment', 'verify you are human')):
                    raise ValueError('source returned an access challenge, not evidence')
                digest = hashlib.sha256(response.content).hexdigest()
                filename = 'documents/'+digest+'.html'
                (output/filename).write_bytes(response.content)
                item = {'id': label, 'url': str(response.url), 'path': filename, 'sha256': digest,
                        'retrieved_at': datetime.now(timezone.utc).isoformat(), 'bytes': len(response.content),
                        'source_kind': 'supplier_public_document' if 'perfumersworld.com' in url else 'official_public_reference'}
                documents.append(item)
                print(json.dumps({'source': label, 'bytes': len(response.content), 'status': 'archived'}), flush=True)
                return soup, item
            except (httpx.HTTPError, ValueError) as error:
                failures.append({'id': label, 'url': url, 'reason': type(error).__name__, 'message': str(error)[:180]})
                print(json.dumps(failures[-1]), flush=True)
                return None, None

        for sku in PRODUCTS:
            product, product_doc = fetch(sku, 'https://www.perfumersworld.com/view.php?pro_id='+sku)
            docs, docs_doc = fetch(sku+'_documents', 'https://www.perfumersworld.com/document-list.php?pro_id='+sku)
            if product is None:
                continue
            text = product.get_text(' ', strip=True)
            identity = re.search(r'CAS\s*No\.?\s*[:|]?\s*('+CAS.pattern+r')', text, re.I)
            price = re.search(r'US\$\s*(\d+(?:\.\d+)?)\s*/\s*gram', text, re.I)
            name = product.find('h1')
            if not name or not identity or sku not in text or not price:
                failures.append({'id': sku, 'reason': 'missing_or_ambiguous_product_identity_or_price'})
                continue
            cas = identity.group(1)
            limits = []
            if docs is not None:
                for row in docs.select('tr'):
                    cells = [cell.get_text(' ', strip=True) for cell in row.find_all(['td', 'th'])]
                    if len(cells) == 3 and re.fullmatch(r'\d+(?:\.\d+)?', cells[-1]):
                        category = ('eau_de_parfum' if 'Fine Fragrance unshaved skin' in cells[0] else
                                    'body_lotion' if 'Body Creams/Oil/Lotion' in cells[0] else
                                    'body_wash' if 'Rinse-off toiletries' in cells[0] else None)
                        if category:
                            limits.append({'product_category': category, 'ifra_category': cells[1],
                                           'maximum_finished_product_percent': float(cells[-1]),
                                           'source_id': docs_doc['id'], 'scope': 'supplier_specimen_not_batch_certificate'})
            materials.append({'cas_number': cas, 'ingredient_ids': by_cas.get(cas, []),
                'supplier': 'PerfumersWorld', 'sku': sku, 'name': name.get_text(' ', strip=True),
                'source_id': product_doc['id'], 'document_source_id': docs_doc['id'] if docs_doc else None,
                'currency': 'USD', 'unit_price_usd_per_gram': float(price.group(1)),
                'price_per_kg': 1000.*float(price.group(1)), 'price_kind': 'public_unit_price_not_binding_quote',
                'stock_indicator': 'in_stock' if 'IN STOCK' in text else 'unknown',
                'available_kg': None, 'lead_time_days': None, 'minimum_order_kg': None, 'quote_valid_until': None,
                'supplier_ifra_limits': limits, 'business_registration_verified': False,
                'frameworks_verified': [], 'reviewer': None})
        for framework, url in GENERAL.items():
            soup, document = fetch(framework, url)
            if soup is None:
                continue
            if framework == 'IFRA':
                # Read only the rows actually present. Pagination not downloaded
                # is reported as partial; absence cannot establish unrestricted use.
                for row in soup.select('tr'):
                    cells = [cell.get_text(' ', strip=True) for cell in row.find_all('td')]
                    href = row.find('a', href=True)
                    if len(cells) >= 6 and CAS.search(' '.join(cells)):
                        standards.append({'cas_numbers': CAS.findall(' '.join(cells)), 'cells': cells,
                                          'document_url': urljoin(url, href['href']) if href else None,
                                          'source_id': document['id'], 'dose_limits_parsed': False})
    history = []
    if args.previous:
        previous = json.loads(args.previous.read_text(encoding='utf-8'))
        if previous['schema_version'] != 'public-regulatory-supply-1':
            raise ValueError('wrong previous observation schema')
        history = [*previous.get('previous_snapshots', []), {key: previous[key] for key in ('version', 'created_at', 'documents', 'materials')}][-12:]
        for snapshot in history:
            for document in snapshot['documents']:
                path = (args.previous.parent/document['path']).resolve()
                if not path.is_relative_to(args.previous.parent.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != document['sha256']:
                    raise ValueError('previous document unavailable or changed')
                shutil.copy2(path, output/document['path'])
    payload = {'schema_version': 'public-regulatory-supply-1', 'version': 'public-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'),
               'created_at': datetime.now(timezone.utc).isoformat(), 'documents': documents,
               'materials': materials, 'ifra_index_rows': standards, 'download_failures': failures,
               'scope': 'public_source_facts_not_operator_reviewed_evidence',
               'previous_snapshots': history,
               'coverage_complete': False, 'manufacturing_approval': False,
               'missing_operator_fields': ['scoped_regulatory_review', 'supplier_business_registration_or_exemption',
                   'binding_quote_validity', 'quantitative_available_stock', 'confirmed_lead_time', 'minimum_order_quantity']}
    path = output/'public_evidence.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'documents': len(documents), 'materials': len(materials), 'ifra_index_rows': len(standards), 'failures': len(failures)}))


if __name__ == '__main__':
    main()
