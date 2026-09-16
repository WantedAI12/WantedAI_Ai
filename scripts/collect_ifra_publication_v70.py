"""Archive whole IFRA 51 publications linked by the official documentation page."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import httpx

URLS = {
    'IFRA_51_OVERVIEW': 'https://d3t14p1xronwr0.cloudfront.net/docs/Standards-Documentation/ifra-51st-amendment-ifra-standards-overview.xlsx',
    'IFRA_51_OTHER_SOURCE_CONTRIBUTIONS': 'https://d3t14p1xronwr0.cloudfront.net/docs/Standards-Documentation/ifra-51st-amendment-annex-on-contributions-from-other-sources.xlsx',
    'IFRA_51_FULL_STANDARDS': 'https://d3t14p1xronwr0.cloudfront.net/docs/Standards-Documentation/ifra-standards-51st-amendment.pdf',
    'IFRA_51_GUIDANCE': 'https://d3t14p1xronwr0.cloudfront.net/docs/Standards-Documentation/ifra-51st-amendment-guidance-for-the-use-of-ifra-standards.pdf',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    documents, failures = [], []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for label, url in URLS.items():
            try:
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    raw = bytearray()
                    for part in response.iter_bytes():
                        raw.extend(part)
                        if len(raw) > 32 * 1024 * 1024:
                            raise ValueError('source publication larger than 32 MiB')
                    extension = Path(url).suffix
                    if not bytes(raw).startswith(b'%PDF-' if extension == '.pdf' else b'PK\x03\x04'):
                        raise ValueError('source content does not match PDF/XLSX format')
                    digest = hashlib.sha256(raw).hexdigest()
                    path = digest + extension
                    (out / path).write_bytes(raw)
                    documents.append({'id': label, 'url': str(response.url), 'path': path, 'sha256': digest,
                        'bytes': len(raw), 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                        'source_kind': 'official_IFRA_51_publication_not_supplier_certificate',
                        'source_listing_url': 'https://ifrafragrance.org/initiatives-positions/safe-use-fragrance-science/ifra-standards/ifra-standards-documentation',
                        'scope': 'published_51st_amendment_no_claim_of_52nd_final_rules'})
                    print(json.dumps(documents[-1]), flush=True)
            except (httpx.HTTPError, ValueError) as error:
                failures.append({'id': label, 'url': url, 'error': str(error)[:250]})
    (out / 'manifest.json').write_text(json.dumps({'sources': documents, 'failures': failures}, indent=2), encoding='utf-8')
    print(json.dumps({'documents': len(documents), 'failures': failures}))


if __name__ == '__main__':
    main()
