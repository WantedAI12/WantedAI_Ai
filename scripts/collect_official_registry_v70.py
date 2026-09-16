"""Download public regulator publications without changing any approval record.

The Korea export endpoint and form fields are discovered from the public page's
own download button. A response is archived verbatim before any parsing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources, failures = [], []

    def archive(identifier, response, extension, *, request_fields=None):
        raw = response.content
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError('public source exceeds bounded download size')
        digest = hashlib.sha256(raw).hexdigest()
        path = digest + extension
        (out / path).write_bytes(raw)
        row = {'id': identifier, 'url': str(response.url), 'path': path,
               'sha256': digest, 'bytes': len(raw),
               'retrieved_at': datetime.now(timezone.utc).isoformat(),
               'content_type': response.headers.get('content-type'),
               'http_status': response.status_code,
               'source_kind': 'official_publication_not_business_registration'}
        if request_fields is not None:
            row['request_method'] = 'POST'
            row['request_fields'] = request_fields
        sources.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        url = 'https://kreach.mcee.go.kr/repwrt/mttr/kr/mttrList.do'
        try:
            response = client.get(url)
            response.raise_for_status()
            archive('K_REACH_PUBLIC_INDEX', response, '.html')
            soup = BeautifulSoup(response.content, 'html.parser')
            form = soup.select_one('form#searchForm')
            script = '\n'.join(item.text for item in soup.select('script:not([src])'))
            if (form is None or '/repwrt/portalExcel/excelDownload.do' not in script
                    or "portalExcelDownload('mttrList')" not in script):
                raise ValueError('public export contract was not present')
            fields = {item['name']: item.get('value', '') for item in form.select('input[name]')
                      if item.get('type', '') not in ('checkbox', 'radio')}
            fields.update(page='mttrList', searchExcelYn='Y', searchOper='AND',
                          currentPageNo='1', recordCountPerPage='40')
            response = client.post('https://kreach.mcee.go.kr/repwrt/portalExcel/excelDownload.do',
                                   data=fields)
            response.raise_for_status()
            # Some portals return an HTML table with an XLS extension. Preserve
            # its actual bytes and report its type instead of trusting headers.
            raw = response.content
            extension = '.xlsx' if raw.startswith(b'PK\x03\x04') else '.xls' if raw.startswith(b'\xd0\xcf\x11\xe0') else '.html'
            archive('K_REACH_PUBLIC_EXPORT', response, extension, request_fields=fields)
            if extension == '.html' and b'<table' not in raw.lower():
                raise ValueError('export returned neither workbook nor data table')
        except (httpx.HTTPError, ValueError) as error:
            failures.append({'source': 'K_REACH_PUBLIC_EXPORT', 'error': str(error)[:250]})

        url = 'https://www.fda.gov/cosmetics/cosmetics-laws-regulations/prohibited-restricted-ingredients-cosmetics'
        try:
            response = client.get(url)
            response.raise_for_status()
            text = BeautifulSoup(response.content, 'html.parser').get_text(' ', strip=True)
            for required in ('Bithionol', 'Hexachlorophene', 'Methylene chloride', 'Vinyl chloride'):
                if required not in text:
                    raise ValueError('FDA publication identity incomplete')
            archive('FDA_PROHIBITED_RESTRICTED_COSMETICS', response, '.html')
        except (httpx.HTTPError, ValueError) as error:
            failures.append({'source': 'FDA_PROHIBITED_RESTRICTED_COSMETICS', 'error': str(error)[:250]})

    result = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'sources': sources,
              'failures': failures, 'operator_approval_modified': False}
    (out / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'sources': len(sources), 'failures': failures}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
