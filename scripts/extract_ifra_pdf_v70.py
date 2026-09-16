"""Extract every page of archived IFRA PDFs; preserve page numbers and hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / '.benchmarks/v70_repair/ifra-official-01'
manifest = json.loads((path / 'manifest.json').read_text())
for source in manifest['sources']:
    if not source['path'].endswith('.pdf'):
        continue
    raw = (path / source['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('source PDF changed')
    reader = PdfReader(path / source['path'])
    pages = [{'page': i + 1, 'text': page.extract_text(extraction_mode='layout')}
             for i, page in enumerate(reader.pages)]
    (path / (source['id'] + '_pages.json')).write_text(json.dumps({'source_sha256': source['sha256'],
        'extracted_at': datetime.now(timezone.utc).isoformat(), 'pages': pages}, ensure_ascii=False), encoding='utf-8')
    (path / (source['id'] + '.txt')).write_text('\n\f\n'.join(row['text'] for row in pages), encoding='utf-8')
    print(json.dumps({'id': source['id'], 'pages': len(pages), 'empty_pages': sum(not row['text'].strip() for row in pages)}), flush=True)
