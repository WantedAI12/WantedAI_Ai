"""Acquire public gas-phase intensity study sources; never synthesize observations."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

ARTICLE = 'PMC12363845'
ROOT_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/' + ARTICLE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('new source directory required')
    args.output.mkdir(parents=True)
    files = {}
    def save(name, data, **metadata):
        (args.output/name).write_bytes(data)
        files[name] = {**metadata, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    def fetch(endpoint, name, maximum):
        url = ROOT_URL + '/' + endpoint
        with urlopen(url, timeout=40) as response:
            data = response.read(maximum+1)
        if len(data) > maximum:
            raise ValueError('source download exceeds the fixed size bound')
        save(name, data, url=url)
        return data
    raw = fetch('fullTextXML', 'article.xml', 4_000_000)
    document = ET.fromstring(raw)
    article_ids = {node.text for node in document.findall('.//article-id')}
    if ARTICLE not in article_ids and ARTICLE.removeprefix('PMC') not in article_ids:
        raise ValueError('source article identity mismatch')
    names = [node.attrib.get('{http://www.w3.org/1999/xlink}href')
             for node in document.findall('.//supplementary-material/media')]
    if names != ['media-1.pdf']:
        raise ValueError('review changed supplementary-material inventory before import')
    package = fetch('supplementaryFiles', 'supplementary.zip', 64_000_000)
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == 'media-1.pdf']
        if len(matches) != 1 or archive.getinfo(matches[0]).file_size > 8_000_000:
            raise ValueError('ambiguous or oversized supplement')
        data = archive.read(matches[0])
    if len(data) != 600103 or hashlib.md5(data).hexdigest() != '87e4dfa8eaf50d697f4534fe915d8125':
        raise ValueError('supplement no longer matches the reviewed article metadata')
    save('media-1.pdf', data, archive_member=matches[0])
    manifest = {'schema': 'gas-phase-intensity-source/v1', 'article': ARTICLE,
        'doi': '10.1101/2025.08.08.668954', 'files': files,
        'raw_observations_imported': False, 'parameters_imported': False,
        'checkpoint_reproduced': False, 'lotion_calibration': False}
    (args.output/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
