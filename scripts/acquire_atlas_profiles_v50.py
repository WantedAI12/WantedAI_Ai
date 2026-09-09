"""Acquire only pinned public Atlas files for local model development."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = '7d1e2da13ecfa0054386eea9aa5edd915205c9e7'
FILES = {
    'behavior_1.csv': '09a82a31c168a9c562a6ddbde8fdd5ccba11fc5d',
    'behavior_2.csv': 'bcec145fa50fb23c7ce3f8933e7516be3d979e84',
    'molecules.csv': '3d29a8640e26bbf484126ad62ca751ae77b40720',
    'stimuli.csv': '553f6769ab8d9b1e70a97334beb313b419adbec2',
    'manifest.toml': '0fd1491502941bf7beee259752110bf1b6eaa1d8',
    'main.py': '7069eb18326ccc67fcede7900b1c2530b3eec40f',
}


def acquire(destination):
    destination.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        name,blob = item
        path = destination/name
        url = f'https://raw.githubusercontent.com/pyrfume/pyrfume-data/{COMMIT}/dravnieks_1985/{name}'
        if path.exists():
            raw = path.read_bytes()
        else:
            with urlopen(url,timeout=30) as response:
                raw = response.read(2*1024*1024+1)
        if len(raw) > 2*1024*1024 or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != blob:
            raise ValueError('pinned source mismatch: '+name)
        if not path.exists():
            with path.open('xb') as handle: handle.write(raw)
        return name,{'url':url,'git_blob':blob,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(fetch,FILES.items()))
    result = {'schema':'atlas-source-snapshot/v1','commit':COMMIT,'files':files,
        'source_specific_redistribution_authorized':False,'downloaded_code_executed':False,
        'scope':'local_research_not_packaged_data','doi':'10.1520/DS61-EB'}
    path = destination/'source_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != result: raise ValueError('source manifest changed')
    else:
        with path.open('x',encoding='utf-8') as handle: json.dump(result,handle,indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.output),indent=2))
