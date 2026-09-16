"""Preserve two public formulation datasets with licenses and immutable hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'PerfumeryAI-local-research/1.0'})
    with urllib.request.urlopen(request,timeout=45) as r:
        return r.read()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    records={}
    def save(name,raw,url):
        (a.output/name).write_bytes(raw)
        records[name]={'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    meta_url='https://api.figshare.com/v2/articles/25451878'
    meta_raw=fetch(meta_url)
    meta=json.loads(meta_raw)
    if meta['license']['name']!='CC0':
        raise ValueError('formulation data license changed')
    file=next(row for row in meta['files'] if row['id']==45187180)
    raw=fetch(file['download_url'])
    # Source MD5 is used only for downloaded-file integrity, not security.
    if len(raw)!=1464630 or hashlib.md5(raw).hexdigest()!='8898e7b460c71e334f6ece5daebc6a6e':
        raise ValueError('inspected experimental data changed')
    save('liquid-metadata.json',meta_raw,meta_url)
    save('liquid-formulations.json',raw,file['download_url'])
    commit_meta=fetch('https://api.github.com/repos/odor-pair/odor-pair/commits/main')
    commit=json.loads(commit_meta)['sha']
    save('odor-pair-commit.json',commit_meta,'https://api.github.com/repos/odor-pair/odor-pair/commits/main')
    base='https://raw.githubusercontent.com/odor-pair/odor-pair/'+commit+'/'
    license_raw=fetch(base+'LICENSE')
    if b'MIT License' not in license_raw:
        raise ValueError('odor-pair repository license changed')
    for name,path in [('odor-pair-LICENSE.txt','LICENSE'),('odor-pair-README.md','README.md'),('odor-pairs.json','dataset/full.json')]:
        raw=license_raw if path=='LICENSE' else fetch(base+path)
        save(name,raw,base+path)
    manifest={'schema':'formulation-public-sources/v76','files':records,
        'liquid':{'doi':'10.6084/m9.figshare.25451878.v1','license':'CC0',
                  'scope':'experimental_rinse_off_formulations_not_cosmetic_lotion_or_fragrance_quality'},
        'odor_pairs':{'doi':'10.1021/acsomega.4c07078','repository_license':'MIT','commit':commit,
                      'scope':'published_pair_odor_annotations_not_known_concentration_or_recipe_similarity'},
        'model_code_imported':False,'source_measurement_classes_kept_separate':True}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({name:{'bytes':row['bytes'],'sha256':row['sha256']} for name,row in records.items()}))


if __name__=='__main__':
    main()
