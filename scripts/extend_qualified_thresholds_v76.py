"""Join all verifiable gas-phase HSDB thresholds from the existing raw cache.

Commercially restricted annotation sources and ambiguous aqueous/recognition
values are not included. A heterogeneous range is a literature prior, not a lot
assay; its complete numeric observations remain in the output evidence record.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_physical_evidence_v76 import identity

NUMBER=r'\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?'
UNIT=r'(?:ppm|ppb|mg\s*/\s*(?:m\^?3|cu\s*m))'
VALUES=re.compile(rf'({NUMBER})\s*(?:\[\s*)?({UNIT})(?:\s*\])?',re.I)
RANGES=re.compile(rf'({NUMBER})\s*(?:to|–|-)\s*({NUMBER})\s*({UNIT})',re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measurements(text,mw):
    if ';' in text or '\n' in text:
        return [v for clause in re.split('[;\n]',text) for v in measurements(clause,mw)]
    lower=text.casefold()
    if any(word in lower for word in ('water','aqueous','recognition','irritat','taste','odor high')):
        return []
    if not any(word in lower for word in ('odor','detection','threshold','air')):
        return []
    records=[]
    for match in RANGES.finditer(text):
        unit=match.group(3).casefold()
        if 'air' not in lower and not unit.startswith('mg'):
            continue
        records.extend((float(match.group(i).replace(',','')),unit) for i in (1,2))
    if not records:
        for match in VALUES.finditer(text):
            prefix=lower[max(0,match.start()-22):match.start()]
            unit=match.group(2).casefold()
            if 'high' in prefix or ('air' not in lower and not unit.startswith('mg')):
                continue
            records.append((float(match.group(1).replace(',','')),unit))
    result=[]
    for value,unit in records:
        if unit=='ppb':
            value/=1000.
        elif unit.startswith('mg'):
            value*=8.314462618*298.15*1000/(101325.*mw)
        if math.isfinite(value) and 0<value<1e6:
            result.append(value)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','registry','cache-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('new immutable output required')
    doc=json.loads(a.base.read_text(encoding='utf8'))
    registry=json.loads(a.registry.read_text(encoding='utf8'))
    checked=[];added=0
    for row in registry['records']:
        cid=row['preferred_cid']
        if type(cid) is not int or cid<=0:
            raise ValueError('explicit positive PubChem CID required')
        cache=a.cache_root/str(cid)/'odor_threshold.json'
        audit=row['annotation_audit']
        if audit.get('status')!='ok' or not cache.is_file() or sha(cache)!=audit['cache_sha256']:
            continue
        envelope=json.loads(cache.read_text(encoding='utf8'))
        record=envelope['payload']['Record']
        ident=identity(row['canonical_smiles'])
        if record['RecordNumber']!=cid or ident is None:
            continue
        key,graph,mw=ident
        refs={r['ReferenceNumber']:r for r in record.get('Reference',[]) if r.get('SourceName')=='Hazardous Substances Data Bank (HSDB)'}
        observations=[]
        def walk(node):
            if isinstance(node,dict):
                if node.get('ReferenceNumber') in refs and 'Value' in node:
                    for entry in node['Value'].get('StringWithMarkup',[]):
                        for value in measurements(entry.get('String',''),mw):
                            observations.append({'value':value,'unit':'ppmv','source_ref':refs[node['ReferenceNumber']]['URL'],
                                                 'source_class':'gas_phase_HSDB_literature_annotation'})
                for value in node.values():
                    walk(value)
            elif isinstance(node,list):
                for value in node:
                    walk(value)
        walk(record.get('Section',[]))
        checked.append({'cid':cid,'cache_sha256':sha(cache),'qualified_observations':len(observations)})
        if not observations:
            continue
        value=float(math.exp(float(statistics.median(math.log(r['value']) for r in observations))))
        target=doc['by_inchikey'].setdefault(key,{'graph':graph,'molecular_weight':mw,'properties':{}})
        if 'odor_threshold_ppm' in target['properties']:
            continue
        target['properties']['odor_threshold_ppm']={'value':value,'unit':'ppmv',
            'source_ref':'pubchem-HSDB:CID:'+str(cid),'source_class':'heterogeneous_public_gas_threshold_prior',
            'aggregation':'geometric_median_of_qualified_gas_observations',
            'reference_temperature_k':298.15,'temperature_note':'conversion_reference_not_every_original_assay_temperature',
            'observations':observations,'source_cache_sha256':sha(cache),
            'minimum_ppmv':min(r['value'] for r in observations),'maximum_ppmv':max(r['value'] for r in observations)}
        added+=1
    doc['counts']['odor_threshold_ppm']=sum('odor_threshold_ppm' in r['properties'] for r in doc['by_inchikey'].values())
    doc['qualified_HSDB_extension']={'registry_sha256':sha(a.registry),'parent_sha256':sha(a.base),'checked':checked,'added':added,
                                   'restricted_sources_used':False,'ambiguous_aqueous_or_recognition_values_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'added':added,'checked_cids':len(checked),'counts':doc['counts'],'sha256':sha(a.output)}))


if __name__=='__main__':
    main()
