"""Acquire the complete public heading, qualify gas endpoints, resolve exact CID.

Uses a small number of batched public data requests, never one request per
recipe ingredient. Only HSDB numeric observations enter the derived index.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.extend_qualified_thresholds_v76 import measurements
from scripts.build_physical_evidence_v76 import identity


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records={}
    def get(url,name):
        time.sleep(.3)
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PerfumeryAI-Research'}),timeout=45) as r:
            raw=r.read()
        (a.output/name).write_bytes(raw)
        records[name]={'url':url,'sha256':sha(raw),'bytes':len(raw)}
        return json.loads(raw)
    first=get('https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON/?heading=Odor%20Threshold&page=1','heading-1.json')
    annotations=list(first['Annotations']['Annotation'])
    for page in range(2,first['Annotations']['TotalPages']+1):
        annotations.extend(get('https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON/?heading=Odor%20Threshold&page='+str(page),f'heading-{page}.json')['Annotations']['Annotation'])
    qualified=[r for r in annotations if r['SourceName']=='Hazardous Substances Data Bank (HSDB)' and len(r.get('LinkedRecords',{}).get('CID',[]))==1]
    cids=sorted({r['LinkedRecords']['CID'][0] for r in qualified})
    properties={}
    for start in range(0,len(cids),100):
        url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+','.join(map(str,cids[start:start+100]))+'/property/MolecularWeight,InChIKey,IsomericSMILES/JSON'
        values=get(url,f'properties-{start//100}.json')['PropertyTable']['Properties']
        properties.update({row['CID']:row for row in values})
    by_key=defaultdict(list);graphs={};rejected=[]
    for row in qualified:
        cid=row['LinkedRecords']['CID'][0]
        prop=properties.get(cid,{})
        ident=identity(prop.get('SMILES',prop.get('IsomericSMILES')))
        if ident is None or ident[0]!=prop.get('InChIKey'):
            rejected.append({'cid':cid,'reason':'not_exact_neutral_single_molecule'})
            continue
        key,graph,mw=ident;graphs[key]={'graph':graph,'molecular_weight':mw}
        for datum in row.get('Data',[]):
            for field in datum.get('Value',{}).get('StringWithMarkup',[]):
                for value in measurements(field.get('String',''),mw):
                    by_key[key].append({'value':value,'unit':'ppmv','source_ref':row['URL'],'source_cid':cid,
                                        'source_class':'qualified_gas_phase_HSDB_literature'})
    doc=json.loads(a.base.read_text(encoding='utf8'));added=[]
    for key,observations in by_key.items():
        row=doc['by_inchikey'].setdefault(key,{**graphs[key],'properties':{}})
        if 'odor_threshold_ppm' in row['properties']:
            continue
        values=[r['value'] for r in observations]
        row['properties']['odor_threshold_ppm']={'value':math.exp(statistics.median(map(math.log,values))),
            'unit':'ppmv','source_ref':'pubchem-HSDB:heading:CID:'+str(observations[0]['source_cid']),
            'source_class':'heterogeneous_public_gas_threshold_prior','observations':observations,
            'reference_temperature_k':298.15,'temperature_note':'unit_conversion_reference_not_original_assay_temperature',
            'minimum_ppmv':min(values),'maximum_ppmv':max(values),'aggregation':'geometric_median_of_qualified_gas_observations'}
        added.append(key)
    doc['counts']['odor_threshold_ppm']=sum('odor_threshold_ppm' in r['properties'] for r in doc['by_inchikey'].values())
    doc['whole_heading_extension']={'all_annotations':len(annotations),'all_pages':first['Annotations']['TotalPages'],
        'single_cid_HSDB_annotations':len(qualified),'resolved_cids':len(properties),'gas_threshold_identities':len(by_key),
        'added':len(added),'identity_rejections':rejected,'sources':records,'parent_sha256':sha(a.base.read_bytes()),
        'restricted_sources_used':False,'recipe_outcomes_used':False}
    raw=json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8')
    (a.output/'index.json').write_bytes(raw)
    print(json.dumps({'whole_heading_annotations':len(annotations),'qualified_identity_count':len(by_key),
                      'added_thresholds':len(added),'counts':doc['counts'],'sha256':sha(raw)}))


if __name__=='__main__':
    main()
