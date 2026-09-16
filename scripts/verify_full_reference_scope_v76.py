"""Check every frozen brief against the complete source-reference route."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('core','reference','bank','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['PERFUMERY_AI_LOCAL_PROFILE']='disabled'
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.recommender.lotion_reference_objective import ObservedReferenceBank
    from fragrance_ai.recommender.formulation_guidance import SharedPerfumeGuidance
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.models import Ingredient,SCENT_DIMENSIONS
    core=FormulationCore(a.core,sha(a.core),allow_candidate=True)
    original=json.loads(a.reference.read_text(encoding='utf8'));value=deepcopy(original)
    value['parent_atlas_sha256']=core.sha256
    target=a.output/'reference.json';target.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    for key in ('profiles','background','concept_metadata','endpoints'):
        if value[key]!=original[key]:
            raise ValueError('reference vectors changed')
    bank=ObservedReferenceBank(target,sha(target))
    provider=SharedPerfumeGuidance(core,reference_bank=bank)
    items=[Ingredient(**r) for r in json.loads((a.bank/'materials.json').read_text(encoding='utf8'))]
    parser=NaturalLanguageBriefParser(IngredientCatalog(items,{}))
    cases=json.loads(a.protocol.read_text(encoding='utf8'))['cases']
    if len(cases)!=400 or len({r['id'] for r in cases})!=400:
        raise ValueError('all frozen 400 cases required')
    rows=[]
    for case in cases:
        session=provider.begin(parser.parse(case['brief']))
        rows.append({**case,'enabled':session.enabled,'supported':session.supported_intent,
                     'unsupported':session.unsupported_intent,
                     'missing_positive_target':any(t is None for t in session.reference_targets)})
    report={'core_sha256':core.sha256,'source_reference_sha256':sha(a.reference),'reference_vectors_unchanged':True,
        'cases':len(rows),'all_19_reference_axes_available':set(SCENT_DIMENSIONS)<=set(bank.profiles),
        'old_unmapped_axes_now_available':[x for x in ('fresh','aquatic','white_floral','amber') if x in bank.profiles],
        'enabled_cases':sum(r['enabled'] for r in rows), 'disabled_cases':[r for r in rows if not r['enabled']],
        'scope':'complete_reference_routing_not_recipe_success_or_human_accuracy','rows':rows}
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
