"""Write a new local research profile; never replace the current selection."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-profile',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().parent!=ROOT:
        raise ValueError('new root-local research profile name required')
    profile=json.loads(a.parent_profile.read_text(encoding='utf8'))
    def binding(path):
        return {'path':path.resolve().relative_to(ROOT).as_posix(),'sha256':sha(path)}
    profile['physical_evidence']=binding(a.evidence)
    if a.model:
        model=json.loads(a.model.read_text(encoding='utf8'))
        if model.get('accepted_for_local_inference') is not True:
            raise ValueError('cannot bind an unaccepted model as a runtime profile')
        profile['formulation_core']=binding(a.model)
        original=ROOT/profile['lotion_target_reference']['path']
        reference=json.loads(original.read_text(encoding='utf8'))
        rebound=deepcopy(reference);rebound['parent_atlas_sha256']=sha(a.model)
        target=a.model.parent/'target-reference.json'
        if target.exists():
            raise ValueError('new target binding path required')
        target.write_text(json.dumps(rebound,ensure_ascii=False,indent=2),encoding='utf8')
        profile['lotion_target_reference']=binding(target)
    a.output.write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'profile':str(a.output.resolve()),'sha256':sha(a.output),'runtime_default_changed':False}))


if __name__=='__main__':
    main()
