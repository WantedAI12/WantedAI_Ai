"""Freeze separated/shared model selections with identical request-time inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepared = json.loads(args.preparation.read_text(encoding='utf-8'))
    if sha(prepared['wheel']) != prepared['wheel_sha256'] or sha(prepared['profile']) != prepared['profile_sha256']:
        raise ValueError('candidate preparation changed')
    shared = json.loads(Path(prepared['profile']).read_text(encoding='utf-8'))
    separated = json.loads((ROOT/'perfumery.v68.rollback.json').read_text(encoding='utf-8'))
    reference = json.loads((ROOT/shared['lotion_target_reference']['path']).read_text(encoding='utf-8'))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Binding an independently observed target to a different decoder does not
    # refit or alter any target vector or definition.
    split_reference = deepcopy(reference)
    split_reference['parent_atlas_sha256'] = separated['odor_backbone']['sha256']
    target = output/'separated-target-reference.json'
    target.write_text(json.dumps(split_reference, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for key in ('profiles','background','concept_metadata','endpoints'):
        assert split_reference[key] == reference[key]
    separated['lotion_target_reference'] = {'path': target.relative_to(ROOT).as_posix(), 'sha256': sha(target)}
    for key in ('catalog','public_evidence'):
        separated[key] = deepcopy(shared[key])
    # Natural-language helper latency is not part of quantitative generation.
    separated['language'] = shared['language'] = None
    versions = {}
    for label, value in (('separated',separated),('shared',shared)):
        path = ROOT/f'perfumery.v72-{output.name}-{label}.local.json'
        if path.exists():
            raise ValueError('new profile names required')
        for name, binding in value.items():
            if isinstance(binding, dict) and 'path' in binding and 'sha256' in binding:
                source = (ROOT/binding['path']).resolve()
                if not source.is_relative_to(ROOT) or sha(source) != binding['sha256']:
                    raise ValueError('missing or changed model binding: '+name)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        versions[label] = {'profile': str(path), 'profile_sha256': sha(path),
            'wheel':prepared['wheel'], 'wheel_sha256':prepared['wheel_sha256'],
            'model_configuration': {k:v['sha256'] for k,v in value.items() if isinstance(v,dict) and 'sha256' in v}}
    report = {'versions': versions, 'shared_request_time_source_wheel':prepared['wheel_sha256'],
        'same_catalog':True, 'same_118_reference_vectors':True, 'same_threshold':95.,
        'same_physics_sampling':'ingredient-coupled-prior-sampling/v1',
        'contrast':'pre-integration configured model suite versus single shared V69 checkpoint',
        'pure_architecture_causal_effect_claimed':False, 'runtime_default_changed':False,
        'reference_vectors_sha256':hashlib.sha256(json.dumps(reference['profiles'], sort_keys=True).encode()).hexdigest()}
    (output/'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
