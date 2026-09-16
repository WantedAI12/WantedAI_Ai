"""Source-closed, explicitly scoped V80 deployment and rollback boundary."""
import argparse
import json
from pathlib import Path

from deploy import runtime_release_v78 as base

ROOT = base.ROOT
RELEASE_ID = 'v80-supported-scope-20260915'
CORE_SHA256 = base.CORE_SHA256
sha = base.sha
registry_path = base.registry_path


def verify(root, expected):
    manifest = base.verify(root, expected, release_id=RELEASE_ID)
    root = Path(root)
    profile = json.loads((root/'perfumery.local.json').read_text(encoding='utf8'))
    space = json.loads((root/profile['odor_space']['path']).read_text(encoding='utf8'))
    reference = json.loads((root/profile['lotion_target_reference']['path']).read_text(encoding='utf8'))
    if space.get('version') != 'v80':
        raise ValueError('V80 generation support policy required')
    from fragrance_ai.recommender.odor_release_scope import validate_scope
    validate_scope(space, reference['profiles'])
    return manifest


def prepare(preparation, output, root=ROOT):
    result = base.prepare(preparation, output, root, release_id=RELEASE_ID)
    verify(output, result['bundle_sha256'])
    return result


def check_installed(root, expected):
    verify(root, expected)
    result = base.check_installed(root, expected, release_id=RELEASE_ID, data_version='v80')
    result['user_authorized_supported_scope'] = True
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--verify', type=Path)
    p.add_argument('--check-installed', type=Path)
    p.add_argument('--sha256')
    a = p.parse_args()
    if a.check_installed:
        check_installed(a.check_installed, a.sha256)
    elif a.verify:
        verify(a.verify, a.sha256)
    elif a.preparation and a.output:
        prepare(a.preparation, a.output)
    else:
        p.error('select prepare, verify, or installed check')
