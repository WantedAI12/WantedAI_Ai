"""Reproject verified active descriptors; retain identities, measurements and safety."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reproject(catalog):
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.odor_integrity import (
        CONCEPT_ODOR_PROJECTION, POSITIVE_ODOR_STATUS, REGISTRY_ODOR_SOURCE,
        odor_assertion_representation, registry_odor_rejection,
    )
    materials, details = [], []
    for item in catalog.ingredients:
        # Measured profiles, pending data and rejected materials are immutable.
        if not (item.formulation_ready and not item.blocked and item.data_source == REGISTRY_ODOR_SOURCE):
            materials.append(item)
            continue
        if reason := registry_odor_rejection(item):
            raise ValueError('unverified source material: ' + reason)
        representation = odor_assertion_representation(item.odor_assertions)
        if representation['status'] != POSITIVE_ODOR_STATUS:
            raise ValueError('concept projection lost positive source evidence')
        new = replace(item, profile=representation['coarse_profile'], odor_projection_version=CONCEPT_ODOR_PROJECTION)
        if reason := registry_odor_rejection(new):
            raise ValueError('invalid projected material: ' + reason)
        after = asdict(new)
        changed = {key for key, value in asdict(item).items() if value != after[key]}
        if changed - {'profile', 'odor_projection_version'}:
            raise ValueError('non-profile material fields changed')
        details.append({'ingredient_id': item.ingredient_id,
                        'profile_changed': 'profile' in changed,
                        'profile_l1_delta': sum(abs(item.profile.get(k, 0) - v) for k, v in new.profile.items()),
                        'representation': representation})
        materials.append(new)
    return IngredientCatalog(materials, {**catalog.metadata,
        'odor_concept_projection_version': CONCEPT_ODOR_PROJECTION}), details


def main():
    from fragrance_ai.recommender.odor_integrity import ODOR_INTEGRITY_VERSION, CONCEPT_ODOR_PROJECTION
    from fragrance_ai.recommender.registry_activation import load_runtime_catalog, write_runtime_catalog
    from scripts.build_odor_integrity_catalog import verify_wheel_sources, digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--wheel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rebind-only', action='store_true', help='preserve every material field; update only package/source binding')
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output directory')
    if digest(args.source_manifest) != args.source_manifest_sha256:
        p.error('source manifest hash mismatch')
    source = json.loads(args.source_manifest.read_text(encoding='utf-8'))['runtime_catalog']
    target = (args.source_manifest.resolve().parent/source['path']).resolve()
    if not target.is_relative_to(args.source_manifest.resolve().parent):
        p.error('source catalog escapes its manifest directory')
    catalog, report, stats = load_runtime_catalog(target, expected_sha256=source['sha256'],
        expected_wheel_sha256=source['wheel_sha256'], expected_registry_sha256=source['registry_sha256'])
    sources = verify_wheel_sources(args.wheel)
    new, details = (catalog, []) if args.rebind_only else reproject(catalog)
    from fragrance_ai.recommender.runtime import _material_digest
    before_materials, after_materials = _material_digest(catalog), _material_digest(new)
    if args.rebind_only and before_materials != after_materials:
        raise ValueError('package-only rebinding changed material data')
    args.output.mkdir(parents=True)
    wheel_sha = digest(args.wheel)
    catalog_sha = write_runtime_catalog(args.output/'runtime_catalog_v3.json.gz', new, report, stats, wheel_sha256=wheel_sha)
    with zipfile.ZipFile(args.wheel) as archive:
        assets = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                  if name.startswith('fragrance_ai/data/') and Path(name).suffix in {'.db', '.json', '.npz'}}
    audit = {'rows': details, 'reprojected_materials': len(details),
        'changed_profiles': sum(r['profile_changed'] for r in details),
        'active_materials': sum(i.formulation_ready and not i.blocked for i in new.ingredients),
        'source_manifest_sha256': args.source_manifest_sha256,
        'profile_projection_changes_are_not_new_sensory_measurements': True,
        'raw_assertions_lineage_identity_price_availability_caps_risk_unchanged': True,
        'package_rebinding_only': args.rebind_only,
        'all_material_fields_unchanged': before_materials == after_materials,
        'source_material_snapshot_sha256': before_materials, 'material_snapshot_sha256': after_materials}
    audit_path = args.output/'profile_audit.json'
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'schema': 'perfumery-odor-integrity-catalog/v1', 'status': 'local_research_candidate_not_deployed',
        'odor_integrity_version': ODOR_INTEGRITY_VERSION, 'odor_projection_version': CONCEPT_ODOR_PROJECTION,
        'runtime_catalog': {'path': 'runtime_catalog_v3.json.gz', 'sha256': catalog_sha,
            'wheel_sha256': wheel_sha, 'registry_sha256': source['registry_sha256'],
            'connected_rows': len(new.ingredients), 'active_odorants': audit['active_materials']},
        'runtime_source_sha256': sources, 'runtime_data_sha256': assets,
        'source_manifest_sha256': args.source_manifest_sha256, 'profile_audit_sha256': digest(audit_path),
        'package_rebinding_only': args.rebind_only, 'material_snapshot_sha256': after_materials,
        'old_profile_scores_comparable': False, 'actual_human_accuracy_claim': False,
        'source_specific_redistribution_cleared': False}
    manifest_path = args.output/'catalog_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({**{k: v for k, v in audit.items() if k != 'rows'},
                      'manifest_sha256': digest(manifest_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
