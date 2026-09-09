"""Local runtime integration and paired profile diagnostics with fixed language targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_catalog(path):
    from fragrance_ai.recommender.registry_activation import load_runtime_catalog
    b = json.loads(path.read_text(encoding='utf-8'))['runtime_catalog']
    return load_runtime_catalog(path.parent/b['path'], expected_sha256=b['sha256'],
        expected_wheel_sha256=b['wheel_sha256'], expected_registry_sha256=b['registry_sha256'])[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output file')
    if any(os.environ.get(name) for name in ('PERFUMERY_AI_PERCEPTION_MANIFEST', 'PERFUMERY_AI_PERCEPTION_MANIFEST_SHA256',
                                           'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST', 'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST_SHA256')):
        p.error('this representation ablation requires the learned runtime to be unset')
    from fragrance_ai.recommender.runtime import RuntimeAIFactory
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    from fragrance_ai.recommender.odor_integrity import assess_odor_assertions, EXPANDED_ODOR_PROJECTION
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    factory = RuntimeAIFactory.from_environment(manifest_path=args.manifest, expected_manifest_sha256=manifest_sha)
    # Factory construction validates the real package, data and inference engine.
    with factory() as ai:
        assert ai.catalog.ingredients == factory.catalog.ingredients
    old = read_catalog(ROOT/'dist/body-lotion-v32/profile-extension/catalog_manifest.json')
    new = factory.catalog
    frozen_parser = NaturalLanguageBriefParser(old)
    texts = ('clean fresh citrus woody', 'floral fruity woody', 'green aromatic woody',
             'minty woody', 'camphoreous woody', 'winey fruity', 'green fruity woody',
             'tropical woody', 'white floral woody', 'rose without floral')
    report = {'scope': 'ten_predeclared_profile_diagnostics_not_full400_or_human_accuracy',
        'targets': 'same_prechange_catalog_parser_for_both_arms_using_corrected_language_model',
        'learned_model_enabled': False, 'fixed_oil_percent': 10,
        'manifest_sha256': manifest_sha, 'runtime_factory_connected': True,
        'language_representation_version': frozen_parser.parse('clean woody').language_representation_version,
        'rows': []}
    paired_old = {i.ingredient_id: i for i in old.ingredients}
    report['canonical_dedup_changes_relative_to_v2'] = sum(
        any(abs(dict(assess_odor_assertions(tuple(i.odor_assertions), EXPANDED_ODOR_PROJECTION)[1]).get(k, 0)-v) > 1e-12
            for k, v in i.profile.items())
        for i in new.ingredients if i.odor_projection_version == 'explicit-odor-projection-3')
    report['changed_existing_profiles'] = sum(i.profile != paired_old[i.ingredient_id].profile for i in new.ingredients)
    with args.output.open('x', encoding='utf-8') as output:
        for text in texts:
            row = {'brief': text}
            for label, catalog in (('before', old), ('after', new)):
                start = time.perf_counter()
                try:
                    result = estimate_lotion_recipe(LotionEstimateRequest(brief=text,
                        registry_pool='conditional_research', max_risk_tier=2), catalog, parser=frozen_parser)
                    row[label] = {k: result.get(k) for k in ('score', 'status', 'profile_target_met')}
                    row[label]['candidate_count'] = result['estimation']['candidate_count']
                except Exception as error:
                    row[label] = {'error': type(error).__name__ + ': ' + str(error)}
                row[label]['seconds'] = time.perf_counter()-start
            report['rows'].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report['before_passed95'] = sum(bool(r['before'].get('profile_target_met')) for r in report['rows'])
        report['after_passed95'] = sum(bool(r['after'].get('profile_target_met')) for r in report['rows'])
        report['errors'] = sum('error' in r[arm] for r in report['rows'] for arm in ('before', 'after'))
        factory.assert_current_snapshot()
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
