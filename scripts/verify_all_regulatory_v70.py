"""Exercise every active material, framework and supported product mapping."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    preparation = json.loads(args.preparation.read_text())
    os.environ.update(PERFUMERY_AI_LOCAL_PROFILE=preparation['profile'], PERFUMERY_AI_ENV='research',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    from fragrance_ai.platform.public_evidence import PublicEvidenceStore
    from fragrance_ai.recommender.runtime import load_configured_catalog
    from fragrance_ai.recommender.safety import PRODUCT_CATEGORY_MAP
    store = PublicEvidenceStore.configured()
    catalog, _ = load_configured_catalog()
    active = [item for item in catalog.ingredients if item.formulation_ready and not item.blocked]
    assert len(active) == 3830
    coverage = store.coverage(catalog)
    assert coverage['active_material_count'] == len(active)
    assert len({row['ingredient_id'] for row in coverage['rows']}) == len(active)
    index = store.registry_indexes[store.value['version']]
    assert set(index.frameworks) == {'IFRA', 'EU_REACH', 'K_REACH', 'FDA'}
    products = sorted(set(PRODUCT_CATEGORY_MAP) | {'body_lotion'})
    ifra = index.frameworks['IFRA'][0]
    assert all(set(row['category_limits']) == set(products) for row in ifra['observations'])
    calls, exceeded, checks = 0, Counter(), Counter()
    # This calls the exact common numerical path used by public.screen. Check
    # all pinned bytes before and after the batch without millions of repeated
    # filesystem stat calls. No approvals or external API calls are made.
    store.assert_current()
    for material in active:
        lines = [{'ingredient_id': material.ingredient_id, 'cas_number': material.cas_number, 'concentrate_percent': 100.}]
        for framework in ('IFRA', 'EU_REACH', 'K_REACH', 'FDA'):
            for product in products if framework == 'IFRA' else [None]:
                value = index.findings(framework, lines, store.documents, category=product, concentration=1.)
                assert value is not None and not value['compliance_verified']
                assert len(value['material_coverage']) == 1
                calls += 1
                if framework == 'IFRA':
                    result = value['formula_rule_checks']
                    exceeded[product] += result['source_limit_exceeded']
                    checks[product] += len(result['checks'])
                    assert not result['full_IFRA_conformity_verified']
                    assert all(math.isfinite(row['finished_product_percent']) for row in result['checks'])
    # All 3,830 IDs in one software-only fixture tests cross-material group sums.
    lines = [{'ingredient_id': item.ingredient_id, 'cas_number': item.cas_number,
              'concentrate_percent': 100. / len(active)} for item in active]
    together = store.screen(lines, category='eau_de_parfum', concentration=15.)
    total_check = together['frameworks'][0]['public_registry_check']['formula_rule_checks']
    expected = defaultdict(float)
    for row in ifra['observations']:
        expected[row['rule_id']] += 100. / len(active) * .15 * row['material_active_strength_percent'] / 100. * row['contribution_coefficient_percent'] / 100.
    actual = {row['rule_id']: row['finished_product_percent'] for row in total_check['checks']}
    assert set(expected) == set(actual)
    assert all(abs(expected[key] - actual[key]) < 1e-12 for key in expected)
    assert not together['gate_passed'] and not together['manufacturing_approval']
    store.assert_current()
    result = {'passed': True, 'scope': 'all_active_materials_and_all_supported_product_categories_software_only',
        'active_materials': len(active), 'frameworks': list(index.frameworks), 'product_categories': products,
        'single_material_framework_product_checks': calls, 'all_material_group_sum_checks': len(actual),
        'ifra_matched_rule_checks_by_product': dict(checks), 'ifra_source_exceedances_at_1_percent_test_dose': dict(exceeded),
        'public_bundle_sha256': store.digest, 'wheel_sha256': preparation['wheel_sha256'],
        'catalog_sha256': preparation['catalog_sha256'], 'coverage': {k: v for k, v in coverage.items() if k != 'rows'},
        'source_population': index.contract(), 'external_api_calls': 0, 'manufacturing_approvals_created': 0}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
