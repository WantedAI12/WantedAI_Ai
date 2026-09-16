"""Bind the final API/data repair to the numerical candidate used in all 400 cases."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ANNOTATION_OR_EVIDENCE_FILES = {
    'fragrance_ai/platform/ai_extensions.py',
    'fragrance_ai/platform/public_evidence.py',
    'fragrance_ai/platform/public_registry_index.py',
    'fragrance_ai/platform/rd_api.py',
    'fragrance_ai/platform/rd_evidence.py',
    'fragrance_ai/recommender/regulatory_status.py',
    'fragrance_ai/recommender/lotion_regulatory.py',
}
CACHE_FILE = 'fragrance_ai/recommender/formulation_views.py'


def package_hashes(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('package digest changed')
    with zipfile.ZipFile(path) as package:
        return {name: hashlib.sha256(package.read(name)).hexdigest() for name in package.namelist()
                if name.startswith('fragrance_ai/')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--cache-equivalence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads(path.read_text()) for path in (args.before, args.after)]
    old = package_hashes(Path(before['wheel']), before['wheel_sha256'])
    new = package_hashes(Path(after['wheel']), after['wheel_sha256'])
    changed = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
    if changed - ANNOTATION_OR_EVIDENCE_FILES - {CACHE_FILE}:
        raise ValueError('additional scientific/model/data code changed after full benchmark candidate')
    cache = json.loads(args.cache_equivalence.read_text())
    if (cache['active_materials'] != 3830 or cache['endpoints'] != 450
            or not cache['all_predictions_exactly_equal'] or not cache['all_provenance_rows_exactly_equal']
            or cache['current_source_sha256'] != new[CACHE_FILE]):
        raise ValueError('cache implementation is not covered by full-material equivalence')
    if before['model_sha256'] != after['model_sha256'] or before['material_digest'] != after['material_digest']:
        raise ValueError('model weights or material values changed')
    result = {'passed': True, 'benchmark_wheel_sha256': before['wheel_sha256'],
        'final_wheel_sha256': after['wheel_sha256'], 'model_sha256': after['model_sha256'],
        'material_digest': after['material_digest'], 'changed_files': sorted(changed),
        'other_scientific_modules_and_packaged_data_byte_identical': True,
        'cache_all_materials_exact_output_equivalence': cache,
        'scope': 'numerical_search_and_prediction_equivalence_not_identical_API_JSON_or_regulatory_outcomes',
        'final_API_and_regulatory_tests_required_separately': True}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
