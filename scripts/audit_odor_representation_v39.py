"""Snapshot language targets; compare versioned representations, not human accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def snapshot():
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.models import RecipeConstraints
    from scripts.evaluate_request_space import request_cases
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    cases = request_cases() + [
        {'id': 'compound-' + str(i), 'brief': text} for i, text in enumerate((
            'green fruity scent', 'green-fruity scent', '풋과일 향',
            'tropical scent', '열대과일 향', '열대 과일 향',
            'white floral scent', '화이트플로럴 향',
            'woody without green fruity', 'floral without tropical',
            'jasmine without floral', 'rose without floral',
            'green fruity and woody scent', '풋과일 향 우디',
        ))]
    rows = []
    for case in cases:
        try:
            brief = parser.parse(case['brief'], RecipeConstraints(enable_semantic_ontology=False))
            rows.append({**case, 'profile': brief.target_profile, 'avoided': brief.avoided_dimensions,
                         'descriptors': brief.recognized_descriptors})
        except ValueError as error:
            rows.append({**case, 'error': str(error)})
    return {'rows': rows, 'parser_sha256': hashlib.sha256(
        (ROOT/'fragrance_ai/recommender/brief_parser.py').read_bytes()).hexdigest(),
        'scope': 'language_representation_diagnostic_not_recipe_pass_rate_or_human_accuracy',
        'semantic_ontology_enabled': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output path')
    result = snapshot()
    if args.baseline:
        before = {r['id']: r for r in json.loads(args.baseline.read_text(encoding='utf-8'))['rows']}
        changed = [r['id'] for r in result['rows'] if r.get('profile') != before[r['id']].get('profile')]
        result.update(changed_target_ids=changed, changed_target_count=len(changed),
                      old_recipe_scores_directly_comparable=not changed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
