"""Freeze conditional human odor references without reading recipe outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fragrance_ai.research.atlas_profiles import load_atlas
from fragrance_ai.recommender.lotion_reference_objective import VERSION, fit_references


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--atlas-model',type=Path)
    p.add_argument('--atlas-sha256')
    args = p.parse_args()
    if bool(args.atlas_model) != bool(args.atlas_sha256):
        p.error('atlas model and SHA256 must be provided together')
    if args.output.exists():
        p.error('a new output directory is required')
    rows, endpoints, source = load_atlas(ROOT/'.benchmarks/atlas_profiles_v50/source')
    rows = [r for r in rows if r['level'] == 'high']
    parent = args.atlas_model or ROOT/'.benchmarks/quantitative_profiles_v54/run-01/model.json'
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    if args.atlas_model:
        from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor
        predictor = AtlasProfilePredictor(parent,sha256=args.atlas_sha256,experimental=True)
        if predictor.endpoints != tuple(endpoints):
            raise ValueError('reference source and molecular output axes differ')
    artifact = {'schema': VERSION, 'endpoints': endpoints, **fit_references(rows, endpoints),
        'source': source, 'parent_atlas_sha256': parent_sha, 'source_stimuli': [r['id'] for r in rows],
        'recipe_outcomes_used': False, 'source_scope': 'local_research_not_redistribution',
        'fit': 'squared_descriptor_fraction_weighted_conditional_mean_no_hyperparameter_search',
        'human_lotion_mixture_calibrated': False,
        'code_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__), ROOT/'fragrance_ai/recommender/lotion_reference_objective.py')}}
    args.output.mkdir(parents=True)
    (args.output/'model.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Retrospective grouped reference discrimination: a held-out profile must
    # prefer its source-observed strongest concept to a mismatched concept.
    # The query is derived from an observation, not a user-authored brief.
    from fragrance_ai.recommender.lotion_reference_objective import ROUTES, normalize
    groups = [r['graph'] or r['id'] for r in rows]
    folds = [int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 5 for g in groups]
    records = []
    for fold in range(5):
        train = [r for r, f in zip(rows, folds) if f != fold]
        fitted = fit_references(train, endpoints)
        concepts = list(fitted['profiles'])
        for r, f in zip(rows, folds):
            if f != fold:
                continue
            actual = normalize(r['applicability'])
            strength = [float(actual[[endpoints.index(e) for e in ROUTES[c]]].sum()) for c in concepts]
            truth = concepts[int(np.argmax(strength))]
            scores = {c: float(np.dot(actual, v[0])/(np.linalg.norm(actual)*np.linalg.norm(v[0])))
                      for c, v in fitted['profiles'].items()}
            ordered = sorted(scores, key=scores.get, reverse=True)
            records.append({'id': r['id'], 'group': r['graph'] or r['id'], 'fold': fold,
                'query': truth, 'rank': ordered.index(truth)+1, 'top1': ordered[0],
                'scope': 'heldout_source_descriptor_retrieval_not_recipe_or_user_accuracy'})
    report = {'rows': len(rows), 'identity_groups': len(set(groups)), 'concept_count': len(artifact['profiles']),
        'concepts': list(artifact['profiles']), 'unsupported_primary_axes': [k for k in ('aquatic','amber','white_floral') if k not in artifact['profiles']],
        'grouped_retrieval_count': len(records), 'top1': sum(r['rank'] == 1 for r in records),
        'top5': sum(r['rank'] <= 5 for r in records), 'records': records,
        'comparison_scope': 'reference_construction_diagnostic_not_new_Atlas_holdout',
        'reference_sha256': hashlib.sha256((args.output/'model.json').read_bytes()).hexdigest()}
    (args.output/'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k,v in report.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
