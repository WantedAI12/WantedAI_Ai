"""Freeze the CURRENT source-bound lotion coefficient pool before training.

The output is engineering input data, never measured lotion observations.
No model, catalog, safety policy or runtime selection is changed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import RuntimeAIFactory, _material_digest
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    profile = local_profile()
    if profile is None or profile['lotion_reference'] != 'atlas':
        raise ValueError('current pinned V54 Atlas lotion runtime is required')
    parents = {}
    for role in ('perfume', 'body_lotion', 'atlas', 'stock_mixture'):
        path, expected = profile[role]
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('parent checkpoint mismatch: ' + role)
        parents[role] = {'path': path, 'sha256': actual}
    factory = RuntimeAIFactory.from_environment()
    request = LotionEstimateRequest(brief='citrus woody body lotion',
        registry_pool='conditional_research', max_risk_tier=2,
        max_formula_cost_per_kg=1e6, max_ingredient_price_per_kg=1e6,
        min_availability=0.)
    inputs, scenarios, metadata = build_estimated_lotion_inputs(request, factory.catalog)
    lookup = {item.ingredient_id: item for item in factory.catalog.ingredients}
    materials = []
    for row in inputs.simulation.materials:
        item = lookup[row.ingredient_id]
        smiles = item.structure_smiles
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is not None:
            identity = 'smiles:' + Chem.MolToSmiles(molecule, isomericSmiles=True)
        elif item.cas_number:
            identity = 'cas:' + item.cas_number
        else:
            identity = 'unresolved:' + item.ingredient_id
        materials.append({'ingredient_id': item.ingredient_id,
            'identity_group': hashlib.sha256(identity.encode()).hexdigest(),
            'identity_basis': identity.split(':', 1)[0], 'structure_smiles': smiles,
            'profile': item.profile, 'parameters': row.model_dump(mode='json')})
    factory.assert_current_snapshot()
    value = {'schema': 'lotion-synthetic-training-inputs/v1',
        'training_executed': False, 'label_kind': 'engineering_parameters_not_observations',
        'measured_lotion_observations': 0, 'parent_models': parents,
        'catalog_sha256': profile['catalog'][1], 'material_snapshot_sha256': _material_digest(factory.catalog),
        'catalog_count': len(factory.catalog.ingredients), 'coefficient_pool_count': len(materials),
        'identity_group_count': len({x['identity_group'] for x in materials}),
        'current_model_and_material_pool_not_substituted': True,
        'reference_request': inputs.simulation.model_dump(mode='json'),
        'coefficient_provenance': metadata, 'materials': materials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'catalog_count': value['catalog_count'], 'coefficient_pool_count': len(materials),
        'identity_groups': value['identity_group_count']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
