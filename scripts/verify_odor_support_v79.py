"""Installed-wheel checks of recovered support and partial candidate contracts."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    meta = json.loads(a.preparation.read_text(encoding='utf8'))
    if not a.worker:
        a.output.mkdir(parents=True, exist_ok=False)
        installed = a.output/'installed'
        installed.mkdir()
        wheel = Path(meta['wheel'])
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != meta['wheel_sha256']:
            raise ValueError('source-bound wheel changed')
        with zipfile.ZipFile(wheel) as z:
            if any(not (installed/n).resolve().is_relative_to(installed.resolve()) for n in z.namelist()):
                raise ValueError('unsafe package path')
            z.extractall(installed)
        env = dict(os.environ, PERFUMERY_AI_LOCAL_PROFILE=meta['profile'], PERFUMERY_AI_ENV='research',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf8')
        subprocess.run([sys.executable, __file__, '--preparation', str(a.preparation.resolve()),
            '--output', str(a.output.resolve()), '--worker'], check=True, env=env)
        return
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(a.output/'installed'))
    import fragrance_ai
    assert Path(fragrance_ai.__file__).resolve().is_relative_to((a.output/'installed').resolve())
    from fastapi.testclient import TestClient
    from deploy.system_runtime_v76 import create_local_app
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.odor_space import configured_odor_space, target_report
    from fragrance_ai.recommender.lotion_reference_objective import load_configured_reference_bank
    from fragrance_ai.recommender.hierarchical_perfume import target_rows
    space, bank = configured_odor_space(), load_configured_reference_bank()
    parser = NaturalLanguageBriefParser(IngredientCatalog([]))
    results = []
    for key in space.value['resolution_extension']['required_reference_ids_before_deployment']:
        text = space.rows[key]['label_en']+' scent'
        try:
            brief = parser.parse(text)
            report = target_report(bank, brief, target_rows(brief))
            results.append({'id': key, 'text': text, 'status': report['status'],
                'wanted': brief.expression_targets, 'unsupported': report['unsupported'],
                'identity_preserved': key in brief.expression_targets})
        except ValueError as e:
            results.append({'id': key, 'text': text, 'status': 'clarification_required', 'error': str(e)})
    write(a.output/'all-134-parser.json', results)
    print(json.dumps({'all_134_parser': dict((s, sum(r['status']==s for r in results))
        for s in sorted({r['status'] for r in results}))}), flush=True)
    checks = []
    common = {'max_risk_tier': 2, 'enable_registry_trace_candidates': True,
        'target_similarity': 95, 'max_ingredients': 12}
    cases = [
        ('interpret_iris', '/v1/odor-expressions/interpret', {'text': 'iris scent'}),
        ('interpret_linalool', '/v1/odor-expressions/interpret', {'text': 'linalool scent'}),
        ('interpret_carvone', '/v1/odor-expressions/interpret', {'text': 'carvone scent'}),
        ('interpret_composition', '/v1/odor-expressions/interpret', {'text': 'green-fruity scent'}),
        ('interpret_partial', '/v1/odor-expressions/interpret', {'text': 'lemon and quasifloral scent'}),
        ('interpret_ambiguous', '/v1/odor-expressions/interpret', {'text': 'skin scent'}),
        ('interpret_absence', '/v1/odor-expressions/interpret', {'text': 'odorless scent'}),
        ('perfume_sparse', '/v1/formulas', {**common, 'brief': 'absinthe scent'}),
        ('perfume_alternatives', '/v1/formulas', {**common, 'brief': 'linalool scent'}),
        ('perfume_partial', '/v1/formulas', {**common, 'brief': 'lemon and quasifloral scent'}),
        ('lotion_partial', '/v1/applications/body-lotion/design',
            {'brief': 'lemon and quasifloral scent', 'max_risk_tier': 2, 'target_similarity': 95}),
        ('perfume_unknown_exclusion', '/v1/formulas', {**common, 'brief': 'lemon without quasifloral scent'})]
    with TestClient(create_local_app(), raise_server_exceptions=False) as client:
        for name, path, payload in cases:
            start = time.perf_counter()
            response = client.post(path, json=payload)
            try:
                value = response.json()
            except ValueError:
                value = {'text': response.text}
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            (a.output/(name+'.json.gz')).write_bytes(gzip.compress(raw, mtime=0))
            row = {'name': name, 'http': response.status_code, 'seconds': time.perf_counter()-start,
                'response_sha256': hashlib.sha256(raw).hexdigest(), 'recipe_count': len(value.get('recipe', [])),
                'candidate_count': len(value.get('closest_candidate', []))}
            checks.append(row)
            write(a.output/'progress.json', checks)
            print(json.dumps(row), flush=True)
            assert response.status_code == 200, (name, value)
            if name.startswith('interpret_'):
                if name == 'interpret_absence':
                    assert value['representation']['status'] == 'odor_absence_condition'
                    assert not value['representation']['positive_odor_profile_applicable']
                else:
                    target = value['representation']['hierarchical_target']
                    assert target['searchable']
                    assert target['status'] == ('partial_source_reference' if name in ('interpret_partial', 'interpret_ambiguous') else 'resolved_source_reference')
            if name in ('perfume_sparse', 'perfume_alternatives'):
                assert value['recipe'] or value['closest_candidate']
                assert value['full_profile_assessment']['reference_assessment']['intent']['searchable']
            if name == 'perfume_partial':
                assert not value['recipe'] and value['closest_candidate']
                assert value['calculated_profile_similarity'] is None and not value['full_profile_target_met']
                assert value['full_profile_assessment']['partial_profile_score'] is not None
            if name == 'lotion_partial':
                assert not value['recipe'] and value['closest_candidate']
                assert value['score'] is None and not value['profile_target_met']
                assert value['partial_profile_score'] is not None
            if name == 'perfume_unknown_exclusion':
                assert not value['recipe'] and not value['closest_candidate']
                assert not value['full_profile_target_met']
    write(a.output/'report.json', {'passed': True, 'checks': checks, 'parser_cases': len(results),
        'wheel_sha256': meta['wheel_sha256'], 'profile': meta['profile'], 'deployed': False})


if __name__ == '__main__':
    main()
