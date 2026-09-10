"""Real local learned-model and quantized-LLM/API integration; no cloud writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.perception_guidance import PerceptionGuidance
    from fragrance_ai.recommender.compact_language import local_completion
    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    path = ROOT/'benchmarks/core_language_v33_integration.json'
    if path.exists():
        raise ValueError('do not overwrite integration evidence')
    report = json.loads((ROOT/'.benchmarks/perception_core_v3/run-01/report.json').read_text(encoding='utf-8'))
    provider = PerceptionGuidance(ROOT/'.benchmarks/conditional_profiles_v2/final-01/models.json',
        ROOT/'benchmarks/industrial_ingredient_registry_v1.db', solvent='pg', experimental=True,
        component_model_path=ROOT/'.benchmarks/perception_core_v3/run-01/model.json', component_model_sha256=report['model_sha256'])
    start = time.perf_counter()
    with NaturalLanguagePerfumeryAI(perception_guidance=provider, require_full_profile_match=True) as ai:
        result = ai.create_recipe('floral fruity woody', RecipeConstraints(target_similarity=95., simulation_draws=64,
                                 physics_search_population=2)).to_dict()
    core = {'seconds': time.perf_counter()-start, 'requested_target': 95., 'accepted_recipe_count': len(result['recipe']),
            'calculated_profile_similarity': result.get('calculated_profile_similarity'),
            'full_profile_target_met': result.get('full_profile_target_met'), 'guidance': result['perception_guidance']}
    work = ROOT/'tmp/compact-llm-v33'
    log = (work/'integration-server.log').open('w', encoding='utf-8')
    proc = subprocess.Popen([str(work/'bin/llama-server.exe'), '-m', str(work/'Qwen3-0.6B-Q8_0.gguf'),
        '--host', '127.0.0.1', '--port', '18089', '-c', '2048', '-t', '1', '-tb', '1', '-np', '1', '-ngl', '0', '--jinja'],
        stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('local LLM startup failed')
            try:
                with urlopen('http://127.0.0.1:18089/health', timeout=1) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(.2)
        else:
            raise TimeoutError('local LLM startup deadline')
        class Formula(BaseModel):
            brief: str
        app = FastAPI()
        def no_generation(*args, **kwargs):
            raise AssertionError('assistant must not bypass recipe pipeline')
        register_ai_extensions(app, Formula, IngredientCatalog.load_builtin(), no_generation, lambda: None,
                               language_backend=local_completion)
        with TestClient(app) as client:
            response = client.post('/v1/ai/assistant', json={'message': '시트러스 향수를 원해요'})
        assert response.status_code == 200
        payload = response.json()
        assert payload['source'] == 'quantized_language_model'
        assert payload['intent_proposal']['desired'] == ['citrus']
        assert payload['requires_confirmation'] and not payload['formula_generated']
        evidence = {'scope': 'actual local learned component model and CPU LLM through ASGI; not Modal deployment',
                    'core': core, 'assistant': payload, 'modal_deployed': False}
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'core_guidance': core['guidance']['status'], 'strict_target_met': core['full_profile_target_met'],
                          'assistant': payload, 'report': str(path)}, ensure_ascii=False))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


if __name__ == '__main__':
    main()
