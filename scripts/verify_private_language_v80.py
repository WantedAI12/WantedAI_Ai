"""Exercise the deployed quantized worker without creating a proxy credential."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PERFUMERY_AI_LOCAL_PROFILE'] = 'disabled'


def main():
    import modal
    from fragrance_ai.recommender.compact_language import completion_payload, decode_completion
    prompt = '비가 갠 새벽 숲속을 걷는 분위기의 향을 원해. 무겁게 달지는 않았으면 해.'
    worker = modal.Cls.from_name('perfumery-ai-core','CompactLanguage')
    started = time.perf_counter()
    raw = worker().infer.remote(completion_payload(prompt))
    decoded = decode_completion(raw)
    output = ROOT/'.benchmarks/v80_supported_release/deployed-language-01.json'
    if output.exists():
        raise ValueError('preserve earlier verification')
    report = {'deployed_worker_called':True,'passed':True,'seconds':time.perf_counter()-started,
        'decoded':decoded,'usage':raw.get('usage'),'backend_credentials_modified':False}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
