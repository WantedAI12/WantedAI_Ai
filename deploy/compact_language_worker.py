"""Optional private CPU worker in the existing Modal app; never auto-deploys."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'tmp/compact-llm-v33/Qwen3-0.6B-Q8_0.gguf'
MODEL_SHA256 = '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031'
LLAMA_COMMIT = '9dcf84e5ae2718947188b539aab8b9c2b15d3ba1'


def _worker_image():
    if modal.is_local():
        with MODEL.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != MODEL_SHA256:
                raise ValueError('quantized model hash mismatch')
    image = (modal.Image.debian_slim(python_version='3.11')
             .apt_install('git', 'cmake', 'g++', 'libcurl4-openssl-dev')
             .run_commands('git clone --depth 1 --branch b10853 https://github.com/ggml-org/llama.cpp.git /opt/llama',
                           f'test "$(git -C /opt/llama rev-parse HEAD)" = "{LLAMA_COMMIT}"',
                           'cmake -S /opt/llama -B /opt/llama/build -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF',
                           'cmake --build /opt/llama/build --config Release -j 2 --target llama-server')
             .add_local_file(MODEL, '/opt/model.gguf', copy=True)
             .add_local_python_source('deploy'))

    return image


language_app = modal.App()


@language_app.cls(image=_worker_image(), cpu=1., memory=2048, min_containers=0, max_containers=1,
                  scaledown_window=60, timeout=180)
@modal.concurrent(max_inputs=1)
class CompactLanguage:
    @modal.enter()
    def load(self):
        with open('/opt/model.gguf', 'rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != MODEL_SHA256:
                raise ValueError('runtime model hash mismatch')
        self.process = subprocess.Popen(['/opt/llama/build/bin/llama-server', '-m', '/opt/model.gguf',
            '--host', '127.0.0.1', '--port', '18089', '-c', '2048', '-t', '1', '-tb', '1', '-np', '1', '-ngl', '0', '--jinja'])
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('CPU language worker failed to start')
            try:
                with urlopen('http://127.0.0.1:18089/health', timeout=1) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(.2)
        self.process.terminate()
        raise TimeoutError('CPU language worker startup deadline')

    @modal.method()
    def infer(self, payload):
        if not isinstance(payload, dict) or len(json.dumps(payload)) > 24000:
            raise ValueError('bounded language request required')
        payload = {**payload, 'max_tokens': 160, 'stream': False}
        request = Request('http://127.0.0.1:18089/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        # A one-core cold prefix evaluation can exceed the old 40-second
        # deadline even before decoding. Keep output bounded at 160 tokens.
        with urlopen(request, timeout=150) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('oversized language result')
        return json.loads(raw)

    @modal.exit()
    def stop(self):
        if getattr(self, 'process', None) is not None:
            self.process.terminate()



def register_worker(app):
    app.include(language_app)
    return CompactLanguage


def backend_for(worker):
    """Use only behind the existing authenticated web app, not a public LLM URL."""
    from fragrance_ai.recommender.compact_language import completion_payload, decode_completion
    def respond(message):
        return decode_completion(worker().infer.remote(completion_payload(message)))
    respond.contract = lambda: {'backend':'private_modal_cpu_llama_cpp','model':'Qwen3-0.6B-Q8_0',
        'model_sha256':MODEL_SHA256,'cpu_only':True,'on_demand':True,
        'third_party_llm_api_calls':0,'formula_generation_by_llm':False}
    return respond
