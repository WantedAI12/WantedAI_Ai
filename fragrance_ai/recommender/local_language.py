"""Owned, lazy CPU-only llama.cpp process; loopback only, no model downloads."""
import hashlib
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.request import urlopen

from .compact_language import local_completion


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class LocalLanguageBackend:
    def __init__(self, settings, root):
        if not isinstance(settings, dict):
            raise ValueError('local language settings must be pinned artifacts')
        self._paths = {}
        for name in ('executable', 'model'):
            item = settings.get(name)
            if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                    or not isinstance(item.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
                raise ValueError('local language requires a path and SHA256: '+name)
            path = (Path(root)/item['path']).resolve(strict=True)
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError('local language artifact escapes workspace')
            self._paths[name] = (path, item['sha256'])
        self._lock = threading.Lock()
        self._process = None
        self._closed = False
        self._port = None

    def contract(self):
        return {'backend': 'local_cpu_llama_cpp', 'model': self._paths['model'][0].name,
                'model_sha256': self._paths['model'][1], 'lazy_start': True,
                'external_llm_api_calls': 0, 'owned_process': True}

    def _start(self):
        if self._closed:
            raise ValueError('local language backend is closed')
        if self._process is not None and self._process.poll() is None:
            return
        for path, digest in self._paths.values():
            if _hash(path) != digest:
                raise ValueError('local language artifact hash mismatch')
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            self._port = reservation.getsockname()[1]
        self._process = subprocess.Popen([
            str(self._paths['executable'][0]), '-m', str(self._paths['model'][0]),
            '--host', '127.0.0.1', '--port', str(self._port), '-c', '2048',
            '-t', '1', '-tb', '1', '-np', '1', '-ngl', '0', '--jinja'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        deadline = time.monotonic()+45
        try:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError('local language process exited before readiness')
                try:
                    with urlopen(f'http://127.0.0.1:{self._port}/health', timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
                time.sleep(.1)
            raise TimeoutError('local language startup timed out')
        except BaseException:
            self._stop()
            raise

    def __call__(self, message):
        with self._lock:
            self._start()
            return local_completion(message, port=self._port)

    def _stop(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def close(self):
        with self._lock:
            self._closed = True
            self._stop()


def configured_language():
    from .local_runtime import local_profile
    profile = local_profile()
    if not profile or profile['language'] is None:
        return None
    return LocalLanguageBackend(profile['language'], Path(profile['profile_path']).parent)
