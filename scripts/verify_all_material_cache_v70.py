"""Exact before/after fine-head equivalence over the complete active catalog."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PERFUMERY_AI_LOCAL_PROFILE'] = 'disabled'


def main():
    from fragrance_ai.recommender.registry_activation import load_runtime_catalog
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.recommender.formulation_views import FineView
    old_file = ROOT/'.benchmarks/v70_repair/full400-paired/baseline-package/fragrance_ai/recommender/formulation_views.py'
    with zipfile.ZipFile(ROOT/'dist/shared-formulation-v69/build-04/perfumery_ai_core-1.4.0-py3-none-any.whl') as archive:
        assert archive.read('fragrance_ai/recommender/formulation_views.py') == old_file.read_bytes()
    spec = importlib.util.spec_from_file_location('fragrance_ai.recommender._old_fine_v69',old_file)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    manifest_path = ROOT/'dist/shared-formulation-v69/build-04/catalog/catalog_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    binding = manifest['runtime_catalog']
    catalog,_,_ = load_runtime_catalog(manifest_path.parent/binding['path'], expected_sha256=binding['sha256'],
        expected_wheel_sha256=binding['wheel_sha256'],expected_registry_sha256=binding['registry_sha256'])
    items = [item for item in catalog.ingredients if item.formulation_ready and not item.blocked]
    core = FormulationCore(ROOT/'.benchmarks/formulation_core_v69/train-04/model.json',
        '3186e467446abc71f249c052c8248c17c97d0e337976abf3666628af47dd997f')
    old, new = old_module.FineView(core), FineView(core)
    expected, evidence = old.materials(items)
    actual, new_evidence = new.materials(items)
    assert np.array_equal(expected,actual) and evidence == new_evidence
    times = {}
    for label, model in [('old_warm',old),('new_warm',new)]:
        started = time.perf_counter()
        value, metadata = model.materials(items)
        times[label] = time.perf_counter()-started
        assert np.array_equal(expected,value) and evidence == metadata
    report = {'active_materials':len(items),'endpoints':actual.shape[1],
              'all_predictions_exactly_equal':True,'all_provenance_rows_exactly_equal':True,
              'timings_seconds':times, 'timing_scope':'fine_material_view_only_not_total_request_latency',
              'current_source_sha256':hashlib.sha256((ROOT/'fragrance_ai/recommender/formulation_views.py').read_bytes()).hexdigest()}
    output = ROOT/'.benchmarks/v70_repair/all-material-cache-equivalence.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
