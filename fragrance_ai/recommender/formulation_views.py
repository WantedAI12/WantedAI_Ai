"""Compatibility views over ONE FormulationCore instance, not model ensembles."""
from __future__ import annotations

from collections import OrderedDict
import threading
import numpy as np

from .formulation_core import VERSION


class QuantitativeView:
    def __init__(self, core):
        self.core = core
        self.path, self.sha256 = core.path, core.sha256
        self.endpoints = core.quantitative_endpoints
        self.fine, self.native = core.manifest['fine_features'], core.manifest['native_profiles']
        self.artifact_version = VERSION

    def assert_current(self):
        self.core.assert_current()

    def predict(self, graphs, *, reference_level='high'):
        output = self.core.molecular(graphs, reference_level)
        # Neural weights/forward use float32, but LP feasibility and normalized
        # physical mixture rows use float64. Rounding the reference sums in
        # float32 can change the active faces of a tight full-profile LP.
        return {k: np.asarray(output[k], np.float64) for k in ('applicability', 'use')}

    def predict_with_diagnostics(self, graphs, *, reference_level='high'):
        result = self.predict(graphs, reference_level=reference_level)
        rows = [{'source_native_present': g in self.native,
                 'source_annotation_present': g in self.fine['by_structure'],
                 'model_kind': VERSION, 'diagnostic_kind': 'source_availability_not_calibrated_uncertainty'} for g in graphs]
        return result, {k: rows for k in result}


class FineView:
    def __init__(self, core):
        self.core = core
        self.path, self.sha256 = core.path, core.sha256
        self.endpoints = core.fine_endpoints
        self.structures = core.manifest['structures']
        self.annotations = core.manifest['source_annotations']
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def assert_current(self):
        self.core.assert_current()

    def predict(self, graphs):
        return self.core.molecular(graphs)['fine']

    def materials(self, items):
        with self._lock:
            return self._materials(items)

    def _materials(self, items):
        from rdkit import Chem
        self.assert_current()
        graphs, evidence = [], []
        for item in items:
            supplied = item.structure_smiles
            binding = self.structures.get(item.ingredient_id)
            graph = supplied or (binding[0] if binding else None)
            if binding and binding[1] is not None and binding[1] != item.cas_number:
                raise ValueError('shared model material/CAS identity mismatch')
            parsed = Chem.MolFromSmiles(graph) if graph and '.' not in graph else None
            if parsed is not None and binding and supplied:
                bound = Chem.MolFromSmiles(binding[0])
                if bound is None or Chem.MolToSmiles(bound) != Chem.MolToSmiles(parsed):
                    raise ValueError('shared model material structure identity mismatch')
            graphs.append(Chem.MolToSmiles(parsed, isomericSmiles=True) if parsed else None)
        missing = sorted({g for g in graphs if g and g not in self._cache})
        for offset in range(0, len(missing), 128):
            batch = missing[offset:offset+128]
            for graph, value in zip(batch, self.predict(batch)):
                self._cache[graph] = value
        result = []
        for item, graph in zip(items, graphs):
            value = np.zeros(len(self.endpoints)) if graph is None else self._cache[graph].copy()
            supported = self.annotations.get(graph, [])
            value[supported] = 1.
            result.append(value)
            evidence.append({'ingredient_id': item.ingredient_id, 'source_positive_labels': len(supported),
                'status': 'missing_or_multicomponent_structure' if graph is None else
                    'source_support_plus_predicted_unlisted' if supported else 'structure_prediction_only'})
        while len(self._cache) > 8192:
            self._cache.popitem(last=False)
        self.assert_current()
        return np.asarray(result).reshape(len(items), len(self.endpoints)), evidence

    def contract(self):
        return {**self.core.contract(), 'model_sha256': self.sha256, 'predicted_concepts': len(self.endpoints),
                'cpu_only_runtime': True, 'label_kind': 'public_descriptor_annotations_not_measured_absence_or_intensity',
                'view': 'fine_annotations_of_single_shared_network'}


class TransportView:
    def __init__(self, core):
        self.core = core
        self.path, self.sha256, self.manifest = core.path, core.sha256, core.manifest

    def assert_current(self):
        self.core.assert_current()

    def kernel(self, raw):
        return self.core.kernel(raw)

    def stable_kernel(self, raw):
        from .unified_transport import UnifiedTransportModel
        # Reuses only the mathematical error-control operation, not any V60
        # weights, loader or trained predictor.
        return UnifiedTransportModel.stable_kernel(self, raw)

    def contract(self):
        return {**self.core.contract(), 'shared_transport_checkpoint': True,
                'inference_runtime': 'numpy_cpu', 'products': ['perfume', 'body_lotion', 'body_wash'],
                'label_kind': 'synthetic_transition_operators', 'trained_transition_blend': 1.,
                'view': 'transport_of_single_shared_network',
                'transport_teacher_checkpoint_sha256': self.manifest['teacher_bindings']['unified_product']['sha256'],
                'trajectory_constraints': ['mass_conservation', 'nonnegative', 'absorbing_cumulative_sinks'],
                'recipe_acceptance_score_modified': False}


def shared_views(core):
    from .formulation_core import _LOCK
    with _LOCK:
        if not hasattr(core, '_views'):
            core._views = (QuantitativeView(core), FineView(core), TransportView(core))
        return core._views
