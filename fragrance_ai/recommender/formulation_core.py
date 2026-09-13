"""Safe NumPy runtime for the shared formulation checkpoint (v69).

Predictive heads share one trained trunk/weight archive. No old neural or kernel
checkpoint is called by this runtime. Physical conservation remains an explicit
operator, not something a language model or a neural score is allowed to waive.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import threading
import zipfile

import numpy as np

VERSION = 'shared-formulation-core/v69'
PRODUCTS = ('perfume', 'body_lotion', 'body_wash', 'stock_assay')
_LOCK = threading.RLock()


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -40., 40.)))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    value = np.exp(x)
    return value / value.sum(axis=-1, keepdims=True)


def forward_arrays(arrays, molecules, masses, context, step_ids, step_values):
    """Pure forward operation; used for independent PyTorch export parity."""
    def linear(x, key):
        out = x @ arrays[key + '.weight'].T
        bias = arrays.get(key + '.bias')
        return out if bias is None else out + bias

    mol = np.maximum(0., linear(np.maximum(0., linear(molecules, 'molecule_in')), 'molecule_out'))
    weights = masses / np.maximum(masses.sum(-1, keepdims=True), 1e-30)
    mean = (weights[..., None] * mol).sum(1)
    variance = (weights[..., None] * (mol - mean[:, None])**2).sum(1)
    ctx = np.maximum(0., linear(context, 'context_in'))
    score = (linear(mol, 'attention_key') * linear(ctx, 'attention_query')[:, None]).sum(-1) / 8.
    score += np.log(np.maximum(masses, 1e-30))
    score = np.where(masses > 0, score, -1e30)
    aw = softmax(score) * (masses > 0)
    aw /= np.maximum(aw.sum(-1, keepdims=True), 1e-30)
    attended = (aw[..., None] * mol).sum(1)
    state = np.zeros((len(molecules), 96), np.float32)
    for j in range(step_ids.shape[1]):
        embedded = np.concatenate((arrays['step_embedding.weight'][step_ids[:, j]], step_values[:, j]), -1)
        updated = np.tanh(linear(embedded, 'process_input') + linear(state, 'process_state'))
        state = np.where((step_ids[:, j] != 0)[:, None], updated, state)
    fused = np.concatenate((mean, variance, attended, ctx, state), -1)
    h = np.maximum(0., linear(np.maximum(0., linear(fused, 'fusion')), 'shared_in'))
    for i in range(2):
        prefix = f'shared.{i}'
        mean_h = h.mean(-1, keepdims=True)
        normalized = (h - mean_h) / np.sqrt(((h - mean_h)**2).mean(-1, keepdims=True) + 1e-5)
        normalized = normalized * arrays[prefix + '.norm.weight'] + arrays[prefix + '.norm.bias']
        h = h + linear(np.maximum(0., linear(normalized, prefix + '.up')), prefix + '.down')
    return {name: linear(h, name + '_head') for name in
            ('fine', 'quantitative', 'transport', 'action', 'check', 'revision', 'emulsion')}


def transport_distribution(logits, raw):
    from .unified_transport import baseline_kernel, correction_gate, kernel_mask, validate_raw
    raw = validate_raw(raw)
    base = baseline_kernel(raw)
    values = np.log(np.maximum(base, 1e-30)) + logits.reshape(-1, 2, 5) * correction_gate(raw)[:, None, None]
    values[~kernel_mask(raw)] = -np.inf
    result = softmax(values)
    constant = correction_gate(raw) == 0
    result[constant] = base[constant]
    return result


class FormulationCore:
    def __init__(self, path, sha256, *, allow_candidate=False):
        self.path = Path(path).resolve(strict=True)
        raw = self.path.read_bytes()
        if len(raw) > 24_000_000 or hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('formulation core manifest hash/size mismatch')
        self.manifest = m = json.loads(raw)
        if (m.get('schema') != VERSION or m.get('training_executed') is not True
                or m.get('single_shared_checkpoint') is not True
                or (not allow_candidate and not m.get('accepted_for_local_inference'))):
            raise ValueError('untrained/unaccepted formulation checkpoint')
        gates = m.get('evaluation', {}).get('gates', {})
        required_gates = {'quantitative_fidelity','fine_fidelity','transport_fidelity',
                          'procedure_fidelity','mixture_proxy_fidelity','revision_deficit','cpu_export',
                          'measured_emulsion_vs_mean'}
        if not allow_candidate and (not required_gates <= set(gates)
                                    or any(gates[k] is not True for k in required_gates)):
            raise ValueError('formulation training gates are missing or failed')
        if m.get('process_context_schema') != 'source_relative_context/v2':
            raise ValueError('formulation process feature version mismatch')
        self.sha256 = sha256
        self.weights_path = (self.path.parent / m['weights']['path']).resolve(strict=True)
        if not self.weights_path.is_relative_to(self.path.parent):
            raise ValueError('formulation weights escape checkpoint directory')
        content = self.weights_path.read_bytes()
        if len(content) > 64_000_000 or hashlib.sha256(content).hexdigest() != m['weights']['sha256']:
            raise ValueError('formulation weights hash/size mismatch')
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 128_000_000:
                raise ValueError('formulation weight archive too large')
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            self.arrays = {name: archive[name].copy() for name in archive.files}
        if any(not np.isfinite(v).all() for v in self.arrays.values()):
            raise ValueError('nonfinite formulation weights')
        self.feature_width = int(m['architecture']['molecule_features'])
        self.actions = tuple(m['actions'])
        self.quantitative_endpoints = tuple(m['quantitative_endpoints'])
        self.fine_endpoints = tuple(m['fine_endpoints'])
        if self.feature_width != 1063 + len(m['fine_features']['vocabulary']):
            raise ValueError('formulation feature vocabulary mismatch')
        from .formulation_process import ACTIONS
        if (self.actions != ACTIONS or len(set(self.quantitative_endpoints)) != 146
                or len(set(self.fine_endpoints)) != 450):
            raise ValueError('formulation output identity mismatch')
        self._validate_shapes()
        for v in self.arrays.values():
            v.setflags(write=False)
        self._metadata = self._stat()
        self._cache = {}

    def _validate_shapes(self):
        a = self.arrays
        expected = {}
        def dense(name, source, dest, bias=True):
            expected[name + '.weight'] = (dest, source)
            if bias:
                expected[name + '.bias'] = (dest,)
        dense('molecule_in', self.feature_width, 384)
        dense('molecule_out', 384, 256)
        dense('context_in', 64, 128)
        dense('attention_key', 256, 64, False)
        dense('attention_query', 128, 64, False)
        expected['step_embedding.weight'] = (len(self.actions) + 1, 48)
        dense('process_input', 60, 96)
        dense('process_state', 96, 96, False)
        dense('fusion', 992, 384)
        dense('shared_in', 384, 256)
        for i in range(2):
            expected[f'shared.{i}.norm.weight'] = (256,)
            expected[f'shared.{i}.norm.bias'] = (256,)
            dense(f'shared.{i}.up', 256, 384)
            dense(f'shared.{i}.down', 384, 256)
        for name, width in [('fine', 450), ('quantitative', 292), ('transport', 10),
                            ('action', len(self.actions)), ('check', 8), ('revision', 19), ('emulsion', 101)]:
            dense(name + '_head', 256, width)
        for name, shape in [('feature_mean', (self.feature_width,)), ('feature_scale', (self.feature_width,)),
                            ('quantitative_scale', (292,)), ('transport_mean', (8,)), ('transport_scale', (8,))]:
            expected[name] = shape
        if set(a) != set(expected) or any(a[k].shape != v for k, v in expected.items()):
            raise ValueError('formulation weight names/dimensions mismatch')
        if any(np.any(a[k] <= 0) for k in ('feature_scale', 'quantitative_scale', 'transport_scale')):
            raise ValueError('invalid formulation normalization')

    def _stat(self):
        return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (self.path, self.weights_path))

    def assert_current(self):
        if self._stat() != self._metadata:
            raise ValueError('formulation core changed; reload the runtime')

    def features(self, graphs, level='high'):
        from ..research.atlas_profiles import atlas_features
        from rdkit import Chem
        canonical = []
        for graph in graphs:
            molecule = Chem.MolFromSmiles(graph) if isinstance(graph, str) and graph and '.' not in graph else None
            if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                raise ValueError('one explicit molecular graph required')
            canonical.append(Chem.MolToSmiles(molecule, isomericSmiles=True))
        return atlas_features(canonical, [level] * len(canonical), self.manifest['native_profiles'],
                              self.manifest['fine_features']).astype(np.float32)

    def forward(self, molecules, masses, context=None, step_ids=None, step_values=None):
        self.assert_current()
        x, w = np.asarray(molecules, np.float32), np.asarray(masses, np.float32)
        if x.ndim != 3 or x.shape[-1] != self.feature_width or w.shape != x.shape[:2] or x.shape[1] == 0:
            raise ValueError('expected batched molecule set and matching masses')
        b = len(x)
        c = np.zeros((b, 64), np.float32) if context is None else np.asarray(context, np.float32)
        ids = np.zeros((b, 0), np.int64) if step_ids is None else np.asarray(step_ids)
        values = np.zeros((*ids.shape, 12), np.float32) if step_values is None else np.asarray(step_values, np.float32)
        if (c.shape != (b, 64) or ids.ndim != 2 or ids.shape[0] != b
                or ids.dtype.kind not in 'iu' or values.shape != (*ids.shape, 12)
                or np.any(ids < 0) or np.any(ids > len(self.actions))):
            raise ValueError('invalid process history/context shape')
        if any(not np.isfinite(v).all() for v in (x, w, c, values)) or np.any(w < 0):
            raise ValueError('finite inputs and nonnegative component masses required')
        if ids.shape[1] > 512 or x.size > 32_000_000:
            raise ValueError('formulation request exceeds memory work budget')
        z = (x - self.arrays['feature_mean']) / self.arrays['feature_scale']
        # Training uses this explicit normalization range; it is not a physical
        # coefficient clamp. Raw values remain available to domain diagnostics.
        z = np.clip(z, -12., 12.)
        outputs = forward_arrays(self.arrays, z, w, c, ids.astype(np.int64), values)
        if any(not np.isfinite(v).all() for v in outputs.values()):
            raise ArithmeticError('nonfinite shared neural output')
        self.assert_current()
        return outputs

    def molecular(self, graphs, level='high'):
        if not len(graphs):
            return {'applicability': np.empty((0, 146)), 'use': np.empty((0, 146)), 'fine': np.empty((0, 450))}
        x = self.features(graphs, level)
        result = self.forward(x[:, None], np.ones((len(x), 1), np.float32))
        q = np.maximum(0., np.expm1(np.clip(result['quantitative'] * self.arrays['quantitative_scale'], -20, 20)))
        return {'applicability': q[:, :146], 'use': q[:, 146:], 'fine': sigmoid(result['fine'])}

    def mixture(self, graphs, fractions, *, product='perfume', target=None, current=None, concentration_percent=1.):
        if product not in PRODUCTS or not np.isfinite(concentration_percent) or concentration_percent <= 0:
            raise ValueError('supported product and positive concentration required')
        fractions = np.asarray(fractions, np.float32)
        if (fractions.shape != (len(graphs),) or not np.isfinite(fractions).all()
                or np.any(fractions < 0) or fractions.sum() <= 0):
            raise ValueError('a nonempty positive-mass mixture is required')
        x = self.features(graphs)
        context = np.zeros((1, 64), np.float32)
        context[0, PRODUCTS.index(product)] = 1.
        for offset, values in ((25, target), (44, current)):
            if values is not None:
                values = np.asarray(values, np.float32)
                if values.shape != (19,) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
                    raise ValueError('valid 19-axis target/current profile required')
                context[0, offset:offset+19] = values/values.sum()
        context[0, 63] = np.log10(concentration_percent/100.)
        output = self.forward(x[None], np.asarray(fractions, np.float32)[None], context)
        q = np.maximum(0., np.expm1(np.clip(output['quantitative'][0]*self.arrays['quantitative_scale'], -20, 20)))
        return {'applicability': q[:146], 'use': q[146:], 'fine': sigmoid(output['fine'][0]),
                'revision_direction': output['revision'][0],
                'scope': 'learned_additive_proxy_and_profile_deficit_not_measured_mixture_interaction'}

    def kernel(self, raw):
        from .unified_transport import features, validate_raw
        raw = validate_raw(raw)
        context = np.zeros((len(raw), 64), np.float32)
        context[:, 4:12] = (features(raw) - self.arrays['transport_mean']) / self.arrays['transport_scale']
        context[:, 12] = 1.
        outputs = self.forward(np.zeros((len(raw), 1, self.feature_width)), np.zeros((len(raw), 1)), context)
        return transport_distribution(outputs['transport'], raw)

    def procedure(self, template, completed=(), *, values=None):
        from .formulation_process import process_context, process_history
        context = process_context(template, values or {})[None]
        ids, numbers = process_history(completed, self.actions)
        output = self.forward(np.zeros((1, 1, self.feature_width)), np.zeros((1, 1)), context,
                              ids[None], numbers[None])
        probabilities = softmax(output['action'])[0]
        index = int(np.argmax(probabilities))
        return {'next_action': self.actions[index], 'model_probability': float(probabilities[index]),
                'check_propensities': sigmoid(output['check'])[0].tolist(),
                'prediction_kind': 'learned_source_procedure_not_manufacturing_outcome',
                'checkpoint_sha256': self.sha256}

    def emulsion(self, raw):
        from .emulsion_science import baseline, context, report, validate_raw
        specification = self.manifest.get('emulsion')
        if specification is None:
            raise ValueError('this checkpoint has no measured emulsification training')
        raw = validate_raw(raw)
        output = self.forward(np.zeros((len(raw), 1, self.feature_width)), np.zeros((len(raw), 1)),
                              context(raw, specification))
        p = softmax(baseline(raw, specification) + output['emulsion'])
        return {'checkpoint_sha256': self.sha256, 'results': report(raw, p, specification),
                'evaluation': self.manifest['evaluation']['test'].get('emulsion'),
                'application_scope': 'reference_geometry_steady_state_emulsions_only',
                'fragrance_release_coefficients_inferred': False}

    def contract(self):
        self.assert_current()
        return {'schema': VERSION, 'checkpoint_sha256': self.sha256, 'single_shared_checkpoint': True,
                'training_executed': True, 'parameter_count': self.manifest['parameter_count'],
                'architecture': self.manifest['architecture'], 'runtime': 'numpy_cpu',
                'old_model_forward_calls': 0, 'external_api_calls': 0,
                'training_sources': self.manifest['training_sources'],
                'evaluation': self.manifest['evaluation'],
                'complete_manufacturing_outcomes_learned': False,
                'human_similarity_percent': None}


@lru_cache(maxsize=2)
def _load(path, digest, size, mtime):
    return FormulationCore(path, digest)


def configured_formulation_core():
    from .local_runtime import local_profile
    profile = local_profile()
    pair = profile.get('formulation_core') if profile else None
    if pair is None:
        return None
    path, digest = pair
    stat = Path(path).stat()
    with _LOCK:
        model = _load(path, digest, stat.st_size, stat.st_mtime_ns)
        model.assert_current()
        return model
