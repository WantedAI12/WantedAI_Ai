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
SYSTEM_VERSION = 'shared-formulation-core/v76'
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
    physical_context=context
    if 'blend_head.weight' in arrays:
        goal_free=context.copy()
        goal_free[:,25:63]=0.
        physical_context=np.where((context[:,12]==2)[:,None],goal_free,context)
    ctx = np.maximum(0., linear(physical_context, 'context_in'))
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
    heads=['fine', 'quantitative', 'transport', 'action', 'check', 'revision', 'emulsion']
    heads.extend(name for name in ('blend','aqueous') if name+'_head.weight' in arrays)
    result={name: linear(h, name + '_head') for name in heads}
    if 'blend_head.weight' in arrays:
        result['revision']=np.where((context[:,12]==2)[:,None],context[:,25:44]-context[:,44:63],result['revision'])
    return result


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


def transport_forward_arrays(arrays, context):
    """Exact zero-mass, empty-history transport subgraph of forward_arrays.

    kernel() supplies zero molecular masses. Its weighted mean, variance and
    attention are therefore exactly zero, regardless of molecular features.
    Reuse the trained context/shared layers and transport head, but do not
    calculate unused molecular, odor, procedure or revision outputs. Preserve
    the original fused shape, dtype promotion and operation order for parity.
    """
    def linear(x, key):
        out = x @ arrays[key + '.weight'].T
        bias = arrays.get(key + '.bias')
        return out if bias is None else out + bias

    c = np.asarray(context, np.float32)
    if c.ndim != 2 or c.shape[1] != 64 or not np.isfinite(c).all() or np.any(c[:, 12] != 1.):
        raise ValueError('transport-only context with finite coefficients required')
    ctx = np.maximum(0., linear(c, 'context_in'))
    molecular_width = arrays['molecule_out.weight'].shape[0]
    state_width = arrays['process_state.weight'].shape[0]
    offset = 3 * molecular_width
    # Zero values still carry dtype in the full path. Keep float64 or mixed
    # checkpoints from being silently rounded through a float32 placeholder.
    upstream = ('feature_mean', 'feature_scale', 'molecule_in.weight', 'molecule_in.bias',
                'molecule_out.weight', 'molecule_out.bias', 'attention_key.weight',
                'attention_key.bias', 'attention_query.weight', 'attention_query.bias')
    fused_dtype = np.result_type(np.float32, ctx.dtype, *(arrays[k].dtype for k in upstream if k in arrays))
    fused = np.zeros((len(c), offset + ctx.shape[1] + state_width), fused_dtype)
    fused[:, offset:offset + ctx.shape[1]] = ctx
    h = np.maximum(0., linear(np.maximum(0., linear(fused, 'fusion')), 'shared_in'))
    for i in range(2):
        prefix = f'shared.{i}'
        mean_h = h.mean(-1, keepdims=True)
        normalized = (h - mean_h) / np.sqrt(((h - mean_h)**2).mean(-1, keepdims=True) + 1e-5)
        normalized = normalized * arrays[prefix + '.norm.weight'] + arrays[prefix + '.norm.bias']
        h = h + linear(np.maximum(0., linear(normalized, prefix + '.up')), prefix + '.down')
    return linear(h, 'transport_head')


class FormulationCore:
    def __init__(self, path, sha256, *, allow_candidate=False):
        self.path = Path(path).resolve(strict=True)
        raw = self.path.read_bytes()
        if len(raw) > 24_000_000 or hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('formulation core manifest hash/size mismatch')
        self.manifest = m = json.loads(raw)
        from .autoregressive_neural import SCHEMA as AUTOREGRESSIVE_VERSION
        from .constrained_autoregressive import SCHEMA as CONSTRAINED_VERSION
        from .aligned_autoregressive import SCHEMA as ALIGNED_VERSION
        self.version = m.get('schema')
        if (self.version not in (VERSION, AUTOREGRESSIVE_VERSION, CONSTRAINED_VERSION, ALIGNED_VERSION, SYSTEM_VERSION) or m.get('training_executed') is not True
                or m.get('single_shared_checkpoint') is not True
                or (not allow_candidate and not m.get('accepted_for_local_inference'))):
            raise ValueError('untrained/unaccepted formulation checkpoint')
        gates = m.get('evaluation', {}).get('gates', {})
        required_gates = {'quantitative_fidelity','fine_fidelity','transport_fidelity',
                          'procedure_fidelity','mixture_proxy_fidelity','revision_deficit','cpu_export',
                          'measured_emulsion_vs_mean'}
        if self.version in (AUTOREGRESSIVE_VERSION, CONSTRAINED_VERSION, ALIGNED_VERSION, SYSTEM_VERSION):
            required_gates |= {'autoregressive_vs_initial','autoregressive_vs_fixed_optimizer',
                              'autoregressive_cpu_export','frozen_backbone_identity'}
            refit=m.get('profile_refit') if self.version==ALIGNED_VERSION else None
            if self.version==SYSTEM_VERSION:
                if m.get('joint_observed_refit',{}).get('schema')!='source_separated_joint_refit/v76':
                    raise ValueError('V76 joint data and physical-training provenance required')
                required_gates.discard('frozen_backbone_identity')
                required_gates |= {'observed_pair_holdout','measured_formulation_holdout','nonlinear_inverse_fidelity'}
                required_gates |= {'autoregressive_pass_count_retention','autoregressive_productwise_validation'}
            elif refit is not None:
                if (refit.get('schema')!='isolated-molecular-profile-refit/v75'
                        or refit.get('nonmolecular_parameter_arrays_exact') is not True
                        or refit.get('source_disjoint_profile_improvement') is not True):
                    raise ValueError('source-verified profile refit retention required')
                required_gates.discard('frozen_backbone_identity')
                required_gates |= {'profile_refit_verified','autoregressive_refit_revalidated'}
            elif m.get('autoregressive',{}).get('base_heads_frozen_and_byte_identical') is not True:
                raise ValueError('autoregressive parent retention evidence required')
        if self.version in (CONSTRAINED_VERSION, ALIGNED_VERSION, SYSTEM_VERSION):
            required_gates |= {'autoregressive_constraints', 'autoregressive_vs_v73'}
        if self.version==SYSTEM_VERSION:
            required_gates.discard('autoregressive_vs_v73')
            required_gates.add('autoregressive_vs_parent')
        if not allow_candidate and (not required_gates <= set(gates)
                                    or any(gates[k] is not True for k in required_gates)):
            raise ValueError('formulation training gates are missing or failed')
        if m.get('process_context_schema') != 'source_relative_context/v2':
            raise ValueError('formulation process feature version mismatch')
        process_graph = m.get('process_graph_training')
        if process_graph is not None:
            from . import formulation_process_graph as graph_module
            if (process_graph.get('schema') != graph_module.VERSION
                    or process_graph.get('source_graph_sha256') != hashlib.sha256(Path(graph_module.__file__).read_bytes()).hexdigest()):
                raise ValueError('process graph and trained checkpoint source mismatch')
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
        if self.version==SYSTEM_VERSION:
            if len(set(self.manifest.get('blend_labels',[])))!=109 or len(self.manifest.get('aqueous',{}).get('ingredient_names',[]))!=18:
                raise ValueError('observed V76 endpoint identity mismatch')
            dense('blend_head',256,109)
            dense('aqueous_head',256,18)
        for name, shape in [('feature_mean', (self.feature_width,)), ('feature_scale', (self.feature_width,)),
                            ('quantitative_scale', (292,)), ('transport_mean', (8,)), ('transport_scale', (8,))]:
            expected[name] = shape
        if self.version != VERSION:
            from .constrained_autoregressive import SCHEMA as CONSTRAINED_VERSION
            from .aligned_autoregressive import SCHEMA as ALIGNED_VERSION
            constrained = self.version in (CONSTRAINED_VERSION,ALIGNED_VERSION,SYSTEM_VERSION)
            hidden = self.manifest.get('autoregressive',{}).get('hidden_size',48)
            if (isinstance(hidden,bool) or hidden not in (48,96,128)
                    or hidden!=48 and self.version!=SYSTEM_VERSION):
                raise ValueError('invalid versioned autoregressive width')
            dense('autoregressive.chemistry', 256, 64)
            expected.update({'autoregressive.cell.weight_ih':(3*hidden,88 if constrained else 80),
                'autoregressive.cell.weight_hh':(3*hidden,hidden),
                'autoregressive.cell.bias_ih':(3*hidden,), 'autoregressive.cell.bias_hh':(3*hidden,)})
            dense('autoregressive.controls',hidden,3 if constrained else 2)
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

    def mixture(self, graphs, fractions, *, product='perfume', target=None, current=None, concentration_percent=1., minutes=0.):
        if product not in PRODUCTS or isinstance(concentration_percent,bool) or not np.isfinite(concentration_percent) or not 0<concentration_percent<=100:
            raise ValueError('supported product and positive concentration required')
        fractions = np.asarray(fractions, np.float32)
        if (fractions.shape != (len(graphs),) or not np.isfinite(fractions).all()
                or np.any(fractions < 0) or fractions.sum() <= 0):
            raise ValueError('a nonempty positive-mass mixture is required')
        x = self.features(graphs)
        context = np.zeros((1, 64), np.float32)
        context[0, PRODUCTS.index(product)] = 1.
        if self.version==SYSTEM_VERSION:
            if not np.isfinite(minutes) or not 0<=minutes<=480:
                raise ValueError('mixture time must be within trained 0 to 480 minute domain')
            context[0,12]=2.
            context[0,13]=minutes/480.
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
                'revision_direction': (None if self.version==SYSTEM_VERSION and (target is None or current is None)
                                       else output['revision'][0]),
                'scope': ('learned_nonlinear_physical_prior_and_exact_goal_deficit_not_measured_mixture_interaction'
                          if self.version==SYSTEM_VERSION else 'learned_additive_proxy_and_profile_deficit_not_measured_mixture_interaction')}

    def kernel(self, raw):
        from .unified_transport import features, validate_raw
        raw = validate_raw(raw)
        self.assert_current()
        if len(raw) * self.feature_width > 32_000_000:
            raise ValueError('formulation request exceeds memory work budget')
        context = np.zeros((len(raw), 64), np.float32)
        context[:, 4:12] = (features(raw) - self.arrays['transport_mean']) / self.arrays['transport_scale']
        context[:, 12] = 1.
        logits = transport_forward_arrays(self.arrays, context)
        if not np.isfinite(logits).all():
            raise ArithmeticError('nonfinite shared neural output')
        self.assert_current()
        return transport_distribution(logits, raw)

    def procedure(self, template, completed=(), *, values=None):
        from .formulation_process import process_context, process_history
        context = process_context(template, values or {})[None]
        ids, numbers = process_history(completed, self.actions)
        output = self.forward(np.zeros((1, 1, self.feature_width)), np.zeros((1, 1)), context,
                              ids[None], numbers[None])
        probabilities = softmax(output['action'])[0]
        index = int(np.argmax(probabilities))
        extra = {}
        policy = self.manifest.get('process_graph_training')
        if policy is not None:
            from .formulation_process_graph import VERSION as GRAPH_VERSION, process_state
            if policy.get('schema') != GRAPH_VERSION or policy.get('source_legal_holdout_passed') is not True:
                raise ValueError('verified source dependency graph training is required')
            actions = [item if isinstance(item, str) else item['action'] for item in completed]
            legal = process_state(template, actions, values or {})
            raw_action = self.actions[index]
            mask = np.array([action in legal['allowed'] for action in self.actions])
            index = int(np.argmax(np.where(mask, probabilities, -np.inf)))
            extra = {'unconstrained_next_action': raw_action,
                     'unconstrained_action_source_consistent': raw_action in legal['allowed'],
                     'source_action_mask_applied': True,
                     'allowed_next_actions': legal['allowed'],
                     'source_graph_version': GRAPH_VERSION,
                     'history_outside_training_length': len(completed)>policy.get('max_training_history_steps',0),
                     'conditional_source_probability': float(probabilities[index]/probabilities[mask].sum())}
        return {'next_action': self.actions[index], 'model_probability': float(probabilities[index]),
                'check_propensities': sigmoid(output['check'])[0].tolist(),
                'prediction_kind': 'learned_source_procedure_not_manufacturing_outcome',
                'checkpoint_sha256': self.sha256, **extra}

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

    def autoregressive_latents(self, items):
        """Exact known chemical encodings; unknown identity stays explicit."""
        with _LOCK:
            return self._autoregressive_latents(items)

    def _autoregressive_latents(self, items):
        from rdkit import Chem
        self.assert_current()
        cache = self._cache.setdefault('autoregressive_latents',{})
        graphs, missing = [], []
        for item in items:
            binding = self.manifest['structures'].get(item.ingredient_id)
            supplied = item.structure_smiles
            if binding and binding[1] is not None and binding[1] != item.cas_number:
                raise ValueError('autoregressive material/CAS mismatch')
            graph = supplied or (binding[0] if binding else None)
            molecule = Chem.MolFromSmiles(graph) if graph and '.' not in graph else None
            canonical = Chem.MolToSmiles(molecule,isomericSmiles=True) if molecule is not None else None
            if supplied and binding:
                original=Chem.MolFromSmiles(binding[0])
                if original is None or canonical != Chem.MolToSmiles(original,isomericSmiles=True):
                    raise ValueError('autoregressive material/structure mismatch')
            graphs.append(canonical)
            if canonical is None:
                missing.append(item.ingredient_id)
        pending=sorted({g for g in graphs if g is not None and g not in cache})
        for start in range(0,len(pending),128):
            batch=pending[start:start+128]
            x=self.features(batch)
            z=np.clip((x-self.arrays['feature_mean'])/self.arrays['feature_scale'],-12.,12.)
            h=np.maximum(0.,z@self.arrays['molecule_in.weight'].T+self.arrays['molecule_in.bias'])
            h=np.maximum(0.,h@self.arrays['molecule_out.weight'].T+self.arrays['molecule_out.bias'])
            for graph,value in zip(batch,h):
                value.setflags(write=False)
                cache[graph]=value
        result=np.array([cache[g] if g is not None else np.zeros(256,np.float32) for g in graphs])
        while len(cache)>8192:
            cache.pop(next(iter(cache)))
        self.assert_current()
        return result,missing

    def autoregressive_proposal(self, items, profiles, targets, responses, fractions, *, product, steps=None,
            minimum_fractions=None, maximum_fractions=None, prices=None, price_budget=None,
            time_weights=None, avoided=None, physics_properties=None, concentration_percent=None):
        from .autoregressive_neural import SCHEMA, refine_arrays
        from .constrained_autoregressive import SCHEMA as CONSTRAINED_VERSION, rollout
        from .aligned_autoregressive import SCHEMA as ALIGNED_VERSION, rollout as aligned_rollout
        if self.version not in (SCHEMA, CONSTRAINED_VERSION, ALIGNED_VERSION, SYSTEM_VERSION):
            return None
        if product not in ('perfume','body_lotion'):
            raise ValueError('autoregressive product has not been trained')
        p,q=(np.asarray(v,np.float32) for v in (profiles,targets))
        r,w=(np.asarray(v,np.float64) for v in (responses,fractions))
        n=len(items)
        if (p.ndim!=3 or p.shape[1]!=n or q.ndim!=3 or q.shape!=(p.shape[0],len(r),p.shape[2])
                or r.shape!=(q.shape[1],n) or w.shape!=(n,) or not n
                or any(not np.isfinite(v).all() or np.any(v<0) for v in (p,q,r,w))
                or w.sum()<=0 or np.any(r@w<=0)
                or not np.allclose(p.sum(-1),1.,atol=1e-5)
                or not np.allclose(q.sum(-1),1.,atol=1e-5)
                or p.size+q.size+r.size>16_000_000):
            raise ValueError('complete finite normalized autoregressive physical basis required')
        latent,missing=self.autoregressive_latents(items)
        steps=int(self.manifest['autoregressive']['default_steps']) if steps is None else steps
        product_index=np.array([0 if product=='perfume' else 1])
        if self.version in (CONSTRAINED_VERSION,ALIGNED_VERSION,SYSTEM_VERSION):
            if any(v is None for v in (minimum_fractions,maximum_fractions,prices,price_budget)):
                raise ValueError('V74 requires actual mass bounds and price budget')
            lower,upper,cost=map(lambda v:np.asarray(v,float),(minimum_fractions,maximum_fractions,prices))
            if (any(v.shape!=(n,) or not np.isfinite(v).all() or np.any(v<0) for v in (lower,upper,cost))
                    or not np.isfinite(price_budget) or price_budget<=0 or np.any(upper>1)):
                raise ValueError('finite physical V74 constraints required')
            options={}
            function=rollout
            if self.version in (ALIGNED_VERSION,SYSTEM_VERSION):
                function=aligned_rollout
                if self.version==SYSTEM_VERSION:
                    options['gradient_geometry']=self.manifest['autoregressive'].get('gradient_geometry')
                if time_weights is not None:
                    time_weights=np.asarray(time_weights,float)
                    if time_weights.shape!=(len(r),):
                        raise ValueError('time weights do not match physical scenarios')
                    options['time_weights']=time_weights[None]
                if avoided is not None:
                    avoided=np.asarray(avoided,float)
                    if avoided.shape!=q.shape or not np.isfinite(avoided).all() or np.any((avoided<0)|(avoided>1)):
                        raise ValueError('avoidance masks must match full odor coordinates')
                    options['avoided']=avoided[None]
                if self.version==SYSTEM_VERSION and product=='perfume':
                    if physics_properties is None or concentration_percent is None:
                        raise ValueError('V76 inverse perfume requires the same scoped physical properties and concentration as final evaluation')
                    from .nonlinear_inverse import NonlinearDoseObjective
                    options['physics']=NonlinearDoseObjective(items,physics_properties,concentration_percent,
                        draws=int(self.manifest['autoregressive'].get('physical_draws',16)))
            value,report=function(self.arrays,latent[None],p[None],q[None],r[None],
                (w/w.sum())[None],product_index,lower[None],upper[None],cost[None],np.array([price_budget]),steps=steps,**options)
        else:
            value,report=refine_arrays(self.arrays,latent[None],p[None],q[None],r[None],
                (w/w.sum())[None],product_index,steps=steps)
        self.assert_current()
        return value[0].astype(np.float64),{**report,'checkpoint_sha256':self.sha256,
            'trained_recurrent_decoder_used':True,'single_shared_checkpoint':True,
            'candidate_materials':n,'missing_molecular_identity':missing,
            'missing_identity_mode':'zero_missing_chemical_embedding_physical_features_retained_not_identity_imputation',
            'original_acceptance_gates_still_required':True}

    def contract(self):
        self.assert_current()
        return {'schema': self.version, 'checkpoint_sha256': self.sha256, 'single_shared_checkpoint': True,
                'training_executed': True, 'parameter_count': self.manifest['parameter_count'],
                'architecture': self.manifest['architecture'], 'runtime': 'numpy_cpu',
                'old_model_forward_calls': 0, 'external_api_calls': 0,
                'training_sources': self.manifest['training_sources'],
                'evaluation': self.manifest['evaluation'],
                'scientific_coverage':self.manifest.get('scientific_coverage'),
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
        expected_evidence=model.manifest.get('scientific_evidence_index_sha256')
        if expected_evidence is not None and (profile.get('physical_evidence') is None or profile['physical_evidence'][1]!=expected_evidence):
            raise ValueError('shared formulation model and physical evidence index mismatch')
        model.assert_current()
        return model
