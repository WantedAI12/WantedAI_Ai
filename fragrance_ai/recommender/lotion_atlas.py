"""Atlas quantitative reference profiles in the real lotion recipe search.

This is an explicit research provider, not a conversion from ordinal source
conditions to lotion doses. Descriptor applicability and use retain separate
heads; full endpoint mass (including off-notes) survives normalization.
"""
from collections import OrderedDict
from copy import deepcopy
from types import MappingProxyType
import threading

import numpy as np

from .perception_runtime import assert_provider_product

VERSION = 'lotion-atlas-reference-shape/v1'
# A declared semantic routing, not weights fitted to the 400 recipe outcomes.
# Each source column has one destination at most. Ambiguous/unmapped columns
# remain in the full 146-column shape and count against target agreement.
ATLAS_PROJECTION = {
    'citrus': ('FRUITY,CITRUS', 'LEMON', 'GRAPEFRUIT', 'ORANGE'),
    'clean': ('SOAPY',),
    'green': ('FRESH GREEN VEGETABLES', 'CRUSHED WEEDS', 'CRUSHED GRASS',
              'HERBAL, GREEN,CUTGRASS', 'RAW CUCUMBER'),
    'floral': ('FLORAL', 'VIOLETS'),
    'rose': ('ROSE',),
    'fruity': ('FRUITY,OTHER THAN CITRUS', 'PINEAPPLE', 'GRAPE JUICE', 'STRAWBERRY',
               'APPLE, FRUIT', 'PEAR', 'CANTALOUPE, HONEYDEW MELON', 'PEACH FRUIT',
               'BANANA', 'CHERRY, BERRY'),
    'spicy': ('SPICY', 'CLOVE', 'CINNAMON', 'BLACK PEPPER'),
    'aromatic': ('AROMATIC', 'LAVENDER', 'LAUREL LEAVES', 'TEA LEAVES', 'DILL',
                 'CARAWAY', 'MINTY, PEPPERMINT', 'CAMPHOR', 'EUCALYPTUS', 'ANISE, LICORICE'),
    'woody': ('OAK WOOD,COGNAC', 'WOODY, RESINOUS', 'CEDARWOOD'),
    'musky': ('MUSK',),
    'gourmand': ('HONEY', 'CHOCOLATE', 'VANILLA', 'SWEET', 'MAPLE SYRUP',
                 'CARAMEL', 'MALTY', 'RAISINS', 'MOLASSES', 'COCONUT'),
    'powdery': ('DRY, POWDERY',),
    'smoky': ('BURNT,SMOKY', 'FRESH TOBACCO SMOKE', 'INCENSE', 'STALE TOBACCO SMOKE'),
    'earthy': ('MUSTY, EARTHY, MOLDY',),
    'leathery': ('LEATHER',),
}
REFERENCES = (
    {'measurement': 'applicability', 'source_ordinal_level': 'high'},
    {'measurement': 'use', 'source_ordinal_level': 'high'},
)


class AtlasReferenceGuidance:
    """Shared frozen odor-reference backbone with verified graph bindings.

    ``structures`` uses the existing registry mapping: material id ->
    (canonical molecular graph, expected CAS or None). No name guessing.
    """
    projection = ATLAS_PROJECTION

    def __init__(self, model, structures, *, experimental=False):
        if not experimental:
            raise ValueError('Atlas lotion guidance requires explicit research opt-in')
        columns = [name for names in ATLAS_PROJECTION.values() for name in names]
        if len(columns) != len(set(columns)) or not set(columns) <= set(model.endpoints):
            raise ValueError('Atlas lotion descriptor contract mismatch')
        self.model, self.endpoints = model, tuple(model.endpoints)
        self.structures = MappingProxyType(dict(structures))
        self.runtime_product = 'shared_odor_reference'
        self.component_model_sha256 = model.sha256
        self.component_model_version = getattr(model, 'artifact_version', VERSION)
        self.model_scope = 'quantitative_Atlas_reference_shapes_not_product_sensory_calibration'
        self.solvent, self.weight = None, 0.
        self.fine_odor_features = model.fine
        self.lotion_shape_cache = OrderedDict()
        self.lotion_domain_cache = {}
        self.lotion_shape_cache_lock = threading.RLock()

    def assert_current(self):
        check = getattr(self.model, 'assert_current', None)
        if check is not None:
            check()
        component = getattr(self, 'runtime_component_provider', None)
        if component is not None:
            from .perception_runtime import assert_provider_current
            from .local_runtime import local_profile
            assert_provider_current(component)
            profile = local_profile()
            if profile is None or profile['profile_sha256'] != self.runtime_local_profile_sha256:
                raise ValueError('local Atlas selection changed; reload API and workers')

    def begin_reference_shapes(self):
        self.assert_current()
        return AtlasLotionShapes(self)


class AtlasLotionGuidance(AtlasReferenceGuidance):
    """Backward-compatible, separately bound lotion use of the shared backbone."""
    def __init__(self, model, structures, *, experimental=False):
        super().__init__(model, structures, experimental=experimental)
        self.runtime_product = 'body_lotion'
        self.model_scope = 'quantitative_Atlas_reference_shapes_not_lotion_sensory_calibration'

    def begin_lotion_shapes(self):
        assert_provider_product(self, 'body_lotion')
        return self.begin_reference_shapes()


class AtlasLotionShapes:
    """Batched, request-local views over bounded graph-level model caches."""
    projection = ATLAS_PROJECTION
    reference_scenarios = REFERENCES
    reference_count = len(REFERENCES)
    nominal_reference_index = 0
    bridge_version = VERSION

    def __init__(self, provider):
        self.provider = provider
        self.shapes, self.basis, self.graphs, self.missing = {}, {}, {}, set()
        self.calls = self.forward_batches = self.shared_cache_hits = 0

    def _graph(self, item):
        from rdkit import Chem
        if item.ingredient_id in self.graphs:
            return self.graphs[item.ingredient_id]
        row = self.provider.structures.get(item.ingredient_id)
        graph = None
        if (row and isinstance(row[0], str) and '.' not in row[0]
                and (row[1] is None or row[1] == item.cas_number)):
            molecule = Chem.MolFromSmiles(row[0])
            if molecule is not None:
                graph = Chem.MolToSmiles(molecule, isomericSmiles=True)
                supplied = item.structure_smiles
                if supplied:
                    parsed = Chem.MolFromSmiles(supplied)
                    if parsed is None or Chem.MolToSmiles(parsed, isomericSmiles=True) != graph:
                        graph = None
        self.graphs[item.ingredient_id] = graph
        return graph

    def _store(self, item, shape, diagnostics=None):
        self.shapes[item.ingredient_id] = shape
        if shape is None:
            self.missing.add(item.ingredient_id)
            return
        self.basis[item.ingredient_id] = {
            'ingredient_id': item.ingredient_id, 'graph_identity': self.graphs[item.ingredient_id],
            'profile_basis': 'model_predicted_ordinal_Atlas_reference_not_measured_lotion',
            'reference_predictions': [dict(row) for row in REFERENCES],
            'numeric_stock_dilution_known': False, 'intensity_model': False,
            'catalog_profile_feature_available': self.graphs[item.ingredient_id] in self.provider.model.native,
            'model_applicability_diagnostics': deepcopy(diagnostics),
        }

    def prefetch(self, items):
        pending = {}
        for item in items:
            if item.ingredient_id in self.shapes:
                continue
            graph = self._graph(item)
            if graph is None:
                self._store(item, None)
                continue
            with self.provider.lotion_shape_cache_lock:
                cache = self.provider.lotion_shape_cache
                if graph in cache:
                    self._store(item, cache[graph], self.provider.lotion_domain_cache.get(graph))
                    cache.move_to_end(graph)
                    self.shared_cache_hits += 1
                    continue
            pending.setdefault(graph, []).append(item)
        graphs = list(pending)
        for offset in range(0, len(graphs), 128):
            batch = graphs[offset:offset+128]
            diagnostic_forward = getattr(self.provider.model, 'predict_with_diagnostics', None)
            if diagnostic_forward is None:
                outputs, diagnostics = self.provider.model.predict(batch, reference_level='high'), None
            else:
                outputs, diagnostics = diagnostic_forward(batch, reference_level='high')
            if set(outputs) != {'applicability', 'use'}:
                raise ValueError('distinct Atlas reference predictions required')
            raw = np.stack([outputs[row['measurement']] for row in REFERENCES])
            if (raw.shape != (self.reference_count, len(batch), len(self.provider.endpoints))
                    or not np.isfinite(raw).all() or np.any(raw < 0)):
                raise ValueError('invalid Atlas reference profile prediction')
            self.calls += len(batch)
            self.forward_batches += 1
            for i, graph in enumerate(batch):
                domain = {head: rows[i] for head, rows in diagnostics.items()} if diagnostics is not None else None
                totals = raw[:, i].sum(axis=1)
                shape = None if np.any(totals <= 0) else raw[:, i]/totals[:, None]
                if shape is not None:
                    shape.setflags(write=False)
                for item in pending[graph]:
                    self._store(item, shape, domain)
                with self.provider.lotion_shape_cache_lock:
                    self.provider.lotion_shape_cache[graph] = shape
                    self.provider.lotion_domain_cache[graph] = domain
                    self.provider.lotion_shape_cache.move_to_end(graph)
                    while len(self.provider.lotion_shape_cache) > 3840:
                        expired, _ = self.provider.lotion_shape_cache.popitem(last=False)
                        self.provider.lotion_domain_cache.pop(expired, None)

    def shape(self, item):
        self.prefetch([item])
        return self.shapes[item.ingredient_id]
