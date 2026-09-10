"""Explicit local inference for a cloned V54-MX experiment; no API registration."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .atlas_profiles import AtlasProfilePredictor
from .physsim_mx import VERSION, DOMAIN, MODES, mx_features, predict_mx


class V54MXPredictor:
    def __init__(self, path, *, sha256, experimental=False):
        if not experimental:
            raise ValueError('V54-MX requires explicit research opt-in')
        self.path = Path(path).resolve(strict=True)
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('V54-MX model hash mismatch')
        self.artifact = json.loads(raw)
        a = self.artifact
        if (a.get('schema') != VERSION or a.get('input_domain') != DOMAIN
                or a.get('runtime_promotion_allowed') is not False
                or a.get('data_redistribution_authorized') is not False
                or a.get('release_calibrated') is not False
                or a.get('gas_to_human_perception_calibrated') is not False
                or a.get('selected_mode') not in MODES or set(a.get('models', {})) != set(MODES)
                or a.get('reference_blocks') != ['applicability_high','use_high','applicability_low','use_low']):
            raise ValueError('invalid V54-MX experiment contract')
        parent = (self.path.parent/a['parent']['path']).resolve(strict=True)
        if not parent.is_relative_to(self.path.parent):
            raise ValueError('V54 parent must remain inside the experiment directory')
        self.atlas = AtlasProfilePredictor(parent, sha256=a['parent']['sha256'], experimental=True)
        self.sha256 = sha256
        stat = self.path.stat()
        self._metadata = (stat.st_size, stat.st_mtime_ns)

    def predict(self, stocks, *, mode=None):
        """stocks: explicit smiles/dilution/carrier/relative_aliquot records.

        No numeric gas concentration, body-lotion matrix or natural material is
        coerced into the nominal stock assay. Unresolved graphs fail explicitly.
        """
        from rdkit import Chem
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._metadata:
            raise ValueError('V54-MX checkpoint changed; reload the experiment')
        mode = self.artifact['selected_mode'] if mode is None else mode
        if mode not in MODES or not isinstance(stocks, (list,tuple)) or not stocks:
            raise ValueError('nonempty explicit nominal-stock mixture required')
        graphs, dilutions, carriers, weights = [], [], [], []
        for stock in stocks:
            if (not isinstance(stock, dict) or set(stock)-{'smiles','dilution','carrier','relative_aliquot'}
                    or not {'smiles','dilution','carrier'} <= set(stock)):
                raise ValueError('nominal-stock fields required; gas/formulation units are not accepted')
            smiles = stock['smiles']
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles,str) and '.' not in smiles else None
            if molecule is None:
                raise ValueError('unsupported fragmented or unresolved molecular graph')
            graphs.append(Chem.MolToSmiles(molecule,isomericSmiles=True))
            dilutions.append(stock['dilution'])
            carriers.append(stock['carrier'])
            weights.append(stock.get('relative_aliquot',1.))
        unique = sorted(set(graphs))
        high, low = self.atlas.predict(unique, reference_level='high'), self.atlas.predict(unique, reference_level='low')
        features = np.concatenate([high['applicability'],high['use'],low['applicability'],low['use']],axis=1)
        lookup = dict(zip(unique,features))
        x = mx_features([lookup[g] for g in graphs], graphs, dilutions, carriers, weights,
                        mode=mode, reference=self.artifact['feature_reference'])
        prediction = predict_mx(self.artifact['models'][mode],x[None,:])[0]
        endpoints = self.artifact['endpoints']
        if len(prediction) != len(endpoints):
            raise ValueError('V54-MX endpoint width mismatch')
        return {'schema': VERSION, 'mode': mode, 'input_domain': DOMAIN,
                'predicted_rata': dict(zip(endpoints,prediction.tolist())),
                'unscored_source_endpoints': ['Ozone','Metallic'],
                'parent_v54_sha256': self.artifact['parent']['sha256'], 'model_sha256': self.sha256,
                'release_calibrated': False, 'human_recipe_similarity_measured': False,
                'runtime_promoted': False}
