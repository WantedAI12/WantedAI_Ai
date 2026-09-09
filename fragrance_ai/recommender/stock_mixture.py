"""Hash-bound SDK for explicit stock-aliquot assays, never lotion transport."""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from collections import OrderedDict
import threading

import numpy as np

from .models import Ingredient,RecipeConstraints,ScentBrief
from .perception_runtime import assert_provider_current
from ..research.mixture_profiles import CROSS_FEATURE_VERSION,MODEL_KIND,mixture_features,predict_mixture_model


@dataclass(frozen=True)
class StockAliquot:
    ingredient: Ingredient
    stock_dilution: float
    relative_volume: float
    solvent: str


@dataclass(frozen=True)
class StockMass:
    ingredient: Ingredient
    stock_dilution: float
    supplied_mass_g: float
    stock_density_g_ml: float
    solvent: str


@dataclass(frozen=True)
class StockControlAliquot:
    """An exact frozen study control, not a purchasable molecular ingredient."""
    study_control_id: str
    stock_dilution: float
    relative_volume: float
    solvent: str


def _finite_number(value):
    return isinstance(value,Real) and not isinstance(value,(bool,np.bool_)) and math.isfinite(value)


class StockMixturePredictor:
    """Explicit local research loading. No paths or claims from an HTTP body."""
    def __init__(self, provider, model_path, *, sha256, experimental=False, atlas_predictor=None):
        if not experimental:
            raise ValueError('stock mixture model requires research opt-in')
        assert_provider_current(provider)
        self.path = Path(model_path).resolve(strict=True)
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('stock mixture checkpoint hash mismatch')
        artifact = json.loads(raw)
        from ..research.unified_mixture import KIND, predict_unified
        self.integrated = artifact.get('schema') == 'v54-integrated-candidate/v1'
        if self.integrated:
            if (artifact.get('local_development_accepted') is not True
                    or atlas_predictor is None or artifact.get('atlas_sha256') != atlas_predictor.sha256
                    or len(atlas_predictor.endpoints) != 146
                    or artifact.get('model',{}).get('kind') != KIND):
                raise ValueError('accepted V54 mixture and its exact molecular parent required')
            atlas_predictor.assert_current()
        elif (artifact.get('schema') != 'stock-mixture-candidate/v1'
                or artifact.get('model',{}).get('kind') != MODEL_KIND):
            raise ValueError('unsupported stock mixture model')
        if (artifact.get('component_sha256') != provider.component_model_sha256
                or artifact.get('endpoints') != list(provider.endpoints)
                or artifact.get('runtime_promotion_allowed') is not False
                or artifact.get('data_redistribution_authorized') is not False
                or artifact.get('lotion_headspace_calibrated') is not False
                or artifact.get('application_domain') != 'relative_aliquots_of_explicit_stock_conditions'):
            raise ValueError('stock mixture model or component provenance mismatch')
        self.provider,self.sha256,self.model = provider,sha256,artifact['model']
        self.component_sha256 = artifact['component_sha256']
        self.atlas = atlas_predictor if self.integrated else None
        self._atlas_sha256 = self.atlas.sha256 if self.atlas is not None else None
        self._atlas_cache, self._atlas_lock = OrderedDict(), threading.RLock()
        stat = self.path.stat()
        self._file_metadata = (stat.st_size, stat.st_mtime_ns)
        # Compile once. Forward calls neither reopen models nor query a registry.
        arrays = ('support','center','scale','coefficients','intercept')
        if self.integrated:
            arrays += ('atlas_support','atlas_center','atlas_scale')
        for key in arrays:
            value = np.asarray(self.model[key],float)
            value.setflags(write=False)
            self.model[key] = value
        if self.integrated:
            predict_unified(self.model,self.model['support'][:1],self.model['atlas_support'][:1])
        else:
            predict_mixture_model(self.model,self.model['center'][None,:])

    def assert_current(self):
        assert_provider_current(self.provider)
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._file_metadata:
            raise ValueError('stock mixture checkpoint changed; reload the local runtime')
        if self.atlas is not None:
            self.atlas.assert_current()
            if self.atlas.sha256 != self._atlas_sha256:
                raise ValueError('Atlas parent binding changed')
        if self.provider.component_model_sha256 != self.component_sha256:
            raise ValueError('stock mixture component checkpoint changed')

    def contract(self):
        self.assert_current()
        return {'model_kind':self.model['kind'], 'model_sha256':self.sha256,
                'component_model_sha256':self.component_sha256,
                'atlas_model_sha256':self.atlas.sha256 if self.atlas is not None else None,
                'integrated_v54':self.integrated, 'application_domain':'stock_aliquot_assay',
                'human_similarity_percent':None, 'recipe_acceptance_modified':False}

    def predict_molecular(self, graphs, *, reference_level='high'):
        """Preserve the original V54 applicability/use outputs without mixing.

        This SDK method is not a new unrestricted public graph HTTP endpoint.
        """
        self.assert_current()
        if self.atlas is None:
            raise ValueError('molecular prediction requires the integrated V54 model')
        return self.atlas.predict(graphs, reference_level=reference_level)

    def _molecular_vectors(self, graphs):
        supported = np.asarray(['.' not in graph and not graph.startswith('unresolved:') for graph in graphs], bool)
        wanted = list(dict.fromkeys(g for g,ok in zip(graphs,supported) if ok))
        forwards = 0
        with self._atlas_lock:
            missing = [g for g in wanted if g not in self._atlas_cache]
            for offset in range(0,len(missing),128):
                batch = missing[offset:offset+128]
                high = self.atlas.predict(batch,reference_level='high')
                low = self.atlas.predict(batch,reference_level='low')
                values = np.concatenate([high['applicability'],high['use'],low['applicability'],low['use']],axis=1)
                if values.shape != (len(batch),584) or not np.isfinite(values).all():
                    raise ValueError('invalid V54 molecular output')
                for graph,value in zip(batch,values):
                    value.setflags(write=False); self._atlas_cache[graph] = value
                forwards += 2
            vectors = np.asarray([self._atlas_cache[g] if ok else np.zeros(584) for g,ok in zip(graphs,supported)])
            for graph in wanted:
                self._atlas_cache.move_to_end(graph)
            while len(self._atlas_cache) > 4096:
                self._atlas_cache.popitem(last=False)
        return vectors,supported,forwards

    def predict_masses(self, stocks):
        """Convert supplied STOCK masses to aliquot volumes, never gas doses.

        Density must describe that diluted stock, not its neat odorant. Inputs
        are not claimed to be measured. No missing density is replaced by 1.
        """
        stocks = tuple(stocks)
        if not stocks or any(not isinstance(row,StockMass)
                or not _finite_number(row.supplied_mass_g) or row.supplied_mass_g < 0
                or not _finite_number(row.stock_density_g_ml) or not .01 <= row.stock_density_g_ml <= 30
                for row in stocks):
            raise ValueError('nonnegative stock masses and explicit stock densities required')
        largest = max(row.supplied_mass_g for row in stocks)
        if largest <= 0:
            raise ValueError('positive total stock mass required')
        mass = np.array([row.supplied_mass_g/largest for row in stocks])
        mass /= mass.sum()
        volumes = mass/np.array([row.stock_density_g_ml for row in stocks])
        result = self.predict([StockAliquot(row.ingredient,row.stock_dilution,float(volume),row.solvent)
                               for row,volume in zip(stocks,volumes)])
        normalized_volumes = volumes/volumes.sum()
        result['input_basis'] = 'supplied_stock_mass_with_explicit_stock_density'
        result['mass_balance'] = {'density_basis':'caller_supplied_stock_density_not_verified_neat_density',
            'density_measured_verified':False,'active_mass_fraction':float(sum(
                fraction*row.stock_dilution for row,fraction in zip(stocks,mass))),
            'components':[{'ingredient_id':row.ingredient.ingredient_id,
                'supplied_mass_fraction':float(fraction),'aliquot_volume_fraction':float(volume),
                'active_mass_fraction':float(fraction*row.stock_dilution),
                'stock_density_g_ml':float(row.stock_density_g_ml)}
                for row,fraction,volume in zip(stocks,mass,normalized_volumes)]}
        return result

    def predict(self, aliquots, *, application='stock_aliquot_assay', controls=()):
        from ..research.conditional_profiles import CARRIERS
        if application != 'stock_aliquot_assay':
            raise ValueError('stock aliquot weights are not lotion gas or finished-product mass fractions')
        self.assert_current()
        aliquots, controls = tuple(aliquots), tuple(controls)
        if not aliquots and not controls:
            raise ValueError('at least one explicit stock aliquot required')
        for row in controls:
            if (not isinstance(row,StockControlAliquot) or not isinstance(row.study_control_id,str)
                    or not row.study_control_id.startswith('-') or not row.study_control_id[1:].isdigit()
                    or int(row.study_control_id) >= 0
                    or row.solvent not in CARRIERS or row.solvent == 'other'
                    or not _finite_number(row.stock_dilution) or not 0 < row.stock_dilution <= 1
                    or not _finite_number(row.relative_volume) or row.relative_volume < 0):
                raise ValueError('explicit frozen control identity and stock condition required')
        for row in aliquots:
            if (not isinstance(row,StockAliquot)
                    or not isinstance(row.ingredient,Ingredient) or not isinstance(row.solvent,str)
                    or row.solvent not in CARRIERS or row.solvent == 'other'
                    or not _finite_number(row.stock_dilution) or not _finite_number(row.relative_volume)
                    or not 0 < row.stock_dilution <= 1 or row.relative_volume < 0):
                raise ValueError('finite dilution and nonnegative relative aliquot volume required')
        active = [row for row in aliquots if row.relative_volume > 0]
        active_controls = [row for row in controls if row.relative_volume > 0]
        if not active and not active_controls:
            raise ValueError('positive total aliquot volume required')
        # Deduplicate equivalent submitted rows before graph/model work.
        unique,positions,material_records = [],{},{}
        for row in active:
            # Ingredient contains mapping fields, so use its identity plus an
            # equality check; contradictory records are not silently merged.
            identity = (row.ingredient.ingredient_id,row.stock_dilution,row.solvent)
            identifier = row.ingredient.ingredient_id
            if identifier in material_records and material_records[identifier] != row.ingredient:
                raise ValueError('conflicting records for one ingredient')
            material_records[identifier] = row.ingredient
            if identity not in positions:
                positions[identity] = len(unique); unique.append(row)
            elif unique[positions[identity]].ingredient != row.ingredient:
                raise ValueError('conflicting records for one ingredient')
        brief = ScentBrief('',{},[],[],[],[],'medium',{},RecipeConstraints())
        session = self.provider.begin(brief)
        predictions,details,keys = [],[],[]
        for offset in range(0,len(unique),128):
            rows = unique[offset:offset+128]
            values,basis,identity = session.predict_stock_conditions([r.ingredient for r in rows],
                [r.stock_dilution for r in rows],[r.solvent for r in rows])
            predictions.extend(values); details.extend(basis); keys.extend(identity)
        indices = [positions[(r.ingredient.ingredient_id,r.stock_dilution,r.solvent)] for r in active]
        control_unique,control_positions = [],{}
        for row in active_controls:
            # Numeric comparison is only against a finite positive dilution;
            # controls may not interpolate/extrapolate or borrow another stock.
            match = [a for a in self.provider.model.get('anchors',())
                     if a['key'][0] == row.study_control_id and a['key'][2] == row.solvent
                     and float(a['key'][1]) == row.stock_dilution]
            if len(match) != 1:
                raise ValueError('control has no exact frozen measured stock condition')
            key = tuple(match[0]['key'])
            if key not in control_positions:
                value = np.asarray(match[0]['profile'],float)
                if value.shape != (len(self.provider.endpoints),) or not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError('invalid frozen control profile')
                control_positions[key] = len(predictions)
                control_unique.append(row)
                predictions.append(value); keys.append(key)
            indices.append(control_positions[key])
        active_rows = active+active_controls
        profiles = np.asarray(predictions)[indices]
        feature = mixture_features(profiles,[keys[i] for i in indices],[r.relative_volume for r in active_rows],
            cross_moments=not self.integrated and self.model['features'] == CROSS_FEATURE_VERSION)
        molecular_report = {}
        if self.integrated:
            from ..research.unified_mixture import atlas_mixture_features,predict_unified
            graphs = [row[1]['canonical_smiles'] for row in session.prepare_stock_molecules([r.ingredient for r in unique])]
            graphs += ['unresolved:'+row.study_control_id for row in control_unique]
            vectors,supported,forwards = self._molecular_vectors(graphs)
            atlas_feature = atlas_mixture_features(vectors[indices],[graphs[i] for i in indices],
                [r.stock_dilution for r in active_rows],[r.solvent for r in active_rows],
                [r.relative_volume for r in active_rows],supported=supported[indices])
            predicted = predict_unified(self.model,feature[None,:],atlas_feature[None,:])[0]
            molecular_report = {'integrated_v54':True,'atlas_model_sha256':self.atlas.sha256,
                'atlas_forward_batches':forwards,'atlas_supported_volume_fraction':float(atlas_feature[-1]),
                'atlas_weight':self.model['atlas_weight'],'output_transform':self.model['output_transform']}
        else:
            predicted = predict_mixture_model(self.model,feature[None,:])[0]
        if not np.isfinite(predicted).all():
            raise ValueError('nonfinite mixture prediction')
        return {'schema':'stock-mixture-prediction/v1','application_domain':application,
            'status':'research_stock_mixture_prediction','model_sha256':self.sha256,
            'model_kind':self.model['kind'],**molecular_report,
            'component_model_sha256':self.provider.component_model_sha256,
            'predicted_rata_profile':dict(zip(self.provider.endpoints,predicted.tolist())),
            'unique_input_stock_count':len(unique)+len(control_unique),'effective_stock_count':float(feature[-4]),
            'component_forward_batches':getattr(session,'stock_batch_forward_calls',0),
            'component_basis':[{'ingredient_id':r.ingredient.ingredient_id,'stock_dilution':r.stock_dilution,
                'solvent':r.solvent,**status} for r,status in zip(unique,details)],
            'study_control_basis':[{'study_control_id':r.study_control_id,'stock_dilution':r.stock_dilution,
                'solvent':r.solvent,'basis':'exact_frozen_measured_solvent_control'} for r in control_unique],
            'general_unequal_ratio_validated':False,'assay_conditions_verified':False,
            'lotion_headspace_calibrated':False,'human_similarity_percent':None,
            'manufacturing_approved':False,'recipe_acceptance_modified':False}
