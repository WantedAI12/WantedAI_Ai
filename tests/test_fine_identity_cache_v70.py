from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from rdkit import Chem
from fragrance_ai.recommender.formulation_views import FineView
from tests.test_full_pool_search import material


def fixture():
    calls = []
    core = SimpleNamespace(path='fixture',sha256='fixture',fine_endpoints=('a','b','c'),
        manifest={'structures':{'a':['CCO','64-17-5']},'source_annotations':{'CCO':[1]}},
        assert_current=lambda:calls.append('guard'),
        molecular=lambda graphs:{'fine':np.tile([.1,.2,.3],(len(graphs),1))})
    return FineView(core), replace(material('a'),cas_number='64-17-5',structure_smiles='CCO'), calls


def test_repeated_material_identity_is_not_reparsed_and_outputs_remain_owned(monkeypatch):
    view, item, guards = fixture()
    real = Chem.MolFromSmiles
    calls = []
    def parse(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)
    monkeypatch.setattr(Chem,'MolFromSmiles',parse)
    expected, evidence = view.materials([item])
    count = len(calls)
    value, repeated_evidence = view.materials([item])
    assert count > 0 and len(calls) == count
    np.testing.assert_array_equal(value,expected)
    assert repeated_evidence == evidence and len(guards) == 4
    value[:] = 99
    np.testing.assert_array_equal(view.materials([item])[0],expected)


def test_cas_and_structure_changes_cannot_hit_old_identity_cache():
    view, item, _ = fixture()
    view.materials([item])
    with pytest.raises(ValueError,match='CAS'):
        view.materials([replace(item,cas_number='wrong')])
    with pytest.raises(ValueError,match='structure'):
        view.materials([replace(item,structure_smiles='CCN')])
