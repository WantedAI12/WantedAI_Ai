import numpy as np
import pytest
from dataclasses import replace

from tests.test_dose_refinement import materials
from fragrance_ai.recommender.nonlinear_inverse import NonlinearDoseObjective
from fragrance_ai.recommender.batched_inverse_values import numpy_values,torch_values


@pytest.mark.parametrize('dimensions,heads',[(19,1),(146,2)])
@pytest.mark.parametrize('count',[3,48])
def test_batched_trials_preserve_each_scalar_physical_score(dimensions,heads,count):
    rng=np.random.default_rng(39)
    source=materials()
    items=[replace(source[i%3],ingredient_id=f'item-{i}') for i in range(count)]
    physics=NonlinearDoseObjective(items,{},15.,draws=16)
    p=rng.dirichlet(np.ones(dimensions),size=(heads,count))
    q=rng.dirichlet(np.ones(dimensions),size=(heads,6))
    w=rng.dirichlet(np.ones(count),size=10)
    tw=np.array([0,.25,.25,.2,.18,.12])
    mask=np.zeros_like(q);mask[:,:,4]=1.
    scalar=np.array([physics(p[None],q[None],None,row[None],np.array([0]),time_weights=tw[None],avoided=mask[None],compute_gradient=False)[0][0] for row in w])
    batch=numpy_values(physics.model,p,q,w,tw,mask)
    np.testing.assert_allclose(batch,scalar,atol=3e-14,rtol=3e-14)
    import torch
    values=[torch.tensor(x,dtype=torch.float32) for x in (p,q,w,tw,mask)]
    result=torch_values(physics.torch('cpu'),*values)
    np.testing.assert_allclose(result,scalar,atol=2e-6,rtol=1e-5)
