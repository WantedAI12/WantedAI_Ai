import numpy as np
import pytest

from fragrance_ai.recommender.autoregressive_neural import residual_features,refine_arrays


@pytest.mark.parametrize('dimensions,heads,components',[(19,1,3),(146,2,9),(19,2,31)])
def test_torch_numpy_recurrent_export_and_physical_feature_parity(dimensions,heads,components):
    torch=pytest.importorskip('torch')
    from fragrance_ai.research.autoregressive_network import AutoregressiveDoseCell,residual_features as torch_features
    torch.set_num_threads(1)
    torch.manual_seed(73)
    rng=np.random.default_rng(730)
    n=components
    profiles=rng.uniform(.01,1,(2,heads,n,dimensions)).astype(np.float32)
    profiles/=profiles.sum(-1,keepdims=True)
    targets=rng.uniform(.01,1,(2,heads,3,dimensions)).astype(np.float32)
    targets/=targets.sum(-1,keepdims=True)
    responses=rng.uniform(.1,3,(2,3,n)).astype(np.float32)
    weights=rng.dirichlet(np.ones(n),size=2).astype(np.float32)
    previous=np.zeros_like(weights)
    product=np.array([0,1])
    mask=np.ones_like(weights,dtype=bool)
    latent=rng.normal(size=(2,n,256)).astype(np.float32)
    model=AutoregressiveDoseCell().eval()
    with torch.no_grad():
        model.controls.weight.normal_(0,.1)
    values=[profiles,targets,responses,weights,previous,product,0,mask]
    tf,tp=torch_features(*[torch.as_tensor(x) if isinstance(x,np.ndarray) else x for x in values])
    f,p=residual_features(*values)
    np.testing.assert_allclose(f,tf.numpy(),atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(p,tp.numpy(),atol=2e-6,rtol=2e-6)
    arrays={'autoregressive.'+k:v.detach().numpy() for k,v in model.state_dict().items()}
    actual,report=refine_arrays(arrays,latent,profiles,targets,responses,weights,product,steps=8,mask=mask)
    with torch.no_grad():
        expected,_=model(*[torch.as_tensor(x) for x in (latent,profiles,targets,responses,weights,product,mask)],steps=8)
    np.testing.assert_allclose(actual,expected.numpy(),atol=2e-5,rtol=2e-5)
    assert report['autoregressive_state_reused']
    np.testing.assert_allclose(actual.sum(-1),1.,atol=1e-6)
    assert np.all(actual>=0)


def test_feedback_gradient_matches_finite_difference_of_full_profile_loss():
    rng=np.random.default_rng(733)
    p=rng.uniform(.1,1,(1,2,4,19))
    p/=p.sum(-1,keepdims=True)
    q=rng.uniform(.1,1,(1,2,3,19))
    q/=q.sum(-1,keepdims=True)
    r=rng.uniform(.1,2,(1,3,4))
    w=np.full((1,4),.25)
    f,_=residual_features(p,q,r,w,np.zeros_like(w),np.array([1]),0,np.ones_like(w,bool))
    def loss(v):
        _,predicted=residual_features(p,q,r,v,np.zeros_like(v),np.array([1]),0,np.ones_like(v,bool))
        return .5*((predicted-q)**2).sum(-1).mean()
    numerical=[]
    for i in range(4):
        step=np.zeros_like(w)
        step[0,i]=1e-6
        numerical.append((loss(w+step)-loss(w-step))/2e-6)
    np.testing.assert_allclose(f[0,:,0]*f[0,:,13],numerical,atol=1e-9,rtol=1e-4)
