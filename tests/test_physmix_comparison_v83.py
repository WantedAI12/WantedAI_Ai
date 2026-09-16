import pytest


def network():
    torch = pytest.importorskip('torch')
    from fragrance_ai.research.formulation_network import FormulationNetwork
    torch.manual_seed(83)
    model = FormulationNetwork(12,450,292,4).eval()
    return model,{k:v.detach().numpy().copy() for k,v in model.state_dict().items()}


def test_frozen_aggregation_matches_the_current_shared_forward_exactly():
    import torch
    from fragrance_ai.research.physmix_comparison import FrozenAggregation
    model,arrays = network()
    features = torch.randn(2,7,12)
    masses = torch.rand(2,7)
    context = torch.zeros(2,64)
    with torch.no_grad():
        expected = model(features,masses,context,torch.zeros(2,0,dtype=torch.long),torch.zeros(2,0,12))
        embedded = torch.relu(model.molecule_out(torch.relu(model.molecule_in(features))))
        latent,_,_ = FrozenAggregation(arrays)(embedded,masses,context)
        actual = model.quantitative_head(latent)
    torch.testing.assert_close(actual,expected['quantitative'],rtol=1e-6,atol=1e-6)


def test_all_arms_preserve_the_same_initial_predictor_and_pair_symmetry():
    import torch
    from fragrance_ai.research.physmix_comparison import PairComparison,parameter_counts
    _,arrays = network()
    torch.manual_seed(99)
    reference = PairComparison(arrays,'current').eval()
    head = reference.head.state_dict()
    a,b = torch.randn(2,6,256),torch.randn(2,6,256)
    wa,wb = torch.rand(2,6),torch.rand(2,6)
    context = torch.zeros(2,64)
    with torch.no_grad():
        expected = reference(a,wa,b,wb,context)
        counts = {}
        for mode in PairComparison.MODES:
            model = PairComparison(arrays,mode).eval()
            model.head.load_state_dict(head)
            value = model(a,wa,b,wb,context)
            torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(value,model(b,wb,a,wa,context),rtol=1e-6,atol=1e-6)
            counts[mode] = parameter_counts(model)['interaction']
    assert abs(counts['physmix']-counts['capacity_control'])/counts['physmix']<.01


def test_latent_dynamics_is_order_zero_dose_and_row_split_invariant():
    import torch
    from fragrance_ai.research.physmix_comparison import DoseLatentDynamics
    torch.manual_seed(8301)
    model = DoseLatentDynamics(16,4).double()
    x = torch.randn(1,3,256,dtype=torch.double)
    w = torch.tensor([[.2,.3,.5]],dtype=torch.double)
    expected = model(x,w)
    torch.testing.assert_close(model(x[:,[2,0,1]],w[:,[2,0,1]]),expected,rtol=1e-9,atol=1e-9)
    ghost = torch.cat((x,torch.full((1,1,256),1e3,dtype=torch.double)),1)
    torch.testing.assert_close(model(ghost,torch.cat((w,torch.zeros(1,1,dtype=torch.double)),1)),expected,rtol=1e-9,atol=1e-9)
    split = x[:,[0,1,1,2]]
    torch.testing.assert_close(model(split,torch.tensor([[.2,.1,.2,.5]],dtype=torch.double)),expected,rtol=1e-9,atol=1e-9)


def test_latent_dynamics_retains_absolute_dose_and_has_correct_weight_gradients():
    import torch
    from fragrance_ai.research.physmix_comparison import DoseLatentDynamics
    torch.manual_seed(832)
    model = DoseLatentDynamics(4,2).double()
    x = torch.randn(1,3,256,dtype=torch.double)
    w = torch.tensor([[.2,.3,.5]],dtype=torch.double,requires_grad=True)
    assert not torch.allclose(model(x,w),model(x,w*.1))
    assert torch.autograd.gradcheck(lambda value:model(x,value),w,fast_mode=True)


def test_grouped_protocol_keeps_swapped_pairs_and_duplicates_out_of_training():
    from fragrance_ai.research.r2_physsim import MixturePair
    from scripts.compare_physmix_v83 import partitions,pair_key
    pairs = [MixturePair((str(i),),(str(i+1),),.5,str(i)) for i in range(30)]
    pairs += [MixturePair(pairs[0].mixture_b,pairs[0].mixture_a,.5,'reverse')]
    folds = partitions(pairs,42)
    assert sorted(i for row in folds for i in row['test'])==list(range(len(pairs)))
    for row in folds:
        test = {pair_key(pairs[i]) for i in row['test']}
        assert not test.intersection(pair_key(pairs[i]) for i in row['train']+row['validation'])


def test_pair_model_rejects_empty_and_nonfinite_inputs():
    import torch
    from fragrance_ai.research.physmix_comparison import PairComparison
    _,arrays = network()
    model = PairComparison(arrays,'current')
    x,w,context = torch.zeros(1,2,256),torch.zeros(1,2),torch.zeros(1,64)
    with pytest.raises(ValueError):
        model(x,w,x,w,context)
    w[:] = .5
    x[0,0,0] = float('nan')
    with pytest.raises(ValueError):
        model(x,w,x,w,context)


@pytest.mark.parametrize('mode',['current','capacity_control','physmix'])
def test_numpy_export_matches_trained_shape_with_active_interactions(mode):
    import torch
    import numpy as np
    from fragrance_ai.research.physmix_comparison import PairComparison
    from fragrance_ai.research.physmix_numpy import NumpyPairComparison
    _,arrays = network()
    torch.manual_seed(8305)
    model = PairComparison(arrays,mode,steps=4).eval()
    if model.interaction is not None:
        with torch.no_grad():
            model.gate.fill_(.8)
    portable = NumpyPairComparison({k:v.detach().numpy() for k,v in model.state_dict().items()},mode,4)
    a,b = torch.randn(2,5,256),torch.randn(2,5,256)
    wa,wb = torch.rand(2,5),torch.rand(2,5)
    wa[:,-1] = 0
    context = torch.zeros(2,64)
    args = (a,wa,b,wb,context)
    with torch.no_grad():
        expected = model(*args).numpy()
    np.testing.assert_allclose(portable(*[v.numpy() for v in args]),expected,atol=2e-6,rtol=2e-6)


def test_laplacian_reduction_preserves_the_original_vector_force():
    import numpy as np
    from fragrance_ai.research.physmix_numpy import radial_acceleration
    rng = np.random.default_rng(837)
    position = rng.normal(size=(2,23,128))*.2
    position[:,3] = position[:,2]
    mass = rng.uniform(.1,2,(2,23,1))
    charge = rng.uniform(-1,1,(2,23,1))
    radius = rng.uniform(.1,.5,(2,23,1))
    activity = rng.uniform(0,.1,(2,23))
    difference = position[:,:,None]-position[:,None,:]
    distance = np.sqrt((difference**2).sum(-1,keepdims=True)+.25)
    power = ((radius[:,:,None]+radius[:,None,:])*.5/distance)**6
    radial = (-.7*mass[:,:,None]*mass[:,None,:]+.8*charge[:,:,None]*charge[:,None,:])/distance**2
    radial += 24*.6/distance*(2*power**2-power)
    expected = (activity[:,None,:,None]*radial*difference/distance).sum(2)/mass
    actual = radial_acceleration(position,mass,charge,radius,activity,.7,.8,.6)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)


def test_checkpointed_dynamics_preserves_values_and_derivatives():
    import torch
    from torch.utils.checkpoint import checkpoint
    from fragrance_ai.research.physmix_comparison import DoseLatentDynamics
    torch.manual_seed(838)
    model = DoseLatentDynamics(4,2).double()
    values = [torch.randn(1,3,4,dtype=torch.double)*.1,torch.randn(1,3,4,dtype=torch.double)*.1,
        torch.ones(1,3,1,dtype=torch.double),torch.full((1,3),.2,dtype=torch.double),
        torch.randn(1,3,1,dtype=torch.double)*.2,torch.full((1,3,1),.3,dtype=torch.double),
        torch.ones(5,dtype=torch.double)]
    first = [v.clone().requires_grad_() for v in values]
    second = [v.clone().requires_grad_() for v in values]
    direct = model.advance(*first)
    reduced = checkpoint(model.advance,*second,use_reentrant=False)
    for x,y in zip(direct,reduced):
        torch.testing.assert_close(x,y)
    gradient = torch.autograd.grad(sum(v.square().sum() for v in direct),first)
    checkpoint_gradient = torch.autograd.grad(sum(v.square().sum() for v in reduced),second)
    for x,y in zip(gradient,checkpoint_gradient):
        torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-12)
