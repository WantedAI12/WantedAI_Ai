import numpy as np
import pytest

from fragrance_ai.research.mixture_profiles import mixture_features,fit_mixture_model,predict_mixture_model,CROSS_FEATURE_VERSION

KEYS = [('a','.1','pg'),('b','.01','pg')]


def test_cross_moments_retain_associations_missing_from_marginal_features():
    together = np.array([[1.,1.],[0.,0.]])
    separate = np.array([[1.,0.],[0.,1.]])
    np.testing.assert_array_equal(mixture_features(together,KEYS),mixture_features(separate,KEYS))
    assert not np.array_equal(mixture_features(together,KEYS,cross_moments=True),
                              mixture_features(separate,KEYS,cross_moments=True))


def test_cross_moments_are_split_permutation_and_trace_continuous():
    profiles = np.array([[1.,2.],[3.,.1]])
    expected = mixture_features(profiles,KEYS,[2.,1.],cross_moments=True)
    np.testing.assert_allclose(mixture_features(profiles[[1,0,0]],[KEYS[1],KEYS[0],KEYS[0]],
        [1.,1.,1.],cross_moments=True),expected,atol=1e-14,rtol=0)
    pure = mixture_features(profiles[:1],KEYS[:1],cross_moments=True)
    trace = mixture_features(profiles,KEYS,[1.,1e-12],cross_moments=True)
    np.testing.assert_allclose(trace[:-1],pure[:-1],atol=1e-8,rtol=0)
    # Standard deviation approaches zero as sqrt(volume), unlike the
    # covariance and mean, which approach their limits linearly in volume.
    epsilon = 1e-12/(1.+1e-12)
    np.testing.assert_allclose(trace[-1],np.sqrt(epsilon*(1.-epsilon)),atol=1e-10,rtol=0)


@pytest.mark.parametrize('scaling',['per_feature','shared_sensory','balanced_sensory'])
def test_cross_checkpoint_prediction_and_single_stock_constraint(scaling):
    profiles = np.array([[1.,2.],[3.,.1]])
    x = np.array([mixture_features(profiles,KEYS,[w,1-w],cross_moments=True) for w in np.linspace(.1,.9,10)])
    model = fit_mixture_model(x,x[:,:2],.1,1.,scaling)
    assert model['features'] == CROSS_FEATURE_VERSION
    np.testing.assert_allclose(predict_mixture_model(model,x),x[:,:2],atol=1e-12)
    pure = mixture_features(profiles[:1],KEYS[:1],cross_moments=True)
    np.testing.assert_array_equal(predict_mixture_model(model,pure[None,:])[0],profiles[0])


def test_cross_feature_contract_rejects_old_width():
    profiles = np.array([[1.,2.],[3.,.1]])
    x = np.array([mixture_features(profiles,KEYS,[w,1-w],cross_moments=True) for w in (.2,.5,.8)])
    model = fit_mixture_model(x,x[:,:2],.1,1.)
    with pytest.raises(ValueError): predict_mixture_model(model,x[:,:10])
