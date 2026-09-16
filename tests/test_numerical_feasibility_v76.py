import numpy as np

from fragrance_ai.recommender.autoregressive_refinement import ResidualFeedback
from fragrance_ai.recommender.numerical_feasibility import canonical_solver_weights


def test_observed_negative_roundoff_is_repaired_before_feedback():
    x=np.array([.6,.4000000000000053,-1.6688970107281406e-15])
    original=x.copy()
    validate=lambda w: abs(w.sum()-1)<1e-8 and w[0]<=.6
    result=canonical_solver_weights(x,np.zeros(3),np.ones(3),validate)
    assert result is not None and not np.any(result<0)
    np.testing.assert_array_equal(x,original)
    history=ResidualFeedback(target=95.)
    assert history.observe(result,[.01,-.01],95.00009999845729,qualified=True)


def test_real_negative_or_broken_physical_constraint_is_not_repaired():
    assert canonical_solver_weights([1.,-1e-5],[0,0],[1,1],lambda x:True) is None
    assert canonical_solver_weights([1.,-1e-15],[0,0],[1,1],lambda x:x.sum()<.99) is None
    assert canonical_solver_weights([np.nan,1.],[0,0],[1,1],lambda x:True) is None


def test_threshold_parser_never_treats_irritation_or_water_as_odor_threshold():
    from scripts.extend_qualified_thresholds_v76 import measurements
    values=measurements('Odor low= 9.00 mg/cu m; Odor high= 90 mg/cu m; Irritating Concn= 2.0 mg/cu m',100.)
    assert len(values)==1
    assert measurements('odor detection threshold in water 0.1 ppm',100.)==[]
    assert measurements('odor recognition threshold in air 0.1 ppm',100.)==[]
