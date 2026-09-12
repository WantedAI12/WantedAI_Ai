"""Small-data, multi-output RKHS regression with an unpenalized intercept.

This is our numerical implementation, not an imported estimator. The output
metric is learned ONLY from training responses and shrunk toward identity.
Nothing here treats descriptor covariance as molecular binding chemistry.
"""
import numpy as np


def fit_structured_kernel(kernel, targets, *, alpha, output_coupling=0.):
    """Solve Kc C + alpha C B^-1 = Yc in two orthogonal spectral bases.

    It minimizes ||Y-KC-1b||_F² + alpha*tr(C.T K C B^-1). Centering
    handles the unpenalized constant exactly; no explicit matrix inverse or
    Kronecker matrix is formed. B=(1-rho)I+rho*d*Cov(Y)/tr(Cov(Y)).
    """
    k, y = np.asarray(kernel, float), np.asarray(targets, float)
    if (k.ndim != 2 or len(k) < 2 or k.shape != (len(k), len(k))
            or y.ndim != 2 or len(y) != len(k) or y.shape[1] < 1
            or not np.isfinite(k).all() or not np.isfinite(y).all()
            or not np.isfinite([alpha, output_coupling]).all()
            or alpha <= 0 or not 0 <= output_coupling < 1):
        raise ValueError('finite kernel/targets, positive alpha and coupling in [0,1) required')
    tolerance = 128*np.finfo(float).eps*max(1., np.linalg.norm(k, ord=np.inf))
    if np.max(np.abs(k-k.T)) > tolerance:
        raise ValueError('symmetric positive semidefinite kernel required')
    k = (k+k.T)*.5
    # Checking only H K H would miss a negative constant mode: for example
    # -ones(n,n) centers to zero, despite not being a valid RKHS Gram matrix.
    if np.linalg.eigvalsh(k)[0] < -tolerance:
        raise ValueError('kernel is not positive semidefinite')
    column_mean = k.mean(axis=0)
    centered = k-column_mean[None,:]-column_mean[:,None]+column_mean.mean()
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    if eigenvalues[0] < -tolerance:
        raise ValueError('kernel is not positive semidefinite')
    # Only eigenvalues within the explicitly checked roundoff envelope are zeroed.
    eigenvalues = np.maximum(eigenvalues, 0.)
    mean = y.mean(axis=0)
    response = y-mean
    d = y.shape[1]
    covariance = response.T@response/max(1, len(y)-1)
    trace = float(np.trace(covariance))
    metric = np.eye(d)
    if output_coupling and trace > 0:
        metric = (1-output_coupling)*metric+output_coupling*d*covariance/trace
    if output_coupling and trace > 0:
        output_values, output_vectors = np.linalg.eigh(metric)
    else:
        output_values, output_vectors = np.ones(d), np.eye(d)
    if not np.isfinite(output_values).all() or np.any(output_values <= 0):
        raise ValueError('output metric must remain numerically positive definite')
    transformed = eigenvectors.T@response@output_vectors
    denominators = eigenvalues[:,None]+alpha/output_values[None,:]
    coefficients = eigenvectors@(transformed/denominators)@output_vectors.T
    # Project away numerical constant-mode leakage; this is the exact constraint
    # 1.T C=0, not a score correction or a target change.
    coefficients -= coefficients.mean(axis=0, keepdims=True)
    intercept = mean-column_mean@coefficients
    precision_action = ((coefficients@output_vectors)/output_values[None,:])@output_vectors.T
    residual = centered@coefficients+alpha*precision_action-response
    relative_residual = float(np.linalg.norm(residual)/max(np.linalg.norm(response), 1.))
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all() or relative_residual > 1e-8:
        raise ValueError('structured kernel solve failed numerical residual check')
    return {'coefficients': coefficients, 'intercept': intercept,
        'diagnostics': {'solver': 'centered_two_sided_spectral_ridge_v66',
            'normal_equation_relative_residual': relative_residual,
            'intercept_stationarity_max_abs': float(np.max(np.abs(coefficients.sum(axis=0)))),
            'output_coupling': float(output_coupling),
            'output_metric_min_eigenvalue': float(output_values.min()),
            'output_metric_max_eigenvalue': float(output_values.max()),
            'effective_spectral_condition': float(denominators.max()/denominators.min()),
            'training_rows': len(y), 'outputs': d}}
