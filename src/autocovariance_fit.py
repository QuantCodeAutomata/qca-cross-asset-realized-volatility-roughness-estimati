"""
Autocovariance fitting and two-estimator correction for Hurst estimation.
Exp_3: Measurement-error attenuation and two-estimator correction.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, List
import statsmodels.api as sm


def fgn_autocovariance(H: float, nu: float, k: int) -> float:
    """Fractional Gaussian Noise autocovariance at lag k.

    gamma(k) = (nu^2 / 2) * (|k+1|^(2H) - 2*|k|^(2H) + |k-1|^(2H))

    Under additive noise (Y_t = X_t + eps_t, eps ~ iid N(0, omega^2)):
    The increment dY_t = dX_t + eps_t - eps_{t-1} has autocovariance:
        gamma_obs(0) = gamma(0) + 2*omega^2
        gamma_obs(1) = gamma(1) - omega^2
        gamma_obs(k) = gamma(k) for k >= 2

    Args:
        H: Hurst exponent
        nu: vol of vol
        k: lag (0, 1, 2, ...)

    Returns:
        autocovariance scalar
    """
    k = abs(k)
    return (nu**2 / 2.0) * (
        (k + 1) ** (2 * H) - 2 * k ** (2 * H) + abs(k - 1) ** (2 * H)
    )


def _model_gamma_obs(H: float, nu: float, omega2: float, k: int) -> float:
    """Model autocovariance of observed fGn + iid noise increments at lag k."""
    base = fgn_autocovariance(H, nu, k)
    if k == 0:
        return base + 2.0 * omega2
    if k == 1:
        return base - omega2
    return base


def compute_empirical_autocovariances(log_rv_series: np.ndarray, max_lag: int = 2) -> np.ndarray:
    """Compute empirical autocovariances of one-day increments.

    Increments: dX_t = X_t - X_{t-1}
    Autocovariance at lag k: (1/(n-k)) * sum_{t=k}^{n-1} dX_t * dX_{t-k}

    Args:
        log_rv_series: daily log-RV series
        max_lag: maximum lag (inclusive)

    Returns:
        Array of autocovariances [gamma(0), gamma(1), ..., gamma(max_lag)]
    """
    dx = np.diff(log_rv_series)
    n = len(dx)
    gammas = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        gammas[k] = np.mean(dx[k:] * dx[:n - k]) if k < n else 0.0
    return gammas


def compute_second_moment_from_series(log_rv_series: np.ndarray, delta_max: int = 10) -> np.ndarray:
    """Compute non-overlapping second moments m(2, Delta) for a log-RV series.

    m(2, Delta) = (1/N) * sum_{k=0}^{N-1} |X_{(k+1)*Delta} - X_{k*Delta}|^2
    where N = floor(len(X) / Delta).

    Returns array of m(2, Delta) for Delta = 1, ..., delta_max
    """
    x = np.asarray(log_rv_series, dtype=np.float64)
    m2 = np.empty(delta_max)
    for d in range(1, delta_max + 1):
        n_blocks = len(x) // d
        if n_blocks < 2:
            m2[d - 1] = np.nan
            continue
        inds = np.arange(n_blocks + 1) * d
        inds = inds[inds < len(x)]
        if len(inds) < 2:
            m2[d - 1] = np.nan
            continue
        diffs = x[inds[1:]] - x[inds[:-1]]
        m2[d - 1] = np.mean(diffs**2)
    return m2


def _estimate_h_from_moments(m2: np.ndarray, delta_max: int) -> float:
    """OLS estimate of H from log(m2) ~ log(Delta) regression.

    Returns H_hat = b/2 where b is the OLS slope.
    """
    deltas = np.arange(1, delta_max + 1, dtype=np.float64)
    valid = np.isfinite(m2) & (m2 > 0)
    if valid.sum() < 2:
        return np.nan
    log_d = np.log(deltas[valid])
    log_m = np.log(m2[valid])
    X = sm.add_constant(log_d)
    ols = sm.OLS(log_m, X).fit()
    return ols.params[1] / 2.0


def fit_two_estimator_correction(log_rv_a: np.ndarray, log_rv_b: np.ndarray,
                                   delta_max: int = 10) -> dict:
    """Implement the two-estimator correction framework.

    Estimator a = RV5m (noisier), estimator b = RK (cleaner).

    Steps:
    1. Compute average second-moment gap: delta_m = mean_Delta [m_a(2,Delta) - m_b(2,Delta)]
       which estimates 2*(omega_a^2 - omega_b^2)
    2. Compute first 3 empirical autocovariances of one-day increments for each estimator
    3. Jointly fit (H, nu, omega_a) by minimizing sum of squared differences between
       empirical and model autocovariances for both series, with constraint
       omega_a^2 = omega_b^2 + delta_m/2
    4. Compute corrected second moments: m_corr(Delta) = m(2,Delta) - 2*omega^2
    5. Re-estimate H from corrected moments (delta_max=10 and delta_max=40)

    Args:
        log_rv_a: daily log-RV series from estimator a (RV5m)
        log_rv_b: daily log-RV series from estimator b (RK)
        delta_max: max lag for second moment computation

    Returns:
        dict with keys: H_raw_a, H_raw_b, omega_a, omega_b, H_nu, H_corrected_a,
                        H_corrected_b, H_corrected_lag40_a, H_corrected_lag40_b,
                        convergence_status
    """
    # Raw H estimates from OLS on m(2,Delta)
    m2_a = compute_second_moment_from_series(log_rv_a, delta_max)
    m2_b = compute_second_moment_from_series(log_rv_b, delta_max)
    H_raw_a = _estimate_h_from_moments(m2_a, delta_max)
    H_raw_b = _estimate_h_from_moments(m2_b, delta_max)

    # Step 1: Second-moment gap → estimate 2*(omega_a^2 - omega_b^2)
    valid = np.isfinite(m2_a) & np.isfinite(m2_b)
    delta_m = float(np.mean((m2_a - m2_b)[valid]))  # ≈ 2*(omega_a^2 - omega_b^2)
    omega_ab_diff2 = max(delta_m / 2.0, 0.0)  # omega_a^2 - omega_b^2, clamped

    # Step 2: Empirical autocovariances of increments (lags 0, 1, 2)
    emp_a = compute_empirical_autocovariances(log_rv_a, max_lag=2)
    emp_b = compute_empirical_autocovariances(log_rv_b, max_lag=2)

    # Step 3: Joint fit: parameters = [H, nu, log(omega_a^2)]
    # omega_b^2 = omega_a^2 - omega_ab_diff2 (constrained >= 0)
    def objective(params: np.ndarray) -> float:
        H_p, nu_p, log_oa2 = params
        if H_p <= 0.0 or H_p >= 1.0 or nu_p <= 0.0:
            return 1e10
        oa2 = np.exp(log_oa2)
        ob2 = max(oa2 - omega_ab_diff2, 1e-12)
        resid = 0.0
        for k in range(3):
            resid += (_model_gamma_obs(H_p, nu_p, oa2, k) - emp_a[k]) ** 2
            resid += (_model_gamma_obs(H_p, nu_p, ob2, k) - emp_b[k]) ** 2
        return resid

    # Initialise with reasonable guesses
    H_init = float(np.clip(0.5 * (H_raw_a + H_raw_b), 0.05, 0.49))
    nu_init = float(np.sqrt(max(emp_a[0], 1e-6)))
    oa2_init = max(omega_ab_diff2 + 1e-6, emp_a[0] * 0.01)

    x0 = np.array([H_init, nu_init, np.log(oa2_init)])
    bounds = [(0.01, 0.49), (1e-4, 10.0), (-30.0, 0.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8})

    if res.success or res.fun < 1e-6:
        H_nu, nu_fit, log_oa2_fit = res.x
        convergence_status = "converged"
    else:
        # Fall back to initial guesses if minimisation failed
        H_nu, nu_fit, log_oa2_fit = H_init, nu_init, np.log(oa2_init)
        convergence_status = f"failed: {res.message}"

    omega_a = float(np.sqrt(np.exp(log_oa2_fit)))
    oa2_fit = np.exp(log_oa2_fit)
    ob2_fit = max(oa2_fit - omega_ab_diff2, 0.0)
    omega_b = float(np.sqrt(ob2_fit))

    # Step 4 & 5: Corrected second moments and re-estimated H
    def corrected_h(m2: np.ndarray, omega2: float, d_max: int) -> float:
        m2_corr = m2[:d_max] - 2.0 * omega2
        # Clip to a small positive floor to avoid log(negative)
        m2_corr = np.maximum(m2_corr, 1e-12)
        return _estimate_h_from_moments(m2_corr, d_max)

    # For delta_max = 10
    H_corr_a10 = corrected_h(m2_a, oa2_fit, delta_max)
    H_corr_b10 = corrected_h(m2_b, ob2_fit, delta_max)

    # For delta_max = 40 (recompute moments at longer horizon)
    m2_a40 = compute_second_moment_from_series(log_rv_a, 40)
    m2_b40 = compute_second_moment_from_series(log_rv_b, 40)
    H_corr_a40 = corrected_h(m2_a40, oa2_fit, 40)
    H_corr_b40 = corrected_h(m2_b40, ob2_fit, 40)

    return {
        "H_raw_a": H_raw_a,
        "H_raw_b": H_raw_b,
        "omega_a": omega_a,
        "omega_b": omega_b,
        "H_nu": float(H_nu),
        "H_corrected_a": H_corr_a10,
        "H_corrected_b": H_corr_b10,
        "H_corrected_lag40_a": H_corr_a40,
        "H_corrected_lag40_b": H_corr_b40,
        "convergence_status": convergence_status,
    }
