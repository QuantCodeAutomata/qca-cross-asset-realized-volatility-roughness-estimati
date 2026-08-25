"""Two-estimator measurement-error correction for realised-volatility Hurst fits.

The implementation follows Section 4.3 of the attached paper.  It keeps the
signed second-moment gap, enforces the variance-difference constraint exactly,
and never hides infeasible corrected moments behind a positive clipping floor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

try:
    from .hurst_realized import compute_second_moment_scaling
except ImportError:  # support direct imports when ``src`` is on sys.path
    from hurst_realized import compute_second_moment_scaling


def fgn_autocovariance(H: float, nu: float, k: int) -> float:
    """Fractional-Gaussian-noise autocovariance at integer lag ``k``."""
    k = abs(int(k))
    return (nu**2 / 2.0) * (
        (k + 1) ** (2.0 * H)
        - 2.0 * k ** (2.0 * H)
        + abs(k - 1) ** (2.0 * H)
    )


def _model_gamma_obs(H: float, nu: float, omega2: float, k: int) -> float:
    base = fgn_autocovariance(H, nu, k)
    if k == 0:
        return base + 2.0 * omega2
    if k == 1:
        return base - omega2
    return base


def compute_empirical_autocovariances(
    log_rv_series: np.ndarray,
    max_lag: int = 2,
    *,
    demean: bool = True,
) -> np.ndarray:
    """Sample autocovariances of one-day log-RV increments.

    Pairwise finite observations are used at each lag.  Demeaning is the
    standard covariance convention and avoids attributing a small sample drift
    in the increment series to the fGn/noise model.
    """
    x = np.asarray(log_rv_series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("log_rv_series must be one-dimensional")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    dx = np.diff(x)
    finite_dx = dx[np.isfinite(dx)]
    mean_dx = float(np.mean(finite_dx)) if demean and finite_dx.size else 0.0
    dx = dx - mean_dx

    gammas = np.full(max_lag + 1, np.nan, dtype=np.float64)
    n = dx.size
    for k in range(max_lag + 1):
        if k >= n:
            continue
        left = dx[k:]
        right = dx[: n - k]
        valid = np.isfinite(left) & np.isfinite(right)
        if np.any(valid):
            gammas[k] = float(np.mean(left[valid] * right[valid]))
    return gammas


def compute_second_moment_from_series(
    log_rv_series: np.ndarray,
    delta_max: int = 10,
) -> np.ndarray:
    """Return the paper's non-overlapping ``m(2,Δ)`` values."""
    _, m2 = compute_second_moment_scaling(
        log_rv_series, delta_max=delta_max, overlapping=False
    )
    return m2


def _estimate_h_from_moments(m2: np.ndarray, delta_max: Optional[int] = None) -> float:
    """OLS estimate ``H = slope/2`` from an array of second moments."""
    moments = np.asarray(m2, dtype=np.float64)
    if delta_max is None:
        delta_max = moments.size
    delta_max = min(int(delta_max), moments.size)
    moments = moments[:delta_max]
    deltas = np.arange(1, delta_max + 1, dtype=np.float64)
    valid = np.isfinite(moments) & (moments > 0.0)
    if int(valid.sum()) < 2:
        return np.nan
    design = sm.add_constant(np.log(deltas[valid]))
    result = sm.OLS(np.log(moments[valid]), design).fit()
    return float(result.params[1]) / 2.0


def _noise_variances_from_base(base: float, difference: float) -> tuple[float, float]:
    """Map a non-negative base variance to ``(omega_a², omega_b²)``.

    ``difference`` is the signed constraint ``omega_a² - omega_b²``.
    """
    if difference >= 0.0:
        return base + difference, base
    return base, base - difference


def _invalid_result(
    H_raw_a: float,
    H_raw_b: float,
    delta_m: float,
    status: str,
) -> dict:
    return {
        "H_raw_a": H_raw_a,
        "H_raw_b": H_raw_b,
        "omega_a": np.nan,
        "omega_b": np.nan,
        "omega_a2": np.nan,
        "omega_b2": np.nan,
        "delta_m": delta_m,
        "omega2_difference": delta_m / 2.0 if np.isfinite(delta_m) else np.nan,
        "H_nu": np.nan,
        "nu_fit": np.nan,
        "H_corrected_a": np.nan,
        "H_corrected_b": np.nan,
        "H_corrected_lag40_a": np.nan,
        "H_corrected_lag40_b": np.nan,
        "correction_valid_10": False,
        "correction_valid_40": False,
        "invalid_lags_a_10": [],
        "invalid_lags_b_10": [],
        "invalid_lags_a_40": [],
        "invalid_lags_b_40": [],
        "objective": np.nan,
        "convergence_status": status,
    }


def _corrected_h_with_diagnostics(
    moments: np.ndarray,
    omega2: float,
    delta_max: int,
) -> tuple[float, bool, list[int]]:
    corrected = np.asarray(moments[:delta_max], dtype=np.float64) - 2.0 * omega2
    invalid = np.flatnonzero(~np.isfinite(corrected) | (corrected <= 0.0)) + 1
    if invalid.size:
        return np.nan, False, invalid.astype(int).tolist()

    h = _estimate_h_from_moments(corrected, delta_max)
    # H is a regularity exponent.  Values outside (0,1) are not an
    # interpretable correction and usually signal a near-zero moment curve.
    if not np.isfinite(h) or not (0.0 < h < 1.0):
        return np.nan, False, []
    return h, True, []


def fit_two_estimator_correction(
    log_rv_a: np.ndarray,
    log_rv_b: np.ndarray,
    delta_max: int = 10,
    *,
    long_delta_max: int = 40,
    h_bounds: tuple[float, float] = (0.01, 0.49),
) -> dict:
    """Fit the paper's two-estimator additive-noise correction.

    Estimator ``a`` is normally RV5m and estimator ``b`` RK.  The fitted
    relation is

    ``omega_a² = omega_b² + delta_m/2``, where
    ``delta_m = mean_Δ(m_a(2,Δ)-m_b(2,Δ))``.

    Unlike the submitted version, ``delta_m`` is not clamped to be positive:
    a sign reversal is itself a useful misspecification diagnostic.  The
    optimiser is constrained so that corrected moments remain positive over
    both requested refit windows.  If that is impossible, corrected H values
    are returned as ``NaN`` rather than as pathological values generated by
    clipping moments to ``1e-12``.
    """
    a = np.asarray(log_rv_a, dtype=np.float64)
    b = np.asarray(log_rv_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("both input series must be one-dimensional")
    if delta_max < 2 or long_delta_max < delta_max:
        raise ValueError("require 2 <= delta_max <= long_delta_max")

    n = min(a.size, b.size)
    if n < long_delta_max + 2:
        long_delta_max = min(long_delta_max, max(delta_max, n - 2))
    a = a[:n]
    b = b[:n]
    # Preserve the calendar grid.  Compressing out missing dates would turn
    # multi-day gaps into artificial one-day increments and corrupt both the
    # moment curve and the increment autocovariances.
    common_count = int(np.count_nonzero(np.isfinite(a) & np.isfinite(b)))
    if common_count < delta_max + 2:
        return _invalid_result(np.nan, np.nan, np.nan, "insufficient_data")

    m2_a_long = compute_second_moment_from_series(a, long_delta_max)
    m2_b_long = compute_second_moment_from_series(b, long_delta_max)
    m2_a = m2_a_long[:delta_max]
    m2_b = m2_b_long[:delta_max]
    H_raw_a = _estimate_h_from_moments(m2_a, delta_max)
    H_raw_b = _estimate_h_from_moments(m2_b, delta_max)

    gap_valid = np.isfinite(m2_a) & np.isfinite(m2_b)
    if not np.any(gap_valid):
        return _invalid_result(H_raw_a, H_raw_b, np.nan, "no_valid_moment_gap")
    delta_m = float(np.mean(m2_a[gap_valid] - m2_b[gap_valid]))
    difference = delta_m / 2.0

    emp_a = compute_empirical_autocovariances(a, max_lag=2, demean=True)
    emp_b = compute_empirical_autocovariances(b, max_lag=2, demean=True)
    if not np.all(np.isfinite(emp_a)) or not np.all(np.isfinite(emp_b)):
        return _invalid_result(H_raw_a, H_raw_b, delta_m, "invalid_autocovariances")

    # Feasibility for subtracting 2*omega² from every moment used in either
    # refit.  A small margin keeps log moments away from numerical zero.
    margin = 1.0 - 1e-8
    max_oa2 = 0.5 * float(np.nanmin(m2_a_long)) * margin
    max_ob2 = 0.5 * float(np.nanmin(m2_b_long)) * margin
    if not np.isfinite(max_oa2) or not np.isfinite(max_ob2):
        return _invalid_result(H_raw_a, H_raw_b, delta_m, "invalid_moments")

    if difference >= 0.0:
        base_upper = min(max_ob2, max_oa2 - difference)
    else:
        base_upper = min(max_oa2, max_ob2 + difference)
    if base_upper <= 0.0:
        return _invalid_result(
            H_raw_a, H_raw_b, delta_m, "infeasible_noise_variance_constraint"
        )

    h_lo, h_hi = h_bounds
    if not (0.0 < h_lo < h_hi < 1.0):
        raise ValueError("h_bounds must lie inside (0,1)")

    scale = max(float(np.max(np.abs(np.r_[emp_a, emp_b]))), 1e-8)

    def objective(params: np.ndarray) -> float:
        h, log_nu, base = map(float, params)
        nu = float(np.exp(log_nu))
        oa2, ob2 = _noise_variances_from_base(base, difference)
        model_a = np.array([_model_gamma_obs(h, nu, oa2, k) for k in range(3)])
        model_b = np.array([_model_gamma_obs(h, nu, ob2, k) for k in range(3)])
        residuals = np.r_[model_a - emp_a, model_b - emp_b] / scale
        return float(np.dot(residuals, residuals))

    raw_center = np.nanmean([H_raw_a, H_raw_b])
    if not np.isfinite(raw_center):
        raw_center = 0.2
    h_starts = np.unique(
        np.clip(np.array([raw_center, 0.08, 0.15, 0.25, 0.35]), h_lo, h_hi)
    )
    nu_guess = np.sqrt(max(0.5 * (emp_a[0] + emp_b[0]), 1e-8))
    base_starts = np.unique(
        np.clip(np.array([0.0, 0.1, 0.4, 0.8]) * base_upper, 0.0, base_upper)
    )
    bounds = [(h_lo, h_hi), (np.log(1e-6), np.log(100.0)), (0.0, base_upper)]

    best = None
    for h0 in h_starts:
        for base0 in base_starts:
            candidate = minimize(
                objective,
                np.array([h0, np.log(nu_guess), base0], dtype=np.float64),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 4000, "ftol": 1e-14, "gtol": 1e-10},
            )
            if np.isfinite(candidate.fun) and (
                best is None or candidate.fun < best.fun
            ):
                best = candidate

    if best is None:
        return _invalid_result(H_raw_a, H_raw_b, delta_m, "optimizer_failed")

    H_nu, log_nu_fit, base_fit = map(float, best.x)
    nu_fit = float(np.exp(log_nu_fit))
    oa2_fit, ob2_fit = _noise_variances_from_base(base_fit, difference)

    h_a10, valid_a10, invalid_a10 = _corrected_h_with_diagnostics(
        m2_a_long, oa2_fit, delta_max
    )
    h_b10, valid_b10, invalid_b10 = _corrected_h_with_diagnostics(
        m2_b_long, ob2_fit, delta_max
    )
    h_a40, valid_a40, invalid_a40 = _corrected_h_with_diagnostics(
        m2_a_long, oa2_fit, long_delta_max
    )
    h_b40, valid_b40, invalid_b40 = _corrected_h_with_diagnostics(
        m2_b_long, ob2_fit, long_delta_max
    )

    valid10 = bool(valid_a10 and valid_b10)
    valid40 = bool(valid_a40 and valid_b40)
    status = "converged" if best.success else f"best_effort: {best.message}"
    if not valid10:
        status += "; invalid_primary_corrected_curve"
    elif not valid40:
        status += "; invalid_long_corrected_curve"

    return {
        "H_raw_a": H_raw_a,
        "H_raw_b": H_raw_b,
        "omega_a": float(np.sqrt(oa2_fit)),
        "omega_b": float(np.sqrt(ob2_fit)),
        "omega_a2": float(oa2_fit),
        "omega_b2": float(ob2_fit),
        "delta_m": delta_m,
        "omega2_difference": difference,
        "H_nu": H_nu,
        "nu_fit": nu_fit,
        "H_corrected_a": h_a10,
        "H_corrected_b": h_b10,
        "H_corrected_lag40_a": h_a40,
        "H_corrected_lag40_b": h_b40,
        "correction_valid_10": valid10,
        "correction_valid_40": valid40,
        "invalid_lags_a_10": invalid_a10,
        "invalid_lags_b_10": invalid_b10,
        "invalid_lags_a_40": invalid_a40,
        "invalid_lags_b_40": invalid_b40,
        "objective": float(best.fun),
        "convergence_status": status,
    }
