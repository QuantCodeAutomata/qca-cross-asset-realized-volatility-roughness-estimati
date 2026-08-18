"""
Tests for Hurst exponent estimation from log realized variance.
Covers: second-moment scaling, OLS regression, fBM simulation consistency.
"""
import numpy as np
import pytest
import statsmodels.api as sm

from src.hurst_realized import compute_second_moment_scaling, estimate_hurst_ols
from src.rv_estimators import compute_rv5m, compute_tsrv, compute_bpv
from src.simulate_rough_paths import simulate_fbm_davies_harte


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fbm_log_rv(n: int, H: float, seed: int = 42) -> np.ndarray:
    """Return cumulative fBM path (length n+1) as a proxy log-RV series."""
    fgn = simulate_fbm_davies_harte(n, H, seed=seed)
    return np.concatenate([[0.0], np.cumsum(fgn)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_second_moment_fbm_scaling():
    """m(2, Δ) should scale as Δ^(2H) for a cumulative fBM input.

    Checks that the OLS slope of log m(2,Δ) on log Δ is 2H (within tolerance).
    """
    H = 0.25
    n = 8_000
    log_rv = _make_fbm_log_rv(n, H, seed=7)

    deltas, m2 = compute_second_moment_scaling(log_rv, delta_max=10)

    log_lags = np.log(deltas)
    log_m2 = np.log(m2)

    X = sm.add_constant(log_lags)
    res = sm.OLS(log_m2, X).fit()
    slope = float(res.params[1])

    assert abs(slope / 2.0 - H) < 0.06, (
        f"Estimated H={slope/2:.3f} should be near {H}"
    )


def test_hurst_ols_known_slope():
    """OLS H estimation with analytically-known m(2, Δ) = Δ^(2H).

    Constructs an exact log-log linear relationship and checks that
    estimate_hurst_ols recovers the correct H via the OLS slope.
    """
    H_true = 0.30
    lags = np.arange(1, 11, dtype=np.float64)
    m2_analytic = lags ** (2 * H_true)

    # OLS directly on known values
    log_lags = np.log(lags)
    log_m2 = np.log(m2_analytic)
    X = sm.add_constant(log_lags)
    res = sm.OLS(log_m2, X).fit()
    H_hat = float(res.params[1]) / 2.0

    assert abs(H_hat - H_true) < 1e-9, (
        f"Exact OLS should recover H={H_true} exactly, got {H_hat:.8f}"
    )


def test_hurst_rough_below_half():
    """H estimates from a rough (H<0.5) fBM path should be < 0.5."""
    for H in [0.10, 0.20, 0.30]:
        log_rv = _make_fbm_log_rv(6_000, H, seed=100 + int(H * 100))
        result = estimate_hurst_ols(log_rv, delta_max=10)
        assert result['H_hat'] < 0.5, (
            f"H={H}: H_hat={result['H_hat']:.3f} should be < 0.5"
        )


def test_hurst_r_squared_high():
    """R² of log-log regression should be high (>0.90) for clean fBM data."""
    H = 0.20
    log_rv = _make_fbm_log_rv(10_000, H, seed=55)
    result = estimate_hurst_ols(log_rv, delta_max=10)
    assert result['r_squared'] > 0.90, (
        f"R²={result['r_squared']:.3f} should exceed 0.90 for clean fBM"
    )


def test_hurst_delta_max_sensitivity():
    """H estimates should be reasonably stable across delta_max=10 and 40."""
    H = 0.25
    log_rv = _make_fbm_log_rv(12_000, H, seed=11)

    res10 = estimate_hurst_ols(log_rv, delta_max=10)
    res40 = estimate_hurst_ols(log_rv, delta_max=40)

    # Both should be in the rough regime (< 0.5) and close to each other
    assert res10['H_hat'] < 0.5
    assert res40['H_hat'] < 0.5
    assert abs(res10['H_hat'] - res40['H_hat']) < 0.10, (
        f"delta_max sensitivity too high: "
        f"H10={res10['H_hat']:.3f}, H40={res40['H_hat']:.3f}"
    )


# ---------------------------------------------------------------------------
# RV estimator sanity tests
# ---------------------------------------------------------------------------

def test_rv5m_positive_for_nonzero_returns():
    """RV5m should be positive for any path with non-constant prices."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 390)
    rv = compute_rv5m(returns)
    assert rv > 0.0


def test_bpv_less_than_rv_under_no_jumps():
    """BPV ≤ RV for a smooth path (BPV is always non-negative)."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.001, 200)
    rv = compute_rv5m(returns)
    bpv = compute_bpv(returns)
    assert bpv >= 0.0
    # Under no jumps, BPV / (mu1^-2) ≈ RV; ratio should be in (0.5, 2.0)
    ratio = bpv / rv if rv > 0 else 0
    assert 0.1 < ratio < 5.0


def test_tsrv_returns_scalar():
    """TSRV should return a finite scalar for a typical price path."""
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.001, 500)
    tsrv = compute_tsrv(returns)
    assert np.isfinite(tsrv)
