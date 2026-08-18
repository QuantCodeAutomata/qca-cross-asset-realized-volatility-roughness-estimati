"""
Tests for measurement-error attenuation and two-estimator correction (exp_3).
Covers: fGn autocovariance model, noise model properties, fBM simulation,
and the two-estimator H correction.
"""
import numpy as np
import pytest

from src.autocovariance_fit import (
    fgn_autocovariance,
    compute_empirical_autocovariances,
    fit_two_estimator_correction,
)
from src.simulate_rough_paths import simulate_fbm_davies_harte, simulate_fou_log_vol
from src.hurst_realized import estimate_hurst_ols


# ---------------------------------------------------------------------------
# fGn autocovariance model
# ---------------------------------------------------------------------------

def test_fgn_autocovariance_lag0_positive():
    """Lag-0 autocovariance (variance) should be positive for all H."""
    for H in [0.05, 0.20, 0.35, 0.49]:
        gamma0 = fgn_autocovariance(H, nu=1.0, k=0)
        assert gamma0 > 0.0, f"H={H}: lag-0 autocovariance should be positive"


def test_fgn_autocovariance_lag0_equals_nu_squared():
    """Lag-0 autocovariance should equal ν² (variance of each fGn increment)."""
    for nu in [0.5, 1.0, 2.0]:
        gamma0 = fgn_autocovariance(0.25, nu=nu, k=0)
        assert abs(gamma0 - nu ** 2) < 1e-12, (
            f"lag-0 should be ν²={nu**2}, got {gamma0}"
        )


def test_fgn_autocovariance_rough_negative_lag1():
    """For H < 0.5, lag-1 autocovariance should be negative (anti-persistence)."""
    for H in [0.10, 0.20, 0.30, 0.40]:
        gamma1 = fgn_autocovariance(H, nu=1.0, k=1)
        assert gamma1 < 0.0, (
            f"H={H}: lag-1 should be negative for rough fGn, got {gamma1:.4f}"
        )


def test_noise_adds_to_lag0():
    """Additive i.i.d. noise increases the lag-0 autocovariance by 2·ω².

    Under Y_t = X_t + ε_t, increments dY_t = dX_t + dε_t have:
        Var(dY) = Var(dX) + Var(dε) = γ(0) + 2·ω²
    """
    H, nu, omega = 0.25, 1.0, 0.40
    n = 80_000
    seed = 42

    # fBM as the 'latent' log-vol path (cumulative fGn)
    fgn = simulate_fbm_davies_harte(n, H, seed=seed)
    log_vol = np.concatenate([[0.0], np.cumsum(fgn)])

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, omega, n + 1)
    log_rv_noisy = log_vol + noise
    log_rv_clean = log_vol

    gamma_noisy = compute_empirical_autocovariances(log_rv_noisy, max_lag=0)
    gamma_clean = compute_empirical_autocovariances(log_rv_clean, max_lag=0)

    # Expected increase at lag 0
    expected_increase = 2.0 * omega ** 2

    measured_increase = gamma_noisy[0] - gamma_clean[0]
    assert abs(measured_increase - expected_increase) < 0.04, (
        f"Expected lag-0 increase of {expected_increase:.3f}, "
        f"measured {measured_increase:.3f}"
    )


def test_noise_subtracts_at_lag1():
    """Additive i.i.d. noise decreases the lag-1 autocovariance by ω².

    Under Y_t = X_t + ε_t: Cov(dY_t, dY_{t+1}) = γ_X(1) − ω²
    """
    H, nu, omega = 0.25, 1.0, 0.40
    n = 80_000
    seed = 43

    fgn = simulate_fbm_davies_harte(n, H, seed=seed)
    log_vol = np.concatenate([[0.0], np.cumsum(fgn)])

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, omega, n + 1)
    log_rv_noisy = log_vol + noise
    log_rv_clean = log_vol

    gamma_noisy = compute_empirical_autocovariances(log_rv_noisy, max_lag=1)
    gamma_clean = compute_empirical_autocovariances(log_rv_clean, max_lag=1)

    expected_decrease = omega ** 2  # lag-1 decreases by ω²
    measured_decrease = gamma_clean[1] - gamma_noisy[1]

    assert abs(measured_decrease - expected_decrease) < 0.04, (
        f"Expected lag-1 decrease of {expected_decrease:.3f}, "
        f"measured {measured_decrease:.3f}"
    )


# ---------------------------------------------------------------------------
# fBM simulation
# ---------------------------------------------------------------------------

def test_fbm_simulation_variance():
    """fGn increments should have variance ≈ 1 (ν=1 embedded in the algorithm)."""
    for H in [0.15, 0.25, 0.35]:
        fgn = simulate_fbm_davies_harte(10_000, H, seed=0)
        var = np.var(fgn)
        assert abs(var - 1.0) < 0.05, (
            f"H={H}: fGn variance={var:.4f} should be ≈ 1.0"
        )


def test_fbm_simulation_mean_zero():
    """fGn increments should have mean ≈ 0."""
    fgn = simulate_fbm_davies_harte(20_000, 0.25, seed=1)
    assert abs(np.mean(fgn)) < 0.05


def test_fbm_simulation_hurst_consistency():
    """Simulated fBM should yield H estimates close to the specified H value."""
    for H_true in [0.15, 0.25, 0.35]:
        fgn = simulate_fbm_davies_harte(8_000, H_true, seed=99)
        # Cumulate to get fBM path (the quantity whose differences scale as Δ^(2H))
        fbm = np.concatenate([[0.0], np.cumsum(fgn)])
        result = estimate_hurst_ols(fbm, delta_max=10)
        assert abs(result['H_hat'] - H_true) < 0.07, (
            f"H_true={H_true}: H_hat={result['H_hat']:.3f} should be within 0.07"
        )


def test_fou_log_vol_shape():
    """simulate_fou_log_vol should return array of correct length."""
    n_days = 500
    result = simulate_fou_log_vol(n_days, H=0.20, kappa=0.02, nu=0.5, seed=42)
    assert len(result) == n_days


def test_fou_log_vol_finite():
    """simulate_fou_log_vol output should be finite."""
    result = simulate_fou_log_vol(1_000, H=0.25, kappa=0.01, nu=0.5, seed=7)
    assert np.all(np.isfinite(result)), "fOU log-vol path should be finite"


# ---------------------------------------------------------------------------
# Two-estimator correction
# ---------------------------------------------------------------------------

def test_correction_raises_h():
    """Two-estimator correction should raise H above the raw noisy estimate.

    With high noise on estimator a and low noise on estimator b, the raw H
    from the noisier series is attenuated downward.  The correction exploits
    the cross-series information to recover a larger, less-biased H estimate.
    """
    H_true = 0.25
    n = 4_000
    seed_fbm = 42
    seed_noise = 100

    fgn = simulate_fbm_davies_harte(n, H_true, seed=seed_fbm)
    log_vol = np.concatenate([[0.0], np.cumsum(fgn)])  # fBM path

    rng = np.random.default_rng(seed_noise)
    omega_a = 0.50   # noisier estimator
    omega_b = 0.10   # cleaner estimator
    log_rv_a = log_vol + rng.normal(0.0, omega_a, n + 1)
    log_rv_b = log_vol + rng.normal(0.0, omega_b, n + 1)

    # Raw H from the noisy series (attenuated downward)
    result_raw = estimate_hurst_ols(log_rv_a, delta_max=10)
    H_raw = result_raw['H_hat']

    # Two-estimator correction
    result_corr = fit_two_estimator_correction(log_rv_a, log_rv_b, delta_max=10)
    H_corr = result_corr['H_corrected_a']

    assert H_corr > H_raw, (
        f"Correction should raise H: H_corr={H_corr:.3f} vs H_raw={H_raw:.3f}"
    )
    # Both should be in the rough domain
    assert H_raw < 0.5 and H_corr < 0.5
