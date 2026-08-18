"""
Tests for option-implied Hurst estimation (exp_4).
Covers: Black-76 pricing, ATM skew estimation, pooled implied-H regression.
"""
import numpy as np
import pandas as pd
import pytest

from src.option_pricing import black76_price, black76_implied_vol
from src.atm_skew_estimator import compute_log_moneyness, estimate_atm_skew
from src.implied_hurst import estimate_pooled_implied_hurst, estimate_daily_implied_hurst


# ---------------------------------------------------------------------------
# Black-76 model tests
# ---------------------------------------------------------------------------

def test_black76_call_positive():
    """Black-76 call price should be positive for any positive sigma."""
    price = black76_price(F=100.0, K=100.0, T=0.5, r=0.03, sigma=0.20, option_type='call')
    assert price > 0.0


def test_black76_put_call_parity():
    """Black-76 call and put must satisfy put-call parity.

    C − P = exp(−rT) · (F − K)
    """
    F, K, T, r, sigma = 105.0, 100.0, 0.25, 0.02, 0.25

    call = black76_price(F, K, T, r, sigma, option_type='call')
    put = black76_price(F, K, T, r, sigma, option_type='put')

    lhs = call - put
    rhs = np.exp(-r * T) * (F - K)

    assert abs(lhs - rhs) < 1e-10, (
        f"PCP failed: C-P={lhs:.8f}, exp(-rT)*(F-K)={rhs:.8f}"
    )


def test_black76_put_call_parity_atm():
    """PCP holds at-the-money (F = K)."""
    F = K = 100.0
    T, r, sigma = 0.5, 0.05, 0.30

    call = black76_price(F, K, T, r, sigma, 'call')
    put = black76_price(F, K, T, r, sigma, 'put')

    # C - P = exp(-rT)*(F - K) = 0 when F = K
    assert abs(call - put) < 1e-10, (
        f"ATM: C={call:.8f}, P={put:.8f}, diff={call-put:.2e}"
    )


def test_black76_implied_vol_roundtrip():
    """Implied vol computed from Black-76 price should round-trip to the same price."""
    F, K, T, r = 100.0, 98.0, 0.5, 0.03
    sigma_true = 0.22

    for option_type in ('call', 'put'):
        price = black76_price(F, K, T, r, sigma_true, option_type)
        sigma_implied = black76_implied_vol(price, F, K, T, r, option_type)

        assert sigma_implied is not None, f"IV inversion returned None for {option_type}"
        assert abs(sigma_implied - sigma_true) < 1e-7, (
            f"{option_type}: σ_implied={sigma_implied:.8f} vs σ_true={sigma_true}"
        )


def test_black76_implied_vol_call_put_same_iv():
    """European call and put with same inputs should imply the same volatility."""
    F, K, T, r, sigma = 100.0, 102.0, 0.25, 0.02, 0.30

    price_call = black76_price(F, K, T, r, sigma, 'call')
    price_put = black76_price(F, K, T, r, sigma, 'put')

    iv_call = black76_implied_vol(price_call, F, K, T, r, 'call')
    iv_put = black76_implied_vol(price_put, F, K, T, r, 'put')

    assert iv_call is not None and iv_put is not None
    assert abs(iv_call - iv_put) < 1e-6, (
        f"Call IV={iv_call:.6f} vs put IV={iv_put:.6f} should agree"
    )


def test_black76_implied_vol_returns_none_below_intrinsic():
    """Should return None when price is below intrinsic value."""
    # Deep ITM call: intrinsic = exp(-rT)*(F-K), set price below that
    F, K, T, r = 120.0, 100.0, 0.25, 0.02
    intrinsic = np.exp(-r * T) * (F - K)
    below_intrinsic = intrinsic * 0.5

    iv = black76_implied_vol(below_intrinsic, F, K, T, r, 'call')
    assert iv is None


# ---------------------------------------------------------------------------
# ATM skew estimator tests
# ---------------------------------------------------------------------------

def test_compute_log_moneyness_atm_is_zero():
    """Log-moneyness at K = F should be zero."""
    F = 100.0
    km = compute_log_moneyness(np.array([F]), F)
    assert km[0] == 0.0


def test_compute_log_moneyness_otm_call_positive():
    """Log-moneyness should be positive for OTM calls (K > F)."""
    km = compute_log_moneyness(np.array([110.0]), forward=100.0)
    assert km[0] > 0.0


def test_atm_skew_linear_smile():
    """For a linearly decreasing smile, the estimated skew should match the slope.

    IV(k) = IV_ATM + slope · k → slope = dIV/dk|_{k=0} = psi
    """
    F = 100.0
    slope_true = -0.10   # typical equity-skew sign

    # Dense grid of strikes near ATM
    k_vals = np.linspace(-0.09, 0.09, 50)
    strikes = F * np.exp(k_vals)
    ivols = 0.20 + slope_true * k_vals  # exact linear smile

    result = estimate_atm_skew(strikes, ivols, F,
                                moneyness_band=0.08, min_strikes=10)

    assert result is not None, "Skew estimation should succeed with 50 strikes"
    assert abs(result['psi'] - slope_true) < 0.005, (
        f"Estimated psi={result['psi']:.4f} vs slope={slope_true}"
    )
    assert result['r_squared'] > 0.999, (
        f"R² should be near 1 for exact linear smile: {result['r_squared']:.5f}"
    )


def test_atm_skew_insufficient_strikes():
    """Should return None when not enough strikes pass the moneyness filter."""
    F = 100.0
    # All strikes far OTM (|k| >> 0.08)
    strikes = np.array([60.0, 65.0, 70.0, 75.0, 130.0, 135.0, 140.0, 145.0])
    ivols = np.array([0.45, 0.40, 0.35, 0.30, 0.22, 0.24, 0.26, 0.28])

    result = estimate_atm_skew(strikes, ivols, F,
                                moneyness_band=0.08, min_strikes=10)
    assert result is None, "Should return None with all strikes outside band"


def test_atm_skew_returns_dict_keys():
    """Result should contain all required keys when estimation succeeds."""
    F = 100.0
    k_vals = np.linspace(-0.07, 0.07, 30)
    strikes = F * np.exp(k_vals)
    ivols = 0.20 - 0.08 * k_vals

    result = estimate_atm_skew(strikes, ivols, F,
                                moneyness_band=0.08, min_strikes=10)

    assert result is not None
    for key in ('psi', 'atm_iv', 'r_squared', 'n_strikes', 'valid'):
        assert key in result, f"Missing key: {key}"
    assert result['valid'] is True


# ---------------------------------------------------------------------------
# Implied Hurst tests
# ---------------------------------------------------------------------------

def _make_power_law_skew_df(H: float, c: float = 0.5, n_dates: int = 300,
                              noise_std: float = 0.03, seed: int = 42
                              ) -> pd.DataFrame:
    """Construct synthetic skew term structure with known power-law scaling."""
    rng = np.random.default_rng(seed)
    T_values = np.array([14, 21, 30, 60, 90, 120, 180, 252]) / 252.0
    dates = pd.date_range('2020-01-02', periods=n_dates, freq='B')

    rows = []
    for date in dates:
        for T in T_values:
            psi = -c * T ** (H - 0.5) * np.exp(rng.normal(0.0, noise_std))
            rows.append({'date': date, 'T': T, 'psi': psi, 'valid': True})

    return pd.DataFrame(rows)


def test_implied_hurst_from_power_law_skew():
    """For |ψ(T)| = c·T^(H−0.5), pooled regression should recover H.

    Uses within-group demeaning to absorb date fixed effects and then
    regresses demeaned log|ψ| on demeaned log T.
    """
    H_true = 0.25
    skew_ts = _make_power_law_skew_df(H_true, c=0.50, n_dates=300,
                                       noise_std=0.03, seed=42)

    result = estimate_pooled_implied_hurst(skew_ts, maturity_window=(14 / 365, 1.0),
                                            min_obs_per_date=2)

    assert result['n_obs'] > 100
    assert abs(result['H_hat_IV'] - H_true) < 0.04, (
        f"H_hat={result['H_hat_IV']:.4f} should be within 0.04 of H={H_true}"
    )
    assert result['identified'], (
        f"Should be identified (R²_ψ={result['r_psi_squared']:.3f} should ≥ 0.3)"
    )
    assert result['r_psi_squared'] >= 0.3


def test_identification_threshold():
    """R²_ψ < 0.3 should mark 'identified = False'.

    Pure noise skew data has no power-law structure, so the within-group
    regression should produce near-zero R² and fail the identification test.
    """
    rng = np.random.default_rng(77)
    n_dates = 200
    dates = pd.date_range('2020-01-02', periods=n_dates, freq='B')
    T_values = np.array([30, 60, 90, 120, 180]) / 252.0

    rows = []
    for date in dates:
        for T in T_values:
            # Pure i.i.d. noise: no relationship with T
            psi = rng.normal(0.0, 0.005)
            rows.append({'date': date, 'T': T, 'psi': psi, 'valid': True})

    skew_ts = pd.DataFrame(rows)
    result = estimate_pooled_implied_hurst(skew_ts)

    assert not result['identified'], (
        f"Pure-noise skew should not be identified: "
        f"R²_ψ={result['r_psi_squared']:.3f}"
    )


def test_daily_implied_hurst_recovery():
    """estimate_daily_implied_hurst should recover H close to the true value."""
    H_true = 0.20
    skew_ts = _make_power_law_skew_df(H_true, c=0.60, n_dates=10,
                                       noise_std=0.01, seed=55)

    date = skew_ts['date'].unique()[0]
    result = estimate_daily_implied_hurst(skew_ts, date,
                                           maturity_window=(14 / 365, 1.0))

    assert result is not None
    assert 'H_hat' in result
    # With low noise, single-date estimate should be reasonably close
    assert abs(result['H_hat'] - H_true) < 0.10, (
        f"Daily H_hat={result['H_hat']:.3f} vs H_true={H_true}"
    )


def test_pooled_h_greater_than_daily_average_stability():
    """Pooled estimator should produce a valid H in (0, 1) for clean data."""
    H_true = 0.30
    skew_ts = _make_power_law_skew_df(H_true, n_dates=100, noise_std=0.02, seed=13)
    result = estimate_pooled_implied_hurst(skew_ts)

    assert 0.0 < result['H_hat_IV'] < 1.0
    assert result['n_dates'] > 0
