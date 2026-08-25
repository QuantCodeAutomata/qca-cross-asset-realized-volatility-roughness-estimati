"""
Tests for fOU spectral analysis (exp_2) and mean-reversion bias.
Covers: exact M_2 computation, asymptotic slope, OLS bias.
"""
import numpy as np
import pytest

from src.fou_spectral import compute_exact_m2, compute_m2_grid, local_slope_asymptotic
from src.mean_reversion_bias import compute_ols_bias_approximation, compute_exact_ols_h


# ---------------------------------------------------------------------------
# compute_exact_m2 tests
# ---------------------------------------------------------------------------

def test_m2_kappa_zero_scaling():
    """When kappa=0, M_2(Δ) should scale exactly as Δ^(2H).

    For the fOU with no mean reversion, the process reduces to fractional
    Brownian motion: M_2(Δ) = ν² · Δ^(2H).
    """
    H = 0.25
    nu = 1.0
    kappa = 0.0
    deltas = np.array([1.0, 2.0, 5.0, 10.0])

    m2 = compute_m2_grid(deltas, H, kappa, nu)
    expected = nu ** 2 * deltas ** (2 * H)

    np.testing.assert_allclose(
        m2, expected, rtol=1e-3,
        err_msg="M_2(Δ) should equal Δ^(2H) for kappa=0"
    )


def test_m2_kappa_zero_various_h():
    """M_2(Δ=1) = ν² for kappa=0, regardless of H."""
    nu = 2.5
    for H in [0.10, 0.20, 0.30, 0.40]:
        m2_val = compute_exact_m2(1.0, H, kappa=0.0, nu=nu)
        assert abs(m2_val - nu ** 2) < 0.005, (
            f"H={H}: M_2(1) = {m2_val:.5f} should be ν²={nu**2:.4f}"
        )


def test_m2_positive():
    """M_2(Δ) should be strictly positive for all valid parameters."""
    params = [
        (0.10, 0.00, 1.0),
        (0.20, 0.01, 1.0),
        (0.25, 0.05, 2.0),
        (0.30, 0.10, 0.5),
        (0.40, 0.00, 1.0),
    ]
    for H, kappa, nu in params:
        for Delta in [1.0, 5.0, 20.0]:
            val = compute_exact_m2(Delta, H, kappa, nu)
            assert val > 0.0, (
                f"M_2({Delta}) < 0 for H={H}, kappa={kappa}, nu={nu}: {val}"
            )




def test_m2_supports_smooth_h_above_half_without_endpoint_failure():
    """The spectral integral is finite on the whole H in (0, 1)."""
    values = [compute_exact_m2(delta, H=0.80, kappa=0.01) for delta in (1.0, 2.0, 5.0)]
    assert np.all(np.isfinite(values))
    assert np.all(np.asarray(values) > 0.0)
    assert values[0] < values[1] < values[2]

def test_m2_mean_reversion_reduces_variance():
    """Mean reversion (kappa>0) should reduce M_2(Δ) relative to kappa=0."""
    H, nu = 0.25, 1.0
    deltas = np.array([1.0, 5.0, 10.0])

    m2_no_mr = compute_m2_grid(deltas, H, kappa=0.0, nu=nu)
    m2_mr = compute_m2_grid(deltas, H, kappa=0.10, nu=nu)

    assert np.all(m2_mr <= m2_no_mr * 1.01), (
        "Mean reversion should not increase M_2 relative to kappa=0"
    )


# ---------------------------------------------------------------------------
# OLS bias (exact and approximate)
# ---------------------------------------------------------------------------

def test_benchmark_cell_h_estimate():
    """Benchmark cell (H=0.20, kappa=0.010): H_hat should be ~0.198 at delta_max=10.

    For small kappa·Δ_max = 0.1, the OLS estimate should be very close to
    the true H, with bias < 0.005.
    """
    H_true = 0.20
    kappa = 0.010
    H_hat, bias = compute_exact_ols_h(H_true, kappa, delta_max=10, nu=1.0)

    assert abs(H_hat - 0.198) < 0.005, (
        f"Benchmark cell: H_hat={H_hat:.4f} should be ≈ 0.198"
    )
    assert abs(bias) < 0.005, (
        f"Benchmark cell: bias={bias:.4f} should be < 0.005"
    )


def test_bias_negligible_at_short_lag():
    """For kappa ≤ 0.02, bias at delta_max=10 should be ≤ 0.01.

    Theoretical bias scales as O(kappa * Delta_max).  For kappa=0.02 and
    delta_max=10 the product is 0.20, allowing bias up to ~0.008 (verified
    via exact spectral integration; see RESULTS.md Exp 2).
    """
    for H in [0.15, 0.20, 0.30]:
        for kappa in [0.005, 0.010, 0.020]:
            H_hat, bias = compute_exact_ols_h(H, kappa, delta_max=10, nu=1.0)
            assert abs(bias) <= 0.01, (
                f"H={H}, kappa={kappa}: bias={bias:.4f} should be ≤ 0.01"
            )


def test_bias_grows_with_kappa():
    """Larger kappa should produce larger (negative) bias in H_hat."""
    H = 0.25
    _, bias_small = compute_exact_ols_h(H, kappa=0.01, delta_max=10)
    _, bias_large = compute_exact_ols_h(H, kappa=0.05, delta_max=10)

    assert abs(bias_large) >= abs(bias_small), (
        f"Larger kappa should give larger bias: "
        f"|bias(0.05)|={abs(bias_large):.4f}, |bias(0.01)|={abs(bias_small):.4f}"
    )


def test_asymptotic_expansion_leading_order():
    """Verify the asymptotic expansion is accurate for small kappa·Δ.

    For kappa·Δ = 0.005 (very small), the local slope from the asymptotic
    formula and the exact numerical derivative should agree within 0.01.
    """
    H = 0.25
    kappa = 0.005
    Delta = 1.0   # kappa*Delta = 0.005  (well within asymptotic regime)

    # Exact local slope via finite difference on log M_2
    eps = 0.001
    m2_hi = compute_exact_m2(Delta * (1.0 + eps), H, kappa)
    m2_lo = compute_exact_m2(Delta * (1.0 - eps), H, kappa)
    slope_exact = (np.log(m2_hi) - np.log(m2_lo)) / (2.0 * np.log(1.0 + eps))

    slope_asymp = local_slope_asymptotic(Delta, H, kappa)

    assert abs(slope_exact - slope_asymp) < 0.01, (
        f"Asymptotic slope={slope_asymp:.4f} vs exact={slope_exact:.4f}; "
        f"difference={abs(slope_exact-slope_asymp):.4f} exceeds 0.01"
    )


def test_asymptotic_expansion_kappa_zero_limit():
    """local_slope_asymptotic(kappa=0) should return exactly 2H."""
    for H in [0.10, 0.25, 0.40]:
        slope = local_slope_asymptotic(1.0, H, kappa=0.0)
        assert abs(slope - 2 * H) < 1e-10, (
            f"H={H}: slope={slope} should equal 2H={2*H}"
        )


def test_ols_bias_approximation_kappa_zero():
    """Bias approximation should be 0 for kappa=0."""
    for H in [0.15, 0.25, 0.35]:
        bias = compute_ols_bias_approximation(H, kappa=0.0, delta_max=10)
        assert bias == 0.0, f"Bias should be 0 for kappa=0, got {bias}"


def test_ols_bias_approximation_negative():
    """Bias approximation should be negative (H_hat < H) for kappa > 0."""
    for H in [0.15, 0.25, 0.35]:
        bias = compute_ols_bias_approximation(H, kappa=0.05, delta_max=10)
        assert bias < 0.0, f"Bias should be < 0 for kappa=0.05, got {bias}"
