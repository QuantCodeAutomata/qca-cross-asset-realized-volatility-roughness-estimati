"""
Fractional Ornstein-Uhlenbeck (fOU) Spectral Density and Exact Second Moment Computation.
Used in exp_2: Mean-reversion contamination of short-lag RV Hurst estimation.

Model: dX_t = -kappa * X_t * dt + nu * dB_t^H
Spectral density: S(lambda) = [nu^2 * Gamma(2H+1) * sin(H*pi) / (2*pi)] * |lambda|^(1-2H) / (lambda^2 + kappa^2)
"""
import numpy as np
from scipy import integrate, special


def spectral_density(lambda_: np.ndarray, H: float, kappa: float, nu: float = 1.0) -> np.ndarray:
    """Compute fOU spectral density S(lambda).

    S(lambda) = [nu^2 * Gamma(2H+1) * sin(H*pi) / (2*pi)] * |lambda|^(1-2H) / (lambda^2 + kappa^2)
    For kappa=0, reduces to fBM spectral density.

    Args:
        lambda_: frequency array (positive values)
        H: Hurst exponent in (0,1)
        kappa: mean reversion rate >= 0
        nu: volatility of volatility (scale parameter)

    Returns:
        S(lambda) values
    """
    lambda_ = np.asarray(lambda_, dtype=np.float64)
    C = nu**2 * special.gamma(2 * H + 1) * np.sin(H * np.pi) / (2 * np.pi)
    abs_lam = np.abs(lambda_)
    if kappa == 0.0:
        # fBM spectral density: S(lambda) = C * |lambda|^{-(1+2H)}
        return C * abs_lam ** (-(1 + 2 * H))
    return C * abs_lam ** (1 - 2 * H) / (lambda_**2 + kappa**2)


def compute_exact_m2(Delta: float, H: float, kappa: float, nu: float = 1.0,
                      tol: float = 1e-10) -> float:
    """Compute exact second moment M_2(Delta) = E[(X_{t+Delta} - X_t)^2] via spectral integration.

    M_2(Delta) = 4 * integral_0^infinity (1 - cos(Delta * lambda)) * S(lambda) d_lambda

    For kappa=0 (pure fBM), the analytical result M_2(Delta) = nu^2 * Delta^(2H) is used.
    For kappa>0 (fOU), the integral is split as:
        M_2 = 4 * [integral_0^∞ S(λ) dλ - integral_0^∞ cos(Δλ) S(λ) dλ]
    The first term is evaluated analytically.  The covariance integral is
    regularised near zero and QAWF is used only on the oscillatory tail.

    Args:
        Delta: lag in trading days
        H: Hurst exponent
        kappa: mean reversion rate
        nu: volatility of volatility
        tol: integration tolerance

    Returns:
        M_2(Delta) scalar
    """
    if Delta <= 0.0:
        raise ValueError("Delta must be positive")
    if not (0.0 < H < 1.0):
        raise ValueError("H must lie in (0,1)")
    if kappa < 0.0 or nu <= 0.0:
        raise ValueError("require kappa>=0 and nu>0")

    # Exact analytical formula for the pure-fBM case:
    # M_2(Delta) = nu^2 * Delta^(2H)
    if kappa == 0.0:
        return nu ** 2 * Delta ** (2.0 * H)

    # For fOU (kappa > 0): split M_2 = 4*(gamma0_half - gammaD_half)
    # where gamma0_half = ∫_0^∞ S(λ) dλ  and  gammaD_half = ∫_0^∞ cos(Δλ) S(λ) dλ.
    C = nu ** 2 * special.gamma(2.0 * H + 1.0) * np.sin(H * np.pi) / (2.0 * np.pi)

    def S(lam: float) -> float:
        return C * lam ** (1.0 - 2.0 * H) / (lam ** 2 + kappa ** 2)

    # The non-oscillatory half-line integral is available in closed form:
    #   ∫_0^∞ λ^(1-2H)/(λ²+κ²) dλ
    #       = π κ^(-2H) / (2 sin(πH)).
    # Multiplying by C gives half of the stationary variance.  Using the
    # identity removes an avoidable endpoint/round-off warning and prevents
    # cancellation error in the small-kappa cells.
    gamma0_half = (
        nu ** 2
        * special.gamma(2.0 * H + 1.0)
        / (4.0 * kappa ** (2.0 * H))
    )

    # Regularise the integrable origin singularity with
    # lambda = u**p, p = 1/(2-2H), on [0, 1].  The powers cancel exactly:
    # S(lambda) d lambda = C*p/(u**(2p)+kappa**2) du.  This keeps the
    # calculation stable for the whole H in (0, 1), including H > 1/2.
    p = 1.0 / (2.0 - 2.0 * H)

    def covariance_near_zero(u: float) -> float:
        lam = u**p
        return C * p * np.cos(Delta * lam) / (u ** (2.0 * p) + kappa**2)

    gamma_near, _ = integrate.quad(
        covariance_near_zero, 0.0, 1.0,
        limit=1000, epsabs=tol, epsrel=tol,
    )
    gamma_tail, _ = integrate.quad(
        S, 1.0, np.inf,
        weight="cos", wvar=Delta,
        limit=1000, limlst=500,
        epsabs=tol, epsrel=tol,
    )
    gammaD_half = gamma_near + gamma_tail

    return 4.0 * (gamma0_half - gammaD_half)


def compute_m2_grid(deltas: np.ndarray, H: float, kappa: float, nu: float = 1.0) -> np.ndarray:
    """Compute M_2(Delta) for an array of lags.

    Args:
        deltas: array of lag values [1, 2, ..., delta_max]
        H: Hurst exponent
        kappa: mean reversion rate
        nu: volatility

    Returns:
        Array of M_2 values
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    return np.array([compute_exact_m2(d, H, kappa, nu) for d in deltas])


def local_slope_asymptotic(Delta: float, H: float, kappa: float) -> float:
    """Compute the asymptotic local log-log slope.

    alpha(Delta) = 2H - (1-H)*Gamma(1+2H)*(kappa*Delta)^(2-2H) + o((kappa*Delta)^(2-2H))

    Derivation: M_2(Delta) ≈ C * Delta^(2H) * [1 - Gamma(1+2H)/2 * (kappa*Delta)^(2-2H)]
    Taking d/d(log Delta) and expanding to first order in (kappa*Delta)^(2-2H) gives the result.

    Args:
        Delta: lag value
        H: Hurst exponent
        kappa: mean reversion rate

    Returns:
        alpha(Delta) scalar
    """
    if kappa == 0.0:
        return 2.0 * H
    correction = (1.0 - H) * special.gamma(1.0 + 2.0 * H) * (kappa * Delta) ** (2.0 - 2.0 * H)
    return 2.0 * H - correction
