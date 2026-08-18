"""
Option pricing utilities for implied volatility computation.
Exp_4: Option-implied Hurst estimation from ATM skew term structure.
Custom implementation -- Context7 found no exact library for all required models.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gbs_d1d2(S: float, K: float, T: float, r: float, q: float,
               sigma: float) -> Tuple[float, float]:
    """d1, d2 for generalized Black-Scholes (continuous dividend yield q)."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _gbs_call(S: float, K: float, T: float, r: float, q: float,
               sigma: float) -> float:
    """European call via generalized Black-Scholes."""
    if T <= 0.0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1, d2 = _gbs_d1d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _gbs_put(S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
    """European put via generalized Black-Scholes."""
    if T <= 0.0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1, d2 = _gbs_d1d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Black-76 (futures options)
# ---------------------------------------------------------------------------

def black76_price(F: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = 'call') -> float:
    """Black-76 option price for futures / forward options.

    Args:
        F: futures price (forward).
        K: strike price.
        T: time to expiry in years.
        r: risk-free rate (continuous compounding).
        sigma: implied volatility.
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Option price.
    """
    if T <= 0.0:
        if option_type == 'call':
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = np.exp(-r * T)

    if option_type == 'call':
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    # put
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_implied_vol(price: float, F: float, K: float, T: float, r: float,
                         option_type: str = 'call') -> Optional[float]:
    """Compute implied volatility from a Black-76 option price using Brent's method.

    Args:
        price: observed option price.
        F: futures price.
        K: strike.
        T: time to expiry in years.
        r: risk-free rate.
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Implied volatility, or ``None`` if no solution is found.
    """
    if T <= 0.0 or price <= 0.0:
        return None

    def objective(sigma: float) -> float:
        return black76_price(F, K, T, r, sigma, option_type) - price

    # Intrinsic value check
    disc = np.exp(-r * T)
    if option_type == 'call':
        intrinsic = disc * max(F - K, 0.0)
    else:
        intrinsic = disc * max(K - F, 0.0)

    if price < intrinsic - 1e-8:
        return None

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=1e-10, maxiter=500)
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Barone-Adesi-Whaley (American options)
# ---------------------------------------------------------------------------

def _baw_critical_call(K: float, T: float, r: float, q: float,
                        sigma: float, q2: float) -> float:
    """Bisect for the critical call exercise price S* > K."""
    def equation(s: float) -> float:
        d1_s, _ = _gbs_d1d2(s, K, T, r, q, sigma)
        bs = _gbs_call(s, K, T, r, q, sigma)
        alpha = (s / q2) * (1.0 - np.exp(-q * T) * norm.cdf(d1_s))
        return (s - K) - bs - alpha

    lo, hi = K * (1.0 + 1e-6), K * 1e4
    # Ensure sign change; if equation(hi) < 0, no early exercise
    try:
        if equation(lo) >= 0.0:
            return lo
        if equation(hi) <= 0.0:
            return hi * 1e4  # effectively ∞ → no early exercise
        return brentq(equation, lo, hi, xtol=1e-8, maxiter=500)
    except (ValueError, RuntimeError):
        return hi * 1e4


def _baw_critical_put(K: float, T: float, r: float, q: float,
                       sigma: float, q1: float) -> float:
    """Bisect for the critical put exercise price S** ∈ (0, K)."""
    def equation(s: float) -> float:
        d1_s, _ = _gbs_d1d2(s, K, T, r, q, sigma)
        bs = _gbs_put(s, K, T, r, q, sigma)
        # q1 < 0  →  -s/q1 > 0
        alpha = -(s / q1) * (1.0 - np.exp(-q * T) * norm.cdf(-d1_s))
        return (K - s) - bs - alpha

    lo, hi = K * 1e-6, K * (1.0 - 1e-7)
    try:
        if equation(lo) <= 0.0:
            return lo
        if equation(hi) >= 0.0:
            return 0.0  # no early exercise
        return brentq(equation, lo, hi, xtol=1e-8, maxiter=500)
    except (ValueError, RuntimeError):
        return 0.0


def baw_approximation(S: float, K: float, T: float, r: float, q: float,
                       sigma: float, option_type: str = 'call') -> float:
    """Barone-Adesi-Whaley quadratic approximation for American options.

    Implemented for both calls and puts.  For calls with zero dividend (q=0)
    the European price is returned directly (never optimal to exercise early).

    Args:
        S: spot price.
        K: strike.
        T: time to expiry in years.
        r: risk-free rate (continuous).
        q: continuous dividend yield.
        sigma: volatility.
        option_type: ``'call'`` or ``'put'``.

    Returns:
        American option price approximation.
    """
    if T <= 0.0:
        if option_type == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # Quadratic equation parameters
    M = 2.0 * r / sigma ** 2
    N_ = 2.0 * (r - q) / sigma ** 2
    h = 1.0 - np.exp(-r * T)
    discriminant = (N_ - 1.0) ** 2 + 4.0 * M / h

    if option_type == 'call':
        if q == 0.0:
            # No early exercise for calls with no dividends
            return _gbs_call(S, K, T, r, q, sigma)

        q2 = (-(N_ - 1.0) + np.sqrt(discriminant)) / 2.0
        s_star = _baw_critical_call(K, T, r, q, sigma, q2)

        if S >= s_star:
            return S - K  # immediate exercise is optimal

        d1_star, _ = _gbs_d1d2(s_star, K, T, r, q, sigma)
        A2 = (s_star / q2) * (1.0 - np.exp(-q * T) * norm.cdf(d1_star))
        return _gbs_call(S, K, T, r, q, sigma) + A2 * (S / s_star) ** q2

    else:  # put
        q1 = (-(N_ - 1.0) - np.sqrt(discriminant)) / 2.0  # q1 < 0
        s_star = _baw_critical_put(K, T, r, q, sigma, q1)

        if s_star <= 0.0 or S <= s_star:
            return max(K - S, 0.0)

        d1_star, _ = _gbs_d1d2(s_star, K, T, r, q, sigma)
        # A1 > 0: q1 < 0 → -s_star/q1 > 0
        A1 = -(s_star / q1) * (1.0 - np.exp(-q * T) * norm.cdf(-d1_star))
        return _gbs_put(S, K, T, r, q, sigma) + A1 * (S / s_star) ** q1


def baw_implied_vol(price: float, S: float, K: float, T: float, r: float,
                    q: float, option_type: str = 'call') -> Optional[float]:
    """Compute implied volatility from a BAW option price using Brent's method.

    Args:
        price: observed American option price.
        S: spot price.
        K: strike.
        T: time to expiry in years.
        r: risk-free rate.
        q: dividend yield.
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Implied volatility, or ``None`` if no solution is found.
    """
    if T <= 0.0 or price <= 0.0:
        return None

    def objective(sigma: float) -> float:
        return baw_approximation(S, K, T, r, q, sigma, option_type) - price

    # Intrinsic-value bounds
    if option_type == 'call':
        intrinsic = max(S - K, 0.0)
    else:
        intrinsic = max(K - S, 0.0)

    if price < intrinsic - 1e-8:
        return None

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=1e-10, maxiter=500)
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Forward / discount inference from put-call parity
# ---------------------------------------------------------------------------

def infer_forward_and_discount_pcp(calls: np.ndarray, puts: np.ndarray,
                                    strikes: np.ndarray) -> Tuple[float, float]:
    """Infer forward price F and discount factor D from put-call parity.

    For European / futures options:
        C − P = D · (F − K)

    This is linearised as:
        (C − P) = D · F − D · K

    Fit by OLS to get slope (= −D) and intercept (= D · F), then:
        D = −slope  (clamped to (0, 1])
        F = intercept / D

    Args:
        calls: array of call prices.
        puts:  array of put prices.
        strikes: array of strike prices.

    Returns:
        ``(F_estimate, D_estimate)`` tuple.
    """
    calls = np.asarray(calls, dtype=np.float64)
    puts = np.asarray(puts, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)

    cp_diff = calls - puts   # D*(F-K) = D*F - D*K

    # OLS: cp_diff = a - b*K  →  a = D*F, b = D
    A = np.column_stack([np.ones(len(strikes)), -strikes])
    coeff, _, _, _ = np.linalg.lstsq(A, cp_diff, rcond=None)
    DF = float(coeff[0])   # D * F
    D = float(coeff[1])    # D

    D = np.clip(D, 1e-6, 1.0)
    F = DF / D

    return float(F), float(D)
