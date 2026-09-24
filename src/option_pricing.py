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

class PricingError(ValueError):
    """Unsupported inputs or a price for which no admissible IV was found."""


class UnsupportedPricingDomain(PricingError):
    """An explicitly unsupported model domain."""


class BoundarySolveError(PricingError):
    """The numerical BAW exercise-boundary calculation failed."""


def _validate(S, K, T, r, sigma, option_type, q=0.0):
    if option_type not in ("call", "put"):
        raise PricingError("option_type must be call or put")
    if not np.all(np.isfinite([S, K, T, r, q, sigma])):
        raise PricingError("non_finite_pricing_input")
    if S <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise PricingError("require positive underlying/strike and nonnegative maturity/volatility")


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
    _validate(F, K, T, r, sigma, option_type)
    if sigma == 0.0:
        intrinsic = max(F - K, 0.0) if option_type == 'call' else max(K - F, 0.0)
        return float(np.exp(-r * T) * intrinsic)
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


def _invert(price, pricer, lower_bound, upper_bound):
    if not np.isfinite(price) or price <= 0:
        raise PricingError("nonpositive_or_nonfinite_option_price")
    if price < lower_bound - 1e-10 or price >= upper_bound:
        raise PricingError("option_price_outside_model_bounds")
    try:
        lo, hi = 1e-6, 1.0
        f_lo = pricer(lo) - price
        f_hi = pricer(hi) - price
        while f_hi < 0 and hi < 10:
            hi = min(10.0, hi * 2)
            f_hi = pricer(hi) - price
        if f_lo >= 0 or f_hi < 0:
            raise PricingError("implied_volatility_not_bracketed")
        return float(brentq(lambda v: pricer(v) - price, lo, hi, xtol=1e-10, maxiter=300))
    except PricingError:
        raise
    except (ValueError, RuntimeError, OverflowError) as exc:
        raise PricingError("implied_volatility_solver_failed") from exc


def black76_implied_vol(price: float, F: float, K: float, T: float, r: float,
                         option_type: str = 'call', *, raise_errors: bool = False) -> Optional[float]:
    """Compute implied volatility from a Black-76 option price using Brent's method.

    Args:
        price: observed option price.
        F: futures price.
        K: strike.
        T: time to expiry in years.
        r: risk-free rate.
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Implied volatility, or ``None`` if no solution is found. Set
        ``raise_errors=True`` to retain an explicit rejection reason.
    """
    try:
        _validate(F, K, T, r, 1.0, option_type)
        if T <= 0:
            raise PricingError("expired_option")
        disc = np.exp(-r * T)
        intrinsic = disc * (max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0))
        return _invert(price, lambda v: black76_price(F, K, T, r, v, option_type),
                       intrinsic, disc * (F if option_type == "call" else K))
    except PricingError:
        if raise_errors:
            raise
        return None


# ---------------------------------------------------------------------------
# Barone-Adesi-Whaley (American options)
# ---------------------------------------------------------------------------

def _baw_roots(T, r, q, sigma):
    """Stable roots of Q² + (N-1)Q - M/h = 0, including the r=0 limit."""
    z = r * T
    ratio = 1.0 + z / 2.0 + z * z / 12.0 if abs(z) < 1e-7 else z / -np.expm1(-z)
    b = 2.0 * ratio / (sigma * sigma * T)
    a = 2.0 * (r - q) / (sigma * sigma) - 1.0
    root = np.hypot(a, 2.0 * np.sqrt(b))
    if a >= 0:
        return -0.5 * (a + root), 2.0 * b / (root + a)
    return -2.0 * b / (root - a), 0.5 * (root - a)


def _solve_log_boundary(equation, option_type):
    """Bracket in log(S*/K). Infinite boundaries must be handled analytically."""
    try:
        zero = float(equation(0.0))
        if not np.isfinite(zero):
            raise BoundarySolveError("non_finite_boundary_residual")
        # The ATM continuation premium gives a negative residual on both sides.
        if zero >= 0:
            raise BoundarySolveError("invalid_boundary_at_the_money")
        direction = 1 if option_type == "call" else -1
        endpoint = direction * np.log(2.0)
        for _ in range(10):
            value = float(equation(endpoint))
            if not np.isfinite(value):
                raise BoundarySolveError("non_finite_boundary_residual")
            if value > 0:
                lo, hi = sorted([0.0, endpoint])
                answer = brentq(equation, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=300)
                if not np.isfinite(answer):
                    raise BoundarySolveError("non_finite_boundary_solution")
                return float(answer)
            endpoint *= 2
        raise BoundarySolveError("exercise_boundary_not_bracketed")
    except BoundarySolveError:
        raise
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        raise BoundarySolveError("exercise_boundary_solver_failed") from exc


def _baw_critical_call(K: float, T: float, r: float, q: float,
                        sigma: float, q2: float) -> float:
    """Bisect for the critical call exercise price S* > K."""
    def equation(y):
        d1 = (y + (r - q + sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        a = -np.expm1(-q * T) + np.exp(-q * T) * norm.sf(d1)
        b = -np.expm1(-r * T) + np.exp(-r * T) * norm.sf(d2)
        return (1 - 1 / q2) * a - np.exp(-y) * b
    return float(K * np.exp(_solve_log_boundary(equation, "call")))


def _baw_critical_put(K: float, T: float, r: float, q: float,
                       sigma: float, q1: float) -> float:
    """Bisect for the critical put exercise price S** ∈ (0, K)."""
    def equation(y):
        d1 = (y + (r - q + sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        a = -np.expm1(-r * T) + np.exp(-r * T) * norm.cdf(d2)
        b = -np.expm1(-q * T) + np.exp(-q * T) * norm.cdf(d1)
        return a - np.exp(y) * (1 - 1 / q1) * b
    return float(K * np.exp(_solve_log_boundary(equation, "put")))


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
    _validate(S, K, T, r, sigma, option_type, q)
    if r < 0:
        raise UnsupportedPricingDomain("BAW_negative_rates_unsupported")
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if T == 0:
        return intrinsic
    if sigma == 0:
        raise UnsupportedPricingDomain("BAW_requires_positive_volatility")
    european = (_gbs_call if option_type == "call" else _gbs_put)(S, K, T, r, q, sigma)
    if (option_type == "call" and q <= 0) or (option_type == "put" and r == 0 and q >= 0):
        return european
    q1, q2 = _baw_roots(T, r, q, sigma)
    if option_type == "call":
        star = _baw_critical_call(K, T, r, q, sigma, q2)
        if not np.isfinite(star) or star <= K:
            raise BoundarySolveError("invalid_call_exercise_boundary")
        if S >= star:
            value = intrinsic
        else:
            d1, _ = _gbs_d1d2(star, K, T, r, q, sigma)
            premium = (star / q2) * (-np.expm1(-q * T) + np.exp(-q * T) * norm.sf(d1))
            value = european + premium * np.exp(q2 * np.log(S / star))
    else:
        star = _baw_critical_put(K, T, r, q, sigma, q1)
        if not np.isfinite(star) or not 0 < star < K:
            raise BoundarySolveError("invalid_put_exercise_boundary")
        if S <= star:
            value = intrinsic
        else:
            d1, _ = _gbs_d1d2(star, K, T, r, q, sigma)
            premium = -(star / q1) * (-np.expm1(-q * T) + np.exp(-q * T) * norm.cdf(d1))
            value = european + premium * np.exp(q1 * np.log(S / star))
    tolerance = 1e-10 * max(S, K, 1.0)
    if not np.isfinite(value) or value < max(european, intrinsic) - tolerance:
        raise BoundarySolveError("BAW_price_violates_lower_bound")
    return float(value)


def baw_implied_vol(price: float, S: float, K: float, T: float, r: float,
                    q: float, option_type: str = 'call', *, raise_errors: bool = False) -> Optional[float]:
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
        Implied volatility, or ``None`` if no solution is found. Set
        ``raise_errors=True`` to retain an explicit rejection reason.
    """
    try:
        _validate(S, K, T, r, 1.0, option_type, q)
        if r < 0:
            raise UnsupportedPricingDomain("BAW_negative_rates_unsupported")
        if T <= 0:
            raise PricingError("expired_option")
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        upper = S * max(1.0, np.exp(-q * T)) if option_type == "call" else K
        return _invert(price, lambda v: baw_approximation(S, K, T, r, q, v, option_type), intrinsic, upper)
    except PricingError:
        if raise_errors:
            raise
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
        D = −slope  (must be positive; D > 1 is supported)
        F = intercept / D

    Args:
        calls: array of call prices.
        puts:  array of put prices.
        strikes: array of strike prices.

    Returns:
        ``(F_estimate, D_estimate)`` tuple.
    """
    result = fit_put_call_parity(calls, puts, strikes)
    return result["forward"], result["discount"]


def fit_put_call_parity(calls, puts, strikes, *, known_forward: Optional[float] = None) -> dict:
    """OLS parity with rank checks, no rate clipping and explicit diagnostics.

    CME: C-P = D(F-K), with supplied F and no intercept.
    OPRA: C-P = DF-DK. On American prices this is an approximation, not an identity.
    """
    c, p, k = (np.asarray(x, dtype=float) for x in (calls, puts, strikes))
    if c.ndim != 1 or p.shape != c.shape or k.shape != c.shape:
        raise PricingError("parity_array_shape_mismatch")
    if not np.all(np.isfinite(np.r_[c, p, k])) or np.any(k <= 0) or np.any(c < 0) or np.any(p < 0):
        raise PricingError("invalid_parity_inputs")
    if len(k) < 2 or len(np.unique(k)) < 2:
        raise PricingError("parity_requires_two_distinct_strikes")
    y = c - p
    if known_forward is not None:
        if not np.isfinite(known_forward) or known_forward <= 0:
            raise PricingError("invalid_known_forward")
        design = (known_forward - k)[:, None]
        mode = "known_forward_no_intercept"
    else:
        center, scale = float(k.mean()), float(np.ptp(k))
        design = np.column_stack([np.ones(len(k)), (k - center) / scale])
        mode = "joint_forward_discount"
    coef, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise PricingError("rank_deficient_parity_regression")
    residual = y - design @ coef
    dof = len(k) - rank
    mse = float(residual @ residual / dof) if dof > 0 else np.nan
    cov = mse * np.linalg.inv(design.T @ design)
    if known_forward is not None:
        D, F = float(coef[0]), float(known_forward)
        d_se, f_se = float(np.sqrt(max(cov[0, 0], 0))), 0.0
        ss_tot = float(y @ y)
        r2_kind = "uncentered"
    else:
        D = float(-coef[1] / scale)
        if not np.isfinite(D) or D <= 0:
            raise PricingError("nonpositive_or_nonfinite_discount")
        F = float(center + coef[0] / D)
        d_se = float(np.sqrt(max(cov[1, 1], 0)) / scale)
        grad = np.array([1 / D, coef[0] / (scale * D * D)])
        f_se = float(np.sqrt(max(grad @ cov @ grad, 0)))
        ss_tot = float(np.sum((y - y.mean())**2))
        r2_kind = "centered"
    if not np.isfinite(D) or D <= 0 or not np.isfinite(F) or F <= 0:
        raise PricingError("nonpositive_or_nonfinite_forward_or_discount")
    return {"forward": F, "discount": D, "forward_se": f_se, "discount_se": d_se,
            "r_squared": 1 - float(residual @ residual) / ss_tot if ss_tot > 0 else np.nan,
            "r_squared_kind": r2_kind, "rmse": float(np.sqrt(np.mean(residual**2))),
            "rank": int(rank), "n_pairs": len(k), "n_unique_strikes": len(np.unique(k)),
            "condition_number": float(singular[0] / singular[-1]), "mode": mode}
