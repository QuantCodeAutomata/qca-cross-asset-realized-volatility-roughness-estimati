"""
Seven realized-variance estimators for high-frequency intraday return data.

All functions accept a 1-D numpy array of *within-session* log returns and
return a non-negative float64 scalar (the daily realized variance estimate).

Estimators
----------
1. RV5m   - 5-minute subsampled realized variance
2. RK     - Realized Kernel with Parzen weights (Barndorff-Nielsen et al. 2008)
3. TSRV   - Two-Scale Realized Variance (Zhang et al. 2005)
4. BPV    - Bipower Variation (Barndorff-Nielsen & Shephard 2004)
5. PAV    - Pre-Averaged Variance (Jacod et al. 2009)
6. PABPV  - Pre-Averaged Bipower Variation (Jacod et al. 2009)
7. C-TRV  - Corrected Threshold Realized Variance (Corsi et al. 2010)
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Pre-averaging weight constants
# ---------------------------------------------------------------------------

def compute_psi_constants() -> Tuple[float, float]:
    """Compute the pre-averaging weight constants for g(x) = min(x, 1-x).

    psi_1 = integral_0^1 g(x)^2 dx
      Analytic: two equal integrals over [0,1/2] and [1/2,1], each = 1/24,
      giving psi_1 = 1/12.
    psi_2 = integral_0^1 (g\'(x))^2 dx
      g\'(x) = +1 on [0,1/2) and -1 on (1/2,1], so (g\')**2 = 1 everywhere
      and psi_2 = 1.

    Returns
    -------
    psi_1 : float  (= 1/12)
    psi_2 : float  (= 1.0)
    """
    return 1.0 / 12.0, 1.0


def _preaveraged_returns(returns: np.ndarray, k_n: int) -> np.ndarray:
    """Pre-averaged returns Ybar_i = sum_{j=0}^{k_n-1} g(j/k_n) * r_{i+j}.

    g(x) = min(x, 1-x) per Jacod et al. (2009).

    Parameters
    ----------
    returns : np.ndarray   length n
    k_n     : int          pre-averaging window

    Returns
    -------
    np.ndarray  shape (n - k_n + 1,)
    """
    j = np.arange(k_n, dtype=np.float64)
    g = np.minimum(j / k_n, 1.0 - j / k_n)
    # Equivalent to: Ybar[i] = dot(g, returns[i:i+k_n])
    # Vectorised via convolution: convolve(returns, g[::-1], mode=valid)
    Ybar = np.convolve(returns, g[::-1], mode="valid")
    return Ybar


# ---------------------------------------------------------------------------
# 1. RV5m
# ---------------------------------------------------------------------------

def compute_rv5m(
    returns: np.ndarray,
    freq_minutes: int = 1,
    target_minutes: int = 5,
) -> float:
    """RV5m: realized variance at 5-minute sampling frequency.

    Aggregates consecutive 1-minute returns into non-overlapping
    target_minutes-minute returns by summing blocks of `step` elements,
    then sums the squared coarse returns.

    Parameters
    ----------
    returns       : 1-minute within-session log returns (~390 values). NaN dropped.
    freq_minutes  : input frequency in minutes (default 1).
    target_minutes: target sampling interval in minutes (default 5).

    Returns
    -------
    float  (>= 0)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    step = target_minutes // freq_minutes
    n_blocks = len(r) // step
    if n_blocks == 0:
        return 0.0
    r_coarse = r[: n_blocks * step].reshape(n_blocks, step).sum(axis=1)
    return float(np.sum(r_coarse ** 2))


# ---------------------------------------------------------------------------
# 2. Realized Kernel (Parzen, BN et al. 2008)
# ---------------------------------------------------------------------------

def _parzen_kernel(x: np.ndarray) -> np.ndarray:
    """Parzen kernel: piecewise cubic with compact support on [-1, 1]."""
    ax = np.abs(x)
    return np.where(
        ax <= 0.5,
        1.0 - 6.0 * ax**2 + 6.0 * ax**3,
        np.where(ax <= 1.0, 2.0 * (1.0 - ax)**3, 0.0),
    )


def compute_realized_kernel(
    returns: np.ndarray,
    max_lag: Optional[int] = None,
) -> float:
    """Realized Kernel with Parzen weights (Barndorff-Nielsen et al. 2008).

    RK = gamma_0 + 2 * sum_{l=1}^L k(l/(L+1)) * gamma_l
    gamma_l = sum_{t=l}^{n-1} r[t] * r[t-l]

    Bandwidth L uses the data-driven rule of Barndorff-Nielsen et al. (2009):
      omega2 = max(-gamma_1, 0)     (negated first-lag autocovariance)
      IQ     = (n/3) * sum(r**4)   (realized quarticity)
      xi2    = omega2 / IQ**0.5
      L      = c* * (xi2 * n)**(2/5), c* = 3.5134

    Parameters
    ----------
    returns : np.ndarray   within-session log returns
    max_lag : int, optional  overrides data-driven bandwidth

    Returns
    -------
    float  (>= 0)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 4:
        return float(np.sum(r**2))

    gamma_0 = float(np.dot(r, r))

    if max_lag is not None:
        L = max(1, int(max_lag))
    else:
        gamma_1 = float(np.dot(r[1:], r[:-1]))
        omega2 = max(-gamma_1, 1e-20)
        IQ = max(float(np.sum(r**4)) * n / 3.0, 1e-40)
        xi2 = omega2 / (IQ**0.5)
        L = max(1, int(np.ceil(3.5134 * (xi2 * n)**0.4)))
        L = min(L, n - 1)

    lags = np.arange(1, L + 1, dtype=np.float64)
    weights = _parzen_kernel(lags / (L + 1))
    rk = gamma_0
    for li, w in zip(range(1, L + 1), weights):
        if w == 0.0:
            continue
        rk += 2.0 * w * float(np.dot(r[li:], r[: n - li]))
    return max(float(rk), 0.0)


# ---------------------------------------------------------------------------
# 3. TSRV (Zhang et al. 2005)
# ---------------------------------------------------------------------------

def compute_tsrv(returns: np.ndarray, K: int = 5) -> float:
    """Two-Scale Realized Variance.

    TSRV = (1/K) sum_{k=0}^{K-1} RV_k  -  (bar_n / n) * RV_full
    bar_n = (n - K + 1) / K

    Parameters
    ----------
    returns : np.ndarray  within-session 1-minute log returns
    K       : int         slow-scale window in minutes (default 5)

    Returns
    -------
    float  (clipped to 0 from below)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        return 0.0
    rv_slow = sum(float(np.sum(r[k::K]**2)) for k in range(K)) / K
    rv_fast = float(np.sum(r**2))
    bar_n = (n - K + 1) / K
    return max(float(rv_slow - (bar_n / n) * rv_fast), 0.0)


# ---------------------------------------------------------------------------
# 4. BPV (Barndorff-Nielsen & Shephard 2004)
# ---------------------------------------------------------------------------

def compute_bpv(returns: np.ndarray) -> float:
    """Bipower Variation.

    BPV = (pi/2) * sum_{i=2}^n |r_i| * |r_{i-1}|

    Jump-robust, not noise-robust.

    Parameters
    ----------
    returns : np.ndarray  within-session log returns

    Returns
    -------
    float  (>= 0)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    return max(float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))), 0.0)


# ---------------------------------------------------------------------------
# 5. PAV (Jacod et al. 2009)
# ---------------------------------------------------------------------------

def compute_pav(returns: np.ndarray, theta: float = 1.0) -> float:
    """Pre-Averaged Variance.

    PAV = (1 / (n * k_n * psi_2)) * sum_i Ybar_i^2
          - (psi_1 / (2 * psi_2 * k_n)) * RV_full

    k_n = ceil(theta * sqrt(n)), psi_1 = 1/12, psi_2 = 1.

    Simultaneously noise-robust and consistent for integrated variance.

    Parameters
    ----------
    returns : np.ndarray  within-session log returns
    theta   : float       window scaling constant (default 1)

    Returns
    -------
    float  (>= 0)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 4:
        return float(np.sum(r**2))
    psi_1, psi_2 = compute_psi_constants()
    k_n = max(1, int(np.ceil(theta * np.sqrt(n))))
    Ybar = _preaveraged_returns(r, k_n)
    main = float(np.sum(Ybar**2)) / (n * k_n * psi_2)
    bias = (psi_1 / (2.0 * psi_2 * k_n)) * float(np.sum(r**2))
    return max(float(main - bias), 0.0)


# ---------------------------------------------------------------------------
# 6. PABPV (Jacod et al. 2009)
# ---------------------------------------------------------------------------

def compute_pabpv(returns: np.ndarray, theta: float = 1.0) -> float:
    """Pre-Averaged Bipower Variation.

    PABPV = (pi / (2 * k_n * psi_2)) * sum_{i=0}^{n-2k_n} |Ybar_i| * |Ybar_{i+k_n}|

    Both noise-robust and jump-robust.

    Parameters
    ----------
    returns : np.ndarray  within-session log returns
    theta   : float       window scaling constant (default 1)

    Returns
    -------
    float  (>= 0)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 4:
        return max(float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))), 0.0)
    _, psi_2 = compute_psi_constants()
    k_n = max(1, int(np.ceil(theta * np.sqrt(n))))
    Ybar = _preaveraged_returns(r, k_n)
    absY = np.abs(Ybar)
    n_pairs = len(absY) - k_n
    if n_pairs <= 0:
        return 0.0
    products = absY[:n_pairs] * absY[k_n : k_n + n_pairs]
    return max(float((np.pi / 2.0) / (k_n * psi_2) * np.sum(products)), 0.0)


# ---------------------------------------------------------------------------
# 7. C-TRV (Corsi et al. 2010)
# ---------------------------------------------------------------------------

def _ctrv_constant() -> float:
    """Exceedance replacement constant c for C-TRV.

    c = E[X^2 | |X| > 3] for X ~ N(0,1)
      = (3*phi(3) + (1 - Phi(3))) / (1 - Phi(3))  ~= 10.849

    Note: uses the PDF phi, not the CDF Phi, in the numerator.
    """
    phi3 = norm.pdf(3.0)
    tail = 1.0 - norm.cdf(3.0)
    return float((3.0 * phi3 + tail) / tail)


_C_TRV_C: float = _ctrv_constant()


def compute_ctrv(returns: np.ndarray) -> float:
    """Corrected Threshold Realized Variance (C-TRV).

    vartheta2 = BPV / n
    threshold = 9 * vartheta2

    C-TRV = sum_{r^2 <= threshold} r^2  +  count_{r^2 > threshold} * c * vartheta2

    Parameters
    ----------
    returns : np.ndarray  within-session log returns

    Returns
    -------
    float  (>= 0)
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2:
        return float(np.sum(r**2))
    bpv = compute_bpv(r)
    vartheta2 = bpv / n
    threshold = 9.0 * vartheta2
    r2 = r**2
    below = r2 <= threshold
    return max(
        float(np.sum(r2[below])) + float((~below).sum()) * _C_TRV_C * vartheta2,
        0.0,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_estimators(returns: np.ndarray) -> dict:
    """Compute all 7 RV estimators for one day's within-session return series.

    Parameters
    ----------
    returns : np.ndarray  1-minute within-session log returns (~390 values)

    Returns
    -------
    dict with keys: rv5m, rk, tsrv, bpv, pav, pabpv, ctrv
    """
    r = np.asarray(returns, dtype=np.float64)
    return {
        "rv5m":  compute_rv5m(r),
        "rk":    compute_realized_kernel(r),
        "tsrv":  compute_tsrv(r),
        "bpv":   compute_bpv(r),
        "pav":   compute_pav(r),
        "pabpv": compute_pabpv(r),
        "ctrv":  compute_ctrv(r),
    }
