"""Realised-variance estimators used by the replication.

All public functions accept a one-dimensional array of within-session *log
returns*.  The implementations follow the definitions cited by the paper while
keeping finite-sample choices explicit.

Two corrections relative to the submitted repository are important:

* TSRV slow-grid returns must be K-step *price differences*, not every K-th
  one-minute return.
* For ``g(x)=min(x, 1-x)``, Jacod et al.'s notation is
  ``psi_1 = ∫(g')² = 1`` and ``psi_2 = ∫g² = 1/12``.  The submitted code had
  the constants and the PAV normalisation reversed, which drove PAV to zero.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


def _clean_returns(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    return r[np.isfinite(r)]


# ---------------------------------------------------------------------------
# Pre-averaging constants and helper
# ---------------------------------------------------------------------------

def compute_psi_constants() -> Tuple[float, float]:
    """Return ``(psi_1, psi_2)`` for ``g(x)=min(x,1-x)``.

    In Jacod et al. (2009), ``psi_1 = ∫_0^1 (g'(x))² dx = 1`` and
    ``psi_2 = ∫_0^1 g(x)² dx = 1/12``.
    """
    return 1.0, 1.0 / 12.0


def _preaveraged_returns(returns: np.ndarray, k_n: int) -> np.ndarray:
    """Compute moving pre-averaged returns.

    ``Ybar_i = Σ_{j=1}^{k_n-1} g(j/k_n) r_{i+j}`` in the usual observation
    indexing.  In zero-based NumPy indexing this is a convolution with the
    ``k_n-1`` interior weights.  There is deliberately no extra ``1/k_n``:
    the normalisation is supplied by the estimator itself.
    """
    r = _clean_returns(returns)
    if k_n < 2:
        raise ValueError("k_n must be at least two")
    if r.size < k_n - 1:
        return np.empty(0, dtype=np.float64)

    j = np.arange(1, k_n, dtype=np.float64)
    g = np.minimum(j / k_n, 1.0 - j / k_n)
    return np.convolve(r, g[::-1], mode="valid")


# ---------------------------------------------------------------------------
# 1. RV5m
# ---------------------------------------------------------------------------

def compute_rv5m(
    returns: np.ndarray,
    freq_minutes: int = 1,
    target_minutes: int = 5,
) -> float:
    """Realised variance from non-overlapping five-minute returns."""
    r = _clean_returns(returns)
    if freq_minutes <= 0 or target_minutes <= 0:
        raise ValueError("sampling frequencies must be positive")
    if target_minutes % freq_minutes != 0:
        raise ValueError("target_minutes must be a multiple of freq_minutes")

    step = target_minutes // freq_minutes
    n_blocks = r.size // step
    if n_blocks == 0:
        return 0.0
    coarse = r[: n_blocks * step].reshape(n_blocks, step).sum(axis=1)
    return float(np.dot(coarse, coarse))


# ---------------------------------------------------------------------------
# 2. Realised Kernel (Parzen)
# ---------------------------------------------------------------------------

def _parzen_kernel(x: np.ndarray) -> np.ndarray:
    """Parzen kernel with support on ``[-1, 1]``."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    return np.where(
        ax <= 0.5,
        1.0 - 6.0 * ax**2 + 6.0 * ax**3,
        np.where(ax <= 1.0, 2.0 * (1.0 - ax) ** 3, 0.0),
    )


def _parzen_bandwidth(returns: np.ndarray) -> int:
    """BNHLS data-driven Parzen bandwidth, in return lags.

    The practical rule is ``H = 3.5134 * xi^(4/5) * n^(3/5)``.  With
    ``xi2 = omega² / sqrt(IQ) = xi²``, this is
    ``3.5134 * xi2^(2/5) * n^(3/5)``.
    """
    r = _clean_returns(returns)
    n = r.size
    if n < 4:
        return max(n - 1, 0)

    gamma1_sum = float(np.dot(r[1:], r[:-1]))
    # Under additive price noise, E[r_t r_{t-1}] = -omega².  The dot product
    # is a sum, hence divide by the number of pairs.
    omega2 = max(-gamma1_sum / (n - 1), 0.0)
    iq = max((n / 3.0) * float(np.sum(r**4)), np.finfo(float).tiny)
    xi2 = omega2 / np.sqrt(iq)

    if xi2 <= 0.0:
        return 1
    bandwidth = int(np.ceil(3.5134 * xi2 ** (2.0 / 5.0) * n ** (3.0 / 5.0)))
    return int(np.clip(bandwidth, 1, n - 1))


def compute_realized_kernel(
    returns: np.ndarray,
    max_lag: Optional[int] = None,
    *,
    clip: bool = True,
) -> float:
    """Parzen realised kernel.

    This is the non-flat-top kernel sum used by the paper's simplified
    one-minute pipeline.  The full BNHLS empirical procedure also discusses
    end effects and timestamp regularisation; those require raw quote/trade
    data and are outside this synthetic harness.
    """
    r = _clean_returns(returns)
    n = r.size
    if n == 0:
        return 0.0
    if n < 4:
        value = float(np.dot(r, r))
        return max(value, 0.0) if clip else value

    if max_lag is None:
        bandwidth = _parzen_bandwidth(r)
    else:
        bandwidth = int(np.clip(int(max_lag), 1, n - 1))

    value = float(np.dot(r, r))
    lags = np.arange(1, bandwidth + 1, dtype=np.float64)
    weights = _parzen_kernel(lags / (bandwidth + 1.0))
    for lag, weight in zip(range(1, bandwidth + 1), weights):
        value += 2.0 * float(weight) * float(np.dot(r[lag:], r[:-lag]))

    return max(value, 0.0) if clip else value


# ---------------------------------------------------------------------------
# 3. TSRV
# ---------------------------------------------------------------------------

def compute_tsrv(
    returns: np.ndarray,
    K: int = 5,
    *,
    finite_sample_rescale: bool = False,
    clip: bool = True,
) -> float:
    """Two-scale realised variance.

    The slow component averages realised variances over all ``K`` staggered
    price subgrids.  A K-step subgrid return is a difference of observed
    prices K minutes apart; selecting every K-th one-minute return, as the
    submitted implementation did, is not equivalent.

    The paper writes

    ``TSRV = mean_k(RV_slow,k) - (bar_n/n) RV_fast``.

    ``finite_sample_rescale=True`` applies the optional division by
    ``1 - bar_n/n`` used in some implementations.  It is a constant daily
    scale when ``n`` is fixed and therefore does not affect H directly.
    """
    r = _clean_returns(returns)
    n = r.size
    if n == 0:
        return 0.0
    if K < 2:
        raise ValueError("K must be at least two")
    K = min(int(K), n)

    prices = np.concatenate(([0.0], np.cumsum(r)))
    slow_rvs: list[float] = []
    slow_counts: list[int] = []
    for offset in range(K):
        grid = prices[offset::K]
        if grid.size < 2:
            continue
        grid_returns = np.diff(grid)
        slow_rvs.append(float(np.dot(grid_returns, grid_returns)))
        slow_counts.append(int(grid_returns.size))

    if not slow_rvs:
        return 0.0

    rv_slow = float(np.mean(slow_rvs))
    rv_fast = float(np.dot(r, r))
    bar_n = float(np.mean(slow_counts))
    ratio = bar_n / n
    value = rv_slow - ratio * rv_fast
    if finite_sample_rescale and ratio < 1.0:
        value /= 1.0 - ratio

    return max(value, 0.0) if clip else value


# ---------------------------------------------------------------------------
# 4. BPV
# ---------------------------------------------------------------------------

def compute_bpv(returns: np.ndarray) -> float:
    """Bipower variation, robust to finite-activity price jumps."""
    r = _clean_returns(returns)
    if r.size < 2:
        return 0.0
    return float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))


# ---------------------------------------------------------------------------
# 5. PAV
# ---------------------------------------------------------------------------

def compute_pav(
    returns: np.ndarray,
    theta: float = 1.0,
    *,
    clip: bool = True,
) -> float:
    """Pre-averaged realised variance.

    For ``k_n = ceil(theta*sqrt(n))`` and ``g(x)=min(x,1-x)``, a finite-sample
    form of Jacod et al.'s estimator is

    ``sum(Ybar_i²)/(k_n*psi_2) - psi_1*RV/(2*k_n²*psi_2)``.
    """
    r = _clean_returns(returns)
    n = r.size
    if n == 0:
        return 0.0
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if n < 4:
        return float(np.dot(r, r))

    psi_1, psi_2 = compute_psi_constants()
    k_n = int(np.clip(np.ceil(theta * np.sqrt(n)), 2, n))
    ybar = _preaveraged_returns(r, k_n)
    if ybar.size == 0:
        return 0.0

    main = float(np.dot(ybar, ybar)) / (k_n * psi_2)
    bias = (psi_1 / (2.0 * k_n**2 * psi_2)) * float(np.dot(r, r))
    value = main - bias
    return max(value, 0.0) if clip else value


# ---------------------------------------------------------------------------
# 6. PABPV
# ---------------------------------------------------------------------------

def compute_pabpv(returns: np.ndarray, theta: float = 1.0) -> float:
    """Pre-averaged bipower variation using non-overlapping blocks."""
    r = _clean_returns(returns)
    n = r.size
    if n == 0:
        return 0.0
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if n < 4:
        return compute_bpv(r)

    _, psi_2 = compute_psi_constants()
    k_n = int(np.clip(np.ceil(theta * np.sqrt(n)), 2, n))
    ybar = _preaveraged_returns(r, k_n)
    n_pairs = ybar.size - k_n
    if n_pairs <= 0:
        return 0.0

    products = np.abs(ybar[:n_pairs]) * np.abs(ybar[k_n : k_n + n_pairs])
    return float((np.pi / (2.0 * k_n * psi_2)) * np.sum(products))


# ---------------------------------------------------------------------------
# 7. C-TRV
# ---------------------------------------------------------------------------

def _ctrv_constant() -> float:
    """``E[Z² | |Z|>3]`` for ``Z~N(0,1)`` (approximately 10.849)."""
    phi3 = norm.pdf(3.0)
    tail = 1.0 - norm.cdf(3.0)
    return float((3.0 * phi3 + tail) / tail)


_C_TRV_C: float = _ctrv_constant()


def compute_ctrv(returns: np.ndarray) -> float:
    """Corrected threshold realised variance of Corsi et al. (2010)."""
    r = _clean_returns(returns)
    n = r.size
    if n == 0:
        return 0.0
    if n < 2:
        return float(np.dot(r, r))

    vartheta2 = compute_bpv(r) / n
    threshold = 9.0 * vartheta2
    r2 = np.square(r)
    below = r2 <= threshold
    return float(np.sum(r2[below]) + np.count_nonzero(~below) * _C_TRV_C * vartheta2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_estimators(returns: np.ndarray) -> dict:
    """Compute all seven daily realised-variance estimators."""
    r = _clean_returns(returns)
    return {
        "rv5m": compute_rv5m(r),
        "rk": compute_realized_kernel(r),
        "tsrv": compute_tsrv(r),
        "bpv": compute_bpv(r),
        "pav": compute_pav(r),
        "pabpv": compute_pabpv(r),
        "ctrv": compute_ctrv(r),
    }
