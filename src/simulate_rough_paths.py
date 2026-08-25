"""Simulation helpers for the rough-volatility validation experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np


def simulate_fbm_davies_harte(
    n: int,
    H: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate ``n`` unit-variance fGn increments by circulant embedding."""
    if n < 1:
        raise ValueError("n must be positive")
    if not (0.0 < H < 1.0):
        raise ValueError("H must lie in (0,1)")

    rng = np.random.default_rng(seed)
    m = 2 * n

    j = np.arange(n + 1, dtype=np.float64)
    gamma = 0.5 * (
        (j + 1.0) ** (2.0 * H)
        - 2.0 * j ** (2.0 * H)
        + np.abs(j - 1.0) ** (2.0 * H)
    )
    gamma[0] = 1.0

    first_row = np.empty(m, dtype=np.float64)
    first_row[: n + 1] = gamma
    first_row[n + 1 :] = gamma[n - 1 : 0 : -1]

    eigenvalues = np.real(np.fft.fft(first_row))
    # Tiny negative values are numerical.  Material negatives would indicate
    # an invalid embedding; the 2n embedding is non-negative for the grids used
    # here, so fail rather than silently changing the covariance materially.
    tolerance = 1e-10 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance:
        raise RuntimeError("Davies-Harte circulant embedding is not positive")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    half = m // 2
    spectral = np.zeros(m, dtype=np.complex128)
    spectral[0] = np.sqrt(eigenvalues[0]) * rng.standard_normal()
    spectral[half] = np.sqrt(eigenvalues[half]) * rng.standard_normal()

    z_real = rng.standard_normal(half - 1)
    z_imag = rng.standard_normal(half - 1)
    interior = np.sqrt(eigenvalues[1:half] / 2.0) * (z_real + 1j * z_imag)
    spectral[1:half] = interior
    spectral[half + 1 :] = np.conj(interior[::-1])

    # NumPy's ifft contains 1/m.  Multiplication by sqrt(m) yields the target
    # covariance for the first n real components.
    fgn = np.real(np.fft.ifft(spectral)) * np.sqrt(float(m))
    return fgn[:n]


def simulate_fou_log_vol(
    n_days: int,
    H: float,
    kappa: float,
    nu: float = 0.5,
    dt: float = 1.0,
    seed: Optional[int] = None,
    burn_in: Optional[int] = None,
) -> np.ndarray:
    """Simulate a daily fractional-OU approximation.

    Parameters are expressed in the same time unit as ``dt``.  For the paper's
    tables, ``dt=1`` and ``kappa=0.010`` mean a mean-reversion speed of 0.010
    *per trading day*.  The submitted experiment instead combined
    ``kappa=0.010`` with ``dt=1/252``, making effective mean reversion 252 times
    too slow.

    The discrete recursion is

    ``X_{t+1} = exp(-kappa*dt) X_t + nu * dt^H * dB_t^H``.

    It is a filtered-fGn approximation to stationary fOU.  For ``kappa>0`` a
    burn-in of ten mean-reversion time constants is used by default.
    """
    if n_days < 1:
        raise ValueError("n_days must be positive")
    if not (0.0 < H < 1.0):
        raise ValueError("H must lie in (0,1)")
    if kappa < 0.0 or nu <= 0.0 or dt <= 0.0:
        raise ValueError("require kappa>=0, nu>0 and dt>0")

    if burn_in is None:
        if kappa == 0.0:
            burn_in = 0
        else:
            burn_in = max(252, int(np.ceil(10.0 / (kappa * dt))))
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")

    n_total = n_days + burn_in
    fgn = simulate_fbm_davies_harte(n_total, H, seed=seed)
    innovations = nu * (dt**H) * fgn

    x = np.empty(n_total, dtype=np.float64)
    x[0] = 0.0
    rho = float(np.exp(-kappa * dt)) if kappa > 0.0 else 1.0
    for t in range(n_total - 1):
        x[t + 1] = rho * x[t] + innovations[t]

    return x[burn_in:]


def simulate_noisy_intraday_prices(
    log_vol: np.ndarray,
    n_bars: int = 390,
    varpi: float = 1e-4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate observed intraday log prices with additive price noise.

    The output has shape ``(n_days, n_bars+1)`` so that differencing produces
    exactly ``n_bars`` one-minute returns per day.
    """
    x = np.asarray(log_vol, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("log_vol must be one-dimensional")
    if n_bars < 1 or varpi < 0.0:
        raise ValueError("require n_bars>=1 and varpi>=0")

    rng = np.random.default_rng(seed)
    n_days = x.size
    daily_sigma = np.exp(x / 2.0)
    efficient_returns = (
        rng.standard_normal((n_days, n_bars))
        * daily_sigma[:, None]
        / np.sqrt(float(n_bars))
    )
    efficient_prices = np.concatenate(
        [np.zeros((n_days, 1), dtype=np.float64), np.cumsum(efficient_returns, axis=1)],
        axis=1,
    )
    noise = rng.normal(0.0, varpi, size=efficient_prices.shape)
    return efficient_prices + noise


def compute_latent_iv(log_vol: np.ndarray) -> np.ndarray:
    """Daily latent integrated variance under the simulation convention."""
    return np.exp(np.asarray(log_vol, dtype=np.float64))
