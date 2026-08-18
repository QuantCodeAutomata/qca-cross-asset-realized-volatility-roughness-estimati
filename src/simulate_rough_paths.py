"""
Rough volatility path simulation using Davies-Harte (circulant embedding) for fBM.
Exp_3: Measurement-error attenuation and two-estimator correction.
"""
import numpy as np
from typing import Tuple, Optional


def simulate_fbm_davies_harte(n: int, H: float, seed: Optional[int] = None) -> np.ndarray:
    """Simulate fractional Brownian motion increments using Davies-Harte algorithm.

    Uses circulant embedding of the autocovariance matrix via FFT.

    The fGn autocovariance at lag k:
        gamma(k) = 0.5 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))

    A circulant matrix of size m=2n is constructed with first row
        c = [gamma(0), ..., gamma(n), gamma(n-1), ..., gamma(1)].

    Its eigenvalues (via FFT) are clipped to [0, ∞) if any are slightly negative,
    then used to generate a Gaussian vector with the correct covariance structure.

    Args:
        n: number of time steps
        H: Hurst exponent in (0,1)
        seed: random seed for reproducibility

    Returns:
        fBm increments (fGn) of length n, scaled to variance 1 per step
    """
    rng = np.random.default_rng(seed)
    m = 2 * n

    # fGn autocovariance
    j = np.arange(n + 1, dtype=np.float64)
    gamma_vals = 0.5 * ((j + 1) ** (2 * H) - 2 * j ** (2 * H) + np.abs(j - 1) ** (2 * H))

    # First row of circulant: [gamma(0),...,gamma(n), gamma(n-1),...,gamma(1)]
    c = np.empty(m)
    c[:n + 1] = gamma_vals
    c[n + 1:] = gamma_vals[n - 1:0:-1]

    # Eigenvalues via FFT (real for a symmetric circulant)
    eigenvalues = np.real(np.fft.fft(c))
    if np.any(eigenvalues < 0):
        eigenvalues = np.maximum(eigenvalues, 0.0)

    # Build conjugate-symmetric random vector W so that IFFT(Y) is real
    # W[0], W[m//2] are real; W[k] = (z1 + i*z2)/sqrt(2) for k=1,...,m//2-1
    half = m // 2
    W = np.empty(m, dtype=complex)
    W[0] = rng.standard_normal()
    W[half] = rng.standard_normal()
    z1 = rng.standard_normal(half - 1)
    z2 = rng.standard_normal(half - 1)
    W[1:half] = (z1 + 1j * z2) / np.sqrt(2)
    W[half + 1:] = np.conj(W[1:half][::-1])

    # Scale by sqrt(eigenvalues), IFFT, then multiply by sqrt(m) for correct variance
    Y = np.sqrt(eigenvalues) * W
    fgn = np.real(np.fft.ifft(Y)) * np.sqrt(m)

    return fgn[:n]


def simulate_fou_log_vol(n_days: int, H: float, kappa: float, nu: float = 0.5,
                          dt: float = 1.0 / 252, seed: Optional[int] = None) -> np.ndarray:
    """Simulate fOU log-volatility process at daily frequency.

    dX_t = -kappa * X_t * dt + nu * dB_t^H  (Euler-Maruyama)

    The fBm increment at step dt satisfies dB_t^H ~ dt^H * fGn, where fGn has unit variance.
    Mean reversion starts from the stationary mean X_0 = 0.
    A 252-day burn-in period is discarded to reduce initial-condition effects.

    Args:
        n_days: number of trading days
        H: Hurst exponent
        kappa: mean reversion rate (annualized)
        nu: vol of vol (annualized)
        dt: time step (1/252 for daily)
        seed: random seed

    Returns:
        Log-volatility array of length n_days
    """
    burn_in = 252
    n_total = n_days + burn_in

    fgn = simulate_fbm_davies_harte(n_total, H, seed=seed)
    # fBm increment with correct dt scaling: dB^H = dt^H * fGn
    dB = nu * dt ** H * fgn

    X = np.empty(n_total)
    X[0] = 0.0
    mean_rev = 1.0 - kappa * dt
    for t in range(n_total - 1):
        X[t + 1] = mean_rev * X[t] + dB[t]

    return X[burn_in:]


def simulate_noisy_intraday_prices(log_vol: np.ndarray, n_bars: int = 390,
                                    varpi: float = 1e-4,
                                    seed: Optional[int] = None) -> np.ndarray:
    """Simulate intraday price paths with microstructure noise.

    For each day d:
    - Efficient log-returns: r_i ~ N(0, sigma_d^2 / n_bars) where sigma_d^2 = exp(log_vol[d])
    - Efficient log-price: p_eff[0] = 0, p_eff[i] = p_eff[i-1] + r_i
    - Noise: epsilon_i ~ N(0, varpi^2) i.i.d.
    - Observed log-price: p_obs[i] = p_eff[i] + epsilon_i

    Args:
        log_vol: daily log-variance from fOU (shape: n_days)
        n_bars: number of intraday bars per day (390 for 1-min equity)
        varpi: noise standard deviation
        seed: random seed

    Returns:
        observed log-price array of shape (n_days, n_bars+1)
    """
    rng = np.random.default_rng(seed)
    n_days = len(log_vol)
    p_obs = np.empty((n_days, n_bars + 1))

    # sigma_d = sqrt(exp(log_vol[d])) = exp(log_vol[d] / 2)
    sigma = np.exp(log_vol / 2.0)  # shape (n_days,)

    for d in range(n_days):
        daily_sigma = sigma[d]
        # Efficient returns: N(0, sigma_d^2 / n_bars)
        r_eff = rng.normal(0.0, daily_sigma / np.sqrt(n_bars), size=n_bars)
        # Efficient cumulative log-price
        p_eff = np.empty(n_bars + 1)
        p_eff[0] = 0.0
        p_eff[1:] = np.cumsum(r_eff)
        # Microstructure noise
        noise = rng.normal(0.0, varpi, size=n_bars + 1)
        p_obs[d] = p_eff + noise

    return p_obs


def compute_latent_iv(log_vol: np.ndarray) -> np.ndarray:
    """Extract latent daily integrated variance from log-vol path.

    For fOU: IV_d = exp(log_vol[d]) (by convention in the model).

    Returns: IV array of length n_days
    """
    return np.exp(log_vol)
