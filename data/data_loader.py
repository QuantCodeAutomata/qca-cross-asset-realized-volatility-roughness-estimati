"""
Data loading utilities for the cross-asset realized-volatility roughness project.

Real data is fetched from the Massive API (MASSIVE_TOKEN env variable).
Synthetic data is generated using the Davies-Harte (circulant embedding) algorithm
for exact fGn/fBm simulation, which underpins all unit-test and benchmark work.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Massive API loaders
# ---------------------------------------------------------------------------

def load_equity_intraday(
    ticker: str,
    from_date: str,
    to_date: str,
    timespan: str = "minute",
) -> pd.DataFrame:
    """Load intraday OHLCV data from the Massive API for an equity ticker.

    Parameters
    ----------
    ticker:
        Equity ticker symbol (e.g. ``"SPY"``).
    from_date:
        Start date in ``"YYYY-MM-DD"`` format (inclusive).
    to_date:
        End date in ``"YYYY-MM-DD"`` format (inclusive).
    timespan:
        Bar size passed to the Massive aggregates endpoint (``"minute"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (ms epoch, int), ``open``, ``high``, ``low``,
        ``close``, ``volume`` (all float64).  Sorted ascending by timestamp.
    """
    from massive.rest import RESTClient

    token = os.environ.get("MASSIVE_TOKEN")
    client = RESTClient(api_key=token)

    bars = list(
        client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            sort="asc",
            limit=50_000,
        )
    )

    if not bars:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    records = [
        {
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    df = df.astype(
        {
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )
    return df


def load_futures_intraday(
    ticker: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Load intraday futures aggregates from the Massive API.

    Parameters
    ----------
    ticker:
        Futures contract ticker (e.g. ``"ESH24"`` for E-mini S&P March 2024).
    from_date:
        Start date filter in ``"YYYY-MM-DD"`` format (inclusive).
    to_date:
        End date filter in ``"YYYY-MM-DD"`` format (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ``window_start`` (int, ms epoch), ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``session_end_date`` (str).
    """
    from massive.rest import FuturesClient

    token = os.environ.get("MASSIVE_TOKEN")
    client = FuturesClient(api_key=token)

    bars = list(
        client.list_futures_aggregates(
            ticker=ticker,
            window_start_gte=from_date,
            window_start_lte=to_date,
            sort="asc",
            limit=50_000,
        )
    )

    if not bars:
        return pd.DataFrame(
            columns=[
                "window_start",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "session_end_date",
            ]
        )

    records = [
        {
            "window_start": b.window_start,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "session_end_date": b.session_end_date,
        }
        for b in bars
    ]
    df = pd.DataFrame(records).sort_values("window_start").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Davies-Harte fBm / fGn simulation
# ---------------------------------------------------------------------------

def _fgn_davies_harte(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate one path of fractional Gaussian noise (fGn) via circulant embedding.

    Implements the exact Wood & Chan (1994) / Davies & Harte (1987) algorithm.
    The returned series has length ``n`` and unit variance per increment
    (i.e. the covariance at lag k is
    ``gamma(k) = 0.5*(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})``).

    Parameters
    ----------
    n:
        Number of fGn increments to generate.
    H:
        Hurst exponent in (0, 1).
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``, dtype float64.
    """
    # ------------------------------------------------------------------
    # Step 1: autocovariance of fGn at lags 0, 1, ..., n
    # gamma(k) = 0.5*(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})
    # gamma(0) = 1  (unit-variance increments)
    # ------------------------------------------------------------------
    k = np.arange(n + 1, dtype=np.float64)
    km1 = np.maximum(k - 1.0, 0.0)
    gamma = 0.5 * ((k + 1.0) ** (2.0 * H) - 2.0 * k ** (2.0 * H) + km1 ** (2.0 * H))
    # Enforce exact unit variance at lag 0
    gamma[0] = 1.0

    # ------------------------------------------------------------------
    # Step 2: embed in 2n-circulant
    # Row: [gamma(0), gamma(1), ..., gamma(n-1), gamma(n), gamma(n-1), ..., gamma(1)]
    # ------------------------------------------------------------------
    m = 2 * n
    c = np.empty(m, dtype=np.float64)
    c[: n + 1] = gamma          # positions 0 … n
    c[n + 1 :] = gamma[n - 1 : 0 : -1]  # positions n+1 … 2n-1 (symmetric fill)

    # ------------------------------------------------------------------
    # Step 3: eigenvalues of circulant via FFT (real-valued because c is symmetric)
    # ------------------------------------------------------------------
    lam = np.real(np.fft.fft(c))
    lam = np.maximum(lam, 0.0)  # clip tiny negatives from numerical error

    # ------------------------------------------------------------------
    # Step 4: generate spectral-domain complex Gaussian with conjugate symmetry
    # so that the IFFT output is real.
    #
    # For j=0 and j=n (Nyquist): W[j] = sqrt(lam[j]) * z_real
    # For j=1,...,n-1:           W[j] = sqrt(lam[j]/2) * (z_r + i*z_i)
    #                            W[m-j] = conj(W[j])
    # ------------------------------------------------------------------
    sqrt_lam = np.sqrt(lam)

    W = np.zeros(m, dtype=np.complex128)

    # DC and Nyquist — real
    W[0] = sqrt_lam[0] * rng.standard_normal()
    W[n] = sqrt_lam[n] * rng.standard_normal()

    # Interior frequencies — complex with conjugate symmetry
    js = np.arange(1, n, dtype=int)
    z_r = rng.standard_normal(n - 1)
    z_i = rng.standard_normal(n - 1)
    W[js] = (sqrt_lam[js] / np.sqrt(2.0)) * (z_r + 1j * z_i)
    W[m - js] = np.conj(W[js])

    # ------------------------------------------------------------------
    # Step 5: synthesise via FFT
    # X = Re(FFT(W)) / sqrt(m)  gives the desired stationary fGn.
    # ------------------------------------------------------------------
    X = np.real(np.fft.fft(W)) / np.sqrt(float(m))
    return X[:n]


def _simulate_fou_log_vol(
    n_total: int,
    H: float,
    kappa: float,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate log-volatility as a fractional Ornstein-Uhlenbeck (fOU) process.

    Euler-Maruyama discretisation at unit time steps (minutes):

        X_{t+1} = X_t * (1 - kappa) + nu * dB^H_t

    where ``dB^H_t`` are the fGn increments produced by Davies-Harte.

    Parameters
    ----------
    n_total:
        Total number of time steps (n_days * n_bars).
    H, kappa, nu:
        fOU parameters (Hurst, mean-reversion speed per step, vol-of-vol).
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Shape ``(n_total,)``, log-volatility path.
    """
    dBH = _fgn_davies_harte(n_total, H, rng)

    X = np.empty(n_total, dtype=np.float64)
    X[0] = 0.0
    one_minus_kappa = 1.0 - kappa
    for t in range(n_total - 1):
        X[t + 1] = one_minus_kappa * X[t] + nu * dBH[t]
    return X


def generate_synthetic_intraday(
    n_days: int = 252,
    n_bars: int = 390,
    H: float = 0.1,
    kappa: float = 0.01,
    nu: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic intraday data using a rough volatility model.

    Architecture
    ------------
    Log-volatility is driven by a fractional Ornstein-Uhlenbeck (fOU) process
    **at daily frequency** (one step per trading day):

        X_{d+1} = (1 - kappa) * X_d + nu * dB_d^H

    so ``kappa`` is a *daily* mean-reversion rate (e.g. ``0.01`` ≈ 100-day
    half-life) and ``nu`` is the daily vol-of-vol amplitude.

    The daily spot-volatility is ``sigma_d = exp(X_d / 2)``.  Within each day,
    ``n_bars`` intraday returns are drawn i.i.d. from
    ``N(0, sigma_d^2 / n_bars)`` so that the daily realized variance satisfies

        RV_d = sum_i r_{d,i}^2  ≈  sigma_d^2  =  exp(X_d)

    and therefore

        log(RV_d)  ≈  X_d  +  chi2 sampling noise

    The scaling of ``m(2, Delta) = E[(log RV_{d+Delta} - log RV_d)^2]`` thus
    reflects the roughness *H* of the underlying fOU, making this simulation
    appropriate for Hurst estimation via the second-moment regression.

    The fGn driver ``dB^H`` is simulated exactly via the Davies-Harte
    (circulant embedding) algorithm.

    Parameters
    ----------
    n_days:
        Number of trading days to simulate.
    n_bars:
        Intraday bars per day (default 390 for the regular NYSE session).
    H:
        Hurst exponent of the log-volatility fOU driver (roughness parameter).
        Values below 0.5 give rough (anti-persistent) volatility.
    kappa:
        Daily mean-reversion rate of the fOU process (e.g. 0.01 → ~100-day
        memory, much longer than ``delta_max=10`` so the roughness scaling is
        not distorted by mean-reversion).
    nu:
        Daily vol-of-vol (amplitude of the fGn driver).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:

        * ``date``      — trading date (pd.Timestamp, one value per day)
        * ``bar_idx``   — bar index within the session (0 … n_bars-1)
        * ``log_price`` — simulated log price
        * ``return_``   — within-session log return (NaN at bar_idx 0 since
                          the opening bar has no predecessor within the session)
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1.  Simulate log-vol at DAILY frequency (n_days steps).
    #     kappa is now the daily mean-reversion rate.
    # ------------------------------------------------------------------
    log_vol_daily = _simulate_fou_log_vol(n_days, H, kappa, nu, rng)

    # Daily spot variance and per-bar standard deviation
    sigma_sq_daily = np.exp(log_vol_daily)          # exp(X_d) — daily IV
    sigma_per_bar = np.sqrt(sigma_sq_daily / n_bars) # std per intraday bar

    # ------------------------------------------------------------------
    # 2.  Simulate intraday returns: r_{d,i} ~ N(0, sigma_d^2 / n_bars)
    #     All bars within a day share the same daily sigma_d (constant
    #     intraday vol), so daily RV = sum_i r^2 ≈ sigma_d^2.
    # ------------------------------------------------------------------
    eps = rng.standard_normal((n_days, n_bars))
    returns_2d = sigma_per_bar[:, None] * eps         # shape (n_days, n_bars)

    # ------------------------------------------------------------------
    # 3.  Build intraday log-prices (cumulative within each day, reset
    #     at session open so overnight gaps are excluded from RV).
    # ------------------------------------------------------------------
    log_price_2d = np.zeros((n_days, n_bars), dtype=np.float64)
    log_price_2d[:, 1:] = np.cumsum(returns_2d[:, 1:], axis=1)

    # ------------------------------------------------------------------
    # 4.  Flatten and assemble DataFrame
    # ------------------------------------------------------------------
    base_date = pd.Timestamp("2010-01-04")
    trading_dates = pd.bdate_range(start=base_date, periods=n_days)

    dates_col = np.repeat(trading_dates, n_bars)
    bar_idx_col = np.tile(np.arange(n_bars), n_days)
    log_price_col = log_price_2d.ravel()

    # Within-session returns: NaN at bar_idx 0 (opening bar has no predecessor)
    returns_col = np.empty((n_days, n_bars), dtype=np.float64)
    returns_col[:, 0] = np.nan
    returns_col[:, 1:] = np.diff(log_price_2d, axis=1)
    returns_col = returns_col.ravel()

    return pd.DataFrame(
        {
            "date": dates_col,
            "bar_idx": bar_idx_col.astype(np.int32),
            "log_price": log_price_col,
            "return_": returns_col,
        }
    )
