"""
Hurst exponent estimation from daily log realized-variance series.

Method: second-moment scaling regression (Gatheral, Jaisson & Rosenbaum 2018).

    m(2, Delta) = (1/N) * sum_{k=0}^{N-1} |X_{(k+1)*Delta} - X_{k*Delta}|^2

    log m(2, Delta) = a + 2H * log(Delta) + u_Delta
    => H_hat = slope / 2
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Optional


def compute_second_moment_scaling(
    log_rv_series: np.ndarray,
    delta_max: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical second moments m(2, Delta) for Delta = 1,...,delta_max.

    Uses non-overlapping blocks so each day contributes at most once per lag:

        m(2, Delta) = (1/N) sum_{k=0}^{N-1} |X_{(k+1)*Delta} - X_{k*Delta}|^2

    where N = floor(T / Delta).

    Parameters
    ----------
    log_rv_series : np.ndarray
        Daily log realized-variance series X_t (length T).
    delta_max : int
        Maximum lag in trading days.

    Returns
    -------
    deltas   : np.ndarray  shape (delta_max,)   [1, 2, ..., delta_max]
    m2_values: np.ndarray  shape (delta_max,)   m(2, Delta) for each delta
    """
    X = np.asarray(log_rv_series, dtype=np.float64)
    T = len(X)
    deltas = np.arange(1, delta_max + 1, dtype=np.int64)
    m2 = np.empty(len(deltas), dtype=np.float64)
    for i, d in enumerate(deltas):
        N = T // d
        if N < 1:
            m2[i] = np.nan
            continue
        idx = np.arange(N) * d
        diffs = X[idx + d - 1] - X[idx]   # X_{(k+1)*d - 1} - X_{k*d}
        # non-overlapping step differences (use end-points of each block)
        ends = np.minimum(idx + d, T - 1)
        starts = idx
        diffs = X[ends] - X[starts]
        m2[i] = float(np.mean(diffs ** 2))
    return deltas.astype(np.float64), m2


def estimate_hurst_ols(
    log_rv_series: np.ndarray,
    delta_max: int = 10,
) -> dict:
    """Estimate the Hurst exponent by OLS log-log regression.

    Fits:  log m(2, Delta) = a + b * log(Delta)
    Returns H_hat = b / 2.

    Parameters
    ----------
    log_rv_series : np.ndarray
        Daily log realized-variance series.
    delta_max : int
        Maximum lag in trading days.

    Returns
    -------
    dict with keys: H_hat, b, a, r_squared, n_lags, n_obs
    """
    X = np.asarray(log_rv_series, dtype=np.float64)
    deltas, m2 = compute_second_moment_scaling(X, delta_max)

    valid = np.isfinite(m2) & (m2 > 0)
    if valid.sum() < 2:
        return {"H_hat": np.nan, "b": np.nan, "a": np.nan,
                "r_squared": np.nan, "n_lags": 0, "n_obs": len(X)}

    log_d = np.log(deltas[valid])
    log_m2 = np.log(m2[valid])

    Xmat = sm.add_constant(log_d)
    res = sm.OLS(log_m2, Xmat).fit()

    return {
        "H_hat":     float(res.params[1]) / 2.0,
        "b":         float(res.params[1]),
        "a":         float(res.params[0]),
        "r_squared": float(res.rsquared),
        "n_lags":    int(valid.sum()),
        "n_obs":     len(X),
    }


def estimate_hurst_cross_section(
    log_rv_panel: pd.DataFrame,
    delta_max: int = 10,
    estimator: str = "tsrv",
) -> pd.DataFrame:
    """Estimate the Hurst exponent for each asset in a panel.

    Parameters
    ----------
    log_rv_panel : pd.DataFrame
        Dates as index, assets as columns; values are log-RV.
    delta_max : int
        Maximum lag.
    estimator : str
        Name of the RV estimator used (stored in output).

    Returns
    -------
    pd.DataFrame
        Columns: asset, H_hat, r_squared, n_obs, estimator
    """
    records = []
    for asset in log_rv_panel.columns:
        series = log_rv_panel[asset].dropna().values
        if len(series) < delta_max + 1:
            continue
        result = estimate_hurst_ols(series, delta_max=delta_max)
        records.append({
            "asset":     str(asset),
            "H_hat":     result["H_hat"],
            "r_squared": result["r_squared"],
            "n_obs":     result["n_obs"],
            "estimator": estimator,
        })
    return pd.DataFrame(records)
