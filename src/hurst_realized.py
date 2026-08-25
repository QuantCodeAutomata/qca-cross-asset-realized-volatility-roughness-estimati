"""Hurst estimation from daily log realised-variance series.

The paper's primary estimator fits

    m(2, Δ) = mean_k (X_{(k+1)Δ} - X_{kΔ})²,
    log m(2, Δ) = a + 2 H log Δ + u_Δ,

on the short-lag window Δ = 1, …, Δ_max.  By default the increment
pairs are non-overlapping, exactly as in equation (1) of the paper.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_second_moment_scaling(
    log_rv_series: np.ndarray,
    delta_max: int = 10,
    *,
    overlapping: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical second moments for lags ``1, …, delta_max``.

    For the paper specification (``overlapping=False``), valid pairs are

    ``(0, Δ), (Δ, 2Δ), …``

    and their number is ``floor((T - 1) / Δ)`` for a series with ``T``
    observations.  This endpoint convention matters: the previous
    implementation accidentally included a zero increment at lag one and
    sometimes used a final block shorter than ``Δ``.

    Parameters
    ----------
    log_rv_series:
        One-dimensional daily log realised-variance series.
    delta_max:
        Largest lag in trading days.
    overlapping:
        If ``True``, use all pairs ``(t, t+Δ)``.  The paper's reported
        estimator uses the default non-overlapping convention.

    Returns
    -------
    deltas, m2_values:
        Float arrays of length ``delta_max``.  A lag is ``NaN`` when fewer
        than one finite increment pair is available.
    """
    x = np.asarray(log_rv_series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("log_rv_series must be one-dimensional")
    if delta_max < 1:
        raise ValueError("delta_max must be at least one")

    t = x.size
    deltas = np.arange(1, delta_max + 1, dtype=np.float64)
    m2 = np.full(delta_max, np.nan, dtype=np.float64)

    for i, d_float in enumerate(deltas):
        d = int(d_float)
        if t <= d:
            continue

        if overlapping:
            starts = np.arange(0, t - d, dtype=np.int64)
        else:
            # k = 0, …, floor((T-1)/d)-1, with exact endpoint distance d.
            n_pairs = (t - 1) // d
            if n_pairs < 1:
                continue
            starts = np.arange(n_pairs, dtype=np.int64) * d

        ends = starts + d
        diffs = x[ends] - x[starts]
        finite = np.isfinite(diffs)
        if np.any(finite):
            m2[i] = float(np.mean(np.square(diffs[finite])))

    return deltas, m2


def estimate_hurst_ols(
    log_rv_series: np.ndarray,
    delta_max: int = 10,
    *,
    overlapping: bool = False,
) -> dict:
    """Estimate ``H`` by OLS of ``log m(2,Δ)`` on ``log Δ``.

    ``R²`` is a shape diagnostic, not a standard-error estimate: regression
    residuals are correlated across lags.
    """
    x = np.asarray(log_rv_series, dtype=np.float64)
    deltas, m2 = compute_second_moment_scaling(
        x, delta_max=delta_max, overlapping=overlapping
    )

    valid = np.isfinite(m2) & (m2 > 0.0)
    if int(valid.sum()) < 2:
        return {
            "H_hat": np.nan,
            "b": np.nan,
            "a": np.nan,
            "r_squared": np.nan,
            "n_lags": int(valid.sum()),
            "n_obs": int(x.size),
        }

    log_d = np.log(deltas[valid])
    log_m2 = np.log(m2[valid])
    design = sm.add_constant(log_d)
    result = sm.OLS(log_m2, design).fit()

    return {
        "H_hat": float(result.params[1]) / 2.0,
        "b": float(result.params[1]),
        "a": float(result.params[0]),
        "r_squared": float(result.rsquared),
        "n_lags": int(valid.sum()),
        "n_obs": int(x.size),
    }


def estimate_hurst_cross_section(
    log_rv_panel: pd.DataFrame,
    delta_max: int = 10,
    estimator: str = "tsrv",
    *,
    overlapping: bool = False,
) -> pd.DataFrame:
    """Estimate the Hurst exponent for each column in a log-RV panel."""
    records = []
    for asset in log_rv_panel.columns:
        # Keep the shared calendar grid.  Dropping missing dates would turn
        # multi-day gaps into one-day increments.
        series = log_rv_panel[asset].to_numpy(dtype=np.float64)
        if series.size < delta_max + 1 or np.count_nonzero(np.isfinite(series)) < 2:
            continue
        result = estimate_hurst_ols(
            series, delta_max=delta_max, overlapping=overlapping
        )
        records.append(
            {
                "asset": str(asset),
                "H_hat": result["H_hat"],
                "r_squared": result["r_squared"],
                "n_obs": result["n_obs"],
                "estimator": estimator,
            }
        )
    return pd.DataFrame(records)
