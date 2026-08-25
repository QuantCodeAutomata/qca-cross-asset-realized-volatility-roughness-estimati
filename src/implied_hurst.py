"""Estimate an option-implied Hurst exponent from ATM-skew term structure."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _valid_skew_rows(
    skew_ts: pd.DataFrame,
    maturity_window: Tuple[float, float],
) -> pd.DataFrame:
    required = {"date", "T", "psi"}
    missing = required.difference(skew_ts.columns)
    if missing:
        raise ValueError(f"skew_ts is missing columns: {sorted(missing)}")
    t_min, t_max = maturity_window
    valid = skew_ts["valid"].astype(bool) if "valid" in skew_ts else True
    mask = (
        valid
        & np.isfinite(skew_ts["T"])
        & (skew_ts["T"] >= t_min)
        & (skew_ts["T"] <= t_max)
        & np.isfinite(skew_ts["psi"])
        & (skew_ts["psi"] != 0.0)
    )
    return skew_ts.loc[mask].copy()


def estimate_daily_implied_hurst(
    skew_ts: pd.DataFrame,
    date: pd.Timestamp,
    maturity_window: Tuple[float, float] = (14 / 365, 1.0),
) -> Optional[dict]:
    """Fit ``log|psi| = a_t + beta_t log(T)`` on one date."""
    subset = _valid_skew_rows(skew_ts, maturity_window)
    subset = subset[subset["date"] == date]
    if len(subset) < 3 or subset["T"].nunique() < 2:
        return None

    log_t = np.log(subset["T"].to_numpy(dtype=np.float64))
    log_abs_psi = np.log(np.abs(subset["psi"].to_numpy(dtype=np.float64)))
    result = sm.OLS(log_abs_psi, sm.add_constant(log_t)).fit()
    beta = float(result.params[1])
    return {
        "date": date,
        "H_hat": beta + 0.5,
        "beta": beta,
        "beta_se": float(result.bse[1]),
        "H_se": float(result.bse[1]),
        "r_squared": float(result.rsquared),
        "n_obs": int(len(subset)),
    }


def estimate_pooled_implied_hurst(
    skew_ts: pd.DataFrame,
    maturity_window: Tuple[float, float] = (14 / 365, 1.0),
    min_obs_per_date: int = 2,
) -> dict:
    """Pooled date-fixed-effect skew regression.

    Date effects are removed by within-date demeaning.  The returned standard
    error is cluster-robust by date, matching the paper's statement that dates
    are treated as independent units.
    """
    df = _valid_skew_rows(skew_ts, maturity_window)
    counts = df.groupby("date")["T"].count()
    valid_dates = counts[counts >= min_obs_per_date].index
    df = df[df["date"].isin(valid_dates)].copy()

    empty = {
        "H_hat_IV": np.nan,
        "beta": np.nan,
        "beta_se": np.nan,
        "H_se": np.nan,
        "r_psi_squared": np.nan,
        "identified": False,
        "n_obs": int(len(df)),
        "n_dates": int(df["date"].nunique()) if len(df) else 0,
    }
    if len(df) < 4 or df["date"].nunique() < 2:
        return empty

    df["log_T"] = np.log(df["T"].to_numpy(dtype=np.float64))
    df["log_abs_psi"] = np.log(np.abs(df["psi"].to_numpy(dtype=np.float64)))
    df["x"] = df["log_T"] - df.groupby("date")["log_T"].transform("mean")
    df["y"] = (
        df["log_abs_psi"]
        - df.groupby("date")["log_abs_psi"].transform("mean")
    )

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    xx = float(np.dot(x, x))
    if xx <= np.finfo(float).eps:
        return empty

    beta = float(np.dot(x, y) / xx)
    residual = y - beta * x
    ss_tot = float(np.dot(y, y))
    ss_res = float(np.dot(residual, residual))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    # One-regressor cluster sandwich variance.  The small-sample multiplier is
    # the standard CR1 correction.
    score_by_date = (
        pd.DataFrame({"date": df["date"].to_numpy(), "score": x * residual})
        .groupby("date")["score"]
        .sum()
        .to_numpy(dtype=np.float64)
    )
    n_obs = len(df)
    n_dates = int(df["date"].nunique())
    meat = float(np.dot(score_by_date, score_by_date))
    correction = 1.0
    if n_dates > 1:
        correction = n_dates / (n_dates - 1.0)
    beta_var = correction * meat / (xx**2)
    beta_se = float(np.sqrt(max(beta_var, 0.0)))

    return {
        "H_hat_IV": beta + 0.5,
        "beta": beta,
        "beta_se": beta_se,
        "H_se": beta_se,
        "r_psi_squared": r2,
        "identified": bool(r2 >= 0.3),
        "n_obs": int(n_obs),
        "n_dates": n_dates,
    }
