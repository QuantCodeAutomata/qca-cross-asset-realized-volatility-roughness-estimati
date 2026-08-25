"""Equity intraday preprocessing for the realised-volatility pipeline."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd

SESSION_MINUTES: int = 390


def build_minute_grid(
    date: pd.Timestamp,
    *,
    timestamp_convention: str = "close",
) -> pd.DatetimeIndex:
    """Build the 390-point regular-session minute grid.

    ``timestamp_convention='close'`` returns 09:31, …, 16:00, which is the
    actual close timestamp of each one-minute interval.  ``'start'`` returns
    09:30, …, 15:59.  The submitted code documented close timestamps but
    silently built the start-timestamp grid.
    """
    day = pd.Timestamp(date).normalize()
    if timestamp_convention == "close":
        start = day + pd.Timedelta(hours=9, minutes=31)
        end = day + pd.Timedelta(hours=16)
    elif timestamp_convention == "start":
        start = day + pd.Timedelta(hours=9, minutes=30)
        end = day + pd.Timedelta(hours=15, minutes=59)
    else:
        raise ValueError("timestamp_convention must be 'close' or 'start'")
    grid = pd.date_range(start=start, end=end, freq="1min")
    if len(grid) != SESSION_MINUTES:
        raise RuntimeError("regular-session grid must contain 390 minutes")
    return grid


def forward_fill_minute_prices(
    prices: pd.Series,
    date: pd.Timestamp,
    *,
    timestamp_convention: str = "close",
    fill_leading: bool = False,
) -> Tuple[pd.Series, float]:
    """Reindex to the session grid and forward-fill missing bars.

    Leading missing bars have no prior in-session observation and therefore
    remain missing by default.  ``fill_leading=True`` reproduces the previous
    back-fill behaviour, but that introduces information from the future and
    should not be used in causal preprocessing.
    """
    grid = build_minute_grid(date, timestamp_convention=timestamp_convention)
    observed = prices.reindex(grid)
    fill_fraction = float(observed.notna().sum()) / SESSION_MINUTES
    filled = observed.ffill()
    if fill_leading:
        filled = filled.bfill()
    return filled, fill_fraction


def compute_intraday_returns(
    prices: pd.Series | np.ndarray,
    *,
    input_is_log: bool = False,
) -> np.ndarray:
    """Compute consecutive within-session log returns.

    The input representation is explicit.  Inferring it from positivity is
    unsafe because ordinary log prices are often positive and were previously
    logged a second time.
    """
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    if p.size < 2:
        return np.empty(0, dtype=np.float64)

    if input_is_log:
        log_prices = p
    else:
        if np.any(p <= 0.0):
            raise ValueError("price levels must be strictly positive")
        log_prices = np.log(p)
    return np.diff(log_prices)


def build_daily_rv_panel(
    intraday_df: pd.DataFrame,
    estimator_fn: Callable[[np.ndarray], float],
) -> pd.DataFrame:
    """Build a one-column daily realised-variance panel."""
    records = []
    for date, group in intraday_df.groupby("date"):
        group = group.sort_values("bar_idx")
        if "return_" in group.columns:
            returns = group["return_"].dropna().to_numpy(dtype=np.float64)
        else:
            returns = compute_intraday_returns(
                group["log_price"].to_numpy(dtype=np.float64), input_is_log=True
            )
        if returns.size < 5:
            continue
        records.append({"date": date, "rv": estimator_fn(returns)})

    if not records:
        return pd.DataFrame(columns=["rv"], index=pd.DatetimeIndex([], name="date"))
    result = pd.DataFrame(records).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result
