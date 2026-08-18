"""
Equity intraday preprocessing: minute-bar grid, forward-filling, and
within-session return computation for realized-variance estimation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Callable, Optional

SESSION_MINUTES: int = 390  # Regular NYSE/Nasdaq session (9:30 AM – 4:00 PM ET)


def build_minute_grid(date: pd.Timestamp) -> pd.DatetimeIndex:
    """Build the 390-bar regular-session minute grid for a given trading date.

    Session: 09:30 AM – 04:00 PM Eastern Time (close-bar timestamps).

    Parameters
    ----------
    date : pd.Timestamp
        The trading date (date part used only).

    Returns
    -------
    pd.DatetimeIndex
        Length 390, minute-frequency, timezone-naive.
    """
    open_time  = pd.Timestamp(date.date()) + pd.Timedelta(hours=9, minutes=30)
    close_time = pd.Timestamp(date.date()) + pd.Timedelta(hours=16, minutes=0)
    return pd.date_range(start=open_time, end=close_time, freq="1min")[: SESSION_MINUTES]


def forward_fill_minute_prices(
    prices: pd.Series,
    date: pd.Timestamp,
) -> Tuple[pd.Series, float]:
    """Reindex a price series onto the 390-bar minute grid and forward-fill gaps.

    Parameters
    ----------
    prices : pd.Series
        Observed bar prices indexed by pd.DatetimeIndex (subset of the grid).
    date : pd.Timestamp
        Trading date for the grid.

    Returns
    -------
    filled_prices : pd.Series
        Length 390, forward-filled; remaining leading NaN bars are back-filled
        from the first observed price.
    fill_fraction : float
        Fraction of grid bars that had an observed price before filling
        (observed_bars / 390).
    """
    grid = build_minute_grid(date)
    observed = prices.reindex(grid)
    fill_fraction = float(observed.notna().sum()) / SESSION_MINUTES
    filled = observed.ffill().bfill()
    return filled, fill_fraction


def compute_intraday_returns(prices: pd.Series) -> np.ndarray:
    """Compute within-session log returns, excluding the overnight gap.

    Takes consecutive differences of log prices starting at bar index 1
    (bar 0 is the session open; its return relative to the prior close is
    excluded to avoid overnight contamination).

    Parameters
    ----------
    prices : pd.Series
        Log prices or price levels on the session grid (length >= 2).

    Returns
    -------
    np.ndarray
        Within-session log returns, length (len(prices) - 1).
        If prices are already log-prices, this is simply np.diff(prices).
        If prices are levels, np.diff(np.log(prices)) is applied.
    """
    p = np.asarray(prices, dtype=np.float64)
    if np.all(p > 0):
        # Price levels: take log first
        lp = np.log(p)
    else:
        # Already log-prices
        lp = p
    return np.diff(lp)


def build_daily_rv_panel(
    intraday_df: pd.DataFrame,
    estimator_fn: Callable[[np.ndarray], float],
) -> pd.DataFrame:
    """Build a panel of daily realized-variance estimates from intraday data.

    Parameters
    ----------
    intraday_df : pd.DataFrame
        Must have columns: date (or be groupable by date), bar_idx
        (0-based), log_price.  The return_ column is used if present,
        otherwise computed from log_price.
    estimator_fn : callable
        Function mapping a 1-D return array to a RV scalar.

    Returns
    -------
    pd.DataFrame
        Index: unique dates; single column rv with daily RV estimates.
    """
    records = []
    for date, group in intraday_df.groupby("date"):
        group = group.sort_values("bar_idx")
        if "return_" in group.columns:
            returns = group["return_"].dropna().values
        else:
            lp = group["log_price"].values
            returns = np.diff(lp.astype(np.float64))
        if len(returns) < 5:
            continue
        rv = estimator_fn(returns)
        records.append({"date": date, "rv": rv})
    df = pd.DataFrame(records).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df
