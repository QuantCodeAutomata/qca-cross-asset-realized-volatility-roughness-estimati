"""
Temporal aggregation of daily realized variance to weekly and monthly log-RV.

Weekly  aggregation: sum daily RV within each ISO trading week, then log-transform.
Monthly aggregation: sum daily RV within each calendar month, then log-transform.
"""

import numpy as np
import pandas as pd


def aggregate_rv_to_weekly(daily_rv: pd.Series) -> pd.Series:
    """Aggregate daily RV to weekly log-RV.

    Within each ISO calendar week (Monday-Friday) the daily RV values are summed
    to produce the weekly RV, then natural-log transformed.  Weeks with zero or
    NaN total are dropped.

    Parameters
    ----------
    daily_rv : pd.Series
        Daily realized variance, DatetimeIndex.

    Returns
    -------
    pd.Series
        Weekly log-RV indexed by the Monday of each week.
    """
    rv = daily_rv.copy().dropna()
    rv.index = pd.DatetimeIndex(rv.index)
    # Group by ISO year-week using the Monday as the period label
    weekly_sum = rv.resample("W-MON", closed="left", label="left").sum()
    weekly_sum = weekly_sum[weekly_sum > 0]
    return np.log(weekly_sum)


def aggregate_rv_to_monthly(daily_rv: pd.Series) -> pd.Series:
    """Aggregate daily RV to monthly log-RV.

    Within each calendar month the daily RV values are summed to produce
    monthly RV, then natural-log transformed.  Months with zero or NaN total
    are dropped.

    Parameters
    ----------
    daily_rv : pd.Series
        Daily realized variance, DatetimeIndex.

    Returns
    -------
    pd.Series
        Monthly log-RV indexed by the first calendar day of each month.
    """
    rv = daily_rv.copy().dropna()
    rv.index = pd.DatetimeIndex(rv.index)
    monthly_sum = rv.resample("MS").sum()
    monthly_sum = monthly_sum[monthly_sum > 0]
    return np.log(monthly_sum)
