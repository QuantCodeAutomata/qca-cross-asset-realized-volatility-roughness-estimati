"""
Futures preprocessing: continuous front-month series via volume-overtake roll,
and within-session return computation.
"""

import numpy as np
import pandas as pd


def build_continuous_front_month(contracts_df: pd.DataFrame) -> pd.DataFrame:
    """Build a continuous front-month futures series using the volume-overtake roll rule.

    The active contract switches to the next contract on the first date where
    its traded volume exceeds the current front contract's volume.

    Parameters
    ----------
    contracts_df : pd.DataFrame
        Columns required: date, expiry (YYYYMMDD str or int),
        close, volume.

    Returns
    -------
    pd.DataFrame
        Columns: date, close, volume, active_expiry, rolled
        (bool, True on roll dates).
    """
    df = contracts_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "expiry"]).reset_index(drop=True)

    records = []
    dates = df["date"].unique()
    dates.sort()

    current_expiry = df.loc[df["date"] == dates[0], "expiry"].min()

    for date in dates:
        day = df[df["date"] == date]
        current_row = day[day["expiry"] == current_expiry]

        if current_row.empty:
            # Current contract no longer traded; roll to lowest expiry available
            current_expiry = day["expiry"].min()
            current_row = day[day["expiry"] == current_expiry]

        current_vol = float(current_row["volume"].iloc[0]) if not current_row.empty else 0.0
        current_close = float(current_row["close"].iloc[0]) if not current_row.empty else np.nan

        # Check for volume overtake by any later-expiry contract
        later = day[day["expiry"] > current_expiry]
        rolled = False
        if not later.empty:
            front_later = later.loc[later["expiry"].idxmin()]
            if float(front_later["volume"]) > current_vol:
                current_expiry = front_later["expiry"]
                current_close  = float(front_later["close"])
                current_vol    = float(front_later["volume"])
                rolled = True

        records.append({
            "date":          date,
            "close":         current_close,
            "volume":        current_vol,
            "active_expiry": current_expiry,
            "rolled":        rolled,
        })

    return pd.DataFrame(records)


def compute_futures_intraday_returns(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Compute within-session log returns for futures, excluding overnight gaps.

    Parameters
    ----------
    intraday_df : pd.DataFrame
        Columns: date, bar_idx (0-based), close (or log_price).
        Bars are sorted by bar_idx within each date group.

    Returns
    -------
    pd.DataFrame
        Columns: date, bar_idx, return_ (within-session log return;
        NaN for bar_idx == 0 of each day).
    """
    price_col = "log_price" if "log_price" in intraday_df.columns else "close"
    records = []
    for date, group in intraday_df.groupby("date"):
        group = group.sort_values("bar_idx").reset_index(drop=True)
        prices = group[price_col].values.astype(np.float64)
        if price_col == "close":
            lp = np.log(prices)
        else:
            lp = prices
        returns = np.empty(len(lp), dtype=np.float64)
        returns[0] = np.nan
        returns[1:] = np.diff(lp)
        for i, row in group.iterrows():
            records.append({
                "date":    date,
                "bar_idx": int(row["bar_idx"]),
                "return_": returns[i],
            })
    return pd.DataFrame(records)
