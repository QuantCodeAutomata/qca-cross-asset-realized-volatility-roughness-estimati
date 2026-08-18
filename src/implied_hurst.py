"""
Implied Hurst Estimation from ATM Skew Term Structure.
Exp_4: Estimate H from |ψ(T)| ~ c · T^(H − 1/2) power law in the skew term
structure.  Pooled OLS with date fixed effects is the primary estimator.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Tuple, Dict


def estimate_daily_implied_hurst(skew_ts: pd.DataFrame, date: pd.Timestamp,
                                  maturity_window: Tuple[float, float] = (14 / 365, 1.0)
                                  ) -> Optional[dict]:
    """Estimate implied H for a single observation date.

    Power-law model:  log|ψ(T)| = a_t + β_t · log T + u_{t,T}
    Hurst estimate:   Ĥ_t = β̂_t + 0.5

    Args:
        skew_ts: DataFrame with columns ``date``, ``T``, ``psi``, ``valid``.
        date: date to estimate H for.
        maturity_window: ``(T_min, T_max)`` in years; rows outside are excluded.

    Returns:
        dict with keys ``date``, ``H_hat``, ``beta``, ``r_squared``, ``n_obs``,
        or ``None`` if fewer than 3 valid observations are available.
    """
    T_min, T_max = maturity_window
    mask = (
        (skew_ts['date'] == date)
        & skew_ts.get('valid', pd.Series(True, index=skew_ts.index))
        & (skew_ts['T'] >= T_min)
        & (skew_ts['T'] <= T_max)
        & np.isfinite(skew_ts['psi'])
        & (skew_ts['psi'] != 0.0)
    )
    subset = skew_ts.loc[mask]

    if len(subset) < 3:
        return None

    log_T = np.log(subset['T'].values)
    log_abs_psi = np.log(np.abs(subset['psi'].values))

    X = sm.add_constant(log_T)
    res = sm.OLS(log_abs_psi, X).fit()

    beta = float(res.params[1])
    H_hat = beta + 0.5

    return {
        'date': date,
        'H_hat': H_hat,
        'beta': beta,
        'r_squared': float(res.rsquared),
        'n_obs': len(subset),
    }


def estimate_pooled_implied_hurst(skew_ts: pd.DataFrame,
                                   maturity_window: Tuple[float, float] = (14 / 365, 1.0),
                                   min_obs_per_date: int = 2) -> dict:
    """Estimate pooled implied H across all dates with date fixed effects.

    Panel regression:
        log|ψ_{t,T}| = α_t + β · log T + u_{t,T}

    where α_t are date-specific intercepts (within-group demeaning).

    Pooled R²_ψ ≥ 0.3 threshold: ``identified = True``.

    Args:
        skew_ts: DataFrame with columns ``date``, ``T``, ``psi``, ``valid``.
        maturity_window: ``(T_min, T_max)`` in years.
        min_obs_per_date: minimum maturity observations per date to include.

    Returns:
        dict with keys ``H_hat_IV``, ``beta``, ``r_psi_squared``,
        ``identified``, ``n_obs``, ``n_dates``.
    """
    T_min, T_max = maturity_window

    valid_col = skew_ts.get('valid', pd.Series(True, index=skew_ts.index))
    mask = (
        valid_col.values.astype(bool)
        & (skew_ts['T'].values >= T_min)
        & (skew_ts['T'].values <= T_max)
        & np.isfinite(skew_ts['psi'].values)
        & (skew_ts['psi'].values != 0.0)
    )
    df = skew_ts.loc[mask].copy()

    # Keep dates with enough maturities
    counts = df.groupby('date')['T'].count()
    valid_dates = counts[counts >= min_obs_per_date].index
    df = df[df['date'].isin(valid_dates)].copy()

    if len(df) < 4:
        return {
            'H_hat_IV': np.nan,
            'beta': np.nan,
            'r_psi_squared': np.nan,
            'identified': False,
            'n_obs': len(df),
            'n_dates': 0,
        }

    df['log_T'] = np.log(df['T'])
    df['log_abs_psi'] = np.log(np.abs(df['psi']))

    # Within-group demeaning to absorb date fixed effects
    df['log_T_dm'] = df['log_T'] - df.groupby('date')['log_T'].transform('mean')
    df['log_abs_psi_dm'] = (
        df['log_abs_psi'] - df.groupby('date')['log_abs_psi'].transform('mean')
    )

    X = df['log_T_dm'].values
    y = df['log_abs_psi_dm'].values

    # OLS (no constant since demeaned)
    beta = float(np.dot(X, y) / np.dot(X, X))

    y_hat = beta * X
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum(y ** 2))
    r_psi_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    H_hat_IV = beta + 0.5
    n_dates = int(df['date'].nunique())
    identified = bool(r_psi_squared >= 0.3)

    return {
        'H_hat_IV': H_hat_IV,
        'beta': beta,
        'r_psi_squared': r_psi_squared,
        'identified': identified,
        'n_obs': len(df),
        'n_dates': n_dates,
    }
