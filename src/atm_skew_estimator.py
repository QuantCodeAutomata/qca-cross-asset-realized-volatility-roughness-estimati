"""
ATM Implied Volatility Skew Estimator.
Exp_4: Estimate ATM skew ψ_{t,T} = dIV/dk|_{k=0} for each date-expiry pair,
where k = log(K/F) is forward log-moneyness.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Tuple, Dict


def compute_log_moneyness(strikes: np.ndarray, forward: float) -> np.ndarray:
    """Compute forward log-moneyness k = log(K / F).

    Args:
        strikes: array of strike prices.
        forward: forward price or futures settlement price.

    Returns:
        log-moneyness array of same shape as ``strikes``.
    """
    return np.log(np.asarray(strikes, dtype=np.float64) / forward)


def estimate_atm_skew(strikes: np.ndarray, ivols: np.ndarray, forward: float,
                       moneyness_band: float = 0.08,
                       min_strikes: int = 10) -> Optional[dict]:
    """Estimate ATM skew by regressing IV on forward log-moneyness.

    Near-ATM linear approximation:
        IV(k) ≈ IV_ATM + ψ · k,   |k| ≤ moneyness_band

    The OLS slope ψ = dIV/dk|_{k=0} is the ATM skew in IV-per-log-moneyness
    units.  A negative ψ reflects the typical equity skew (puts more expensive
    than calls).

    Args:
        strikes: array of strike prices.
        ivols: array of implied volatilities (same length as ``strikes``).
        forward: forward / futures price.
        moneyness_band: half-width |k| ≤ moneyness_band filter (default 0.08).
        min_strikes: minimum number of valid strikes required.

    Returns:
        dict with keys ``psi``, ``atm_iv``, ``r_squared``, ``n_strikes``,
        ``valid``, or ``None`` if there are fewer than ``min_strikes`` valid
        observations.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    ivols = np.asarray(ivols, dtype=np.float64)

    k = compute_log_moneyness(strikes, forward)
    mask = (np.abs(k) <= moneyness_band) & np.isfinite(ivols) & (ivols > 0.0)

    if mask.sum() < min_strikes:
        return None

    k_sel = k[mask]
    iv_sel = ivols[mask]

    X = sm.add_constant(k_sel)
    res = sm.OLS(iv_sel, X).fit()

    atm_iv = float(res.params[0])      # intercept = IV at k=0
    psi = float(res.params[1])         # slope = dIV/dk|_{k=0}
    r_squared = float(res.rsquared)
    n_strikes = int(mask.sum())

    return {
        'psi': psi,
        'atm_iv': atm_iv,
        'r_squared': r_squared,
        'n_strikes': n_strikes,
        'valid': True,
    }


def build_skew_term_structure(option_chain: pd.DataFrame,
                               forward_map: dict,
                               discount_map: dict,
                               maturity_window: Tuple[float, float] = (14 / 365, 1.0),
                               min_strikes: int = 10) -> pd.DataFrame:
    """Build ATM skew term structure from an option chain.

    For every (date, expiry) pair in ``option_chain`` whose time-to-maturity
    lies within ``maturity_window``, compute the ATM skew ψ_{t,T}.

    Args:
        option_chain: DataFrame with columns ``date``, ``expiry``, ``strike``,
                      ``iv``, ``option_type``, ``volume``.
        forward_map: ``{(date, expiry): forward_price}`` mapping.
        discount_map: ``{(date, expiry): discount_factor}`` mapping.
        maturity_window: ``(T_min, T_max)`` in years; pairs outside are skipped.
        min_strikes: minimum strikes required per (date, expiry) slice.

    Returns:
        DataFrame with columns ``date``, ``expiry``, ``T``, ``psi``,
        ``atm_iv``, ``r_squared``, ``n_strikes``, ``valid``.
    """
    T_min, T_max = maturity_window
    records = []

    for (date, expiry), group in option_chain.groupby(['date', 'expiry']):
        # Time to maturity
        if hasattr(date, 'to_pydatetime') and hasattr(expiry, 'to_pydatetime'):
            T = (expiry - date).days / 365.0
        else:
            T = float(expiry - date)

        if not (T_min <= T <= T_max):
            continue

        forward = forward_map.get((date, expiry), None)
        if forward is None or forward <= 0.0:
            continue

        strikes = group['strike'].values
        ivols = group['iv'].values

        result = estimate_atm_skew(strikes, ivols, forward,
                                    min_strikes=min_strikes)

        if result is None:
            records.append({
                'date': date, 'expiry': expiry, 'T': T,
                'psi': np.nan, 'atm_iv': np.nan,
                'r_squared': np.nan, 'n_strikes': 0, 'valid': False,
            })
        else:
            records.append({
                'date': date, 'expiry': expiry, 'T': T, **result,
            })

    if not records:
        return pd.DataFrame(columns=['date', 'expiry', 'T', 'psi',
                                      'atm_iv', 'r_squared', 'n_strikes', 'valid'])
    return pd.DataFrame(records)
