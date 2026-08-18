"""
Mean-reversion bias analysis for the H estimator under fOU dynamics.
Exp_2: Quantify OLS bias in short-lag realized-volatility Hurst estimation.
"""
import numpy as np
import pandas as pd
from scipy import special
from typing import List, Tuple
import statsmodels.api as sm

from src.fou_spectral import compute_m2_grid, local_slope_asymptotic


def compute_ols_weights(delta_max: int) -> np.ndarray:
    """Compute OLS weights w_Delta for the log-log regression.

    w_Delta = (log(Delta) - mean(log(Delta))) / sum_{Delta'} (log(Delta') - mean(log(Delta')))^2

    These are the standard OLS projection weights onto log(Delta).

    Args:
        delta_max: maximum lag

    Returns:
        weights array of shape (delta_max,)
    """
    deltas = np.arange(1, delta_max + 1, dtype=np.float64)
    log_deltas = np.log(deltas)
    centered = log_deltas - log_deltas.mean()
    return centered / np.sum(centered**2)


def compute_ols_bias_approximation(H: float, kappa: float, delta_max: int) -> float:
    """Compute the OLS bias approximation for H_hat.

    E[H_hat] approximately H - [Gamma(1+2H)/4] * kappa^(2-2H) * sum_{Delta=1}^{delta_max} w_Delta * Delta^(2-2H)

    Derivation: log M_2(Delta) ≈ const + 2H*log(Delta) - (Gamma(1+2H)/2)*(kappa*Delta)^(2-2H)
    so the OLS slope b estimates 2H - (Gamma(1+2H)/2)*kappa^(2-2H)*sum_Delta w_Delta*Delta^(2-2H),
    and H_hat = b/2 gives the formula above.

    Args:
        H: true Hurst exponent
        kappa: mean reversion rate
        delta_max: maximum lag window

    Returns:
        Approximate bias (H_hat - H)
    """
    if kappa == 0.0:
        return 0.0
    weights = compute_ols_weights(delta_max)
    deltas = np.arange(1, delta_max + 1, dtype=np.float64)
    coeff = special.gamma(1.0 + 2.0 * H) / 4.0
    weighted_sum = np.sum(weights * deltas ** (2.0 - 2.0 * H))
    return -coeff * kappa ** (2.0 - 2.0 * H) * weighted_sum


def compute_exact_ols_h(H: float, kappa: float, delta_max: int = 10,
                         nu: float = 1.0) -> Tuple[float, float]:
    """Compute exact finite-window OLS H estimate from numerical M_2(Delta) values.

    1. Compute M_2(Delta) for Delta=1,...,delta_max via spectral integration
    2. Fit OLS: log(M_2(Delta)) = a + b*log(Delta)
    3. Return H_hat = b/2 and bias = H_hat - H

    Args:
        H: true Hurst exponent
        kappa: mean reversion rate
        delta_max: max lag
        nu: vol of vol

    Returns:
        (H_hat, bias) tuple
    """
    deltas = np.arange(1, delta_max + 1, dtype=np.float64)
    m2 = compute_m2_grid(deltas, H, kappa, nu)
    log_delta = np.log(deltas)
    log_m2 = np.log(m2)
    # OLS: log_m2 = a + b * log_delta
    X = sm.add_constant(log_delta)
    ols = sm.OLS(log_m2, X).fit()
    b = ols.params[1]
    H_hat = b / 2.0
    bias = H_hat - H
    return H_hat, bias


def build_bias_table(H_values: List[float], kappa_values: List[float],
                      delta_max_values: List[int] = None) -> pd.DataFrame:
    """Build comprehensive bias table over (H, kappa, delta_max) grid.

    Args:
        H_values: list of H values to test
        kappa_values: list of kappa values
        delta_max_values: list of delta_max values

    Returns:
        DataFrame with columns: H, kappa, delta_max, H_hat_exact, bias_exact, bias_approx, alpha_delta1
    """
    if delta_max_values is None:
        delta_max_values = [10, 40]

    rows = []
    total = len(H_values) * len(kappa_values) * len(delta_max_values)
    count = 0
    for H in H_values:
        for kappa in kappa_values:
            for delta_max in delta_max_values:
                count += 1
                print(f"  [{count}/{total}] H={H:.2f}, kappa={kappa:.3f}, delta_max={delta_max}")
                H_hat, bias_exact = compute_exact_ols_h(H, kappa, delta_max)
                bias_approx = compute_ols_bias_approximation(H, kappa, delta_max)
                alpha1 = local_slope_asymptotic(1.0, H, kappa)
                rows.append({
                    "H": H,
                    "kappa": kappa,
                    "delta_max": delta_max,
                    "H_hat_exact": H_hat,
                    "bias_exact": bias_exact,
                    "bias_approx": bias_approx,
                    "alpha_delta1": alpha1,
                })

    return pd.DataFrame(rows)
