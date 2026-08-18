"""
Publication-quality figures for realized-volatility roughness analysis.

All plotting functions save to RESULTS_DIR and optionally display inline.
Non-interactive Agg backend is set at import so the module is safe in headless
environments.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Global style
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


def plot_hurst_distribution(
    hurst_df: pd.DataFrame,
    title: str = "Hurst Exponent Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot histogram of H estimates across assets.

    Parameters
    ----------
    hurst_df : pd.DataFrame
        Must contain column H_hat.  Optionally estimator for facets.
    title : str
        Figure title.
    save_path : Path, optional
        If provided, the figure is saved there; otherwise saved to RESULTS_DIR.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    if "estimator" in hurst_df.columns and hurst_df["estimator"].nunique() > 1:
        for est, sub in hurst_df.groupby("estimator"):
            ax.hist(sub["H_hat"].dropna(), bins=20, alpha=0.5, label=est, density=True)
        ax.legend()
    else:
        ax.hist(hurst_df["H_hat"].dropna(), bins=20, color="steelblue",
                edgecolor="white", density=True)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="H=0.5 (BM)")
    ax.set_xlabel("H estimate")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()

    out = save_path or (RESULTS_DIR / "hurst_distribution.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_loglog_fit(
    deltas: np.ndarray,
    m2_values: np.ndarray,
    H_hat: float,
    asset_name: str = "Asset",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot log-log scaling fit for a single asset.

    Shows the empirical m(2, Delta) points and the OLS fitted line
    log m(2, Delta) = a + 2*H*log(Delta).

    Parameters
    ----------
    deltas     : array of lag values (x-axis)
    m2_values  : array of empirical second moments (y-axis)
    H_hat      : estimated Hurst exponent (for the legend)
    asset_name : label for the title
    save_path  : optional output path

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid = np.isfinite(m2_values) & (m2_values > 0)
    log_d  = np.log(deltas[valid])
    log_m2 = np.log(m2_values[valid])

    # OLS fit for the line
    b = np.polyfit(log_d, log_m2, deg=1)
    log_d_grid = np.linspace(log_d.min(), log_d.max(), 100)
    fit_line   = np.polyval(b, log_d_grid)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(log_d, log_m2, color="steelblue", zorder=3, label="Empirical")
    ax.plot(log_d_grid, fit_line, color="tomato", linewidth=1.5,
            label=f"Fit  H={H_hat:.3f}")
    ax.set_xlabel("log Delta")
    ax.set_ylabel("log m(2, Delta)")
    ax.set_title(f"Log-log scaling — {asset_name}")
    ax.legend()
    fig.tight_layout()

    out = save_path or (RESULTS_DIR / f"loglog_{asset_name.replace(' ','_')}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_futures_class_medians(
    futures_hurst: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot median Hurst estimates grouped by futures asset class.

    Parameters
    ----------
    futures_hurst : pd.DataFrame
        Columns: H_hat, asset_class (e.g. equity_index, energy, metals,
        fixed_income, fx).
    save_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "asset_class" not in futures_hurst.columns:
        raise ValueError("futures_hurst must have an 'asset_class' column.")

    order = (
        futures_hurst.groupby("asset_class")["H_hat"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=futures_hurst,
        x="asset_class",
        y="H_hat",
        order=order,
        palette="colorblind",
        width=0.5,
        ax=ax,
    )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Asset class")
    ax.set_ylabel("H estimate")
    ax.set_title("Realized Hurst exponent by futures asset class")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out = save_path or (RESULTS_DIR / "futures_class_medians.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig
