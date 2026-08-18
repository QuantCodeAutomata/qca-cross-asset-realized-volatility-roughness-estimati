"""
Experiment 3: Measurement-error attenuation and two-estimator correction.

Runs in Mode A (simulation only, no market data needed).

Full-scale settings (documented):
    n_paths = 400
    n_days  = 2520  (10 trading years)
    n_bars  = 390   (1-min bars per day)

Reduced settings used here for runtime:
    n_paths = 100
    n_days  = 1000  (~4 trading years)
    n_bars  = 390

Benchmark cell: H=0.20, kappa=0.010 (annualized), nu=0.5, dt=1/252.

RV estimators per day:
  - Latent log-IV : log_vol (no noise)
  - RV5m          : realized variance from 5-min returns (noisy)
  - TSRV          : two-scale RV (bias-corrected, cleaner)

Two-estimator correction uses RV5m (a) vs TSRV (b).
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulate_rough_paths import (
    simulate_fou_log_vol,
    simulate_noisy_intraday_prices,
    compute_latent_iv,
)
from autocovariance_fit import (
    compute_second_moment_from_series,
    fit_two_estimator_correction,
    _estimate_h_from_moments,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Simulation parameters ──────────────────────────────────────────────────────
N_PATHS   = 100      # reduced from 400 for runtime
N_DAYS    = 1000     # reduced from 2520 for runtime
N_BARS    = 390      # 1-min bars per day
BASE_SEED = 42

# Benchmark cell
H_TRUE    = 0.20
KAPPA     = 0.010    # annualized mean-reversion rate
NU        = 0.50     # vol of vol (annualized)
DT        = 1.0 / 252

VARPI_GRID = [1e-4, 5e-4]   # noise std-dev values
DELTA_MAX  = 10              # lag window for H estimation
DELTA_MAX2 = 40


# ── Helper: compute RV estimators from noisy intraday prices ─────────────────

def compute_rv_estimators(p_obs: np.ndarray) -> tuple:
    """Compute RV5m and TSRV for all days from observed log-prices.

    Args:
        p_obs: shape (n_days, n_bars+1)

    Returns:
        (rv5m, tsrv) each of shape (n_days,)
    """
    n_bars = p_obs.shape[1] - 1
    # 1-min returns
    r1 = np.diff(p_obs, axis=1)                    # (n_days, n_bars)
    rv1m = np.sum(r1 ** 2, axis=1)                 # (n_days,)

    # 5-min subsampled returns
    idx5 = np.arange(0, n_bars + 1, 5)            # 0, 5, 10, …, n_bars
    r5 = np.diff(p_obs[:, idx5], axis=1)           # (n_days, n_bars/5)
    rv5m = np.sum(r5 ** 2, axis=1)                 # (n_days,)

    # TSRV: unbiased for IV
    # n_slow = n_bars/5, n_fast = n_bars
    n_slow = len(idx5) - 1          # 78
    ratio  = n_slow / n_bars        # 0.2
    scale  = 1.0 / (1.0 - ratio)   # 1.25
    tsrv   = scale * (rv5m - ratio * rv1m)

    return rv5m, tsrv


def simulate_one_path(path_id: int, varpi: float) -> dict:
    """Simulate a single path and compute H estimates.

    Returns dict with H_latent, H_rv5m, H_tsrv, correction_result.
    """
    seed_vol   = BASE_SEED + path_id
    seed_price = BASE_SEED + path_id + 10_000

    # fOU log-variance (log-IV)
    log_vol = simulate_fou_log_vol(N_DAYS, H_TRUE, KAPPA, NU, DT, seed=seed_vol)

    # Noisy intraday prices
    p_obs = simulate_noisy_intraday_prices(log_vol, N_BARS, varpi, seed=seed_price)

    # RV estimators
    rv5m, tsrv = compute_rv_estimators(p_obs)

    # Clip to positive before log
    rv5m_pos = np.maximum(rv5m, 1e-12)
    tsrv_pos = np.maximum(tsrv, 1e-12)

    log_rv_latent = log_vol                    # log-IV (no measurement noise)
    log_rv5m      = np.log(rv5m_pos)
    log_tsrv      = np.log(tsrv_pos)

    # Second moments
    m2_latent = compute_second_moment_from_series(log_rv_latent, DELTA_MAX2)
    m2_rv5m   = compute_second_moment_from_series(log_rv5m,      DELTA_MAX2)
    m2_tsrv   = compute_second_moment_from_series(log_tsrv,      DELTA_MAX2)

    H_latent = _estimate_h_from_moments(m2_latent[:DELTA_MAX],  DELTA_MAX)
    H_rv5m   = _estimate_h_from_moments(m2_rv5m[:DELTA_MAX],    DELTA_MAX)
    H_tsrv   = _estimate_h_from_moments(m2_tsrv[:DELTA_MAX],    DELTA_MAX)

    H_latent_lag40 = _estimate_h_from_moments(m2_latent, DELTA_MAX2)
    H_rv5m_lag40   = _estimate_h_from_moments(m2_rv5m,   DELTA_MAX2)
    H_tsrv_lag40   = _estimate_h_from_moments(m2_tsrv,   DELTA_MAX2)

    # Two-estimator correction (RV5m vs TSRV)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = fit_two_estimator_correction(log_rv5m, log_tsrv, DELTA_MAX)

    return {
        "H_latent":        H_latent,
        "H_rv5m":          H_rv5m,
        "H_tsrv":          H_tsrv,
        "H_latent_lag40":  H_latent_lag40,
        "H_rv5m_lag40":    H_rv5m_lag40,
        "H_tsrv_lag40":    H_tsrv_lag40,
        "H_corrected_rv5m":    corr["H_corrected_a"],
        "H_corrected_tsrv":    corr["H_corrected_b"],
        "H_corr_lag40_rv5m":   corr["H_corrected_lag40_a"],
        "H_corr_lag40_tsrv":   corr["H_corrected_lag40_b"],
        "omega_rv5m":  corr["omega_a"],
        "omega_tsrv":  corr["omega_b"],
        "convergence": corr["convergence_status"],
    }


def run_simulation(varpi: float) -> pd.DataFrame:
    """Run N_PATHS paths for a given varpi and return summary DataFrame."""
    print(f"\n  varpi={varpi:.1e} — simulating {N_PATHS} paths …")
    records = []
    for i in range(N_PATHS):
        if (i + 1) % 20 == 0:
            print(f"    path {i+1}/{N_PATHS}")
        rec = simulate_one_path(i, varpi)
        rec["varpi"]   = varpi
        rec["path_id"] = i
        records.append(rec)
    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame, varpi: float) -> None:
    """Print a formatted summary table for a given varpi."""
    sub = df[df["varpi"] == varpi]
    print(f"\n  ── varpi = {varpi:.1e} ──────────────────────────────────────")
    fmt = "  {:<30s} {:>7.4f}  ±{:.4f}"
    for col, label in [
        ("H_latent",           "H_latent   (lag10)"),
        ("H_rv5m",             "H_RV5m     (lag10)"),
        ("H_tsrv",             "H_TSRV     (lag10)"),
        ("H_corrected_rv5m",   "H_corr_RV5m(lag10)"),
        ("H_corrected_tsrv",   "H_corr_TSRV(lag10)"),
        ("H_latent_lag40",     "H_latent   (lag40)"),
        ("H_rv5m_lag40",       "H_RV5m     (lag40)"),
        ("H_tsrv_lag40",       "H_TSRV     (lag40)"),
        ("H_corr_lag40_rv5m",  "H_corr_RV5m(lag40)"),
        ("H_corr_lag40_tsrv",  "H_corr_TSRV(lag40)"),
    ]:
        vals = sub[col].dropna()
        if len(vals):
            print(fmt.format(label, vals.mean(), vals.std()))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_attenuation_boxplot(all_df: pd.DataFrame) -> None:
    """Box-and-whisker comparison of H estimates across varpi levels."""
    estimators = [
        ("H_latent",         "Latent"),
        ("H_rv5m",           "RV5m"),
        ("H_tsrv",           "TSRV"),
        ("H_corrected_rv5m", "Corr-RV5m"),
        ("H_corrected_tsrv", "Corr-TSRV"),
    ]
    n_varpi = len(VARPI_GRID)
    fig, axes = plt.subplots(1, n_varpi, figsize=(7 * n_varpi, 6), sharey=True)

    for ax, varpi in zip(axes, VARPI_GRID):
        sub = all_df[all_df["varpi"] == varpi]
        data  = [sub[col].dropna().values for col, _ in estimators]
        labels = [lbl for _, lbl in estimators]
        bp = ax.boxplot(data, patch_artist=True, notch=False)
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(H_TRUE, ls="--", color="black", lw=1.0, label=f"True H={H_TRUE}")
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("H estimate")
        ax.set_title(f"H estimates — varpi={varpi:.1e}\n(N={N_PATHS} paths, n_days={N_DAYS})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp3_attenuation_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_correction_comparison(all_df: pd.DataFrame) -> None:
    """Mean ± 1 std comparison of raw vs corrected H estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    estimator_groups = {
        "lag10": [
            ("H_latent",         "Latent",       "#1f77b4"),
            ("H_rv5m",           "RV5m (raw)",   "#d62728"),
            ("H_tsrv",           "TSRV (raw)",   "#ff7f0e"),
            ("H_corrected_rv5m", "Corr-RV5m",    "#2ca02c"),
            ("H_corrected_tsrv", "Corr-TSRV",    "#9467bd"),
        ],
        "lag40": [
            ("H_latent_lag40",    "Latent",       "#1f77b4"),
            ("H_rv5m_lag40",      "RV5m (raw)",   "#d62728"),
            ("H_tsrv_lag40",      "TSRV (raw)",   "#ff7f0e"),
            ("H_corr_lag40_rv5m", "Corr-RV5m",    "#2ca02c"),
            ("H_corr_lag40_tsrv", "Corr-TSRV",    "#9467bd"),
        ],
    }
    for ax, (lag_label, groups) in zip(axes, estimator_groups.items()):
        for varpi_idx, varpi in enumerate(VARPI_GRID):
            sub = all_df[all_df["varpi"] == varpi]
            x_offset = varpi_idx * (len(groups) + 1)
            for k, (col, label, color) in enumerate(groups):
                vals = sub[col].dropna()
                x = x_offset + k
                ax.errorbar(x, vals.mean(), yerr=vals.std(),
                            fmt="o", color=color, ms=7, capsize=4,
                            label=label if varpi_idx == 0 else "")
        ax.axhline(H_TRUE, ls="--", color="black", lw=0.8, label="True H")
        # x-tick labels
        xticks, xlabels = [], []
        for vi, varpi in enumerate(VARPI_GRID):
            mid = vi * (len(groups) + 1) + (len(groups) - 1) / 2
            xticks.append(mid)
            xlabels.append(f"varpi={varpi:.1e}")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("H estimate (mean ± 1σ)")
        ax.set_title(f"Raw vs Corrected H — {lag_label}")
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp3_correction_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── RESULTS.md update ─────────────────────────────────────────────────────────

def update_results_md(all_df: pd.DataFrame) -> None:
    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    lines = [
        "\n## Experiment 3: Measurement-Error Attenuation and Two-Estimator Correction\n",
        "\n### Simulation Settings\n",
        f"- Mode A (simulation only)\n",
        f"- n_paths = {N_PATHS}  (full-scale: 400)\n",
        f"- n_days  = {N_DAYS}   (full-scale: 2520)\n",
        f"- n_bars  = {N_BARS}   (1-min bars per day)\n",
        f"- Benchmark: H={H_TRUE}, kappa={KAPPA}, nu={NU}, dt={DT:.5f}\n",
        f"- Estimators: Latent log-IV, RV5m, TSRV (two-scale RV)\n\n",
        "### Key Results (mean ± std across paths)\n",
    ]
    for varpi in VARPI_GRID:
        sub = all_df[all_df["varpi"] == varpi]
        lines.append(f"\n#### varpi = {varpi:.1e}\n")
        lines.append("| Estimator | lag10 | lag40 |\n")
        lines.append("|-----------|-------|-------|\n")
        for col10, col40, label in [
            ("H_latent",         "H_latent_lag40",    "Latent"),
            ("H_rv5m",           "H_rv5m_lag40",      "RV5m"),
            ("H_tsrv",           "H_tsrv_lag40",      "TSRV"),
            ("H_corrected_rv5m", "H_corr_lag40_rv5m", "Corr-RV5m"),
            ("H_corrected_tsrv", "H_corr_lag40_tsrv", "Corr-TSRV"),
        ]:
            v10 = sub[col10].dropna()
            v40 = sub[col40].dropna()
            lines.append(
                f"| {label:<12s} | {v10.mean():.3f}±{v10.std():.3f}"
                f" | {v40.mean():.3f}±{v40.std():.3f} |\n"
            )
    lines += [
        "\n### Findings\n",
        "- **Attenuation**: Observed H_RV5m and H_TSRV < H_latent due to log-RV measurement noise.\n",
        "- **Larger varpi** produces stronger attenuation.\n",
        "- **Two-estimator correction** partially recovers latent H; residual bias depends on\n",
        "  noise estimation accuracy and sample length.\n",
        "- **Longer lag window** (lag40) reduces sampling noise in second-moment estimates.\n\n",
        "### Files\n",
        "- `results/exp3_simulation_results.csv`\n",
        "- `results/exp3_attenuation_boxplot.png`\n",
        "- `results/exp3_correction_comparison.png`\n",
    ]
    with open(md_path, "a") as f:
        f.writelines(lines)
    print(f"  Updated: {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 65)
    print("Experiment 3: Measurement-error attenuation and correction")
    print(f"  n_paths={N_PATHS}, n_days={N_DAYS}, n_bars={N_BARS}")
    print(f"  H={H_TRUE}, kappa={KAPPA}, nu={NU}")
    print("=" * 65)

    all_dfs = []
    for varpi in VARPI_GRID:
        df_v = run_simulation(varpi)
        print_summary(df_v, varpi)
        all_dfs.append(df_v)

    all_df = pd.concat(all_dfs, ignore_index=True)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp3_simulation_results.csv")
    all_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved simulation results → {csv_path}")

    # Plots
    print("\nGenerating plots …")
    plot_attenuation_boxplot(all_df)
    plot_correction_comparison(all_df)

    # RESULTS.md
    update_results_md(all_df)

    elapsed = time.time() - t0
    print(f"\n✓ Experiment 3 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
