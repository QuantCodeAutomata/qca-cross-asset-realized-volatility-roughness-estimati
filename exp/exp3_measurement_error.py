#!/usr/bin/env python3
"""Experiment 3: measurement-error attenuation and two-estimator correction.

This is a simulation validation of Sections 5.2-5.3 of the paper, not an
empirical replication.  It uses the paper's daily time convention, 2,520 days,
390 returns per day, and the empirical correction pair RV5m (leg a) / RK (leg
b).  The default Monte Carlo count is deliberately modest for a runnable audit;
pass ``--n-paths 400`` for the paper-scale correction grid.

The fOU path generator is a filtered-fGn discrete approximation.  Consequently,
this script validates the estimator/correction pipeline and its qualitative
benchmarks; it is not a bit-for-bit reproduction of the paper's stationary fOU
Monte Carlo engine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.autocovariance_fit import fit_two_estimator_correction
from src.hurst_realized import estimate_hurst_ols
from src.rv_estimators import compute_realized_kernel, compute_rv5m, compute_tsrv
from src.simulate_rough_paths import simulate_fou_log_vol, simulate_noisy_intraday_prices

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

H_TRUE = 0.20
KAPPA_PER_DAY = 0.010
TARGET_LOGVAR_STD = 1.05
TARGET_ANNUALIZED_VOL = 0.128
ANNUALIZATION_DAYS = 252.0
DELTA_MAX = 10
LONG_DELTA_MAX = 40
PAPER_REFERENCE = {
    "H_latent": 0.198,
    "H_rv5m": 0.181,
    "H_rk": 0.180,
    "H_corr_rk_10": 0.278,
    "H_corr_rk_40": 0.238,
}


def _positive_log(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-transform positive observations without collapsing the date grid."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    invalid_fraction = 1.0 - float(np.mean(valid))
    logged = np.full(values.shape, np.nan, dtype=np.float64)
    logged[valid] = np.log(values[valid])
    return logged, invalid_fraction


def _hurst(values: np.ndarray, delta_max: int) -> float:
    result = estimate_hurst_ols(values, delta_max=delta_max)
    return float(result["H_hat"])


def _daily_realized_measures(observed_prices: np.ndarray) -> dict[str, np.ndarray]:
    returns = np.diff(observed_prices, axis=1)
    n_days = returns.shape[0]
    rv5m = np.empty(n_days, dtype=np.float64)
    rk = np.empty(n_days, dtype=np.float64)
    tsrv = np.empty(n_days, dtype=np.float64)
    for day, intraday_returns in enumerate(returns):
        rv5m[day] = compute_rv5m(intraday_returns)
        rk[day] = compute_realized_kernel(intraday_returns)
        # Kept as a diagnostic because the paper reports it, but the correction
        # pair below is RV5m/RK.
        tsrv[day] = compute_tsrv(intraday_returns)
    return {"rv5m": rv5m, "rk": rk, "tsrv": tsrv}


def simulate_one_path(
    path_id: int,
    *,
    n_days: int,
    n_bars: int,
    varpi: float,
    seed: int,
) -> dict:
    log_iv = simulate_fou_log_vol(
        n_days=n_days,
        H=H_TRUE,
        kappa=KAPPA_PER_DAY,
        nu=0.5,
        dt=1.0,
        seed=seed + path_id,
    )
    log_iv = log_iv - float(np.mean(log_iv))
    std = float(np.std(log_iv, ddof=1))
    if std <= 0.0:
        raise RuntimeError("degenerate latent log-variance path")
    log_iv *= TARGET_LOGVAR_STD / std
    # The price-noise scale is meaningful only relative to the efficient return
    # scale.  Section 5.1 calibrates the mean annualized volatility to about
    # 12.8%; leaving log variance centered at zero would imply roughly 100%
    # daily volatility and make varpi almost irrelevant.
    current_ann_vol = float(np.mean(np.exp(log_iv / 2.0)) * np.sqrt(ANNUALIZATION_DAYS))
    log_iv += 2.0 * np.log(TARGET_ANNUALIZED_VOL / current_ann_vol)

    observed_prices = simulate_noisy_intraday_prices(
        log_iv,
        n_bars=n_bars,
        varpi=varpi,
        seed=seed + 1_000_000 + path_id,
    )
    measures = _daily_realized_measures(observed_prices)

    log_rv5m, invalid_rv5m = _positive_log(measures["rv5m"])
    log_rk, invalid_rk = _positive_log(measures["rk"])
    log_tsrv, invalid_tsrv = _positive_log(measures["tsrv"])
    if not (len(log_rv5m) == len(log_rk) == n_days):
        raise RuntimeError("realized-measure arrays lost their calendar alignment")
    if np.any(~np.isfinite(log_rv5m)) or np.any(~np.isfinite(log_rk)):
        raise RuntimeError("RV5m/RK contains invalid days; cannot fit correction pair")

    correction = fit_two_estimator_correction(
        log_rv5m,
        log_rk,
        delta_max=DELTA_MAX,
        long_delta_max=LONG_DELTA_MAX,
    )

    error_rv5m = log_rv5m - log_iv
    error_rk = log_rk - log_iv
    tsrv_valid = np.isfinite(log_tsrv)
    error_tsrv = log_tsrv[tsrv_valid] - log_iv[tsrv_valid]

    return {
        "path_id": path_id,
        "varpi": varpi,
        "H_latent_10": _hurst(log_iv, DELTA_MAX),
        "H_latent_40": _hurst(log_iv, LONG_DELTA_MAX),
        "H_rv5m_10": _hurst(log_rv5m, DELTA_MAX),
        "H_rv5m_40": _hurst(log_rv5m, LONG_DELTA_MAX),
        "H_rk_10": _hurst(log_rk, DELTA_MAX),
        "H_rk_40": _hurst(log_rk, LONG_DELTA_MAX),
        "H_tsrv_10": _hurst(log_tsrv, DELTA_MAX),
        "H_tsrv_40": _hurst(log_tsrv, LONG_DELTA_MAX),
        "H_corr_rv5m_10": correction["H_corrected_a"],
        "H_corr_rv5m_40": correction["H_corrected_lag40_a"],
        "H_corr_rk_10": correction["H_corrected_b"],
        "H_corr_rk_40": correction["H_corrected_lag40_b"],
        "omega2_fit_rv5m": correction["omega_a2"],
        "omega2_fit_rk": correction["omega_b2"],
        "omega2_true_rv5m": float(np.var(error_rv5m, ddof=1)),
        "omega2_true_rk": float(np.var(error_rk, ddof=1)),
        "omega2_true_tsrv": float(np.var(error_tsrv, ddof=1)),
        "delta_m": correction["delta_m"],
        "correction_valid_10": correction["correction_valid_10"],
        "correction_valid_40": correction["correction_valid_40"],
        "correction_status": correction["convergence_status"],
        "invalid_fraction_rv5m": invalid_rv5m,
        "invalid_fraction_rk": invalid_rk,
        "invalid_fraction_tsrv": invalid_tsrv,
        "mean_annualized_vol": float(
            np.mean(np.exp(log_iv / 2.0)) * np.sqrt(ANNUALIZATION_DAYS)
        ),
    }


def run_simulation(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    for varpi in args.varpi:
        print(f"varpi={varpi:.1e}: {args.n_paths} paths")
        for path_id in range(args.n_paths):
            rows.append(
                simulate_one_path(
                    path_id,
                    n_days=args.n_days,
                    n_bars=args.n_bars,
                    varpi=varpi,
                    # Pair the latent path, efficient returns, and noise draws
                    # across varpi levels.  Only the noise amplitude changes.
                    seed=args.seed,
                )
            )
            if (path_id + 1) % max(1, min(10, args.n_paths)) == 0:
                print(f"  completed {path_id + 1}/{args.n_paths}")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "H_latent_10",
        "H_rv5m_10",
        "H_rk_10",
        "H_tsrv_10",
        "H_corr_rv5m_10",
        "H_corr_rk_10",
        "H_latent_40",
        "H_rv5m_40",
        "H_rk_40",
        "H_corr_rv5m_40",
        "H_corr_rk_40",
        "omega2_true_rv5m",
        "omega2_true_rk",
        "omega2_fit_rv5m",
        "omega2_fit_rk",
    ]
    rows = []
    for varpi, group in results.groupby("varpi"):
        for metric in metrics:
            values = group[metric].dropna()
            rows.append(
                {
                    "varpi": varpi,
                    "metric": metric,
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(ddof=1),
                    "n_valid": len(values),
                    "n_paths": len(group),
                }
            )
        rows.append(
            {
                "varpi": varpi,
                "metric": "correction_valid_10_rate",
                "mean": group["correction_valid_10"].mean(),
                "median": np.nan,
                "std": np.nan,
                "n_valid": int(group["correction_valid_10"].sum()),
                "n_paths": len(group),
            }
        )
    return pd.DataFrame(rows)


def _plot_estimates(results: pd.DataFrame) -> None:
    metrics = [
        ("H_latent_10", "latent"),
        ("H_rv5m_10", "RV5m raw"),
        ("H_rk_10", "RK raw"),
        ("H_corr_rv5m_10", "RV5m corrected"),
        ("H_corr_rk_10", "RK corrected"),
    ]
    for varpi, group in results.groupby("varpi"):
        data = [group[col].dropna().to_numpy() for col, _ in metrics]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.boxplot(data, tick_labels=[label for _, label in metrics])
        ax.axhline(H_TRUE, linestyle="--", linewidth=1.0, label="true H")
        ax.set_ylabel("H estimate")
        ax.set_title(f"Measurement-error validation, varpi={varpi:.1e}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"exp3_h_estimates_varpi_{varpi:.0e}.png", dpi=150)
        plt.close(fig)


def write_summary_markdown(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Experiment 3 - measurement-error validation\n\n",
        "This is a synthetic estimator validation, not an empirical replication.\n\n",
        "## Configuration\n\n",
        f"- paths: {args.n_paths}\n",
        f"- days per path: {args.n_days}\n",
        f"- one-minute returns per day: {args.n_bars}\n",
        f"- H: {H_TRUE}; kappa: {KAPPA_PER_DAY} per trading day\n",
        "- correction pair: RV5m / RK\n",
        "- latent log-variance rescaled to std 1.05\n",
        "- mean annualized volatility calibrated to 12.8%\n",
        "- paths and Gaussian draws are paired across varpi levels\n\n",
        "## Results (means)\n\n",
        "| varpi | latent H10 | RV5m H10 | RK H10 | corr RV5m H10 | corr RK H10 | corr RK H40 | valid correction |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for varpi in sorted(summary["varpi"].unique()):
        sub = summary[summary["varpi"] == varpi].set_index("metric")
        value = lambda name: float(sub.loc[name, "mean"])
        lines.append(
            f"| {varpi:.1e} | {value('H_latent_10'):.3f} | {value('H_rv5m_10'):.3f} "
            f"| {value('H_rk_10'):.3f} | {value('H_corr_rv5m_10'):.3f} "
            f"| {value('H_corr_rk_10'):.3f} | {value('H_corr_rk_40'):.3f} "
            f"| {value('correction_valid_10_rate'):.1%} |\n"
        )
    lines.extend(
        [
            "\n## Distribution diagnostic\n\n",
        ]
    )
    baseline = summary[summary["varpi"] == min(summary["varpi"])].set_index("metric")
    median = lambda name: float(baseline.loc[name, "median"])
    lines.extend(
        [
            f"At the baseline noise level, median corrected RK estimates are "
            f"{median('H_corr_rk_10'):.3f} at lag 10 and "
            f"{median('H_corr_rk_40'):.3f} at lag 40. The corresponding means are "
            "higher because the correction distribution is right-skewed. This is "
            "why the corrected output should be read as a noisy bracket, not as a "
            "stable point estimate.\n\n",
            "\n## Paper benchmark for varpi=1e-4\n\n",
            "Paper Table 4: latent about 0.198, RV5m 0.181, RK 0.180. "
            "Paper Table 5: the RK leg corrected to 0.278 at lag 10 and 0.238 at lag 40. "
            "The lag-10 correction is explicitly interpreted as an upper bracket, not a point estimate.\n\n",
            "At varpi=5e-4 the raw attenuation remains directionally consistent, "
            "but the iid additive log-noise premise breaks down and the correction "
            "becomes unstable. The stressed corrected values must not be interpreted "
            "as recovery of the latent H.\n\n",
            "The local simulation uses a filtered-fGn approximation and a simplified one-minute RK, "
            "so differences from the paper's Monte Carlo are expected and must not be presented as empirical evidence.\n",
        ]
    )
    (RESULTS_DIR / "exp3_summary.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=20)
    parser.add_argument("--n-days", type=int, default=2520)
    parser.add_argument("--n-bars", type=int, default=390)
    parser.add_argument("--varpi", type=float, nargs="+", default=[1e-4, 5e-4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_paths < 1 or args.n_days <= LONG_DELTA_MAX + 2 or args.n_bars < 20:
        raise ValueError("invalid simulation dimensions")

    start = time.time()
    results = run_simulation(args)
    summary = summarize(results)
    results.to_csv(RESULTS_DIR / "exp3_simulation_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "exp3_summary.csv", index=False)
    metadata = {
        "experiment": "synthetic measurement-error validation",
        "n_paths": args.n_paths,
        "n_days": args.n_days,
        "n_bars": args.n_bars,
        "varpi": args.varpi,
        "H": H_TRUE,
        "kappa_per_day": KAPPA_PER_DAY,
        "target_logvar_std": TARGET_LOGVAR_STD,
        "target_annualized_vol": TARGET_ANNUALIZED_VOL,
        "correction_pair": ["rv5m", "rk"],
        "paper_reference": PAPER_REFERENCE,
        "elapsed_seconds": time.time() - start,
    }
    (RESULTS_DIR / "exp3_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not args.no_plots:
        _plot_estimates(results)
    write_summary_markdown(summary, args)
    print(summary.to_string(index=False))
    print(f"completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
