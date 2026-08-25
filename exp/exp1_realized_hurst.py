#!/usr/bin/env python3
"""Experiment 1: synthetic realized-H estimator recovery validation.

The paper's first empirical experiment uses 3,926 equities and 34 CME futures
roots.  This repository has no such market panel.  The script therefore has a
narrower and honest purpose: generate paper-like H targets by asset class and
check whether the seven daily realized-variance implementations recover rough
scaling from simulated intraday returns.

It must not be described as a cross-asset empirical replication.
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

from data.data_loader import generate_synthetic_intraday
from src.hurst_realized import compute_second_moment_scaling, estimate_hurst_ols
from src.rv_estimators import compute_all_estimators

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ESTIMATORS = ["rv5m", "rk", "tsrv", "pav", "pabpv", "bpv", "ctrv"]
DELTA_MAX = 10

# Table 7 and Section 6 references from the paper.
CLASS_TARGETS = {
    "livestock": 0.048,
    "rates": 0.074,
    "fx": 0.081,
    "agriculture": 0.085,
    "energy": 0.088,
    "metals": 0.091,
    "single_stock": 0.131,
    "equity_index": 0.195,
}


def build_asset_universe(n_per_class: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    assets: list[dict] = []
    for asset_class, target in CLASS_TARGETS.items():
        for index in range(n_per_class):
            h = float(np.clip(target + rng.normal(0.0, 0.008), 0.03, 0.30))
            assets.append(
                {
                    "asset": f"{asset_class}_{index + 1:02d}",
                    "asset_class": asset_class,
                    "H_target": h,
                    "class_reference": target,
                    "kappa": 0.010,
                }
            )
    return assets


def daily_measures(asset: dict, *, n_days: int, n_bars: int, seed: int) -> pd.DataFrame:
    intraday = generate_synthetic_intraday(
        n_days=n_days,
        n_bars=n_bars,
        H=asset["H_target"],
        kappa=asset["kappa"],
        nu=0.5,
        seed=seed,
    )
    rows = []
    for date, group in intraday.groupby("date", sort=True):
        values = compute_all_estimators(group["return_"].to_numpy(dtype=np.float64))
        values["date"] = date
        rows.append(values)
    return pd.DataFrame(rows).set_index("date")


def estimate_asset(asset: dict, measures: pd.DataFrame) -> list[dict]:
    rows = []
    for estimator in ESTIMATORS:
        values = measures[estimator].to_numpy(dtype=np.float64)
        valid = np.isfinite(values) & (values > 0.0)
        log_values = np.full(values.shape, np.nan, dtype=np.float64)
        log_values[valid] = np.log(values[valid])
        result = estimate_hurst_ols(log_values, delta_max=DELTA_MAX)
        rows.append(
            {
                **asset,
                "estimator": estimator,
                "H_hat": result["H_hat"],
                "bias": result["H_hat"] - asset["H_target"],
                "r_squared": result["r_squared"],
                "n_obs": result["n_obs"],
                "invalid_day_fraction": 1.0 - float(np.mean(valid)),
            }
        )
    return rows


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    assets = build_asset_universe(args.n_per_class, args.seed)
    all_rows: list[dict] = []
    cached: dict[str, pd.DataFrame] = {}
    for index, asset in enumerate(assets):
        print(f"[{index + 1}/{len(assets)}] {asset['asset']} H={asset['H_target']:.3f}")
        measures = daily_measures(
            asset,
            n_days=args.n_days,
            n_bars=args.n_bars,
            seed=args.seed + 10_000 + index,
        )
        cached[asset["asset"]] = measures
        all_rows.extend(estimate_asset(asset, measures))
    return pd.DataFrame(all_rows), cached


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimator_summary = (
        results.groupby("estimator")
        .agg(
            median_H=("H_hat", "median"),
            mean_H=("H_hat", "mean"),
            median_bias=("bias", "median"),
            mean_abs_error=("bias", lambda x: float(np.mean(np.abs(x)))),
            median_r_squared=("r_squared", "median"),
            n=("H_hat", "count"),
        )
        .reset_index()
    )
    class_summary = (
        results.groupby(["asset_class", "estimator"])
        .agg(
            H_reference=("class_reference", "first"),
            median_target=("H_target", "median"),
            median_H=("H_hat", "median"),
            median_bias=("bias", "median"),
            median_r_squared=("r_squared", "median"),
            n=("H_hat", "count"),
        )
        .reset_index()
    )
    return estimator_summary, class_summary


def make_plots(results: pd.DataFrame, cached: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for estimator, group in results.groupby("estimator"):
        ax.scatter(group["H_target"], group["H_hat"], s=24, alpha=0.65, label=estimator)
    line = np.linspace(0.03, 0.24, 100)
    ax.plot(line, line, linestyle="--", linewidth=1.0, label="target")
    ax.set_xlabel("synthetic H target")
    ax.set_ylabel("estimated H")
    ax.set_title("Synthetic realized-H estimator recovery")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_estimator_recovery.png", dpi=150)
    plt.close(fig)

    representative = results[results["estimator"] == "tsrv"].copy()
    representative["distance"] = np.abs(representative["H_target"] - 0.131)
    row = representative.sort_values("distance").iloc[0]
    values = cached[row["asset"]]["tsrv"].to_numpy(dtype=np.float64)
    valid_values = np.isfinite(values) & (values > 0.0)
    log_values = np.full(values.shape, np.nan, dtype=np.float64)
    log_values[valid_values] = np.log(values[valid_values])
    deltas, moments = compute_second_moment_scaling(log_values, DELTA_MAX)
    fit = estimate_hurst_ols(log_values, DELTA_MAX)
    valid = np.isfinite(moments) & (moments > 0.0)
    coefficients = np.polyfit(np.log(deltas[valid]), np.log(moments[valid]), 1)
    grid = np.linspace(np.log(deltas[valid]).min(), np.log(deltas[valid]).max(), 100)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(np.log(deltas[valid]), np.log(moments[valid]), label="moments")
    ax.plot(grid, np.polyval(coefficients, grid), label=f"H={fit['H_hat']:.3f}")
    ax.set_xlabel("log lag")
    ax.set_ylabel("log second moment")
    ax.set_title(f"Representative synthetic fit: {row['asset']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_representative_loglog.png", dpi=150)
    plt.close(fig)


def write_summary(
    estimator_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Experiment 1 - synthetic realized-H estimator validation\n\n",
        "This is not the paper's empirical cross-asset replication. It uses synthetic assets centered on the paper's class medians.\n\n",
        f"Configuration: {args.n_per_class} synthetic assets per class, {args.n_days} days, {args.n_bars} returns/day.\n\n",
        "## Estimator summary\n\n",
        "| estimator | median H | median bias | MAE | median R2 | n |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for _, row in estimator_summary.iterrows():
        lines.append(
            f"| {row['estimator']} | {row['median_H']:.3f} | {row['median_bias']:+.3f} "
            f"| {row['mean_abs_error']:.3f} | {row['median_r_squared']:.3f} | {int(row['n'])} |\n"
        )
    lines.extend(
        [
            "\n## TSRV class recovery\n\n",
            "| class | paper reference | synthetic target | estimated median | median R2 |\n",
            "|---|---:|---:|---:|---:|\n",
        ]
    )
    for _, row in class_summary[class_summary["estimator"] == "tsrv"].iterrows():
        lines.append(
            f"| {row['asset_class']} | {row['H_reference']:.3f} | {row['median_target']:.3f} "
            f"| {row['median_H']:.3f} | {row['median_r_squared']:.3f} |\n"
        )
    lines.append(
        "\nThe output can validate implementation scale and rough-regime recovery, but it cannot validate the paper's cross-sectional ordering, quality filters, or empirical R2 distribution.\n"
    )
    (RESULTS_DIR / "exp1_summary.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=2)
    parser.add_argument("--n-days", type=int, default=1000)
    parser.add_argument("--n-bars", type=int, default=390)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_per_class < 1 or args.n_days <= DELTA_MAX + 2 or args.n_bars < 20:
        raise ValueError("invalid simulation dimensions")
    start = time.time()
    results, cached = run(args)
    estimator_summary, class_summary = summarize(results)
    results.to_csv(RESULTS_DIR / "exp1_results.csv", index=False)
    estimator_summary.to_csv(RESULTS_DIR / "exp1_summary.csv", index=False)
    class_summary.to_csv(RESULTS_DIR / "exp1_class_summary.csv", index=False)
    metadata = {
        "experiment": "synthetic realized-H estimator validation",
        "n_per_class": args.n_per_class,
        "n_days": args.n_days,
        "n_bars": args.n_bars,
        "class_targets": CLASS_TARGETS,
        "seed": args.seed,
    }
    (RESULTS_DIR / "exp1_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not args.no_plots:
        make_plots(results, cached)
    write_summary(estimator_summary, class_summary, args)
    print(estimator_summary.to_string(index=False))
    print(f"completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
