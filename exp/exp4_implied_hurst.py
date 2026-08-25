#!/usr/bin/env python3
"""Experiment 4: end-to-end synthetic option-skew identification validation.

The paper's empirical experiment starts from option chains, estimates an ATM
smile slope for each date/expiry, and only then estimates H from the maturity
term structure.  The submitted script skipped the first stage and generated the
exact power law it later regressed, making the successful cases tautological.

This replacement generates strike-level synthetic implied-volatility smiles,
runs :func:`estimate_atm_skew`, and then runs the daily/panel H regressions.  It
is still a synthetic *method validation*, not a replication of the paper's
CME/OPRA evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.atm_skew_estimator import estimate_atm_skew
from src.implied_hurst import estimate_pooled_implied_hurst

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MATURITY_DAYS = np.array([14, 21, 30, 45, 60, 90, 120, 180, 252, 365])
MATURITY_WINDOW = (14 / 365, 1.0)
LOG_MONEYNESS = np.linspace(-0.075, 0.075, 21)

# Paper-like underlier examples, not asset-class aggregate outputs.
UNDERLIERS = {
    "ES_like": {
        "kind": "power_law",
        "H_target": 0.25,
        "H_realized_reference": 0.201,
        "c": 0.075,
        "atm_iv": 0.20,
        "skew_log_noise": 0.18,
        "iv_noise": 0.0008,
        "expected_identified": True,
        "seed_offset": 11,
    },
    "ZW_like": {
        "kind": "power_law",
        "H_target": 0.08,
        "H_realized_reference": 0.088,
        "c": 0.035,
        "atm_iv": 0.24,
        "skew_log_noise": 0.38,
        "iv_noise": 0.0012,
        "expected_identified": True,
        "seed_offset": 23,
    },
    "ZN_like": {
        "kind": "weak",
        "H_target": np.nan,
        "H_realized_reference": 0.083,
        "c": 0.012,
        "atm_iv": 0.10,
        "skew_log_noise": 0.0,
        "iv_noise": 0.0010,
        "expected_identified": False,
        "seed_offset": 37,
    },
    "6E_like": {
        "kind": "weak",
        "H_target": np.nan,
        "H_realized_reference": 0.072,
        "c": 0.015,
        "atm_iv": 0.12,
        "skew_log_noise": 0.0,
        "iv_noise": 0.0011,
        "expected_identified": False,
        "seed_offset": 41,
    },
    "NG_like": {
        "kind": "seasonal",
        "H_target": np.nan,
        "H_realized_reference": 0.065,
        "c": 0.055,
        "atm_iv": 0.42,
        "skew_log_noise": 0.0,
        "iv_noise": 0.0014,
        "expected_identified": False,
        "seed_offset": 53,
    },
}


def _true_skew(
    config: dict,
    T: float,
    date_index: int,
    rng: np.random.Generator,
) -> float:
    kind = config["kind"]
    if kind == "power_law":
        date_level = rng.normal(0.0, 0.12)
        maturity_noise = rng.normal(0.0, config["skew_log_noise"])
        magnitude = (
            config["c"]
            * T ** (config["H_target"] - 0.5)
            * np.exp(date_level + maturity_noise)
        )
        return -float(magnitude)

    if kind == "weak":
        # Symmetric smiles: the tiny fitted slope changes sign and has no
        # systematic relation with maturity.
        return float(rng.normal(0.0, config["c"]))

    if kind == "seasonal":
        # A composition/seasonal maturity pattern dominates any monotone power
        # law.  The phase changes with the observation date.
        phase = 2.0 * np.pi * ((date_index % 252) / 252.0)
        log_magnitude = (
            np.log(config["c"])
            + 1.10 * np.sin(phase + 8.0 * np.pi * T)
            + rng.normal(0.0, 0.35)
        )
        sign = -1.0 if rng.random() > 0.15 else 1.0
        return sign * float(np.exp(log_magnitude))

    raise ValueError(f"unknown synthetic underlier kind: {kind}")


def generate_skew_observations(
    underlier: str,
    config: dict,
    *,
    n_dates: int,
    seed: int,
) -> pd.DataFrame:
    """Generate strike-level smiles and estimate one ATM skew per date/maturity."""
    rng = np.random.default_rng(seed + config["seed_offset"])
    dates = pd.bdate_range("2018-01-02", periods=n_dates)
    rows: list[dict] = []

    for date_index, date in enumerate(dates):
        # Date effects in the ATM level are deliberately independent of the skew
        # slope and are absorbed by the panel regression.
        date_atm_shift = rng.normal(0.0, 0.012)
        forward = 100.0 * np.exp(rng.normal(0.0, 0.01))
        strikes = forward * np.exp(LOG_MONEYNESS)

        for maturity_days in MATURITY_DAYS:
            T = float(maturity_days / 365.0)
            true_psi = _true_skew(config, T, date_index, rng)
            atm_iv = max(
                0.04,
                config["atm_iv"]
                + date_atm_shift
                + 0.015 * np.sqrt(T)
                + rng.normal(0.0, 0.004),
            )
            curvature = 0.35 + rng.normal(0.0, 0.05)
            iv = (
                atm_iv
                + true_psi * LOG_MONEYNESS
                + curvature * LOG_MONEYNESS**2
                + rng.normal(0.0, config["iv_noise"], LOG_MONEYNESS.size)
            )
            result = estimate_atm_skew(
                strikes,
                iv,
                forward,
                moneyness_band=0.08,
                min_strikes=10,
            )
            if result is None:
                rows.append(
                    {
                        "underlier": underlier,
                        "date": date,
                        "T": T,
                        "maturity_days": int(maturity_days),
                        "true_psi": true_psi,
                        "psi": np.nan,
                        "smile_r_squared": np.nan,
                        "valid": False,
                    }
                )
            else:
                rows.append(
                    {
                        "underlier": underlier,
                        "date": date,
                        "T": T,
                        "maturity_days": int(maturity_days),
                        "true_psi": true_psi,
                        "psi": result["psi"],
                        "smile_r_squared": result["r_squared"],
                        "n_strikes": result["n_strikes"],
                        "valid": result["valid"],
                    }
                )
    return pd.DataFrame(rows)


def run_experiment(n_dates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_observations = []
    summaries = []
    for underlier, config in UNDERLIERS.items():
        observations = generate_skew_observations(
            underlier,
            config,
            n_dates=n_dates,
            seed=seed,
        )
        all_observations.append(observations)
        result = estimate_pooled_implied_hurst(
            observations[["date", "T", "psi", "valid"]],
            maturity_window=MATURITY_WINDOW,
            min_obs_per_date=3,
        )
        target = config["H_target"]
        h_close = (
            np.nan
            if np.isnan(target)
            else bool(abs(result["H_hat_IV"] - target) <= 0.08)
        )
        summaries.append(
            {
                "underlier": underlier,
                "synthetic_kind": config["kind"],
                "H_hat_IV": result["H_hat_IV"],
                "H_target": target,
                "H_realized_reference": config["H_realized_reference"],
                "beta": result["beta"],
                "beta_se_cluster_date": result["beta_se"],
                "R_psi_squared": result["r_psi_squared"],
                "identified": result["identified"],
                "expected_identified": config["expected_identified"],
                "identification_match": bool(
                    result["identified"] == config["expected_identified"]
                ),
                "target_recovered_within_0.08": h_close,
                "median_smile_r_squared": observations["smile_r_squared"].median(),
                "n_obs": result["n_obs"],
                "n_dates": result["n_dates"],
            }
        )
    return pd.concat(all_observations, ignore_index=True), pd.DataFrame(summaries)


def plot_term_structures(observations: pd.DataFrame) -> None:
    for underlier in ("ES_like", "ZN_like", "NG_like"):
        subset = observations[observations["underlier"] == underlier]
        grouped = subset.groupby("T")["psi"].apply(
            lambda x: float(np.median(np.abs(x[np.isfinite(x)])))
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(grouped.index, grouped.values)
        ax.plot(grouped.index, grouped.values)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("maturity T (years)")
        ax.set_ylabel("median |ATM skew|")
        ax.set_title(f"Synthetic strike-to-skew validation: {underlier}")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"exp4_term_structure_{underlier}.png", dpi=150)
        plt.close(fig)


def plot_implied_vs_realized(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for _, row in summary.iterrows():
        marker = "o" if row["identified"] else "x"
        ax.scatter(row["H_realized_reference"], row["H_hat_IV"], marker=marker, s=90)
        ax.annotate(
            row["underlier"],
            (row["H_realized_reference"], row["H_hat_IV"]),
            xytext=(5, 4),
            textcoords="offset points",
        )
    line = np.linspace(0.0, 0.35, 100)
    ax.plot(line, line, linestyle="--", linewidth=1.0, label="parity")
    ax.set_xlabel("paper realized-H reference")
    ax.set_ylabel("synthetic implied-H estimate")
    ax.set_title("Only identified implied-H points are interpretable")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_implied_vs_realized.png", dpi=150)
    plt.close(fig)


def write_summary(summary: pd.DataFrame, n_dates: int) -> None:
    lines = [
        "# Experiment 4 - synthetic option-skew identification validation\n\n",
        "This experiment starts from strike-level synthetic smiles and therefore tests both the ATM-skew estimator and the maturity regression. It is not a CME/OPRA replication.\n\n",
        f"Dates per synthetic underlier: {n_dates}. Maturity convention: calendar days / 365.\n\n",
        "| underlier | H_hat_IV | target | R2_psi | identified | expected | median smile R2 | validation |\n",
        "|---|---:|---:|---:|:---:|:---:|---:|:---:|\n",
    ]
    for _, row in summary.iterrows():
        target = "-" if np.isnan(row["H_target"]) else f"{row['H_target']:.3f}"
        target_ok = (
            True
            if np.isnan(row["H_target"])
            else bool(row["target_recovered_within_0.08"])
        )
        passed = bool(row["identification_match"]) and target_ok
        lines.append(
            f"| {row['underlier']} | {row['H_hat_IV']:.3f} | {target} | "
            f"{row['R_psi_squared']:.3f} | {bool(row['identified'])} | "
            f"{bool(row['expected_identified'])} | {row['median_smile_r_squared']:.3f} "
            f"| {'PASS' if passed else 'FAIL'} |\n"
        )
    lines.extend(
        [
            "\nThe paper-level interpretation is deliberately stricter than the original repository: rates and FX estimates are not meaningful when R2 is near zero, and a stylized energy power law is not evidence for the paper's seasonal commodity result.\n",
        ]
    )
    (RESULTS_DIR / "exp4_summary.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dates", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_dates < 20:
        raise ValueError("n_dates must be at least 20")
    observations, summary = run_experiment(args.n_dates, args.seed)
    observations.to_csv(RESULTS_DIR / "exp4_skew_observations.csv", index=False)
    summary.to_csv(RESULTS_DIR / "exp4_implied_hurst_results.csv", index=False)
    metadata = {
        "experiment": "synthetic strike-level option-skew validation",
        "n_dates": args.n_dates,
        "maturity_day_count": 365,
        "underliers": list(UNDERLIERS),
        "seed": args.seed,
    }
    (RESULTS_DIR / "exp4_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not args.no_plots:
        plot_term_structures(observations)
        plot_implied_vs_realized(summary)
    write_summary(summary, args.n_dates)
    print(summary.to_string(index=False))
    if not summary["identification_match"].all():
        raise SystemExit("synthetic identification validation failed")


if __name__ == "__main__":
    main()
