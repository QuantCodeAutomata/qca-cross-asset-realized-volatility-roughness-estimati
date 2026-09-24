"""Resumable, local-only runners for paper Tables 3--6.

Dry-run is available for the full design. The paper profile is deliberately
opt-in with --execute-paper; smoke is small and is NOT a statistical replication.
Each completed (including failed) path has an atomic, checksummed receipt.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..autocovariance_fit import fit_two_estimator_correction
from ..hurst_realized import estimate_hurst_ols, compute_second_moment_scaling
from ..rv_estimators import (compute_all_estimators, compute_rv5m, compute_realized_kernel)
from ..simulate_rough_paths import simulate_noisy_intraday_prices
from .common import atomic_json, digest, private_output, sha256_file
from .gaussian import calibrate, generate_log_variance

H_VALUES = (.05, .10, .15, .20, .30, .50)
KAPPA_VALUES = (0., .003, .010, .020, .035)
ESTIMATORS = ("rv5m", "rk", "tsrv", "pav", "pabpv", "bpv", "ctrv")
PAIR_ESTIMATORS = tuple(e for e in ESTIMATORS if e != "rv5m")


@dataclass(frozen=True)
class RunConfig:
    profile: str = "smoke"
    seed: int = 20260924
    generator: str = "gaussian"
    overlapping: bool = False
    specification: str = "paper"
    calibration_horizon: int = 2520

    def __post_init__(self):
        if self.profile not in ("smoke", "paper") or self.generator not in ("gaussian", "legacy"):
            raise ValueError("invalid run profile or generator")
        if self.specification not in ("paper", "guarded") or self.calibration_horizon < 2:
            raise ValueError("invalid correction or calibration specification")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    @property
    def n_days(self):
        return 2520 if self.profile == "paper" else 96

    @property
    def n_bars(self):
        return 390


@dataclass(frozen=True)
class Scenario:
    table: int
    H: float
    kappa: float
    noise: float
    n_paths: int

    @property
    def id(self):
        return f"t{self.table}_h{self.H:.2f}_k{self.kappa:.3f}_noise{self.noise:.0e}"

    @property
    def estimators(self):
        if self.table == 3:
            return ("rv5m",)
        if self.table == 6:
            return ("rv5m", "rk")
        return ESTIMATORS

    @property
    def pairs(self):
        return PAIR_ESTIMATORS if self.table == 5 else (("rk",) if self.table == 6 else ())

    @property
    def h_bounds(self):
        return (1e-6, .49 if self.H <= .20 else .99)


def build_design(profile="paper", tables=(3, 4, 5, 6)):
    if profile not in ("smoke", "paper") or not set(tables).issubset({3, 4, 5, 6}):
        raise ValueError("unknown profile or table")
    scenarios = []
    count = 1000 if profile == "paper" else 2
    for table in sorted(set(tables)):
        if table == 3:
            grid = [(h, k) for h in H_VALUES for k in KAPPA_VALUES] if profile == "paper" else [(.20, .010)]
            scenarios += [Scenario(3, h, k, 1e-4, count) for h, k in grid]
        elif table == 4:
            scenarios += [Scenario(4, .20, .010, noise, count) for noise in (1e-4, 5e-4)]
        elif table == 5:
            # All six pairings use the same path and are separately summarized.
            scenarios.append(Scenario(5, .20, .010, 1e-4, count))
        else:
            grid = [(h, k) for h in H_VALUES for k in (0., .010, .035)] if profile == "paper" else [(.10, 0.), (.50, .035)]
            scenarios += [Scenario(6, h, k, 1e-4, 400 if profile == "paper" else 2) for h, k in grid]
    return scenarios


def path_seed(master_seed: int, scenario: Scenario, path_id: int, stream: str) -> int:
    # Excludes path count, selection order and any Python process-randomized hash.
    semantic = dict(master_seed=master_seed, table=scenario.table, H=scenario.H,
                    kappa=scenario.kappa, noise=scenario.noise, path_id=path_id, stream=stream)
    return int(digest(semantic)[:16], 16)


def metric_names(scenario):
    names = ["latent.H10", "latent.H40"]
    for est in scenario.estimators:
        names += [f"{est}.H10", f"{est}.H40", f"{est}.omega2", f"{est}.invalid_day_fraction"]
    for second in scenario.pairs:
        prefix = "pair_" + second
        names += [prefix + "." + key for key in ("Hcorr10_a", "Hcorr10_b", "Hcorr40_a", "Hcorr40_b",
                 "omega_fit_a2", "omega_fit_b2", "fit_H", "delta_m", "true_variance_gap_twice", "gap_consistency_error")]
    return names


def simulate_path(config: RunConfig, scenario: Scenario, path_id: int, cache_dir: Path):
    seeds = {stream: path_seed(config.seed, scenario, path_id, stream) for stream in ("latent", "intraday")}
    record = {"scenario_id": scenario.id, "path_id": path_id, "seeds": seeds,
              "metrics": dict.fromkeys(metric_names(scenario)), "corrections": {},
              "status": "failed", "valid": False, "failure_reason": None}
    try:
        calibration = calibrate(scenario.H, scenario.kappa, horizon=config.calibration_horizon)
        x = generate_log_variance(config.n_days, calibration, seeds["latent"],
                                  cache_dir=str(cache_dir), generator=config.generator)
        record["calibration"] = asdict(calibration)
        prices = simulate_noisy_intraday_prices(x, n_bars=config.n_bars,
                                               varpi=scenario.noise, seed=seeds["intraday"])
        returns = np.diff(prices, axis=1)
        daily = {name: np.empty(config.n_days) for name in scenario.estimators}
        for day, r in enumerate(returns):
            if scenario.table == 3:
                measures = {"rv5m": compute_rv5m(r)}
            elif scenario.table == 6:
                measures = {"rv5m": compute_rv5m(r), "rk": compute_realized_kernel(r)}
            else:
                measures = compute_all_estimators(r)
            for est in scenario.estimators:
                daily[est][day] = measures[est]
        logs = {}
        for est, values in daily.items():
            valid = np.isfinite(values) & (values > 0)
            logs[est] = np.full(values.shape, np.nan)
            logs[est][valid] = np.log(values[valid])
            record["metrics"][est + ".invalid_day_fraction"] = float(1-valid.mean())
            error = logs[est] - x
            record["metrics"][est + ".omega2"] = float(np.var(error[valid], ddof=1)) if valid.sum() > 1 else np.nan
        for est, series in {"latent": x, **logs}.items():
            for end in (10, 40):
                fit = estimate_hurst_ols(series, end, overlapping=config.overlapping)
                record["metrics"][f"{est}.H{end}"] = fit["H_hat"]
        for second in scenario.pairs:
            result = fit_two_estimator_correction(logs["rv5m"], logs[second],
                delta_max=10, long_delta_max=40, h_bounds=scenario.h_bounds,
                specification=config.specification, overlapping=config.overlapping)
            record["corrections"][second] = result
            common = np.isfinite(logs["rv5m"]) & np.isfinite(logs[second])
            gap = 2*(np.var((logs["rv5m"]-x)[common], ddof=1) - np.var((logs[second]-x)[common], ddof=1)) if common.sum() > 1 else np.nan
            values = {"delta_m": result["delta_m"],
                      "true_variance_gap_twice": gap, "gap_consistency_error": result["delta_m"]-gap}
            for name, field in {"omega_fit_a2": "omega_a2", "omega_fit_b2": "omega_b2", "fit_H": "H_nu",
                                "Hcorr10_a": "H_corrected_a", "Hcorr10_b": "H_corrected_b",
                                "Hcorr40_a": "H_corrected_lag40_a", "Hcorr40_b": "H_corrected_lag40_b"}.items():
                # Keep failed-optimizer numbers in diagnostics, not valid MC means.
                values[name] = result[field] if result["optimizer_success"] else np.nan
            record["metrics"].update({f"pair_{second}.{name}": value for name, value in values.items()})
        # Figure 2 uses the smooth truth and exactly these three mean reversions.
        if scenario.table == 3 and scenario.H == .50 and scenario.kappa in (0., .010, .035):
            end = min(250, config.n_days-1)
            record["figure2_moments"] = {est: compute_second_moment_scaling(series, end, overlapping=config.overlapping)[1].tolist()
                                          for est, series in (("latent", x), ("rv5m", logs["rv5m"]))}
        record["status"] = "completed"
        record["valid"] = bool(all(value is not None and np.isfinite(value) for value in record["metrics"].values()))
        if not record["valid"]:
            record["failure_reason"] = "one_or_more_invalid_metrics; inspect_per_metric_counts_and_corrections"
    except Exception as exc:
        # Operational/numerical errors are completed failure receipts, not missing paths.
        record["failure_reason"] = type(exc).__name__ + ": " + str(exc)
    return record


def code_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = ["fou_spectral.py", "hurst_realized.py", "rv_estimators.py", "autocovariance_fit.py",
             "simulate_rough_paths.py", "replication/gaussian.py", "replication/monte_carlo.py", "replication/common.py"]
    return {"source_hashes": {path: sha256_file(root / "src" / path) for path in paths},
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "statsmodels")}}


@contextmanager
def _run_lock(output):
    import fcntl
    with (output / ".runner.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another runner owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_receipt(path, fingerprint, scenario, path_id, config):
    result = json.loads(path.read_text())
    expected = result.pop("record_sha256", None)
    if expected != digest(result):
        raise ValueError("path receipt checksum mismatch")
    if result["run_fingerprint"] != fingerprint or result["scenario_id"] != scenario.id or result["path_id"] != path_id:
        raise ValueError("path receipt identity/configuration mismatch")
    expected_seeds = {s: path_seed(config.seed, scenario, path_id, s) for s in ("latent", "intraday")}
    if result["seeds"] != expected_seeds or result["status"] not in ("completed", "failed"):
        raise ValueError("invalid completed path receipt")
    return result


def summarize(scenario, records):
    rows = []
    for metric in metric_names(scenario):
        values = [r["metrics"].get(metric) for r in records]
        values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        variance = float(np.var(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append({"scenario_id": scenario.id, **asdict(scenario), "metric": metric,
            "n_requested": scenario.n_paths, "n_completed": len(records), "n_valid": len(values),
            "n_invalid": len(records)-len(values), "mean": float(values.mean()) if len(values) else np.nan,
            "variance": variance, "mc_standard_error": float(np.sqrt(variance/len(values))) if len(values) > 1 else np.nan})
    diagnostics = {"scenario_id": scenario.id, "n_requested": scenario.n_paths,
        "n_completed": len(records), "n_valid_all_metrics": sum(bool(r["valid"]) for r in records),
        "path_status_counts": dict(Counter(r["status"] for r in records)),
        "failure_reasons": dict(Counter(r["failure_reason"] for r in records if r["failure_reason"])), "pairings": {}}
    for second in scenario.pairs:
        fits = [r["corrections"][second] for r in records if second in r["corrections"]]
        diagnostics["pairings"][second] = {"n_requested": scenario.n_paths, "n_attempted": len(fits),
            "n_converged": sum(bool(f["optimizer_success"]) for f in fits),
            "n_valid_short": sum(bool(f["optimizer_success"] and f["correction_valid_10"]) for f in fits),
            "n_valid_long": sum(bool(f["optimizer_success"] and f["correction_valid_40"]) for f in fits),
            "n_boundary": sum(any(f.get(k, False) for k in ("h_at_lower_bound", "h_at_upper_bound", "noise_at_lower_bound", "noise_at_upper_bound", "nu_at_lower_bound", "nu_at_upper_bound")) for f in fits),
            "convergence_status_counts": dict(Counter(f["convergence_status"] for f in fits))}
    return rows, diagnostics


def run(config: RunConfig, output_dir, tables=(3, 4, 5, 6), *, resume=False, max_new_paths=None):
    """Run serially. max_new_paths allows controlled interruption/resume tests."""
    scenarios = build_design(config.profile, tables)
    output = private_output(output_dir)
    spec = {"configuration": asdict(config), "n_days": config.n_days, "n_bars": config.n_bars,
            "code": code_fingerprint(), "receipt_schema_version": 1}
    fingerprint = digest(spec)
    summary_rows, diagnostics, figure_rows = [], [], []
    new_paths = 0
    with _run_lock(output):
        run_file = output / "run_spec.json"
        if run_file.exists():
            if not resume:
                raise ValueError("output already contains a run; use resume")
            if json.loads(run_file.read_text()) != spec:
                raise ValueError("resume configuration or code fingerprint mismatch")
        else:
            if any((output / "paths").glob("**/*.json")):
                raise ValueError("orphaned path receipts without run specification")
            atomic_json(run_file, spec)
        for scenario in scenarios:
            records = []
            for path_id in range(scenario.n_paths):
                path = output / "paths" / scenario.id / f"{path_id:06d}.json"
                if path.exists():
                    record = _read_receipt(path, fingerprint, scenario, path_id, config)
                elif max_new_paths is not None and new_paths >= max_new_paths:
                    continue
                else:
                    record = simulate_path(config, scenario, path_id, output / "covariance_cache")
                    record["run_fingerprint"] = fingerprint
                    atomic_json(path, {**record, "record_sha256": digest(record)})
                    new_paths += 1
                records.append(record)
            rows, diagnostic = summarize(scenario, records)
            summary_rows.extend(rows); diagnostics.append(diagnostic)
            for est in ("latent", "rv5m"):
                moments = [r["figure2_moments"][est] for r in records if "figure2_moments" in r]
                if moments:
                    for lag, column in enumerate(np.asarray(moments, dtype=float).T, 1):
                        finite = column[np.isfinite(column)]
                        figure_rows.append({"scenario_id": scenario.id, "estimator": est, "lag": lag,
                                            "mean_moment": float(finite.mean()) if len(finite) else np.nan,
                                            "n_valid": len(finite), "n_requested": scenario.n_paths})
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output / "mc_summary.csv", index=False)
        pd.DataFrame(figure_rows).to_csv(output / "figure2_moments.csv", index=False)
        status = {"profile": config.profile, "run_fingerprint": fingerprint, "new_paths": new_paths,
                  "n_requested": sum(s.n_paths for s in scenarios),
                  "n_completed": sum(d["n_completed"] for d in diagnostics),
                  "n_valid_all_metrics": sum(d["n_valid_all_metrics"] for d in diagnostics),
                  "empirical_data_used": False, "paper_results_reproduced": False,
                  "scenarios": diagnostics}
        atomic_json(output / "run_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--tables", nargs="+", type=int, choices=[3, 4, 5, 6], default=[3, 4, 5, 6])
    parser.add_argument("--output-dir", type=Path, default=Path(".local/replication_mc"))
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--generator", choices=["gaussian", "legacy"], default="gaussian")
    parser.add_argument("--specification", choices=["paper", "guarded"], default="paper")
    parser.add_argument("--overlapping", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-paper", action="store_true")
    args = parser.parse_args()
    config = RunConfig(args.profile, args.seed, args.generator, args.overlapping, args.specification)
    if args.dry_run:
        print(json.dumps({"configuration": asdict(config), "n_days": config.n_days, "n_bars": config.n_bars,
                          "scenarios": [{**asdict(s), "id": s.id, "estimators": s.estimators,
                                         "pairings": s.pairs, "fitted_H_bounds": s.h_bounds}
                                         for s in build_design(config.profile, args.tables)]}, indent=2))
    elif config.profile == "paper" and not args.execute_paper:
        parser.error("paper-scale execution requires explicit --execute-paper; use --dry-run to inspect")
    else:
        print(json.dumps(run(config, args.output_dir, args.tables, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
