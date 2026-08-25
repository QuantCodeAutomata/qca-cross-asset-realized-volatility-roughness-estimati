#!/usr/bin/env python3
"""Run the audited rough-volatility estimators on real Massive minute bars.

The primary sample follows the paper's equity convention as closely as the
public text allows: regular-session one-minute closes, causal forward fill, at
least 120 observed minutes, close-to-close intraday log returns, and no
overnight return.  Raw bars and daily market-derived outputs are written only
to the caller-selected output directory, which should remain outside Git.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.autocovariance_fit import fit_two_estimator_correction
from src.hurst_realized import compute_second_moment_scaling
from src.rv_estimators import compute_all_estimators


SESSION_START_MINUTE = 9 * 60 + 30
SESSION_MINUTES = 390
ESTIMATORS = ("rv5m", "rk", "tsrv", "pav", "pabpv", "bpv", "ctrv")
PRIMARY_SAMPLE = "paper_like_min120_ffill"
STRICT_SAMPLE = "strict_full_390"
PRIMARY_RETURN_CONVENTION = "close_only_389"
OPEN_SEEDED_RETURN_CONVENTION = "open_seeded_390"
FIT_WINDOWS = ((1, 10), (1, 40), (40, 250))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_to_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def month_ranges(start: date, end: date) -> Iterator[tuple[date, date]]:
    if end < start:
        raise ValueError("end must not precede start")
    cursor = start.replace(day=1)
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        month_end = date(cursor.year, cursor.month, last_day)
        yield max(cursor, start), min(month_end, end)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def credential() -> str:
    for name in ("MASSIVE_TOKEN", "MASSIVE_API_KEY", "POLYGON_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def manifest_identity(offline: bool, chunk_receipts: list[dict]) -> tuple[str, str]:
    """Choose a receipt name without overwriting the original network receipt."""

    if offline:
        return "completed_offline_replay", "manifest_offline_rerun.json"
    if any(item.get("source") == "network" for item in chunk_receipts):
        return "completed_real_data_empirical_pilot", "manifest_network_run.json"
    return "completed_cached_replay", "manifest_cached_rerun.json"


def normalize_massive_rows(records: list[dict]) -> pd.DataFrame:
    columns = ["timestamp_ms", "date", "bar_idx", "open", "high", "low", "close", "volume"]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame.from_records(records)
    timestamp = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    local = timestamp.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    regular = (minute >= SESSION_START_MINUTE) & (minute < SESSION_START_MINUTE + SESSION_MINUTES)
    frame = frame.loc[regular].copy()
    local = local.loc[regular]
    minute = minute.loc[regular]
    frame["date"] = local.dt.tz_localize(None).dt.normalize()
    frame["bar_idx"] = (minute - SESSION_START_MINUTE).astype("int16")
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values(["date", "bar_idx", "timestamp_ms"])
    frame = frame.drop_duplicates(["date", "bar_idx"], keep="last")
    return frame[columns].reset_index(drop=True)


def fetch_month(
    client: object | None,
    *,
    ticker: str,
    start: date,
    end: date,
    cache_dir: Path,
    offline: bool,
) -> tuple[pd.DataFrame, dict]:
    safe_ticker = "".join(char if char.isalnum() or char in ".-_" else "_" for char in ticker)
    cache_path = cache_dir / f"{safe_ticker}_{start.isoformat()}_{end.isoformat()}_1min_rth.parquet"
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        return frame, {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": "cache",
            "rows": int(len(frame)),
            "cache_file": cache_path.name,
            "sha256": sha256_file(cache_path),
        }
    if offline:
        raise FileNotFoundError(f"offline cache miss: {cache_path}")
    if client is None:
        raise RuntimeError("Massive client is unavailable")

    rows: list[dict] = []
    for bar in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="minute",
        from_=start.isoformat(),
        to=end.isoformat(),
        adjusted=True,
        sort="asc",
        limit=50_000,
    ):
        rows.append(
            {
                "timestamp_ms": int(bar.timestamp),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": np.nan if bar.volume is None else float(bar.volume),
            }
        )
    frame = normalize_massive_rows(rows)
    atomic_to_parquet(frame, cache_path)
    return frame, {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source": "network",
        "rows": int(len(frame)),
        "cache_file": cache_path.name,
        "sha256": sha256_file(cache_path),
    }


def _daily_row(
    group: pd.DataFrame,
    *,
    sample: str,
    return_convention: str,
    compute_all_estimators: Callable[[np.ndarray], dict],
) -> dict:
    day = pd.Timestamp(group["date"].iloc[0]).normalize()
    grid = np.full(SESSION_MINUTES, np.nan, dtype=np.float64)
    opens = np.full(SESSION_MINUTES, np.nan, dtype=np.float64)
    positions = group["bar_idx"].to_numpy(dtype=np.int64)
    closes = group["close"].to_numpy(dtype=np.float64)
    open_values = group["open"].to_numpy(dtype=np.float64)
    valid_position = (positions >= 0) & (positions < SESSION_MINUTES)
    positions = positions[valid_position]
    closes = closes[valid_position]
    open_values = open_values[valid_position]
    positive = np.isfinite(closes) & (closes > 0.0) & np.isfinite(open_values) & (open_values > 0.0)
    positions = positions[positive]
    closes = closes[positive]
    open_values = open_values[positive]
    grid[positions] = closes
    opens[positions] = open_values
    observed = int(np.count_nonzero(np.isfinite(grid)))
    first_bar = int(np.flatnonzero(np.isfinite(grid))[0]) if observed else -1
    last_bar = int(np.flatnonzero(np.isfinite(grid))[-1]) if observed else -1

    if sample == PRIMARY_SAMPLE:
        eligible = observed >= 120
        filled = pd.Series(grid).ffill().to_numpy(dtype=np.float64)
    elif sample == STRICT_SAMPLE:
        eligible = observed == SESSION_MINUTES
        filled = grid
    else:
        raise ValueError(f"unknown sample: {sample}")

    with np.errstate(divide="ignore", invalid="ignore"):
        if return_convention == PRIMARY_RETURN_CONVENTION:
            # Literal Appendix-B convention: returns between the 390 minute closes.
            returns = np.diff(np.log(filled))
        elif return_convention == OPEN_SEEDED_RETURN_CONVENTION:
            # Count-consistent sensitivity: the first interval uses its 09:30 bar
            # open, followed by close-to-close returns.  This yields 390 returns
            # on a complete session and excludes the overnight move.
            returns = np.full(SESSION_MINUTES, np.nan, dtype=np.float64)
            if first_bar >= 0:
                returns[first_bar] = np.log(grid[first_bar]) - np.log(opens[first_bar])
                if first_bar + 1 < SESSION_MINUTES:
                    returns[first_bar + 1 :] = np.diff(np.log(filled[first_bar:]))
        else:
            raise ValueError(f"unknown return convention: {return_convention}")
    finite_returns = int(np.count_nonzero(np.isfinite(returns)))
    row = {
        "date": day,
        "sample": sample,
        "return_convention": return_convention,
        "observed_bars": observed,
        "coverage_fraction": observed / SESSION_MINUTES,
        "first_bar_idx": first_bar,
        "last_bar_idx": last_bar,
        "finite_returns": finite_returns,
        "eligible": bool(eligible and finite_returns >= 5),
        "status": "eligible" if eligible and finite_returns >= 5 else "insufficient_session",
    }
    row.update({name: np.nan for name in ESTIMATORS})
    if row["eligible"]:
        row.update(compute_all_estimators(returns))
    return row


def build_daily_measures(
    bars: pd.DataFrame,
    compute_all_estimators: Callable[[np.ndarray], dict],
    *,
    return_convention: str = PRIMARY_RETURN_CONVENTION,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    records: list[dict] = []
    for _, group in bars.groupby("date", sort=True):
        for sample in (PRIMARY_SAMPLE, STRICT_SAMPLE):
            records.append(
                _daily_row(
                    group,
                    sample=sample,
                    return_convention=return_convention,
                    compute_all_estimators=compute_all_estimators,
                )
            )
    return pd.DataFrame.from_records(records).sort_values(["sample", "date"]).reset_index(drop=True)


def fit_hurst_window(
    log_rv_series: np.ndarray,
    *,
    delta_min: int,
    delta_max: int,
    overlapping: bool,
    compute_second_moment_scaling: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Fit the log-moment slope on an explicit inclusive lag window."""

    x = np.asarray(log_rv_series, dtype=np.float64)
    deltas, moments = compute_second_moment_scaling(
        x,
        delta_max=delta_max,
        overlapping=overlapping,
    )
    valid = (
        (deltas >= delta_min)
        & np.isfinite(moments)
        & (moments > 0.0)
    )
    if int(valid.sum()) < 2:
        return {
            "H_hat": np.nan,
            "b": np.nan,
            "a": np.nan,
            "r_squared": np.nan,
            "n_lags": int(valid.sum()),
            "n_obs": int(x.size),
        }
    design = sm.add_constant(np.log(deltas[valid]))
    result = sm.OLS(np.log(moments[valid]), design).fit()
    return {
        "H_hat": float(result.params[1]) / 2.0,
        "b": float(result.params[1]),
        "a": float(result.params[0]),
        "r_squared": float(result.rsquared),
        "n_lags": int(valid.sum()),
        "n_obs": int(x.size),
    }


def estimate_tables(
    daily: pd.DataFrame,
    *,
    compute_second_moment_scaling: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimate_rows: list[dict] = []
    moment_rows: list[dict] = []
    for sample, sample_frame in daily.groupby("sample", sort=True):
        sample_frame = sample_frame.sort_values("date")
        for estimator in ESTIMATORS:
            values = sample_frame[estimator].to_numpy(dtype=np.float64)
            valid = np.isfinite(values) & (values > 0.0)
            log_rv = np.full(values.shape, np.nan, dtype=np.float64)
            log_rv[valid] = np.log(values[valid])
            deltas, moments = compute_second_moment_scaling(
                log_rv,
                delta_max=max(delta_max for _, delta_max in FIT_WINDOWS),
            )
            for delta, moment in zip(deltas, moments):
                moment_rows.append(
                    {
                        "sample": sample,
                        "return_convention": str(sample_frame["return_convention"].iloc[0]),
                        "estimator": estimator,
                        "delta": int(delta),
                        "m2": float(moment),
                    }
                )
            for delta_min, delta_max in FIT_WINDOWS:
                fit = fit_hurst_window(
                    log_rv,
                    delta_min=delta_min,
                    delta_max=delta_max,
                    overlapping=False,
                    compute_second_moment_scaling=compute_second_moment_scaling,
                )
                estimate_rows.append(
                    {
                        "sample": sample,
                        "return_convention": str(sample_frame["return_convention"].iloc[0]),
                        "estimator": estimator,
                        "delta_min": delta_min,
                        "delta_max": delta_max,
                        "H_hat": fit["H_hat"],
                        "slope": fit["b"],
                        "intercept": fit["a"],
                        "r_squared": fit["r_squared"],
                        "n_lags": fit["n_lags"],
                        "n_calendar_rows": int(len(log_rv)),
                        "n_valid_days": int(valid.sum()),
                    }
                )
    return pd.DataFrame(estimate_rows), pd.DataFrame(moment_rows)


def correction_results(
    daily: pd.DataFrame,
    fit_two_estimator_correction: Callable[..., dict],
) -> dict:
    output: dict[str, dict] = {}
    for sample, frame in daily.groupby("sample", sort=True):
        frame = frame.sort_values("date")
        series: dict[str, np.ndarray] = {}
        for estimator in ("rv5m", "rk"):
            values = frame[estimator].to_numpy(dtype=np.float64)
            valid = np.isfinite(values) & (values > 0.0)
            log_rv = np.full(values.shape, np.nan, dtype=np.float64)
            log_rv[valid] = np.log(values[valid])
            series[estimator] = log_rv
        raw = fit_two_estimator_correction(
            series["rv5m"],
            series["rk"],
            delta_max=10,
            long_delta_max=40,
        )
        output[str(sample)] = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in raw.items()
        }
    return output


def increment_convention_table(
    daily: pd.DataFrame,
    compute_second_moment_scaling: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Record the paper's unresolved overlapping-increment sensitivity."""

    rows: list[dict] = []
    for sample, frame in daily.groupby("sample", sort=True):
        frame = frame.sort_values("date")
        for estimator in ("tsrv", "rv5m", "rk"):
            values = frame[estimator].to_numpy(dtype=np.float64)
            valid = np.isfinite(values) & (values > 0.0)
            log_rv = np.full(values.shape, np.nan, dtype=np.float64)
            log_rv[valid] = np.log(values[valid])
            for overlapping in (False, True):
                for delta_min, delta_max in FIT_WINDOWS:
                    fit = fit_hurst_window(
                        log_rv,
                        delta_min=delta_min,
                        delta_max=delta_max,
                        overlapping=overlapping,
                        compute_second_moment_scaling=compute_second_moment_scaling,
                    )
                    rows.append(
                        {
                            "sample": sample,
                            "estimator": estimator,
                            "return_convention": str(frame["return_convention"].iloc[0]),
                            "increments": "overlapping" if overlapping else "non_overlapping",
                            "delta_min": delta_min,
                            "delta_max": delta_max,
                            **fit,
                            "n_valid_days": int(valid.sum()),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def write_loglog_plot(moments: pd.DataFrame, estimates: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = moments[(moments["sample"] == PRIMARY_SAMPLE) & (moments["estimator"] == "tsrv")]
    subset = subset[np.isfinite(subset["m2"]) & (subset["m2"] > 0.0)].copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(np.log(subset["delta"]), np.log(subset["m2"]), s=20, color="#345995", label="SPY TSRV moments")
    colors = {(1, 10): "#e4572e", (1, 40): "#17b890", (40, 250): "#7b2cbf"}
    for delta_min, delta_max in FIT_WINDOWS:
        fit = estimates[
            (estimates["sample"] == PRIMARY_SAMPLE)
            & (estimates["estimator"] == "tsrv")
            & (estimates["delta_min"] == delta_min)
            & (estimates["delta_max"] == delta_max)
        ].iloc[0]
        x = np.log(np.arange(delta_min, delta_max + 1, dtype=np.float64))
        y = float(fit["intercept"]) + float(fit["slope"]) * x
        ax.plot(
            x,
            y,
            color=colors[(delta_min, delta_max)],
            linewidth=1.6,
            label=f"fit {delta_min}-{delta_max}: H={fit['H_hat']:.3f}",
        )
    ax.set_xlabel("log trading-day lag")
    ax.set_ylabel("log second moment")
    ax.set_title("SPY real-data rough-volatility scaling (TSRV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, path)


def dependency_versions() -> dict[str, str]:
    names = ("massive", "numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pyarrow")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    versions["python"] = sys.version.split()[0]
    return versions


def json_safe(value: object) -> object:
    """Convert NumPy values and non-finite floats to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SPY real-data rough-volatility pilot."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".local" / "spy_empirical",
        help="Private output/cache directory (default: .local/spy_empirical).",
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 5, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "raw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing_cache = [
        (month_start, month_end)
        for month_start, month_end in month_ranges(args.start, args.end)
        if not (
            cache_dir
            / f"{args.ticker}_{month_start.isoformat()}_{month_end.isoformat()}_1min_rth.parquet"
        ).exists()
    ]
    token = credential()
    if missing_cache and not args.offline and not token:
        raise RuntimeError("MASSIVE_TOKEN, MASSIVE_API_KEY, or POLYGON_API_KEY is required")
    client = None
    if missing_cache and not args.offline:
        from massive import RESTClient

        client = RESTClient(api_key=token)

    chunks: list[pd.DataFrame] = []
    chunk_receipts: list[dict] = []
    for index, (month_start, month_end) in enumerate(month_ranges(args.start, args.end), start=1):
        frame, receipt = fetch_month(
            client,
            ticker=args.ticker,
            start=month_start,
            end=month_end,
            cache_dir=cache_dir,
            offline=args.offline,
        )
        chunks.append(frame)
        chunk_receipts.append(receipt)
        print(
            f"[{index:02d}] {month_start:%Y-%m}: {len(frame):,} RTH rows ({receipt['source']})",
            flush=True,
        )

    bars = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if bars.empty:
        raise RuntimeError("Massive returned no regular-session bars")
    bars = bars.sort_values(["date", "bar_idx", "timestamp_ms"])
    bars = bars.drop_duplicates(["date", "bar_idx"], keep="last").reset_index(drop=True)
    daily = build_daily_measures(
        bars,
        compute_all_estimators,
        return_convention=PRIMARY_RETURN_CONVENTION,
    )
    open_seeded_daily = build_daily_measures(
        bars,
        compute_all_estimators,
        return_convention=OPEN_SEEDED_RETURN_CONVENTION,
    )
    estimates, moments = estimate_tables(
        daily,
        compute_second_moment_scaling=compute_second_moment_scaling,
    )
    open_seeded_estimates, _ = estimate_tables(
        open_seeded_daily,
        compute_second_moment_scaling=compute_second_moment_scaling,
    )
    increment_sensitivity = increment_convention_table(
        daily,
        compute_second_moment_scaling,
    )
    correction = correction_results(daily, fit_two_estimator_correction)
    open_seeded_correction = correction_results(
        open_seeded_daily,
        fit_two_estimator_correction,
    )
    return_sensitivity = pd.concat(
        [estimates, open_seeded_estimates],
        ignore_index=True,
    )

    atomic_to_csv(daily, args.output_dir / "daily_realized_variance.csv")
    atomic_to_parquet(daily, args.output_dir / "daily_realized_variance.parquet")
    atomic_to_csv(estimates, args.output_dir / "hurst_estimates.csv")
    atomic_to_csv(
        increment_sensitivity,
        args.output_dir / "increment_convention_sensitivity.csv",
    )
    atomic_to_csv(
        return_sensitivity,
        args.output_dir / "return_convention_sensitivity.csv",
    )
    atomic_to_csv(moments, args.output_dir / "second_moment_scaling.csv")
    atomic_write_text(
        args.output_dir / "measurement_error_correction.json",
        json.dumps(json_safe(correction), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(
        args.output_dir / "return_convention_correction_sensitivity.json",
        json.dumps(
            json_safe(
                {
                    PRIMARY_RETURN_CONVENTION: correction,
                    OPEN_SEEDED_RETURN_CONVENTION: open_seeded_correction,
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    write_loglog_plot(moments, estimates, args.output_dir / "spy_tsrv_loglog.png")

    quality = (
        daily.groupby("sample", sort=True)
        .agg(
            calendar_rows=("date", "size"),
            eligible_days=("eligible", "sum"),
            median_observed_bars=("observed_bars", "median"),
            median_coverage=("coverage_fraction", "median"),
            minimum_observed_bars=("observed_bars", "min"),
        )
        .reset_index()
    )
    atomic_to_csv(quality, args.output_dir / "data_quality_summary.csv")
    manifest_status, manifest_name = manifest_identity(args.offline, chunk_receipts)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": manifest_status,
        "ticker": args.ticker,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "provider": "Massive adjusted stock aggregates",
        "provider_endpoint_contract": "v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}",
        "adjusted": True,
        "timezone": "America/New_York",
        "regular_session": "09:30 <= start timestamp < 16:00",
        "session_grid_minutes": SESSION_MINUTES,
        "primary_sample": {
            "name": PRIMARY_SAMPLE,
            "minimum_observed_bars": 120,
            "missing_price_policy": "causal forward fill only",
            "return_convention": PRIMARY_RETURN_CONVENTION,
            "return_count_on_full_session": 389,
            "overnight_return": "excluded",
        },
        "return_convention_sensitivity": {
            "name": OPEN_SEEDED_RETURN_CONVENTION,
            "return_count_on_full_session": 390,
            "first_interval": "09:30 bar open to first minute close",
            "reason": "paper close-only wording conflicts with its 390-return simulation count",
        },
        "strict_sensitivity": {
            "name": STRICT_SAMPLE,
            "required_observed_bars": 390,
            "missing_price_policy": "none",
        },
        "hurst_specification": {
            "increments": "non-overlapping exact pairs",
            "fit_windows": [
                {"delta_min": delta_min, "delta_max": delta_max}
                for delta_min, delta_max in FIT_WINDOWS
            ],
            "primary_estimator": "tsrv",
        },
        "code": {
            "entrypoint": "exp/exp5_spy_empirical.py",
            "repository": "QuantCodeAutomata/qca-cross-asset-realized-volatility-roughness-estimati",
        },
        "counts": {
            "rth_bars": int(len(bars)),
            "observed_session_dates": int(bars["date"].nunique()),
            "monthly_chunks": int(len(chunk_receipts)),
            "network_chunks": int(sum(item["source"] == "network" for item in chunk_receipts)),
            "cached_chunks": int(sum(item["source"] == "cache" for item in chunk_receipts)),
        },
        "dependencies": dependency_versions(),
        "chunks": chunk_receipts,
        "credential_values_persisted": False,
        "limitations": [
            "single-instrument pilot, not the paper's 3,926-equity cross-section",
            "Massive aggregates are not the paper's XNAS.ITCH source",
            "paper leaves return-boundary, session, bandwidth, and corporate-action conventions underspecified",
        ],
    }
    atomic_write_text(
        args.output_dir / manifest_name,
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    primary = estimates[
        (estimates["sample"] == PRIMARY_SAMPLE) & (estimates["estimator"] == "tsrv")
    ].sort_values("delta_max")
    print("\nPrimary SPY TSRV results")
    print(primary[["delta_min", "delta_max", "H_hat", "r_squared", "n_valid_days"]].to_string(index=False))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
