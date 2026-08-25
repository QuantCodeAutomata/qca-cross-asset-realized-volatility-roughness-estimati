from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "exp" / "exp5_spy_empirical.py"
SPEC = importlib.util.spec_from_file_location("qca_real_pilot", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_month_ranges_preserve_requested_boundaries() -> None:
    assert list(MODULE.month_ranges(date(2024, 1, 15), date(2024, 3, 2))) == [
        (date(2024, 1, 15), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 2)),
    ]


def test_cached_online_rerun_does_not_overwrite_network_receipt() -> None:
    status, name = MODULE.manifest_identity(False, [{"source": "cache"}])
    assert status == "completed_cached_replay"
    assert name == "manifest_cached_rerun.json"


def _raw_day(day: str, bars: int) -> list[dict]:
    local = pd.date_range(f"{day} 09:30", periods=bars, freq="1min", tz="America/New_York")
    utc_ms = (local.tz_convert("UTC").astype("int64") // 1_000_000).to_numpy()
    prices = 100.0 * np.exp(np.linspace(0.0, 0.01, bars))
    return [
        {
            "timestamp_ms": int(timestamp),
            "open": float(price * 0.9999),
            "high": float(price * 1.0002),
            "low": float(price * 0.9998),
            "close": float(price),
            "volume": 1_000.0,
        }
        for timestamp, price in zip(utc_ms, prices)
    ]


def test_regular_session_normalization_and_duplicate_removal() -> None:
    rows = _raw_day("2024-01-02", 390)
    rows.append(dict(rows[10]))
    rows.append(
        {
            "timestamp_ms": int(pd.Timestamp("2024-01-02 08:00", tz="America/New_York").tz_convert("UTC").timestamp() * 1000),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    frame = MODULE.normalize_massive_rows(rows)
    assert len(frame) == 390
    assert frame["bar_idx"].tolist() == list(range(390))


def test_primary_keeps_120_bar_day_and_strict_rejects_it() -> None:
    bars = MODULE.normalize_massive_rows(_raw_day("2024-01-02", 120))

    def fake_estimators(returns: np.ndarray) -> dict:
        assert np.isfinite(returns).sum() == 389
        return {name: float(np.dot(returns, returns) + 1e-9) for name in MODULE.ESTIMATORS}

    daily = MODULE.build_daily_measures(bars, fake_estimators)
    primary = daily[daily["sample"] == MODULE.PRIMARY_SAMPLE].iloc[0]
    strict = daily[daily["sample"] == MODULE.STRICT_SAMPLE].iloc[0]
    assert bool(primary["eligible"])
    assert not bool(strict["eligible"])
    assert np.isfinite(primary["tsrv"])
    assert np.isnan(strict["tsrv"])


def test_open_seeded_sensitivity_has_390_returns() -> None:
    bars = MODULE.normalize_massive_rows(_raw_day("2024-01-02", 390))

    def fake_estimators(returns: np.ndarray) -> dict:
        assert np.isfinite(returns).sum() == 390
        return {name: float(np.dot(returns, returns) + 1e-9) for name in MODULE.ESTIMATORS}

    daily = MODULE.build_daily_measures(
        bars,
        fake_estimators,
        return_convention=MODULE.OPEN_SEEDED_RETURN_CONVENTION,
    )
    assert set(daily["return_convention"]) == {MODULE.OPEN_SEEDED_RETURN_CONVENTION}
    assert set(daily["finite_returns"]) == {390}


def test_explicit_long_lag_window_fit() -> None:
    def exact_moments(_series: np.ndarray, delta_max: int, *, overlapping: bool):
        assert overlapping
        deltas = np.arange(1, delta_max + 1, dtype=np.float64)
        return deltas, deltas**0.4

    fit = MODULE.fit_hurst_window(
        np.ones(300),
        delta_min=40,
        delta_max=250,
        overlapping=True,
        compute_second_moment_scaling=exact_moments,
    )
    assert np.isclose(fit["H_hat"], 0.2)
    assert np.isclose(fit["r_squared"], 1.0)
    assert fit["n_lags"] == 211
