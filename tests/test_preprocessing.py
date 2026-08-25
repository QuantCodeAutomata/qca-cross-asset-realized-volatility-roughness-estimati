"""Regression tests for causal intraday preprocessing."""

import numpy as np
import pandas as pd

from data.data_loader import generate_synthetic_intraday
from src.preprocessing.equity import (
    build_minute_grid,
    compute_intraday_returns,
    forward_fill_minute_prices,
)


def test_close_timestamp_grid_has_390_points_and_correct_endpoints():
    grid = build_minute_grid(pd.Timestamp("2024-01-03"), timestamp_convention="close")
    assert len(grid) == 390
    assert grid[0].strftime("%H:%M") == "09:31"
    assert grid[-1].strftime("%H:%M") == "16:00"


def test_positive_log_prices_are_not_logged_twice():
    log_prices = np.array([4.0, 4.1, 4.05])
    np.testing.assert_allclose(
        compute_intraday_returns(log_prices, input_is_log=True),
        np.array([0.1, -0.05]),
    )


def test_forward_fill_does_not_backfill_leading_missing_bars_by_default():
    date = pd.Timestamp("2024-01-03")
    grid = build_minute_grid(date)
    observed = pd.Series([100.0, 101.0], index=[grid[2], grid[4]])
    filled, fill_fraction = forward_fill_minute_prices(observed, date)
    assert np.isnan(filled.iloc[0]) and np.isnan(filled.iloc[1])
    assert filled.iloc[2] == 100.0
    assert filled.iloc[3] == 100.0
    assert fill_fraction == 2 / 390


def test_synthetic_generator_matches_annualized_volatility_level():
    panel = generate_synthetic_intraday(
        n_days=80, n_bars=20, H=0.2, kappa=0.01, seed=123
    )
    daily_log_iv = panel.groupby("date")["latent_log_iv"].first().to_numpy()
    annualized = np.mean(np.exp(daily_log_iv / 2.0)) * np.sqrt(252.0)
    np.testing.assert_allclose(annualized, 0.128, rtol=0.0, atol=1e-12)
