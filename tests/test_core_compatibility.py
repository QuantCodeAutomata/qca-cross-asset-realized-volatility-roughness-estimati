"""Regression checks through the pre-extension public interfaces."""
import numpy as np
import pytest
import inspect

from src.option_pricing import baw_approximation, infer_forward_and_discount_pcp
from src.atm_skew_estimator import estimate_atm_skew
from src.autocovariance_fit import fit_two_estimator_correction
from src.hurst_realized import compute_second_moment_scaling, estimate_hurst_ols


def test_zero_rate_put_keeps_time_value():
    assert baw_approximation(100, 100, 1, 0, 0, .2, "put") == pytest.approx(
        7.965567455405804, abs=1e-10)


def test_parity_tuple_supports_discount_above_one():
    strikes = np.array([90., 100., 110.])
    assert infer_forward_and_discount_pcp(
        np.array([15.2, 10., 4.8]), np.array([5., 10., 15.]), strikes
    ) == pytest.approx((100., 1.02))


def test_skew_minimum_is_unique_strikes_not_quote_rows():
    strikes = np.repeat(np.linspace(95., 105., 10), 2)
    assert estimate_atm_skew(strikes, .2 - .1 * np.log(strikes / 100.),
                             100., min_strikes=20) is None


def test_legacy_defaults_and_return_contracts():
    from src.option_pricing import black76_implied_vol, baw_implied_vol
    assert inspect.signature(compute_second_moment_scaling).parameters["overlapping"].default is False
    parameters = inspect.signature(fit_two_estimator_correction).parameters
    assert parameters["specification"].default == "guarded"
    assert parameters["overlapping"].default is False
    assert parameters["h_bounds"].default == (.01, .49)
    for function in (black76_implied_vol, baw_implied_vol):
        parameter = inspect.signature(function).parameters["raise_errors"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is False
    assert black76_implied_vol(-1, 100, 100, 1, .02) is None
    assert baw_implied_vol(-1, 100, 100, 1, .02, 0) is None
    x = np.random.default_rng(10).normal(size=60).cumsum()
    deltas, moments = compute_second_moment_scaling(x)
    assert len(deltas) == len(moments) == 10
    assert set(estimate_hurst_ols(x)) == {"H_hat", "b", "a", "r_squared", "n_lags", "n_obs"}


@pytest.mark.parametrize("n,expected_h,expected_gap,effective_window", [
    (30, .49, .4708298494885213, 28),
    (100, .4793753266268174, .7812484655479546, 40),
    (500, .49, 1.7443824838783892, 40),
])
def test_guarded_matches_pinned_base_on_complete_calendar(n, expected_h, expected_gap, effective_window):
    # Frozen from 537ae369; only missing-mask, invalid-input and explicit paper
    # mode cases are intended to change the original correction's behavior.
    rng = np.random.default_rng(18)
    x = np.cumsum(rng.normal(size=n))
    a, b = x + rng.normal(0, .8, n), x + rng.normal(0, .2, n)
    result = fit_two_estimator_correction(a, b)
    assert result["H_nu"] == pytest.approx(expected_h, abs=1e-7)
    assert result["delta_m"] == pytest.approx(expected_gap, abs=1e-12)
    assert result["long_delta_max"] == effective_window
    assert result["requested_long_delta_max"] == 40
