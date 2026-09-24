"""Regression checks through the pre-extension public interfaces."""
import numpy as np
import pytest

from src.option_pricing import baw_approximation, infer_forward_and_discount_pcp
from src.atm_skew_estimator import estimate_atm_skew


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
