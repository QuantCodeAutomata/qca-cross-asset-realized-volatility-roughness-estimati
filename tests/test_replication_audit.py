"""Independent audit tests for the pinned roughness repository.

Run from anywhere with:
  ROUGHNESS_REPO=/absolute/path/to/checkout python -m pytest -q test_replication_audit.py

These are the 35 previous audit requirements, integrated into the suite.
The two skew-filter tests are acceptance requirements for a complete OPRA
pipeline; a generic estimator may instead require a separate validated adapter.
"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest

repo_root = Path(os.environ.get("ROUGHNESS_REPO", ".")).resolve()
if not (repo_root / "src" / "option_pricing.py").is_file():
    raise RuntimeError("Set ROUGHNESS_REPO to the root of the target checkout.")
sys.path.insert(0, str(repo_root))

from src.fou_spectral import compute_exact_m2
from src.option_pricing import (
    _gbs_put, baw_approximation, baw_implied_vol,
    black76_price, black76_implied_vol, infer_forward_and_discount_pcp,
)
from src.atm_skew_estimator import build_skew_term_structure


@pytest.mark.parametrize("kappa", [.003, .01, .02, .035])
@pytest.mark.parametrize("delta", [1., 10., 40., 100.])
def test_fou_matches_brownian_ou_closed_form(kappa, delta):
    reference = -np.expm1(-kappa * delta) / kappa
    assert compute_exact_m2(delta, .5, kappa) == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("h", [.1, .2, .5])
@pytest.mark.parametrize("delta", [1., 10.])
def test_fou_zero_mean_reversion_limit(h, delta):
    assert compute_exact_m2(delta, h, 0.) == pytest.approx(delta ** (2. * h))


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [90., 100., 110.])
def test_black76_price_iv_roundtrip(option_type, strike):
    price = black76_price(100., strike, 1., .02, .2, option_type)
    result = black76_implied_vol(price, 100., strike, 1., .02, option_type)
    assert result == pytest.approx(.2, abs=1e-8)


def test_baw_positive_rate_put_iv_roundtrip():
    price = baw_approximation(100., 100., 1., .05, .02, .2, "put")
    result = baw_implied_vol(price, 100., 100., 1., .05, .02, "put")
    assert result == pytest.approx(.2, abs=1e-8)


def _parity_panel(discount):
    strikes = np.linspace(94., 106., 20)
    rate = -np.log(discount)
    calls = np.array([black76_price(100., k, 1., rate, .2, "call") for k in strikes])
    puts = np.array([black76_price(100., k, 1., rate, .2, "put") for k in strikes])
    return calls, puts, strikes


def test_parity_positive_rate_control():
    f, d = infer_forward_and_discount_pcp(*_parity_panel(.98))
    assert f == pytest.approx(100., abs=1e-8)
    assert d == pytest.approx(.98, abs=1e-10)


def test_parity_does_not_clip_valid_discount_above_one():
    f, d = infer_forward_and_discount_pcp(*_parity_panel(1.02))
    assert (f, d) == pytest.approx((100., 1.02), abs=1e-8), (
        "Clipping D to 1 also changes the inferred forward from 100 to 102."
    )


def test_baw_zero_rate_put_is_at_least_european_value():
    european = _gbs_put(100., 100., 1., 0., 0., .2)
    american = baw_approximation(100., 100., 1., 0., 0., .2, "put")
    assert np.isfinite(american) and american + 1e-8 >= european, (
        f"American={american}, European lower bound={european}; "
        "r=0 leads to 0/0 and an intrinsic-only fallback."
    )


def _chain(n_unique, repeats, positive_volume):
    # Acceptance now exercises the connected OPRA adapter, beginning with prices.
    # Both a call and a put are needed to identify parity at every strike.
    from tests.option_fixtures import price_fixture
    return price_fixture(n_strikes=n_unique, positive_volume=positive_volume,
                         n_dates=1, days=(30,))


def _accepted(*args):
    from src.replication.options import prices_to_implied_hurst
    result = prices_to_implied_hurst(*args)["skews"]
    return bool(not result.empty and result["valid"].any())


def test_skew_valid_positive_volume_control():
    assert _accepted(*_chain(20, 1, True))


def test_skew_minimum_counts_unique_strikes_not_rows():
    assert not _accepted(*_chain(10, 2, True)), (
        "20 call/put rows contain only 10 distinct strikes."
    )


def test_opra_zero_volume_chain_is_not_accepted():
    assert not _accepted(*_chain(20, 1, False)), (
        "Paper Section 3 requires positive-volume OPRA strikes."
    )
