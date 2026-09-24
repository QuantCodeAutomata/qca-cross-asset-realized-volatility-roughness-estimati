import numpy as np
import pandas as pd
import pytest
from src.replication.options import OptionConfig, prices_to_implied_hurst, write_option_results
from src import option_pricing as pricing
from tests.option_fixtures import price_fixture


@pytest.mark.parametrize("venue", ["CME", "OPRA"])
@pytest.mark.parametrize("rate", [-.02, 0., .02])
def test_prices_to_h_end_to_end(venue, rate):
    result = prices_to_implied_hurst(*price_fixture(venue=venue, rate=rate))
    assert result["skews"].valid.all()
    assert result["pooled_h"].iloc[0].H_hat_IV == pytest.approx(.23, abs=1e-7)
    assert result["pooled_h"].iloc[0].identified
    assert (result["skews"].n_strikes == 21).all()
    assert (result["skews"].n_quotes == 42).all()
    assert np.isfinite(result["skews"].psi_se).all()
    assert np.isfinite(result["pooled_h"].H_se).all()
    expected = "known_forward_no_intercept" if venue == "CME" else "joint_forward_discount"
    assert set(result["parity"]["mode"]) == {expected}


def test_opra_unique_strikes_and_volume():
    result = prices_to_implied_hurst(*price_fixture(n_strikes=10))
    assert not result["skews"].valid.any()
    assert set(result["skews"].status) == {"insufficient_unique_strikes"}
    result = prices_to_implied_hurst(*price_fixture(positive_volume=False))
    assert not result["skews"].valid.any()
    assert "OPRA_requires_positive_volume" in set(result["rejections"].reason)
    # CME accepts nontraded settlement marks and has a ten-strike minimum.
    result = prices_to_implied_hurst(*price_fixture(venue="CME", n_strikes=10, positive_volume=False))
    assert result["skews"].valid.all()


def test_duplicate_quotes_are_not_double_counted():
    defs, prices, context = price_fixture()
    baseline = prices_to_implied_hurst(defs, prices, context)
    duplicated = prices_to_implied_hurst(defs, pd.concat([prices, prices], ignore_index=True), context)
    pd.testing.assert_frame_equal(baseline["pooled_h"], duplicated["pooled_h"])
    assert len(duplicated["iv_quotes"]) == len(prices)
    assert len(duplicated["rejections"]) == len(prices)


def test_conflicting_duplicate_quote_is_rejected():
    defs, prices, ctx = price_fixture(n_dates=1)
    extra = prices.iloc[[0]].copy()
    extra.price += 1
    result = prices_to_implied_hurst(defs, pd.concat([prices, extra]), ctx)
    assert "ambiguous_equal_volume_quotes" in set(result["rejections"].reason)


def test_spx_monthly_weekly_merged_by_expiry_volume():
    result = prices_to_implied_hurst(*price_fixture(roots=("SPX", "SPXW")))
    assert set(result["series_selection"].selected_root) == {"SPXW"}
    assert set(result["iv_quotes"].root) == {"SPXW"}
    assert result["skews"].valid.all()
    assert (result["skews"].n_quotes == 42).all()


def test_maturity_boundaries_inclusive_and_alternative():
    fixture = price_fixture(days=(13, 14, 29, 30, 365, 366))
    result = prices_to_implied_hurst(*fixture)
    assert set(result["skews"].loc[result["skews"].valid, "maturity_days"]) == {14, 29, 30, 365}
    other = prices_to_implied_hurst(*fixture, OptionConfig(min_days=30))
    assert set(other["skews"].loc[other["skews"].valid, "maturity_days"]) == {30, 365}
    assert not other["pooled_h"].iloc[0].identified  # only two maturities per day


def test_otm_only_has_distinct_strike_counts():
    result = prices_to_implied_hurst(*price_fixture(), OptionConfig(otm_only=True))
    assert result["skews"].valid.all()
    assert (result["skews"].n_strikes == 21).all()
    assert (result["skews"].n_quotes == 22).all()  # call/put both retained at exactly ATM


def test_weak_identification_is_reported_not_dropped():
    result = prices_to_implied_hurst(*price_fixture(n_dates=30, weak=True))
    row = result["pooled_h"].iloc[0]
    assert not row.identified and np.isfinite(row.H_hat_IV)
    assert row.status == "weak_identification"
    assert row.n_dates == 30
    assert len(result["daily_h"]) == 30


def test_american_context_and_solver_failure(monkeypatch):
    defs, prices, context = price_fixture(n_dates=1)
    defs.exercise_style = "american"
    missing = prices_to_implied_hurst(defs, prices, context)
    assert not missing["skews"].valid.any()
    assert "American_requires_spot_and_explicit_dividend_assumption" in set(missing["rejections"].reason)
    context["spot"], context["dividend_yield"], context["dividend_assumption"] = 100., .02, "continuous yield fixed at 2% (synthetic)"
    def failed(*args, **kwargs):
        raise pricing.BoundarySolveError("injected_boundary_failure")
    monkeypatch.setattr(pricing, "baw_implied_vol", failed)
    result = prices_to_implied_hurst(defs, prices, context)
    assert not result["skews"].valid.any()
    assert set(result["rejections"].reason) == {"injected_boundary_failure"}
    assert set(result["rejections"].pricing_model) == {"BAW"}


def test_option_writer_is_local_and_private(tmp_path):
    result = prices_to_implied_hurst(*price_fixture(n_dates=1))
    hashes = write_option_results(result, tmp_path)
    assert "pooled_h.csv" in hashes
    assert (tmp_path / "option_run.json").exists()


def test_opra_volume_filter_applies_to_skew_not_both_parity_legs():
    defs, prices, ctx = price_fixture(n_dates=1)
    puts = set(defs.loc[defs.option_type=="put","contract_id"])
    prices.loc[prices.contract_id.isin(puts),"volume"] = 0
    result = prices_to_implied_hurst(defs,prices,ctx)
    assert result["skews"].valid.all()
    assert (result["skews"].n_strikes == 21).all()
    assert (result["skews"].n_quotes == 21).all()
    assert (result["parity"].n_pairs == 21).all()


def test_unpaired_quote_can_have_iv_after_parity_identification():
    defs,prices,ctx = price_fixture(n_dates=1)
    removed = defs[(defs.option_type=="put")].groupby("expiry").head(1).contract_id
    prices = prices[~prices.contract_id.isin(removed)]
    result = prices_to_implied_hurst(defs,prices,ctx)
    assert result["skews"].valid.all()
    assert (result["skews"].n_strikes == 21).all()
    assert (result["skews"].n_pairs == 20).all()


def test_declared_underlying_retained_without_any_quotes():
    defs,prices,ctx = price_fixture(n_dates=2)
    result = prices_to_implied_hurst(defs,prices.iloc[:0],ctx)
    assert len(result["pooled_h"]) == 1
    assert result["pooled_h"].iloc[0].status == "insufficient_data"
    assert len(result["daily_h"]) == 2


def test_cme_uses_the_settlement_of_the_defined_underlying_contract():
    defs,prices,ctx=price_fixture(venue="CME",n_dates=1)
    ctx["underlying_contract_id"]="wrong_futures_delivery"
    result=prices_to_implied_hurst(defs,prices,ctx)
    assert not result["skews"].valid.any()
    assert set(result["rejections"].reason)=={"CME_underlying_contract_mapping_mismatch"}


def test_american_zero_rate_price_to_h_not_just_iv_smiles():
    defs,prices,context = price_fixture(n_dates=2,rate=0,days=(14,30,90))
    defs["exercise_style"] = "american"
    context["spot"] = context.futures_settlement
    context["dividend_yield"] = 0.
    context["dividend_assumption"] = "zero continuous yield, synthetic"
    # Start with actual BAW prices. The pipeline must invert them itself.
    for i,row in prices.iterrows():
        definition = defs.iloc[i]
        T = (definition.expiry-definition.date).days/365
        spot = context.loc[(context.date==definition.date)&(context.expiry==definition.expiry),"spot"].iloc[0]
        k = np.log(definition.strike/spot)
        iv = .25-.08*T**(.23-.5)*k+.1*k*k
        prices.loc[i,"price"] = pricing.baw_approximation(spot,definition.strike,T,0,0,iv,definition.option_type)
    result = prices_to_implied_hurst(defs,prices,context)
    assert result["skews"].valid.all()
    assert set(result["iv_quotes"].pricing_model) == {"BAW"}
    assert result["pooled_h"].iloc[0].H_hat_IV == pytest.approx(.23,abs=1e-7)
    assert result["parity"].BAW_zero_rate_roundoff.all()
    assert np.max(np.abs(result["parity"].raw_parity_rate)) < 1e-12


def test_american_negative_parity_rate_rejected_not_black_fallback():
    defs,prices,context = price_fixture(n_dates=1,rate=-.02)
    defs["exercise_style"] = "american"
    context["spot"],context["dividend_yield"],context["dividend_assumption"] = 100.,0.,"synthetic zero yield"
    result = prices_to_implied_hurst(defs,prices,context)
    assert not result["skews"].valid.any()
    assert set(result["rejections"].reason) == {"BAW_negative_rates_unsupported"}
    assert (result["parity"].discount > 1).all()
    assert not result["parity"].BAW_zero_rate_roundoff.any()
