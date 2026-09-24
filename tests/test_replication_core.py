"""Numerical regression tests; paper-table disagreement is not a fit target."""
import numpy as np
import pytest
from src import option_pricing as op
from src.autocovariance_fit import fit_two_estimator_correction
from src.replication.table2 import independent_m2, PAPER_ROWS
from src.fou_spectral import compute_exact_m2
from src.simulate_rough_paths import simulate_fbm_davies_harte


@pytest.mark.parametrize("r", [0., 1e-14, 1e-9, .01, .05])
@pytest.mark.parametrize("q", [0., .02])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_baw_lower_bound_and_round_trip(r, q, kind):
    price = op.baw_approximation(100, 100, 1, r, q, .2, kind)
    euro = (op._gbs_call if kind == "call" else op._gbs_put)(100, 100, 1, r, q, .2)
    assert price >= euro - 1e-10
    assert op.baw_implied_vol(price, 100, 100, 1, r, q, kind, raise_errors=True) == pytest.approx(.2, abs=1e-8)


def test_zero_rate_put_value():
    assert op.baw_approximation(100, 100, 1, 0, 0, .2, "put") == pytest.approx(7.965567455405804, abs=1e-10)


def test_baw_boundary_failure_is_not_intrinsic(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate solver failure")
    monkeypatch.setattr(op, "brentq", fail)
    with pytest.raises(op.BoundarySolveError):
        op.baw_approximation(100, 100, 1, .05, .02, .2, "put")
    with pytest.raises(op.BoundarySolveError):
        op.baw_implied_vol(10, 100, 100, 1, .05, .02, "put", raise_errors=True)
    assert op.baw_implied_vol(10, 100, 100, 1, .05, .02, "put") is None
    # Analytically absent early exercise does not consult the failed solver.
    assert op.baw_approximation(100, 100, 1, 0, 0, .2, "put") > 7


@pytest.mark.parametrize("kind", ["call", "put"])
def test_negative_baw_explicit_rejection(kind):
    with pytest.raises(op.UnsupportedPricingDomain):
        op.baw_approximation(100, 100, 1, -.01, 0, .2, kind)


@pytest.mark.parametrize("discount", [.98, 1., 1.02])
@pytest.mark.parametrize("known", [False, True])
def test_parity_all_rate_signs(discount, known):
    k = np.linspace(90, 110, 21)
    r = -np.log(discount)
    c, p = [np.array([op.black76_price(100, x, 1, r, .2, kind) for x in k]) for kind in ("call", "put")]
    out = op.fit_put_call_parity(c, p, k, known_forward=100 if known else None)
    assert (out["forward"], out["discount"]) == pytest.approx((100, discount), abs=1e-10)
    assert out["rank"] == (1 if known else 2)
    assert out["n_unique_strikes"] == 21


@pytest.mark.parametrize("k", [[100, 100], [100], [0, 100], [100, np.nan]])
def test_parity_invalid_strikes(k):
    with pytest.raises(op.PricingError):
        op.fit_put_call_parity(np.ones(len(k)), np.ones(len(k)), k)


@pytest.mark.parametrize("h", [.05, .2, .5, .8])
@pytest.mark.parametrize("delta", [1., 10., 40.])
def test_independent_integral(h, delta):
    assert compute_exact_m2(delta, h, .01) == pytest.approx(independent_m2(delta, h, .01), rel=2e-8, abs=1e-9)


def test_table2_reference_shape():
    assert len(PAPER_ROWS) * 4 == 48


@pytest.fixture
def noisy_pair():
    rng = np.random.default_rng(18)
    x = np.r_[0, np.cumsum(simulate_fbm_davies_harte(1000, .2, seed=17))]
    return x + rng.normal(0, .2, len(x)), x + rng.normal(0, .1, len(x))


@pytest.mark.parametrize("specification", ["paper", "guarded"])
def test_correction_common_masks(noisy_pair, specification):
    a, b = (x.copy() for x in noisy_pair)
    a[[3, 40, 55]] = np.nan
    b[[4, 41, 65]] = np.nan
    result = fit_two_estimator_correction(a, b, specification=specification)
    common = np.isfinite(a) & np.isfinite(b)
    control = fit_two_estimator_correction(np.where(common, a, np.nan), np.where(common, b, np.nan), specification=specification)
    assert result["n_common_observations"] == int(common.sum())
    assert result["moment_pair_counts"][0] == np.sum(common[:-1] & common[1:])
    for key in ("delta_m", "omega_a2", "omega_b2", "H_nu"):
        assert result[key] == pytest.approx(control[key])
    assert result["omega_a2"] - result["omega_b2"] == pytest.approx(result["delta_m"] / 2, abs=1e-12)


@pytest.mark.parametrize("overlapping", [False, True])
def test_paper_short_fit_independent_of_long_window(noisy_pair, overlapping):
    a = fit_two_estimator_correction(*noisy_pair, specification="paper", long_delta_max=40, overlapping=overlapping)
    b = fit_two_estimator_correction(*noisy_pair, specification="paper", long_delta_max=300, overlapping=overlapping)
    for key in ("H_nu", "nu_fit", "omega_a2", "omega_b2", "H_corrected_a", "H_corrected_b", "objective"):
        assert a[key] == b[key]  # identical optimization, not a looser tolerance


def test_correction_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        fit_two_estimator_correction(np.zeros(100), np.zeros(99))
