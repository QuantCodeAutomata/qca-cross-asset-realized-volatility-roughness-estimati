import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.fou_spectral import compute_exact_m2
from src.replication import gaussian as g
from src.replication import monte_carlo as mc


@pytest.mark.parametrize("h", [.05, .10, .15, .20, .30, .50])
def test_stationary_nu_and_covariance_matches_spectral_m2(h):
    c = g.calibrate(h, .01)
    assert g.stationary_covariance(0, h, .01, c.nu) == pytest.approx(1.05**2, abs=1e-12)
    for lag in (1, 10, 40):
        moment = 2 * (g.stationary_covariance(0, h, .01, c.nu) - g.stationary_covariance(lag, h, .01, c.nu))
        assert moment == pytest.approx(compute_exact_m2(lag, h, .01, c.nu), rel=1e-8, abs=1e-9)


@pytest.mark.parametrize("kappa", [0., .01])
def test_calibration_is_ensemble_not_pathwise(kappa):
    c = g.calibrate(.2, kappa)
    if kappa == 0:
        assert c.nu**2 * g.fbm_expected_sample_variance(2520, .2) == pytest.approx(1.05**2)
        variances = c.nu**2 * np.arange(2520)**.4
    else:
        variances = np.full(2520, 1.05**2)
    expected_mean = np.mean(np.exp(c.mean/2 + variances/8)) * np.sqrt(252)
    assert expected_mean == pytest.approx(.128, abs=1e-12)
    a = g.generate_log_variance(96, c, 11)
    b = g.generate_log_variance(96, c, 12)
    assert not np.isclose(np.std(a), np.std(b))
    assert not np.isclose(np.std(a, ddof=1), 1.05)


def test_fbm_calibration_brownian_closed_form_and_origin():
    assert g.fbm_expected_sample_variance(2520, .5) == pytest.approx(2521/6)
    c = g.calibrate(.5, 0)
    x = g.generate_log_variance(30, c, 17)
    assert x[0] == c.mean
    matrix = g.covariance_matrix(5, .5, 0, 2)
    np.testing.assert_allclose(matrix, 4*np.minimum.outer(np.arange(5), np.arange(5)))


@pytest.mark.parametrize("h", [.05, .2, .5])
def test_gaussian_factor_covariance_and_empirical_small_fixture(h):
    c = g.calibrate(h, .01)
    covariance = g.covariance_matrix(6, h, .01, c.nu)
    factor = g.checked_cholesky(covariance)
    np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-12, rtol=1e-12)
    batch = factor @ np.random.default_rng(51).normal(size=(6, 2048))
    observed = np.cov(batch)
    # Known Gaussian covariance sampling error; not calibrated to a paper table.
    se = np.sqrt((np.outer(np.diag(covariance), np.diag(covariance)) + covariance**2)/2047)
    assert np.all(np.abs(observed-covariance) < 6*se)


def test_nonpositive_covariance_not_repaired():
    with pytest.raises(g.CovarianceError, match="no_eigenvalue_repair"):
        g.checked_cholesky(np.array([[1., 2.], [2., 1.]]))


def test_factor_cache_integrity(tmp_path):
    factor = g.covariance_factor(20, .2, .01, .5, cache_dir=str(tmp_path)).copy()
    g.covariance_factor.cache_clear()
    np.testing.assert_array_equal(factor, g.covariance_factor(20, .2, .01, .5, cache_dir=str(tmp_path)))
    path = next(tmp_path.glob("*.npz"))
    with np.load(path, allow_pickle=False) as data:
        bad, metadata = data["factor"].copy(), data["metadata"].copy()
    bad[0, 0] += 1
    np.savez(path, factor=bad, metadata=metadata)
    g.covariance_factor.cache_clear()
    with pytest.raises(g.CovarianceError, match="integrity"):
        g.covariance_factor(20, .2, .01, .5, cache_dir=str(tmp_path))


def test_complete_paper_design_without_execution():
    design = mc.build_design("paper")
    assert len(design) == 51
    assert sum(s.n_paths for s in design) == 40200
    t3 = [s for s in design if s.table == 3]
    assert {(s.H,s.kappa) for s in t3} == {(h,k) for h in mc.H_VALUES for k in mc.KAPPA_VALUES}
    assert all(s.n_paths == 1000 and s.estimators == ("rv5m",) for s in t3)
    t4 = [s for s in design if s.table == 4]
    assert {s.noise for s in t4} == {1e-4,5e-4}
    assert all(len(s.estimators) == 7 for s in t4)
    t5 = [s for s in design if s.table == 5]
    assert len(t5) == 1 and len(t5[0].pairs) == 6 and t5[0].n_paths == 1000
    t6 = [s for s in design if s.table == 6]
    assert len(t6) == 18 and all(s.n_paths == 400 for s in t6)
    assert all(s.h_bounds == (1e-6, .49 if s.H <= .2 else .99) for s in t6)
    assert mc.RunConfig("paper").n_days == 2520 and mc.RunConfig("paper").n_bars == 390


def test_seed_independent_of_execution_order_and_path_count():
    a = mc.build_design("paper", [6,3])
    b = mc.build_design("paper", [3,6])
    assert {s.id:mc.path_seed(1,s,7,"latent") for s in a} == {s.id:mc.path_seed(1,s,7,"latent") for s in reversed(b)}
    scenario = a[0]
    changed = mc.Scenario(scenario.table, scenario.H, scenario.kappa, scenario.noise, 2)
    assert mc.path_seed(1,scenario,7,"latent") == mc.path_seed(1,changed,7,"latent")
    assert mc.path_seed(1,scenario,7,"latent") != mc.path_seed(1,scenario,7,"intraday")


def test_resume_and_partial_completion(tmp_path):
    config = mc.RunConfig()
    first = mc.run(config, tmp_path, [3], max_new_paths=1)
    assert first["n_requested"] == 2 and first["n_completed"] == 1
    result = mc.run(config, tmp_path, [3], resume=True)
    assert result["new_paths"] == 1 and result["n_completed"] == 2
    paths = sorted((tmp_path / "paths").glob("**/*.json"))
    before = {p:p.read_bytes() for p in paths}
    again = mc.run(config, tmp_path, [3], resume=True)
    assert again["new_paths"] == 0 and again["n_completed"] == 2
    assert all(p.read_bytes() == value for p,value in before.items())
    with pytest.raises(ValueError, match="fingerprint"):
        mc.run(mc.RunConfig(seed=8), tmp_path, [3], resume=True)
    paths[0].write_text(paths[0].read_text().replace('"path_id": 0','"path_id": 99'))
    with pytest.raises(ValueError, match="checksum"):
        mc.run(config, tmp_path, [3], resume=True)


def test_failures_are_counted_and_not_rerun(monkeypatch, tmp_path):
    def failed(*a, **kw):
        raise g.CovarianceError("synthetic non-PD test")
    monkeypatch.setattr(mc, "generate_log_variance", failed)
    out = mc.run(mc.RunConfig(), tmp_path, [3])
    assert out["n_completed"] == 2 and out["n_valid_all_metrics"] == 0
    summary = pd.read_csv(tmp_path / "mc_summary.csv")
    assert (summary.n_valid == 0).all() and (summary.n_invalid == 2).all()
    assert out["scenarios"][0]["path_status_counts"] == {"failed":2}
    assert mc.run(mc.RunConfig(), tmp_path, [3], resume=True)["new_paths"] == 0


def test_smoke_each_table_small_complete_design(tmp_path):
    result = mc.run(mc.RunConfig(), tmp_path)
    assert result["n_requested"] == result["n_completed"] == 12
    assert len(result["scenarios"]) == 6
    assert all(s["path_status_counts"] == {"completed":2} for s in result["scenarios"])
    t5 = [s for s in result["scenarios"] if s["scenario_id"].startswith("t5_")][0]
    assert set(t5["pairings"]) == set(mc.PAIR_ESTIMATORS)
    assert all(p["n_attempted"] == 2 for p in t5["pairings"].values())
