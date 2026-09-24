"""Behavioral regressions for defects found in the delivery archive."""
import numpy as np
import pandas as pd

from src import autocovariance_fit
from src.replication import monte_carlo as mc


def test_failed_optimizer_iterates_are_not_valid_mc_estimates(monkeypatch, tmp_path):
    # Inject failure at the third-party solver boundary, retaining a real,
    # finite last iterate. A missing/NaN iterate would not expose this bug.
    solve = autocovariance_fit.minimize

    def unsuccessful(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.success = False
        result.message = "forced unsuccessful termination with finite iterate"
        return result

    monkeypatch.setattr(autocovariance_fit, "minimize", unsuccessful)
    scenario = mc.Scenario(6, .2, .01, 1e-4, 1)
    record = mc.simulate_path(mc.RunConfig(), scenario, 0, tmp_path)
    fit = record["corrections"]["rk"]
    assert not fit["optimizer_success"]
    assert np.isfinite([fit["H_nu"], fit["omega_a2"], fit["omega_b2"]]).all()
    rows, diagnostics = mc.summarize(scenario, [record])
    summary = pd.DataFrame(rows).set_index("metric")
    for metric in ("fit_H", "omega_fit_a2", "omega_fit_b2", "Hcorr10_b"):
        row = summary.loc["pair_rk." + metric]
        assert row.n_valid == 0 and row.n_invalid == 1 and pd.isna(row["mean"])
    # Observed moment gaps and uncorrected H do not require a converged fit.
    assert summary.loc["pair_rk.delta_m", "n_valid"] == 1
    assert summary.loc["rv5m.H10", "n_valid"] == 1
    assert diagnostics["pairings"]["rk"]["n_converged"] == 0
