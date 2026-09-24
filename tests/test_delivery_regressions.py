"""Behavioral regressions for defects found in the delivery archive."""
import json
import numpy as np
import pandas as pd
import pytest

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


def comparison_fixture(tmp_path, *, min_estimation_days=1):
    from tests.panel_fixtures import (calendar_fixture, instrument_fixture,
                                     bars_fixture, write_manifest)
    from tests.option_fixtures import price_fixture
    days = pd.bdate_range("2024-01-02", periods=45).strftime("%Y-%m-%d").tolist()
    calendar = calendar_fixture(days, minutes=10)
    instruments = instrument_fixture().iloc[[0]].copy()
    instruments["valid_from"], instruments["valid_to"] = days[0], days[-1]
    definitions, prices, context = price_fixture(n_dates=3)
    path = write_manifest(tmp_path, {
        "sessions": calendar, "instruments": instruments,
        "equity_bars": bars_fixture(calendar), "option_definitions": definitions,
        "option_prices": prices, "option_context": context,
    }, selections={"option_underlyings": ["U"],
                   "comparison_map": [{"realized_id": "E1", "option_id": "U"}]})
    manifest = json.loads(path.read_text())
    manifest["base_universe"]["min_estimation_days"] = min_estimation_days
    path.write_text(json.dumps(manifest))
    return path


def test_base_excluded_asset_is_diagnostic_not_primary_match(tmp_path):
    from src.replication.empirical import run
    path = comparison_fixture(tmp_path / "inputs", min_estimation_days=46)
    result = run(path, tmp_path / "output")
    assert not result["frames"]["equities_universe"].iloc[0].base_eligible
    row = result["frames"]["realized_implied_match"].iloc[0]
    assert np.isfinite(row.H_realized_TSRV) and np.isfinite(row.H_hat_IV)
    assert not row.raw_pair_available and not row.corrected_pair_available
    assert not row.base_eligible
    assert "insufficient_estimation_days" in row.base_exclusion_reasons
    assert result["report"]["counts"]["matched_assets"]["observed"] == 0


def test_quality_exclusion_does_not_remove_base_match(tmp_path):
    from src.replication.empirical import run
    path = comparison_fixture(tmp_path / "inputs")
    result = run(path, tmp_path / "output")
    universe = result["frames"]["equities_universe"].iloc[0]
    assert universe.base_eligible and not universe.quality_eligible
    row = result["frames"]["realized_implied_match"].iloc[0]
    assert row.raw_pair_available
    assert result["report"]["counts"]["matched_assets"]["observed"] == 1


@pytest.mark.parametrize("data_kind", ["synthetic_fixture", "market"])
def test_raw_and_corrected_coverage_are_independent(monkeypatch, tmp_path, data_kind):
    from src.replication.empirical import run
    from src.replication.figures import render_empirical
    solve = autocovariance_fit.minimize

    def unsuccessful(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.success = False
        result.message = "forced optimizer failure"
        return result

    monkeypatch.setattr(autocovariance_fit, "minimize", unsuccessful)
    path = comparison_fixture(tmp_path / "inputs")
    # Both receipt-label branches use the same synthetic inputs; no market access.
    manifest = json.loads(path.read_text())
    manifest["data_kind"] = data_kind
    path.write_text(json.dumps(manifest))
    result = run(path, tmp_path / "output")
    matched = result["frames"]["realized_implied_match"]
    assert matched.raw_pair_available.any() and not matched.corrected_pair_available.any()
    coverage = {row["item"]: row for row in result["report"]["coverage"]}
    flag = "checked_on_this_fixture" if data_kind == "synthetic_fixture" else "executed_on_supplied_market_data"
    assert coverage["figure6"][flag]
    assert not coverage["figure7"][flag]
    assert coverage["figure7"]["requires_source_or_separate_run"] == ["comparison_corrected"]
    assert result["report"]["comparison_counts"] == {"raw": 1, "corrected": 0}
    status = render_empirical(result, tmp_path / "figures")
    assert status["figure6"]["status"] == "rendered_from_supplied_fixture_or_data"
    assert status["figure7"]["status"] == "no_valid_corrected_pairs"
    assert not (tmp_path / "figures" / "figure7.png").exists()


@pytest.mark.parametrize("mode", ["all_parquet", "mixed_csv_parquet"])
def test_real_parquet_panels_match_csv_results(tmp_path, mode):
    pytest.importorskip("pyarrow")
    from src.replication.empirical import run
    from src.replication.manifest import FileManifest
    from src.replication.common import sha256_file
    path = comparison_fixture(tmp_path / "inputs")
    expected = run(path, tmp_path / "csv_output")
    reader = FileManifest.load(path)
    manifest = json.loads(path.read_text())
    for role in list(manifest["files"]):
        if mode == "mixed_csv_parquet" and role not in ("equity_bars", "option_context"):
            continue
        frame = reader.read(role)
        if role.startswith("option_"):
            for column in ("date", "expiry"):
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column])
        parts = np.array_split(np.arange(len(frame)), 2) if role == "equity_bars" else [np.arange(len(frame))]
        entries = []
        for index, positions in enumerate(parts):
            target = path.parent / f"{role}_{index}.parquet"
            frame.iloc[positions].to_parquet(target, index=False)
            assert target.read_bytes()[:4] == b"PAR1"
            entries.append({"path": target.name, "sha256": sha256_file(target)})
        manifest["files"][role] = entries
    path.write_text(json.dumps(manifest))
    actual = run(path, tmp_path / "parquet_output")
    assert actual["report"]["counts"] == expected["report"]["counts"]
    assert actual["report"]["comparison_counts"] == expected["report"]["comparison_counts"]
    for name in ("equities_H", "equities_corrections", "realized_implied_match"):
        pd.testing.assert_frame_equal(actual["frames"][name], expected["frames"][name],
                                      check_dtype=False, rtol=1e-9, atol=1e-10)
