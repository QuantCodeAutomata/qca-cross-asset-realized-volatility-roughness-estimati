import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
from src.replication.manifest import FileManifest, ManifestError, MissingSource, parse_timestamps
from src.replication.panels import ESTIMATORS, equity_panel, futures_panel, front_month_schedule, physical_realized_measures, aggregate_variance
from src.replication.empirical import run, export_approved_aggregates
from src.replication.common import private_output
from src.rv_estimators import compute_rv5m, compute_tsrv
from tests.panel_fixtures import *
from tests.option_fixtures import price_fixture


def test_timezone_dst_rename_and_delisting():
    calendar = calendar_fixture()
    bars = bars_fixture(calendar)
    daily, universe, excluded = equity_panel(bars,instrument_fixture(),calendar,dataset_config(),universe_rules())
    assert daily.active.tolist() == [True,True,False]
    assert daily.symbol.iloc[:2].tolist() == ["OLD","NEW"]
    assert daily.observed_bars.iloc[:2].tolist() == [390,390]
    assert universe.iloc[0].n_symbol_intervals == 2
    assert universe.iloc[0].exits_before_calendar_end
    assert daily.status.iloc[2] == "not_listed"
    assert "bars_outside_definition_lifetime" in set(excluded.reason)
    timestamps = parse_timestamps(bars.timestamp, dataset_config()["timestamp"])
    assert timestamps[0].hour == 14 and timestamps[390].hour == 13


def test_naive_timestamps_require_zone_and_dst_gap_rejected():
    with pytest.raises(ManifestError):
        parse_timestamps(["2024-03-08 09:30"],{"encoding":"iso8601"})
    with pytest.raises(Exception):
        parse_timestamps(["2024-03-10 02:30"],{"encoding":"iso8601","naive_timezone":"America/New_York"})


def test_missing_trading_day_is_not_compressed():
    calendar = calendar_fixture()
    bars = bars_fixture(calendar)
    bars = bars.iloc[np.r_[np.arange(390),np.arange(780,1170)]]
    instruments = instrument_fixture().iloc[[0]].copy()
    instruments["valid_to"] = "2024-03-12"
    daily,_,_ = equity_panel(bars,instruments,calendar,dataset_config(),universe_rules())
    assert len(daily) == 3 and daily.status.iloc[1] == "missing_session"
    assert np.isnan(daily.tsrv.iloc[1])


def test_incomplete_and_early_close_sessions_have_no_backfill():
    calendar = calendar_fixture(days=("2024-03-08",),minutes=210)
    bars = bars_fixture(calendar).iloc[2:100]
    daily,_,_ = equity_panel(bars,instrument_fixture(),calendar,dataset_config(),universe_rules())
    row = daily.iloc[0]
    assert row.n_close_prices == 210 and row.n_returns == 209
    assert row.leading_missing_bars == 2
    assert row.observed_bars == 98 and row.forward_filled_bars == 110
    assert row.n_finite_returns == 207
    assert row.observed_fraction == pytest.approx(98/210)


def test_price_node_conventions_and_physical_frequency():
    p = np.exp(np.linspace(4.,4.01,390))
    values,counts = physical_realized_measures(p,bar_seconds=60)
    assert counts["n_returns"] == 389
    assert values["rv5m"] == pytest.approx(compute_rv5m(np.diff(np.log(p))))
    assert values["tsrv"] == pytest.approx(compute_tsrv(np.diff(np.log(p))))
    _,counts = physical_realized_measures(p,bar_seconds=60,return_convention="open_seeded",open_price=np.exp(4.-.01/389))
    assert counts["n_price_nodes"] == 391 and counts["n_returns"] == 390
    second_prices = np.exp(np.arange(601)*1e-5)
    values,counts = physical_realized_measures(second_prices,bar_seconds=1)
    assert counts["five_minute_step"] == 300 and counts["rv5m_increments"] == 2
    assert values["rv5m"] == pytest.approx(2*(300e-5)**2)
    assert values["rv5m"] != pytest.approx(compute_rv5m(np.diff(np.log(second_prices))))


def test_quality_500_days_half_observed_eighty_percent():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2021-01-04",periods=501)]
    calendar = calendar_fixture(days,minutes=10)
    bars = bars_fixture(calendar)
    instruments = instrument_fixture().iloc[[0]].copy()
    instruments["valid_from"], instruments["valid_to"] = days[0], days[-1]
    daily,universe,_ = equity_panel(bars,instruments,calendar,dataset_config(),universe_rules())
    assert universe.iloc[0].n_estimation_days == 501
    assert universe.iloc[0].quality_eligible
    # Keep only the first 499 active sessions to test the exact minimum.
    _, short,_ = equity_panel(bars.iloc[:4990],instruments,calendar,dataset_config(),universe_rules())
    assert short.iloc[0].n_estimation_days == 499 and not short.iloc[0].quality_eligible


def futures_fixture():
    calendar = calendar_fixture()
    contracts = pd.DataFrame([{"root":"ES","contract_id":"A","expiry":"2024-06-20","calendar_id":"X","asset_class":"equity_index"},
                              {"root":"ES","contract_id":"B","expiry":"2024-09-20","calendar_id":"X","asset_class":"equity_index"}])
    rows = []
    for i,date in enumerate(calendar.session_date):
        rows += [{"root":"ES","date":date,"contract_id":"A","volume":[100,50,40][i]},
                 {"root":"ES","date":date,"contract_id":"B","volume":[50,100,100][i]}]
    a,b = bars_fixture(calendar,level=100), bars_fixture(calendar,level=300)
    a["contract_id"], b["contract_id"] = "A","B"
    return calendar,contracts,pd.DataFrame(rows),pd.concat([a,b],ignore_index=True)


def test_rollover_then_intraday_no_roll_price_jump():
    calendar,contracts,volumes,bars = futures_fixture()
    daily,schedule,excluded = futures_panel(bars,contracts,volumes,calendar,dataset_config(),["ES"],roll_timing="same_session")
    assert schedule.active_contract.tolist() == ["A","B","B"]
    assert schedule.rolled.tolist() == [False,True,False]
    # Exact scale invariance: choosing B's level 300 cannot add a log(3) return.
    control,_,_ = futures_panel(bars.assign(close=bars.close.where(bars.contract_id=="A",bars.close/3),
                                          open=bars.open.where(bars.contract_id=="A",bars.open/3)),
                               contracts,volumes,calendar,dataset_config(),["ES"],roll_timing="same_session")
    np.testing.assert_allclose(daily.tsrv,control.tsrv,atol=1e-15,rtol=1e-10)
    later = front_month_schedule(volumes,contracts,calendar,["ES"],roll_timing="next_session")
    assert later.active_contract.tolist() == ["A","A","B"]
    assert "not_selected_front_contract" in set(excluded.reason)


def test_futures_nonpositive_price_is_logged():
    calendar,contracts,volumes,bars = futures_fixture()
    bfirst = bars.index[bars.contract_id=="B"][390]
    bars.loc[bfirst,"close"] = -1
    daily,_,excluded = futures_panel(bars,contracts,volumes,calendar,dataset_config(),["ES"],roll_timing="same_session")
    assert daily.status.iloc[1] == "nonpositive_price" and np.isnan(daily.tsrv.iloc[1])
    assert "nonpositive_price" in set(excluded.reason)


def test_aggregation_keeps_missing_week_and_sums_variance():
    dates = pd.bdate_range("2024-01-01",periods=15)
    daily = pd.DataFrame({"instrument_id":"E", "date":dates,"active":True,**{name:np.ones(15) for name in ESTIMATORS}})
    daily.loc[5:9,list(ESTIMATORS)] = np.nan
    result = aggregate_variance(daily,frequency="weekly",missing_day_policy="invalidate",edge_period_policy="include")
    assert len(result)==3 and result.tsrv.iloc[0]==5 and np.isnan(result.tsrv.iloc[1]) and result.tsrv.iloc[2]==5


def test_manifest_hashes_units_and_missing_sources(tmp_path):
    path = write_manifest(tmp_path,{"sessions":calendar_fixture()})
    manifest = FileManifest.load(path)
    assert len(manifest.read("sessions")) == 3
    with pytest.raises(MissingSource):
        manifest.read("equity_bars")
    with pytest.raises(MissingSource):
        manifest.selection("seconds")
    with (tmp_path/"sessions.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ManifestError,match="hash mismatch"):
        manifest.read("sessions")


def test_empty_panels_report_missing_not_invented(tmp_path):
    manifest_path = write_manifest(tmp_path/"input",{})
    out = run(manifest_path,tmp_path/"output")
    assert out["report"]["counts"]["equities"]["observed"] == 0
    assert out["report"]["counts"]["equities"]["publication_reference"] == 3926
    assert not out["report"]["full_empirical_replication"]
    assert "options" in out["report"]["missing_sources"] and "seconds" in out["report"]["missing_sources"]
    assert all(not x["executed_on_supplied_market_data"] for x in out["report"]["coverage"])


def test_public_export_and_private_directory_guards(tmp_path):
    target = tmp_path/"repo"; target.mkdir(); (target/".git").mkdir()
    with pytest.raises(ValueError):
        private_output(target/"results")
    assert private_output(target/".local"/"private").is_dir()
    manifest_path = write_manifest(tmp_path/"input",{})
    result = run(manifest_path,tmp_path/"output")
    public = tmp_path/"public.json"
    export_approved_aggregates(result,public)
    assert set(json.loads(public.read_text())) == {"counts","manifest_sha256"}
    result["manifest"].content["publication"]["approved_aggregates"] = ["option_daily_H"]
    with pytest.raises(ManifestError):
        export_approved_aggregates(result,public)


def test_file_end_to_end_equity_futures_options(tmp_path):
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02",periods=45)]
    calendar = calendar_fixture(days,minutes=10)
    instruments = instrument_fixture().iloc[[0]].copy()
    instruments["valid_from"], instruments["valid_to"] = days[0], days[-1]
    definitions,prices,context = price_fixture(n_dates=3)
    _,contracts,_,_ = futures_fixture()
    volume = pd.DataFrame([{"date":day,"root":"ES","contract_id":c,"volume":v} for day in days for c,v in [("A",100),("B",20)]])
    a,b = bars_fixture(calendar,level=100),bars_fixture(calendar,level=300)
    a["contract_id"],b["contract_id"] = "A","B"
    files = {"sessions":calendar,"instruments":instruments,"equity_bars":bars_fixture(calendar),
        "futures_contracts":contracts,"futures_daily":volume,"futures_bars":pd.concat([a,b]),
        "option_definitions":definitions,"option_prices":prices,"option_context":context}
    path = write_manifest(tmp_path/"input",files,datasets={"equity":dataset_config(),"futures":dataset_config()},
        selections={"futures_roots":["ES"],"option_underlyings":["U"],"comparison_map":[{"realized_id":"E1","option_id":"U"}],"figure1_ids":["E1","ES"]},
        analysis={"futures_roll_timing":"same_session","lag_windows":[[1,10],[1,40],[40,250]],"overlapping":[False],
                  "aggregation":{"missing_day_policy":"invalidate","edge_period_policy":"include"}})
    result = run(path,tmp_path/"output")
    assert result["report"]["counts"]["equities"]["observed"] == 1
    assert result["report"]["counts"]["futures_roots"]["observed"] == 1
    assert result["report"]["counts"]["option_underlyings"]["observed"] == 1
    assert len(result["frames"]["realized_implied_match"]) == 1
    assert result["frames"]["realized_implied_match"].iloc[0].H_hat_IV == pytest.approx(.23,abs=1e-7)
    assert "equities_monthly_H" in result["frames"]
    assert "seconds" in result["report"]["missing_sources"]
    assert all(not r["executed_on_supplied_market_data"] for r in result["report"]["coverage"])
    from src.replication.figures import render_empirical
    status = render_empirical(result, tmp_path/"figures")
    for number in (1,3,4,5,6,7):
        assert any(key.startswith(f"figure{number}") and value["status"]=="rendered_from_supplied_fixture_or_data"
                   for key,value in status.items())


def test_parquet_adapter_engine_diagnostic_or_roundtrip(tmp_path):
    source = tmp_path/"toy.parquet"
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        pd.DataFrame({"x":[1,2]}).to_parquet(source)
        path = write_manifest(tmp_path,{})
        data = json.loads(path.read_text())
        data["files"]["toy"] = {"path":source.name,"sha256":sha256_file(source)}
        path.write_text(json.dumps(data))
        assert FileManifest.load(path).read("toy").x.tolist() == [1,2]
    else:
        source.write_bytes(b"PAR1synthetic-not-a-real-parquet-filePAR1")
        path = write_manifest(tmp_path,{})
        data = json.loads(path.read_text())
        data["files"]["toy"] = {"path":source.name,"sha256":sha256_file(source)}
        path.write_text(json.dumps(data))
        with pytest.raises(MissingSource,match="Parquet requires"):
            FileManifest.load(path).read("toy")


def test_seconds_file_scenario_uses_manifest_selection_and_300_second_grid(tmp_path):
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02",periods=15)]
    calendar = calendar_fixture(days,minutes=10)
    instruments = instrument_fixture().iloc[[0]].copy()
    instruments["valid_from"],instruments["valid_to"] = days[0],days[-1]
    files={"sessions":calendar,"instruments":instruments,"second_bars":bars_fixture(calendar,bar_seconds=1)}
    manifest=write_manifest(tmp_path/"input",files,datasets={"seconds":dataset_config(1)},selections={"seconds":["E1"]})
    result=run(manifest,tmp_path/"output")
    assert result["report"]["counts"]["second_frequency_assets"]["observed"] == 1
    assert (result["frames"]["seconds_daily_RV"].five_minute_step == 300).all()
    assert set(result["frames"]["seconds_H"].estimator) == set(ESTIMATORS)


def test_figure_renderers_use_products_and_keep_weak_identification(tmp_path):
    from src.replication.figures import render_empirical, render_figure2
    frame=pd.DataFrame({"H_realized_TSRV":[.2,.1],"H_realized_RK_corrected":[.24,.15],
        "H_hat_IV":[.23,.6],"identified":[True,False],"raw_pair_available":[True,True],"corrected_pair_available":[True,True]})
    status=render_empirical({"frames":{"realized_implied_match":frame}},tmp_path)
    assert (tmp_path/"figure6.png").exists() and (tmp_path/"figure7.png").exists()
    assert status["figure1"]["status"] == "requires_source_or_selection"
    moments=pd.DataFrame({"scenario_id":"synthetic_h0.5","estimator":"latent","lag":np.arange(1,11),"mean_moment":np.arange(1,11)})
    assert len(render_figure2(moments,tmp_path)) == 64



def test_csv_stable_id_leading_zero_is_not_lost(tmp_path):
    path=write_manifest(tmp_path,{"ids":pd.DataFrame({"instrument_id":["001","1"]})})
    assert FileManifest.load(path).read("ids").instrument_id.tolist() == ["001","1"]


def test_unread_sources_are_not_reported_as_verified(tmp_path):
    path=write_manifest(tmp_path,{"sessions":calendar_fixture(),"unused":pd.DataFrame({"x":[1]})})
    manifest=FileManifest.load(path)
    manifest.read("sessions")
    assert set(manifest.verified_hashes) == {"sessions"}


def test_explicit_conservative_aggregation_edge_policy():
    dates = pd.bdate_range("2024-01-01",periods=15)
    daily = pd.DataFrame({"instrument_id":"E","date":dates,"active":True,**{e:np.ones(15) for e in ESTIMATORS}})
    result=aggregate_variance(daily,frequency="weekly",missing_day_policy="invalidate",edge_period_policy="exclude")
    assert result.sample_boundary_period.tolist() == [True,False,True]
    assert np.isnan(result.tsrv.iloc[0]) and result.tsrv.iloc[1]==5 and np.isnan(result.tsrv.iloc[2])


@pytest.mark.parametrize("policy,n_days,base_eligible",[("keep",2,True),("exclude_day",1,True),("exclude_asset",2,False)])
def test_zero_volume_base_universe_policy_is_explicit(policy,n_days,base_eligible):
    calendar=calendar_fixture()
    bars=bars_fixture(calendar)
    bars.loc[:389,"volume"] = 0
    rules=universe_rules(); rules["zero_volume_policy"]=policy
    _,universe,_=equity_panel(bars,instrument_fixture(),calendar,dataset_config(),rules)
    assert universe.iloc[0].n_estimation_days == n_days
    assert bool(universe.iloc[0].base_eligible) == base_eligible


def test_futures_only_file_run_has_its_own_input_counts(tmp_path):
    calendar, contracts, volumes, bars = futures_fixture()
    path = write_manifest(tmp_path / "input",
        {"sessions":calendar,"futures_contracts":contracts,"futures_daily":volumes,"futures_bars":bars},
        datasets={"futures":dataset_config()}, selections={"futures_roots":["ES"]},
        analysis={"futures_roll_timing":"same_session","lag_windows":[[1,2]],"overlapping":[False]})
    result = run(path, tmp_path / "output")
    summary = result["frames"]["table1_panel_summary"]
    assert summary.loc[summary.panel == "futures", "n_input_bars"].iloc[0] == len(bars)
    assert "equities" in result["report"]["missing_sources"]
