"""File-only empirical scenarios with explicit coverage and publication controls.

Outputs are private. Counts in the publication are reference comparisons, never
universe-size constraints. No source membership list is inferred from tickers.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..hurst_realized import compute_second_moment_scaling
from ..autocovariance_fit import fit_two_estimator_correction
from .common import atomic_json, private_output, write_frames
from .manifest import FileManifest, MissingSource, ManifestError
from .options import OptionConfig, prices_to_implied_hurst, write_option_results
from .panels import ESTIMATORS, equity_panel, futures_panel, aggregate_variance

REFERENCES = {"equities":3926, "quality_equities":1409, "futures_roots":34,
              "second_frequency_assets":40, "option_underlyings":44, "matched_assets":41}
COVERAGE = {
    "table1":(["equities","futures","seconds","options"], "source composition and dates must be audited"),
    "table2":(["table2"], "14 printed cells disagree with independently checked integrals"),
    "table3":(["mc3"], "author calibration convention is unresolved"),
    "table4":(["mc4"], "author calibration and PAV normalization are unresolved"),
    "table5":(["mc5"], "report invalid paper-mode corrections; do not clip moments"),
    "table6":(["mc6"], "author calibration convention is unresolved"),
    "table7":(["futures"], "explicit roll timing and composition; counts are references"),
    "figure1":(["figure1"], "overlap convention and prose/plot benchmark numbers differ"),
    "figure2":(["mc_figure2"], "Gaussian convention is not the verified author simulation"),
    "figure3":(["equities"], "Sections 3.1/3.2 base-universe rules conflict"),
    "figure4":(["options"], "daily estimates and rolling medians preserve weak identification"),
    "figure5":(["futures"], "raw TSRV versus corrected RK; do not conflate estimators"),
    "figure6":(["comparison_raw"], "render raw TSRV on x; paper caption/axis conflicts with prose"),
    "figure7":(["comparison_corrected"], "render corrected RK on x, not corrected TSRV"),
}


def fit_window(log_variance, low, high, *, overlapping):
    if not 1 <= low < high:
        raise ValueError("require an inclusive lag window with at least two lags")
    x = np.asarray(log_variance, dtype=float)
    deltas, moments = compute_second_moment_scaling(x, high, overlapping=overlapping)
    valid = (deltas >= low) & np.isfinite(moments) & (moments > 0)
    result = {"H_hat":np.nan,"r_squared":np.nan,"slope":np.nan,"intercept":np.nan,
              "n_lags":int(valid.sum()),"n_calendar_rows":len(x),"n_estimation_days":int(np.isfinite(x).sum())}
    if valid.sum() >= 2:
        fit = sm.OLS(np.log(moments[valid]), sm.add_constant(np.log(deltas[valid]))).fit()
        result.update(H_hat=float(fit.params[1]/2), r_squared=float(fit.rsquared),
                      slope=float(fit.params[1]), intercept=float(fit.params[0]))
    return result, pd.DataFrame({"lag":deltas,"m2":moments})


def estimate_daily_panel(daily, *, windows=((1,10),(1,40),(40,250)), overlaps=(False,True),
                         correction_specification="paper", h_bounds=(1e-6,.49), frequency="daily"):
    estimates, moments, corrections = [], [], []
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for instrument_id, group in daily.groupby("instrument_id", sort=True):
        group = group.sort_values("date")
        if group.duplicated("date").any():
            raise ManifestError("duplicate instrument/date in realized panel")
        logs = {}
        for estimator in ESTIMATORS:
            values = group[estimator].to_numpy(dtype=float)
            mask = np.isfinite(values) & (values > 0)
            logs[estimator] = np.full(len(values), np.nan)
            logs[estimator][mask] = np.log(values[mask])
            for overlapping in overlaps:
                max_lag = max(high for _,high in windows)
                _, curve = fit_window(logs[estimator], 1, max_lag, overlapping=overlapping)
                curve = curve.assign(instrument_id=instrument_id,estimator=estimator,
                                     overlapping=overlapping,frequency=frequency)
                moments.append(curve)
                for low,high in windows:
                    fit, _ = fit_window(logs[estimator], low, high, overlapping=overlapping)
                    estimates.append({"instrument_id":instrument_id,"estimator":estimator,
                        "lag_min":low,"lag_max":high,"overlapping":overlapping,"frequency":frequency,**fit})
        if frequency == "daily":
            for overlapping in overlaps:
                correction = fit_two_estimator_correction(logs["rv5m"], logs["rk"],
                    specification=correction_specification, overlapping=overlapping,
                    h_bounds=tuple(h_bounds))
                corrections.append({"instrument_id":instrument_id,"frequency":frequency,**correction})
    return pd.DataFrame(estimates), pd.concat(moments, ignore_index=True), pd.DataFrame(corrections)


def distribution_summary(estimates, universe):
    if estimates.empty or universe.empty:
        return pd.DataFrame()
    combined = estimates.merge(universe, on="instrument_id", validate="many_to_one")
    rows = []
    for scope, flag in (("all_base","base_eligible"),("quality","quality_eligible")):
        eligible = combined[combined[flag]]
        for key, group in eligible.groupby(["estimator","frequency","lag_min","lag_max","overlapping"]):
            values = group.H_hat[np.isfinite(group.H_hat)]
            rows.append(dict(zip(["estimator","frequency","lag_min","lag_max","overlapping"],key),
                scope=scope,n_assets=len(group),n_valid=len(values),median_H=float(values.median()) if len(values) else np.nan,
                q25=float(values.quantile(.25)) if len(values) else np.nan,q75=float(values.quantile(.75)) if len(values) else np.nan,
                mean_H=float(values.mean()) if len(values) else np.nan,median_r_squared=float(group.r_squared.dropna().median()) if group.r_squared.notna().any() else np.nan))
    return pd.DataFrame(rows)


def make_coverage(executed, data_kind, missing):
    rows = []
    for item, (requirements,note) in COVERAGE.items():
        completed = all(executed.get(name,False) for name in requirements)
        required = [name for name in requirements if not executed.get(name,False)]
        rows.append({"item":item,"implemented":True,"checked_on_this_fixture":completed and data_kind=="synthetic_fixture",
            "executed_on_supplied_market_data":completed and data_kind=="market",
            "requires_source_or_separate_run":required,"requires_methodology_clarification":note,
            "status":"checked_on_fixture" if completed and data_kind=="synthetic_fixture" else
                     "executed_on_supplied_data" if completed else "requires_source_or_separate_run",
            "full_empirical_replication":False})
    return rows


def run(manifest_path, output_dir):
    manifest = FileManifest.load(manifest_path)
    output = private_output(output_dir)
    analysis = manifest.content.get("analysis", {})
    windows = analysis.get("lag_windows", [[1,10],[1,40],[40,250]])
    overlaps = analysis.get("overlapping", [False, True])
    if not overlaps or any(type(x) is not bool for x in overlaps) or len(set(overlaps)) != len(overlaps):
        raise ManifestError("overlapping must list distinct boolean specifications")
    if not windows or any(len(w)!=2 or not 1 <= w[0] < w[1] for w in windows):
        raise ManifestError("invalid lag windows")
    spec = analysis.get("correction_specification", "paper")
    bounds = analysis.get("fitted_H_bounds", [1e-6,.49])
    primary_overlap = analysis.get("primary_overlapping", False)
    if primary_overlap not in overlaps:
        raise ManifestError("primary_overlapping must be included in specifications")
    frames, missing, executed = {}, {}, {}
    panel_coverage = []
    daily_parts, estimate_parts, correction_parts = [], [], []
    eligibility_parts = []
    counts = dict.fromkeys(REFERENCES, 0)
    comparison_counts = {"raw": 0, "corrected": 0}
    sessions = None
    try:
        sessions = manifest.read("sessions")
    except MissingSource as exc:
        missing["sessions"] = str(exc)
    for kind in ("equities", "seconds", "futures"):
        try:
            if sessions is None:
                raise MissingSource("sessions are required")
            dataset_name = {"equities":"equity", "seconds":"seconds", "futures":"futures"}[kind]
            dataset = manifest.dataset(dataset_name)
            if kind in ("equities", "seconds"):
                selected = manifest.selection("seconds") if kind=="seconds" else None
                bars = manifest.read("second_bars" if kind=="seconds" else "equity_bars")
                daily, universe, excluded = equity_panel(bars, manifest.read("instruments"), sessions,
                              dataset, manifest.base_universe(), selected_ids=selected)
                frames[kind+"_universe"] = universe
                counts["second_frequency_assets" if kind=="seconds" else "equities"] = int(universe.base_eligible.sum())
                if kind=="equities":
                    counts["quality_equities"] = int(universe.quality_eligible.sum())
            else:
                roots = manifest.selection("futures_roots")
                if "futures_roll_timing" not in analysis:
                    raise ManifestError("explicit futures_roll_timing is required")
                bars = manifest.read("futures_bars")
                daily, schedule, excluded = futures_panel(bars, manifest.read("futures_contracts"),
                    manifest.read("futures_daily"), sessions, dataset, roots, roll_timing=analysis["futures_roll_timing"])
                frames["futures_roll_schedule"] = schedule
                counts["futures_roots"] = int(daily.loc[daily.eligible,"instrument_id"].nunique())
            panel_coverage.append({"panel":kind, "source":manifest.content["source"],
                "snapshot":manifest.content["snapshot"], "n_input_bars":len(bars),
                "n_defined_assets":int(daily.instrument_id.nunique()),
                "n_eligible_asset_days":int(daily.eligible.sum()),
                "calendar_start":str(daily.date.min()), "calendar_end":str(daily.date.max()),
                "bar_seconds":dataset["bar_seconds"], "timestamp_convention":dataset["timestamp_convention"],
                "return_convention":dataset["return_convention"], "data_kind":manifest.content["data_kind"]})
            frames[kind+"_daily_RV"] = daily
            frames[kind+"_exclusions"] = excluded
            estimates, moments, corrections = estimate_daily_panel(daily, windows=windows, overlaps=overlaps,
                                          correction_specification=spec, h_bounds=bounds)
            frames[kind+"_H"] = estimates
            frames[kind+"_moments"] = moments
            frames[kind+"_corrections"] = corrections
            if kind in ("equities", "seconds"):
                frames[kind+"_distribution"] = distribution_summary(estimates, universe)
            else:
                classes = daily[["instrument_id","asset_class"]].drop_duplicates()
                frame = estimates.merge(classes, on="instrument_id", validate="many_to_one")
                frames["class_medians"] = frame.groupby(["asset_class","estimator","frequency","lag_min","lag_max","overlapping"], as_index=False).agg(
                    median_H=("H_hat","median"),n_valid=("H_hat","count"),median_r_squared=("r_squared","median"))
            # Only minute equity/futures series enter the cross-asset primary match.
            if kind != "seconds":
                daily_parts.append(daily.assign(panel=kind))
                estimate_parts.append(estimates.assign(panel=kind))
                correction_parts.append(corrections.assign(panel=kind))
                if kind == "equities":
                    eligibility = universe[["instrument_id", "base_eligible", "base_exclusion_reasons"]]
                else:
                    valid_days = daily.eligible & np.isfinite(daily.tsrv) & (daily.tsrv > 0)
                    eligibility = daily.assign(base_eligible=valid_days).groupby(
                        "instrument_id", as_index=False).base_eligible.any()
                    eligibility["base_exclusion_reasons"] = eligibility.base_eligible.map(
                        lambda ok: [] if ok else ["no_eligible_estimation_days"])
                eligibility_parts.append(eligibility.assign(panel=kind))
            if kind in ("equities", "seconds") and "aggregation" in analysis:
                aggregation = analysis["aggregation"]
                if not {"missing_day_policy","edge_period_policy"}.issubset(aggregation):
                    raise ManifestError("weekly/monthly aggregation requires explicit missing and edge policies")
                for frequency in ("weekly", "monthly"):
                    aggregated = aggregate_variance(daily, frequency=frequency, **aggregation)
                    frames[f"{kind}_{frequency}_RV"] = aggregated
                    h, m, _ = estimate_daily_panel(aggregated, windows=windows, overlaps=overlaps, frequency=frequency)
                    frames[f"{kind}_{frequency}_H"] = h
                    frames[f"{kind}_{frequency}_distribution"] = distribution_summary(h, universe)
            executed[kind] = bool(not estimates.empty and np.isfinite(estimates.H_hat).any())
        except MissingSource as exc:
            missing[kind] = str(exc); executed[kind] = False
    options_result = None
    try:
        selected = manifest.selection("option_underlyings")
        definitions = manifest.read("option_definitions")
        if not set(selected).issubset(set(definitions.underlying_id)):
            raise ManifestError("selected option underlyings absent from definitions")
        definitions = definitions[definitions.underlying_id.isin(selected)]
        prices = manifest.read("option_prices")
        prices = prices[prices.contract_id.isin(set(definitions.contract_id))]
        # Calendar dates are validated and normalized jointly inside the option adapter,
        # so mixed CSV strings / Parquet datetime dtypes do not break this selection.
        context = manifest.read("option_context")
        config = OptionConfig(**analysis.get("options", {}))
        options_result = prices_to_implied_hurst(definitions, prices, context, config)
        write_option_results(options_result, output / "options" / "primary")
        frames["option_pooled_H"] = options_result["pooled_h"]
        frames["option_daily_H"] = options_result["daily_h"]
        frames["option_daily_H"]["rolling_median_21_observation_days"] = frames["option_daily_H"].groupby("underlying_id")["H_hat"].transform(lambda x:x.rolling(21,min_periods=1).median())
        panel_coverage.append({"panel":"options", "source":manifest.content["source"],
            "snapshot":manifest.content["snapshot"], "n_input_quotes":len(prices),
            "n_defined_assets":len(selected), "calendar_start":str(definitions.date.min()),
            "calendar_end":str(definitions.date.max()), "data_kind":manifest.content["data_kind"]})
        counts["option_underlyings"] = int(options_result["pooled_h"].H_hat_IV.notna().sum())
        executed["options"] = bool(counts["option_underlyings"])
        sensitivity = []
        for index, alternative in enumerate(analysis.get("option_sensitivities", [])):
            alternate_config = OptionConfig(**{**asdict(config), **alternative})
            alternate = prices_to_implied_hurst(definitions, prices, context, alternate_config)
            write_option_results(alternate, output / "options" / f"sensitivity_{index:02d}")
            sensitivity.append(alternate["pooled_h"].assign(specification_id=index,configuration=json.dumps(asdict(alternate_config))))
        if sensitivity:
            frames["option_sensitivity_H"] = pd.concat(sensitivity,ignore_index=True)
    except MissingSource as exc:
        missing["options"] = str(exc); executed["options"] = False
    if estimate_parts:
        combined = pd.concat(estimate_parts,ignore_index=True)
        corrections = pd.concat(correction_parts,ignore_index=True)
        try:
            figure_ids = manifest.selection("figure1_ids")
            primary_moments = pd.concat([v for k,v in frames.items() if k in ("equities_moments","futures_moments")],ignore_index=True)
            if not set(figure_ids).issubset(set(primary_moments.instrument_id)):
                raise MissingSource("Figure 1 IDs missing from realized panels")
            frames["figure1_moments"] = primary_moments[primary_moments.instrument_id.isin(figure_ids) & (primary_moments.estimator=="tsrv")]
            frames["figure1_fits"] = combined[combined.instrument_id.isin(figure_ids) & (combined.estimator=="tsrv")]
            executed["figure1"] = not frames["figure1_fits"].empty
        except MissingSource as exc:
            missing["figure1"] = str(exc)
        try:
            mapping = manifest.selection("comparison_map")
            if options_result is None:
                raise MissingSource("option estimates required for realized/implied match")
            mapped = pd.DataFrame(mapping)
            if not {"realized_id","option_id"}.issubset(mapped) or mapped.duplicated(["realized_id","option_id"]).any():
                raise ManifestError("comparison_map requires distinct realized_id/option_id pairs")
            raw = combined[(combined.estimator=="tsrv") & (combined.lag_min==1) & (combined.lag_max==10) & (combined.overlapping==primary_overlap)]
            corr = corrections[corrections.overlapping==primary_overlap]
            if raw.duplicated("instrument_id").any() or corr.duplicated("instrument_id").any():
                raise ManifestError("realized IDs collide between equity/futures panels; use namespaced stable IDs")
            matched = mapped.merge(raw[["instrument_id","H_hat","r_squared"]].rename(columns={"instrument_id":"realized_id","H_hat":"H_realized_TSRV"}),on="realized_id",how="left",validate="many_to_one")
            matched = matched.merge(corr[["instrument_id","H_corrected_b","optimizer_success","correction_valid_10"]].rename(columns={"instrument_id":"realized_id","H_corrected_b":"H_realized_RK_corrected"}),on="realized_id",how="left",validate="many_to_one")
            matched = matched.merge(options_result["pooled_h"].rename(columns={"underlying_id":"option_id"}),on="option_id",how="left",validate="many_to_one")
            eligibility = pd.concat(eligibility_parts, ignore_index=True).rename(columns={"instrument_id":"realized_id"})
            matched = matched.merge(eligibility, on="realized_id", how="left", validate="many_to_one")
            matched["base_eligible"] = matched.base_eligible.eq(True)
            matched["base_exclusion_reasons"] = matched.base_exclusion_reasons.map(
                lambda reasons: reasons if isinstance(reasons, list) else ["missing_realized_universe"])
            matched["raw_pair_available"] = matched.base_eligible & np.isfinite(matched.H_realized_TSRV) & np.isfinite(matched.H_hat_IV)
            matched["corrected_pair_available"] = matched.base_eligible & np.isfinite(matched.H_realized_RK_corrected) & np.isfinite(matched.H_hat_IV) & matched.optimizer_success.eq(True) & matched.correction_valid_10.eq(True)
            frames["realized_implied_match"] = matched  # Weak identification is never a join filter.
            counts["matched_assets"] = int(matched.raw_pair_available.sum())
            comparison_counts = {"raw": int(matched.raw_pair_available.sum()),
                                 "corrected": int(matched.corrected_pair_available.sum())}
            executed["comparison_raw"] = bool(comparison_counts["raw"])
            executed["comparison_corrected"] = bool(comparison_counts["corrected"])
        except MissingSource as exc:
            missing["comparison_raw"] = missing["comparison_corrected"] = str(exc)
    frames["table1_panel_summary"] = pd.DataFrame(panel_coverage)
    if "equities_H" in frames and "seconds_H" in frames:
        keys = ["instrument_id","estimator","frequency","lag_min","lag_max","overlapping"]
        comparison = frames["seconds_H"][keys+["H_hat","r_squared"]].merge(
            frames["equities_H"][keys+["H_hat","r_squared"]], on=keys, how="left", suffixes=("_seconds","_minutes"), validate="one_to_one")
        comparison["difference_H"] = comparison.H_hat_seconds - comparison.H_hat_minutes
        frames["sampling_frequency_comparison"] = comparison
    coverage = make_coverage(executed, manifest.content["data_kind"], missing)
    counts_comparison = {key:{"observed":value,"publication_reference":REFERENCES[key],"difference":value-REFERENCES[key],"forced_target":False} for key,value in counts.items()}
    hashes = write_frames(output, frames)
    resolved = {**analysis, "lag_windows":windows, "overlapping":overlaps,
                "primary_overlapping":primary_overlap, "correction_specification":spec,
                "fitted_H_bounds":bounds, "log_scale":"log_daily_unannualized_variance",
                "nonoverlap_phase_origin":"first session of the supplied calendar; inactive dates remain NaN"}
    report = {"manifest_sha256":manifest.file_sha256, "manifest_content_fingerprint":manifest.fingerprint,
              "data_kind":manifest.content["data_kind"],
              "source_snapshot_hashes":manifest.verified_hashes,
              "counts":counts_comparison,"comparison_counts":comparison_counts,
              "coverage":coverage,"missing_sources":missing,
              "analysis_configuration":resolved,"output_hashes":hashes,"full_empirical_replication":False}
    atomic_json(output / "empirical_run.json", report)
    return {"manifest":manifest,"frames":frames,"report":report}


def export_approved_aggregates(result, path):
    """Explicit publication whitelist. Never serialize manifests or time series."""
    permitted = result["manifest"].content.get("publication", {}).get("approved_aggregates", [])
    available = {"counts":result["report"]["counts"],
                 "manifest_sha256":result["report"]["manifest_sha256"],
                 "source_snapshot_hashes":result["report"]["source_snapshot_hashes"]}
    for key in ("equities_distribution", "class_medians"):
        if key in result["frames"]:
            available[key] = result["frames"][key].to_dict("records")
    if not permitted or not set(permitted).issubset(available):
        raise ManifestError("public output requires an explicit approved aggregate whitelist")
    atomic_json(path, {key:available[key] for key in permitted})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approved-public-json", type=Path)
    args = parser.parse_args()
    result = run(args.manifest, args.output_dir)
    if args.approved_public_json:
        export_approved_aggregates(result, args.approved_public_json)
    print(json.dumps({"counts":result["report"]["counts"], "missing_sources":result["report"]["missing_sources"],
                      "full_empirical_replication":False}, indent=2))


if __name__ == "__main__":
    main()
