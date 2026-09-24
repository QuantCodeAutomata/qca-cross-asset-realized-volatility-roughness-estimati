"""Definitions + prices -> parity -> IV -> ATM skew -> daily/pooled H.

Section 3.2 and equations (6)-(8) of arXiv:2608.16749v1. All tables, including
rejected and weakly identified observations, are retained. This module does
not fetch data. European parity on American marks is an explicit approximation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import option_pricing as pricing
from ..atm_skew_estimator import estimate_atm_skew
from ..implied_hurst import estimate_daily_implied_hurst, estimate_pooled_implied_hurst
from .common import atomic_json, write_frames, private_output


@dataclass(frozen=True)
class OptionConfig:
    moneyness_band: float = .08
    min_days: int = 14
    max_days: int = 365
    cme_min_strikes: int = 10
    opra_min_strikes: int = 20
    otm_only: bool = False
    identification_r2: float = .3
    min_maturities_per_date: int = 3
    require_cme_contract_mapping: bool = True
    # Definitions must already map these roots to the same stable underlying ID.
    merge_series_groups: tuple = (("SPX", "SPXW"),)

    def __post_init__(self):
        if not 0 < self.moneyness_band < 1 or not 1 <= self.min_days < self.max_days:
            raise ValueError("invalid moneyness or maturity window")
        if min(self.cme_min_strikes, self.opra_min_strikes) < 2:
            raise ValueError("at least two unique strikes are required")
        if self.min_maturities_per_date < 3 or not 0 <= self.identification_r2 <= 1:
            raise ValueError("invalid H-regression configuration")


def _require(frame, columns, name):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _dates(frame, columns):
    frame = frame.copy()
    for col in columns:
        parsed = pd.to_datetime(frame[col], errors="raise")
        if parsed.isna().any() or parsed.dt.tz is not None:
            raise ValueError(f"{col} must be explicit timezone-free calendar dates")
        if (parsed != parsed.dt.normalize()).any():
            raise ValueError(f"{col} must not contain intraday times")
        frame[col] = parsed
    return frame


def _consistent_scalar(group, field):
    values = group[field].drop_duplicates()
    if len(values) != 1 or pd.isna(values.iloc[0]):
        raise ValueError(f"inconsistent_{field}")
    return values.iloc[0]


def prices_to_implied_hurst(definitions: pd.DataFrame, prices: pd.DataFrame,
                            context: pd.DataFrame, config: OptionConfig | None = None) -> dict:
    """Process daily definitions and raw prices, never precomputed IVs.

    definitions: date, contract_id, underlying_id, root, venue (CME/OPRA),
        expiry, strike, option_type (call/put), exercise_style (european/american);
        CME also requires underlying_contract_id matching the context settlement.
    prices: date, contract_id, price, volume. Multiple quotes are deduplicated
        by highest volume; equal-volume conflicting prices are rejected.
    context: date, underlying_id, expiry; CME futures_settlement; American
        spot, dividend_yield, dividend_assumption (nonempty descriptive text).

    After IV inversion, call/put IVs are averaged *within each unique strike*,
    then equally weighted across strikes. Quote and strike counts are separate.
    OTM-only changes the IV/skew sample, not the call/put pairing for parity.
    """
    config = config or OptionConfig()
    _require(definitions, ["date", "contract_id", "underlying_id", "root", "venue", "expiry", "strike", "option_type", "exercise_style"], "definitions")
    _require(prices, ["date", "contract_id", "price", "volume"], "prices")
    _require(context, ["date", "underlying_id", "expiry"], "context")
    defs, raw, ctx = _dates(definitions, ["date", "expiry"]), _dates(prices, ["date"]), _dates(context, ["date", "expiry"])
    for col in ("contract_id", "underlying_id", "root", "venue", "option_type", "exercise_style"):
        if defs[col].isna().any():
            raise ValueError(f"definitions contain missing {col}")
    if not defs.venue.isin(["CME", "OPRA"]).all() or not defs.option_type.isin(["call", "put"]).all():
        raise ValueError("unsupported venue or option_type in definitions")
    if not defs.exercise_style.isin(["european", "american"]).all():
        raise ValueError("unsupported exercise_style")
    defs["strike"] = pd.to_numeric(defs.strike, errors="coerce")
    defs = defs.drop_duplicates()
    if defs.duplicated(["date", "contract_id"]).any():
        raise ValueError("conflicting daily contract definitions")
    if ctx.duplicated(["date", "underlying_id", "expiry"]).any():
        raise ValueError("conflicting option context rows")
    raw = raw.reset_index(drop=True)
    raw["input_quote_id"] = np.arange(len(raw))
    raw["price"] = pd.to_numeric(raw.price, errors="coerce")
    raw["volume"] = pd.to_numeric(raw.volume, errors="coerce")
    rejection, selections, parity_rows, iv_rows, skew_rows = [], [], [], [], []

    def reject(frame, stage, reason, **extra):
        for row in frame.to_dict("records"):
            rejection.append({"input_quote_id": row.get("input_quote_id"),
                "date": row.get("date"), "underlying_id": row.get("underlying_id"),
                "contract_id": row.get("contract_id"), "expiry": row.get("expiry"),
                "stage": stage, "reason": reason, **extra})

    clean = []
    for _, group in raw.groupby(["date", "contract_id"], sort=True, dropna=False):
        # Identical repeated quotes cannot inflate volume or strike counts.
        unique = group.drop_duplicates(["price", "volume"], keep="first")
        reject(group.loc[~group.index.isin(unique.index)], "quotes", "duplicate_quote")
        top = unique.loc[unique.volume.fillna(-np.inf) == unique.volume.fillna(-np.inf).max()]
        if len(top) > 1:
            reject(unique, "quotes", "ambiguous_equal_volume_quotes")
            continue
        selected = top.iloc[[0]]
        reject(unique.loc[~unique.index.isin(selected.index)], "quotes", "lower_volume_duplicate_quote")
        clean.append(selected)
    clean = pd.concat(clean, ignore_index=True) if clean else raw.iloc[:0]
    joined = clean.merge(defs, on=["date", "contract_id"], how="left", validate="one_to_one", indicator=True)
    reject(joined[joined._merge != "both"], "definitions", "missing_definition")
    joined = joined[joined._merge == "both"].drop(columns="_merge")
    # We retain all declared underlyings even when every quote is rejected.
    underlyings = sorted(defs.underlying_id.unique(), key=str)
    keys = ["underlying_id", "date", "expiry"]
    for key, all_group in joined.groupby(keys, sort=True):
        underlying, date, expiry = key
        base = dict(zip(keys, key))
        days = int((expiry - date).days)
        base.update(T=days / 365.0, maturity_days=days)
        slice_row = {**base, "valid": False, "psi": np.nan, "psi_se": np.nan,
                     "r_squared": np.nan, "n_strikes": 0, "n_quotes": 0,
                     "n_input_quotes": len(all_group)}
        group = all_group.copy()
        try:
            venue = _consistent_scalar(group, "venue")
            style = _consistent_scalar(group, "exercise_style")
        except ValueError as exc:
            reject(group, "slice", str(exc)); slice_row["status"] = str(exc)
            skew_rows.append(slice_row); continue
        slice_row.update(venue=venue, exercise_style=style)
        if not config.min_days <= days <= config.max_days:
            reject(group, "maturity", "outside_maturity_window")
            skew_rows.append({**slice_row, "status": "outside_maturity_window"}); continue
        numeric = np.isfinite(group.price) & (group.price >= 0) & np.isfinite(group.strike) & (group.strike > 0)
        reject(group[~numeric], "quotes", "invalid_price_or_strike")
        group = group[numeric]
        roots = set(group.root)
        if len(roots) > 1:
            if not any(roots.issubset(set(rule)) for rule in config.merge_series_groups):
                reject(group, "series", "multiple_series_requires_merge_rule")
                skew_rows.append({**slice_row, "status": "multiple_series_requires_merge_rule"}); continue
            totals = group.groupby("root").volume.sum(min_count=1).fillna(-np.inf)
            winner = sorted(totals.index, key=lambda root: (-totals[root], str(root)))[0]
            selections.append({**base, "selected_root": winner, "volume_by_root": totals.to_dict(),
                               "tie_break": "lexicographic_root", "reason": "largest_total_series_volume"})
            reject(group[group.root != winner], "series", "lower_volume_expiry_series")
            group = group[group.root == winner]
        # Same-strike multiple contract IDs must not create many-to-many pairing.
        dup = group.duplicated(["strike", "option_type"], keep=False)
        reject(group[dup], "pairing", "duplicate_contract_at_strike_type")
        group = group[~dup]
        calls = group[group.option_type == "call"]
        puts = group[group.option_type == "put"]
        paired = calls[["strike", "price"]].merge(puts[["strike", "price"]], on="strike", suffixes=("_call", "_put"), validate="one_to_one")
        pair_mask = group.strike.isin(paired.strike)
        # Parity uses matched prices. Unpaired valid quotes can still enter IV
        # once F,D are identified; the paper's positive-volume rule is applied
        # to OPRA skew quotes, not to both legs of the parity sample.
        slice_row.update(n_pairs=len(paired), n_paired_quotes=int(pair_mask.sum()),
                         n_unpaired_quotes=int((~pair_mask).sum()))
        context_row = ctx[(ctx.underlying_id == underlying) & (ctx.date == date) & (ctx.expiry == expiry)]
        info = context_row.iloc[0].to_dict() if len(context_row) else {}
        model = "BAW" if venue == "OPRA" and style == "american" else "Black76"
        approximation = ("European_parity_on_American_marks" if style == "american" else "European_parity")
        slice_row.update(pricing_model=model, parity_assumption=approximation)
        try:
            if venue == "CME" and ("futures_settlement" not in info or pd.isna(info["futures_settlement"])):
                raise pricing.PricingError("missing_futures_settlement")
            if venue == "CME" and config.require_cme_contract_mapping:
                if "underlying_contract_id" not in group or "underlying_contract_id" not in info:
                    raise pricing.PricingError("missing_CME_underlying_contract_mapping")
                contract_ids = group.underlying_contract_id.dropna().unique()
                if len(contract_ids) != 1 or group.underlying_contract_id.isna().any() or contract_ids[0] != info["underlying_contract_id"]:
                    raise pricing.PricingError("CME_underlying_contract_mapping_mismatch")
            if model == "BAW":
                if not all(field in info and pd.notna(info[field]) for field in ("spot", "dividend_yield", "dividend_assumption")):
                    raise pricing.PricingError("American_requires_spot_and_explicit_dividend_assumption")
                if not str(info["dividend_assumption"]).strip():
                    raise pricing.PricingError("American_requires_spot_and_explicit_dividend_assumption")
                info["spot"], info["dividend_yield"] = float(info["spot"]), float(info["dividend_yield"])
                if not np.isfinite(info["spot"]) or info["spot"] <= 0 or not np.isfinite(info["dividend_yield"]):
                    raise pricing.PricingError("invalid_American_spot_or_dividend_yield")
            parity = pricing.fit_put_call_parity(paired.price_call, paired.price_put, paired.strike,
                       known_forward=float(info["futures_settlement"]) if venue == "CME" else None)
            rate_raw = -np.log(parity["discount"]) / base["T"]
            # Joint parity on exactly zero-rate prices can return 1 +/- a few
            # floating-point ulps. Keep D and the raw rate unchanged in evidence;
            # only select the analytic r=0 BAW limit within eight ulps of D=1.
            # This is not support for economically negative BAW rates.
            zero_rate_roundoff = bool(model == "BAW" and abs(parity["discount"]-1.) <= 8*np.finfo(float).eps)
            rate = 0. if zero_rate_roundoff else rate_raw
            parity_rows.append({**base, **parity, "rate": rate, "raw_parity_rate": rate_raw,
                                "BAW_zero_rate_roundoff": zero_rate_roundoff, "status": "ok", "pricing_model": model,
                                "parity_assumption": approximation,
                                "underlying_contract_id": info.get("underlying_contract_id"),
                                "CME_contract_mapping_required": bool(venue=="CME" and config.require_cme_contract_mapping)})
        except (pricing.PricingError, ValueError, TypeError) as exc:
            reject(group, "parity_or_context", str(exc), pricing_model=model)
            parity_rows.append({**base, "status": str(exc), "pricing_model": model})
            skew_rows.append({**slice_row, "status": str(exc)}); continue
        slice_row.update(forward=parity["forward"], discount=parity["discount"], rate=rate,
                         parity_r_squared=parity["r_squared"])
        iv_slice = []
        for _, row in group.iterrows():
            if venue == "OPRA" and (not np.isfinite(row.volume) or row.volume <= 0):
                reject(row.to_frame().T, "volume", "OPRA_requires_positive_volume"); continue
            k = float(np.log(row.strike / parity["forward"]))
            if abs(k) > config.moneyness_band + 1e-12:
                reject(row.to_frame().T, "moneyness", "outside_moneyness_band"); continue
            if config.otm_only and ((row.option_type == "call" and k < -1e-12) or (row.option_type == "put" and k > 1e-12)):
                reject(row.to_frame().T, "OTM", "in_the_money_quote"); continue
            try:
                if model == "Black76":
                    iv = pricing.black76_implied_vol(float(row.price), parity["forward"], float(row.strike), base["T"], rate, row.option_type, raise_errors=True)
                else:
                    iv = pricing.baw_implied_vol(float(row.price), float(info["spot"]), float(row.strike), base["T"], rate,
                        float(info["dividend_yield"]), row.option_type, raise_errors=True)
                item = {**base, "contract_id": row.contract_id, "input_quote_id": int(row.input_quote_id),
                        "root": row.root, "strike": float(row.strike), "option_type": row.option_type,
                        "price": float(row.price), "volume": float(row.volume), "iv": iv,
                        "paired_for_parity": bool(row.strike in set(paired.strike)),
                        "log_moneyness": k, "pricing_model": model,
                        "forward": parity["forward"], "discount": parity["discount"],
                        "spot": info.get("spot"), "dividend_yield": info.get("dividend_yield"),
                        "dividend_assumption": info.get("dividend_assumption")}
                iv_rows.append(item); iv_slice.append(item)
            except pricing.PricingError as exc:
                reject(row.to_frame().T, "IV", str(exc), pricing_model=model)
        minimum = config.cme_min_strikes if venue == "CME" else config.opra_min_strikes
        slice_row.update(n_quotes=len(iv_slice), minimum_unique_strikes=minimum)
        if not iv_slice:
            skew_rows.append({**slice_row, "status": "no_valid_IV"}); continue
        iv_frame = pd.DataFrame(iv_slice)
        unique_strikes = iv_frame.groupby("strike", sort=True).iv.mean().reset_index()
        slice_row["n_strikes"] = len(unique_strikes)
        fit = estimate_atm_skew(unique_strikes.strike.to_numpy(), unique_strikes.iv.to_numpy(),
                               parity["forward"], moneyness_band=config.moneyness_band + 1e-12,
                               min_strikes=minimum)
        if fit is None:
            skew_rows.append({**slice_row, "status": "insufficient_unique_strikes"})
        else:
            skew_rows.append({**slice_row, **fit, "n_quotes": len(iv_slice), "status": "ok"})
    skews = pd.DataFrame(skew_rows)
    if skews.empty:
        skews = pd.DataFrame(columns=keys + ["T", "psi", "valid", "status"])
        skews = skews.astype({"T":float,"psi":float,"valid":bool})
    daily, pooled = [], []
    for underlying in underlyings:
        subset = skews[skews.underlying_id == underlying].copy()
        # Declared dates remain visible even when all quotes for the date fail.
        dates = sorted(defs.loc[defs.underlying_id == underlying, "date"].unique())
        for date in dates:
            selected = subset[subset.date == date]
            count = int(selected.loc[selected.valid.astype(bool), "T"].nunique())
            result = estimate_daily_implied_hurst(subset, date, (config.min_days/365, config.max_days/365)) if count >= config.min_maturities_per_date else None
            daily.append({"underlying_id": underlying, "date": date,
                **(result if result is not None else {"H_hat": np.nan, "H_se": np.nan, "beta": np.nan, "beta_se": np.nan, "r_squared": np.nan, "n_obs": count}),
                "status": "ok" if result else "insufficient_maturities",
                "identified": bool(result is not None and result["r_squared"] >= config.identification_r2)})
        result = estimate_pooled_implied_hurst(subset, (config.min_days/365, config.max_days/365),
                                            min_obs_per_date=config.min_maturities_per_date)
        result["identified"] = bool(np.isfinite(result["r_psi_squared"]) and result["r_psi_squared"] >= config.identification_r2)
        values = [r["H_hat"] for r in daily if r["underlying_id"] == underlying and np.isfinite(r["H_hat"])]
        pooled.append({"underlying_id": underlying, **result,
            "median_daily_H": float(np.median(values)) if values else np.nan,
            "n_valid_daily_H": len(values), "n_declared_dates": len(dates),
            "status": "identified" if result["identified"] else ("weak_identification" if np.isfinite(result["H_hat_IV"]) else "insufficient_data")})
    return {"config": asdict(config), "counts": {"input_quotes": len(raw), "input_definitions": len(defs),
             "IV_quotes": len(iv_rows), "rejected_quotes": len(rejection), "expiry_slices": len(skews)},
            "iv_quotes": pd.DataFrame(iv_rows), "parity": pd.DataFrame(parity_rows), "skews": skews,
            "daily_h": pd.DataFrame(daily), "pooled_h": pd.DataFrame(pooled),
            "rejections": pd.DataFrame(rejection), "series_selection": pd.DataFrame(selections)}


def write_option_results(result: dict, output_dir: str | Path):
    output = private_output(output_dir)
    hashes = write_frames(output, {key: value for key, value in result.items() if isinstance(value, pd.DataFrame)})
    atomic_json(output / "option_run.json", {"config": result["config"], "counts": result["counts"],
                                           "output_hashes": hashes, "empirical_replication_claimed": False})
    return hashes
