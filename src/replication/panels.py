"""Calendar-preserving equity/futures file adapters and physical-frequency RV.

These functions accept private tabular sources; they never query providers.
Each RV is computed inside a single session and a single futures contract.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..rv_estimators import compute_all_estimators
from .manifest import ManifestError, calendar_sessions, parse_timestamps

ESTIMATORS = ("rv5m", "rk", "tsrv", "pav", "pabpv", "bpv", "ctrv")


def physical_realized_measures(closes, *, bar_seconds: int, return_convention="close_only", open_price=None):
    """Five minutes means 300 seconds. Preserve the session-grid sampling phase.

    Leading NaNs are not backfilled. Forward filling belongs to session building.
    RV5m uses endpoints on the original grid; TSRV uses all K staggered grids.
    Other estimators use the contiguous finite suffix of adjacent log returns.
    """
    p = np.asarray(closes, dtype=float)
    if p.ndim != 1 or bar_seconds <= 0 or 300 % bar_seconds:
        raise ValueError("invalid price grid or physical bar interval")
    if np.any(np.isfinite(p) & (p <= 0)):
        raise ValueError("nonpositive_price")
    if return_convention == "open_seeded":
        start = np.nan if open_price is None else float(open_price)
        if np.isfinite(start) and start <= 0:
            raise ValueError("nonpositive_open_price")
        p = np.r_[start, p]
    elif return_convention != "close_only":
        raise ValueError("unknown return convention")
    with np.errstate(invalid="ignore"):
        lp = np.log(p)
    r = np.diff(lp)
    finite = np.isfinite(r)
    positions = np.flatnonzero(finite)
    if positions.size and np.any(np.diff(positions) != 1):
        raise ValueError("internal_missing_returns_must_not_be_compressed")
    values = compute_all_estimators(r[finite])
    step = 300 // bar_seconds
    coarse = np.diff(lp[::step])
    coarse = coarse[np.isfinite(coarse)]
    values["rv5m"] = float(coarse @ coarse)
    slow, counts = [], []
    for offset in range(step):
        diffs = np.diff(lp[offset::step])
        diffs = diffs[np.isfinite(diffs)]
        # Include empty phase grids so short/partial grids are not overweighted.
        slow.append(float(diffs @ diffs)); counts.append(len(diffs))
    if finite.sum():
        values["tsrv"] = max(float(np.mean(slow) - np.mean(counts)/finite.sum() * np.sum(r[finite]**2)), 0.)
    else:
        values["tsrv"] = 0.
    return values, {"n_close_prices": len(closes), "n_price_nodes": len(p),
                    "n_returns": len(r), "n_finite_returns": int(finite.sum()),
                    "five_minute_step": step, "rv5m_increments": len(coarse),
                    "bar_seconds": bar_seconds, "return_convention": return_convention}


def _session_grid(session, config):
    step = pd.Timedelta(seconds=config["bar_seconds"])
    duration = session["close"] - session["open"]
    if duration % step:
        raise ManifestError("session duration is not a multiple of bar interval")
    if config["timestamp_convention"] == "start":
        return pd.date_range(session["open"], session["close"], freq=step, inclusive="left")
    return pd.date_range(session["open"]+step, session["close"], freq=step, inclusive="both")


def _session_measure(bars, session, config, minimum_minutes, zero_volume_policy):
    grid = _session_grid(session, config)
    log = []
    if config["timestamp_convention"] == "start":
        within = (bars.timestamp >= session["open"]) & (bars.timestamp < session["close"])
    else:
        within = (bars.timestamp > session["open"]) & (bars.timestamp <= session["close"])
    sample = bars[within].copy()
    offgrid = ~sample.timestamp.isin(grid)
    if offgrid.any():
        log.append({"reason":"off_grid_bars", "n_rows":int(offgrid.sum())})
    sample = sample[~offgrid]
    # Identical repeated records are harmless, conflicting prices are not.
    sample = sample.drop_duplicates(["timestamp", "close", "volume"] + (["open"] if "open" in sample else []))
    duplicated = sample.duplicated("timestamp", keep=False)
    if duplicated.any():
        return _empty_day("conflicting_duplicate_bars", len(grid)), log + [{"reason":"conflicting_duplicate_bars", "n_rows":int(duplicated.sum())}]
    closes = pd.to_numeric(sample.close, errors="coerce")
    if (np.isfinite(closes) & (closes <= 0)).any():
        return _empty_day("nonpositive_price", len(grid)), log + [{"reason":"nonpositive_price", "n_rows":int((closes<=0).sum())}]
    valid_price = np.isfinite(closes) & (closes > 0)
    observed = pd.Series(closes[valid_price].to_numpy(), index=pd.DatetimeIndex(sample.loc[valid_price,"timestamp"])).reindex(grid)
    # Counting observed physical time avoids treating five seconds as five minutes.
    observed_seconds = int(observed.notna().sum()) * config["bar_seconds"]
    denominator = len(grid)*config["bar_seconds"] if config["observed_fraction_denominator"] == "session" else config["regular_session_seconds"]
    volume = float(pd.to_numeric(sample.volume, errors="coerce").sum(min_count=1)) if len(sample) else np.nan
    covered = int(observed.notna().sum())
    row = {**_empty_day("insufficient_session", len(grid)), "observed_bars":covered,
           "observed_minutes":observed_seconds/60, "observed_fraction":observed_seconds/denominator,
           "session_volume":volume, "zero_volume":bool(np.isfinite(volume) and volume <= 0)}
    if not covered:
        row["status"] = "missing_session"
        return row, log
    if zero_volume_policy == "exclude_day" and (not np.isfinite(volume) or volume <= 0):
        row["status"] = "zero_or_missing_session_volume"
        return row, log
    if observed_seconds/60 < minimum_minutes:
        return row, log
    filled = observed.ffill()  # no backfill; never carries prices across sessions
    open_price = None
    first = sample[sample.timestamp == grid[0]]
    if len(first) and "open" in first:
        open_price = float(first.open.iloc[0])
    try:
        measures, counts = physical_realized_measures(filled.to_numpy(), bar_seconds=config["bar_seconds"],
                          return_convention=config["return_convention"], open_price=open_price)
    except ValueError as exc:
        row["status"] = str(exc)
        return row, log + [{"reason":str(exc), "n_rows":len(sample)}]
    row.update(counts)
    row["forward_filled_bars"] = int(filled.notna().sum() - covered)
    row["leading_missing_bars"] = int(filled.isna().sum())
    # Enough observed endpoints for at least one 5-minute increment are required.
    if counts["rv5m_increments"] < 1 or counts["n_finite_returns"] < 2:
        return row, log
    row.update(measures, eligible=True, status="ok")
    return row, log


def _empty_day(reason, grid_size=0):
    return {**dict.fromkeys(ESTIMATORS, np.nan), "status":reason, "eligible":False,
            "observed_bars":0, "observed_minutes":0., "observed_fraction":0.,
            "n_close_prices":grid_size, "session_volume":np.nan, "zero_volume":False}


def _instruments(frame):
    required = {"instrument_id", "symbol", "valid_from", "valid_to", "calendar_id", "asset_class"}
    if required-set(frame):
        raise ManifestError("instrument definitions require stable IDs, symbol intervals, calendar and asset class")
    frame = frame.copy()
    frame["valid_from"] = pd.to_datetime(frame.valid_from, errors="raise").dt.normalize()
    frame["valid_to"] = pd.to_datetime(frame.valid_to, errors="raise").dt.normalize()
    if frame[["instrument_id", "symbol", "valid_from", "calendar_id", "asset_class"]].isna().any().any():
        raise ManifestError("missing required instrument identity")
    if (frame.valid_to < frame.valid_from).any():
        raise ManifestError("reversed instrument validity interval")
    for _, group in frame.groupby("instrument_id"):
        group = group.sort_values("valid_from")
        if group.calendar_id.nunique() != 1 or group.asset_class.nunique() != 1:
            raise ManifestError("calendar/class changes require an explicit separate analysis segment")
        ends = group.valid_to.fillna(pd.Timestamp.max.normalize()).to_numpy()
        if np.any(group.valid_from.to_numpy()[1:] <= ends[:-1]):
            raise ManifestError("overlapping symbol intervals for one stable ID")
    return frame


def assign_sessions(timestamps, calendar, timestamp_convention):
    """Vectorized exchange-calendar assignment, including overnight sessions."""
    calendar = calendar.sort_values("open")
    times = pd.DatetimeIndex(timestamps).asi8
    opens, closes = pd.DatetimeIndex(calendar.open).asi8, pd.DatetimeIndex(calendar.close).asi8
    side = "right" if timestamp_convention == "start" else "left"
    positions = np.searchsorted(opens, times, side=side)-1
    safe = np.clip(positions, 0, len(calendar)-1)
    valid = positions >= 0
    valid &= times < closes[safe] if timestamp_convention == "start" else times <= closes[safe]
    dates = np.full(len(times), np.datetime64("NaT"), dtype="datetime64[ns]")
    dates[valid] = calendar.session_date.to_numpy()[positions[valid]]
    return dates


def equity_panel(bars, instruments, sessions, config, universe, *, selected_ids=None):
    instruments = _instruments(instruments)
    sessions = calendar_sessions(sessions)
    if selected_ids is not None:
        unknown = set(selected_ids)-set(instruments.instrument_id)
        if unknown:
            raise ManifestError("selected stable IDs absent from instrument definitions")
        instruments = instruments[instruments.instrument_id.isin(selected_ids)]
    for col in ("timestamp", "close", "volume"):
        if col not in bars:
            raise ManifestError("bars missing " + col)
    bars = bars.copy()
    bars["timestamp"] = parse_timestamps(bars.timestamp, config["timestamp"])
    if "instrument_id" not in bars:
        raise ManifestError("bars require stable instrument_id, not an inferred ticker history")
    invalid_ids = bars[~bars.instrument_id.isin(instruments.instrument_id)]
    exclusions = [{"instrument_id":i, "reason":"unselected_or_undefined_instrument", "n_rows":len(g)} for i,g in invalid_ids.groupby("instrument_id")]
    rows, summaries = [], []
    for instrument_id, history in instruments.groupby("instrument_id", sort=True):
        calendar = sessions[sessions.calendar_id == history.calendar_id.iloc[0]]
        if calendar.empty:
            raise ManifestError("calendar missing for stable instrument ID")
        instrument_bars = bars[bars.instrument_id == instrument_id].copy()
        instrument_bars["session_date"] = assign_sessions(instrument_bars.timestamp, calendar, config["timestamp_convention"])
        outside = int(instrument_bars.session_date.isna().sum())
        if outside:
            exclusions.append({"instrument_id":instrument_id, "reason":"outside_supplied_sessions", "n_rows":outside})
        session_groups = {date:group for date,group in instrument_bars.groupby("session_date")}
        own_rows = []
        for _, session in calendar.iterrows():
            date = session.session_date
            active = history[(history.valid_from <= date) & (history.valid_to.isna() | (history.valid_to >= date))]
            base = {"instrument_id":instrument_id, "date":date, "calendar_id":session.calendar_id,
                    "asset_class":history.asset_class.iloc[0], "active":not active.empty,
                    "symbol":active.symbol.iloc[0] if len(active) else None}
            session_bars = session_groups.get(date, instrument_bars.iloc[:0])
            if active.empty:
                if len(session_bars):
                    exclusions.append({**base, "reason":"bars_outside_definition_lifetime", "n_rows":len(session_bars)})
                row = {**base, **_empty_day("not_listed")}
            else:
                row, logs = _session_measure(session_bars, session, config, universe["min_observed_minutes_per_day"], universe["zero_volume_policy"])
                row = {**base, **row}
                exclusions.extend({**base, **log} for log in logs)
            own_rows.append(row)
        own = pd.DataFrame(own_rows)
        valid = own.eligible & np.isfinite(own.tsrv) & (own.tsrv > 0)
        runs, longest = 0, 0
        for flag in valid:
            runs = runs+1 if flag else 0
            longest = max(longest, runs)
        n_days = int(valid.sum())
        base_exclusions = []
        if n_days == 0:
            base_exclusions.append("no_eligible_estimation_days")
        if n_days < universe["min_estimation_days"]:
            base_exclusions.append("insufficient_estimation_days")
        if longest < universe["min_consecutive_days"]:
            base_exclusions.append("insufficient_consecutive_days")
        if universe["zero_volume_policy"] == "exclude_asset" and own.zero_volume.any():
            base_exclusions.append("zero_volume_asset_excluded")
        base_ok = not base_exclusions
        good_fraction = float((own.loc[valid,"observed_fraction"] >= .8).mean()) if n_days else 0.
        quality = bool(base_ok and n_days >= 500 and good_fraction >= .5)
        summaries.append({"instrument_id":instrument_id, "asset_class":history.asset_class.iloc[0],
            "base_eligible":base_ok, "quality_eligible":quality, "n_estimation_days":n_days,
            "base_exclusion_reasons":base_exclusions,
            "longest_consecutive_estimation_days":longest, "fraction_days_observed_ge_80pct":good_fraction,
            "n_symbol_intervals":len(history), "n_active_calendar_days":int(own.active.sum()),
            "exits_before_calendar_end":bool(history.valid_to.notna().all() and history.valid_to.max() < calendar.session_date.max())})
        rows.extend(own_rows)
    return pd.DataFrame(rows), pd.DataFrame(summaries), pd.DataFrame(exclusions)


def front_month_schedule(daily_volume, contracts, sessions, roots, *, roll_timing):
    """Forward-only volume overtake. Same-session selection is retrospective.

    Missing volume does not prove a contract stopped trading. A still-live
    contract is held when its volume is absent; expiry forces a forward roll.
    """
    if roll_timing not in ("same_session", "next_session"):
        raise ManifestError("futures roll_timing must be explicit")
    for frame, required in ((daily_volume,{"date","root","contract_id","volume"}),
                            (contracts,{"contract_id","root","expiry","calendar_id","asset_class"})):
        if required-set(frame):
            raise ManifestError("incomplete futures contract/volume schema")
    contracts = contracts.copy(); daily_volume = daily_volume.copy()
    contracts["expiry"] = pd.to_datetime(contracts.expiry, errors="raise").dt.normalize()
    daily_volume["date"] = pd.to_datetime(daily_volume.date, errors="raise").dt.normalize()
    if contracts.duplicated("contract_id").any() or daily_volume.duplicated(["date","contract_id"]).any():
        raise ManifestError("duplicate futures contract definitions or daily volumes")
    if not set(roots).issubset(set(contracts.root)):
        raise ManifestError("selected futures roots missing from definitions")
    sessions = calendar_sessions(sessions)
    records = []
    for root in roots:
        definitions = contracts[contracts.root == root].sort_values(["expiry", "contract_id"])
        if definitions.calendar_id.nunique() != 1:
            raise ManifestError("one calendar per root is required")
        calendar = sessions[sessions.calendar_id == definitions.calendar_id.iloc[0]]
        if calendar.empty:
            raise ManifestError("missing futures calendar")
        current = pending = None
        previous_expiry = pd.Timestamp.min
        for date in calendar.session_date:
            old = current
            reason, decision_date = "hold", date
            live = definitions[definitions.expiry >= date]
            if pending is not None:
                current, decision_date = pending
                pending = None; reason = "prior_session_volume_overtake"
            current_def = definitions[definitions.contract_id == current]
            if current is None or current_def.empty or current_def.expiry.iloc[0] < date:
                possible = live[live.expiry >= previous_expiry]
                current = possible.contract_id.iloc[0] if len(possible) else None
                reason = "initial_front_month" if old is None else "forced_expiry_roll"
            if current is None:
                records.append({"root":root,"date":date,"active_contract":None,"rolled":False,"reason":"no_live_contract","decision_date":decision_date})
                continue
            current_def = definitions[definitions.contract_id == current].iloc[0]
            previous_expiry = current_def.expiry
            volumes = daily_volume[(daily_volume.root == root) & (daily_volume.date == date)].set_index("contract_id").volume
            current_volume = volumes.get(current, np.nan)
            later = live[live.expiry > current_def.expiry]
            next_contract = later.contract_id.iloc[0] if len(later) else None
            next_volume = volumes.get(next_contract, np.nan)
            if np.isfinite(current_volume) and current_volume < 0 or np.isfinite(next_volume) and next_volume < 0:
                raise ManifestError("negative futures volume")
            if np.isfinite(current_volume) and np.isfinite(next_volume) and next_volume > current_volume:
                if roll_timing == "same_session":
                    current = next_contract
                    previous_expiry = definitions.loc[definitions.contract_id == current,"expiry"].iloc[0]
                    reason = "same_session_volume_overtake"
                else:
                    pending = (next_contract, date)
                    reason = "scheduled_next_session_volume_overtake"
            elif not np.isfinite(current_volume):
                reason = "missing_current_volume_hold_live_contract"
            records.append({"root":root, "date":date, "active_contract":current,
                "previous_contract":old, "rolled":bool(old is not None and current != old),
                "active_expiry":previous_expiry, "decision_date":decision_date, "reason":reason,
                "front_volume_before_decision":current_volume, "next_volume":next_volume,
                "roll_timing":roll_timing})
    return pd.DataFrame(records)


def futures_panel(bars, contracts, daily_volume, sessions, config, roots, *, roll_timing):
    schedule = front_month_schedule(daily_volume, contracts, sessions, roots, roll_timing=roll_timing)
    bars = bars.copy()
    if not {"contract_id", "timestamp", "close", "volume"}.issubset(bars):
        raise ManifestError("incomplete futures intraday schema")
    bars["timestamp"] = parse_timestamps(bars.timestamp, config["timestamp"])
    sessions = calendar_sessions(sessions)
    rows, exclusions = [], []
    grouped_bars = {}
    known_contracts = set(contracts.contract_id)
    for contract_id, group in bars.groupby("contract_id"):
        if contract_id not in known_contracts:
            exclusions.append({"active_contract":contract_id, "reason":"undefined_contract", "n_rows":len(group)})
            continue
        definition = contracts[contracts.contract_id == contract_id].iloc[0]
        calendar = sessions[sessions.calendar_id == definition.calendar_id]
        if calendar.empty:
            raise ManifestError("missing contract calendar")
        group = group.copy()
        group["session_date"] = assign_sessions(group.timestamp, calendar, config["timestamp_convention"])
        outside = int(group.session_date.isna().sum())
        if outside:
            exclusions.append({"active_contract":contract_id, "reason":"outside_supplied_sessions", "n_rows":outside})
        for date, session_group in group.groupby("session_date"):
            grouped_bars[(contract_id,date)] = session_group
    for _, selected in schedule.iterrows():
        root_defs = contracts[contracts.root == selected.root]
        calendar_id = root_defs.calendar_id.iloc[0]
        session = sessions[(sessions.calendar_id == calendar_id) & (sessions.session_date == selected.date)].iloc[0]
        base = {"instrument_id":selected.root, "root":selected.root, "date":selected.date,
                "calendar_id":calendar_id, "asset_class":root_defs.asset_class.iloc[0],
                "active_contract":selected.active_contract, "rolled":bool(selected.rolled), "active":True}
        intraday = grouped_bars.pop((selected.active_contract, selected.date), bars.iloc[:0])
        row, logs = _session_measure(intraday, session, config, config.get("min_observed_minutes_per_day",0), "keep")
        rows.append({**base, **row})
        if row["status"] != "ok":
            exclusions.append({**base, "reason":row["status"], "n_rows":len(intraday)})
        exclusions.extend({**base, **log} for log in logs)
    for (contract_id,date), unused in grouped_bars.items():
        exclusions.append({"active_contract":contract_id,"date":date,"reason":"not_selected_front_contract", "n_rows":len(unused)})
    return pd.DataFrame(rows), schedule, pd.DataFrame(exclusions)


def aggregate_variance(daily, *, frequency, missing_day_policy, edge_period_policy):
    """Sum daily variance, never log variance. Preserve every calendar bin.

    An incomplete active-day bin is invalidated by default. The explicit
    "exclude" edge policy conservatively excludes the first/last supplied
    calendar bins and partially unlisted bins. It does not infer whether
    absent pre-snapshot dates were holidays; "include" keeps boundary bins.
    """
    if frequency not in ("weekly", "monthly"):
        raise ValueError("aggregation frequency must be weekly or monthly")
    if missing_day_policy not in ("invalidate", "available") or edge_period_policy not in ("include", "exclude"):
        raise ValueError("aggregation missing/edge rules must be explicit")
    rule = "W-FRI" if frequency == "weekly" else pd.offsets.MonthEnd()
    rows = []
    for instrument_id, frame in daily.groupby("instrument_id"):
        frame = frame.copy(); frame["date"] = pd.to_datetime(frame.date)
        if frame.duplicated("date").any():
            raise ValueError("duplicate daily rows during aggregation")
        bins = list(frame.set_index("date").resample(rule))
        for bin_index, (end, group) in enumerate(bins):
            active = group[group.active] if "active" in group else group
            row = {"instrument_id":instrument_id, "date":end, "frequency":frequency,
                   "calendar_days_in_bin":len(group), "expected_active_days":len(active),
                   "active":len(active)>0, "missing_day_policy":missing_day_policy,
                   "edge_period_policy":edge_period_policy,
                   "sample_boundary_period":bin_index in (0,len(bins)-1)}
            for estimator in ESTIMATORS:
                values = active[estimator]
                valid = np.isfinite(values) & (values > 0)
                allow = len(active)>0 and valid.any()
                if missing_day_policy == "invalidate" and not valid.all():
                    allow = False
                if edge_period_policy == "exclude" and (len(active) != len(group) or bin_index in (0,len(bins)-1)):
                    allow = False
                row[estimator] = float(values[valid].sum()) if allow else np.nan
                row[estimator + "_valid_days"] = int(valid.sum())
            rows.append(row)
    return pd.DataFrame(rows)
