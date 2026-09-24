"""Synthetic file fixtures; calendars here are toy inputs, not market calendars."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.replication.common import sha256_file


def calendar_fixture(days=("2024-03-08","2024-03-11","2024-03-12"), minutes=390):
    return pd.DataFrame([{"calendar_id":"X", "session_date":day,
        "open":pd.Timestamp(day+" 09:30",tz="America/New_York").isoformat(),
        "close":(pd.Timestamp(day+" 09:30",tz="America/New_York")+pd.Timedelta(minutes=minutes)).isoformat()}
        for day in days])


def dataset_config(bar_seconds=60):
    return {"bar_seconds":bar_seconds,"timestamp_convention":"start",
            "timestamp":{"encoding":"epoch","unit":"ms"},"return_convention":"close_only",
            "observed_fraction_denominator":"session","regular_session_seconds":23400}


def universe_rules():
    return {"min_estimation_days":1,"min_consecutive_days":0,"min_observed_minutes_per_day":0,
            "zero_volume_policy":"exclude_day","rationale":"Synthetic fixture; not an interpretation of conflicting paper screens"}


def instrument_fixture():
    return pd.DataFrame([{"instrument_id":"E1","symbol":"OLD","valid_from":"2024-03-08","valid_to":"2024-03-10","calendar_id":"X","asset_class":"single_stock"},
                         {"instrument_id":"E1","symbol":"NEW","valid_from":"2024-03-11","valid_to":"2024-03-11","calendar_id":"X","asset_class":"single_stock"}])


def bars_fixture(calendar, instrument_id="E1", bar_seconds=60, level=100.):
    rows = []
    for index, session in calendar.iterrows():
        opened, closed = pd.Timestamp(session.open), pd.Timestamp(session.close)
        times = pd.date_range(opened, closed, freq=f"{bar_seconds}s", inclusive="left")
        returns = np.full(len(times), .0002*(1+.2*np.sin(index)))
        logprices = np.log(level) + np.cumsum(returns)
        for i,timestamp in enumerate(times):
            rows.append({"instrument_id":instrument_id,"timestamp":timestamp.tz_convert("UTC").value//1_000_000,
                "open":float(np.exp(logprices[i]-returns[i])),"close":float(np.exp(logprices[i])),"volume":10.})
    return pd.DataFrame(rows)


def write_manifest(tmp_path, files, *, datasets=None, selections=None, analysis=None, instruments=None):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True,exist_ok=True)
    entries = {}
    for role, frame in files.items():
        path = tmp_path / (role+".csv")
        frame.to_csv(path,index=False)
        entries[role] = {"path":path.name,"sha256":sha256_file(path)}
    data = {"schema_version":1,"data_kind":"synthetic_fixture","source":"generated_fixture",
        "snapshot":"synthetic-v1","files":entries,"units":{"prices":"levels","variance":"daily_unannualized"},
        "price_adjustments":{"equity":"unadjusted synthetic","seconds":"unadjusted synthetic","futures":"unadjusted synthetic"},
        "datasets":datasets if datasets is not None else {"equity":dataset_config()},
        "base_universe":universe_rules(),"selections":selections or {},
        "analysis":analysis or {"lag_windows":[[1,10],[1,40],[40,250]],"overlapping":[False]},
        "publication":{"approved_aggregates":["counts","manifest_sha256"]}}
    target = tmp_path / "private_manifest.json"
    target.write_text(json.dumps(data,indent=2))
    return target
