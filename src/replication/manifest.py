"""Strict local-file manifests: provenance, units, calendars and explicit choices."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import digest, sha256_file


class MissingSource(FileNotFoundError):
    """A declared or required local source is unavailable."""


class ManifestError(ValueError):
    """An ambiguous or inconsistent source specification."""


def _check_private_path(path):
    for parent in (path.parent, *path.parent.parents):
        if (parent / ".git").exists():
            if not path.is_relative_to(parent / ".local"):
                raise ManifestError("private inputs must be outside Git or under .local")
            break


@dataclass
class FileManifest:
    path: Path
    content: dict
    file_sha256: str = ""
    verified_hashes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise MissingSource("manifest file is absent")
        _check_private_path(path)
        raw_manifest = path.read_bytes()
        data = json.loads(raw_manifest)
        required = {"schema_version", "data_kind", "source", "snapshot", "files", "units", "price_adjustments", "datasets"}
        if required - set(data):
            raise ManifestError(f"missing manifest fields: {sorted(required-set(data))}")
        if data["schema_version"] != 1 or data["data_kind"] not in ("market", "synthetic_fixture"):
            raise ManifestError("unsupported manifest schema or data_kind")
        if not str(data["source"]).strip() or not str(data["snapshot"]).strip():
            raise ManifestError("source and snapshot must be explicit")
        if data["units"].get("prices") != "levels" or data["units"].get("variance") != "daily_unannualized":
            raise ManifestError("adapters require price levels and daily unannualized variance")
        if not isinstance(data["price_adjustments"], dict) or not data["price_adjustments"]:
            raise ManifestError("explicit price adjustment rules are required")
        for role, entries in data["files"].items():
            for item in entries if isinstance(entries, list) else [entries]:
                if not isinstance(item, dict) or not {"path", "sha256"}.issubset(item):
                    raise ManifestError(f"invalid file specification: {role}")
                if not re.fullmatch("[0-9a-f]{64}", str(item["sha256"])):
                    raise ManifestError(f"invalid SHA-256 for {role}")
                if "://" in item["path"]:
                    raise ManifestError("only local files are supported; no URL fetch")
        return cls(path, data, file_sha256=hashlib.sha256(raw_manifest).hexdigest())

    @property
    def fingerprint(self):
        return digest(self.content)

    def entries(self, role):
        if role not in self.content["files"]:
            raise MissingSource("missing manifest file role: " + role)
        entries = self.content["files"][role]
        if not entries:
            raise MissingSource("empty manifest file role: " + role)
        return entries if isinstance(entries, list) else [entries]

    def file_paths(self, role):
        paths = []
        for entry in self.entries(role):
            path = (self.path.parent / entry["path"]).resolve()
            _check_private_path(path)
            if not path.is_file():
                raise MissingSource("missing local source for role: " + role)
            if sha256_file(path) != entry["sha256"]:
                raise ManifestError("snapshot hash mismatch for role: " + role)
            paths.append(path)
        self.verified_hashes[role] = [item["sha256"] for item in self.entries(role)]
        return paths

    def read(self, role):
        frames = []
        for path in self.file_paths(role):
            if path.suffix.lower() == ".csv":
                # IDs are opaque strings; never turn "001" into numeric 1.
                identity_columns = ("instrument_id","contract_id","underlying_id","calendar_id",
                                    "root","symbol","option_id","realized_id","underlying_contract_id")
                try:
                    frames.append(pd.read_csv(path, dtype={name:"string" for name in identity_columns}))
                except pd.errors.EmptyDataError as exc:
                    raise MissingSource("empty tabular source for role: " + role) from exc
            elif path.suffix.lower() in (".parquet", ".pq"):
                try:
                    frames.append(pd.read_parquet(path))
                except ImportError as exc:
                    raise MissingSource("Parquet requires an installed pyarrow or fastparquet engine") from exc
            else:
                raise ManifestError("tabular sources must be CSV or Parquet")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def dataset(self, kind):
        if kind not in self.content["datasets"]:
            raise MissingSource("missing dataset convention: " + kind)
        config = self.content["datasets"][kind]
        required = {"bar_seconds", "timestamp_convention", "timestamp", "return_convention", "observed_fraction_denominator", "regular_session_seconds"}
        if required-set(config):
            raise ManifestError(f"missing {kind} dataset fields: {sorted(required-set(config))}")
        seconds = config["bar_seconds"]
        if not isinstance(seconds, int) or seconds <= 0 or 300 % seconds != 0:
            raise ManifestError("bar interval must divide 300 seconds")
        if config["timestamp_convention"] not in ("start", "close") or config["return_convention"] not in ("close_only", "open_seeded"):
            raise ManifestError("invalid bar timestamp or return convention")
        if config["observed_fraction_denominator"] not in ("session", "regular_session"):
            raise ManifestError("observed-fraction denominator must be explicit")
        if config["regular_session_seconds"] <= 0:
            raise ManifestError("invalid regular session duration")
        if kind not in self.content["price_adjustments"]:
            raise ManifestError("missing price adjustment description for " + kind)
        return config

    def base_universe(self):
        rules = self.content.get("base_universe", {})
        required = {"min_estimation_days", "min_consecutive_days", "min_observed_minutes_per_day", "zero_volume_policy", "rationale"}
        if required-set(rules):
            raise ManifestError("equity base universe must explicitly resolve Sections 3.1/3.2")
        if any(rules[x] < 0 for x in required if x not in ("zero_volume_policy", "rationale")):
            raise ManifestError("negative universe threshold")
        if rules["zero_volume_policy"] not in ("keep", "exclude_day", "exclude_asset") or not str(rules["rationale"]).strip():
            raise ManifestError("invalid zero-volume policy or missing rationale")
        return rules

    def selection(self, name):
        selections = self.content.get("selections", {})
        if name not in selections or not isinstance(selections[name], list) or not selections[name]:
            raise MissingSource("explicit nonempty selection required: " + name)
        if name != "comparison_map" and len(selections[name]) != len(set(selections[name])):
            raise ManifestError("duplicate IDs in selection: " + name)
        return selections[name]


def parse_timestamps(values, spec):
    """No silent assumption that naive timestamps are UTC; DST errors are fatal."""
    encoding = spec.get("encoding")
    if encoding == "epoch":
        if spec.get("unit") not in ("s", "ms", "us", "ns"):
            raise ManifestError("epoch timestamp unit required")
        out = pd.to_datetime(pd.to_numeric(values, errors="raise"), unit=spec["unit"], utc=True, errors="raise")
    elif encoding == "iso8601":
        parsed = []
        for value in values:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                raise ManifestError("missing timestamp")
            if ts.tzinfo is None:
                tz = spec.get("naive_timezone")
                if not tz:
                    raise ManifestError("naive timestamp without declared timezone")
                ts = ts.tz_localize(tz, ambiguous="raise", nonexistent="raise")
            parsed.append(ts.tz_convert("UTC"))
        out = pd.DatetimeIndex(parsed)
    else:
        raise ManifestError("timestamp encoding must be epoch or iso8601")
    if pd.isna(out).any():
        raise ManifestError("missing timestamp")
    return pd.DatetimeIndex(out)


def calendar_sessions(frame):
    required = {"calendar_id", "session_date", "open", "close"}
    if required-set(frame):
        raise ManifestError("session calendar requires calendar_id, session_date, open, close")
    frame = frame.copy()
    frame["session_date"] = pd.to_datetime(frame.session_date, errors="raise").dt.normalize()
    frame["open"] = parse_timestamps(frame.open, {"encoding":"iso8601"})
    frame["close"] = parse_timestamps(frame.close, {"encoding":"iso8601"})
    if frame.duplicated(["calendar_id", "session_date"]).any() or (frame.close <= frame.open).any():
        raise ManifestError("duplicate or nonpositive session interval")
    for _, group in frame.groupby("calendar_id"):
        group = group.sort_values("open")
        if np.any(group.open.to_numpy()[1:] < group.close.to_numpy()[:-1]):
            raise ManifestError("overlapping calendar sessions")
    return frame.sort_values(["calendar_id", "session_date"]).reset_index(drop=True)
