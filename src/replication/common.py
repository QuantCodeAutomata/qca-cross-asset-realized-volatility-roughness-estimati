"""Local, strict-JSON artifact helpers. No provider or database clients."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def canonical_json(value):
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def private_output(path):
    """Allow paths outside checkouts, or inside a checkout's ignored .local tree.

    This does not authorize public release. Detailed outputs always stay private.
    Resolving symlinks prevents an apparent outside directory targeting Git.
    """
    path = Path(path).expanduser().resolve()
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            if not path.is_relative_to(parent / ".local"):
                raise ValueError("private output must be outside Git or under .local")
            break
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_frames(output, frames):
    output = private_output(output)
    hashes = {}
    for name, frame in frames.items():
        target = output / (name + ".csv")
        temp = target.with_suffix("." + uuid4().hex + ".tmp")
        serializable = frame.copy()
        for column in serializable.select_dtypes(include="object"):
            serializable[column] = serializable[column].map(
                lambda x: canonical_json(x) if isinstance(x, (dict, list, tuple, np.ndarray)) else x)
        serializable.to_csv(temp, index=False)
        os.replace(temp, target)
        hashes[target.name] = sha256_file(target)
    return hashes
