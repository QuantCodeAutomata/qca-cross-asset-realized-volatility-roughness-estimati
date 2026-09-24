"""Gaussian fOU/fBm paths with calibration fixed before path generation.

The covariance is that of the continuous stationary fOU sampled at daily times,
not a filtered-fGn recursion. The calibration convention is implementation-
defined; the author's exact simulation calibration has not been established.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky, toeplitz
from scipy.special import gamma, logsumexp

from ..simulate_rough_paths import simulate_fbm_davies_harte, simulate_fou_log_vol
from .common import canonical_json, digest

ALGORITHM = "stationary-fou-spectral-cholesky-v1"


class CovarianceError(RuntimeError):
    """Invalid covariance/factorization; no eigenvalue clipping is performed."""


@dataclass(frozen=True)
class Calibration:
    H: float
    kappa: float
    nu: float
    mean: float
    target_std: float
    target_annualized_vol: float
    horizon: int
    dt: float
    annualization_days: float
    convention: str


def fbm_expected_sample_variance(n: int, H: float, dt: float = 1.) -> float:
    """E[unbiased sample variance of B_H(0),...,B_H((n-1)dt)]."""
    if n < 2 or not 0 < H < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid fBm calibration parameters")
    lags = np.arange(1, n, dtype=float)
    return float(np.sum((n - lags) * (lags * dt)**(2 * H)) / (n * (n - 1)))


def calibrate(H: float, kappa: float, *, horizon: int = 2520,
              target_std: float = 1.05, annualized_vol: float = .128,
              annualization_days: float = 252., dt: float = 1.) -> Calibration:
    values = [H, kappa, target_std, annualized_vol, annualization_days, dt]
    if not np.all(np.isfinite(values)) or not 0 < H < 1 or kappa < 0:
        raise ValueError("invalid Gaussian calibration parameters")
    if horizon < 2 or min(target_std, annualized_vol, annualization_days, dt) <= 0:
        raise ValueError("calibration scales and horizon must be positive")
    if kappa > 0:
        unit_variance = gamma(2 * H + 1) / (2 * kappa**(2 * H))
        nu = target_std / np.sqrt(unit_variance)
        log_mean_vol_factor = target_std**2 / 8
        convention = "stationary_logvariance_std"
    else:
        nu = target_std / np.sqrt(fbm_expected_sample_variance(horizon, H, dt))
        time = np.arange(horizon) * dt
        variance = nu**2 * time**(2 * H)
        log_mean_vol_factor = logsumexp(variance / 8) - np.log(horizon)
        convention = "expected_unbiased_fbm_sample_variance"
    mean = 2 * (np.log(annualized_vol / np.sqrt(annualization_days)) - log_mean_vol_factor)
    return Calibration(H, kappa, float(nu), float(mean), target_std, annualized_vol,
                       horizon, dt, annualization_days, convention)


def stationary_covariance(lag: float, H: float, kappa: float, nu: float = 1., tol: float = 1e-10) -> float:
    """2 int cos(lag*lambda) S(lambda) d lambda, using a dimensionless tail.

    Scaling u=lag*lambda keeps the oscillation frequency equal to one at large
    lags. This avoids the increasingly oscillatory near-origin integration that
    would arise from repeatedly subtracting the existing exact M2 integral.
    """
    if not np.all(np.isfinite([lag, H, kappa, nu, tol])) or not 0 < H < 1 or min(kappa, nu, tol) <= 0:
        raise ValueError("stationary fOU requires H in (0,1), kappa,nu,tol > 0")
    lag = abs(lag)
    variance = nu**2 * gamma(2 * H + 1) / (2 * kappa**(2 * H))
    if lag == 0:
        return float(variance)
    if H == .5:
        return float(variance * np.exp(-kappa * lag))
    e = kappa * lag
    coefficient = nu**2 * gamma(2*H+1) * np.sin(np.pi*H) / (2*np.pi) * lag**(2*H)
    power = 1 / (2 - 2*H)
    def near(v):
        u = v**power
        return coefficient * power * np.cos(u) / (u*u + e*e)
    def tail(u):
        return coefficient * u**(1-2*H) / (u*u + e*e)
    a, error_a = quad(near, 0., 1., epsabs=tol, epsrel=tol, limit=500)
    b, error_b = quad(tail, 1., np.inf, weight="cos", wvar=1., epsabs=tol, limlst=500, limit=500)
    if not np.isfinite(a+b) or error_a + error_b > 100 * tol * max(1., abs(a)+abs(b)):
        raise CovarianceError("spectral_covariance_integration_not_accurate")
    return float(2*(a+b))


def covariance_matrix(n: int, H: float, kappa: float, nu: float, dt: float = 1.) -> np.ndarray:
    if n < 2 or dt <= 0:
        raise ValueError("require n>=2, dt>0")
    if kappa == 0:
        time = np.arange(n) * dt
        return .5 * nu**2 * (time[:, None]**(2*H) + time[None, :]**(2*H) - np.abs(time[:, None]-time[None, :])**(2*H))
    return toeplitz([stationary_covariance(i*dt, H, kappa, nu) for i in range(n)])


def checked_cholesky(covariance: np.ndarray) -> np.ndarray:
    c = np.asarray(covariance, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.all(np.isfinite(c)):
        raise CovarianceError("invalid_covariance_matrix")
    if not np.allclose(c, c.T, rtol=0, atol=1e-12 * max(float(np.max(np.abs(c))), 1.)):
        raise CovarianceError("asymmetric_covariance_matrix")
    try:
        factor = cholesky(c, lower=True, check_finite=True)
    except np.linalg.LinAlgError as exc:
        raise CovarianceError("covariance_not_positive_definite; no_eigenvalue_repair") from exc
    if np.any(np.diag(factor) <= 0):
        raise CovarianceError("nonpositive_cholesky_diagonal")
    return factor


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


@lru_cache(maxsize=4)
def covariance_factor(n: int, H: float, kappa: float, nu: float, dt: float = 1., cache_dir: str | None = None):
    """Cache a strictly checked stationary covariance factor, not market data."""
    if kappa <= 0:
        raise ValueError("fBm uses exact fGn increments; its origin covariance is singular")
    spec = dict(algorithm=ALGORITHM, n=n, H=H, kappa=kappa, nu=nu, dt=dt, tolerance=1e-10)
    key = digest(spec)
    path = Path(cache_dir) / (key + ".npz") if cache_dir else None
    if path and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            factor = saved["factor"]
            meta = json.loads(str(saved["metadata"].item()))
        if meta["spec"] != spec or factor.shape != (n, n) or _array_sha(factor) != meta["factor_sha256"]:
            raise CovarianceError("covariance_cache_integrity_failure")
        if not np.all(np.isfinite(factor)) or np.any(np.diag(factor) <= 0) or np.any(np.triu(factor, 1)):
            raise CovarianceError("invalid_cached_factor")
    else:
        covariance = covariance_matrix(n, H, kappa, nu, dt)
        factor = checked_cholesky(covariance)
        meta = dict(spec=spec, factor_sha256=_array_sha(factor), covariance_sha256=_array_sha(covariance),
                    eigenvalue_repair=False, jitter=0.0)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name("." + key + "." + uuid4().hex + ".tmp")
            try:
                with temporary.open("wb") as f:
                    np.savez_compressed(f, factor=factor, metadata=np.array(canonical_json(meta)))
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    factor.setflags(write=False)
    return factor


def generate_log_variance(n: int, calibration: Calibration, seed: int, *,
                          cache_dir: str | None = None, generator: str = "gaussian") -> np.ndarray:
    if n < 2:
        raise ValueError("at least two daily observations required")
    c = calibration
    if generator == "legacy":
        x = simulate_fou_log_vol(n, c.H, c.kappa, nu=.5, dt=c.dt, seed=seed)
        x = (x - x.mean()) * c.target_std / np.std(x, ddof=1)
        x += 2*np.log(c.target_annualized_vol / (np.mean(np.exp(x/2))*np.sqrt(c.annualization_days)))
        return x
    if generator != "gaussian":
        raise ValueError("generator must be gaussian or legacy")
    if c.kappa == 0:
        increments = simulate_fbm_davies_harte(n-1, c.H, seed)
        return c.mean + np.r_[0., np.cumsum(increments)] * c.nu * c.dt**c.H
    factor = covariance_factor(n, c.H, c.kappa, c.nu, c.dt, cache_dir)
    return c.mean + factor @ np.random.default_rng(seed).standard_normal(n)
