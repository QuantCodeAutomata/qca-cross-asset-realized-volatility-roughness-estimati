# Developer notes: audited rough-volatility harness

## Scope invariant

This repository contains synthetic experiments, one exact numerical
calculation, and a repository-native runner for a private SPY real-data pilot.
Do not describe it as a replication of the paper's Nasdaq/CME/OPRA evidence
unless the full real panels are added and processed.

## Core APIs and invariants

### `src/hurst_realized.py`

- `compute_second_moment_scaling(x, delta_max=10, overlapping=False)` returns
  `(deltas, m2)`.
- Default increments are non-overlapping and exactly `Delta` observations apart.
- Missing dates stay on the common calendar grid; never `dropna()` and compress
  time before H estimation.
- `estimate_hurst_ols(...)` returns keys `H_hat`, `b`, `a`, `r_squared`,
  `n_lags`, and `n_obs`.

### `src/rv_estimators.py`

All public estimators consume one intraday array of log returns.

- TSRV uses K-step price differences on each staggered price grid.
- PAV/PABPV use `g(x)=min(x,1-x)`, `psi_1=1`, `psi_2=1/12`.
- RK bandwidth uses the `n^(3/5)` data-driven rate and a normalized lag-one
  covariance estimate.
- Negative finite-sample estimates may be returned when `clip=False`; clipping
  is a reporting choice, not part of the mathematical identity.

### `src/autocovariance_fit.py`

- Intended pairing: leg `a = RV5m`, leg `b = RK`.
- Keep signed `delta_m`; never clamp it to zero.
- Demean increments before autocovariances.
- Preserve missing-date positions.
- Enforce `omega_a^2 - omega_b^2 = delta_m/2` exactly.
- Invalid corrected curves return `NaN` and diagnostics; never manufacture a
  large H by clipping moments to `1e-12`.

### `src/simulate_rough_paths.py`

- `simulate_fbm_davies_harte(n, H, seed)` returns unit-variance fGn increments.
- `simulate_fou_log_vol(n_days, H, kappa, nu, dt=1, ...)` treats `kappa` in the
  same units as `dt`; paper tables use trading-day units.
- A stationary burn-in is applied for `kappa > 0`.
- `simulate_noisy_intraday_prices` returns `n_bars + 1` prices so differencing
  yields exactly `n_bars` returns.

### `src/fou_spectral.py`

- `compute_exact_m2(Delta, H, kappa, nu=1)` uses the pure-fBM analytic result at
  `kappa=0` and a regularized stationary-fOU covariance integral otherwise.
- The origin treatment must work for the full `H in (0,1)` range.

### Option pipeline

- Start with strikes and option-implied volatilities, estimate one ATM skew per
  `(date, expiry)`, then estimate the maturity power law.
- `H_hat_IV` is meaningless when pooled `R2_psi` is below the identification
  threshold. Preserve and report the `identified` flag.
- Use calendar-day maturities divided by 365 consistently.
- Never seed synthetic results with Python's process-randomized `hash()`.

### Preprocessing

- Require the caller to say whether prices are raw or already logarithmic; do
  not infer this from positivity.
- Use only forward fill inside a session. Backfilling leading prices is
  look-ahead.
- Keep the documented timestamp unit aligned with the implementation.

## Validation commands

```bash
pytest -q
python -m compileall -q src data exp tests
python exp/exp1_realized_hurst.py
python exp/exp2_mean_reversion.py
python exp/exp3_measurement_error.py
python exp/exp4_implied_hurst.py
```

The corrected harness contains 85 tests, and the repository-native SPY adapter
adds six synthetic integration tests. Any change to an estimator should add a
scale or identity test, not merely a non-negativity smoke test.

## Real-data boundary

`exp/exp5_spy_empirical.py` writes Massive caches and all market-derived outputs
under `.local/spy_empirical` by default. Keep `.local`, raw minute bars, daily RV
panels, manifests, and credentials out of Git. Only aggregate findings belong in
`AUDIT_REPORT.md`.
