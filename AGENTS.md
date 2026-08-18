# Cross-Asset Realized-Volatility Roughness — Agent Memory

## Project Layout
```
src/                   Core library modules
exp/                   Experiment runner scripts (exp1–exp4)
tests/                 pytest suite (73 tests, 4 files)
results/               CSV outputs + RESULTS.md
data/                  (empty; experiments use synthetic data)
```

## Key Source Modules and Their Public APIs

### `src/hurst_realized.py`
- `compute_second_moment_scaling(log_rv, delta_max) -> (deltas, m2_values)` — returns a **tuple**; second argument is an `int`, not an array.
- `estimate_hurst_ols(log_rv, delta_max=20) -> dict` — keys: `H_hat`, `r_squared`, `beta`, `intercept`. **No `delta_min` parameter.**

### `src/rv_estimators.py`
- All estimators (`compute_rv5m`, `compute_bpv`, `compute_tsrv`, `compute_realized_kernel`) take **returns** (first-differences of log-prices), NOT log-price arrays.
- `compute_all_estimators(returns) -> dict` with keys: `rv5m`, `rk`, `tsrv`, `bpv`, `pav`, `pabpv`, `ctrv`.

### `src/simulate_rough_paths.py`
- `simulate_fbm_davies_harte(n, H, seed) -> np.ndarray` — returns **fGn increments** (length `n`), NOT the fBM path. Cumsum to get the fBM level.
- `simulate_fou_log_vol(n_days, H, kappa, nu, seed) -> np.ndarray` — returns log-vol levels.

### `src/autocovariance_fit.py`
- `fgn_autocovariance(H, nu, k) -> float` — lag-0 = ν², lag-k < 0 for H < 0.5.
- `compute_empirical_autocovariances(series, max_lag) -> np.ndarray` — length `max_lag+1`.
- `fit_two_estimator_correction(log_rv_a, log_rv_b, delta_max) -> dict` — key `H_corrected_a`.

### `src/fou_spectral.py`
- `compute_exact_m2(Delta, H, kappa, nu=1.0) -> float`
- `compute_m2_grid(deltas, H, kappa, nu=1.0) -> np.ndarray`
- `local_slope_asymptotic(Delta, H, kappa) -> float`

### `src/mean_reversion_bias.py`
- `compute_exact_ols_h(H, kappa, delta_max, nu=1.0) -> (H_hat, bias)`
- `compute_ols_bias_approximation(H, kappa, delta_max) -> float`

### `src/option_pricing.py`
- `black76_price(F, K, T, r, sigma, option_type) -> float`
- `black76_implied_vol(price, F, K, T, r, option_type) -> Optional[float]`
- `baw_approximation(S, K, T, r, q, sigma, option_type) -> float`

### `src/atm_skew_estimator.py`
- `compute_log_moneyness(strikes, forward) -> np.ndarray`
- `estimate_atm_skew(strikes, ivols, forward, moneyness_band=0.08, min_strikes=10) -> Optional[dict]`

### `src/implied_hurst.py`
- `estimate_daily_implied_hurst(skew_ts, date, maturity_window) -> Optional[dict]`
- `estimate_pooled_implied_hurst(skew_ts, maturity_window, min_obs_per_date) -> dict` — key `H_hat_IV`, `r_psi_squared`, `identified` (bool: R²≥0.3).

## Test Suite Facts (73 tests)
| File | Count | Coverage |
|---|---|---|
| `test_correction.py` | 11 | fGn autocovariance, noise model, fBM sim, two-estimator correction |
| `test_fou_spectral.py` | 12 | M₂ grid, asymptotic slope, OLS bias |
| `test_hurst_estimation.py` | 8 | second-moment scaling, OLS H, RV estimator sanity |
| `test_implied_hurst.py` | 15 | Black-76, log-moneyness, ATM skew, implied H |
| `test_rv_estimators.py` | 28 | all RV estimators + Hurst OLS |

## Critical Simulation Design Decision

`data/data_loader.py::generate_synthetic_intraday` simulates the fOU at **daily** frequency:

```
X_{d+1} = (1-kappa)*X_d + nu*dB_d^H   (one step per trading day)
sigma_d  = exp(X_d/2)
r_{d,i}  ~ N(0, sigma_d^2 / n_bars)   for i = 0…n_bars-1
```

**Why daily, not minute-by-minute?**  If fOU is driven at minute frequency with `kappa=0.01/bar`
the process loses memory in ~100 bars (≈15 min), which is far shorter than one trading day.
Within-day averaging over 390 bars then makes daily log-RV nearly i.i.d. (effective H → 0).
Driving at daily frequency ensures the kappa mean-reversion operates on the *same* time scale
as the delta_max=10-day Hurst regression, so the fOU roughness H is preserved in daily log-RV.

**PAV behaviour on clean data:**  `compute_pav` returns 0 for noise-free returns.
The bias term `psi_1/(2*k_n)*RV_full ≈ RV_full/(12*k_n)` is ~10× larger than the main
pre-averaged sum because the formula was designed to cancel microstructure noise variance.
Without added noise the over-correction dominates and PAV is clipped to 0.
PAV is excluded from exp1 H estimates but that is the mathematically correct outcome.

## Known Numerical Facts
- `compute_exact_ols_h(H=0.3, kappa=0.02, delta_max=10)` → bias ≈ -0.0079 (not ≤ 0.005).  
  Test threshold for kappa ≤ 0.02 is **0.01**, not 0.005.
- Bias threshold 0.005 is only valid for kappa ≤ 0.01.
- `simulate_fbm_davies_harte` fGn increments have variance ≈ 1 (ν embedded); tolerance 0.05.
- fBM Hurst recovery from 8 000-point simulation: tolerance 0.07.

## Stale-pycache Warning
Running tests after heavy edits without clearing `__pycache__` can produce phantom line-number mismatches in pytest tracebacks. Clear with:
```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

## Python Environment
- Python 3.12.13, pytest 9.1.1
- Key packages: numpy, scipy, statsmodels, pandas, matplotlib
