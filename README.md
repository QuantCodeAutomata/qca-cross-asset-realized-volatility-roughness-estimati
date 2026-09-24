# Audited Rough-Volatility Replication Harness

This repository is an audited **synthetic and numerical validation harness** for
Saad Mouti's *Rough Volatility Across Assets* (2026). It is not the paper's
empirical Nasdaq/CME/OPRA replication: the market-data panels used in the paper
are not included here.

The distinction matters:

- **Experiment 2** directly reproduces the paper's exact fOU mean-reversion
  calculation.
- **Experiments 1, 3, and 4** test the estimator pipeline on controlled synthetic
  data and compare the outputs with paper benchmarks.
- None of the synthetic experiments can establish the paper's cross-sectional
  asset-class ranking or option-market findings in real data.

The submitted version was audited and corrected for estimator normalization,
time-unit, missing-date, simulation-calibration, correction-stability, and
option-identification errors. A private SPY minute-bar run now also validates
the central roughness result on real data without committing the market panel.
See [`AUDIT_REPORT.md`](AUDIT_REPORT.md) for the English audit and real-data
validation report, and [`results/RESULTS.md`](results/RESULTS.md) for the
synthetic experiment outputs.

## Repository map

```text
.
├── data/
│   └── data_loader.py                 # optional Massive loaders + synthetic panel wrapper
├── exp/
│   ├── exp1_realized_hurst.py         # synthetic realized-H estimator recovery
│   ├── exp2_mean_reversion.py         # exact fOU mean-reversion calculation
│   ├── exp3_measurement_error.py      # measurement error + two-estimator correction
│   ├── exp4_implied_hurst.py          # strike-level synthetic option-skew identification
│   └── exp5_spy_empirical.py          # private-output Massive SPY real-data pilot
├── src/
│   ├── autocovariance_fit.py          # additive log-noise correction
│   ├── fou_spectral.py                # exact stationary-fOU second moment
│   ├── hurst_realized.py              # non-overlapping short-lag H estimator
│   ├── implied_hurst.py               # daily and pooled ATM-skew regressions
│   ├── preprocessing/                 # causal grid construction
│   ├── rv_estimators.py               # RV5m, RK, TSRV, PAV, PABPV, BPV, C-TRV
│   └── simulate_rough_paths.py        # canonical fGn/fOU and intraday simulation
├── tests/                              # regression and behavioral tests
└── results/                            # generated CSV, JSON, Markdown, and figures
```

## Corrected methodological conventions

### Realized-H estimator

For daily log realized variance `X_t`, the paper's primary estimator is

```text
m(2, Delta) = mean_k (X_(k+1)Delta - X_kDelta)^2
log m(2, Delta) = a + 2 H log Delta + u_Delta,
Delta = 1, ..., 10.
```

`src/hurst_realized.py` uses exact **non-overlapping** increments by default and
preserves missing calendar dates. It does not add a zero increment or a short
terminal block.

### Realized-variance estimators

`src/rv_estimators.py` implements:

| Key | Estimator | Main role |
|---|---|---|
| `rv5m` | five-minute realized variance | benchmark |
| `rk` | Parzen realized kernel | microstructure-noise robust |
| `tsrv` | two-scale realized variance | microstructure-noise robust |
| `pav` | pre-averaged realized variance | microstructure-noise robust |
| `pabpv` | pre-averaged bipower variation | noise and jump robust |
| `bpv` | bipower variation | jump robust |
| `ctrv` | corrected threshold realized variance | jump robust |

Inputs are intraday **log returns**. TSRV is formed from staggered K-step price
grids; RK uses the corrected data-driven bandwidth; PAV/PABPV use the standard
triangular-weight constants `psi_1 = 1` and `psi_2 = 1/12`.

### Measurement-error correction

The correction keeps the signed moment gap

```text
delta_m = mean_Delta(m_a(2, Delta) - m_b(2, Delta))
omega_a^2 - omega_b^2 = delta_m / 2
```

and jointly fits lags 0, 1, and 2 of the increment autocovariances. Infeasible or
pathological corrected moment curves return `NaN` plus diagnostics rather than
being clipped to a tiny positive number. The intended empirical pair is RV5m
and RK.

## Install and verify

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

The Massive client is optional and only needed for the loader functions:

```bash
pip install massive
export MASSIVE_TOKEN="..."
```

The experiment scripts default to synthetic data and require no API key.

## Run the real-data SPY pilot without committing market data

The empirical runner writes all raw bars, daily market-derived series, receipts,
and figures to `.local/spy_empirical` by default. That directory and the known
empirical output basenames are gitignored.

```bash
pip install -r requirements-empirical.txt
export MASSIVE_TOKEN="..."
python exp/exp5_spy_empirical.py

# Recompute from an existing private cache with network access disabled:
python exp/exp5_spy_empirical.py --offline
```

Use `--output-dir /private/path` to keep the cache outside the checkout. Never
commit that directory. The repository contains only aggregate findings in
[`AUDIT_REPORT.md`](AUDIT_REPORT.md).

## Run the experiments

```bash
python exp/exp1_realized_hurst.py
python exp/exp2_mean_reversion.py
python exp/exp3_measurement_error.py
python exp/exp4_implied_hurst.py
```

Useful scale controls:

```bash
python exp/exp1_realized_hurst.py --n-per-class 4 --n-days 2520
python exp/exp3_measurement_error.py --n-paths 400 --no-plots
python exp/exp4_implied_hurst.py --n-dates 500
```

Experiment 3 is the expensive step. The checked-in audit result uses 20 paths;
the paper uses hundreds to thousands of paths depending on the table.

## Current audited results

- **Exp. 1:** all estimator medians remain in the rough regime, approximately
  `0.076-0.092`, but median synthetic fit quality is only `R² ≈ 0.58-0.74`.
  This validates estimator scale, not the paper's empirical cross-section.
- **Exp. 2:** `H=0.20`, `kappa=0.010`, `Delta_max=10` gives
  `H_hat=0.198305`, closely reproducing the paper's table.
- **Exp. 3:** at price noise `1e-4`, mean latent/RV5m/RK estimates are
  `0.192/0.175/0.183`; correction raises RK to `0.244` at lag 10 and `0.214`
  at lag 40, with a right-skewed distribution. At `5e-4`, raw attenuation is
  reproduced but the correction becomes unstable, as expected when the iid
  additive log-noise premise fails.
- **Exp. 4:** the strike-level synthetic pipeline identifies the ES-like and
  ZW-like power laws and correctly refuses to interpret rate, FX, and seasonal
  natural-gas analogs with near-zero pooled `R²`.
- **SPY real-data pilot:** all seven realized-variance estimators remain in the
  rough regime (`H=0.156-0.215`). Figure-compatible overlapping increments
  reproduce the paper's SPY fits at `H=0.188`, `0.163`, and `0.071` for windows
  `1-10`, `1-40`, and `40-250`, respectively. See the English audit report for
  scope and sensitivity details.

## Data and replication boundary

The optional API loaders are not a substitute for the paper's data construction.
A full empirical replication still requires, among other things:

- the 3,926-equity panel and its survivorship/quality screens;
- 34 continuous CME roots with the paper's volume-overtake roll rule;
- one-second liquid-equity checks;
- CME and OPRA option chains, put-call-parity forwards, and contract-style
  handling;
- asset-level diagnostics, robustness windows, and uncertainty estimates.

Until those panels are supplied and run through the audited functions, the
correct label is **audited validation harness with a single-instrument
real-data pilot**, not **full empirical replication**.

## Additive replication extension — September 24, 2026

`src/replication` adds a price-to-implied-H pipeline, a Gaussian fOU/fBm
generator, complete Table 3–6 scenario designs, and private CSV/Parquet panel
adapters. Existing `exp/` commands, result files and the historical
`AUDIT_REPORT.md` remain unchanged. The core correction still defaults to
`guarded`; the new runners explicitly select the documented `paper` mode.

See the [implementation guide](docs/REPLICATION_GUIDE.md) for input contracts,
filtering, calibration and known conventions, and the
[dated integration report](docs/AUDIT_REPORT_2026-09-24.md) for tests and limits.
These additions do not certify the paper's empirical results. In particular,
the historical Exp. 2 summary above is not a full Table 2 match: 14 of 48
printed cells disagree with independently checked integrals; the original
spectral integrator is retained.

## Local extension checks (no market access)

```bash
python -m pytest -q
python -m src.replication.monte_carlo --profile paper --dry-run
python -m src.replication.monte_carlo --profile smoke --output-dir .local/mc_smoke
python -m src.replication.monte_carlo --profile smoke --output-dir .local/mc_smoke --resume
python -m tests.run_file_smoke --output-dir .local/file_smoke
python -m src.replication.table2 --output-dir .local/table2_check
```

Parquet support is optional: see `requirements-replication.txt`. Detailed inputs
and outputs belong outside Git or in ignored `.local`. The paper dry-run only
lists the design; the smoke run is a small synthetic check, not a replication
of published Monte Carlo averages. Use a fresh output directory after code or
configuration changes; resume deliberately rejects mismatched fingerprints.
