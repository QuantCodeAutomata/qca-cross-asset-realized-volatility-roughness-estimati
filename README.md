# Cross-Asset Realized-Volatility Roughness Estimation

Empirical study of realized-volatility roughness across equities, equity indices,
fixed-income, commodity, and FX futures using the Hurst exponent framework of
Gatheral, Jaisson & Rosenbaum (2018).  Seven noise-robust realized-variance
estimators are used to construct daily log-RV series, whose Hurst exponent is
estimated via second-moment log-log regression.

---

## Project Structure

```
.
├── data/
│   ├── __init__.py
│   └── data_loader.py          # Massive API loader + synthetic data generator
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── equity.py           # Equity intraday preprocessing
│   │   └── futures.py          # Futures roll & preprocessing
│   ├── rv_estimators.py        # 7 realized-variance estimators
│   ├── hurst_realized.py       # Hurst estimation via scaling regressions
│   ├── aggregation.py          # Daily → weekly/monthly RV aggregation
│   └── plots_realized.py       # Publication-quality figures
├── exp/
│   └── exp1_realized_hurst.py  # Experiment 1: cross-sectional Hurst
├── tests/
│   ├── __init__.py
│   └── test_rv_estimators.py   # Pytest unit tests
├── results/                    # Auto-generated outputs (gitignored except RESULTS.md)
├── README.md
└── .gitignore
```

---

## Realized-Variance Estimators

| Symbol  | Name                                 | Noise-robust | Jump-robust |
|---------|--------------------------------------|:------------:|:-----------:|
| RV5m    | 5-minute subsampled RV               | ✓ (mild)     | ✗           |
| RK      | Realized Kernel (Parzen)             | ✓✓           | ✗           |
| TSRV    | Two-Scale Realized Variance          | ✓✓           | ✗           |
| BPV     | Bipower Variation                    | ✗            | ✓           |
| PAV     | Pre-Averaged Variance                | ✓✓           | ✗           |
| PABPV   | Pre-Averaged Bipower Variation       | ✓✓           | ✓           |
| C-TRV   | Corrected Threshold Realized Variance| ✗            | ✓✓          |

---

## Experiments

### Experiment 1 — Cross-Sectional Realized Hurst (Equities)
**Script:** `exp/exp1_realized_hurst.py`

Estimates the Hurst exponent H for a cross-section of equities (or synthetic
analogs) using all seven RV estimators.  The scaling regression

    log m(2, Δ) = a + b · log Δ,   H = b / 2

is run for Δ = 1, …, 10 trading days.  Output: distribution of H across assets,
table of median H by estimator, and representative log-log scaling plots.

**Key finding (target):** Equity realized volatility is rough (H ≈ 0.10–0.15),
consistent with Gatheral et al. (2018), and this roughness is robust across all
seven estimators.

---

### Experiment 2 — Futures Asset-Class Comparison
**Script:** `exp/exp2_futures_hurst.py` *(forthcoming)*

Extends the Hurst analysis to continuous front-month futures across five asset
classes: equity indices, fixed income, energy, metals, and FX.  Uses the
volume-overtake roll rule implemented in `src/preprocessing/futures.py`.

**Key finding (target):** Equity-index futures share the low-H regime of single
stocks.  Fixed-income and FX futures exhibit slightly higher H (≈ 0.15–0.25)
while energy and metals span a wider range.

---

### Experiment 3 — Temporal Aggregation and Hurst Stability
**Script:** `exp/exp3_aggregation_hurst.py` *(forthcoming)*

Tests whether the estimated H is stable across daily, weekly, and monthly
aggregation of realized variance using the utilities in `src/aggregation.py`.
Under the rough-volatility model, H should be invariant to aggregation level.

**Key finding (target):** H estimates are stable (within one standard error)
across aggregation frequencies, confirming the power-law scaling property of the
rough volatility model.

---

### Experiment 4 — Estimator Robustness to Microstructure Noise
**Script:** `exp/exp4_noise_robustness.py` *(forthcoming)*

Uses the synthetic data generator (`data/data_loader.py::generate_synthetic_intraday`)
with a known ground-truth H to benchmark estimator bias and variance under
increasing levels of microstructure noise.

**Key finding (target):** Noise-robust estimators (RK, TSRV, PAV, PABPV)
recover the true H with negligible bias; RV5m shows upward bias for H when noise
is large, while BPV and C-TRV are noisy but unbiased.

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scipy statsmodels matplotlib seaborn massive

# Run Experiment 1
python exp/exp1_realized_hurst.py
```

Results are written to `results/exp1_results.csv` and `results/RESULTS.md`.

---

## Data

Real data is loaded from the Massive API.  Set the `MASSIVE_TOKEN` environment
variable before calling any loader function:

```bash
export MASSIVE_TOKEN="your_api_key"
```

Synthetic data requires no API key and is the default for all experiment scripts.

---

## References

- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). *Volatility is rough.*
  Quantitative Finance, 18(6), 933–949.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008).
  *Designing realized kernels to measure the ex-post variation of equity prices
  in the presence of noise.* Econometrica, 76(6), 1481–1536.
- Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). *A tale of two time
  scales.* JASA, 100(472), 1394–1411.
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). *Power and bipower variation
  with stochastic volatility and jumps.* Journal of Financial Econometrics.
- Jacod, J., Li, Y., Mykland, P. A., Podolskij, M., & Vetter, M. (2009).
  *Microstructure noise in the continuous case.* Stochastic Processes and their
  Applications, 119(8), 2249–2276.
- Corsi, F., Pirino, D., & Renò, R. (2010). *Threshold bipower variation and
  the impact of jumps on volatility forecasting.* Journal of Econometrics.
- Wood, A. T. A., & Chan, G. (1994). *Simulation of stationary Gaussian
  processes in [0, 1]^d.* Journal of Computational and Graphical Statistics.
