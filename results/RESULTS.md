# Audited experiment results

## Overall status

**Repository status after audit:** credible synthetic/numerical validation harness.

**Paper-replication status:** **not empirically replicated**. The repository does
not contain the paper's Nasdaq, CME, or OPRA panels. Only Experiment 2 is a
direct numerical reproduction of a paper calculation; Experiments 1, 3, and 4
are controlled synthetic validations.

## Experiment 1 - realized-H estimator recovery

### Question tested

Can the seven realized-variance estimators and the short-lag H regression recover
rough synthetic daily volatility at approximately the right scale?

### Configuration

- 8 synthetic asset classes centered on the paper's reported class medians;
- 2 assets per class, 16 assets total;
- 1,000 trading days and 390 returns per day;
- daily fOU parameters in trading-day units;
- log-variance standard deviation 1.05 and mean annualized volatility 12.8%;
- non-overlapping `Delta=1,...,10` scaling regression.

### Estimator results

| estimator | median H | median bias | mean absolute error | median R2 |
|---|---:|---:|---:|---:|
| BPV | 0.0917 | +0.0037 | 0.0223 | 0.7416 |
| C-TRV | 0.0921 | +0.0037 | 0.0229 | 0.7385 |
| PABPV | 0.0759 | -0.0158 | 0.0247 | 0.5752 |
| PAV | 0.0792 | -0.0110 | 0.0247 | 0.5928 |
| RK | 0.0892 | -0.0001 | 0.0237 | 0.6906 |
| RV5m | 0.0865 | -0.0019 | 0.0245 | 0.6863 |
| TSRV | 0.0866 | -0.0032 | 0.0243 | 0.6559 |

### Interpretation

The implementation now recovers the rough regime without the previous PAV,
PABPV, TSRV, and RK normalization defects. It does **not** reproduce the paper's
empirical cross-section:

- the class labels are simulation targets, not observed asset classes;
- two paths per class are insufficient to validate a cross-sectional ordering;
- median synthetic `R2` is about `0.58-0.74`, materially below the paper's
  quality-equity median of about `0.988`;
- no equity quality subset, one-second subset, delisting treatment, or futures
  roll panel is present.

**Classification:** synthetic implementation validation only.

## Experiment 2 - mean-reversion contamination

### Question tested

Does the exact stationary-fOU second moment produce the paper's reported small
short-lag downward bias and increasing long-window contamination?

### Key cells

| H | kappa | Delta max | exact H hat | bias |
|---:|---:|---:|---:|---:|
| 0.10 | 0.020 | 10 | 0.097871 | -0.002129 |
| 0.20 | 0.010 | 10 | 0.198305 | -0.001695 |
| 0.20 | 0.020 | 10 | 0.195405 | -0.004595 |
| 0.20 | 0.010 | 40 | 0.191172 | -0.008828 |
| 0.20 | 0.020 | 40 | 0.178486 | -0.021514 |
| 0.20 | 0.020 | 100 | 0.145799 | -0.054201 |

The primary benchmark, `H=0.20`, `kappa=0.010`, `Delta_max=10`, gives
`H_hat=0.198305`, very close to the paper's `0.1982` table value. The lag-10
bias remains below 0.005 over the paper's core rough region
`H<=0.20, kappa<=0.020`, while longer windows bend downward strongly.

**Classification:** strong direct numerical reproduction.

## Experiment 3 - measurement error and correction

### Question tested

Does one-minute realized-volatility measurement error attenuate H, and does the
paper's RV5m/RK two-estimator correction raise it under the baseline noise
regime without manufacturing pathological outputs?

### Configuration

- 20 paired Monte Carlo paths;
- 2,520 days and 390 one-minute returns per day;
- `H=0.20`, `kappa=0.010` per trading day;
- latent log-variance standard deviation 1.05;
- mean annualized volatility 12.8%;
- price noise `varpi in {1e-4, 5e-4}`;
- correction pair RV5m/RK;
- corrected curves refit at lags 10 and 40.

### Results

| varpi | latent H10 | RV5m H10 | RK H10 | corrected RK H10 | corrected RK H40 | valid primary correction |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4, mean | 0.1918 | 0.1750 | 0.1827 | 0.2444 | 0.2145 | 100% |
| 1e-4, median | 0.1903 | 0.1739 | 0.1822 | 0.2042 | 0.1902 | - |
| 5e-4, mean | 0.1918 | 0.1463 | 0.1724 | 0.3965 | 0.2897 | 95% |
| 5e-4, median | 0.1903 | 0.1443 | 0.1714 | 0.3595 | 0.2669 | - |

At baseline noise, raw RV5m and RK are close to the paper's Table 4 values
(approximately 0.181 and 0.180). The correction moves H upward, but its mean is
right-skewed; the median lag-10 correction is 0.204 rather than 0.244. This is a
weakly identified bracket, not a stable point estimate.

At stressed noise, the raw attenuation remains informative, but the iid additive
log-noise assumption fails. The high corrected values are therefore evidence of
correction breakdown, not successful recovery.

The submitted implementation previously emitted corrected H values above 4 and
6 after clipping non-positive moments to `1e-12`. The audited implementation
returns `NaN` and diagnostics for infeasible or non-interpretable corrected
curves, and all finite corrected H values are constrained to `(0,1)`.

**Classification:** partial synthetic reproduction of attenuation; correction
behavior remains fragile, consistently with the paper's warning.

## Experiment 4 - option-skew identification

### Question tested

Can a strike-level synthetic option pipeline first estimate the ATM skew and then
correctly identify, or reject, its maturity power law?

### Results

| underlier | synthetic structure | H hat IV | target | pooled R2 | identified |
|---|---|---:|---:|---:|:---:|
| ES-like | power law | 0.2486 | 0.25 | 0.6031 | yes |
| ZW-like | power law | 0.0827 | 0.08 | 0.5335 | yes |
| ZN-like | weak skew | 0.5231 | - | 0.0005 | no |
| 6E-like | weak skew | 0.5191 | - | 0.0004 | no |
| NG-like | seasonal maturity pattern | 0.5148 | - | 0.0003 | no |

For the unidentified rows, the reported numerical H is intentionally treated as
meaningless: the fitted slope explains essentially none of the within-date
maturity variation. This is the correct interpretation of the paper's rate, FX,
and seasonal-commodity failure taxonomy.

The submitted experiment had generated class-level power laws directly, so the
estimated H was largely encoded in the data generator; it also declared a
stylized energy class strongly identified. The audited experiment starts from
strike-level smiles and deliberately includes weak and seasonal failure cases.

**Classification:** synthetic end-to-end identification test, not an option-data
replication.

## Final assessment

The corrected code can support the following claims:

1. the short-lag H estimator and seven RV measures are implemented at a credible
   numerical scale;
2. the paper's fOU mean-reversion table is reproduced closely;
3. the simulated measurement-error attenuation is reproduced near the paper's
   benchmark after fixing units and volatility calibration;
4. the correction is fragile and should be reported as a bracket with validity
   diagnostics;
5. the option-skew method must be rejected when its pooled maturity regression
   has negligible explanatory power.

It cannot support the following claims without real data:

1. realized volatility is rough across 3,926 equities and 34 CME roots;
2. the paper's asset-class ranking is independently reproduced;
3. the quality-equity median and `R2` distribution are reproduced;
4. the empirical implied-versus-realized results for 41 underlyings are
   reproduced;
5. robustness to the paper's data filters, contract rolls, option-chain
   construction, and one-second sampling is established.

**Independent verdict:** the submitted repository was not a paper replication;
it was a partly broken synthetic demonstration. After the audit it is a useful
and testable validation harness, but the empirical replication remains open.
