# H and nu forecasts improve on persistence, without establishing superiority to HAR

## Outcome

The forecasting continuation completed on the existing sample through
**September 23, 2026**. It repeated the earlier experiment, evaluated a later
fixed-parameter window, and produced a single-origin forecast snapshot for
**102 securities**. No additional market observations were acquired.

- In the January–August repeat, the overlap-H/moment-nu model reduced one-day
  QLIKE by **31.49% versus persistence**, but was effectively tied with HAR:
  **0.150578 versus 0.150479**. The paired confidence interval includes zero.
- On the 17 later sessions, one-day performance was again close to HAR.
  At 22 sessions, rough-model QLIKE was **33.20% higher than HAR**. This short
  window does not establish a durable ranking, but it argues against claiming
  universal rough-model superiority.
- Interval calibration was horizon-dependent: stock WIS improved **1.49% at
  22 sessions**, but worsened **1.01% at five sessions**. Higher coverage alone
  is not evidence of better intervals.
- SPY's latest all-history estimate is **H = 0.1804**, with innovation
  **nu_logVol = 0.3710**. The separate moment-nu estimate is **0.3732**.

```yaml
status: available_sample_forecasts_evaluated
fully_reproduced: false
```

This is a forecasting extension on the available equity sample, not confirmation
of the paper's full empirical results or a trading-performance claim.

## Scope and interpretation

The [fixed protocol](FORECAST_PROTOCOL_2026-09-25.md) separates three calculations.
The **January 1–August 28** target window has already been inspected in earlier
work; its new calculation is a retrospective repeat, not a fresh untouched test.
The **August 31–September 23** window uses previously saved rough parameters and
HAR coefficients. The final snapshot reconstructs predictions **as of the close
of September 23**; it does not claim they were issued in real time. In particular,
the September 24 target is already historical at this report's preparation date,
but no observation for it enters this experiment.

The target is single-session positive daily TSRV at t+h, not cumulative variance
over h sessions. Forecasting operates on log(TSRV), preserving missing calendar
positions. The same audited session processing is retained. Full-model fitting
ends in 2025 for retrospective scoring; strict split models fit through 2024 and
calibrate on 2025. H-grid selection uses only 2024 validation. No reported test
score chooses a model or setting.

Across 106 security definitions, 76,331,183 existing minute observations were
processed, including 7,207,790 in 2026. The shared calendar has 2,111 sessions.
The run evaluated 423,636 native retrospective predictions and 19,992 native
later-window predictions. Counts include models and horizons, not independent
market observations. The released snapshot contains 918 predictions:
102 securities × three model variants × three horizons.

The matched retrospective quality cohort contains **92 stocks**; the later
quality cohort contains **90**. Stocks have equal weight; SPY is separate.
Within each window and horizon, models use identical security/origin/target
rows. The windows have different eligibility and must not be treated as a
constant-composition trend. Native counts and the quality-without-history-boundary
sensitivity are retained in the aggregate files.

The earlier repeat used 90 quality stocks; the current one includes two repaired
histories. Old reports are unchanged. ARM, BKNG and LIN lack at least one model
or sufficient calibration history for the nine-model common intersection.
New securities and repaired Fiserv/LRCX histories are excluded from the later
fixed-parameter comparison; inactive EA is also excluded. The 97-stock later
intersection includes 90 original-quality stocks, plus SPY reported separately.

## Retrospective point forecasts

Equal-stock QLIKE on the 92-stock quality intersection; lower is better.
“Rough” here means overlap H with moment nu. Target-session counts are 165,
161 and 144 for horizons 1, 5 and 22 because origin cutoffs are respected.

| Horizon | Persistence | HAR | Rough | Rough minus HAR, paired 95% interval |
|---|---:|---:|---:|---:|
| 1 | 0.219786 | 0.150479 | 0.150578 | +0.000099 [-0.001313, 0.001539] |
| 5 | 0.303080 | 0.190044 | 0.187344 | -0.002699 [-0.009178, 0.004546] |
| 22 | 0.450192 | 0.259016 | 0.243822 | -0.015194 [-0.049101, 0.019043] |

All three rough-minus-persistence intervals are below zero under the specified
calendar-block bootstrap. All three rough-minus-HAR intervals contain zero.
The evidence supports improvement over persistence, not an established edge
over HAR. These are approximate, unadjusted, dependence-aware intervals on a
previously inspected test window, not proof of a universal ranking.

Replacing moment nu by innovation nu makes only small descriptive differences:
one-day QLIKE is 0.150652 and WIS is 0.190948, versus 0.150578 and 0.191028 for
moment nu. The original non-overlap variant and training-selected H-grid variant
are retained in the complete comparison; neither is silently discarded.

## What calibration did to prediction intervals

This comparison changes only interval radii for the same strict-split rough
model. Prediction levels are 50%, 80%, 90% and 95%; WIS combines those levels
and penalizes both excess width and missed targets.

| Horizon | 90% coverage, Gaussian → calibrated | 95% coverage, Gaussian → calibrated | WIS, Gaussian → calibrated | 90% log-width, Gaussian → calibrated |
|---|---:|---:|---:|---:|
| 1 | 91.42% → 92.73% | 94.91% → 96.96% | 0.190740 → 0.191191 | 1.6608 → 1.7504 |
| 5 | 92.85% → 94.82% | 95.86% → 98.05% | 0.214855 → 0.217020 | 1.9985 → 2.2175 |
| 22 | 93.58% → 94.27% | 96.37% → 97.44% | 0.249068 → 0.245361 | 2.4212 → 2.4751 |

The paired calibrated-minus-Gaussian WIS intervals are:

- One session: +0.000451 [-0.000104, 0.001064], inconclusive.
- Five sessions: +0.002164 [0.000647, 0.003514], worse under this protocol.
- Twenty-two sessions: -0.003706 [-0.005341, -0.002142], better under this protocol.

The stock intervals already overcover at several levels. Making them wider
can worsen accuracy even when fewer observations fall outside. Results also
differ by security: SPY's one-day coverage rises from 87.27% to 92.12%, while
its 22-session WIS worsens from 0.357317 to 0.363085 after calibration.
The stock-average 22-session improvement is not a claim about every security.

For subsequent experiments, retain **innovation nu as a scale estimate**, and
keep calibration disjoint from fitting. Assess radii by horizon with both
coverage and WIS; do not select a new calibration rule on these test outcomes
and continue calling the same observations untouched OOS. No automatic switch
to the best-looking variant was made here.

## Later fixed-parameter window

The 90 quality stocks provide 1,530 matched rows per model/horizon across
17 target sessions. Parameters are frozen; the table is descriptive only.
No confidence interval based on 22-session blocks is claimed for this shorter
window. Multi-session horizons overlap and are not independent observations.

| Horizon | Persistence QLIKE | HAR QLIKE | Rough moment-nu QLIKE | Rough innovation-nu QLIKE | 90% coverage, HAR / rough moment |
|---|---:|---:|---:|---:|---:|
| 1 | 0.222383 | 0.141378 | 0.141785 | 0.141747 | 92.81% / 92.55% |
| 5 | 0.272045 | 0.157656 | 0.160095 | 0.159877 | 94.64% / 95.29% |
| 22 | 0.315737 | 0.188891 | 0.251598 | 0.250672 | 96.14% / 95.62% |

At 22 sessions HAR also has better WIS: 0.229394 versus 0.248624 for rough
moment nu. For SPY alone rough moment nu has lower QLIKE than HAR at all three
horizons in this later window (0.237778/0.209423/0.507001 versus
0.250840/0.254035/0.561023). Neither the favorable SPY result nor the unfavorable
stock-average result warrants extrapolating from 17 sessions.

## Updated H, nu and parameter uncertainty

All-history estimates through the cutoff, using overlap increments:

| Security | H | 95% block-bootstrap interval for H | Innovation nu_logVol |
|---|---:|---:|---:|
| SPY | 0.1804 | [0.1543, 0.2096] | 0.3710 |
| AAPL | 0.1629 | [0.1296, 0.1904] | 0.3097 |
| MSFT | 0.1584 | [0.1327, 0.1851] | 0.2960 |
| NVDA | 0.1394 | [0.1141, 0.1641] | 0.3039 |
| AMZN | 0.1576 | [0.1329, 0.1840] | 0.3024 |
| TSLA | 0.1448 | [0.1232, 0.1665] | 0.3066 |
| GEHC | 0.1187 | [0.0733, 0.1693] | 0.3323 |
| XEL | 0.1586 | [0.1188, 0.1979] | 0.2580 |

For SPY, moment nu_logVol is 0.3732 with a 95% bootstrap interval
**[0.3542, 0.3902]**. This interval does **not** cover the separately estimated
innovation nu by construction. The bootstrap jointly resamples lag-moment
vectors: its H/nu correlation is -0.4864, so treating the parameter estimates as
independent would lose relevant dependence. Common-start bootstrap point H is
0.1811, slightly different from the all-pairs 0.1804 above.

The trailing-504 sensitivity gives SPY H = 0.1801 and innovation nu_logVol =
0.4076: similar estimated roughness but a larger innovation scale. Full-history,
trailing-window and March-split estimates are retained separately. The intervals
describe measured TSRV scaling, not noise-free latent volatility parameters;
conditional prediction intervals do not integrate parameter uncertainty.

## Single-origin forecast snapshot

SPY, reconstructed at the September 23 close, using the March-31 fit with
subsequent disjoint calibration. Values are **annualized volatility percent**
for the single target session; the point is expected volatility, not the square
root of expected variance. These snapshot outcomes have not been scored here.

| Horizon | Target date | Expected volatility | Central 90% prediction interval |
|---|---|---:|---:|
| 1 | 2026-09-24 | 6.06% | [3.27%, 10.05%] |
| 5 | 2026-09-30 | 6.30% | [2.89%, 11.37%] |
| 22 | 2026-10-23 | 7.01% | [2.78%, 12.69%] |

All 102 eligible securities have all nine planned snapshot predictions. No
snapshot is issued for ALGN (last available observation January 31, 2024), EA
(August 4, 2026), CDW or VRSK (no supplied observations). Historical parameter
estimates for stale securities are retained and explicitly flagged; they are
not fresh forecasts. GEHC and XEL have current snapshots but are not added to
the original-stock score comparisons.

## Verification and released evidence

The standard suite passed **260 tests**, plus **21 forecasting tests** covering
the inherited models, prepared refresh and continuation. Future-value
perturbations leave fitted pre-OOS parameters unchanged. Missing-day and
stale-origin cases, model intersections and short-window uncertainty are tested.

All 106 paper-period source fingerprints match the preceding validation.
The independent OOS scan found zero invalid time-grid or price/volume rows.
TSRV was cross-checked against the physical-grid adapter on 2,254 sessions,
maximum absolute difference 8.67 × 10⁻¹⁹. Independent dense conditioning matched
all 918 latest forecasts, maximum mean/variance difference 5.33 × 10⁻¹⁵.
QLIKE arithmetic and temporal bounds were checked on every evaluated row;
WIS was independently checked on every retrospective row;
all published equal-security QLIKE/WIS/coverage aggregates were independently
reconciled. Complete packet hashes and deployed-code hashes were verified.

Released files contain aggregates and a single-origin forecast snapshot, not
observed bars, returns, daily RV, dated historical predictions or per-date scores:

- [Acceptance and fingerprints](validation/2026-09-25/forecast/acceptance.json)
- [All securities, H/nu and parameter intervals](validation/2026-09-25/forecast/securities.csv)
- [Snapshot forecasts and prediction intervals](validation/2026-09-25/forecast/forecast_snapshot.csv)
- [Model/cohort comparisons](validation/2026-09-25/forecast/model_comparison.csv)
- [Paired score uncertainty](validation/2026-09-25/forecast/score_uncertainty.json)
- [Per-security aggregate scores and native/matched counts](validation/2026-09-25/forecast/security_scores.csv)
- [Eligibility and exclusions](validation/2026-09-25/forecast/exclusions.json)

The selected historical universe, incomplete extensions, adjusted-history
vintages, proxy-model assumptions and short later window remain limitations.
The existing architecture, estimator defaults and historical results are
unchanged. Full cross-asset replication remains unclaimed.
