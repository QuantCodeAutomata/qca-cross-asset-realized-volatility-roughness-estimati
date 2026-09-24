# Equity roughness on the available minute-price sample

## Outcome

The updated methods were run on the available sample for **May 1, 2018 through
December 31, 2025**. The original comparison group is SPY plus 100 stocks;
five additional stock definitions are reported separately rather than silently
changing that group. 104 of the 106 definitions have observations
in this window. No additional market observations were acquired.

The quality-screened original stocks have median TSRV H **0.1323**
with non-overlapping increments and **0.1356** with overlapping
increments, compared with **0.131** in the paper. This is a comparison on a
different, fixed liquid-stock sample, not equality of the underlying populations.

All 4,368 attempted H/R² fits passed an independent numerical
cross-check. This establishes execution and numerical consistency on the
available sample; it does not establish every conclusion of the paper.

The low-H equity finding is supported on this sample, but not every benchmark
matches: the primary quality-sample median R² is 0.7890 versus 0.988 in the
paper, and the lag-10 corrected RK median is 0.1611 versus 0.140. That primary
correction is available for 91/93 quality stocks; two optimizations failed.

```yaml
status: available_sample_validated
fully_reproduced: false
```

The repository remains suitable for this scoped empirical validation. The full
cross-asset and option claims, author simulation conventions and unresolved
Table 2 discrepancies are not promoted to reproduced results.

## Sample and computation

- 69,123,393 adjusted one-minute price observations were processed,
  on a shared calendar of 1,929 SPY sessions.
- Each session uses a fixed 390-minute regular-session grid; intraday gaps are
  forward-filled only. Leading gaps remain missing; there is no overnight return.
  At least 120 observed minutes and positive total volume are required per day.
- Five-minute phases remain anchored to the original session grid. All seven
  estimators are evaluated: RV5m, RK, TSRV, PAV, PABPV, BPV and C-TRV.
- Missing daily values remain on the calendar. Nonpositive variance estimates
  are not logged or replaced by artificial positive constants.
- H is half the OLS slope of log second moments against log lag, for windows
  1–10, 1–40 and 40–250 sessions. A reported summary requires the complete lag
  window. Non-overlapping increments remain primary; overlap is a sensitivity,
  not a convention selected after seeing agreement with the paper.
- The main quality screen requires at least 500 positive-TSRV estimation days
  and at least half of those days with positive volume in 80% of grid minutes.
  A separate observed-minute screen and a stricter 500-consecutive-positive-day
  screen expose the differing wording of Sections 3.1, 3.2 and 6.1.

The fixed original stock list was selected using historical proxy membership
and 2024 liquidity. It is not a point-in-time reconstruction of the paper's
3,926-asset universe. Existing history boundaries and incomplete extensions are
retained, not interpreted as verified listing/delisting dates. These selection
and availability limits preclude generalizing the sample median to all equities.

## SPY and Figure 1

| Lag window | Paper H | Non-overlapping H / R² | Overlapping H / R² |
|---|---:|---:|---:|
| 1–10 | 0.182 | 0.1784 / 0.9391 | 0.1877 / 0.9922 |
| 1–40 | 0.160 | 0.1366 / 0.6785 | 0.1631 / 0.9883 |
| 40–250 | 0.071 | 0.0587 / 0.0119 | 0.0709 / 0.7356 |

Overlap better matches the joint H/R² pattern in Figure 1, especially at the
medium and long windows, but is not an exact match. At lag 10, non-overlap H
alone is closer to the printed H, while overlap R² is closer. The paper's R² values are
0.991, 0.991 and 0.753. The non-overlapping long-window fit is much weaker.
This preserves the unresolved increment-convention issue rather than hiding it.

## Stock cross-section

SPY is excluded from all stock medians. The extension is excluded from the
original-sample comparisons. IQR describes dispersion across securities, not
a confidence interval for a latent H or for the cross-sectional median.

| Original-sample screen | Valid stocks | TSRV H median, non-overlap | IQR | Median R², non-overlap | TSRV H median, overlap |
|---|---:|---:|---:|---:|---:|
| All available original stocks | 100 | 0.1309 | [0.1147, 0.1456] | 0.7893 | 0.1354 |
| Quality: traded minutes | 93 | 0.1323 | [0.1162, 0.1454] | 0.7890 | 0.1356 |
| Quality: observed minutes | 93 | 0.1323 | [0.1162, 0.1454] | 0.7890 | 0.1356 |
| 500 consecutive positive days | 98 | 0.1309 | [0.1152, 0.1459] | 0.7893 | 0.1351 |
| Consecutive + traded quality | 93 | 0.1323 | [0.1162, 0.1454] | 0.7890 | 0.1356 |

For reference, Section 6.1 reports 0.119 for all stocks and 0.131 for its
quality subset, IQR [0.117, 0.145] and median R² 0.988 in that quality subset.
The range of primary lag-10 TSRV H across our original sample is
[0.0705, 0.1886]. The estimates are descriptive;
this run does not add dependence-aware parameter confidence intervals.
The overlapping quality-sample median R² is 0.9882, close to the paper's 0.988;
this agreement does not resolve which increment convention the author used.

### Estimator sensitivity in the original traded-minute quality subset

| Estimator | Valid stocks | Median H, non-overlap | Median H, overlap |
|---|---:|---:|---:|
| RV5M | 93 | 0.1191 | 0.1274 |
| RK | 93 | 0.1463 | 0.1528 |
| TSRV | 93 | 0.1323 | 0.1356 |
| PAV | 93 | 0.1073 | 0.1129 |
| PABPV | 93 | 0.1090 | 0.1163 |
| BPV | 93 | 0.1598 | 0.1647 |
| CTRV | 93 | 0.1755 | 0.1847 |

The paper gives endpoint medians 0.114 for PAV and 0.175 for C-TRV. These are
reference values, not an acceptance tolerance chosen for this different sample.

## RV5m/RK measurement-error correction

Both `paper` and `guarded` modes are retained. Failed optimization iterates
never enter corrected medians. Invalid corrected curves remain unavailable.
The count in each row is the valid RK-leg count; short/long outcomes are separate.
For lag 10 the raw comparison uses precisely the same successful instruments
and the same finite-observation mask as the corrected RK leg.

| Mode | Increments | Window | Valid / attempted | Optimizer failures | Corrected RK median | Matched raw RK median (lag 10) |
|---|---|---:|---:|---:|---:|---:|
| paper | non-overlap | 1–10 | 91 / 93 | 2 | 0.1611 | 0.1463 |
| paper | non-overlap | 1–40 | 91 / 93 | 2 | 0.1235 | unavailable |
| paper | overlap | 1–10 | 89 / 93 | 4 | 0.1629 | 0.1523 |
| paper | overlap | 1–40 | 89 / 93 | 4 | 0.1414 | unavailable |
| guarded | non-overlap | 1–10 | 89 / 93 | 4 | 0.1615 | 0.1464 |
| guarded | non-overlap | 1–40 | 89 / 93 | 4 | 0.1235 | unavailable |
| guarded | overlap | 1–10 | 90 / 93 | 3 | 0.1648 | 0.1533 |
| guarded | overlap | 1–40 | 90 / 93 | 3 | 0.1413 | unavailable |

Section 6.1 reports corrected RK medians 0.140 (lag 10) and 0.123 (lag 40).
Differences here must not be removed by changing bounds, deleting unsuccessful
cases, or switching the displayed specification. The correction is a sensitivity
to measurement error, not a directly observed latent-volatility parameter.

## Extension and instrument list

The extension is a separate availability check, not a source of additional
observations for the headline original-sample comparison.

| Extension stock | Observed days | Positive TSRV days | Traded-minute quality | TSRV H10, non-overlap |
|---|---:|---:|---|---:|
| ALGN | 1448 | 1448 | True | 0.0801 |
| CDW | 0 | 0 | False | unavailable |
| GEHC | 733 | 733 | True | 0.1035 |
| VRSK | 0 | 0 | False | unavailable |
| XEL | 1929 | 1929 | True | 0.1668 |

Original stocks: AAL, AAPL, ABNB, ADBE, ADI, ADP, ADSK, AEP, AMAT, AMD, AMGN, AMZN, APP, ARM, ASML, AVGO, AZN, BIIB, BKNG, BKR, CDNS, CEG, CHTR, CMCSA, COST, CPRT, CRWD, CSCO, CSX, CTAS, CTSH, DASH, DDOG, DLTR, DXCM, EA, EBAY, EXC, EXPE, FANG, FAST, FISV, FTNT, GILD, GOOG, GOOGL, HON, IDXX, ILMN, INTC, INTU, ISRG, KDP, KHC, KLAC, LIN, LRCX, LULU, MAR, MCHP, MDB, MDLZ, MELI, META, MNST, MRNA, MRVL, MSFT, MU, NFLX, NVDA, NXPI, ODFL, ON, ORLY, PANW, PAYX, PCAR, PDD, PEP, PYPL, QCOM, REGN, ROP, ROST, SBUX, SNPS, SWKS, TEAM, TMUS, TSLA, TTD, TTWO, TXN, ULTA, VRTX, WBD, WDAY, WDC, ZS. Benchmark: SPY.

Defined instruments without paper-period rows: CDW, VRSK.
Instruments with observations but fewer than 500 positive-TSRV days: ARM.

The [security-level aggregate table](validation/2026-09-25/available_sample_securities.csv)
includes coverage endpoints, counts, screening flags and H/R² summaries. It
contains no price, return, daily-RV or forecast time series.

## Validation and limits

- The unchanged implementation passed **260 tests**, with no failures or skips.
- All 4,368 attempted H/R² fits were compared with separately
  constructed lag pairs and a polynomial least-squares fit. Maximum absolute
  errors were 5.27e-16 for H and 6.66e-16 for R²;
  valid/unavailable status was checked as well.
- The streamed session mapping agreed with the accepted file adapter on
  520 sampled sessions across observed securities,
  for all seven RV estimators. This is a sampled adapter check, not an
  independent reimplementation of every estimator on every day.
- An independent aggregate scan reconciled the consumed observation count and
  found zero timestamp-grid and zero finite-price/OHLC/volume violations.
  The input's last update preceded the calculation snapshot.
- Aggregate packets were checksummed. The [machine-readable receipt](validation/2026-09-25/available_sample.json)
  records code/input digests, dependencies, valid counts and scope.

No implementation, APIs, historical experiments or results were replaced.
Only aggregate evidence, this report and a short README status update are added. No price or volatility time
series, connection details or operational infrastructure metadata are included.

This run does **not** execute the original futures or option panels, the
one-second experiment, out-of-sample forecasting or the full Table 3–6 Monte
Carlo design. Those are not inferred from successful minute-equity calculations.
The earlier [integration report](AUDIT_REPORT_2026-09-24.md) remains the record
for method tests and synthetic scenarios. Its Table 2 finding remains unchanged:
34/48 printed cells agree; 14 differ, with independent numerical support for
the retained integrator.

Paper comparisons were checked against the supplied *Rough Volatility Across
Assets*, Figure 1 (page 7) and Section 6.1/Figure 3 (page 16). They are not
source-equivalent replication tests or proof of a forecasting advantage.
