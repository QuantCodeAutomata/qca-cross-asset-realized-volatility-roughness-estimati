# Audit and Real-Data Validation Report

## Technical summary

The corrected repository is a credible synthetic and numerical validation
harness, and its realized-volatility estimators now run end to end on real
one-minute market data. A private SPY pilot over the paper's full equity window
(`2018-05-01` through `2025-12-31`) confirms the central short-lag result:
realized volatility is rough, with a TSRV estimate near `H = 0.18`.

The strongest result is a near-reproduction of all three SPY fits shown in the
paper's Figure 1 when daily increments are overlapping. The repository's
audited primary convention remains non-overlapping because that is the literal
reading of the paper's displayed estimator. Both conventions are therefore
reported; neither is silently treated as uniquely specified by the paper.

This is not a full empirical replication. The private run covers one instrument
from Massive adjusted aggregates, not the paper's 3,926-equity XNAS.ITCH panel,
CME futures panel, or option data. No raw bars, daily realized-variance panel,
cache files, credentials, or machine-local paths are committed. This report
contains aggregate estimates only.

## Figure 1 is nearly reproduced under overlapping increments

The paper's third SPY regression uses lags `40-250`, not `1-100`. Using that
window and all overlapping daily pairs produces a close match in both H and fit
quality:

| Specification | H(1-10) / R2 | H(1-40) / R2 | H(40-250) / R2 |
|---|---:|---:|---:|
| Paper, Figure 1 | 0.182 / 0.991 | 0.160 / 0.991 | 0.071 / 0.753 |
| SPY, overlapping | 0.1877 / 0.9922 | 0.1631 / 0.9883 | 0.0709 / 0.7356 |
| SPY, strict 390-bar days, overlapping | 0.1883 / 0.9908 | 0.1578 / 0.9848 | 0.0766 / 0.7689 |
| SPY, audited non-overlapping primary | 0.1784 / 0.9391 | 0.1366 / 0.6785 | 0.0587 / 0.0119 |

The non-overlapping curve becomes noisy at long lags because it uses only one
fixed phase and the number of available pairs falls sharply. The overlapping
result shows that increment convention explains most of the earlier SPY gap;
small residual differences may still reflect provider, adjustment, timestamp,
and trade-aggregation conventions.

No market-derived chart is checked in because the handoff intentionally excludes
the underlying and derived market panel. The exact aggregate values needed to
audit the conclusion are included in the table instead.

## All seven realized-variance estimators remain in the rough regime

For the primary sample and non-overlapping lags `1-10`, all daily estimators
produce H estimates well below `0.5`:

| Estimator | H | R2 | Valid sessions |
|---|---:|---:|---:|
| TSRV | 0.1784 | 0.9391 | 1,929 |
| RV5m | 0.1839 | 0.9361 | 1,929 |
| RK | 0.1940 | 0.9293 | 1,929 |
| PAV | 0.1560 | 0.9620 | 1,929 |
| PABPV | 0.1594 | 0.9678 | 1,929 |
| BPV | 0.2124 | 0.9506 | 1,929 |
| C-TRV | 0.2148 | 0.9461 | 1,929 |

The estimator range is `0.156-0.215`. Restricting the analysis to the 1,904
sessions with all 390 bars changes TSRV H(1-10) from `0.1784` to `0.1760`, so
the headline conclusion is not driven by incomplete sessions.

## Scope, data, and metric definitions

- **Instrument:** SPY.
- **Provider:** Massive adjusted one-minute stock aggregates.
- **Window:** `2018-05-01` through `2025-12-31`.
- **Observed data:** 749,014 regular-session bars across 1,929 session dates.
- **Primary session rule:** at least 120 observed minutes, followed by causal
  forward fill on the 390-minute regular-session grid.
- **Strict sensitivity:** exactly 390 observed minute bars; no fill.
- **Overnight returns:** excluded.
- **Primary H estimator:** OLS slope of `log m(2, Delta)` on `log Delta`, divided
  by two, using exact non-overlapping pairs.
- **Figure-compatible sensitivity:** the same regression using all overlapping
  pairs.

Of the 1,929 sessions, 1,904 contain all 390 bars. The other 25 include 18
scheduled early closes with 211 observations and seven ordinary sessions with
376-386 observations. Early-close forward filling creates zero returns after
the observed close; the strict-sample result above bounds the effect of that
choice on the headline TSRV estimate.

## Return-boundary ambiguity does not change headline TSRV roughness

The paper is internally ambiguous about the first intraday return. It describes
a grid of 390 one-minute OHLC prices and defines returns between adjacent
closes, which yields 389 returns. Its simulation section instead uses 390
returns per day. The empirical runner therefore preserves two conventions:

1. `close_only_389`: returns between the 390 minute closes. This is primary.
2. `open_seeded_390`: the first return is the 09:30 bar open to its close,
   followed by close-to-close returns. No overnight return is introduced.

| Estimator or correction | Close-only 389 | Open-seeded 390 |
|---|---:|---:|
| TSRV H(1-10) | 0.1784 | 0.1785 |
| RV5m H(1-10) | 0.1839 | 0.1778 |
| RK H(1-10) | 0.1940 | 0.1947 |
| Corrected RK H(1-10) | 0.2798 | 0.3169 |
| Corrected RK H(1-40) | 0.2025 | 0.2203 |

TSRV is effectively invariant to the boundary choice. RV5m moves modestly,
while the measurement-error correction is materially boundary-sensitive.

## Measurement-error correction converges but is not empirically replicated

The RV5m/RK correction produces positive corrected moments and converges under
both return-boundary conventions. Under the primary convention it estimates
RK noise variance `0.1052`, fitted latent `H = 0.3976`, corrected RK
`H(1-10) = 0.2798`, and corrected RK `H(1-40) = 0.2025`.

Those values should be treated as an unstable upper bracket, not as a recovered
point estimate. The correction is much more sensitive than the raw TSRV result,
its uplift is larger than the paper's equity-panel shift, and a single SPY
series cannot validate a panel-level correction claim.

## Corrected implementation and validation

The audited code corrects the original submission in the areas that materially
affected interpretation:

- exact non-overlapping H increments, without a zero first increment or short
  terminal block;
- missing-date preservation rather than calendar compression;
- TSRV price-grid construction and RK bandwidth normalization;
- standard PAV/PABPV constants and estimator scaling;
- signed two-estimator moment gaps and fail-closed correction diagnostics;
- consistent fOU time units, stationary burn-in, and 390-return simulation;
- strike-level option-skew identification with explicit rejection when pooled
  fit quality is too low.

The supplied corrected harness passed 85 tests. The repository-native empirical
adapter adds six synthetic unit tests for session normalization, the 120-bar
primary rule, the 389/390 return conventions, the explicit `40-250` fit, and
network-receipt preservation on cached reruns.
The private real-data run additionally passed 206 independent assertions and
verified all 92 monthly cache files by SHA-256. Those market-data caches and
receipts are intentionally excluded from this repository handoff.

## Limitations and uncertainty

The evidence supports a real-data SPY validation, not the paper's complete
cross-asset replication. The following remain open:

- the 3,926-equity point-in-time panel, survivorship and delisting treatment,
  and the paper's quality screen;
- XNAS.ITCH rather than Massive adjusted aggregates;
- the 34-root CME continuous-futures construction;
- one-second robustness checks;
- OPRA option chains and the empirical implied-versus-realized channel;
- stable panel-level measurement-error correction and uncertainty estimates.

## Recommended next steps

1. Treat overlapping and non-overlapping increments as named specifications in
   all future tables rather than choosing one silently.
2. Run the repository-native empirical adapter on a point-in-time equity panel
   with the paper's source and filters while keeping raw data outside Git.
3. Report correction estimates only with boundary, provider, fit-window, and
   validity sensitivities.
4. Add a sanitized aggregate panel summary only after its source snapshot and
   survivorship rules can be independently audited.

## Further questions

- Does the near-exact SPY Figure 1 match survive on the paper's XNAS.ITCH feed?
- Which increment convention was used to generate the published empirical
  figures and tables?
- How much of the correction instability is instrument-specific versus caused
  by provider and return-boundary conventions?
