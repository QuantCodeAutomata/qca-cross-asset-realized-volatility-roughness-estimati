# Replication implementation guide

## 1. Sources and scope

[P] Saad Mouti, *Rough Volatility Across Assets*, arXiv:2608.16749v1, supplied
29-page `rv.pdf`. SHA-256:
`af309e997c261b0e8be55be7365c0fa03c970a01c7b102da35280f9d44ba8c16`.

[B] Repository base commit `537ae369776c3a4bbdd8bc91de1c43baeb53e594`.
`docs/BASE_PROVENANCE.json` records the original tree and blob hashes.

This extension documents the archive's simulation and acceptance conventions
where [P] is ambiguous. This guide distinguishes
those conventions from reproducing the author's unpublished code. The new
entrypoints never query a market-data provider or database. Legacy network
loaders are unchanged but are not invoked by the new runners.

## 2. Pricing and parity

`src/option_pricing.py` validates finite positive price inputs and distinguishes
`PricingError`, `UnsupportedPricingDomain` and `BoundarySolveError`.
Black-76 and parity support negative rates. BAW explicitly does not.

For BAW, `M/h = 2*r / (sigma²*(1-exp(-r*T)))` is evaluated using `expm1` and
its analytic limit `2/(sigma²*T)` at zero rate. Critical exercise prices are
bracketed in log-price coordinates. A failed bracket or root solve raises a
numerical error; it is not interpreted as no early exercise and is not replaced
by intrinsic value. Proven no-exercise cases use the European price: calls with
nonpositive continuous dividend yield and nonnegative rate; puts with zero rate
and nonnegative dividend yield. Positive time value is therefore preserved in
the zero-rate ATM put regression case. Legitimate exercise regions still return
intrinsic value after an exercise boundary has been established.

IV inversion preserves the old optional-`None` interface for non-strict callers.
The connected option adapter uses `raise_errors=True`, records the rejection
reason and pricing model, and never falls back from failed BAW to Black-76.
The numerical IV search interval is `[1e-6, 10]`; prices outside the supported
root range are rejected, not assigned a boundary volatility.

Parity operates on coherent call/put pairs of the same date, expiry, strike,
underlying and selected series. It validates finite positive inferred discount
and forward, distinct strikes, regression rank and diagnostics. It does **not**
cap the discount factor at one. CME uses a known futures settlement and fits
only the coefficient of `F-K`, without an intercept. OPRA fits an intercept and
strike slope, then recovers both `D` and `F`. Standard errors, rank, condition
number, residual RMSE and R² are retained. At least two distinct strikes are
required even for the one-coefficient CME regression.

The CME context settlement must match `underlying_contract_id` in the option
definitions. It is not automatically the continuous front-month settlement.
`require_cme_contract_mapping=False` is an explicit relaxed diagnostic option;
the default is strict and the output records the setting.

**Floating-point zero-rate boundary.** Exact zero-rate prices can give an OLS
parity discount one or two floating-point ulps above one. Only in BAW mode,
when `abs(D-1) <= 8*machine_epsilon`, the pricing rate is set to its analytic
zero limit. The original discount and `raw_parity_rate` remain in the parity
record alongside `BAW_zero_rate_roundoff`. No discount is capped. Economically
negative rates remain rejected. The core BAW function itself rejects every
negative supplied rate.

American OPRA valuation requires positive `spot`, finite `dividend_yield` and a
nonempty `dividend_assumption` in the context. The dividend model is a continuous
yield, not a discrete-dividend schedule. European parity on American marks is
an explicitly reported approximation; it need not recover a perfect forward
or discount on real American chains. CME near-ATM American-style marks use
Black-76 as stated in [P, §3.2], with that approximation recorded.

## 3. Option input and output contracts

All IDs are opaque UTF-8 strings. Prices, strikes, spot and futures settlements
must already be normalized to mutually consistent per-unit price units; applying
contract multipliers is the source preparation's responsibility. Dates and
expiries are timezone-free calendar dates `YYYY-MM-DD`, not intraday timestamps.

| Private role | Required columns |
|---|---|
| `option_definitions` | `date, contract_id, underlying_id, root, venue, expiry, strike, option_type, exercise_style` |
| `option_prices` | `date, contract_id, price, volume` |
| `option_context` | `date, underlying_id, expiry`, plus model-specific fields below |

`venue` is `CME` or `OPRA`; `option_type` is `call` or `put`; `exercise_style`
is `european` or `american`. CME definitions and context additionally carry
`underlying_contract_id`, and context has `futures_settlement`. American OPRA
context has `spot, dividend_yield, dividend_assumption`.

Duplicate identical quotes are not counted twice. Different quotes for one
contract/date are selected by highest volume; a conflicting equal-volume tie is
rejected. Ambiguous multiple contract IDs at one strike/type are rejected rather
than creating a many-to-many call/put join. SPX and SPXW must be explicitly mapped
to the same `underlying_id`. At a duplicate expiry the larger total-volume series
is retained; an exact tie uses a documented lexicographic root rule.

Parity is estimated from all valid matched prices in the selected series.
The positive-volume OPRA filter applies to the quotes used for IV/skew, not to
both legs of every parity pair. This permits a traded call with a valid but
zero-volume put mark to contribute one strike. An unpaired quote may also be
inverted once other matched pairs identify parity. Pair participation is saved.

The main skew profile is `abs(log(K/F)) <= 0.08`, inclusive calendar maturities
14–365 days, 10 unique CME strikes, or 20 unique OPRA strikes with positive
volume. Call/put IVs are averaged within each unique strike; the ATM regression
then weights unique strikes equally. `n_quotes` and `n_strikes` are separate.
This equal-strike averaging is an explicit finite-sample convention. The tests
for duplicate strikes and zero OPRA volume exercise the connected price adapter,
not only the generic OLS helper.

`OptionConfig` exposes `moneyness_band`, `min_days`, `max_days`,
`cme_min_strikes`, `opra_min_strikes`, `otm_only`, `identification_r2`,
`min_maturities_per_date`, `require_cme_contract_mapping`, `merge_series_groups`.
The file runner accepts a primary `analysis.options` object and a list of partial
overrides in `analysis.option_sensitivities`. For example:

```json
{
  "options": {"min_days": 14, "max_days": 365, "moneyness_band": 0.08},
  "option_sensitivities": [
    {"min_days": 30},
    {"otm_only": true},
    {"moneyness_band": 0.06, "opra_min_strikes": 16}
  ]
}
```

An OTM-only specification retains both call and put at exactly ATM, but they
still count as one strike. Daily H uses at least three maturities; pooled H uses
date effects and date-cluster standard errors. Results retain all declared
underlyings and dates, including missing or weakly identified fits. Outputs are
`iv_quotes, parity, skews, daily_h, pooled_h, rejections, series_selection`.
The fitted H, standard errors, R² and `identified` are kept separately; a weak
fit is not erased because its H is implausible. This is important for [P, §8].

## 4. RV correction and log-scale conventions

`fit_two_estimator_correction(a, b, specification="guarded", overlapping=False)`
requires equal-length one-dimensional arrays. The caller first aligns stable
IDs and trading calendars. Both legs are masked to their common finite daily
observations, without deleting calendar positions. The same eligible increments
and covariance pairs are used on both legs.

For `delta_m = mean_delta(m_a - m_b)`, the constraint is exactly
`omega_a² - omega_b² = delta_m/2`, including negative gaps. The objective jointly
fits increment covariances at lags 0, 1 and 2. Both modes retain signed moments,
fit/convergence status, optimizer boundary flags, invalid lag indices, R²,
common pair counts and full corrected moment curves.

`guarded` preserves the original default's feasibility constraint across both
requested windows. `paper` does not constrain noise variances by corrected
empirical moments, and its optimization and starting points do not depend on
the long refit window. Tests compare the same short estimate under long windows
40 and 300. The longer window is diagnostic only in paper mode. Negative or
nonfinite corrected moments never receive a positive logarithm floor. Physical
corrected-H validity and unconstrained OLS diagnostic values are separate.
The old `H_corrected_lag40_*` key names remain for compatibility; metadata also
records the actual requested/effective windows.

New runners pass fitted-H bounds `(1e-6, 0.49)` for rough cells and use an upper
bound `0.99` for Table 6 cells with true H above 0.20. The lower bound is numerical,
not an extra theoretical restriction. The legacy public default remains
`(0.01, 0.49)`. Positive fitted nu has numerical bounds `[1e-6,100]`; hitting a
bound is reported, not concealed. A failed optimizer's last fitted H and noise variances, as well as corrected H,
are preserved in diagnostics but excluded from valid Monte Carlo means.
Observed moment gaps and raw H remain independent of optimizer convergence.

The implementation's state is **log daily unannualized variance**. Replacing
log variance by log volatility divides the state by two: H is unchanged,
nu and noise standard deviations halve, and their variances divide by four.
Mixing these scales in noise calibration is not harmless. [P, §§4.1–4.3] uses
both notations; this implementation fixes and records one convention.

## 5. Gaussian simulation and complete table designs

For positive kappa, the stationary covariance is obtained from [P, Appendix A]
and factored by Cholesky. At H=1/2 it agrees with the ordinary OU covariance;
`2*(gamma(0)-gamma(d))` is checked against the unchanged exact-moment integrator.
A materially nonpositive covariance raises an error. There is no eigenvalue
clipping or hidden jitter. Cached factors contain parameter/algorithm metadata
and factor/covariance hashes; altered caches are rejected.

Calibration is performed **before** drawing paths. Let `s=1.05`, annualization
`A=252`, target mean annualized volatility `v=0.128`, and daily observation
times `t=0,...,N-1` with the paper calibration horizon `N=2520`.

For stationary fOU,

```text
Var(X) = nu² * Gamma(2H+1) / (2*kappa^(2H))
nu² = s² * 2*kappa^(2H) / Gamma(2H+1).
```

For fBm with zero kappa, define the expected unbiased sample variance of a
unit-scale path by

```text
V_N = sum_{d=1}^{N-1} (N-d)*d^(2H) / (N*(N-1))
nu² = s² / V_N.
```

For both cases set a constant mean

```text
mu = 2 * [log(v/sqrt(A)) - log(mean_t(exp(Var(X_t)/8)))].
```

This enforces the desired expected average `sqrt(A)*exp(X_t/2)`, not its
realization-by-realization value. Stationary variance is constant; fBm variance
is `nu²*t^(2H)` and its first random component is zero. The smoke profile uses
a 96-day prefix with the same 2520-day calibration, not a new 96-day fit.
These choices are **documented implementation conventions**, not
confirmed details of the author's generator. The old filtered-fGn and pathwise
rescaling remain available through `generator="legacy"` and legacy examples.

| Table | Paper design | Paths |
|---|---|---:|
| 3 | H `.05,.10,.15,.20,.30,.50`; kappa `0,.003,.010,.020,.035`; latent and RV5m, lags 10/40, baseline noise | 1000 per cell |
| 4 | H `.20`, kappa `.010`; all seven RV measures; noise `1e-4,5e-4` | 1000 per noise |
| 5 | RV5m paired with each of the other six measures, benchmark and baseline noise, refits 10/40 | 1000 shared paths; each pair evaluated on every path |
| 6 | Six H values; kappa `0,.010,.035`; RV5m/RK with the specified H bounds | 400 per cell |

All paper paths have 2520 days and 390 returns/day (391 price nodes). The small
smoke profile has six selected scenarios and two 96-day paths each; it is a code
exercise, not evidence for published Monte Carlo means or standard errors.

Seeds derive from SHA-256 of master seed, table, H, kappa, noise, path ID and
stream. They do not depend on process-randomized `hash`, execution order or path
count. Table 5 reuses each trajectory across all six pairings. Different table
or noise scenarios intentionally have different semantic seeds.

Each completed path, including a failed numerical attempt, has an atomic JSON
receipt. Resume validates the checksum, semantic seed, config and code/dependency
fingerprint. It neither double-counts nor silently retries failed paths. Changing
code or configuration requires a fresh output directory. A Unix file lock prevents
concurrent writers to one run directory. This is a sequential runner; Windows
locking and distributed execution have not been implemented or tested.

Per-metric outputs include requested/completed/valid/invalid counts, means,
unbiased sample variances and Monte Carlo standard errors `sd/sqrt(n_valid)`.
Convergence counts, boundary solutions, invalid refits and failure reasons remain
visible. These means are conditional on the recorded valid set; invalid paths
must not be interpreted as nonexistent. No published Table 3–6 numerical result
is certified by merely executing the matching design.

## 6. Private file manifest

`FileManifest` supports local CSV and Parquet. Each role may name one file or a
list of files to concatenate; every file has a required SHA-256. URLs are rejected.
Only hashes actually checked are called verified. The full manifest remains
private; its exact byte hash and a separate canonical-content fingerprint are
recorded. A source snapshot hash verifies file identity, not provider correctness.

Minimal top-level structure (placeholders below are deliberately not executable
market data or a guessed asset universe):

```json
{
  "schema_version": 1,
  "data_kind": "market",
  "source": "YOUR_SOURCE_AND_FEED",
  "snapshot": "YOUR_IMMUTABLE_SNAPSHOT_ID",
  "files": {
    "sessions": {"path": "sessions.csv", "sha256": "REPLACE_WITH_64_HEX_DIGITS"},
    "instruments": {"path": "instruments.csv", "sha256": "REPLACE_WITH_64_HEX_DIGITS"},
    "equity_bars": {"path": "equity_bars.parquet", "sha256": "REPLACE_WITH_64_HEX_DIGITS"}
  },
  "units": {"prices": "levels", "variance": "daily_unannualized"},
  "price_adjustments": {"equity": "DESCRIBE_UPSTREAM_ADJUSTMENTS_EXACTLY"},
  "datasets": {
    "equity": {
      "bar_seconds": 60,
      "timestamp_convention": "start",
      "timestamp": {"encoding": "epoch", "unit": "ms"},
      "return_convention": "close_only",
      "observed_fraction_denominator": "session",
      "regular_session_seconds": 23400
    }
  },
  "base_universe": {
    "min_estimation_days": 500,
    "min_consecutive_days": 500,
    "min_observed_minutes_per_day": 120,
    "zero_volume_policy": "exclude_day",
    "rationale": "EXPLICIT_USER_CHOICE; THESE VALUES DO NOT RESOLVE THE PAPER'S CONFLICT"
  },
  "selections": {},
  "analysis": {
    "lag_windows": [[1,10],[1,40],[40,250]],
    "overlapping": [false,true],
    "primary_overlapping": false,
    "correction_specification": "paper",
    "fitted_H_bounds": [0.000001,0.49],
    "aggregation": {"missing_day_policy": "invalidate", "edge_period_policy": "exclude"}
  },
  "publication": {"approved_aggregates": ["counts", "manifest_sha256"]}
}
```

This example chooses numerical universe screens explicitly; it is **not** a
claim that these conflicting screens are the author's base universe. A real
manifest must give the actual decision and rationale. A fully executable toy
manifest with computed hashes is generated by `python -m tests.run_file_smoke`.

Additional file roles and columns:

| Role | Column contract |
|---|---|
| `sessions` | `calendar_id, session_date, open, close`; timezone-aware session open/close |
| `instruments` | `instrument_id, symbol, valid_from, valid_to, calendar_id, asset_class` |
| `equity_bars`, `second_bars` | `instrument_id, timestamp, close, volume`; `open` for open-seeded sensitivity |
| `futures_contracts` | `contract_id, root, expiry, calendar_id, asset_class` |
| `futures_daily` | `date, root, contract_id, volume` |
| `futures_bars` | `contract_id, timestamp, close, volume`; optional `open` |
| Option roles | Section 3 above |

Each dataset (`equity`, `seconds`, `futures`) needs its own convention object and
price-adjustment description. Physical bar intervals must divide 300 seconds;
one-second and one-minute grids are exercised. At a five-minute base grid,
TSRV's fast and slow scales coincide and its subtraction degenerates; it is not
an informative two-scale estimate. Coarser grids are outside this adapter's
supported two-scale specification.

Epoch units may be seconds, milliseconds, microseconds or nanoseconds. ISO8601
bars are also supported; naive values require `naive_timezone`. Ambiguous and
nonexistent local DST times raise errors. Session calendars are supplied files,
not guessed business-day calendars; they include half days and overnight
futures sessions. Adjustment rules are declarations of upstream processing, not
a corporate-action engine implemented here.

`valid_to` may be blank for a still-active equity interval. Renames use multiple
nonoverlapping symbol intervals with one stable ID. Delisted assets remain in
the panel and are not filtered by current ticker availability. Calendar/class
changes for one ID require explicit separate analysis segments. Leading zeros
in IDs are preserved in CSV; Parquet producers should store IDs as strings too.

## 7. Panel semantics and empirical scenarios

Equity bars are matched to supplied sessions, reindexed to a physical grid and
forward-filled inside that session only. There is no backfill and no overnight
carry. Missing trading days retain rows with NaNs, as do inactive calendar
positions before listing or after exit; those do not become artificial adjacent
days. Conflicting repeated bars and nonpositive prices are logged/rejected.

Base screens are entirely explicit: minimum estimation days, longest consecutive
valid trading-day run, minimum observed physical minutes and zero-volume policy
(`keep`, `exclude_day`, `exclude_asset`). The quality subset, independently of
those manifest choices, requires at least 500 valid TSRV estimation days and an
observed fraction of at least 0.80 on at least half those days. Half-day fractions
use either actual session duration or the declared regular-session duration,
according to `observed_fraction_denominator`.

Futures processing first constructs a forward-only volume-overtake schedule,
then selects intraday bars for that contract/session. It never differences prices
across contracts or across sessions. The initial contract is the earliest
unexpired declared contract. Expiry forces a forward roll; missing current
volume alone does not prove the contract disappeared. `analysis.futures_roll_timing`
must explicitly be `same_session` (retrospective volume-based daily selection)
or `next_session` (decision applied on the following supplied session).
All selected contracts, decision dates, volumes and roll reasons are retained.
Nonselected bars and nonpositive prices, including negative futures prices,
are logged; they cannot enter a logarithm.

Five-minute RV uses `300/bar_seconds`-spaced price endpoints on the original
session grid, not five elements of every array. TSRV uses all staggered price
subgrids on that physical scale. Complete 390 close prices yield 389 returns.
`open_seeded` adds the actual first in-session bar open to yield 390, and never
uses the previous day's close. The Monte Carlo simulator directly creates 391
price nodes and 390 returns. Leading missing endpoints do not shift the
five-minute phase to a later first observation.

Weekly and monthly aggregation sums daily **variance**, then takes logs for H;
it does not sum log variance. `missing_day_policy="invalidate"` rejects a period
with missing active observations; `available` explicitly uses the available
ones. Every calendar bin remains. `edge_period_policy="exclude"` conservatively
excludes first/last supplied calendar bins and partially unlisted bins without
guessing whether absent pre-snapshot dates were holidays. `include` retains
boundary periods. These are explicit implementation choices, not recovered
author conventions. Lags for aggregated series are weeks/months, not days;
outputs record their frequency. Insufficient or missing lags remain visible in
`n_lags`; a partial fit must not be presented as a complete requested window.

Explicit selection keys are:
`seconds`, `futures_roots`, `option_underlyings`, `figure1_ids`, and
`comparison_map` (a list of `{realized_id, option_id}` objects). No membership is
inferred from the paper's target counts. The primary realized/implied join uses
minute-based raw TSRV and corrected RK, with weak implied fits preserved and
availability flags for both comparisons. Base-excluded equities remain in the
diagnostic join with `base_eligible=false` and `base_exclusion_reasons`, but
cannot contribute to either primary comparison. The quality subset is not an
additional primary-match filter. Selected futures require eligible estimation
days. `comparison_counts.raw` and `.corrected` and the Figure 6/7 coverage flags
are independent. An empty comparison renderer reports `no_valid_raw_pairs` or
`no_valid_corrected_pairs`, without claiming a completed figure. Read the latest
`figure_status.json`; when a comparison becomes empty, any old image is retained
under a unique `previous` filename, recorded as `previous_image`, instead of
remaining at the current figure's pathname.
Second-frequency matches are reported
separately. Cross-panel ID collisions fail rather than mixing unrelated assets.

Generated products cover source/count summaries, full/quality equity
cross-sections, per-root/class futures comparisons, all seven estimators,
second-frequency comparisons, weekly/monthly aggregation, lag windows, option
sensitivities and explicit realized/implied matches. Plot renderers consume
these products. Figure 2 additionally consumes the smooth-truth Table 3 moment
curves. The toy driver exercises a separate three-path, 96-day Figure 2 fixture.
It does not reproduce a 1000-path ensemble plot.

## 8. Publication, coverage and known paper ambiguities

Private output guards allow paths outside Git or inside ignored `.local` only.
Do not commit raw bars, option marks, derived daily series or detailed manifests.
A public export can contain only manifest-approved items from the built-in
whitelist: `counts`, `manifest_sha256`, `source_snapshot_hashes`,
`equities_distribution`, `class_medians`. Merely passing an output pathname
is not permission to publish a time series.

Every table and figure has separate implementation, fixture, numerical-run and
market-data-run status. Missing sources and methodology issues are separate
fields. Even a completed supplied-data run does not automatically certify the
source's universe, author conventions or published numerical match. Full
empirical replication always requires a separate review.

Important unresolved or explicitly corrected readings of [P]:

| Source location | Treatment in this branch |
|---|---|
| §3.1 versus §3.2 | 500 consecutive days versus permissive minute coverage is not silently reconciled; manifest screens and rationale are mandatory. |
| Eq. (1) and Figure 1 | Non-overlapping is the literal equation; both overlap conventions are named and retained. The figure's prose and printed labels also differ. |
| §§4.1–4.3 | Log variance is fixed here; H is scale invariant but noise and nu normalization are not. |
| Appendix B return notation | Use differences of log prices, not a second logarithm of already logged prices. |
| Appendix B PAV | Preserve the standard tent-weight implementation: no extra `1/k_n` in the pre-averaged return; `psi1=1`, `psi2=1/12`, and the outer/bias normalization already corrected in the base. |
| Appendix B C-TRV | The printed `10.849` corresponds to `1+3*phi(3)/(1-Phi(3))`; the printed numerator uses the wrong normal-function symbol. Keep the density-based implementation. |
| Table 2 | Keep all 48 printed numbers as comparison data; 14 disagreements in H=.20 are not fitted away. Independent integrals support the retained code. |
| §5.1 | The listed grid omits .20 in prose but includes it in tables/benchmark; the requested six-value grid includes .20. Calibration and stationary generator are documented, not attributed to unseen author code. |
| Figures 6/7 | Prose distinguishes raw from corrected, while a caption/axis conflicts. Outputs explicitly name raw TSRV and corrected RK rather than relabeling both as corrected TSRV. |
| Aggregation and roll timing | Source text does not fully settle edge/missing/decision timing policies; these are manifest choices. |

## 9. Operating limits and reproducibility

The September 24 integration validates CSV and genuine Parquet with pyarrow,
including split bar files and mixed CSV/Parquet option date dtypes. The optional
engine test skips when pyarrow is absent; the manifest reader also gives an
explicit missing-engine diagnostic. See [the dated integration report](AUDIT_REPORT_2026-09-24.md)
for the actual environment, checks and evidence boundary.

File roles are materialized in memory and then grouped; this is not a streaming
billion-row ingestion service. The full source universe and 2520-by-2520 dense
Gaussian factor caches have not been scale/load tested. Full data runs need
appropriate memory, sharding/operational planning and separate authorization.
No market provider, TimescaleDB, forecast experiment, push, PR, dependency
installation or paper-scale Monte Carlo is needed for these local checks.

The delivery archive was used as a source, not applied as a five-commit series.
This checkout starts from the complete pinned base, keeps the original
`AUDIT_REPORT.md` and historical results, and adds the extension separately.
