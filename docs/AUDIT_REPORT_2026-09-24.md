# Integration audit — September 24, 2026

## Outcome and scope

The original repository has been extended, not replaced. The delivery archive
was used selectively as implementation material; its five commits were not
cherry-picked wholesale. The original source layout, legacy experiments and
historical results are retained.

This is an implementation and synthetic-validation milestone. It is **not**
a full empirical replication of *Rough Volatility Across Assets*, and it adds
no new market evidence to the historical SPY pilot.

Base: `537ae369776c3a4bbdd8bc91de1c43baeb53e594`.
Validated code: `86d3fe02e4b98bfb71f5a8c25b41265d5a748563`.
The final documentation commit is separate from that code commit.

Sources:

- Delivery: `qca_replication_audit_20260924.zip`, SHA-256
  `4a3d2e3d81c0db8055fc5bfd3a4645371e42db756cf87b694121b1855d1c6a19`.
  All 36 payload checksums and the incremental Git bundle were verified.
- Supplied 29-page `rv.pdf`, *Rough Volatility Across Assets*,
  arXiv:2608.16749v1, SHA-256
  `af309e997c261b0e8be55be7365c0fa03c970a01c7b102da35280f9d44ba8c16`.
- All 64 original blob identities were checked against
  [base provenance](BASE_PROVENANCE.json).

## What changed

Only three existing Python modules changed: `option_pricing.py`,
`atm_skew_estimator.py` and `autocovariance_fit.py`. README has 33 appended
lines; ignore rules have a small addition. New implementation is isolated
under `src/replication`; no original module or experiment was relocated.

Pricing retains existing entry points, signatures and explanatory sections.
BAW now handles the zero-rate limit and reports boundary-solver failures
instead of returning a spurious intrinsic-only value. The zero-rate ATM put
regression returns 7.9655674554, not zero. Black-76 and parity support negative
rates; BAW explicitly rejects them. Strict IV errors are opt-in through
`raise_errors=True`; the default invalid-IV result remains `None`.
Parity no longer caps a valid discount above one.

The generic skew minimum counts unique strikes. The separate option pipeline
enforces venue-specific volume/strike/maturity rules and connects prices,
parity, IV, ATM skew and daily/pooled H. Rejections, standard errors and weak
identification are retained.

Correction uses one finite-observation mask for both legs without compressing
the calendar. Unequal-length input now raises an explicit error instead of
silently truncating. The public default remains `guarded`, including its
short-series effective-window behavior. The optional `paper` mode removes
the empirical-moment feasibility cap; its short fit is independent of the
long diagnostic window. New runners select that mode explicitly.

The new modules also provide continuous-covariance Gaussian fOU/fBm,
ensemble calibration, full Table 3–6 designs, deterministic seeds, checksummed
resume, private CSV/Parquet inputs, session/universe/roll handling, aggregation
and figure data products. See the [implementation guide](REPLICATION_GUIDE.md)
for schemas and conventions.

### Delivery defects corrected

| Defect | Correction and regression |
|---|---|
| Finite last iterates from unsuccessful fits entered MC means | Fitted H, both noise variances and corrected H are excluded from valid per-metric means when optimization fails. Last iterates remain in diagnostics; observed gaps and raw H remain available. |
| Base-excluded equities entered primary RV/IV comparisons | Base eligibility and exclusion reasons travel into the diagnostic join; excluded rows cannot set raw/corrected availability. Quality exclusion alone does not remove a base-eligible asset. |
| Figure 7 inherited the raw comparison's success | Raw and corrected counts and coverage requirements are separate. Empty corrected output reports `no_valid_corrected_pairs`, not a completed figure. |
| Additional review finding: stale plot after an empty rerender | Only the previous renderer-owned Figure 6/7 image is renamed to a unique recoverable filename, recorded in `previous_image`. It cannot remain at the current output pathname. |

The failure regressions were observed red before the corresponding fixes and
then passed. Solver-failure tests retain a real finite iterate while injecting
unsuccessful termination at the third-party numerical-solver boundary.

## Validation evidence

Environment: Python 3.12.10; NumPy 2.3.5;
SciPy 1.16.3; pandas 2.3.3;
statsmodels 0.14.6; pytest 9.0.1;
pyarrow 23.0.1; matplotlib 3.10.7.

| Check | Observed result |
|---|---|
| Original suite before changes | 91 passed |
| Final full suite | **260 passed**, 0 failed, 0 skipped |
| Test composition | 91 original + 153 archive + 16 integration/regression cases |
| Bytecode compilation | `python -m compileall -q src data exp tests` passed |
| Public API inspection | Previous parameter names, kinds and defaults preserved for 34 functions; added parameters are optional |
| Guarded numerical compatibility | All old result fields matched the pinned base on complete calendars of 30, 100 and 500 observations |
| New MC smoke | 6 scenarios, 12/12 completed paths, 0 failed path receipts |
| MC validity | 8/12 paths valid across every metric; invalid corrections remain visible |
| Resume | 0 new paths on completed-run resume; tests also cover interrupted resume, checksum/config rejection and failed-path accounting |
| File smoke | Toy minute equities, seconds, futures, options, aggregation, sensitivities and figure products completed |
| Parquet | Genuine pyarrow read/write, all-Parquet and mixed CSV/Parquet pipelines, split bar files and native option-date dtypes passed |
| Full MC dry-run | 51 scenarios / 40,200 planned paths; no paper-scale simulation executed |
| Preservation | All 27 original `results/` files, original tests, legacy experiments and historical audit report unchanged; no moves, renames or deletions |

The paper-design dry-run has 30,000 Table 3 paths, 2,000 Table 4 paths,
1,000 shared Table 5 paths covering all six pairings, and 7,200 Table 6 paths.
Every paper path is configured for 2,520 days and 390 returns/day. This verifies
design coverage, not published Monte Carlo estimates.

The small MC run is deliberately not promoted to a statistical result.
In Table 5, all six pairings had invalid long corrected curves on both toy
paths. The H=0.10, kappa=0 Table 6 toy paths had invalid short/long corrected
curves. All optimization attempts completed, and per-metric valid counts
make these limitations inspectable.

Legacy commands were executed in a separate temporary snapshot so their
hard-coded result paths could not overwrite historical outputs:

```bash
python exp/exp1_realized_hurst.py --n-per-class 1 --n-days 96 --n-bars 390 --no-plots
python exp/exp2_mean_reversion.py
python exp/exp3_measurement_error.py --n-paths 1 --n-days 96 --n-bars 390 --no-plots
python exp/exp4_implied_hurst.py --n-dates 20 --no-plots
python exp/exp5_spy_empirical.py --help
```

All returned zero. Exp. 5 was checked only through CLI help and its six
existing synthetic integration tests: no private SPY cache or provider was used.
Small legacy runs check compatibility, not stability of their statistical
estimates. The initial temporary legacy runs emitted only matplotlib cache
location warnings; final validation used an explicit writable cache directory.

Machine-readable counts, command checks, hashes and smoke diagnostics are in
[acceptance.json](validation/2026-09-24/acceptance.json). Detailed generated
receipts are kept locally in ignored `.local/validation_20260924`; the tracked
receipt lists their SHA-256 hashes and relative paths.

## Table 2: discrepancy retained

The unchanged `src/fou_spectral.py` was used to recompute all 48 printed
cells. **34 match four-decimal rounding; 14 do not**, all in the H=0.20 block.
Maximum absolute H discrepancy: **0.004942396639**.

An independent difference-from-fBm integral agrees with the retained
integrator within approximately **1.21e-12 relative error** over the 400
H=0.20 moment checks. The original H=1/2 OU identity tests also pass.
No integrator parameter or formula was tuned to the printed table.

See the [48-cell comparison](validation/2026-09-24/table2_independent_check.csv)
and [numerical summary](validation/2026-09-24/table2_summary.json).
The provenance of the discrepant printed values remains unresolved.

## Independent diff review

### Standards

No hard violations of the repository's AGENTS.md were found. Guarded defaults,
calendar positions, signed moment gaps, variance-difference constraints,
invalid-correction diagnostics, identification flags and private-output
boundaries are preserved.

One nonblocking judgement-call smell remains: the correction wrapper repeats
moment/OLS calculations to expose diagnostics. It was retained instead of
adding unrelated cleanup to this targeted integration. The review found no
confirmed correctness defect on this axis.

### Spec

No missing code requirement or unrequested architectural change was found.
The reviewer identified the stale comparison image on an empty rerender.
That finding was fixed, and the reviewer independently reran both raw and
corrected rerender regressions: 2 passed. No Spec finding remains open.

Summary: Standards — 0 hard findings, 1 nonblocking smell; Spec — 1 finding
resolved, 0 outstanding.

## Remaining boundaries

- No new market-data panel was loaded or executed. Source membership,
  survivorship handling, provider correctness and published empirical
  conclusions have not been independently certified.
- The author's exact simulation/calibration convention, overlap convention,
  conflicting base-universe descriptions, PAV/C-TRV notation and other PDF
  ambiguities remain documented, not silently assumed resolved.
- BAW uses continuous dividend yield. European parity on American marks and
  near-ATM CME pricing remain explicit approximations.
- File inputs are materialized in memory. Full-panel load, dense 2,520-day
  covariance scaling, distributed execution and Windows locking were not tested.
- MC standard errors are conditional on per-metric valid paths. This change
  does not establish forecast interval coverage or out-of-sample performance.
- No push, PR, database access/change, market download, forecast experiment,
  dependency installation or paper-scale Monte Carlo was performed.

The correct status remains an expanded, tested implementation and validation
harness with historical SPY evidence—not a claim of full empirical replication.
