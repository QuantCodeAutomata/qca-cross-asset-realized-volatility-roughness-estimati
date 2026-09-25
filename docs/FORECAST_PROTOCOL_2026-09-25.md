# H and nu forecasting continuation — fixed protocol

Status: execution protocol, not evidence of predictive success. `fully_reproduced: false`.
This extends the September 3 forecasting experiment using only the existing
sample through September 23, 2026. No additional market observations are acquired.
Only aggregate scores, parameter estimates, security coverage and a single
forecast-origin snapshot may be released; observed or reconstructed time series
are excluded. Scientific conclusions remain limited to this selected sample.

## Three distinct calculations

1. **Retrospective repeat:** January–August 28, 2026 targets, with fitting ending
   in 2025 or earlier. This period has already been inspected; it is not a new
   untouched test. Recompute daily TSRV with the accepted implementation and
   retain the original quality and history-boundary cohort flags. Repaired
   histories can change estimates and eligibility; historical reports are not overwritten.
2. **Later fixed-parameter evaluation:** August 31–September 23 targets, using
   previously saved 2025 rough parameters and HAR coefficients. Compare four
   predeclared models: persistence, HAR, overlap-H/moment-nu and
   overlap-H/innovation-nu. Exclude new securities, repaired Fiserv/LRCX histories
   and inactive EA. Persistence uses the unchanged pre-2026 horizon-error scale.
   These are reconstructed forecasts, not proof of real-time issuance or
   point-in-time vendor vintages. This window selects no settings.
3. **Latest snapshot:** fit full-history and trailing-504-session overlap H and
   innovation nu through September 23; separately fit through March 31 and
   calibrate on subsequent errors through September 23. Report endpoint
   forecasts at 1, 5 and 22 trading sessions. No fresh forecast is produced
   without a positive TSRV observation at the common cutoff. The snapshot is
   not scored against unavailable future targets.

## Target, preprocessing and estimators

The target is **single-session daily TSRV at t+h**, not accumulated variance.
Use the common SPY trading calendar, a 390-minute regular-session grid, at least
120 observed minutes, positive total volume, causal intraday forward fill and
close-only returns; no overnight returns, leading backfill or cross-day fill.
Non-positive TSRV is unavailable in log space. Missing days retain their calendar
positions. Five-minute TSRV averages all staggered grids without finite-sample
rescaling. The fast TSRV calculation is checked against the accepted physical-grid
adapter on five initial sessions and every later-window session per security.
Paper-period source fingerprints must match the preceding sample validation.

For y = log(TSRV), H is half the log-variogram slope at lags 1–10.
Moment nu_y = exp(intercept/2); innovation nu_y squared is the mean squared
one-step conditional error divided by its unit-scale conditional variance.
**nu_logVol = nu_logRV / 2**, with one trading session as the time unit.
Nu is not an annualized volatility percentage. H is invariant to this scaling.
The model is a finite-past Gaussian fBm proxy, not an identified latent-volatility law.

Retain the original nine-model retrospective comparison: persistence; direct
log HAR fit through 2025; non-overlap and overlap paper-H/moment-nu; overlap
paper-H/innovation-nu; split rough Gaussian and calibrated variants; selected-H
calibrated rough; split calibrated HAR. Split models fit through 2024 and
calibrate on 2025. The H grid remains 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
0.70, 0.90, selected on 2024 one-step log score with scale fit through 2023.

Rough forecasts use up to 64 preceding calendar lags, requiring 22 observed
endpoints. HAR uses current, trailing-5 and trailing-22 mean log RV, with direct
horizon-specific regressions. All training targets must precede their cutoff;
forecast origins must follow fitting and calibration cutoffs.

## Intervals, uncertainty and scores

Central prediction levels are 50%, 80%, 90%, 95%. Split calibration uses absolute
standardized errors with origins after the fitting boundary and targets no later
than the calibration cutoff. Require 80 scores per horizon. The radius uses
rank ceil((n+1) times level), capped at n. Calibration changes intervals, not
point forecasts. Serial dependence and regime changes preclude an unconditional
finite-sample coverage guarantee.

Rough/HAR mean RV is exp(mu+v/2); persistence predicts the current RV. For the
latest snapshot, median annualized volatility is sqrt(252) exp(mu/2), whereas
its mean is sqrt(252) exp(mu/2+v/8). Neither is sqrt of expected annualized RV.

Separate 95% parameter intervals use 300 paired moving-block bootstrap samples
of joint lag-moment vectors, block length 20. SPY additionally uses blocks 5
and 60. The bootstrap point fit uses common-start lag support; it can differ
from the all-pairs H fit. The nu interval covers **moment nu**, not innovation
nu. Parameter uncertainty is not integrated into the conditional Gaussian
prediction intervals. Bootstrap seeds are deterministic by security and stage.

Evaluate QLIKE, log squared error, empirical coverage, 90% interval width, and
weighted interval score (WIS; lower is better). Compare identical security,
origin and target intersections across all models within each experiment;
also retain native model counts. Stocks are equally weighted and SPY is separate.
Report original stocks, original quality stocks and quality stocks without
history boundaries. No model is selected on the reported test scores.

Retrospective score uncertainty uses 1,000 circular calendar-date block samples,
block length 22, preserving cross-security shocks and equal-security weights.
Paired differences are reported against persistence and HAR; calibrated WIS
also against split Gaussian WIS. **No block-bootstrap confidence interval is
claimed for the 17-session later window.** It is descriptive evidence only.

## Acceptance

Require read-only execution, complete instrument receipts, verified aggregate
packet hashes, source-fingerprint reconciliation, calendar and exclusion counts,
past-only synthetic tests, independent score arithmetic on every retrospective
row, and independent dense linear solves for all latest rough forecasts.
Release neither observations, per-date scores nor infrastructure identifiers.
Keep the original repository architecture and historical outputs unchanged.
