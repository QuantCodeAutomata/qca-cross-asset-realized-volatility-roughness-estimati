# Experiment 3 - measurement-error validation

This is a synthetic estimator validation, not an empirical replication.

## Configuration

- paths: 20
- days per path: 2520
- one-minute returns per day: 390
- H: 0.2; kappa: 0.01 per trading day
- correction pair: RV5m / RK
- latent log-variance rescaled to std 1.05
- mean annualized volatility calibrated to 12.8%
- paths and Gaussian draws are paired across varpi levels

## Results (means)

| varpi | latent H10 | RV5m H10 | RK H10 | corr RV5m H10 | corr RK H10 | corr RK H40 | valid correction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0e-04 | 0.192 | 0.175 | 0.183 | 0.226 | 0.244 | 0.214 | 100.0% |
| 5.0e-04 | 0.192 | 0.146 | 0.172 | 0.160 | 0.397 | 0.290 | 95.0% |

## Distribution diagnostic

At the baseline noise level, median corrected RK estimates are 0.204 at lag 10 and 0.190 at lag 40. The corresponding means are higher because the correction distribution is right-skewed. This is why the corrected output should be read as a noisy bracket, not as a stable point estimate.


## Paper benchmark for varpi=1e-4

Paper Table 4: latent about 0.198, RV5m 0.181, RK 0.180. Paper Table 5: the RK leg corrected to 0.278 at lag 10 and 0.238 at lag 40. The lag-10 correction is explicitly interpreted as an upper bracket, not a point estimate.

At varpi=5e-4 the raw attenuation remains directionally consistent, but the iid additive log-noise premise breaks down and the correction becomes unstable. The stressed corrected values must not be interpreted as recovery of the latent H.

The local simulation uses a filtered-fGn approximation and a simplified one-minute RK, so differences from the paper's Monte Carlo are expected and must not be presented as empirical evidence.
