# Experiment 1 - synthetic realized-H estimator validation

This is not the paper's empirical cross-asset replication. It uses synthetic assets centered on the paper's class medians.

Configuration: 2 synthetic assets per class, 1000 days, 390 returns/day.

## Estimator summary

| estimator | median H | median bias | MAE | median R2 | n |
|---|---:|---:|---:|---:|---:|
| bpv | 0.092 | +0.004 | 0.022 | 0.742 | 16 |
| ctrv | 0.092 | +0.004 | 0.023 | 0.739 | 16 |
| pabpv | 0.076 | -0.016 | 0.025 | 0.575 | 16 |
| pav | 0.079 | -0.011 | 0.025 | 0.593 | 16 |
| rk | 0.089 | -0.000 | 0.024 | 0.691 | 16 |
| rv5m | 0.087 | -0.002 | 0.024 | 0.686 | 16 |
| tsrv | 0.087 | -0.003 | 0.024 | 0.656 | 16 |

## TSRV class recovery

| class | paper reference | synthetic target | estimated median | median R2 |
|---|---:|---:|---:|---:|
| agriculture | 0.085 | 0.082 | 0.066 | 0.433 |
| energy | 0.088 | 0.085 | 0.105 | 0.690 |
| equity_index | 0.195 | 0.191 | 0.185 | 0.909 |
| fx | 0.081 | 0.076 | 0.042 | 0.267 |
| livestock | 0.048 | 0.043 | 0.043 | 0.258 |
| metals | 0.091 | 0.084 | 0.105 | 0.723 |
| rates | 0.074 | 0.080 | 0.057 | 0.518 |
| single_stock | 0.131 | 0.124 | 0.142 | 0.854 |

The output can validate implementation scale and rough-regime recovery, but it cannot validate the paper's cross-sectional ordering, quality filters, or empirical R2 distribution.
