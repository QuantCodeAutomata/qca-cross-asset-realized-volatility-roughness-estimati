# Experiment 2 - exact mean-reversion contamination validation

This is the strongest direct numerical reproduction in the repository: it evaluates the paper's exact fOU second-moment integral on the reported parameter grid.

## Benchmark

- H=0.20, kappa=0.010, delta_max=10: H_hat=0.198305, bias=-0.001695.
- Maximum absolute lag-10 bias over H<=0.20 and kappa<=0.020: 0.004595.

## Interpretation

- Short-lag estimation is only weakly contaminated in the paper's rough benchmark region.
- Extending the fit through delta_max=20, 40, and 100 progressively increases the magnitude of the downward bias; higher lags are more, not less, affected by mean reversion.
- The asymptotic approximation is useful for small kappa*delta, while the exact spectral integral is the reference calculation.
