#!/usr/bin/env python3
"""Recompute all 48 printed Table 2 cells and cross-check the H=.20 integral.

No market data or network access is needed. This script imports the repository's
own compute_exact_m2/compute_m2_grid rather than substituting an implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gamma

# Transcribed from Table 2, printed PDF page 10, arXiv:2608.16749v1.
# Each list corresponds to kappa = .003, .010, .020, .035.
PAPER_ROWS = {(0.1, 10): [0.0999, 0.0993, 0.0979, 0.0951],
 (0.1, 20): [0.0997, 0.0982, 0.0951, 0.0893],
 (0.1, 40): [0.0993, 0.0956, 0.0887, 0.0777],
 (0.1, 100): [0.0974, 0.0864, 0.0704, 0.0519],
 (0.15, 10): [0.1498, 0.1488, 0.1467, 0.1426],
 (0.15, 20): [0.1496, 0.1473, 0.1426, 0.1344],
 (0.15, 40): [0.1489, 0.1435, 0.1336, 0.118],
 (0.15, 100): [0.1461, 0.1304, 0.1077, 0.081],
 (0.2, 10): [0.1997, 0.1982, 0.1952, 0.1896],
 (0.2, 20): [0.1993, 0.196, 0.1896, 0.1786],
 (0.2, 40): [0.1983, 0.1908, 0.1775, 0.1569],
 (0.2, 100): [0.1944, 0.1733, 0.1431, 0.1068]}

def independent_m2(delta: float, h: float, kappa: float) -> float:
    """An independent, dimensionless difference-from-fBm representation.

    Let e=kappa*delta. Then
      M2 = delta^(2H) * [1 - C*e^2*J],
      C = 2*Gamma(2H+1)*sin(pi*H)/pi,
      J = integral (1-cos(u))/(u^(1+2H)*(u^2+e^2)) du.
    This does not subtract stationary covariances, unlike the repo routine.
    """
    if kappa == 0.:
        return delta ** (2*h)
    e = kappa * delta
    p = 1. / (2. - 2.*h)

    def near(v):
        u = v**p
        return .5 * np.sinc(u/(2.*np.pi))**2 * p / (u*u+e*e)

    def tail(u):
        return u**(-1.-2.*h) / (u*u+e*e)

    near_int = quad(near, 0., 1., epsabs=1e-12, epsrel=1e-12, limit=500)[0]
    tail_int = quad(tail, 1., np.inf, epsabs=1e-12, epsrel=1e-12, limit=500)[0]
    tail_cos = quad(tail, 1., np.inf, weight="cos", wvar=1.,
                    epsabs=1e-12, epsrel=1e-12, limlst=500)[0]
    c = 2.*gamma(2.*h+1.)*np.sin(np.pi*h)/np.pi
    return delta**(2.*h) * (1.-c*e*e*(near_int+tail_int-tail_cos))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2], help="Target checkout root.")
    parser.add_argument("--output-dir", type=Path, default=Path("numerical_check"))
    args = parser.parse_args()
    root = args.repo.resolve()
    if not (root/"src"/"fou_spectral.py").is_file():
        parser.error("--repo must contain src/fou_spectral.py")
    sys.path.insert(0, str(root))
    from src.fou_spectral import compute_m2_grid

    args.output_dir.mkdir(parents=True, exist_ok=True)
    deltas = np.arange(1., 101.)
    kappas = [.003, .010, .020, .035]
    rows = []
    cross_checks = []
    for h in [.10, .15, .20]:
        for ki, kappa in enumerate(kappas):
            m2 = compute_m2_grid(deltas, h, kappa)
            if np.any(~np.isfinite(m2)) or np.any(m2 <= 0.):
                raise RuntimeError(f"Invalid moments for H={h}, kappa={kappa}")
            for n in [10, 20, 40, 100]:
                h_hat = float(np.polyfit(np.log(deltas[:n]), np.log(m2[:n]), 1)[0]/2.)
                ref = PAPER_ROWS[(h, n)][ki]
                rows.append({
                    "H": h, "kappa": kappa, "delta_max": n,
                    "paper_table2": ref, "repo_exact_m2_OLS": h_hat,
                    "difference": h_hat-ref,
                    "matches_published_rounding": format(h_hat, ".4f") == format(ref, ".4f"),
                })
            if h == .20:
                alternative = np.array([independent_m2(d, h, kappa) for d in deltas])
                cross_checks.append({
                    "kappa": kappa,
                    "max_m2_relative_difference": float(np.max(np.abs(alternative/m2-1.))),
                    "H_hat_100_independent": float(np.polyfit(
                        np.log(deltas), np.log(alternative), 1)[0]/2.),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir/"table2_independent_check.csv", index=False)
    summary = {
        "cells": len(frame),
        "match_published_four_decimal_rounding": int(frame.matches_published_rounding.sum()),
        "max_absolute_difference": float(frame.difference.abs().max()),
        "independent_integral_checks": cross_checks,
        "interpretation": "A printed-table discrepancy is not automatically a code defect. Compare independent integrals before changing the implementation.",
    }
    (args.output_dir/"table2_summary.json").write_text(json.dumps(summary, indent=2))
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
