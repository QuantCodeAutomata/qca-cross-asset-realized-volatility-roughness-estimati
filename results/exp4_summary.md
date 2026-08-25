# Experiment 4 - synthetic option-skew identification validation

This experiment starts from strike-level synthetic smiles and therefore tests both the ATM-skew estimator and the maturity regression. It is not a CME/OPRA replication.

Dates per synthetic underlier: 250. Maturity convention: calendar days / 365.

| underlier | H_hat_IV | target | R2_psi | identified | expected | median smile R2 | validation |
|---|---:|---:|---:|:---:|:---:|---:|:---:|
| ES_like | 0.249 | 0.250 | 0.603 | True | True | 0.964 | PASS |
| ZW_like | 0.083 | 0.080 | 0.534 | True | True | 0.857 | PASS |
| ZN_like | 0.523 | - | 0.000 | False | False | 0.103 | PASS |
| 6E_like | 0.519 | - | 0.000 | False | False | 0.133 | PASS |
| NG_like | 0.515 | - | 0.000 | False | False | 0.744 | PASS |

The paper-level interpretation is deliberately stricter than the original repository: rates and FX estimates are not meaningful when R2 is near zero, and a stylized energy power law is not evidence for the paper's seasonal commodity result.
