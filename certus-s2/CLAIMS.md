# CERTUS-S2 Formal Claim Audit

Every substantive scientific and engineering claim made by CERTUS-S2 is categorized below.
Statuses: 
- `SAFE`: Holds as stated, given assumptions.
- `NEEDS EXPERIMENT`: Design is sound, evidence pending.
- `ASSUMPTION-DEPENDENT`: Valid only while a stated assumption holds; must always be quoted with it.
- `DO NOT CLAIM`: Unsupported — appears nowhere in our materials, UI, or presentations.

## Claims Ledger

| # | Claim | Type | Assumptions | Evidence required | Status |
|---|---|---|---|---|---|
| 1 | Track A gives a finite-sample bound `E[FD/N] ≤ α` | Mathematical | Exchangeable calibration **scenes**; loss monotone + bounded | Realized ≤ nominal on site-disjoint TEST | **SAFE** |
| 2 | Track B gives finite-sample **FDR** control over the selected set | Mathematical | Exchangeability of **null** units (B1); PRDS (B2); candidate-level null exchangeability (B3) | Realized FDR ≤ α on site-disjoint TEST; BY reported alongside | **ASSUMPTION-DEPENDENT** |
| 3 | Track B with **BY** controls FDR under arbitrary dependence | Mathematical | B1 only (exchangeability of nulls) | Same, BY column | **SAFE** given B1 |
| 4 | Standard CRC controls **FDP/FDR** directly | Mathematical | — | — | **DO NOT CLAIM** |
| 5 | A realized empirical FDP is itself a guarantee | Empirical | — | — | **DO NOT CLAIM** |
| 6 | `s` estimates how strongly the observation constrains local structure | Physical/statistical | Assumed effective operator + noise model; Fourier approximation | Operator fit + CIs; sensitivity | **ASSUMPTION-DEPENDENT** |
| 7 | `s = 0` indicates hallucination / `s = 1` indicates correctness | Interpretive | — | — | **DO NOT CLAIM** |
| 8 | Above-Nyquist support is *identifiable* | Mathematical | — | — | **DO NOT CLAIM** |
| 9 | `s`/`ν_θ` improve the risk score over variance-only | Empirical | — | Ablation on real data | **NEEDS EXPERIMENT** |
| 10 | `D̂` is Sentinel-2's physical MTF | Physical | — | — | **DO NOT CLAIM** |
| 11 | `D̂` is an *effective observation operator* fitted with reported CIs | Empirical | Same-day pairing; joint sub-pixel shift; regularization | Fitted `σ_b` ± bootstrap CI | **NEEDS EXPERIMENT** |
| 12 | Geographic generalization | Empirical | Site-disjoint splits, machine-checked | Realized ≤ nominal on TEST | **NEEDS EXPERIMENT** |
| 13 | Robustness to distribution shift | Empirical | — | Shift sweep; ablations | **ASSUMPTION-DEPENDENT** (diagnosis and abstention, never robustness) |
| 14 | `Δ̂` is a certified bound on the coverage gap | Mathematical | — | — | **DO NOT CLAIM** |
| 15 | Certified guarantee over the Indian AOI | Empirical | — | — | **DO NOT CLAIM** (out-of-domain demo) |
| 16 | 2.5 m **sampling** output | Engineering | 4× from 10 m | Grid-alignment assertions | **SAFE** |
| 17 | 2.5 m **resolved detail** everywhere | Physical | — | — | **DO NOT CLAIM** |
| 18 | Decision-level certification for a satellite reconstruction product | Novelty | Literature search | Evaluation of guarantees | **NEEDS EXPERIMENT** (SAFE as scoped novelty) |
| 19 | Validity is independent of risk-model quality | Mathematical | Same as claims 1–3 | Ablation with constant score | **SAFE** |
| 20 | Certified **change** detection (bi-temporal) | Empirical | Bi-temporal HR refs | Bi-temporal evaluation | **DO NOT CLAIM** in MVP |
| 21 | Better imagery than SEN2SR/LDSR-S2 | Empirical | — | — | **DO NOT CLAIM** |
| 22 | Spectral consistency preserved | Empirical | Inherited from backbone + `𝒞(Y)` | SAM, per-band MAE | **NEEDS EXPERIMENT** |
| 23 | Abstention is principled, not heuristic | Methodological | Shift score tied to measured breakpoint | Abstention-vs-shift curve | **NEEDS EXPERIMENT** |
| 24 | Real-time full-scene processing | Engineering | — | — | **DO NOT CLAIM** (batch job) |
| 25 | Exact data consistency | Physical | — | — | **DO NOT CLAIM** |
| 26 | Hallucination probability output | Interpretive | — | — | **DO NOT CLAIM** |

> **RULE:** No code, README, or UI element may violate these boundaries. Any presentation of `NEEDS EXPERIMENT` must be accompanied by the actual generated experimental artifact proving it. Any `ASSUMPTION-DEPENDENT` claim must state the assumption explicitly alongside it.
