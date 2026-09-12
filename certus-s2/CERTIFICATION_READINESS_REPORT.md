# CERTUS-S2 Certification Readiness & Audit Report

## 1. Executive Summary & Objective

This report constitutes the formal scientific and engineering audit of the **Certification Layer** of CERTUS-S2, consisting of:
- **Track A Certifier** ([`src/certification/track_a.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/certification/track_a.py)): Conformal Risk Control (CRC) on a monotone normalized false-discovery loss.
- **Track B Certifier** ([`src/certification/track_b.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/certification/track_b.py)): Conformal p-values with Benjamini–Hochberg (BH) and Benjamini–Yekutieli (BY) multiple testing.
- **D1 Candidate Proposal Stage** ([`src/decision/proposal.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/decision/proposal.py)): Fixed $\lambda$-independent candidate generation for task D1 (binary built-structure presence verification).
- **Abstention & Shift Evaluator** ([`src/shift/abstention.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/shift/abstention.py)): Distribution shift detection and abstention gating.
- **Site-Disjoint Split Manifest** ([`data/split_manifest.json`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/data/split_manifest.json)): Machine-checked site separation.

### MVP Certified Scope
The MVP certification target is **strictly restricted to Task D1** (binary built-structure presence verification). Generic bi-temporal "change detection" is explicitly out of scope for this MVP (Claim 20: `DO NOT CLAIM in MVP`).

---

## 2. Track A Audit: Conformal Risk Control on Monotone Loss

### 2.1 Mathematical Formulation
For calibration or test scene $i$ with $N_i = |\mathcal{K}_i|$ proposed candidates:
$$L_i(\lambda) = \frac{\text{FD}_i(\lambda)}{\max(1, N_i)}, \quad \text{FD}_i(\lambda) = \#\{k \in \mathcal{K}_i : r_k \le \lambda \text{ and } k \text{ is false}\}$$
Given $n$ independent calibration scenes, finite-sample conformal risk control selects:
$$\hat{\lambda} = \sup \left\{ \lambda \in [0, 1] : \frac{n \hat{L}_n(\lambda) + B}{n + 1} \le \alpha \right\}, \quad \hat{L}_n(\lambda) = \frac{1}{n} \sum_{i=1}^n L_i(\lambda)$$
Where $B = 1.0$ is the uniform upper bound on $L_i(\lambda)$.

### 2.2 Proof of Monotonicity
Let $\lambda_1 < \lambda_2$.
1. For any candidate $k$, $r_k \le \lambda_1 \implies r_k \le \lambda_2$.
2. The retained candidate sets are nested: $\mathcal{R}_i(\lambda_1) \subseteq \mathcal{R}_i(\lambda_2)$.
3. Filtering by ground-truth false discoveries preserves nesting:
   $$\{k \in \mathcal{K}_i : r_k \le \lambda_1 \text{ and false}\} \subseteq \{k \in \mathcal{K}_i : r_k \le \lambda_2 \text{ and false}\}$$
4. Hence $\text{FD}_i(\lambda_1) \le \text{FD}_i(\lambda_2)$.
5. Because $N_i = |\mathcal{K}_i|$ is fixed by the proposal stage ($sr\_image > c_0$) and has zero dependence on $\lambda$, the denominator $\max(1, N_i)$ is constant with respect to $\lambda$.
6. Therefore:
   $$L_i(\lambda_1) \le L_i(\lambda_2) \quad \forall \lambda_1 < \lambda_2$$
   **The loss is strictly non-decreasing in $\lambda$ everywhere on $[0, 1]$.**

### 2.3 Why Raw FDP Cannot Be Certified by CRC
Raw False Discovery Proportion is defined as:
$$\text{FDP}_i(\lambda) = \frac{\text{FD}_i(\lambda)}{\max(1, R_i(\lambda))}, \quad R_i(\lambda) = \#\{k \in \mathcal{K}_i : r_k \le \lambda\}$$
As $\lambda$ increases, both $\text{FD}_i(\lambda)$ and $R_i(\lambda)$ increase. If a true detection enters the selection, $R_i$ increases while $\text{FD}_i$ remains unchanged, causing $\text{FDP}_i(\lambda)$ to **decrease**. Because FDP is non-monotone, standard CRC cannot be applied to it. Track A avoids this mathematical fallacy by fixing the denominator to $N_i$.

### 2.4 Identified Mathematical Mismatch in Code
An audit of [`src/certification/track_a.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/certification/track_a.py#L70-L73) identified a concrete formula mismatch:
- **Specification Formula** (`SIH26142_v4_hardened.md` line 334 & line 625):
  $$\text{Bound}_{\text{spec}}(\lambda) = \frac{n \hat{L}_n(\lambda) + B}{n + 1} = \frac{n}{n+1} \hat{L}_n(\lambda) + \frac{B}{n+1}$$
- **Current Code Implementation** (`src/certification/track_a.py` line 71):
  ```python
  empirical_risk = np.mean(losses)
  bound = empirical_risk + self.B / (n + 1)
  ```
  Which evaluates to:
  $$\text{Bound}_{\text{code}}(\lambda) = \hat{L}_n(\lambda) + \frac{B}{n + 1}$$
- **Discrepancy Analysis**:
  $$\text{Bound}_{\text{code}}(\lambda) - \text{Bound}_{\text{spec}}(\lambda) = \hat{L}_n(\lambda) \left(1 - \frac{n}{n+1}\right) = \frac{\hat{L}_n(\lambda)}{n+1} \ge 0$$
  The current code omits the $\frac{n}{n+1}$ shrinkage factor on $\hat{L}_n(\lambda)$, making it strictly more conservative than the exact CRC theorem.
- **Proposed Minimal Fix (Reported for approval, NOT yet implemented)**:
  Modify line 71 of `src/certification/track_a.py` to:
  `bound = (n * empirical_risk + self.B) / (n + 1)`
  and update the class docstring on line 15 accordingly.

---

## 3. Track B Audit: Conformal p-values + Multiple Testing (BH / BY)

### 3.1 Null Hypothesis & Conformal p-values
- **Null Hypothesis per Candidate**:
  $$H_k: \text{Candidate } k \text{ is a FALSE detection}$$
- Evidence against the null (i.e. that $k$ is a genuine built structure) is a **low risk score** $r_k$.
- From calibration scenes, risk scores of verified false detections form the null pool:
  $$R^0 = \{r_1^0, \dots, r_m^0\}$$
- For a test candidate score $r_k$, the conformal p-value is:
  $$p_k = \frac{1 + \#\{j \in \{1, \dots, m\} : r_j^0 \le r_k\}}{m + 1}$$
- **Verification in Code**:
  [`src/certification/track_b.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/certification/track_b.py#L47-L51) uses:
  ```python
  sorted_nulls = np.sort(self.null_scores)
  counts = np.searchsorted(sorted_nulls, test_scores, side='right')
  p_values = (1.0 + counts) / (m + 1.0)
  ```
  Because `side='right'`, `counts` is exactly $\#\{j : r_j^0 \le r_k\}$. Small risk scores produce small p-values, which correctly reject the null $H_k$ (certifying the detection).

### 3.2 Benjamini–Hochberg (BH) & Benjamini–Yekutieli (BY) Implementation
- **Sorting & Thresholds**:
  p-values are sorted $p_{(1)} \le \dots \le p_{(N)}$.
  - Under **BH**: $T_k = \frac{k}{N} \alpha$.
  - Under **BY**: $T_k = \frac{k}{N \cdot c(N)} \alpha$, where $c(N) = \sum_{i=1}^N \frac{1}{i}$.
- **Indexing Audit**:
  ```python
  k_array = np.arange(1, N + 1)
  thresholds = (k_array / N) * self.alpha  # or / (N * cm)
  valid_k = np.where(p_sorted <= thresholds)[0]
  if len(valid_k) > 0:
      k_max = valid_k[-1]
      certified_indices = sorted_indices[:k_max + 1]
      certified_mask[certified_indices] = True
  ```
  The 0-based index `k_max` corresponds to the largest rank $k = k_{\max} + 1$ satisfying $p_{(k)} \le T_k$. Slicing `[:k_max + 1]` retains all hypotheses with rank $\le k$, which exactly matches the Benjamini–Hochberg and Benjamini–Yekutieli algorithms.
- **Distinction between FDR and Per-Scene FDP**:
  Track B guarantees $\text{FDR} = \mathbb{E}[\text{FDP}] \le \alpha$. It does **NOT** guarantee that realized FDP on any individual scene will be $\le \alpha$.

---

## 4. Proposal Stage Audit (Task D1)

- **Class**: [`D1ProposalStage`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/decision/proposal.py)
- **Selection Rule**: `candidate_mask = sr_image > self.c_0`
- **Audit Findings**:
  1. `c_0` is calibrated **exclusively on the TRAIN split**:
     `self.c_0 = float(np.quantile(all_pixels, 1.0 - target_yield))`
  2. Once fitted on TRAIN, `c_0` is frozen and never tuned per scene, track, or $\alpha$.
  3. `sr_image > self.c_0` fixes the candidate set $\mathcal{K}_i$ and its cardinality $N_i = |\mathcal{K}_i|$.
  4. $\lambda$ is solely used downstream for candidate retention ($r_k \le \lambda$).
  5. **Conclusion**: $N_i$ is strictly fixed independently of $\lambda$, satisfying the core mathematical assumption of Track A.

---

## 5. Calibration Split & Site Disjointness Audit

- **Split Manifest**: [`data/split_manifest.json`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/data/split_manifest.json)
  - `TRAIN`: `SEN2NAIP_US_SITE_01`, `SEN2NAIP_US_SITE_02`, `SEN2VENUS_SITE_FR_1`, `SEN2VENUS_SITE_IT_1`
  - `CALIBRATE`: `OPENSR_SPOT_SITE_01`, `OPENSR_SPAIN_CROPS_01`
  - `TEST`: `OPENSR_SPAIN_URBAN_01`, `SEN2VENUS_SITE_DE_1`, `WORLDSTRAT_AOI_01`
  - `DEMO`: `INDIAN_AOI_HYDERABAD`, `INDIAN_AOI_PUNJAB`
- **Disjointness Invariant**:
  [`tests/test_splits.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/tests/test_splits.py) executes automated set-intersection assertions:
  $$\text{TRAIN} \cap \text{CALIBRATE} = \emptyset, \quad \text{TRAIN} \cap \text{TEST} = \emptyset, \quad \text{CALIBRATE} \cap \text{TEST} = \emptyset$$
  Whole geographic sites are segregated. No random tiles or patches from calibration sites are permitted in test sets.

---

## 6. Ground-Truth Separation Audit

- **Inference Pipeline**: [`src/pipeline/runner.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/src/pipeline/runner.py)
  Takes only low-resolution Sentinel-2 L2A data.
  Data flow: $y \to \hat{x} \to \hat{D}(\hat{x}) \to e \to s(p) \to r(p) \to \text{Certification}$.
- **Ground-Truth Boundaries**:
  HR reference imagery enters only:
  1. During operator fitting on calibration pairs.
  2. As ground-truth labels `is_false_discovery` on CALIBRATE sites to calibrate $\hat{\lambda}$ and pool $R^0$.
  3. During offline evaluation of TEST scenes.
- **Audit Finding**: Zero HR reference data is ingested or accessible during test scene inference.

---

## 7. Edge-Case Verification

Deterministic testing of edge cases in [`scratch/test_edge_cases.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/certus-s2/scratch/test_edge_cases.py) yielded 100% expected behavior:

| Edge Case | Track A Behavior | Track B Behavior | Status |
| :--- | :--- | :--- | :--- |
| **Zero proposals ($N=0$)** | Returns empty boolean mask `(0,)` | Returns empty boolean mask `(0,)` | PASS |
| **Zero proposals in calibration ($N_i=0$)** | $L_i = 0 / \max(1, 0) = 0.0$ | N/A (no null candidates generated) | PASS |
| **Zero selections** | Returns all False | Returns all False | PASS |
| **All candidates selected** | Certified mask is all True | Certified mask is all True | PASS |
| **No calibration nulls ($m=0$)** | N/A (loss evaluates based on FDs) | Correctly raises `AssertionError` | PASS |
| **Tied risk scores** | Tied candidates selected together | Tied p-values rejected together | PASS |
| **$\alpha \to 0$ (infeasible)** | $\hat{\lambda} = -1.0 \implies$ 0 certified | All $p_k > T_k \implies$ 0 certified | PASS |
| **$\alpha \to 1$** | $\hat{\lambda} = 1.0 \implies$ all certified | Most/all candidates certified | PASS |

---

## 8. Unit Test Results

Execution of certification-specific tests and full suite:
```
tests/test_track_a.py::test_track_a_calibration PASSED                   [ 20%]
tests/test_track_b.py::test_track_b_calibration_and_certification PASSED [ 40%]
tests/test_proposal.py::test_d1_proposal PASSED                          [ 60%]
tests/test_splits.py::test_split_disjointness PASSED                     [ 80%]
tests/test_shift.py::test_shift_evaluator PASSED                         [100%]
============================== 5 passed in 0.12s ===============================

Full Suite:
============================== 17 passed in 2.86s ==============================
```

---

## 9. Claims Status Audit (against `CLAIMS.md`)

### Claims that are Supported
- **Claim 1 (Track A CRC Bound $\mathbb{E}[FD/N] \le \alpha$)**: SAFE. Proved mathematically and bounded on site-disjoint calibration scenes.
- **Claim 2 (Track B FDR Control under PRDS)**: ASSUMPTION-DEPENDENT. Valid when calibration nulls are exchangeable and PRDS holds.
- **Claim 3 (Track B with BY under Arbitrary Dependence)**: SAFE given null exchangeability (B1).
- **Claim 16 (2.5 m Sampling Output)**: SAFE. Spatial scaling $4\times$ verified.
- **Claim 19 (Validity Independent of Risk-Model Quality)**: SAFE. Conformal validity holds even with arbitrary scores.

### Claims that Remain Unsupported / DO NOT CLAIM
- **Claim 4 (Standard CRC controls FDP/FDR directly)**: DO NOT CLAIM. Raw FDP is non-monotone.
- **Claim 5 (Realized Empirical FDP is a guarantee)**: DO NOT CLAIM. It is a finite-sample estimator of expected FDR.
- **Claim 7 ($s=0$ hallucination / $s=1$ correctness)**: DO NOT CLAIM. Support quantifies observational constraint only.
- **Claim 20 (Certified Change Detection)**: DO NOT CLAIM in MVP. The scope is strictly task D1 (binary built-structure presence verification).
- **Claim 26 (Hallucination Probability Output)**: DO NOT CLAIM. $r(p)$ is a decision ranking metric, not a calibrated probability.

---

## 10. Audit Conclusion & Recommendations

1. **Architecture & Specification Alignment**:
   Track A, Track B, D1 Proposal Stage, Abstention Evaluator, and Split Manifest align with the hardened specification.
2. **Identified Issue**:
   `TrackACertifier.calibrate` in `src/certification/track_a.py` uses `bound = empirical_risk + self.B / (n + 1)` instead of `bound = (n * empirical_risk + self.B) / (n + 1)`.
   - The current code is strictly more conservative than the spec.
   - Recommended minimal fix: update line 71 to `bound = (n * empirical_risk + self.B) / (n + 1)`.
3. **Execution Gate**:
   Per instructions, **no production code changes have been made**. Awaiting user instruction before applying the minimal fix or proceeding to the evaluation stage.
