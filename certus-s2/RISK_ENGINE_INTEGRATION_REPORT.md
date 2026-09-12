# Risk Engine Integration & Validation Report

## 1. Overview & Milestone Scope
This report documents the integration and validation of the verified **Measurement Support Map** $s(p)$ into the **Risk Engine** (`src/risk/engine.py`).

The milestone strictly focused on integrating observational evidence into downstream decision risk scoring while maintaining architecture stability, mathematical rigor, pluggable backend abstraction, and strict non-claims regarding probabilities.

---

## 2. Mathematical Formulation & Feature Mapping

### Complete Data Flow
1. **Low-Resolution Observation ($y$)**: Real Sentinel-2 L2A surface reflectance at 10m ($[C, H, W]$).
2. **Super-Resolution Reconstruction ($\hat{x}$)**: SEN2SR 4× super-resolved reflectance at 2.5m ($[C, 4H, 4W]$).
3. **Effective Observation Operator ($\hat{D}$)**: Depthwise blur + sub-pixel shift + 4× block-averaging downsampling mapping $2.5\text{m} \to 10\text{m}$.
4. **Measurement Residual ($e$)**:
   $$e = |y - \hat{D}(\hat{x})| \in \mathbb{R}^{C \times H \times W} \quad (10\text{m})$$
5. **Measurement Support Map ($s(p)$)**:
   $$s(p_{10\text{m}}) = \exp\left(-\frac{e(p_{10\text{m}})^2}{2\sigma_b^2}\right) \in [0, 1]$$
   Broadcasted via $4\times 4$ nearest-neighbor blocks to match the 2.5m reconstruction grid:
   $$s(p) = \text{repeat}_{4\times 4}(s(p_{10\text{m}})) \in [0, 1]$$
6. **Uncertainty Maps**:
   - Model / Epistemic uncertainty: $\sigma_{\text{ep}}(p) \ge 0$
   - Sensor / Aleatoric uncertainty: $\sigma_{\text{al}}(p) \ge 0$
7. **Downstream Decision Risk Score ($r(p)$)**:
   Feature vector per pixel $p$:
   $$X(p) = \begin{bmatrix} s(p) & \sigma_{\text{ep}}(p) & \sigma_{\text{al}}(p) \end{bmatrix} \in \mathbb{R}^3$$
   Mapped via risk scoring model $g_\phi$:
   $$z(p) = w_s s(p) + w_{\text{ep}} \sigma_{\text{ep}}(p) + w_{\text{al}} \sigma_{\text{al}}(p) + b$$
   $$r(p) = \text{clip}\left(\frac{1}{1 + e^{-z(p)}}, 0.0, 1.0\right) \in [0, 1]$$
   Default parameters under `NumpyRiskBackend`:
   $$w_s = -2.0, \quad w_{\text{ep}} = +3.0, \quad w_{\text{al}} = +1.5, \quad b = 0.5$$

---

## 3. Strict Conceptual Distinctions (Non-Claims)

In compliance with `SIH26142_v4_hardened.md` and `CLAIMS.md`:
- **Measurement Support $s(p) \in [0, 1]$**: Quantifies how strongly the Sentinel-2 observation constrains the 2.5m structural hypothesis under $\hat{D}$. It is **NOT** a probability of correctness, and $s \approx 0$ is **NOT** a proof of hallucination.
- **Measurement Residual $e$**: Quantifies the direct discrepancy between reality and the degraded reconstruction under $\hat{D}$.
- **Uncertainties $\sigma_{\text{ep}}, \sigma_{\text{al}}$**: Quantify reconstruction model ambiguity and sensor noise floors.
- **Risk Score $r(p) \in [0, 1]$**: Downstream decision metric used for candidate retention filtering ($r_k \le \lambda$). It is **NOT** a calibrated probability of hallucination or error unless established by formal empirical calibration.

---

## 4. Architectural Stability & Backend Abstraction

The existing pluggable `RiskBackend` ABC and `RiskEngine` architecture are fully preserved:
- **`RiskBackend` (ABC)**: Defines `fit(features, labels)` and `predict_proba(features) -> np.ndarray`.
- **`NumpyRiskBackend`**: Native deterministic fallback with verified monotonicity and bounds enforcement.
- **`SklearnRiskBackend`**: Production-ready wrapper for scikit-learn `LogisticRegression` with automatic fallback when dependencies are missing.
- **Interface Stability**: `compute_risk(support, sigma_ep, sigma_al, residual=None)` preserves 100% backward compatibility for the existing 3-feature signature while optionally supporting direct residual inputs.
- **Zero New Dependencies**: Implemented using pure NumPy and existing standard library tools.

---

## 5. Verification on Verified Real Sentinel-2 Baseline Data

Executing the complete end-to-end chain on the verified real Sentinel-2 L2A crop (`data/input_crop`) and real SEN2SR 2.5m reconstruction (`data/output_2_5m`) yields numerically stable results:

| Band | Residual ($e$) Mean | Residual ($e$) Max | Support ($s(p)$) Mean | Support ($s(p)$) Min | Decision Risk ($r(p)$) Mean | Decision Risk ($r(p)$) Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B02 (Blue)** | 0.000425 | 0.006330 | 0.999908 | 0.987558 | 0.193908 | 0.197798 |
| **B03 (Green)** | 0.000464 | 0.006984 | 0.999889 | 0.984874 | 0.193914 | 0.198651 |
| **B04 (Red)** | 0.000297 | 0.005671 | 0.999954 | 0.990000 | 0.193893 | 0.197024 |
| **B08 (NIR)** | 0.000164 | 0.003557 | 0.999985 | 0.996054 | 0.193884 | 0.195115 |

### Observations:
1. **Mathematical Consistency**: The pixel with the largest residual ($e = 0.006984$ in B03) achieves the lowest support ($s = 0.984874$), which corresponds to the highest decision risk ($r = 0.198651$).
2. **Analytical Match**: For baseline pixels ($s=1.0, \sigma_{\text{ep}}=0.02, \sigma_{\text{al}}=0.01$), the analytical risk is $\sigma(-1.425) = 0.193879$, which matches the empirical output exactly to 6 decimal places.
3. **Absence of NaNs**: All output grids are completely free of `NaN` or `Inf` values.

---

## 6. Deterministic Test Suite Results

The unit test suite was expanded in `tests/test_risk.py` with 6 deterministic test cases:
1. **`test_risk_engine`**: Verifies baseline forward pass and fitting compatibility.
2. **`test_risk_bounds_and_extremes`**: Confirms $r(p) \in [0.0, 1.0]$ across normal and extreme inputs ($\pm 10^5$).
3. **`test_risk_nan_resilience`**: Confirms that `NaN`, $+\infty$, and $-\infty$ in support or uncertainties do not produce `NaN` in $r(p)$ (invalid support treated as zero observational constraint $s=0.0$).
4. **`test_risk_monotonicity_support`**: Confirms $\frac{\partial r}{\partial s} \le 0$ (stronger observational support never increases decision risk).
5. **`test_risk_monotonicity_uncertainty`**: Confirms $\frac{\partial r}{\partial \sigma_{\text{ep}}} \ge 0$ and $\frac{\partial r}{\partial \sigma_{\text{al}}} \ge 0$ (greater uncertainty never decreases decision risk).
6. **`test_risk_residual_to_support_chain`**: Confirms end-to-end consistency from residual $e \to$ support $s \to$ decision risk $r$.

### Full Regression Check
```
============================== 17 passed in 3.04s ==============================
tests/test_observation.py ...             [17%]
tests/test_pipeline.py .                  [23%]
tests/test_proposal.py .                  [29%]
tests/test_risk.py ......                 [64%]
tests/test_shift.py .                     [70%]
tests/test_splits.py .                    [76%]
tests/test_support.py ..                  [88%]
tests/test_track_a.py .                   [94%]
tests/test_track_b.py .                   [100%]
```
Zero regressions across Track A, Track B, proposal, observation operator, split, or shift modules.

---

## 7. Next Steps
1. **Candidate Proposal Stage**: Filter 2.5m reconstruction into candidate detections $\mathcal{K}_i$.
2. **Conformal Certification (Track A & Track B)**: Calibrate risk threshold $\hat{\lambda}$ on calibration sites using the newly wired $r_k$ scores.
3. **Abstention & Shift Testing**: Evaluate covariate shift on test scenes.
