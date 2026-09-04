# SIH26142 — Definitive Technical Design (v4, hardened)
## CERTUS-S2 — *Certified Reconstruction and Trust for Sentinel-2*
### Decision-certified super-resolution: we do not certify pixels, we certify the analytical decisions made from them

**Problem statement:** Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries — NTRO, SIH 2026.

**Supersedes:** v1 (Real-ESRGAN pipeline), v2 (null-space provenance), v3 (decision-level CRC).
**What changed from v3 — and only this changed:** the core thesis, architecture and contributions are unaltered. v3 applied standard conformal risk control directly to the false-discovery proportion. FDP is a ratio whose denominator moves with the threshold, so it is not monotone and standard CRC does not reach it. v4 splits that single overloaded claim into **two separately valid guarantees** (§9.6), forbids tile-level calibration (§9.8), enforces site-disjoint splits (§16), and adds a formal claim audit (§29). Every change is a tightening. Nothing is redesigned.

---

# 1. Executive Summary

Sentinel-2 super-resolution is an underdetermined inverse problem. Every method that outputs a 2.5 m image from a 10 m measurement is *inventing* most of the fine detail, because the measurement does not contain it. The field's leading effort — ESA's OpenSR programme — has already recognized this and shipped strong answers: **SEN2SR** (2.5 m, low-frequency hard constraint), **LDSR-S2** (latent diffusion with an uncertainty map), and **`opensr-test`** (hallucination/omission/improvement metrics against real references). A team proposing "hallucination-aware Sentinel-2 SR" in 2026 is proposing the state of the art, not advancing it.

**So we move the question.** The operationally important question is not *"is this pixel sharp and honest?"* It is:

> **"If an analyst acts on this product — flags a new structure, reports a crop-stress area, triggers a disaster response — what is the probability that decision is wrong, and can that probability be guaranteed rather than hoped for?"**

CERTUS-S2 answers that. It wraps a strong existing SR backbone in three layers that do not currently exist together anywhere we could find:

1. **An effective observation operator, fitted with its own uncertainty**, giving a measurement-support estimate — how strongly the Sentinel-2 observation constrains each spatial-frequency component, under the assumed operator and noise model.
2. **Conformal certification applied at the decision level**, not the pixel level. Two guarantees, each with its own stated loss and assumptions: a finite-sample conformal risk bound on a **monotone normalized false-discovery loss**, and — separately — **false-discovery-rate control over the selected detections** via conformal p-values with Benjamini–Hochberg. Empirical FDP is reported as a diagnostic and is never itself the certified quantity.
3. **An explicit guarantee-transfer and abstention layer.** In Earth observation, high-resolution references exist in the US, Europe and a handful of Venµs sites — and essentially nowhere you actually need to deploy. Calibration is therefore *always* under geographic shift. We quantify that shift, report the resulting estimated inflation instead of hiding it, and **abstain** where the guarantee cannot be honestly transferred.

**The product is not an image.** It is a georeferenced multi-layer product in which every analytical output carries a risk certificate, a provenance record, and an explicit statement of where the system declines to answer.

**Honest ownership:** the SR backbone is ESA's. Conformal risk control is Angelopoulos et al.'s; conformal-p-value selection is Bates et al.'s and Jin & Candès's. Effective-rank inverse-problem analysis is classical. **Ours is the composition, the operator anchoring of the risk score, the decision-level target, and the shift-aware transfer with abstention** — and we say exactly that, in §31 and on a slide.

**One scoping statement, made first rather than extracted under questioning:** we do not claim CERTUS-S2 produces sharper or more accurate imagery than SEN2SR or LDSR-S2. The reconstruction is theirs. The novelty is the trust and decision layer around it.

---

# 2. SIH26142 Problem Interpretation

Official PS (NTRO): transform 10 m Sentinel-2 into "sharper, information-rich products (<4 m) while preserving geospatial and spectral consistency," covering "preprocessing, model training, accuracy assessment, and validation against high-resolution references," for crop monitoring, urban analysis and disaster assessment, "while accounting for uncertainty and error components."

Four clauses drive the whole design:

| PS clause | What most teams will do | What it actually demands |
|---|---|---|
| "<4 m" | Pick 4× and stop | A *sampling* target, not a resolution guarantee. We must say what is actually resolved vs merely sampled. |
| "preserving spectral consistency" | Report SAM | A constraint on the *estimator*, not a metric added at the end |
| "validation against high-resolution references" | Compute PSNR on a held-out split | References exist only in specific geographies — so this clause silently implies a **transfer problem** |
| **"accounting for uncertainty and error components"** | Ship a confidence heatmap | *Components*, plural: aleatoric, epistemic, operator, and observation error are different things and must be reported separately |

The last clause is the one nearly everyone will under-read. It is written into the expected outcome, and it is where we build.

---

# 3. Real-World Problem

An NTRO-style analyst receives a 2.5 m super-resolved scene. They see a rectangular structure that is not in the previous acquisition. Three possibilities:

1. A real new building, genuinely recoverable from the measurement.
2. A real new building, *not* determined by the measurement — the model's prior placed a plausible rectangle where the data only supports "something bright and blob-like."
3. No building. The prior produced a rectangle because rectangles are what its training data contains.

Cases 2 and 3 are visually identical to case 1. **The image cannot distinguish them, and no amount of sharpening will.** Under current products the analyst has two options: distrust everything (in which case the SR product has no operational value), or trust everything (in which case the product is a liability). Both are failures.

This is not a hypothetical framing. It is exactly why ESA named its programme *trustworthy* SR. The unsolved part is that current trust outputs are **descriptive** (here is a variance map) rather than **operational** (here is the error rate you will incur if you act on this, and here is where you must not act at all).

---

# 4. Existing State of the Art

### 4.1 Sentinel-2 super-resolution

| System | What it does | Relevance |
|---|---|---|
| **SEN2SR** (ESA OpenSR) | 10 m + 20 m bands → 2.5 m; CNN/Swin/Mamba backbones; **low-frequency hard constraint** so the model only adds high-frequency detail while coarse reflectance is preserved | **Our backbone and primary baseline.** Solves radiometric consistency well. |
| **LDSR-S2 / `opensr-model`** | Latent diffusion, RGB-NIR 4-ch, 128→512 (4×); **already produces an uncertainty map** via posterior sampling | Directly comparable trust output. Our risk layer must beat *this*, not TTA. |
| **DiffFuSR** | Diffusion SR on RGB + learned fusion for remaining bands, all 12 bands → 2.5 m; evaluated on the OpenSR benchmark | Strong all-band baseline |
| **Sen4x** ("Beyond Pretty Pictures", 2025) | Hybrid SISR+MISR; found **pure MISR underperforms SISR** (38.7 vs 48.9 mIoU), hybrid best (51.6), HR reference ceiling 66.3; **PSNR/SSIM correlate poorly with downstream utility** | Our evidence base for rejecting naive multi-temporal, and for downstream-first evaluation |
| **L1BSR** | Self-supervised SR from L1B **detector overlap** — real subpixel-shifted observations, no synthetic degradation | Evidence that genuine sub-pixel information exists in S2; a roadmap data source |
| **`opensr-test`** | Standardized benchmark: consistency, synthesis, and **correctness (hallucination / omission / improvement)** against real HR references; 5 datasets, 178 scenes | **Our evaluation harness.** Non-negotiable for credibility. |

### 4.2 Uncertainty, conformal and risk control

| Work | What it provides | Assumption / limit |
|---|---|---|
| **RCPS / im2im-uq** (Angelopoulos et al., ICML 2022) | Distribution-free per-pixel intervals for image-to-image regression with risk control | Exchangeable calibration data |
| **Conformal Risk Control** (Angelopoulos et al., 2023) | Controls the expectation of **any bounded, monotone loss** by calibrating a scalar λ; public PyTorch code | Exchangeability; **loss must be monotone in λ and bounded**. FDP is a ratio and does not satisfy this — see §9.6. |
| **Testing for Outliers with Conformal p-values** (Bates, Candès, Lei, Romano, Sesia; AoS 2023) | Conformal p-values are super-uniform under the null; **BH applied to them controls FDR** under the PRDS property induced by a shared calibration set | Requires exchangeability of **null** units between calibration and test |
| **Selection by Prediction with Conformal p-values** (Jin & Candès, 2023) | Extends conformal-p-value selection to picking units whose outcome exceeds a threshold, with FDR control | Same exchangeability requirement |
| **Benjamini–Yekutieli** (2001) | FDR control under **arbitrary** dependence, at a `Σ 1/i` power cost | Conservative |
| **Conformalized generative SR** (arXiv 2502.09664, Feb 2025) | Risk-controlled "confidence mask" for SR; bounds fidelity error; black-box models | Natural photographs; **no observation model**; assumes exchangeable calibration |
| **QUTCC** (2507.14760) | Conformal, spatially-adaptive pixel-marginal coverage for imaging inverse problems | Pixel-marginal only; fails on rare OOD events |
| **Self-supervised conformal via SURE** (2502.05127) | Conformal calibration **without ground truth**, via Stein's estimator on the measurements | Stated for **ill-conditioned** linear problems |
| **Task-driven UQ in inverse problems** (ECCV 2024) | Conformal intervals **in task-output space** (MRI → meniscus-tear detection) | Scalar task output; marginal coverage, **explicitly not FDR control**; names spatial tasks and RCPS as **future work** |
| **Conformal prediction beyond exchangeability** (Barber et al., AoS 2023) | Coverage-gap bound under distribution shift | Needs a shift measure; the bound is not estimable in high dimensions from few scenes |
| **Weighted conformal under covariate shift** (Tibshirani et al., 2019) | Restores validity with density-ratio weights | Needs estimable likelihood ratio and support overlap |
| **Conformal UQ in EO** (Sci. Rep. 2024), **GeoConformal** | Conformal applied to EO classification / spatial prediction | Not reconstruction; not SR |

---

# 5. Gap Analysis

**Solved — claim none of it:**
- Radiometrically consistent S2 SR to 2.5 m (SEN2SR).
- Posterior-sample uncertainty maps for S2 SR (LDSR-S2).
- Measuring hallucination against HR references (`opensr-test`).
- Risk-controlled trust masks for SR in general (conformalized generative SR, Feb 2025).
- Conformal UQ for EO classification/segmentation.
- FDR control via conformal p-values in the abstract (Bates et al.; Jin & Candès).

**Partially solved:**
- Trust maps exist but are **descriptive**: an uncertainty map with no stated error rate tells an analyst nothing actionable.
- Conformal SR exists but is **physics-free**: its uncertainty is generative-sample variance, with no observation model and no data-consistency residual.
- Task-level conformal UQ exists but for a **scalar** output on **medical** data, with marginal coverage and explicitly **not** FDR control — the authors themselves name spatial tasks and RCPS as future work.
- Conformal calibration without ground truth exists (SURE) but is posed for **ill-conditioned** problems. Our operator is **rank-deficient**: 4× decimation gives ~16× fewer measurements than unknowns per band. In the genuinely unobserved subspace there is *no measurement information*, so a measurement-domain risk estimator cannot bound the error there. **This is the precise technical reason we calibrate on real HR references rather than self-calibrating.**

**Genuinely missing — this is where we build:**

| # | Gap | Why it matters |
|---|---|---|
| **G1** | No SR trust score is **anchored in a fitted effective observation model** with its own quantified uncertainty. Existing scores are generative variance. | A variance-only score cannot distinguish "the model is unsure" from "the observation weakly constrains this." |
| **G2** | No one controls the **risk of the analytical decision** made from a super-resolved satellite product. Certification stops at the image. | Analysts act on decisions, not pixels. An image certificate does not bound the false-discovery rate of a structure report. |
| **G3** | Every conformal SR method assumes **exchangeable calibration data**. In EO, HR references exist in the US, Europe and 29 Venµs sites; deployment is Indian terrain, monsoon seasons, unfamiliar land cover. **Deployment is structurally out of domain.** Nobody addresses guarantee transfer, and nobody abstains. | This is the single largest barrier to operational use, and it is invisible in every benchmark-based paper. §19.2 measures how fast the guarantee breaks. |

---

# 6. Our Final Product

**CERTUS-S2** is a georeferenced Sentinel-2 reconstruction product in which every analytical layer carries a certified error rate or an explicit refusal.

```
   Sentinel-2 L2A  (10 m / 20 m / 60 m + SCL + metadata)
             │
             ▼
   ┌──────────────────────────────────────────────────┐
   │  A. RECONSTRUCTION                               │
   │     SEN2SR / LDSR-S2 backbone (EXISTING TECH)    │
   │     + soft data-consistency projection under a   │
   │       FITTED effective observation operator D̂    │
   └────────────────────┬─────────────────────────────┘
                        ▼
   ┌──────────────────────────────────────────────────┐
   │  B. MEASUREMENT SUPPORT       (CONTRIBUTION C1)  │
   │     support estimate s(p): how strongly the      │
   │     observation constrains local structure,      │
   │     under noise + operator-fit uncertainty       │
   └────────────────────┬─────────────────────────────┘
                        ▼
   ┌──────────────────────────────────────────────────┐
   │  C. RECONSTRUCTION RISK SCORE                    │
   │     r(p) = f(support, residual, epistemic,       │
   │             aleatoric, operator-uncertainty)     │
   │     — a *score*, not a probability               │
   └────────────────────┬─────────────────────────────┘
                        ▼
   ┌──────────────────────────────────────────────────┐
   │  D. DECISION CERTIFICATION    (CONTRIBUTION C2)  │
   │     Track A: CRC on a MONOTONE normalized        │
   │       false-discovery loss   → E[L] ≤ α          │
   │     Track B: conformal p-values + BH             │
   │       → FDR of the selected set ≤ α              │
   │     Diagnostic (not certified): empirical FDP    │
   └────────────────────┬─────────────────────────────┘
                        ▼
   ┌──────────────────────────────────────────────────┐
   │  E. TRANSFER + ABSTENTION     (CONTRIBUTION C3)  │
   │     shift score → estimated inflation → DEGRADED │
   │     or ABSTAIN. In-domain: guarantee.            │
   │     Out-of-domain: diagnosis, never a guarantee. │
   └────────────────────┬─────────────────────────────┘
                        ▼
   ┌──────────────────────────────────────────────────┐
   │  F. CERTIFIED PRODUCT                            │
   │     imagery + support + risk + certificate +     │
   │     abstention mask + provenance (COG + STAC)    │
   └──────────────────────────────────────────────────┘
```

**Selective super-resolution is the operating mode.** The system does not release fine detail uniformly. For a given analytical task and a given tolerated error rate α, it releases detail only where the certificate holds, marks the rest as prior-dependent, and refuses entirely where the calibration domain does not cover the input. That refusal is a feature: *a system that never abstains is a system that has not understood its own limits.*

---

# 7. Core Research Question

> Can the reliability of a super-resolved satellite reconstruction be certified **at the level of the analytical decision** — with a finite-sample, distribution-free guarantee under stated exchangeability assumptions — using only quantities computable at inference time without a local high-resolution reference, and can the *limits* of that certificate be honestly diagnosed where no reference exists?

Three sub-questions, each independently testable:

- **Q1 (support):** Does an operator-anchored measurement-support estimate carry information about reconstruction error beyond what generative variance already provides?
- **Q2 (certification):** Do the two certification tracks deliver their promised error rates on site-disjoint held-out data?
- **Q3 (transfer):** Under geographic shift, how fast does the guarantee degrade, and does abstention fire before it breaks?

Note the deliberate wording of Q3. We do not ask whether the certificate *survives* shift. We ask how it *fails*, and whether we detect the failure in time. §19.2 answers this quantitatively.

---

# 8. Novelty and Contributions

Three contributions. Each states what exists, what does not, and how it is validated.

### C1 — Operator-anchored measurement support under operator uncertainty

- **Problem:** Existing SR trust signals are generative-sample variance. They conflate "the model is unsure" with "the observation weakly constrains this."
- **What exists:** LDSR-S2's posterior-sample uncertainty; conformalized generative SR's variance-based mask; classical inverse-problem effective-rank analysis.
- **What does not:** an S2-specific, *fitted* effective observation operator whose **own parameter uncertainty is propagated** into a per-pixel, per-frequency support estimate, delivered as a product layer.
- **What we do:** fit `D̂` from same-day cross-sensor pairs with a jointly fitted sub-pixel shift and a regularized kernel; bootstrap **over independent sites** to get operator uncertainty; compute spectral support under the joint noise + operator-uncertainty budget; emit `s(p)`.
- **Why it matters:** it is the only trust input that speaks to *"how strongly was this constrained by the observation?"* rather than *"is the model confident?"*
- **Validation:** ablation E-C1 — does adding `s` improve risk-score AUSE and raise recall at a fixed certified error rate versus variance-only? Plus an operator-sensitivity ablation (A4b) perturbing `θ̂` within its bootstrap CI.
- **Honest bound on the claim:** `s` is **not** a fabrication detector. In our controlled-simulator ablation (§19.1) the operator-anchored features alone selected nothing at α = 0.10; they contribute as part of a feature set, not as a standalone discriminator. We report this rather than implying otherwise.

### C2 — Decision-level certification for a satellite reconstruction product ← **the headline**

- **Problem:** Certifying image fidelity does not bound the error rate of the decision an analyst makes.
- **What exists:** conformal risk control for monotone losses; conformal-p-value selection with FDR control (Bates et al.; Jin & Candès); conformalized generative SR (certifies *image fidelity*, natural photos); task-driven UQ in inverse problems (certifies a *scalar* task output for MRI, marginal coverage, explicitly not FDR).
- **What does not:** either machinery applied to a **spatial analytical decision** derived from a **satellite reconstruction**, with an operator-anchored risk score, and with the shift behaviour measured rather than assumed.
- **What we do:** define one tightly specified decision task (§21.1); certify it two ways, each with its own valid loss and stated assumptions (§9.6); report empirical FDP separately as an uncertified diagnostic.
- **Why it matters:** it converts an SR product from "looks trustworthy" into "carries a stated, tested error rate" — the minimum bar for entry into an operational analysis chain.
- **Validation:** realized loss vs nominal α on site-disjoint test scenes, for both tracks, across α ∈ {0.05, 0.1, 0.2}.

### C3 — Guarantee transfer under geographic shift, with principled abstention

- **Problem:** Conformal guarantees require exchangeability. HR references exist where deployment doesn't. Every existing conformal SR result is therefore, in operational terms, an in-domain result.
- **What exists:** weighted conformal under covariate shift (Tibshirani et al.); conformal beyond exchangeability with a coverage-gap bound (Barber et al.); OOD detection in EO.
- **What does not:** their application to satellite reconstruction, with a **measured** shift-sensitivity curve, an explicitly **estimated** (never certified) inflation, and an **abstention rule** tied to it.
- **What we do:** estimate a domain-shift score from measurement-domain features; where support overlaps and a ratio is estimable, reweight; otherwise report an estimated inflation and abstain past tolerance.
- **Why it matters:** it is the difference between a benchmark result and a deployable product — and it is the honest answer to *"what happens over India?"*
- **Validation:** §19.2's shift-sensitivity sweep, plus A10/A11 on induced geographic shift.

**What we explicitly do NOT claim as novel:** the SR backbone, conformal prediction, conformal risk control, conformal p-values, Benjamini–Hochberg, the hallucination benchmark, PSF fitting as a technique, or "uncertainty-aware SR" as a concept. The novelty is the composition, its operator anchoring, its decision-level target, and its measured transfer behaviour. **If a judge says "you assembled known parts into a new operational guarantee," that is our claim, stated first by us.**

---

# 9. Mathematical Formulation

## 9.1 What was wrong in earlier versions — and the fixes

**v2's error (fixed in v3):** an exact range/null decomposition giving `D X̂ = Y` exactly, with null-space content treated as "hallucination." Exactness is false under noise and operator error; and null-space content may be entirely correct. Both corrected in §9.3 and §9.4.

**v3's error (fixed here):** v3 defined the CRC loss as

```
L(λ) = |{false detections at λ}| / max(1, |{reported at λ}|)          ← FDP
```

and applied standard conformal risk control to it. **This is not valid.** CRC requires a loss that is monotone in λ and bounded. FDP is a *ratio* whose denominator is itself a function of λ: tightening λ removes detections from numerator and denominator simultaneously, so FDP can rise or fall non-monotonically. The CRC theorem does not apply, and any FDP/FDR claim derived that way is unsupported.

The fix is not to weaken the project. It is to use the right tool for each quantity, which §9.6 does with two separate, individually valid tracks.

## 9.2 Observation model

For band `b`, with `X_b` the latent 2.5 m reflectance field:

```
Y_b  =  S_{s_b} ( h_b(θ_b) * X_b )  +  e_b ,        e_b = n_b + m_b
```

- `h_b(θ_b)` — **effective observation operator** with fitted parameters `θ_b`, absorbing optical MTF, detector footprint, L2A resampling, residual co-registration error, and cross-sensor radiometric and atmospheric differences. **It is not the physical Sentinel-2 MTF and is never described as such** (§32 attack A7).
- `S_{s_b}` — decimation: `s_b = 4` (10 m bands), `s_b = 8` (20 m bands).
- `n_b` — sensor noise, modelled signal-dependent Gaussian.
- `m_b` — **model error**: operator misspecification, residual registration error, cross-sensor radiometric mismatch. Estimated empirically from calibration residuals, not assumed zero.

Stacked: `Y = D̂(X) + e`, with combined error covariance `Σ = Σ_n + Σ_m`, and `θ` carrying its own posterior uncertainty from the fit.

## 9.3 The feasible set, not a point constraint

```
𝒞(Y) = { X : ‖ D̂(X) − Y ‖²_{Σ⁻¹}  ≤  τ }
```

with `τ` from the χ² quantile of the residual distribution at the fitted noise + model-error level. Any `X ∈ 𝒞(Y)` is compatible with the observation. **The reconstruction problem is the selection of one element of 𝒞(Y) using a prior — and the size of 𝒞(Y) in a given direction is exactly how much freedom the prior had.**

Implementation: a soft projection — `k ≈ 5–10` conjugate-gradient steps on `min_X ‖D̂X − Y‖²_{Σ⁻¹} + ρ‖X − G_θ(Y,C)‖²`, with a Morozov-style stopping rule at the discrepancy level `τ`. **We report the achieved residual; we do not claim exactness.**

## 9.4 Taxonomy of information

| Term | Definition | Computable at inference without HR? |
|---|---|---|
| **Measurement-constrained** | Directions in which `𝒞(Y)` is narrow relative to tolerance ε | ✅ yes |
| **Measurement-underdetermined** | Directions where `𝒞(Y)` is wide: true null space *plus* directions attenuated below the noise floor. **May still be correct.** | ✅ yes |
| **Prior-dependent** | The portion of `X̂` selected by `G_θ` within the underdetermined subspace | ✅ yes |
| **Aleatoric uncertainty** | Irreducible scene/measurement noise | ✅ predicted |
| **Epistemic uncertainty** | Model/prior uncertainty; reducible with more data | ✅ sampled |
| **Operator uncertainty** | Uncertainty in `θ̂` itself, propagated | ✅ bootstrap over sites |
| **Reconstruction error** | `‖X̂ − X_true‖` | ❌ needs HR reference |
| **Hallucination** | *Structured, confident, incorrect* detail — a specific failure mode | ❌ needs HR reference (`opensr-test`'s `ha_metric`) |

**Consequence, stated plainly:** we predict **reconstruction risk**, not "hallucination probability." Calling our output a hallucination probability would be unjustified, so we do not.

## 9.5 Measurement support — conservative interpretation

Because `S∘H` is periodically shift-varying, a clean Fourier analysis is approximate; we use it deliberately and state the approximation. Per band, the pre-decimation transfer is `Ĥ_b(f) = FFT(h_b)`. Define the **measurement-support estimate**

```
s_b(f)  =  |Ĥ_b(f)|²  /  ( |Ĥ_b(f)|²  +  SNR_b⁻¹  +  ν_θ(f) )        ∈ [0,1]
```

where `ν_θ(f)` is the variance of `|Ĥ_b(f)|²` induced by the operator-fit posterior.

**What `s` means, stated conservatively and used consistently throughout this document:**

> `s_b(f)` is an estimate of **how strongly the observation constrains the component at frequency `f`, under the assumed effective operator and noise model.** It is a statement about the *observation process*, not about the correctness of any particular reconstructed structure.

**What `s` does NOT mean — and these phrasings appear nowhere in this document, the slides, or the demo:**

- `s = 0` does **not** mean "hallucinated." Weakly constrained content may be entirely correct; the prior can guess right.
- `s = 1` does **not** mean "perfectly measured." It means strongly constrained *under the assumed operator*, which is itself fitted and uncertain.
- `s` is **not** a hallucination probability, and **not** a per-structure fabrication score.

**Aliasing caveat, preserved everywhere `s` is discussed:** above the post-decimation Nyquist frequency, decimation *folds* frequencies. Aliased components are present in the measurement and are what makes any super-resolution possible at all, but folded components **are not independently identifiable** without a prior. Above Nyquist, `s` therefore measures *potential* support, not identifiability. The two regimes are reported separately and are never averaged into a single number.

A per-pixel support map `s(p)` is obtained by local windowed spectral analysis of the reconstruction weighted by `s_b(f)`.

## 9.6 Decision certification — two tracks, two guarantees

This section replaces v3's single FDP claim. **Read the two tracks as separate products with separate assumptions.** Neither is presented as the other.

### Common setup

Fix a decision task (§21.1). For calibration or test unit `i` (a **scene** — see §9.8):

- A **λ-independent proposal stage** produces a candidate set `𝒦_i` with `N_i = |𝒦_i|`. The proposal threshold `c₀` is fixed once on the TRAIN split and is **never** re-tuned per α, per track, or per scene. This fixity is what makes `N_i` a legitimate λ-independent denominator.
- Each candidate `k` carries a risk score `r_k ∈ [0,1]` from `g_φ` (low = trustworthy).
- The HR reference labels each candidate **true** or **false** (§21.1 defines both precisely).

### Track A — Conformal risk control on a monotone loss ✅ standard CRC, fully valid

Retention rule: retain `k` iff `r_k ≤ λ`. Define the **normalized false-discovery count**

```
L_i(λ)  =  FD_i(λ) / max(1, N_i) ,      FD_i(λ) = #{ k ∈ 𝒦_i : r_k ≤ λ  and  k is false }
```

**Monotonicity, by construction:** raising λ can only enlarge the retained set (the sets are nested), so `FD_i(λ)` is non-decreasing in λ, and `N_i` does not depend on λ. Hence `L_i` is non-decreasing in λ and bounded by `B = 1`. Both CRC hypotheses hold.

Given `n` exchangeable calibration **scenes**, CRC selects

```
λ̂  =  sup { λ :  ( n · L̂_n(λ)  +  B ) / (n + 1)   ≤  α } ,       L̂_n(λ) = (1/n) Σ_i L_i(λ)
```

(the supremum rather than the infimum because our loss increases with λ; this is the standard CRC statement under the substitution λ → −λ), yielding the finite-sample, distribution-free guarantee

```
E[ L_test(λ̂) ]  ≤  α
```

**What this certifies, in words an analyst can act on:** *on a fresh scene from the calibration distribution, the expected number of false detections that survive is at most α times the size of the candidate pool.* Equivalently, at most `α · N` false detections per scene in expectation.

**What it does NOT certify:** the false-discovery *proportion* among reported detections. `FD/N ≤ α` does not imply `FD/R ≤ α`, because the retained count `R ≤ N`. We never present Track A's number as an FDP or an FDR.

### Track B — FDR control via conformal p-values + Benjamini–Hochberg ✅ valid, assumptions stated

Track A bounds a count. An analyst looking at a list of flagged structures wants the *proportion* of that list that is wrong. That is FDR, and it is reachable — not by forcing CRC onto a non-monotone loss, but by treating detection as a multiple-testing problem.

Define the null hypothesis per candidate:

```
H_k :  candidate k is a FALSE detection.
```

From the calibration scenes, collect the risk scores of candidates labelled false by the HR reference: `R⁰ = { r⁰_1, …, r⁰_m }`. For a test candidate with score `r_k`, define the **conformal p-value**

```
p_k  =  ( 1 + #{ j : r⁰_j ≤ r_k } ) / ( m + 1 )
```

**Validity (super-uniformity).** If `H_k` holds, `r_k` is exchangeable with `R⁰`, so its rank among `R⁰ ∪ {r_k}` is uniform, giving `P(p_k ≤ t) ≤ t` for all `t ∈ [0,1]`. This is the standard conformal p-value construction (Vovk; Bates et al. 2023).

Apply **Benjamini–Hochberg** at level α to `{p_k}`: reject `H_k` (i.e. *select* candidate `k` as a reportable detection) for the largest `K` with `p_(K) ≤ αK/M`. Then

```
FDR  =  E[ #{selected and false} / max(1, #selected) ]   ≤   α · M₀/M   ≤   α
```

**The three assumptions, each stated and each tested:**

| # | Assumption | Status | Where tested |
|---|---|---|---|
| B1 | **Exchangeability of null units** between calibration and test | The binding assumption. Broken by geographic shift. | §19.2 measures exactly how fast it breaks |
| B2 | **PRDS dependence** among the p-values, which BH requires | Conformal p-values from a *shared calibration set* are PRDS (Bates et al. 2023, Thm 1). Spatial correlation among test candidates is *additional* dependence not covered by that theorem. | §19.1 clustering sweep |
| B3 | Calibration null scores are exchangeable **at the candidate level** | Violated by within-scene clustering: candidates from one scene share illumination, land cover and geometry. | §19.1 clustering sweep |

**Our handling of B2 and B3 — the assumption-free fallback.** Where we cannot argue PRDS, we report the **Benjamini–Yekutieli** variant, which controls FDR under *arbitrary* dependence by running BH at level `α / Σ_{i=1}^{M} (1/i)`. It costs roughly a `ln M` factor in power and buys unconditional validity. **Both BH and BY numbers are reported in every results table.** The BH number is the headline only where the PRDS argument is defensible; the BY number is the one we stand behind unconditionally.

**Empirically (§19.1), within-scene clustering degrades power severely but did not breach validity** — realized FDR stayed at or below α across clustering strengths, while recall fell from 0.665 to 0.056. The dangerous failure is not clustering. It is shift, which is Track B's assumption B1 and is measured in §19.2.

### Empirical FDP — its status differs between the tracks, and the difference matters

`FDP = #false / max(1, #reported)` is computed and reported at every operating point of both tracks. **Its status is not the same in each, and conflating them would be exactly the slippage this revision exists to remove:**

- **Under Track A, empirical FDP is an uncertified diagnostic.** Track A bounds `E[FD/N]`. Because the retained count `R ≤ N`, a bound on `FD/N` does **not** bound `FD/R`. Track A's certified number is never presented as an FDP or an FDR, and FDP sits beside it in a column explicitly headed *uncertified diagnostic*.
- **Under Track B, empirical FDP is the natural finite-sample estimator of the certified quantity**, because Track B certifies `FDR = E[FDP]`. It is an estimate of something we do hold a guarantee on — not itself a guarantee. Two consequences, stated wherever Track B numbers appear: (i) the bound is on the **expectation**, so FDP on an individual scene may exceed α without any violation; (ii) a realized FDP below α on a finite test set is **evidence consistent with** the guarantee, never a proof of it.

The certificate's `certified_quantity` field (§18) names which of the two a consumer is holding, so the distinction survives contact with downstream users who never read this document.

### One-sentence summary, for the abstract and the slide

> **We provide (i) a finite-sample conformal risk bound on a monotone normalized false-discovery loss, and (ii) finite-sample FDR control over the selected detections via conformal p-values with Benjamini–Hochberg — under stated exchangeability assumptions, with a Benjamini–Yekutieli variant valid under arbitrary dependence. Empirical FDP is reported throughout: as an uncertified diagnostic for the first guarantee, and as the finite-sample estimator of the second.**

## 9.7 Transfer under shift — what is guaranteed and what is not

**Absolute distinction, applied everywhere in this document:**

| Regime | What we provide | What we call it |
|---|---|---|
| **In-domain** (shift score within calibration support) | Finite-sample conformal guarantee under the stated exchangeability and dependence assumptions of §9.6 | **Guarantee** |
| **Out-of-domain** | Shift diagnosis, an **estimated** risk inflation `Δ̂`, and abstention past tolerance | **Diagnostic** — never a guarantee |

Two mechanisms, both implemented:

1. **Weighted conformal** (Tibshirani et al.): reweight calibration scenes by an estimated likelihood ratio `w(x) = dP_test/dP_cal` from a domain classifier on measurement-domain features. Valid **only** where the ratio is estimable and the supports overlap. We apply it only when a support-overlap test passes.
2. **Estimated inflation reporting**: report `α + Δ̂` where `Δ̂` is an empirical distributional-distance estimate. **`Δ̂` is explicitly not a certified bound** — the Barber et al. coverage-gap term is not reliably estimable in high dimensions from a handful of scenes (§32.1 W2).

**Abstention rule:** given tolerance `α_max`, if `α + Δ̂ > α_max`, or if the shift score exceeds the calibration support, the system **abstains** for that region and task. Abstention is a delivered mask layer with a reason code, not a silent quality drop.

**We do not claim arbitrary geographic-shift guarantees anywhere.** The Indian AOI work is labelled throughout as an **out-of-domain transfer demonstration**, reporting shift score, behaviour, estimated inflation and abstention rate — not as a certified deployment.

## 9.8 Calibration units — scenes, never tiles

**The rule, stated once and never contradicted:**

> A calibration unit is an **independent scene**, or an explicitly **spatially blocked group of scenes** from disjoint sites. Tiles drawn from a shared scene are **never** treated as independent calibration units.

Tiles from one scene share illumination, atmospheric state, land cover, acquisition geometry and sensor unit. Treating them as independent inflates the effective `n`, shrinks the `B/(n+1)` finite-sample term, and makes the bound look tighter than it is. The finite-sample correction uses `n = ` the number of **independent calibration units**, and that number is printed on the certificate.

**When `n` is too small, we say so.** The consequence is a **more conservative, less useful certificate** — a smaller `λ̂`, lower retention, and possibly no feasible threshold at all. The remedy is **more geographic diversity in the reference archive**, not a redefinition of the sampling unit. In our controlled simulator (§19.1), `n = 5` calibration units admitted *no* feasible threshold at α = 0.10, `n = 10` gave 0.099 retention, and `n = 40` gave 0.199 — the finite-sample penalty `1/(n+1)` consumes 17%, 9% and 2.4% of the risk budget respectively. **This is a concrete, quantified ask of any agency wanting a certified deployment: supply referenced scenes from `n` distinct sites, and here is what each additional site buys.**

There is **no fallback anywhere in this document that increases `n` by subdividing scenes into tiles.** Any earlier text suggesting that (v3 §27 R4) is withdrawn.

---

# 10. Effective Observation Operator — Fitting Procedure

We do **not** quote Sentinel-2 MTF specification values, and we do not describe the result as the physical MTF. Per band, `h_b` is a small separable parametric kernel (Gaussian `σ_b` plus an optional fitted 5×5 residual), fitted jointly with a sub-pixel registration shift:

```
(θ̂_b, δ̂_b)  =  argmin_{θ,δ}  Σ_pairs  ‖ S_{s_b}( h_b(θ) * T_δ(X_HR) )  −  Y_obs ‖₁  +  γ·R(θ)
```

where `T_δ` is a sub-pixel translation initialized by phase correlation, and `R(θ)` is a regularizer penalizing kernel roughness and departure from radial symmetry (γ selected on the TRAIN split only).

**What the fitted kernel absorbs — stated explicitly because it bears directly on how much the result can be claimed:**

| Absorbed into `θ̂` | Consequence for interpretation |
|---|---|
| True optical blur + detector footprint | The part we actually want |
| L2A resampling | Legitimate — it is part of the delivered product's observation chain |
| Residual co-registration error | Inflates apparent blur; partly mitigated by the joint `δ` fit |
| Cross-sensor radiometric differences (VENµS vs S2 spectral response) | Biases the fitted amplitude, not primarily the width |
| Atmospheric and BRDF differences | Reduced by same-day pairing, not eliminated |
| Temporal differences | Reduced to near-zero by same-day pairing — the reason SEN2VENµS is the fitting set |

**Therefore `D̂` is an *effective observation operator*, never "the Sentinel-2 MTF."** This terminology is used consistently in §9.2, §11, §12, §19, §28, §29 and §31.

**Uncertainty quantification:** `θ̂` uncertainty comes from a **bootstrap over independent sites** (never over tiles within a site — §9.8). Fitted `σ_b` and their bootstrap confidence intervals are **reported in the submission and printed on the product's provenance record**.

**Operator sensitivity ablation (A4b), mandatory:** we re-run the full certification pipeline with `θ̂` perturbed to the edges of its bootstrap CI, and report how `λ̂`, retention and realized risk move. If the certificate is fragile to operator misspecification, that is a finding we publish. This experiment is what converts "we fitted an operator" into "we know what our operator error costs."

---

# 11. Reconstruction Architecture

```
INPUT   S2 L2A: 10 m [B02 B03 B04 B08] · 20 m [B05 B06 B07 B8A B11 B12]
        60 m [B01 B09] → conditioning only (B10 absent in L2A)
        + SCL validity mask + CRS/geotransform
   │
   ├─ PREPROCESS   rasterio → float reflectance; SCL mask; 20 m co-grid;
   │               tile 128² LR / 32 px overlap
   │
   ├─ BACKBONE  G_θ   [EXISTING TECH — ESA]
   │   SEN2SR (primary) or LDSR-S2 (diffusion, for posterior samples)
   │   frozen or LoRA-adapted;  K=8 stochastic samples → σ_ep
   │
   ├─ EFFECTIVE OPERATOR  D̂(θ̂)   [OURS — C1]
   │   per-band depthwise PSF conv + strided pooling; site-bootstrap ν_θ
   │
   ├─ SOFT DATA-CONSISTENCY PROJECTION
   │   k≈5–10 CG steps, Morozov stop at discrepancy τ  → X̂ ∈ 𝒞(Y)
   │
   ├─ SUPPORT + RISK   [OURS — C1]
   │   s(p) support estimate · residual · σ_ep · σ_al · ν_θ → r(p)
   │
   ├─ DECISION CERTIFICATION   [OURS — C2]
   │   Track A: CRC on monotone loss → λ̂
   │   Track B: conformal p-values + BH/BY → selected set
   │
   ├─ TRANSFER + ABSTENTION   [OURS — C3]
   │   shift score → weighted conformal or estimated inflation; abstain
   │
   └─ STITCH + GEOREFERENCE → COG stack + STAC provenance
```

**Shapes:** LR tile `10×128×128` → HR `10×512×512` at 2.5 m. Backbone unchanged; our added parameters < 0.5 M. **The contribution is small in parameters and large in consequence — that is intentional.**

**Band policy (defensible, stated unprompted):** 10 m → 2.5 m (×4). 20 m → 2.5 m (×8), guided by 10 m structure, flagged with systematically lower support. 60 m atmospheric bands **never emitted as super-resolved surface products** — conditioning only.

**Tiles vs scenes, to avoid any ambiguity:** tiling at 128² with 32 px overlap is an *inference and memory* strategy. It has nothing to do with calibration units. Certification always aggregates to the scene level (§9.8).

---

# 12. Measurement / Inference / Provenance Framework

Every output pixel carries three orthogonal descriptors, and the distinction between them is the product:

| Layer | Question answered | Source |
|---|---|---|
| `s(p)` **support estimate** | "How strongly did the observation constrain this, under our fitted operator?" | Effective operator + noise + operator-fit uncertainty |
| `r(p)` **risk score** | "How likely is this materially wrong?" | Learned, then conformally calibrated |
| `a(p)` **abstention/reason** | "Should this be used at all?" | Domain shift vs calibration set |

A pixel can be **weakly constrained but low-risk** (smooth field interior — the prior interpolates safely) or **weakly constrained and high-risk** (a fabricated structure edge). Collapsing these into one "confidence" number is precisely the error v2 made and the field generally makes.

---

# 13. Reliability and Uncertainty Architecture

**What is predicted, stated precisely:**

> `r(p)` is a score. Its meaning comes entirely from the certification step. At the product level we certify a **monotone normalized false-discovery loss** (Track A) and the **FDR of the selected detection set** (Track B). `r` is **not** a hallucination probability, because hallucination is not identifiable without a reference.

**Inputs, each justified:**

| Input | Justification | Kept? |
|---|---|---|
| `1 − s(p)` support deficit | Operator-anchored; the only input speaking to observation constraint | ✅ C1 |
| `\|D̂X̂ − Y\|` residual | Detects operator/registration failure; nearly free | ✅ |
| `σ_ep` (K posterior samples) | Epistemic; LDSR-S2 already supports this | ✅ |
| `σ_al` (heteroscedastic head, Gaussian NLL) | Aleatoric; separates irreducible noise | ✅ |
| `ν_θ` operator-fit uncertainty | Almost universally ignored; cheap via site bootstrap | ✅ |
| Local structure statistics | Edges are where hallucination concentrates | ✅ |
| Low-resolution evidence at the candidate location | Does the measurement show *anything* here? Strongest single feature in our simulator | ✅ |
| **TTA variance** | Measures equivariance, not correctness | ❌ baseline only |
| **Deep ensembles** | Best epistemic estimate, N× training cost | ❌ roadmap |
| **Full Bayesian / VI** | Not justifiable in budget | ❌ rejected |
| **SURE self-calibration** | Cannot bound risk in the unobserved subspace of a rank-deficient operator | ❌ rejected, with reason |

**Calibration procedure:** train `g_φ` on HR-referenced scenes from TRAIN **sites** → calibrate on a **site-disjoint** CALIBRATE set → evaluate on a **third, site-disjoint** TEST set. Site-level three-way separation is what makes the result mean anything (§16).

---

# 14. OOD and Abstention

**Shift score:** Mahalanobis distance in **measurement-domain** feature space to the calibration distribution, plus a domain-classifier probability.

*Why measurement-domain and not backbone-feature space:* a descriptor computed on the reconstruction drifts with the backbone and can be gamed by a model that hallucinates familiar-looking texture. The measurement is the only thing actually observed. Features: per-band reflectance moments and percentiles, gradient statistics, edge density, and radial power-spectrum slope.

**Threshold calibration, done correctly:** Mahalanobis scores computed on the same scenes that estimated the mean and covariance are optimistically small. Thresholds are therefore set on a **held-out portion of the calibration sites**, never in-sample. In our simulator this reduced the false-alarm rate from 42% to 8% at the same nominal operating point.

**Three-tier release policy:**

| Tier | Condition | Behaviour | Claim status |
|---|---|---|---|
| **CERTIFIED** | in-domain, `α + Δ̂ ≤ α_max` | Full detail released; certificate attached | **Guarantee** under §9.6 assumptions |
| **DEGRADED** | moderate shift, support overlap holds | Detail released at inflated level `α + Δ̂`, stated on the product | **Estimate**, labelled as such |
| **ABSTAIN** | shift beyond calibration support, or inflated level exceeds tolerance | Fine detail withheld; product falls back to measurement-supported content; reason code emitted | **No claim** |

**Abstention is scientifically defensible** because it is triggered by a measurable quantity tied to a stated assumption whose violation has a *measured* consequence (§19.2). It is not a heuristic "looks weird" flag.

---

# 15. Spectral and Physical Constraints

- **Reflectance box constraint** `X̂ ∈ [0,1]`, enforced by projection.
- **Aggregation consistency** — enforced softly through `𝒞(Y)`, not as a separate hand-written rule.
- **Spectral shape** — SAM in the loss; per-band reflectance MAE reported.
- **Explicitly rejected: NDVI-consistency as a training constraint.** NDVI is nonlinear, so `NDVI(mean(X)) ≠ mean(NDVI(X))`; constraining LR-NDVI to the spatial mean of HR-NDVI is mathematically wrong and injects bias. Index agreement is an **evaluation metric**, never a loss. *(Raise this unprompted.)*
- **Explicitly rejected: adversarial loss.** It rewards plausible invention, which is the failure mode the project exists to bound.

**Training objective:**

```
L = λ_rec‖X̂−X_HR‖₁ + λ_spec·SAM + λ_grad‖∇X̂−∇X_HR‖₁
    + λ_dc‖D̂X̂−Y‖²_{Σ⁻¹} + λ_unc·GaussianNLL(σ̂) + λ_reg·TV(prior-dependent part only)
```

TV restricted to the prior-dependent component so it never smooths measurement-supported content.

---

# 16. Data and Datasets — site-disjoint by construction

| Dataset | Reference | Role | Notes |
|---|---|---|---|
| **SEN2VENµS** | VENµS 5 m, **same-day** | **Operator fitting** + training | **29 named sites.** Splits are performed at **SITE level**, never at patch level. S2 Etalab OL2.0; **VENµS CC BY-NC 4.0** — non-commercial, must be stated. Same-day pairing removes phenology/illumination confounds. ×2 only. |
| **SEN2NAIP v2** | NAIP aerial sub-metre | Training (×4) | 2,851 real cross-sensor pairs + 17,657 synthetic; **US only**; aerial BRDF/radiometry differs from satellite — a real limitation to state. |
| **WorldStrat** | SPOT 1.5 m | Validation; geographic diversity | ~10,000 km², globally stratified; split by AOI. |
| **`opensr-test` suite** | NAIP / SPOT / Venµs / Spain Crops / Spain Urban | **Conformal calibration + final test** | 178 scenes. Calibration scenes and test scenes are **disjoint at scene and site level** — enforced by an explicit assertion in code, not by convention. |
| **Indian AOI** (peri-urban Hyderabad; Punjab croplands) | **none** | **Out-of-domain transfer demonstration** | No HR reference exists. Deliverable: shift score, tier assignment, estimated inflation, abstention rate. **Explicitly not a certified deployment.** |
| **SEN2NEON** | — | **Excluded** | Could not verify as a released standard S2-paired benchmark. Do not name-drop it. |

### The split rule — stated as an enforceable invariant

```
sites(TRAIN)  ∩  sites(CALIBRATE)  =  ∅
sites(TRAIN)  ∩  sites(TEST)       =  ∅
sites(CALIBRATE) ∩ sites(TEST)     =  ∅
```

Concretely: `TRAIN` = SEN2NAIP (US) + a **site-disjoint** SEN2VENµS subset · `CALIBRATE` = `opensr-test` SPOT + Spain Crops scenes · `TEST` = `opensr-test` Spain Urban + **Venµs sites not used in TRAIN** + WorldStrat holdout AOIs · `TRANSFER DEMO` = Indian AOI.

**Critical, because it is the easiest place to accidentally cheat:** SEN2VENµS is used for operator fitting *and* appears in `opensr-test`'s Venµs subset. Any Venµs site used for operator fitting or backbone training is **removed from TEST**. The split is materialized as an explicit site-ID manifest checked into the repository, and a unit test fails the build if the three sets intersect.

**We do not claim geographic generalization from a split that shares sites.** If site-level disjointness cannot be established for a dataset, that dataset is used for training only, never for the headline certification result.

**Train/validate separation:** synthetic-degradation data is used **only** for training. Real cross-sensor pairs are used **only** for calibration and validation. No model is ever validated against its own degradation assumption.

---

# 17. Training Strategy

| Stage | Trains | Data | Cost |
|---|---|---|---|
| 0 · Operator fit | `θ̂_b`, `δ̂_b` + site-bootstrap `ν_θ` | SEN2VENµS same-day pairs, TRAIN sites | Hours, light |
| 1 · Backbone adaptation | LoRA on SEN2SR | SEN2NAIP + SEN2VENµS (TRAIN sites) | 4–8 h, 1 GPU |
| 2 · Aleatoric head | `σ̂` | same | 1–2 h |
| 3 · Risk model `g_φ` | monotone GBM / small MLP | TRAIN sites, HR-referenced | Minutes |
| 4 · Certification | scalar `λ̂` (Track A); null score set `R⁰` (Track B) | **CALIBRATE sites** | **Seconds** |
| 5 · Shift model | domain classifier + Mahalanobis stats, thresholds on held-out calibration sites | TRAIN vs held-out features | Minutes |

**Note the shape of this:** the scientific contribution is calibrated in stages 3–5, which cost minutes. The expensive stages are the ones we *did not invent*. That is deliberate scoping, and it is the correct answer to "can you build this in a week?"

---

# 18. Calibration Strategy

1. Compute `r` for every candidate on every CALIBRATE **scene** (HR used only as ground truth for labels, never as model input).
2. Fix the decision task and proposal threshold `c₀` from TRAIN (§21.1). Never re-tune per α.
3. **Track A:** scan λ; select `λ̂` satisfying `(n·L̂_n(λ) + B)/(n+1) ≤ α` with `n` = number of independent calibration **units**.
4. **Track B:** collect null scores `R⁰` from calibration candidates labelled false; store as the conformal calibration set.
5. Estimate shift between CALIBRATE and TEST/deployment features; compute weights (if support overlaps) or `Δ̂`.
6. Emit the certificate.

**Certificate fields — the certificate is a product artifact written into STAC metadata, not a slide claim:**

```
{ task, track, alpha, lambda_hat | n_null_scores, procedure: BH|BY,
  n_calibration_units, calibration_unit_type: "scene",
  calibration_sites: [...], shift_score, delta_hat, effective_level,
  tier: CERTIFIED|DEGRADED|ABSTAIN,
  certified_quantity: "E[normalized false-discovery count]" | "FDR of selected set",
  uncertified_diagnostics: { empirical_FDP, empirical_recall },
  operator_version, theta_hat, theta_CI, model_version, timestamp }
```

Note `certified_quantity` is a **required, explicit field**. A consumer never has to infer which guarantee they hold.

---

# 19. Validation Framework

Designed to survive hostile review.

**A · Spatial quality** — PSNR, SSIM, MAE/RMSE vs real HR. *Reported, explicitly not headline* (Sen4x: poor correlation with utility).
**B · Spectral fidelity** — SAM, per-band reflectance MAE, ERGAS, band-correlation preservation.
**C · Radiometric consistency** — `opensr-test` reflectance + spectral consistency metrics.
**D · Measurement consistency** — `‖D̂X̂ − Y‖_{Σ⁻¹}` vs the χ² tolerance `τ`; report achieved residual and fraction of tiles inside `𝒞(Y)`. **This is the physics slide.**
**E · Reconstruction error / hallucination** — `opensr-test` `ha_metric`, `om_metric`, `im_metric`.
**F · Uncertainty calibration** — AUSE / sparsification; reliability diagram; ECE. Baselines: TTA variance, LDSR-S2's native uncertainty, `σ_ep` alone.
**G · Decision certification — the core result.** For **both tracks**, realized loss vs nominal α on site-disjoint TEST scenes, α ∈ {0.05, 0.1, 0.2}. **Realized ≤ nominal is the pass condition.** BH and BY reported side by side. Empirical FDP in a column marked *uncertified diagnostic*.
**H · OOD behaviour** — abstention rate vs shift; does abstention fire *before* the certificate breaks? Uses §19.2's breakpoint.
**I · Downstream utility** — frozen task head on: bilinear · Real-ESRGAN · SEN2SR · LDSR-S2 · ours. Sen4x's published band (≈38.7–48.9 SISR, 51.6 hybrid, 66.3 HR ceiling mIoU) sets the honest expectation — **we do not promise to exceed it**.
**J · Geospatial correctness** — CRS/geotransform round-trip, corner coordinates, 4×4 grid alignment (no half-pixel drift), seam artifact measurement.
**K · Operator sensitivity** — A4b: re-certify with `θ̂` at its bootstrap CI edges; report movement in `λ̂`, retention and realized risk.

## 19.1 Machinery verification on a controlled simulator — *not a performance claim*

**Read this subsection's scope before quoting any number from it.** These results come from a controlled simulator with a known forward operator, a mock reconstruction that fabricates structure preferentially where the measurement is ambiguous, and exact labels. They verify that **the certification machinery is implemented correctly and behaves as the theory predicts**. They say **nothing** about how good any real super-resolution model is, and no number here transfers to Sentinel-2. The real numbers come from §19's protocol on `opensr-test` data.

Base rate of fabricated candidates before certification: **86.4%**.

| α | Track B selected | Realized FDP *(estimator of the certified `E[FDP]`)* | Recall of real structures | Bound respected |
|---|---|---|---|---|
| 0.05 | 859 | **0.022** | 0.637 | ✅ |
| 0.10 | 1004 | **0.043** | 0.729 | ✅ |
| 0.20 | 1246 | **0.139** | 0.813 | ✅ |

Track A held on **24 / 24** feasible α settings.

**Dependence sweep (assumption B2/B3).** Injecting scene-level clustering into the null scores:

| Scene clustering | BH realized FDR (α=0.10) | BY realized FDR | BH recall |
|---|---|---|---|
| none (exchangeable) | 0.067 | 0.008 | 0.665 |
| sd 0.05 | 0.072 | 0.006 | 0.634 |
| sd 0.10 | 0.070 | 0.005 | 0.492 |
| sd 0.20 | 0.073 | 0.008 | 0.184 |
| sd 0.35 | 0.055 | 0.000 | 0.056 |

**Finding, and it is the useful kind:** within-scene clustering costs **power**, not **validity** — realized FDR stayed at or below α throughout while recall collapsed by an order of magnitude. Clustering is therefore an efficiency problem, handled by honest unit definition (§9.8), not a validity emergency.

**Calibration-unit sweep.** At α = 0.10: `n = 5` → **no feasible threshold**; `n = 10` → 0.099 retention; `n = 20` → 0.175; `n = 30` → 0.189; `n = 40` → 0.199. The `1/(n+1)` penalty consumes 17%, 9%, 4.8%, 3.2% and 2.4% of the risk budget respectively.

**Feature ablation, including a negative result we report rather than bury.** Recall at fixed certified error rate: full model 0.729; without the support term 0.716; without novelty energy 0.729; without the consistency residual 0.728; without low-resolution evidence 0.306; **operator-anchored features alone 0.000**; no risk signal at all 0.000.

**Interpretation, stated plainly:** in this simulator the operator-anchored features do **not** separate real from fabricated structures on their own. Low-resolution measurement evidence carries the discriminative load. This is consistent with §9.5's conservative reading of `s`: support says where the prior was unconstrained, not whether a given structure is fake. **C1's claim is scoped accordingly, and the real-data ablation (A6/A7) is the experiment that decides whether C1 survives.** If it does not, C2 and C3 stand on a variance-plus-evidence score, and we say so.

## 19.2 Shift sensitivity — how fast the guarantee breaks

This is the experiment that justifies C3's existence. Holding everything else fixed, we shift the **test null-score distribution** relative to calibration — the precise violation of assumption B1 — and measure realized FDR at a promised α = 0.10.

| Null-score shift | BH realized FDR | BY realized FDR | Verdict |
|---|---|---|---|
| 0.00 | 0.066 | 0.004 | holds |
| 0.05 | **0.127** | 0.022 | **BREACHED** |
| 0.10 | **0.209** | 0.049 | **BREACHED** |
| 0.20 | **0.413** | 0.198 | **BREACHED** |
| 0.30 | **0.592** | 0.439 | **BREACHED** |
| 0.45 | **0.722** | 0.685 | **BREACHED** |

**The headline finding, and we lead with it rather than hiding it:** a null-score shift of **0.05 is enough to break a 10% FDR promise**, delivering 12.7%. At shift 0.20 the realized FDR is **four times** the promise. The Benjamini–Yekutieli variant is materially more robust — it absorbs shift up to roughly 0.10 before breaching — but it is not immune either.

**Why this strengthens rather than weakens the submission:** it converts C3 from a nice-to-have into a *load-bearing* component with a measured justification. Anyone who applies conformal certification to Earth observation without shift handling is shipping a number that is wrong by a factor of four over unfamiliar terrain, and now we can show by how much. It also sets the abstention threshold on evidence: the shift score at which abstention fires is chosen to sit **below** the measured breakpoint, not at an arbitrary quantile.

### Hypothesis confirmation / falsification

**Confirmed if:** on site-disjoint TEST, realized loss ≤ nominal α for both tracks across α ∈ {0.05, 0.1, 0.2}; **and** the risk score with `s`/`ν_θ` beats variance-only and TTA on AUSE and on recall-at-fixed-α; **and** under induced shift the naive certificate's realized risk exceeds α while the abstaining version does not.

**Falsified if:** realized loss exceeds nominal α on site-disjoint held-out data; **or** `s` and `ν_θ` add nothing over generative variance (C1 dead — C2/C3 survive on a variance-only score); **or** abstention fires essentially everywhere or nowhere.

**We will report either outcome.** A falsified sub-hypothesis with clean evidence beats an unfalsifiable demo. §19.1 already contains one negative result we did not have to disclose.

---

# 20. Ablation Plan

| ID | Configuration | Isolates |
|---|---|---|
| A0 | Bicubic | Floor |
| A1 | Real-ESRGAN | v1's original core — expect sharp, spectrally poor |
| A2 | SEN2SR unmodified | Backbone alone |
| A3 | LDSR-S2 + its native uncertainty | **The trust baseline we must beat** |
| A4 | A2 + fitted `D̂` vs hand-set Gaussian | Does operator fitting matter? (C1a) |
| **A4b** | **A4 with `θ̂` perturbed to bootstrap CI edges** | **Operator sensitivity — how fragile is the certificate?** |
| A5 | A4 + soft data-consistency projection | Does `𝒞(Y)` membership help? |
| A6 | Risk score without `s`, `ν_θ` (variance only) | **C1's value** |
| A7 | Full risk score | C1 |
| A8 | Pixel-level conformal only | Baseline certification |
| **A9a** | **Track A — CRC on the monotone loss** | **C2, guarantee 1** |
| **A9b** | **Track B — conformal p-values + BH** | **C2, guarantee 2** |
| **A9c** | **Track B with BY correction** | **Validity under arbitrary dependence** |
| A10 | Naive certification under induced geographic shift | Shows the failure C3 fixes (§19.2) |
| A11 | Shift-corrected + abstention | **C3's value** |
| A12 | Full CERTUS-S2 | Everything |

**Discipline:** any component whose ablation shows no gain is removed from the final architecture and reported as a negative result. **Note that validity is guaranteed for every variant by construction — so ablations are scored on recall at a fixed certified error rate, never on validity.** Comparing validity across ablations would prove nothing.

---

# 21. Downstream Applications

## 21.1 The certified decision task — defined precisely

Certification attaches to a concrete decision, never to "any change." The MVP task is fixed as follows and is the **only** task certified in the MVP.

### MVP task D1 — **binary built-structure presence verification**

*Chosen as MVP over bi-temporal change because it is the task for which the available HR references can provide defensible ground truth.* `opensr-test`, SEN2NAIP and WorldStrat supply **single-date** S2/HR pairs; bi-temporal HR reference pairs over the same site are rare and would reduce `n` below the point where any certificate is useful (§9.8). Scoping to the task the data actually supports is a deliberate choice, not an omission.

| Element | Definition |
|---|---|
| **Question certified** | "Of the compact built structures visible in this reconstruction, which are really present on the ground?" |
| **Spatial unit** | A **connected candidate region** (object), not a pixel. Area in `[16, 400]` HR pixels — at 2.5 m, ≈100 m² to 2500 m². Regions outside this band are not candidates. |
| **Candidate generation (λ-independent)** | High-pass response of the reconstruction thresholded at a **fixed** `c₀`, then connected-component labelling. `c₀` is selected once on TRAIN and frozen; it is never re-tuned per α, per track, per scene, or after seeing calibration results. |
| **Positive (true) detection** | A candidate whose centroid lies within `d = 2` HR pixels of a structure in the reference mask, obtained by applying **the same detector with the same `c₀`** to the co-registered HR reference. Using an identical detector on both sides is what keeps the label definition from smuggling in a second, un-audited threshold. |
| **False detection** | Any candidate that is not a positive detection. |
| **What λ controls** | Retention of candidates by risk score: retain iff `r_k ≤ λ`. Nothing else in the pipeline depends on λ. |
| **Certified loss — Track A** | `L_i(λ) = FD_i(λ) / max(1, N_i)`, `N_i = |𝒦_i|` fixed. Guarantee: `E[L] ≤ α`. |
| **Certified quantity — Track B** | FDR of the BH-selected subset ≤ α (BY variant under arbitrary dependence). |
| **Diagnostics** | `FDP = #false / max(1, #selected)` — *uncertified* under Track A, the finite-sample estimator of the certified `E[FDP]` under Track B (§9.6) — and recall of real structures (always uncertified). |
| **Evaluation unit for the guarantee** | The **scene**. Per-scene FDP may exceed α on individual scenes without violating a guarantee that is an expectation over scenes — this is stated wherever per-scene numbers are shown. |

### Extension task D2 — bi-temporal built-change detection *(not MVP)*

Two dates → reconstruction + risk for each → radiometric normalization on invariant pixels → candidate change objects → identical certification machinery. Requires bi-temporal HR references at the same site. **Built only if such references are secured for at least 15 independent sites**; otherwise it remains a documented extension with the machinery demonstrated on D1. We do not claim certified change detection on the strength of a single-date experiment.

### Secondary — certified index products *(strongly recommended, not MVP)*

NDVI/NDWI at 2.5 m with prediction intervals propagated from `σ` by first-order error propagation, coverage validated conformally. Ties to the PS's crop-monitoring application.

**Roadmap (not built, greyed out):** building/road extraction, field-boundary delineation, land-cover mapping, multi-sensor fusion.

---

# 22. Geospatial Data Pipeline

- CRS + geotransform read and propagated; output exactly 2.5 m, grid-aligned so each 10 m pixel maps to exactly 4×4 HR pixels (**no half-pixel drift** — asserted in tests).
- Tiling 128² LR with 32 px overlap; cosine-window blending; seam artifacts *measured*, not assumed absent. **Tiling is an inference strategy only and never defines a calibration unit** (§9.8).
- SCL-derived validity and abstention masks propagated as bands, never silently filled.
- Outputs as **Cloud-Optimized GeoTIFF** with overviews; **STAC** sidecar carrying full provenance including the certificate block of §18.
- Scale-out: a full 110×110 km tile at 2.5 m is ~44,000² px/band — windowed chunked processing, never whole-scene RAM loads.

# 23. Product Outputs

| # | Layer | Reason it exists |
|---|---|---|
| 1 | SR reflectance, 10 bands @ 2.5 m | The PS deliverable |
| 2 | RGB + NIR renders | Human inspection |
| 3 | **Support estimate `s`** | "How strongly was this constrained?" — C1 |
| 4 | **Risk score `r`** | Input to certification — C2 |
| 5 | **Certificate** (incl. explicit `certified_quantity` field) | Makes the claim auditable — C2 |
| 6 | **Abstention mask + reason codes** | "Should this be used?" — C3 |
| 7 | Measurement residual map | Physics evidence; detects registration failure |
| 8 | Validity mask (SCL-derived) | Cloud/shadow honesty |
| 9 | Certified indices with intervals | Uncertainty reaches the analytic product |
| 10 | Certified detection layer (task D1) | The decision product |
| 11 | Provenance: S2 product ID, `θ̂` + CIs, model + operator version, calibration site list, `n` units, timestamp | Auditability |

Every layer answers a question a user actually asks. Nothing is included to look enterprise-grade.

---

# 24. System Architecture

| Layer | Choice | Justification |
|---|---|---|
| Geospatial I/O | `rasterio` / GDAL | Windowed R/W, CRS handling |
| SR backbone | `sen2sr`, `opensr-model` (pip) | Pretrained, released, ESA-maintained |
| Geo wrapper | `opensr-utils` | Tiling/blending/metadata |
| Evaluation | `opensr-test` (pip) | Third-party standardized |
| Conformal | `conformal-risk` (Angelopoulos) + our BH/BY selection module | Reference implementation for Track A; Track B is ~60 lines and is ours to write |
| DL | PyTorch | Backbone ecosystem |
| Risk model | scikit-learn / LightGBM (monotone constraints) | Small, interpretable, fast |
| Data access | `cdsetool` / `cdse-client` → CDSE | `sentinelsat` targets the retired Open Access Hub |
| UI | **Streamlit** | Correct at prototype scale |
| Packaging | Docker on `osgeo/gdal` | Reproducibility; protects the live demo |
| Tracking | W&B or CSV + git tags | Ablations are the deliverable — they must be logged |

# 25. Compute Requirements

**Prototype:** single 16 GB GPU (T4/P100/Colab-class), fp16, LoRA-only. Backbone unchanged; +<0.5 M parameters. Data footprint <50 GB. Inference ≈ seconds per 128→512 tile with K=8 samples + CG steps; a demo AOI is interactive, a full S2 tile is a batch job — **say so rather than implying real-time**. The certification layer itself is milliseconds and runs on CPU.

**Production:** queue-driven tile-parallel workers; COG to object storage; STAC catalogue; deep ensembles replacing MC-dropout; diffusion posterior sampling; per-region recalibration with automated OOD gating; operator refitting per satellite unit (S2A/S2B/S2C are distinct instruments; S2C took primary duty in January 2025).

---

# 26. Implementation Roadmap

### Build tiers — MVP is a short list, deliberately

| Tier | Items |
|---|---|
| **MVP — MUST BUILD (9 items)** | 1. L2A ingest + SCL. 2. SEN2SR reconstruction through `opensr-test`. 3. Fitted effective operator `D̂` + site bootstrap. 4. Measurement residual + support map `s`. 5. Model uncertainty (`σ_ep`, `σ_al`). 6. Risk score `g_φ`. 7. **Conformal certification — Track A and Track B** on task D1. 8. Certificate + abstention. 9. Streamlit demo. |
| **EXTENSIONS / BASELINES — build if time, never cut from the narrative** | LDSR-S2 as a second backbone (also baseline A3); elaborate aleatoric head; additional generative backbones; extensive multi-model comparison; weighted conformal (vs estimated-inflation reporting only); soft-projection ablation A5; certified NDVI intervals; bi-temporal change task D2. |
| **FUTURE / PRODUCTION** | Multi-temporal fusion; deep ensembles; diffusion posterior; L1B detector-overlap self-supervision; multi-sensor fusion. |

**These are marked as extensions, not deleted.** They remain in the architecture diagram and the comparison table as planned capability, clearly labelled. Presenting an honest MVP boundary is stronger than presenting nine days of work as seven.

**Cut rule applied:** multi-temporal fusion is cut from the MVP because published evidence gives it a modest, uncertain benefit (Sen4x: pure MISR *worse* than SISR by 12.9 mIoU) at high implementation risk. Neural fields are cut: a representation choice that adds no information and complicates the observation operator.

### 7-day plan

- **Day 1** — Docker; `pip install sen2sr opensr-test opensr-utils conformal-risk`; **one scene end-to-end through `opensr-test` with printed metrics**. Baselines A0/A2. Materialize the **site-ID split manifest** and its failing unit test *before any modelling*. *If Day 1 does not end with a third-party metric on screen and a passing split assertion, fix that before anything else.*
- **Day 2** — SEN2VENµS TRAIN sites; co-registration; operator fit with joint sub-pixel shift + site bootstrap; report `σ_b` ± CI. Support map `s`.
- **Day 3** — Soft data-consistency projection; residual metric; risk features assembled; `g_φ` trained on TRAIN sites. Freeze `c₀` for task D1.
- **Day 4** — **Certification day.** Track A CRC calibration; Track B null-score set and BH/BY selection; realized-vs-nominal curves on site-disjoint TEST for both tracks. *This is where the contribution lives — protect it.*
- **Day 5** — Shift score, abstention tiers, induced-shift experiment (A10/A11), operator sensitivity (A4b); Indian AOI out-of-domain transfer demonstration.
- **Day 6** — Streamlit, COG/STAC output with the certificate block, geospatial assertions; complete ablation table.
- **Day 7** — Buffer, rehearsal, judge-attack drill, claim-audit and "what we do not claim" slides.

**5-day fallback:** drop Day 5 (C3 becomes roadmap with §19.2's sensitivity result shown as the justification) and Day 6's index products. **Never drop Day 4.**

---

# 27. Risks and Failure Modes

| # | Risk | Mitigation |
|---|---|---|
| R1 | Install/CUDA friction eats a day | Docker Day 1; fall back to SEN2SRLite or Swin2SR — the wrapper is backbone-agnostic by design |
| R2 | Realized risk exceeds nominal | Almost always non-monotone loss, site leakage, or shift. Track A's loss is monotone by construction; the split manifest test catches leakage; §19.2 characterizes shift |
| R3 | `s` and `ν_θ` add nothing over generative variance | C1 dies; C2/C3 survive on a variance-plus-evidence score. §19.1 already shows this is a live possibility — we report it either way |
| R4 | **Too few independent calibration units → loose or infeasible bound** | **Report `n` and the resulting bound honestly. The certificate becomes more conservative and possibly infeasible; that is the correct behaviour.** The remedy is **more geographic diversity in the reference archive** — additional independent sites. **We do not subdivide scenes into tiles to inflate `n`** (§9.8). If no feasible threshold exists at the requested α, the system abstains and says why |
| R5 | Abstention fires everywhere on the Indian AOI | That is a *finding*, not a failure — the honest statement that no calibration data exists for that terrain, which is itself the operational argument |
| R6 | Exact-consistency claim attacked | Removed in §9.3. We claim membership in `𝒞(Y)` at a stated tolerance and report the residual |
| R7 | "You just used SEN2SR" | §31 ledger + the ablation table. A6/A7/A9a/A9b/A11 show exactly what each layer adds |
| R8 | Operator fit degenerate or absorbing registration error | Joint sub-pixel shift fit, kernel regularization, inspect kernels, report bootstrap CIs, and A4b sensitivity |
| R9 | **BH's PRDS assumption questioned** | Report the **BY variant** in every table; it is valid under arbitrary dependence. The headline can retreat to BY without losing the contribution |

---

# 28. What We Explicitly Do NOT Claim

1. We do not claim to beat SEN2SR, LDSR-S2 or DiffFuSR on PSNR/SSIM or sharpness. **We do not claim CERTUS-S2 produces better imagery at all** — the reconstruction is the backbone's.
2. We do not claim to have invented hallucination-aware super-resolution.
3. We do not claim to have invented conformal prediction, conformal risk control, conformal p-values, or Benjamini–Hochberg.
4. **We do not claim standard conformal risk control gives FDP or FDR control.** It does not; FDP is non-monotone. Track A certifies a monotone normalized false-discovery loss; FDR comes separately from Track B.
5. **We do not claim that a realized empirical FDP measured on a finite test set is itself a guarantee.** Under Track A it is an uncertified diagnostic that the Track A bound does not imply. Under Track B it is the finite-sample estimator of the certified quantity `FDR = E[FDP]` — evidence consistent with the bound, not a proof of it, and the bound is on the expectation, so individual scenes may exceed α.
6. We do not claim our fitted PSF is Sentinel-2's true physical MTF — it is an **effective observation operator**, reported with error bars, absorbing registration, radiometric and atmospheric differences.
7. We do not claim exact data consistency — we claim membership in `𝒞(Y)` at a stated tolerance, and we report the residual.
8. We do not claim to output a hallucination probability. **We do not claim `s = 0` means hallucinated or `s = 1` means correctly measured.**
9. We do not claim guarantees hold under arbitrary distribution shift. §19.2 shows a 0.05 null-shift already breaches a 10% promise — which is precisely why the system abstains.
10. **We do not claim a certified deployment over India.** The Indian AOI is an out-of-domain transfer *demonstration*.
11. **We do not claim geographic generalization from any split that shares sites.** Splits are site-disjoint and machine-checked.
12. We do not claim SR closes the gap to real high-resolution imagery. Published evidence puts the best SR at 51.6 mIoU against a 66.3 HR ceiling.
13. We do not claim multi-temporal fusion helps; we cite evidence that naive MISR **hurts**.
14. We do not claim real-time full-scene processing.

**This slide wins more credibility than any performance number**, because almost no team presents one.

---

# 29. Formal Claim Audit

Every substantive claim, its type, its assumptions, the evidence required, and its status. **Statuses:** `SAFE` (holds as stated, given assumptions), `NEEDS EXPERIMENT` (design is sound, evidence pending), `ASSUMPTION-DEPENDENT` (valid only while a stated assumption holds; must always be quoted with it), `DO NOT CLAIM` (unsupported — appears nowhere in our materials).

| # | Claim | Type | Assumptions | Evidence required | Status |
|---|---|---|---|---|---|
| 1 | Track A gives a finite-sample bound `E[FD/N] ≤ α` | Mathematical | Exchangeable calibration **scenes**; loss monotone + bounded (both hold by construction) | Realized ≤ nominal on site-disjoint TEST | **SAFE** |
| 2 | Track B gives finite-sample **FDR** control over the selected set | Mathematical | Exchangeability of **null** units (B1); PRDS (B2); candidate-level null exchangeability (B3) | Realized FDR ≤ α on site-disjoint TEST; BY reported alongside | **ASSUMPTION-DEPENDENT** — quote with B1–B3 |
| 3 | Track B with **BY** controls FDR under arbitrary dependence | Mathematical | B1 only (exchangeability of nulls) | Same, BY column | **SAFE** given B1 |
| 4 | Standard CRC controls **FDP/FDR** directly | Mathematical | — | — | **DO NOT CLAIM** (FDP non-monotone; v3's error) |
| 5 | A realized empirical FDP is itself a guarantee | Empirical | — | — | **DO NOT CLAIM** — under Track A it is an uncertified diagnostic; under Track B it estimates the certified `E[FDP]`, and the bound is on the expectation, not on any single scene |
| 6 | `s` estimates how strongly the observation constrains local structure | Physical/statistical | Assumed effective operator + noise model; Fourier approximation to a shift-varying operator | Operator fit + CIs; A4b sensitivity | **ASSUMPTION-DEPENDENT** — quote with the operator model |
| 7 | `s = 0` indicates hallucination / `s = 1` indicates correctness | Interpretive | — | — | **DO NOT CLAIM** (§9.5) |
| 8 | Above-Nyquist support is *identifiable* | Mathematical | — | — | **DO NOT CLAIM** — aliasing folds frequencies; `s` there is *potential* support |
| 9 | `s`/`ν_θ` improve the risk score over variance-only | Empirical | — | Ablation A6 vs A7 on real data | **NEEDS EXPERIMENT** — simulator (§19.1) shows operator terms alone are insufficient |
| 10 | `D̂` is Sentinel-2's physical MTF | Physical | — | — | **DO NOT CLAIM** (§10) |
| 11 | `D̂` is an *effective observation operator* fitted with reported CIs | Empirical | Same-day pairing; joint sub-pixel shift; regularization | Fitted `σ_b` ± bootstrap CI over sites; A4b | **NEEDS EXPERIMENT** (design SAFE) |
| 12 | Geographic generalization | Empirical | Site-disjoint splits, machine-checked | Realized ≤ nominal on site-disjoint TEST | **NEEDS EXPERIMENT** — and **DO NOT CLAIM** if site disjointness fails |
| 13 | Robustness to distribution shift | Empirical | — | §19.2 sweep; A10/A11 | **ASSUMPTION-DEPENDENT** — we claim *diagnosis and abstention*, never robustness |
| 14 | `Δ̂` is a certified bound on the coverage gap | Mathematical | — | — | **DO NOT CLAIM** — estimate only (W2) |
| 15 | Certified guarantee over the Indian AOI | Empirical | — | — | **DO NOT CLAIM** — out-of-domain demonstration |
| 16 | 2.5 m **sampling** output | Engineering | 4× from 10 m | Grid-alignment assertions | **SAFE** |
| 17 | 2.5 m **resolved detail** everywhere | Physical | — | — | **DO NOT CLAIM** — 2.5 m is a sampling choice; certified detail is less and varies spatially |
| 18 | Decision-level certification for a satellite reconstruction product | Novelty | Literature search §4 current at submission | §4 survey; A9a/A9b | **NEEDS EXPERIMENT** for the result; **SAFE** as a scoped novelty claim |
| 19 | Validity is independent of risk-model quality | Mathematical | Same as claims 1–3 | Ablation A0 (constant score → valid but near-zero yield) | **SAFE** |
| 20 | Certified **change** detection (bi-temporal) | Empirical | Bi-temporal HR references at ≥15 independent sites | D2 protocol | **DO NOT CLAIM** in MVP — extension only (§21.1) |
| 21 | Better imagery than SEN2SR/LDSR-S2 | Empirical | — | — | **DO NOT CLAIM** |
| 22 | Spectral consistency preserved | Empirical | Inherited from backbone + `𝒞(Y)` | SAM, per-band MAE, `opensr-test` consistency | **NEEDS EXPERIMENT** |
| 23 | Abstention is principled, not heuristic | Methodological | Shift score tied to a measured breakpoint (§19.2) | H-track abstention-vs-shift curve | **NEEDS EXPERIMENT** (design SAFE) |
| 24 | Real-time full-scene processing | Engineering | — | — | **DO NOT CLAIM** — batch job |

**How this table is used:** any sentence in the deck, the demo or the paper that asserts a row must carry that row's assumptions in the same breath. Rows marked `DO NOT CLAIM` are checked against the slide text before submission.

---

# 30. Comparison with Existing Systems

| Capability | Real-ESRGAN | SEN2SR | LDSR-S2 | Conformal SR (2502.09664) | **CERTUS-S2** |
|---|---|---|---|---|---|
| S2-specific, spectral consistency | ✗ | ✓ | ✓ | ✗ | ✓ (via backbone) |
| 2.5 m all-band product | ✗ | ✓ | RGB-NIR | ✗ | ✓ (via backbone) |
| Fitted **effective** observation operator | ✗ | partial (hard constraint) | ✗ | ✗ | **✓ with error bars** |
| Uncertainty map | ✗ | ✗ | ✓ | ✓ (variance) | ✓ (decomposed) |
| Measurement-support layer | ✗ | ✗ | ✗ | ✗ | **✓** |
| Risk-controlled guarantee | ✗ | ✗ | ✗ | ✓ (image fidelity) | ✓ (decision-level) |
| **Decision-level certification** | ✗ | ✗ | ✗ | ✗ | **✓ (two tracks)** |
| **FDR control over reported detections** | ✗ | ✗ | ✗ | ✗ | **✓ (BH / BY)** |
| Measured shift-sensitivity curve | ✗ | ✗ | ✗ | ✗ | **✓** |
| Abstention | ✗ | ✗ | ✗ | ✗ | **✓** |
| Geospatial certificate/provenance | ✗ | partial | partial | ✗ | **✓** |
| **Sharper imagery than the backbone** | — | — | — | — | **✗ — not claimed** |

The last row is deliberate. A comparison table where our column is all ticks is a table nobody believes.

# 31. Research Contributions — Ownership Ledger

| Component | Status |
|---|---|
| SEN2SR / LDSR-S2 backbone, `opensr-utils` | **EXISTING — ESA OpenSR.** Used as-is. |
| `opensr-test` metrics and reference datasets | **EXISTING — ESA OpenSR.** Our evaluation harness. |
| Conformal prediction, CRC, weighted conformal, coverage-gap bounds | **EXISTING** — Vovk; Angelopoulos et al.; Tibshirani et al.; Barber et al. |
| Conformal p-values, BH-over-conformal-p-values FDR control, BY correction | **EXISTING** — Bates et al.; Jin & Candès; Benjamini–Hochberg; Benjamini–Yekutieli |
| SEN2VENµS / SEN2NAIP / WorldStrat | **EXISTING** datasets |
| Effective operator fitting **with site-bootstrap uncertainty propagated into a support estimate** | **OURS — C1** |
| **Decision-level certification of a satellite reconstruction product: a monotone CRC loss for detections plus conformal-p-value FDR control, with an operator-anchored risk score** | **OURS — C2 (headline)** |
| **Measured shift-sensitivity characterization and abstention for EO reconstruction certification** | **OURS — C3** |
| Certified geospatial product format (support + risk + certificate with explicit `certified_quantity` + abstention + provenance) | **OURS — engineering contribution** |
| Multi-temporal fusion, neural fields, diffusion posterior, deep ensembles, L1B self-supervision | **FUTURE / ROADMAP** — explicitly not built |

---

# 32. Judge Attack Test — and the Answers

**A1 · "Isn't this just SEN2SR?"** SEN2SR is our backbone and we say so on slide 1. It outputs an image with consistent radiometry. It does not tell you how strongly the sensor constrained each region, does not bound the error rate of any decision, and does not know when it is outside its calibration domain. Ablations A2 → A7 → A9a/A9b → A11 quantify exactly what each layer adds.

**A2 · "Isn't this just uncertainty estimation?"** Uncertainty estimation produces a number with no operational meaning. We produce two *guarantees* with stated losses: a conformal risk bound on a monotone normalized false-discovery loss, and FDR control over the reported detection set. LDSR-S2 already ships an uncertainty map — it is baseline A3.

**A3 · "You applied conformal risk control to FDP. FDP isn't monotone."** Correct, and that is exactly why we do not do it. That was an error in our previous revision and we fixed it (§9.1, §9.6). Track A certifies `FD/N` with a **λ-independent denominator**, monotone by construction. FDR comes from a different machine entirely — conformal p-values plus Benjamini–Hochberg. Empirical FDP is reported as a diagnostic and never certified. **Being asked this question is the reason the document is structured the way it is.**

**A4 · "BH needs PRDS. Your detections are spatially correlated."** Conformal p-values from a shared calibration set are PRDS (Bates et al. 2023). Spatial correlation among *test* candidates is additional dependence that theorem does not cover, so we report the **Benjamini–Yekutieli** variant in every table — valid under arbitrary dependence at a `ln M` power cost. Our simulator sweep (§19.1) found clustering degrades power, not validity: realized FDR stayed ≤ α while recall fell from 0.665 to 0.056.

**A5 · "Aren't your calibration tiles correlated?"** Calibration units are **scenes or spatially blocked site groups, never tiles from a shared scene** (§9.8). Tiling exists only for inference memory. When `n` is small the certificate becomes conservative or infeasible and we report that; we never subdivide scenes to inflate `n`.

**A6 · "How do you know prior-dependent detail is wrong?"** We don't, and we never claim it. Prior-dependence is necessary but not sufficient for error. `s` says how strongly the observation constrained something, not whether it is fake — §9.5, and claim 7 in the audit is marked DO NOT CLAIM.

**A7 · "Is your PSF Sentinel-2's actual physical MTF?"** No. It is an **effective observation operator** absorbing optics, detector footprint, L2A resampling, residual registration error, and cross-sensor radiometric and atmospheric differences. Fitted from same-day pairs with a joint sub-pixel shift and a regularized kernel, reported with bootstrap CIs over sites, and stress-tested in A4b.

**A8 · "Isn't your calibration secretly using HR references?"** Yes — deliberately and openly. Calibration consumes references once, where they exist. Deployment consumes none. The scientific risk is entirely in the *transfer*, which C3 measures rather than assumes.

**A9 · "Your train and test both use Venµs sites."** Splits are enforced at **site level** with a checked-in site-ID manifest and a unit test that fails the build on intersection (§16). Any Venµs site used for operator fitting or training is removed from TEST.

**A10 · "What happens over India?"** The shift score rises, the tier drops to DEGRADED or ABSTAIN, and we report shift score, estimated inflation and abstention rate. **It is labelled an out-of-domain transfer demonstration, not a certified deployment.** §19.2 is why: a null-score shift of 0.05 already turns a 10% promise into 12.7%, and 0.20 turns it into 41%. If the system abstains everywhere over our Indian AOI, that is a true and useful statement.

**A11 · "What if your trust model is wrong?"** Validity does not depend on `r` being good — a bad score yields a valid but *useless* certificate (near-zero yield). Ablation A0 demonstrates it: a constant risk score selects nothing at all. **The failure mode of this system is uselessness, not false confidence.**

**A12 · "Why 2.5 m?"** The PS requires <4 m; 4× from 10 m gives 2.5 m, matching SEN2SR/LDSR-S2/DiffFuSR so comparison is apples-to-apples. But 2.5 m is a *sampling* choice. Certified detail is less and varies spatially — audit claim 17.

**A13 · "What exactly did you invent?"** §31's ledger. In one sentence: the first decision-level, operator-anchored, shift-characterized risk certificate for a satellite super-resolution product, with abstention.

**A14 · "What if your novelty already exists?"** We searched (§4). If a paper published before the event does exactly this for Sentinel-2, our fallback contribution is the operator-anchored support estimate and the measured shift-sensitivity analysis — and we would say so rather than restate the claim.

## 32.1 Weaknesses in our own method that a sharp evaluator will find

Presenting these before they are asked is worth more than defending them afterwards.

**W1 · FDP is not monotone, so it cannot be certified by CRC.** *Handling:* Track A certifies `FD/N` with a λ-independent denominator (monotone by construction); Track B obtains genuine FDR via conformal p-values and BH; empirical FDP is a reported diagnostic. **The two tracks are never conflated, and `certified_quantity` is an explicit certificate field so no consumer can mistake one for the other.**

**W2 · The coverage gap `Δ̂` under shift is an estimate, not a certified bound.** The beyond-exchangeability results bound the gap by a distributional distance that is not reliably estimable in high dimensions from a handful of scenes. *Handling:* the guarantee is **exact in-domain, diagnostic out-of-domain**. `Δ̂` is reported as an estimated inflation, and abstention is deliberately conservative. **The honest claim is "certified in-domain, diagnosed and abstaining out-of-domain."**

**W3 · Calibration units are spatially correlated, and pooled null scores inherit that correlation.** Track B's assumption B3 asks for candidate-level exchangeability of null scores; candidates within one scene share illumination, land cover and geometry, so B3 is violated in the strict sense. *Handling:* calibration units are scenes (§9.8); we report the **BY** variant which needs no dependence assumption; we run a **scene-level block bootstrap** on realized FDR as a sensitivity check; and §19.1's clustering sweep shows the empirical cost is power rather than validity. **We do not claim to have solved scene-clustered conformal FDR in general — that is an open problem and we name it as one.**

**W4 · The number of independent calibration units is small, and there is no honest way to make it larger.** Real HR-referenced scenes number in the low hundreds globally, and site-disjointness cuts that further. *Handling:* report `n` on every certificate; accept the looser `B/(n+1)` term; quantify what each additional site buys (§9.8). **The remedy is geographic diversity in the reference archive, not statistical creativity.**

**W5 · The support estimate may not survive its own ablation.** §19.1 found the operator-anchored features insufficient on their own in a controlled simulator. *Handling:* A6 vs A7 on real data decides it. If C1 fails, C2 and C3 stand on a variance-plus-evidence score and we report C1 as a negative result. **The architecture degrades gracefully by design, and we would rather publish that than defend a dead component.**

**Why raising these strengthens the submission:** the property that makes this design robust is that **certification validity does not depend on our risk score being good** — a poor score produces a valid certificate with poor yield, not an invalid one. The failure mode is *uselessness*, not *false confidence*. That is the right failure mode for anything intended for operational use, and it is the sentence to end the technical defence on.

---

# 33. Final Technical Summary and Single Recommendation

**Build CERTUS-S2, exactly as specified above. One architecture, one headline claim.**

> **We do not certify pixels. We certify decisions.**
> A Sentinel-2 reconstruction at 2.5 m, produced by an existing ESA backbone that we do not pretend to have invented or improved, wrapped in three layers we did build: a fitted **effective observation operator** that estimates how strongly the sensor constrained each region; **decision-level certification** giving a finite-sample conformal bound on a monotone false-discovery loss *and* FDR control over the reported detections; and a **shift-characterized transfer layer** that abstains rather than exporting a guarantee it cannot honour.

**Why this and nothing else:**
- **Novelty** — the composition is unclaimed, and the decision-level target is named as future work by the closest adjacent paper.
- **Rigor** — every guarantee is finite-sample and distribution-free *under assumptions we state, test and audit* (§29), and valid even if our risk model is bad.
- **User value** — an analyst gets a stated error rate, an explicit statement of *which* error rate, and an explicit refusal where none applies.
- **Validatability** — every claim maps to a plotted curve on site-disjoint data, with a stated falsification condition and a claim audit.
- **Feasibility** — the expensive components are pretrained; our contributions calibrate in minutes on CPU.
- **SIH competitiveness** — the contribution cannot be reduced to "you used an existing SR model," because the ablation table separates the backbone from every layer we added, and because no other team will bring two separately-valid guarantees, a measured shift-breakpoint, a claim audit, and a "what we do not claim" slide.

**Build first, today, before any model code:** Docker + `pip install sen2sr opensr-test opensr-utils conformal-risk`, then one `opensr-test` scene through one SEN2SR forward pass with metrics printed — and the site-ID split manifest with its failing-on-intersection unit test. That de-risks the largest schedule threat and the largest credibility threat in the same afternoon.

---

## Sources

- [SIH26142 problem statement](https://sih2026-ps-viewer.vercel.app/ps/SIH26142)
- [OpenSR — Trustworthy Super-Resolution (ESA)](https://opensr.eu/) · [SEN2SR](https://github.com/ESAOpenSR/SEN2SR) · [opensr-model / LDSR-S2](https://github.com/ESAOpenSR/opensr-model) · [opensr-test](https://github.com/ESAOpenSR/opensr-test)
- [Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion (LDSR-S2)](https://www.semanticscholar.org/paper/27fa48af71d55c671c498649b5a65d57fbed13f4)
- [A radiometrically and spatially consistent SR framework for Sentinel-2 (SEN2SR preprint)](https://opensr.eu/news/new-preprint-a-radiometrically-and-spatially-consistent-super-resolution-framework-for-sentinel-2/)
- [DiffFuSR](https://arxiv.org/pdf/2506.11764) · [Beyond Pretty Pictures / Sen4x](https://arxiv.org/html/2505.24799v1) · [Alias and Band-Shift for S2 SR](https://ar5iv.labs.arxiv.org/html/2302.11494) · [L1BSR](https://arxiv.org/abs/2304.06871)
- [Image-to-Image Regression with Distribution-Free UQ (RCPS)](https://proceedings.mlr.press/v162/angelopoulos22a.html) · [im2im-uq code](https://github.com/aangelopoulos/im2im-uq)
- [Conformal Risk Control](https://arxiv.org/pdf/2208.02814) · [conformal-risk code](https://github.com/aangelopoulos/conformal-risk)
- **[Testing for Outliers with Conformal p-values (Bates, Candès, Lei, Romano, Sesia)](https://arxiv.org/abs/2104.08279)** — Track B's validity and the PRDS result
- **[Selection by Prediction with Conformal p-values (Jin & Candès)](https://arxiv.org/abs/2210.01408)** — conformal-p-value selection with FDR control
- **[Benjamini & Yekutieli (2001), FDR under dependency](https://projecteuclid.org/euclid.aos/1013699998)** — the arbitrary-dependence fallback
- [Image Super-Resolution with Guarantees via Conformalized Generative Models](https://arxiv.org/html/2502.09664v1)
- [QUTCC](https://arxiv.org/html/2507.14760) · [Self-supervised Conformal Prediction for Imaging](https://arxiv.org/abs/2502.05127)
- [Task-Driven Uncertainty Quantification in Inverse Problems (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07734.pdf)
- [Conformal Prediction Under Covariate Shift](https://arxiv.org/pdf/1904.06019) · [Conformal Prediction Beyond Exchangeability](https://www.stat.cmu.edu/~ryantibs/papers/nexcp.pdf)
- [UQ for probabilistic ML in Earth observation using conformal prediction](https://www.nature.com/articles/s41598-024-65954-w) · [GeoConformal](https://arxiv.org/html/2412.08661v1)
- [SEN2VENµS](https://zenodo.org/records/6514159) · [SEN2NAIP v2](https://opensr.eu/news/sen2naip-v2-0-released-a-major-boost-for-sentinel-2-super-resolution/) · [WorldStrat](https://worldstrat.github.io/)
- [Sentinel-2 mission — bands and constellation status (SentiWiki)](https://sentiwiki.copernicus.eu/web/s2-mission) · [CDSETool](https://github.com/CDSETool/CDSETool)
