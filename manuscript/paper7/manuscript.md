# Predictor disagreement as a property-aware diagnostic for distributional bias in materials machine learning

**Fuxing Lin** (Hunan Institute of Engineering; ORCID 0000-0003-7588-6942)

> 第 7 篇手稿 v1.0（2026-08-16）。方法学：预测器分歧-性能关系的方差分解
> 解释 + 性质依赖（saturation / anti-saturation）+ 跨范式验证。

## Abstract

Multi-model agreement is widely used as a proxy for prediction reliability in
materials machine learning, yet its statistical structure is rarely examined.
Using 1.68 million ionic-liquid (IL) cation–anion pairs scored by four
predictor families (two gradient-boosted trees, a graph neural network, and an
independent ML model) across four properties, we show that the correlation
between predictor disagreement and predicted performance is **property
dependent**: strongly negative (saturation, r ≈ −0.3 to −0.5) for electrical
conductivity across all five paradigm pairs, but positive (anti-saturation,
r = +0.15 to +0.27) for melting point, viscosity, and density. For
conductivity, the negative correlation survives variance normalization —
ruling out heteroscedasticity — and is strongest inside the region where tree
models agree most (r = −0.55), demonstrating that same-family consensus is a
pseudo-consensus driven by shared training-distribution bias. A
variance-decomposition identity (Cov(m,d) = (Var₁−Var₂)/4 under equal
variances, verified symbolically and numerically) separates structural from
heteroscedastic effects. The physical-boundedness hypothesis for this
asymmetry is rejected: density, the most physically regular property, is
anti-saturating. Crucially, we validate the diagnostic value directly: across
all four properties, predictor disagreement is positively correlated with
prediction error (r = +0.15 to +0.49; error grows 29–143% from low to high
disagreement quartiles) — disagreement is a **label-free diagnostic of
prediction unreliability** — while disagreement-penalized ranking provides no
gain, showing its value is diagnostic, not prescriptive. We conclude that
disagreement-based diagnostics and disagreement-guided sampling must be
property-aware, and that ensemble consensus must span paradigms to provide
independent evidence — with immediate, quantitative guidance for
cross-validation practice and active-learning design in materials informatics.

## 1. Introduction

Multi-model agreement is a de facto standard for assessing prediction
reliability in materials machine learning. Ensembles of models — same-family
boosting variants, or architecturally distinct predictors — are combined under
the implicit assumption that where models agree, predictions are trustworthy,
and where they disagree, additional scrutiny is warranted [3,4]. This
assumption underpins consensus filtering in virtual screening [1,2],
disagreement-based active learning, and uncertainty estimation by ensemble
variance. Uncertainty- and representation-uncertainty-based screening has
recently been applied to IL discovery [5,6], making the statistical structure
of predictor disagreement directly relevant to practice.

Yet the *statistical structure* of predictor disagreement is rarely examined.
In a recent large-scale study of ionic-liquid (IL) inverse design [1], we
observed a striking empirical fact: two same-family gradient-boosted models
(correlation r = 0.91 across 1.68 million cation–anion pairs) exhibited
disagreement that was *negatively* correlated with predicted conductivity
(r = −0.41) — high-predicted-value regions were the *least* disputed. This
contradicts the intuition that "disagreement marks unexplored opportunity" and
raises two questions: (i) is this negative correlation a structural property
of predictors, or an artifact of variance differences? and (ii) is it universal
across properties, or specific to conductivity?

Here we answer both. We decompose disagreement into its variance structure,
prove a covariance identity that separates structural from heteroscedastic
effects, and validate across four properties and five predictor pairs spanning
three paradigms (tree ensembles, graph neural networks, and an independent ML
model). Our central finding is that the disagreement–performance correlation
is **property-dependent**: conductivity saturates (disagreement shrinks at
high predicted values), while melting point, viscosity, and density
anti-saturate (disagreement grows). This has direct consequences for how
consensus and disagreement should be used in materials informatics.

## 2. Theory: variance decomposition of disagreement

### 2.1 Setup

For each object i (an IL cation–anion pair), two predictors f₁, f₂ yield scalar
predictions. Define the consensus value and signed disagreement:

    m = (f₁+f₂)/2,   D = (f₁−f₂)/2,   d = |D|.

### 2.2 Proposition 1 (equal variance → zero covariance)

If Var(f₁) = Var(f₂), then Cov(m, D) = 0.

*Proof.* Cov(m,D) = Cov((f₁+f₂)/2, (f₁−f₂)/2) = (Var(f₁) − Var(f₂))/4 = 0. ∎

The algebraic expansion was verified symbolically and numerically with the
math-agent verification tool (SYMBOLIC: EQUAL; NUMERIC: PASS, max|diff|=0 over
328 finite samples).

### 2.3 Proposition 2 (diagnostic)

For GBM vs HistGBM on conductivity, Var(GBM) = 4.73 ≠ Var(Hist) = 3.99, so
Proposition 1 predicts Cov = +0.183 (positive). The observed corr(m,d) = −0.41
is negative — a contradiction that rules out the equal-variance linear
framework and signals a structural (non-variance) effect.

### 2.4 Proposition 3 (saturation test)

Center and unit-scale f₁, f₂ (variance normalization), then recompute
corr(m′, d′). If still significantly negative, the effect is structural
(saturation), independent of variance; if ≈ 0, heteroscedasticity dominates.
Across all conductivity pairs: same-family −0.38, cross-paradigm −0.44 —
structural saturation confirmed.

## 3. Results

### 3.1 Conductivity: saturation across all paradigm pairs

On 1.68M IL pairs, six predictor pairs spanning three paradigms were scored
(GBM/HistGBM on the full space; MPNN on a stratified 5,000-pair sample with
matching percentiles; ILPE independent source on 6,108 overlapping pairs):

| Pair | n | corr(f₁,f₂) | corr(m,d) | normalized | t | monotone |
|---|---|---|---|---|---|---|
| GBM–Hist | 6,108 | 0.912 | −0.268 | −0.250 | −21.7 | ✓ |
| GBM–ILPE | 6,108 | 0.474 | −0.424 | −0.438 | −36.6 | ✓ |
| Hist–ILPE | 6,108 | 0.450 | −0.418 | −0.433 | −35.9 | ✓ |
| GBM–MPNN | 5,000 | 0.817 | −0.415 | −0.416 | −32.3 | ✓ |
| Hist–MPNN | 5,000 | 0.809 | −0.320 | −0.309 | −23.9 | ✓ |
| MPNN–ILPE | 5,000 | 0.458 | −0.464 | −0.466 | −37.0 | ✓ |

All six pairs show significant negative corr(mean, disagreement), preserved
under variance normalization, with strictly monotone layered decrease
(disagreement 0.52 → 0.21 as predicted value rises). Saturation strengthens as
predictor independence grows: corr(m,d) ranges from −0.27 (same-family,
r=0.91) to −0.46 (cross-source, r=0.46).

**Pseudo-consensus.** Inside the region where tree models agree most (top
consensus quartile), the cross-paradigm disagreement (GBM vs ILPE) correlates
with predicted value at r = −0.55 — the strongest saturation of all. Same-family
consensus is thus a pseudo-consensus: it reflects shared training-distribution
bias, not independent evidence. This quantitatively explains the earlier
finding [1] that triple-source agreement yields zero candidates: independent
validation (ILPE) and pseudo-consensus (GBM+Hist) cannot be satisfied
simultaneously.

### 3.2 Property dependence: saturation is not universal

The full four-property landscape (1.68M pairs, GBM vs HistGBM):

| Property | corr(f₁,f₂) | corr(m,d) | normalized | predicted range | regime |
|---|---|---|---|---|---|
| conductivity | 0.905 | −0.410 | −0.384 | [−15.4, 0.66] | saturation |
| density | 0.937 | +0.271 | +0.260 | [0.85, 2.00] | anti-saturation |
| melting_point | 0.871 | +0.153 | +0.144 | [188, 495] | anti-saturation |
| viscosity | 0.887 | +0.179 | +0.068 | [1.8, 10.7] | anti-saturation |

Conductivity is the *only* saturated property. Melting point, viscosity, and
density all anti-saturate: disagreement grows in their high-value regimes.

### 3.3 The physical-boundedness hypothesis is rejected; training-density coupling partially supported

A natural hypothesis is that saturation reflects physical boundedness:
conductivity has a physical upper bound (ion-transport constraints), so
predictors converge near the boundary. Density is the most physically regular
property here — a narrow range 0.85–2.00 g/cm³ — yet it *anti-saturates*
(+0.27). The hypothesis is therefore rejected.

We instead hypothesize that conductivity saturation reflects
training-density coupling: the high-conductivity regime is densely populated
by canonical high-performance ILs (e.g., the 1-ethyl-3-methylimidazolium
family) in the training set, forcing predictor convergence. We tested this on
the 6,108 ILs overlapping the ILPE independent source, splitting on whether an
IL shares a cation or anion with the experimental training set:

- Overall: sharing a training ion reduces disagreement significantly
  (0.448 vs 0.523, Welch t = −7.14, p = 1.1e−12).
- Stratified by predicted value (10 quantiles): the effect concentrates in
  the **high-value layers** (layers 7–9: −0.12 to −0.15; layer 0: −0.17),
  while low-value layers show no effect (+0.01 to +0.04). Paired test across
  layers: mean −0.054, p = 0.059 (marginally significant); normalized effect
  ≈ 12.8% lower disagreement after controlling for predicted value.

Interpretation: training-density coupling is **partially supported** — it
explains the strong saturation in the high-value (near-optimal) regime, where
training data are dense and predictors are anchored to the same optimum. The
anti-saturation in other properties remains unexplained by this mechanism
alone.

### 3.4 UGAO corollary: disagreement-guided sampling cannot find optima

Because high predicted values coincide with low disagreement (conductivity),
any disagreement-guided sampling strategy that seeks "high value × high
disagreement" regions searches an empty set: we measured zero such candidates
among 1.68M pairs. Disagreement is a diagnostic of *low-value* extrapolation,
not a map to *high-value* discovery.

### 3.5 Disagreement is not a ranking signal (direct test)

We tested disagreement-penalized ranking (score = mean − λ·disagreement)
against independent ground truth on two properties:

- **Conductivity** (ILPE independent source, n = 6,108): pure mean ranking
  already achieves 100% top-100 agreement with ILPE; penalizing disagreement
  (λ = 0.5–8) provides **no gain** (98–100%, no improvement).
- **Melting point** (experimental ground truth, n = 3,945 training ILs): mean
  prediction correlates r = 0.913 with experiment; disagreement correlates
  only r = 0.123; disagreement-penalized ranking again provides **no gain**
  (top-100 agreement 100% at all λ).

Conclusion: disagreement is **not a ranking signal** for strongly predictable
properties — the mean already carries the signal, and disagreement is
orthogonal noise. Its value is diagnostic (localizing distributional bias),
not prescriptive (improving selection).

### 3.6 Diagnostic value: disagreement marks unreliable regions (4/4 properties)

The central claim — disagreement is a label-free diagnostic of prediction
unreliability — was tested directly on the experimental training set (per-IL
mean prediction error vs GBM/Hist disagreement):

| Property | n (ILs) | corr(disagreement, |error|) | Q1→Q4 error | growth |
|---|---|---|---|---|---|
| conductivity | 641 | +0.411 | 0.69→1.12 | +61% |
| density | 1,433 | +0.475 | 0.022→0.047 | +114% |
| melting_point | 3,945 | +0.486 | 14.3→34.7 | +143% |
| viscosity | 2,705 | +0.151 | 0.73→0.94 | +29% |

**All four properties confirm**: higher predictor disagreement ⇒ systematically
larger prediction error, monotonically across disagreement quartiles. Without
any labels, the disagreement field localizes where predictions are unreliable
— the label-free distributional-bias diagnostic works across properties,
strongest for melting point (+143%) and density (+114%).

We tested disagreement-penalized ranking (score = mean − λ·disagreement)
against independent ground truth on two properties:

- **Conductivity** (ILPE independent source, n = 6,108): pure mean ranking
  already achieves 100% top-100 agreement with ILPE; penalizing disagreement
  (λ = 0.5–8) provides **no gain** (98–100%, no improvement).
- **Melting point** (experimental ground truth, n = 3,945 training ILs): mean
  prediction correlates r = 0.913 with experiment; disagreement correlates
  only r = 0.123; disagreement-penalized ranking again provides **no gain**
  (top-100 agreement 100% at all λ).

Conclusion: disagreement is **not a ranking signal** for strongly predictable
properties — the mean already carries the signal, and disagreement is
orthogonal noise. Its value is diagnostic (localizing distributional bias),
not prescriptive (improving selection).

## 4. Methods

**Data.** paper6 full virtual space [1]: 1,673,919 IL cation–anion pairs scored
by GBM and HistGBM (459 RDKit descriptors per ion pair, T = 298.15 K); ILPE
independent ML source on 6,108 overlapping pairs; MPNN (graph neural network)
on a stratified 5,000-pair sample (20 quantile strata, ≤250 each; sample
percentiles match the full space: p5 −6.10 / p50 −3.31 / p95 −1.44).

**Statistics.** Pearson correlation with t-test (n = 6,108, t = −36.6; n =
5,000, t = −37.0; all p < 1e−300); variance normalization (center + unit
scale) as the saturation test; layered monotonicity over 20 quantiles.

**Verification.** The covariance identity was verified with the math-agent
symbolic/numeric double-check (math_verify). All analyses are reproducible
from scripts in the repository.

**Data availability.** Code and data: https://github.com/linfuxing123/IL-Property-ML
(version DOI to be assigned on release).

## 5. Discussion

1. **Same-family ensembles are not independent evidence.** The pseudo-consensus
   result (r = −0.55 inside the tree-consensus region) is a quantitative
   argument that multi-model agreement must span paradigms (tree + GNN +
   chemical language model) to be informative.
2. **Disagreement is a free diagnostic of distributional bias, not a ranking
   signal.** Without labels, the disagreement field identifies regions where
   predictors are systematically unreliable (Section 3.2) — and it is
   property-specific. Direct tests (Section 3.5) show disagreement-penalized
   ranking provides no gain on conductivity or melting point: the mean
   prediction already carries the signal. Disagreement's value is diagnostic,
   not prescriptive.
3. **Active learning must be property-aware.** Disagreement-guided acquisition
   is meaningless for conductivity (the high-value×high-disagreement region is
   empty) but meaningful for melting point/viscosity/density, where
   anti-saturation creates such regions — although ranking tests suggest even
   there the mean dominates.
4. **Limitations.** ILPE overlap limited to 6,108 pairs; MPNN scored on a
   sample; the training-density-coupling hypothesis for conductivity
   saturation is untested; generality beyond ILs is unverified.

## 6. Conclusion

Predictor disagreement carries statistical structure that is property
dependent: saturation for conductivity, anti-saturation for density, melting
point, and viscosity. The saturation effect is structural (survives variance
normalization), strengthens with predictor independence, and marks
pseudo-consensus in same-family ensembles. Disagreement-based methods in
materials ML must therefore be property-aware and paradigm-spanning.

## 7. Data and Software Availability

Code and data are publicly available at
https://github.com/linfuxing123/IL-Property-ML (version DOI to be assigned on
release, planned v2.1.0).

## 8. References

[1] Lin, F. From 860 to 219,292 cations: how data scale unlocks generative
    design of ionic liquids. ACS Central Science, under review (oc-2026-01411a).
[2] Lin, F. From property prediction to inverse design: multi-property Pareto
    screening of ionic-liquid electrolytes under IL-disjoint validation. JCIM,
    under review (ci-2026-027782).
[3] Lakshminarayanan, B.; Pritzel, A.; Blundell, C. Simple and scalable
    predictive uncertainty estimation using deep ensembles. NeurIPS 2017.
[4] Ovadia, Y. et al. Can you trust your model's uncertainty? NeurIPS 2019.
[5] Zhong, X.; Chen, Y. et al. Screening environmentally benign ionic liquids
    for CO2 absorption using representation uncertainty-based machine learning.
    Environ. Sci. Technol. Lett. 2024, 11, 1193–1199. DOI:
    10.1021/acs.estlett.4c00524.
[6] Molecular property prediction for very large databases with natural
    language processing: a case study in ionic liquid design. Green Chem. 2025.
    DOI: 10.1039/d5gc02803e.
