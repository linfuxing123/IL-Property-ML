# Supplementary Materials

**Predictor disagreement as a property-aware diagnostic for distributional bias
in materials machine learning**

Fuxing Lin, Hunan Institute of Engineering

---

## Table S1. Complete six-pair statistics (conductivity)

| Pair | n | corr(f₁,f₂) | corr(m,d) | normalized | t | monotone | layered seq |
|---|---|---|---|---|---|---|---|
| GBM–Hist | 6,108 | 0.912 | −0.268 | −0.250 | −21.7 | ✓ | 0.316, 0.223, 0.212, 0.201 |
| GBM–ILPE | 6,108 | 0.474 | −0.424 | −0.438 | −36.6 | ✓ | 0.930, 0.604, 0.510, 0.372 |
| Hist–ILPE | 6,108 | 0.450 | −0.418 | −0.433 | −35.9 | ✓ | 0.985, 0.632, 0.575, 0.428 |
| GBM–MPNN | 5,000 | 0.817 | −0.415 | −0.416 | −32.3 | ✓ | 0.502, 0.308, 0.251, 0.207 |
| Hist–MPNN | 5,000 | 0.809 | −0.320 | −0.309 | −23.9 | ✓ | 0.488, 0.374, 0.295, 0.248 |
| MPNN–ILPE | 5,000 | 0.458 | −0.464 | −0.466 | −37.0 | ✓ | 0.900, 0.576, 0.502, 0.316 |

## Table S2. Four-property landscape (GBM vs HistGBM, 1,673,919 pairs)

| Property | corr(f₁,f₂) | corr(m,d) | normalized | predicted range | regime |
|---|---|---|---|---|---|
| conductivity | 0.905 | −0.410 | −0.384 | [−15.4, 0.66] | saturation |
| density | 0.937 | +0.271 | +0.260 | [0.85, 2.00] | anti-saturation |
| melting_point | 0.871 | +0.153 | +0.144 | [188, 495] | anti-saturation |
| viscosity | 0.887 | +0.179 | +0.068 | [1.8, 10.7] | anti-saturation |

## Table S3. Diagnostic value: disagreement vs |error| (experimental training set)

| Property | n (ILs) | corr(disagreement, |error|) | Q1→Q4 error | growth |
|---|---|---|---|---|---|
| conductivity | 641 | +0.411 | 0.69→1.12 | +61% |
| density | 1,433 | +0.475 | 0.022→0.047 | +114% |
| melting_point | 3,945 | +0.486 | 14.3→34.7 | +143% |
| viscosity | 2,705 | +0.151 | 0.73→0.94 | +29% |

## Table S4. Training-density coupling test (6,108 ILPE-overlap ILs)

| Split | n | mean disagreement |
|---|---|---|
| Shares training ion | 3,857 | 0.4483 |
| No shared ion | 2,251 | 0.5229 |
| Difference | — | −0.0747 (Welch t = −7.14, p = 1.1e−12) |

Stratified by predicted value (10 quantiles): effect concentrated in high-value
layers (layers 7–9: −0.12 to −0.15; layer 0: −0.17); low-value layers ≈ 0.
Paired test across layers: mean −0.054, p = 0.059; normalized ≈ 12.8% lower
disagreement after controlling for predicted value.

## Table S5. Disagreement-penalized ranking (no gain)

| Property | GT source | mean corr w/ GT | disagreement corr w/ GT | ranking gain |
|---|---|---|---|---|
| conductivity | ILPE (n=6,108) | high | — | none (λ=0–8, 98–100%) |
| melting_point | experiment (n=3,945) | 0.913 | 0.123 | none (100% at all λ) |

## Methods details

- **Data**: paper6 virtual space [1], 1,673,919 IL cation–anion pairs; GBM and
  HistGBM scored on 459 RDKit descriptors per ion pair, T = 298.15 K.
- **MPNN sample**: stratified 5,000-pair sample (20 quantile strata, ≤250
  each); sample percentiles match the full space (p5 −6.10 / p50 −3.31 /
  p95 −1.44).
- **ILPE source**: 6,108 pairs overlapping with positive conductivity
  predictions, log-transformed.
- **Statistics**: Pearson correlation + Welch t-test; variance normalization
  (center + unit scale) as the saturation test; layered monotonicity over
  quantiles; stratified matching for the coupling test.
- **Verification**: covariance identity verified symbolically and numerically
  (math-agent math_verify: SYMBOLIC EQUAL, NUMERIC PASS).

## Reproducibility

All analyses are reproducible from scripts in the repository:
`code/paper7/` (divergence_theory.py, cross_paradigm_divergence.py,
mpnn_paradigm.py, multiprop_saturation.py, diagnostic_value.py,
train_coupling_significance.py, disagreement_ranking.py,
candidate_uncertainty.py, final_audit.py).
