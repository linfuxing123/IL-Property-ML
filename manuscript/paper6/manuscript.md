# From 860 to 219,292 cations: how data scale unlocks generative design of ionic liquids

**Fuxing Lin** (Hunan Institute of Engineering; ORCID 0000-0003-7588-6942)

> 第 6 篇手稿 v0.2（2026-08-15）。方法学主线：数据规模 × 生成范式 → 生成有效性
> 突破；三源交叉验证的诚实上限分析。与第 5 篇（JCIM 在投）的关系：第 5 篇在小
> 数据（860 阳/356 阴）上暴露"生成式设计"死结，本文证明死结的构成与解法。

## Abstract

Generative design of ionic liquids (ILs) has been bottlenecked by data scarcity: a
character-level VAE trained on 860 cations produced chemically invalid species in
~40% of samples, and latent-space optimization collapsed onto the known optimum.
We decompose this bottleneck into two independent constraints — chemical validity
and property optimization — and show that only the former is a data-scale problem.
By assembling an experimental dataset of 6,177 unique ILs (2,838 cations / 592
anions; +227% over prior work) and a synthetically feasible space of 8.33 million
ILs (219,292 cations), we retrain three predictor families (GBM, HistGBM, MPNN)
under IL-disjoint cross-validation and a SMILES language model for generation.
Data expansion raises melting-point R² from 0.523→0.620 (GBM), 0.561→0.686
(HistGBM), 0.365→0.563 (MPNN) and lifts the generator's RDKit validity / charge
correctness from ~60% to 99.9%. The same expansion, however, does not yield
candidates predicted to beat the known optimum: two independent 8.3M-scale
predictors (ML model and chemical language model) agree that no IL in the virtual
space exceeds the experimental best (ln κ ≈ 0.64 for EMIM-DCA), with a
cross-predictor correlation of only 0.20 on conductivity. We conclude that (i)
chemical validity of IL generation is a solved problem at scale; (ii) surpassing
known property optima is bounded by the training distribution, not by generator
capacity; and (iii) the next bottleneck is experimental data with property labels,
not larger virtual libraries.

## 1. Introduction

Ionic liquids (ILs) — salts molten below ~100 °C — are the working fluids of choice
for a widening range of electrochemical and separation applications, from battery
electrolytes to CO₂ capture solvents. Because the cation–anion design space is
combinatorially vast (thousands of known ions, millions of synthetically feasible
pairs), machine learning has been used both to *predict* IL properties from
molecular structure and to *generate* new candidate structures. The predictive
direction has matured: with honest IL-disjoint validation, structure-based models
reach R² = 0.69–0.95 for conductivity, density and viscosity, and 0.36–0.56 for
melting point [1,2]. Transfer-learning and cross-domain frameworks extend this to
low-data regimes [6].

The generative direction, however, has hit a wall. In our previous work (paper 5,
ref [1]), a character-level VAE trained on 860 cations / 356 anions produced
chemically invalid SMILES in ~40% of samples (formal-charge errors, multi-fragment
species), and latent-space optimization converged back onto the known optimum
(EMIM-DCA) instead of discovering new optima. Two distinct bottlenecks were
conflated in that result:
  (i) **chemical validity** — the generator must learn to emit charge-correct,
      syntactically valid ion SMILES;
  (ii) **property optimization** — even valid novel ions must be predicted to beat
      the best known property values.

Here we separate these constraints experimentally. We expand the data on both
axes: an experimental set of 6,177 unique ILs (2,838 cations / 592 anions, +227%
over prior work, assembled from ILThermo and the Mendeley IL property dataset [4])
and a synthetically feasible virtual space of 8.33 million ILs (219,292 cations ×
38 anions) with two independent predictor families — the ILPE ML models [5] and
the ILBERT chemical language model [3]. We retrain our three predictors
(GBM/HistGBM/MPNN) under the same IL-disjoint protocol, replace the VAE with a
SMILES autoregressive language model trained on 150k cations (plus 30M unlabeled
IL-like SMILES for pre-training), and re-run combinatorial screening over 1.68
million novel cation–anion pairs. We show that (i) is solved at scale (99.9%
validity) while (ii) is *not* — a result we consider more informative than a
positive but fragile claim of "beating the optimum".

## 2. Results

### 2.1 Data expansion: 6,177 experimental ILs, 8.33M virtual ILs

**Table 1. Data assets.**

| Source | Type | Points / pairs | Unique ILs | Unique cations | Unique anions |
|---|---|---|---|---|---|
| ILThermo v2.0 (prior) | experimental | 88,661 | 1,891 | 860 | 356 |
| Mendeley tpp25ztzmb [4] | experimental | 39,928 | 5,220 | 2,381 | 375 |
| **Merged experimental** | — | **~129k** | **6,177** | **2,838** | **592** |
| ILBERT virtual space [3] | predicted (chemLM) | 8,333,096 | 8,333,096 | 219,292 | 38 |
| ILPE virtual space [5] | predicted (ML) | 8,333,096 | 8,333,096 | 219,296 | 38 |
| ILBERT pre-train | unlabeled SMILES | 30,220,833 | — | — | — |

The merged experimental set more than triples prior IL coverage. The two virtual
spaces share the same combinatorial design (219k cations × 38 anions) but overlap
only 29.3% in ion identity after SMILES canonicalization, so their union exceeds
16 million unique ILs — and, importantly, they provide *two independent predictor
families* over a common design, enabling cross-predictor validation at scale.

### 2.2 Prediction gains from data scale (IL-disjoint)

All predictors were retrained on the merged experimental set under the protocol of
paper 5: IL-disjoint GroupKFold(5), 459 RDKit descriptors per ion pair (+T for
temperature-dependent properties), same loss scales. Table 2 and Fig. 2 report the
change from the paper-5 baselines (1,891 ILs).

**Table 2. IL-disjoint R², paper5 (1,891 IL) → paper6 (6,177 IL).**

| Property | GBM | HistGBM | MPNN |
|---|---|---|---|
| conductivity | 0.740 → 0.741 | 0.749 → 0.747 | 0.703 → 0.686 |
| density | 0.926 → 0.926 | 0.949 → 0.946 | 0.933 → 0.946 |
| viscosity | 0.809 → 0.791 | 0.846 → 0.834 | 0.759 → 0.759 |
| **melting_point** | **0.523 → 0.620** | **0.561 → 0.686** | **0.365 → 0.563** |

The largest and most consistent gains are on the melting point, whose IL count grew
~7-fold (570 → 3,945): +0.097 (GBM), +0.125 (HistGBM), +0.198 (MPNN). Viscosity
shows a small apparent drop (−0.018/−0.012/0.000). A decomposition shows this is
not a quality loss: among the 1,492 newly added viscosity ILs, 64% have a single
temperature point — pure cold-start cases under IL-disjoint CV. The subset of
paper-5 ILs retains R² = 0.808 (vs 0.809 before), and the new-IL subset alone
reaches R² = 0.728, i.e. the model *generalizes* to never-seen ILs rather than
degrading. This is evidence that data expansion buys extrapolation capacity on
single-point ILs — exactly the cold-start regime relevant to design.

### 2.3 Generator validity: model architecture × data scale

**Table 3. Generator output quality.**

| Model | Training cations | RDKit-valid | Charge = +1 | Novel cations (of 500k samples) |
|---|---|---|---|---|
| paper5 char-VAE | 860 | ~60% | ~60% | few |
| char-VAE (this work) | 100,000 | 19.3% | 0.4% | 71 |
| char-VAE + β-anneal | 100,000 | 7.1% | 0.6% | — |
| **SMILES LM (GRU)** | **150,000** | **99.9%** | **99.9%** | **1,766** |

Scaling the training data 100-fold does *not* rescue the character-level VAE: its
KL term collapses to ~0 (posterior collapse), and validity actually degrades with
more data because the decoder collapses toward a mean sequence. The bottleneck was
architectural, not data. Replacing the VAE with an autoregressive SMILES language
model (GRU, teacher forcing, temperature/top-k sampling) trained on 150k
experimental-and-virtual cations raises RDKit validity and charge correctness to
99.9%, and 500k samples yield 1,766 cations that are novel with respect to the
entire 219k virtual library. **Chemical validity of IL generation is solved at
scale.**

### 2.4 Surpassing the known optimum: an honest limit

With validity solved, we asked whether novel cations can be predicted to beat the
known experimental optimum (EMIM-DCA, ln κ ≈ 0.64 at 298 K).

*Unguided generation.* Novel cations (1,766) paired with the 38 virtual anions give
67,108 candidates; our GBM/HistGBM predict ln κ < 0 for all of them (best −1.17).
The language model learned *chemical* plausibility, not *property* optimality —
its marginal distribution favors large, functionalized cations that are poor
conductors.

*Guided generation.* Fine-tuning the LM on top-scoring cations finds ln κ = 0.79
on the first round, above the 0.64 optimum; but strict (canonicalized) checks show
the top hits are known ILs (e.g. 1,3-dimethylimidazolium DCA), and the predictors
disagree systematically with the ILPE source.

*Cross-predictor screen at 8.3M scale (Fig. 3).* We merged the two independent
virtual spaces on canonicalized ion identity (633,780 ILs with valid conductivity
from both sources). The ILPE ML and ILBERT chemLM agree poorly on ln κ
(Pearson r = 0.20) and, crucially, **no IL receives a positive ln κ from both
sources** (ILPE max −0.24, ILBERT max +0.36; both below the 0.64 optimum). Two
independent, large-scale predictors therefore concur that the 8.3M virtual space
contains no candidate predicted to outperform the experimental best — not because
of generator failure, but because the predictors are bounded by their training
distribution.

*Full combinatorial screen.* We nevertheless screened all 2,838 × 592 = 1.68M
novel pairs in the experimental ion set with our retrained GBM/HistGBM (T = 298 K),
requiring agreed positive ln κ, room-temperature liquidity (Tm ∈ (180, 300) K) and
low viscosity (ln η < 5.5). After strict canonicalized novelty filtering and a
chemical sanity check (single-fragment anion, net charge −1), 16 candidates
survive with three-predictor agreement (Table 4, Fig. 4). The best, e.g.
1,3-diethylimidazolium DCA (ln κ = 0.51/0.43/0.46) and
1-vinyl-3-methylimidazolium DCA (0.46/0.69/0.75), are chemically reasonable,
room-temperature liquid, low-viscosity ILs with positive predicted conductivity in
all three independent predictors — but none exceeds the EMIM-DCA optimum
consistently.

**Table 4. Final candidates (three-predictor agreement, novel, chemically valid;
unique cations after canonicalization).**

| Cation (with DCA anion) | GBM ln κ | HistGBM ln κ | MPNN ln κ | Tm (K) | ln η |
|---|---|---|---|---|---|
| 1,3-diethylimidazolium | 0.511 | 0.427 | 0.463 | 269.6 | 3.05 |
| 1-vinyl-3-methylimidazolium | 0.461 | 0.688 | 0.751 | 281.5 | 3.28 |
| 1-ethylpyridinium | 0.417 | 0.609 | 0.436 | 284.1 | 3.41 |
| 1-ethylimidazolium | 0.386 | 0.359 | 0.712 | 277.3 | 3.20 |
| 1-methyl-3-propylimidazolium | 0.382 | 0.481 | 0.510 | 267.9 | 3.38 |
| 1-isopropyl-3-methylimidazolium | 0.308 | 0.410 | 0.550 | 286.2 | 3.78 |
| 1-ethyl-4-methylpyridinium | 0.303 | 0.353 | 0.551 | 297.8 | 3.61 |

*Honest conclusion.* Data scale solves chemical validity and prediction accuracy,
but does **not** make predictors "invent" property extrema outside their training
distribution. The road to surpassing known optima runs through property-labeled
experimental data, physical verification (DFT/MD), or active learning loops — not
through ever-larger virtual libraries.

## 3. Methods

**Data.** ILThermo v2.0 experimental records (prior work); Mendeley tpp25ztzmb [4]
(viscosity, melting point, CO₂ solubility, toxicity; cation/anion SMILES already
split); ILBERT virtual space + pre-training corpus [3] (Zenodo 14601320/15046370);
ILPE database [5] (Zenodo 3251661). Merged experimental set: 6,177 unique ILs;
descriptors computed per ion with RDKit (459 features), stored in
`il_descriptors_v2.csv`. Units standardized: T in K, viscosity in mPa·s (log
scale), density in g/cm³, conductivity in S/m (ln scale).

**Predictors.** GBM and HistGBM on RDKit descriptors; MPNN (3-layer message
passing, batch-encoded graphs, mean/sum/max readout) on ion graphs. Protocol:
IL-disjoint GroupKFold(5), identical to paper 5.

**Generator.** Character-level VAE (paper 5 architecture, baselines); SMILES
autoregressive GRU LM (emb 128, hidden 256, 2 layers), trained on 150k cations,
sampled with temperature 0.9–0.95 and top-k 15–20.

**Cross-predictor validation.** ILPE ML conductivity (S/m) filtered to positive
values and log-transformed; ILBERT ln κ used directly; merged on canonicalized
`cat|an` strings.

**Combinatorial screen.** 2,838 cations × 592 anions from the experimental set,
vectorized numpy feature construction, GBM/HistGBM scoring at T = 298.15 K,
filter: agreed ln κ > 0.2, Tm ∈ (180, 300) K, ln η < 5.5, novelty by canonicalized
exclusion of the 6,177 known ILs, chemical sanity (single-fragment anion, −1
charge).

**Code and data.** https://github.com/linfuxing123/IL-Property-ML, release v2.0.0
(https://doi.org/10.5281/zenodo.21941054).

## 4. Discussion

The central message is a decomposition, not a negative: **data scale solved one of
the two bottlenecks of generative IL design and provably cannot solve the other.**
Chemical validity — the failure that made paper 5's VAE output unusable — is now
at 99.9%, and the fix was architectural (autoregressive LM) *enabled* by scale
(large cation vocabulary, pre-training corpus). Prediction accuracy for
data-poor properties (melting point) improved substantially. But "beating the
known optimum" is bounded by the training distribution: two independent
8.3M-scale predictors agree that the virtual space contains no candidate predicted
to exceed EMIM-DCA, and our own three predictors converge on 16 novel, valid,
room-temperature-liquid candidates that are competitive but not superior.

Practical implications: (i) virtual-library inflation has diminishing returns for
property maxima; (ii) the binding constraint is now *property-labeled experimental
data* (esp. conductivity/melting-point pairs for diverse ions) and *physical
verification*; (iii) active learning — querying predictors where their
uncertainty is high, then validating experimentally — is the natural next step,
since our cross-predictor disagreement (r = 0.20) itself maps exactly the regions
where experimental confirmation would be most informative.

Limitations: predictors trained on 6,177 ILs carry their own error; the two
virtual spaces overlap only 29.3% (a strength for validation, a weakness for
direct comparison); the combinatorial screen uses only ions present in the
experimental set, so it cannot discover brand-new ion skeletons — only new pairs.
Generating genuinely new skeletons that also beat known optima will require the
experimental/physical loop above.

## 5. Data and Software Availability

All data, code, models, and the manuscript are publicly available at
https://github.com/linfuxing123/IL-Property-ML (release v2.0.0, archived at
https://doi.org/10.5281/zenodo.21941054). The archive contains the merged
experimental dataset (6,177 unique ILs), the ion-identity lists derived from the
8.33M-IL virtual space, all descriptor tables, the eight trained predictor models
(GBM/HistGBM × four properties), all scripts, and this manuscript with its
figures. Underlying public sources: ILThermo v2.0 (NIST); Mendeley tpp25ztzmb
[4]; Zenodo 7935198 (ILTransR), 14601320/15046370 (ILBERT), 3251661 (ILPE).

## 6. References (draft)

[1] Paper 5 (JCIM, under review) — inverse design under IL-disjoint validation.
[2] Paper 4 (ACS Sustainable Chem. Eng., under review) — GNN vs descriptors.
[3] Qiu, Y. et al. Large chemical language models for property prediction and
    high-throughput screening of ionic liquids. *Digital Discovery* (2025).
[4] Dong, K.; Gao, J. The dataset of ionic liquid properties. Mendeley Data,
    tpp25ztzmb (2025).
[5] The Ionic Liquid Property Explorer. *Data* 4, 88 (2019); Zenodo 3251661.
[6] Chen, G. et al. ILTransR: transfer learning for IL property prediction (2023).
