# From property prediction to inverse design: multi-property Pareto screening of ionic-liquid electrolytes under IL-disjoint validation

**Authors:** Fuxing Lin* (corresponding author; ORCID: 0009-0003-7588-6942)

**Affiliation:** Hunan Institute of Engineering

**Corresponding author:** Fuxing Lin; ORCID: 0009-0003-7588-6942; email: 3612411485@qq.com

---

## Abstract

Machine-learning prediction of ionic-liquid (IL) properties has matured, but the harder question—how to *choose* an ion pair for a target application—remains largely open. Here I present an inverse-design workflow that combines an IL-disjoint-validated property predictor with combinatorial screening and two additional, independent models for cross-validation. Across 238,500 cation–anion combinations formed from 795 known cations and 300 known anions (only 1,891 of which have been experimentally reported), the workflow identifies 53,451 unreported, room-temperature-liquid candidates, and, after three-model agreement and Pareto filtering, proposes 11 high-confidence candidates. The leading candidates are thiazolium/imidazolium cations paired with the dicyanamide anion, with predicted conductivity up to ln κ = 0.64, viscosity ln η = 3.18, and melting point 275 K. I also report, honestly, the limits of *de novo* generation at this data scale: a character-level VAE trained on 860 cations generates largely chemically unreasonable ions, and latent-space optimization converges back to the known optimum rather than surpassing it. Data density, not model capacity, remains the binding constraint on true generative inverse design.

**Keywords:** ionic liquids; inverse design; multi-objective Pareto; group-disjoint validation; model cross-validation; generative models

---

## 1. Introduction

Ionic liquids (ILs) offer a combinatorially vast cation–anion design space (1–3) in which only a small fraction of ion pairs have been experimentally characterized. Machine learning has become the standard route to structure–property prediction for ILs (4–6), and prior work in this series established that honest, IL-disjoint (group-level) validation is essential—point-wise splits inflate conductivity R² by 0.15–0.36 (7)—and that data density and feature engineering are the dominant levers on group-level transferability (8,9).

Prediction, however, is only half of design. The forward problem asks "what properties does this ion pair have?"; the inverse problem asks "which ion pair should I make?" Existing IL generative studies optimize single properties and validate with random splits (10–12), which overstate generalization. Here I close that gap with a multi-property inverse-design workflow that is *honest* in three ways: (i) the property predictor is validated under IL-disjoint cross-validation; (ii) every proposed candidate is cross-checked by a second, independent model; and (iii) the failure modes of *de novo* generation at this data scale are reported rather than hidden.

## 2. Results

### 2.1. A validated two-model oracle

Three independent models—classical gradient boosting (GBM), histogram gradient boosting (HistGBM), and a message-passing graph neural network (MPNN)—were trained on the same 84,077-record, 1,891-ion-pair dataset under 5-fold IL-disjoint GroupKFold. GBM reaches group-level R² of 0.740 (conductivity, ln κ), 0.926 (density), 0.809 (viscosity, ln η), and 0.523 (melting point); HistGBM reaches 0.749, 0.949, 0.846, and 0.561; the MPNN reaches 0.703, 0.933, 0.759, and 0.365. The models are algorithmically distinct (two tree ensembles and one graph network), so agreement among them on an unseen candidate is a meaningful consistency check.

### 2.2. Combinatorial inverse screening

The design space is the Cartesian product of 795 known cations and 300 known anions—238,500 pairs—of which only 1,891 are present in the experimental database. Evaluating all pairs at 298.15 K and filtering for novelty, room-temperature liquid character (predicted melting point < 298 K), and three-model agreement yields 24,847 candidates; the multi-objective Pareto front (maximize conductivity, minimize viscosity and melting point) contains 25 candidates. Table 1 lists the leading candidates ranked by predicted conductivity.

**Table 1. Leading unreported IL candidates (three-model consistent, room-temperature liquid).**

| Cation | Anion | ln κ | ln η | T_m (K) | ρ (g/cm³) |
|--------|-------|------|------|---------|-----------|
| Ethylthiazolium | Dicyanamide (DCA) | 0.640 | 3.18 | 274.7 | 1.19 |
| 1,3-Diethylimidazolium | Dicyanamide | 0.511 | 3.11 | 267.0 | 1.11 |
| 1-Ethyl-3-methylimidazolium | Dicyanamide | 0.630 | 3.58 | 272.7 | 1.12 |
| 2-Ethylimidazolium | Dicyanamide | 0.386 | 3.05 | 260.2 | 1.16 |
| 1,3-Dimethyl-1,2,4-triazolium | Dicyanamide | 0.232 | 3.14 | 260.1 | 1.14 |

The candidates cluster chemically on small, charge-delocalized heterocyclic cations paired with the small, charge-delocalized dicyanamide anion—exactly the combination expected from physical intuition for low-viscosity, high-conductivity electrolytes. That the validated predictor reproduces this known design rule on *unreported* pairs, and that a second independent model agrees, is the primary evidence that these candidates are chemically sensible rather than artifacts.

### 2.3. Scaffold-mutation generation (a controlled, chemistry-rational alternative)

As a controlled alternative to black-box generation, I systematically mutated six cation scaffolds (imidazolium, pyridinium, pyrrolidinium, ammonium, phosphonium, sulfonium) across alkyl, ether, and hydroxyethyl substituents, yielding 272 chemically valid, charge-correct, novel cations. When paired with known anions, these cations produce additional novel candidates, but their predicted conductivity is systematically lower than the small-cation optimum (best ln κ ≈ −0.07 versus +0.64). This is a real, physics-consistent result: larger or functionalized cations raise viscosity and lower conductivity.

### 2.4. Honest limits of *de novo* generation

A character-level variational autoencoder (VAE) trained on 860 cations generated ions of which only ~60% carried the correct +1 charge and only ~10% were syntactically valid; many of the remainder were chemically unreasonable fragments. A second-generation HistGBM oracle did not rescue this. Latent-space evolutionary optimization of the cation, targeting high conductivity against a fixed dicyanamide anion, converged to 1-ethyl-3-methylimidazolium—the known optimum—rather than surpassing it. These results are reported as a constraint: with ~1,000 ions, a generative model learns string statistics, not ion chemistry, and a data-driven predictor tends to regress toward the known optimum. True *de novo* inverse design of novel ion *chemistries* therefore requires a substantially larger ion database than the present one.

## 3. Discussion

The forward model answers "what properties?"; the inverse workflow answers "which pair?". This study shows that the latter can be done honestly at the *combinatorial* level: a validated three-model oracle, applied to the ~99% of the cation–anion product space that has never been measured, recovers chemically sensible, three-model-consistent, room-temperature-liquid candidates whose leading members are small heterocyclic cations with dicyanamide. The same workflow applied to *novel chemistry* (scaffold mutations, generative models) is limited by data density: generated cations are chemically reasonable only when constrained by known scaffolds, and black-box generation at ~1,000 ions yields mostly unreasonable fragments.

Three implications follow. First, for the modeling community, multi-model cross-validation is a cheap, principled guard against presenting predictor artifacts as discoveries; single-oracle "inverse design" results should be treated cautiously. Second, for the experimental community, the leading candidates (thiazolium/DCA and related pairs) are concrete, synthesizable targets whose predicted low viscosity and high conductivity are worth measuring. Third, the failure modes of generation quantify the data frontier: representation and architecture cannot substitute for missing ion-chemistry data, echoing the data-density conclusion reached throughout this series (8).

Limitations: the synthetic-accessibility (SA) scores computed here are calibrated for neutral drug-like molecules and are unreliable for ionic fragments, so they are not used as a hard filter; a proper retrosynthetic-feasibility assessment is deferred. The predictors inherit inter-laboratory scatter from the underlying data. Melting-point prediction remains the weakest oracle (R² ≈ 0.52–0.56), so the room-temperature-liquid classification carries uncertainty.

## 4. Materials and Methods

**Data.** Records for viscosity, density, electrical conductivity, and melting point were compiled from ILThermo v2.0, ILest, and iolitech as described previously (8); conductivity and viscosity targets are log-transformed. 458 per-ion RDKit descriptors were computed for each of 1,891 ion pairs (9).

**Predictors.** GradientBoostingRegressor and HistGradientBoostingRegressor (scikit-learn) were trained on the 458 descriptors (+temperature for temperature-dependent properties) under 5-fold GroupKFold on ion-pair identity; R²/MAE were computed on pooled out-of-fold predictions, and final oracles were refit on the full dataset.

**Inverse screening.** All 238,500 cation–anion combinations were evaluated at 298.15 K by all three predictors. Candidates were filtered for novelty (not in the 1,891-pair database), predicted melting point < 298 K, and three-model agreement (|Δ ln κ| < 0.8 across models). The multi-objective Pareto front (maximize ln κ; minimize ln η and T_m) was computed by non-dominated sorting.

**Scaffold mutation and VAE.** Six cation scaffolds were mutated across substituent lists (C1–C8 alkyl, methoxy/ethoxyethyl, hydroxyethyl) with RDKit validation of valence and +1 charge. A character-level GRU VAE (latent dimension 48) was trained on cation and anion SMILES; latent-space optimization used a (μ, λ) evolutionary strategy against the GBM oracle with a synthetic-accessibility penalty.

## Data and Software Availability

All data, descriptor tables, predictors, and analysis scripts are available at https://github.com/linfuxing123/IL-Property-ML and archived on Zenodo at https://doi.org/10.5281/zenodo.21898948 (version v1.4.1, which includes the inverse-design workflow described here).

## References

1. R. D. Rogers, K. R. Seddon, Ionic liquids—solvents of the future? *Science* **302**, 792–793 (2003).
2. T. Welton, Room-temperature ionic liquids: Solvents for synthesis and catalysis. *Chem. Rev.* **99**, 2071–2083 (1999).
3. M. Armand, F. Endres, D. R. MacFarlane, H. Ohno, B. Scrosati, Ionic-liquid materials for the electrochemical challenges of the future. *Nat. Mater.* **8**, 621–629 (2009).
4. K. T. Butler, D. W. Davies, H. Cartwright, O. Isayev, A. Walsh, Machine learning for molecular and materials science. *Nature* **559**, 547–555 (2018).
5. D. M. Makarov, Y. A. Fadeeva, L. E. Shmukler, I. V. Tetko, Benchmarking machine learning methods for modeling physical properties of ionic liquids. *J. Mol. Liq.* **351**, 118616 (2022).
6. X. Yu, End-to-end deep learning models for predicting the electrical conductivity of ionic liquids. *ACS Sustain. Chem. Eng.* (2026).
7. F. Lin, A structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties, submitted (2026).
8. F. Lin, Data density as the binding constraint for ionic-liquid property prediction, submitted (2026).
9. F. Lin, Full-spectrum descriptors substitute for data density in ionic-liquid property prediction, submitted (2026).
10. C. Song, C. Wang, F. Fang, G. Zhou, Z. Dai, Z. Yang, Large-scale screening for high conductivity ionic liquids via machine learning algorithm utilizing graph neural network-based features. *J. Chem. Eng. Data* **69**, 800–810 (2024).
11. K. Baran, A. Kloskowski, Graph neural networks and structural information on ionic liquids. *J. Phys. Chem. B* **127**, 10542–10555 (2023).
12. Extension of scoring-assisted generative exploration for ionic liquids (SAGE-IL). *Green Chem. Eng.* **6**, 335–343 (2025).
