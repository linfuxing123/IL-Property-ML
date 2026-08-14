# Supplementary Materials

From property prediction to inverse design: multi-property Pareto screening of ionic-liquid electrolytes under IL-disjoint validation

## Table S1. Three-model oracle (IL-disjoint 5-fold R²)

| Property | GBM R² | HistGBM R² | MPNN R² |
|----------|--------|------------|---------|
| Conductivity (ln κ) | 0.740 | 0.749 | 0.703 |
| Density | 0.926 | 0.949 | 0.933 |
| Viscosity (ln η) | 0.809 | 0.846 | 0.759 |
| Melting point (K) | 0.523 | 0.561 | 0.365 |

## Table S2. Final unreported candidates (three-model consistent, T_m < 298 K)

| # | Cation | Anion | ln κ (G/H/N) | ln η (G) | T_m (G) | ρ (G) | |Δlnκ(GBM−GNN)| | SA |
|---|--------|-------|---------------|----------|---------|--------|----------------|----|
| 1 | CC[n+]1ccsc1 | N#C[N-]C#N | 0.640/0.280/0.434 | 3.175 | 274.7 | 1.175 | 0.206 | 9.7 |
| 2 | CC[n+]1ccn(C)c1 | N#C[N-]C#N.N#C[N-]C#N | 0.630/0.715/0.734 | 3.585 | 272.7 | 1.183 | 0.104 | 9.7 |
| 3 | CCn1cc[n+](C)c1 | N#C[N-]C#N.N#C[N-]C#N | 0.588/0.634/0.760 | 3.743 | 270.0 | 1.183 | 0.171 | 9.3 |
| 4 | CCn1cc[n+](CC)c1 | N#C[N-]C#N | 0.511/0.493/0.326 | 3.112 | 267.0 | 1.111 | 0.185 | 9.4 |
| 5 | CC[n+]1cc[nH]c1 | N#C[N-]C#N | 0.386/0.383/0.501 | 3.050 | 260.2 | 1.158 | 0.116 | 9.6 |
| 6 | CCn1cn[n+](C)c1 | N#C[N-]C#N | 0.232/0.225/0.430 | 3.142 | 260.1 | 1.139 | 0.197 | 10.3 |
| 7 | C[n+]1cn(N)cn1 | N#C[N-]C#N | 0.138/-0.268/0.332 | 2.988 | 273.4 | 1.212 | 0.194 | 10.8 |
| 8 | CC[n+]1cc[nH]c1 | N#C[B-](C#N)(C#N)C#N | 0.018/-0.070/0.327 | 2.973 | 289.2 | 1.117 | 0.310 | 9.0 |
| 9 | CC[S+](C)CC | O=S(=O)([O-])C(F)(F)F.O=S(=O)([O-])C(F)(F)F | -0.954/-0.513/-1.333 | 4.553 | 237.5 | 1.487 | 0.379 | 7.0 |
| 10 | CC[S+](CC)CC | O=S(=O)([O-])C(F)(F)F.O=S(=O)([O-])C(F)(F)F | -0.999/-0.487/-1.323 | 4.475 | 240.7 | 1.468 | 0.324 | 7.0 |
| 11 | CCOCn1cc[n+](C)c1 | O=S(=O)([O-])C(F)(F)F.O=S(=O)([O-])C(F)(F)F | -1.227/-0.895/-1.039 | 4.427 | 243.9 | 1.526 | 0.187 | 6.3 |

## Table S3. Scaffold-mutation cation inventory (by core)

| Core | Count |
|------|-------|
| Imidazolium | 96 |
| Pyrrolidinium | 96 |
| Sulfonium | 96 |
| Pyridinium | 11 |
| Ammonium | 11 |
| Phosphonium | 11 |
| **Total (novel, charge +1, valid)** | **272** |

## Methods supplement

**Predictors.** GBM (GradientBoostingRegressor, default) and HistGBM
(HistGradientBoostingRegressor, max_iter=300) on 458 per-ion RDKit descriptors
(+ temperature for temperature-dependent properties); 5-fold GroupKFold on ion-pair
identity; final oracles refit on the full 84,077-record dataset.

**Combinatorial screening.** 795 canonical cations × 300 canonical anions = 238,500
pairs, evaluated at 298.15 K by both models. Filters: novelty (not in the 1,891-pair
database), predicted T_m < 298 K, |Δ ln κ| < 0.6, |ΔT_m| < 15 K. Multi-objective Pareto
front (maximize ln κ; minimize ln η and T_m) by non-dominated sorting.

**Scaffold mutation.** Six cation cores (imidazolium, pyridinium, pyrrolidinium,
ammonium, phosphonium, sulfonium) mutated across C1–C8 alkyl, methoxy/ethoxyethyl,
and hydroxyethyl substituents; RDKit-validated for valence and +1 charge;
novelty enforced against the 867 known cations.

**VAE.** Character-level GRU VAE (latent dim 48, max length 80) trained on 860 cations
and 356 anions; validity ~10%, correct-charge fraction ~60%.

**Latent optimization.** (μ, λ) evolutionary strategy in the 48-d latent space against
the GBM oracle with an SA penalty; converged to 1-ethyl-3-methylimidazolium (the known
optimum).
