# Supplementary Materials

**From 860 to 219,292 cations: how data scale unlocks generative design of ionic liquids**

Fuxing Lin, Hunan Institute of Engineering

---

## Table S1. Full final candidate set (three-predictor agreement, novel, chemically valid)

All candidates pair the listed cation with the dicyanamide (DCA) anion,
`N#C[N-]C#N`. Predictions at T = 298.15 K. "Novel" = canonicalized IL string not
present in the 6,177-IL experimental training set. Chemical sanity: single-fragment
anion, net formal charge −1.

| Cation (canonical SMILES) | GBM ln κ | HistGBM ln κ | MPNN ln κ | Tm (K) | ln η (mPa·s) |
|---|---|---|---|---|---|
| CCn1cc[n+](CC)c1 (1,3-diethylimidazolium) | 0.511 | 0.427 | 0.463 | 269.6 | 3.05 |
| C=C[n+]1ccn(C)c1 (1-vinyl-3-methylimidazolium) | 0.461 | 0.688 | 0.751 | 281.5 | 3.28 |
| CC[n+]1cccnc1 (1-ethylpyridinium) | 0.417 | 0.609 | 0.436 | 284.1 | 3.41 |
| CC[n+]1cc[nH]c1 (1-ethylimidazolium) | 0.386 | 0.359 | 0.712 | 277.3 | 3.20 |
| CCC[n+]1ccn(C)c1 (1-methyl-3-propylimidazolium) | 0.382 | 0.481 | 0.510 | 267.9 | 3.38 |
| CC(C)n1cc[n+](C)c1 (1-isopropyl-3-methylimidazolium) | 0.308 | 0.410 | 0.550 | 286.2 | 3.78 |
| CC[n+]1ccc(C)cc1 (1-ethyl-4-methylpyridinium) | 0.303 | 0.353 | 0.551 | 297.8 | 3.61 |

Selection pipeline: 2,838 cations × 592 anions = 1,680,096 novel pairs →
GBM/HistGBM agreed ln κ > 0.2 → Tm ∈ (180, 300) K → ln η < 5.5 → canonicalized
novelty vs 6,177 known ILs → chemical sanity → MPNN agreement > 0.3. 85 raw
survivors → 47 chemically valid → 16 triple-consistent → 7 unique cations after
SMILES canonicalization.

## Table S2. Per-fold IL-disjoint R² (GBM, 5-fold)

| Property | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|---|
| conductivity | — | — | — | — | — | 0.740 |
| density | — | — | — | — | — | 0.926 |
| viscosity | — | — | — | — | — | 0.791 |
| melting_point | — | — | — | — | — | 0.620 |

(Per-fold values available in the code repository; mean reported in main text.)

## Table S3. Data asset inventory

| File | Description | Size |
|---|---|---|
| data/mendeley_normalized.csv | Mendeley tpp25ztzmb normalized (viscosity/melting/CO2/toxicity) | 6.8 MB |
| data/il_descriptors_v2.csv | 6,177 ILs × 459 RDKit descriptors | 22 MB |
| data/ilpe_props_canon.csv | ILPE 8.33M ILs × 12 ML properties (canonicalized) | 1.2 GB |
| data/ilbert_props.csv | ILBERT 8.33M ILs × 12 chemLM properties | 1.1 GB |
| data/ions/cations_83m.csv | 219,292 unique cations | 9.5 MB |
| data/ions/anions_83m.csv | 38 unique anions | <0.1 MB |
| data/pretrain_train_set.txt | 30.2M unlabeled IL-like SMILES | 1.6 GB |

## Table S4. Generator validation details

- SMILES LM: GRU (emb 128, hidden 256, 2 layers), teacher forcing, Adam lr 1e-3,
  25 epochs, batch 512; sampling temperature 0.90–0.95, top-k 15–20.
- Validity check: RDKit `MolFromSmiles`; charge check: `GetFormalCharge == +1`.
- 500,000 samples → 499,388 valid (99.9%), 499,361 charge +1 (99.9%),
  1,766 novel cations (not in the 219k virtual library).

## Table S5. Cross-predictor conductivity agreement (Fig. 3 data)

| Quantity | Value |
|---|---|
| Overlapping ILs (both sources, ILPE κ > 0) | 633,780 |
| After excluding 6,177 known experimental ILs | 633,733 |
| Pearson r (ILPE ln κ vs ILBERT ln κ) | 0.20 |
| ILs with both sources ln κ > 0 | **0** |
| ILPE ln κ max | −0.24 |
| ILBERT ln κ max | +0.36 |
| Known experimental optimum (EMIM-DCA) ln κ | ≈ +0.64 |

## Table S6. Mendeley dataset contribution (per property)

| Property | Points | Unique ILs | Source sheet |
|---|---|---|---|
| viscosity (ln mPa·s) | 10,718 | 1,877 | vis_clean (LOG10 → ×ln10) |
| melting_point (K) | 4,742 | 3,593 | mp_clean |
| CO₂ solubility (mole fraction) | 24,067 | 337 | xco2_clean |
| toxicity (logEC50) | 401 | 331 | tox_clean |

## Table S7. Viscosity cold-start decomposition (Section 2.2)

| Subset | Points | ILs | IL-disjoint R² |
|---|---|---|---|
| Full expanded set | 34,191 | 2,705 | 0.791 |
| Paper-5 ILs only | 27,712 | 1,213 | 0.808 |
| New (Mendeley-only) ILs | 6,479 | 1,492 | 0.728 |
| — of which single-point ILs | — | 956 / 1,492 (64%) | — |

The slight decrease in overall viscosity R² (0.809 → 0.791) is fully explained by
the cold-start fraction: 64% of newly added ILs have a single temperature point and
are therefore never seen in IL-disjoint training. The model nevertheless reaches
R² = 0.728 on these unseen ILs — evidence of genuine extrapolation, not data
quality degradation.
