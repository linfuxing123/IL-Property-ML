# Supplementary Materials for

## Data density as the binding constraint: a 7.7-fold expansion of ionic-liquid property data lifts group-disjoint prediction from cold start to transferable accuracy

Fuxing Lin, Hunan Institute of Engineering

---

## S1. Data acquisition details

All records were retrieved from ILThermo v2.0 (National Institute of Standards and Technology, https://ilthermo.boulder.nist.gov/) using the ilthermopy v1.1.2 client. Enumeration used the search API restricted to single-component entries (`n_compounds = 1`) with the following property keys: viscosity (`tplC`), density (`jBwV`), electrical conductivity (`LCor`), and normal melting temperature (`LPuZ`). Data entries were downloaded with a 48-worker concurrent client with retry logic (3 attempts, backoff) and resumable state stored in SQLite, so interrupted runs resume without data loss.

**Table S1. Download statistics.**

| Property | Entries enumerated | Entries downloaded | Entries without SMILES | Records parsed |
|----------|-------------------|--------------------|------------------------|----------------|
| Viscosity | 3,054 | 2,653 | 401 | 26,533 |
| Density | 4,706 | 4,205 | 501 | 51,369 |
| Conductivity | 1,300 | 1,136 | 164 | 9,860 |
| Melting point | 1,122 | 899 | 223 | 899 |
| **Total** | **10,182** | **8,893** | **1,256** | **88,661** |

† 1,256 entries lacked SMILES; 33 additional entries contained no parseable data (10,182 − 8,893 = 1,289).

## S2. Dataset statistics

**Table S2. Model-ready dataset scale (after filtering and deduplication).**

| Property | Records | Unique ILs | Multi-temperature ILs | Temperature range (K) |
|----------|---------|-----------|-----------------------|-----------------------|
| Viscosity | 25,260 | 1,213 | 84% | 214–573 |
| Density | 48,428 | 1,433 | 78% | 90–571 |
| Conductivity | 9,470 | 641 | 77% | 197–484 |
| Melting point | 919 | 642 | — | — |

Cation family distribution (by number of ILs): imidazolium dominates (42–47% depending on property), followed by pyridinium, phosphonium, pyrrolidinium, ammonium, piperidinium, guanidinium, and cholinium derivatives; "other" includes sulfonium, morpholinium, and functionalized variants. Anions are dominated by bis(trifluoromethanesulfonyl)imide, tetrafluoroborate, hexafluorophosphate, dicyanamide, halides, alkyl sulfates, and carboxylates.

## S3. Per-fold validation results

All models: `GradientBoostingRegressor` (scikit-learn defaults, `random_state=0`), 10 RDKit descriptors plus temperature (except melting point, structure-only), 5-fold GroupKFold (group) or KFold (point). Targets log-transformed for viscosity and conductivity.

**Table S3. Per-fold R² (group-disjoint, unseen ILs in test folds).**

| Property | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|----------|--------|--------|--------|--------|--------|-----------|
| Viscosity (ln η) | 0.748 | 0.630 | 0.642 | 0.645 | 0.667 | 0.666 ± 0.047 |
| Density | 0.870 | 0.882 | 0.755 | 0.872 | 0.834 | 0.843 ± 0.052 |
| Conductivity (ln κ) | 0.787 | 0.797 | 0.576 | 0.703 | 0.623 | 0.697 ± 0.098 |
| Melting point (K) | 0.404 | 0.420 | 0.346 | 0.147 | 0.556 | 0.375 ± 0.149 |

**Table S4. Per-fold R² (point-wise, leakage allowed).**

| Property | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|----------|--------|--------|--------|--------|--------|-----------|
| Viscosity (ln η) | 0.796 | 0.790 | 0.804 | 0.802 | 0.776 | 0.794 ± 0.011 |
| Density | 0.935 | 0.934 | 0.936 | 0.937 | 0.935 | 0.936 ± 0.001 |
| Conductivity (ln κ) | 0.832 | 0.853 | 0.844 | 0.856 | 0.834 | 0.844 ± 0.011 |
| Melting point (K) | 0.419 | 0.488 | 0.276 | 0.581 | 0.536 | 0.460 ± 0.119 |

Leakage rates (fraction of test records whose IL also appears in training under point-wise splitting): viscosity 70%, density 67%, conductivity 67%, melting point 0% (single record per IL).

## S4. Unit standardization

ILThermo column headers encode property, unit, and phase (e.g., "Viscosity, Pa&#8226;s => Liquid"); HTML entities and `<SUP>` tags were cleaned before parsing. Conversions applied: temperature °C→K (add 273.15); viscosity Pa·s→mPa·s (multiply 1,000); density kg/m³→g/cm³ (divide 1,000); conductivity mS/cm→S/m (multiply 0.1), µS/cm→S/m (multiply 10⁻⁴). Legacy records (companion study) were merged only for ILs absent from ILThermo: viscosity (48 ILs) and density (37 ILs) with temperature converted from °C to K; melting points (72 ILs) with °C→K conversion verified against 34 shared ILs (offset 273.4 ± 1.7 K). Legacy conductivity records were excluded from modeling because unit consistency could not be guaranteed (see main text, Section 2.4).

## S5. Handling of entries without SMILES

1,256 entries lacked ILThermo SMILES. These were attempted with a name-based resolver (`ilthermo_resolver.py`) that maps anion names to SMILES and parses cation families (imidazolium, pyridinium, pyrrolidinium, piperidinium, ammonium, phosphonium, cholinium, guanidinium) with alkyl-chain expansion; candidates were accepted only if RDKit parsed the combined SMILES. After supplementing the anion table (fatty-acid carboxylates, dimethyl phosphate, fluorosulfonyl amides, 1,2,4-triazolate), 29% of the remaining names were resolved; the rest were excluded. The 87–90% of entries with official SMILES carry no resolver uncertainty.

## S6. Coverage planner targets

For viscosity, density, and conductivity the expanded dataset reaches a median of 9 records per IL (target ≥3–5), with single-sample-group fractions of 16–24% and cold-start scores of 11–16/100 (low). For melting point, the binding constraint is IL diversity rather than per-IL replication: 642 ILs yield group-level R² = 0.39; acquiring several thousand diverse ILs (prioritizing non-imidazolium families) is the identified path to design-grade accuracy.
