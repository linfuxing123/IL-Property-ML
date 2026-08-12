# Data density as the binding constraint: a 7.7-fold expansion of ionic-liquid property data lifts group-disjoint prediction from cold start to transferable accuracy

**One-sentence summary:** Expanding the ionic-liquid property database 7.7-fold converts cold-start viscosity prediction from R² = −0.09 to 0.68 under strict IL-disjoint validation.

**Authors:** Fuxing Lin\* (corresponding author; ORCID: 0009-0003-7588-6942)

**Affiliation:** Hunan Institute of Engineering

**Corresponding author:** Fuxing Lin; ORCID: 0009-0003-7588-6942; email: 3612411485@qq.com

---

## Abstract

Machine-learning prediction of ionic-liquid (IL) properties for previously unseen ion pairs is a key enabler of high-throughput electrolyte design, but extrapolation accuracy is limited by data density rather than model capacity (1). Here I test this hypothesis directly by expanding a curated IL property database 7.7-fold—from 11,511 to 88,077 experimental records spanning 1,891 unique ion pairs—through systematic harvesting of the NIST ILThermo v2.0 repository with standardized units and verified SMILES (2). Under strict IL-disjoint 5-fold cross-validation with identical gradient-boosting models and open RDKit descriptors, viscosity (ln η) improves from R² = −0.09 to 0.68, electrical conductivity (ln κ) from 0.55 to 0.70, density from 0.83 to 0.85, and melting point from ≈0 to 0.39 with 642 ILs. Point-wise validation continues to inflate R² by 0.09–0.15, showing that honest group-level evaluation remains mandatory at any data scale. The results establish a quantitative relationship between per-IL coverage and group-level transferability and provide the largest openly available multi-property IL dataset with verified chemistry.

**Keywords:** ionic liquids; data density; group-disjoint validation; QSPR; ILThermo; leakage

---

## 1. Introduction

Ionic liquids—salts that are liquid below ~373 K—offer a combinatorially vast design space, with roughly 10⁶ cation–anion pairs synthetically accessible but only a few thousand characterized experimentally (3). Reliable property prediction for unseen ion pairs is therefore central to electrolyte screening, yet conventional point-wise (random) train/test splits overstate extrapolation accuracy because records of the same IL leak across the split boundary (4). Our companion study (1) introduced an IL-disjoint validation protocol and showed on 11,511 records that (i) point-wise validation inflates conductivity R² from 0.552 to 0.908, and (ii) the properties that fail under IL-disjoint splits—viscosity and melting point, with roughly one record per IL—are precisely those with the lowest per-IL data density. This suggested a sharp hypothesis: *data density, not model capacity, is the binding constraint on group-level extrapolation.*

Testing that hypothesis requires a controlled experiment: keep the model, features, and evaluation protocol fixed, and increase per-IL data coverage. Here I assemble 88,077 records for 1,891 unique ILs by systematically harvesting the NIST ILThermo v2.0 repository (2), the most comprehensive curated database of IL thermophysical properties, and re-run the identical honest-evaluation pipeline. Three questions frame the analysis: (i) Does multi-temperature coverage convert cold-start properties into predictable ones? (ii) Does the leakage penalty persist at scale? (iii) Which property remains data-limited, and what kind of data would close the gap?

![Figure 1](figures/fig1_groupR2_comparison.png)

**Fig. 1. Data density converts cold-start IL properties into predictable ones.** Group-disjoint 5-fold R² (GBM, 10 RDKit features + temperature; log-transformed targets for viscosity and conductivity) for the previous dataset (11,511 records, gray) and the expanded dataset (88,077 records, blue). Horizontal ticks mark point-wise R². Viscosity rises from −0.09 to 0.68; conductivity from 0.55 to 0.70; density from 0.83 to 0.85; melting point from ≈0 to 0.39.

## 2. Results

### 2.1. A 7.7-fold expansion with verified chemistry

ILThermo v2.0 (NIST) curates experimental data with reference, method, phase, and—for 87–90% of compounds—manually verified SMILES (2). Using the ilthermopy client, I enumerated all single-component entries for four properties (viscosity, density, electrical conductivity, and normal melting temperature), downloaded 8,893 entries (10,182 enumerated; 1,256 lacked SMILES; 33 contained no parseable data), and standardized every record to a common schema: temperature in kelvin, viscosity in mPa·s, density in g/cm³, conductivity in S/m, with cation/anion SMILES and source reference preserved (Table 1).

**Table 1. Dataset scale before and after expansion.**

| Property | Previous records | Previous ILs | Expanded records | Expanded ILs | Multi-temperature ILs |
|----------|-----------------|--------------|------------------|--------------|----------------------|
| Viscosity | 122 | 121 | 25,260 | 1,213 | 84% |
| Density | 102 | 101 | 48,428 | 1,433 | 78% |
| Conductivity | 11,181 | 1,560 | 9,470 | 641 | 77% |
| Melting point | 106 | 106 | 919 | 642 | — |
| **Total** | **11,511** | **1,658** | **88,077** | **1,891 unique (3,929 prop–IL)** | — |

The expansion was most dramatic for the previously data-starved properties: viscosity grew 207-fold in records and 10-fold in IL count; density grew 475-fold in records. Conductivity, already well covered previously, was re-curated from the authoritative source, which also served as a unit and value cross-check (Section 2.4).

### 2.2. Data density converts cold-start properties into predictable ones

All models are gradient-boosting regressors with ten RDKit descriptors of the ion pair (molecular weight, log P, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic rings, heavy-atom count, fraction of Csp³, ring count) plus temperature, evaluated by 5-fold GroupKFold on IL identity. Figure 1 and Table 2 show the central result: under identical IL-disjoint validation, all three temperature-dependent properties are now predictable. Viscosity, the flagship cold-start failure of the companion study (1), reaches R² = 0.68; conductivity reaches 0.70 with 641 ILs—fewer than the previous 1,560 because the previous dataset mixed conductivity units (Section 2.4); density, already regular, improves slightly to 0.85.

**Table 2. Group-disjoint 5-fold R² (GBM, 10 RDKit features + T).**

| Property | Previous R² | Expanded R² | Point-wise R² | ΔR² (inflation) |
|----------|-------------|-------------|---------------|------------------|
| Viscosity (ln η) | −0.09 | **0.68** | 0.79 | +0.12 |
| Density | 0.83 | **0.85** | 0.94 | +0.09 |
| Conductivity (ln κ) | 0.55 | **0.70** | 0.84 | +0.15 |
| Melting point (K) | ≈0 | **0.39** | — | — |

Figure 2 places these four points in the coverage–transferability plane, using the median records per IL as the coverage axis. The pattern is monotone within each property family: increasing per-IL coverage lifts group-level R², with density the notable exception—it was already regular at one record per IL because density is strongly composition-dominated.

![Figure 2](figures/fig2_coverage_transferability.png)

**Fig. 2. Per-IL coverage versus group-disjoint R².** Each marker is a property at a given data stage (open: previous dataset; filled: expanded dataset). Viscosity and conductivity move right and upward as median records per IL rise from 1 to 9; melting point moves upward through IL diversity (642 ILs) despite remaining single-valued per IL; density is coverage-insensitive because composition dominates.

### 2.3. The leakage penalty persists at scale

Point-wise validation continues to overstate performance after the expansion, with inflation ΔR² = +0.09 to +0.15 and 66–70% of test records sharing an IL with training data. This is a cautionary result: larger datasets do not by themselves discipline evaluation. Group-disjoint splitting must remain the reporting standard as IL databases grow (4).

### 2.4. Data quality audit: unit inconsistency in legacy conductivity

Cross-comparing the previous dataset against ILThermo on the 111 shared conductivity ILs, values agreed exactly in most cases but differed by a factor of ~10 in others, revealing that the previous dataset mixed S/m and mS/cm units. All expanded records were therefore taken from the single authoritative source (ILThermo, S/m). Legacy records were retained only for ILs absent from ILThermo, after temperature-unit calibration: melting points in °C were converted to K (cross-source offset 273.4 ± 1.7 K over 34 shared ILs, confirming the conversion); viscosity (48 ILs) and density (37 ILs) legacy records were converted from °C to K. Legacy conductivity records were excluded from modeling. This audit is itself a methodological contribution: unit inconsistency is an invisible source of model degradation in aggregated IL datasets.

### 2.5. Melting point: a structurally different cold start

Melting temperature is single-valued per IL—there is no multi-temperature analog—so the relevant coverage dimension is the number of distinct ILs. Expanding from 106 to 642 ILs with structure-only features yields R² = 0.39, a meaningful first step for a property long considered difficult to predict (5). We estimate that several thousand ILs spanning diverse cation families will be required for design-grade accuracy; melting-point acquisition should prioritize chemical diversity over replication.

## 3. Discussion

This study provides the first direct, controlled test of the data-density hypothesis in IL property prediction. The pattern is unambiguous: properties with ~1 record per IL were unpredictable; properties with ~9 records per IL across 77–84% of ILs are predictable at group level. Viscosity—the flagship cold-start failure—rose from −0.09 to 0.68 with no architectural change, isolating data coverage as the causal factor.

Three implications follow. First, for the modeling community, database curation is a first-class activity: the same model, features, and evaluation that produced R² = 0.55 on sparse data produced R² = 0.70 on dense, quality-controlled data. Second, for the experimental community, the marginal value of a new measurement is highest for single-sample groups in chemically diverse regions; our coverage planner identifies those ILs explicitly. Third, the persistence of leakage inflation at scale argues that performance claims must be tied to the split protocol, not to dataset size.

Limitations: ILThermo, while curated, inherits literature scatter; inter-laboratory consistency filtering was not applied. Density, although highly predictable, is dominated by a few cation families and may overstate chemical diversity. Melting-point prediction remains below design grade and awaits broader IL coverage.

## 4. Materials and Methods

**Data acquisition.** All records were retrieved from ILThermo v2.0 (https://ilthermo.boulder.nist.gov/) via ilthermopy v1.1.2, restricted to single-component entries (`n_compounds = 1`). Enumeration used the search API with property keys for viscosity (`tplC`), density (`jBwV`), electrical conductivity (`LCor`), and normal melting temperature (`LPuZ`); data were downloaded entry-by-entry with a 48-worker concurrent client, retry logic, and resumable state stored in SQLite. Unit standardization parsed ILThermo column headers (temperature, pressure, value) with HTML-entity and `<SUP>` cleanup; °C→K, Pa·s→mPa·s, kg/m³→g/cm³, and mS/cm→S/m conversions were applied.

**SMILES handling.** ILThermo provides manually verified SMILES for 87–90% of compounds; cation and anion fragments were split on charged fragments. Entries without SMILES (1,256) were attempted with a name-based resolver (29% success after supplementing the anion table with fatty-acid carboxylates, dimethyl phosphate, fluorosulfonyl amides, and cholinium/guanidinium cations) and excluded otherwise.

**Legacy merge.** Legacy records (companion study (1)) were merged only for ILs absent from ILThermo: viscosity (48 ILs) and density (37 ILs) with temperature converted from °C to K; melting points (72 ILs) with °C→K conversion verified against 34 shared ILs. Legacy conductivity records were excluded due to mixed units.

**Descriptors.** Ten RDKit descriptors were computed for the combined cation–anion SMILES: molecular weight, Crippen log P, topological polar surface area, H-bond donor and acceptor counts, rotatable-bond count, aromatic-ring count, heavy-atom count, fraction of Csp³ carbons, and total ring count.

**Models and validation.** GradientBoostingRegressor (scikit-learn defaults, seed 0) with 5-fold GroupKFold on the canonical cation|anion key. Point-wise controls used 5-fold KFold. Targets for viscosity and conductivity were log-transformed (ln η, ln κ). Reported metrics: R², RMSE, MAE; leakage rate; inflation ΔR² = R²(point) − R²(group).

## Data and code availability

All data (88,077 records, standardized units, verified SMILES), scripts (`ilthermo_fetch.py`, `ilt_validate.py`, `ilt_merge_old.py`), honest-evaluation tools (`honest_cv.py`, `coverage_planner.py`), and per-fold results are openly available at https://github.com/linfuxing123/IL-Property-ML (release v1.1.0) and archived at Zenodo (concept DOI 10.5281/zenodo.21898948; the latest version contains the companion dataset). ILThermo v2.0 is the primary data source (2).

## References

1. F. Lin, *A unified structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties*, submitted (2026); preprint: ChemRxiv (2026). Companion study: methodology, previous dataset, and evaluation tools.
2. ILThermo v2.0, National Institute of Standards and Technology, https://ilthermo.boulder.nist.gov/.
3. K. R. Seddon, Ionic liquids for clean technology, *J. Chem. Technol. Biotechnol.* **68**, 351 (1997).
4. S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, *Patterns* **4**, 100804 (2023).
5. D. M. Eike, J. F. Brennecke, E. J. Maginn, Predicting melting points of quaternary ammonium ionic liquids, *Green Chem.* **5**, 323 (2003).

---

*Draft v0.2 — Science-format companion manuscript. 2026-08-12.*
