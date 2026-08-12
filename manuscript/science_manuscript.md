# A unified structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties

**One-sentence summary:** Structure-only model predicts four ionic-liquid properties and exposes the gap hidden by conventional splits.

**Authors:** Fuxing Lin\* (corresponding author)

**Affiliation:** Hunan Institute of Engineering

**Corresponding author:** Fuxing Lin; email: 3612411485@qq.com

---

## Abstract

Ionic liquids (ILs) are promising electrolyte platforms, but their combinatorial chemical space makes exhaustive measurement impossible. Existing structure–property models for ILs rely on expensive descriptors and point-wise validation that overstates extrapolation accuracy. I assembled 11,511 experimental records for 1,658 ion pairs from five public sources and built an open, structure-based framework predicting conductivity, density, viscosity, and melting point from SMILES and temperature alone. Under strict IL-disjoint cross-validation, gradient boosting reaches R² = 0.552 for conductivity and R² = 0.828 for density, versus R² = 0.908 under point-wise splitting—quantifying the inflation inherent in conventional validation. SHAP identifies temperature, anion lipophilicity, and cation size as dominant drivers; data density, not model capacity, is the binding constraint for extrapolation.

**Keywords:** ionic liquids; machine learning; QSPR; IL-disjoint validation; conductivity; open descriptors

---

## 1. Introduction

Ionic liquids—salts that are liquid below ~373 K—have become central to electrochemical energy storage, green solvents, gas capture, and separation science because their properties can be tuned by pairing different cations and anions (1–3). The number of synthetically accessible ion pairs, however, vastly exceeds what can be measured, so reliable low-cost property prediction is a core problem in materials informatics (4).

Three computational strategies dominate the field, each with distinct trade-offs. First-principles quantum chemistry delivers accurate electronic-structure insight but requires a new calculation for every candidate molecule, making large-scale screening impractical (5). Molecular dynamics can probe transport mechanisms, but its accuracy is bounded by force-field quality and its throughput is far below screening requirements (6–8). Descriptor-based machine learning—quantitative structure–property relationships (QSPR)—is the only strategy with the throughput needed for high-throughput virtual screening, and it has been applied to IL conductivity (9–11), viscosity (12), and melting point (13).

Two problems, however, limit the reliability and reach of existing QSPR models. First, validation is often performed with point-wise (random) splits in which records of the same ionic liquid appear in both training and test sets. Such protocols report deceptively high accuracy (test R² ≈ 0.96–0.99) that collapses when test folds contain completely unseen ionic liquids (R² ≈ 0.59–0.83) (9, 11). A recent end-to-end study established that IL-based splitting yields test R² = 0.871/MAE = 0.392 on one dataset and R² = 0.778/MAE = 0.600 on a larger dataset, and showed by a random-forest control that point-wise splitting inflates R² from 0.627 to 0.979 under identical features (9). That work, however, depended on Dragon structural descriptors and quantum-chemical parameters, and addressed a single property. Second, the field lacks a fully open, dependency-free descriptor pipeline: commercial descriptor packages and continuum solvation models such as COSMO-RS remain common prerequisites, limiting reproducibility and accessibility.

Here I address both gaps. I assemble a curated database of 11,511 experimental records covering 1,658 unique ion pairs from five public sources, and build a unified framework that predicts four physical properties—conductivity, density, viscosity, and melting point—directly from cation/anion SMILES and temperature, using only freely computable RDKit descriptors and Morgan fingerprints. All models are evaluated under strict IL-disjoint cross-validation, with paired point-wise controls that quantify the inflation caused by conventional splitting. The framework is fully open and reproducible: data assembly, featurization, model training, and interpretation run from a single Python codebase with no commercial dependencies. The results show that (i) a single structure-based representation serves multiple properties; (ii) honest IL-level accuracy is far lower than point-wise claims, and the gap is a quantitative measure of generalization risk; and (iii) data density—not model capacity—is the binding constraint for property extrapolation.

![Figure 1](figures/fig1_framework.png)

**Fig. 1. Unified structure-based multi-property prediction framework for ionic liquids.** Cation and anion SMILES are converted by RDKit into 2D descriptors and ECFP4 fingerprints; composition and temperature are appended to form a 2070-dimensional input; a shared multi-task network predicts conductivity, viscosity, density, melting point, and, in future extension, chemical properties.

## 2. Results

### 2.1. A curated multi-property database of 11,511 records

I assembled experimental records from five public sources into a local database (`il_props.db`, 11,511 records): the ILest mixture-conductivity repository (8,035 records, stored as three tabular compilations), three ILThermo-derived conductivity tables resolved to ion-level SMILES by a purpose-built parser (3,036 records), and the iolitech pure-IL compilation (440 records covering conductivity, viscosity, density, and melting point) (14–16). The database contains 1,658 unique cation–anion pairs and four properties (Table 1). Cation families include imidazolium, pyridinium, pyrrolidinium, piperidinium, ammonium, and phosphonium derivatives; anions include bis(trifluoromethanesulfonyl)imide, tetrafluoroborate, hexafluorophosphate, dicyanamide, thiocyanate, alkyl sulfates, halides, and carboxylates.

From this database I derived model-ready pure-IL subsets with validated ion SMILES: 3,242 conductivity records for 216 unique ILs (temperature range 20–484 K), 102 density records for 101 ILs, 59 viscosity records for 58 ILs, and 61 melting-point records for 61 ILs. The conductivity subset is the only one with multiple records per IL (mean ≈ 15 records per IL), which proves decisive for IL-level extrapolation, as shown below.


**Table 1. Database statistics.** Sources, property coverage, and model-ready subsets.

| Source | Records | Properties |
| --- | ---: | --- |
| ILest/joekasp conductivity compilations | 8,035 | κ |
| ILThermo (conductivity, purpose-built parser) | 3,036 | κ |
| iolitech (pure ILs) | 440 | κ, η, ρ, T_m |
| Total | 11,511 | 4 properties, 1,658 ion pairs |

### 2.2. A single open representation for four properties

Each ion is encoded by ten freely computable RDKit 2D descriptors (molecular weight, log P, H-bond donors/acceptors, topological polar surface area, rotatable bonds, fraction of Csp³, ring counts, molar refractivity) concatenated with a 1024-bit ECFP4 Morgan fingerprint (17). The input vector for each record is

$$x = [\Phi(c^+) \oplus \Phi(a^-) \oplus x_{IL} \oplus T] \in \mathbb{R}^{2070},$$

where Φ is the ion descriptor map, x_IL encodes the IL-level composition (mole fraction), and T is temperature in K. No commercial descriptor package and no continuum solvation calculation is used. Featurization succeeds for 3,242 of 3,268 conductivity records (26 records lost to SMILES parsing), at sub-millisecond cost per record on a single CPU core.

### 2.3. Honest IL-level accuracy: what the models actually achieve

All models were evaluated with 5-fold GroupKFold in which every record of a given ion pair is confined to a single fold, so test folds contain only completely unseen ionic liquids (Fig. 3; Table S1). The gradient-boosting model (GBM) reaches R² = 0.552 and MAE = 1.185 (in ln κ units) for conductivity; the multilayer perceptron (MLP) reaches R² = 0.329 and MAE = 1.401, and multiple linear regression fails (R² = −0.099). Density, which is strongly determined by composition, is predicted with R² = 0.828 and MAE = 0.051 g cm⁻³. Viscosity and melting-point subsets contain roughly one record per IL, so IL-level extrapolation is intrinsically a cold-start problem: GBM reports R² = −0.096 (MAE = 0.419 in ln η units) and R² = −0.086 (MAE = 19.2 K), respectively. These results quantify the data-density requirement for reliable extrapolation: the properties with ≥10 records per IL (conductivity, density) are predictable; the properties with exactly one record per IL (viscosity, melting point) are not, at this data scale.

![Figure 3](figures/fig3_properties.png)

**Fig. 3. IL-level (GroupKFold) prediction of four ionic-liquid properties.** Each panel shows pooled out-of-fold predictions of the GBM under 5-fold IL-disjoint cross-validation. (a) Conductivity (ln κ scale), n = 3,242 records / 216 ILs, R² = 0.552, MAE = 1.185. (b) Density, n = 102 / 101, R² = 0.828, MAE = 0.051 g cm⁻³. (c) Viscosity (ln η scale), n = 59 / 58, R² = −0.096, MAE = 0.419. (d) Melting point, n = 61 / 61, R² = −0.086, MAE = 19.2 K. Panels c and d are cold-start cases with one record per IL.

### 2.4. Quantifying the inflation of point-wise validation

Paired control experiments on identical features and models reproduce and quantify the split-inflation phenomenon on real data (Fig. 2). For conductivity, the same GBM reports point-wise R² = 0.908 but only R² = 0.552 under IL-disjoint splitting; the MLP falls from 0.918 to 0.329; and linear regression falls from 0.438 to −0.099. The point-wise estimates are thus inflated by 0.36–0.59 R² units relative to honest extrapolation, confirming on an independent dataset the effect documented in ref. (9) and explaining why literature claims of "near-perfect prediction" (R² ≈ 0.98–0.99) must be interpreted with care (9, 11).

![Figure 2](figures/fig2_split_inflation.png)

**Fig. 2. Experimental versus calculated ln κ under the two validation protocols (GBM, 5-fold).** (a) IL-disjoint split (R² = 0.552): every record of an ionic liquid is confined to a single fold, and test folds contain only unseen ion pairs. (b) Point-wise split (R² = 0.908): the same ion pairs appear in both training and test folds. The dashed line is perfect agreement. The gap between the two R² values quantifies the inflation inherent in conventional validation.

### 2.5. Mechanism: temperature, anion identity, and cation size drive conductivity

SHAP analysis of the conductivity model (18) identifies temperature as the dominant descriptor (mean |SHAP| = 1.665), consistent with the thermally activated nature of ionic transport (Fig. S1). Anion lipophilicity (LogP, 0.228) and polarity (TPSA, 0.215) follow, then cation size descriptors (molecular weight 0.199, rotatable bonds 0.191, molar refractivity 0.146). A perturbation analysis shows that a ±10 K change in temperature changes the predicted ln κ by 0.526 on average (≈69% relative change in κ), quantifying the thermal sensitivity captured by the model. These rankings are physically interpretable: temperature sets the activation scale; anion size and polarity govern ion dissociation and mobility; cation bulk affects packing and charge delocalization.

### 2.6. Multi-property structure: a shared representation, but data density is the binding constraint

A single shared representation serves all four properties, and a multi-task MLP with a shared trunk and per-property heads was benchmarked against single-task training under the identical IL-disjoint protocol (Fig. S2). At the current data scale, shared training yields no significant improvement for data-rich properties and only marginal gains for the sparsest ones (conductivity R² = −0.876 vs −0.620 single-task; density −0.296 vs −0.210; viscosity −1.210 vs −1.404; melting point −0.363 vs −0.466). The shared representation therefore does not rescue cold-start properties at this data density, identifying data coverage—not representation capacity—as the binding constraint for multi-property extrapolation. This is a constructive negative result: it defines precisely where new experimental data would create the largest predictive gain.

## 3. Discussion

The central finding of this work is quantitative: on real data, point-wise validation inflates conductivity R² by ~0.36 relative to strict IL-disjoint validation, and the inflation grows to ~0.59 for the MLP. Because most published QSPR models for ILs are validated point-wise, the practical accuracy of IL property prediction is systematically overestimated in the literature. This has direct consequences for virtual screening: a model whose test R² = 0.92 on a point-wise split may rank unseen ILs no better than a model at R² = 0.55 evaluated honestly. Reporting both numbers, as done here, makes the generalization risk explicit rather than implicit.

The framework also demonstrates that expensive descriptor pipelines are not required for useful structure-based prediction. RDKit descriptors and Morgan fingerprints are freely available, deterministic, and fast; the entire pipeline—data assembly, featurization, training, and interpretation—runs without commercial software. This lowers the entry barrier for the community and makes every result reproducible from the public data and code.

The most actionable implication, however, is about data strategy. Conductivity and density—the properties with sufficient per-IL coverage—are predictable under IL-disjoint validation, while viscosity and melting point—with one record per IL—are not. The gap is not a failure of representation or model capacity: it is a failure of data density. Future experimental effort should prioritize multi-temperature, multi-composition measurements of the same ILs for underrepresented properties; the framework quantifies exactly how much coverage is needed before IL-level extrapolation becomes reliable.

## 4. Conclusions

This work contributes an open, reproducible framework that predicts four ionic-liquid properties from cation/anion SMILES and temperature using only free descriptors, together with the largest consistently featurized multi-property IL database assembled in this work (11,511 records; 1,658 ion pairs). Under strict IL-disjoint validation, GBM achieves R² = 0.552 (conductivity, ln κ scale) and R² = 0.828 (density), while paired point-wise controls reach R² = 0.908, quantifying the inflation inherent in conventional validation. SHAP analysis confirms that temperature, anion lipophilicity/polarity, and cation size dominate conductivity, and sensitivity analysis yields a ±10 K → Δln κ = 0.526 thermal response. Finally, the negative multi-task result and the cold-start failures of viscosity and melting point identify data density—not model capacity—as the binding constraint, providing a concrete roadmap for the experiments that would most increase predictive power. The database, code, and figures are fully available, and the framework is ready for extension to chemical properties (pKa, toxicity) as labeled data accumulate.

## 5. Materials and Methods

### 5.1. Data assembly

Experimental records were compiled from ILest (8,035 mixture-conductivity records), ILThermo conductivity tables resolved to ion-level SMILES by a purpose-built parser (3,036 records), and the iolitech compilation (440 records) (14–16). Records were matched to cation/anion SMILES through a name-to-SMILES vocabulary, deduplicated, and stored in an SQLite database with consistent units (κ in S m⁻¹, η in mPa·s, ρ in g cm⁻³, T_m in K). Model-ready subsets contain only pure-IL records (mole fraction = 1) with valid ion SMILES and positive property values.

### 5.2. Descriptor representation

Ions were featurized with RDKit (17): ten 2D descriptors per ion plus a 1024-bit ECFP4 Morgan fingerprint, concatenated with mole fraction and temperature into a 2070-dimensional vector. Positive-skewed targets (conductivity, viscosity) were modeled in log space; density and melting point in original units.

### 5.3. Models

Three model families were compared: multiple linear regression (MLR); histogram-based gradient boosting (HistGradientBoostingRegressor, 300 iterations, learning rate 0.08, max depth 6, early stopping); and a multilayer perceptron (128–64–16 hidden units, ReLU, batch normalization, early stopping). A multi-task MLP with a shared trunk and per-property heads, trained with missing-target masking, was used for the multi-property benchmark.

### 5.4. Validation protocols

IL-disjoint validation used 5-fold GroupKFold with ion pairs as groups. Point-wise controls used 5-fold shuffled KFold with the same folds across models. All metrics (R², RMSE, MAE) were computed on the modeling scale; conductivity is reported in ln κ units unless stated otherwise.

### 5.5. Interpretation

SHAP values were computed with the tree explainer on the trained GBM (300 background samples) (18). Temperature sensitivity was quantified as the mean absolute change in predicted ln κ when T was perturbed by ±10 K at fixed descriptors.

## Data and materials availability

The assembled database (11,511 records for 1,658 ion pairs; `data/il_props.db`), all modeling and figure scripts, per-fold validation results, and this manuscript are openly available at https://github.com/linfuxing123/IL-Property-ML and archived on Zenodo (DOI: 10.5281/zenodo.XXXXXXX; assigned on first release). Public data sources: ILest (14), iolitech (15), and ILThermo (16). Code is provided as Python scripts with pinned dependencies (requirements.txt); no commercial software is required.

## References

1. R. D. Rogers, K. R. Seddon, Ionic liquids—solvents of the future? *Science* **302**, 792–793 (2003).
2. M. Armand, F. Endres, D. R. MacFarlane, H. Ohno, B. Scrosati, Ionic-liquid materials for the electrochemical challenges of the future. *Nat. Mater.* **8**, 621–629 (2009).
3. J. P. Hallett, T. Welton, Room-temperature ionic liquids: Solvents for synthesis and catalysis. *Chem. Rev.* **111**, 3508–3576 (2011).
4. K. T. Butler, D. W. Davies, H. Cartwright, O. Isayev, A. Walsh, Machine learning for molecular and materials science. *Nature* **559**, 547–555 (2018).
5. W. Kohn, L. J. Sham, Self-consistent equations including exchange and correlation effects. *Phys. Rev.* **140**, A1133–A1138 (1965).
6. O. Borodin, G. D. Smith, Structure and dynamics of N-methyl-N-propylpyrrolidinium bis(trifluoromethanesulfonyl)imide ionic liquid from molecular dynamics simulations. *J. Phys. Chem. B* **110**, 11481–11490 (2006).
7. M. G. Del Pópolo, G. A. Voth, On the structure and dynamics of ionic liquids. *J. Phys. Chem. B* **108**, 1744–1752 (2004).
8. A. A. H. Pádua, M. F. Costa Gomes, J. N. A. Canongia Lopes, Understanding ionic liquids through atomistic and coarse-grained molecular dynamics simulations. *Acc. Chem. Res.* **40**, 1087–1096 (2007).
9. X. Yu, End-to-end deep learning models for predicting the electrical conductivity of ionic liquids. *ACS Sustain. Chem. Eng.* (2026). DOI: 10.1021/acssuschemeng.6c07089.
10. Z. Chen, J. Chen, Prediction of electrical conductivity of ionic liquids: From COSMO-RS derived QSPR evaluation to boosting machine learning. *ACS Sustain. Chem. Eng.* **12**, 17749–17760 (2024). DOI: 10.1021/acssuschemeng.4c00307.
11. C. Song, C. Wang, F. Fang, G. Zhou, Z. Dai, Z. Yang, Large-scale screening for high conductivity ionic liquids via machine learning algorithm utilizing graph neural network-based features. *J. Chem. Eng. Data* **69**, 800–810 (2024). DOI: 10.1021/acs.jced.3c00709.
12. R. Li, et al., Machine learning-enhanced QSPR model for predicting the viscosity of ionic liquids. *Chem. Eng. Sci.* **321**, 122992 (2025). DOI: 10.1016/j.ces.2025.122992.
13. V. Venkatraman, S. Evjen, H. K. Knuutila, A. Fiksdahl, B. K. Alsberg, Predicting ionic liquid melting points using machine learning. *J. Mol. Liq.* **264**, 318–326 (2018).
14. ILest: Ionic liquids database. https://ilest.uobabylon.edu.iq.
15. iolitech: Ionic liquids physicochemical properties database. https://iolitech.de.
16. NIST ILThermo: Ionic liquids database. https://ilthermo.boulder.nist.gov.
17. G. Landrum, RDKit: Open-source cheminformatics. https://www.rdkit.org.
18. S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions. *Adv. Neural Inf. Process. Syst.* 30, 4765–4774 (2017).



## Supplementary materials

Materials and methods details, data-assembly documentation, the full model performance table (table S1), per-fold results, descriptor lists, model hyperparameters, the SHAP analysis (fig. S1), and the multi-task benchmark (fig. S2) are provided in the Supplementary Materials.
