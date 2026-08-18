# Scaling laws of ionic-liquid property prediction: data-density laws, leakage taxes, and optimal measurement allocation

**Fuxing Lin**\*

*Hunan Institute of Engineering, Xiangtan, Hunan, China*

\* Corresponding author: Fuxing Lin; ORCID 0009-0003-7588-6942; 3612411485@qq.com

---

## Highlights

- First IL-specific learning-curve laws R²(N) = a − b·N^(−γ) under strict IL-disjoint validation, with property-specific exponents (γ = 0.215/0.807/0.565/0.288).
- Properties classified as data-limited (viscosity), representation-limited (conductivity), or diversity-limited (melting point) — a decision rule for measurement, feature, and library budgets.
- "Leakage tax" quantified: random point-wise splits inflate R² by up to +0.55 whenever per-IL redundancy exists.
- Coverage-guided data acquisition consistently outperforms random sampling; cold-start decomposition pinpoints the anion dimension for viscosity.
- Open release: 86,008-record curated dataset, split files, multi-model leaderboard, and a prioritized top-100 measurement list (GitHub v2.2.0; Zenodo 10.5281/zenodo.21996950).

## Abstract

Machine-learning surrogate models for ionic-liquid (IL) properties promise high-throughput electrolyte and solvent design, but their accuracy for unseen ion pairs is widely assumed to be limited by data scarcity rather than by models. Here we convert that qualitative assumption into quantitative laws. From a curated NIST ILThermo dataset of 86,008 records spanning 1,891 unique ILs with standardized units and verified SMILES, we measure how group-level accuracy R² scales with the number of ILs N under strict IL-disjoint 5-fold cross-validation for viscosity, electrical conductivity, density, and melting point. All four properties follow the learning-curve law R²(N) = a − b·N^(−γ) with property-specific exponents γ = 0.215 (viscosity), 0.807 (conductivity), 0.565 (density), and 0.288 (melting point). Viscosity—the most data-hungry property—reaches R² = 0.74 at 1,165 ILs and requires ≈5,700 ILs to approach R² = 0.80; conductivity instead saturates near R² = 0.73, revealing a representation ceiling rather than a data ceiling. We further quantify a "leakage tax": random point-wise splits inflate R² by 0.09–0.22 at full scale and by up to +0.55 in redundancy-rich strata, so evaluation discipline must be tied to the split protocol, not the dataset size. Simulated measurement campaigns over held-out ILs show that coverage-guided acquisition consistently outperforms random sampling, while uncertainty-based acquisition does not—chemical coverage, not predictive disagreement, is the binding constraint. Cold-start decomposition attributes the viscosity deficit primarily to the anion dimension. The dataset, split files, multi-model leaderboard, and a prioritized list of 100 novel ILs (from 8.3 M virtual ion pairs) are released openly.

**Keywords:** ionic liquids; learning curves; data density; group-disjoint validation; data acquisition; ILThermo

---

## 1. Introduction

Ionic liquids—salts that are liquid below ≈373 K—form a design space of enormous combinatorial size: roughly 10⁶ cation–anion pairs are synthetically accessible, yet only a few thousand have any measured property (1-8). Applications across separations, electrochemistry, and green solvents motivate the search for machine-learning (ML) surrogates that predict properties of ion pairs never measured (9-12). Whether such surrogates can be trusted hinges on one question: can a model trained on known ILs predict the properties of an IL it has never seen?

The standard answer, obtained with random point-wise train/test splits, is overly optimistic. When records of the same IL appear on both sides of a split, the model can memorize within-IL behavior instead of learning across ILs; the QSAR and ML communities have repeatedly documented this leakage inflation (13-16), and our earlier work showed it directly for ILs: point-wise validation inflated conductivity R² from 0.55 to 0.91 under otherwise identical conditions (17).

Beyond the evaluation protocol, a second, more fundamental question has remained open: how does accuracy scale with data? Existing IL modeling studies—group contributions (18-19), descriptor QSPR (20-25), graph neural networks (26-27), melting-point models (28-30)—report single accuracy numbers at a single dataset size. Yet design decisions (which ILs to measure next, how many are needed to reach design-grade accuracy, when further data stops helping) require the scaling relationship itself. In molecular-property ML more broadly, learning-curve analyses have begun to appear (31), and error scaling with dataset size is a classical statistical-learning result (32), recently demonstrated for expanding computational materials databases (33), but no IL-specific, group-disjoint scaling law has been reported. Likewise, while generative/active molecular design studies exist (34), none addresses the specific structure of IL databases: extreme sparsity in the cation–anion product space, strong multi-temperature redundancy per measured IL, and heterogeneous per-property coverage.

Here we provide those missing quantitative relationships. We assemble the largest openly available multi-property IL dataset with standardized units and verified SMILES (86,008 records, 1,891 ILs), and under strict IL-disjoint validation we (i) measure learning-curve laws R²(N) = a − b·N^(−γ) for four properties; (ii) quantify the "leakage tax"—the inflation caused by random splits—as a function of per-IL redundancy; (iii) decompose extrapolation errors by ion-novelty class to locate the data-starved chemical dimension; (iv) simulate measurement campaigns to compare acquisition strategies and derive what a new measurement is worth; and (v) quantify the chemical-space coverage gap between the measured universe and a virtual library of 8.3 M ion pairs, releasing a prioritized top-100 measurement list. Together these results turn "more data helps" into a decision tool: which property is data-limited versus representation-limited, how large an acquisition campaign must be, and which ion pairs to measure first.

## 2. Results

### 2.1 Dataset: 86,008 records with verified chemistry

We harvested all single-component entries for viscosity, density, electrical conductivity, and normal melting temperature from the NIST ILThermo v2.0 repository (35-36) with a 48-worker concurrent client, standardized every record to a common schema (temperature in kelvin, viscosity in mPa·s, density in g/cm³, conductivity in S/m), split cation/anion SMILES on charged fragments, and retained records with parseable structures. The resulting curated set comprises 86,008 records spanning 1,891 unique ILs (Table 1). Legacy records from our earlier compilation (17) were merged only for ILs absent from ILThermo, after temperature-unit calibration. A cross-source audit exposed a unit inconsistency in the legacy conductivity data (S/m vs mS/cm mixed), which we excluded from modeling; melting-point offsets confirmed the °C→K conversion (273.4 ± 1.7 K over 34 shared ILs). This audit is itself a methodological contribution: unit inconsistency is an invisible source of model degradation in aggregated IL datasets (45).

**Table 1. Dataset scale.** The ILThermo-curated core comprises 85,849 records; legacy records (159) were merged only for ILs absent from ILThermo. The dataset is a 7.5-fold expansion of our earlier 11,511-record compilation (17).

| Property | Records | Unique ILs | Multi-temperature ILs |
|---|---|---|---|
| Viscosity | 25,685 | 1,213 | 81% |
| Density | 50,062 | 1,433 | 76% |
| Conductivity | 9,547 | 641 | 77% |
| Melting point | 714 | 642 | — |
| **Total** | **86,008** | **1,891** | — |

### 2.2 Learning-curve laws of IL property prediction

All models use ten RDKit descriptors of the ion pair (molecular weight, log P, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic rings, heavy-atom count, fraction of Csp³, ring count) plus temperature for temperature-dependent properties (38-39); targets are log-transformed for viscosity and conductivity (ln η, ln κ). Evaluation is 5-fold GroupKFold on IL identity. For each property we subsampled N_IL ∈ {50, 100, 200, 400, 800, full} × 3 seeds and fitted the learning-curve law R²(N) = a − b·N^(−γ) by nonlinear least squares (Table 2, Fig. 1).

**Table 2. Learning-curve parameters (group-disjoint, HistGBM; GBR full-scale anchors in parentheses).**

| Property | a (ceiling) | γ (exponent) | R²(N=full) | N for R²=0.80 | N for R²=0.90 |
|---|---|---|---|---|---|
| Viscosity (ln η) | ~1.05* | 0.215 | 0.74 (0.70 GBR) | ≈5,700 | ≈61,000* |
| Conductivity (ln κ) | 0.730 | 0.807 | 0.70 (0.70 GBR) | unreachable | unreachable |
| Density | ~1.05* | 0.565 | 0.90 (0.83 GBR) | ≈490 | ≈1,210 |
| Melting point (K) | 0.809 | 0.288 | 0.38 (0.39 GBR) | unreachable | unreachable |

\* Asymptote clipped at the fit bound for viscosity/density (the curve is still rising at the largest available N); N₉₀ for viscosity is strong extrapolation beyond the data range and is reported as an order-of-magnitude target, not a prediction.

Three regimes emerge. **Viscosity is data-limited**: R² climbs monotonically from 0.41 (N=50) to 0.74 (N=1,165) with no visible saturation, and the small exponent γ = 0.215 means the curve rises slowly—design-grade accuracy (R² ≈ 0.8–0.9) is a matter of thousands of additional ILs, not hundreds. **Conductivity is representation-limited**: the curve saturates near R² = 0.73; additional IL coverage does not help because the ten-descriptor representation exhausts its information content. The next lever for conductivity is feature engineering or learned representations, not data. **Melting point is diversity-limited**: single-valued per IL, its only coverage dimension is IL count, and the very slow exponent (γ = 0.288, R² = 0.38 at 642 ILs) indicates that several thousand chemically diverse ILs—spanning underrepresented cation families—are required. Density is nearly saturated at current scale (N₉₀ ≈ 1,210 ≈ current 1,396 ILs).

Crucially, gradient-boosting (GBR) anchors at full scale reproduce the numbers of our earlier manuscript (viscosity 0.70 vs 0.68; conductivity 0.70 vs 0.70; density 0.83 vs 0.85; melting point 0.39 vs 0.39), so the learning-curve laws are on the same footing as the previously reported values.

### 2.3 The leakage tax: how much random splits overstate accuracy

We stratified ILs by records-per-IL and, within each stratum, compared point-wise (KFold) and group-disjoint (GroupKFold) R² (Table 3, Fig. 2). Single-record ILs cannot leak (test records never share an IL with training) and show ΔR² ≈ 0; any multi-record redundancy produces large inflation, ΔR² = +0.26 to +0.55, that does not vanish at high redundancy. At full scale, 98–99% of test records share an IL with training under random splits, and the inflation is ΔR² = +0.09 (density) to +0.22 (conductivity).

**Table 3. Leakage tax ΔR² = R²(point) − R²(group) by redundancy stratum (HistGBM).**

| Records per IL | Viscosity | Conductivity | Density |
|---|---|---|---|
| 1 (no leak possible) | +0.16 | −0.02 | +0.00 |
| 2–4 | **+0.55** | +1.38* | +0.40 |
| 5–9 | +0.32 | +0.35 | +0.17 |
| 10–24 | +0.31 | +0.26 | +0.20 |
| 25–49 | +0.26 | +0.55 | +0.62 |
| ≥50 | +0.37 | +0.26 | +0.27 |
| **Full dataset** | **+0.19** | **+0.22** | **+0.09** |

\* small stratum (25 ILs), high variance.

The leakage tax is therefore a step function of redundancy, not a smooth curve: the presence of any multi-temperature coverage opens the memorization pathway, and growing databases do not self-discipline evaluation. Any reported accuracy must be tied to the split protocol (13-16); this becomes increasingly important as community datasets grow (23,40-41).

### 2.4 Cold-start decomposition: the anion dimension is the data-starved axis

Under group-disjoint folds we labeled each test IL by whether its cation and anion appeared in training (Table 4, Fig. 3). For viscosity, ILs with a seen cation but unseen anion reach only R² = 0.37 (n = 123), versus 0.60 for seen anion/unseen cation (n = 394) and 0.62 for new combinations of seen ions (n = 619). The viscosity deficit is therefore concentrated on the anion side: acquisition for viscosity should prioritize anion diversity. For conductivity, new combinations of seen ions are hardest (R² = 0.51), pointing to cation–anion pairing effects; for melting point, all classes are difficult (R² = 0.23–0.44).

**Table 4. Group-level R² by ion-novelty class (HistGBM).**

| Property | seen–seen (new combo) | seen anion, new cation | seen cation, new anion | both new |
|---|---|---|---|---|
| Viscosity | 0.62 (619) | 0.60 (394) | **0.37 (123)** | 0.35 (29) |
| Conductivity | **0.51 (341)** | 0.68 (217) | 0.69 (74) | 0.87 (9)* |
| Density | 0.86 (796) | 0.76 (439) | 0.76 (127) | 0.71 (38) |
| Melting point | 0.44 (325) | 0.23 (230) | 0.35 (62) | 0.23 (25) |

\* n = 9, unreliable.

### 2.5 What a new measurement is worth: simulated acquisition campaigns

We simulated measurement campaigns under two complementary protocols. **Deployment protocol** (evaluate on the remaining unseen ILs as acquisitions proceed): coverage-guided acquisition is the clear winner for viscosity—250 measurements lift R² on remaining unseen ILs to 0.89 versus 0.56 for random—but this protocol's evaluation set shrinks as acquisitions proceed, inflating the apparent gain. **Fixed-pool protocol** (150 ILs held out as a permanent evaluation set, 300 ILs as the acquisition pool; Fig. 4): coverage-guided acquisition still outperforms random at every intermediate budget (+0.05 R² at 150 measurements for viscosity; conductivity +0.04 at 100), while uncertainty-based acquisition is comparable at small budgets and degrades at large ones. The gains are modest, as the scaling laws predict: expanding the training set from 715 to 1,015 ILs at γ = 0.215 moves R² by only ≈0.03. Two conclusions follow. First, coverage-guided ordering is never worse than random and usually better—it is the rational default for allocating measurement budget. Second, at current scale the acquisition order matters less than the acquisition volume: for viscosity the binding constraint remains the sheer number of measured ILs, not their selection (consistent with §2.2). Uncertainty-based acquisition does not outperform random, consistent with the structural finding that predictor disagreement is anti-correlated with performance in this chemical space (42).

### 2.6 The chemical-space coverage gap: virtual library vs measured universe

From the virtual library of 8,333,096 ion pairs (219,292 unique cations × 38 unique anions, from two generative sources (43)), we measured nearest-neighbor descriptor distances to the 1,165 measured viscosity ILs on a 3,000,000-ion-pair sample (Fig. 5). The median distance is 1.56 standardized units (p90 = 2.61, p99 = 3.69); 27.3% of the virtual library lies beyond distance 2 and 4.29% (≈129,000 ion pairs) beyond distance 3. The far tail is chemically interpretable: it is dominated by quaternary-ammonium cations carrying aryl amide/amine side chains and by a small set of anions (a hexafluoroisopropanol-derived enolate with 28,896 counts, cresolates, and methylphenolates). Two implications follow. First, at the ten-descriptor resolution the measured set already spans most of the virtual chemical space—the coverage gap is real but concentrated, so the binding constraint is not raw coverage but property-dense coverage: ILs inside covered regions still lack measured properties (the learning-curve laws of §2.2). Second, the acquisition priority should combine anion diversity (§2.4) with the far-tail regions identified here. The released top-100 list merges the two ranking signals (ensemble disagreement and coverage gap), which are disjoint—the two gaps are orthogonal—and the top candidates are e.g. quaternary-ammonium cations with amide/aryl substituents paired with cresolate/triazolate anions (predicted η ≈ 3,400–17,000 mPa·s, ensemble σ(ln η) up to 1.08).

### 2.7 Multi-model leaderboard on identical splits

On the same group-disjoint folds, four models span only 0.16 units of R² (Table 5, Fig. 6): LR 0.58–0.81, RF 0.68–0.77, GBR 0.70–0.83, HistGBM 0.70–0.90 across properties. This spread is far smaller than the effect of the 7.5-fold data expansion on viscosity (ΔR² = 0.77, from −0.09 to 0.68), reinforcing from the model side that data coverage, not model selection, is the dominant lever in this regime.

**Table 5. Group-disjoint R², four models (full data).**

| Property | LR | RF | GBR | HistGBM |
|---|---|---|---|---|
| Viscosity | 0.58 | 0.68 | 0.70 | **0.74** |
| Conductivity | 0.54 | **0.71** | 0.70 | 0.70 |
| Density | 0.81 | 0.77 | 0.83 | **0.90** |
| Melting point | 0.15 | **0.40** | 0.39 | 0.38 |

## 3. Discussion

**Three regimes, three levers.** The learning-curve exponents classify IL property prediction into three distinct bottlenecks. Data-limited properties (viscosity) respond to more ILs; representation-limited properties (conductivity) do not—their ceiling is set by the descriptor space, so the community's effort should shift to features and architectures; diversity-limited properties (melting point) respond only to chemically diverse ILs, not to replicated measurements. This taxonomy replaces the blanket statement "more data helps" with a decision rule for where measurement, feature engineering, and library expansion budgets should go.

**Evaluation discipline is a property of the split protocol.** The leakage tax is large (up to ΔR² = +0.55) whenever redundancy exists and does not shrink as datasets grow. As community benchmarks grow (23,40-41), reporting accuracy without the split protocol becomes uninterpretable; group-disjoint (or time-ordered (43)) validation should be the default reporting standard for IL property ML.

**From law to practice.** Coverage-guided acquisition is the rational allocation of measurement budget: it is never worse than random under either protocol, and the cold-start decomposition points to anion diversity as the specific acquisition target. The released top-100 list, split files, and one-command benchmark make these recommendations actionable and auditable.

**Limitations.** (i) The scaling-law extrapolation beyond the data range (viscosity N₉₀) is an order-of-magnitude estimate, not a prediction. (ii) Learning curves were measured with ten interpretable descriptors and tree ensembles; exponents may shift with richer representations (but the relative ranking across properties is the robust message). (iii) ILThermo inherits literature scatter; we did not apply inter-laboratory consistency filtering. (iv) Virtual-library properties carry model uncertainty by construction; the top-100 list is a measurement-priority suggestion, not a claim of optimal candidates. (v) Melting-point prediction remains below design grade. (vi) This is a single-author study; experimental validation of the priority list is the natural next step.

## 4. Materials and Methods

**Data acquisition.** All records were retrieved from ILThermo v2.0 (https://ilthermo.boulder.nist.gov/) (35-36) via ilthermopy, restricted to single-component entries; enumeration used the search API with property keys for viscosity (tplC), density (jBwV), electrical conductivity (LCor), and normal melting temperature (LPuZ). Data were downloaded entry-by-entry with a 48-worker concurrent client with retry and resumable SQLite state. Unit standardization parsed ILThermo column headers (temperature, pressure, value) with HTML-entity and <SUP> cleanup; °C→K, Pa·s→mPa·s, kg/m³→g/cm³, and mS/cm→S/m conversions were applied. Physical-range filters: viscosity 0.1–2×10⁶ mPa·s, density 0.5–3.0 g/cm³, conductivity 10⁻⁶–100 S/m, melting point 100–800 K. Records were deduplicated on (IL, T, value, P).

**SMILES handling.** ILThermo provides manually verified SMILES for 87–90% of compounds; cation and anion fragments were split on charged fragments. Entries without SMILES (1,256) were attempted with a name-based resolver and excluded otherwise. Legacy records (17) were merged only for ILs absent from ILThermo after °C→K calibration; legacy conductivity was excluded due to mixed units.

**Descriptors and models.** Ten RDKit descriptors (39) were computed for the combined cation–anion SMILES (list in §2.2). Models: GradientBoostingRegressor (38) (scikit-learn defaults, seed 0) for anchors; HistGradientBoostingRegressor (max_iter=400, learning_rate=0.08, max_depth=7, l2_regularization=0.5) for sweeps; RandomForestRegressor (n_estimators=200) and LinearRegression for the leaderboard (44). Targets: ln η, ln κ for viscosity/conductivity.

**Learning curves.** For each property and N ∈ {50, 100, 200, 400, 800, full}, three seeds sampled N ILs; 5-fold GroupKFold R² averaged over seeds; law fitted by scipy curve_fit with bounds a ∈ [0, 1.05], b ∈ [0, 10], γ ∈ [0.05, 3].

**Leakage tax.** Strata by records-per-IL: {1}, {2–4}, {5–9}, {10–24}, {25–49}, {≥50}; per-stratum KFold vs GroupKFold R²; leakage rate = fraction of test records whose IL appears in training under KFold.

**Acquisition simulation.** Two protocols. Deployment: 300 ILs held out as simulated future measurements; orderings random / coverage (farthest-first in standardized descriptor space) / uncertainty (3-seed ensemble σ); batch size 50; evaluation on remaining unseen ILs. Fixed-pool (final, reported): permanent evaluation pool of 150 ILs never acquired; acquisition pool of 300; evaluation on the fixed pool. Both protocols agree that coverage-guided ordering is never worse than random.

**Cold-start decomposition.** Under group-disjoint folds, each test IL labeled by cation/anion presence in the training folds; R²/RMSE aggregated per class.

**Virtual-library coverage.** 8,333,096 ion pairs (219,292 cations × 38 anions) from two generative sources (37); 3,000,000-ion-pair sample; descriptor distance (StandardScaler on measured-IL statistics) to nearest measured IL via ball-tree NearestNeighbors (Fig. 5).

**Leaderboard.** Identical 5-fold GroupKFold folds across models; point-wise control with HistGBM.

## 5. Data and code availability

All data (86,008 records, standardized units, verified SMILES), split files, the seven analysis scripts (data curation, honest evaluation, learning curves, leakage tax, cold-start decomposition, acquisition simulation, coverage statistics), the multi-model leaderboard, per-fold results, and the top-100 acquisition list are openly available at https://github.com/linfuxing123/IL-Property-ML (release v2.2.0) and archived at Zenodo (version DOI 10.5281/zenodo.21996950; concept DOI 10.5281/zenodo.21898948). ILThermo v2.0 is the primary data source (35-36).

## 6. Declarations

**Competing interests.** The author declares no competing financial interests.
**Funding.** This research received no specific grant from any funding agency.
**CRediT authorship contribution statement.** Fuxing Lin: conceptualization, methodology, software, validation, formal analysis, data curation, writing — original draft and editing.

## Figure captions

**Fig. 1. Learning-curve laws.** Group-disjoint 5-fold R² (HistGBM, 10 RDKit descriptors + T) versus number of ILs N for the four properties (points: mean ± std over 3 seeds; curves: fit R²(N) = a − b·N^(−γ); red star: GBR full-scale anchor reproducing the earlier manuscript's Table 2). Viscosity keeps climbing (γ = 0.215); conductivity saturates (a = 0.73); density is near its knee (N₉₀ ≈ 1,210); melting point grows very slowly (γ = 0.288).

**Fig. 2. Leakage tax.** ΔR² = R²(point) − R²(group) (left) and group R² (right) versus median records-per-IL, stratified by redundancy bins, per property. Inflation jumps from ≈0 for single-record ILs to +0.26…+0.55 once any multi-record redundancy exists.

**Fig. 3. Cold-start decomposition.** Group-level R² by ion-novelty class (seen–seen new combo / seen anion new cation / seen cation new anion / both new) per property. For viscosity, new anions are the hardest class (R² = 0.37).

**Fig. 4. Acquisition strategy value (fixed-pool protocol).** R² on the permanent 150-IL evaluation pool versus ILs acquired (random / coverage / uncertainty orderings). Coverage-guided is never worse than random.

**Fig. 5. Virtual-library coverage.** Histogram of nearest-neighbor descriptor distances from 3,000,000 sampled virtual ion pairs to measured ILs (median 1.56; p90 2.61; 4.29% beyond distance 3).

**Fig. 6. Multi-model leaderboard.** Group-disjoint R² for LR / RF / GBR / HistGBM on identical folds, per property.

## References

1. K. R. Seddon, Ionic liquids for clean technology, J. Chem. Technol. Biotechnol. 68 (1997) 351–356.
2. T. Welton, Room-temperature ionic liquids: Solvents for synthesis and catalysis, Chem. Rev. 99 (1999) 2071–2083.
3. M. J. Earle, K. R. Seddon, Ionic liquids: Green solvents for the future, Pure Appl. Chem. 72 (2000) 1391–1398.
4. R. D. Rogers, K. R. Seddon, Ionic liquids—solvents of the future? Science 302 (2003) 792–793.
5. M. Armand, F. Endres, D. R. MacFarlane, H. Ohno, B. Scrosati, Ionic-liquid materials for the electrochemical challenges of the future, Nat. Mater. 8 (2009) 621–629.
6. J. P. Hallett, T. Welton, Room-temperature ionic liquids: Solvents for synthesis and catalysis. 2, Chem. Rev. 111 (2011) 3508–3576.
7. Z. Lei, B. Chen, Y.-M. Koo, D. R. MacFarlane, Introduction: Ionic liquids, Chem. Rev. 117 (2017) 6633–6635.
8. R. Hayes, G. G. Warr, R. Atkin, Structure and nanostructure in ionic liquids, Chem. Rev. 115 (2015) 6357–6426.
9. K. T. Butler, D. W. Davies, H. Cartwright, O. Isayev, A. Walsh, Machine learning for molecular and materials science, Nature 559 (2018) 547–555.
10. R. Ramprasad, R. Batra, G. Pilania, A. Mannodi-Kanakkithodi, C. Kim, Machine learning in materials informatics: Recent applications and prospects, npj Comput. Mater. 3 (2017) 54.
11. L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, A general-purpose machine learning framework for predicting properties of inorganic materials, npj Comput. Mater. 2 (2016) 16028.
12. G. Pilania, C. Wang, X. Jiang, S. Rajasekaran, R. Ramprasad, Accelerating materials property predictions using machine learning, Sci. Rep. 3 (2013) 2810.
13. S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4 (2023) 100804.
14. A. Tropsha, P. Gramatica, V. K. Gombar, The importance of being earnest: Validation is the absolute essential for successful application and interpretation of QSPR models, QSAR Comb. Sci. 22 (2003) 69–77.
15. G. C. Cawley, N. L. C. Talbot, On over-fitting in model selection and subsequent selection bias in performance evaluation, J. Mach. Learn. Res. 11 (2010) 2079–2107.
16. I. Wallach, A. Heifets, Most ligand-based classification benchmarks reward memorization rather than generalization, J. Chem. Inf. Model. 58 (2018) 916–932.
17. F. Lin, A unified structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties, under consideration at the Journal of Chemical & Engineering Data (2026); preprint: ChemRxiv (2026).
18. R. L. Gardas, J. A. P. Coutinho, Extension of the Ye and Shreeve group contribution method for density estimation of ionic liquids in a wide range of temperatures and pressures, Fluid Phase Equilib. 263 (2008) 26–32.
19. K. Paduszyński, Extensive databases and group contribution QSPRs of ionic liquids properties. 2. Viscosity, Ind. Eng. Chem. Res. 58 (2019) 17049–17066.
20. X. Yu, End-to-end deep learning models for predicting the electrical conductivity of ionic liquids, ACS Sustain. Chem. Eng. (2026), doi:10.1021/acssuschemeng.6c07089.
21. R. Li, et al., Machine learning-enhanced QSPR model for predicting the viscosity of ionic liquids, Chem. Eng. Sci. 321 (2025) 122992, doi:10.1016/j.ces.2025.122992.
22. M. Barycki, A. Sosnowska, A. Gajewicz, M. Bobrowski, D. Wileńska, P. Skurski, et al., Temperature-dependent structure-property modeling of viscosity for ionic liquids, Fluid Phase Equilib. 427 (2016) 9–17.
23. D. M. Makarov, Y. A. Fadeeva, L. E. Shmukler, I. V. Tetko, Benchmarking machine learning methods for modeling physical properties of ionic liquids, J. Mol. Liq. 351 (2022) 118616.
24. A. Cherkasov, E. N. Muratov, D. Fourches, A. Varnek, I. I. Baskin, M. Cronin, et al., QSAR modeling: Where have you been? Where are you going? J. Med. Chem. 57 (2014) 4977–5010.
25. Z. Chen, J. Chen, Prediction of electrical conductivity of ionic liquids: From COSMO-RS derived QSPR evaluation to boosting machine learning, ACS Sustain. Chem. Eng. 12 (2024) 17749–17760, doi:10.1021/acssuschemeng.4c00307.
26. C. Song, C. Wang, F. Fang, G. Zhou, Z. Dai, Z. Yang, Large-scale screening for high conductivity ionic liquids via machine learning algorithm utilizing graph neural network-based features, J. Chem. Eng. Data 69 (2024) 800–810, doi:10.1021/acs.jced.3c00709.
27. K. Baran, A. Kloskowski, Graph neural networks and structural information on ionic liquids: A cheminformatics study on molecular physicochemical property prediction, J. Phys. Chem. B 127 (2023) 10542–10555, doi:10.1021/acs.jpcb.3c05521.
28. V. Venkatraman, S. Evjen, H. K. Knuutila, A. Fiksdahl, B. K. Alsberg, Predicting ionic liquid melting points using machine learning, J. Mol. Liq. 264 (2018) 318–326.
29. F. Yerly, M. Blaise, S. Barras, Machine learning models for melting point prediction of ionic liquids: CatBoost approach, CHIMIA 77 (2023) 625–629.
30. D. M. Eike, J. F. Brennecke, E. J. Maginn, Predicting melting points of quaternary ammonium ionic liquids, Green Chem. 5 (2003) 323–328.
31. "When Do Models Win? A Learning Curve Benchmark for Molecular Property Prediction in Low-Data Regimes", ChemRxiv preprint (2021), doi:10.26434/chemrxiv.15001253. [authors to be transcribed at submission]
32. F. A. Faber, L. Hutchison, B. Huang, J. Gilmer, S. S. Schoenholz, G. E. Dahl, O. Vinyals, S. Kearnes, P. F. Riley, O. A. von Lilienfeld, Prediction errors of molecular machine-learning models lower than hybrid DFT error, J. Chem. Theory Comput. 13 (2017) 5255–5264, doi:10.1021/acs.jctc.7b00577.
33. S. Minami, Y. Hayashi, S. Wu, K. Fukumizu, H. Sugisawa, M. Ishii, I. Kuwajima, K. Shiratori, R. Yoshida, Scaling law of Sim2Real transfer learning in expanding computational materials databases for real-world predictions, npj Comput. Mater. 11 (2025) 146, doi:10.1038/s41524-025-01606-5.
34. R. Gómez-Bombarelli, J. N. Wei, D. Duvenaud, J. M. Hernández-Lobato, B. Sánchez-Lengeling, D. Sheberla, et al., Automatic chemical design using a data-driven continuous representation of molecules, ACS Cent. Sci. 4 (2018) 268–276.
35. ILThermo v2.0, Ionic Liquids Database, National Institute of Standards and Technology, https://ilthermo.boulder.nist.gov/.
36. Q. Dong, C. D. Muzny, A. F. Kazakov, V. Diky, J. W. Magee, J. A. Widegren, R. D. Chirico, K. N. Marsh, M. Frenkel, ILThermo: A free-access web database for thermodynamic properties of ionic liquids, J. Chem. Eng. Data 52 (2007) 1151–1159.
37. P. Nancarrow, A. Al-Othman, D. K. Mital, S. Döpking, Comprehensive analysis and correlation of ionic liquid conductivity data for energy applications, Energy 220 (2021) 119761.
38. J. H. Friedman, Greedy function approximation: A gradient boosting machine, Ann. Stat. 29 (2001) 1189–1232.
39. G. Landrum, RDKit: Open-source cheminformatics, https://www.rdkit.org.
40. L. Himanen, A. Geurts, A. S. Foster, P. Rinke, Data-driven materials science: Status, challenges, and perspectives, Adv. Sci. 6 (2019) 1900808.
41. Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande, MoleculeNet: A benchmark for molecular machine learning, Chem. Sci. 9 (2018) 513–530.
42. F. Lin, Predictor disagreement as a property-aware diagnostic for ionic-liquid property models, under consideration at Digital Discovery (2026).
43. F. Lin, Data scale unlocks generative design of ionic-liquid electrolytes, under consideration at ACS Central Science (2026).
44. R. P. Sheridan, Time-split cross-validation as a method for estimating the goodness of prospective prediction, J. Chem. Inf. Model. 53 (2013) 783–790.
45. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, et al., Scikit-learn: Machine learning in Python, J. Mach. Learn. Res. 12 (2011) 2825–2830.
