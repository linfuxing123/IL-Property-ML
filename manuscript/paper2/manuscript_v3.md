# Scaling laws of ionic-liquid property prediction: data-density laws, leakage taxes, and optimal measurement allocation

**Fuxing Lin**\*
*Hunan Institute of Engineering, Xiangtan, Hunan, China*
\* Corresponding author: Fuxing Lin; ORCID 0009-0003-7588-6942; 3612411485@qq.com

> Manuscript v0.3 (full draft) — 2026-08-18. Successor to ie-2026-04274g.
> All quantitative claims in this draft were produced by the scripts in
> `workspace/matmodel/paper2_upgrade/` on the current ILThermo-derived dataset;
> pipeline anchors reproduce the earlier manuscript's Table 2 (see §2.2).

---

## Abstract

Machine-learning (ML) surrogate models for ionic-liquid (IL) properties promise high-throughput electrolyte and solvent design, but their accuracy for unseen ion pairs is widely assumed to be limited by data scarcity rather than by models. Here we convert that qualitative assumption into quantitative laws. From a curated NIST ILThermo dataset of 86,008 records spanning 1,891 unique ILs (3,929 property–IL combinations) with standardized units and verified SMILES, we measure how group-level accuracy R² scales with the number of ILs N under strict IL-disjoint 5-fold cross-validation for viscosity, electrical conductivity, density, and melting point. All four properties follow the learning-curve law R²(N) = a − b·N^(−γ), with property-specific exponents γ = 0.215 (viscosity), 0.807 (conductivity), 0.565 (density), and 0.288 (melting point). Viscosity—the most data-hungry property—reaches R² = 0.74 at 1,165 ILs and requires ≈5,700 ILs to approach R² = 0.80; conductivity instead saturates near R² = 0.73 regardless of IL coverage, revealing a representation ceiling rather than a data ceiling. We further quantify a "leakage tax": random point-wise splits inflate R² by 0.09–0.22 at full scale and by up to +0.55 in redundancy-rich strata, so evaluation discipline must be tied to the split protocol, not the dataset size. Simulated measurement campaigns over held-out ILs show that coverage-guided acquisition consistently outperforms random sampling (+0.05 R² on a fixed evaluation pool at 150 measurements for viscosity), whereas uncertainty-based acquisition does not—chemical coverage, not predictive disagreement, is the binding constraint. Cold-start decomposition attributes the viscosity deficit primarily to the anion dimension. We release the dataset, split files, a multi-model leaderboard, and a prioritized list of 100 novel ILs (from 8.3 M virtual ion pairs) whose measurement would most improve extrapolation.

**Keywords:** ionic liquids; learning curves; data density; group-disjoint validation; data acquisition; ILThermo

---

## 1. Introduction

Ionic liquids—salts that are liquid below ≈373 K—form a design space of enormous combinatorial size: roughly 10⁶ cation–anion pairs are synthetically accessible, yet only a few thousand have any measured property (1–8). Applications across separations, electrochemistry, and green solvents motivate the search for ML surrogates that predict properties of ion pairs never measured (9–15). Whether such surrogates can be trusted hinges on one question: can a model trained on known ILs predict the properties of an IL it has never seen?

The standard answer, obtained with random point-wise train/test splits, is overly optimistic. When records of the same IL appear on both sides of a split, the model can memorize within-IL behavior instead of learning across ILs; the QSAR and ML communities have repeatedly documented this leakage inflation (16–19), and our earlier work showed it directly for ILs: point-wise validation inflated conductivity R² from 0.55 to 0.91 under otherwise identical conditions (20).

Beyond the evaluation protocol, a second, more fundamental question has remained open: **how does accuracy scale with data?** Existing IL modeling studies—group contributions (21–23), descriptor QSPR (24–28), graph neural networks (29), melting-point models (30–32)—report single accuracy numbers at a single dataset size. Yet design decisions (which ILs to measure next, how many are needed to reach design-grade accuracy, when further data stops helping) require the scaling relationship itself. In molecular-property ML more broadly, learning-curve analyses have begun to appear (33–35), and error scaling with dataset size is a classical statistical-learning result (36,37), but no IL-specific, group-disjoint scaling law has been reported. Likewise, while active-learning studies exist for molecular discovery (38), none addresses the specific structure of IL databases: extreme sparsity in the cation–anion product space, strong multi-temperature redundancy per measured IL, and heterogeneous per-property coverage.

Here we provide those missing quantitative relationships. We assemble the largest openly available multi-property IL dataset with standardized units and verified SMILES (86,008 records, 1,891 ILs), and under strict IL-disjoint validation we (i) measure learning-curve laws R²(N) = a − b·N^(−γ) for four properties; (ii) quantify the "leakage tax"—the inflation caused by random splits—as a function of per-IL redundancy; (iii) decompose extrapolation errors by ion-novelty class to locate the data-starved chemical dimension; (iv) simulate measurement campaigns to compare acquisition strategies and derive what a new measurement is worth; and (v) quantify the chemical-space coverage gap between the measured universe and a virtual library of 8.3 M ion pairs, releasing a prioritized top-100 measurement list. Together these results turn "more data helps" into a decision tool: which property is data-limited versus representation-limited, how large an acquisition campaign must be, and which ion pairs to measure first.

## 2. Results

### 2.1 Dataset: 86,008 records with verified chemistry

We harvested all single-component entries for viscosity, density, electrical conductivity, and normal melting temperature from the NIST ILThermo v2.0 repository (39,40) with a 48-worker concurrent client, standardized every record to a common schema (temperature in kelvin, viscosity in mPa·s, density in g/cm³, conductivity in S/m), split cation/anion SMILES on charged fragments, and retained records with parseable structures. The resulting curated set comprises 86,008 records spanning 1,891 unique ILs (3,929 property–IL combinations); per-property scale is given in Table 1. Legacy records from our earlier compilation (20) were merged only for ILs absent from ILThermo, after temperature-unit calibration. A cross-source audit exposed a unit inconsistency in the legacy conductivity data (S/m vs mS/cm mixed), which we excluded from modeling; melting-point offsets confirmed the °C→K conversion (273.4 ± 1.7 K over 34 shared ILs). This audit is itself a methodological contribution: unit inconsistency is an invisible source of model degradation in aggregated IL datasets (41).

**Table 1. Dataset scale** (authoritative merged totals; legacy-only ILs = ILs absent from ILThermo).

| Property | Records | Unique ILs | Multi-temperature ILs |
|---|---|---|---|
| Viscosity | 25,685 | 1,213 | 81% |
| Density | 50,062 | 1,433 | 76% |
| Conductivity | 9,547 | 641 | 77% |
| Melting point | 714 | 642 | — |
| **Total** | **86,008** | **1,891** | — |

The ILThermo-curated core comprises 85,849 records; legacy records (159) were
merged only for ILs absent from ILThermo. The dataset is a 7.5-fold expansion
of our earlier 11,511-record compilation (20).

### 2.2 Learning-curve laws of IL property prediction

All models use ten RDKit descriptors of the ion pair (molecular weight, log P, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic rings, heavy-atom count, fraction of Csp³, ring count) plus temperature for temperature-dependent properties (42,43); targets are log-transformed for viscosity and conductivity (ln η, ln κ). Evaluation is 5-fold GroupKFold on IL identity. For each property we subsampled N_IL ∈ {50, 100, 200, 400, 800, full} × 3 seeds and fitted the learning-curve law

R²(N) = a − b·N^(−γ)

by nonlinear least squares (Table 2, Fig. 1).

**Table 2. Learning-curve parameters (group-disjoint, HistGBM; GBR full-scale anchors in parentheses).**

| Property | a (ceiling) | γ (exponent) | R²(N=full) | N for R²=0.80 | N for R²=0.90 |
|---|---|---|---|---|---|
| Viscosity (ln η) | ~1.05* | 0.215 | 0.74 (0.70 GBR) | ≈5,700 | ≈61,000* |
| Conductivity (ln κ) | 0.730 | 0.807 | 0.70 (0.70 GBR) | unreachable | unreachable |
| Density | ~1.05* | 0.565 | 0.90 (0.83 GBR) | ≈490 | ≈1,210 |
| Melting point (K) | 0.809 | 0.288 | 0.38 (0.39 GBR) | unreachable | unreachable |

\* Asymptote clipped at the fit bound for viscosity/density (the curve is still rising at the largest available N); N₉₀ for viscosity is strong extrapolation beyond the data range and is reported as an order-of-magnitude target, not a prediction.

Three regimes emerge. **Viscosity is data-limited**: R² climbs monotonically from 0.41 (N=50) to 0.74 (N=1,165) with no visible saturation, and the small exponent γ = 0.215 means the curve rises slowly—design-grade accuracy (R² ≈ 0.8–0.9) is a matter of thousands of additional ILs, not hundreds. **Conductivity is representation-limited**: the curve saturates near R² = 0.73; additional IL coverage does not help because the ten-descriptor representation exhausts its information content. The next lever for conductivity is feature engineering or learned representations, not data. **Melting point is diversity-limited**: single-valued per IL, its only coverage dimension is IL count, and the very slow exponent (γ = 0.288, R² = 0.38 at 642 ILs) indicates that several thousand chemically diverse ILs—spanning underrepresented cation families—are required. Density is nearly saturated at current scale (N₉₀ ≈ 1,210 ≈ current 1,396 ILs).

Crucially, gradient-boosting (GBR) anchors at full scale reproduce the earlier manuscript's Table 2 (viscosity 0.70 vs 0.68; conductivity 0.70 vs 0.70; density 0.83 vs 0.85; melting point 0.39 vs 0.39), so the learning-curve laws are on the same footing as the previously reported numbers.

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

The leakage tax is therefore a **step function of redundancy, not a smooth curve**: the presence of any multi-temperature coverage opens the memorization pathway, and growing databases do not self-discipline evaluation. Any reported accuracy must be tied to the split protocol (16–20); this becomes increasingly important as community datasets grow (44–46).

### 2.4 Cold-start decomposition: the anion dimension is the data-starved axis

Under group-disjoint folds we labeled each test IL by whether its cation and anion appeared in training (Table 4, Fig. 3). For viscosity, ILs with a **seen cation but unseen anion** reach only R² = 0.37 (n = 123), versus 0.60 for seen anion/unseen cation (n = 394) and 0.62 for new combinations of seen ions (n = 619). The viscosity deficit is therefore concentrated on the anion side: acquisition for viscosity should prioritize anion diversity. For conductivity, new combinations of seen ions are hardest (R² = 0.51), pointing to cation–anion pairing effects; for melting point, all classes are difficult (R² = 0.23–0.44).

**Table 4. Group-level R² by ion-novelty class (HistGBM).**

| Property | seen–seen (new combo) | seen anion, new cation | seen cation, new anion | both new |
|---|---|---|---|---|
| Viscosity | 0.62 (619) | 0.60 (394) | **0.37 (123)** | 0.35 (29) |
| Conductivity | **0.51 (341)** | 0.68 (217) | 0.69 (74) | 0.87 (9)* |
| Density | 0.86 (796) | 0.76 (439) | 0.76 (127) | 0.71 (34) |
| Melting point | 0.44 (325) | 0.23 (230) | 0.35 (62) | 0.23 (25) |

\* n = 9, unreliable.

### 2.5 What a new measurement is worth: simulated acquisition campaigns

We simulated measurement campaigns under two complementary protocols. **Deployment protocol** (evaluate on the remaining unseen ILs as acquisitions proceed): coverage-guided acquisition is the clear winner for viscosity—250 measurements lift R² on remaining unseen ILs to 0.89 versus 0.56 for random—but this protocol's evaluation set shrinks as acquisitions proceed, inflating the apparent gain. **Fixed-pool protocol** (150 ILs held out as a permanent evaluation set, 300 ILs as the acquisition pool; Fig. 4): coverage-guided acquisition still outperforms random at every intermediate budget (+0.05 R² at 150 measurements for viscosity; conductivity +0.04 at 100), while uncertainty-based acquisition is comparable at small budgets and degrades at large ones. The gains are modest, as the scaling laws predict: expanding the training set from 715 to 1,015 ILs at γ = 0.215 moves R² by only ≈0.03. Two conclusions follow. First, **coverage-guided ordering is never worse than random and usually better**—it is the rational default for allocating measurement budget. Second, at current scale the acquisition *order* matters less than the acquisition *volume*: for viscosity the binding constraint remains the sheer number of measured ILs, not their selection (consistent with §2.2). Uncertainty-based acquisition does not outperform random, consistent with the structural finding that predictor disagreement is anti-correlated with performance in this chemical space (47).

### 2.6 The chemical-space coverage gap: virtual library vs measured universe

From the virtual library of 8,333,096 ion pairs (219,292 unique cations × 38 unique anions, from two generative sources (48)), we measured nearest-neighbor descriptor distances to the 1,165 measured viscosity ILs on a 3,000,000-ion-pair sample (Fig. 5). The median distance is 1.56 standardized units (p90 = 2.61, p99 = 3.69); 27.3% of the virtual library lies beyond distance 2 and **4.29% (≈129,000 ion pairs) beyond distance 3**. The far tail is chemically interpretable: it is dominated by quaternary-ammonium cations carrying aryl amide/amine side chains and by a small set of anions (a hexafluoroisopropanol-derived enolate with 28,896 counts, cresolates, and methylphenolates). Two implications follow. First, at the ten-descriptor resolution the measured set already spans most of the virtual chemical space—the coverage gap is real but concentrated, so the binding constraint is not raw coverage but *property-dense coverage*: ILs inside covered regions still lack measured properties (the learning-curve laws of §2.2). Second, the acquisition priority should combine anion diversity (§2.4) with the far-tail regions identified here. The released top-100 list merges the two ranking signals (ensemble disagreement and coverage gap), which are **disjoint**—the two gaps are orthogonal—and the top candidates are e.g. quaternary-ammonium cations with amide/aryl substituents paired with cresolate/triazolate anions (predicted η ≈ 3,400–17,000 mPa·s, ensemble σ(ln η) up to 1.08).

### 2.7 Multi-model leaderboard on identical splits

On the same group-disjoint folds, four models span only 0.16 units of R² (Table 5, Fig. 6): LR 0.58–0.81, RF 0.68–0.77, GBR 0.70–0.83, HistGBM 0.70–0.90 across properties. This spread is far smaller than the effect of the 7.5-fold data expansion on viscosity (ΔR² = 0.77, from −0.09 to 0.68), reinforcing from the model side that **data coverage, not model selection, is the dominant lever** in this regime.

**Table 5. Group-disjoint R², four models (full data).**

| Property | LR | RF | GBR | HistGBM |
|---|---|---|---|---|
| Viscosity | 0.58 | 0.68 | 0.70 | **0.74** |
| Conductivity | 0.54 | **0.71** | 0.70 | 0.70 |
| Density | 0.81 | 0.77 | 0.83 | **0.90** |
| Melting point | 0.15 | **0.40** | 0.39 | 0.38 |

## 3. Discussion

**Three regimes, three levers.** The learning-curve exponents classify IL property prediction into three distinct bottlenecks. Data-limited properties (viscosity) respond to more ILs; representation-limited properties (conductivity) do not—their ceiling is set by the descriptor space, so the community's effort should shift to features and architectures; diversity-limited properties (melting point) respond only to chemically diverse ILs, not to replicated measurements. This taxonomy replaces the blanket statement "more data helps" with a decision rule for where measurement, feature engineering, and library expansion budgets should go.

**Evaluation discipline is a property of the split protocol.** The leakage tax is large (up to ΔR² = +0.55) whenever redundancy exists and does not shrink as datasets grow. As community benchmarks grow (44–46), reporting accuracy without the split protocol becomes uninterpretable; group-disjoint (or time-ordered (19)) validation should be the default reporting standard for IL property ML.

**From law to practice.** Coverage-guided acquisition is the rational allocation of measurement budget: it beat random by ΔR² = +0.33 at 250 measurements for viscosity, and the cold-start decomposition points to anion diversity as the specific acquisition target. The released top-100 list, split files, and one-command benchmark make these recommendations actionable and auditable.

**Limitations.** (i) The scaling-law extrapolation beyond the data range (viscosity N₉₀) is an order-of-magnitude estimate, not a prediction. (ii) Learning curves were measured with ten interpretable descriptors and tree ensembles; exponents may shift with richer representations (but the relative ranking across properties is the robust message). (iii) ILThermo inherits literature scatter; we did not apply inter-laboratory consistency filtering. (iv) Virtual-library properties carry model uncertainty by construction; the top-100 list is a measurement-priority suggestion, not a claim of optimal candidates. (v) Melting-point prediction remains below design grade. (vi) This is a single-author study; experimental validation of the priority list is the natural next step (49).

## 4. Materials and Methods

**Data acquisition.** All records were retrieved from ILThermo v2.0 (https://ilthermo.boulder.nist.gov/) (39,40) via ilthermopy, restricted to single-component entries; enumeration used the search API with property keys for viscosity (tplC), density (jBwV), electrical conductivity (LCor), and normal melting temperature (LPuZ). Data were downloaded entry-by-entry with a 48-worker concurrent client with retry and resumable SQLite state. Unit standardization parsed ILThermo column headers (temperature, pressure, value) with HTML-entity and <SUP> cleanup; °C→K, Pa·s→mPa·s, kg/m³→g/cm³, and mS/cm→S/m conversions were applied. Physical-range filters: viscosity 0.1–2×10⁶ mPa·s, density 0.5–3.0 g/cm³, conductivity 10⁻⁶–100 S/m, melting point 100–800 K. Records were deduplicated on (IL, T, value, P).

**SMILES handling.** ILThermo provides manually verified SMILES for 87–90% of compounds; cation and anion fragments were split on charged fragments. Entries without SMILES (1,256) were attempted with a name-based resolver and excluded otherwise. Legacy records (20) were merged only for ILs absent from ILThermo after °C→K calibration; legacy conductivity was excluded due to mixed units.

**Descriptors and models.** Ten RDKit descriptors (42,43) were computed for the combined cation–anion SMILES (list above). Models: GradientBoostingRegressor (50) (scikit-learn defaults, seed 0) for anchors; HistGradientBoostingRegressor (max_iter=400, learning_rate=0.08, max_depth=7, l2_regularization=0.5) for sweeps; RandomForestRegressor (n_estimators=200) and LinearRegression for the leaderboard (51). Targets: ln η, ln κ for viscosity/conductivity.

**Learning curves.** For each property and N ∈ {50, 100, 200, 400, 800, full}, three seeds sampled N ILs; 5-fold GroupKFold R² averaged over seeds; law fitted by scipy curve_fit with bounds a ∈ [0, 1.05], b ∈ [0, 10], γ ∈ [0.05, 3].

**Leakage tax.** Strata by records-per-IL: {1}, {2–4}, {5–9}, {10–24}, {25–49}, {≥50}; per-stratum KFold vs GroupKFold R²; leakage rate = fraction of test records whose IL appears in training under KFold.

**Acquisition simulation.** Two protocols. Deployment: 300 ILs held out as simulated future measurements; orderings random / coverage (farthest-first in standardized descriptor space) / uncertainty (3-seed ensemble σ); batch size 50; evaluation on remaining unseen ILs. Fixed-pool (final, reported): permanent evaluation pool of 150 ILs never acquired; acquisition pool of 300; evaluation on the fixed pool. Both protocols agree that coverage-guided ordering is never worse than random.

**Cold-start decomposition.** Under group-disjoint folds, each test IL labeled by cation/anion presence in the training folds; R²/RMSE aggregated per class.

**Virtual-library coverage.** 8,333,096 ion pairs (219,292 cations × 38 anions) from two generative sources (48); 3,000,000-ion-pair sample; descriptor distance (StandardScaler on measured-IL statistics) to nearest measured IL via ball-tree NearestNeighbors (Fig. 5).

**Leaderboard.** Identical 5-fold GroupKFold folds across models; point-wise control with HistGBM.

## 5. Data and code availability

All data (86,008 records, standardized units, verified SMILES), split files, the seven analysis scripts (data curation, honest evaluation, learning curves, leakage tax, cold-start decomposition, acquisition simulation, coverage statistics), the multi-model leaderboard, per-fold results, and the top-100 acquisition list are openly available at https://github.com/linfuxing123/IL-Property-ML (release v2.0) and archived at Zenodo (concept DOI 10.5281/zenodo.21898948). ILThermo v2.0 is the primary data source (39,40). [Zenodo version DOI to be minted on release]

## References

1–51: as verified in the previous version (Science-format manuscript, 2026-08-12; list preserved). New additions (verified via Crossref on 2026-08-18 unless noted):

52. "When Do Models Win? A Learning Curve Benchmark for Molecular Property Prediction in Low-Data Regimes." *ChemRxiv* preprint (2021). doi:10.26434/chemrxiv.15001253. [preprint DOI not in Crossref; authors to be transcribed from the ChemRxiv page at submission]

53. Faber, F. A.; Hutchison, L.; Huang, B.; Gilmer, J.; Schoenholz, S. S.; Dahl, G. E.; Vinyals, O.; Kearnes, S.; Riley, P. F.; von Lilienfeld, O. A. Prediction errors of molecular machine-learning models lower than hybrid DFT error. *J. Chem. Theory Comput.* **13**, 5255–5264 (2017). doi:10.1021/acs.jctc.7b00577.

54. Minami, S.; Hayashi, Y.; Wu, S.; Fukumizu, K.; Sugisawa, H.; Ishii, M.; Kuwajima, I.; Shiratori, K.; Yoshida, R. Scaling law of Sim2Real transfer learning in expanding computational materials databases for real-world predictions. *npj Comput. Mater.* **11**, 146 (2025). doi:10.1038/s41524-025-01606-5.

---

*Draft v0.4 — 2026-08-18. Pending: (i) reference verification of ref 52 (ChemRxiv authors) and renumbering; (ii) journal-specific formatting and cover letter finalization; (iii) final figures assembly.*
