# Graph neural networks and engineered descriptors trade on temperature in ionic-liquid property prediction

**One-sentence summary:** Message-passing networks beat engineered descriptors for temperature-dependent properties (density, conductivity) but lose for cold-start melting point under IL-disjoint validation.

**Authors:** Fuxing Lin* (corresponding author; ORCID: 0009-0003-7588-6942)

**Affiliation:** Hunan Institute of Engineering

**Corresponding author:** Fuxing Lin; ORCID: 0009-0003-7588-6942; email: 3612411485@qq.com

---

## Abstract

Machine-learning prediction of ionic-liquid (IL) properties for unseen ion pairs is constrained by per-IL data density and feature engineering. Here I test whether end-to-end graph neural networks (GNNs) trained from cation and anion molecular graphs can substitute for engineered descriptors. Under identical data, folds, and IL-disjoint 5-fold cross-validation, a message-passing network reaches group-level R² of 0.751 for conductivity (ln κ) and 0.941 for density, exceeding both the ten-descriptor (0.698, 0.850) and 458-descriptor (0.740, 0.926) gradient-boosting baselines. For viscosity the GNN (0.780) beats ten descriptors (0.676) but not 458 descriptors (0.809); for cold-start melting point it matches ten descriptors (0.399 versus 0.397) but falls short of 458 descriptors (0.523). The GNN advantage scales with temperature dependence and per-IL data density, while engineered descriptors remain essential for cold-start extrapolation. Learned representations and feature engineering are thus complementary levers on group-level transferability.

**Keywords:** ionic liquids; graph neural networks; message passing; group-disjoint validation; descriptor versus representation; property prediction

---

## 1. Introduction

Ionic liquids (ILs) are salts liquid below roughly 373 K, with a combinatorial cation–anion design space (1–4) that makes machine learning the dominant route to quantitative structure–property prediction (5,6). Descriptor-based QSPR remains the workhorse: group-contribution schemes (7), temperature-dependent QSPRs (8,9), multi-property benchmarks (10), and dedicated conductivity (11,12), graph-feature (13,14), and melting point (15–17) studies. In parallel, graph neural networks (GNNs) promise end-to-end learning directly from molecular structure, bypassing hand-crafted descriptors (18,19), and have been applied to IL properties (13,14).

The companion studies established that, under an IL-disjoint validation protocol that removes ion-pair leakage, point-wise validation inflates conductivity R² by 0.15–0.36 (20), and that the binding constraints on group-level extrapolation are (i) per-IL data density and (ii) feature engineering. A 7.7-fold data expansion converted cold-start properties into predictable ones with unchanged features (21); expanding ten descriptors to 458 per-ion descriptors then lifted all four properties under fixed data, with gains concentrating where data are scarcest (22). The latter study explicitly left open whether learned graph representations push the substitution frontier further.

Here I close that gap: I train a message-passing GNN end-to-end from cation and anion molecular graphs and compare it, under identical data, folds, and protocol, against the ten-descriptor and 458-descriptor gradient-boosting baselines. The answer is graded: learned representations win where temperature varies and data are abundant, and hand-engineered descriptors win at cold start.

## 2. Results

### 2.1. A controlled three-way comparison under IL-disjoint validation

All models are evaluated on the same 84,077-record, 1,891-ion-pair dataset with 5-fold GroupKFold on ion-pair identity; conductivity and viscosity targets are log-transformed, and the two gradient-boosting baselines are reproduced verbatim from the companion study (22). Table 1 reports group-disjoint R² and MAE for the ten-descriptor baseline, the 458-descriptor baseline, and the message-passing GNN.

**Table 1. Group-disjoint 5-fold R² under fixed data, folds, and protocol.**

| Property | Records | ILs | GBM10 R² | GBM458 R² | GNN R² | Δ(GNN−458) | Δ(GNN−10) |
|----------|---------|-----|----------|-----------|--------|------------|-----------|
| Conductivity (ln κ) | 9,470 | 641 | 0.698 | 0.740 | 0.751 | +0.010 | +0.053 |
| Density | 48,428 | 1,433 | 0.850 | 0.926 | 0.941 | +0.015 | +0.091 |
| Viscosity (ln η) | 25,260 | 1,213 | 0.676 | 0.809 | 0.780 | −0.028 | +0.104 |
| Melting point (K) | 919 | 642 | 0.397 | 0.523 | 0.399 | −0.124 | +0.002 |

### 2.2. GNNs win where temperature varies and data are abundant

The GNN outperforms both descriptor baselines for the two most data-dense, temperature-dependent properties. Density improves from 0.850 (ten descriptors) and 0.926 (458 descriptors) to 0.941, and conductivity from 0.698 and 0.740 to 0.751, with MAE falling to 0.029 g/cm³ and 0.769 ln-units respectively. The gains are largest relative to the ten-descriptor baseline (+0.091 for density, +0.053 for conductivity), indicating that learned molecular-graph features capture structure–temperature interactions that a handful of hand-picked descriptors cannot.

### 2.3. Descriptors retain an edge for viscosity and cold-start melting point

The GNN is not uniformly superior. For viscosity it beats ten descriptors (0.780 versus 0.676, +0.104) but falls 0.028 below the 458-descriptor baseline, suggesting that the strong, non-Arrhenius temperature dependence of viscosity is better captured by an explicit descriptor set than by the GNN's temperature-concatenation head. For melting point, the single cold-start property with roughly one record per IL and no temperature variable, the GNN matches ten descriptors (0.399 versus 0.397) but substantially underperforms 458 descriptors (0.523, −0.124), with the highest fold-to-fold variance (standard deviation 0.10). The message is monotone across the property set: the GNN advantage grows with temperature coverage and per-IL record count, and hand-engineered descriptors remain the better lever where data are scarcest (Figure 1 and Figure 3).

![Figure 1](figures/fig1_models_comparison.png)

**Fig. 1. Group-disjoint R² for three models across four properties.** Ten-descriptor gradient boosting (gray), 458-descriptor gradient boosting (blue), and message-passing GNN (red); error bars show GNN fold-level standard deviation. The GNN leads for density and conductivity, ties for viscosity, and trails for melting point.

![Figure 2](figures/fig2_parity.png)

**Fig. 2. Predicted versus measured for the message-passing GNN** under IL-disjoint 5-fold cross-validation, one panel per property.

![Figure 3](figures/fig3_density_gap.png)

**Fig. 3. GNN advantage versus data density.** ΔR² (GNN − 458-descriptor GBM) plotted against the number of records per property (log scale). The GNN advantage increases with data abundance and vanishes (and reverses) for the data-sparse cold-start melting point.

## 3. Discussion

The companion studies established that per-IL data density (21) and feature engineering (22) are the two dominant levers on group-disjoint IL prediction, and explicitly left open whether learned graph representations could push the substitution frontier further. The present result answers with a qualification: yes for temperature-dependent, data-rich properties, and no for cold start. End-to-end message passing beats hand-crafted descriptors where the model has enough multi-temperature data to learn the structure–temperature interaction—density and conductivity—while the 458-descriptor set remains essential for the one-record-per-IL melting point and competitive for viscosity.

This reconciles an apparent tension in the literature. GNN studies of IL properties have reported strong performance (13,14), but those results rest on point-wise validation, which the field has shown to overstate generalization to unseen chemistry (20,23,24). Under honest IL-disjoint evaluation, the GNN advantage is real but conditional: it scales with data density and temperature coverage rather than holding universally. The practical implication is that representation choice should follow data regime—descriptors for sparse screening libraries, learned graphs for well-sampled families—and that reported GNN accuracies must be re-examined under group-disjoint protocols.

Limitations: the GNN uses a fixed three-layer architecture with a scalar temperature appended at the readout head; richer temperature conditioning (attention, physics-informed Arrhenius terms) may close the viscosity gap and deserves separate study. Inter-laboratory scatter in the underlying literature data is inherited unchanged. Melting-point prediction remains below design grade under every representation (R² ≤ 0.52), reinforcing that this cold-start property is a data problem, not a model problem.

## 4. Materials and Methods

**Data.** Records for viscosity, density, electrical conductivity, and normal melting temperature were compiled in the companion studies (21) from ILThermo v2.0, ILest, and iolitech, with units standardized (K, mPa·s, g/cm³, S/m), SMILES verified, and legacy conductivity records excluded on unit-inconsistency grounds. The four property tables (84,077 records after joining descriptor coverage, 1,891 unique ion pairs) are released with this manuscript; conductivity and viscosity targets are log-transformed as before (22).

**Graph representation.** Each ion pair is represented as two independent molecular graphs, one for the cation and one for the anion. Nodes are atoms with features encoding element identity (one-hot over the elements appearing in the dataset plus an "other" slot), degree, formal charge, hydrogen count, hybridization, aromaticity, ring membership, and atomic mass (29 features). Edges are covalent bonds with features encoding bond order (single/double/triple/aromatic) and conjugation (5 features). Graphs are built with RDKit (25) from SMILES.

**Model.** A three-layer message-passing neural network (18) with residual updates encodes each ion graph into a hidden-96 embedding; the readout concatenates mean, sum, and max node embeddings. Cation and anion embeddings are concatenated (with standardized temperature appended for the three temperature-dependent properties) and passed through a three-layer multilayer perceptron (576→128→64→1) with dropout to predict the property. Training uses Adam (learning rate 1e-3, weight decay 1e-4) with mean-squared-error loss, a 10% within-fold validation split for early stopping (patience 20), and a 200-epoch budget.

**Baselines and evaluation.** The two descriptor baselines—gradient boosting on ten merged RDKit descriptors and on the full 458 per-ion RDKit descriptor set—are taken verbatim from the companion study (22), ensuring identical data and folds. All models use 5-fold GroupKFold on ion-pair identity (26). R², MAE, and RMSE are computed on pooled out-of-fold predictions; fold-level standard deviation is reported for the GNN. All code uses PyTorch, RDKit, and scikit-learn.

## Data and code availability

All data, descriptor tables, graph featurization, and training scripts will be made available at GitHub (linfuxing123/IL-Property-ML) and archived on Zenodo (concept DOI 10.5281/zenodo.21898948), following FAIR principles.

## References

1. R. D. Rogers, K. R. Seddon, Ionic liquids—solvents of the future? *Science* **302**, 792–793 (2003).
2. T. Welton, Room-temperature ionic liquids: Solvents for synthesis and catalysis. *Chem. Rev.* **99**, 2071–2083 (1999).
3. M. Armand, F. Endres, D. R. MacFarlane, H. Ohno, B. Scrosati, Ionic-liquid materials for the electrochemical challenges of the future. *Nat. Mater.* **8**, 621–629 (2009).
4. J. P. Hallett, T. Welton, Room-temperature ionic liquids: Solvents for synthesis and catalysis. 2. *Chem. Rev.* **111**, 3508–3576 (2011).
5. K. T. Butler, D. W. Davies, H. Cartwright, O. Isayev, A. Walsh, Machine learning for molecular and materials science. *Nature* **559**, 547–555 (2018).
6. R. Ramprasad, R. Batra, G. Pilania, A. Mannodi-Kanakkithodi, C. Kim, Machine learning in materials informatics: Recent applications and prospects. *npj Comput. Mater.* **3**, 54 (2017).
7. R. L. Gardas, J. A. P. Coutinho, Extension of the Ye and Shreeve group contribution method for density estimation of ionic liquids in a wide range of temperatures and pressures. *Fluid Phase Equilib.* **263**, 26–32 (2008).
8. K. Paduszyński, Extensive databases and group contribution QSPRs of ionic liquids properties. 2. Viscosity. *Ind. Eng. Chem. Res.* **58**, 17049–17066 (2019).
9. M. Barycki, A. Sosnowska, A. Gajewicz, M. Bobrowski, D. Wileńska, P. Skurski, et al., Temperature-dependent structure-property modeling of viscosity for ionic liquids. *Fluid Phase Equilib.* **427**, 9–17 (2016).
10. D. M. Makarov, Y. A. Fadeeva, L. E. Shmukler, I. V. Tetko, Benchmarking machine learning methods for modeling physical properties of ionic liquids. *J. Mol. Liq.* **351**, 118616 (2022).
11. X. Yu, End-to-end deep learning models for predicting the electrical conductivity of ionic liquids. *ACS Sustain. Chem. Eng.* (2026). DOI: 10.1021/acssuschemeng.6c07089.
12. Z. Chen, J. Chen, Prediction of electrical conductivity of ionic liquids: From COSMO-RS derived QSPR evaluation to boosting machine learning. *ACS Sustain. Chem. Eng.* **12**, 17749–17760 (2024). DOI: 10.1021/acssuschemeng.4c00307.
13. C. Song, C. Wang, F. Fang, G. Zhou, Z. Dai, Z. Yang, Large-scale screening for high conductivity ionic liquids via machine learning algorithm utilizing graph neural network-based features. *J. Chem. Eng. Data* **69**, 800–810 (2024). DOI: 10.1021/acs.jced.3c00709.
14. K. Baran, A. Kloskowski, Graph neural networks and structural information on ionic liquids: A cheminformatics study on molecular physicochemical property prediction. *J. Phys. Chem. B* **127**, 10542–10555 (2023). DOI: 10.1021/acs.jpcb.3c05521.
15. V. Venkatraman, S. Evjen, H. K. Knuutila, A. Fiksdahl, B. K. Alsberg, Predicting ionic liquid melting points using machine learning. *J. Mol. Liq.* **264**, 318–326 (2018).
16. F. Yerly, M. Blaise, S. Barras, Machine learning models for melting point prediction of ionic liquids: CatBoost approach. *CHIMIA* **77**, 625–629 (2023).
17. D. M. Eike, J. F. Brennecke, E. J. Maginn, Predicting melting points of quaternary ammonium ionic liquids. *Green Chem.* **5**, 323–328 (2003).
18. J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl, Neural message passing for quantum chemistry. *Proc. Mach. Learn. Res.* **70**, 1263–1272 (2017).
19. D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, R. P. Adams, Convolutional networks on graphs for learning molecular fingerprints. *Adv. Neural Inf. Process. Syst.* **28**, 2224–2232 (2015).
20. F. Lin, A structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties, submitted (2026).
21. F. Lin, Data density as the binding constraint for ionic-liquid property prediction, submitted (2026); preprint: ChemRxiv (2026).
22. F. Lin, Full-spectrum descriptors substitute for data density in ionic-liquid property prediction, submitted (2026).
23. A. Tropsha, P. Gramatica, V. K. Gombar, The importance of being earnest: Validation is the absolute essential for successful application and interpretation of QSPR models. *QSAR Comb. Sci.* **22**, 69–77 (2003).
24. I. Wallach, A. Heifets, Most ligand-based classification benchmarks reward memorization rather than generalization. *J. Chem. Inf. Model.* **58**, 916–932 (2018).
25. G. Landrum, RDKit: Open-source cheminformatics. https://www.rdkit.org.
26. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, et al., Scikit-learn: Machine learning in Python. *J. Mach. Learn. Res.* **12**, 2825–2830 (2011).
