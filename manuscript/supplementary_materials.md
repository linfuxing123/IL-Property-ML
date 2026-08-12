# Supplementary Materials

## A unified structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties

**Fuxing Lin**, Hunan Institute of Engineering

Correspondence: 3612411485@qq.com

**This PDF file includes:**

- Supplementary text S1 to S6
- Figs. S1 and S2
- Tables S1 to S7
- References (19 to 21)

---

## S1. Data assembly

### S1.1. Sources

The database (`il_props.db`) was assembled from five public tabular compilations
spanning three primary sources (main-text refs. 14–16): the ILest mixture-conductivity
repository (8,035 records, stored as three tabular compilations), three ILThermo-derived
conductivity tables resolved to ion-level SMILES by a purpose-built parser (3,036 records),
and the iolitech pure-IL compilation (440 records covering conductivity, viscosity,
density, and melting point). Table S2 summarizes the sources.

### S1.2. ILThermo name-to-SMILES resolution

ILThermo tables report ion identities as names rather than structures. A purpose-built
parser (`ilthermo_resolver.py`) maps each cation/anion name to a canonical SMILES string
through a hand-curated vocabulary, and every mapping is validated by comparing the
RDKit-computed molecular formula of the resolved SMILES with the formula string in the
original ILThermo record. Mappings that fail formula validation are rejected rather than
silently retained. This resolution step increased the number of ILThermo ionic liquids
with usable SMILES from 74 to 97, and added 3,036 conductivity records to the database.

### S1.3. Deduplication and curation

Records were deduplicated on (cation SMILES, anion SMILES, temperature, value,
property). Model-ready subsets contain only pure-IL records (mole fraction = 1) with
valid ion SMILES and positive property values. Conductivity is stored in S m⁻¹,
viscosity in mPa·s, density in g cm⁻³, and melting point in K.

## S2. Descriptor representation

Each ion is encoded by ten freely computable RDKit 2D descriptors (table S3) concatenated
with a 1024-bit ECFP4 Morgan fingerprint (radius 2, bit-vector form) (main-text ref. 17).
The full input for a record is

$$x = [\Phi(c^+) \oplus \Phi(a^-) \oplus x_{IL} \oplus T] \in \mathbb{R}^{2070},$$

where Φ is the 1034-dimensional ion map (10 descriptors + 1024 fingerprint bits),
x_IL is the IL-level mole fraction, and T is temperature in K. No commercial descriptor
package and no continuum solvation calculation is used. Featurization succeeds for
3,242 of 3,268 conductivity records (26 records lost to SMILES parsing), at
sub-millisecond cost per record on a single CPU core.

## S3. Models and hyperparameters

Three model families are compared, implemented with scikit-learn (ref. 19):
multiple linear regression (MLR); histogram-based gradient boosting
(`HistGradientBoostingRegressor`); and a multilayer perceptron
(`MLPRegressor`, 128–64–16 hidden units, ReLU, early stopping). Positive-skewed
targets (conductivity, viscosity) are modeled in log space; density and melting point
in original units. The multi-task extension is a PyTorch network (ref. 20) with a
shared 128–64 trunk and per-property heads, trained with missing-target masking and a
weighted mean-squared-error loss. Table S4 lists the final hyperparameters.

## S4. Validation protocols and metrics

IL-disjoint validation uses 5-fold GroupKFold with the (cation, anion) pair as the
group key, so every record of a given ionic liquid is confined to a single fold and
test folds contain only completely unseen ion pairs. Point-wise controls use 5-fold
shuffled KFold with the same folds across models. Metrics are pooled over all folds:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}, \quad
RMSE = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}, \quad
MAE = \frac{1}{n}\sum_i |y_i - \hat{y}_i|.$$

Pooled metrics are reported in table S1; per-fold metrics in table S5.

## S5. SHAP analysis and temperature sensitivity

SHAP values were computed with the tree explainer on the gradient-boosting conductivity
model (300 background samples) (main-text ref. 18). Fig. S1 shows the beeswarm plot for
the top 15 features; table S6 lists their mean |SHAP| values. Temperature is the dominant
feature (mean |SHAP| = 1.665), followed by anion lipophilicity (LogP, 0.228) and polarity
(TPSA, 0.215), then cation size descriptors. A perturbation analysis on the test records
shows that a ±10 K change in temperature changes the predicted ln κ by 0.526 on average
(≈69% relative change in κ), quantifying the thermal sensitivity of ionic conductivity.

## S6. Multi-task benchmark

The multi-task MLP (shared trunk + per-property heads, missing-target masking) was
benchmarked against single-task training under the identical IL-disjoint protocol.
Fig. S2 and table S7 show that shared training yields no significant improvement at the
current data scale, identifying data coverage rather than representation capacity as the
binding constraint for multi-property extrapolation.

---

## Table S1. Full model performance under point-wise and IL-disjoint validation (5-fold)

| Property (scale) | Records (ILs) | Model | R² (IL) | R² (point) | RMSE | MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Conductivity (ln κ) | 3,242 (216) | GBM | 0.552 | 0.908 | 2.252 | 1.185 |
| | | MLP | 0.329 | 0.918 | 2.754 | 1.401 |
| | | MLR | −0.099 | 0.438 | 3.526 | 2.421 |
| Density (g cm⁻³) | 102 (101) | GBM | 0.828 | 0.752 | 0.083 | 0.057 |
| Viscosity (ln η) | 59 (58) | GBM | −0.096 | −0.125 | 0.491 | 0.419 |
| Melting point (K) | 61 (61) | GBM | −0.086 | 0.022 | 22.2 | 18.5 |

Metrics are pooled over the five folds on the modeling scale of each property.

## Table S2. Data sources

| Source | Records | Properties | Access |
| --- | ---: | --- | --- |
| ILest (mixture conductivity; three tabular compilations) | 8,035 | κ | https://ilest.uobabylon.edu.iq |
| ILThermo (conductivity; purpose-built parser) | 3,036 | κ | https://ilthermo.boulder.nist.gov |
| iolitech (pure ILs) | 440 | κ, η, ρ, T_m | https://iolitech.de |
| Total | 11,511 | 4 properties, 1,658 ion pairs | `data/il_props.db` |

## Table S3. RDKit 2D descriptors used for each ion

| # | Descriptor | Description | RDKit function |
| ---: | --- | --- | --- |
| 1 | MW | Exact molecular weight | `CalcExactMolWt` |
| 2 | LogP | Wildman–Crippen octanol–water partition coefficient | `CalcCrippenDescriptors` |
| 3 | HBD | Number of H-bond donors | `CalcNumHBD` |
| 4 | HBA | Number of H-bond acceptors | `CalcNumHBA` |
| 5 | TPSA | Topological polar surface area (Å²) | `CalcTPSA` |
| 6 | RotatableBonds | Number of rotatable bonds | `CalcNumRotatableBonds` |
| 7 | FractionCsp3 | Fraction of sp³ carbons | `CalcFractionCSP3` |
| 8 | RingCount | Number of rings | `CalcNumRings` |
| 9 | AromaticRings | Number of aromatic rings | `CalcNumAromaticRings` |
| 10 | MR | Molar refractivity | `Descriptors.MolMR` |

Each descriptor vector is concatenated with a 1024-bit ECFP4 Morgan fingerprint
(radius 2) per ion.

## Table S4. Model hyperparameters

| Model | Hyperparameter | Value |
| --- | --- | --- |
| MLR | — | Ordinary least squares, no regularization |
| HistGBM | max_iter / learning_rate / max_depth | 300 / 0.08 / 6 |
| HistGBM | early_stopping | on (10% validation, 10 rounds) |
| HistGBM | random_state | 42 |
| MLP | hidden_layer_sizes | (128, 64, 16) |
| MLP | activation / solver | ReLU / Adam |
| MLP | max_iter / early_stopping | 300 / on |
| MLP | random_state | 42 |
| Multi-task MLP | shared trunk | 128 → 64 |
| Multi-task MLP | per-property heads | 64 → 1 (one head per property) |
| Multi-task MLP | loss | masked weighted MSE (log scale for κ, η) |
| Multi-task MLP | gradient clipping | 1.0 |

## Table S5. Per-fold results (pooled metrics in table S1)

| Property | Model | Split | Fold | n | R² | RMSE | MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| conductivity | mlr | point | 1 | 649 | 0.415 | 2.780 | 1.583 |
| conductivity | mlr | point | 2 | 649 | 0.480 | 2.415 | 1.461 |
| conductivity | mlr | point | 3 | 648 | 0.448 | 2.383 | 1.504 |
| conductivity | mlr | point | 4 | 648 | 0.458 | 2.544 | 1.551 |
| conductivity | mlr | point | 5 | 648 | 0.380 | 2.462 | 1.524 |
| conductivity | mlr | group | 1 | 649 | -3.128 | 2.810 | 2.419 |
| conductivity | mlr | group | 2 | 649 | 0.208 | 3.904 | 2.668 |
| conductivity | mlr | group | 3 | 648 | 0.084 | 3.743 | 2.245 |
| conductivity | mlr | group | 4 | 648 | -1.038 | 3.303 | 2.180 |
| conductivity | mlr | group | 5 | 648 | -0.045 | 3.754 | 2.593 |
| conductivity | gbm | point | 1 | 649 | 0.904 | 1.125 | 0.608 |
| conductivity | gbm | point | 2 | 649 | 0.912 | 0.992 | 0.488 |
| conductivity | gbm | point | 3 | 648 | 0.912 | 0.949 | 0.497 |
| conductivity | gbm | point | 4 | 648 | 0.930 | 0.915 | 0.491 |
| conductivity | gbm | point | 5 | 648 | 0.879 | 1.089 | 0.551 |
| conductivity | gbm | group | 1 | 649 | 0.314 | 1.146 | 0.689 |
| conductivity | gbm | group | 2 | 649 | 0.726 | 2.295 | 1.483 |
| conductivity | gbm | group | 3 | 648 | 0.587 | 2.513 | 1.116 |
| conductivity | gbm | group | 4 | 648 | 0.463 | 1.695 | 1.038 |
| conductivity | gbm | group | 5 | 648 | 0.289 | 3.097 | 1.598 |
| conductivity | mlp | point | 1 | 649 | 0.921 | 1.020 | 0.590 |
| conductivity | mlp | point | 2 | 649 | 0.895 | 1.086 | 0.651 |
| conductivity | mlp | point | 3 | 648 | 0.912 | 0.950 | 0.570 |
| conductivity | mlp | point | 4 | 648 | 0.912 | 1.024 | 0.570 |
| conductivity | mlp | point | 5 | 648 | 0.906 | 0.958 | 0.544 |
| conductivity | mlp | group | 1 | 649 | 0.296 | 1.161 | 0.875 |
| conductivity | mlp | group | 2 | 649 | 0.883 | 1.499 | 1.047 |
| conductivity | mlp | group | 3 | 648 | 0.645 | 2.328 | 1.067 |
| conductivity | mlp | group | 4 | 648 | 0.119 | 2.173 | 1.355 |
| conductivity | mlp | group | 5 | 648 | -0.070 | 3.799 | 1.891 |
| density | gbm | point | 1 | 21 | 0.766 | 0.084 | 0.064 |
| density | gbm | point | 2 | 21 | 0.867 | 0.054 | 0.041 |
| density | gbm | point | 3 | 20 | 0.911 | 0.050 | 0.040 |
| density | gbm | point | 4 | 20 | 0.444 | 0.135 | 0.096 |
| density | gbm | point | 5 | 20 | 0.809 | 0.064 | 0.047 |
| density | gbm | group | 1 | 21 | 0.881 | 0.050 | 0.037 |
| density | gbm | group | 2 | 21 | 0.764 | 0.076 | 0.061 |
| density | gbm | group | 3 | 20 | 0.835 | 0.067 | 0.046 |
| density | gbm | group | 4 | 20 | 0.798 | 0.076 | 0.055 |
| density | gbm | group | 5 | 20 | 0.853 | 0.074 | 0.053 |
| viscosity | gbm | point | 1 | 12 | -0.165 | 0.373 | 0.323 |
| viscosity | gbm | point | 2 | 12 | -0.103 | 0.551 | 0.439 |
| viscosity | gbm | point | 3 | 12 | -0.169 | 0.604 | 0.546 |
| viscosity | gbm | point | 4 | 12 | -0.062 | 0.389 | 0.323 |
| viscosity | gbm | point | 5 | 11 | -1.067 | 0.529 | 0.487 |
| viscosity | gbm | group | 1 | 12 | -0.158 | 0.578 | 0.517 |
| viscosity | gbm | group | 2 | 12 | -0.330 | 0.484 | 0.409 |
| viscosity | gbm | group | 3 | 12 | -0.873 | 0.458 | 0.430 |
| viscosity | gbm | group | 4 | 12 | -0.222 | 0.437 | 0.345 |
| viscosity | gbm | group | 5 | 11 | 0.099 | 0.484 | 0.388 |
| melting_point | gbm | point | 1 | 13 | -0.387 | 22.362 | 19.183 |
| melting_point | gbm | point | 2 | 12 | -0.112 | 22.082 | 17.531 |
| melting_point | gbm | point | 3 | 12 | 0.296 | 17.736 | 14.329 |
| melting_point | gbm | point | 4 | 12 | -0.038 | 24.831 | 21.446 |
| melting_point | gbm | point | 5 | 12 | 0.161 | 23.107 | 20.014 |
| melting_point | gbm | group | 1 | 13 | -0.270 | 26.701 | 22.062 |
| melting_point | gbm | group | 2 | 12 | -0.127 | 17.317 | 13.596 |
| melting_point | gbm | group | 3 | 12 | 0.115 | 23.478 | 20.243 |
| melting_point | gbm | group | 4 | 12 | -0.218 | 23.056 | 18.325 |
| melting_point | gbm | group | 5 | 12 | -0.028 | 24.780 | 21.522 |

## Table S6. SHAP mean |SHAP| ranking (top 15, conductivity GBM)

| Rank | Feature | Mean \|SHAP\| |
| ---: | --- | ---: |
| 1 | T(K) | 1.6654 |
| 2 | A_LogP | 0.2281 |
| 3 | A_TPSA | 0.2147 |
| 4 | C_MW | 0.1994 |
| 5 | C_RotatableBonds | 0.1913 |
| 6 | C_MR | 0.1464 |
| 7 | C_FractionCsp3 | 0.0641 |
| 8 | A_MW | 0.0545 |
| 9 | C_LogP | 0.0509 |
| 10 | ECFPan_33 | 0.0429 |
| 11 | ECFPcat_887 | 0.0402 |
| 12 | ECFPan_360 | 0.0307 |
| 13 | ECFPan_210 | 0.0304 |
| 14 | ECFPan_790 | 0.0297 |
| 15 | A_MR | 0.0296 |

## Table S7. Multi-task versus single-task under IL-disjoint validation

| Property | Multi-task R² | Single-task R² |
| --- | ---: | ---: |
| Conductivity | −0.876 | −0.620 |
| Density | −0.296 | −0.210 |
| Viscosity | −1.210 | −1.404 |
| Melting point | −0.363 | −0.466 |

---

## Fig. S1. SHAP beeswarm plot (top 15 features, conductivity GBM)

![Fig. S1](D:\Codex\MEC-Workspace\workspace\science-paper\supplementary\figS1_shap.png)

## Fig. S2. Multi-task versus single-task under IL-disjoint validation

![Fig. S2](D:\Codex\MEC-Workspace\workspace\science-paper\supplementary\figS2_multitask.png)

---

## Code availability and reproducibility

All scripts are provided in `workspace/matmodel/`:

| Script | Purpose |
| --- | --- |
| `model.py` | featurization + MLR/GBM/MLP + point/IL/Kennard–Stone splits |
| `multitask.py` | PyTorch multi-task MLP with missing-target masking |
| `shap_sensitivity.py` | SHAP top-20 + ±10 K temperature sensitivity |
| `figures.py` | main-text figures (figs. 1 and 2) |
| `fig3_properties.py` | main-text fig. 3 (four-property IL-level scatter) |
| `perfold_results.py` | per-fold results (table S5) |
| `ilthermo_resolver.py` | ILThermo ion-name → SMILES resolution + formula validation |
| `il_db.py` | database construction (`il_props.db`) |

Reproduction: create the pinned environment (Python 3.12, RDKit 2026.03.5,
numpy/scipy/pandas/scikit-learn 1.9.0/torch 2.13.0+cpu/shap 0.52.0/matplotlib),
then run, in order: `il_db.py` (database), `model.py` (baseline results),
`multitask.py` (multi-task results), `shap_sensitivity.py` (interpretation),
`perfold_results.py` (per-fold results), and the three figure scripts.

## References

19. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel,
    P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau,
    M. Brucher, M. Perrot, E. Duchesnay, Scikit-learn: Machine learning in Python.
    *J. Mach. Learn. Res.* **12**, 2825–2830 (2011).
20. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin,
    N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison,
    A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, PyTorch:
    An imperative style, high-performance deep learning library.
    *Adv. Neural Inf. Process. Syst.* **32**, 8026–8037 (2019).
21. C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau,
    E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk,
    M. Brett, A. Haldane, J. F. del Rí­o, M. Wiebe, P. Peterson, P. Gérard-Marchant, K. Sheppard,
    T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, T. E. Oliphant, Array programming with NumPy.
    *Nature* **585**, 357–362 (2020).
