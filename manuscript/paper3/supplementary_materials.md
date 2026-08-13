# Supplementary Materials for

## Full-spectrum descriptors substitute for data density in ionic-liquid property prediction

Fuxing Lin, Hunan Institute of Engineering

---

## S1. Data and descriptor construction

The model-ready dataset comprises 84,077 experimental records for 1,891 unique ion pairs across four properties (viscosity, density, electrical conductivity, melting point), compiled and unit-standardized in the companion study from ILThermo v2.0, ILest, and iolitech (K, mPa·s, g/cm³, S/m; verified SMILES). For every unique ion pair, the complete RDKit descriptor list was computed independently for the cation and the anion (229 per fragment; 458 total), and temperature was appended for the three temperature-dependent properties. Descriptor columns undefined for all ions were removed; remaining missing values were imputed by column median, then zero.

## S2. Complete stratified interaction table

All stratified results use 5-fold GroupKFold on IL identity with histogram gradient boosting (200 trees, default hyperparameters); metrics are computed on pooled out-of-fold predictions within each per-IL record-count bucket. `base` = ten hand-picked merged descriptors (+ temperature); `full` = 458 per-ion descriptors (+ temperature).

**Table S1. Group-disjoint R² within per-IL record-count buckets.**

| Property | Bucket | Set | R² | MAE | Samples | ILs |
|---|---|---|---|---|---|---|
| Conductivity (ln κ) | 1 | base | 0.7582 | 0.7680 | 149 | 149 |
| Conductivity (ln κ) | 1 | full | 0.7789 | 0.6642 | 149 | 149 |
| Conductivity (ln κ) | 2-3 | base | 0.0750 | 1.4667 | 32 | 14 |
| Conductivity (ln κ) | 2-3 | full | 0.8142 | 0.7711 | 32 | 14 |
| Conductivity (ln κ) | 4-9 | base | 0.6587 | 0.7073 | 1429 | 200 |
| Conductivity (ln κ) | 4-9 | full | 0.7964 | 0.5294 | 1429 | 200 |
| Conductivity (ln κ) | 10-24 | base | 0.6689 | 0.7035 | 2591 | 199 |
| Conductivity (ln κ) | 10-24 | full | 0.7387 | 0.5611 | 2591 | 199 |
| Conductivity (ln κ) | 25+ | base | 0.7264 | 0.7836 | 5269 | 79 |
| Conductivity (ln κ) | 25+ | full | 0.7392 | 0.6964 | 5269 | 79 |

| Density | 1 | base | 0.6933 | 0.0692 | 346 | 346 |
| Density | 1 | full | 0.7970 | 0.0455 | 346 | 346 |
| Density | 2-3 | base | 0.5900 | 0.0777 | 79 | 35 |
| Density | 2-3 | full | 0.7801 | 0.0519 | 79 | 35 |
| Density | 4-9 | base | 0.8554 | 0.0424 | 2691 | 378 |
| Density | 4-9 | full | 0.9046 | 0.0291 | 2691 | 378 |
| Density | 10-24 | base | 0.8636 | 0.0438 | 6013 | 430 |
| Density | 10-24 | full | 0.9438 | 0.0247 | 6013 | 430 |
| Density | 25+ | base | 0.8928 | 0.0380 | 39299 | 244 |
| Density | 25+ | full | 0.9532 | 0.0237 | 39299 | 244 |

| Viscosity (ln η) | 1 | base | 0.1398 | 0.8677 | 233 | 233 |
| Viscosity (ln η) | 1 | full | 0.5598 | 0.5300 | 233 | 233 |
| Viscosity (ln η) | 2-3 | base | 0.5458 | 0.6676 | 116 | 46 |
| Viscosity (ln η) | 2-3 | full | 0.8190 | 0.4123 | 116 | 46 |
| Viscosity (ln η) | 4-9 | base | 0.6707 | 0.6850 | 2567 | 355 |
| Viscosity (ln η) | 4-9 | full | 0.8417 | 0.4356 | 2567 | 355 |
| Viscosity (ln η) | 10-24 | base | 0.7300 | 0.6012 | 5584 | 412 |
| Viscosity (ln η) | 10-24 | full | 0.8721 | 0.3772 | 5584 | 412 |
| Viscosity (ln η) | 25+ | base | 0.7029 | 0.5399 | 16760 | 167 |
| Viscosity (ln η) | 25+ | full | 0.8163 | 0.3538 | 16760 | 167 |

| Melting point (K) | 1 | base | 0.3314 | 27.2836 | 540 | 540 |
| Melting point (K) | 1 | full | 0.4965 | 22.9242 | 540 | 540 |
| Melting point (K) | 2-3 | base | 0.5422 | 20.3595 | 168 | 74 |
| Melting point (K) | 2-3 | full | 0.6231 | 17.8301 | 168 | 74 |
| Melting point (K) | 4-9 | base | 0.1398 | 24.7149 | 125 | 22 |
| Melting point (K) | 4-9 | full | 0.4990 | 19.0934 | 125 | 22 |
| Melting point (K) | 10-24 | base | 0.8201 | 14.3153 | 86 | 6 |
| Melting point (K) | 10-24 | full | 0.8351 | 13.6893 | 86 | 6 |

## S3. Descriptor list

Each ion pair is represented by 229 cation descriptors and 229 anion descriptors (458 total), listed below. Temperature is appended for conductivity, density, and viscosity.

**Table S2. Cation descriptors.**

| # | Descriptor |
|---|---|
| 1 | smiles |
| 2 | ok |
| 3 | MaxAbsEStateIndex |
| 4 | MaxEStateIndex |
| 5 | MinAbsEStateIndex |
| 6 | MinEStateIndex |
| 7 | qed |
| 8 | SPS |
| 9 | MolWt |
| 10 | HeavyAtomMolWt |
| 11 | ExactMolWt |
| 12 | NumValenceElectrons |
| 13 | NumRadicalElectrons |
| 14 | MaxPartialCharge |
| 15 | MinPartialCharge |
| 16 | MaxAbsPartialCharge |
| 17 | MinAbsPartialCharge |
| 18 | FpDensityMorgan1 |
| 19 | FpDensityMorgan2 |
| 20 | FpDensityMorgan3 |
| 21 | BCUT2D_MWHI |
| 22 | BCUT2D_MWLOW |
| 23 | BCUT2D_CHGHI |
| 24 | BCUT2D_CHGLO |
| 25 | BCUT2D_LOGPHI |
| 26 | BCUT2D_LOGPLOW |
| 27 | BCUT2D_MRHI |
| 28 | BCUT2D_MRLOW |
| 29 | AvgIpc |
| 30 | BalabanJ |
| 31 | BertzCT |
| 32 | Chi0 |
| 33 | Chi0n |
| 34 | Chi0v |
| 35 | Chi1 |
| 36 | Chi1n |
| 37 | Chi1v |
| 38 | Chi2n |
| 39 | Chi2v |
| 40 | Chi3n |
| 41 | Chi3v |
| 42 | Chi4n |
| 43 | Chi4v |
| 44 | HallKierAlpha |
| 45 | Ipc |
| 46 | Kappa1 |
| 47 | Kappa2 |
| 48 | Kappa3 |
| 49 | LabuteASA |
| 50 | PEOE_VSA1 |
| 51 | PEOE_VSA10 |
| 52 | PEOE_VSA11 |
| 53 | PEOE_VSA12 |
| 54 | PEOE_VSA13 |
| 55 | PEOE_VSA14 |
| 56 | PEOE_VSA2 |
| 57 | PEOE_VSA3 |
| 58 | PEOE_VSA4 |
| 59 | PEOE_VSA5 |
| 60 | PEOE_VSA6 |
| 61 | PEOE_VSA7 |
| 62 | PEOE_VSA8 |
| 63 | PEOE_VSA9 |
| 64 | SMR_VSA1 |
| 65 | SMR_VSA10 |
| 66 | SMR_VSA2 |
| 67 | SMR_VSA3 |
| 68 | SMR_VSA4 |
| 69 | SMR_VSA5 |
| 70 | SMR_VSA6 |
| 71 | SMR_VSA7 |
| 72 | SMR_VSA8 |
| 73 | SMR_VSA9 |
| 74 | SlogP_VSA1 |
| 75 | SlogP_VSA10 |
| 76 | SlogP_VSA11 |
| 77 | SlogP_VSA12 |
| 78 | SlogP_VSA2 |
| 79 | SlogP_VSA3 |
| 80 | SlogP_VSA4 |
| 81 | SlogP_VSA5 |
| 82 | SlogP_VSA6 |
| 83 | SlogP_VSA7 |
| 84 | SlogP_VSA8 |
| 85 | SlogP_VSA9 |
| 86 | TPSA |
| 87 | EState_VSA1 |
| 88 | EState_VSA10 |
| 89 | EState_VSA11 |
| 90 | EState_VSA2 |
| 91 | EState_VSA3 |
| 92 | EState_VSA4 |
| 93 | EState_VSA5 |
| 94 | EState_VSA6 |
| 95 | EState_VSA7 |
| 96 | EState_VSA8 |
| 97 | EState_VSA9 |
| 98 | VSA_EState1 |
| 99 | VSA_EState10 |
| 100 | VSA_EState2 |
| 101 | VSA_EState3 |
| 102 | VSA_EState4 |
| 103 | VSA_EState5 |
| 104 | VSA_EState6 |
| 105 | VSA_EState7 |
| 106 | VSA_EState8 |
| 107 | VSA_EState9 |
| 108 | FractionCSP3 |
| 109 | HeavyAtomCount |
| 110 | NHOHCount |
| 111 | NOCount |
| 112 | NumAliphaticCarbocycles |
| 113 | NumAliphaticHeterocycles |
| 114 | NumAliphaticRings |
| 115 | NumAmideBonds |
| 116 | NumAromaticCarbocycles |
| 117 | NumAromaticHeterocycles |
| 118 | NumAromaticRings |
| 119 | NumAtomStereoCenters |
| 120 | NumBridgeheadAtoms |
| 121 | NumHAcceptors |
| 122 | NumHDonors |
| 123 | NumHeteroatoms |
| 124 | NumHeterocycles |
| 125 | NumRotatableBonds |
| 126 | NumSaturatedCarbocycles |
| 127 | NumSaturatedHeterocycles |
| 128 | NumSaturatedRings |
| 129 | NumSpiroAtoms |
| 130 | NumUnspecifiedAtomStereoCenters |
| 131 | Phi |
| 132 | RingCount |
| 133 | MolLogP |
| 134 | MolMR |
| 135 | fr_Al_COO |
| 136 | fr_Al_OH |
| 137 | fr_Al_OH_noTert |
| 138 | fr_ArN |
| 139 | fr_Ar_COO |
| 140 | fr_Ar_N |
| 141 | fr_Ar_NH |
| 142 | fr_Ar_OH |
| 143 | fr_COO |
| 144 | fr_COO2 |
| 145 | fr_C_O |
| 146 | fr_C_O_noCOO |
| 147 | fr_C_S |
| 148 | fr_HOCCN |
| 149 | fr_Imine |
| 150 | fr_NH0 |
| 151 | fr_NH1 |
| 152 | fr_NH2 |
| 153 | fr_N_O |
| 154 | fr_Ndealkylation1 |
| 155 | fr_Ndealkylation2 |
| 156 | fr_Nhpyrrole |
| 157 | fr_SH |
| 158 | fr_aldehyde |
| 159 | fr_alkyl_carbamate |
| 160 | fr_alkyl_halide |
| 161 | fr_allylic_oxid |
| 162 | fr_amide |
| 163 | fr_amidine |
| 164 | fr_aniline |
| 165 | fr_aryl_methyl |
| 166 | fr_azide |
| 167 | fr_azo |
| 168 | fr_barbitur |
| 169 | fr_benzene |
| 170 | fr_benzodiazepine |
| 171 | fr_bicyclic |
| 172 | fr_diazo |
| 173 | fr_dihydropyridine |
| 174 | fr_epoxide |
| 175 | fr_ester |
| 176 | fr_ether |
| 177 | fr_furan |
| 178 | fr_guanido |
| 179 | fr_halogen |
| 180 | fr_hdrzine |
| 181 | fr_hdrzone |
| 182 | fr_imidazole |
| 183 | fr_imide |
| 184 | fr_isocyan |
| 185 | fr_isothiocyan |
| 186 | fr_ketone |
| 187 | fr_ketone_Topliss |
| 188 | fr_lactam |
| 189 | fr_lactone |
| 190 | fr_methoxy |
| 191 | fr_morpholine |
| 192 | fr_nitrile |
| 193 | fr_nitro |
| 194 | fr_nitro_arom |
| 195 | fr_nitro_arom_nonortho |
| 196 | fr_nitroso |
| 197 | fr_oxazole |
| 198 | fr_oxime |
| 199 | fr_para_hydroxylation |
| 200 | fr_phenol |
| 201 | fr_phenol_noOrthoHbond |
| 202 | fr_phos_acid |
| 203 | fr_phos_ester |
| 204 | fr_piperdine |
| 205 | fr_piperzine |
| 206 | fr_priamide |
| 207 | fr_prisulfonamd |
| 208 | fr_pyridine |
| 209 | fr_quatN |
| 210 | fr_sulfide |
| 211 | fr_sulfonamd |
| 212 | fr_sulfone |
| 213 | fr_term_acetylene |
| 214 | fr_tetrazole |
| 215 | fr_thiazole |
| 216 | fr_thiocyan |
| 217 | fr_thiophene |
| 218 | fr_unbrch_alkane |
| 219 | fr_urea |
| 220 | core_ExactMolWt |
| 221 | core_CrippenLogP |
| 222 | core_NumHBD |
| 223 | core_NumHBA |
| 224 | core_TPSA |
| 225 | core_RotatableBonds |
| 226 | core_FractionCSP3 |
| 227 | core_Rings |
| 228 | core_AromaticRings |
| 229 | core_MR |

**Table S3. Anion descriptors.**

| # | Descriptor |
|---|---|
| 1 | smiles |
| 2 | ok |
| 3 | MaxAbsEStateIndex |
| 4 | MaxEStateIndex |
| 5 | MinAbsEStateIndex |
| 6 | MinEStateIndex |
| 7 | qed |
| 8 | SPS |
| 9 | MolWt |
| 10 | HeavyAtomMolWt |
| 11 | ExactMolWt |
| 12 | NumValenceElectrons |
| 13 | NumRadicalElectrons |
| 14 | MaxPartialCharge |
| 15 | MinPartialCharge |
| 16 | MaxAbsPartialCharge |
| 17 | MinAbsPartialCharge |
| 18 | FpDensityMorgan1 |
| 19 | FpDensityMorgan2 |
| 20 | FpDensityMorgan3 |
| 21 | BCUT2D_MWHI |
| 22 | BCUT2D_MWLOW |
| 23 | BCUT2D_CHGHI |
| 24 | BCUT2D_CHGLO |
| 25 | BCUT2D_LOGPHI |
| 26 | BCUT2D_LOGPLOW |
| 27 | BCUT2D_MRHI |
| 28 | BCUT2D_MRLOW |
| 29 | AvgIpc |
| 30 | BalabanJ |
| 31 | BertzCT |
| 32 | Chi0 |
| 33 | Chi0n |
| 34 | Chi0v |
| 35 | Chi1 |
| 36 | Chi1n |
| 37 | Chi1v |
| 38 | Chi2n |
| 39 | Chi2v |
| 40 | Chi3n |
| 41 | Chi3v |
| 42 | Chi4n |
| 43 | Chi4v |
| 44 | HallKierAlpha |
| 45 | Ipc |
| 46 | Kappa1 |
| 47 | Kappa2 |
| 48 | Kappa3 |
| 49 | LabuteASA |
| 50 | PEOE_VSA1 |
| 51 | PEOE_VSA10 |
| 52 | PEOE_VSA11 |
| 53 | PEOE_VSA12 |
| 54 | PEOE_VSA13 |
| 55 | PEOE_VSA14 |
| 56 | PEOE_VSA2 |
| 57 | PEOE_VSA3 |
| 58 | PEOE_VSA4 |
| 59 | PEOE_VSA5 |
| 60 | PEOE_VSA6 |
| 61 | PEOE_VSA7 |
| 62 | PEOE_VSA8 |
| 63 | PEOE_VSA9 |
| 64 | SMR_VSA1 |
| 65 | SMR_VSA10 |
| 66 | SMR_VSA2 |
| 67 | SMR_VSA3 |
| 68 | SMR_VSA4 |
| 69 | SMR_VSA5 |
| 70 | SMR_VSA6 |
| 71 | SMR_VSA7 |
| 72 | SMR_VSA8 |
| 73 | SMR_VSA9 |
| 74 | SlogP_VSA1 |
| 75 | SlogP_VSA10 |
| 76 | SlogP_VSA11 |
| 77 | SlogP_VSA12 |
| 78 | SlogP_VSA2 |
| 79 | SlogP_VSA3 |
| 80 | SlogP_VSA4 |
| 81 | SlogP_VSA5 |
| 82 | SlogP_VSA6 |
| 83 | SlogP_VSA7 |
| 84 | SlogP_VSA8 |
| 85 | SlogP_VSA9 |
| 86 | TPSA |
| 87 | EState_VSA1 |
| 88 | EState_VSA10 |
| 89 | EState_VSA11 |
| 90 | EState_VSA2 |
| 91 | EState_VSA3 |
| 92 | EState_VSA4 |
| 93 | EState_VSA5 |
| 94 | EState_VSA6 |
| 95 | EState_VSA7 |
| 96 | EState_VSA8 |
| 97 | EState_VSA9 |
| 98 | VSA_EState1 |
| 99 | VSA_EState10 |
| 100 | VSA_EState2 |
| 101 | VSA_EState3 |
| 102 | VSA_EState4 |
| 103 | VSA_EState5 |
| 104 | VSA_EState6 |
| 105 | VSA_EState7 |
| 106 | VSA_EState8 |
| 107 | VSA_EState9 |
| 108 | FractionCSP3 |
| 109 | HeavyAtomCount |
| 110 | NHOHCount |
| 111 | NOCount |
| 112 | NumAliphaticCarbocycles |
| 113 | NumAliphaticHeterocycles |
| 114 | NumAliphaticRings |
| 115 | NumAmideBonds |
| 116 | NumAromaticCarbocycles |
| 117 | NumAromaticHeterocycles |
| 118 | NumAromaticRings |
| 119 | NumAtomStereoCenters |
| 120 | NumBridgeheadAtoms |
| 121 | NumHAcceptors |
| 122 | NumHDonors |
| 123 | NumHeteroatoms |
| 124 | NumHeterocycles |
| 125 | NumRotatableBonds |
| 126 | NumSaturatedCarbocycles |
| 127 | NumSaturatedHeterocycles |
| 128 | NumSaturatedRings |
| 129 | NumSpiroAtoms |
| 130 | NumUnspecifiedAtomStereoCenters |
| 131 | Phi |
| 132 | RingCount |
| 133 | MolLogP |
| 134 | MolMR |
| 135 | fr_Al_COO |
| 136 | fr_Al_OH |
| 137 | fr_Al_OH_noTert |
| 138 | fr_ArN |
| 139 | fr_Ar_COO |
| 140 | fr_Ar_N |
| 141 | fr_Ar_NH |
| 142 | fr_Ar_OH |
| 143 | fr_COO |
| 144 | fr_COO2 |
| 145 | fr_C_O |
| 146 | fr_C_O_noCOO |
| 147 | fr_C_S |
| 148 | fr_HOCCN |
| 149 | fr_Imine |
| 150 | fr_NH0 |
| 151 | fr_NH1 |
| 152 | fr_NH2 |
| 153 | fr_N_O |
| 154 | fr_Ndealkylation1 |
| 155 | fr_Ndealkylation2 |
| 156 | fr_Nhpyrrole |
| 157 | fr_SH |
| 158 | fr_aldehyde |
| 159 | fr_alkyl_carbamate |
| 160 | fr_alkyl_halide |
| 161 | fr_allylic_oxid |
| 162 | fr_amide |
| 163 | fr_amidine |
| 164 | fr_aniline |
| 165 | fr_aryl_methyl |
| 166 | fr_azide |
| 167 | fr_azo |
| 168 | fr_barbitur |
| 169 | fr_benzene |
| 170 | fr_benzodiazepine |
| 171 | fr_bicyclic |
| 172 | fr_diazo |
| 173 | fr_dihydropyridine |
| 174 | fr_epoxide |
| 175 | fr_ester |
| 176 | fr_ether |
| 177 | fr_furan |
| 178 | fr_guanido |
| 179 | fr_halogen |
| 180 | fr_hdrzine |
| 181 | fr_hdrzone |
| 182 | fr_imidazole |
| 183 | fr_imide |
| 184 | fr_isocyan |
| 185 | fr_isothiocyan |
| 186 | fr_ketone |
| 187 | fr_ketone_Topliss |
| 188 | fr_lactam |
| 189 | fr_lactone |
| 190 | fr_methoxy |
| 191 | fr_morpholine |
| 192 | fr_nitrile |
| 193 | fr_nitro |
| 194 | fr_nitro_arom |
| 195 | fr_nitro_arom_nonortho |
| 196 | fr_nitroso |
| 197 | fr_oxazole |
| 198 | fr_oxime |
| 199 | fr_para_hydroxylation |
| 200 | fr_phenol |
| 201 | fr_phenol_noOrthoHbond |
| 202 | fr_phos_acid |
| 203 | fr_phos_ester |
| 204 | fr_piperdine |
| 205 | fr_piperzine |
| 206 | fr_priamide |
| 207 | fr_prisulfonamd |
| 208 | fr_pyridine |
| 209 | fr_quatN |
| 210 | fr_sulfide |
| 211 | fr_sulfonamd |
| 212 | fr_sulfone |
| 213 | fr_term_acetylene |
| 214 | fr_tetrazole |
| 215 | fr_thiazole |
| 216 | fr_thiocyan |
| 217 | fr_thiophene |
| 218 | fr_unbrch_alkane |
| 219 | fr_urea |
| 220 | core_ExactMolWt |
| 221 | core_CrippenLogP |
| 222 | core_NumHBD |
| 223 | core_NumHBA |
| 224 | core_TPSA |
| 225 | core_RotatableBonds |
| 226 | core_FractionCSP3 |
| 227 | core_Rings |
| 228 | core_AromaticRings |
| 229 | core_MR |

## S4. Reproducibility

Environment: Python 3.12, pandas 3.0.5, scikit-learn 1.9.0, RDKit 2026.03.5. Models: `GradientBoostingRegressor(random_state=0)` for the overall comparison (Table 1 of the main text) and `HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)` for the stratified analysis (Fig. 1 and Table S1); 5-fold `GroupKFold` on IL identity. Viscosity and conductivity targets are log-transformed. Scripts: `feat_scale_exp.py` (overall comparison) and `feat_density_interaction.py` (stratified interaction); descriptors generated by `il_descriptors.py`. Data and code are available at GitHub (linfuxing123/IL-Property-ML, v1.2.0) and archived on Zenodo (concept DOI 10.5281/zenodo.21898948).
