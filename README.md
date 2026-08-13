# IL-Property-ML: A structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties

**Author:** Fuxing Lin (蔺复兴) · Hunan Institute of Engineering, Xiangtan, Hunan, China

This repository contains the complete data, code, figures, and manuscript for the
submission *"A structure-based framework for honest, IL-disjoint prediction
of ionic-liquid properties"* and its companion study *"Data density as the binding
constraint: a 7.7-fold expansion of ionic-liquid property data lifts group-disjoint
prediction from cold start to transferable accuracy."*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21898949.svg)](https://doi.org/10.5281/zenodo.21898949)

## What this work does

We assemble the largest consistently featurized multi-property ionic-liquid (IL)
database used in this work (11,511 records; 1,658 ion pairs) from three public
sources — ILest, iolitech, and ILThermo — and evaluate structure-based machine
learning (MLR / GBM / MLP) on cation/anion SMILES + temperature with RDKit
descriptors and Morgan fingerprints (2,069 features, all free and deterministic).

The central methodological point is **honest validation**: under IL-disjoint
(GroupKFold) splits, GBM reaches R² = 0.552 (conductivity, ln κ scale) and
R² = 0.828 (density), while paired point-wise controls reach R² = 0.908 for
conductivity — quantifying the inflation that conventional random point splits
introduce. Viscosity and melting point, with only one record per IL, fail under
IL-disjoint validation (cold-start), identifying **data density, not model
capacity, as the binding constraint**.

## Companion study (v1.1+)

The companion manuscript tests the data-density hypothesis directly: the database
is expanded 7.7-fold to **88,077 records / 1,891 unique ion pairs** by systematic
harvesting of the NIST ILThermo v2.0 repository (`data/ilt/`, four properties,
standardized units, verified SMILES). Under the identical model, features, and
IL-disjoint 5-fold protocol, viscosity rises from R² = −0.09 to 0.68,
conductivity from 0.55 to 0.70, density from 0.83 to 0.85, and melting point
from ≈0 to 0.39 (642 ILs). Key tools: `ilthermo_fetch.py` (concurrent downloader),
`ilt_validate.py` (export + honest validation), `ilt_merge_old.py` (legacy merge).
Manuscript, supplementary materials, and figures are in `manuscript/paper2/`
and `figures/paper2/`.

## Repository layout

```
IL-Property-ML/
├── data/
│   ├── il_props.db              # assembled multi-property IL database (SQLite)
│   └── perfold_results.csv      # per-fold validation results (real numbers)
│   └── ilt/                     # ILThermo v2.0 expanded dataset (4 properties,
│                                # 88,077 records, standardized units)
├── code/
│   ├── model.py                 # main pipeline: featurization + MLR/GBM/MLP + GroupKFold
│   ├── multitask.py             # multi-task shared-trunk experiments
│   ├── figures.py               # official figure pipeline (single data path)
│   ├── fig3_properties.py       # per-property parity figure
│   ├── shap_sensitivity.py      # SHAP + temperature sensitivity analysis
│   ├── perfold_results.py       # per-fold result export
│   ├── ilthermo_fetch.py        # ILThermo v2.0 concurrent downloader (companion)
│   ├── ilt_validate.py          # export + honest_cv/coverage validation (companion)
│   ├── ilt_merge_old.py         # legacy dataset merge (companion)
│   └── perfold_si.py            # per-fold SI results (companion)
├── manuscript/
│   └── paper2/                  # companion manuscript + SI + cover letter
├── figures/
│   └── paper2/                  # companion figures (300 dpi)
│   ├── il_db.py                 # database assembly helpers
│   ├── ilthermo_resolver.py     # ILThermo name → SMILES resolver
│   ├── chem_tools.py            # RDKit chemistry utilities
│   └── requirements.txt         # pinned Python dependencies
├── figures/                     # 300-dpi publication figures
├── manuscript/                  # manuscript + supplementary text (Markdown)
├── LICENSE                      # CC-BY-4.0
└── README.md
```

## Reproduce

```bash
# 1. environment (Python ≥3.10, all free packages)
pip install -r code/requirements.txt

# 2. train and evaluate the main pipeline (prints per-split and IL-level metrics)
python code/model.py

# 3. multi-task experiments
python code/multitask.py

# 4. regenerate all publication figures
python code/figures.py
python code/fig3_properties.py

# 5. SHAP analysis and temperature sensitivity
python code/shap_sensitivity.py

# 6. export per-fold results
python code/perfold_results.py
```

All scripts read directly from `data/il_props.db`; no commercial software is
required.

## Data sources

- ILest: https://ilest.nju.edu.cn/
- iolitech: https://iolitech.org/
- ILThermo (NIST): https://ilthermo.boulder.nist.gov/

## Citation

If you use this repository, please cite:

Lin, Fuxing (2026). IL-Property-ML: A structure-based framework for
honest, IL-disjoint prediction of ionic-liquid properties. Zenodo.
https://doi.org/10.5281/zenodo.21898949

## License

Code and data are released under CC-BY-4.0 (see [LICENSE](LICENSE)).
