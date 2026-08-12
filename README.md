# IL-Property-ML: A unified structure-based framework for honest, IL-disjoint prediction of ionic-liquid properties

**Author:** Fuxing Lin (蔺复兴) · Hunan Institute of Engineering, Xiangtan, Hunan, China

This repository contains the complete data, code, figures, and manuscript for the
submission *"A unified structure-based framework for honest, IL-disjoint prediction
of ionic-liquid properties."*

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

## Repository layout

```
IL-Property-ML/
├── data/
│   ├── il_props.db              # assembled multi-property IL database (SQLite)
│   └── perfold_results.csv      # per-fold validation results (real numbers)
├── code/
│   ├── model.py                 # main pipeline: featurization + MLR/GBM/MLP + GroupKFold
│   ├── multitask.py             # multi-task shared-trunk experiments
│   ├── figures.py               # official figure pipeline (single data path)
│   ├── fig3_properties.py       # per-property parity figure
│   ├── shap_sensitivity.py      # SHAP + temperature sensitivity analysis
│   ├── perfold_results.py       # per-fold result export
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

If you use this repository, please cite the associated manuscript (Zenodo DOI to
be assigned on first release).

## License

Code and data are released under CC-BY-4.0 (see [LICENSE](LICENSE)).
