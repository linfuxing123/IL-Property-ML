# Supplementary Materials

Graph neural networks and engineered descriptors trade on temperature in ionic-liquid property prediction

## Table S1. Group-disjoint 5-fold metrics (R² / RMSE / MAE)

| Property | Records | ILs | Model | R² | RMSE | MAE |
|----------|---------|-----|-------|-----|------|-----|
| Conductivity (ln κ) | 9470 | 641 | 10-desc GBM | 0.6978 | — | 0.8071 |
|  |  |  | 458-desc GBM | 0.7404 | — | 0.7051 |
|  |  |  | GNN (MPNN) | 0.7507 | 1.2972 | 0.7691 |
| Density | 48428 | 1433 | 10-desc GBM | 0.8499 | — | 0.0480 |
|  |  |  | 458-desc GBM | 0.9260 | — | 0.0295 |
|  |  |  | GNN (MPNN) | 0.9408 | 0.0415 | 0.0286 |
| Viscosity (ln η) | 25260 | 1213 | 10-desc GBM | 0.6763 | — | 0.6365 |
|  |  |  | 458-desc GBM | 0.8085 | — | 0.4580 |
|  |  |  | GNN (MPNN) | 0.7804 | 0.7267 | 0.5073 |
| Melting point (K) | 919 | 642 | 10-desc GBM | 0.3972 | — | 24.6128 |
|  |  |  | 458-desc GBM | 0.5233 | — | 21.7230 |
|  |  |  | GNN (MPNN) | 0.3989 | 33.4020 | 24.8494 |

## Table S2. Per-fold GNN R² (fold-level stability)

| Property | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
|----------|--------|--------|--------|--------|--------|-----------|
| Conductivity (ln κ) | 0.7927 | 0.8388 | 0.6942 | 0.7735 | 0.6377 | 0.7474 ± 0.0721 |
| Density | 0.9320 | 0.9345 | 0.9427 | 0.9339 | 0.9590 | 0.9404 ± 0.0100 |
| Viscosity (ln η) | 0.7783 | 0.7753 | 0.7632 | 0.7972 | 0.7768 | 0.7782 ± 0.0109 |
| Melting point (K) | 0.3347 | 0.4529 | 0.3743 | 0.2222 | 0.5151 | 0.3799 ± 0.1006 |

Note: Mean is the arithmetic mean of the five fold-level R² values; the pooled R² in
Table S1 is computed on concatenated out-of-fold predictions and differs slightly.


## Graph featurization

Node features (29): element one-hot (13), degree one-hot (7), formal charge (1),
hydrogen count (1), hybridization one-hot (4), aromaticity (1), ring membership (1),
atomic mass/100 (1). Edge features (5): bond order one-hot (single/double/triple/aromatic)
and conjugation (1). Cation and anion graphs are encoded by a 3-layer message-passing
network (hidden 96, residual updates, mean/sum/max readout); embeddings are concatenated
with standardized temperature (for temperature-dependent properties) and passed through a
3-layer MLP head (576→128→64→1, dropout 0.2/0.1). Adam (lr 1e-3, weight decay 1e-4),
MSE loss, 10% within-fold validation for early stopping (patience 20), 200-epoch budget.

Reproducibility: all scripts in the companion GitHub repository (linfuxing123/IL-Property-ML).
