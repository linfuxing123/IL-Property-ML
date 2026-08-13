#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_si.py — 第四篇 SI（Table S1 全指标 + Table S2 逐折 GNN）"""
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = pathlib.Path(__file__).resolve().parent
PROPS = ["conductivity", "density", "viscosity", "melting_point"]
LABEL = {"conductivity": "Conductivity (ln κ)", "density": "Density",
         "viscosity": "Viscosity (ln η)", "melting_point": "Melting point (K)"}


def main():
    res = pd.read_csv(ROOT / "gnn_results.csv")
    lines = [
        "# Supplementary Materials",
        "",
        "Graph neural networks and engineered descriptors trade on temperature in ionic-liquid property prediction",
        "",
        "## Table S1. Group-disjoint 5-fold metrics (R² / RMSE / MAE)",
        "",
        "| Property | Records | ILs | Model | R² | RMSE | MAE |",
        "|----------|---------|-----|-------|-----|------|-----|",
    ]
    for _, r in res.iterrows():
        p = r["prop"]
        lines.append(f"| {LABEL[p]} | {r['n']} | {r['n_il']} | 10-desc GBM | {r['gbm10_R2']:.4f} | — | {r['gbm10_MAE']:.4f} |")
        lines.append(f"|  |  |  | 458-desc GBM | {r['gbm458_R2']:.4f} | — | {r['gbm458_MAE']:.4f} |")
        oof = pd.read_csv(ROOT / f"oof_{p}.csv")
        rmse = float(np.sqrt(mean_squared_error(oof["truth"], oof["pred"])))
        lines.append(f"|  |  |  | GNN (MPNN) | {r['gnn_R2']:.4f} | {rmse:.4f} | {r['gnn_MAE']:.4f} |")
    lines += ["", "## Table S2. Per-fold GNN R² (fold-level stability)", "",
              "| Property | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |",
              "|----------|--------|--------|--------|--------|--------|-----------|"]
    for p in PROPS:
        oof = pd.read_csv(ROOT / f"oof_{p}.csv")
        fr = []
        for f in range(5):
            sub = oof[oof["fold"] == f]
            fr.append(r2_score(sub["truth"], sub["pred"]) if len(sub) else float("nan"))
        mean = float(np.nanmean(fr))
        sd = float(np.nanstd(fr))
        cells = " | ".join(f"{v:.4f}" for v in fr)
        lines.append(f"| {LABEL[p]} | {cells} | {mean:.4f} ± {sd:.4f} |")
    lines += ["",
              "Note: Mean is the arithmetic mean of the five fold-level R² values; the pooled R² in",
              "Table S1 is computed on concatenated out-of-fold predictions and differs slightly.",
              ""]
    lines += ["", "## Graph featurization",
              "",
              "Node features (29): element one-hot (13), degree one-hot (7), formal charge (1),",
              "hydrogen count (1), hybridization one-hot (4), aromaticity (1), ring membership (1),",
              "atomic mass/100 (1). Edge features (5): bond order one-hot (single/double/triple/aromatic)",
              "and conjugation (1). Cation and anion graphs are encoded by a 3-layer message-passing",
              "network (hidden 96, residual updates, mean/sum/max readout); embeddings are concatenated",
              "with standardized temperature (for temperature-dependent properties) and passed through a",
              "3-layer MLP head (576→128→64→1, dropout 0.2/0.1). Adam (lr 1e-3, weight decay 1e-4),",
              "MSE loss, 10% within-fold validation for early stopping (patience 20), 200-epoch budget.",
              "",
              "Reproducibility: all scripts in the companion GitHub repository (linfuxing123/IL-Property-ML)."]
    out = "\n".join(lines) + "\n"
    (ROOT / "supplementary_materials.md").write_text(out, encoding="utf-8")
    print("SI 已写入 supplementary_materials.md")
    print(out)


if __name__ == "__main__":
    main()
