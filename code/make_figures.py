#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures.py — 第四篇图（GNN vs 描述符）

读 gnn_results.csv + oof_<prop>.csv：
  fig1_models_comparison.png  三模型 × 四性质 R² 分组柱状图（GNN 带逐折 std 误差棒）
  fig2_parity.png             四性质 2×2 parity 散点（GNN 预测 vs 实测）
  fig3_density_gap.png         GNN-vs-458 差距 vs 数据密度（对数记录数）
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

PROPS = ["conductivity", "density", "viscosity", "melting_point"]
LABEL = {"conductivity": "Conductivity (ln κ)",
         "density": "Density",
         "viscosity": "Viscosity (ln η)",
         "melting_point": "Melting point"}
COLORS = {"gbm10": "#9aa0a6", "gbm458": "#4285f4", "gnn": "#ea4335"}


def fig1(res):
    x = np.arange(len(PROPS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
    for j, (key, col, lab) in enumerate([
            ("gbm10_R2", COLORS["gbm10"], "10 descriptors (GBM)"),
            ("gbm458_R2", COLORS["gbm458"], "458 descriptors (GBM)"),
            ("gnn_R2", COLORS["gnn"], "Graph neural network")]):
        vals = [res.loc[res.prop == p, key].iloc[0] for p in PROPS]
        err = None
        if key == "gnn_R2":
            err = [res.loc[res.prop == p, "gnn_R2_std"].iloc[0] for p in PROPS]
        ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=3, label=lab,
               color=col, edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + (j - 1) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in PROPS], fontsize=8)
    ax.set_ylabel("Group-disjoint R²", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_models_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2(res):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.8), dpi=300)
    for ax, p in zip(axes.ravel(), PROPS):
        oof = pd.read_csv(ROOT / f"oof_{p}.csv")
        r2 = res.loc[res.prop == p, "gnn_R2"].iloc[0]
        ax.scatter(oof["truth"], oof["pred"], s=4, alpha=0.18, color=COLORS["gnn"],
                   edgecolors="none", rasterized=True)
        lo = min(oof["truth"].min(), oof["pred"].min())
        hi = max(oof["truth"].max(), oof["pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("Measured", fontsize=8)
        ax.set_ylabel("Predicted (GNN)", fontsize=8)
        ax.set_title(f"{LABEL[p]}  (R² = {r2:.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_parity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3(res):
    rows = []
    for _, r in res.iterrows():
        rows.append({"prop": r["prop"], "n": r["n"], "dR2": r["dR2_gnn_vs_458"]})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    ax.scatter(d["n"], d["dR2"], s=80, color=COLORS["gnn"], zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for _, r in d.iterrows():
        ax.annotate(LABEL[r["prop"]], (r["n"], r["dR2"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Records (log scale)", fontsize=9)
    ax.set_ylabel("ΔR² (GNN − 458-descriptor GBM)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_density_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    res = pd.read_csv(ROOT / "gnn_results.csv")
    fig1(res)
    fig2(res)
    fig3(res)
    print("图已生成:", FIG)


if __name__ == "__main__":
    main()
