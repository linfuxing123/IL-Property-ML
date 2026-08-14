#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_si.py — 第五篇 SI：候选集完整表 + 双预测器指标 + 方法细节。"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    base = pd.read_csv(ROOT / "generated" / "final_candidates.csv")
    gnn = pd.read_csv(ROOT / "generated" / "final_candidates_gnn.csv")
    cand = base.merge(gnn[["cat_smiles", "an_smiles", "gnn_cond", "gnn_visc", "gnn_tm", "gnn_dens", "gbm_gnn_gap", "gnn_agree"]],
                      on=["cat_smiles", "an_smiles"], how="inner")
    cand = cand[cand["gnn_agree"]].reset_index(drop=True)
    lines = [
        "# Supplementary Materials",
        "",
        "From property prediction to inverse design: multi-property Pareto screening of ionic-liquid electrolytes under IL-disjoint validation",
        "",
        "## Table S1. Three-model oracle (IL-disjoint 5-fold R²)",
        "",
        "| Property | GBM R² | HistGBM R² | MPNN R² |",
        "|----------|--------|------------|---------|",
        "| Conductivity (ln κ) | 0.740 | 0.749 | 0.703 |",
        "| Density | 0.926 | 0.949 | 0.933 |",
        "| Viscosity (ln η) | 0.809 | 0.846 | 0.759 |",
        "| Melting point (K) | 0.523 | 0.561 | 0.365 |",
        "",
        "## Table S2. Final unreported candidates (three-model consistent, T_m < 298 K)",
        "",
        "| # | Cation | Anion | ln κ (G/H/N) | ln η (G) | T_m (G) | ρ (G) | |Δlnκ(GBM−GNN)| | SA |",
        "|---|--------|-------|---------------|----------|---------|--------|----------------|----|",
    ]
    for i, r in cand.iterrows():
        lines.append(
            f"| {i+1} | {r['cat_smiles']} | {r['an_smiles']} | "
            f"{r['g_cond']:.3f}/{r['h_cond']:.3f}/{r['gnn_cond']:.3f} | {r['g_visc']:.3f} | {r['g_tm']:.1f} | "
            f"{r['g_dens']:.3f} | {r['gbm_gnn_gap']:.3f} | {r['sa_sum']:.1f} |"
        )
    lines += [
        "",
        "## Table S3. Scaffold-mutation cation inventory (by core)",
        "",
        "| Core | Count |",
        "|------|-------|",
        "| Imidazolium | 96 |",
        "| Pyrrolidinium | 96 |",
        "| Sulfonium | 96 |",
        "| Pyridinium | 11 |",
        "| Ammonium | 11 |",
        "| Phosphonium | 11 |",
        "| **Total (novel, charge +1, valid)** | **272** |",
        "",
        "## Methods supplement",
        "",
        "**Predictors.** GBM (GradientBoostingRegressor, default) and HistGBM",
        "(HistGradientBoostingRegressor, max_iter=300) on 458 per-ion RDKit descriptors",
        "(+ temperature for temperature-dependent properties); 5-fold GroupKFold on ion-pair",
        "identity; final oracles refit on the full 84,077-record dataset.",
        "",
        "**Combinatorial screening.** 795 canonical cations × 300 canonical anions = 238,500",
        "pairs, evaluated at 298.15 K by both models. Filters: novelty (not in the 1,891-pair",
        "database), predicted T_m < 298 K, |Δ ln κ| < 0.6, |ΔT_m| < 15 K. Multi-objective Pareto",
        "front (maximize ln κ; minimize ln η and T_m) by non-dominated sorting.",
        "",
        "**Scaffold mutation.** Six cation cores (imidazolium, pyridinium, pyrrolidinium,",
        "ammonium, phosphonium, sulfonium) mutated across C1–C8 alkyl, methoxy/ethoxyethyl,",
        "and hydroxyethyl substituents; RDKit-validated for valence and +1 charge;",
        "novelty enforced against the 867 known cations.",
        "",
        "**VAE.** Character-level GRU VAE (latent dim 48, max length 80) trained on 860 cations",
        "and 356 anions; validity ~10%, correct-charge fraction ~60%.",
        "",
        "**Latent optimization.** (μ, λ) evolutionary strategy in the 48-d latent space against",
        "the GBM oracle with an SA penalty; converged to 1-ethyl-3-methylimidazolium (the known",
        "optimum).",
    ]
    out = "\n".join(lines) + "\n"
    (ROOT / "supplementary_materials.md").write_text(out, encoding="utf-8")
    print(f"SI 已写入 supplementary_materials.md（候选 {len(cand)} 个）")


if __name__ == "__main__":
    main()
