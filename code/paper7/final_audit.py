# -*- coding: utf-8 -*-
"""final_audit.py — 手稿最终数字审计。"""
import numpy as np
import pandas as pd

GEN = r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6\data\generated"
DS = r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6\data\expanded\paper_dataset"

# 1. 训练耦合数字
df = pd.read_csv(GEN + r"\consensus_layers.csv")
df = df[df["ln_kappa_ilpe"].notna()].copy()
df["cat"] = df["il"].str.split("|").str[0]
df["an"] = df["il"].str.split("|").str[1]
tr = pd.read_csv(DS + r"\conductivity.csv")
tc = set(tr["cat_smiles"].astype(str).unique())
ta = set(tr["an_smiles"].astype(str).unique())
df["shares"] = df["cat"].isin(tc) | df["an"].isin(ta)
s = df[df["shares"]]["divergence"]
ns = df[~df["shares"]]["divergence"]
print(f"共享 n={len(s):,} 分歧={s.mean():.4f} | 不共享 n={len(ns):,} 分歧={ns.mean():.4f}")
print(f"差异 {s.mean()-ns.mean():+.4f}")

# 2. 手稿 Table 数字核对（4 性质 corr）
full = pd.read_csv(GEN + r"\full_space_all_scored.csv")
props = {"conductivity": ("gbm_conductivity", "hist_conductivity"),
         "density": ("gbm_density", "hist_density"),
         "melting_point": ("gbm_melting_point", "hist_melting_point"),
         "viscosity": ("gbm_viscosity", "hist_viscosity")}
print("\n4 性质 corr(m,d):")
for p, (a, b) in props.items():
    sub = full[[a, b]].dropna()
    x, y = sub[a].to_numpy(), sub[b].to_numpy()
    m = (x + y) / 2
    d = np.abs(x - y) / 2
    print(f"  {p}: {np.corrcoef(m, d)[0,1]:+.3f}")

# 3. 6 对预测器（Table 1）核对
df2 = pd.read_csv(GEN + r"\consensus_layers.csv")
df2 = df2[df2["ln_kappa_ilpe"].notna()]
pairs = [("GBM-Hist", df2["gbm_conductivity"], df2["hist_conductivity"]),
         ("GBM-ILPE", df2["gbm_conductivity"], df2["ln_kappa_ilpe"]),
         ("Hist-ILPE", df2["hist_conductivity"], df2["ln_kappa_ilpe"])]
mp = pd.read_csv(GEN + r"\mpnn_paradigm_sample.csv")
pairs += [("GBM-MPNN", mp["gbm_conductivity"], mp["mpnn_conductivity"]),
          ("Hist-MPNN", mp["hist_conductivity"], mp["mpnn_conductivity"]),
          ("MPNN-ILPE", mp["mpnn_conductivity"], mp["ln_kappa_ilpe"])]
print("\nTable 1 六对 corr(m,d):")
for name, a, b in pairs:
    m = (a + b) / 2
    d = (a - b).abs() / 2
    print(f"  {name}: {np.corrcoef(m, d)[0,1]:+.3f}")
print("\n审计完成")
