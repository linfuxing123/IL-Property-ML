# -*- coding: utf-8 -*-
"""cross_paradigm_divergence.py — 跨范式分歧验证（树 vs ILPE 独立源）。

问题：饱和效应（高预测值处分歧小）是树模型特有，还是跨范式普适？
用 GBM vs ILPE（独立 ML 范式）重测 corr(mean, divergence)。
"""
import pathlib

import numpy as np
import pandas as pd

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"

df = pd.read_csv(GEN / "consensus_layers.csv")
# consensus_layers.csv 已含 ln_kappa_ilpe（ILPE 匹配行），直接用
df = df[df["ln_kappa_ilpe"].notna()].copy()
print(f"GBM-ILPE 交集: {len(df):,}", flush=True)

f1 = df["gbm_conductivity"].to_numpy()
f2 = df["ln_kappa_ilpe"].to_numpy()
m = (f1 + f2) / 2
d = np.abs(f1 - f2) / 2

print(f"Var(GBM) = {np.var(f1):.4f}, Var(ILPE) = {np.var(f2):.4f}")
print(f"corr(GBM,ILPE) = {np.corrcoef(f1,f2)[0,1]:.4f}")
print(f"corr(mean, divergence) = {np.corrcoef(m,d)[0,1]:.4f}")

# 同方差对照
f1c = (f1 - f1.mean()) / f1.std()
f2c = (f2 - f2.mean()) / f2.std()
m2 = (f1c + f2c) / 2
d2 = np.abs(f1c - f2c) / 2
print(f"同方差化后 corr(m',d') = {np.corrcoef(m2,d2)[0,1]:.4f}")

# 分层
print("\n=== 分歧随均值分层（GBM vs ILPE）===")
for q in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    lo, hi = np.quantile(m, q[0]), np.quantile(m, q[1])
    mask = (m >= lo) & (m <= hi)
    print(f"  m∈[{lo:.2f},{hi:.2f}]: 平均分歧={d[mask].mean():.4f}, n={mask.sum():,}")

# 树共识区（Q1）内，GBM-ILPE 分歧 vs GBM 预测值
q1 = df[df["div_q"] == "Q1共识最高"]
print(f"\n=== 树共识区（Q1）内跨范式结构 ===")
print(f"  Q1: n={len(q1):,}, corr(GBM均值, GBM-ILPE分歧) = "
      f"{np.corrcoef(q1['gbm_conductivity'], (q1['gbm_conductivity']-q1['ln_kappa_ilpe']).abs())[0,1]:.4f}")
df.to_csv(GEN / "cross_paradigm_divergence.csv", index=False)
print("\n已保存 cross_paradigm_divergence.csv")
