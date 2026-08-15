# -*- coding: utf-8 -*-
"""divergence_theory.py — 预测器分歧的方差分解理论 + 实证验证。

核心洞察（math-agent 类比同构：分歧↔方差分解）：
  对同一 IL，两个预测器 f1(GBM), f2(Hist) 的预测值：
    mean m = (f1+f2)/2，分歧 d = |f1-f2|/2
  方差分解：Var(f1) 与 Var(f2) 的关系约束了 corr(m, d) 的可能符号。
  
  关键定理：若 corr(f1, f2) = r > 0 且 Var(f1)=Var(f2)=σ²，则
    corr(m, d) = 0（m 与 d 独立）——因为 m=(f1+f2)/2, d=(f1-f2)/2，
    Cov(m,d) = (Var(f1)-Var(f2))/4 = 0。
  因此实测 corr(mean, divergence) = -0.41 意味着：
    (a) Var(f1) ≠ Var(f2)（异方差），或
    (b) corr(m,d) 受高阶结构影响（非线性），或
    (c) f1,f2 在"高值区"分歧系统性小于"低值区"（预测饱和效应）
  本脚本用真实数据验证：-0.41 主要来自 (c) 饱和效应还是 (a) 异方差。
"""
import pathlib

import numpy as np
import pandas as pd

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"

df = pd.read_csv(GEN / "full_space_uncertainty.csv")
f1 = df["gbm_conductivity"].to_numpy()
f2 = df["hist_conductivity"].to_numpy()
m = (f1 + f2) / 2
d = np.abs(f1 - f2) / 2

print("=== 方差分解验证 ===")
print(f"Var(f1=GBM) = {np.var(f1):.4f}")
print(f"Var(f2=Hist) = {np.var(f2):.4f}")
print(f"corr(f1,f2) = {np.corrcoef(f1,f2)[0,1]:.4f}")
print(f"理论 Cov(m,d) = (Var1-Var2)/4 = {(np.var(f1)-np.var(f2))/4:.4f}")
print(f"实测 corr(m,d) = {np.corrcoef(m,d)[0,1]:.4f}")

# 如果异方差是主因，残差化后的偏相关应接近 0
# 用分位数看：高值区 vs 低值区的平均分歧
print("\n=== 分歧随预测值的结构 ===")
for q in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    lo, hi = np.quantile(m, q[0]), np.quantile(m, q[1])
    mask = (m >= lo) & (m <= hi)
    print(f"  m∈[{lo:.2f},{hi:.2f}]: 平均分歧={d[mask].mean():.4f}, n={mask.sum():,}")

# 关键检验：去均值（centering）后 corr 是否消失
print("\n=== 饱和效应检验 ===")
# 若高值区分歧系统性小（饱和），则 corr(m,d)<0 来自结构而非方差
# 构造对照：同方差的 f1',f2'（分别中心化后加相同σ）→ corr(m',d') 应≈0
f1c = (f1 - f1.mean()) / f1.std()
f2c = (f2 - f2.mean()) / f2.std()
m2 = (f1c + f2c) / 2
d2 = np.abs(f1c - f2c) / 2
print(f"对照（同方差化后）corr(m',d') = {np.corrcoef(m2,d2)[0,1]:.4f}")
print(f"→ 若仍显著负，说明是结构饱和效应；若≈0，说明是异方差")
df.to_csv(GEN / "divergence_theory.csv", index=False)
print("\n已保存 divergence_theory.csv")
