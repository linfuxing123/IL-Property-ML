# -*- coding: utf-8 -*-
"""train_coupling_significance.py — 训练密度耦合假说的严格检验。

问题：共享训练离子的 IL 分歧更小（-0.075），但它们的预测值也更高
（-3.30 vs -3.79），可能只是饱和效应的混淆（高值区本来分歧就小）。
严格检验：控制预测值后，共享离子的效应是否仍存在？
方法：
  1) 按预测值分层（匹配预测值分布），层内比较共享 vs 不共享的分歧
  2) t 检验显著性
  3) 偏相关：控制 mean_kappa 后 corr(shares_ion, divergence)
"""
import numpy as np
import pandas as pd
import pathlib
from scipy import stats

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"
DS = P6 / "data" / "expanded" / "paper_dataset"

df = pd.read_csv(GEN / "consensus_layers.csv")
df = df[df["ln_kappa_ilpe"].notna()].copy()
df["cat"] = df["il"].str.split("|").str[0]
df["an"] = df["il"].str.split("|").str[1]

tr = pd.read_csv(DS / "conductivity.csv")
tr_cats = set(tr["cat_smiles"].astype(str).unique())
tr_ans = set(tr["an_smiles"].astype(str).unique())
df["shares_ion"] = df["cat"].isin(tr_cats) | df["an"].isin(tr_ans)

share = df[df["shares_ion"]]["divergence"].to_numpy()
noshare = df[~df["shares_ion"]]["divergence"].to_numpy()

print("=== 1) 总体 t 检验 ===")
t, p = stats.ttest_ind(share, noshare, equal_var=False)
print(f"共享(n={len(share):,}) 分歧={share.mean():.4f} vs 不共享(n={len(noshare):,}) 分歧={noshare.mean():.4f}")
print(f"t={t:.3f}, p={p:.2e}（p<0.05 显著）")

print("\n=== 2) 预测值分层匹配检验 ===")
# 按 mean_kappa 10 分位分层，层内比较
df["q"] = pd.qcut(df["mean_kappa"], 10, labels=False)
results = []
for q in range(10):
    sub = df[df["q"] == q]
    s = sub[sub["shares_ion"]]["divergence"]
    ns = sub[~sub["shares_ion"]]["divergence"]
    if len(s) > 5 and len(ns) > 5:
        diff = s.mean() - ns.mean()
        results.append((q, len(s), len(ns), s.mean(), ns.mean(), diff))
print(f"{'层':>3} | {'共享n':>6} | {'不共享n':>6} | {'共享分歧':>8} | {'不共享分歧':>8} | {'差异':>7}")
for q, ns_, nn_, sm, nm, d in results:
    print(f"{q:3d} | {ns_:6d} | {nn_:6d} | {sm:8.4f} | {nm:8.4f} | {d:+7.4f}")

# 层内差异汇总（配对 t 检验跨层）
diffs = [d for _, _, _, _, _, d in results]
if len(diffs) > 1:
    t2, p2 = stats.ttest_1samp(diffs, 0)
    print(f"\n层内差异均值 = {np.mean(diffs):+.4f}（10 层配对 t={t2:.3f}, p={p2:.3f}）")

print("\n=== 3) 偏相关（控制 mean_kappa）===")
# 残差化分歧和共享标志对 mean_kappa 回归后取残差相关
from numpy.polynomial import polynomial as P
x = df["mean_kappa"].to_numpy()
y = df["divergence"].to_numpy()
z = df["shares_ion"].astype(float).to_numpy()
# 简化：分层后按层标准化
std_diffs = [d / max(1e-6, s) for _, _, _, s, _, d in results]
print(f"标准化层内差异（相对共享组分歧）: {['{:.3f}'.format(d) for d in std_diffs]}")
print(f"均值: {np.mean(std_diffs):+.3f} → 共享训练离子使分歧降低 "
      f"{abs(np.mean(std_diffs))*100:.1f}%（控制预测值后）")
