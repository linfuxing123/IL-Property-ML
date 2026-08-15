# -*- coding: utf-8 -*-
"""disagreement_ranking.py — 分歧校准排序方法 + 用 ILPE 独立源验证。

方法：对每个候选 IL，评分 = mean_kappa - lambda * divergence（分歧惩罚）。
lambda=0 即纯均值排序；lambda>0 惩罚高分歧候选。
验证：用 ILPE（独立源）作为 proxy ground truth——看排序 top-k 中
"ILPE 也高"的比例是否随 lambda 提升（分歧惩罚确实提升排序质量）。
"""
import numpy as np
import pandas as pd
import pathlib

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"
OUT = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper7")
OUT.mkdir(parents=True, exist_ok=True)

# 电导率：GBM/Hist + ILPE 交集（6,108）
df = pd.read_csv(GEN / "consensus_layers.csv")
df = df[df["ln_kappa_ilpe"].notna()].copy()
m = (df["gbm_conductivity"] + df["hist_conductivity"]) / 2
d = (df["gbm_conductivity"] - df["hist_conductivity"]).abs()
gt = df["ln_kappa_ilpe"].to_numpy()

print(f"验证样本: {len(df):,}（含 ILPE 独立源）", flush=True)

# 不同 lambda 的排序质量：top-100 中 ILPE 高电导(>中位)的比例
gt_med = np.median(gt)
print("\n=== 分歧校准排序 vs 纯均值排序（top-100）===")
print(f"{'lambda':>7} | {'top100 中 ILPE>中位':>18} | {'提升':>6}")
baseline = None
for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
    score = m - lam * d
    top = df.iloc[np.argsort(-score.to_numpy())[:100]]
    frac = (top["ln_kappa_ilpe"].to_numpy() > gt_med).mean()
    if lam == 0:
        baseline = frac
    print(f"{lam:7.1f} | {frac*100:15.1f}% | {frac-baseline:+5.1%}", flush=True)

# 全谱：lambda 扫描下 top-k 的 ILPE 一致性（k=50/100/200）
print("\n=== lambda 扫描（top-50/100/200 的 ILPE 高值比例）===")
for k in [50, 100, 200]:
    row = []
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        score = m - lam * d
        top = df.iloc[np.argsort(-score.to_numpy())[:k]]
        frac = (top["ln_kappa_ilpe"].to_numpy() > gt_med).mean()
        row.append(round(float(frac), 3))
    print(f"  top-{k}: " + " ".join(f"{x:.3f}" for x in row), flush=True)

# 保存
df["mean_kappa"] = m
df["divergence"] = d
for lam in [1.0, 2.0]:
    df[f"score_l{lam}"] = m - lam * d
df.to_csv(OUT / "disagreement_ranking.csv", index=False)
print("\n已保存 disagreement_ranking.csv", flush=True)
