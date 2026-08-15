# -*- coding: utf-8 -*-
"""mpnn_paradigm.py — 补 MPNN(GNN 范式) 打分，验证三范式饱和效应。

目标：把方差分解定理推广到第三范式（GNN）。
1) 对 full_space 打分结果抽样子集（MPNN 逐图编码慢，抽 5000 个覆盖全值域）
2) 计算 MPNN vs GBM/HistGBM/ILPE 的分歧-均值相关
3) 验证饱和效应在 GNN 范式是否成立
"""
import pathlib
import sys

import numpy as np
import pandas as pd

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"
sys.path.insert(0, str(P6))
from gnn_score_v2 import score_mpnn_v2

# 全空间打分（含 GBM/Hist + ILPE 匹配）
df = pd.read_csv(GEN / "consensus_layers.csv")
df = df[df["ln_kappa_ilpe"].notna()].copy()
print(f"可用（含 ILPE）: {len(df):,}", flush=True)

# 分层抽样：按 mean_kappa 分位数抽 5000，保证覆盖全值域
df["q"] = pd.qcut(df["mean_kappa"], 20, labels=False)
samp = df.groupby("q", observed=True).apply(
    lambda g: g.sample(min(250, len(g)), random_state=0), include_groups=False
).reset_index()
# 修正：apply 后可能丢列，改用循环
rows = []
for qv in range(20):
    sub = df[df["q"] == qv]
    if len(sub):
        rows.append(sub.sample(min(250, len(sub)), random_state=0))
samp = pd.concat(rows).reset_index(drop=True)
print(f"抽样子集: {len(samp):,}", flush=True)

# MPNN 打分
samp["mpnn_conductivity"] = score_mpnn_v2(samp)
samp.to_csv(GEN / "mpnn_paradigm_sample.csv", index=False)
print("MPNN 打分完成", flush=True)

# 三范式分歧分析
f_g = samp["gbm_conductivity"].to_numpy()
f_h = samp["hist_conductivity"].to_numpy()
f_m = samp["mpnn_conductivity"].to_numpy()
f_i = samp["ln_kappa_ilpe"].to_numpy()

pairs = [("GBM-Hist", f_g, f_h), ("GBM-MPNN", f_g, f_m),
         ("Hist-MPNN", f_h, f_m), ("GBM-ILPE", f_g, f_i),
         ("MPNN-ILPE", f_m, f_i)]
print("\n=== 各对预测器的分歧-均值相关 ===")
for name, a, b in pairs:
    m = (a + b) / 2
    d = np.abs(a - b) / 2
    corr_raw = np.corrcoef(m, d)[0, 1]
    # 同方差化
    ac = (a - a.mean()) / a.std()
    bc = (b - b.mean()) / b.std()
    m2 = (ac + bc) / 2
    d2 = np.abs(ac - bc) / 2
    corr_norm = np.corrcoef(m2, d2)[0, 1]
    r_ab = np.corrcoef(a, b)[0, 1]
    print(f"  {name}: corr(f1,f2)={r_ab:.3f} | corr(m,d)={corr_raw:.3f} "
          f"| 同方差化={corr_norm:.3f}")
