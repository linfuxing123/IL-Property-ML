# -*- coding: utf-8 -*-
"""fig1_saturation.py — 第 7 篇 Fig 1：饱和效应三面板。

(a) GBM-ILPE 分歧 vs 预测均值散点（同方差化后仍负相关的核心证据）
(b) 分歧随预测值分层的单调性（五对预测器）
(c) 五对预测器的 corr(m,d) 汇总条形（raw vs 同方差化）
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"
OUT = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper7")
OUT.mkdir(parents=True, exist_ok=True)

# 数据：GBM-ILPE 跨范式（6,108 交集）
df = pd.read_csv(GEN / "consensus_layers.csv")
df = df[df["ln_kappa_ilpe"].notna()].copy()
f1, f2 = df["gbm_conductivity"].to_numpy(), df["ln_kappa_ilpe"].to_numpy()
m = (f1 + f2) / 2
d = np.abs(f1 - f2) / 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) 散点 + 拟合
ax = axes[0]
ax.scatter(m, d, s=2, alpha=0.2, c="#4C72B0")
bins = np.quantile(m, np.linspace(0, 1, 21))
cents, means = [], []
for i in range(20):
    mask = (m >= bins[i]) & (m < bins[i+1])
    if mask.sum() > 10:
        cents.append((bins[i]+bins[i+1])/2)
        means.append(d[mask].mean())
ax.plot(cents, means, "r-o", lw=2, ms=4, label="mean |disagreement|")
ax.set_xlabel("Predicted mean ln κ (GBM & ILPE)")
ax.set_ylabel("|disagreement| / 2")
ax.set_title("(a) Saturation: high-value region → low disagreement")
ax.legend(fontsize=8)

# (b) 分层单调性（五对）
pairs = {
    "GBM-Hist": ("gbm_conductivity", "hist_conductivity"),
    "GBM-MPNN": ("gbm_conductivity", "mpnn_conductivity"),
    "Hist-MPNN": ("hist_conductivity", "mpnn_conductivity"),
    "GBM-ILPE": ("gbm_conductivity", "ln_kappa_ilpe"),
    "MPNN-ILPE": ("mpnn_conductivity", "ln_kappa_ilpe"),
}
mp = pd.read_csv(GEN / "mpnn_paradigm_sample.csv")
ax = axes[1]
for name, (ca, cb) in pairs.items():
    sub = mp[mp[cb].notna()]
    a, b = sub[ca].to_numpy(), sub[cb].to_numpy()
    mm = (a + b) / 2
    dd = np.abs(a - b) / 2
    qs = np.quantile(mm, np.linspace(0, 1, 5))
    xs, ys = [], []
    for i in range(4):
        mask = (mm >= qs[i]) & (mm < qs[i+1])
        if mask.sum() > 5:
            xs.append((qs[i]+qs[i+1])/2)
            ys.append(dd[mask].mean())
    ax.plot(xs, ys, "-o", ms=4, lw=1.5, label=name)
ax.set_xlabel("Predicted mean (centered)")
ax.set_ylabel("mean |disagreement|")
ax.set_title("(b) Monotone decrease across all paradigm pairs")
ax.legend(fontsize=7)

# (c) 汇总条形
ax = axes[2]
names = list(pairs.keys())
raw = [-0.274, -0.415, -0.320, -0.426, -0.464]
norm = [-0.249, -0.416, -0.309, -0.440, -0.466]
x = np.arange(len(names))
ax.bar(x-0.18, raw, 0.36, label="raw corr(m,d)", color="#4C72B0")
ax.bar(x+0.18, norm, 0.36, label="variance-normalized", color="#DD8452")
ax.axhline(0, c="gray", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8, rotation=15)
ax.set_ylabel("corr(mean, disagreement)")
ax.set_title("(c) Universal negative correlation (5/5 pairs)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "fig1_saturation.png", dpi=300)
print("saved", OUT / "fig1_saturation.png")
