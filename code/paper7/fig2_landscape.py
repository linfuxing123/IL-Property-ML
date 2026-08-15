# -*- coding: utf-8 -*-
"""fig2_landscape.py — 第 7 篇 Fig 2：性质依赖景观 + 诊断价值。

(a) 4 性质 corr(m,d) 条形（raw vs 同方差化）——饱和/反饱和
(b) 4 性质分歧分层 vs 预测误差（诊断价值：Q1→Q4 误差增长）
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

# (a) 数据：4 性质 corr(m,d)
df = pd.read_csv(GEN / "full_space_all_scored.csv")
props = ["conductivity", "density", "melting_point", "viscosity"]
cols = {
    "conductivity": ("gbm_conductivity", "hist_conductivity"),
    "density": ("gbm_density", "hist_density"),
    "melting_point": ("gbm_melting_point", "hist_melting_point"),
    "viscosity": ("gbm_viscosity", "hist_viscosity"),
}
raw, norm = [], []
for p in props:
    ca, cb = cols[p]
    sub = df[[ca, cb]].dropna()
    a, b = sub[ca].to_numpy(), sub[cb].to_numpy()
    m = (a + b) / 2
    d = np.abs(a - b) / 2
    raw.append(np.corrcoef(m, d)[0, 1])
    ac = (a - a.mean()) / a.std()
    bc = (b - b.mean()) / b.std()
    m2 = (ac + bc) / 2
    d2 = np.abs(ac - bc) / 2
    norm.append(np.corrcoef(m2, d2)[0, 1])

# (b) 诊断价值（来自 diagnostic_value 结果）
diag = {
    "conductivity": (0.411, 61), "density": (0.475, 114),
    "melting_point": (0.486, 143), "viscosity": (0.151, 29),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
x = np.arange(4)
ax.bar(x - 0.18, raw, 0.36, label="corr(mean, disagreement)", color="#4C72B0")
ax.bar(x + 0.18, norm, 0.36, label="variance-normalized", color="#DD8452")
ax.axhline(0, c="gray", lw=0.8)
ax.axhline(0, c="red", ls="--", lw=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(props, fontsize=10)
ax.set_ylabel("corr(mean, disagreement)")
ax.set_title("(a) Property-dependent saturation / anti-saturation")
ax.legend(fontsize=8)

ax = axes[1]
names = list(diag.keys())
errs = [diag[n][0] for n in names]
growths = [diag[n][1] for n in names]
xx = np.arange(4)
bars = ax.bar(xx, errs, 0.5, color="#55A868")
for i, (e, g) in enumerate(zip(errs, growths)):
    ax.text(i, e + 0.01, f"+{g}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xx)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("corr(disagreement, |error|)")
ax.set_title("(b) Diagnostic value: disagreement marks error (4/4)")
ax.set_ylim(0, 0.6)

fig.tight_layout()
fig.savefig(OUT / "fig2_landscape.png", dpi=300)
print("saved", OUT / "fig2_landscape.png")
