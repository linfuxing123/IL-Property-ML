# -*- coding: utf-8 -*-
"""multiprop_saturation.py — 饱和效应多性质普适性验证。

全空间打分含 3 性质（conductivity/melting_point/viscosity）的 GBM+Hist 预测。
对每个性质重算 corr(mean, disagreement) + 同方差化 + 分层单调：
  - 若 3/3 性质都负相关 → 饱和效应跨性质普适
  - 若某性质不成立 → 边界（可能与性质分布形状有关）
"""
import numpy as np
import pandas as pd
import pathlib

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"
OUT = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper7")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(GEN / "full_space_all_scored.csv")
print(f"全空间: {len(df):,} 组合", flush=True)

props = {
    "conductivity": ("gbm_conductivity", "hist_conductivity"),
    "melting_point": ("gbm_melting_point", "hist_melting_point"),
    "viscosity": ("gbm_viscosity", "hist_viscosity"),
}

rows = []
for prop, (ca, cb) in props.items():
    if ca not in df.columns or cb not in df.columns:
        print(f"{prop}: 列缺失，跳过", flush=True)
        continue
    sub = df[[ca, cb]].dropna()
    a, b = sub[ca].to_numpy(), sub[cb].to_numpy()
    m = (a + b) / 2
    d = np.abs(a - b) / 2
    r_ab = np.corrcoef(a, b)[0, 1]
    r_md = np.corrcoef(m, d)[0, 1]
    ac = (a - a.mean()) / a.std()
    bc = (b - b.mean()) / b.std()
    m2 = (ac + bc) / 2
    d2 = np.abs(ac - bc) / 2
    r_norm = np.corrcoef(m2, d2)[0, 1]
    n = len(a)
    t = r_md * np.sqrt((n - 2) / (1 - r_md**2))
    # 分层单调
    qs = np.quantile(m, np.linspace(0, 1, 5))
    seq = []
    for i in range(4):
        mask = (m >= qs[i]) & (m < qs[i+1])
        if mask.sum() > 5:
            seq.append(round(float(d[mask].mean()), 4))
    mono = len(seq) == 4 and all(seq[i] > seq[i+1] for i in range(3))
    rows.append({"property": prop, "n": n, "corr_predictors": round(r_ab, 3),
                 "corr_mean_div": round(r_md, 3), "corr_normalized": round(r_norm, 3),
                 "t_stat": round(t, 1), "monotone": mono, "layered_seq": str(seq)})
    print(f"{prop}: corr(f1,f2)={r_ab:.3f} corr(m,d)={r_md:.3f} "
          f"norm={r_norm:.3f} t={t:.1f} mono={mono}", flush=True)

tbl = pd.DataFrame(rows)
tbl.to_csv(OUT / "table2_multiprop.csv", index=False)
print("\n已保存 table2_multiprop.csv", flush=True)
