#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scaling_law.py — IL 性质预测的学习曲线标度律（paper2 升级，2026-08-18）

核心：把"数据密度是硬约束"从断言变成定律。
对 4 个属性做 IL-disjoint (GroupKFold 5) 多规模子采样（N_IL ∈ 网格），
拟合 R²(N) = a − b·N^(−γ)，给出每属性的 γ、渐近上限 a、达到设计精度
（R²=0.8/0.9）所需 IL 数。

口径（与论文一致）：
  - 特征: 10 个 RDKit 描述符 (mw logp tpsa hbd hba rotb ar_rings heavy fcsp3 rings)
  - 粘度/电导率目标做 ln 变换；密度/熔点用原始值
  - 温度相关属性含 T 特征；熔点无 T
  - 组级评估: 5 折 GroupKFold(IL 身份)；点级对照: 5 折 KFold

输出 (results/)：
  scaling_law_results.csv  每 (属性, N, seed) 的组级/点级 R²
  scaling_law_fits.csv     拟合参数 a/b/γ + N80/N90 + GBR 全量锚点
  fig_scaling_laws.png     2x2 面板学习曲线 + 拟合

用法: python scaling_law.py [--model hgb|gbm] [--seeds 0,1,2] [--no-anchor]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "workspace" / "matmodel" / "data" / "ilt"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]

# 属性配置: (csv, 是否 ln, 是否含 T, N 网格, 中文名)
PROPS = {
    "viscosity":     ("viscosity.csv", True,  True,  [50, 100, 200, 400, 800, None], "Viscosity (ln η)"),
    "conductivity":  ("conductivity.csv", True,  True,  [50, 100, 200, 400, 800, None], "Conductivity (ln κ)"),
    "density":       ("density.csv", False, True,  [50, 100, 200, 400, 800, None], "Density"),
    "melting_point": ("melting_point_all.csv", False, False, [50, 100, 200, 300, 450, None], "Melting point (K)"),
}


def make_model(name, seed):
    if name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=seed)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.08, max_depth=7,
        l2_regularization=0.5, random_state=seed)


def load(prop):
    csv, use_log, has_t, _, _ = PROPS[prop]
    df = pd.read_csv(DATA / csv)
    df = df.dropna(subset=FEATS)
    if has_t:
        df = df.dropna(subset=["T"])
        feats = FEATS + ["T"]
    else:
        feats = list(FEATS)
    y = np.log(df["value"].to_numpy(dtype=float)) if use_log else df["value"].to_numpy(dtype=float)
    X = df[feats].to_numpy(dtype=float)
    g = df["il"].to_numpy()
    return X, y, g, feats


def group_r2(model_name, X, y, g, seed, group_disjoint):
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, KFold
    cv = GroupKFold(n_splits=5) if group_disjoint else KFold(n_splits=5, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tr, te in cv.split(X, y, groups=g if group_disjoint else None):
        m = make_model(model_name, seed)
        m.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te]))
    return r2_score(np.asarray(yt), np.asarray(yp))


def subsample(X, y, g, n_il, seed):
    if n_il is None:
        return X, y, g
    ils = np.unique(g)
    chosen = set(np.random.RandomState(seed).choice(ils, size=min(n_il, len(ils)), replace=False))
    mask = np.isin(g, list(chosen))
    return X[mask], y[mask], g[mask]


def fit_law(ns, r2s):
    from scipy.optimize import curve_fit
    f = lambda N, a, b, g: a - b * np.power(N, -g)
    ns = np.asarray(ns, dtype=float); r2s = np.asarray(r2s, dtype=float)
    p0 = [0.9, 0.6, 0.5]
    try:
        popt, _ = curve_fit(f, ns, r2s, p0=p0,
                            bounds=([0.0, 0.0, 0.05], [1.05, 10.0, 3.0]),
                            maxfev=20000)
    except Exception as e:
        print(f"  [fit failed] {e}")
        return None
    a, b, g = popt
    resid = f(ns, *popt) - r2s
    r2_fit = 1 - np.sum(resid ** 2) / np.sum((r2s - r2s.mean()) ** 2)
    def n_for(target):
        if a < target + 0.02:
            return None
        return float((b / (a - target)) ** (1.0 / g))
    return {"a": a, "b": b, "gamma": g, "N80": n_for(0.80), "N90": n_for(0.90), "r2_fit": r2_fit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb", choices=["hgb", "gbm"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--no-anchor", action="store_true", help="跳过 GBR 全量锚点（复现 Table 2）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows, fits = [], []
    for prop, (csv, use_log, has_t, ngrid, label) in PROPS.items():
        print(f"\n===== {prop} =====")
        X, y, g, feats = load(prop)
        n_il_total = len(np.unique(g))
        print(f"  {len(X)} records / {n_il_total} ILs / {len(feats)} features")
        for n in ngrid:
            n_eff = n_il_total if n is None else min(n, n_il_total)
            r2g, r2p = [], []
            for s in seeds:
                Xs, ys, gs = subsample(X, y, g, n, s)
                if len(np.unique(gs)) < 6:
                    continue
                rg = group_r2(args.model, Xs, ys, gs, s, group_disjoint=True)
                rp = group_r2(args.model, Xs, ys, gs, s, group_disjoint=False)
                r2g.append(rg); r2p.append(rp)
                rows.append({"property": prop, "N_IL": n_eff, "seed": s,
                             "r2_group": rg, "r2_point": rp})
                print(f"  N={n_eff:5d} seed={s}  R2_group={rg:+.3f}  R2_point={rp:+.3f}")
            pd.DataFrame(rows).to_csv(OUT / "scaling_law_results.csv", index=False, encoding="utf-8-sig")
            if len(r2g) >= 2:
                print(f"  -> N={n_eff:5d} mean R2_group={np.mean(r2g):+.3f} ± {np.std(r2g):.3f}")
        # 拟合
        sub = pd.DataFrame(rows)
        sub = sub[sub["property"] == prop]
        means = sub.groupby("N_IL")["r2_group"].mean().reset_index()
        fit = fit_law(means["N_IL"].to_numpy(), means["r2_group"].to_numpy())
        if fit:
            fit.update({"property": prop, "label": label})
            fits.append(fit)
            n90 = f"{fit['N90']:.0f}" if fit["N90"] else "unreachable"
            n80 = f"{fit['N80']:.0f}" if fit["N80"] else "unreachable"
            print(f"  FIT  a={fit['a']:.3f}  b={fit['b']:.3f}  gamma={fit['gamma']:.3f}  "
                  f"N80={n80}  N90={n90}  (fit R2={fit['r2_fit']:.3f})")
        # GBR 全量锚点（复现 Table 2）
        if not args.no_anchor:
            print("  [anchor] GBR full-data GroupKFold(5) seed=0 ...")
            rg = group_r2("gbm", X, y, g, 0, group_disjoint=True)
            rp = group_r2("gbm", X, y, g, 0, group_disjoint=False)
            print(f"  [anchor] GBR full: R2_group={rg:+.3f}  R2_point={rp:+.3f}  (paper Table2: "
                  f"{'0.68/0.79' if prop=='viscosity' else ('0.70/0.84' if prop=='conductivity' else ('0.85/0.94' if prop=='density' else '0.39/—'))})")
            fits.append({"property": prop, "label": label, "anchor_gbm_r2_group": rg,
                         "anchor_gbm_r2_point": rp})

    pd.DataFrame(fits).to_csv(OUT / "scaling_law_fits.csv", index=False, encoding="utf-8-sig")

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    res = pd.DataFrame(rows)
    for ax, prop in zip(axes.ravel(), PROPS):
        s = res[res["property"] == prop]
        ns = sorted(s["N_IL"].unique())
        means, stds = [], []
        for n in ns:
            v = s[s["N_IL"] == n]["r2_group"]
            means.append(v.mean()); stds.append(v.std())
        ax.errorbar(ns, means, yerr=stds, fmt="o", ms=4, capsize=2, label="group R² (mean±std)")
        f = next((x for x in fits if x.get("property") == prop and "gamma" in x), None)
        anchor = next((x for x in fits if x.get("property") == prop and "anchor_gbm_r2_group" in x), None)
        if f:
            nn = np.linspace(min(ns), max(ns), 100)
            ax.plot(nn, f["a"] - f["b"] * nn ** (-f["gamma"]), "-", lw=1.5,
                    label=f"fit: a={f['a']:.2f}, γ={f['gamma']:.2f}")
        if anchor:
            ax.scatter([max(ns)], [anchor["anchor_gbm_r2_group"]], marker="*", s=80,
                       color="red", label="GBR full (paper anchor)")
        ax.set_xscale("log"); ax.set_xlabel("N_IL"); ax.set_ylabel("group R²")
        ax.set_title(PROPS[prop][4]); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling_laws.png", dpi=300)
    print(f"\n图已存: {OUT / 'fig_scaling_laws.png'}")
    print(f"结果已存: {OUT / 'scaling_law_results.csv'} / {OUT / 'scaling_law_fits.csv'}")


if __name__ == "__main__":
    main()
