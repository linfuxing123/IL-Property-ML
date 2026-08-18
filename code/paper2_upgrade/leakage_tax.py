#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""leakage_tax.py — 泄漏税：点级 vs 组级膨胀随冗余度（每 IL 记录数）的变化（paper2 升级）

科学问题：数据规模变大后，随机点级划分的虚高（泄漏税 ΔR² = R2_point − R2_group）
是否随"每 IL 记录数"（冗余度）增长？→ 量化"更大的数据库需要更严的评估纪律"。

方法：按每 IL 记录数分桶（[1],[2-4],[5-9],[10-24],[25-49],[50+]），桶内分别跑
5 折 GroupKFold（组级）与 5 折 KFold（点级），报告 R²、ΔR²、泄漏率
（点级划分下测试记录中同 IL 出现在训练集的比例）。

输出 (results/)：leakage_tax_results.csv + fig_leakage_tax.png
用法: python leakage_tax.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "workspace" / "matmodel" / "data" / "ilt"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]
BINS = [(1, 1), (2, 4), (5, 9), (10, 24), (25, 49), (50, None)]


def load(prop, use_log):
    df = pd.read_csv(DATA / f"{prop}.csv").dropna(subset=FEATS + ["T"])
    feats = FEATS + ["T"]
    y = np.log(df["value"].to_numpy(dtype=float)) if use_log else df["value"].to_numpy(dtype=float)
    return df, df[feats].to_numpy(dtype=float), y


def run_cv(X, y, g, seed, group_disjoint):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, KFold
    cv = GroupKFold(n_splits=5) if group_disjoint else KFold(n_splits=5, shuffle=True, random_state=seed)
    yt, yp = [], []
    leak_records = 0
    for tr, te in cv.split(X, y, groups=g if group_disjoint else None):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                          max_depth=7, l2_regularization=0.5,
                                          random_state=seed)
        m.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te]))
        if not group_disjoint:
            g_tr = set(g[tr]); g_te = set(g[te])
            leak_records += sum(1 for gi in g[te] if gi in g_tr)
    r2 = r2_score(np.asarray(yt), np.asarray(yp))
    leak = leak_records / len(yt) if not group_disjoint else 0.0
    return r2, leak


def main():
    props = {"viscosity": True, "conductivity": True, "density": False}
    rows = []
    for prop, use_log in props.items():
        df, X, y = load(prop, use_log)
        g = df["il"].to_numpy()
        sizes = df.groupby("il").size()
        print(f"\n===== {prop}: {len(X)} records / {len(sizes)} ILs =====")
        for lo, hi in BINS:
            il_bin = sizes[(sizes >= lo) & (sizes <= hi if hi else True)].index
            mask = np.isin(g, il_bin)
            if mask.sum() < 30 or len(il_bin) < 10:
                print(f"  bin [{lo}-{hi or '∞'}]: skip ({len(il_bin)} ILs / {mask.sum()} recs)")
                continue
            r2g, _ = run_cv(X[mask], y[mask], g[mask], 0, True)
            r2p, leak = run_cv(X[mask], y[mask], g[mask], 0, False)
            med = float(np.median(sizes[il_bin]))
            rows.append({"property": prop, "bin": f"[{lo}-{hi or 'inf'}]", "n_il": len(il_bin),
                         "n_records": int(mask.sum()), "median_records_per_il": med,
                         "r2_group": r2g, "r2_point": r2p, "delta_r2": r2p - r2g,
                         "leak_rate": leak})
            print(f"  bin [{lo}-{hi or '∞':>3}] ILs={len(il_bin):4d} recs={mask.sum():6d} "
                  f"med={med:5.1f}  R2g={r2g:+.3f} R2p={r2p:+.3f} Δ={r2p - r2g:+.3f} leak={100 * leak:.0f}%")
        # 全量总览（对照论文 Table 2 的 ΔR²）
        r2g, _ = run_cv(X, y, g, 0, True)
        r2p, leak = run_cv(X, y, g, 0, False)
        rows.append({"property": prop, "bin": "ALL", "n_il": len(sizes),
                     "n_records": len(X), "median_records_per_il": float(np.median(sizes)),
                     "r2_group": r2g, "r2_point": r2p, "delta_r2": r2p - r2g, "leak_rate": leak})
        print(f"  ALL        ILs={len(sizes):4d} recs={len(X):6d} "
              f"R2g={r2g:+.3f} R2p={r2p:+.3f} Δ={r2p - r2g:+.3f} leak={100 * leak:.0f}%")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "leakage_tax_results.csv", index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key in zip(axes, ["delta_r2", "r2_group"]):
        for prop in props:
            s = res[(res["property"] == prop) & (res["bin"] != "ALL")]
            ax.plot(s["median_records_per_il"], s[key], "o-", label=prop)
        ax.set_xscale("log"); ax.set_xlabel("median records per IL")
        ax.set_ylabel("ΔR² (point − group)" if key == "delta_r2" else "R²")
        ax.set_title("Leakage tax vs redundancy" if key == "delta_r2" else "R² by redundancy bin")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_leakage_tax.png", dpi=300)
    print(f"\n图已存: {OUT / 'fig_leakage_tax.png'}")
    print(f"结果已存: {OUT / 'leakage_tax_results.csv'}")


if __name__ == "__main__":
    main()
