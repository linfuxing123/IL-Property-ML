#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""feat_density_interaction.py — 特征增益 vs 数据密度 交互分析（第三篇核心图）

协议：IL 级 GroupKFold(5)，HistGradientBoostingRegressor（多线程，快 10~50 倍），
      baseline（10 核心）vs full（全套 458 描述符）。
分层：按每个 IL 在全集的样本数分桶（1 / 2-3 / 4-9 / 10-24 / 25+），
      桶内样本级 R²/MAE，观察特征升级在稀疏组 vs 稠密组的增益差异。
输出：feat_density_interaction.md + feat_density_interaction.png（2x2，300dpi）
"""
import concurrent.futures
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from feat_scale_exp import PROPS, featsets, load_prop

OUT = pathlib.Path(__file__).resolve().parent
BUCKETS = ["1", "2-3", "4-9", "10-24", "25+"]


def bucket_of(n):
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 9:
        return "4-9"
    if n <= 24:
        return "10-24"
    return "25+"


def run_prop(prop):
    df = load_prop(prop)
    gsize = df.groupby("il").size().rename("gsize")
    df = df.merge(gsize, on="il")
    df["bucket"] = df["gsize"].map(bucket_of)
    base, full = featsets(df, prop)
    cv = GroupKFold(n_splits=5)
    cols = ["il", "bucket", "value", "pred", "set"]
    parts = []
    for name, feats in (("base", base), ("full", full)):
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all").fillna(X.median()).fillna(0.0)
        y = df["value"].to_numpy(dtype=float)
        g = df["il"].to_numpy()
        Xn = X.to_numpy()
        for tr, te in cv.split(Xn, y, groups=g):
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
            m.fit(Xn[tr], y[tr])
            sub = df.iloc[te].copy()
            sub["pred"] = m.predict(Xn[te])
            sub["set"] = name
            parts.append(sub[cols])
    res = pd.concat(parts, ignore_index=True)
    rows = []
    for b in BUCKETS:
        d = res[res["bucket"] == b]
        if len(d) < 2:
            continue
        for name in ("base", "full"):
            dd = d[d["set"] == name]
            if len(dd) < 2:
                continue
            rows.append({
                "prop": prop, "bucket": b, "set": name, "n": len(dd),
                "n_il": dd["il"].nunique(),
                "R2": r2_score(dd["value"], dd["pred"]),
                "MAE": mean_absolute_error(dd["value"], dd["pred"]),
            })
    print(f"[{prop}] 分层完成", flush=True)
    return prop, pd.DataFrame(rows)


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(PROPS)) as ex:
        results = dict(ex.map(run_prop, PROPS))
    all_rows = pd.concat(results.values(), ignore_index=True)
    all_rows.to_csv(OUT / "feat_density_interaction.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 特征增益 vs 数据密度（交互分析）",
        "",
        "协议：IL 级 GroupKFold(5) + HistGradientBoostingRegressor（多线程）；",
        "分层 = 每组样本数：1 / 2-3 / 4-9 / 10-24 / 25+，桶内样本级 R²。",
        "",
        "| 性质 | 桶 | 基线 R² | 全套 R² | ΔR² | 基线 MAE | 全套 MAE | 样本数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for prop in PROPS:
        for b in BUCKETS:
            d = all_rows[(all_rows["prop"] == prop) & (all_rows["bucket"] == b)]
            if len(d) < 2:
                continue
            rb = d[d["set"] == "base"].iloc[0]
            rf = d[d["set"] == "full"].iloc[0]
            lines.append(
                f"| {prop} | {b} | {rb['R2']:.3f} | {rf['R2']:.3f} | {rf['R2']-rb['R2']:+.3f} "
                f"| {rb['MAE']:.3f} | {rf['MAE']:.3f} | {rb['n']} |"
            )
    md = "\n".join(lines) + "\n"
    (OUT / "feat_density_interaction.md").write_text(md, encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, prop in zip(axes.ravel(), PROPS):
        d = all_rows[all_rows["prop"] == prop]
        base_r2 = [d[(d["bucket"] == b) & (d["set"] == "base")]["R2"].iloc[0] if len(d[(d["bucket"] == b) & (d["set"] == "base")]) else np.nan for b in BUCKETS]
        full_r2 = [d[(d["bucket"] == b) & (d["set"] == "full")]["R2"].iloc[0] if len(d[(d["bucket"] == b) & (d["set"] == "full")]) else np.nan for b in BUCKETS]
        xs = np.arange(len(BUCKETS))
        ax.plot(xs, base_r2, "o--", label="10 core desc")
        ax.plot(xs, full_r2, "s-", label="full 458 desc")
        for i, (bb, ff) in enumerate(zip(base_r2, full_r2)):
            if not (np.isnan(bb) or np.isnan(ff)):
                ax.annotate(f"{ff-bb:+.2f}", (i, ff), textcoords="offset points", xytext=(0, 8), fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(BUCKETS)
        ax.set_title(prop)
        ax.set_xlabel("samples per IL")
        ax.set_ylabel("group-disjoint R2")
        ax.legend(fontsize=8)
    fig.suptitle("Feature scale gain vs data density (IL-disjoint CV)")
    fig.tight_layout()
    fig.savefig(OUT / "feat_density_interaction.png", dpi=300)
    print("结果: feat_density_interaction.md + png", flush=True)


if __name__ == "__main__":
    main()
