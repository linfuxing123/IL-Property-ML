#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""feat_scale_exp.py — 特征规模升级实验（第二篇延续：10 核心 vs 全套 458 描述符）

协议与第二篇一致：IL 级 GroupKFold(5) + GradientBoostingRegressor（honest_cv 同款），
目标列 value（电导率/粘度已是 ln 尺度）。

对比：基线 = paper_dataset 自带 10 个组合描述符 (+T)
      全套 = il_descriptors.csv 的 cat_* + an_* (+T，熔点无 T)
输出：workspace/matmodel/feat_scale_results.csv + md
"""
import concurrent.futures
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
DS = ROOT / "workspace" / "matmodel" / "data" / "ilt" / "paper_dataset"
DESC = ROOT / "data" / "il_descriptors.csv"
OUT = ROOT / "workspace" / "matmodel"

BASE10 = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]
PROPS = ["conductivity", "density", "viscosity", "melting_point"]
NO_T = {"melting_point"}


def load_prop(prop):
    df = pd.read_csv(DS / f"{prop}.csv")
    desc = pd.read_csv(DESC)
    df = df.merge(desc.drop(columns=["cat_smiles", "an_smiles"]), on="il", how="inner")
    drop = [c for c in df.columns if c in ("cat_ok", "an_ok")]
    df = df.drop(columns=drop)
    return df


def featsets(df, prop):
    cat = [c for c in df.columns if c.startswith("cat_")]
    an = [c for c in df.columns if c.startswith("an_")]
    full = cat + an
    if prop not in NO_T:
        full = full + ["T"]
    base = BASE10 + ([] if prop in NO_T else ["T"])
    return base, full


def run_protocol(df, feats, prop):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")      # 整列无效（对离子全 NaN）直接删
    X = X.fillna(X.median()).fillna(0.0)  # 其余缺值：中位数，仍缺则补 0
    y = df["value"].to_numpy(dtype=float)
    g = df["il"].to_numpy()
    cv = GroupKFold(n_splits=5)
    yt, yp = [], []
    for tr, te in cv.split(X, y, groups=g):
        m = GradientBoostingRegressor(random_state=0)
        m.fit(X.iloc[tr], y[tr])
        yp.extend(m.predict(X.iloc[te]))
        yt.extend(y[te])
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    return {
        "R2": r2_score(yt, yp),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "n": len(yt),
        "n_il": len(np.unique(g)),
    }


def one_prop(prop):
    df = load_prop(prop)
    base, full = featsets(df, prop)
    print(f"[{prop}] 数据 {len(df)} 点 / {df['il'].nunique()} IL | 基线 {len(base)} 特征 / 全套 {len(full)} 特征", flush=True)
    rb = run_protocol(df, base, prop)
    rf = run_protocol(df, full, prop)
    row = {
        "prop": prop, "n": rb["n"], "n_il": rb["n_il"],
        "base_feats": len(base), "full_feats": len(full),
        "base_R2": round(rb["R2"], 4), "base_RMSE": round(rb["RMSE"], 4), "base_MAE": round(rb["MAE"], 4),
        "full_R2": round(rf["R2"], 4), "full_RMSE": round(rf["RMSE"], 4), "full_MAE": round(rf["MAE"], 4),
        "dR2": round(rf["R2"] - rb["R2"], 4),
    }
    print(f"  -> base R2={rb['R2']:.4f}  full R2={rf['R2']:.4f}  dR2={row['dR2']:+.4f}", flush=True)
    return row


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(PROPS)) as ex:
        rows = list(ex.map(one_prop, PROPS))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "feat_scale_results.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 特征规模升级实验（第二篇延续）",
        "",
        "协议：IL 级 GroupKFold(5) + GradientBoostingRegressor（与第二篇一致）；",
        "基线 = 10 个组合描述符(+T)，全套 = il_descriptors.csv 阳/阴各自 ~229 描述符(+T)。",
        "",
        "| 性质 | 点/IL | 基线特征 | 全套特征 | 基线 R² | 全套 R² | ΔR² | 基线 MAE | 全套 MAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {r['prop']} | {r['n']}/{r['n_il']} | {r['base_feats']} | {r['full_feats']} "
            f"| {r['base_R2']:.4f} | {r['full_R2']:.4f} | {r['dR2']:+.4f} | {r['base_MAE']:.4f} | {r['full_MAE']:.4f} |"
        )
    md = "\n".join(lines) + "\n"
    (OUT / "feat_scale_results.md").write_text(md, encoding="utf-8")
    print("结果表已写入 feat_scale_results.md", flush=True)


if __name__ == "__main__":
    main()
