#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_oracle.py — 训练并保存 4 性质 GBM(458 描述符) 预测器（逆向设计 oracle）。

协议与第四篇一致：IL-disjoint GroupKFold(5)，输入 = il_descriptors.csv 的 cat_*/an_*
（+T 对温度依赖性质），目标 = paper_dataset 的 value（电导/粘度已是 ln 尺度）。
产出：oracle/ 下每个性质一个 joblib 模型 + 训练集 IL 清单（用于新颖性判定）。
"""
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[3]
DS = ROOT / "workspace" / "matmodel" / "data" / "ilt" / "paper_dataset"
DESC = ROOT / "data" / "il_descriptors.csv"
OUT = pathlib.Path(__file__).resolve().parent / "oracle"

PROPS = ["conductivity", "density", "viscosity", "melting_point"]
NO_T = {"melting_point"}


def load_prop(prop):
    df = pd.read_csv(DS / f"{prop}.csv")
    desc = pd.read_csv(DESC)
    df = df.merge(desc.drop(columns=["cat_smiles", "an_smiles"]), on="il", how="inner")
    df = df.drop(columns=[c for c in df.columns if c in ("cat_ok", "an_ok")])
    return df


def featset(df, prop):
    cat = [c for c in df.columns if c.startswith("cat_")]
    an = [c for c in df.columns if c.startswith("an_")]
    return cat + an + (["T"] if prop not in NO_T else [])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known_il = set()
    for prop in PROPS:
        df = load_prop(prop)
        known_il |= set(df["il"].unique())
        feats = featset(df, prop)
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all").fillna(X.median()).fillna(0.0)
        y = df["value"].to_numpy(dtype=float)
        g = df["il"].to_numpy()
        cv = GroupKFold(n_splits=5)
        yt, yp = [], []
        for tr, te in cv.split(X, y, groups=g):
            m = GradientBoostingRegressor(random_state=0)
            m.fit(X.iloc[tr], y[tr])
            yp.extend(m.predict(X.iloc[te]))
            yt.extend(y[te])
        yt, yp = np.asarray(yt), np.asarray(yp)
        r2 = 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        # 全量重训（最终 oracle）
        final = GradientBoostingRegressor(random_state=0)
        final.fit(X, y)
        joblib.dump({"model": final, "feats": list(X.columns)}, OUT / f"gbm_{prop}.joblib")
        print(f"[{prop}] n={len(df)} feats={len(X.columns)} IL-disjoint R2={r2:.4f} -> 已保存", flush=True)
    # 保存训练集 IL 清单（新颖性判定用）
    pd.Series(sorted(known_il)).to_csv(OUT / "known_il.csv", index=False, header=["il"])
    print(f"训练集已知 IL 数: {len(known_il)}", flush=True)


if __name__ == "__main__":
    main()
