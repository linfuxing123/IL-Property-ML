#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_hist_v2.py — 第 6 篇：扩充数据集上重训 HistGBM 预测器（第二预测器）。

协议与 train_oracle_v2 一致（IL-disjoint GroupKFold(5)，同特征/尺度），
产出 data/expanded/oracle_v2/hist_{prop}.joblib + hist_r2_summary.csv
对比基准（paper5 HistGBM）：0.749/0.949/0.846/0.561
"""
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

P6 = pathlib.Path(__file__).resolve().parent
DS = P6 / "data" / "expanded" / "paper_dataset"
DESC = P6 / "data" / "il_descriptors_v2.csv"
OUT = P6 / "data" / "expanded" / "oracle_v2"

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
    rows = []
    for prop in PROPS:
        df = load_prop(prop)
        feats = featset(df, prop)
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all").fillna(X.median()).fillna(0.0)
        y = df["value"].to_numpy(dtype=float)
        g = df["il"].to_numpy()
        cv = GroupKFold(n_splits=5)
        yt, yp = [], []
        for tr, te in cv.split(X, y, groups=g):
            m = HistGradientBoostingRegressor(random_state=0)
            m.fit(X.iloc[tr], y[tr])
            yp.extend(m.predict(X.iloc[te]))
            yt.extend(y[te])
        yt, yp = np.asarray(yt), np.asarray(yp)
        r2 = 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        final = HistGradientBoostingRegressor(random_state=0)
        final.fit(X, y)
        joblib.dump({"model": final, "feats": list(X.columns)}, OUT / f"hist_{prop}.joblib")
        rows.append((prop, len(df), df["il"].nunique(), r2))
        print(f"[{prop}] n={len(df)} IL={df['il'].nunique()} R2={r2:.4f}", flush=True)
    pd.DataFrame(rows, columns=["prop", "n", "n_il", "r2"]).to_csv(
        OUT / "hist_r2_summary.csv", index=False)


if __name__ == "__main__":
    main()
