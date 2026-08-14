#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_oracle_hist.py — 训练 HistGradientBoosting 作为第二预测器（正交验证）。
与 GBM 同数据同协议（IL-disjoint GroupKFold），输出 oracle/gbm_hist_<prop>.joblib。
"""
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for prop in PROPS:
        df = load_prop(prop)
        cat = [c for c in df.columns if c.startswith("cat_")]
        an = [c for c in df.columns if c.startswith("an_")]
        feats = cat + an + (["T"] if prop not in NO_T else [])
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all").fillna(X.median()).fillna(0.0)
        y = df["value"].to_numpy(dtype=float)
        g = df["il"].to_numpy()
        cv = GroupKFold(5)
        yt, yp = [], []
        for tr, te in cv.split(X, y, groups=g):
            m = HistGradientBoostingRegressor(random_state=0, max_iter=300)
            m.fit(X.iloc[tr], y[tr])
            yp.extend(m.predict(X.iloc[te]))
            yt.extend(y[te])
        yt, yp = np.asarray(yt), np.asarray(yp)
        r2 = 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        final = HistGradientBoostingRegressor(random_state=0, max_iter=300)
        final.fit(X, y)
        joblib.dump({"model": final, "feats": list(X.columns)}, OUT / f"hist_{prop}.joblib")
        print(f"[{prop}] IL-disjoint R2={r2:.4f} -> 已保存", flush=True)


if __name__ == "__main__":
    main()
