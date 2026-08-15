#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_oracle_v2.py — 第 6 篇：扩充数据集上重训 4 性质 GBM 预测器。

协议与 paper5 完全一致：IL-disjoint GroupKFold(5)，输入 = il_descriptors_v2.csv
的 cat_*/an_*（+T），目标 = data/expanded/paper_dataset 的 value（同尺度）。
对比基准（paper5 GBM，1,891 IL）：
  conductivity 0.740 / density 0.926 / viscosity 0.809 / melting_point 0.523
产出：data/expanded/oracle_v2/gbm_{prop}.joblib + r2_summary.csv
"""
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[3]
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
        rows.append((prop, len(df), df["il"].nunique(), len(X.columns), r2))
        print(f"[{prop}] n={len(df)} IL={df['il'].nunique()} feats={len(X.columns)} "
              f"IL-disjoint R2={r2:.4f}", flush=True)
    pd.DataFrame(rows, columns=["prop", "n", "n_il", "n_feats", "r2"]).to_csv(
        OUT / "r2_summary.csv", index=False)
    pd.Series(sorted(known_il)).to_csv(OUT / "known_il_v2.csv", index=False, header=["il"])
    print(f"训练集已知 IL 总数: {len(known_il)}", flush=True)


if __name__ == "__main__":
    main()
