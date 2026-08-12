#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""perfold_si.py - 生成 SI 的 per-fold 结果 CSV

4 属性 × GroupKFold 5 折 GBM (10 RDKit 特征 + T), 输出 per-fold R²/RMSE/MAE
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]  # matmodel/
DATA = ROOT / "data" / "ilt" / "paper_dataset"
OUT = ROOT / "paper2"
FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]


def run_prop(prop, use_t=True):
    df = pd.read_csv(DATA / f"{prop}.csv", encoding="utf-8-sig")
    feats = FEATS + (["T"] if use_t and "T" in df.columns else [])
    df = df.dropna(subset=feats + ["value", "il"])
    X = df[feats].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    g = df["il"].to_numpy()
    rows = []
    # 组级
    gkf = GroupKFold(n_splits=5)
    for fi, (tr, te) in enumerate(gkf.split(X, y, g)):
        m = GradientBoostingRegressor(random_state=0)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        rows.append({"property": prop, "split": "group", "fold": fi + 1,
                     "n_train": len(tr), "n_test": len(te),
                     "R2": r2_score(y[te], p),
                     "RMSE": mean_squared_error(y[te], p) ** 0.5,
                     "MAE": mean_absolute_error(y[te], p)})
    # 点级
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for fi, (tr, te) in enumerate(kf.split(X)):
        m = GradientBoostingRegressor(random_state=0)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        rows.append({"property": prop, "split": "point", "fold": fi + 1,
                     "n_train": len(tr), "n_test": len(te),
                     "R2": r2_score(y[te], p),
                     "RMSE": mean_squared_error(y[te], p) ** 0.5,
                     "MAE": mean_absolute_error(y[te], p)})
    return pd.DataFrame(rows)


def main():
    frames = []
    for prop, use_t in [("viscosity", True), ("density", True),
                        ("conductivity", True), ("melting_point", False)]:
        print(f"running {prop} ...", flush=True)
        frames.append(run_prop(prop, use_t))
    df = pd.concat(frames, ignore_index=True)
    out = OUT / "perfold_results_si.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {out}")
    print(df.groupby(["property", "split"])["R2"].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
