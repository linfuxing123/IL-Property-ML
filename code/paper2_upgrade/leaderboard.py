#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""leaderboard.py — 同一组级分割下的多模型对照（paper2 升级）

同一 5 折 GroupKFold(seed=0) 分割下比较：LinearRegression / RandomForest /
GBR（论文口径）/ HistGBM，逐属性报告 R² / RMSE / MAE；
并给出点级对照（KFold）展示泄漏膨胀（每属性用 HistGBM）。

输出 (results/)：leaderboard.csv + fig_leaderboard.png
用法: python leaderboard.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "workspace" / "matmodel" / "data" / "ilt"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]
PROPS = {"viscosity": True, "conductivity": True, "density": False, "melting_point": False}


def load(prop, use_log):
    csv = f"{prop}.csv" if prop != "melting_point" else "melting_point_all.csv"
    df = pd.read_csv(DATA / csv).dropna(subset=FEATS)
    has_t = prop != "melting_point"
    if has_t:
        df = df.dropna(subset=["T"])
        feats = FEATS + ["T"]
    else:
        feats = list(FEATS)
    y = np.log(df["value"].to_numpy(dtype=float)) if use_log else df["value"].to_numpy(dtype=float)
    return df, df[feats].to_numpy(dtype=float), y, feats


def make_model(name, seed=0):
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import LinearRegression
    if name == "LR":
        return LinearRegression()
    if name == "RF":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "GBR":
        return GradientBoostingRegressor(random_state=seed)
    if name == "HistGBM":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                             max_depth=7, l2_regularization=0.5,
                                             random_state=seed)
    raise ValueError(name)


def cv_metrics(X, y, g, name, group_disjoint, seed=0):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold, KFold
    cv = GroupKFold(n_splits=5) if group_disjoint else KFold(n_splits=5, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tr, te in cv.split(X, y, groups=g if group_disjoint else None):
        m = make_model(name, seed)
        m.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te]))
    yt, yp = np.asarray(yt), np.asarray(yp)
    return {"R2": r2_score(yt, yp), "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp))}


def main():
    rows = []
    for prop, use_log in PROPS.items():
        df, X, y, feats = load(prop, use_log)
        g = df["il"].to_numpy()
        print(f"\n===== {prop}: {len(X)} records / {len(np.unique(g))} ILs =====")
        for name in ["LR", "RF", "GBR", "HistGBM"]:
            m = cv_metrics(X, y, g, name, group_disjoint=True)
            rows.append({"property": prop, "model": name, "split": "group", **m})
            print(f"  {name:8s} group  R2={m['R2']:+.3f} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f}")
        mp = cv_metrics(X, y, g, "HistGBM", group_disjoint=False)
        rows.append({"property": prop, "model": "HistGBM", "split": "point", **mp})
        print(f"  {'HistGBM':8s} point  R2={mp['R2']:+.3f} RMSE={mp['RMSE']:.3f} MAE={mp['MAE']:.3f} "
              f"dR2={mp['R2'] - rows[-2]['R2']:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "leaderboard.csv", index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    w = 0.18
    for i, prop in enumerate(PROPS):
        s = res[res["property"] == prop]
        names = ["LR", "RF", "GBR", "HistGBM"]
        vals = [s[(s["model"] == n) & (s["split"] == "group")]["R2"].iloc[0] for n in names]
        ax.bar([i + j * w for j in range(4)], vals, width=w, label=prop if i == 0 else None)
        ax.set_xticks([i + 1.5 * w for i in range(4)])
        ax.set_xticklabels(PROPS)
    ax.set_ylabel("group R²"); ax.legend(title="model"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_leaderboard.png", dpi=300)
    print(f"\n图已存: {OUT / 'fig_leaderboard.png'}")
    print(f"结果已存: {OUT / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
