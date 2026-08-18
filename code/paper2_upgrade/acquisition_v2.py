#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""acquisition_v2.py — 获取策略 v2：固定评估集协议（paper2 升级）

v1 的曲线中段非单调，原因是"评估集随获取而缩小漂移"。v2 修正：
  - 评估池 = 150 IL（固定，永不参与获取）
  - 获取池 = 300 IL（模拟"未来可测"，按策略排序分批"补测"）
  - 训练集 = 其余全部
在固定评估池上报告组级 R² / RMSE，曲线应单调。

输出 (results/)：acquisition_v2_results.csv + fig_acquisition_v2.png
用法: python acquisition_v2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "workspace" / "matmodel" / "data" / "ilt"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]
T_REF = 298.15
EVAL_N, POOL_N = 150, 300


def make_model(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                         max_depth=7, l2_regularization=0.5,
                                         random_state=seed)


def load_prop(prop):
    use_log = prop in ("viscosity", "conductivity")
    df = pd.read_csv(DATA / f"{prop}.csv").dropna(subset=FEATS + ["T"])
    y = np.log(df["value"].to_numpy(dtype=float)) if use_log else df["value"].to_numpy(dtype=float)
    return df, df[FEATS + ["T"]].to_numpy(dtype=float), y


def il_rep(X, g):
    df = pd.DataFrame(X, columns=FEATS + ["T"])
    df["il"] = g
    agg = df.groupby("il")[FEATS].mean()
    agg["T"] = T_REF
    return agg


def run(prop):
    print(f"\n===== Part A(v2): {prop} =====")
    df, X, y = load_prop(prop)
    g = df["il"].to_numpy()
    ils = np.unique(g)
    rng = np.random.RandomState(0)
    perm = rng.permutation(ils)
    eval_ils = set(perm[:EVAL_N])          # 固定评估池
    pool_ils = set(perm[EVAL_N:EVAL_N + POOL_N])  # 获取池
    train_ils = set(perm[EVAL_N + POOL_N:])
    print(f"  train {len(train_ils)} / pool {len(pool_ils)} / eval {len(eval_ils)} ILs")

    tr_mask = np.isin(g, list(train_ils))
    ev_mask = np.isin(g, list(eval_ils))
    Xtr, ytr, gtr = X[tr_mask], y[tr_mask], g[tr_mask]

    rep = il_rep(X, g)
    ens0 = [make_model(s) for s in (0, 1, 2)]
    for m in ens0:
        m.fit(Xtr, ytr)
    pool_rep = np.vstack([rep.loc[i, FEATS + ["T"]].to_numpy(dtype=float) for i in sorted(pool_ils)])
    train_rep = np.vstack([rep.loc[i, FEATS + ["T"]].to_numpy(dtype=float) for i in sorted(train_ils)])

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(train_rep)
    dists, _ = nn.kneighbors(pool_rep)
    order_cov = [sorted(pool_ils)[i] for i in np.argsort(-dists[:, 0])]
    _, stds = np.mean([m.predict(pool_rep) for m in ens0], axis=0), \
              np.std([m.predict(pool_rep) for m in ens0], axis=0)
    order_unc = [sorted(pool_ils)[i] for i in np.argsort(-stds)]
    order_rnd = list(pool_ils); rng.shuffle(order_rnd)

    from sklearn.metrics import mean_squared_error, r2_score
    rows = []
    for strat, order in [("random", order_rnd), ("coverage", order_cov), ("uncertainty", order_unc)]:
        for n_acq in [0, 50, 100, 150, 200, 250, 300]:
            acq = set(order[:n_acq])
            tmask = tr_mask | np.isin(g, list(acq))
            ens = [make_model(s) for s in (0, 1, 2)]
            for m in ens:
                m.fit(X[tmask], y[tmask])
            pred = np.mean([m.predict(X[ev_mask]) for m in ens], axis=0)
            r2 = r2_score(y[ev_mask], pred)
            rmse = float(np.sqrt(mean_squared_error(y[ev_mask], pred)))
            rows.append({"property": prop, "strategy": strat, "n_acquired": n_acq,
                         "r2_eval_fixed": r2, "rmse_eval_fixed": rmse})
            print(f"  {strat:11s} n={n_acq:3d}  R2(fixed eval)={r2:+.3f}  RMSE={rmse:.3f}")
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = [run(p) for p in ["viscosity", "conductivity"]]
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(OUT / "acquisition_v2_results.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, prop in zip(axes, ["viscosity", "conductivity"]):
        for strat in ["random", "coverage", "uncertainty"]:
            s = res[(res["property"] == prop) & (res["strategy"] == strat)].sort_values("n_acquired")
            ax.plot(s["n_acquired"], s["r2_eval_fixed"], "o-", label=strat)
        ax.set_xlabel("ILs acquired"); ax.set_ylabel("R² on fixed eval pool")
        ax.set_title(prop); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_acquisition_v2.png", dpi=300)
    print(f"\n图已存: {OUT / 'fig_acquisition_v2.png'}")
    print(f"结果已存: {OUT / 'acquisition_v2_results.csv'}")


if __name__ == "__main__":
    main()
