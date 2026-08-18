#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""acquisition.py — 最优数据获取：模拟测量战役 + 虚拟库优先测量 Top-100（paper2 升级）

科学问题：预算有限时"下一个该测哪些 IL？"——把"数据密度是硬约束"变成
"如何最高效地填补数据密度"。

Part A（模拟测量战役，只用真实标签，不伪造）：
  随机留出 300 个 IL 模拟"未来可测的 IL"，训练集上三种获取顺序
  (random / coverage 最远优先 / uncertainty 集成分歧)，
  分批"补测"50/100/.../300 个，逐批评估对仍未测 IL 的组级 R²，
  输出"每补测 100 个 IL 涨多少 R²"曲线。

Part B（虚拟库 Top-100 优先测量清单）：
  从 8.33M 虚拟 IL（paper6 data/ions/il_ions_83m.csv）随机采样 30 万，
  过滤已测 IL，用 10 个 RDKit 描述符计算：
    - coverage gap: 到最近已测 IL 的特征距离（StandardScaler 后）
    - uncertainty:  3-seed HistGBM 集成在候选上的预测分歧（T=298.15 K）
  输出 uncertainty 排序 Top-100 与 coverage 排序 Top-100。

输出 (results/)：acquisition_results.csv / acquisition_top100.csv / fig_acquisition_curves.png
用法: python acquisition.py [--sample 300000] [--holdout 300]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "workspace" / "matmodel" / "data" / "ilt"
VIRTUAL = ROOT / "workspace" / "matmodel" / "paper6" / "data" / "ions" / "il_ions_83m.csv"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]
T_REF = 298.15

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

_FEAT_CACHE = {}


def mol_feats(mol):
    return np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), mol.GetNumHeavyAtoms(),
        Descriptors.FractionCSP3(mol), rdMolDescriptors.CalcNumRings(mol)], dtype=float)


def il_feats(cat, an):
    key = cat + "|" + an
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    mol = Chem.MolFromSmiles(cat + "." + an)
    v = mol_feats(mol) if mol is not None else None
    _FEAT_CACHE[key] = v
    return v


def make_model(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                         max_depth=7, l2_regularization=0.5,
                                         random_state=seed)


def load_prop(prop):
    use_log = prop in ("viscosity", "conductivity")
    df = pd.read_csv(DATA / f"{prop}.csv").dropna(subset=FEATS + ["T"])
    y = np.log(df["value"].to_numpy(dtype=float)) if use_log else df["value"].to_numpy(dtype=float)
    X = df[FEATS + ["T"]].to_numpy(dtype=float)
    return df, X, y


def il_rep(X, g, with_T=True):
    """每 IL 的代表特征：记录均值，温度固定 T_REF。"""
    df = pd.DataFrame(X, columns=FEATS + (["T"] if with_T else []))
    df["il"] = g
    agg = df.groupby("il")[FEATS].mean()
    if with_T:
        agg["T"] = T_REF
    return agg.reset_index(), agg


def ensemble_pred(seed_models, Xq):
    return np.mean([m.predict(Xq) for m in seed_models], axis=0), \
           np.std([m.predict(Xq) for m in seed_models], axis=0)


def part_a(prop, holdout, seeds=(0, 1, 2)):
    print(f"\n===== Part A: {prop} (holdout {holdout} ILs) =====")
    df, X, y = load_prop(prop)
    g = df["il"].to_numpy()
    ils = np.unique(g)
    rng = np.random.RandomState(0)
    held = set(rng.choice(ils, size=holdout, replace=False))
    tr_mask = np.isin(g, list(set(ils) - held))
    Xtr, ytr, gtr = X[tr_mask], y[tr_mask], g[tr_mask]
    held_mask = ~tr_mask

    # 训练初始集成（用于 uncertainty 排序）
    ens = [make_model(s) for s in seeds]
    for m in ens:
        m.fit(Xtr, ytr)
    Xrep, rep = il_rep(X, g)          # 每 IL 代表特征（含 T_REF）；rep 索引即 il
    held_ils = sorted(held)
    held_rep = np.vstack([rep.loc[i, FEATS + ["T"]].to_numpy(dtype=float) for i in held_ils])

    # 三种排序
    order_random = list(held_ils); rng.shuffle(order_random)
    from sklearn.neighbors import NearestNeighbors
    tr_rep = np.vstack([rep.loc[i, FEATS + ["T"]].to_numpy(dtype=float)
                        for i in (set(ils) - held)])
    nn = NearestNeighbors(n_neighbors=1).fit(tr_rep)
    dists, _ = nn.kneighbors(held_rep)
    order_coverage = [held_ils[i] for i in np.argsort(-dists[:, 0])]
    _, stds = ensemble_pred(ens, held_rep)
    order_unc = [held_ils[i] for i in np.argsort(-stds)]

    strategies = {"random": order_random, "coverage": order_coverage, "uncertainty": order_unc}
    rows = []
    for strat, order in strategies.items():
        for n_acq in [0, 50, 100, 150, 200, 250, 300]:
            if n_acq > len(order):
                continue
            acq = set(order[:n_acq])
            train_mask = tr_mask | np.isin(g, list(acq))
            ens2 = [make_model(s) for s in seeds]
            for m in ens2:
                m.fit(X[train_mask], y[train_mask])
            te_mask = held_mask & ~np.isin(g, list(acq))
            if te_mask.sum() == 0:
                continue
            from sklearn.metrics import mean_squared_error, r2_score
            pred = np.mean([m.predict(X[te_mask]) for m in ens2], axis=0)
            r2 = r2_score(y[te_mask], pred)
            rmse = float(np.sqrt(mean_squared_error(y[te_mask], pred)))
            rows.append({"property": prop, "strategy": strat, "n_acquired": n_acq,
                         "r2_unseen": r2, "rmse_unseen": rmse})
            print(f"  {strat:11s} acquired={n_acq:3d}  R2(unseen)={r2:+.3f}  RMSE={rmse:.3f}")
    return pd.DataFrame(rows)


def canon_il(cat, an):
    m = Chem.MolFromSmiles(cat + "." + an)
    return Chem.MolToSmiles(m) if m is not None else None


def part_b(prop, sample_n):
    print(f"\n===== Part B: virtual-library Top-100 for {prop} (sample {sample_n}) =====")
    df, X, y = load_prop(prop)
    g = df["il"].to_numpy()
    ils = np.unique(g)
    measured = set()
    for il in ils:
        cat, an = il.split("|")
        c = canon_il(cat, an)
        if c:
            measured.add(c)
    print(f"  measured canonical ILs: {len(measured)}")

    virt = pd.read_csv(VIRTUAL, usecols=["il", "cat", "an"])
    virt = virt.sample(n=min(sample_n, len(virt)), random_state=0)
    virt["canon"] = [canon_il(r.cat, r.an) for r in virt.itertuples()]
    virt = virt[virt["canon"].notna() & ~virt["canon"].isin(measured)].copy()
    print(f"  sampled {len(virt)} novel virtual ILs (after overlap filter)")

    # 描述符
    feats = np.vstack([il_feats(r.cat, r.an) for r in virt.itertuples()])
    ok = ~np.isnan(feats).any(axis=1)
    feats, virt = feats[ok], virt[ok].reset_index(drop=True)
    print(f"  descriptors ok: {len(virt)}")

    # 已测 IL 代表特征 + 缩放
    Xrep, rep = il_rep(X, g)          # rep 索引即 il
    meas_rep = np.vstack([rep.loc[i, FEATS + ["T"]].to_numpy(dtype=float) for i in ils])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(meas_rep)
    virt_feats = np.hstack([feats, np.full((len(feats), 1), T_REF)])
    virt_scaled = sc.transform(virt_feats)
    meas_scaled = sc.transform(meas_rep)

    # coverage gap
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(meas_scaled)
    dists, _ = nn.kneighbors(virt_scaled)
    virt["dist_nearest_measured"] = dists[:, 0]

    # uncertainty (3-seed ensemble, trained on all measured records)
    ens = [make_model(s) for s in (0, 1, 2)]
    for m in ens:
        m.fit(X, y)
    pred, std = ensemble_pred(ens, virt_feats)
    virt["pred_ln"] = pred
    virt["ens_std"] = std
    if prop == "viscosity":
        virt["pred_eta_mPas"] = np.exp(pred)

    # 输出 Top-100（uncertainty 与 coverage 各一份）
    top_unc = virt.sort_values("ens_std", ascending=False).head(100)
    top_cov = virt.sort_values("dist_nearest_measured", ascending=False).head(100)
    top_unc["rank_uncertainty"] = range(1, 101)
    top_cov["rank_coverage"] = range(1, 101)
    top_unc = top_unc.merge(top_cov[["canon", "rank_coverage"]], on="canon", how="left")
    cols = ["rank_uncertainty", "il", "cat", "an", "pred_ln", "ens_std",
            "dist_nearest_measured", "rank_coverage"]
    if prop == "viscosity":
        cols.insert(4, "pred_eta_mPas")
    top_unc[cols].to_csv(OUT / "acquisition_top100.csv", index=False, encoding="utf-8-sig")
    print(f"  Top-100 已存: {OUT / 'acquisition_top100.csv'}")
    print(f"  ens_std range: {virt['ens_std'].min():.3f}..{virt['ens_std'].max():.3f}")
    print(f"  dist range:    {virt['dist_nearest_measured'].min():.3f}..{virt['dist_nearest_measured'].max():.3f}")
    return virt


def plot_a(df_a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strat in ["random", "coverage", "uncertainty"]:
        s = df_a[df_a["strategy"] == strat].sort_values("n_acquired")
        ax.plot(s["n_acquired"], s["r2_unseen"], "o-", label=strat)
    ax.set_xlabel("ILs acquired (simulated measurements)"); ax.set_ylabel("group R² on unseen ILs")
    ax.set_title("Acquisition strategy value (simulated)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_acquisition_curves.png", dpi=300)
    print(f"图已存: {OUT / 'fig_acquisition_curves.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300000)
    ap.add_argument("--holdout", type=int, default=300)
    args = ap.parse_args()

    frames = []
    for prop in ["viscosity", "conductivity"]:
        frames.append(part_a(prop, args.holdout))
    df_a = pd.concat(frames, ignore_index=True)
    df_a.to_csv(OUT / "acquisition_results.csv", index=False, encoding="utf-8-sig")
    plot_a(df_a)

    part_b("viscosity", args.sample)
    print("\nDone.")


if __name__ == "__main__":
    main()
