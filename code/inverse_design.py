#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inverse_design.py — 逆向设计核心：组合空间 + 多性质预测 + 多目标 Pareto 筛选。

流程：
  1. 从 il_descriptors.csv 提取去重后的阳/阴离子描述符表（867 阳 × 356 阴）。
  2. 全量组合（约 30.9 万）→ 拼接描述符 + T=298.15K → 4 性质 GBM 预测。
  3. 过滤掉已有 IL，再按约束（室温液态 Tm<298K）过滤。
  4. 多目标 Pareto（max 电导率, min 粘度, min 熔点）+ 目标排序，输出 Top 候选。
"""
import argparse
import pathlib

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[3]
DESC = ROOT / "data" / "il_descriptors.csv"
ORACLE = pathlib.Path(__file__).resolve().parent / "oracle"
OUT = pathlib.Path(__file__).resolve().parent

PROPS = ["conductivity", "density", "viscosity", "melting_point"]
T_REF = 298.15


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s


def load_ion_tables():
    df = pd.read_csv(DESC)
    cat_cols = [c for c in df.columns if c.startswith("cat_") and c not in ("cat_smiles", "cat_ok")]
    an_cols = [c for c in df.columns if c.startswith("an_") and c not in ("an_smiles", "an_ok")]
    df["cat_smiles"] = df["cat_smiles"].map(canon)
    df["an_smiles"] = df["an_smiles"].map(canon)
    cat_tab = df[["cat_smiles"] + cat_cols].drop_duplicates("cat_smiles").reset_index(drop=True)
    an_tab = df[["an_smiles"] + an_cols].drop_duplicates("an_smiles").reset_index(drop=True)
    return cat_tab, an_tab, cat_cols, an_cols


def predict_batch(oracles, Xcat, Xan, T=T_REF):
    """Xcat/Xan 为描述符 DataFrame（列已对齐）。返回 4 性质预测 DataFrame。"""
    # 每个预测器的特征列顺序（joblib 里保存）
    out = {}
    for prop in PROPS:
        m = oracles[prop]["model"]
        feats = oracles[prop]["feats"]
        no_t = prop == "melting_point"
        # 构建特征矩阵：cat 特征 + an 特征 + T
        # 特征名形如 cat_xxx / an_xxx / T
        cols = []
        X = np.zeros((len(Xcat), len(feats)))
        for j, f in enumerate(feats):
            if f == "T":
                X[:, j] = T
            elif f.startswith("cat_"):
                v = Xcat[f].to_numpy() if f in Xcat.columns else np.zeros(len(Xcat))
                X[:, j] = np.nan_to_num(v, nan=0.0)
            elif f.startswith("an_"):
                v = Xan[f].to_numpy() if f in Xan.columns else np.zeros(len(Xan))
                X[:, j] = np.nan_to_num(v, nan=0.0)
        out[prop] = m.predict(X)
    return out


def pareto_frontier(df, maximize, minimize):
    """简单非支配排序：返回是否在 Pareto 前沿。"""
    n = len(df)
    objs = np.column_stack([df[m].to_numpy() for m in minimize] +
                           [-df[m].to_numpy() for m in maximize])
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                dominated[i] = True
                break
    return ~dominated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-combos", type=int, default=0, help="0=全量")
    ap.add_argument("--tm-max", type=float, default=298.0, help="熔点上限（室温液态约束）")
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args()

    cat_tab, an_tab, cat_cols, an_cols = load_ion_tables()
    print(f"阳离子 {len(cat_tab)} / 阴离子 {len(an_tab)}", flush=True)

    oracles = {p: joblib.load(ORACLE / f"gbm_{p}.joblib") for p in PROPS}
    known_raw = pd.read_csv(ORACLE / "known_il.csv")["il"].tolist()
    known = set()
    for k in known_raw:
        if "|" in k:
            c, an = k.split("|", 1)
            known.add(canon(c) + "|" + canon(an))
        else:
            known.add(canon(k))

    # 组合
    n_cat, n_an = len(cat_tab), len(an_tab)
    total = n_cat * n_an
    if a.max_combos and a.max_combos < total:
        rng = np.random.default_rng(0)
        idx = rng.choice(total, size=a.max_combos, replace=False)
    else:
        idx = np.arange(total)
    ci = idx // n_an
    ai = idx % n_an

    Xcat = cat_tab.iloc[ci][cat_cols].reset_index(drop=True)
    Xan = an_tab.iloc[ai][an_cols].reset_index(drop=True)
    cat_s = cat_tab["cat_smiles"].iloc[ci].reset_index(drop=True)
    an_s = an_tab["an_smiles"].iloc[ai].reset_index(drop=True)

    pred = predict_batch(oracles, Xcat, Xan)
    res = pd.DataFrame({
        "cat_smiles": cat_s,
        "an_smiles": an_s,
        "ln_cond": pred["conductivity"],
        "density": pred["density"],
        "ln_visc": pred["viscosity"],
        "tm": pred["melting_point"],
    })
    res["il"] = res["cat_smiles"] + "|" + res["an_smiles"]
    res["novel"] = ~res["il"].isin(known)
    # 约束：室温液态（熔点 < 阈值）
    res = res[res["novel"] & (res["tm"] < a.tm_max)].reset_index(drop=True)
    print(f"组合 {len(idx)} -> 新颖且 Tm<{a.tm_max}K: {len(res)}", flush=True)

    # 综合排序（电导率 + 低粘度 + 低熔点 的加权）
    res["score"] = res["ln_cond"] - 0.5 * res["ln_visc"] - 0.01 * res["tm"]
    # 先取 top 3000 做 Pareto（O(n^2) 只在小集合上）
    sub = res.sort_values("score", ascending=False).head(3000).reset_index(drop=True)
    sub["pareto"] = pareto_frontier(sub, maximize=["ln_cond"], minimize=["ln_visc", "tm"])
    n_pareto = int(sub["pareto"].sum())
    print(f"top3000 内 Pareto 前沿: {n_pareto}", flush=True)

    top = sub.sort_values(["pareto", "score"], ascending=[False, False]).head(a.top)
    top.to_csv(OUT / "inverse_design_top.csv", index=False, encoding="utf-8-sig")
    print(top[["cat_smiles", "an_smiles", "ln_cond", "ln_visc", "tm", "density", "pareto"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
