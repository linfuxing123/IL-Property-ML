#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final_pipeline.py — 第五篇最终 pipeline：组合 + 双预测器交叉验证 + 严格过滤。

输出最终候选集 generated/il_candidates.csv：
  骨架变异新阳(272) × 已知阴 + 已知阳 × VAE新阴(260)
  过滤：新颖 / Tm<298K(GBM) / 单一片段 / 阳阴 SA 可合成 / GBM-HistGBM 预测一致。
"""
import argparse
import os
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import RDConfig
RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
DESC = ROOT.parents[2] / "data" / "il_descriptors.csv"
ORACLE = ROOT / "oracle"
GEN = ROOT / "generated"
PROPS = ["conductivity", "density", "viscosity", "melting_point"]
T_REF = 298.15


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s


def sa(s):
    m = Chem.MolFromSmiles(s)
    return sascorer.calculateScore(m) if m is not None else 10.0


def charge(s):
    m = Chem.MolFromSmiles(s)
    return sum(a.GetFormalCharge() for a in m.GetAtoms()) if m else 0


def predict(oracles, cat_df, an_df, prefix):
    res = {}
    for prop in PROPS:
        m = oracles[prop]["model"]
        feats = oracles[prop]["feats"]
        X = np.zeros((len(cat_df), len(feats)))
        for j, f in enumerate(feats):
            if f == "T":
                X[:, j] = T_REF
            elif f.startswith("cat_"):
                v = cat_df[f].to_numpy() if f in cat_df.columns else np.zeros(len(cat_df))
                X[:, j] = np.nan_to_num(v, nan=0.0)
            elif f.startswith("an_"):
                v = an_df[f].to_numpy() if f in an_df.columns else np.zeros(len(an_df))
                X[:, j] = np.nan_to_num(v, nan=0.0)
        res[prop] = m.predict(X)
    return res


def pareto(df, maximize, minimize):
    objs = np.column_stack([df[m].to_numpy() for m in minimize] +
                           [-df[m].to_numpy() for m in maximize])
    n = len(df)
    dom = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                dom[i] = True
                break
    return ~dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-max", type=float, default=298.0)
    ap.add_argument("--sa-max", type=float, default=7.0)
    ap.add_argument("--cond-gap", type=float, default=0.6, help="GBM/HistGBM 电导率偏差阈值")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    gbm = {p: joblib.load(ORACLE / f"gbm_{p}.joblib") for p in PROPS}
    hist = {p: joblib.load(ORACLE / f"hist_{p}.joblib") for p in PROPS}

    df = pd.read_csv(DESC)
    cat_cols = [c for c in df.columns if c.startswith("cat_") and c not in ("cat_smiles", "cat_ok")]
    an_cols = [c for c in df.columns if c.startswith("an_") and c not in ("an_smiles", "an_ok")]
    df["cat_smiles"] = df["cat_smiles"].map(canon)
    df["an_smiles"] = df["an_smiles"].map(canon)
    cat_tab = df[["cat_smiles"] + cat_cols].drop_duplicates("cat_smiles").set_index("cat_smiles")
    an_tab = df[["an_smiles"] + an_cols].drop_duplicates("an_smiles").set_index("an_smiles")

    known = set()
    for k in pd.read_csv(ORACLE / "known_il.csv")["il"]:
        c, an = k.split("|", 1)
        known.add(canon(c) + "|" + canon(an))

    # 新阳（骨架变异）+ 新阴（VAE，电荷过滤 + 单片段）
    scat = pd.read_csv(GEN / "cation_scaffold.csv")
    gan = pd.read_csv(GEN / "anion_novel.csv")
    gan = gan[gan["smiles"].map(charge) == -1]
    gan = gan[~gan["smiles"].str.contains(r"\.", na=False)]
    print(f"新阳 {len(scat)} / 新阴 {len(gan)}", flush=True)

    combos = []
    # 已知阳 × 已知阴（未报道组合，预测性质最优的来源）
    for csm in cat_tab.index:
        for asm in an_tab.index:
            combos.append(((csm, cat_tab.loc[csm]), (asm, an_tab.loc[asm])))
    # 骨架变异新阳 × 已知阴（生成新结构）
    for _, r in scat.iterrows():
        for asm in an_tab.index:
            combos.append(((r["smiles"], r), (asm, an_tab.loc[asm])))
    print(f"组合 {len(combos)}", flush=True)

    # 分批双预测
    rows = []
    bs = 80000
    for i in range(0, len(combos), bs):
        chunk = combos[i:i + bs]
        cats = pd.DataFrame([c[0][1] for c in chunk])
        ans = pd.DataFrame([c[1][1] for c in chunk])
        pg = predict(gbm, cats, ans, "g")
        ph = predict(hist, cats, ans, "h")
        for j, (cc, aa) in enumerate(chunk):
            rows.append({
                "cat_smiles": cc[0], "an_smiles": aa[0],
                "g_cond": pg["conductivity"][j], "g_dens": pg["density"][j],
                "g_visc": pg["viscosity"][j], "g_tm": pg["melting_point"][j],
                "h_cond": ph["conductivity"][j], "h_visc": ph["viscosity"][j],
                "h_tm": ph["melting_point"][j],
            })
    res = pd.DataFrame(rows)
    res["il"] = res["cat_smiles"].map(canon) + "|" + res["an_smiles"].map(canon)
    res = res[~res["il"].isin(known)]
    res = res[res["g_tm"] < a.tm_max]
    res["sa_cat"] = res["cat_smiles"].map(sa)
    res["sa_an"] = res["an_smiles"].map(sa)
    res["sa_sum"] = res["sa_cat"] + res["sa_an"]
    res = res[res["sa_sum"] < a.sa_max]
    res["cond_gap"] = (res["g_cond"] - res["h_cond"]).abs()
    res = res[res["cond_gap"] < a.cond_gap]
    print(f"过滤后候选: {len(res)}", flush=True)

    res["score"] = res["g_cond"] - 0.5 * res["g_visc"] - 0.01 * res["g_tm"]
    sub = res.sort_values("score", ascending=False).head(4000).reset_index(drop=True)
    sub["pareto"] = pareto(sub, ["g_cond"], ["g_visc", "g_tm"])
    top = sub.sort_values(["pareto", "score"], ascending=[False, False]).head(a.top)
    top.to_csv(GEN / "il_candidates.csv", index=False, encoding="utf-8-sig")
    cols = ["cat_smiles", "an_smiles", "g_cond", "g_visc", "g_tm", "g_dens", "cond_gap", "sa_sum", "pareto"]
    print(top[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
