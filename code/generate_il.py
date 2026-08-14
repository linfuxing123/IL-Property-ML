#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_il.py — 把 VAE 生成的新离子接入逆向设计：组合 + 预测 + Pareto + 可合成性。

组合空间：
  新阳(395) × 已知阴(300)  +  已知阳(795) × 新阴(468)  +  新阳 × 新阴
过滤：新颖（不在 1,891 已知 IL）+ Tm<298K。
排序：多目标 score = ln_cond - 0.5*ln_visc - 0.01*tm，取 Pareto + score top。
验证：RDKit 有效性 + SA 可合成性（阳/阴各自打分）。
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


def load_ion_tables():
    df = pd.read_csv(DESC)
    cat_cols = [c for c in df.columns if c.startswith("cat_") and c not in ("cat_smiles", "cat_ok")]
    an_cols = [c for c in df.columns if c.startswith("an_") and c not in ("an_smiles", "an_ok")]
    df["cat_smiles"] = df["cat_smiles"].map(canon)
    df["an_smiles"] = df["an_smiles"].map(canon)
    cat_tab = df[["cat_smiles"] + cat_cols].drop_duplicates("cat_smiles").reset_index(drop=True)
    an_tab = df[["an_smiles"] + an_cols].drop_duplicates("an_smiles").reset_index(drop=True)
    return cat_tab, an_tab


def sa(smiles):
    m = Chem.MolFromSmiles(smiles)
    return sascorer.calculateScore(m) if m is not None else 10.0


def charge(smiles):
    m = Chem.MolFromSmiles(smiles)
    return sum(a.GetFormalCharge() for a in m.GetAtoms()) if m is not None else 0


def predict(oracles, cat_df, an_df):
    """cat_df/an_df 含描述符列；返回 4 性质预测。"""
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


def combine_predict(oracles, cats, ans):
    """cats/ans: list of (smiles, desc_series)。返回预测结果 DataFrame。"""
    n = len(cats)
    cat_s = [c[0] for c in cats]
    an_s = [a[0] for a in ans]
    cat_df = pd.DataFrame([c[1] for c in cats])
    an_df = pd.DataFrame([a[1] for a in ans])
    pred = predict(oracles, cat_df, an_df)
    return pd.DataFrame({
        "cat_smiles": cat_s, "an_smiles": an_s,
        "ln_cond": pred["conductivity"], "density": pred["density"],
        "ln_visc": pred["viscosity"], "tm": pred["melting_point"],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-max", type=float, default=298.0)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    oracles = {p: joblib.load(ORACLE / f"gbm_{p}.joblib") for p in PROPS}
    cat_tab, an_tab = load_ion_tables()
    known_raw = pd.read_csv(ORACLE / "known_il.csv")["il"].tolist()
    known = set()
    for k in known_raw:
        if "|" in k:
            c, an = k.split("|", 1)
            known.add(canon(c) + "|" + canon(an))

    gen_cat = pd.read_csv(GEN / "cation_novel.csv")
    gen_an = pd.read_csv(GEN / "anion_novel.csv")
    gen_cat = gen_cat[gen_cat["smiles"].map(charge) == 1].reset_index(drop=True)
    gen_an = gen_an[gen_an["smiles"].map(charge) == -1].reset_index(drop=True)
    print(f"新阳 {len(gen_cat)} / 新阴 {len(gen_an)}", flush=True)

    # 已知描述符 dict
    cat_desc = {r["cat_smiles"]: r for _, r in cat_tab.iterrows()}
    an_desc = {r["an_smiles"]: r for _, r in an_tab.iterrows()}

    combos = []
    # 新阳 × 已知阴
    for _, r in gen_cat.iterrows():
        for asm, ad in an_desc.items():
            combos.append(((r["smiles"], r), (asm, ad)))
    # 已知阳 × 新阴
    for csm, cd in cat_desc.items():
        for _, r in gen_an.iterrows():
            combos.append(((csm, cd), (r["smiles"], r)))
    print(f"组合总数: {len(combos)}", flush=True)

    # 分批预测（避免内存爆）
    res = []
    bs = 100000
    for i in range(0, len(combos), bs):
        chunk = combos[i:i + bs]
        cats = [c[0] for c in chunk]
        ans = [c[1] for c in chunk]
        r = combine_predict(oracles, cats, ans)
        res.append(r)
    res = pd.concat(res, ignore_index=True)
    res["il"] = res["cat_smiles"].map(canon) + "|" + res["an_smiles"].map(canon)
    res = res[~res["il"].isin(known) & (res["tm"] < a.tm_max)].reset_index(drop=True)
    print(f"新颖且 Tm<{a.tm_max}K: {len(res)}", flush=True)

    res["score"] = res["ln_cond"] - 0.5 * res["ln_visc"] - 0.01 * res["tm"]
    sub = res.sort_values("score", ascending=False).head(4000).reset_index(drop=True)
    sub["pareto"] = pareto(sub, ["ln_cond"], ["ln_visc", "tm"])
    top = sub.sort_values(["pareto", "score"], ascending=[False, False]).head(a.top)

    top["sa_cat"] = top["cat_smiles"].map(sa)
    top["sa_an"] = top["an_smiles"].map(sa)
    top["sa_sum"] = top["sa_cat"] + top["sa_an"]
    top.to_csv(ROOT / "generated_il_top.csv", index=False, encoding="utf-8-sig")
    cols = ["cat_smiles", "an_smiles", "ln_cond", "ln_visc", "tm", "density", "sa_sum", "pareto"]
    print(top[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
