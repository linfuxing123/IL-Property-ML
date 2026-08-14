#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_candidates.py — 对组合筛选的高分候选做 HistGBM 交叉验证，产出最终候选集。

读 inverse_design_top.csv（GBM 高分候选），用第二预测器 HistGBM 复测 4 性质，
计算一致性（cond_gap / visc_gap / tm_gap），保留双预测一致的候选，附 SA。
"""
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
ORACLE = ROOT / "oracle"
DESC = ROOT.parents[2] / "data" / "il_descriptors.csv"
PROPS = ["conductivity", "density", "viscosity", "melting_point"]
T_REF = 298.15


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s


def sa(s):
    m = Chem.MolFromSmiles(s)
    return sascorer.calculateScore(m) if m else 10.0


def predict(oracles, cat_desc, an_desc):
    out = {}
    for prop in PROPS:
        m = oracles[prop]["model"]
        feats = oracles[prop]["feats"]
        X = np.zeros((1, len(feats)))
        for j, f in enumerate(feats):
            if f == "T":
                X[0, j] = T_REF
            elif f.startswith("cat_"):
                X[0, j] = np.nan_to_num(cat_desc.get(f, 0.0), nan=0.0)
            elif f.startswith("an_"):
                X[0, j] = np.nan_to_num(an_desc.get(f, 0.0), nan=0.0)
        out[prop] = float(m.predict(X)[0])
    return out


def main():
    gbm = {p: joblib.load(ORACLE / f"gbm_{p}.joblib") for p in PROPS}
    hist = {p: joblib.load(ORACLE / f"hist_{p}.joblib") for p in PROPS}
    df = pd.read_csv(DESC)
    cat_cols = [c for c in df.columns if c.startswith("cat_") and c not in ("cat_smiles", "cat_ok")]
    an_cols = [c for c in df.columns if c.startswith("an_") and c not in ("an_smiles", "an_ok")]
    df["cat_smiles"] = df["cat_smiles"].map(canon)
    df["an_smiles"] = df["an_smiles"].map(canon)
    cat_tab = df[["cat_smiles"] + cat_cols].drop_duplicates("cat_smiles").set_index("cat_smiles")
    an_tab = df[["an_smiles"] + an_cols].drop_duplicates("an_smiles").set_index("an_smiles")

    top = pd.read_csv(ROOT / "inverse_design_top.csv")
    rows = []
    for _, r in top.iterrows():
        cs = canon(r["cat_smiles"])
        asm = canon(r["an_smiles"])
        if cs not in cat_tab.index or asm not in an_tab.index:
            continue
        cat_desc = cat_tab.loc[cs].to_dict()
        an_desc = an_tab.loc[asm].to_dict()
        pg = predict(gbm, cat_desc, an_desc)
        ph = predict(hist, cat_desc, an_desc)
        rows.append({
            "cat_smiles": cs, "an_smiles": asm,
            "g_cond": pg["conductivity"], "g_visc": pg["viscosity"], "g_tm": pg["melting_point"], "g_dens": pg["density"],
            "h_cond": ph["conductivity"], "h_visc": ph["viscosity"], "h_tm": ph["melting_point"],
            "cond_gap": abs(pg["conductivity"] - ph["conductivity"]),
            "visc_gap": abs(pg["viscosity"] - ph["viscosity"]),
            "tm_gap": abs(pg["melting_point"] - ph["melting_point"]),
            "sa_sum": sa(cs) + sa(asm),
        })
    res = pd.DataFrame(rows)
    res = res[(res["cond_gap"] < 0.6) & (res["tm_gap"] < 15)]
    res = res.sort_values("g_cond", ascending=False)
    res.to_csv(ROOT / "generated" / "final_candidates.csv", index=False, encoding="utf-8-sig")
    cols = ["cat_smiles", "an_smiles", "g_cond", "g_visc", "g_tm", "cond_gap", "tm_gap", "sa_sum"]
    print(f"双预测一致候选 {len(res)}：")
    print(res[cols].head(25).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
