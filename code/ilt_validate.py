#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ilt_validate.py - 从 ilt_records 导出属性数据并跑诚实验证 (honest_cv + coverage)

用法:
  python ilt_validate.py --export [--log]   # 导出 (--log: 粘度/电导率用 ln 变换)
  python ilt_validate.py --validate [--log]
"""
import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "il_props.db"
OUT = ROOT / "workspace" / "matmodel" / "data" / "ilt"
SKILL = ROOT / ".codex" / "skills" / "math-agent" / "scripts"
PY = sys.executable

# 值域合理性过滤 (物理范围)
RANGE = {
    "viscosity": (0.1, 2e6),      # mPa.s
    "density": (0.5, 3.0),        # g/cm3
    "conductivity": (1e-6, 100),  # S/m
    "melting_point": (100, 800),  # K
}


def export(use_log=False):
    OUT.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    for prop in RANGE:
        df = pd.read_sql_query("""
            SELECT cat_smiles, an_smiles, T, P, value, unit, ref
            FROM ilt_records WHERE prop=?
        """, c, params=(prop,))
        if df.empty:
            print(f"[{prop}] 无数据")
            continue
        df = df.dropna(subset=["cat_smiles", "an_smiles"])
        df["il"] = df["cat_smiles"] + "|" + df["an_smiles"]
        lo, hi = RANGE[prop]
        df = df[(df["value"] >= lo) & (df["value"] <= hi)]
        if use_log and prop in ("viscosity", "conductivity"):
            df["value"] = df["value"].apply(lambda v: __import__("math").log(v))
        if prop != "melting_point":
            df = df.dropna(subset=["T"])
        # 温度去重: 同 IL 同温度多压力点保留 (压力是有效维度)
        df = df.drop_duplicates(subset=["il", "T", "value", "P"])
        df = add_features(df)
        df.to_csv(OUT / f"{prop}.csv", index=False, encoding="utf-8-sig")
        n_il = df["il"].nunique()
        n_grp_multi = df.groupby("il").size()
        print(f"[{prop}] 数据点 {len(df)} | 唯一IL {n_il} | "
              f"多温度IL {n_grp_multi[n_grp_multi > 1].count()} "
              f"({100 * n_grp_multi[n_grp_multi > 1].count() / max(n_il,1):.1f}%)")
    return c


def add_features(df):
    """用 RDKit 给 cat+an 组合分子算结构描述符, 用于组级外推特征"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        return df
    import contextlib
    import io
    feats = {"mw": [], "logp": [], "tpsa": [], "hbd": [], "hba": [],
             "rotb": [], "ar_rings": [], "heavy": [], "fcsp3": [], "rings": []}
    cache = {}
    for il in df["il"]:
        if il not in cache:
            smi = il.replace("|", ".")
            with contextlib.redirect_stderr(io.StringIO()):
                mol = Chem.MolFromSmiles(smi)
            if mol is None:
                cache[il] = [None] * len(feats)
            else:
                cache[il] = [
                    Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol), rdMolDescriptors.CalcNumHBD(mol),
                    rdMolDescriptors.CalcNumHBA(mol),
                    rdMolDescriptors.CalcNumRotatableBonds(mol),
                    rdMolDescriptors.CalcNumAromaticRings(mol),
                    mol.GetNumHeavyAtoms(), Descriptors.FractionCSP3(mol),
                    rdMolDescriptors.CalcNumRings(mol)]
        for i, k in enumerate(feats):
            feats[k].append(cache[il][i])
    for k, v in feats.items():
        df[k] = v
    return df


def validate():
    export(use_log=True)
    feat_cols = "mw logp tpsa hbd hba rotb ar_rings heavy fcsp3 rings T"
    for prop in RANGE:
        csv = OUT / f"{prop}.csv"
        if not csv.exists():
            continue
        print(f"\n===== {prop} =====")
        subprocess.run([PY, str(SKILL / "coverage_planner.py"), str(csv),
                        "--group", "il", "--value", "value", "--property", prop],
                       check=False)
        if prop != "melting_point":
            subprocess.run([PY, str(SKILL / "honest_cv.py"), str(csv),
                            "--group", "il", "--target", "value",
                            "--features", *feat_cols.split()],
                           check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--log", action="store_true", help="导出时对粘度/电导率做 ln 变换")
    a = ap.parse_args()
    if a.export:
        export(use_log=a.log)
    if a.validate:
        validate()


if __name__ == "__main__":
    main()
