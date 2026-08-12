#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ilt_merge_old.py - 把旧库独有 IL 的属性并入 ILThermo 数据集

支持:
  --melting    melting_point (旧库 °C -> K, +273.15)
  --thermo     viscosity/density (旧库 T °C -> K; 值 mPa.s / g/cm3)
用法:
  python ilt_merge_old.py --melting
  python ilt_merge_old.py --thermo
"""
import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "il_props.db"
OUT = ROOT / "workspace" / "matmodel" / "data" / "ilt"
SKILL = ROOT / ".codex" / "skills" / "math-agent" / "scripts"
PY = sys.executable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ilt_validate import add_features, export as export_base


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def merge_thermo(prop, unit):
    """并入旧库独有 IL 的 viscosity/density (T °C -> K)"""
    c = sqlite3.connect(DB)
    have = set()
    for cat, an in c.execute("select distinct cat_smiles, an_smiles from ilt_records where prop=?", (prop,)):
        have.add((canon(cat), canon(an)))
    rows = []
    skipped = 0
    for cat, an, T, v in c.execute(
            "select cat_smiles, an_smiles, T, value from records where property=? and value>0", (prop,)):
        k = (canon(cat), canon(an))
        if k in have or T is None:
            continue
        # 旧库 T 为 °C: 合理范围 -50..350 °C
        if not (-50 <= T <= 350):
            skipped += 1
            continue
        rows.append({"cat_smiles": cat, "an_smiles": an,
                     "T": T + 273.15, "P": None, "value": v,
                     "unit": unit, "ref": "legacy_db", "il": cat + "|" + an})
    df = pd.DataFrame(rows)
    print(f"[{prop}] 旧库独有 IL 数据点: {len(df)} (T 越界跳过 {skipped})")
    if df.empty:
        return
    df = add_features(df)
    path = OUT / f"{prop}_merged.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"写入 {path}")
    subprocess.run([PY, str(SKILL / "coverage_planner.py"), str(path),
                    "--group", "il", "--value", "value", "--property", prop],
                   check=False)


def build_all():
    """合并 ILThermo 基础 CSV + 旧库补充 -> paper_dataset/, 跑完整验证"""
    export_base(use_log=False)  # 先刷新原始值基础 CSV
    out_dir = OUT / "paper_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    merge_map = {"viscosity": "viscosity_merged.csv",
                 "density": "density_merged.csv",
                 "melting_point": "melting_point_merged.csv",
                 "conductivity": None}
    for prop, merged_name in merge_map.items():
        base_path = OUT / f"{prop}.csv"
        merged_path = OUT / merged_name if merged_name else None
        if not base_path.exists():
            print(f"[{prop}] 无基础 CSV")
            continue
        base = pd.read_csv(base_path, encoding="utf-8-sig")
        if merged_path and merged_path.exists():
            extra = pd.read_csv(merged_path, encoding="utf-8-sig")
            keep = [x for x in base.columns if x in extra.columns]
            df = pd.concat([base, extra[keep]], ignore_index=True)
        else:
            df = base
        # 去重: 同 IL 同 T 同值
        df = df.drop_duplicates(subset=["il", "T", "value"])
        # 每属性转 ln 目标 (粘度/电导率)
        if prop in ("viscosity", "conductivity"):
            df["value"] = df["value"].apply(lambda v: __import__("math").log(v))
        out_path = out_dir / f"{prop}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        n_il = df["il"].nunique()
        print(f"[{prop}] paper_dataset: {len(df)} 点 / {n_il} IL -> {out_path}")
        # 验证
        subprocess.run([PY, str(SKILL / "coverage_planner.py"), str(out_path),
                        "--group", "il", "--value", "value", "--property", prop],
                       check=False)
        if prop != "melting_point":
            subprocess.run([PY, str(SKILL / "honest_cv.py"), str(out_path),
                            "--group", "il", "--target", "value",
                            "--features", *"mw logp tpsa hbd hba rotb ar_rings heavy fcsp3 rings T".split()],
                           check=False)


def merge_melting():
    c = sqlite3.connect(DB)
    # ILThermo 已有 IL
    have = set()
    for cat, an in c.execute("select distinct cat_smiles, an_smiles from ilt_records where prop='melting_point'"):
        have.add((canon(cat), canon(an)))
    # 旧库 melting 独有 IL
    rows = []
    for cat, an, v in c.execute("select cat_smiles, an_smiles, value from records where property='melting_point'"):
        k = (canon(cat), canon(an))
        if k in have:
            continue
        rows.append({"cat_smiles": cat, "an_smiles": an,
                     "T": None, "P": None, "value": v + 273.15,
                     "unit": "K", "ref": "legacy_db", "il": cat + "|" + an})
    df = pd.DataFrame(rows)
    print(f"旧库独有熔点 IL: {len(df)}")
    if df.empty:
        return
    df = add_features(df)
    path = OUT / "melting_point_merged.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"写入 {path}")
    # 合并 ILThermo + 旧库独有 -> 全量熔点数据集
    ilt_csv = OUT / "melting_point.csv"
    if ilt_csv.exists():
        base = pd.read_csv(ilt_csv, encoding="utf-8-sig")
        keep = [x for x in base.columns if x in df.columns]
        all_df = pd.concat([base[keep], df[keep]], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=["il"])
        all_path = OUT / "melting_point_all.csv"
        all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        print(f"全量熔点: ILThermo {len(base)} + 旧库独有 {len(df)} = {len(all_df)} IL")
        subprocess.run([PY, str(SKILL / "coverage_planner.py"), str(all_path),
                        "--group", "il", "--value", "value", "--property", "melting_point"],
                       check=False)
        subprocess.run([PY, str(SKILL / "honest_cv.py"), str(all_path),
                        "--group", "il", "--target", "value",
                        "--features", *"mw logp tpsa hbd hba rotb ar_rings heavy fcsp3 rings".split()],
                       check=False)
    # 验证
    subprocess.run([PY, str(SKILL / "coverage_planner.py"), str(path),
                    "--group", "il", "--value", "value", "--property", "melting_point"],
                   check=False)
    subprocess.run([PY, str(SKILL / "honest_cv.py"), str(path),
                    "--group", "il", "--target", "value",
                    "--features", *"mw logp tpsa hbd hba rotb ar_rings heavy fcsp3 rings".split()],
                   check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--melting", action="store_true")
    ap.add_argument("--thermo", action="store_true")
    ap.add_argument("--build-all", action="store_true")
    a = ap.parse_args()
    if a.melting:
        merge_melting()
    if a.thermo:
        merge_thermo("viscosity", "mPa.s")
        merge_thermo("density", "g/cm3")
    if a.build_all:
        build_all()


if __name__ == "__main__":
    main()
