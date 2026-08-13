#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""il_descriptors.py - 批量生成 1,891 种 IL 的 RDKit 描述符（备建模）

来源：第二篇论文最终数据集 workspace/matmodel/data/ilt/paper_dataset/*.csv
      （1,891 个唯一 cation|anion SMILES 对）
特征：阳离子、阴离子分别计算 RDKit 全套描述符（descList ~208 个）
      + 论文 10 个核心（ExactMolWt/Crippen LogP/HBD/HBA/TPSA/Rotatable/
      FractionCsp3/Ring/AromaticRings/MR）+ ECFP4 可选（默认关）
输出：data/il_descriptors.csv + il_props.db 新表 il_descriptors
并发：ProcessPoolExecutor 按 CPU 核并行（默认 min(16, cpu)）

用法: python il_descriptors.py [--out data/il_descriptors.csv] [--workers N]
"""
import argparse
import csv
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "il_props.db"
DATA_CSV = ROOT / "workspace" / "matmodel" / "data" / "ilt" / "paper_dataset"


def load_unique_ils():
    """从 paper_dataset 四个 CSV 汇总唯一 cation|anion 对"""
    import pandas as pd

    union = {}
    for f in sorted(DATA_CSV.glob("*.csv")):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            cat = str(row.get("cat_smiles", "")).strip()
            an = str(row.get("an_smiles", "")).strip()
            if cat and an and cat != "nan" and an != "nan":
                union[f"{cat}|{an}"] = (cat, an)
    return union


def ion_descriptors(smiles):
    """计算单个离子的 RDKit 全套描述符 + 论文 10 个核心"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = {}
    if not smiles:
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    # RDKit 全套
    for name, fn in Descriptors.descList:
        try:
            v = fn(mol)
            out[name] = None if v is None else float(v)
        except Exception:
            out[name] = None
    # 论文 10 个核心（显式，避免命名差异）
    core = {
        "ExactMolWt": rdMolDescriptors.CalcExactMolWt(mol),
        "CrippenLogP": rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
        "NumHBD": rdMolDescriptors.CalcNumHBD(mol),
        "NumHBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "MR": Descriptors.MolMR(mol),
    }
    for k, v in core.items():
        out["core_" + k] = None if v is None else float(v)
    return out


def compute_one(item):
    """worker: 计算单个 IL 的描述符（阳离子 + 阴离子）"""
    key, (cat, an) = item
    row = {"il": key, "cat_smiles": cat, "an_smiles": an}
    cd, ad = ion_descriptors(cat), ion_descriptors(an)
    row["cat_ok"] = 1 if cd else 0
    row["an_ok"] = 1 if ad else 0
    for k, v in cd.items():
        row["cat_" + k] = v
    for k, v in ad.items():
        row["an_" + k] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "il_descriptors.csv"))
    ap.add_argument("--workers", type=int, default=min(16, __import__("os").cpu_count() or 4))
    args = ap.parse_args()

    ils = load_unique_ils()
    print(f"唯一 IL 数: {len(ils)}")
    items = list(ils.items())

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(compute_one, it) for it in items]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 300 == 0:
                print(f"  已完成 {i}/{len(items)}")

    # 统一列（以第一条非空为准，保证 CSV 表头完整）
    all_cols = []
    for r in results:
        for k in r:
            if k not in all_cols:
                all_cols.append(k)
    ok = sum(1 for r in results if r["cat_ok"] and r["an_ok"])
    print(f"完成: {len(results)} | 阳+阴都成功: {ok} | 失败: {len(results) - ok}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("CSV:", args.out, Path(args.out).stat().st_size, "bytes")

    # 入 SQLite
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS il_descriptors")
    con.execute(
        f"""CREATE TABLE il_descriptors (
            il TEXT PRIMARY KEY, cat_smiles TEXT, an_smiles TEXT,
            cat_ok INTEGER, an_ok INTEGER,
            {", ".join(f'"{c}" REAL' for c in all_cols if c not in
                       ("il", "cat_smiles", "an_smiles", "cat_ok", "an_ok"))}
        )"""
    )
    for r in results:
        cols = [c for c in all_cols if c not in
                ("il", "cat_smiles", "an_smiles", "cat_ok", "an_ok")]
        con.execute(
            "INSERT INTO il_descriptors (il,cat_smiles,an_smiles,cat_ok,an_ok,"
            + ",".join(f'"{c}"' for c in cols) + ") VALUES (?,?,?,?,?,"
            + ",".join("?" for _ in cols) + ")",
            [r["il"], r["cat_smiles"], r["an_smiles"], r["cat_ok"], r["an_ok"]]
            + [r.get(c) for c in cols],
        )
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM il_descriptors").fetchone()[0]
    con.close()
    print(f"SQLite il_descriptors 表: {n} 行")


if __name__ == "__main__":
    main()
