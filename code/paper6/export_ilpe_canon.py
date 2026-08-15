# -*- coding: utf-8 -*-
"""export_ilpe_canon.py — 重建 ILPE 为 canonical SMILES 的 il 键。

问题：ILPE Molecule 用原始 SMILES（非 canonical），与 ILBERT 对不上。
修法：仅对 219,334 个离子做 RDKit canonical，然后经 CATION_ID/ANION_ID
      join 重建 8.3M 行的 il（cat|an canonical）。
产出：data/ilpe_props_canon.csv
"""
import os
import sqlite3

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

RAW = r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6\data\raw"
OUT = r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6\data"
DB = os.path.join(RAW, "ILPE.db")
OUTCSV = os.path.join(OUT, "ilpe_props_canon.csv")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s


def main():
    con = sqlite3.connect(DB)
    mols = pd.read_sql("SELECT MOLID, SMILES, CATEGORY FROM Molecule", con)
    mols["canon"] = mols["SMILES"].map(canon)
    cat_map = dict(zip(mols.loc[mols.CATEGORY == "Cation", "MOLID"].astype(str),
                       mols.loc[mols.CATEGORY == "Cation", "canon"]))
    an_map = dict(zip(mols.loc[mols.CATEGORY == "Anion", "MOLID"].astype(str),
                      mols.loc[mols.CATEGORY == "Anion", "canon"]))
    print(f"ions canon: {len(cat_map)} cat / {len(an_map)} an", flush=True)

    ml_cols = [c for c in pd.read_sql("PRAGMA table_info(Properties)", con)["name"]
               if c.startswith("ML_")]
    cols = ["CATION_ID", "ANION_ID"] + ml_cols
    first = True
    n = 0
    with open(OUTCSV, "w", newline="", encoding="utf-8") as fo:
        for i in range(0, 8_333_096, 1_000_000):
            q = (f"SELECT {', '.join(cols)} FROM Properties "
                 f"WHERE rowid > {i} AND rowid <= {i + 1_000_000}")
            ch = pd.read_sql(q, con)
            ch["CATION_ID"] = ch["CATION_ID"].astype(str)
            ch["ANION_ID"] = ch["ANION_ID"].astype(str)
            ch["cat"] = ch["CATION_ID"].map(cat_map)
            ch["an"] = ch["ANION_ID"].map(an_map)
            ch["il"] = ch["cat"] + "|" + ch["an"]
            ch = ch[["il"] + ml_cols]
            ch.to_csv(fo, index=False, header=first)
            first = False
            n += len(ch)
            print(f"  {n:,} rows", flush=True)
    con.close()
    print("saved", OUTCSV, flush=True)


if __name__ == "__main__":
    main()
