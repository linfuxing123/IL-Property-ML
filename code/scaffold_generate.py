#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scaffold_generate.py — 骨架变异生成化学合理的新阳离子（替代质量差的 VAE 生成）。

用已知 IL 阳离子的 6 类核（咪唑/季铵/吡啶/季磷/吡咯烷/硫）+ 取代基（不同链长、
醚链、羟乙基）系统构造新阳离子，RDKit 校验有效性 + 电荷 + 新颖 + 描述符。
输出：generated/cation_scaffold.csv（smiles + cat_* 描述符，格式同 il_descriptors）。
"""
import pathlib
import sys

import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parent
DESC = ROOT.parents[2] / "data" / "il_descriptors.csv"
sys.path.insert(0, str(ROOT.parents[2] / "data"))
from il_descriptors import ion_descriptors  # noqa: E402

# 取代基（SMILES 片段）
R = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCCC", "COCC", "CCOCC", "COCCOCC", "CCO"]

# 核模板（返回 SMILES 列表）
def imidazolium():
    return [f"{r1}n1cc[n+]({r2})c1" for r1 in R for r2 in R]

def pyridinium():
    return [f"{r1}[n+]1ccccc1" for r1 in R]

def pyrrolidinium():
    return [f"{r1}[N+]1({r2})CCCC1" for r1 in R for r2 in R]

def ammonium():
    return [f"{r1}[N+](C)(C)C" for r1 in R]

def phosphonium():
    return [f"{r1}[P+](C)(C)C" for r1 in R]

def sulfonium():
    return [f"{r1}[S+]({r2})C" for r1 in R for r2 in R]


def charge(s):
    m = Chem.MolFromSmiles(s)
    return sum(a.GetFormalCharge() for a in m.GetAtoms()) if m else 0


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s


def main():
    gen = imidazolium() + pyridinium() + pyrrolidinium() + ammonium() + phosphonium() + sulfonium()
    gen = list(dict.fromkeys(gen))
    valid = [canon(s) for s in gen if Chem.MolFromSmiles(s) is not None and charge(s) == 1]
    valid = list(dict.fromkeys(valid))

    df = pd.read_csv(DESC)
    known = set(df["cat_smiles"].dropna().map(canon).unique())
    novel = [s for s in valid if s not in known]
    print(f"生成 {len(gen)} -> 有效+电荷+1 {len(valid)} -> 新颖 {len(novel)}", flush=True)

    rows = []
    for s in novel:
        d = ion_descriptors(s)
        if not d:
            continue
        row = {"smiles": s}
        for k, v in d.items():
            row["cat_" + k] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    outdir = ROOT / "generated"
    outdir.mkdir(exist_ok=True)
    out.to_csv(outdir / "cation_scaffold.csv", index=False, encoding="utf-8-sig")
    print(f"描述符完成 {len(out)} -> {outdir / 'cation_scaffold.csv'}", flush=True)


if __name__ == "__main__":
    main()
