#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_mendeley.py — Mendeley tpp25ztzmb 数据集标准化并入 paper_dataset 格式。

Mendeley 提供 4 性质（clean 版）：
  vis_clean   : 粘度 LOG10(mPa.s)  @ T(K), P(bar)      -> ln 尺度 = LOG10*ln10
  mp_clean    : 熔点 MP(K)  + LOG10(MP)
  xco2_clean  : CO2 溶解度 Xco2 @ T(K), P(bar)         -> 新性质（第 6 篇亮点）
  tox_clean   : 毒性 logEC50 (IPC-81)                    -> 新性质（绿色设计约束）
统一输出 schema（与 paper_dataset 一致）：
  cat_smiles, an_smiles, T, P, value, unit, ref, il, source
  value 尺度：conductivity/viscosity = ln，density = g/cm3，melting_point = K
"""
import math
import pathlib
import sqlite3
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW = pathlib.Path(__file__).resolve().parent / "data" / "raw"
OUT = pathlib.Path(__file__).resolve().parent / "data"
DB = ROOT / "data" / "il_props.db"
XLSX = RAW / "mendeley_il_props.xlsx"

LN10 = math.log(10.0)


def canon(smiles: str) -> str:
    """RDKit canonical SMILES；失败返回 None。"""
    if not smiles or not isinstance(smiles, str):
        return None
    m = Chem.MolFromSmiles(smiles.strip())
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """规范化 cation/anion/IL 三列 SMILES，返回带 il 键的 df。"""
    df = df.copy()
    df["cat_smiles"] = df["Cation_SMILES"].map(canon)
    df["an_smiles"] = df["Anion_SMILES"].map(canon)
    df = df.dropna(subset=["cat_smiles", "an_smiles"])
    df["il"] = df["cat_smiles"] + "|" + df["an_smiles"]
    return df


def main():
    xl = pd.ExcelFile(XLSX)

    # 所有 sheet 第 0 行是描述标题，第 1 行才是表头
    def parse_sheet(name):
        return xl.parse(name, header=1)

    # ---- 粘度 (LOG10 -> ln) ----
    vis = prep_df(parse_sheet("vis_clean"))
    vis = vis.rename(columns={"Temperature(K)": "T", "Pressure(bar)": "P",
                              "LOG10(VISCOSITY)(mPa.s)": "log10v"})
    vis["value"] = vis["log10v"] * LN10
    vis["unit"] = "mPa.s"
    vis["prop"] = "viscosity"
    vis["ref"] = "Mendeley tpp25ztzmb (Dong & Gao 2025)"
    vis = vis[["cat_smiles", "an_smiles", "T", "P", "value", "unit", "ref", "il", "prop"]]
    print(f"viscosity: {len(vis)} points, {vis['il'].nunique()} unique IL", flush=True)

    # ---- 熔点 (K) ----
    mp = prep_df(parse_sheet("mp_clean"))
    mp = mp.rename(columns={"MP(K)": "value"})
    mp["T"] = np.nan
    mp["P"] = np.nan
    mp["unit"] = "K"
    mp["prop"] = "melting_point"
    mp["ref"] = "Mendeley tpp25ztzmb (Dong & Gao 2025)"
    mp = mp[["cat_smiles", "an_smiles", "T", "P", "value", "unit", "ref", "il", "prop"]]
    print(f"melting_point: {len(mp)} points, {mp['il'].nunique()} unique IL", flush=True)

    # ---- CO2 溶解度 (新性质) ----
    xc = prep_df(parse_sheet("xco2_clean"))
    xc = xc.rename(columns={"Temperature(K)": "T", "Pressure(bar)": "P", "Xco2(mole fraction)": "value"})
    xc["unit"] = "mole_fraction"
    xc["prop"] = "co2_solubility"
    xc["ref"] = "Mendeley tpp25ztzmb (Dong & Gao 2025)"
    xc = xc[["cat_smiles", "an_smiles", "T", "P", "value", "unit", "ref", "il", "prop"]]
    print(f"co2_solubility: {len(xc)} points, {xc['il'].nunique()} unique IL", flush=True)

    # ---- 毒性 (新性质) ----
    tx = prep_df(parse_sheet("tox_clean"))
    tx = tx.rename(columns={"logEC50": "value"})
    tx["T"] = np.nan
    tx["P"] = np.nan
    tx["unit"] = "logEC50"
    tx["prop"] = "toxicity"
    tx["ref"] = "Mendeley tpp25ztzmb (Dong & Gao 2025)"
    tx = tx[["cat_smiles", "an_smiles", "T", "P", "value", "unit", "ref", "il", "prop"]]
    print(f"toxicity: {len(tx)} points, {tx['il'].nunique()} unique IL", flush=True)

    all_df = pd.concat([vis, mp, xc, tx], ignore_index=True)
    all_df.to_csv(OUT / "mendeley_normalized.csv", index=False)
    print(f"total: {len(all_df)} points, {all_df['il'].nunique()} unique IL", flush=True)

    # ---- 并入 il_props.db 新表（不覆盖旧表，保留溯源）----
    con = sqlite3.connect(DB)
    all_df.to_sql("mendeley_records", con, if_exists="replace", index=False)
    # 同步现有 known IL 交叉统计
    cur = con.cursor()
    cur.execute("SELECT COUNT(DISTINCT il) FROM mendeley_records")
    n_il = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT cat_smiles) FROM mendeley_records")
    n_cat = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT an_smiles) FROM mendeley_records")
    n_an = cur.fetchone()[0]
    con.commit()
    con.close()
    print(f"db 入库完成: {n_il} IL / {n_cat} 阳 / {n_an} 阴", flush=True)


if __name__ == "__main__":
    main()
