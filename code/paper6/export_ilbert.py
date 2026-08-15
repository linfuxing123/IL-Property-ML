#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_ilbert.py — 解析 ILBERT IL.csv（8.3M × 29 列）为 il|性质 格式。

Normalized_SMILES 为 "anion.cation" 点分隔 → 统一为 cat|an（与 ilpe_props 一致）。
性质：熔点/玻璃化/热分解/logEC50/电导(ln)/折射率/热导/密度/粘度(ln)/表面张力/
      CO2(ln)/热容，含 STD。
产出：data/ilbert_props.csv（分块流式）
"""
import os

import numpy as np
import pandas as pd

RAW = r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6\data\raw"
OUT = r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6\data"
CSV = os.path.join(RAW, "ilbert_IL.csv")
OUTCSV = os.path.join(OUT, "ilbert_props.csv")

CHUNK = 500_000
RENAME = {
    "Melting_point(K)": "ML_MELTINGPOINT",
    "Glass_transition_temperature(K)": "ML_GLASSTRANSITION",
    "Thermal_decomposition_temperature(K)": "ML_THERMALDECOMPOSITION",
    "logEC50": "ML_CYTOTOXICITY",
    "ln_Electrical_conductivity(S/m)": "ML_LN_CONDUCTIVITY",
    "Refractive index": "ML_REFRACTIVEINDEX",
    "Thermal_conductivity(W/m/K)": "ML_THERMALCONDUCTIVITY",
    "Density(kg/m3)": "ML_DENSITY",
    "ln_Viscosity(mPas)": "ML_LN_VISCOSITY",
    "Surface tension(mN/m)": "ML_SURFACETENSION",
    "ln(xCO2)_298K_1bar": "ML_LN_XCO2_298",
    "Heat capacity(J/mol/K)": "ML_HEATCAPACITY",
    "ln(xCO2)_328K_1bar": "ML_LN_XCO2_328",
}


def split_il(smiles):
    """'anion.cation' → 'cat|an'。"""
    parts = smiles.split(".")
    if len(parts) != 2:
        return None
    an, cat = parts
    return f"{cat}|{an}"


def main():
    first = True
    n = 0
    with open(OUTCSV, "w", newline="", encoding="utf-8") as fo:
        for chunk in pd.read_csv(CSV, chunksize=CHUNK, usecols=lambda c: c in RENAME or c == "Normalized_SMILES"):
            chunk["il"] = chunk["Normalized_SMILES"].map(split_il)
            chunk = chunk.dropna(subset=["il"])
            chunk = chunk.drop(columns=["Normalized_SMILES"])
            chunk = chunk.rename(columns=RENAME)
            chunk.to_csv(fo, index=False, header=first)
            first = False
            n += len(chunk)
            print(f"  {n:,} rows", flush=True)
    print("saved", OUTCSV, flush=True)


if __name__ == "__main__":
    main()
