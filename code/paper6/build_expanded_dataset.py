#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_expanded_dataset.py — 第 6 篇：构建扩充后的实验数据集 + 诚实验证。

输入（全部实验值，金标准）：
  1) paper5 现有 4 性质 paper_dataset（ILThermo + 旧库）
  2) Mendeley tpp25ztzmb 实验值（粘度/熔点/CO2/毒性，已标准化为 mendeley_normalized.csv）
输出：
  data/expanded/paper_dataset/{prop}.csv  —— 与 paper5 同 schema 的可比数据集
  data/expanded/stats.md                   —— 数据规模对比表
协议：完全复用 paper5（GBM/HistGBM/MPNN IL-disjoint GroupKFold(5)），
     仅扩充数据，模型与验证不变 → 干净对照"数据规模 → 精度"。
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
P5 = ROOT / "workspace" / "matmodel" / "data" / "ilt" / "paper_dataset"
OUT = pathlib.Path(__file__).resolve().parent / "data" / "expanded"
MEN = pathlib.Path(__file__).resolve().parent / "data" / "mendeley_normalized.csv"

PROPS = ["conductivity", "density", "viscosity", "melting_point"]
NEW_PROPS = {"co2_solubility", "toxicity"}  # 第 6 篇新增性质


def main():
    men = pd.read_csv(MEN)
    men_v = men[men["prop"] == "viscosity"].copy()
    men_m = men[men["prop"] == "melting_point"].copy()
    men_c = men[men["prop"] == "co2_solubility"].copy()
    men_t = men[men["prop"] == "toxicity"].copy()

    (OUT / "paper_dataset").mkdir(parents=True, exist_ok=True)
    stats = []
    for prop in PROPS:
        old = pd.read_csv(P5 / f"{prop}.csv")
        if prop == "viscosity":
            new = men_v.drop(columns=["prop"])
        elif prop == "melting_point":
            new = men_m.drop(columns=["prop"])
        else:
            new = pd.DataFrame()
        if len(new):
            # paper5 原样保留；只补 Mendeley 中 (il,T) 不重复的记录
            common = [c for c in new.columns if c in old.columns]
            new = new[common]
            key = ["il"] + (["T"] if prop != "melting_point" else ["il"])
            old_keys = set(map(tuple, old[key].drop_duplicates().values))
            new = new[~new[key].apply(tuple, axis=1).isin(old_keys)]
            merged = pd.concat([old, new], ignore_index=True)
        else:
            merged = old
        merged.to_csv(OUT / "paper_dataset" / f"{prop}.csv", index=False)
        stats.append((prop, len(old), len(merged), merged["il"].nunique()))

    # 新性质单独导出
    for prop, df in [("co2_solubility", men_c), ("toxicity", men_t)]:
        df = df.drop(columns=["prop"])
        df.to_csv(OUT / "paper_dataset" / f"{prop}.csv", index=False)
        stats.append((prop, 0, len(df), df["il"].nunique()))

    # stats 报告
    lines = ["# 扩充后数据集统计（paper5 vs paper6）\n"]
    lines.append("| 性质 | paper5 点数 | 扩充后点数 | 扩充后唯一 IL |")
    lines.append("|---|---|---|---|")
    for prop, n_old, n_new, n_il in stats:
        lines.append(f"| {prop} | {n_old} | {n_new} | {n_il} |")
    (OUT / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
