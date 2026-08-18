#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_numbers.py — 稿件数字 vs 结果 CSV 口径审计（paper2_upgrade）

逐项核对 manuscript_ces.md 中的关键数字与 results/*.csv 是否一致。
用法: python audit_numbers.py
"""
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RES = BASE / "results"
SRC = BASE / "manuscript_ces.md"

text = SRC.read_text(encoding="utf-8")
fails = []


def check(label, actual, expected, tol=0.005):
    ok = abs(float(actual) - float(expected)) <= tol
    if not ok:
        fails.append(f"{label}: manuscript={actual} vs csv={expected}")
    print(f"[{'OK' if ok else 'FAIL'}] {label}: manuscript={actual} | csv={expected}")


print("== 1. 标度律 (scaling_law_fits.csv) ==")
fits = pd.read_csv(RES / "scaling_law_fits.csv")
print(fits[["property", "a", "gamma", "N80", "N90"]].to_string(index=False))
for _, r in fits.iterrows():
    p = r["property"]
    if pd.notna(r.get("gamma")):
        # 稿件 Table 2 的值
        table = {"viscosity": 0.215, "conductivity": 0.807, "density": 0.565, "melting_point": 0.288}
        if p in table:
            check(f"{p} gamma", table[p], r["gamma"], 0.01)

print("\n== 2. 泄漏税 (leakage_tax_results.csv) ==")
leak = pd.read_csv(RES / "leakage_tax_results.csv")
allrows = leak[leak["bin"] == "ALL"]
for _, r in allrows.iterrows():
    print(f"  {r['property']}: R2g={r['r2_group']:.3f} R2p={r['r2_point']:.3f} dR2={r['delta_r2']:+.3f}")
print("  稿件称全量 ΔR² = +0.19(粘)/+0.22(电)/+0.09(密)")
for p, d in [("viscosity", 0.187), ("conductivity", 0.223), ("density", 0.085)]:
    v = allrows[allrows["property"] == p]["delta_r2"].iloc[0]
    check(f"{p} full dR2", round(d, 3), v, 0.01)

print("\n== 3. leaderboard (leaderboard.csv) ==")
lb = pd.read_csv(RES / "leaderboard.csv")
for p in ["viscosity", "conductivity", "density", "melting_point"]:
    s = lb[(lb["property"] == p) & (lb["split"] == "group")]
    row = {m: s[s["model"] == m]["R2"].iloc[0] for m in ["LR", "RF", "GBR", "HistGBM"]}
    best = max(row, key=row.get)
    print(f"  {p}: LR={row['LR']:.3f} RF={row['RF']:.3f} GBR={row['GBR']:.3f} HistGBM={row['HistGBM']:.3f} best={best}")

print("\n== 4. 获取策略 v2 (acquisition_v2_results.csv) ==")
a2 = pd.read_csv(RES / "acquisition_v2_results.csv")
for p in ["viscosity", "conductivity"]:
    s = a2[a2["property"] == p]
    r0 = s[s["strategy"] == "random"].sort_values("n_acquired").iloc[0]["r2_eval_fixed"]
    print(f"  {p}: random@0 R2={r0:.3f}")
    cov = s[(s["strategy"] == "coverage") & (s["n_acquired"] == 150)]
    rnd = s[(s["strategy"] == "random") & (s["n_acquired"] == 150)]
    if len(cov) and len(rnd):
        print(f"  {p} @150: coverage={cov['r2_eval_fixed'].iloc[0]:.3f} random={rnd['r2_eval_fixed'].iloc[0]:.3f}")

print("\n== 5. 冷启动 (cold_start_results.csv) ==")
cs = pd.read_csv(RES / "cold_start_results.csv")
for p in ["viscosity", "conductivity", "density", "melting_point"]:
    s = cs[cs["property"] == p]
    for _, r in s.iterrows():
        print(f"  {p} {r['class'][:28]:28s} n={int(r['n_IL']):4d} R2={r['R2']:+.3f}")

print("\n== 6. 覆盖 (coverage_stats.csv) ==")
cov = pd.read_csv(RES / "coverage_stats.csv")
print(cov.to_string(index=False))
print("  稿件称：median 1.56 / p90 2.61 / 4.29% beyond 3")
check("median", 1.56, cov["dist_median"].iloc[0], 0.02)
check("p90", 2.61, cov["dist_p90"].iloc[0], 0.05)
check("frac>3", 4.29, cov["frac_dist_gt_3"].iloc[0] * 100, 0.1)

print("\n" + ("=" * 20))
print("FAILURES:", len(fails))
for f in fails:
    print("  -", f)
