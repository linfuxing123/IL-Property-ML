#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""triple_source.py — 三源电导率交叉验证（ILPE vs ILBERT vs 我们）。

样本：8.3M 库全量（ILPE/ILBERT 同空间）；我们 GBM/HistGBM 在 top 子集上。
问题：
  1) ILPE 与 ILBERT 电导率预测的一致性（同库两个独立模型）
  2) 高电导 top 候选（两源一致）在我们预测器上是否也高
  3) 三源一致候选 = 可信的"预测超越已知最优"对象
产出：data/generated/triple_source_report.csv
"""
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

P6 = pathlib.Path(__file__).resolve().parent
OUT = P6 / "data" / "generated"
ILPE = P6 / "data" / "ilpe_props.csv"
ILBERT = P6 / "data" / "ilbert_props.csv"
ORACLE = P6 / "data" / "expanded" / "oracle_v2"
KNOWN = ORACLE / "known_il_v2.csv"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "data"))
from il_descriptors import ion_descriptors  # noqa: E402

TARGET_T = 298.15


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = set(pd.read_csv(KNOWN)["il"])
    print(f"已知实验 IL: {len(known)}", flush=True)

    # 两源合并（ILPE: ML_ELECTRICALCONDUCTIVITY 单位 S/m → 过滤>0 后 ln；
    #             ILBERT: ML_LN_CONDUCTIVITY 已是 ln 尺度）
    ilpe = pd.read_csv(P6 / "data" / "ilpe_props_canon.csv",
                       usecols=["il", "ML_ELECTRICALCONDUCTIVITY", "ML_MELTINGPOINT", "ML_VISCOSITY"])
    ilbert = pd.read_csv(ILBERT, usecols=["il", "ML_LN_CONDUCTIVITY", "ML_MELTINGPOINT", "ML_LN_VISCOSITY"])
    ilpe = ilpe[ilpe["ML_ELECTRICALCONDUCTIVITY"] > 0]
    ilpe["ln_kappa_ilpe"] = np.log(ilpe["ML_ELECTRICALCONDUCTIVITY"])
    m = ilpe.merge(ilbert, on="il", how="inner", suffixes=("_ilpe", "_ilbert"))
    print(f"两源交集（ILPE 有效电导）: {len(m):,}", flush=True)
    m = m[~m["il"].isin(known)].reset_index(drop=True)
    print(f"排除已知后: {len(m):,}", flush=True)

    # 两源相关性与一致性
    corr = m["ln_kappa_ilpe"].corr(m["ML_LN_CONDUCTIVITY"])
    print(f"ILPE vs ILBERT lnκ 相关系数: {corr:.4f}", flush=True)
    m["agree_pos"] = (m["ln_kappa_ilpe"] > 0) & (m["ML_LN_CONDUCTIVITY"] > 0)
    print(f"两源一致电导>0: {m['agree_pos'].sum():,} / {len(m):,}", flush=True)

    # 两源都认为高电导的 top
    m["mean_kappa"] = (m["ln_kappa_ilpe"] + m["ML_LN_CONDUCTIVITY"]) / 2
    top = m[m["agree_pos"]].sort_values("mean_kappa", ascending=False).head(3000).reset_index(drop=True)
    print(f"两源一致高电导 top-3000: lnκ 范围 "
          f"{top['mean_kappa'].min() if len(top) else 'NA'} ~ "
          f"{top['mean_kappa'].max() if len(top) else 'NA'}", flush=True)
    if len(top) == 0:
        print("两源一致>0 候选为 0 —— 检查两源电导率分布：", flush=True)
        print(f"  ILPE lnκ: p50={m['ln_kappa_ilpe'].median():.3f} p90={m['ln_kappa_ilpe'].quantile(.9):.3f} "
              f"p99={m['ln_kappa_ilpe'].quantile(.99):.3f} max={m['ln_kappa_ilpe'].max():.3f}", flush=True)
        print(f"  ILBERT lnκ: p50={m['ML_LN_CONDUCTIVITY'].median():.3f} "
              f"p90={m['ML_LN_CONDUCTIVITY'].quantile(.9):.3f} "
              f"p99={m['ML_LN_CONDUCTIVITY'].quantile(.99):.3f} max={m['ML_LN_CONDUCTIVITY'].max():.3f}", flush=True)
        m.to_csv(OUT / "triple_source_full.csv", index=False)
        return

    # 我们 GBM/HistGBM 在 top-3000 上打分
    rows = []
    for _, r in top.iterrows():
        cat, an = r["il"].split("|")
        row = {}
        cd, ad = ion_descriptors(cat), ion_descriptors(an)
        if not cd or not ad:
            continue
        for k, v in cd.items():
            row["cat_" + k] = v
        for k, v in ad.items():
            row["an_" + k] = v
        row["il"] = r["il"]
        rows.append(row)
    fdf = pd.DataFrame(rows)
    top = top[top["il"].isin(fdf["il"])].reset_index(drop=True)
    fdf = fdf.reset_index(drop=True)
    print(f"描述符有效: {len(top)}", flush=True)

    for prop, suffix in [("conductivity", "gbm"), ("conductivity", "hist")]:
        pkg = joblib.load(ORACLE / f"{suffix}_{prop}.joblib")
        feats = pkg["feats"]
        for c in feats:
            if c not in fdf.columns:
                fdf[c] = np.nan
        X = fdf[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        top[f"{suffix}_conductivity"] = pkg["model"].predict(X)
    print("自训练预测器打分完成", flush=True)

    # 三源一致
    top["triple"] = (top["gbm_conductivity"] > 0) & (top["hist_conductivity"] > 0)
    triple = top[top["triple"]].sort_values("gbm_conductivity", ascending=False)
    print(f"\n三源一致电导>0: {len(triple)}", flush=True)
    cols = ["il", "ln_kappa_ilpe", "ML_LN_CONDUCTIVITY", "mean_kappa",
            "gbm_conductivity", "hist_conductivity", "ML_MELTINGPOINT_ilpe"]
    triple.to_csv(OUT / "triple_source_report.csv", index=False)
    if len(triple):
        print(triple.head(20)[cols].to_string(), flush=True)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
