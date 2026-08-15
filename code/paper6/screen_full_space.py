# -*- coding: utf-8 -*-
"""screen_full_space.py — 第 6 篇：扩充库全组合空间逆向筛选（向量化版）。

空间：实验库唯一阳 × 唯一阴（2,838 × 592 ≈ 168 万组合）。
方法：GBM/HistGBM 双预测器（扩充数据训练，IL-disjoint 验证），298.15K 打分。
筛选：新颖（不在 6,177 已知 IL）+ 高电导（ln κ>0.2）+ 室温液态 + 低粘 + 双预测一致。
关键优化：唯一离子描述符矩阵一次性构建 → 组合矩阵 numpy 索引拼接（无逐行循环）。
产出：data/generated/full_space_all_scored.csv + full_space_candidates.csv
"""
import pathlib

import joblib
import numpy as np
import pandas as pd

P6 = pathlib.Path(__file__).resolve().parent
OUT = P6 / "data" / "generated"
ORACLE = P6 / "data" / "expanded" / "oracle_v2"
DESC = P6 / "data" / "il_descriptors_v2.csv"
KNOWN = ORACLE / "known_il_v2.csv"

TARGET_T = 298.15
CHUNK = 100_000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = set(pd.read_csv(KNOWN)["il"])
    desc = pd.read_csv(DESC)

    # 唯一离子描述符矩阵（只取数值列）
    cat_cols = [c for c in desc.columns if c.startswith("cat_") and c not in ("cat_smiles", "cat_ok")]
    an_cols = [c for c in desc.columns if c.startswith("an_") and c not in ("an_smiles", "an_ok")]
    cat_uniq = desc.drop_duplicates("cat_smiles").set_index("cat_smiles", drop=False)
    an_uniq = desc.drop_duplicates("an_smiles").set_index("an_smiles", drop=False)
    cats = cat_uniq.index.tolist()
    ans = an_uniq.index.tolist()
    cat_mat = np.nan_to_num(cat_uniq[cat_cols].to_numpy(dtype=float), nan=0.0)
    an_mat = np.nan_to_num(an_uniq[an_cols].to_numpy(dtype=float), nan=0.0)
    print(f"阳离子: {len(cats)} | 阴离子: {len(ans)} | 组合: {len(cats)*len(ans):,}", flush=True)

    # 加载模型
    models = {}
    for suffix in ["gbm", "hist"]:
        for prop in ["conductivity", "melting_point", "viscosity"]:
            pkg = joblib.load(ORACLE / f"{suffix}_{prop}.joblib")
            models[f"{suffix}_{prop}"] = pkg

    # 各模型特征列映射（cat 列 / an 列 / T）
    def feat_indices(feats):
        ci = np.array([cat_cols.index(c) if c in cat_cols else -1 for c in feats])
        ai = np.array([an_cols.index(c) if c in an_cols else -1 for c in feats])
        ti = np.array([1 if c == "T" else 0 for c in feats])
        return ci, ai, ti

    plan = {k: feat_indices(pkg["feats"]) for k, pkg in models.items()}

    def build(c_i, a_i, key):
        ci, ai, ti = plan[key]
        n = len(c_i)
        Xm = np.zeros((n, len(ci)), dtype=np.float64)
        for j in range(len(ci)):
            if ci[j] >= 0:
                Xm[:, j] = cat_mat[c_i, ci[j]]
            elif ai[j] >= 0:
                Xm[:, j] = an_mat[a_i, ai[j]]
            elif ti[j]:
                Xm[:, j] = TARGET_T
        return Xm

    # 枚举组合（分块，cat 主序）
    total = len(cats) * len(ans)
    out_path = OUT / "full_space_all_scored.csv"
    cand_path = OUT / "full_space_candidates.csv"
    header_written = False
    all_sel = []

    for start in range(0, total, CHUNK):
        end = min(start + CHUNK, total)
        c_i = np.arange(start, end) // len(ans)
        a_i = np.arange(start, end) % len(ans)
        n = end - start

        chunk_df = pd.DataFrame({
            "il": [f"{cats[i]}|{ans[j]}" for i, j in zip(c_i, a_i)],
        })
        chunk_df["gbm_conductivity"] = models["gbm_conductivity"]["model"].predict(build(c_i, a_i, "gbm_conductivity"))
        chunk_df["hist_conductivity"] = models["hist_conductivity"]["model"].predict(build(c_i, a_i, "hist_conductivity"))
        chunk_df["gbm_melting_point"] = models["gbm_melting_point"]["model"].predict(build(c_i, a_i, "gbm_melting_point"))
        chunk_df["gbm_viscosity"] = models["gbm_viscosity"]["model"].predict(build(c_i, a_i, "gbm_viscosity"))

        # 排除已知
        chunk_df = chunk_df[~chunk_df["il"].isin(known)]
        chunk_df.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True

        # 筛选
        agree = (chunk_df["gbm_conductivity"] > 0.2) & (chunk_df["hist_conductivity"] > 0.2)
        liquid = (chunk_df["gbm_melting_point"] < 300) & (chunk_df["gbm_melting_point"] > 180)
        lowvis = chunk_df["gbm_viscosity"] < 5.5
        sel = chunk_df[agree & liquid & lowvis]
        all_sel.append(sel)
        print(f"  已打分 {end:,} / {total:,} | 本批候选 {len(sel)}", flush=True)

    if all_sel:
        cand = pd.concat(all_sel, ignore_index=True).sort_values("gbm_conductivity", ascending=False)
        cand.to_csv(cand_path, index=False)
        print(f"\n通过全部约束候选: {len(cand):,}", flush=True)
        if len(cand):
            print(cand.head(30).to_string(), flush=True)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
