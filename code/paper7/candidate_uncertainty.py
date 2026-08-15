# -*- coding: utf-8 -*-
"""candidate_uncertainty.py — 第 7 篇应用：候选不确定性标注 + 训练耦合检验。

1) 对 paper6 的 7 个最终候选，标注分歧/不确定性（GBM-Hist 分歧 + 跨源参考）
2) 检验"训练密度耦合"：6,108 交集 IL 中，与训练集共享阳/阴离子的 IL
   是否分歧更小（支持"高电导区=训练密集区→低分歧"假说）
"""
import numpy as np
import pandas as pd
import pathlib

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
GEN = P6 / "data" / "generated"
DS = P6 / "data" / "expanded" / "paper_dataset"

# 1) 候选不确定性标注
cand = pd.read_csv(GEN / "final_candidates_v2_unique.csv")
cand["divergence"] = (cand["gbm_conductivity"] - cand["hist_conductivity"]).abs()
cand["mean_kappa"] = (cand["gbm_conductivity"] + cand["hist_conductivity"]) / 2
cand["span_3model"] = cand[["gbm_conductivity", "hist_conductivity", "mpnn_conductivity"]].max(axis=1) - \
                      cand[["gbm_conductivity", "hist_conductivity", "mpnn_conductivity"]].min(axis=1)
print("=== paper6 候选 + 不确定性标注 ===")
print(cand[["il_canon", "mean_kappa", "divergence", "span_3model"]].to_string(index=False))
cand.to_csv(pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper7") / "candidates_uncertainty.csv", index=False)

# 2) 训练密度耦合检验（共享离子 → 邻近训练 → 低分歧？）
df = pd.read_csv(GEN / "consensus_layers.csv")
df = df[df["ln_kappa_ilpe"].notna()].copy()
df["cat"] = df["il"].str.split("|").str[0]
df["an"] = df["il"].str.split("|").str[1]

tr = pd.read_csv(DS / "conductivity.csv")
tr["cat"] = tr["cat_smiles"].astype(str)
tr["an"] = tr["an_smiles"].astype(str)
tr_cats = set(tr["cat"].unique())
tr_ans = set(tr["an"].unique())

df["cat_in_train"] = df["cat"].isin(tr_cats)
df["an_in_train"] = df["an"].isin(tr_ans)
df["shares_ion"] = df["cat_in_train"] | df["an_in_train"]

print("\n=== 训练密度耦合检验（6,108 交集 IL）===")
share = df[df["shares_ion"]]
noshare = df[~df["shares_ion"]]
print(f"共享训练离子: {len(share):,} IL | 平均分歧 = {share['divergence'].mean():.3f} | 平均预测 = {share['mean_kappa'].mean():.3f}")
print(f"不共享:        {len(noshare):,} IL | 平均分歧 = {noshare['divergence'].mean():.3f} | 平均预测 = {noshare['mean_kappa'].mean():.3f}")
if len(share) > 10 and len(noshare) > 10:
    print(f"\n分歧差异: 共享 vs 不共享 = {share['divergence'].mean() - noshare['divergence'].mean():+.3f}")
    print(f"（负值 = 共享训练离子的 IL 分歧更小 → 支持训练密度耦合）")
