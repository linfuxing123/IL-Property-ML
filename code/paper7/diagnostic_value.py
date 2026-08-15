# -*- coding: utf-8 -*-
"""diagnostic_value.py — 验证"分歧标记不可靠区"的诊断价值。

核心检验：训练集 IL 上，GBM/Hist 分歧大的区域，预测误差（|均值-实验|）是否
确实更大。若成立 → 分歧是分布偏差的有效诊断（无需标签即可定位不可靠区）。
覆盖 4 性质（电导/密度/熔点/粘度）。
"""
import joblib
import numpy as np
import pandas as pd
import pathlib

P6 = pathlib.Path(r"D:\Codex\MEC-Workspace\workspace\matmodel\paper6")
DS = P6 / "data" / "expanded" / "paper_dataset"
ORACLE = P6 / "data" / "expanded" / "oracle_v2"
DESC = P6 / "data" / "il_descriptors_v2.csv"

desc = pd.read_csv(DESC)
cat_cols = [c for c in desc.columns if c.startswith("cat_") and c not in ("cat_smiles", "cat_ok")]
an_cols = [c for c in desc.columns if c.startswith("an_") and c not in ("an_smiles", "an_ok")]
cat_uniq = desc.drop_duplicates("cat_smiles").set_index("cat_smiles", drop=False)
an_uniq = desc.drop_duplicates("an_smiles").set_index("an_smiles", drop=False)
cat_idx = {c: i for i, c in enumerate(cat_uniq.index.tolist())}
an_idx = {a: i for i, a in enumerate(an_uniq.index.tolist())}
cat_mat = np.nan_to_num(cat_uniq[cat_cols].to_numpy(dtype=float), nan=0.0)
an_mat = np.nan_to_num(an_uniq[an_cols].to_numpy(dtype=float), nan=0.0)

props = {
    "conductivity": ("gbm_conductivity", "hist_conductivity", False),
    "density": ("gbm_density", "hist_density", True),
    "melting_point": ("gbm_melting_point", "hist_melting_point", False),
    "viscosity": ("gbm_viscosity", "hist_viscosity", True),
}

print("=== 诊断价值验证：分歧分层 vs 预测误差（|mean - 实验|）===")
for prop, (c1, c2, needT) in props.items():
    tr = pd.read_csv(DS / f"{prop}.csv")
    tr_il = tr.groupby("il")["value"].mean().reset_index()
    ci = tr_il["il"].str.split("|").str[0].map(cat_idx).to_numpy()
    ai = tr_il["il"].str.split("|").str[1].map(an_idx).to_numpy()
    valid = (ci != -1) & (ai != -1)
    tr_il = tr_il[valid].reset_index(drop=True)
    ci, ai = ci[valid], ai[valid]
    if len(tr_il) < 30:
        print(f"{prop}: 样本不足 {len(tr_il)}，跳过")
        continue

    preds = {}
    for name, col in [("gbm", c1), ("hist", c2)]:
        pkg = joblib.load(ORACLE / f"{name}_{prop}.joblib")
        feats = pkg["feats"]
        n = len(tr_il)
        X = np.zeros((n, len(feats)), dtype=np.float32)
        for j, f in enumerate(feats):
            if f in cat_cols:
                X[:, j] = cat_mat[ci, cat_cols.index(f)]
            elif f in an_cols:
                X[:, j] = an_mat[ai, an_cols.index(f)]
            elif f == "T":
                X[:, j] = 298.15
        preds[name] = pkg["model"].predict(X)

    m = (preds["gbm"] + preds["hist"]) / 2
    d = np.abs(preds["gbm"] - preds["hist"])
    gt = tr_il["value"].to_numpy()
    err = np.abs(m - gt)

    # 分歧分层 → 每层平均误差
    qs = np.quantile(d, np.linspace(0, 1, 5))
    layers = []
    for i in range(4):
        mask = (d >= qs[i]) & (d < qs[i+1])
        if mask.sum() > 5:
            layers.append((i, round(float(err[mask].mean()), 3), int(mask.sum())))
    corr_d_err = np.corrcoef(d, err)[0, 1]
    print(f"\n{prop} (n={len(tr_il):,}): corr(分歧, |误差|) = {corr_d_err:+.3f}")
    seq = [f"Q{i+1}:err={e}(n={n})" for i, e, n in layers]
    print("   " + " | ".join(seq), flush=True)
