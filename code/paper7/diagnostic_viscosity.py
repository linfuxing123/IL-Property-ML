# -*- coding: utf-8 -*-
"""diagnostic_viscosity.py — 补 viscosity 的诊断价值验证。"""
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

tr = pd.read_csv(DS / "viscosity.csv")
tr_il = tr.groupby("il")["value"].mean().reset_index()
ci = tr_il["il"].str.split("|").str[0].map(cat_idx).to_numpy()
ai = tr_il["il"].str.split("|").str[1].map(an_idx).to_numpy()
valid = (ci != -1) & (ai != -1)
tr_il = tr_il[valid].reset_index(drop=True)
ci, ai = ci[valid], ai[valid]
print(f"viscosity IL: {len(tr_il):,}")

preds = {}
for name in ["gbm", "hist"]:
    pkg = joblib.load(ORACLE / f"{name}_viscosity.joblib")
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
corr = np.corrcoef(d, err)[0, 1]
print(f"viscosity: corr(分歧, |误差|) = {corr:+.3f}")

qs = np.quantile(d, np.linspace(0, 1, 5))
for i in range(4):
    mask = (d >= qs[i]) & (d < qs[i+1])
    if mask.sum() > 5:
        print(f"  Q{i+1}: err={err[mask].mean():.3f} (n={mask.sum():,})")
