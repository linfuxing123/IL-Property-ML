#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SHAP 机理分析 + 温度敏感性（math-agent 工作流：验证 + 敏感性）"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import featurize, DESC_NAMES  # noqa: E402


def load():
    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "il_pure_cond.csv",
                     encoding="utf-8-sig")
    X, y, names, rows = [], [], [], []
    for _, r in df.iterrows():
        if pd.isna(r["smiles_cation"]) or pd.isna(r["smiles_anion"]):
            continue
        x = featurize(r["smiles_cation"], r["smiles_anion"], float(r["temperature"]),
                      float(r["mole_fraction"]) if not pd.isna(r["mole_fraction"]) else 0.0)
        if x is None:
            continue
        X.append(x)
        y.append(math.log(float(r["value"])))
        rows.append(r)
    feats = ([f"C_{n}" for n in DESC_NAMES] + [f"A_{n}" for n in DESC_NAMES] +
             [f"ECFPcat_{i}" for i in range(1024)] +
             [f"ECFPan_{i}" for i in range(1024)] + ["mole_frac", "T(K)"])
    return np.asarray(X), np.asarray(y), feats, rows


def main():
    X, y, feats, rows = load()
    print(f"样本 {len(X)} | 特征 {X.shape[1]}")
    model = GradientBoostingRegressor(
        n_estimators=80, max_depth=4, learning_rate=0.1,
        subsample=0.9, random_state=42)
    model.fit(X, y)
    tr_r2 = model.score(X, y)
    print(f"GBM 全量拟合 R² = {tr_r2:.3f}")

    import shap
    print("计算 SHAP（TreeExplainer）...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1][:20]
    print("\n=== 电导率 Top-20 特征（mean|SHAP|）===")
    for rank, i in enumerate(order, 1):
        print(f"{rank:>2}. {feats[i]:<20} {mean_abs[i]:.4f}")

    # 温度敏感性：T ±10K
    idx = np.arange(len(X))
    rng = np.random.default_rng(0)
    sample = rng.choice(idx, min(200, len(X)), replace=False)
    deltas = []
    for i in sample:
        x0 = X[i].copy()
        xp = x0.copy()
        xp[-1] += 10.0
        deltas.append(abs(model.predict([xp])[0] - model.predict([x0])[0]))
    print(f"\n=== 温度敏感性（±10 K，lnκ 变化）===")
    print(f"样本 {len(sample)}：mean |Δlnκ| = {np.mean(deltas):.4f}（≈ κ 相对变化 {100*(np.exp(np.mean(deltas))-1):.2f}%）")

    Path("workspace/matmodel/shap_top20.txt").write_text(
        "\n".join(f"{rank}. {feats[i]}  {mean_abs[i]:.4f}" for rank, i in enumerate(order, 1)),
        encoding="utf-8")
    print("\n已存 workspace/matmodel/shap_top20.txt")


if __name__ == "__main__":
    main()
