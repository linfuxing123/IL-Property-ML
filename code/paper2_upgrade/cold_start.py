#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cold_start.py — 冷启动分解：按离子新颖性分解组级外推误差（paper2 升级）

对每个测试 IL 按"离子在训练集是否出现过"分类：
  SS = 阳/阴都见过（新组合）       —— 组合外推
  SC = 阳见过/阴没见过（新阴）     —— 阴离子冷启动
  CS = 阳没见过/阴见过（新阳）     —— 阳离子冷启动
  CC = 阳/阴都没见过               —— 双冷启动
逐类报告 R² / RMSE / 样本数 —— 回答"数据密度在哪个化学维度上卡脖子"。

方法：HistGBM + 5 折 GroupKFold；每折记录每个测试 IL 的预测/真值 + 离子
新颖性标签；最后按类汇总。

输出 (results/)：cold_start_results.csv + fig_cold_start.png
用法: python cold_start.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "workspace" / "matmodel" / "data" / "ilt"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]
PROPS = {"viscosity": True, "conductivity": True, "density": False, "melting_point": False}


def load(prop, use_log):
    csv = f"{prop}.csv" if prop != "melting_point" else "melting_point_all.csv"
    df = pd.read_csv(DATA / csv).dropna(subset=FEATS)
    has_t = prop != "melting_point"
    if has_t:
        df = df.dropna(subset=["T"])
        feats = FEATS + ["T"]
    else:
        feats = list(FEATS)
    y = np.log(df["value"].to_numpy(dtype=float)) if use_log else df["value"].to_numpy(dtype=float)
    return df, df[feats].to_numpy(dtype=float), y


def classify(cat, an, tr_ils):
    """返回新颖性类别。tr_ils: 训练 IL 的 (cat, an) 集合。"""
    seen_cat = any(c == cat for c, _ in tr_ils)
    seen_an = any(a == an for _, a in tr_ils)
    if seen_cat and seen_an:
        return "SS(seen-seen, new combo)"
    if seen_cat:
        return "CS(seen cat, new anion)"
    if seen_an:
        return "SC(seen anion, new cation)"
    return "CC(both new)"


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold

    rows = []
    for prop, use_log in PROPS.items():
        df, X, y = load(prop, use_log)
        g = df["il"].to_numpy()
        cats = df["cat_smiles"].to_numpy()
        ans = df["an_smiles"].to_numpy()
        print(f"\n===== {prop}: {len(X)} records / {len(np.unique(g))} ILs =====")
        kf = GroupKFold(n_splits=5)
        # per-IL 汇总
        il_pred, il_true, il_cls = {}, {}, {}
        for tr, te in kf.split(X, y, groups=g):
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                              max_depth=7, l2_regularization=0.5,
                                              random_state=0)
            m.fit(X[tr], y[tr])
            tr_ils = set(zip(cats[tr], ans[tr]))
            pred = m.predict(X[te])
            for idx, pi, yi, gi in zip(te, pred, y[te], g[te]):
                il_pred.setdefault(gi, []).append(pi)
                il_true.setdefault(gi, []).append(yi)
                il_cls[gi] = classify(cats[idx], ans[idx], tr_ils)
        # 注意：上面按记录聚合，IL 级预测=其记录均值；IL 真值=记录均值
        df_il = pd.DataFrame({
            "il": list(il_pred), "pred": [np.mean(v) for v in il_pred.values()],
            "true": [np.mean(v) for v in il_true.values()],
            "cls": [il_cls[i] for i in il_pred]})
        print("  类别分布:", df_il["cls"].value_counts().to_dict())
        for cls, sub in df_il.groupby("cls"):
            if len(sub) < 5:
                print(f"  {cls:28s} n={len(sub):4d}  (skip <5)")
                continue
            r2 = r2_score(sub["true"], sub["pred"])
            rmse = float(np.sqrt(mean_squared_error(sub["true"], sub["pred"])))
            rows.append({"property": prop, "class": cls, "n_IL": len(sub),
                         "R2": r2, "RMSE": rmse})
            print(f"  {cls:28s} n={len(sub):4d}  R2={r2:+.3f}  RMSE={rmse:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cold_start_results.csv", index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    order = ["SS(seen-seen, new combo)", "CS(seen cat, new anion)",
             "SC(seen anion, new cation)", "CC(both new)"]
    for ax, prop in zip(axes.ravel(), PROPS):
        s = res[res["property"] == prop]
        d = {r["class"]: r["R2"] for _, r in s.iterrows()}
        vals = [d.get(c, np.nan) for c in order]
        ax.bar(range(len(order)), vals, color=["#4C72B0" if not np.isnan(v) else "#cccccc" for v in vals])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([c.split("(")[0] for c in order], rotation=15, fontsize=8)
        ax.set_ylabel("R²"); ax.set_title(prop); ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cold_start.png", dpi=300)
    print(f"\n图已存: {OUT / 'fig_cold_start.png'}")
    print(f"结果已存: {OUT / 'cold_start_results.csv'}")


if __name__ == "__main__":
    main()
