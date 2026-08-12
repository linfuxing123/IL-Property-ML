#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figures.py — Science 手稿图表（模板 Figure 1/4/5-6/7-8 版式）

产出:
  fig1_framework.png    框架示意图（SMILES→描述符→模型→性质）
  fig2_split_inflation  点级 vs IL 级划分散点（模板 Figure 5/6 版式）
  fig3_shap.png         SHAP beeswarm（模板 Figure 7/8 版式）
  fig4_multitask.png    多任务 vs 单任务 R² 对比柱状图
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DESC_NAMES, build_Xy, load_data  # noqa: E402

OUT = Path(__file__).resolve().parent


def load_cond():
    """与 model.py 完全同口径：同一加载器 + 同一特征化 + 同一 IL 分组。"""
    rows = load_data(OUT / "data" / "il_pure_cond.csv")
    return build_Xy(rows, log_target=True)


def fig1_framework():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, "SMILES\n(cation / anion)", "#dbeafe"),
        (0.24, "RDKit 2D descriptors\n+ ECFP4 (1024 bit)", "#dcfce7"),
        (0.46, "x = [Φ(cat) ⊕ Φ(an)\n⊕ x_IL ⊕ T] ∈ R^2070", "#fef9c3"),
        (0.68, "Multi-task MLP\nshared trunk + heads", "#fee2e2"),
        (0.90, "κ, η, ρ,\nT_m, pKa, ...", "#ede9fe"),
    ]
    for x0, label, color in boxes:
        ax.add_patch(plt.Rectangle((x0, 0.35), 0.12, 0.30, fc=color, ec="#334155", lw=1.2))
        ax.text(x0 + 0.06, 0.50, label, ha="center", va="center", fontsize=9)
    for x0 in (0.14, 0.36, 0.58, 0.80):
        ax.annotate("", xy=(x0 + 0.11, 0.5), xytext=(x0 + 0.005, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6))
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1. Unified structure-based multi-property prediction framework for ionic liquids.",
                 fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_framework.png", dpi=300)
    plt.close(fig)
    print("fig1_framework.png OK")


def _fit_and_collect(X, y, groups, mode):
    ys, ps, trs, tes = [], [], [], []
    if mode == "point":
        splits = KFold(5, shuffle=True, random_state=42).split(X)
    else:
        splits = GroupKFold(5).split(X, y, groups)
    for tr, te in splits:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6,
                                          random_state=42, early_stopping=True)
        m.fit(X[tr], y[tr])
        ps.extend(m.predict(X[te]))
        ys.extend(y[te])
        trs.extend([False] * len(te))
    return np.asarray(ys), np.asarray(ps)


def fig2_split_inflation():
    X, y, groups = load_cond()
    y_il, p_il = _fit_and_collect(X, y, groups, "il")
    y_pt, p_pt = _fit_and_collect(X, y, groups, "point")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (yy, pp, title) in zip(axes, [
            (y_il, p_il, "(a) IL-based split  R² = %.3f" %
             (1 - ((y_il - p_il) ** 2).sum() / ((y_il - y_il.mean()) ** 2).sum())),
            (y_pt, p_pt, "(b) Point-wise split  R² = %.3f" %
             (1 - ((y_pt - p_pt) ** 2).sum() / ((y_pt - y_pt.mean()) ** 2).sum()))]):
        lim = [min(yy.min(), pp.min()) - 0.5, max(yy.max(), pp.max()) + 0.5]
        ax.plot(lim, lim, "k--", lw=1)
        ax.scatter(yy, pp, s=12, alpha=0.6, c="#1f77b4", edgecolors="none")
        ax.set_xlabel("Experimental ln κ values")
        ax.set_ylabel("Calculated ln κ values")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Figure 2. Experimental vs calculated ln κ (conductivity, GBM).",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig2_split_inflation.png", dpi=300)
    plt.close(fig)
    print("fig2_split_inflation.png OK")


def fig3_shap():
    import shap
    from sklearn.ensemble import GradientBoostingRegressor
    X, y, _ = load_cond()
    m = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)
    m.fit(X, y)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(300, len(X)), replace=False)
    Xs = X[idx]
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(Xs)
    order = np.argsort(np.abs(sv).mean(0))[::-1][:15]
    feats = ([f"C_{n}" for n in DESC_NAMES] + [f"A_{n}" for n in DESC_NAMES] +
             [f"ECFPcat_{i}" for i in range(1024)] +
             [f"ECFPan_{i}" for i in range(1024)] + ["x_IL", "T(K)"])
    feat_names = [feats[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    shap.summary_plot(sv[:, order], Xs[:, order], feature_names=feat_names,
                      show=False, max_display=15)
    plt.title("Figure 3. SHAP beeswarm plot (top 15 features, conductivity model).",
              fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_shap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig3_shap.png OK")


def fig4_multitask():
    props = ["conductivity", "density", "viscosity", "melting_point"]
    mt = [-0.876, -0.296, -1.210, -0.363]
    st = [-0.620, -0.210, -1.404, -0.466]
    x = np.arange(len(props))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.35
    ax.bar(x - w / 2, mt, w, label="Multi-task", color="#60a5fa")
    ax.bar(x + w / 2, st, w, label="Single-task", color="#f87171")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(props, fontsize=9)
    ax.set_ylabel("Test R² (IL-based split)")
    ax.set_title("Figure 4. Multi-task vs single-task under IL-based splitting.",
                 fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig4_multitask.png", dpi=300)
    plt.close(fig)
    print("fig4_multitask.png OK")


def main():
    fig1_framework()
    fig2_split_inflation()
    fig4_multitask()
    fig3_shap()
    print("全部图表已生成 ->", OUT)


if __name__ == "__main__":
    main()
