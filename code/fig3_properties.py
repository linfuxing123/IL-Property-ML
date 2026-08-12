#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig3_properties.py — 四性质 IL 级划分预测 vs 实验散点图（Figure 3）

与 model.py 完全同口径：load_data + build_Xy + GroupKFold(5) + HistGBM。
每个面板输出 R² / MAE（以该性质的建模尺度），供手稿引用。
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_Xy, load_data  # noqa: E402

OUT = Path(__file__).resolve().parent

PROPS = [
    ("conductivity", "il_pure_cond.csv", True, "ln κ (κ in S m⁻¹)"),
    ("density", "il_pure_dens.csv", False, "Density (g cm⁻³)"),
    ("viscosity", "il_pure_visc.csv", True, "ln η (η in mPa·s)"),
    ("melting_point", "il_pure_mp.csv", False, "Melting point (K)"),
]


def load_prop(fname, log_target):
    rows = load_data(OUT / "data" / fname)
    return build_Xy(rows, log_target=log_target)


def cv_metrics(X, y, groups):
    """5 折 GroupKFold + HistGBM（与 results-cond-v2.txt 同超参）。"""
    ys, ps = [], []
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=6,
            random_state=42, early_stopping=True)
        m.fit(X[tr], y[tr])
        ys.extend(y[te])
        ps.extend(m.predict(X[te]))
    ys = np.asarray(ys)
    ps = np.asarray(ps)
    return r2_score(ys, ps), mean_absolute_error(ys, ps), ys, ps


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    for ax, (name, fname, log_target, axis_label) in zip(axes.ravel(), PROPS):
        X, y, groups = load_prop(fname, log_target)
        r2, mae, ys, ps = cv_metrics(X, y, groups)
        lim = [min(ys.min(), ps.min()) - 0.05 * abs(min(ys.min(), ps.min())),
               max(ys.max(), ps.max()) + 0.05 * abs(max(ys.max(), ps.max()))]
        ax.plot(lim, lim, "k--", lw=1)
        ax.scatter(ys, ps, s=14, alpha=0.65, c="#1f77b4", edgecolors="none")
        ax.set_xlabel(f"Experimental {axis_label}")
        ax.set_ylabel(f"Calculated {axis_label}")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{name}  R² = {r2:.3f}  MAE = {mae:.3f}", fontsize=10)
        print(f"{name}: n={len(ys)} IL={len(set(groups))} R²={r2:.4f} MAE={mae:.4f}")
    fig.suptitle("Figure 3. IL-level (GroupKFold) prediction of four ionic-liquid properties.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig3_properties.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig3_properties.png OK ->", OUT / "fig3_properties.png")


if __name__ == "__main__":
    main()
