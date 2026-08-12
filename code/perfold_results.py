#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""perfold_results.py — SI 用的 per-fold 真实结果（与 model.py 同口径）

输出: workspace\science-paper\supplementary\perfold_results.csv
性质尺度（与正文一致）：电导率 lnκ；密度 g/cm³；粘度 lnη；熔点 K。
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_Xy, load_data  # noqa: E402

OUT = Path(__file__).resolve().parent / ".." / "science-paper" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

DATA = {
    "conductivity": ("il_pure_cond.csv", True),
    "density": ("il_pure_dens.csv", False),
    "viscosity": ("il_pure_visc.csv", True),
    "melting_point": ("il_pure_mp.csv", False),
}


def model_for(kind, Xtr, ytr, seed=42):
    if kind == "mlr":
        return LinearRegression().fit(Xtr, ytr)
    if kind == "gbm":
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=6,
            random_state=seed, early_stopping=True).fit(Xtr, ytr)
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(128, 64, 16), activation="relu",
                            batch_size=64, max_iter=300, random_state=seed,
                            early_stopping=True).fit(Xtr, ytr)
    raise ValueError(kind)


def run_prop(prop, models, splits=("point", "group")):
    fname, log_target = DATA[prop]
    rows = load_data(OUT.parent.parent / "matmodel" / "data" / fname)
    X, y, groups = build_Xy(rows, log_target=log_target)
    rows_out = []
    for model in models:
        for split in splits:
            if split == "point":
                folds = KFold(5, shuffle=True, random_state=42).split(X, y)
            else:
                folds = GroupKFold(5).split(X, y, groups)
            for fi, (tr, te) in enumerate(folds, 1):
                m = model_for(model, X[tr], y[tr])
                p = m.predict(X[te])
                rows_out.append({
                    "property": prop, "model": model, "split": split,
                    "fold": fi, "n_test": len(te),
                    "r2": r2_score(y[te], p),
                    "rmse": mean_squared_error(y[te], p) ** 0.5,
                    "mae": mean_absolute_error(y[te], p),
                })
    return rows_out, f"{prop}: n={len(X)} IL={len(set(groups))}"


def main():
    all_rows = []
    for prop, models in [
        ("conductivity", ("mlr", "gbm", "mlp")),
        ("density", ("gbm",)),
        ("viscosity", ("gbm",)),
        ("melting_point", ("gbm",)),
    ]:
        rows, info = run_prop(prop, models)
        all_rows.extend(rows)
        print(info)
    with open(OUT / "perfold_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["property", "model", "split", "fold",
                                          "n_test", "r2", "rmse", "mae"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"per-fold CSV -> {OUT / 'perfold_results.csv'} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
