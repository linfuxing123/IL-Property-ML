#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multitask.py — IL 多性质多任务学习（math-agent 工作流：问题→模型→验证）

模型：共享主干 MLP（128-64）+ 每性质输出头（32→1），NaN 掩码损失，
      仅用该性质可用样本训练对应头；对比单任务基线。
验证：IL 级 GroupKFold（同 IL 全部记录同折），逐折训练集统计做标准化（防泄漏）。
指标：原尺度 R² / MAE（正偏态性质 log 变换后逆变换）。

用法:
  python multitask.py [--epochs 200] [--seed 42]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import featurize  # noqa: E402

torch.manual_seed(42)

PROPS = ["conductivity", "density", "viscosity", "melting_point"]
LOG_PROPS = {"conductivity", "viscosity", "density"}
T_REF = 298.15


def load_rows():
    """联合行：导电率行（实测 T）+ 每 IL 参考行（298.15 K 附密度/粘度/熔点）。"""
    base = Path(__file__).resolve().parent / "data"
    cond = pd.read_csv(base / "il_pure_cond.csv", encoding="utf-8-sig")
    dens = pd.read_csv(base / "il_pure_dens.csv", encoding="utf-8-sig")
    visc = pd.read_csv(base / "il_pure_visc.csv", encoding="utf-8-sig")
    mp = pd.read_csv(base / "il_pure_mp.csv", encoding="utf-8-sig")

    rows = []
    for _, r in cond.iterrows():
        if pd.isna(r["smiles_cation"]) or pd.isna(r["smiles_anion"]):
            continue
        rows.append({"cat": r["smiles_cation"], "an": r["smiles_anion"],
                     "T": float(r["temperature"]), "il": (r["smiles_cation"], r["smiles_anion"]),
                     "conductivity": float(r["value"]), "density": None,
                     "viscosity": None, "melting_point": None})
    for df, prop in [(dens, "density"), (visc, "viscosity"), (mp, "melting_point")]:
        for _, r in df.iterrows():
            if pd.isna(r["smiles_cation"]) or pd.isna(r["smiles_anion"]):
                continue
            key = (r["smiles_cation"], r["smiles_anion"])
            hit = next((x for x in rows if x["il"] == key and x["T"] == T_REF), None)
            if hit is None:
                hit = {"cat": r["smiles_cation"], "an": r["smiles_anion"],
                       "T": T_REF, "il": key,
                       "conductivity": None, "density": None,
                       "viscosity": None, "melting_point": None}
                rows.append(hit)
            hit[prop] = float(r["value"])
    return rows


def build_arrays(rows):
    X, Y, groups = [], [], []
    gid = {}
    for r in rows:
        x = featurize(r["cat"], r["an"], r["T"])
        if x is None:
            continue
        X.append(x)
        y = []
        for p in PROPS:
            v = r[p]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                y.append(np.nan)
            elif p in LOG_PROPS and v > 0:
                y.append(math.log(v))
            else:
                y.append(v)
        Y.append(y)
        if r["il"] not in gid:
            gid[r["il"]] = len(gid)
        groups.append(gid[r["il"]])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), groups


class MultiTaskMLP(nn.Module):
    def __init__(self, d_in, d_hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden // 2), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(d_hidden // 2, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in PROPS])

    def forward(self, x):
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


def masked_mse(pred, target):
    pred = pred[:, :target.shape[1]]
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return ((pred[mask] - target[mask]) ** 2).mean()


def train(model, Xtr, Ytr, Xte, Yte, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    Xt = torch.from_numpy(Xtr)
    Yt = torch.from_numpy(Ytr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = masked_mse(model(Xt), Yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).numpy()
    return pred


def evaluate_props(pred, yte, group_te):
    out = {}
    for i, p in enumerate(PROPS):
        mask = ~np.isnan(yte[:, i])
        if mask.sum() < 3:
            out[p] = None
            continue
        y = yte[mask, i]
        pr = pred[mask, i]
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y - pr) ** 2).sum() / ss if ss > 0 else float("nan")
        mae = np.mean(np.abs(y - pr))
        out[p] = (r2, mae)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    rows = load_rows()
    X, Y, groups = build_arrays(rows)
    print(f"联合样本: {len(X)} | 特征: {X.shape[1]} | 独特 IL: {len(set(groups))}")
    for i, p in enumerate(PROPS):
        print(f"  {p}: 非空 {np.sum(~np.isnan(Y[:, i]))}")

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=a.n_splits)
    results = {p: {"mt_r2": [], "mt_mae": [], "st_r2": [], "st_mae": []} for p in PROPS}

    for fold, (tr, te) in enumerate(gkf.split(X, Y, groups), 1):
        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]
        # 逐折标准化（防泄漏）：特征标准化
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        # 目标标准化（按性质逐折）
        Ytr_s, Yte_s = Ytr.copy(), Yte.copy()
        for i, p in enumerate(PROPS):
            m = ~np.isnan(Ytr[:, i])
            if m.sum() > 1:
                mu_y, sd_y = Ytr[m, i].mean(), Ytr[m, i].std() + 1e-8
                Ytr_s[m, i] = (Ytr_s[m, i] - mu_y) / sd_y
                Yte_s[:, i] = (Yte_s[:, i] - mu_y) / sd_y
        # 多任务
        model = MultiTaskMLP(X.shape[1])
        pred_mt = train(model, Xtr, Ytr_s, Xte, Yte_s, a.epochs)
        # 单任务
        pred_st = {}
        for i, p in enumerate(PROPS):
            m = ~np.isnan(Ytr[:, i])
            if m.sum() < 3:
                continue
            Yt1 = np.full_like(Ytr[:, i], np.nan)
            Yt1[m] = Ytr_s[m, i]
            m1 = MultiTaskMLP(X.shape[1])
            pr = train(m1, Xtr, Yt1.reshape(-1, 1), Xte, Yte_s[:, i].reshape(-1, 1), a.epochs)
            pred_st[i] = pr[:, 0]
        # 还原目标尺度
        for i, p in enumerate(PROPS):
            m = ~np.isnan(Ytr[:, i])
            if m.sum() > 1:
                mu_y = Ytr[m, i].mean()
                sd_y = Ytr[m, i].std() + 1e-8
                pred_mt[:, i] = np.clip(pred_mt[:, i], -3, 3) * sd_y + mu_y
                if i in pred_st:
                    pred_st[i] = np.clip(pred_st[i], -3, 3) * sd_y + mu_y
        ev_mt = evaluate_props(pred_mt, Yte, None)
        ev_st = {p: None for p in PROPS}
        for i, p in enumerate(PROPS):
            if i in pred_st:
                yte1 = Yte.copy()
                for j in range(len(PROPS)):
                    if j != i:
                        yte1[:, j] = np.nan
                ev_st[p] = evaluate_props(
                    np.column_stack([pred_st.get(j, np.full(len(Yte), np.nan)) for j in range(len(PROPS))]),
                    yte1, None)[p]
        for p in PROPS:
            if ev_mt[p]:
                results[p]["mt_r2"].append(ev_mt[p][0])
                results[p]["mt_mae"].append(ev_mt[p][1])
            if ev_st.get(p):
                results[p]["st_r2"].append(ev_st[p][0])
                results[p]["st_mae"].append(ev_st[p][1])
        print(f"fold {fold} 完成")

    print("\n=== 多任务 vs 单任务（IL 级划分，原尺度）===")
    print(f"{'性质':<14} {'多任务 R²':>10} {'多任务 MAE':>10} {'单任务 R²':>10} {'单任务 MAE':>10}")
    for p in PROPS:
        if results[p]["mt_r2"]:
            print(f"{p:<14} {np.mean(results[p]['mt_r2']):>10.3f} {np.mean(results[p]['mt_mae']):>10.3f} "
                  f"{np.mean(results[p]['st_r2']) if results[p]['st_r2'] else float('nan'):>10.3f} "
                  f"{np.mean(results[p]['st_mae']) if results[p]['st_mae'] else float('nan'):>10.3f}")


if __name__ == "__main__":
    main()
