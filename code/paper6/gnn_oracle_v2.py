#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gnn_oracle_v2.py — 第 6 篇：扩充数据集上重训 MPNN（第三预测器）。

协议与 gnn_oracle.py 完全一致（batch 化图编码 + IL-disjoint GroupKFold(5)），
数据换成 data/expanded/paper_dataset（6,177 IL 实验库）。
产出：data/expanded/oracle_v2/gnn_{prop}.pt + cfg
对比基准（paper5 GNN，1,891 IL）：0.703/0.933/0.759/0.365
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

P6 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper5"))
from gnn_oracle import ILGNN  # noqa: E402

DS = P6 / "data" / "expanded" / "paper_dataset"
OUT = P6 / "data" / "expanded" / "oracle_v2"
PROPS = ["conductivity", "density", "viscosity", "melting_point"]
NO_T = {"melting_point"}


def load_prop(prop):
    return pd.read_csv(DS / f"{prop}.csv")


def _fit(model, gcache, cat_s, an_s, Ttr, ytr, epochs, device):
    """同 gnn_oracle._fit（早停 + 归一化）。"""
    import torch.nn as nn
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    n = len(ytr)
    nval = max(1, int(0.1 * n))
    perm = torch.randperm(n, device=device)
    val_idx = perm[:nval].tolist()
    tr_idx = perm[nval:].tolist()
    best = float("inf"); patience = 0
    cat_tr, an_tr = cat_s[tr_idx], an_s[tr_idx]
    cat_va, an_va = cat_s[val_idx], an_s[val_idx]
    T_tr = None if Ttr is None else Ttr[tr_idx]
    T_va = None if Ttr is None else Ttr[val_idx]
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(gcache, cat_tr, an_tr, T_tr)
        loss = lossf(pred, ytr[tr_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(gcache, cat_va, an_va, T_va), ytr[val_idx]).item()
        if vl < best - 1e-6:
            best = vl; patience = 0
        else:
            patience += 1
            if patience >= 20:
                break


def train_one(prop, epochs, device):
    use_T = prop not in NO_T
    df = load_prop(prop)
    cat_s = df["cat_smiles"].to_numpy()
    an_s = df["an_smiles"].to_numpy()
    y = df["value"].to_numpy(dtype=np.float32)
    g = df["il"].to_numpy()
    T = df["T"].to_numpy(dtype=np.float32) if use_T else None
    # 图缓存（全量唯一离子）
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper4"))
    from gnn_exp import smiles_to_graph  # noqa: E402
    gcache = {}
    for s in set(cat_s.tolist()) | set(an_s.tolist()):
        gg = smiles_to_graph(s)
        if gg is not None:
            gcache[s] = {k: v.to(device) for k, v in gg.items()}

    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(5)
    yt, yp = [], []
    for tr, te in cv.split(df, y, groups=g):
        model = ILGNN(use_T=use_T).to(device)
        mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
        ytr = torch.tensor((y[tr] - mu_y) / sd_y, dtype=torch.float32, device=device)
        Ttr = None
        if use_T:
            mu_t, sd_t = T[tr].mean(), T[tr].std() + 1e-8
            Ttr = torch.tensor((T[tr] - mu_t) / sd_t, dtype=torch.float32, device=device)
        _fit(model, gcache, cat_s[tr], an_s[tr], Ttr, ytr, epochs, device)
        model.eval()
        with torch.no_grad():
            Tte = None
            if use_T:
                mu_t, sd_t = T[tr].mean(), T[tr].std() + 1e-8
                Tte = torch.tensor((T[te] - mu_t) / sd_t, dtype=torch.float32, device=device)
            pte = model(gcache, cat_s[te], an_s[te], Tte).cpu().numpy()
        yp.extend((pte * sd_y + mu_y).tolist())
        yt.extend(y[te].tolist())
    yt, yp = np.asarray(yt), np.asarray(yp)
    r2 = 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()

    # 全量重训
    model = ILGNN(use_T=use_T).to(device)
    mu_y, sd_y = y.mean(), y.std() + 1e-8
    ytr = torch.tensor((y - mu_y) / sd_y, dtype=torch.float32, device=device)
    Ttr = None
    if use_T:
        mu_t, sd_t = T.mean(), T.std() + 1e-8
        Ttr = torch.tensor((T - mu_t) / sd_t, dtype=torch.float32, device=device)
    _fit(model, gcache, cat_s, an_s, Ttr, ytr, epochs, device)
    torch.save(model.state_dict(), OUT / f"gnn_{prop}.pt")
    with open(OUT / f"gnn_{prop}_cfg.json", "w") as f:
        json.dump({"use_T": use_T, "mu_y": float(mu_y), "sd_y": float(sd_y),
                   "mu_t": float(T.mean()) if use_T else 0.0,
                   "sd_t": float(T.std()) if use_T else 1.0}, f)
    print(f"[{prop}] n={len(df)} IL-disjoint R2={r2:.4f} -> 已保存", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for prop in PROPS:
        train_one(prop, 200, device)


if __name__ == "__main__":
    main()
