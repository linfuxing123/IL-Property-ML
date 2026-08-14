#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gnn_oracle.py — batch 化图编码的 GNN oracle 训练（GPU，第三正交预测器）。

关键优化：把逐图 Python 循环改成 batch 化——合并一组图的 node_feat/edge_index/edge_attr，
一次 MPNN 前向，再用 scatter 做 mean/sum/max 读出，GPU 利用率大幅提升。
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper4"))
from gnn_exp import smiles_to_graph, load_prop, NODE_DIM, EDGE_DIM  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "oracle"
PROPS = ["conductivity", "density", "viscosity", "melting_point"]
NO_T = {"melting_point"}
HIDDEN = 96


class BatchMPNN(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=HIDDEN, layers=3):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden)
        self.msg = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.layers = layers

    def forward(self, x, edge_index, e, batch):
        x = self.node_in(x)
        e = self.edge_in(e)
        for k in range(self.layers):
            src, dst = edge_index
            m = self.msg[k](torch.cat([x[src], e], dim=1))
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, m)
            x = x + self.upd[k](torch.cat([x, agg], dim=1))
            x = F.relu(x)
        n_graphs = int(batch.max()) + 1
        mean = _scatter_mean(x, batch, n_graphs)
        s = torch.zeros(n_graphs, x.shape[1], device=x.device)
        s.index_add_(0, batch, x)
        mx = torch.full((n_graphs, x.shape[1]), -1e9, device=x.device)
        mx = torch.scatter_reduce(mx, 0, batch.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=True)
        return torch.cat([mean, s, mx], dim=1)


def _scatter_mean(x, batch, n):
    s = torch.zeros(n, x.shape[1], device=x.device)
    s.index_add_(0, batch, x)
    c = torch.zeros(n, device=x.device)
    c.index_add_(0, batch, torch.ones(x.shape[0], device=x.device))
    return s / c.clamp(min=1).unsqueeze(-1)


class ILGNN(nn.Module):
    def __init__(self, use_T=True):
        super().__init__()
        self.encoder = BatchMPNN()
        self.head = nn.Sequential(
            nn.Linear(6 * HIDDEN + (1 if use_T else 0), 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.use_T = use_T

    def encode_unique(self, gcache, smiles_arr):
        uni = list(dict.fromkeys(smiles_arr.tolist()))
        graphs = [gcache[s] for s in uni]
        xs, eis, eas, batch = [], [], [], []
        off_n = 0
        for i, g in enumerate(graphs):
            xs.append(g["x"])
            ei = g["edge_index"] + off_n
            eis.append(ei)
            eas.append(g["edge_attr"])
            batch.append(torch.full((g["x"].shape[0],), i, dtype=torch.long, device=g["x"].device))
            off_n += g["x"].shape[0]
        x = torch.cat(xs, 0)
        ei = torch.cat(eis, 1) if eis else torch.zeros((2, 0), dtype=torch.long, device=x.device)
        ea = torch.cat(eas, 0) if eas else torch.zeros((0, EDGE_DIM), device=x.device)
        b = torch.cat(batch, 0)
        emb = self.encoder(x, ei, ea, b)  # (n_uni, 3*hidden)
        idx = {s: i for i, s in enumerate(uni)}
        return emb[[idx[s] for s in smiles_arr]]

    def forward(self, gcache, cat_s, an_s, T):
        hc = self.encode_unique(gcache, cat_s)
        ha = self.encode_unique(gcache, an_s)
        if T is not None:
            return self.head(torch.cat([hc, ha, T.unsqueeze(-1)], dim=1)).squeeze(-1)
        return self.head(torch.cat([hc, ha], dim=1)).squeeze(-1)


def _fit(model, gcache, cat_s, an_s, Ttr, ytr, epochs, device):
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
    print(f"[{prop}] IL-disjoint R2={r2:.4f} -> 已保存", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for prop in PROPS:
        train_one(prop, 200, device)


if __name__ == "__main__":
    main()
