#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gnn_exp.py — 第四篇：端到端图神经网络 vs 手写描述符（IL-disjoint 诚实验证）

问题：在数据、验证、协议都固定的前提下，学出来的图表示（message-passing GNN，
从 SMILES 分子图端到端训练）能否把手写 RDKit 描述符（10 / 458 特征 GBM）的
外推边界再往前推？

协议（与第二、三篇一致）：IL 级 GroupKFold(5)，目标列 value
（电导率/粘度已是 ln 尺度，密度/熔点原尺度）；10 描述符 GBM 与 458 描述符 GBM
作为基线（同折、同参数），GNN 为实验组。

模型：
  阳/阴离子分别构图（原子特征 + 键特征）→ 3 层 MPNN 编码器 → [mean;sum] 读出
  → 拼接 [h_cat, h_an, T?] → MLP 头 → 单值（熔点无 T）。

用法:
  python gnn_exp.py --prop melting_point --folds 2 --epochs 5 --smoke
  python gnn_exp.py --workers 4
"""
import argparse
import concurrent.futures
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[3]
DS = ROOT / "workspace" / "matmodel" / "data" / "ilt" / "paper_dataset"
DESC = ROOT / "data" / "il_descriptors.csv"
OUT = ROOT / "workspace" / "matmodel" / "paper4"

PROPS = ["conductivity", "density", "viscosity", "melting_point"]
NO_T = {"melting_point"}
BASE10 = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]

ATOM_TYPES = ["C", "N", "O", "F", "S", "P", "Cl", "Br", "I", "B", "Si", "H"]
ATOM_OTHER = len(ATOM_TYPES)          # 未知原子专用槽
NODE_DIM = len(ATOM_TYPES) + 1 + 7 + 1 + 1 + 4 + 1 + 1 + 1   # 29
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
EDGE_DIM = len(BOND_TYPES) + 1                                  # 5


# ---------------- 图特征化 ----------------

def _atom_feat(atom):
    sym = atom.GetSymbol()
    typ = np.zeros(len(ATOM_TYPES) + 1, dtype=np.float32)
    if sym in ATOM_TYPES:
        typ[ATOM_TYPES.index(sym)] = 1.0
    else:
        typ[ATOM_OTHER] = 1.0
    deg = np.zeros(7, dtype=np.float32)
    d = min(int(atom.GetDegree()), 6)
    deg[d] = 1.0
    hyb = np.zeros(4, dtype=np.float32)
    hs = str(atom.GetHybridization()).upper()
    hyb[0 if "SP" == hs else (1 if "SP2" == hs else (2 if "SP3" == hs else 3))] = 1.0
    mass = atom.GetMass() / 100.0
    return np.concatenate([
        typ, deg,
        np.array([float(atom.GetFormalCharge()), float(atom.GetTotalNumHs())], dtype=np.float32),
        hyb,
        np.array([float(atom.GetIsAromatic()), float(atom.IsInRing()), mass], dtype=np.float32),
    ])


def _bond_feat(bond):
    bt = str(bond.GetBondType()).upper()
    typ = np.zeros(len(BOND_TYPES), dtype=np.float32)
    if bt in BOND_TYPES:
        typ[BOND_TYPES.index(bt)] = 1.0
    conj = float(bond.GetIsConjugated())
    return np.concatenate([typ, np.array([conj], dtype=np.float32)])


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    nodes = np.stack([_atom_feat(a) for a in mol.GetAtoms()]).astype(np.float32)
    src, dst, edges = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        e = _bond_feat(b)
        src += [i, j]
        dst += [j, i]
        edges += [e, e]
    n = len(nodes)
    if n == 0:
        return None
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(np.stack(edges), dtype=torch.float32) if edges else torch.zeros((0, EDGE_DIM), dtype=torch.float32)
    return {"x": torch.tensor(nodes, dtype=torch.float32),
            "edge_index": edge_index, "edge_attr": edge_attr}


# ---------------- MPNN ----------------

class MPNN(nn.Module):
    """3 层 message passing + [mean;sum;max] 读出，输出图嵌入。"""

    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=96, layers=3, dropout=0.1):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden)
        self.msg = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.layers = layers
        self.drop = nn.Dropout(dropout)

    def forward(self, g):
        x = self.node_in(g["x"])
        e = self.edge_in(g["edge_attr"])
        ei = g["edge_index"]
        for k in range(self.layers):
            if ei.shape[1] == 0:
                agg = torch.zeros_like(x)
            else:
                src, dst = ei[0], ei[1]
                m = self.msg[k](torch.cat([x[src], e], dim=1))
                agg = torch.zeros_like(x)
                agg.index_add_(0, dst, m)
            x = x + self.upd[k](torch.cat([x, agg], dim=1))
            x = torch.relu(x)
            x = self.drop(x)
        return torch.cat([x.mean(0), x.sum(0), x.max(0).values])


class ILGNN(nn.Module):
    def __init__(self, hidden=96, use_T=True, layers=3):
        super().__init__()
        self.encoder = MPNN(hidden=hidden, layers=layers)
        emb = 6 * hidden
        head_in = emb + (1 if use_T else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))

    def forward(self, cat_graphs, an_graphs, T):
        hc = torch.stack([self.encoder(g) for g in cat_graphs])
        ha = torch.stack([self.encoder(g) for g in an_graphs])
        if T is not None:
            return self.head(torch.cat([hc, ha, T], dim=1)).squeeze(-1)
        return self.head(torch.cat([hc, ha], dim=1)).squeeze(-1)


# ---------------- 数据 ----------------

def load_prop(prop):
    df = pd.read_csv(DS / f"{prop}.csv")
    desc = pd.read_csv(DESC)
    df = df.merge(desc.drop(columns=["cat_smiles", "an_smiles"]), on="il", how="inner")
    df = df.drop(columns=[c for c in df.columns if c in ("cat_ok", "an_ok")])
    return df


def featsets(df, prop):
    cat = [c for c in df.columns if c.startswith("cat_")]
    an = [c for c in df.columns if c.startswith("an_")]
    full = cat + an + (["T"] if prop not in NO_T else [])
    base = BASE10 + (["T"] if prop not in NO_T else [])
    return base, full


def run_gbm(df, feats, y):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all").fillna(X.median()).fillna(0.0)
    g = df["il"].to_numpy()
    cv = GroupKFold(n_splits=5)
    yt, yp = [], []
    for tr, te in cv.split(X, y, groups=g):
        m = GradientBoostingRegressor(random_state=0)
        m.fit(X.iloc[tr], y[tr])
        yp.extend(m.predict(X.iloc[te]))
        yt.extend(y[te])
    yt, yp = np.asarray(yt), np.asarray(yp)
    return {"R2": r2_score(yt, yp),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp)))}


def _encode(model, gcache, smiles_arr):
    """按记录编码图，唯一 SMILES 只算一次（保持梯度）。"""
    uni = list(dict.fromkeys(smiles_arr.tolist()))
    emap = {s: model.encoder(gcache[s]) for s in uni}
    return torch.stack([emap[s] for s in smiles_arr])


def _forward(model, gcache, cat_s, an_s, T):
    hc = _encode(model, gcache, cat_s)
    ha = _encode(model, gcache, an_s)
    if T is not None:
        if T.dim() == 1:
            T = T.unsqueeze(-1)
        return model.head(torch.cat([hc, ha, T], dim=1)).squeeze(-1)
    return model.head(torch.cat([hc, ha], dim=1)).squeeze(-1)


def _train_gnn(model, gcache, cat_s, an_s, Ttr, ytr, epochs, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    n = len(ytr)
    nval = max(1, int(0.1 * n))
    perm = torch.randperm(n)
    val_idx = perm[:nval].tolist()
    tr_idx = perm[nval:].tolist()
    cat_tr, an_tr = cat_s[tr_idx], an_s[tr_idx]
    Ttr_tr = None if Ttr is None else Ttr[tr_idx]
    cat_va, an_va = cat_s[val_idx], an_s[val_idx]
    Ttr_va = None if Ttr is None else Ttr[val_idx]
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = _forward(model, gcache, cat_tr, an_tr, Ttr_tr)
        loss = lossf(pred, ytr[tr_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = _forward(model, gcache, cat_va, an_va, Ttr_va)
            vl = lossf(pv, ytr[val_idx]).item()
        if vl < best - 1e-6:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 20:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def run_gnn(df, prop, epochs):
    torch.set_num_threads(6)
    use_T = prop not in NO_T
    cat_smiles = df["cat_smiles"].to_numpy()
    an_smiles = df["an_smiles"].to_numpy()
    y = df["value"].to_numpy(dtype=np.float32)
    g = df["il"].to_numpy()
    T = df["T"].to_numpy(dtype=np.float32) if use_T else None
    n_records = df.groupby("il").size().to_dict()

    # 预构图（每 IL 一次）
    gcache = {}
    for s in set(cat_smiles.tolist()) | set(an_smiles.tolist()):
        gg = smiles_to_graph(s)
        if gg is not None:
            gcache[s] = gg
    assert all(s in gcache for s in cat_smiles), "存在无法构图的阳离子"
    assert all(s in gcache for s in an_smiles), "存在无法构图的阴离子"

    cv = GroupKFold(n_splits=5)
    yt, yp, oof_il, oof_fold, fold_r2 = [], [], [], [], []
    for fold, (tr, te) in enumerate(cv.split(df, y, groups=g)):
        print(f"  [{prop}] fold {fold} 训练开始 ({len(tr)} 训练点)", flush=True)
        # 逐折标准化目标与 T（防泄漏）
        mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
        ytr = torch.tensor((y[tr] - mu_y) / sd_y, dtype=torch.float32)
        yte = torch.tensor((y[te] - mu_y) / sd_y, dtype=torch.float32)
        Ttr = None
        if use_T:
            mu_t, sd_t = T[tr].mean(), T[tr].std() + 1e-8
            Ttr = torch.tensor((T[tr] - mu_t) / sd_t, dtype=torch.float32)
            Tte = torch.tensor((T[te] - mu_t) / sd_t, dtype=torch.float32)
        model = ILGNN(use_T=use_T)
        _train_gnn(model, gcache, cat_smiles[tr], an_smiles[tr], Ttr, ytr, epochs)
        model.eval()
        with torch.no_grad():
            pte = _forward(model, gcache, cat_smiles[te], an_smiles[te],
                           None if not use_T else Tte).numpy()
        pred = pte * sd_y + mu_y
        yp.extend(pred.tolist())
        yt.extend(y[te].tolist())
        oof_il.extend(g[te].tolist())
        oof_fold.extend([fold] * len(te))
        fold_r2.append(r2_score(y[te], pred))
        print(f"  [{prop}] fold {fold} R²={fold_r2[-1]:.4f}", flush=True)
    yt, yp = np.asarray(yt), np.asarray(yp)
    oof = pd.DataFrame({
        "il": oof_il, "truth": yt, "pred": yp, "fold": oof_fold,
        "n_records": [n_records[i] for i in oof_il],
    })
    return {"R2": r2_score(yt, yp),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "fold_r2": [round(v, 4) for v in fold_r2],
            "oof": oof}


def load_gbm_baseline():
    base = ROOT / "workspace" / "matmodel" / "feat_scale_results.csv"
    d = pd.read_csv(base)
    return {r["prop"]: r for _, r in d.iterrows()}


def one_prop(prop, epochs, gbm_base):
    t0 = time.time()
    df = load_prop(prop)
    y = df["value"].to_numpy(dtype=float)
    print(f"[{prop}] {len(df)} 点 / {df['il'].nunique()} IL", flush=True)
    rg = run_gnn(df, prop, epochs)
    b = gbm_base[prop]
    row = {
        "prop": prop, "n": len(df), "n_il": int(df["il"].nunique()),
        "gbm10_R2": round(b["base_R2"], 4), "gbm10_MAE": round(b["base_MAE"], 4),
        "gbm458_R2": round(b["full_R2"], 4), "gbm458_MAE": round(b["full_MAE"], 4),
        "gnn_R2": round(rg["R2"], 4), "gnn_MAE": round(rg["MAE"], 4),
        "gnn_R2_std": round(float(np.std(rg["fold_r2"])), 4),
        "dR2_gnn_vs_458": round(rg["R2"] - b["full_R2"], 4),
        "dR2_gnn_vs_10": round(rg["R2"] - b["base_R2"], 4),
        "sec": round(time.time() - t0, 1),
    }
    oof_csv = OUT / f"oof_{prop}.csv"
    rg["oof"].to_csv(oof_csv, index=False, encoding="utf-8-sig")
    print(f"  -> GBM10 {b['base_R2']:.4f} | GBM458 {b['full_R2']:.4f} | GNN {rg['R2']:.4f} "
          f"| dR2(vs458) {row['dR2_gnn_vs_458']:+.4f} | folds {rg['fold_r2']} | {row['sec']}s", flush=True)
    return row


def _run_one(args):
    prop, epochs, gbm_base = args
    return one_prop(prop, epochs, gbm_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", choices=PROPS + ["all"], default="all")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5, help="仅 smoke 时用（当前固定 5）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    props = PROPS if a.prop == "all" else [a.prop]
    if a.smoke:
        props = props[:1]
        a.epochs = min(a.epochs, 5)
    torch.set_num_threads(max(1, 24 // max(1, len(props))))

    OUT.mkdir(parents=True, exist_ok=True)
    gbm_base = load_gbm_baseline()
    if len(props) == 1:
        rows = [one_prop(props[0], a.epochs, gbm_base)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(props)) as ex:
            rows = list(ex.map(_run_one, [(p, a.epochs, gbm_base) for p in props]))
    res = pd.DataFrame(rows)
    if a.smoke:
        out_csv = OUT / "gnn_results_smoke.csv"
    elif len(props) == 1:
        out_csv = OUT / f"gnn_results_{props[0]}.csv"
    else:
        out_csv = OUT / "gnn_results.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print(f"结果已写入 {out_csv}", flush=True)


if __name__ == "__main__":
    main()
