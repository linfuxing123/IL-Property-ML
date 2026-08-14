#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_gnn.py — 用 GNN 第三预测器对最终候选做交叉验证，更新候选集。

读 final_candidates.csv，对每个候选用 GNN 复测 4 性质，与 GBM/HistGBM 对比一致性。
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

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gnn_oracle import ILGNN  # noqa: E402
sys.path.insert(0, str(ROOT.parents[1] / "paper4"))
from gnn_exp import smiles_to_graph  # noqa: E402

PROPS = ["conductivity", "density", "viscosity", "melting_point"]


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s


def predict_gnn(model, cfg, gcache, cat_s, an_s, T=None):
    model.eval()
    with torch.no_grad():
        p = model(gcache, np.array([cat_s]), np.array([an_s]), T)
    return float(p.cpu().numpy()[0]) * cfg["sd_y"] + cfg["mu_y"]


def main():
    cand = pd.read_csv(ROOT / "generated" / "final_candidates.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for _, r in cand.iterrows():
        cs, asm = canon(r["cat_smiles"]), canon(r["an_smiles"])
        # 构建图 cache（候选的阳/阴）
        gcache = {}
        for s in (cs, asm):
            g = smiles_to_graph(s)
            if g is not None:
                gcache[s] = {k: v.to(device) for k, v in g.items()}
        # 用 GNN 预测（温度依赖性质用 298.15K）
        gnn = {}
        for prop in PROPS:
            model = ILGNN(use_T=prop != "melting_point").to(device)
            model.load_state_dict(torch.load(ROOT / "oracle" / f"gnn_{prop}.pt", map_location=device))
            with open(ROOT / "oracle" / f"gnn_{prop}_cfg.json") as f:
                cfg = json.load(f)
            T = None
            if prop != "melting_point":
                T = torch.tensor([(298.15 - cfg["mu_t"]) / cfg["sd_t"]], dtype=torch.float32, device=device)
            gnn[prop] = predict_gnn(model, cfg, gcache, cs, asm, T)
        rows.append({
            "cat_smiles": cs, "an_smiles": asm,
            "g_cond": r["g_cond"], "h_cond": r["h_cond"],
            "gnn_cond": gnn["conductivity"], "gnn_visc": gnn["viscosity"], "gnn_tm": gnn["melting_point"],
            "gnn_dens": gnn["density"],
        })
    res = pd.DataFrame(rows)
    res["gbm_gnn_gap"] = (res["g_cond"] - res["gnn_cond"]).abs()
    res["gnn_agree"] = res["gbm_gnn_gap"] < 0.8
    res.to_csv(ROOT / "generated" / "final_candidates_gnn.csv", index=False, encoding="utf-8-sig")
    print(f"GNN 第三重验证（|GBM−GNN| lnκ < 0.8 视为一致）: {int(res['gnn_agree'].sum())}/{len(res)} 一致")
    cols = ["cat_smiles", "an_smiles", "g_cond", "h_cond", "gnn_cond", "gbm_gnn_gap", "gnn_agree"]
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
