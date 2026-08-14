#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""latent_optimize.py — 潜空间进化优化：从目标性质反推新阳离子结构。

核心（真正的"逆向设计"）：
  固定一个对离子（默认 DCA，Pareto 前沿常见低粘度阴离子），在阳离子 VAE 的
  连续潜空间 z 上做进化搜索，fitness = 预测性质 score（高电导 + 低粘度 + 低熔点）
  − SA 可合成性惩罚，无效/电荷不对的个体判负无穷。
  迭代后解码最优 z → 新阳离子，并与已知最优（EMIM-DCA）对比。
"""
import argparse
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from smiles_vae import VAE, decode  # noqa: E402
sys.path.insert(0, str(ROOT.parents[2] / "data"))
from il_descriptors import ion_descriptors  # noqa: E402

ORACLE = ROOT / "oracle"
DESC = ROOT.parents[2] / "data" / "il_descriptors.csv"
T_REF = 298.15
PROPS = ["conductivity", "density", "viscosity", "melting_point"]


def charge(s):
    m = Chem.MolFromSmiles(s)
    return sum(a.GetFormalCharge() for a in m.GetAtoms()) if m is not None else 0


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s


def predict_il(oracles, cat_desc, an_desc):
    out = {}
    for prop in PROPS:
        m = oracles[prop]["model"]
        feats = oracles[prop]["feats"]
        X = np.zeros((1, len(feats)))
        for j, f in enumerate(feats):
            if f == "T":
                X[0, j] = T_REF
            elif f.startswith("cat_"):
                X[0, j] = np.nan_to_num(cat_desc.get(f, 0.0), nan=0.0)
            elif f.startswith("an_"):
                X[0, j] = np.nan_to_num(an_desc.get(f, 0.0), nan=0.0)
        out[prop] = float(m.predict(X)[0])
    return out


def score_ion(oracles, model, i2c, z, an_desc, an_smiles):
    ids = model.decode(torch.tensor(z, dtype=torch.float32).unsqueeze(0), sample=False).argmax(-1)[0]
    s = decode(ids, i2c)
    if Chem.MolFromSmiles(s) is None or charge(s) != 1:
        return -1e9, s
    d = ion_descriptors(s)
    if not d:
        return -1e9, s
    cat_desc = {("cat_" + k): v for k, v in d.items()}
    p = predict_il(oracles, cat_desc, an_desc)
    sc = p["conductivity"] - 0.5 * p["viscosity"] - 0.01 * p["melting_point"]
    # SA 惩罚：用 RDKit 可合成性（离子层面）
    sa = _sa(s)
    return sc - 0.1 * sa, s


def _sa(s):
    import os as _os
    from rdkit.Chem import RDConfig as _RC
    sys.path.append(_os.path.join(_RC.RDContribDir, "SA_Score"))
    import sascorer
    m = Chem.MolFromSmiles(s)
    return sascorer.calculateScore(m) if m else 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anion", default="N#C[N-]C#N", help="对离子 SMILES")
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(12)

    with open(ROOT / "vae_models" / "vocab_cation.json", encoding="utf-8") as f:
        vocab = json.load(f)
    i2c = {int(k): v for k, v in vocab["i2c"].items()}
    model = VAE(vocab["vocab_size"], latent=vocab["latent"], max_len=vocab["max_len"])
    model.load_state_dict(torch.load(ROOT / "vae_models" / "vae_cation.pt", map_location="cpu"))
    model.emb_pad_id = vocab["c2i"]["<pad>"]
    model.eval()

    oracles = {p: joblib.load(ORACLE / f"gbm_{p}.joblib") for p in PROPS}
    df = pd.read_csv(DESC)
    an_smiles = canon(a.anion)
    an_row = df[df["an_smiles"].map(canon) == an_smiles]
    if len(an_row) == 0:
        print(f"对离子 {a.anion} 不在描述符表"); sys.exit(1)
    an_cols = [c for c in df.columns if c.startswith("an_") and c not in ("an_smiles", "an_ok")]
    an_desc = {c: float(an_row.iloc[0][c]) for c in an_cols if pd.notna(an_row.iloc[0][c])}
    print(f"对离子: {an_smiles}", flush=True)

    latent = vocab["latent"]
    pop = np.random.randn(a.pop, latent).astype(np.float32)
    best = (-1e9, "")
    for gen in range(a.gens):
        scores = []
        for z in pop:
            sc, s = score_ion(oracles, model, i2c, z, an_desc, an_smiles)
            scores.append(sc)
            if sc > best[0]:
                best = (sc, s)
        scores = np.array(scores)
        idx = np.argsort(scores)[::-1]
        elite = pop[idx[: int(a.pop * 0.2)]]
        # 变异生成子代
        child = []
        for _ in range(a.pop - len(elite)):
            parent = elite[np.random.randint(len(elite))]
            child.append(parent + 0.4 * np.random.randn(latent).astype(np.float32))
        pop = np.vstack([elite, np.array(child)])
        if (gen + 1) % 10 == 0:
            print(f"  代 {gen+1}/{a.gens} 最优 score={scores.max():.3f}", flush=True)

    print(f"\n最优 score={best[0]:.3f} 阳离子={best[1]}", flush=True)
    # 与已知最优 EMIM-DCA 对比
    emim = "CC[n+]1ccn(C)c1"
    emim_row = df[df["cat_smiles"].map(canon) == canon(emim)]
    if len(emim_row):
        cat_cols = [c for c in df.columns if c.startswith("cat_") and c not in ("cat_smiles", "cat_ok")]
        emim_desc = {c: float(emim_row.iloc[0][c]) for c in cat_cols if pd.notna(emim_row.iloc[0][c])}
        p_emim = predict_il(oracles, emim_desc, an_desc)
        sc_emim = p_emim["conductivity"] - 0.5 * p_emim["viscosity"] - 0.01 * p_emim["melting_point"]
        print(f"已知 EMIM-{an_smiles} score={sc_emim:.3f} (ln_cond={p_emim['conductivity']:.3f}, ln_visc={p_emim['viscosity']:.3f}, Tm={p_emim['melting_point']:.0f})", flush=True)


if __name__ == "__main__":
    main()
