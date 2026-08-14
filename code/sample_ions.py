#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample_ions.py — 从保存的 VAE 大规模采样新离子，过滤有效/新颖，计算描述符。

输出：paper5/generated/<ion>_novel.csv（smiles + cat_*/an_* 描述符，与 il_descriptors 对齐）
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from smiles_vae import VAE, decode  # noqa: E402
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "data"))
from il_descriptors import ion_descriptors  # noqa: E402

DESC = ROOT.parents[2] / "data" / "il_descriptors.csv"


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s


def load_model(ion, device):
    with open(ROOT / "vae_models" / f"vocab_{ion}.json", encoding="utf-8") as f:
        vocab = json.load(f)
    model = VAE(vocab["vocab_size"], latent=vocab["latent"], max_len=vocab["max_len"])
    model.load_state_dict(torch.load(ROOT / "vae_models" / f"vae_{ion}.pt", map_location=device))
    model.emb_pad_id = vocab["c2i"]["<pad>"]
    model.eval().to(device)
    return model, vocab


def sample(model, vocab, n, device, batch=500):
    i2c = {int(k): v for k, v in vocab["i2c"].items()}
    out = []
    with torch.no_grad():
        for _ in range(0, n, batch):
            z = torch.randn(batch, vocab["latent"], device=device)
            logits = model.decode(z, sample=True)
            ids = logits.argmax(-1)
            for row in ids:
                s = decode(row, i2c)
                if Chem.MolFromSmiles(s) is not None:
                    out.append(canon(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ion", choices=["cation", "anion"], required=True)
    ap.add_argument("--n", type=int, default=30000)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ion={a.ion} 采样 {a.n}", flush=True)
    model, vocab = load_model(a.ion, device)
    smi = sample(model, vocab, a.n, device)
    print(f"有效 SMILES: {len(smi)}", flush=True)

    # 新颖性：不在训练集
    df = pd.read_csv(DESC)
    col = "cat_smiles" if a.ion == "cation" else "an_smiles"
    known = set(df[col].dropna().map(canon).unique())
    novel = [s for s in dict.fromkeys(smi) if s not in known]
    print(f"去重后新颖: {len(novel)}", flush=True)

    # 计算描述符（前缀 cat_/an_）
    prefix = "cat_" if a.ion == "cation" else "an_"
    rows = []
    for s in novel:
        d = ion_descriptors(s)
        if not d:
            continue
        row = {"smiles": s}
        for k, v in d.items():
            row[prefix + k] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    outdir = ROOT / "generated"
    outdir.mkdir(exist_ok=True)
    out.to_csv(outdir / f"{a.ion}_novel.csv", index=False, encoding="utf-8-sig")
    print(f"描述符完成 {len(out)} -> {outdir / f'{a.ion}_novel.csv'}", flush=True)


if __name__ == "__main__":
    main()
