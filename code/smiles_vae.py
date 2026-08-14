#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smiles_vae.py — 字符级 VAE（阳/阴离子 SMILES），第五篇逆向设计的基础生成器。

目标：在 867 阳离子 / 356 阴离子的 SMILES 上学一个连续潜空间，验证能否生成
化学有效的新离子。这是 MVP，先跑通「生成 → RDKit 有效性」闭环。
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parents[3]
DESC = ROOT / "data" / "il_descriptors.csv"

PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"


def build_vocab(smiles_list):
    chars = sorted(set("".join(smiles_list)))
    vocab = [PAD, SOS, EOS] + chars
    return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}


def encode(smiles, c2i, max_len):
    ids = [c2i[SOS]] + [c2i[c] for c in smiles[:max_len - 2]] + [c2i[EOS]]
    ids = ids + [c2i[PAD]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def decode(ids, i2c):
    out = []
    for i in ids:
        c = i2c[int(i)]
        if c in (EOS, PAD):
            break
        if c != SOS:
            out.append(c)
    return "".join(out)


class VAE(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, latent=32, max_len=80):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.enc = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(2 * hidden, latent)
        self.logvar = nn.Linear(2 * hidden, latent)
        self.z2h = nn.Linear(latent, hidden)
        self.dec = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)
        self.latent = latent
        self.hidden = hidden
        self.max_len = max_len

    def encode(self, x):
        e = self.emb(x)
        _, h = self.enc(e)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x=None, sample=False):
        B = z.shape[0]
        h = self.z2h(z).unsqueeze(0)
        if x is not None:  # teacher forcing
            e = self.emb(x)
            o, _ = self.dec(e, h)
            return self.out(o)
        # 自由生成
        inp = torch.full((B, 1), self.emb_pad_id, dtype=torch.long, device=z.device)
        outs = []
        for _ in range(self.max_len):
            e = self.emb(inp)
            o, h = self.dec(e, h)
            logits = self.out(o[:, -1])
            if sample:
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            outs.append(logits.unsqueeze(1))
            inp = nxt
        return torch.cat(outs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ion", choices=["cation", "anion"], default="cation")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canonical", action="store_true", help="训练前 RDKit 规范化 SMILES")
    ap.add_argument("--save", action="store_true", help="保存模型 + vocab")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(12)

    df = pd.read_csv(DESC)
    col = "cat_smiles" if a.ion == "cation" else "an_smiles"
    smiles = df[col].dropna().unique().tolist()
    if a.canonical:
        can = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                can.append(Chem.MolToSmiles(m))
        smiles = list(dict.fromkeys(can))
    smiles = [s for s in smiles if 3 <= len(s) <= a.max_len - 2]
    print(f"[{a.ion}] 训练 SMILES 数: {len(smiles)}", flush=True)

    c2i, i2c = build_vocab(smiles)
    V = len(c2i)
    model = VAE(V, latent=a.latent, max_len=a.max_len)
    model.emb_pad_id = c2i[PAD]

    data = torch.stack([encode(s, c2i, a.max_len) for s in smiles])
    data = data[torch.randperm(len(data))]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep in range(a.epochs):
        tot = 0.0
        n = 0
        for i in range(0, len(data), a.batch):
            x = data[i:i + a.batch]
            mu, logvar = model.encode(x)
            z = model.reparam(mu, logvar)
            logits = model.decode(z, x)
            # 目标右移一位
            tgt = torch.cat([x[:, 1:], torch.full((x.shape[0], 1), c2i[PAD], dtype=torch.long)], 1)
            recon = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=c2i[PAD])
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
            loss = recon + a.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/{a.epochs} loss={tot/n:.3f}", flush=True)

    # 生成 + RDKit 有效性验证
    model.eval()
    n_gen = 2000
    valid = 0
    unique = set()
    with torch.no_grad():
        for _ in range(n_gen // 200):
            z = torch.randn(200, a.latent)
            logits = model.decode(z, sample=True)
            ids = logits.argmax(-1)
            for row in ids:
                s = decode(row, i2c)
                if Chem.MolFromSmiles(s) is not None:
                    valid += 1
                    unique.add(s)
    print(f"[{a.ion}] 生成 {n_gen} 个，RDKit 有效 {valid}（{100*valid/n_gen:.1f}%），独特 {len(unique)}", flush=True)
    # 新颖性（不在训练集）
    novel = unique - set(smiles)
    print(f"  独特中新颖（不在训练集）: {len(novel)}", flush=True)

    if a.save:
        outdir = pathlib.Path(__file__).resolve().parent / "vae_models"
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), outdir / f"vae_{a.ion}.pt")
        with open(outdir / f"vocab_{a.ion}.json", "w", encoding="utf-8") as f:
            json.dump({"c2i": c2i, "i2c": i2c, "max_len": a.max_len,
                       "latent": a.latent, "vocab_size": V}, f)
        print(f"模型已保存: {outdir}", flush=True)


if __name__ == "__main__":
    main()
