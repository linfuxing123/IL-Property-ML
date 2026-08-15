#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_smiles_lm.py — 第 6 篇：大规模 SMILES 语言模型生成器（对照 VAE）。

科学对照：paper5 字符 VAE（860 阳离子，生成 40% 电荷错误）；
本实验：SMILES 自回归 LM（GRU）+ 大规模语料（219,292 阳离子 + 可选 30M 预训练）。
LM 无 VAE 的 KL 坍缩问题，是生成化学领域（MOSES）的标准基线，有效性通常 >90%。
产出：vae_models/lm_cation_v2.pt + 生成验证报告。
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

P6 = pathlib.Path(__file__).resolve().parent
IONS = P6 / "data" / "ions" / "cations_83m.csv"
PRETRAIN = P6 / "data" / "pretrain_train_set.txt"   # 30M SMILES（可选）
OUT = P6 / "vae_models"

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


class SMILESLM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, vocab_size)
        self.n_layers = n_layers
        self.hidden = hidden

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.rnn(e)
        return self.out(o)

    def sample(self, c2i, i2c, max_len, temp=1.0, topk=20, n=1, device="cuda"):
        """自回归采样：temperature + top-k。"""
        h = torch.zeros(self.n_layers, n, self.hidden, device=device)
        inp = torch.full((n, 1), c2i[SOS], dtype=torch.long, device=device)
        outs = []
        for _ in range(max_len):
            e = self.emb(inp)
            o, h = self.rnn(e, h)
            logits = self.out(o[:, -1]) / temp
            # top-k 过滤
            if topk > 0:
                k = min(topk, logits.shape[-1])
                v, _ = logits.topk(k)
                logits[logits < v[:, -1:]] = -1e9
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            outs.append(nxt)
            inp = nxt
            # 全部 EOS 就停
            if (nxt == c2i[EOS]).all():
                break
        ids = torch.cat(outs, 1)
        return [decode(row, i2c) for row in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=200_000)
    ap.add_argument("--use-pretrain", action="store_true",
                    help="用 30M pretrain 语料扩充训练（更强语法先验）")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)

    df = pd.read_csv(IONS)
    smiles = df["smiles"].dropna().unique().tolist()
    rng = np.random.RandomState(a.seed)
    if len(smiles) > a.max_samples:
        smiles = rng.choice(smiles, a.max_samples, replace=False).tolist()

    if a.use_pretrain:
        # 采样 pretrain 语料（阳离子侧重：过滤含正电荷的 SMILES）
        pre = []
        with open(PRETRAIN, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and ("[N+]" in s or "[n+]" in s or "[NH+]" in s or "[P+]" in s):
                    pre.append(s)
                if len(pre) >= a.max_samples:
                    break
        print(f"pretrain 阳离子类 SMILES: {len(pre)}", flush=True)
        smiles = smiles + pre
    print(f"训练 SMILES 总数: {len(smiles)}", flush=True)

    c2i, i2c = build_vocab(smiles)
    V = len(c2i)
    model = SMILESLM(V).to(dev)

    data = torch.stack([encode(s, c2i, a.max_len) for s in smiles]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_all = len(data)
    n_batches = (n_all + a.batch - 1) // a.batch

    model.train()
    for ep in range(a.epochs):
        perm = torch.randperm(n_all, device=dev)
        tot, n = 0.0, 0
        for bi in range(n_batches):
            idx = perm[bi * a.batch:(bi + 1) * a.batch]
            x = data[idx]
            logits = model(x)
            tgt = torch.cat([x[:, 1:], torch.full((x.shape[0], 1), c2i[PAD], dtype=torch.long, device=dev)], 1)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=c2i[PAD])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        print(f"  epoch {ep+1}/{a.epochs} loss={tot/n:.3f}", flush=True)

    # 生成验证
    model.eval()
    n_gen = 5000
    valid, chg1, unique = 0, 0, set()
    with torch.no_grad():
        for _ in range(n_gen // 500):
            rows = model.sample(c2i, i2c, a.max_len, temp=a.temp, topk=a.topk,
                                n=500, device=dev)
            for s in rows:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    valid += 1
                    unique.add(s)
                    if Chem.GetFormalCharge(m) == 1:
                        chg1 += 1
    novel = unique - set(smiles)
    print(f"[LM 生成] {n_gen} 个：RDKit 有效 {100*valid/n_gen:.1f}% | "
          f"电荷=+1 {100*chg1/n_gen:.1f}% | 独特 {len(unique)} | 新颖 {len(novel)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "lm_cation_v2.pt")
    with open(OUT / "vocab_lm_v2.json", "w", encoding="utf-8") as f:
        json.dump({"c2i": c2i, "i2c": i2c, "max_len": a.max_len}, f)
    print("已保存:", OUT, flush=True)


if __name__ == "__main__":
    main()
