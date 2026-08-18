#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coverage_full.py — 虚拟库化学空间覆盖率统计（paper2 升级）

对 8.33M 虚拟 IL（il_ions_83m.csv）：
  1) 全量离子多样性统计（唯一阳/阴离子数；流式计数，不占内存）
  2) 3M 采样描述符 + 到已测 IL（粘度集）的最近距离分布 → 覆盖缺口量化：
     - 距已测 IL 的最近特征距离（StandardScaler 后）
     - 距离分位数（中位/90 分位/超过 3 的比例）
     - 图：距离直方图 + 累计分布
  3) 输出覆盖缺口最大区域的代表离子（距离 Top 阳/阴组合计数）

输出 (results/)：coverage_stats.csv / fig_coverage_hist.png / coverage_far_ions.csv
用法: python coverage_full.py [--sample 3000000]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "workspace" / "matmodel" / "data" / "ilt"
VIRTUAL = ROOT / "workspace" / "matmodel" / "paper6" / "data" / "ions" / "il_ions_83m.csv"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "ar_rings", "heavy", "fcsp3", "rings"]
T_REF = 298.15

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")


def mol_feats(mol):
    return np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), mol.GetNumHeavyAtoms(),
        Descriptors.FractionCSP3(mol), rdMolDescriptors.CalcNumRings(mol)], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=3000000)
    args = ap.parse_args()

    # 1) 全量离子多样性（流式计数）
    print("counting unique ions over full 8.33M ...")
    cats, ans = set(), set()
    n = 0
    for chunk in pd.read_csv(VIRTUAL, usecols=["cat", "an"], chunksize=2_000_000):
        cats.update(chunk["cat"].dropna().unique())
        ans.update(chunk["an"].dropna().unique())
        n += len(chunk)
    stats = {"total_virtual_il": n, "unique_cations": len(cats), "unique_anions": len(ans)}
    print(stats)

    # 已测 IL 代表特征
    dfm = pd.read_csv(DATA / "viscosity.csv").dropna(subset=FEATS)
    agg = dfm.groupby("il")[FEATS].mean()
    agg["T"] = T_REF
    meas_rep = agg[FEATS + ["T"]].to_numpy(dtype=float)
    print(f"measured ILs: {len(agg)}")

    # 2) 采样描述符 + 距离
    print(f"sampling {args.sample} virtual ILs ...")
    virt = pd.read_csv(VIRTUAL, usecols=["il", "cat", "an"]).sample(n=args.sample, random_state=0)
    feats = np.empty((len(virt), len(FEATS)), dtype=np.float32)
    ok = np.zeros(len(virt), dtype=bool)
    for i, r in enumerate(virt.itertuples()):
        m = Chem.MolFromSmiles(r.cat + "." + r.an)
        if m is not None:
            feats[i] = mol_feats(m)
            ok[i] = True
    virt, feats = virt[ok].reset_index(drop=True), feats[ok]
    print(f"descriptors ok: {len(virt)} / {args.sample}")

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(meas_rep)
    virt_scaled = sc.transform(np.hstack([feats, np.full((len(virt), 1), T_REF)]))
    meas_scaled = sc.transform(meas_rep)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(meas_scaled)
    dists, _ = nn.kneighbors(virt_scaled, n_neighbors=1)
    d = dists[:, 0]
    stats.update({
        "sampled_n": len(d),
        "dist_median": float(np.median(d)),
        "dist_p90": float(np.percentile(d, 90)),
        "dist_p99": float(np.percentile(d, 99)),
        "frac_dist_gt_2": float(np.mean(d > 2)),
        "frac_dist_gt_3": float(np.mean(d > 3)),
    })
    print(stats)

    pd.DataFrame([stats]).to_csv(OUT / "coverage_stats.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "coverage_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 3) 覆盖缺口最大区域的代表离子
    far = virt[d > 3].copy()
    far["cat"] = far["cat"].astype(str)
    far["an"] = far["an"].astype(str)
    top_cats = far["cat"].value_counts().head(20).rename_axis("cation").reset_index(name="n_far")
    top_ans = far["an"].value_counts().head(20).rename_axis("anion").reset_index(name="n_far")
    far_ions = {"n_far_gt3": int(len(far))}
    top_cats.to_csv(OUT / "coverage_far_cations.csv", index=False, encoding="utf-8-sig")
    top_ans.to_csv(OUT / "coverage_far_anions.csv", index=False, encoding="utf-8-sig")
    print(f"ILs beyond dist 3: {len(far)} ({100*len(far)/len(d):.2f}%)")
    print("top far cations:\n", top_cats.head(8).to_string())
    print("top far anions:\n", top_ans.head(8).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(d, bins=60, density=True, alpha=0.7, label="virtual ILs")
    ax.axvline(stats["dist_median"], color="r", ls="--", label=f"median {stats['dist_median']:.2f}")
    ax.axvline(stats["dist_p90"], color="orange", ls="--", label=f"p90 {stats['dist_p90']:.2f}")
    ax.set_xlabel("nearest distance to measured IL (std units)")
    ax.set_ylabel("density"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage_hist.png", dpi=300)
    print(f"图已存: {OUT / 'fig_coverage_hist.png'}")


if __name__ == "__main__":
    main()
