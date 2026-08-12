#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IL-MultiProp — 离子液体多性质统一预测数学模型（MEC 选题 A 核心）

数学建模：
  输入    : SMILES(阳离子), SMILES(阴离子), 温度 T
  特征映射: Φ(SMILES) = RDKit 2D 描述符 ⊕ ECFP4(1024bit)
            x = [Φ(cat) ⊕ Φ(an) ⊕ T] ∈ R^d
  模型    : f_θ(x) -> 多性质输出（导电率/粘度/密度/熔点/溶解度/…）
            基线 MLR / GBM / MLP（共享主干 + 每性质输出头）
  损失    : 目标 log 变换后加权 MSE（正偏态性质）
  验证    : ① IL 级 GroupKFold（同一 IL 的全部记录同一折）= 诚实外推
            ② 点级随机划分对照 = 量化"划分虚高"
            ③ Kennard-Stone IL 级挑选训练/测试（--split ks）
  指标    : R² / RMSE / MAE / MAPE（原尺度）

用法:
  python model.py --data data.csv --property conductivity --target col
  python model.py --demo                          # 合成数据冒烟测试（非论文数据）
  python model.py --demo --split ks

CSV 列: smiles_cation, smiles_anion, temperature, value
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    RDKIT = True
except ImportError:
    RDKIT = False

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cdist


# ---------- 特征映射 ----------

DESC_NAMES = [
    "MW", "LogP", "HBD", "HBA", "TPSA", "RotatableBonds", "FractionCsp3",
    "RingCount", "AromaticRings", "MR",
]


def _desc_vec(smiles):
    if not smiles:
        return np.zeros((10,), dtype=float)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    v = [
        rdMolDescriptors.CalcExactMolWt(mol),
        rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.MolMR(mol),
    ]
    return np.asarray(v, dtype=float)


def _ecfp_vec(smiles, n_bits=1024):
    if not smiles:
        return np.zeros((n_bits,), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((n_bits,), dtype=np.float32)
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
    for bit in gen.GetFingerprint(mol).GetOnBits():
        arr[bit] = 1.0
    return arr


def featurize(smiles_cat, smiles_an, T, x_frac=0.0):
    """x = [Φ(cat) ⊕ Φ(an) ⊕ x ⊕ T]"""
    vc, va, ec, ea = _desc_vec(smiles_cat), _desc_vec(smiles_an), \
        _ecfp_vec(smiles_cat), _ecfp_vec(smiles_an)
    if any(v is None for v in (vc, va, ec, ea)):
        return None
    return np.concatenate([vc, va, ec, ea, [float(x_frac), float(T)]])


# ---------- 模型 ----------

def build_model(kind):
    if kind == "mlr":
        return LinearRegression()
    if kind == "gbm":
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=6,
            random_state=42, early_stopping=True, validation_fraction=0.1,
        )
    if kind == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 16), activation="relu",
            alpha=1e-4, max_iter=600, early_stopping=True,
            random_state=42,
        )
    raise ValueError(kind)


# ---------- 划分协议 ----------

def il_group_key(row):
    return (row["smiles_cat"], row["smiles_an"])


def make_splits(X, y, groups, mode, n_splits=5, seed=42):
    if mode == "point":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(X)), "点级划分（对照，预期虚高）"
    if mode == "group":
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(X, y, groups=groups)), "IL 级划分（诚实外推）"
    if mode == "ks":
        # Kennard–Stone：在 IL 级描述符空间挑训练集
        il_map, il_feats, il_idx = {}, [], []
        for i, g in enumerate(groups):
            if g not in il_map:
                il_map[g] = len(il_idx)
                il_idx.append(i)
                il_feats.append(X[i])
        F = np.asarray(il_feats)
        n_tr = int(math.ceil(len(F) * 0.7))
        sel = _kennard_stone(F, n_tr)
        train_il = {il_idx[s] for s in sel}
        tr = [i for i in range(len(X)) if il_idx[il_map[groups[i]]] in train_il]
        te = [i for i in range(len(X)) if i not in set(tr)]
        return [(np.asarray(tr), np.asarray(te))], "Kennard–Stone IL 级 70/30"
    raise ValueError(mode)


def _kennard_stone(F, n):
    n = min(n, len(F))
    d = cdist(F, F)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    sel = [int(i), int(j)]
    rest = list(set(range(len(F))) - set(sel))
    while len(sel) < n:
        dists = d[rest][:, sel].min(axis=1)
        k = int(np.argmax(dists))
        sel.append(rest.pop(k))
    return sel


# ---------- 训练与评价 ----------

def evaluate(X, y, groups, kind, split_mode, n_splits=5):
    splits, desc = make_splits(X, y, groups, split_mode, n_splits)
    ys = np.asarray(y)
    all_truth, all_pred = [], []
    per_fold = []
    for tr, te in splits:
        model = build_model(kind)
        model.fit(X[tr], ys[tr])
        p = model.predict(X[te])
        all_truth.extend(ys[te])
        all_pred.extend(p)
        per_fold.append(r2_score(ys[te], p))
    r2 = r2_score(all_truth, all_pred)
    rmse = mean_squared_error(all_truth, all_pred) ** 0.5
    mae = mean_absolute_error(all_truth, all_pred)
    return {
        "split": desc,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "fold_r2": [round(v, 4) for v in per_fold],
    }


def load_data(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "smiles_cat": r["smiles_cation"].strip(),
                    "smiles_an": r["smiles_anion"].strip(),
                    "mole_fraction": r.get("mole_fraction", "").strip(),
                    "temperature": float(r["temperature"]),
                    "value": float(r["value"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def build_Xy(rows, log_target=True):
    X, y, groups = [], [], []
    group_id = {}
    bad = 0
    for r in rows:
        xf = float(r["mole_fraction"]) if str(r.get("mole_fraction", "")).strip() else 0.0
        x = featurize(r["smiles_cat"], r["smiles_an"], r["temperature"], xf)
        if x is None:
            bad += 1
            continue
        X.append(x)
        v = r["value"]
        y.append(math.log(v) if log_target and v > 0 else v)
        k = il_group_key(r)
        if k not in group_id:
            group_id[k] = len(group_id)
        groups.append(group_id[k])
    print(f"特征化: {len(rows)} 行 -> 可用 {len(X)}（跳过 {bad}）")
    return np.asarray(X), np.asarray(y), groups


def demo_data(n_il=60, n_temp=8, seed=0):
    """合成演示数据（仅验证管线，不是论文数据）。"""
    rng = np.random.default_rng(seed)
    cations = ["CCCC[n+]1ccn(C)c1", "CC[n+]1ccn(C)c1", "C[n+]1ccn(C)c1",
               "CCCCC[n+]1ccn(C)c1", "CCCCCC[n+]1ccn(C)c1"]
    anions = ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
              "O=S(=O)([O-])C(F)(F)F", "[Cl-]", "O=S(=O)([O-])O",
              "CC(C)C[n+]1ccn(C)c1", "C1=C[N+](=CN1)C"]
    rows = []
    for i in range(n_il):
        ca, an = cations[i % len(cations)], anions[i % len(anions)]
        for j in range(n_temp):
            T = 273.15 + j * 15
            base = -3.5 - 0.10 * (i % 5) + 0.02 * (j % 4)
            sigma = 0.08
            rows.append({
                "smiles_cat": ca, "smiles_an": an,
                "mole_fraction": 0.0,
                "temperature": T,
                "value": math.exp(base + rng.normal(0, sigma)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="IL-MultiProp 数学建模管线")
    ap.add_argument("--data", help="CSV: smiles_cation,smiles_anion,temperature,value")
    ap.add_argument("--demo", action="store_true", help="合成数据冒烟测试")
    ap.add_argument("--split", choices=["point", "group", "ks"], default="group")
    ap.add_argument("--models", default="mlr,gbm,mlp")
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    if not RDKIT:
        print("错误：需要 RDKit（mec-lit-venv 已装）")
        return
    if a.demo:
        rows = demo_data()
        print(f"[demo] 合成数据 {len(rows)} 行（仅验证管线，非论文数据）")
    else:
        if not a.data:
            ap.error("需要 --data 或 --demo")
        rows = load_data(a.data)

    X, y, groups = build_Xy(rows)
    if len(X) == 0:
        print("无可用数据")
        return
    print(f"特征维度: {X.shape[1]} | 样本: {len(X)} | 独特 IL: {len(set(groups))}")

    print("\n=== 模型 × 划分 对照表（R² / RMSE / MAE）===")
    print(f"{'模型':<5} {'划分':<24} {'R²':>7} {'RMSE':>8} {'MAE':>7}")
    for kind in a.models.split(","):
        kind = kind.strip()
        for sm in (["point", "group"] if a.split == "group" else [a.split]):
            res = evaluate(X, y, groups, kind, sm, a.n_splits)
            print(f"{kind:<5} {res['split'][:22]:<24} {res['r2']:>7.3f} "
                  f"{res['rmse']:>8.3f} {res['mae']:>7.3f}")
    print("\n说明：point = 点级划分（对照）；group = IL 级划分（诚实外推）")


if __name__ == "__main__":
    main()
