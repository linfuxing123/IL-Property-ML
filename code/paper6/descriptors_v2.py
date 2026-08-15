#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""descriptors_v2.py — 第 6 篇：为扩充后的唯一 IL 集生成 459 维描述符。

复用 data/il_descriptors.py 的 ion_descriptors 计算逻辑（RDKit 全套 + 10 核心），
但输入改为扩充后数据集（paper5 + Mendeley），只算 paper5 描述符表里没有的新 IL，
输出 paper6/data/il_descriptors_v2.csv（含全部 IL，便于直接建模）。
"""
import argparse
import csv
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data"))
from il_descriptors import ion_descriptors  # noqa: E402

P6 = Path(__file__).resolve().parent
EXP = P6 / "data" / "expanded" / "paper_dataset"
OLD_DESC = ROOT / "data" / "il_descriptors.csv"


def load_unique_ils():
    import pandas as pd
    union = {}
    for f in sorted(EXP.glob("*.csv")):
        if f.name in ("co2_solubility.csv", "toxicity.csv"):
            continue  # 新性质描述符在阳/阴层面已经覆盖（同 IL 族）
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            cat = str(row.get("cat_smiles", "")).strip()
            an = str(row.get("an_smiles", "")).strip()
            if cat and an and cat != "nan" and an != "nan":
                union[f"{cat}|{an}"] = (cat, an)
    return union


def compute_one(item):
    key, (cat, an) = item
    row = {"il": key, "cat_smiles": cat, "an_smiles": an}
    cd, ad = ion_descriptors(cat), ion_descriptors(an)
    row["cat_ok"] = 1 if cd else 0
    row["an_ok"] = 1 if ad else 0
    for k, v in cd.items():
        row["cat_" + k] = v
    for k, v in ad.items():
        row["an_" + k] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()

    ils = load_unique_ils()
    print(f"扩充后唯一 IL 数: {len(ils)}", flush=True)

    # 跳过已在 paper5 描述符表中的 IL（增量）
    old_keys = set()
    if OLD_DESC.exists():
        import pandas as pd
        old_keys = set(pd.read_csv(OLD_DESC, usecols=["il"])["il"])
    todo = {k: v for k, v in ils.items() if k not in old_keys}
    print(f"paper5 已有: {len(old_keys & set(ils))} | 需新算: {len(todo)}", flush=True)

    results = []
    items = list(todo.items())
    if items:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(compute_one, it) for it in items]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 500 == 0:
                    print(f"  已完成 {i}/{len(items)}", flush=True)

    # 合并 paper5 旧表 + 新算的
    out_path = P6 / "data" / "il_descriptors_v2.csv"
    all_cols = []
    for r in results:
        for k in r:
            if k not in all_cols:
                all_cols.append(k)
    # 旧表列顺序继承（保证特征顺序与 paper5 一致）
    if OLD_DESC.exists():
        with open(OLD_DESC, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            old_rows = list(rd)
        old_cols = rd.fieldnames or []
        for k in old_cols:
            if k not in all_cols:
                all_cols.insert(0, k)  # 保持旧列在前面
        for r in results:
            for k in old_cols:
                r.setdefault(k, None)
    ok = sum(1 for r in results if r.get("cat_ok") and r.get("an_ok"))
    print(f"新算完成: {len(results)} | 阳+阴成功: {ok}", flush=True)

    # 写出：新 IL 全量 + 旧表行（保证可独立建模）
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in old_rows:
            w.writerow({k: r.get(k) for k in all_cols})
        for r in results:
            w.writerow({k: r.get(k) for k in all_cols})
    print("CSV:", out_path, out_path.stat().st_size, "bytes", flush=True)

    # 入 SQLite（新表，不覆盖旧表）
    con = sqlite3.connect(ROOT / "data" / "il_props.db")
    con.execute("DROP TABLE IF EXISTS il_descriptors_v2")
    con.execute(
        f"""CREATE TABLE il_descriptors_v2 (
            il TEXT PRIMARY KEY, cat_smiles TEXT, an_smiles TEXT,
            cat_ok INTEGER, an_ok INTEGER,
            {", ".join(f'"{c}" REAL' for c in all_cols if c not in
                       ("il", "cat_smiles", "an_smiles", "cat_ok", "an_ok"))}
        )"""
    )
    all_rows = [{k: r.get(k) for k in all_cols} for r in old_rows] + results
    for r in all_rows:
        cols = [c for c in all_cols if c not in
                ("il", "cat_smiles", "an_smiles", "cat_ok", "an_ok")]
        con.execute(
            "INSERT OR REPLACE INTO il_descriptors_v2 (il,cat_smiles,an_smiles,cat_ok,an_ok,"
            + ",".join(f'"{c}"' for c in cols) + ") VALUES (?,?,?,?,?,"
            + ",".join("?" for _ in cols) + ")",
            [r["il"], r["cat_smiles"], r["an_smiles"], int(r.get("cat_ok") or 0),
             int(r.get("an_ok") or 0)] + [r.get(c) for c in cols],
        )
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM il_descriptors_v2").fetchone()[0]
    con.close()
    print(f"SQLite il_descriptors_v2 表: {n} 行", flush=True)


if __name__ == "__main__":
    main()
