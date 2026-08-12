#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""il_db.py — 离子液体性质数据库（自建）

来源：joekasp/ionic_liquids（MIT）GitHub 数据集
  - data.xlsx / compounddata.xlsx / compounddata2.xlsx / Bigger_Data_Set.xlsx
  - 字段：A(IL 盐 SMILES)、B(溶剂 SMILES)、摩尔分数、T(K)、P、ELE_COD(电导率 S/m ± 不确定度)

用法:
  python il_db.py init
  python il_db.py import --dir workspace\\matmodel\\data
  python il_db.py report
  python il_db.py export-csv --subset dense|sparse|all --out out.csv
"""
import argparse
import re
import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB = Path(__file__).resolve().parent / "il_props.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cat_smiles TEXT,
    an_smiles TEXT,
    b_smiles TEXT,
    mole_frac REAL,
    T REAL,
    P REAL,
    property TEXT,
    value REAL,
    uncertainty REAL,
    source TEXT,
    UNIQUE(cat_smiles, an_smiles, b_smiles, mole_frac, T, property, value)
);
CREATE INDEX IF NOT EXISTS idx_records_il ON records(cat_smiles, an_smiles);
CREATE INDEX IF NOT EXISTS idx_records_prop ON records(property);
"""

SRC_FILES = {
    "data.xlsx": "joekasp/data.xlsx",
    "compounddata.xlsx": "joekasp/compounddata.xlsx",
    "compounddata2.xlsx": "joekasp/compounddata2.xlsx",
    "Bigger_Data_Set.xlsx": "joekasp/Bigger_Data_Set.xlsx",
}

EXTRA_FILES = {
    "iolitech_joined.csv": "iolitech",
    "ilthermo_matched.csv": "ILThermo(iolitech词表匹配)",
}

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
VAL_RE = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
ERR_RE = re.compile(r"±\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def connect():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def split_salt(smiles):
    """把盐 SMILES（. 连接的离子对）拆成 阳离子/阴离子。"""
    s = str(smiles or "").strip()
    if not s:
        return "", ""
    parts = [p.strip() for p in s.split(".") if p.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        p = parts[0]
        if re.search(r"\[\w*[nNH]\+|\[\w*\+", p):
            return p, ""
        if re.search(r"\[.*[-\]]", p) and "+" not in p:
            return "", p
        return p, ""
    cat, an = [], []
    for p in parts:
        if re.search(r"\[\w*[nNH]\+|\[\w*\+", p):
            cat.append(p)
        elif re.search(r"\[.*-\]", p):
            an.append(p)
        else:
            cat.append(p)  # 中性组分默认归阳离子侧（按数据习惯 A 为 IL）
    return ".".join(cat), ".".join(an)


def parse_num(x):
    s = str(x)
    m = VAL_RE.search(s)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def parse_err(x):
    s = str(x)
    m = ERR_RE.search(s)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def cmd_init(_):
    with connect() as c:
        c.executescript(SCHEMA)
    print(f"OK: 数据库就绪 -> {DB}")


def cmd_import(a):
    import pandas as pd
    d = Path(a.dir)
    total = ins = dup = skip = 0
    with connect() as c:
        c.executescript(SCHEMA)
        for fname, src in SRC_FILES.items():
            fp = d / fname
            if not fp.exists():
                print(f"跳过（不存在）: {fname}")
                continue
            df = pd.read_excel(fp)
            xcol = "MOLFRC(for A and B)" if "MOLFRC(for A and B)" in df.columns else "MOLFRC_A"
            for _, r in df.iterrows():
                total += 1
                cat, an = split_salt(r["A"])
                val, err = parse_num(r.get("ELE_COD")), parse_err(r.get("ELE_COD"))
                if val is None or cat == "":
                    skip += 1
                    continue
                try:
                    c.execute(
                        """INSERT OR IGNORE INTO records
                           (cat_smiles, an_smiles, b_smiles, mole_frac, T, P,
                            property, value, uncertainty, source)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (cat, an, str(r.get("B", "")), float(r[xcol]), float(r["T"]),
                         float(r["P"]), "conductivity", val, err, src),
                    )
                    if c.execute("SELECT changes()").fetchone()[0]:
                        ins += 1
                    else:
                        dup += 1
                except Exception:
                    skip += 1
        c.commit()
    print(f"OK: 读取 {total} 行 -> 新入库 {ins}，重复 {dup}，跳过 {skip}")


def cmd_import_extra(a):
    import pandas as pd
    d = Path(a.dir)
    total = ins = dup = skip = 0
    with connect() as c:
        c.executescript(SCHEMA)
        # iolitech 多性质
        fp = d / "iolitech_joined.csv"
        if fp.exists():
            df = pd.read_csv(fp, encoding="utf-8-sig")
            for _, r in df.iterrows():
                for prop, tcol in [("conductivity", "T_conductivity"),
                                   ("viscosity", "T_viscosity"),
                                   ("density", "T_density"),
                                   ("melting_point", None)]:
                    v = r.get(prop)
                    if pd.isna(v):
                        continue
                    try:
                        v = float(v)
                    except Exception:
                        continue
                    if prop == "conductivity" and not (0 < v <= 100):
                        skip += 1
                        continue
                    T = float(r[tcol]) if tcol and not pd.isna(r.get(tcol)) else 298.15
                    total += 1
                    c.execute(
                        """INSERT OR IGNORE INTO records
                           (cat_smiles, an_smiles, b_smiles, mole_frac, T, P,
                            property, value, uncertainty, source)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (str(r["cat_smiles"]).strip(), str(r["an_smiles"]).strip(),
                         "", 1.0, T, 101, prop, v, None, "iolitech"),
                    )
                    if c.execute("SELECT changes()").fetchone()[0]:
                        ins += 1
                    else:
                        dup += 1
        # ILThermo 电导率
        fp2 = d / "ilthermo_matched.csv"
        if fp2.exists():
            df2 = pd.read_csv(fp2, encoding="utf-8-sig")
            for _, r in df2.iterrows():
                v = float(r["value"])
                if not (0 < v <= 100):
                    skip += 1
                    continue
                total += 1
                c.execute(
                    """INSERT OR IGNORE INTO records
                       (cat_smiles, an_smiles, b_smiles, mole_frac, T, P,
                        property, value, uncertainty, source)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (str(r["cat_smiles"]).strip(), str(r["an_smiles"]).strip(),
                     "", 1.0, float(r["T"]), 100, "conductivity", v, None,
                     "ILThermo(iolitech词表匹配)"),
                )
                if c.execute("SELECT changes()").fetchone()[0]:
                    ins += 1
                else:
                    dup += 1
        # ILThermo 自研解析（分子式验证）
        fp3 = d / "ilthermo_resolved.csv"
        if fp3.exists():
            c.execute("DELETE FROM records WHERE source='ILThermo(iolitech词表匹配)'")
            df3 = pd.read_csv(fp3, encoding="utf-8-sig")
            for _, r in df3.iterrows():
                v = float(r["value"])
                if not (0 < v <= 100):
                    skip += 1
                    continue
                total += 1
                c.execute(
                    """INSERT OR IGNORE INTO records
                       (cat_smiles, an_smiles, b_smiles, mole_frac, T, P,
                        property, value, uncertainty, source)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (str(r["cat_smiles"]).strip(), str(r["an_smiles"]).strip(),
                     "", 1.0, float(r["T"]), 100, "conductivity", v, None,
                     "ILThermo(自研解析+分子式验证)"),
                )
                if c.execute("SELECT changes()").fetchone()[0]:
                    ins += 1
                else:
                    dup += 1
        c.commit()
    print(f"OK: 额外来源读取 {total} 行 -> 新入库 {ins}，重复 {dup}，跳过 {skip}")


def cmd_report(_):
    with connect() as c:
        n = c.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        n_il = c.execute("SELECT COUNT(DISTINCT cat_smiles||'/'||an_smiles) FROM records").fetchone()[0]
        n_prop = c.execute("SELECT property, COUNT(*), MIN(value), MAX(value) FROM records GROUP BY property").fetchall()
        by_src = c.execute("SELECT source, COUNT(*) FROM records GROUP BY source").fetchall()
    print(f"总记录: {n} | 独特 IL（阳/阴对）: {n_il}")
    print("按性质:")
    for p in n_prop:
        print(f"  {p['property']}: {p[1]} 行 | 范围 {p[2]} ~ {p[3]}")
    print("按来源:")
    for s in by_src:
        print(f"  {s['source']}: {s[1]}")


def cmd_export(a):
    import pandas as pd
    with connect() as c:
        q = f"SELECT * FROM records WHERE property='{a.property}' AND value>0 AND value<=100"
        n_raw = c.execute(f"SELECT COUNT(*) FROM records WHERE property='{a.property}'").fetchone()[0]
        n_ok = c.execute(
            f"SELECT COUNT(*) FROM records WHERE property='{a.property}' AND value>0 AND value<=100"
        ).fetchone()[0]
        if a.pure:
            q += " AND mole_frac=1"
            n_ok = c.execute(
                f"SELECT COUNT(*) FROM records WHERE property='{a.property}' "
                "AND value>0 AND value<=100 AND mole_frac=1"
            ).fetchone()[0]
        if a.subset == "dense":
            # 稠密子集：至少 50 条记录的 IL
            q += """ AND (cat_smiles||'/'||an_smiles) IN (
                      SELECT cat_smiles||'/'||an_smiles FROM records
                      GROUP BY cat_smiles||'/'||an_smiles HAVING COUNT(*)>=50)"""
        elif a.subset == "sparse":
            q += """ AND (cat_smiles||'/'||an_smiles) IN (
                      SELECT cat_smiles||'/'||an_smiles FROM records
                      GROUP BY cat_smiles||'/'||an_smiles HAVING COUNT(*)<50)"""
        rows = c.execute(q).fetchall()
    print(f"过滤: 总 {n_raw} 行 -> 物理合理（0<κ≤100 S/m）{n_ok} 行")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("smiles_cation,smiles_anion,mole_fraction,temperature,value\n")
        for r in rows:
            f.write(f"{r['cat_smiles']},{r['an_smiles']},{r['mole_frac']},{r['T']},{r['value']}\n")
    print(f"OK: 导出 {len(rows)} 行 -> {out}")


def main():
    ap = argparse.ArgumentParser(description="离子液体性质数据库")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("init")
    p = sub.add_parser("import")
    p.add_argument("--dir", default="workspace\\matmodel\\data")
    p = sub.add_parser("import-extra")
    p.add_argument("--dir", default="workspace\\matmodel\\data")
    sub.add_parser("report")
    p = sub.add_parser("export-csv")
    p.add_argument("--subset", choices=["all", "dense", "sparse"], default="all")
    p.add_argument("--property", default="conductivity")
    p.add_argument("--pure", action="store_true")
    p.add_argument("--out", default="workspace\\matmodel\\data\\il_conductivity.csv")
    a = ap.parse_args()
    if not a.cmd:
        ap.print_help()
        return
    {"init": cmd_init, "import": cmd_import, "import-extra": cmd_import_extra,
     "report": cmd_report,
     "export-csv": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
