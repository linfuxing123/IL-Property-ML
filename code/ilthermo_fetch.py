#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ilthermo_fetch.py - 从 ILThermo v2.0 (NIST) 批量下载多温度属性数据

阶段:
  --list                枚举 4 属性纯组分条目 -> ilt_entries 表
  --fetch -c N          并发下载所有条目 -> ilt_records 表 (单位标准化, 断点续传)
  --merge               拆分/验证 SMILES 并去重统计, 可并入 records

单位标准化:
  T: -> Kelvin (C/F 换算)
  viscosity: -> mPa.s
  density:   -> g/cm3
  conductivity: -> S/m
  melting_point: T 为 NULL (值即熔点温度)

用法:
  python ilthermo_fetch.py --list
  python ilthermo_fetch.py --fetch -c 32
  python ilthermo_fetch.py --merge
"""
import argparse
import html
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ilthermopy as ilt

DB = Path(__file__).resolve().parent / "il_props.db"
_local = threading.local()

# 给 requests 全局加默认超时: ilthermopy 内部裸 requests.get 无 timeout,
# 服务器挂起会让线程无限等待 (2026-08-12 实测 40 并发 19 分钟 0 进度)
import requests as _requests
_REQ_TIMEOUT = 30
_orig_session_request = _requests.sessions.Session.request
def _timed_request(self, *a, **kw):
    kw.setdefault("timeout", _REQ_TIMEOUT)
    return _orig_session_request(self, *a, **kw)
_requests.sessions.Session.request = _timed_request

PROP_KEYS = {
    "viscosity": ("tplC", "Viscosity"),
    "density": ("jBwV", "Density"),
    "conductivity": ("LCor", "Electrical conductivity"),
    "melting_point": ("LPuZ", "Normal melting temperature"),
}


def connect():
    c = sqlite3.connect(DB, timeout=60)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS ilt_entries(
        id TEXT PRIMARY KEY, prop_key TEXT, prop TEXT, cmp1 TEXT, cmp1_id TEXT,
        cmp1_smiles TEXT, npts INTEGER, ref TEXT, status TEXT DEFAULT 'pending',
        fetched_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS ilt_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT, prop TEXT, cat_smiles TEXT, an_smiles TEXT, b_smiles TEXT,
        T REAL, P REAL, value REAL, unit TEXT, ref TEXT, expmeth TEXT,
        UNIQUE(entry_id, prop, T, P, value))""")
    return c


def thread_conn():
    if not hasattr(_local, "c"):
        _local.c = sqlite3.connect(DB, timeout=60)
    return _local.c


def clean_entity(s):
    s = html.unescape(s).replace("\u2022", ".")
    s = re.sub(r"<SUP>(.*?)</SUP>", r"^\1", s, flags=re.I)
    s = re.sub(r"<.*?>", "", s)
    return s.strip()


def unit_from_fullname(fn):
    fn = clean_entity(fn)
    part = fn.split(",", 1)[1] if "," in fn else fn
    part = part.split("=>")[0].strip()
    return part


def parse_entry(e, prop):
    """解析 Entry -> 行列表 [(T_K, P_kPa, value, unit), ...]"""
    hdr = e.header
    tcol = pcol = vcol = None
    for vn, fn in hdr.items():
        base = fn.split(" => ")[0].lower()
        if "melting temperature" in base:
            vcol = vcol or vn
        elif "temperature" in base:
            tcol = tcol or vn
        elif "pressure" in base:
            pcol = pcol or vn
        elif prop == "viscosity" and "viscos" in base:
            vcol = vcol or vn
        elif prop == "density" and "density" in base:
            vcol = vcol or vn
        elif prop == "conductivity" and "conductiv" in base:
            vcol = vcol or vn
    if vcol is None:
        return []
    unit = unit_from_fullname(hdr.get(vcol, ""))
    rows = []
    for _, row in e.data.iterrows():
        val = float(row[vcol])
        # 单位换算 -> 标准
        u = unit.lower()
        if prop == "viscosity":
            if u in ("mpa.s", "mpa", "cp", "cst"):
                pass
            elif "pa" in u or "poise" in u:
                val *= 1000.0
            unit_out = "mPa.s"
        elif prop == "density":
            if ("kg" in u and ("m3" in u or "m-3" in u or "m^3" in u)):
                val /= 1000.0
            unit_out = "g/cm3"
        elif prop == "conductivity":
            if u.startswith("ms") or "m s" in u:
                val *= 0.1
            elif u.startswith("us") or u.startswith("µs") or u.startswith("μs"):
                val *= 1e-4
            elif u.startswith("ks"):
                val *= 1000.0
            unit_out = "S/m"
        else:
            unit_out = unit
        t = None
        if tcol is not None:
            tv = float(row[tcol])
            tu = unit_from_fullname(hdr.get(tcol, "")).lower()
            if "c" in tu or "celsius" in tu:
                tv += 273.15
            elif "f" in tu:
                tv = (tv - 32.0) * 5.0 / 9.0 + 273.15
            t = round(tv, 4)
        p = float(row[pcol]) if pcol is not None else None
        rows.append((t, p, round(float(val), 6), unit_out))
    return rows


def split_il_smiles(smi):
    if not smi:
        return None, None, None
    frags = str(smi).split(".")
    cat = [f for f in frags if "+" in f]
    an = [f for f in frags if "-" in f]
    b = [f for f in frags if "+" not in f and "-" not in f]
    return ".".join(cat) or None, ".".join(an) or None, ".".join(b) or None


def cmd_list(c):
    total = 0
    for prop, (key, _name) in PROP_KEYS.items():
        df = ilt.Search(n_compounds=1, prop_key=key)
        rows = []
        for _, r in df.iterrows():
            rows.append((r["id"], key, prop, r["cmp1"], r["cmp1_id"],
                         r["cmp1_smiles"], int(r["num_data_points"]), r["reference"]))
            total += 1
        c.executemany("""INSERT OR REPLACE INTO ilt_entries
            (id, prop_key, prop, cmp1, cmp1_id, cmp1_smiles, npts, ref, status)
            VALUES (?,?,?,?,?,?,?,?,'pending')""", rows)
        c.commit()
        print(f"[{prop}] 条目 {len(df)} (带SMILES {df['cmp1_smiles'].notna().sum()})")
    print(f"合计条目 {total}")


def fetch_one(eid, prop, c, retries=3):
    c = thread_conn()
    for attempt in range(retries):
        try:
            e = ilt.GetEntry(eid)
            rows = parse_entry(e, prop)
            if not rows:
                c.execute("UPDATE ilt_entries SET status='nodata', fetched_at=? WHERE id=?",
                          (time.strftime("%Y-%m-%d %H:%M:%S"), eid))
                c.commit()
                return eid, "nodata", 0
            smi = next((cp.smiles for cp in e.components if cp.smiles), None)
            if not smi:
                c.execute("UPDATE ilt_entries SET status='nosmiles', fetched_at=? WHERE id=?",
                          (time.strftime("%Y-%m-%d %H:%M:%S"), eid))
                c.commit()
                return eid, "nosmiles", 0
            c.executemany("""INSERT OR IGNORE INTO ilt_records
                (entry_id, prop, cat_smiles, an_smiles, b_smiles, T, P, value, unit, ref, expmeth)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                [(eid, prop, *split_il_smiles(smi),
                  t, p, v, u,
                  (e.ref.full or "")[:300],
                  (e.expmeth or "")[:200])
                 for t, p, v, u in rows])
            c.execute("UPDATE ilt_entries SET status='done', fetched_at=? WHERE id=?",
                      (time.strftime("%Y-%m-%d %H:%M:%S"), eid))
            c.commit()
            return eid, "ok", len(rows)
        except Exception as ex:
            if attempt == retries - 1:
                c.execute("UPDATE ilt_entries SET status='error', fetched_at=? WHERE id=?",
                          (time.strftime("%Y-%m-%d %H:%M:%S"), eid))
                c.commit()
                return eid, f"ERR:{type(ex).__name__}", 0
            time.sleep(1.0 + attempt)
    return eid, "ERR", 0


def cmd_fetch(c, workers, limit=None):
    sql = "SELECT id, prop FROM ilt_entries WHERE status IN ('pending','error')"
    if limit:
        sql += f" LIMIT {int(limit)}"
    todo = c.execute(sql).fetchall()
    done = c.execute("SELECT count(*) FROM ilt_entries WHERE status='done'").fetchone()[0]
    print(f"待抓取 {len(todo)} | 已完成 {done}")
    if not todo:
        print("无待抓取条目")
        return
    t0 = time.time()
    n_ok = n_err = n_nodata = n_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, eid, prop, c): eid for eid, prop in todo}
        for i, f in enumerate(as_completed(futs), 1):
            eid, st, nr = f.result()
            n_rows += nr
            if st == "ok":
                n_ok += 1
            elif st == "nodata":
                n_nodata += 1
            else:
                n_err += 1
            if i % 200 == 0:
                c.commit()
                el = time.time() - t0
                print(f"进度 {i}/{len(todo)} ok={n_ok} err={n_err} rows={n_rows} "
                      f"{(el/60):.1f}min ({i/el:.1f}/s)")
    c.commit()
    el = time.time() - t0
    print(f"完成: ok={n_ok} nodata={n_nodata} err={n_err} 数据点={n_rows} 耗时 {el/60:.1f}min")


def cmd_merge(c):
    """拆分验证 SMILES, 统计并入 records 的规模"""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        print("需要 rdkit")
        return
    rows = c.execute("""SELECT DISTINCT cat_smiles, an_smiles, b_smiles FROM ilt_records
                        WHERE cat_smiles IS NOT NULL AND an_smiles IS NOT NULL""").fetchall()
    canon = {}
    for cat, an, b in rows:
        key = (cat, an, b)
        pair = []
        ok = True
        for s in (cat, an):
            m = Chem.MolFromSmiles(s)
            if m is None:
                ok = False
                break
            pair.append(Chem.MolToSmiles(m))
        if ok:
            canon[key] = (pair[0], pair[1], b)
    print(f"SMILES 验证: 通过 {len(canon)} / {len(rows)}")
    for prop in ["viscosity", "density", "conductivity", "melting_point"]:
        stats = c.execute("""SELECT count(*), count(T), count(DISTINCT entry_id),
            count(DISTINCT cat_smiles||'|'||an_smiles)
            FROM ilt_records WHERE prop=?""", (prop,)).fetchone()
        print(f"[{prop}] 数据点 {stats[0]} | 带T {stats[1]} | 条目 {stats[2]} | 唯一IL {stats[3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("-c", "--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    c = connect()
    if a.list:
        cmd_list(c)
    elif a.fetch:
        cmd_fetch(c, a.concurrency, a.limit)
    elif a.merge:
        cmd_merge(c)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
