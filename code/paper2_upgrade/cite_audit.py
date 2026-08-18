#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cite_audit.py — 引用编号审计：按首次出现顺序重排参考文献（paper2_upgrade）

规则（引用编号审计铁律）：
1) 正文摘出 (n)/(n–m)/(n,m)/(n–m,o) 形式的数字引用组，按首次出现顺序编号
   - 跳过表格行（以 | 开头）与所有数字 > 54 的组（非引用，如样本量 (619)）
2) 参考文献条目按首次出现顺序重排并重编号
3) 原位回写 manuscript_ces.md；输出审计报告
用法: python cite_audit.py
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "manuscript_ces.md"
REFS_MARK = "## References"
MAX_REF = 56  # 已知参考表条目数上限

text = SRC.read_text(encoding="utf-8")
body, refs = text.split(REFS_MARK, 1)

cit_pat = re.compile(r"\((\d+(?:\s*(?:–|-)\s*\d+)?(?:\s*,\s*\d+(?:\s*(?:–|-)\s*\d+)?)*)\)")


def parse(s):
    nums = []
    for part in s.split(","):
        part = part.strip()
        if "–" in part or "-" in part:
            lo, hi = re.split(r"\s*(?:–|-)\s*", part)
            nums.extend(range(int(lo), int(hi) + 1))
        else:
            nums.append(int(part))
    return nums


groups = []  # (start_pos_in_body, nums)
for m in cit_pat.finditer(body):
    nums = parse(m.group(1))
    if not nums or max(nums) > MAX_REF:
        continue  # 非引用（样本量/记录数等）
    line_start = body.rfind("\n", 0, m.start()) + 1
    line = body[line_start:body.find("\n", m.start())]
    if line.strip().startswith("|"):
        continue  # 表格行
    groups.append((m.start(), nums))

order, seen = [], set()
for _, nums in groups:
    for n in nums:
        if n not in seen:
            seen.add(n)
            order.append(n)
print(f"citation groups: {len(groups)}; unique refs: {len(seen)}; max: {max(order)}")

# 参考文献条目
ref_pat = re.compile(r"^(\d+)\.\s+(.*)$", re.M)
entries = {int(m.group(1)): m.group(2).rstrip() for m in ref_pat.finditer(refs)}
missing = [n for n in order if n not in entries]
if missing:
    print(f"参考表缺条目: {missing}")
    sys.exit(1)
unused = sorted(set(entries) - set(order))
print(f"未被引用的条目(保留原编号,追加尾部): {unused}")

old_to_new = {old: i + 1 for i, old in enumerate(order)}

# 正文引用组替换（逐 token 处理，同串多出现用 replace(...,1) 顺序消费）
new_body = body
for m in cit_pat.finditer(body):
    nums = parse(m.group(1))
    if not nums or max(nums) > MAX_REF:
        continue
    line_start = body.rfind("\n", 0, m.start()) + 1
    line = body[line_start:body.find("\n", m.start())]
    if line.strip().startswith("|"):
        continue
    new_nums = sorted(old_to_new[n] for n in nums)

    def fmt(ns):
        out, i = [], 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                j += 1
            out.append(f"{ns[i]}-{ns[j]}" if j > i else str(ns[i]))
            i = j + 1
        return ",".join(out)

    new_body = new_body.replace(m.group(0), "(" + fmt(new_nums) + ")", 1)

# 参考表重排 + 尾部补未引用条目（若引用组遗漏了它们，保序追加）
ordered_items = [(old_to_new[old], entries[old]) for old in order]
next_num = len(order) + 1
for old in unused:
    ordered_items.append((next_num, entries[old]))
    next_num += 1
new_refs = "\n".join(f"{n}. {e}" for n, e in ordered_items)

out = new_body + REFS_MARK + "\n\n" + new_refs + "\n"
SRC.write_text(out, encoding="utf-8")
print("--- old->new 映射 ---")
print(old_to_new)
print("--- 审计通过，已回写 ---")
