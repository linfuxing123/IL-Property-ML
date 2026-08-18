#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finalize_refs.py — 参考表定稿：保留 1-38，追加 8 条选定文献（39-46）

1) 解析当前 References 条目
2) 保留 1..38（已核验语义）
3) 追加：39 Kapoor, 40 Tropsha, 41 Cawley, 42 Wallach, 43 Yu, 44 Li, 45 Himanen, 46 MoleculeNet
4) 回写
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "manuscript_ces.md"
REFS_MARK = "## References"
text = SRC.read_text(encoding="utf-8")
body, refs = text.split(REFS_MARK, 1)

entries = {}
for m in re.finditer(r"^(\d+)\.\s+(.*)$", refs, re.M):
    entries[int(m.group(1))] = m.group(2).rstrip()

new_entries = {n: entries[n] for n in range(1, 39) if n in entries}
addition = {
    39: "S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4 (2023) 100804.",
    40: "A. Tropsha, P. Gramatica, V. K. Gombar, The importance of being earnest: Validation is the absolute essential for successful application and interpretation of QSPR models, QSAR Comb. Sci. 22 (2003) 69–77.",
    41: "G. C. Cawley, N. L. C. Talbot, On over-fitting in model selection and subsequent selection bias in performance evaluation, J. Mach. Learn. Res. 11 (2010) 2079–2107.",
    42: "I. Wallach, A. Heifets, Most ligand-based classification benchmarks reward memorization rather than generalization, J. Chem. Inf. Model. 58 (2018) 916–932.",
    43: "X. Yu, End-to-end deep learning models for predicting the electrical conductivity of ionic liquids, ACS Sustain. Chem. Eng. (2026), doi:10.1021/acssuschemeng.6c07089.",
    44: "R. Li, et al., Machine learning-enhanced QSPR model for predicting the viscosity of ionic liquids, Chem. Eng. Sci. 321 (2025) 122992, doi:10.1016/j.ces.2025.122992.",
    45: "L. Himanen, A. Geurts, A. S. Foster, P. Rinke, Data-driven materials science: Status, challenges, and perspectives, Adv. Sci. 6 (2019) 1900808.",
    46: "Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande, MoleculeNet: A benchmark for molecular machine learning, Chem. Sci. 9 (2018) 513–530.",
}
new_entries.update(addition)

ref_block = "\n".join(f"{n}. {new_entries[n]}" for n in sorted(new_entries))
out = body + REFS_MARK + "\n\n" + ref_block + "\n"
SRC.write_text(out, encoding="utf-8")
print(f"reference entries now: {len(new_entries)} (1-38 kept + 39-46 added)")
