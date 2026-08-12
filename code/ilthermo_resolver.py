#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ilthermo_resolver.py — ILThermo 离子名 → SMILES 解析器（含分子式验证）

思路：
  1. 常见阴离子名字→SMILES 映射表（含拼写变体）
  2. 阳离子按家族正则解析（咪唑/吡啶/吡咯烷/铵/膦/哌啶 + 烷基链）
  3. 生成 IL SMILES = 阳离子.阴离子
  4. RDKit 核对分子式 == ILThermo Formula 才接受（验证铁律）

用法:
  python ilthermo_resolver.py --out workspace\\matmodel\\data\\ilthermo_resolved.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError:
    print("需要 rdkit")
    sys.exit(1)

ANIONS = {
    "bis(trifluoromethylsulfonyl)imide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "bis(trifluoromethanesulfonyl)imide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "bis((trifluoromethyl)sulfonyl)amide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "bis((trifluoromethyl)sulfonyl)imide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "bis(trifluoromethylsulfonyl)imide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "hexafluorophosphate": "F[P-](F)(F)(F)(F)F",
    "tetrafluoroborate": "F[B-](F)(F)F",
    "dicyanamide": "N#C[N-]C#N",
    "tricyanomethanide": "N#CC([C-]#N)C#N",
    "thiocyanate": "[S-]C#N",
    "iodide": "[I-]",
    "bromide": "[Br-]",
    "chloride": "[Cl-]",
    "nitrate": "[O-][N+](=O)[O-]",
    "tetrafluoroborate": "F[B-](F)(F)F",
    "methylsulfate": "COS(=O)(=O)[O-]",
    "ethylsulfate": "CCOS(=O)(=O)[O-]",
    "octylsulfate": "CCCCCCCCOS(=O)(=O)[O-]",
    "trifluoromethanesulfonate": "C(F)(F)(F)S(=O)(=O)[O-]",
    "triflate": "C(F)(F)(F)S(=O)(=O)[O-]",
    "methanesulfonate": "CS(=O)(=O)[O-]",
    "mesylate": "CS(=O)(=O)[O-]",
    "4-methylbenzenesulfonate": "Cc1ccc(S(=O)(=O)[O-])cc1",
    "tosylate": "Cc1ccc(S(=O)(=O)[O-])cc1",
    "p-toluenesulfonate": "Cc1ccc(S(=O)(=O)[O-])cc1",
    "bis(trifluoromethylsulfonyl)amide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "trifluoroacetate": "O=C([O-])C(F)(F)F",
    "acetate": "CC([O-])=O",
    "formate": "[O-]C=O",
    "lactate": "CC(O)C([O-])=O",
    "sulfurochloridate": "O=S(=O)([O-])Cl",
    "perchlorate": "[O-]Cl(=O)(=O)=O",
    "dihydrogenphosphate": "OP(=O)([O-])O",
    "hexafluoroarsenate": "F[As-](F)(F)(F)(F)F",
    "tetracyanoborate": "N#CB([C-]#N)(C#N)C#N",
    "salicylate": "O=C([O-])c1ccccc1O",
    "trifluoromethylsulfonylimide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "sulfate": "[O-]S(=O)(=O)[O-]",
    "hydrogen sulfate": "OS(=O)(=O)[O-]",
    "hydrogensulfate": "OS(=O)(=O)[O-]",
    "carbonate": "[O-]C(=O)[O-]",
    "bicarbonate": "OC(=O)[O-]",
}

ALKYL = {
    "methyl": "C", "ethyl": "CC", "propyl": "CCC", "butyl": "CCCC",
    "pentyl": "CCCCC", "hexyl": "CCCCCC", "heptyl": "CCCCCCC",
    "octyl": "CCCCCCCC", "nonyl": "CCCCCCCCC", "decyl": "CCCCCCCCCC",
    "dodecyl": "CCCCCCCCCCCC", "tetradecyl": "CCCCCCCCCCCCCC",
    "hexadecyl": "CCCCCCCCCCCCCCCC", "octadecyl": "CCCCCCCCCCCCCCCCCC",
}
ALKYL_RE = re.compile(
    r"(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl|"
    r"dodecyl|tetradecyl|hexadecyl|octadecyl)")


def _chains(label):
    """提取 label 中全部烷基链 SMILES（按出现顺序）。"""
    return [ALKYL[m.group(0)] for m in ALKYL_RE.finditer(label)]


def cation_smiles(label):
    s = label.replace("(", "").replace(")", "")
    chains = _chains(s)
    if not chains:
        return None
    # 咪唑鎓：1-A-3-Bimidazolium / 1-A-3-methylimidazolium / 3-A-1H-imidazol-3-ium
    if "imidazol" in s:
        c1, c3 = chains[0], (chains[1] if len(chains) > 1 else "C")
        return f"{c1}[n+]1ccn({c3})c1"
    if "pyrrolidin" in s:  # 吡咯烷鎓：1-A-1-Bpyrrolidinium
        c1, c2 = chains[0], (chains[1] if len(chains) > 1 else "C")
        return f"{c1}[N+]1({c2})CCCC1"
    if "piperidin" in s:
        c1, c2 = chains[0], (chains[1] if len(chains) > 1 else "C")
        return f"{c1}[N+]1({c2})CCCCC1"
    if "pyridin" in s:  # 吡啶鎓
        return f"{chains[0]}[n+]1ccccc1"
    if "ammonium" in s:
        n = 4 if s.startswith("tetra") else (3 if s.startswith("tri") else
             (2 if s.startswith("di") else 1))
        cs = chains[:n]
        while len(cs) < n:
            cs.append("C")
        if n == 4:
            return f"[N+]({cs[0]})({cs[1]})({cs[2]})({cs[3]})"
        if n == 3:
            return f"[NH+]({cs[0]})({cs[1]})({cs[2]})"
        if n == 2:
            return f"[NH2+]({cs[0]})({cs[1]})"
        return f"[NH3+]{cs[0]}"
    if "phosphonium" in s:
        n = 4 if s.startswith("tetra") else 3
        cs = chains[:n]
        while len(cs) < n:
            cs.append("C")
        return f"[P+]({cs[0]})({cs[1]})({cs[2]})({cs[3] if n == 4 else 'C'})"
    return None


def resolve(label):
    s = label.lower().strip()
    for an, smi in sorted(ANIONS.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(an):
            cat_part = s[: -len(an)].strip(" -")
            cs = cation_smiles(cat_part)
            if cs:
                return cs, smi
    return None, None


def formula_ok(smiles, target_formula):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    f = rdMolDescriptors.CalcMolFormula(mol)
    return f.replace(" ", "") == str(target_formula).replace(" ", "").replace("·", ".").split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace\\matmodel\\data\\ilthermo_resolved.csv")
    ap.add_argument("--src", default="workspace\\matmodel\\data\\ionic_liquid_ILthermo_conductivcity.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.src, encoding="utf-8-sig")
    rows = []
    stat = {"resolved": 0, "formula_ok": 0, "fail": 0}
    for _, r in df.iterrows():
        label = str(r["Label"])
        cs, an = resolve(label)
        if not cs:
            stat["fail"] += 1
            continue
        stat["resolved"] += 1
        if not formula_ok(f"{cs}.{an}", r["Formula"]):
            stat["fail"] += 1
            continue
        stat["formula_ok"] += 1
        import re as _re
        mm = _re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(r["Electrical conductivity, S/m"]))
        if not mm:
            continue
        rows.append({"cat_smiles": cs, "an_smiles": an, "T": float(r["Temperature, K"]),
                     "value": float(mm.group(0)), "label": label, "formula": r["Formula"]})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"解析: 尝试 {stat['resolved'] + stat['fail']} 行 | 解析出阳离子 {stat['resolved']} | "
          f"分子式验证通过 {stat['formula_ok']} | 唯一 IL {out.groupby(['cat_smiles','an_smiles']).ngroups}")
    print(f"输出 {len(out)} 行 -> {a.out}")


if __name__ == "__main__":
    main()
