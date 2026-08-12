#!/usr/bin/env python
"""
Organic Chemistry Agent — 核心化学工具库
支持：分子建模、描述符计算、波谱辅助、反应分析、数据拟合
依赖：rdkit, numpy, scipy, matplotlib
"""

import sys, os, json, math
from typing import Optional, Tuple, List, Dict
from io import BytesIO
import base64

# ====== RDKit 导入 ======
try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, Draw, rdMolDescriptors,
        MACCSkeys, rdFMCS, Lipinski, Fragments,
        ChemicalFeatures, rdDistGeom, rdForceFieldHelpers
    )
    from rdkit.Chem.Draw import MolsToGridImage
    from rdkit import DataStructs
    RDKIT_OK = True
except ImportError as e:
    RDKIT_OK = False
    print(f"[WARN] RDKit import failed: {e}", file=sys.stderr)
    print("[WARN] Install: pip install rdkit", file=sys.stderr)

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats


# ====== 一、分子建模 ======

def mol_from_smiles(smiles: str, addH: bool = False) -> dict:
    """从 SMILES 创建分子对象，返回基本信息"""
    if not RDKIT_OK: return {"error": "RDKit未安装"}
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        info = {
            "smiles": Chem.MolToSmiles(mol),
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        }
        if addH:
            mol = Chem.AddHs(mol)
        return {"mol": mol, "info": info}
    except Exception as e:
        return {"error": str(e)}


def compute_descriptors(smiles: str) -> dict:
    """计算分子的全套理化性质描述符"""
    result = mol_from_smiles(smiles)
    if "error" in result: return result
    mol = result["mol"]

    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "FractionCsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "Lipinski_HBA": rdMolDescriptors.CalcNumLipinskiHBA(mol),
        "Lipinski_HBD": rdMolDescriptors.CalcNumLipinskiHBD(mol),
        "MR": round(Descriptors.MolMR(mol), 2),  # 摩尔折射率
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        # 药物相似性
        "Lipinski_Violations": Lipinski_Check(mol),
        "Rule_of_3_Pass": Ro3_Check(mol),
    }


def Lipinski_Check(mol) -> int:
    """Lipinski 五规则违规计数"""
    violations = 0
    if Descriptors.MolWt(mol) > 500: violations += 1
    if Descriptors.MolLogP(mol) > 5: violations += 1
    if rdMolDescriptors.CalcNumHBD(mol) > 5: violations += 1
    if rdMolDescriptors.CalcNumHBA(mol) > 10: violations += 1
    return violations


def Ro3_Check(mol) -> bool:
    """Rule of Three (先导化合物规则)"""
    return (Descriptors.MolWt(mol) <= 300 and
            Descriptors.MolLogP(mol) <= 3 and
            rdMolDescriptors.CalcNumHBD(mol) <= 3 and
            rdMolDescriptors.CalcNumHBA(mol) <= 3 and
            rdMolDescriptors.CalcNumRotatableBonds(mol) <= 3)


def generate_3d_conformer(smiles: str, maxAttempts: int = 100) -> dict:
    """生成 3D 构象并优化"""
    result = mol_from_smiles(smiles, addH=True)
    if "error" in result: return result
    mol = result["mol"]

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    status = rdDistGeom.EmbedMultipleConfs(mol, numConfs=1, params=params)

    if status[0] != 0:
        return {"error": "3D embedding failed"}

    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, rdForceFieldHelpers.MMFFGetMoleculeProperties(mol), confId=0)
    if ff is None:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=0)

    if ff:
        ff.Minimize()
        energy = ff.CalcEnergy()
    else:
        energy = None

    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        atom = mol.GetAtomWithIdx(i)
        coords.append({
            "symbol": atom.GetSymbol(),
            "idx": i,
            "x": round(pos.x, 4),
            "y": round(pos.y, 4),
            "z": round(pos.z, 4)
        })

    return {
        "energy": round(energy, 2) if energy else None,
        "num_atoms": mol.GetNumAtoms(),
        "coordinates": coords[:mol.GetNumAtoms() - mol.GetNumAtoms() // 2],  # heavy atoms only
    }


def draw_molecule(smiles: str, output_path: str = "molecule.png", size: tuple = (600, 400),
                  show_atom_idx: bool = False, kekulize: bool = True) -> str:
    """绘制分子结构图并保存"""
    result = mol_from_smiles(smiles)
    if "error" in result: return f"Error: {result['error']}"
    mol = result["mol"]

    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size, kekulize=kekulize,
                          highlightAtoms=range(mol.GetNumAtoms()) if show_atom_idx else [])
    img.save(output_path)
    return f"Saved: {output_path} ({mol.GetNumAtoms()} atoms)"


def molecular_similarity(smiles1: str, smiles2: str, method: str = "maccs") -> dict:
    """计算两个分子间的相似度"""
    r1 = mol_from_smiles(smiles1)
    r2 = mol_from_smiles(smiles2)
    if "error" in r1: return r1
    if "error" in r2: return r2

    if method == "maccs":
        fp1 = MACCSkeys.GenMACCSKeys(r1["mol"])
        fp2 = MACCSkeys.GenMACCSKeys(r2["mol"])
    else:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(r1["mol"], 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(r2["mol"], 2, nBits=2048)

    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
    dice = DataStructs.DiceSimilarity(fp1, fp2)

    return {
        "method": method,
        "tanimoto": round(tanimoto, 4),
        "dice": round(dice, 4),
    }


def find_common_substructure(smiles_list: list) -> dict:
    """找到一系列分子的最大公共子结构"""
    mols = []
    for s in smiles_list:
        r = mol_from_smiles(s)
        if "error" in r: return r
        mols.append(r["mol"])

    mcs = rdFMCS.FindMCS(mols, timeout=10)
    return {
        "smarts": mcs.smartsString,
        "num_atoms": mcs.numAtoms,
        "num_bonds": mcs.numBonds,
        "smiles": Chem.MolToSmiles(Chem.MolFromSmarts(mcs.smartsString)) if mcs.smartsString else None,
    }


# ====== 二、官能团分析 ======

FUNCTIONAL_GROUPS = {
    "羧酸": "[CX3](=[OX1])[OX2H1]",
    "酯": "[CX3](=[OX1])[OX2][CX4]",
    "酰胺": "[CX3](=[OX1])[NX3]",
    "伯胺": "[NX3;H2;!$(NC=O)]",
    "仲胺": "[NX3;H1;!$(NC=O)]",
    "叔胺": "[NX3;H0;!$(NC=O)]",
    "羟基": "[OX2H]",
    "酚羟基": "[OX2H][c]",
    "醛": "[CX3H1](=O)[#6]",
    "酮": "[CX3](=O)[#6][#6]",
    "醚": "[OD2]([#6])[#6]",
    "腈基": "[CX2]#N",
    "硝基": "[NX3](=O)=O",
    "磺酸": "[SX4](=O)(=O)[OX2H]",
    "卤代(F)": "[F]",
    "卤代(Cl)": "[Cl]",
    "卤代(Br)": "[Br]",
    "卤代(I)": "[I]",
    "烯烃": "[CX3]=[CX3]",
    "炔烃": "[CX2]#[CX2]",
    "苯环": "c1ccccc1",
    "杂芳环": "c1[#7,#8,#16]ccc1",
    "环氧": "[OX2r3]1[#6r3][#6r3]1",
    "硫醇": "[SX2H]",
    "亚胺": "[CX3]=[NX2]",
    "酰氯": "[CX3](=[OX1])[Cl]",
    "酸酐": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "磷酰基": "[PX4](=O)",
}


def analyze_functional_groups(smiles: str) -> dict:
    """分析分子的官能团组成"""
    result = mol_from_smiles(smiles)
    if "error" in result: return result
    mol = result["mol"]

    found = {}
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            found[name] = len(matches)

    return {
        "smiles": result["info"]["canonical_smiles"],
        "functional_groups": found,
        "total_fg_count": sum(found.values()),
        "has_bioisostere_candidates": "羧酸" in found or "酯" in found or "酰胺" in found,
    }


# ====== 三、化学反应分析 ======

def assess_reaction_feasibility(smiles_reactants: list, smiles_product: str) -> dict:
    """
    评估反应的化学可行性
    检查：原子守恒、官能团变化、氧化态变化
    """
    try:
        reactants = [mol_from_smiles(s) for s in smiles_reactants]
        product = mol_from_smiles(smiles_product)

        errors = [r.get("error") for r in reactants if "error" in r]
        if errors: return {"error": errors}
        if "error" in product: return {"error": product["error"]}

        # 原子守恒检查
        from collections import Counter
        reactant_atoms = Counter()
        for r in reactants:
            for atom in r["mol"].GetAtoms():
                reactant_atoms[atom.GetAtomicNum()] += 1

        product_atoms = Counter()
        for atom in product["mol"].GetAtoms():
            product_atoms[atom.GetAtomicNum()] += 1

        # 计算差值
        all_atoms = set(reactant_atoms.keys()) | set(product_atoms.keys())
        balance = {}
        for z in all_atoms:
            balance[z] = reactant_atoms.get(z, 0) - product_atoms.get(z, 0)

        atom_conserved = all(v == 0 for v in balance.values())

        # 官能团分析
        rxn_fgs = {}
        for i, r in enumerate(reactants):
            if "mol" in r:
                rxn_fgs[f"reactant_{i+1}"] = analyze_functional_groups(Chem.MolToSmiles(r["mol"]))

        product_fgs = analyze_functional_groups(Chem.MolToSmiles(product["mol"]))

        return {
            "atom_conserved": atom_conserved,
            "atom_balance": {Chem.Atom.GetElementSymbol(z): b for z, b in balance.items() if b != 0},
            "reactant_fgs": rxn_fgs,
            "product_fgs": product_fgs,
            "verdict": "✅ 原子守恒" if atom_conserved else "❌ 原子不守恒，缺少副产物或试剂"
        }
    except Exception as e:
        return {"error": str(e)}


# ====== 四、数据拟合与动力学 ======

def fit_kinetics(t: list, C: list, model: str = "auto") -> dict:
    """反应动力学数据拟合（零级/一级/二级）"""
    t = np.array(t, dtype=float)
    C = np.array(C, dtype=float)
    C0 = C[0]

    results = {}

    for order, label in [(0, "零级"), (1, "一级"), (2, "二级")]:
        try:
            if order == 0:
                y = C - C0
                slope, intercept, r, p, std = stats.linregress(t, y)
                k = -slope
                C_pred = C0 - k * t
            elif order == 1:
                y = np.log(C / C0)
                slope, intercept, r, p, std = stats.linregress(t, y)
                k = -slope
                C_pred = C0 * np.exp(-k * t)
            else:
                y = 1/C - 1/C0
                slope, intercept, r, p, std = stats.linregress(t, y)
                k = slope
                C_pred = 1 / (1/C0 + k * t)

            r2 = r**2
            rmse = np.sqrt(np.mean((C - C_pred)**2))
            results[label] = {"k": round(k, 6), "r2": round(r2, 4), "rmse": round(rmse, 6)}
        except Exception as e:
            results[label] = {"error": str(e)}

    # 自动选最优
    best = max(
        [(k, v) for k, v in results.items() if "r2" in v],
        key=lambda x: x[1]["r2"], default=(None, None)
    )

    return {
        "C0": C0,
        "models": results,
        "best_fit": best[0],
        "best_params": best[1],
    }


def arrhenius_fit(T: list, k: list) -> dict:
    """Arrhenius 方程拟合 k = A * exp(-Ea/RT)"""
    T = np.array(T, dtype=float)
    k = np.array(k, dtype=float)

    # ln(k) = ln(A) - Ea/(R*T)
    y = np.log(k)
    x = 1 / (8.314 * T)

    slope, intercept, r, p, std = stats.linregress(x, y)

    Ea = -slope / 1000  # kJ/mol
    A = np.exp(intercept)
    r2 = r**2

    return {
        "Ea_kJ_per_mol": round(Ea, 2),
        "A_pre_exponential": f"{A:.2e}",
        "r2": round(r2, 4),
        "equation": f"k = {A:.2e} * exp(-{Ea:.1f} kJ/mol / RT)"
    }


# ====== 五、化工计算 ======

def mass_balance(species: dict, stoichiometry: dict, conversion: float) -> dict:
    """
    反应物料衡算
    species: {"C2H4": 100.0, "H2O": 150.0}  # mol/h 进料
    stoichiometry: {"C2H4": -1, "H2O": -1, "C2H5OH": 1}
    conversion: 0.85  # 85%转化率（基于关键组分）
    """
    key_reactant = next(k for k, v in stoichiometry.items() if v < 0)
    reacted = species.get(key_reactant, 0) * conversion * abs(stoichiometry[key_reactant])

    output = {}
    for sp, coeff in stoichiometry.items():
        output[sp] = species.get(sp, 0) + coeff * reacted

    for sp, amount in species.items():
        if sp not in output:
            output[sp] = amount

    return {
        "input": species,
        "output": output,
        "conversion": conversion,
        "key_reactant_consumed": round(reacted, 3),
        "atom_economy": round(100 * sum(v for v in stoichiometry.values() if v > 0) /
                              sum(abs(v) for v in stoichiometry.values() if v < 0), 1)
    }


def distillation_estimate(alpha: float, xf: float, xd: float, xb: float, R_Rmin_ratio: float = 1.3) -> dict:
    """
    精馏塔估算 (Fenske-Underwood-Gilliland)
    alpha: 相对挥发度
    xf, xd, xb: 进料/塔顶/塔底 轻组分摩尔分数
    """
    # Fenske: 最小理论板数
    import math as m
    Nmin = m.log((xd / (1-xd)) * ((1-xb) / xb)) / m.log(alpha)

    # Underwood: 最小回流比
    Rmin = (xd - xf * alpha / (1 + (alpha-1)*xf)) / (xf * alpha / (1 + (alpha-1)*xf))

    # Gilliland 关联（简化）
    R = Rmin * R_Rmin_ratio
    X = (R - Rmin) / (R + 1)
    Y = 1 - m.exp((1 + 54.4*X) / (11 + 117.2*X) * (X-1) / X**0.5) if X > 0 else 0
    N = (Nmin + Y) / (1 - Y)

    return {
        "Nmin_Fenske": round(Nmin, 1),
        "Rmin_Underwood": round(Rmin, 3),
        "R_operation": round(R, 3),
        "N_theoretical": round(N, 0),
        "RR_ratio": R_Rmin_ratio
    }


# ====== 六、安全评估 ======

def reaction_safety_assessment(delta_H: float, Cp: float, mass_total: float,
                                T_initial: float, T_decomp: float) -> dict:
    """
    反应安全评估
    delta_H: 反应热 (kJ/mol)
    Cp: 比热容 (J/g·K)
    mass_total: 总质量 (g)
    T_initial: 起始温度 (K)
    T_decomp: 分解温度 (K)
    """
    # 绝热温升
    dT_adiabatic = (delta_H * 1000) / (Cp * mass_total)

    # MTSR (最大合成反应温度)
    MTSR = T_initial + dT_adiabatic

    return {
        "dT_adiabatic_K": round(dT_adiabatic, 1),
        "MTSR_K": round(MTSR, 1),
        "T_decomp_gap_K": round(T_decomp - MTSR, 1),
        "risk_level": "HIGH-RISK" if MTSR > T_decomp else
                      "MEDIUM" if MTSR > T_decomp - 20 else
                      "SAFE",
        "warning": "MTSR超过分解温度！可能引发热失控！" if MTSR > T_decomp else None
    }


def green_chemistry_metrics(product_mass: float, total_input_mass: float,
                              stoichiometric_product: float, total_solvent: float) -> dict:
    """绿色化学指标计算"""
    E_factor = (total_input_mass - product_mass) / product_mass
    atom_economy = 100 * stoichiometric_product / total_input_mass
    PMI = total_input_mass / product_mass

    return {
        "E_factor": round(E_factor, 2),
        "atom_economy_pct": round(atom_economy, 1),
        "PMI": round(PMI, 2),
        "E_factor_grade": "优秀" if E_factor < 5 else "一般" if E_factor < 25 else "差",
    }


# ====== 主入口（命令行测试用） ======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python chem_tools.py <func> <args>")
        print("函数: descriptors, draw, fg_analysis, similarity, kinetics, mass_balance")
        sys.exit(1)

    func = sys.argv[1]
    args = sys.argv[2:]

    if func == "descriptors" and args:
        print(json.dumps(compute_descriptors(args[0]), indent=2, ensure_ascii=False))
    elif func == "draw" and len(args) >= 1:
        out = args[1] if len(args) > 1 else "molecule.png"
        print(draw_molecule(args[0], out))
    elif func == "fg_analysis" and args:
        print(json.dumps(analyze_functional_groups(args[0]), indent=2, ensure_ascii=False))
    elif func == "similarity" and len(args) >= 2:
        print(json.dumps(molecular_similarity(args[0], args[1]), indent=2))
    elif func == "kinetics" and args:
        data = json.loads(args[0])
        print(json.dumps(fit_kinetics(data["t"], data["C"]), indent=2))
    elif func == "mass_balance" and args:
        data = json.loads(args[0])
        print(json.dumps(mass_balance(data["species"], data["stoichiometry"], data["conversion"]), indent=2))
    else:
        print(f"未知函数: {func}")
