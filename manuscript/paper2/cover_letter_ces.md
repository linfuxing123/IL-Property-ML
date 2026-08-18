# Cover letter — Chemical Engineering Science（定稿版）

> 2026-08-18 · 目标刊：Chemical Engineering Science (Elsevier, 混合刊, 订阅路线免费)
> 三句话卖点：law → tool → benchmark

---

## Cover letter

**Submission: "Scaling laws of ionic-liquid property prediction: data-density laws, leakage taxes, and optimal measurement allocation"**

Dear Editor,

We are pleased to submit our manuscript for consideration as a Research Article in *Chemical Engineering Science*.

Ionic liquids are a central platform in chemical-process engineering—as electrolytes, separation solvents, and reaction media—and engineering design requires thermophysical property data across a cation–anion space of roughly 10⁶ combinations, of which only a few thousand have been measured. Process groups must therefore decide how to spend a limited measurement budget. This manuscript converts that practical decision into quantitative laws and an actionable tool.

1. **A law.** From 86,008 curated NIST ILThermo records (1,891 ILs) we establish the first IL-specific learning-curve laws R²(N) = a − b·N^(−γ) under strict IL-disjoint validation, with property-specific exponents (γ = 0.215 viscosity, 0.807 conductivity, 0.565 density, 0.288 melting point). The laws classify each property as data-limited (viscosity: ≈5,700 ILs for R² ≈ 0.80), representation-limited (conductivity saturates at R² ≈ 0.73), or diversity-limited (melting point) — a decision rule for where measurement, feature, and library-expansion budgets belong. This directly addresses CES's interest in property prediction and process-design data (cf. the recent IL-viscosity ML QSPR in *Chem. Eng. Sci.* 321 (2025) 122992, cited in our manuscript).
2. **A tool.** We quantify the "leakage tax" of random point-wise splits (up to ΔR² = +0.55 in redundancy-rich strata), decompose cold-start errors to identify the anion dimension as the data-starved axis for viscosity, and show by simulated measurement campaigns that coverage-guided acquisition is never worse than random (fixed-eval protocol: +0.05 R² at 150 measurements). A prioritized top-100 list of novel ILs to measure next—drawn from 8.3 M virtual ion pairs—accompanies the paper.
3. **A benchmark.** The dataset, split files, a multi-model leaderboard on identical folds, and one-command reproducibility are released openly (GitHub release v2.2.1; Zenodo DOI 10.5281/zenodo.21997263), so every number is auditable and extensible.

We believe the combination of quantitative learning-curve laws, honest evaluation methodology, and an engineering-relevant measurement-allocation tool fits *Chemical Engineering Science*'s readership. The manuscript has not been published or submitted elsewhere; there are no competing interests.

Sincerely,

Fuxing Lin
Hunan Institute of Engineering, Xiangtan, Hunan, China
Email: 3612411485@qq.com · ORCID: 0009-0003-7588-6942

---

## 投稿附注（内部用，不随稿）

- **稿件**：manuscript_ces.docx（Elsevier 初始投稿不强制模板；Editorial Manager 上传 main manuscript + highlights 字段 + cover letter 分开填）
- **Highlights**：已在稿件内（5 条）；EM 提交时如有独立字段可复制
- **建议审稿人**（可留空或给 2-3 位）：可建议 IL 热物性/ML 方向学者（如 Makarov、Paduszyński 等的相关作者），谨慎起见投稿时再定
- **费用**：CES 为混合刊，默认订阅路线不收 APC；不要勾选 OA 选项
- **账号**：Elsevier Editorial Manager 需注册（QQ 邮箱可用）；如该邮箱已注册过 Wiley/ACS 无冲突，EM 独立注册
- **冲突检查**：与在投其他稿件（JCED 第 1 篇、DD 第 7 篇等）内容无重叠，封面信已声明"not published or submitted elsewhere"
