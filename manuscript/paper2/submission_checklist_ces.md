# CES 投稿检查清单 + 元数据包（Editorial Manager 全字段就绪）

> 2026-08-18 · 目标刊 Chemical Engineering Science（Elsevier，混合刊，订阅路线免费）
> 状态：稿件/封面信/数据/代码全部就绪，只差 EM 账号 + 提交动作

## 0. 投稿动作检查（✓ = 已就绪）

- [x] 稿件：`manuscript_ces.docx`（20 页预览 PDF 已验证：标题页/表格/参考文献/图题渲染正常）
- [x] 封面信：`cover_letter_ces.md`（law→tool→benchmark 三句话）
- [x] Highlights：5 条，全部 ≤85 字符（Elsevier 硬性要求）
- [x] 摘要：248 词（≤250 合规）
- [x] 关键词：6 个
- [x] 数据/代码：GitHub v2.2.1 + Zenodo 10.5281/zenodo.21997263（已在线验证 published）
- [x] 图：6 张 300 dpi 内嵌；表：5 张
- [x] 声明：无利益冲突/无资助/CRediT 单作者
- [ ] **Editorial Manager 账号注册（QQ 邮箱，需邮箱验证码——用户操作）**
- [ ] **最终提交（用户确认后执行）**

## 1. EM 表单字段（直接复制粘贴）

**Article Type:** Research Article

**Title:**
Scaling laws of ionic-liquid property prediction: data-density laws, leakage taxes, and optimal measurement allocation

**Running head:** Scaling laws of IL property prediction

**Abstract（248 词）:**
Machine-learning surrogate models for ionic-liquid (IL) properties promise high-throughput electrolyte and solvent design, but their accuracy for unseen ion pairs is widely assumed to be limited by data scarcity rather than by models. Here we convert that qualitative assumption into quantitative laws. From a curated NIST ILThermo dataset of 86,008 records spanning 1,891 unique ILs with standardized units and verified SMILES, we measure how group-level accuracy R² scales with the number of ILs N under strict IL-disjoint 5-fold cross-validation for viscosity, electrical conductivity, density, and melting point. All four properties follow the learning-curve law R²(N) = a − b·N^(−γ) with property-specific exponents γ = 0.215 (viscosity), 0.807 (conductivity), 0.565 (density), and 0.288 (melting point). Viscosity—the most data-hungry property—reaches R² = 0.74 at 1,165 ILs and requires ≈5,700 ILs to approach R² = 0.80; conductivity instead saturates near R² = 0.73, revealing a representation ceiling rather than a data ceiling. We further quantify a "leakage tax": random point-wise splits inflate R² by 0.09–0.22 at full scale and by up to +0.55 in redundancy-rich strata, so evaluation discipline must be tied to the split protocol, not the dataset size. Simulated measurement campaigns over held-out ILs show that coverage-guided acquisition consistently outperforms random sampling, while uncertainty-based acquisition does not—chemical coverage, not predictive disagreement, is the binding constraint. Cold-start decomposition attributes the viscosity deficit primarily to the anion dimension. The dataset, split files, multi-model leaderboard, and a prioritized list of 100 novel ILs (from 8.3 M virtual ion pairs) are released openly.

**Highlights（5 条，≤85 字符/条）:**
- IL learning-curve laws R²(N) = a − b·N^(−γ) with property-specific exponents.
- Properties are data-, representation-, or diversity-limited: a budget decision rule.
- Leakage tax: random splits inflate R² by up to +0.55 with per-IL redundancy.
- Coverage-guided acquisition beats random; new anions are the viscosity bottleneck.
- Open release: dataset, split files, leaderboard, and top-100 measurement list.

**Keywords:** ionic liquids; learning curves; data density; group-disjoint validation; data acquisition; ILThermo

**Author:** Fuxing Lin（通讯作者）
- Affiliation: Hunan Institute of Engineering, Xiangtan, Hunan, China
- Email: 3612411485@qq.com
- ORCID: 0009-0003-7588-6942

**Declarations（EM 勾选）:**
- 无利益冲突（No competing interests）✅
- 无资助声明（No funding）✅
- 原创性/未一稿多投声明 ✅
- 数据可用性：GitHub + Zenodo（稿件 §5 已含）

## 2. 文件上传清单（EM "Attach Files"）

| 顺序 | 文件 | 类型 |
|---|---|---|
| 1 | manuscript_ces.docx | Manuscript |
| 2 | cover_letter_ces.md → 转 .txt 或直接粘贴到 "Cover Letter" 字段 | Cover Letter |
| 3 | （可选）fig 源 PNG × 6 | Figure |

## 3. 建议审稿人（投稿时选填，谨慎推荐）

可建议（均为 IL 性质 ML/热物性方向活跃学者，非熟识）：
1. Dmitry M. Makarov（Ivanovo State University，IL 性质 ML benchmark 作者——Makarov et al., J. Mol. Liq. 351 (2022) 118616）
2. Kamil Paduszyński（Adam Mickiewicz University，IL GCM/QSPR 粘度 作者——Paduszyński, IECR 58 (2019) 17049）
3. （备选）Ryo Yoshida（ISM，材料数据库标度律——Minami et al., npj Comput. Mater. 11 (2025) 146）

> 注意：建议审稿人需与作者无利益关系；若不确定可不填，EM 允许留空。

## 4. 费用确认（投稿前必读）

- CES 为**混合刊**：默认订阅路线（不勾选 OA）**不收 APC**；金色 OA 选项（约 USD 3,500）**不要选**。
- 编辑部可能邮件询问是否选 OA——一律回复"subscription route"。
- 录用后按订阅路线出版：无任何版面费。

## 5. 风险预案（若 desk reject）

1. **Fluid Phase Equilibria**（Elsevier 混合刊免费；IL 热物性建模传统阵地；风险低）
2. **JCIM**（ACS 订阅免费；方法学向；注意同作者同刊二拒风险，且第 5 篇 8-17 刚被该刊拒）
3. 同一稿件不并行投多刊（用户既定原则）
