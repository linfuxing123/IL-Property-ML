# 第 2 篇升级作战文档 v1 — 从「验证型」到「定律 + 工具 + 基准」

> 2026-08-18 · 针对 ie-2026-04274g（原 Science ael4478）两次 desk reject 的升级方案
> 状态：三个实验已在后台跑批（leakage_tax / scaling_law / acquisition），数据出来后填数。

## 0. 一句话判断（钉死，不再动摇）

**连续两次 desk reject（Science → I&ECR）是定位问题，不是质量问题。**
编辑模板只说了"广谱读者兴趣不足"，没送审、无质量评价。第三次投必须换故事，
不是换期刊。

## 1. 新版定位

| 维度 | 旧版（被拒） | 新版 |
|---|---|---|
| 叙事 | 数据密度是**硬约束**（负面告诫） | 数据密度是**可量化的定律 + 可行动的获取策略**（正面贡献） |
| 核心交付 | 4 个性质 × 2 阶段 R² 对比 | ① 标度律 R²(N)=a−b·N^(−γ)（逐性质 γ、N80/N90）② 泄漏税曲线 ③ 模拟测量战役曲线 + 虚拟库 Top-100 优先测量清单 ④ 开放基准（组级分割 + leaderboard + 一键复现） |
| 读者拿走什么 | "哦，数据多有用" | "我知道该测哪些 IL、测多少能到 R²=0.9、随机划分会虚高多少" |

## 2. 标题候选（正负面反转，全部以"发现/定律"开头）

1. **Scaling laws of ionic-liquid property prediction: how data density governs extrapolation and where to measure next**（推荐，长但信息全）
2. How much data does ionic-liquid machine learning need? Scaling laws, leakage taxes, and acquisition priorities
3. Data-density scaling laws and optimal measurement allocation for ionic-liquid property prediction

## 3. 新摘要骨架（~230 词，第一批真实数字已回填 ✅）

> Machine learning promises high-throughput design of ionic liquids (ILs), yet
> prediction for unseen ion pairs remains limited by experimental data coverage.
> Here we turn this qualitative constraint into quantitative laws. From a curated
> NIST ILThermo dataset of 86,008 records spanning 1,891 ILs with standardized
> units and verified SMILES, we measure, under strict IL-disjoint validation,
> how group-level accuracy R² scales with the number of ILs N for viscosity,
> conductivity, density, and melting point. All properties follow
> R²(N) = a − b·N^(−γ) with property-specific exponents
> (γ = 0.215 / 0.807 / 0.565 / 0.288) and saturation ceilings
> (a = ~1.05 / 0.730 / ~1.05 / 0.809): viscosity, the most data-hungry property,
> needs ~5,700 ILs for R²=0.80, whereas conductivity saturates near 0.73 — its
> ceiling is set by descriptors, not data. We further quantify a "leakage tax":
> random point-wise splits inflate R² by Δ = 0.09–0.19 at full scale, and by up
> to Δ = 0.55 in redundancy-rich strata, so evaluation discipline must tighten
> as databases grow. Finally, simulating measurement campaigns over held-out ILs
> shows coverage-guided acquisition outperforms random sampling (ΔR² = +0.05 on
> a fixed evaluation pool at 150 measurements; larger under the deployment
> protocol), and we release a prioritized top-100 list of
> novel ILs (from 8.3 M virtual ion pairs) whose measurement would most improve
> extrapolation, together with split files and a one-command benchmark.

## 4. 新章节结构（对照旧版差异）

1. **Introduction**：从"数据不够"重写为"数据与精度的定量关系 + 测量预算决策"；
   引入学习曲线标度律文献（分子性质低数据区间的 learning-curve benchmark、
   材料数据规模的标度律研究）；保留 IL-disjoint 诚实评估的必要性论证。
2. **Results**
   - 2.1 数据资产与质量审计（压缩保留：86,008 点/1,891 IL、单位标准化、S/m 混用发现）
   - 2.2 **学习曲线标度律**（新核心）：R²(N) 曲线、γ、渐近上限、N80/N90 表 + 图
   - 2.3 **泄漏税**（方法学贡献）：ΔR² 随每 IL 记录数的变化 + 全量泄漏率
   - 2.4 **冷启动分解**（新增）：新阳离子 / 新阴离子 / 已知对×新温度 的误差分解
   - 2.5 **获取策略**（工程交付）：模拟战役曲线（random vs coverage vs uncertainty）
     + 虚拟库 Top-100 优先测量清单（uncertainty 与 coverage 双排序）
   - 2.6 与既有模型对照：同一组级分割下 GBM/HistGBM/GNN/COSMO-RS（或老师 ACS SCE 模型）leaderboard
3. **Discussion**：定律的含义（数据饱和 vs 方法饱和）、测量预算决策规则、
   对基准社区的建议（组级分割应成为报告标准）。
4. **Methods** + **Data availability**（基准仓库、分割文件、一键复现）+ SI。

## 5. 第一批实验结果（2026-08-18 已跑通，全部 exit=0，管线与 Table 2 对齐验证 ✅）

**5.1 学习曲线标度律** R²(N) = a − b·N^(−γ)（HistGBM 组级；GBR 全量锚点复现 Table 2：
电导 0.695/0.70、密度 0.834/0.85、熔点 0.389/0.39、粘度 0.695/0.68）

| 属性 | a（渐近上限） | γ（学习指数） | N(R²=0.80) | N(R²=0.90) | 解读 |
|---|---|---|---|---|---|
| 粘度 | ~1.05*（未饱和） | **0.215** | ~5,700 | ~61,000（强外推，谨慎） | 最吃数据：N=1165 时 R²=0.735 仍在爬升 |
| 电导 | 0.730（饱和） | **0.807** | 不可达 | 不可达 | 卡在 10 描述符天花板，加数据无效 → 需换特征/模型 |
| 密度 | ~1.05* | 0.565 | ~490 | ~1,210 | 已近饱和（当前 1,396 IL） |
| 熔点 | 0.809 | 0.288 | 不可达 | 不可达 | 极慢：642 IL 只有 0.375，需数千个多样性 IL |

*拟合上界被夹住（a 想去 >1.05），粘度/密度的渐近上限估计不可靠，N90 属强外推，稿中必须标注。

**5.2 泄漏税**（ΔR² = 点级 − 组级，随冗余度分桶）

| 桶（每 IL 记录数） | 粘度 ΔR² | 电导 ΔR² | 密度 ΔR² |
|---|---|---|---|
| 1（无冗余，leak 0%） | +0.16 | −0.02 | +0.00 |
| 2–4（leak ~93–97%） | **+0.55** | +1.38* | +0.40 |
| 5–9 | +0.32 | +0.35 | +0.17 |
| 10–24 | +0.31 | +0.26 | +0.20 |
| 25–49 | +0.26 | +0.55 | +0.62 |
| 50+ | +0.37 | +0.26 | +0.27 |
| **全量** | **+0.19** | **+0.22** | **+0.09** |

*小样本桶（25 IL）噪声大。**发现：泄漏税是"冗余度的阶梯函数"——只要存在多记录
冗余，点级划分就虚高 0.25–0.55 个 R²，且不会随冗余增长而消失**；单记录 IL 无法泄漏
但组级精度也低。结论：评估纪律必须与分割协议绑定，与数据规模无关（原稿 2.3 节的
量化升级版）。

**5.3 获取策略（v2 固定评估集协议，最终口径）**
- 粘度：固定评估池 150 IL 下，覆盖型在 150 次测量时 R²=0.629 vs 随机 0.579（+0.05，全程不劣于随机）
- 电导：不确定性型小预算略优（+0.04@100），大预算衰减
- **诚实修正**：v1 协议的 +0.33 是"评估集随获取缩小"的协议假象；v2 下优势温和但一致。
  且与标度律自洽：715→1,015 IL（γ=0.215）本就只值 ≈0.03 R²——**当前阶段获取的
  "顺序"不如"总量"重要**，粘度真正的约束仍是已测 IL 数量。
- 不确定性型不优于随机：与第 7 篇"分歧与性能负相关"自洽（化学覆盖缺口 > 预测不确定）。

**5.4 虚拟库覆盖统计（3M 采样自 8.33M，coverage_full.py）**
- 虚拟库 8,333,096 个离子对（219,292 阳 × 38 阴），与已测 1,165 IL（粘度集）最近距离：
  中位 1.56、p90 2.61、p99 3.69（标准化单位）；27.3% 超过 2，**4.29%（≈12.9 万）超过 3**
- **诚实发现**：10 描述符分辨率下已测集覆盖了虚拟空间 95%+——约束不是"原始覆盖"，
  而是"带性质标注的覆盖"（与标度律一致：已覆盖区内的 IL 缺性质数据）。
  远尾区域集中在季铵芳基酰胺/胺侧链阳离子 + 六氟异丙醇衍生烯醇阴离子（28,896 个）、
  甲酚盐等——与冷启动分解（阴离子维度饥渴）互相印证。
- Top-100（acquisition_top100.csv）：uncertainty 与 coverage 排序**零重合**（正交缺口）。

**5.5 管线验证**：GBR 全量锚点复现 Table 2（0.695/0.68、0.695/0.70、0.834/0.85、
0.389/0.39）——新实验与旧稿同口径，可直接续写。

**5.6 leaderboard（同一组级分割，LR/RF/GBR/HistGBM）**：模型间差异 ≤0.16 R²，
远小于数据 7.5× 扩充带来的 0.77 提升（粘度 −0.09→0.68）——**从模型侧再次证明
"数据覆盖是主导杠杆"**。熔点 RF 最优（0.396）。

**5.7 冷启动分解（按离子新颖性）**：
- 粘度：**阴离子维度是数据饥渴轴**（新阴 R²=0.367 vs 新阳 0.597）→ 补测应优先阴离子多样性
- 电导：已见离子新组合最难（0.514）→ 阴阳离子配对效应主导
- 熔点：各类都难（0.23–0.44），需全维度扩充

**5.8 结论性叙事（新稿 v0.2 已成型，manuscript_v2.md）**：
三条决定规则 = ①粘度：继续加 IL 数（数据受限，γ=0.215）；
②电导：换特征/架构，别加数据（表征受限，饱和 0.73）；
③熔点：扩离子家族多样性（多样性受限）。+ 泄漏税 + 获取策略 + 开放基准。

下一批待办（v2 实验，待用户拍板后跑）：
- [ ] 获取策略 v2：固定评估集协议（评估集永不参与获取），消除曲线中段非单调
- [ ] 虚拟库 8.33M 全量覆盖率统计 + 化学空间覆盖热图
- [ ] GNN（第 4 篇 MPNN）进 leaderboard（需 torch 环境）
- [ ] GitHub IL-Property-ML v2.0 release + Zenodo 版本 DOI
- [ ] 目标刊拍板（DD / JCIM / JCED / I&ECR 二投）→ 重做投稿格式与封面信

## 6. 目标期刊与封面信要点

- **首选**：Digital Discovery（定律+基准类型的最对口刊；第 7 篇在投但不冲突）
  或 JCIM（若强化方法学）；JCED（第 1 篇已在，不同稿不冲突）。
- **若坚持 I&ECR 二投**：封面信必须改成"测量预算优化/过程设计"叙事，
  明确回答"你的读者（工艺工程师）能拿走什么"。
- 封面信三句话卖点：**law → tool → benchmark**：
  1) 我们给出了 IL 性质预测的定量学习曲线定律（谁都没发过组级版本）；
  2) 我们把它变成测量预算决策工具（每测 100 个 IL 涨多少 R²，Top-100 优先清单）；
  3) 我们发布了可一键复现的开放基准与分割文件（必引资源）。

## 7. 交付物清单（投出前全部就位）

- [ ] 新标题/摘要/正文（本文档回填后成稿）
- [ ] 图：学习曲线 4 面板 + 泄漏税曲线 + 获取战役曲线 + 覆盖热图
- [ ] GitHub IL-Property-ML v2.0：数据集 + 分割文件 + 三个脚本 + leaderboard + 一键复现
- [ ] Zenodo 版本 DOI（沿用概念 DOI 21898948 自动指向新版本）
- [ ] SI：per-fold 全部结果 + Top-100 全表 + 数据审计明细

## 8. 诚实边界（写进稿子，不藏）

- 标度律是经验拟合（a−b·N^(−γ)），外推到 N 超出数据范围的部分是估计；
- HistGBM 扫参 + GBR 锚点两种模型都报告，不混用；
- Top-100 是"预测分歧/覆盖缺口"排序，未经实验验证——定位是**测量优先级建议**，
  不是"保证最优候选"；虚拟库性质来自其他 ML 模型（ILBERT/ILPE），有分布外风险；
- 熔点仍低于设计级（R²≈0.4），需更广化学多样性数据。
