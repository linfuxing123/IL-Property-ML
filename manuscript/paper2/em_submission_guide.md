# Elsevier Editorial Manager（CES）提交流程手册

> 2026-08-18 · 目标刊 Chemical Engineering Science · 事件驱动、不轮询
> 前置条件：① QQ 邮箱注册 EM 账号（需邮箱验证码，用户操作）；② 用户授权执行

## 0. 必要文件（已就绪，路径）

| 文件 | 路径 |
|---|---|
| 主稿 | `workspace\matmodel\paper2_upgrade\manuscript_ces.docx`（20 页，含图表） |
| 封面信 | `workspace\matmodel\paper2_upgrade\cover_letter_ces.md` |
| 元数据包 | `workspace\matmodel\paper2_upgrade\submission_checklist_ces.md` |
| 数据/代码 | GitHub v2.2.1 + Zenodo 10.5281/zenodo.21997263（稿件 §5 已含） |

## 1. 注册（用户操作，2 分钟）

1. 打开 https://www.editorialmanager.com/cesc/ （CES 的 EM 入口；若入口变动，
   从 ScienceDirect 期刊页 "Submit your article" 进入）
2. Register → 用 3612411485@qq.com → 收验证码 → 完成（若提示已注册，直接登录）
3. **把登录状态/验证码告知代理或自行按本手册操作**

## 2. 提交（New Submission，分步）

**Step 1 — Select Article Type:** Research Article（若列表无此名，选 Full Length Article）

**Step 2 — Attach Files（顺序建议）：**
1. `manuscript_ces.docx` → 类型选 **Manuscript**
2. （可选）6 张图 PNG → 类型 **Figure**（docx 已内嵌，可不上传）
3. 封面信 → 类型 **Cover Letter**（或粘贴到 Step 3 文本域）

**Step 3 — Enter Data（字段全部来自 submission_checklist_ces.md）：**
- Title / Running head / Abstract（248 词，粘贴）
- Highlights：5 条（EM 有独立 Highlights 字段；若没有，放在稿件内已合规）
- Keywords：6 个
- Author：Fuxing Lin / Hunan Institute of Engineering / 3612411485@qq.com /
  ORCID 0009-0003-7588-6942（通讯作者勾选）
- Suggested Reviewers：3 位（Makarov / Paduszyński / Yoshida，见 checklist §3）

**Step 4 — Additional Information / 声明勾选：**
- 原创性 + 未一稿多投 ✅
- 无利益冲突 ✅
- 无资助 ✅
- 数据可用性：稿件 §5 已声明（GitHub + Zenodo 版本 DOI）
- **不要**勾选任何 Open Access / APC 选项（订阅路线零费用）
- 期刊可能问 "Do you wish to publish open access?" → **No**

**Step 5 — Comments / Review Preferences:** 可留空

**Step 6 — Final Review:** 逐项核对 → Approve Submission

## 3. 提交后

- 收确认邮件（EM 系统邮件，主题含 Manuscript Number，形如 CESC-D-26-XXXXX）
- 状态跟踪：EM Dashboard → Submissions Being Processed
- **若编辑部邮件问 OA/APC**：一律回复 "subscription route"，不产生任何费用

## 4. 自动化执行注意（若用 CDP/Playwright 代投）

- 一律事件驱动：waitForSelector / waitForURL / waitForFunction，禁止固定长等待
- EM 是 AngularJS 老系统：React 外的表单需同步 scope；用真实鼠标事件点击
  （CDP Input.dispatchMouseEvent），el.click() 对 Angular 无效的坑位与
  ScholarOne/ChronosHub 类似（详见 memory/2026-08-15.md 第七节）
- 上传文件用 filechooser 拦截 + Input 事件（ACS 同款坑）
- 提交前最后一步逐页截图 QA（本地 Ollama 识图，8 月内不花云识图费用）

## 5. 风险预案（desk reject 后）

1. Fluid Phase Equilibria（Elsevier 混合刊免费，IL 热物性阵地）
2. JCIM（ACS 免费；注意 8-17 第 5 篇被该刊拒过的同作者风险）
不并行投多刊（用户既定原则）。
