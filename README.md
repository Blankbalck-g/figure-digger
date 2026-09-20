# 📈 DIG

**D**ata from **I**mages and **G**raphs

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-beta-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![VLM](https://img.shields.io/badge/VLM-optional%20(DeepSeek)-blueviolet)
![Vector](https://img.shields.io/badge/vector%20figures-lossless-success)

---

## 🚀 快速开始

```bash
# 0) 依赖（详情见 requirements.txt）
pip install -r requirements.txt

# 1) 单篇：PDF -> 数据（建议先用 --pages 拿一页试）
python run.py all paper.pdf --out out --pages 7
python run.py all paper.pdf --out out

# 1b) 只要某几张图：直接说你想要什么（找不到会如实告诉你）
python run.py all paper.pdf --out out --vlm --want "800 bar 条件下速度随时间的折线图"
python run.py all paper.pdf --out out --vlm --want "Figure 5, 图7"        # 说图号也行
python run.py all paper.pdf --out out --vlm --want "…" --want-deep        # 连正文一起判（更准）

# 2) 批量：一个目录里的所有 PDF
python run.py batch ./papers --out ./out
python run.py batch ./papers --out ./out --recursive      # 含子目录
python run.py batch ./papers --out ./out --jobs 4         # 4 篇并行（每篇一个进程）
python run.py batch ./papers --out ./out --vlm --want "速度随时间的折线图"   # 8 篇里哪些有这张图

# 2b) 还要一份"模板格式"的结果：给一个 txt/dat 模板，模型读一次编译成代码后缓存
python run.py all   paper.pdf --out out --vlm --template my_format.dat
python run.py batch ./papers  --out ./out --vlm --template my_format.dat   # 每篇都出一份
python run.py extract out/paper/paper_config.json --template my_format.dat # 只重出模板文件
python run.py extract out/paper/paper_config.json --no-template            # 不要模板（只出 CSV）

# 3) 启用 VLM（分类 / 轴读数 / 语义规划 / 质检）
copy deepseek_key.txt.example deepseek_key.txt            # 填入 key（或用环境变量 DEEPSEEK_API_KEY）
python vlm_client.py --check                              # 验证 key
python run.py all  paper.pdf --out out --vlm
python run.py batch ./papers --out ./out --vlm --jobs 4   # 推荐：VLM + 并行

# 4) 环境自检
python run.py doctor
```

- 不带 `--vlm` 也能跑：轴范围改由 OCR 提议 + 人工确认（见[轴范围确认](#-轴范围确认)）
- 带 `--vlm` 时 **OCR 默认关闭**；需要两路交叉核对时加 `--ocr`
- 默认终端只打印面板结果、最终路径和一行 **token/费用摘要**；需要逐图细节时加 `--verbose`
- 默认只保留成果文件；需要 gap/review/OCR/选图诊断图时加 `--keep-intermediates`
- 支持的图：折线图、散点图（含**双 Y 轴**、多子图）；照片/示意图/表格自动跳过

## 🎯 只要某几张图：`--want`

```bash
python run.py all  paper.pdf --out out --vlm --want "800 bar 条件下速度随时间的折线图"
python run.py batch ./papers --out ./out --vlm --want "乙醇喷雾贯穿距 vs 时间"
python run.py all  paper.pdf --out out --vlm --want "Figure 5, 图7"     # 说图号也行
```

一句话描述你要的图即可，**不用管内部怎么找**。它的做法是：

1. **先把图注和图一对一配对**。一页常有 2–5 张图、好几条 "Figure N" 文本行（还有正文里的引用句），
   所以按几何位置就近配对，而不是"取本页第一条图注"。
2. 需求里有图号（`Figure 5`/`图7`）→ 本地直接命中，**不花任何调用**。
3. 否则把"图清单（编号 + 图注 + 矢量图的图内文字）"和一张**编号缩略图对照图**一起交给模型：
   坐标轴标题、图例、工况标注都写在图里，只看图注是判断不了的。
4. **说不清时再读正文**（按需）：图注和图里都没写清"哪张图对应哪个工况"时，自动再问一次模型——
   这次给它的是**正文里所有提到图的句子**（`Figure 5 presents the penetration at Pi = 800 bar`），
   而不是全文。这些句子只占全文的 **3–12%**（实测每篇 754–4,943 字符 ≈ 209–1,373 tokens）。
   看图就能定下来时**不跑**这一步；想每次都跑加 `--want-deep`。
5. **论文里没有这张图就直说没有**：不会硬挑一张最像的，而是告诉你"这篇里没有匹配的图"，
   并列出实际的图清单（图号 + 图注），你可以换个说法或直接写图号再来一次。
6. **命中要过"轴标题"这一关**（不额外调用）：模型读刻度时会顺带读出 x 轴与各纵轴的
   **标题/单位**，并判断"这张图的横轴/纵轴是不是你要的那两个量"。要"贯穿距随时间变化"，
   横轴就必须是时间、纵轴必须是贯穿距——横轴是曲轴转角、纵轴是锥角/动量通量的会被
   **标为跳过并写明理由**（`report.md` 里 `⏭ 跳过原因`），不再只靠"模型觉得像"。
   标题看不清时**保留**面板并标 `⚠️ 看不清（保留）`，宁可让人复核也不静默丢数据。

挑选依据会写进 `out/<PDF>/report.md` 的表格（每张候选图列出图号、图注、是否选中）。
若要复核模型看到的缩略图对照图，加 `--keep-intermediates`，文件在
`out/select/contact_sheet.png`。

成本（含正文通道，结果同样进缓存）：

| 环节 | 触发条件 | tokens/篇 | 10 篇费用（高峰价，¥7.2/$） |
|---|---|---|---|
| 说了图号 | 本地匹配 | 0 | **¥0** |
| 看图选图 | 有 `--want` | 2,214–7,055 | ¥0.06–0.20 |
| 正文通道 | 看图没定下来，或 `--want-deep` | +1,290–2,440 | +¥0.03 |
| **合计最坏（10 篇都用 `--want-deep`）** | | | **≈ ¥0.3** |

相比省下的开销这很划算：未选中的图不再做分类、读轴、抽查，而**每张图约 2–3 次调用 / 2.5–4k tokens**——
一篇 57 张图的论文，选图的钱约等于**一张图**的钱。

## ✨ 核心特性

- **PDF → 数据一条命令**：分诊图与图注、切分多子图、定标、提取、质检、报告
- **按你的模板出结果** 📄：`--template my_format.dat` 给一份模板（txt/dat/csv/任意文本），
  模型读**一次**就把它编译成渲染代码并缓存（同模板再来 0 token）；模板里要按曲线标注
  参数（燃料、压力、工况…）时，先由模型从全文建立一次**论文级工况知识库**，
  再让图级 Agent 结合图片/图注/系列名只绑定 condition/fact ID，最后由代码做单位换算与模板填充。
  PDF 文本层漏掉但图内标题明写的值，可以作为受控 `panel_facts` 写回知识库供后续图复用。
  每个值都在 `report.md` 记录证据和 `explicit/derived` 来源，不能唯一归属的条件留空。
  不传 `--template` 就沿用上次的模板，从没用过就只出 CSV
- **一句话指定要哪张图** 🎯：`--want "800 bar 下速度随时间的折线图"`；图注与图先一对一配对（一页多图也不会张冠李戴），再由模型看图选图；**论文里没有这张图就直说没有**，并列出实际有哪些图
- **看不准再读正文** 📖：`--want` 定不下来时，自动拿"正文里提到图的句子"（占全文 3–12%）再判一次——`Figure 5 presents…at Pi = 800 bar` 这种条件归属只有正文写；`--want-deep` 可强制每次都读
- **矢量图无损提取** 🎯：直接读 PDF 路径坐标与真实刻度文字，**实测零误差**（不需要 OCR 与模型）
- **矢量失败自动回退**：矢量区域读不出刻度文字时渲染成位图转交图像路线，而不是直接判死
- **图例硬禁区**：本地检测有框图例，Agent 补充无框图例；整个图例区域都不作为数据证据
- **双 Y 轴**：识别左右两条纵轴，为**每条曲线指派它所属的那条轴**（x 共用，y 各算各的）
- **非数据图自动跳过**：照片、装置图、示意图、表格在读刻度前就被排除
- **VLM 主读数（可选）** 🤖：模型直接按图上刻度读轴范围并自动确认；加 `--ocr` 则恢复"两路独立读数一致才确认"
- **抽查校验**：不让模型搬运数据，只让它用视觉位置独立重读几个点，替代人工抽检
- **强制质检产物** 🔍：每个面板输出"提取点重投影回原图"的验证图，错误一眼可见
- **可审计 + 可省钱** 💰：调用全部落盘（模型/prompt/回答/token），结果带两层缓存
- **并行批量** ⚡：`--jobs N` 每篇一个进程。VLM 是网络等待型任务，实测 7 篇（各限一页、每次调用模拟 500ms 网络）：串行 52s → `--jobs 3` 25s → `--jobs 6` 20s，且曲线数与抽查结果逐项一致

## 📁 输出结构

仓库结构（`run.py` 是唯一入口，库模块都在 `dig/`）：

```
figure-digger/
├── run.py              主入口：analyze / extract / confirm / batch / doctor
├── dig/                核心库（每个模块也能单独跑，见各自 --help）
│   ├── triage_pdf.py         分诊：找出图与图注、判断矢量/位图
│   ├── split_panels.py       多子图切分
│   ├── figure_index.py       图注↔图配对、--want 选图、正文通道
│   ├── axis_ranges.py        OCR 刻度读数与标定
│   ├── ocr_ticks.py          刻度识别的底层实现 + CLI
│   ├── legend_colors.py      图例色块检测
│   ├── extract_lines.py      曲线追踪核心
│   ├── batch_extract.py      单篇内多面板的批量提取
│   ├── vector_extract.py     矢量直读
│   ├── verify_overlay.py     质检叠加图渲染
│   ├── imgio.py              中文路径安全的读写
│   ├── condition_kb.py       论文工况知识库、图级绑定、确定性单位换算
│   ├── template_out.py       模板编译/渲染、全文证据包与参数 Agent
│   ├── vlm_client.py         VLM 客户端（缓存 / 日志 / 用量）
│   └── vlm_tasks.py          全部提示词常量
├── tests/              回归测试与测试素材
├── docs/               README 用的示例图
├── requirements.txt
└── LICENSE             MIT
```

`out/`（运行产物）：

```
out/
├── figures/   从 PDF 抠出的图          panels/  切分后的子图
├── csv/       提取结果（一条曲线一个 CSV）★ 真正的成果
├── templates/ --template 渲染出来的结果文件（子目录 = 模板名）
├── verify/    质检叠加图（人眼复核用）
├── condition_knowledge.json  论文级工况事实/条件与图内补充事实
├── <pdf>_config.json    轴范围配置（人工确认就是改这个文件）
├── extract_results.json / report.md
└── batch_summary.md      批量模式的汇总表
```

默认不落盘 `gaps/`、`review/`、`debug/`、`select/` 等仅供 Agent 调用的中间图片；它们会在临时目录中使用后自动清理。加 `--keep-intermediates` 才保留这些诊断目录，便于开发时审计。除 `csv/` 外其余结果都可由重跑重建；`.vlm_cache/`（VLM 结果缓存，命中则不调 API）、`.templates/`（模板编译出来的渲染代码，命中则 0 token）与 `.vlm_log.jsonl`（审计日志）也在 `.gitignore` 中。
🧹 **提交前清理**：`rm -rf out_* *_out output .vlm_cache .vlm_log.jsonl .templates __pycache__`

## ⌨️ 命令行参考

| 命令 | 作用 |
|---|---|
| `run.py all <pdf> --out DIR [--vlm] [--want "…"] [--pages 7] [--force] [--no-ocr] [--template F] [--verbose]` | 一篇走完：分析 + 提取（+ 模板输出） |
| `run.py batch <dir> --out DIR [--vlm] [--want "…"] [--jobs 4] [--recursive] [--pattern "*.pdf"] [--template F] [--verbose]` | 批量处理目录下所有 PDF + 汇总 |
| `run.py analyze <pdf> --out DIR [--vlm] [--ocr]` | 只做分析，产出 config 供人工确认 |
| `run.py extract <config.json> [--force] [--only ID...] [--vlm] [--template F] [--no-template]` | 按 config 提取（只跑已确认的面板） |
| `run.py confirm <config.json> --id ID [--x 0,8] [--y 0,3] [--skip]` | 确认 / 修正 / 跳过某个面板 |
| `run.py doctor` | 环境自检（解释器、依赖、key） |
| `python dig/vlm_client.py --check / --probe / --stats / --image-tokens / --probe-image` | key、实际模型、token 统计、单图 token |

常用参数：`--vlm` 启用 VLM；`--want "…"` 一句话指定要哪几张图（命中要过轴标题核对）；`--want-deep` 再用正文段落判定一次；`--template F` 按模板文件再输出一份（不给 = 沿用上次的模板，`--no-template` = 不要模板）；`--vlm-model` / `--vlm-base-url` 覆盖默认模型与端点；`--pages` 只处理指定页；`--force` 跳过确认检查；`--jobs N` 并行批量（每篇一个进程，N 建议 ≤ CPU 核数）；`--no-ocr` 显式不再跑刻度 OCR；`--ocr` 即使在 `--vlm` 下也跑 OCR 做交叉核对；`--verbose` 打印完整过程；`--keep-intermediates` 保留 Agent 诊断图片。
🔧 `dig/` 里的每个模块都能单独跑（如 `python dig/triage_pdf.py paper.pdf`、`python dig/legend_colors.py fig.png`），见各自 `--help`。

📦 批量产出：

```
out/<每个 PDF 一个目录>          独立的 figures/panels/csv/verify/report
out/<PDF>/run.log               --jobs>1 时该篇的完整日志（并行时不再往终端打）
out/batch_summary.md / .json     每篇的面板数、曲线数、抽查一致率、token 与合计
```

**批量用法要点**（都实测过）：

1. **文献放哪**：`batch` 的第一个参数是**目录路径**，相对路径按“当前工作目录”解析。所以 `./papers` 指你运行命令时所处目录下的 `papers`；也可以直接给绝对路径。**目录要先自己建好**（不存在时会提示"目录里没有匹配 *.pdf 的文件"）。
   ```bash
   mkdir papers && cp /path/to/*.pdf papers/     # 文献放这里
   python run.py batch ./papers --out ./out --vlm
   ```
2. **怎么命名**：PDF 文件名（去掉 `.pdf`）会成为输出子目录名与所有产物的前缀，所以它要能当路径名用：中文、空格、连字符、下划线都**可以**（已实测）；避免 Windows 非法字符 `\ / : * ? " < > |`，以及结尾的点和空格。推荐用稳定的编号或"作者_年份_关键词"，例如 `2023-01-1635.pdf`。
3. **递归时不要有重名文件**：`--recursive` 下 `a/dup.pdf` 与 `b/dup.pdf` 会写进**同一个** `out/dup/`，后一个会覆盖前一个（已实测）。重名时请自行改名，或分批跑。
4. **不带 `--vlm` 时必须加 `--force`**：否则轴范围处于"未确认"状态，`extract` 会跳过所有面板，结果是 0 条曲线（命令会给出提示）。用 `--vlm` 时 VLM 的轴读数会直接把面板标成已确认（默认不跑 OCR，所以不需要 `--force`）。
5. **`--jobs N` 并行**：默认串行（1）。VLM 是网络等待型任务，并行收益明显——同一批 7 篇（各限一页）：串行 52.1s / `--jobs 3` 25.1s / `--jobs 6` 20.2s（每次调用模拟 500ms 网络），产出**逐面板一致**。本地不联网的 mock 只测出 11.4s → 7.4s，因为剩下的时间都在做本地的图像处理。并行时终端只打每篇小结；加 `--verbose` 才为每篇写 `out/<PDF>/run.log`。token 用量由子进程回传后统一加总；`.vlm_cache/` 共享，重复图仍只花一次钱。

## ⚙️ 工作原理

```
PDF ──→ 分诊 ──→ 图注↔图配对 ──→ [--want?] 按需求选图 ──→ VLM 分类 ─┬─ 非数据图 ──→ 跳过
                                                                    └─ 数据图 ─┬─ 矢量：直接读 PDF 路径 + 真实刻度文字（无损）
                                                                               │            └─ 读不出刻度 → 渲染成位图，转下一条路线
                                                                                └─ 位图：切分面板 → 图例取色 → VLM 读轴（OCR 默认关）
                                                                                              ↓
                    模型指出曲线 + 锚点 ──→ 代码从锚点精确追踪 ──→ VLM 抽查校验 ──→ CSV / 质检图
                    全文建论文工况库 ──→ 当前图绑定 fact/condition ID ──→ 代码换算/填模板 ──→ 报告
```

1. **分诊**：`pdfplumber` 找出每页的图与图注并导出；统计矢量路径对象判断矢量/位图。图注按"基线 + 列间距"自己分组（pdfplumber 的整行提取会把两栏同高度的文字粘成一行），再按几何位置配给具体某张图
2. **子图切分**：矩形轮廓检测找出绘图框，边距刻意给足（100–140px）——边距不足会把 y 刻度标签切掉一半，这是 OCR 读不准的头号原因
3. **图例处理**：图例仍用于读系列颜色，但整个图例框会被硬屏蔽；框里的样本线、marker、文字
   都不能参与追踪。真实曲线若被图例遮住，按不可见缺口处理
4. **轴标定**：VLM 先读（含左右多条纵轴）作为主读数；OCR 只在 `--vlm --ocr` 或纯 OCR 流程里参与。OCR 侧读刻度标签 → 最小二乘拟合 → 外推到坐标框边缘得范围，并用 **R² ≥ 0.999** 作为质量门限。实测 OCR 一个面板要 10.4s（单次 VLM 调用 1–2s），所以 VLM 模式下默认关掉它
5. **轻量曲线 Agent（模型负责懂、代码负责准）**：
   - 正常流程固定为：**一次语义规划 → 代码追踪 → 一次整图质检**。规划器一次返回系列清单、
     line/marker 身份、图例禁区、预期横向范围和粗锚点；不再先单独命名图例再重复读图
   - **模型**读出图里有哪几条数据曲线，给出每条的图例名、颜色、**是不是黑线（dark）**、**是不是闭合回线（closed）**、**怎么画的（draw：只有散点 / 散点连线 / 只有线 / 散点+独立拟合线）**、**与哪几条曲线重合（overlaps）**、**哪几段自己看不见（occluded）**，以及**三个粗锚点**（左/中/右，归一化坐标，±3% 就够）；同时列出它没有当作数据的东西（作者的斜率参考线、示意图、标注文字）
   - **代码**在每个锚点附近取该曲线的真实颜色，**从锚点向两端做连续性追踪**（每列取离上一列最近的暗段，用最近几列估斜率做预测，虚线断口自动跨过），再换算成数值
   - **黑线由模型点，代码追**：黑白论文图（实测 2006/1996 年的 PDF）里模型会说 color=#000000/dark=true，代码改用"深色笔画掩膜 + 你给的锚点"来追——不再因为"颜色是黑的"而整本图丢数据。同时用三条判据挡住背景网格：横跨大半张图且几乎没有起伏的直线、所在高度上整幅图都有同一条线、以及 40% 以上的点都挤在同一行
   - **闭合回线由模型判定**（closed）：只有模型说是"围成一圈的"才走轮廓法。普通曲线走轮廓会得到一条来回折返的假轨迹（实测散点+折线图被当成回线，画出来是一个圈）
   - **一个图例条目 = 一条曲线 = 一份数据**：draw=markers_connected（散点用折线连起来）只出**点**——线就是这些点的连线，不再另出一个 `_line.csv`（否则用户看到的是"一条曲线变成两条"）
   - **散点图不再有"阶梯"**：标记符号宽十几像素，逐列追踪会在它上面走出一条水平段，画出来就是台阶。现在标记检出齐全时整条曲线直接用**标记中心连线**（一个点一个值，散点图上十几个点就是完整数据，不再按"线要 20 点"砍掉）；标记检出不全时退一步，把追踪留下的**平台压成一个点**（683 行 -> 120 行，平台段归零）。两种情况都只改散点类曲线，实心数据线的水平段不受影响
   - **重复轨迹直接合并**：同一根线被模型拆成两条两条上报（实测 63 个面板里 30 条完全相同的轨迹）时，按"同一像素实例 / 轨迹重合 ≥97%"判定为同一条，只出一份数据并写进报告；模型明说"这两条有重合"的（两条不同的线画在一起）不合并
   - 为什么是"锚点 + 代码"而不是"代码猜物件、模型点头"：后者模型只能在我给的碎片里选，遇到"真曲线没被切出来"就只能干看着（实测 E00 的蓝线因此整条丢失），而且"判它 not data"等于静默删数据。锚点路线的输出是一份**要提取的曲线清单**，追不到的会明确报"未追到 + 原因"
   - 实测：同一条红虚线，全局追踪会断成 43%+56% 两段；给出锚点后**一条 99% 覆盖、461 点**；锚点抖动 ±3%（约 30px）仍然命中
   - 标记符号：模型说这条曲线带标记时，在**追踪出的走廊内**用腐蚀法找标记中心（走廊外的一切都不可能进来），用中心出点；模型说是模型/拟合线就出线
   - **重合并列 / 遮挡补全**：两条线画在一起时，压在下面的那条在重合段里**一个像素都没有**（图上只显示上面那条的颜色）。代码拿"另一条曲线真的画在那里"当证据，把断掉的那段沿它的轨迹补回来；补出来的点与实测点分开计数，写进报告（`补全 N 列（被「X」遮挡）`）并画在质检图上（橙色圆圈）。**没有笔画经过的长断口一律不补**——曲线本来就在那里结束时必须保持断开
   - **局部补断只在异常时触发**：像素证据补不上时，只裁缺口小图，让模型给 4~8 个锚点，
     代码仅在缺失区插值；每面板最多 3 次。裁剪图存档在 `gaps/`
   - **一次整图质检**：把提取轨迹编号画回原图，模型只返回 merge/drop/add/extend；代码仍负责
     真正的像素追踪。成功曲线的规划锚点不再画到最终 verify，只有失败位置会标红
   - **轴的数量级标注**：轴旁边写着 ×10⁻⁴ 这类乘数时，模型读出来、代码乘到数据上（报告里注明"数量级标注"），不再让 CSV 里出现"5.6 其实是 5.6×10⁻⁴"这种错
   - **模板输出（`--template`）**：模型把模板编译成一段最简 `render(series, context)` 代码（无注释无打印），按模板内容 sha1 缓存；命中缓存就完全不调模型。代码只拿得到曲线数据 + 参数，渲染结果写到 `out/templates/<模板名>/`
     - **参数由"代码自己说"**：先空跑一遍，把 `render` 实际取用的字段名探出来（模型有时只列了"按曲线"的参数，模板里的工况块会被漏掉），再拿这些字段 + 模板原文里的单位写法去问模型；读不到的字段填空（模板要数值时退成 0），绝不会因为缺参数把整份输出丢掉
     - **论文级工况知识库**：代码先把全文证据分成“当前图直接证据 / 实验条件表 / 字段专项检索 / 相邻页”四路，双栏 PDF 同时保留默认阅读序和左右栏阅读序。模型每篇只建一次 `facts/conditions/global_fact_ids`，原值、原单位、证据与关系持久化到 `condition_knowledge.json`
     - **图级精确绑定**：后续面板不再重读全文，只根据当前图/图注/系列名选 condition/fact ID；同一 condition 含 core/bulk 等竞争值时用 `parameter_fact_ids` 精确到字段。图内标题明写但 PDF 文本缺失的值才能作为 `panel_facts` 受控入库，空的论文知识库也可从图内事实开始建立
     - **代码负责数值真值**：支持 `bar/MPa/kPa/Pa/dyn·cm⁻²`、密度、长度、质量、`°C↔K`、直径→半径和 `same_as`。多个事实换算后冲突就留空；列表/范围/一段话里的多个数字不是标量，严禁默认取第一个
     - **真实回归**：2026 Figure 10 得到 273 K、35 MPa、0.1 MPa 与 363/273 K；2024 三温度图绑定 800/946/1150 K；2007 从图内 `700 BAR / 21 g/L / 5×0.18×150 TIP` 得到 `70 MPa / 0.021 g/cm³ / 0.009 cm`；2023 歧义反例只填唯一明确的 298 K
     - **实测费用**：每篇首面板一次建库 + 一次绑定，后续面板通常只需一次绑定。当前真实定向回归约 `$0.001–$0.009/篇`，10 篇仍显著低于 `$0.3`；论文库、prompt/image 都有本地缓存
6. **矢量直读**：聚类矢量路径成图区域 → 找坐标框 → **用刻度线几何位置 + 刻度文字数值**拟合（文字包围盒有约 0.36pt 系统偏差，用刻度线可消除）→ 映射路径点。区域里没有可读刻度文字时（刻度是路径而非文字）不会直接判死：**有坐标框**就渲染成位图转交位图路线再试一次；覆盖整页、与已导出的位图重叠、或压根没有坐标框（页面线条/表格）的才跳过
7. **VLM 辅助**：只做"小输出"任务（分类、轴读数、命名、抽查），不让模型搬运数据点

## 🎯 轴范围确认

位图的轴范围默认需要确认——这是设计选择，不是缺陷。配置片段：

```json
{
  "id": "paper_p4_img1_p1",
  "frame": [160, 54, 904, 670],
  "legend_colors": ["#4369a7", "#489557", "#a73437"],
  "axis": { "x": [0, 8], "y": [0, 3], "confirmed": false, "source": "ocr" },
  "y_axes": { "left": {"range": [0, 2]}, "right": {"range": [0, 40]} },
  "series_axes": { "#487A88": "left", "#957768": "right" }
}
```

```bash
python run.py confirm out/paper_config.json --id paper_p4_img1_p1 --x 0,8 --y 0,3   # 认可/修正
python run.py confirm out/paper_config.json --id paper_p3_img1_p1 --skip            # 跳过
python run.py extract out/paper_config.json
```

✅ **启用 VLM 后**：VLM 的轴读数直接写入 `confirmed: true`，这一步即可省略；若加了 `--ocr`，则要求两路读数一致（容差 2%）才自动确认，不一致的会写进 `notes` 供复核。

## 🤖 VLM 辅助（可选）

```bash
copy deepseek_key.txt.example deepseek_key.txt   # 填 key
python vlm_client.py --check                     # 验证 + 列出可用模型
```

🔑 key 读取顺序：`deepseek_key.txt` → 环境变量 `DEEPSEEK_API_KEY` → `secrets.json`（代码中无硬编码密钥，文件已在 `.gitignore`）。

接口事实（依据官方文档）：base_url `https://api.deepseek.com`（OpenAI 兼容）；支持图像的模型是 **`deepseek-flash`**（`deepseek-v4-pro` **不支持**图像）；图像用 `image_url` + base64 data URI；结构化输出用 `response_format={"type":"json_object"}`，prompt 中需含 "json" 并给示例。

四项任务：

| 任务 | 作用 |
|---|---|
| 🏷️ 图表分类 | **最先执行**；判为非数据图则整图跳过，不做切分与 OCR |
| 📏 轴范围读数 | **主读数来源**；含左右多条纵轴，加 `--ocr` 时 OCR 随后独立核对 |
| 🧭 语义规划 | 系列数量/身份、line 与 marker、纵轴归属、图例禁区、预期范围与粗锚点 |
| ✅ 抽查校验 | 用**视觉位置**（最左/正中/最右）提问、**不给任何轴数值**，模型自己按图上刻度读数并与追踪结果比对 |

💰 成本与缓存（`deepseek-flash` 实测）：

- 单张图约 **525 token**（官方上界 1024）；一页 4 次调用 ≈ 3,100 input / 350 output tokens ≈ **$0.001**
- 一篇 8 图 23 面板的论文约 50 次调用 → **高峰约 $0.014、闲时约 $0.007**
- **本地缓存** `.vlm_cache/`（键=模型+prompt+图片哈希）：重跑、以及同一张图出现在多个文件时 **0 次调用**；批量实测第二个相同文件 0 token
- **服务端上下文缓存**：模板提示词与重复图片命中，实测一页 3,167 input 中 2,432 命中（命中价 1/50）

> ⚠️ **thinking 模式必须关闭**。DeepSeek 默认开启（effort=high），思维链会吃光 `max_tokens` 让 `content` 返回空——本项目已默认发送 `{"thinking":{"type":"disabled"}}`；确需推理用 `--thinking` 并调大 `max_tokens`。

## 🔧 定制与调参

**提示词全部集中在 [`dig/vlm_tasks.py`](dig/vlm_tasks.py) 顶部的字符串常量**，直接改文本即可：

| 常量 | 行 | 作用 |
|---|---|---|
| `CLASSIFY_SYSTEM` / `CLASSIFY_PROMPT` | 12 / 14 | 判断是不是可提取的数据图 |
| `AXIS_SYSTEM` / `AXIS_PROMPT` | 25 / 27 | 读坐标轴范围（含左右两条纵轴） |
| `NAME_SYSTEM` / `NAME_PROMPT` | 41 / 43 | 图例↔颜色映射 + 每条曲线的轴归属 |
| `SPOT_SYSTEM` / `SPOT_PROMPT` | 53 / 55 | 抽查校验 |
| `SHORTLIST_PROMPT` / `SELECT_PROMPT` | 70 / 81 | `--want` 选图：纯文字粗筛 / 看图选图（含"没有就说没有"的约束） |
| `BODY_SELECT_PROMPT` | 99 | `--want` 的正文通道：只喂"提到图的段落"，要求引用原句 |

⚠️ 改提示词必须注意：

1. **保留 "json" 字样和一份示例输出**——JSON 输出模式要求如此，删了会退化成普通文本
2. **`SPOT_PROMPT` 里不能出现我们的任何数值**（轴范围等）——抽查的价值来自"独立"，早期版本喂了轴范围，模型被锚定，6 点里 4 点答成圆整数
3. **返回字段名与代码解析耦合**：只改文字可以；改字段名（`min`/`max`/`y_axes`/`side`/`read_y`…）要同步改 `run.py` 的 `_norm_axis` / `_norm_y_axes` / 命名遍历
4. **双轴图依赖侧别说明**：`SPOT_PROMPT` 的"按左/右侧纵轴刻度读数"与 `AXIS_PROMPT` 的 `side` 字段要成对保留
5. **改提示词会让本地缓存失效**（缓存键含提示词），下次运行重新计费

其它可调项：

| 想改什么 | 位置 |
|---|---|
| 默认模型 / 端点 / 价格表 / 上传缩放上限 / 密钥与缓存路径 | `dig/vlm_client.py` L40/41、L290、L44/45、L42/43（模型端点也可用命令行覆盖） |
| 抽查位置、抽查条数、容差 | `run.py` L878 `spots`、L883 `[:2]`；`dig/vlm_tasks.py` L137 `axis_tolerance` |
| 非数据图判定（覆盖率/厚度）、子图边距 | `run.py` L668、L131 |
| 矢量回退（裁剪边距、整页/无框/重复判定） | `run.py` L407、L410 |
| 并行批量（每篇一个进程） | `run.py` L975 `_batch_one`、L1035 `_batch_parallel` |
| 选图的图注配对距离、候选上限、缩略图尺寸 | `dig/figure_index.py` L30/31 `SHEET_MAX`/`TILE_W`、`attach_captions` 里的 80/40/45 |
| 正文通道的触发阈值与摘要长度上限 | `dig/figure_index.py` L32 `BODY_TRIGGER`；`run.py` `_body_digest` 的 `max_chars=12000` |
| **让模型找曲线的提示词**（要它给什么、锚点怎么描述） | `dig/vlm_tasks.py` `FIND_SERIES_PROMPT` |
| 锚点搜索半径、追踪容差、标记点走廊 | `dig/series_seed.py` `trace_from_seed`（45px 半径 / `max_gap=90`）、`markers_in_corridor`（`half=22`） |
| 曲线可信度闸门（点太少/跨度太小就不写） | `dig/batch_extract.py` 顶部 `MIN_SPAN_X`、`MIN_POINTS`、`MAX_BIG_JUMPS` |
| 图框与图例/放大子图的识别 | `dig/extract_lines.py` `find_boxes`（长直线组合+四边验真）、`batch_extract.inner_boxes` |
| 回退路线（按颜色全局追踪）的容差 | `dig/extract_lines.py` `trace_series_by_target` 的 `hue_tol`、`_walk` 的 `max_jump/max_gap` |
| 轴读数质量门限、标签带尺寸 | `dig/axis_ranges.py` L74 `min_r2=0.999`、L87/88 |
| 图例色块检测 | `dig/legend_colors.py` L43/56/130/171 |
| 曲线颜色匹配与追踪 | `dig/extract_lines.py` L164/L180 |
| 坐标框检测 | `dig/extract_lines.py` L62 |
| 矢量区域检测 | `dig/vector_extract.py` L39/137/238 |
| PDF 图片过滤 / 渲染 DPI | `dig/triage_pdf.py` L184 / L191 |

💡 **只改某一张图时不必动代码**——改它那份配置即可（`run.py confirm ...`），重跑时 VLM 结果走缓存几乎不花钱。
调参顺序建议：先用 `--pages` 单页试，改一处看一处；优先调"证据阈值"（`min_r2`、容差），不要先动结论——把错误数据放进结果比漏掉更危险。

## ⚠️ 已知限制

- **可疑曲线会被拒绝而不是硬输出**：一条轨迹如果只覆盖横轴的 30% 以内、点数不足 25、或有大跳变，会被判为"标注/图例样本/标记点混入"而不写 CSV，原因记在 `report.md` 与 `extract_results.json` 的 `skipped_series` 里。宁可说"这条没提出来"，也不给一条看起来像数据的垃圾
- **同色且挨得很近的两条线**：同色多实例用"标记半径走廊"区分，间距小于约 20px 的同色线仍会被当成一条；不同颜色但间距只有 8–19px 的邻近曲线（实测三温度曲线就是）能正确分开，因为去重只在**颜色内部**做
- **黑白线**：黑线需要模型确认后才提取（`dark_series`），且要求纵向跨度 ≥12% 量程，否则会被当成轴线/文字行拒绝；灰色误差带仍不提取（与网格同色，按颜色无法区分）
- **对数轴**：目前只检测并标记"疑似对数轴"，未实现 `数值 = 10^(a×像素+b)` 映射
- **启发式阈值**：非数据图判定阈值来自单本期刊实测，换期刊建议先跑 `analyze` 扫一遍报告的"注意："标记
- **被遮挡的段是"补"出来的**：重合段里压在下面的曲线没有像素，代码沿另一条曲线的轨迹补回，并在报告里写明补了多少列、被谁遮挡；质检图上用橙色圆圈标出。只补"有笔画经过"的断口，没有证据的长断口保持断开（报告里标 `⚠️ 长断口未补`）
- **闭合回线只提取可见的弧**：P-V 边界、喷雾边界那种一圈的曲线，被另一圈压住的部分目前补不了（掩膜碎成多块时按可见段出数据），报告里会标 `闭合回线` 并给出实际覆盖范围
- **网格线的三道防线**：黑线路线最容易把背景网格当成曲线。现在挡了三层——横跨大半张图且几乎没有起伏、所在高度整幅图都有同一条线、40% 以上的点挤在同一行；仍然可能漏的（很淡的点状网格、极短的残段）会在报告里以"几乎水平的轨迹"出现，复盘轮也会看到并要求模型确认
- **复盘轮要花 token**：只在有问题的面板上问一次（正常面板不问）。如果不想要这一轮，跑 `extract` 时不要传 `--vlm`；同一面板的复盘结果有缓存，重跑不重复计费
- **模板代码是模型写的一次性代码**：缓存在 `.templates/<哈希>.json`（可直接打开看/改，改完删掉重跑即可重新生成）。它在只给白名单 `import` 的环境里执行，渲染失败只跳过模板输出、不影响 CSV。模板参数来自论文工况库与当前图绑定；报告会保留原始证据、事实 ID、换算方式和冲突，无法唯一判断时留空。`condition_knowledge.json` 是可审计缓存；原 PDF 更改后会自动失效重建
- **`--want` 的轴标题核对看不清时不删**：模型读不出轴标题（扫描图常见）时会保留该面板并标 `⚠️ 看不清（保留）`，需要你按质检图人工确认；要强制只保留核对通过的，可在报告里按 `axis_check` 手工 `run.py confirm --skip`

## 🩺 故障排查

| 现象 | 处理 |
|---|---|
| `依赖导入失败 / No module named 'numpy._utils'` | 用错环境了。`python run.py doctor` 看解释器，`conda activate <装了依赖的环境>` 后重试 |
| `plot frame not found` | 图无闭合坐标框。裁剪过紧时已自动退回 `edge_frac=0` 再试一次（实测把一张 0 曲线的图救成 8 条）；仍失败就调 `extract_lines.find_frame` 的 `dark_thresh` |
| y 轴读数差 | 优先怀疑裁剪边距不足导致标签被切；增大 `split_panels` 的 margin 或 `--y-band` |
| `y 轴拟合质量不足` | OCR 混入错误数值被 R²≥0.999 门限拒绝；用 `confirm` 手填或启用 `--vlm` |
| `[矢量] ... 无刻度文字 → 渲染为位图` | 正常回退：该区域刻度不是文字（或缺文字），已自动改走图像路线；同一篇的位图面板可能因此多一张 |
| 提取出多余/错误系列 | 图例取色失败；跑 `legend_colors.py <fig>` 看读到的颜色 |
| VLM 每次返回空内容 | thinking 模式吃光了 `max_tokens`（日志里 `finish_reason=length`）；确认没有误用 `--thinking` |
| `--jobs` 并行时终端看不到过程输出 | 设计如此：终端只打每篇小结；需要逐篇完整日志时加 `--verbose` |
| `--want` 说"这篇里没有匹配的图" | 不是故障：模型判定没有该图，终端会列出实际的图清单；换个说法或直接写图号（`--want "Figure 5"`） |
| `--want` 选错了图 | 看 `report.md` 的候选表；需要模型所见对照图时用 `--keep-intermediates` 重跑；直接写图号最稳 |
| 指定了 `--vlm` 但提示无 key | 正常降级，走 OCR/图像路线；填 `deepseek_key.txt` 即可 |
| matplotlib 一绘图就崩（`0xC06D007F`） | 环境 DLL 冲突；本项目质检图用 OpenCV 渲染，不依赖 matplotlib |
| numpy 线性代数崩溃（`inv`/`svd`/`polyfit` 杀进程） | pip 升级后 `numpy.libs` 未同步；本项目拟合全用闭式算术，不调 LAPACK。修环境：`conda install -c conda-forge numpy --force-reinstall` |

## 🧪 开发与测试

```bash
# 矢量提取精度回归：生成已知答案的 PDF 并逐点核对（不需要 key）
python tests/make_test_vector_pdf.py tests/test_vector.pdf
python dig/vector_extract.py tests/test_vector.pdf --out test_vector_out

# VLM 链路自检（不需要 key、不联网）
python dig/vlm_client.py --selftest
python tests/test_vlm_mock.py [image.png]           # 本地 mock 服务器端到端
python tests/test_vlm_pipeline.py paper.pdf 7       # 用 mock 跑完整流水线
```

📊 已知实测结果：矢量路线 **0 误差**；一篇 8 图 23 面板的论文 → 17 个数据面板 / 37 条曲线；VLM 抽查通过率 **98%**（唯一的失败项是双 Y 轴图，修正轴指派后 5/5 通过）。

| 模块 | 职责 |
|---|---|
| `run.py` | 主入口与流水线编排（analyze / extract / confirm / batch / doctor），含并行批量 |
| `dig/figure_index.py` | 图注↔图配对、按 `--want` 选图（本地编号匹配 + 缩略图对照图 + 正文通道） |
| `dig/axis_ranges.py` / `dig/ocr_ticks.py` | 刻度读数与标定（实现都在 `axis_ranges.py`，`ocr_ticks.py` 是底层识别 + CLI） |
| `dig/extract_lines.py` | 图像提取核心（轴框、图例排除、颜色匹配、逐列追踪） |
| `dig/legend_colors.py` | 图例色块检测（线段样本 + 圆点标记） |
| `dig/vector_extract.py` | 矢量直读（含图表内文字提取） |
| `dig/vlm_client.py` / `dig/vlm_tasks.py` | VLM 客户端（缓存/日志/用量）与全部提示词 |
| `dig/verify_overlay.py` / `dig/imgio.py` | 质检图渲染（纯 OpenCV）、中文路径安全读写 |
| `dig/triage_pdf.py` / `dig/split_panels.py` / `dig/batch_extract.py` | 分诊（按栏分行、图注跨行合并）、子图切分、单篇多面板批量提取 |

## 🤝 贡献

欢迎 issue 与 PR。提交前请：

1. 跑一遍"开发与测试"中的回归（尤其是矢量零误差测试）
2. 新增启发式阈值请附实测数据说明，不要凭感觉定
3. 不要提交任何 API key、论文原文或提取出的受版权保护数据

## ⚖️ 许可

**MIT License** —— 见 [LICENSE](LICENSE)。可自由使用、修改、分发（包括商用），保留版权声明即可。

> ⚠️ 注意：许可证只覆盖**本仓库的代码**。它不授予任何论文原文或从中提取出的数据的权利。
>
使用前请注意：提取他人论文中的数据涉及**版权与学术规范**，请遵守目标期刊与所在机构的规定，并在成果中正确引用数据来源。
