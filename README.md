# 📈 DIG

**D**ata from **I**mages and **G**raphs

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-beta-orange)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
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

# 3) 启用 VLM（分类 / 轴读数 / 系列命名 / 抽查校验）
copy deepseek_key.txt.example deepseek_key.txt            # 填入 key（或用环境变量 DEEPSEEK_API_KEY）
python vlm_client.py --check                              # 验证 key
python run.py all  paper.pdf --out out --vlm
python run.py batch ./papers --out ./out --vlm --jobs 4   # 推荐：VLM + 并行

# 4) 环境自检
python run.py doctor
```

- 不带 `--vlm` 也能跑：轴范围改由 OCR 提议 + 人工确认（见[轴范围确认](#-轴范围确认)）
- 带 `--vlm` 时 **OCR 默认关闭**；需要两路交叉核对时加 `--ocr`
- 跑完自动打印 **token 用量与预估费用** 💰，并在 `out/verify/` 生成质检图
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

挑选依据会写进 `out/<PDF>/report.md` 的表格（每张候选图列出图号、图注、是否选中）；
模型看到的那张对照图留在 `out/select/contact_sheet.png`，可以复核它是不是看错了。

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
- **一句话指定要哪张图** 🎯：`--want "800 bar 下速度随时间的折线图"`；图注与图先一对一配对（一页多图也不会张冠李戴），再由模型看图选图；**论文里没有这张图就直说没有**，并列出实际有哪些图
- **看不准再读正文** 📖：`--want` 定不下来时，自动拿"正文里提到图的句子"（占全文 3–12%）再判一次——`Figure 5 presents…at Pi = 800 bar` 这种条件归属只有正文写；`--want-deep` 可强制每次都读
- **矢量图无损提取** 🎯：直接读 PDF 路径坐标与真实刻度文字，**实测零误差**（不需要 OCR 与模型）
- **矢量失败自动回退**：矢量区域读不出刻度文字时渲染成位图转交图像路线，而不是直接判死
- **位图图例取色**：从图例读系列真实颜色再逐列追踪，深红/深蓝不会被误分
- **双 Y 轴**：识别左右两条纵轴，为**每条曲线指派它所属的那条轴**（x 共用，y 各算各的）
- **非数据图自动跳过**：照片、装置图、示意图、表格在读刻度前就被排除
- **VLM 主读数（可选）** 🤖：模型直接按图上刻度读轴范围并自动确认；加 `--ocr` 则恢复"两路独立读数一致才确认"
- **抽查校验**：不让模型搬运数据，只让它用视觉位置独立重读几个点，替代人工抽检
- **强制质检产物** 🔍：每个面板输出"提取点重投影回原图"的验证图，错误一眼可见
- **可审计 + 可省钱** 💰：调用全部落盘（模型/prompt/回答/token），结果带两层缓存
- **并行批量** ⚡：`--jobs N` 每篇一个进程。VLM 是网络等待型任务，实测 7 篇（各限一页、每次调用模拟 500ms 网络）：串行 52s → `--jobs 3` 25s → `--jobs 6` 20s，且曲线数与抽查结果逐项一致

## 📁 输出结构

```
out/
├── figures/   从 PDF 抠出的图          panels/  切分后的子图
├── csv/       提取结果（一条曲线一个 CSV）★ 真正的成果
├── verify/    质检叠加图（人眼复核用）
├── debug/     刻度标签带裁剪图（只有跑 OCR 时才有：--ocr 或纯 OCR 流程）
├── select/    --want 时的选图依据（contact_sheet.png = 模型看的缩略图对照图）
├── <pdf>_config.json    轴范围配置（人工确认就是改这个文件）
├── extract_results.json / report.md
└── batch_summary.md      批量模式的汇总表
```

除 `csv/` 外全部可由重跑重建；`.vlm_cache/`（VLM 结果缓存，命中则不调 API）与 `.vlm_log.jsonl`（审计日志）也在 `.gitignore` 中。
🧹 **提交前清理**：`rm -rf out_* *_out output .vlm_cache .vlm_log.jsonl __pycache__`

## ⌨️ 命令行参考

| 命令 | 作用 |
|---|---|
| `run.py all <pdf> --out DIR [--vlm] [--want "…"] [--pages 7] [--force] [--no-ocr]` | 一篇走完：分析 + 提取 |
| `run.py batch <dir> --out DIR [--vlm] [--want "…"] [--jobs 4] [--recursive] [--pattern "*.pdf"]` | 批量处理目录下所有 PDF + 汇总 |
| `run.py analyze <pdf> --out DIR [--vlm] [--ocr]` | 只做分析，产出 config 供人工确认 |
| `run.py extract <config.json> [--force] [--only ID...] [--vlm]` | 按 config 提取（只跑已确认的面板） |
| `run.py confirm <config.json> --id ID [--x 0,8] [--y 0,3] [--skip]` | 确认 / 修正 / 跳过某个面板 |
| `run.py doctor` | 环境自检（解释器、依赖、key） |
| `vlm_client.py --check / --probe / --stats / --image-tokens / --probe-image` | key、实际模型、token 统计、单图 token |

常用参数：`--vlm` 启用 VLM；`--want "…"` 一句话指定要哪几张图；`--want-deep` 再用正文段落判定一次；`--vlm-model` / `--vlm-base-url` 覆盖默认模型与端点；`--pages` 只处理指定页；`--force` 跳过确认检查；`--jobs N` 并行批量（每篇一个进程，N 建议 ≤ CPU 核数）；`--no-ocr` 显式不再跑刻度 OCR；`--ocr` 即使在 `--vlm` 下也跑 OCR 做交叉核对。
🔧 独立工具（`triage_pdf.py`、`split_panels.py`、`legend_colors.py`、`ocr_ticks.py`、`vector_extract.py`、`extract_lines.py`、`verify_overlay.py`）各自都可单跑，见 `--help`。

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
5. **`--jobs N` 并行**：默认串行（1）。VLM 是网络等待型任务，并行收益明显——同一批 7 篇（各限一页）：串行 52.1s / `--jobs 3` 25.1s / `--jobs 6` 20.2s（每次调用模拟 500ms 网络），产出**逐面板一致**。本地不联网的 mock 只测出 11.4s → 7.4s，因为剩下的时间都在做本地的图像处理。并行时：① 每篇日志写进 `out/<PDF>/run.log`，终端只打每篇小结；② token 用量由子进程回传后**统一加总**打印；③ `.vlm_cache/` 是共享的，重复图仍只花一次钱；④ `.vlm_log.jsonl` 由多个进程追加写，极端情况下行可能交错（不影响提取结果）。

## ⚙️ 工作原理

```
PDF ──→ 分诊 ──→ 图注↔图配对 ──→ [--want?] 按需求选图 ──→ VLM 分类 ─┬─ 非数据图 ──→ 跳过
                                                                    └─ 数据图 ─┬─ 矢量：直接读 PDF 路径 + 真实刻度文字（无损）
                                                                               │            └─ 读不出刻度 → 渲染成位图，转下一条路线
                                                                               └─ 位图：切分面板 → 图例取色 → VLM 读轴（OCR 默认关）
                                                                                             ↓
                                          VLM 系列命名 ──→ 曲线追踪 ──→ VLM 抽查校验 ──→ CSV / 质检图 / 报告 / token
```

1. **分诊**：`pdfplumber` 找出每页的图与图注并导出；统计矢量路径对象判断矢量/位图。图注按"基线 + 列间距"自己分组（pdfplumber 的整行提取会把两栏同高度的文字粘成一行），再按几何位置配给具体某张图
2. **子图切分**：矩形轮廓检测找出绘图框，边距刻意给足（100–140px）——边距不足会把 y 刻度标签切掉一半，这是 OCR 读不准的头号原因
3. **图例取色**：读图例里的线段样本与圆点标记（横排/竖排都支持），按色相+颜色距离合并重复；灰/黑条目按设计跳过（与网格文字同色，按颜色提取必然污染）
4. **轴标定**：VLM 先读（含左右多条纵轴）作为主读数；OCR 只在 `--vlm --ocr` 或纯 OCR 流程里参与。OCR 侧读刻度标签 → 最小二乘拟合 → 外推到坐标框边缘得范围，并用 **R² ≥ 0.999** 作为质量门限。实测 OCR 一个面板要 10.4s（单次 VLM 调用 1–2s），所以 VLM 模式下默认关掉它
5. **曲线追踪**：对每个系列颜色做色相+饱和度匹配（黑色文字、灰色网格因饱和度低自动排除），逐列取中位数
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
| 🔖 系列命名 | 图例文字 ↔ 颜色映射 + **每条曲线读哪条纵轴** |
| ✅ 抽查校验 | 用**视觉位置**（最左/正中/最右）提问、**不给任何轴数值**，模型自己按图上刻度读数并与追踪结果比对 |

💰 成本与缓存（`deepseek-flash` 实测）：

- 单张图约 **525 token**（官方上界 1024）；一页 4 次调用 ≈ 3,100 input / 350 output tokens ≈ **$0.001**
- 一篇 8 图 23 面板的论文约 50 次调用 → **高峰约 $0.014、闲时约 $0.007**
- **本地缓存** `.vlm_cache/`（键=模型+prompt+图片哈希）：重跑、以及同一张图出现在多个文件时 **0 次调用**；批量实测第二个相同文件 0 token
- **服务端上下文缓存**：模板提示词与重复图片命中，实测一页 3,167 input 中 2,432 命中（命中价 1/50）

> ⚠️ **thinking 模式必须关闭**。DeepSeek 默认开启（effort=high），思维链会吃光 `max_tokens` 让 `content` 返回空——本项目已默认发送 `{"thinking":{"type":"disabled"}}`；确需推理用 `--thinking` 并调大 `max_tokens`。

## 🔧 定制与调参

**提示词全部集中在 [`vlm_tasks.py`](vlm_tasks.py) 顶部的 8 个字符串常量**，直接改文本即可：

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
| 默认模型 / 端点 / 价格表 / 上传缩放上限 / 缓存与日志路径 | `vlm_client.py` L38/39、L288、L42/43、L40/41（模型端点也可用命令行覆盖） |
| 抽查位置、抽查条数、容差 | `run.py` L878 `spots`、L883 `[:2]`；`vlm_tasks.py` L137 `axis_tolerance` |
| 非数据图判定（覆盖率/厚度）、子图边距 | `run.py` L668、L131 |
| 矢量回退（裁剪边距、整页/无框/重复判定） | `run.py` L407、L410 |
| 并行批量（每篇一个进程） | `run.py` L975 `_batch_one`、L1035 `_batch_parallel` |
| 选图的图注配对距离、候选上限、缩略图尺寸 | `figure_index.py` L30/31 `SHEET_MAX`/`TILE_W`、`attach_captions` 里的 80/40/45 |
| 正文通道的触发阈值与摘要长度上限 | `figure_index.py` L32 `BODY_TRIGGER`（低于该置信度才读正文）；`run.py` L223 `_body_digest` 的 `max_chars=12000` |
| 轴读数质量门限、标签带尺寸 | `axis_ranges.py` L74 `min_r2=0.999`、L87/88 |
| 图例色块检测 | `legend_colors.py` L43/56/130/171 |
| 曲线颜色匹配与追踪 | `extract_lines.py` L164/L180 |
| 坐标框检测 | `extract_lines.py` L62 |
| 矢量区域检测 | `vector_extract.py` L39/137/238 |
| PDF 图片过滤 / 渲染 DPI | `triage_pdf.py` L184 / L191 |

💡 **只改某一张图时不必动代码**——改它那份配置即可（`run.py confirm ...`），重跑时 VLM 结果走缓存几乎不花钱。
调参顺序建议：先用 `--pages` 单页试，改一处看一处；优先调"证据阈值"（`min_r2`、容差），不要先动结论——把错误数据放进结果比漏掉更危险。

## ⚠️ 已知限制

- **灰色误差带 / 黑色点划线**：按颜色无法提取（与网格、文字同色），需按线型/填充模式识别
- **对数轴**：目前只检测并标记"疑似对数轴"，未实现 `数值 = 10^(a×像素+b)` 映射
- **启发式阈值**：非数据图判定阈值来自单本期刊实测，换期刊建议先跑 `analyze` 扫一遍报告的"注意："标记
- **不连续系列**：只提取可见部分，被遮挡段不插值

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
| `--jobs` 并行时终端看不到过程输出 | 设计如此：每篇的完整日志在 `out/<PDF>/run.log`，终端只打每篇小结 |
| `--want` 说"这篇里没有匹配的图" | 不是故障：模型判定没有该图，终端会列出实际的图清单；换个说法或直接写图号（`--want "Figure 5"`） |
| `--want` 选错了图 | 看图注是否缺失（`out/<PDF>/report.md` 的候选表里图注为空说明这篇没图注），以及模型看到的 `out/select/contact_sheet.png`；直接写图号最稳 |
| 指定了 `--vlm` 但提示无 key | 正常降级，走 OCR/图像路线；填 `deepseek_key.txt` 即可 |
| matplotlib 一绘图就崩（`0xC06D007F`） | 环境 DLL 冲突；本项目质检图用 OpenCV 渲染，不依赖 matplotlib |
| numpy 线性代数崩溃（`inv`/`svd`/`polyfit` 杀进程） | pip 升级后 `numpy.libs` 未同步；本项目拟合全用闭式算术，不调 LAPACK。修环境：`conda install -c conda-forge numpy --force-reinstall` |

## 🧪 开发与测试

```bash
# 矢量提取精度回归：生成已知答案的 PDF 并逐点核对（不需要 key）
python make_test_vector_pdf.py test_vector.pdf
python vector_extract.py test_vector.pdf --out test_vector_out

# VLM 链路自检（不需要 key、不联网）
python vlm_client.py --selftest
python test_vlm_mock.py [image.png]         # 本地 mock 服务器端到端
python test_vlm_pipeline.py paper.pdf 7     # 用 mock 跑完整流水线
```

📊 已知实测结果：矢量路线 **0 误差**；一篇 8 图 23 面板的论文 → 17 个数据面板 / 37 条曲线；VLM 抽查通过率 **98%**（唯一的失败项是双 Y 轴图，修正轴指派后 5/5 通过）。

| 模块 | 职责 |
|---|---|
| `run.py` | 主入口与流水线编排（analyze / extract / confirm / batch / doctor），含并行批量 |
| `figure_index.py` | 图注↔图配对、按 `--want` 选图（本地编号匹配 + 缩略图对照图） |
| `axis_ranges.py` / `ocr_ticks.py` | 刻度读数与标定（逻辑在 `axis_ranges.py`，`ocr_ticks.py` 只是 CLI 包装） |
| `extract_lines.py` | 图像提取核心（轴框、图例排除、颜色匹配、逐列追踪） |
| `legend_colors.py` | 图例色块检测（线段样本 + 圆点标记） |
| `vector_extract.py` | 矢量直读 |
| `vlm_client.py` / `vlm_tasks.py` | VLM 客户端（缓存/日志/重试）与四项任务提示词 |
| `verify_overlay.py` | 质检图渲染（纯 OpenCV） |
| `triage_pdf.py` / `split_panels.py` / `batch_extract.py` | 分诊（含按栏分行、图注跨行合并）、子图切分、批量提取 |

## 🤝 贡献

欢迎 issue 与 PR。提交前请：

1. 跑一遍"开发与测试"中的回归（尤其是矢量零误差测试）
2. 新增启发式阈值请附实测数据说明，不要凭感觉定
3. 不要提交任何 API key、论文原文或提取出的受版权保护数据

## ⚖️ 许可

使用前请注意：提取他人论文中的数据涉及**版权与学术规范**，请遵守目标期刊与所在机构的规定，并在成果中正确引用数据来源。
