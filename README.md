# DIG Studio

**把论文 PDF 中的折线图、散点图提取为可核对的 CSV 数据。**

DIG Studio 面向需要复用论文图表数据的研究者。上传一篇或多篇 PDF 后，它会寻找图表、读取坐标轴、提取曲线点，并生成报告与“数据点叠加原图”的核对图。你可以先检查提取结果，再把 CSV 用于绘图或分析。项目也保留命令行入口，适合批量处理。

| 输入 | 处理 | 输出 |
| --- | --- | --- |
| 论文 PDF；可选图号、目标描述或结果模板 | 图表定位、坐标标定、曲线提取、质量核对 | 每条曲线的 CSV、核对图、Markdown 报告、坐标配置 |

支持 PDF 中的矢量图和位图，能处理多子图、部分双 Y 轴图表。矢量路径可直接从 PDF 读取；其他情况使用图像处理。本地 OCR 可以离线运行；启用 DeepSeek 视觉模型后，还能根据文字描述找图、辅助读轴和核对结果。照片、示意图与表格不属于当前提取目标。

> 提取结果是辅助数据，尤其遇到交叉曲线、低分辨率图片或模糊刻度时，应查看核对图和报告。项目没有针对其他工具做统一数据集上的精度基准测试。

## 部署与启动

需要 Python 3.10 或更新版本。以下命令在 Windows PowerShell 中运行：

```powershell
git clone https://github.com/Blankbalck-g/figure-digger.git
cd figure-digger
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS / Linux 使用 `python3 -m venv .venv`、`source .venv/bin/activate`，其余命令相同。启动后打开终端显示的地址，通常是 `http://localhost:8501`。

界面默认只监听本机。若部署到其他机器上并开放访问，请先在前面配置身份验证及访问限制，再通过 `--server.address 0.0.0.0` 显式开放监听；上传的论文、处理日志和结果会保存在运行机器的 `.ui_workspace/` 中。

## 使用图形界面

1. 上传 PDF。建议第一次先指定少数页试跑；一次上传多篇会自动使用批量模式。
2. 如果知道图号，填写 `Figure 5` 或 `图5`；留空会处理检测到的数据图。按图表内容描述目标时，选择“视觉模型（DeepSeek）”并提供 API Key。
3. 选择“自动提取”，运行完成后先查看“图像核对”，确认数据点贴合原曲线，再下载 CSV 或整包 ZIP。若坐标轴难以辨认，可选“仅分析，稍后确认坐标”。

本地 OCR 不需要 API Key。界面输入的 Key 只传给本次运行的子进程，不写入任务文件。你也可以通过 `DEEPSEEK_API_KEY` 环境变量、项目根目录下的 `deepseek_key.txt` 或 `secrets.json` 配置 Key；密钥文件已被 Git 忽略。视觉模型调用外部接口，会产生相应费用。

任务记录及产物保存在 `.ui_workspace/`，关闭浏览器后可在“任务结果”中重新查看。Git 不会提交这个目录。

## 命令行使用

图形界面调用的是同一个 `run.py` 流水线。以下示例可直接在项目根目录运行：

```bash
# 环境自检
python run.py doctor

# 单篇：先处理第 7 页；--force 表示接受自动读出的轴范围
python run.py all paper.pdf --out out --pages 7 --force

# 按图号选择（无需视觉模型）
python run.py all paper.pdf --out out --want "Figure 5" --force

# 按语义选图（需要已配置的 DeepSeek Key）
python run.py all paper.pdf --out out --vlm --want "速度随时间变化的曲线"

# 批量处理 papers/ 中的 PDF
python run.py batch papers --out out --jobs 4 --force
```

要人工确认轴范围，可以分两步：

```bash
python run.py analyze paper.pdf --out out
python run.py confirm out/paper_config.json --id fig5_p1 --x 0,5 --y 0,70
python run.py extract out/paper_config.json
```

实际配置文件名由 PDF 文件名决定；请使用 `analyze` 输出的路径。`--template format.dat` 可以额外生成指定模板格式的结果，首次编译模板需要视觉模型。完整选项运行 `python run.py <命令> --help` 查看。

## 结果在哪里

单篇任务的输出目录包含：

```text
out/
├── csv/                 每条曲线的数值数据
├── verify/              提取点与原图的叠加核对图
├── report.md            选图、坐标轴和提取结果说明
├── <文件名>_config.json  坐标范围与面板状态
└── figures/、panels/    从 PDF 裁出的图和子图
```

批量模式为每篇 PDF 建立子目录，并在根目录生成 `batch_summary.md`。图形界面可以直接预览、下载单个 CSV，也能下载本次任务的全部结果。`--keep-intermediates` 会保留额外诊断图片。

## 与同类项目的关系

以下比较基于各项目公开文档，侧重使用流程，没有进行同一数据集上的准确率或速度排名。

| 项目 | 主要输入与操作 | 适合的场景 |
| --- | --- | --- |
| **DIG Studio** | 论文 PDF；自动定位图表、提取曲线、批量导出并生成核对图 | 从多篇论文中提取并审阅折线/散点数据 |
| [WebPlotDigitizer](https://github.com/automeris-io/WebPlotDigitizer) | 以图表图像为中心的视觉辅助数字化工作流 | 需要在专门的数字化工具里处理多种图表图像 |
| [PlotRedox](https://github.com/elechou/PlotRedox) | 桌面应用；导入图像、四点标定、手动或视觉辅助选点，可保存项目文件 | 需要细致编辑点位与维护可继续修改的项目 |
| [PlotDigitizer](https://github.com/anadb/PlotDigitizer) | 图像输入；网页画布可缩放、移动和拖拽曲线点，导出 CSV | 需要在网页里手动修正识别出的点 |

DIG 当前的优势是从完整 PDF 到批量 CSV 与核对材料的流水线。需要逐点交互编辑时，上表的画布类工具更合适；DIG 的人工修正目前通过 `confirm` 和 `extract` 命令完成。

## 项目结构与来源

- `app.py`：本地图形界面。
- `ui_utils.py`：界面与命令行之间的参数、文件和结果处理。
- `run.py`：分析、提取、人工确认、批量处理和环境自检入口。
- `dig/`：PDF 分析、曲线提取、坐标标定、视觉模型、报告等核心模块。
- `tests/`：回归脚本与合成矢量 PDF 测试素材。

本仓库基于 [Jeff-920/figure-digger](https://github.com/Jeff-920/figure-digger) 改进，保留原项目 MIT 许可证。欢迎在核对真实论文结果后提交问题或改进。

## 调试与常见问题

| 现象 | 处理办法 |
| --- | --- |
| 页面打不开 | 确认 `python -m streamlit run app.py` 仍在运行；打开终端显示的本地地址。 |
| 缺少 Python 依赖 | 激活安装依赖时使用的虚拟环境，再运行 `python -m pip install -r requirements.txt` 和 `python run.py doctor`。 |
| 启用了视觉模型却提示没有 Key | 在界面填写 API Key，或设置 `DEEPSEEK_API_KEY`。无需视觉模型时切回“本地 OCR”。 |
| 成功运行却没有 CSV | 查看 `report.md` 中的选图、跳过原因和坐标状态；OCR 模式下可重新确认轴范围后运行 `extract`。 |
| 识别出的点偏离原图 | 先检查 `verify/` 中的核对图；缩小页码范围，或改用“仅分析”并手动修正坐标。 |
| 批量递归处理存在同名 PDF | 不同目录下的同名文件可能写入同一结果子目录；先重命名或分批运行。 |

开发时可先安装 `pytest`，再运行 `python -m pytest -q` 做界面辅助函数检查。`tests/test_vlm_mock.py`、`tests/test_vlm_pipeline.py` 和 `tests/test_panel_agent_plan.py` 需要另行提供测试图片或论文；运行时把路径作为第一个参数传入。
