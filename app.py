"""DIG Studio: a local workbench for extracting chart data from PDFs."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
import uuid
from collections import deque
from datetime import datetime
from itertools import islice
from pathlib import Path

import streamlit as st

from ui_utils import RunOptions, build_command, collect_artifacts, make_archive, safe_filename


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / ".ui_workspace"

st.set_page_config(page_title="DIG Studio", page_icon="▦", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    :root { --ink:#18292f; --muted:#53676b; --rule:#d9e3e4; --teal:#176b64; }
    .stApp { background:#f6f8f8; color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1220px; padding-top:2.3rem; padding-bottom:4rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
    h1 { font-size:clamp(2.15rem,3.5vw,3.4rem); line-height:1.16; max-width:820px; }
    h2 { font-size:1.35rem; }
    .dig-top { display:flex; align-items:baseline; justify-content:space-between;
      border-bottom:1px solid var(--rule); padding:0 0 1rem; margin-bottom:2rem; }
    .dig-mark { font-size:1.05rem; font-weight:850; letter-spacing:-.055em; color:var(--ink); }
    .dig-mark span { color:var(--teal); font-weight:600; letter-spacing:0; }
    .dig-top small { color:var(--muted); font-size:.83rem; }
    .dig-intro { max-width:720px; color:var(--muted); font-size:1.05rem; line-height:1.7;
      margin:-.6rem 0 2rem; }
    div[data-testid="stFileUploader"] section { background:#fff; border:1.5px dashed #8faeac;
      border-radius:10px; min-height:95px; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--rule); }
    .stButton button[kind="primary"] { background:var(--teal); border:1px solid var(--teal);
      color:#fff; border-radius:8px; font-weight:700; min-height:2.9rem; }
    .stButton button[kind="primary"]:hover { background:#11564f; border-color:#11564f; }
    .stButton button:focus-visible, input:focus-visible, textarea:focus-visible {
      outline:3px solid #8bc9bf; outline-offset:2px; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--rule);
      border-radius:8px; padding:.8rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { border-bottom:1px solid var(--rule); gap:1rem; }
    .stTabs [data-baseweb="tab"] { font-weight:650; padding:.7rem .2rem; }
    @media (max-width:700px) { .block-container { padding-top:1rem; }
      .dig-top small { display:none; } .dig-intro { font-size:1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_run_dir() -> Path:
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = WORKSPACE / f"{label}-{uuid.uuid4().hex[:6]}"
    (run_dir / "input").mkdir(parents=True)
    (run_dir / "output").mkdir()
    return run_dir


def save_uploads(files, directory: Path) -> list[Path]:
    paths = []
    used: set[str] = set()
    for uploaded in files:
        base = safe_filename(uploaded.name)
        name = base
        count = 2
        while name.casefold() in used:
            name = f"{Path(base).stem}_{count}{Path(base).suffix}"
            count += 1
        used.add(name.casefold())
        path = directory / name
        path.write_bytes(uploaded.getvalue())
        paths.append(path)
    return paths


def run_pipeline(command: list[str], api_key: str, log_path: Path, log_slot) -> tuple[int, float]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    if api_key.strip():
        env["DEEPSEEK_API_KEY"] = api_key.strip()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.monotonic()
    tail: deque[str] = deque(maxlen=70)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=env, creationflags=flags,
            )
            assert process.stdout is not None
            last_draw = 0.0
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                if time.monotonic() - last_draw > 0.3:
                    log_slot.code("\n".join(tail), language="text")
                    last_draw = time.monotonic()
            code = process.wait()
    except OSError as exc:
        message = f"无法启动处理程序：{exc}"
        log_path.write_text(message, encoding="utf-8")
        tail.append(message)
        code = 1
    log_slot.code("\n".join(tail) or "处理程序没有输出日志。", language="text")
    return code, time.monotonic() - started


def recent_runs() -> list[tuple[str, dict]]:
    runs = []
    if not WORKSPACE.exists():
        return runs
    for path in WORKSPACE.glob("*/ui-run.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            runs.append((path.parent.name, metadata))
        except (OSError, ValueError):
            continue
    return sorted(runs, reverse=True)


def read_csv_preview(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(islice(csv.DictReader(handle), 200))


def has_vlm_key(entered: str) -> bool:
    if entered.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return True
    key_file = ROOT / "deepseek_key.txt"
    try:
        if any(line.strip() and not line.lstrip().startswith("#")
               for line in key_file.read_text(encoding="utf-8").splitlines()):
            return True
    except OSError:
        pass
    try:
        secrets = json.loads((ROOT / "secrets.json").read_text(encoding="utf-8"))
        return bool(str(secrets.get("deepseek_api_key") or secrets.get("api_key") or "").strip())
    except (OSError, ValueError, AttributeError):
        return False


def show_results(run_id: str, metadata: dict, key: str) -> None:
    output = WORKSPACE / run_id / "output"
    artifacts = collect_artifacts(output)
    state = "已完成" if metadata["code"] == 0 else "处理失败"
    st.subheader(f"{state} · {metadata.get('label', run_id)}")
    st.caption(f"任务 {run_id}  ·  原始文件和结果仅保存在本机 .ui_workspace 目录")
    metrics = st.columns(4)
    metrics[0].metric("CSV 文件", len(artifacts["csv"]))
    metrics[1].metric("核对图", len(artifacts["images"]))
    metrics[2].metric("报告", len(artifacts["reports"]))
    metrics[3].metric("耗时", f"{metadata.get('elapsed', 0):.1f} 秒")

    if any(artifacts.values()):
        st.download_button("下载本次全部结果 (.zip)", make_archive(output),
                           file_name=f"dig-{run_id}.zip", mime="application/zip",
                           key=f"{key}-archive")

    data_tab, image_tab, report_tab, detail_tab = st.tabs(
        ["数据", "图像核对", "报告", "日志与配置"]
    )
    with data_tab:
        if not artifacts["csv"]:
            st.info("暂无 CSV。如果选择了“仅分析”，请先确认坐标轴，再运行提取命令。")
        else:
            selected = st.selectbox("选择数据文件", artifacts["csv"],
                                    format_func=lambda p: str(p.relative_to(output)),
                                    key=f"{key}-csv")
            try:
                st.dataframe(read_csv_preview(selected), width="stretch",
                             hide_index=True)
                st.caption("最多预览前 200 行；下载文件包含全部数据。")
            except (OSError, UnicodeError, csv.Error) as exc:
                st.warning(f"预览失败：{exc}")
            st.download_button("下载当前 CSV", selected.read_bytes(), file_name=selected.name,
                               mime="text/csv", key=f"{key}-csv-download")
    with image_tab:
        if not artifacts["images"]:
            st.info("提取完成后，这里会显示曲线数据重投影到原图的核对图。")
        else:
            selected = st.selectbox("选择核对图", artifacts["images"],
                                    format_func=lambda p: str(p.relative_to(output)),
                                    key=f"{key}-image")
            st.image(str(selected), caption=str(selected.relative_to(output)),
                     width="stretch")
            st.caption("请检查提取点是否贴合原曲线，尤其是遮挡、交叉和坐标轴边缘。")
    with report_tab:
        if not artifacts["reports"]:
            st.info("暂无报告。")
        else:
            selected = st.selectbox("选择报告", artifacts["reports"],
                                    format_func=lambda p: str(p.relative_to(output)),
                                    key=f"{key}-report")
            st.markdown(selected.read_text(encoding="utf-8", errors="replace"))
    with detail_tab:
        for path in artifacts["configs"]:
            st.download_button(f"下载坐标配置 · {path.name}", path.read_bytes(),
                               file_name=path.name, mime="application/json",
                               key=f"{key}-config-{path}")
        log_path = output.parent / "ui-run.log"
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8", errors="replace")[-24000:],
                    language="text")


st.markdown(
    '<div class="dig-top"><div class="dig-mark">DIG <span>/ Studio</span></div>'
    '<small>PDF 图表数据提取工作台 · 本地运行</small></div>',
    unsafe_allow_html=True,
)
st.title("把论文里的曲线，变成可核对的数据。")
st.markdown(
    '<p class="dig-intro">上传 PDF，提取折线图和散点图中的坐标点。'
    '输出 CSV、提取报告与原图核对图，方便检查后再用于分析。</p>',
    unsafe_allow_html=True,
)

new_tab, history_tab, help_tab = st.tabs(["开始提取", "任务结果", "使用说明"])

with new_tab:
    input_col, settings_col = st.columns([1.35, 1], gap="large")
    with input_col:
        st.subheader("论文与目标")
        uploads = st.file_uploader("选择一篇或多篇 PDF", type=["pdf"],
                                   accept_multiple_files=True)
        if uploads:
            st.caption("本次文件：" + "、".join(item.name for item in uploads))
        want = st.text_area("想提取哪张图？", height=100,
                            placeholder="例如：Figure 5，或 800 bar 条件下速度随时间的曲线",
                            help="留空会处理检测到的全部数据图；按文字描述找图需要视觉模型。")
        st.caption("知道图号就填 Figure 5；没有特定目标可以留空。")
        template_upload = st.file_uploader("按模板导出（可选）", type=["txt", "dat", "csv"],
                                           help="此功能需要视觉模型。")

    with settings_col:
        st.subheader("处理方式")
        mode = st.radio("流程", ["自动提取", "仅分析，稍后确认坐标"],
                        help="自动提取会直接使用识别到的轴范围；请在结果页核对原图。")
        method = st.radio("读图方式", ["本地 OCR", "视觉模型（DeepSeek）"], horizontal=True)
        use_vlm = method.startswith("视觉模型")
        api_key = st.text_input("DeepSeek API Key", type="password", disabled=not use_vlm,
                                help="只用于本次子进程，不保存到任务目录。")
        has_key = has_vlm_key(api_key)
        if use_vlm and not has_key:
            st.warning("视觉模型需要 API Key。填写后即可运行；本地 OCR 不需要。")
        if want.strip() and not use_vlm and not re.search(r"(?:fig(?:ure)?\.?|图)\s*\d+", want, re.I):
            st.warning("按图的内容选图需要视觉模型。使用 OCR 时请填写图号，或留空处理全部图。")

        with st.expander("更多设置"):
            pages = st.text_input("指定页码", placeholder="如 3,5,7")
            dpi = st.select_slider("渲染精度 (DPI)", options=[150, 200, 300, 400, 600], value=300)
            jobs = st.slider("批量并行数", 1, 8, 2,
                             disabled=not uploads or len(uploads) < 2)
            use_ocr = st.toggle("视觉模型 + OCR 交叉核对", value=False,
                                disabled=not use_vlm)
            want_deep = st.toggle("结合正文核对图意", value=False, disabled=not use_vlm)
            find_series = st.toggle("自动识别曲线", value=True, disabled=not use_vlm)
            keep_intermediates = st.toggle("保留诊断图片", value=False)
            verbose = st.toggle("记录详细日志", value=False)
            model = st.text_input("模型名称", value="deepseek-flash", disabled=not use_vlm)
            base_url = st.text_input("接口地址", value="https://api.deepseek.com",
                                     disabled=not use_vlm)

        invalid_description = bool(want.strip() and not use_vlm and not re.search(
            r"(?:fig(?:ure)?\.?|图)\s*\d+", want, re.I))
        invalid_batch = bool(uploads and len(uploads) > 1 and mode.startswith("仅分析"))
        if invalid_batch:
            st.info("仅分析模式一次处理一篇 PDF。")
        if template_upload and not use_vlm:
            st.info("模板输出需要视觉模型。")
        can_run = bool(uploads) and not invalid_batch and not invalid_description and (
            not use_vlm or has_key) and (not template_upload or use_vlm)
        clicked = st.button("开始处理 PDF", type="primary", width="stretch",
                            disabled=not can_run)

    if clicked:
        run_dir = new_run_dir()
        input_paths = save_uploads(uploads, run_dir / "input")
        template_path = None
        if template_upload is not None:
            template_path = run_dir / safe_filename(template_upload.name, "template.txt")
            template_path.write_bytes(template_upload.getvalue())
        options = RunOptions(
            dpi=dpi, pages=pages, force=True, use_vlm=use_vlm, use_ocr=use_ocr,
            want=want, want_deep=want_deep, find_series=find_series,
            verbose=verbose, keep_intermediates=keep_intermediates,
            jobs=jobs, model=model, base_url=base_url, template=template_path,
        )
        try:
            command = build_command(ROOT / "run.py", input_paths, run_dir / "output", options,
                                    analyze_only=mode.startswith("仅分析"))
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.status("正在分析 PDF", expanded=True) as status:
                log_slot = st.empty()
                code, elapsed = run_pipeline(command, api_key, run_dir / "ui-run.log", log_slot)
                status.update(label="处理完成" if code == 0 else "处理失败",
                              state="complete" if code == 0 else "error", expanded=code != 0)
            metadata = {"code": code, "elapsed": elapsed,
                        "label": "、".join(path.name for path in input_paths),
                        "mode": mode}
            (run_dir / "ui-run.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            show_results(run_dir.name, metadata, "new")

with history_tab:
    runs = recent_runs()
    if not runs:
        st.info("完成一次处理后，结果会显示在这里；关闭浏览器后也可再次查看。")
    else:
        choices = {run_id: metadata for run_id, metadata in runs}
        selected_id = st.selectbox("本机任务", list(choices),
                                   format_func=lambda item: f"{item}  ·  {choices[item].get('label', '')}",
                                   key="history-selection")
        show_results(selected_id, choices[selected_id], "history")

with help_tab:
    st.markdown(
        """
        ### 怎么使用

        上传 PDF 后，直接点“开始处理 PDF”。建议先选少数页试跑。知道图号时填写 `Figure 5`；
        想按图中含义查找，需要启用视觉模型并提供 API Key。结果页先看“图像核对”，再下载 CSV。

        ### 什么时候选“仅分析”

        坐标轴较模糊或数值要求很严时，先生成配置文件，再用命令行确认范围：

        ```bash
        python run.py confirm <配置文件> --id <面板ID> --x 0,5 --y 0,70
        python run.py extract <配置文件>
        ```

        上传文件、日志和结果保存在本机 `.ui_workspace/`；该目录不会提交到 Git。
        完整安装说明和命令行参数见项目 README。
        """
    )
