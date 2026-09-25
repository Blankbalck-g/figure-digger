"""DIG Studio — visual interface for the figure-digger pipeline."""

from __future__ import annotations

import csv
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from ui_utils import RunOptions, build_command, collect_artifacts, make_archive, safe_filename


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / ".ui_workspace"

st.set_page_config(
    page_title="DIG Studio · 图表数据提取",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#10243e; --muted:#64748b; --line:#dce6ef; --brand:#1967d2; --mint:#18a999; }
    .stApp { background: linear-gradient(135deg,#f7fbff 0%,#f8fafc 48%,#f2fbf8 100%); }
    [data-testid="stSidebar"] { background:#0e2239; }
    [data-testid="stSidebar"] * { color:#eaf2fb; }
    [data-testid="stSidebar"] input { color:#10243e !important; }
    .dig-hero { padding:2.1rem 2.4rem; border:1px solid rgba(25,103,210,.12); border-radius:24px;
      background:radial-gradient(circle at 90% 5%,rgba(24,169,153,.22),transparent 34%),
                 linear-gradient(120deg,#fff,#f4f9ff); box-shadow:0 18px 55px rgba(16,36,62,.08); margin-bottom:1.4rem; }
    .dig-eyebrow { color:#1967d2; font-size:.76rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
    .dig-hero h1 { color:#10243e; font-size:clamp(2rem,4vw,3.8rem); line-height:1.02; letter-spacing:-.045em; margin:.55rem 0 .7rem; }
    .dig-hero p { color:#52657a; max-width:720px; font-size:1.02rem; margin:0; }
    .dig-chip { display:inline-block; margin:.9rem .45rem 0 0; padding:.34rem .7rem; border-radius:99px;
      background:#e8f1ff; color:#174e98; font-size:.78rem; font-weight:700; }
    .dig-card { background:rgba(255,255,255,.88); border:1px solid var(--line); border-radius:18px; padding:1rem 1.1rem; }
    .dig-step { color:#1967d2; font-size:.75rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    div[data-testid="stFileUploader"] { background:rgba(255,255,255,.78); border:1px dashed #9eb9d6; border-radius:18px; padding:.45rem; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem; }
    .stButton > button[kind="primary"] { border:0; border-radius:12px; font-weight:750;
      background:linear-gradient(100deg,#1967d2,#18a999); box-shadow:0 8px 22px rgba(25,103,210,.22); }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:.55rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _new_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = WORKSPACE / f"{stamp}-{uuid.uuid4().hex[:6]}"
    (run_dir / "input").mkdir(parents=True, exist_ok=False)
    (run_dir / "output").mkdir()
    return run_dir


def _save_uploads(uploaded_files, destination: Path) -> list[Path]:
    paths: list[Path] = []
    used: set[str] = set()
    for number, uploaded in enumerate(uploaded_files, start=1):
        name = safe_filename(uploaded.name)
        if name.lower() in used:
            stem, suffix = Path(name).stem, Path(name).suffix
            name = f"{stem}_{number}{suffix}"
        used.add(name.lower())
        path = destination / name
        path.write_bytes(uploaded.getvalue())
        paths.append(path)
    return paths


def _run(command: list[str], api_key: str, log_box) -> tuple[int, str, float]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    if api_key.strip():
        env["DEEPSEEK_API_KEY"] = api_key.strip()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.monotonic()
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=flags,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-120:]) or "正在启动…", language="text")
    return process.wait(), "\n".join(lines), time.monotonic() - started


def _read_csv(path: Path, limit: int = 200) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))[:limit]


def _result_view(output_dir: Path, elapsed: float | None = None) -> None:
    artifacts = collect_artifacts(output_dir)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("数据文件", len(artifacts["csv"]))
    col2.metric("质检图", len(artifacts["images"]))
    col3.metric("报告", len(artifacts["reports"]))
    col4.metric("运行耗时", f"{elapsed:.1f}s" if elapsed is not None else "—")

    if any(artifacts.values()):
        st.download_button(
            "下载全部结果（ZIP）",
            make_archive(output_dir),
            file_name=f"dig-results-{output_dir.parent.name}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    result_tabs = st.tabs(["数据预览", "提取报告", "质检画廊", "配置文件"])
    with result_tabs[0]:
        if not artifacts["csv"]:
            st.info("运行完成后，提取出的 CSV 会显示在这里。")
        for path in artifacts["csv"][:24]:
            with st.expander(str(path.relative_to(output_dir)), expanded=len(artifacts["csv"]) == 1):
                try:
                    st.dataframe(_read_csv(path), use_container_width=True, hide_index=True)
                except (OSError, UnicodeError, csv.Error) as exc:
                    st.warning(f"无法预览：{exc}")
                st.download_button("下载 CSV", path.read_bytes(), file_name=path.name,
                                   mime="text/csv", key=f"csv-{path}")
    with result_tabs[1]:
        if not artifacts["reports"]:
            st.info("暂无报告。")
        for path in artifacts["reports"]:
            st.caption(str(path.relative_to(output_dir)))
            st.markdown(path.read_text(encoding="utf-8", errors="replace"))
    with result_tabs[2]:
        if not artifacts["images"]:
            st.info("暂无质检图；完整提取后会在这里显示曲线重投影结果。")
        gallery = st.columns(2)
        for index, path in enumerate(artifacts["images"][:20]):
            gallery[index % 2].image(str(path), caption=str(path.relative_to(output_dir)),
                                     use_container_width=True)
    with result_tabs[3]:
        if not artifacts["configs"]:
            st.info("暂无配置文件。")
        for path in artifacts["configs"]:
            st.download_button(f"下载 {path.name}", path.read_bytes(), file_name=path.name,
                               mime="application/json", key=f"config-{path}")


with st.sidebar:
    st.markdown("## DIG Studio")
    st.caption("让论文中的曲线重新变成可计算的数据")
    st.divider()
    use_vlm = st.toggle("启用视觉模型", value=False, help="用于智能选图、读轴和结果抽检")
    api_key = st.text_input("DeepSeek API Key", type="password", disabled=not use_vlm,
                            help="只传给本次运行，不写入磁盘或仓库")
    with st.expander("模型连接", expanded=False):
        model = st.text_input("模型", value="deepseek-flash", disabled=not use_vlm)
        base_url = st.text_input("接口地址", value="https://api.deepseek.com", disabled=not use_vlm)
    st.divider()
    st.markdown("**处理精度**")
    dpi = st.select_slider("渲染 DPI", options=[150, 200, 300, 400, 600], value=300)
    use_ocr = st.toggle("OCR 交叉核对", value=not use_vlm,
                        help="启用视觉模型时通常无需开启；会显著增加耗时")
    keep_intermediates = st.toggle("保留诊断素材", value=False)
    verbose = st.toggle("详细日志", value=False)

st.markdown(
    """
    <section class="dig-hero">
      <div class="dig-eyebrow">Research data extraction workspace</div>
      <h1>从论文图像，到可信数据。</h1>
      <p>上传论文，描述你关心的图表。DIG 会完成选图、坐标标定、曲线追踪与可视化质检，结果随时可审计。</p>
      <span class="dig-chip">矢量无损读取</span><span class="dig-chip">多子图识别</span>
      <span class="dig-chip">CSV + 质检报告</span>
    </section>
    """,
    unsafe_allow_html=True,
)

task_tab, results_tab, guide_tab = st.tabs(["新建提取任务", "结果中心", "使用指南"])

with task_tab:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown('<div class="dig-step">01 · 输入论文</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("拖入一篇或多篇 PDF", type=["pdf"], accept_multiple_files=True,
                                   label_visibility="collapsed")
        st.markdown('<div class="dig-step">02 · 描述目标</div>', unsafe_allow_html=True)
        want = st.text_area(
            "你想提取什么？",
            placeholder="例如：800 bar 条件下，喷雾贯穿距随时间变化的曲线；也可以直接写 Figure 5",
            height=110,
        )
        template_upload = st.file_uploader(
            "结果模板（可选）", type=["txt", "dat", "csv"],
            help="上传已有数据格式，系统会按相同结构输出。需启用视觉模型。",
        )
    with right:
        st.markdown('<div class="dig-step">03 · 运行设置</div>', unsafe_allow_html=True)
        mode = st.radio("工作方式", ["一键分析并提取", "仅分析，稍后确认坐标轴"],
                        captions=["适合快速获得结果", "适合对数据精度要求极高的任务"])
        pages = st.text_input("限定页码（可选）", placeholder="例如 3,5,7")
        want_deep = st.toggle("结合正文理解图表", value=False, disabled=not use_vlm)
        find_series = st.toggle("自动识别需要的曲线", value=True, disabled=not use_vlm)
        force = st.toggle("自动确认识别到的坐标范围", value=True,
                          disabled=mode != "一键分析并提取")
        jobs = st.slider("批量并行数", 1, 8, 2, disabled=not uploads or len(uploads) < 2)
        can_run = bool(uploads) and not (mode.startswith("仅分析") and len(uploads) > 1)
        if mode.startswith("仅分析") and uploads and len(uploads) > 1:
            st.warning("仅分析模式一次处理一篇 PDF；请只保留一份文件。")
        if use_vlm and not api_key:
            st.caption("未填写 API Key 时会自动退回本地 OCR 路线。")
        run_clicked = st.button("开始提取", type="primary", use_container_width=True,
                                disabled=not can_run)

    if run_clicked:
        run_dir = _new_run_dir()
        input_paths = _save_uploads(uploads, run_dir / "input")
        template_path = None
        if template_upload is not None:
            template_path = run_dir / safe_filename(template_upload.name, "template.txt")
            template_path.write_bytes(template_upload.getvalue())
        options = RunOptions(
            dpi=dpi, pages=pages, force=force, use_vlm=use_vlm, use_ocr=use_ocr,
            want=want, want_deep=want_deep, find_series=find_series, verbose=verbose,
            keep_intermediates=keep_intermediates, jobs=jobs, model=model,
            base_url=base_url, template=template_path,
        )
        try:
            command = build_command(ROOT / "run.py", input_paths, run_dir / "output", options,
                                    analyze_only=mode.startswith("仅分析"))
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.status("正在解析论文与图表…", expanded=True) as status:
                log_box = st.empty()
                code, log_text, elapsed = _run(command, api_key, log_box)
                (run_dir / "ui-run.log").write_text(log_text, encoding="utf-8")
                if code == 0:
                    status.update(label="处理完成", state="complete", expanded=False)
                else:
                    status.update(label=f"运行失败（退出码 {code}）", state="error", expanded=True)
            st.session_state["last_result"] = {
                "output": str(run_dir / "output"), "elapsed": elapsed, "code": code,
            }
            if code == 0:
                st.success("结果已生成，可以在下方预览或下载。")
                _result_view(run_dir / "output", elapsed)
            else:
                st.error("请查看上方日志。常见原因是依赖、API Key 或 PDF 内容异常。")

with results_tab:
    last = st.session_state.get("last_result")
    if not last:
        st.info("完成一次任务后，最近的结果会保留在这里。")
    else:
        output = Path(last["output"])
        state = "成功" if last["code"] == 0 else "失败"
        st.caption(f"最近任务 · {state} · {output.parent.name}")
        _result_view(output, last["elapsed"])

with guide_tab:
    st.markdown(
        """
        ### 三步得到可复核的数据

        1. 上传论文 PDF。先用单篇、少数页验证效果，再批量处理。
        2. 用自然语言描述目标图；如果知道图号，直接写 `Figure 5` 最快。
        3. 下载 CSV，并查看“质检画廊”中数据点是否贴合原曲线。

        **本地 OCR** 不需要联网，适合图号明确、刻度清晰的图。**视觉模型**更适合按语义选图、
        复杂坐标轴和曲线识别。API Key 只注入当前子进程，不会保存到文件。

        如果轴范围需要人工修正，请选择“仅分析”，下载生成的配置文件，并继续使用原有
        `python run.py confirm ...` / `python run.py extract ...` 工作流。
        """
    )
