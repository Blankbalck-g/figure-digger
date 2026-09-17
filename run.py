"""End-to-end pipeline: paper PDF -> figures -> panels -> axis ranges -> data CSVs.

Two phases, because axis ranges must be confirmed by a human before data is trusted:

  python run.py analyze paper.pdf --out out/          # figures, panels, legend colours, OCR axis ranges
  python run.py confirm out/paper_config.json --id fig5_p1 --x 0,5 --y 0,70
  python run.py extract out/paper_config.json         # only confirmed panels
  python run.py all paper.pdf --out out/ --force      # one shot, trusting OCR as-is

With a VLM the axis reading replaces OCR (which is then off by default, it is 5-10x slower):

  python run.py all paper.pdf --out out/ --vlm
  python run.py batch ./papers --out ./out --vlm --jobs 4    # one process per paper

Outputs under --out:
  figures/     cropped figure images from the PDF
  panels/      individual panel crops (multi-panel figures)
  csv/         extracted data, one CSV per series
  verify/      reprojection overlays for visual QA
  paper_config.json   phase-1 result; edit "axis" and set "confirmed": true
  report.md           what was extracted, with the OCR evidence per axis
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Windows consoles default to GBK; make output UTF-8 before anything prints, including
# the dependency-check message below.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import cv2
    import imgio as iio
    import numpy as np
except ImportError as _exc:  # almost always "wrong Python environment"
    _bar = "=" * 72
    print(_bar)
    print(f"依赖导入失败：{_exc}")
    print(f"当前解释器  ：{sys.executable}")
    print()
    print("这通常不是代码问题，而是用错了 Python 环境——例如用了 conda 的 base，")
    print("而不是装好本项目依赖的那个环境。numpy「装了但导入失败」也会走到这里。")
    print()
    print("请先确认解释器，再激活正确环境：")
    print('    python -c "import sys; print(sys.executable)"')
    print("    conda activate pytorch          # 换成你装依赖的那个环境")
    print("    python run.py doctor            # 一键自检")
    print(_bar)
    raise SystemExit(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import axis_ranges as ar  # noqa: E402
import batch_extract as be  # noqa: E402
import extract_lines as el  # noqa: E402
import figure_index as fi  # noqa: E402
import legend_colors as lc  # noqa: E402
import split_panels as sp  # noqa: E402
import triage_pdf as tp  # noqa: E402
import verify_overlay as vo  # noqa: E402
import vector_extract as vx  # noqa: E402
import vlm_client as vlmc  # noqa: E402
import vlm_tasks as vt  # noqa: E402

# 并行批处理时子进程不往终端打（会互相穿插），而是各写一份 <out>/<pdf>/run.log。
_LOG_FILE = None


def log(msg):
    if _LOG_FILE is not None:
        _LOG_FILE.write(str(msg) + "\n")
        _LOG_FILE.flush()
    else:
        print(msg, flush=True)


def _elapsed(t0):
    """Human-readable elapsed time since t0."""
    d = time.time() - t0
    return f"{d:.1f}s" if d < 60 else f"{d / 60:.1f}min"


def detect_legend_colors(figure_img):
    """Colours of the coloured series, read from the figure's legend (any layout)."""
    gray = cv2.cvtColor(figure_img, cv2.COLOR_BGR2GRAY)
    best = None
    for box in lc.find_boxes(gray):
        sw = lc.find_swatches(figure_img, box)
        if sw and (best is None or len(sw) > len(best[1])):
            best = (box, sw)
    if best is None:
        return [], None
    box, sw = best
    colors = [s["hex"] for s in sw]
    # A photo has no legend: dozens of "swatches" are just colourful image content.
    if len(colors) > 8:
        return [], None
    return colors, box


def split_figure(figure_path, outdir, margin=None):
    """Return [(panel_image_path, frame_in_panel_coords, box_in_figure)] for a figure."""
    img = iio.imread(figure_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    boxes = sp.order_row_major(sp.find_panel_boxes(gray))
    if not boxes:
        # 一个绘图框都没检出：整图当一个面板处理。丢弃才是更糟的选择——
        # 位图路线后面还有"非数据图"门限会把它拦下，静默丢图则连机会都没有。
        try:
            frame = el.find_frame(gray)
        except Exception:  # noqa: BLE001
            frame = (0, 0, w, h)
        log(f"      （未检出子图框，按单面板处理 frame={tuple(frame)}）")
        return [(Path(figure_path), tuple(frame), (0, 0, w, h))]

    if len(boxes) == 1:
        x, y, bw, bh = boxes[0]
        return [(Path(figure_path), (x, y, x + bw, y + bh), (x, y, bw, bh))]

    if margin is None:
        # Generous margins matter: the y tick labels sit left of the axes box and a
        # narrow crop cuts them in half, which quietly ruins the y-axis OCR.
        margin = min(140, max(100, int(0.35 * min(b[3] for b in boxes))))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    panels = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
        crop = img[y0:y1, x0:x1]
        name = f"{Path(figure_path).stem}_p{i + 1:02d}.png"
        path = outdir / name
        iio.imwrite(path, crop)
        panels.append((path, (x - x0, y - y0, x + bw - x0, y + bh - y0), (x, y, bw, bh)))
    return panels


def curve_content(img, frame, legend_colors, max_dist=60):
    """How curve-like the pixels matching the legend colours are.

    A line chart draws continuous strokes: most columns of the plot area contain the
    colour and each such column holds a few pixels (stroke thickness). Photo/schematic
    "matches" are scattered single-pixel edges, and figures whose series do not span the
    axis (e.g. a handful of markers) score low on coverage.

    Returns (coverage, median_thickness) - both 0 when nothing matches.
    """
    left, top, right, bottom = frame
    sub = img[top + 2:bottom - 1, left + 2:right - 1].astype(float)
    if sub.size == 0:
        return 0.0, 0.0
    best = (0.0, 0.0)
    for hexs in legend_colors:
        target = np.array(el.hex_to_bgr(hexs), dtype=float)
        mask = np.sqrt(((sub - target) ** 2).sum(axis=2)) <= max_dist
        counts = mask.sum(axis=0)
        nz = counts[counts > 0]
        if not len(nz):
            continue
        coverage = len(nz) / mask.shape[1]
        thickness = float(np.median(nz))
        if (coverage, thickness) > best:
            best = (coverage, thickness)
    return best


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _overlap_frac(a, b):
    """Overlap area as a fraction of the smaller box (are these two boxes the same figure?)."""
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ox * oy / small if small > 0 else 0.0


def _vector_thumb(page, bbox, outdir, name, wanted):
    """Low-dpi crop of a vector region, only when a selection request needs one.

    The model picks figures from a contact sheet, so every candidate needs *some*
    picture; for a vector figure that picture has to be rendered first (cheap at 100
    dpi). Skipped entirely when nobody asked for a specific figure.
    """
    if not wanted:
        return None
    path = Path(outdir) / "select" / "thumbs" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.crop(tuple(bbox), strict=False).to_image(resolution=100).original.save(path)
    except Exception:  # noqa: BLE001
        return None
    return path


def _vector_inner_text(page, axes):
    """Free text from inside a vector chart: axis titles + legend/sub-plot labels."""
    if not axes:
        return ""
    try:
        t = vx.find_figure_text(page, axes)
    except Exception:  # noqa: BLE001
        return ""
    parts = [f"x轴:{t['x_title']}" if t["x_title"] else "",
             f"y轴:{t['y_title']}" if t["y_title"] else "",
             t["inside"][:80]]
    return " ｜ ".join(p for p in parts if p)


def _body_digest(pdf_path, want_pages=None, max_chars=12000):
    """正文里所有提到图的段落（不是全文）——按需才读。

    图注只说这张图"是什么"，正文才会说它"画的是什么、在什么工况下"：
    "Figure 5 presents the penetration at Pi = 800 bar"。这些句子只占全文的 3-12%，
    所以比灌全文便宜一个量级，而且是选图真正需要的那部分信息。
    """
    lines = []
    with tp.pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if want_pages is not None and i not in want_pages:
                continue
            lines += tp.figure_mentions(page)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（正文摘要已截断）"
    return text


def doctor():
    """Environment self-check: interpreter, dependencies (really importing them), key."""
    import importlib
    import importlib.util

    print("环境自检")
    print(f"  解释器      : {sys.executable}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  工作目录    : {Path.cwd()}")
    print()
    checks = [("numpy", "数组运算", True), ("cv2", "图像处理", True),
              ("pdfplumber", "PDF 解析", True), ("PIL", "图片编码", True),
              ("rapidocr_onnxruntime", "刻度 OCR（可选）", False)]
    bad = 0
    for mod, why, required in checks:
        ok = importlib.util.find_spec(mod) is not None
        detail = ""
        if ok:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail = f"  导入失败: {type(exc).__name__}: {exc}"
        if not ok:
            bad += 1 if required else 0
        tag = "✅" if ok else ("❌" if required else "⚠️ ")
        print(f"  {mod:<22} {tag} {why}{detail}")
    key = vlmc.load_api_key()
    print(f"  {'VLM key':<22} {'✅' if key else '⚠️ '} "
          + ("已配置（内容不会被打印）" if key else "未配置：VLM 功能不可用，其余流程正常"))
    # RapidOCR pulls these in transitively; a --no-deps install misses them and only
    # fails much later, deep inside the OCR engine.
    # 这些是容易被 --no-deps 安装漏掉的传递依赖；缺了往往要到很晚才报错，
    # 甚至只是"某个功能静默失效"（例如缺 pypdfium2 时所有图都导不出来）。
    ocr_deps = ["yaml", "shapely", "pyclipper", "onnxruntime", "six", "tqdm",
                "pypdfium2", "pdfminer"]
    missing = [m for m in ocr_deps if importlib.util.find_spec(m) is None]
    if missing:
        print(f"  {'OCR 传递依赖':<22} ⚠️  缺 {', '.join(missing)}"
              "  → pip install -r requirements.txt")
    print()
    if bad:
        print(f"❌ 有 {bad} 个必需依赖不可用。多半是环境不对——请激活装了依赖的环境"
              "（例如 conda activate pytorch）后重试。")
    else:
        print("✅ 必需依赖齐备，可以运行：python run.py all <pdf> --out out_dir [--vlm]")


def _norm_axis(value):
    """Accept {'min','max'} (VLM) or [lo, hi] (OCR/manual) -> [lo, hi] or None."""
    if isinstance(value, dict):
        lo, hi = value.get("min"), value.get("max")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = value
    else:
        return None
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    return [lo, hi] if hi > lo else None


def _norm_y_axes(vlm_result):
    """VLM may return a list of y axes (left/right) or a single 'y'.

    -> {side: {"range": [lo, hi], "labels": [...]}}
    """
    out = {}
    for item in (vlm_result.get("y_axes") or []):
        if not isinstance(item, dict):
            continue
        side = str(item.get("side") or "left").strip().lower()
        rng = _norm_axis(item)
        if rng:
            out[side] = {"range": rng, "labels": item.get("labels")}
    if not out:
        single = _norm_axis(vlm_result.get("y"))
        if single:
            out["left"] = {"range": single, "labels": None}
    return out


def merge_axis(vlm_axis, ocr_axis, tol_frac=0.02):
    """Decide the final axis range. The VLM is the primary reader; OCR is an
    independent cross-check and the fallback when the model is unavailable.

    Returns (values, confirmed, source, note).
    """
    v, o = _norm_axis(vlm_axis), _norm_axis(ocr_axis)
    if v and o:
        span = max(abs(v[1] - v[0]), 1e-9)
        if max(abs(v[0] - o[0]), abs(v[1] - o[1])) <= tol_frac * span:
            return v, True, "vlm+ocr一致", ""
        return v, True, "vlm", f"OCR 读数不同（VLM {v} vs OCR {o}），采用 VLM，建议抽检"
    if v:
        return v, True, "vlm", "OCR 未读出，仅 VLM 读数"
    if o:
        return o, False, "ocr", "VLM 未读出，回退 OCR（建议人工确认）"
    return None, False, "none", "两轴均未读出"


def print_vlm_usage(client, phase, before=None):
    """Print the VLM spend for a phase, taken from the API's own usage counters.

    Goes through log() so a parallel worker writes it into that paper's run.log
    instead of interleaving it into the shared terminal.
    """
    if client is None:
        return
    u = client.usage
    d = {k: u.get(k, 0) - (before or {}).get(k, 0) for k in u}
    bar = "─" * 66
    title = (f"VLM 用量（{phase}）" if phase.endswith(("合计", "总计"))
             else f"VLM 用量（{phase} 阶段）")
    log("\n" + bar)
    log(title)
    log(f"  调用次数     : {d['calls']} 次   （本地缓存命中 {d['cache_hits']} 次，不消耗 token）")
    line = f"  输入 tokens  : {d['prompt_tokens']:,}"
    if d["cache_hit_tokens"] or d["cache_miss_tokens"]:
        line += (f"   （上下文缓存命中 {d['cache_hit_tokens']:,} / "
                 f"未命中 {d['cache_miss_tokens']:,}）")
    log(line)
    log(f"  输出 tokens  : {d['completion_tokens']:,}")
    if d["calls"]:
        log(f"  每次平均     : 输入 {d['prompt_tokens'] / d['calls']:,.0f} / "
            f"输出 {d['completion_tokens'] / d['calls']:,.0f} tokens")
    p = vlmc.PRICES.get(client.model)
    if p:
        hit = d["cache_hit_tokens"]
        miss = d["cache_miss_tokens"] or max(0, d["prompt_tokens"] - hit)
        lo = (hit / 1e6 * p["in_hit"][0] + miss / 1e6 * p["in_miss"][0]
              + d["completion_tokens"] / 1e6 * p["out"][0])
        hi = (hit / 1e6 * p["in_hit"][1] + miss / 1e6 * p["in_miss"][1]
              + d["completion_tokens"] / 1e6 * p["out"][1])
        log(f"  预估费用     : ${lo:.4f} ~ ${hi:.4f}   （{client.model}，闲时~高峰，官方价）")
    log(f"  逐次明细     : {vlmc.LOG_PATH}")
    log(bar)


def analyze(pdf_path, outdir, dpi=300, vlm=None, pages=None, use_ocr=None, want=None,
            want_deep=False):
    """use_ocr=None means "auto": with a VLM the model is the primary axis reader and
    OCR (5-10x slower, same job) stays off; without one OCR is the only reader left."""
    if use_ocr is None:
        use_ocr = vlm is None
    t0 = time.time()
    pdf_path = Path(pdf_path).resolve()
    outdir = Path(outdir).resolve()
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "panels").mkdir(parents=True, exist_ok=True)
    if use_ocr:
        (outdir / "debug").mkdir(parents=True, exist_ok=True)
    elif vlm is not None:
        log("OCR 交叉核对已关闭（VLM 模式下默认关闭；要两路对照请加 --ocr）")

    with tp.pdfplumber.open(pdf_path) as pdf:
        want_pages = None
        if pages:
            want_pages = {int(v) for v in str(pages).replace(" ", "").split(",") if v}
        report = {
            "pdf": str(pdf_path),
            "pdf_stem": pdf_path.stem,
            "pages": [tp.analyse_page(p, i + 1) for i, p in enumerate(pdf.pages)
                      if want_pages is None or (i + 1) in want_pages],
        }
        report["exported"] = tp.export_figures(pdf, report, outdir, dpi)

        # Vector figures contain their own data: no rasterising, no OCR needed.
        vector_found = []
        vector_fallback = []   # 矢量区域读不出刻度 -> 渲染成位图，交给后面的位图路线
        exported_by_page = {}
        for e in report.get("exported", []):
            if e.get("file") and e.get("bbox"):
                exported_by_page.setdefault(e["page"], []).append(tuple(e["bbox"]))
        for pno, page in enumerate(pdf.pages, start=1):
            if want_pages is not None and pno not in want_pages:
                continue
            for j, reg in enumerate(vx.find_vector_regions(page), start=1):
                vid = f"{pdf_path.stem}_vec_p{pno}_{j}"
                res = vx.extract_vector_figure(page, reg["bbox"])
                if res.get("x_calibration") and res.get("y_calibration"):
                    vector_found.append({"page": pno, "reg": reg, "res": res, "id": vid,
                                         "inner": _vector_inner_text(page, res.get("axes")),
                                         "thumb": _vector_thumb(page, reg["bbox"], outdir, vid, want)})
                    continue
                # 区域里没有 PDF 自带的可读刻度文字。两种常见原因：刻度是路径而非文字；
                # 或者页面边框/分栏线与图并成了一个区域。前者渲染成位图还能救回来，后者不是图。
                bx0, by0, bx1, by1 = reg["bbox"]
                if (bx1 - bx0) > 0.9 * page.width and (by1 - by0) > 0.9 * page.height:
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：区域覆盖整页"
                        "（页面边框/分栏线被并了进来，不是数据图）")
                    continue
                if not res.get("axes"):
                    # 连坐标框都没有，只是一堆页面线条/表格线：回退成位图也读不出坐标轴
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：无坐标框"
                        "（页面线条/表格，非数据图）")
                    continue
                # 留出边距：区域边界常常正好压在坐标框上，紧贴裁剪会让后面的找框
                # 逻辑把框线当成图片边界而丢弃；另外刻度标签也需要这点空间。
                m = 24.0
                crop_box = (max(0.0, bx0 - m), max(0.0, by0 - m),
                            min(page.width, bx1 + m), min(page.height, by1 + m))
                if any(_overlap_frac(crop_box, eb) > 0.5
                       for eb in exported_by_page.get(pno, [])):
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：与已导出的位图图重叠，"
                        "同一张图不重复提取")
                    continue
                fb_path = outdir / "figures" / f"{vid}.png"
                try:
                    page.crop(crop_box, strict=False).to_image(
                        resolution=dpi).original.save(fb_path)
                except Exception as exc:  # noqa: BLE001
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：无刻度文字且渲染失败"
                        f"（{type(exc).__name__}: {exc}）")
                    continue
                # 坐标框在 PDF 里已经找到了，直接换算成像素给位图路线用：
                # 比自己再去图里猜一次框更可靠（刻度线贴边时猜框会失手）。
                scale = dpi / 72.0
                ax = res["axes"]
                frame = [int(round((ax[0] - crop_box[0]) * scale)),
                         int(round((ax[1] - crop_box[1]) * scale)),
                         int(round((ax[2] - crop_box[0]) * scale)),
                         int(round((ax[3] - crop_box[1]) * scale))]
                vector_fallback.append({
                    "path": fb_path, "page": pno, "frame": frame,
                    "note": f"矢量回退 bbox={[round(v, 1) for v in crop_box]}",
                    "id": vid, "bbox": reg["bbox"], "thumb": fb_path,
                })
                log(f"[矢量] p{pno} bbox={reg['bbox']} 无刻度文字 → "
                    f"渲染为位图改走图像路线：{fb_path.name}")

    captions_by_page = {}
    for p in report["pages"]:
        caps = [c for c in p["captions"] if c.get("kind") != "table"]
        if caps:
            captions_by_page[p["page"]] = caps

    config = {"pdf": str(pdf_path), "outdir": str(outdir), "panels": []}
    engine = ar.get_engine() if use_ocr else None   # 不跑 OCR 就别去加载识别引擎（要好几秒）
    vlm_usage0 = dict(vlm.usage) if vlm is not None else None

    # ---- 候选图清单：先把"每张图 + 它自己的图注"配好，再按需求筛 ----
    cards = [{"id": it["id"], "page": it["page"], "bbox": list(it["reg"]["bbox"]),
              "kind": "vector", "thumb": it["thumb"], "inner_text": it.get("inner", "")}
             for it in vector_found]
    cards += [{"id": it["id"], "page": it["page"], "bbox": list(it["bbox"]),
               "kind": "vector-fallback", "thumb": it["thumb"]} for it in vector_fallback]
    exported = []
    for exp in report["exported"]:
        if not exp.get("file"):
            # 不要静默跳过：渲染失败通常意味着缺依赖（如 pypdfium2），
            # 而症状只是"后面一个位图面板都没有"，极难排查。
            log(f"  ⚠ 第 {exp.get('page')} 页的图导出失败：{exp.get('error')}")
            if exp.get("error") and "pypdfium2" in str(exp["error"]):
                log("     ↑ 缺 pypdfium2（pdfplumber 渲染 PDF 用）。"
                    "修：pip install -r requirements.txt")
            continue
        exported.append({"path": Path(exp["file"]), "page": exp["page"],
                         "frame": None, "note": None})
        cards.append({"id": Path(exp["file"]).stem, "page": exp["page"],
                      "bbox": list(exp["bbox"]), "kind": "raster",
                      "thumb": Path(exp["file"])})
    for it in vector_fallback:
        exported.insert(0, {"path": it["path"], "page": it["page"], "frame": it["frame"],
                            "note": it["note"]})
    cards.sort(key=lambda c: (c["page"], round(c["bbox"][1]), round(c["bbox"][0])))
    fi.attach_captions(cards, captions_by_page, log=log)
    caption_of = {c["id"]: c["caption"] for c in cards}

    keep = None
    if want:
        decision = fi.select(want, cards, vlm=vlm, out_dir=outdir / "select", log=log,
                             body_text=lambda: _body_digest(pdf_path, want_pages),
                             deep=want_deep)
        keep = set(decision["ids"])
        config["selection"] = decision
        if not keep:
            log("\n" + "!" * 66)
            log(f"这篇里没有匹配「{want}」的图 —— 没有提取任何数据。")
            if decision.get("available"):
                log(f"  这篇实际有：{decision['available']}")
            if decision.get("reason"):
                log(f"  判断依据：{decision['reason']}")
            log("  候选清单（可用 --want 换个说法，或直接写图号）：")
            for c in decision["candidates"]:
                log(f"    {c['n']:>2}. {c['label'] or '(无编号)':<12} p{c['page']:<3} "
                    f"{c['caption'][:78]}")
            log("!" * 66)

    for it in vector_found:
        pno, reg, res = it["page"], it["reg"], it["res"]
        if keep is not None and it["id"] not in keep:
            continue
        # 只有"坐标框 + 刻度文字都读到了"的区域才会进 vector_found
        xcal, ycal = res["x_calibration"], res["y_calibration"]
        entry = {
            "id": it["id"],
            "page": pno,
            "method": "vector",
            "vector_bbox": reg["bbox"],
            "vector_objects": reg["objects"],
            "frame": res["axes"],
            "caption": caption_of.get(it["id"], ""),
            "legend_colors": [],
            "axis": {
                "x": vx.axis_range_from_calibration(xcal, res["axes"], "x"),
                "y": vx.axis_range_from_calibration(ycal, res["axes"], "y"),
                # values come from the PDF's own text, not OCR -> trust them
                "confirmed": True,
                "source": "vector-text",
            },
            "vector": {
                "x_labels": res["x_labels"], "y_labels": res["y_labels"],
                "x_r2": xcal["r2"], "y_r2": ycal["r2"],
                "x_positions": xcal.get("positions", "label-centres"),
                "y_positions": ycal.get("positions", "label-centres"),
                "series_count": len(res.get("series", [])),
            },
        }
        log(f"\n[矢量图] p{pno} bbox={reg['bbox']}  "
            f"x={entry['axis']['x']} y={entry['axis']['y']}  "
            f"{len(res.get('series', []))} 条路径序列（直接读取，无需 OCR）")
        config["panels"].append(entry)

    for fig in exported:                     # 矢量回退图排在最前，其余是导出的位图图
        fig_path, page, note = fig["path"], fig["page"], fig["note"]
        if keep is not None and fig_path.stem not in keep:
            continue
        fig_img = iio.imread(fig_path)
        if fig_img is None:
            continue
        caption = caption_of.get(fig_path.stem, "")
        log(f"\n[图] {fig_path.name}  (p{page})" + (f"  ← {note}" if note else ""))

        # ---- 第一步就交给 VLM 判断：不是数据图就整图跳过，省掉切分与 OCR 的开销 ----
        vlm_cls, name_map = None, {}
        if vlm is not None:
                try:
                    vlm_cls = vt.classify_chart(vlm, fig_path)
                    log(f"      VLM 分类: {vlm_cls.get('chart_type')}  "
                        f"子图={vlm_cls.get('panel_count')}  双Y轴={vlm_cls.get('dual_y_axis')}  "
                        f"可提取={vlm_cls.get('is_line_chart')}"
                        f"  （{str(vlm_cls.get('reason', ''))[:40]}）")
                    if vlm_cls.get("is_line_chart") is False:
                        log("      ⏭ 跳过：VLM 判定为非数据图（照片/示意图/表格），不再切分与 OCR")
                        continue
                    multi_y = bool(vlm_cls.get("dual_y_axis"))
                    if multi_y:
                        log("      ⚠ 检出双 Y 轴：将为每条曲线单独指派纵轴")
                except vlmc.VLMError as exc:
                    log(f"      VLM 分类失败: {exc}（继续走图像路线）")

        legend_colors, legend_box = detect_legend_colors(fig_img)
        log(f"      图例色: {legend_colors}")
        series_sides = {}
        if vlm is not None and legend_colors:
            try:
                nm = vt.name_series(vlm, fig_path, legend_colors)
                for s in nm.get("series", []):
                    col = (s.get("color") or "").upper()
                    if col and s.get("label"):
                        name_map[col] = s["label"]
                    if col and s.get("y_axis"):
                        series_sides[col] = str(s["y_axis"]).strip().lower()
                log("      VLM 系列命名: " + ", ".join(f"{k}->{v}" for k, v in name_map.items()))
                if series_sides:
                    log("      VLM 轴归属: " + ", ".join(f"{k}->{v}轴" for k, v in series_sides.items()))
            except vlmc.VLMError as exc:
                log(f"      VLM 命名失败: {exc}")

        if fig.get("frame"):
            # 矢量回退：坐标框已知，不必（也不该）再去图里猜子图
            panels = [(fig_path, tuple(fig["frame"]), (0, 0, fig_img.shape[1], fig_img.shape[0]))]
        else:
            panels = split_figure(fig_path, outdir / "panels")
        for idx, (panel_path, frame, box) in enumerate(panels, start=1):
            pid = f"{fig_path.stem}_p{idx}"
            log(f"  面板 {idx}/{len(panels)}: {panel_path.name} frame={frame}")
            entry = {
                "id": pid,
                "page": page,
                "figure": str(fig_path.relative_to(outdir)) if fig_path.is_relative_to(outdir) else str(fig_path),
                "panel_image": str(panel_path.relative_to(outdir)) if panel_path.is_relative_to(outdir) else str(panel_path),
                "frame": list(frame),
                "box_in_figure": list(box),
                "caption": caption,
                "legend_colors": legend_colors,
                "series_names": name_map,
                "vlm_classification": vlm_cls,
                "axis": {"x": None, "y": None, "confirmed": False},
            }
            # ---- 轴读数：VLM 先读（主），OCR 随后独立核对（辅）----
            va = None
            if vlm is not None:
                try:
                    va = vt.read_axis_ranges(vlm, panel_path)
                    y_axes = _norm_y_axes(va)
                    entry["vlm_axis"] = {"x": va.get("x"), "y_axes": va.get("y_axes"),
                                         "y": va.get("y"),
                                         "confidence": va.get("confidence"),
                                         "note": va.get("note")}
                    if y_axes:
                        entry["y_axes"] = y_axes
                    log("      VLM 轴读数: x=" + str(_norm_axis(va.get("x")))
                        + "  y=" + (", ".join(
                            f"{k}轴{[round(d['range'][0], 4), round(d['range'][1], 4)]}"
                            for k, d in y_axes.items()) or "None")
                        + f"  置信度={va.get('confidence')}")
                except vlmc.VLMError as exc:
                    log(f"      VLM 轴读数失败: {exc}（回退 OCR）")
                except Exception as exc:  # noqa: BLE001
                    log(f"      VLM 轴读数异常: {type(exc).__name__}: {exc}")

            ocr_ranges = {}
            if use_ocr:
                try:
                    ocr_ranges = ar.read_axis_ranges(panel_path, frame, engine=engine, log=log,
                                                     debug_crops=outdir / "debug")
                    entry["ocr"] = {k: v for k, v in ocr_ranges.items()
                                    if k in ("x", "y", "bands")}
                    ox = ocr_ranges.get("x", {}).get("range") if "x" in ocr_ranges else None
                    oy = ocr_ranges.get("y", {}).get("range") if "y" in ocr_ranges else None
                    log(f"      OCR 对照: x={ox} y={oy}")
                except Exception as exc:  # noqa: BLE001
                    log(f"      OCR 失败: {type(exc).__name__}: {exc}（不影响 VLM 读数）")
            else:
                log("      OCR 已跳过（VLM 模式下默认关闭）")

            # x 轴全面板共用；y 轴以"左轴"作为面板默认值，逐条曲线再按 VLM 的归属覆盖
            for axis_key in ("x", "y"):
                if axis_key == "x":
                    v_val = va.get("x") if va else None
                else:
                    y_axes = (entry.get("y_axes") or {})
                    first = y_axes.get("left") or (list(y_axes.values())[0] if y_axes else None)
                    v_val = first["range"] if first else None
                o_val = (ocr_ranges.get(axis_key, {}) or {}).get("range") \
                    if axis_key in ocr_ranges else None
                vals, conf, src, note = merge_axis(v_val, o_val)
                if vals:
                    entry["axis"][axis_key] = vals
                entry[f"axis_src_{axis_key}"] = src
                if note:
                    entry.setdefault("notes", []).append(f"{axis_key}: {note}")
                if conf:
                    entry["axis"]["confirmed"] = True
            log(f"      最终轴范围: x={entry['axis'].get('x')} y={entry['axis'].get('y')}"
                f"  来源={entry.get('axis_src_x')}/{entry.get('axis_src_y')}"
                f"  已确认={entry['axis'].get('confirmed')}")
            if entry.get("y_axes") and len(entry["y_axes"]) > 1:
                pairs = []
                for col, side in (series_sides or {}).items():
                    detail = entry["y_axes"].get(side) or {}
                    pairs.append(f"{col}→{side}轴{detail.get('range')}")
                log("      逐曲线轴指派: " + ("; ".join(pairs) if pairs else "（VLM 未给出归属，统一用左轴）"))
            entry["series_axes"] = series_sides

            # Not-a-line-chart filter: needs both no readable ticks AND no curve-like
            # pixel content. Photos/schematics fail on both counts; a line chart only
            # has to pass one of them.
            coverage, thickness = curve_content(fig_img, frame, legend_colors)
            entry["curve_content"] = {"coverage": round(coverage, 3),
                                      "median_thickness": round(thickness, 2)}
            curve_ok = coverage >= 0.75 and thickness >= 2
            ocr = entry.get("ocr", {})
            ocr_ok = all(
                ocr.get(a) and ocr[a]["n_labels"] >= 4 and ocr[a]["r2"] >= 0.9995
                for a in ("x", "y"))
            # 第二重证据 = 两轴范围确实被读出来了。VLM 读的算数（它是主读数来源，
            # "是不是数据图"已由最前面的分类把关），OCR 读的按拟合质量算数。
            # 注意：开 OCR 做交叉核对不该让判定变得更严——那样 VLM 读对的图反被丢掉。
            vlm_ok = vlm is not None and all(
                s in ("vlm", "vlm+ocr一致")
                for s in (entry.get("axis_src_x"), entry.get("axis_src_y")))
            if not (curve_ok or vlm_ok or ocr_ok):
                entry["skip"] = True
                entry["skip_reason"] = (
                    f"非折线图（曲线覆盖率 {coverage:.2f}/厚度 {thickness:.1f} 未达标，"
                    f"且没有可信的轴读数；"
                    f"疑似照片/示意图/表格）")
                log(f"      跳过：{entry['skip_reason']}")
            elif not curve_ok:
                entry["needs_review"] = True
                log(f"      注意：曲线覆盖率 {coverage:.2f}、厚度 {thickness:.1f}，"
                    f"系列可能不连续或不是折线图，建议人工确认")
            config["panels"].append(entry)

    cfg_path = outdir / f"{pdf_path.stem}_config.json"
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(config, outdir / "report.md", phase="analyze")
    log(f"\n共 {len(config['panels'])} 个面板，耗时 {_elapsed(t0)} -> {cfg_path}")
    if want and not keep:
        log(f"没提取任何数据：这篇里没有匹配「{want}」的图（候选清单见上面与 report.md）。")
    elif want:
        log(f"按需求筛出 {len(keep)} 张图（其余图未处理）。可核对 report.md 里的选择依据。")
    else:
        log("下一步：核对 JSON 里的 axis 数值，把 confirmed 改成 true（或用 run.py confirm），再跑 extract")
    print_vlm_usage(vlm, "analyze", vlm_usage0)
    return cfg_path


def write_report(config, path, phase, results=None):
    lines = [f"# 提取报告 ({phase})", "", f"PDF: `{config['pdf']}`", ""]
    sel = config.get("selection")
    if sel:
        lines += ["## 本次需求", "",
                  f"- 需求: {sel.get('want')}",
                  f"- 判定方式: {sel.get('method')}",
                  f"- 命中: {len(sel.get('ids') or [])} 张图"]
        if sel.get("reason"):
            lines.append(f"- 依据: {sel['reason']}")
        if sel.get("body_used"):
            lines.append(f"- 正文通道: 查了 {sel.get('body_chars', 0)} 字符的"
                         f"「提到图的段落」" + (f"，依据: {sel['body_reason']}"
                                              if sel.get("body_reason") else ""))
        if sel.get("available"):
            lines.append(f"- 这篇实际有: {sel['available']}")
        lines.append("")
        if sel.get("candidates"):
            pre = sel.get("screened_n") and sel["screened_n"] < len(sel["candidates"])
            head = ["#", "图号", "页", "图注", "选中"] + (["进对照图"] if pre else [])
            lines += ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
            for c in sel["candidates"]:
                row = [str(c["n"]), c["label"] or "-", str(c["page"]),
                       c["caption"][:70].replace("|", "/"), "✅" if c["selected"] else ""]
                if pre:
                    row.append("●" if c.get("screened") else "—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    for p in config["panels"]:
        ax = p["axis"]
        if p.get("skip"):
            conf = "⏭️ 已跳过"
        else:
            conf = "✅ 已确认" if ax.get("confirmed") else "⚠️ 未确认"
        lines += [
            f"## {p['id']}  (p{p['page']}, {conf})",
            "",
            f"- 提取方式: {'矢量（直接从 PDF 路径读取，无 OCR）' if p.get('method') == 'vector' else '位图（图例取色 + 图像处理）'}",
            f"- 图注: {p['caption'][:120] or '(未匹配到图注)'}",
            (f"- 矢量区域: {p['vector_bbox']}（第 {p['page']} 页）" if p.get("method") == "vector"
             else f"- 面板图: `{p.get('panel_image', '')}`"),
            f"- 坐标框: {p['frame']}",
            f"- 图例颜色: {', '.join(p['legend_colors']) or '(未检出)'}",
            f"- 轴范围: x={ax.get('x')}  y={ax.get('y')}",
        ]
        ocr = p.get("ocr", {})
        for axis in ("x", "y"):
            if axis in ocr:
                o = ocr[axis]
                lines.append(
                    f"  - {axis} 轴 OCR: 读到 {o['labels']} ({o['n_labels']} 个), "
                    f"R²={o['r2']}, 步长={o['step']}, 吸附={'是' if o['snapped'] else '否'}"
                    + (f", 离群={o['outliers']}" if o["outliers"] else "")
                    + (f", 已修复={o['repaired']}" if o["repaired"] else ""))
        if results and p["id"] in results:
            r = results[p["id"]]
            lines.append(f"- 提取结果: {len(r['series'])} 条曲线")
            for s in r["series"]:
                lines.append(f"  - `{s['file']}`  {s['points']} 点, "
                             f"y=[{s['y_range'][0]}, {s['y_range'][1]}]")
            lines.append(f"- 质检图: `verify/{p['id']}_verify.png`")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def make_verify_image(panel_path, frame, axis, legend_colors, series_files, out_path):
    """Thin wrapper so callers do not have to build the series tuples themselves."""
    series = []
    for csv_path, color_hex in zip(series_files, legend_colors):
        xs, ys = vo.read_csv(csv_path)
        series.append((Path(csv_path).stem, color_hex, xs, ys))
    vo.compose_overlay(panel_path, frame, tuple(axis["x"]), tuple(axis["y"]),
                       series, out_path)


def extract(cfg_path, force=False, only=None, vlm=None):
    t0 = time.time()
    cfg_path = Path(cfg_path).resolve()
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    vlm_usage0 = dict(vlm.usage) if vlm is not None else None
    outdir = Path(config.get("outdir") or cfg_path.parent)
    csv_dir = outdir / "csv"
    verify_dir = outdir / "verify"
    csv_dir.mkdir(parents=True, exist_ok=True)
    verify_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    pdf_doc = None
    for p in config["panels"]:
        if only and p["id"] not in only:
            continue
        ax = p["axis"]
        if p.get("skip"):
            log(f"跳过 {p['id']}（{p.get('skip_reason', '已标记跳过')}）")
            continue

        # ---- vector figure: read the polylines straight out of the PDF ----
        if p.get("method") == "vector":
            if pdf_doc is None:
                pdf_doc = tp.pdfplumber.open(config["pdf"])
            page = pdf_doc.pages[p["page"] - 1]
            vres = vx.extract_vector_figure(page, tuple(p["vector_bbox"]), csv_dir, p["id"])
            series = [{
                "file": s["file"], "points": s["n_points"], "color": s["color"],
                "x_range": [round(v, 6) for v in s["x_range"]],
                "y_range": [round(v, 6) for v in s["y_range"]],
            } for s in vres.get("series", [])]
            results[p["id"]] = {"series": series, "method": "vector",
                                "frame": vres.get("axes"), "axis_range": ax}
            log(f"{p['id']} [矢量·直接读取]: {len(series)} 条曲线")
            try:
                scale = 300 / 72.0
                bbox = p["vector_bbox"]
                page.crop(tuple(bbox), strict=False).to_image(resolution=300).original.save(
                    outdir / "figures" / f"{p['id']}.png")
                frame_c = [(p["frame"][0] - bbox[0]) * scale, (p["frame"][1] - bbox[1]) * scale,
                           (p["frame"][2] - bbox[0]) * scale, (p["frame"][3] - bbox[1]) * scale]
                make_verify_image(outdir / "figures" / f"{p['id']}.png", frame_c, ax,
                                  [s["color"] for s in series],
                                  [csv_dir / s["file"] for s in series],
                                  verify_dir / f"{p['id']}_verify.png")
            except Exception as exc:  # noqa: BLE001
                log(f"      （质检图渲染失败：{type(exc).__name__}: {exc}）")
            continue

        if not (ax.get("confirmed") or force):
            log(f"跳过 {p['id']}（轴范围未确认；核对后把 confirmed 设为 true，或用 --force）")
            continue
        if ax.get("x") is None or ax.get("y") is None:
            log(f"跳过 {p['id']}（缺少轴范围）")
            continue
        xr, yr = ax["x"], ax["y"]
        rng = [xr[0], xr[1], yr[0], yr[1]]
        # dual-axis figures: give each series the y range of the axis it reads
        series_y = {}
        tol_by_range = {}
        for col, side in (p.get("series_axes") or {}).items():
            detail = (p.get("y_axes") or {}).get(side) or {}
            r = detail.get("range")
            if r:
                series_y[col.upper()] = r
                step = None
                labels = detail.get("labels")
                if labels and len(labels) > 1:
                    vals = sorted(float(v) for v in labels if _is_number(v))
                    if len(vals) > 1:
                        step = min(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
                tol_by_range[tuple(r)] = vt.axis_tolerance(r, step)
        panel_path = Path(p["panel_image"])
        if not panel_path.is_absolute():
            panel_path = outdir / panel_path
        try:
            res = be.process_panel(panel_path, rng, csv_dir,
                                   sat_min=70, val_min=40, hue_tol=16,
                                   legend_colors=p["legend_colors"] or None,
                                   name_map=p.get("series_names") or None,
                                   series_y=series_y or None,
                                   # 图里已经知道的坐标框（PDF 里量的 / 子图切分检出的），
                                   # 只在自动找框失败时才用得上
                                   frame_hint=p.get("frame"))
        except Exception as exc:  # noqa: BLE001
            # 单个面板失败（典型：图里没有闭合坐标框）不该连累这篇的其余面板
            log(f"{p['id']}: ✗ 提取失败 {type(exc).__name__}: {exc}")
            results[p["id"]] = {"series": [], "error": f"{type(exc).__name__}: {exc}",
                                "axis_range": rng}
            continue
        res["axis_range"] = rng
        results[p["id"]] = res
        log(f"{p['id']}: {len(res.get('series', []))} 条曲线"
            + (f"  frame={res.get('frame')}" if res.get("series") else ""))
        if res.get("series"):
            files = [csv_dir / s["file"] for s in res["series"]]
            colors = [s.get("target_bgr") and "#{:02x}{:02x}{:02x}".format(
                s["target_bgr"][2], s["target_bgr"][1], s["target_bgr"][0]) or "#ff00ff"
                for s in res["series"]]
            make_verify_image(panel_path, res["frame"], ax, colors, files,
                              verify_dir / f"{p['id']}_verify.png")

        # ---- VLM spot check: independent re-reading of a few points ----
        if vlm is not None and res.get("series"):
            try:
                # Positions are described visually (not as our computed x values) so the
                # model's reading is a genuinely independent cross-check of the mapping.
                spots = [("横轴最左端（x 最小值处）", 0.0),
                         ("横轴正中间", 0.5),
                         ("横轴最右端（x 最大值处）", 1.0)]
                queries, expected = [], {}
                qid = 0
                for s in res["series"][:2]:
                    xs, ys = vo.read_csv(csv_dir / s["file"])
                    if not xs:
                        continue
                    rng = s.get("y_axis_range")
                    s_tol = tol_by_range.get(tuple(rng)) if rng else None
                    for where, frac in spots:
                        qid += 1
                        target = xs[0] + frac * (xs[-1] - xs[0])
                        i = min(range(len(xs)), key=lambda k: abs(xs[k] - target))
                        bgr = s.get("target_bgr")
                        hexs = ("#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])).upper() \
                            if bgr else None
                        side = (p.get("series_axes") or {}).get(hexs) if hexs else None
                        queries.append({"id": qid, "where": where, "tol": s_tol,
                                        "axis": side if len(p.get("y_axes") or {}) > 1 else None,
                                        "series": s.get("series_name") or s.get("label")})
                        expected[qid] = ys[i]
                if queries:
                    raw = vt.spot_check(vlm, panel_path, queries)
                    # tolerance from the chart's own tick spacing, not a % of the range
                    step = None
                    for src in (p.get("ocr", {}).get("y"), (p.get("vlm_axis") or {}).get("y")):
                        if isinstance(src, dict) and src.get("step"):
                            step = abs(src["step"]) if step is None else min(step, abs(src["step"]))
                        labels = (src or {}).get("labels") if isinstance(src, dict) else None
                        if labels and len(labels) > 1:
                            vals = sorted(float(v) for v in labels if _is_number(v))
                            if len(vals) > 1:
                                gaps = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
                                g = min(gaps)
                                step = g if step is None else min(step, g)
                    tol = vt.axis_tolerance(ax["y"], step)
                    tol_by_id = {q["id"]: q.get("tol") for q in queries}
                    cmp = vt.compare_spot_check(raw, expected, tol, tol_by_id)
                    res["spot_check"] = cmp
                    tols = sorted({c.get("tol") for c in cmp["checks"] if c.get("tol")})
                    log(f"      VLM 抽查: {cmp['n_agree']}/{cmp['n_compared']} 一致"
                        f"（容差 ±{'/'.join(f'{t:g}' for t in tols) or f'{tol:g}'}）"
                        + (f"  ⚠ 存在不一致" if cmp["n_compared"] and
                           cmp["n_agree"] < cmp["n_compared"] else ""))
            except vlmc.VLMError as exc:
                log(f"      VLM 抽查失败: {exc}")
            except Exception as exc:  # noqa: BLE001
                log(f"      VLM 抽查异常: {type(exc).__name__}: {exc}")

    (outdir / "extract_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if pdf_doc is not None:
        pdf_doc.close()
    write_report(config, outdir / "report.md", phase="extract", results=results)
    total = sum(len(r.get("series", [])) for r in results.values())
    log(f"\n共提取 {total} 条曲线，耗时 {_elapsed(t0)} -> {csv_dir}")
    log(f"质检图 -> {verify_dir}")
    log(f"报告 -> {outdir / 'report.md'}")
    print_vlm_usage(vlm, "extract", vlm_usage0)
    return results


def _run_one(pdf, sub, row, dpi, pages, force, vlm, use_ocr, want=None, want_deep=False):
    """One PDF end-to-end (analyze + extract). Shared by serial and parallel batch."""
    cfg = analyze(pdf, sub, dpi=dpi, vlm=vlm, pages=pages, use_ocr=use_ocr, want=want,
                  want_deep=want_deep)
    config = json.loads(Path(cfg).read_text(encoding="utf-8"))
    if config.get("selection"):
        row["matched"] = len(config["selection"].get("ids") or [])
    row["panels"] = len([p for p in config["panels"] if not p.get("skip")])
    row["panels_skipped"] = len([p for p in config["panels"] if p.get("skip")])
    results = extract(cfg, force=force, vlm=vlm) or {}
    row["curves"] = sum(len(r.get("series", [])) for r in results.values())
    for r in results.values():
        sc = r.get("spot_check") or {}
        row["spot_ok"] += sc.get("n_agree", 0)
        row["spot_n"] += sc.get("n_compared", 0)
    row["panels_extracted"] = len(results)
    return row


def new_row(pdf, sub):
    return {"pdf": Path(pdf).name, "outdir": str(sub), "curves": 0, "panels": 0,
            "panels_skipped": 0, "spot_ok": 0, "spot_n": 0}


def _log_row(row, with_vlm):
    u = row.get("usage") or {}
    log(f"  本文件小结: {row.get('curves', 0)} 条曲线 | 抽查 "
        f"{row['spot_ok']}/{row['spot_n']} | "
        + (f"VLM {u.get('calls', 0)} 次调用/{u.get('prompt_tokens', 0)} tokens"
           if with_vlm else "未启用 VLM")
        + f" | 耗时 {row.get('elapsed', '?')}s"
        + (f" | ✗ {row['error']}" if row.get("error") else ""))


def _batch_one(pdf_str, sub_str, dpi, pages, force, vlm_spec, use_ocr, want=None,
               want_deep=False):
    """Worker body for --jobs > 1: one PDF per process.

    Output goes to that paper's own run.log - several processes printing to one terminal
    interleaves into unreadable text. VLM usage is returned with the row and summed by
    the parent, because a client built in a child cannot report back through pickling.
    """
    global _LOG_FILE
    pdf, sub = Path(pdf_str), Path(sub_str)
    sub.mkdir(parents=True, exist_ok=True)
    row = new_row(pdf, sub)
    t_file = time.time()
    client = None
    try:
        if vlm_spec:
            client = vlmc.DeepSeekVLM(model=vlm_spec["model"],
                                      base_url=vlm_spec["base_url"])
            if not client.api_key:
                client = None
        _LOG_FILE = (sub / "run.log").open("w", encoding="utf-8")
        _run_one(pdf, sub, row, dpi, pages, force, client, use_ocr, want, want_deep)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        log(f"  ✗ 失败: {row['error']}")
        if isinstance(exc, ImportError):
            log("     ↑ 缺依赖（常见于用 --no-deps 装的包）。修：pip install -r requirements.txt")
    finally:
        if _LOG_FILE is not None:
            _LOG_FILE.close()
            _LOG_FILE = None
    if client is not None:
        row["usage"] = dict(client.usage)
    row["elapsed"] = round(time.time() - t_file, 1)
    return row


def _batch_serial(pdfs, outdir, dpi, pages, force, vlm, use_ocr, want=None,
                  want_deep=False):
    rows = []
    for i, pdf in enumerate(pdfs, start=1):
        t_file = time.time()
        sub = outdir / pdf.stem
        log(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        before = dict(vlm.usage) if vlm is not None else None
        row = new_row(pdf, sub)
        try:
            _run_one(pdf, sub, row, dpi, pages, force, vlm, use_ocr, want, want_deep)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            log(f"  ✗ 失败: {row['error']}")
            if isinstance(exc, ImportError):
                log("     ↑ 缺依赖（常见于用 --no-deps 装的包）。"
                    "修：pip install -r requirements.txt")
        if before is not None:
            row["usage"] = {k: vlm.usage.get(k, 0) - before.get(k, 0) for k in vlm.usage}
        row["elapsed"] = round(time.time() - t_file, 1)
        rows.append(row)
        _log_row(row, before is not None)
    return rows


def _batch_parallel(pdfs, outdir, jobs, dpi, pages, force, vlm, use_ocr, want=None,
                    want_deep=False):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    spec = {"model": vlm.model, "base_url": vlm.base_url} if vlm is not None else None
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_batch_one, str(pdf), str(outdir / pdf.stem), dpi, pages,
                             force, spec, use_ocr, want, want_deep): pdf for pdf in pdfs}
        for fut in as_completed(futures):
            pdf = futures[fut]
            done += 1
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 (spawn/pickle failure)
                row = new_row(pdf, outdir / pdf.stem)
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            log(f"[{done}/{len(pdfs)}] {pdf.name}")
            _log_row(row, spec is not None)
    rows.sort(key=lambda r: r["pdf"])
    return rows


def batch(dir_path, outdir, dpi=300, vlm=None, pages=None, force=False,
          pattern="*.pdf", recursive=False, use_ocr=True, jobs=1, want=None,
          want_deep=False):
    """Process every PDF in a folder, then write a combined summary.

    jobs=1 keeps everything in one process (shared OCR engine / VLM client and caches).
    jobs>1 runs one paper per process: the VLM calls are network-bound, so this is where
    the wall-clock speedup comes from. Each paper logs to <out>/<pdf>/run.log.
    """
    src = Path(dir_path).resolve()
    t_batch = time.time()
    pdfs = sorted(src.rglob(pattern) if recursive else src.glob(pattern))
    pdfs = [p for p in pdfs if p.is_file()]
    if not pdfs:
        log(f"目录里没有匹配 {pattern} 的文件: {src}")
        return
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(jobs or 1))
    log(f"批量处理 {len(pdfs)} 个 PDF -> {outdir}"
        + (f"   并行 {jobs} 进程，每篇日志见 <out>/<pdf>/run.log" if jobs > 1 else "")
        + ("   [OCR 已关闭]" if not use_ocr else "")
        + (f"\n需求：{want}" if want else "")
        + "\n" + "=" * 66)

    if jobs > 1:
        rows = _batch_parallel(pdfs, outdir, jobs, dpi, pages, force, vlm, use_ocr, want,
                               want_deep)
        # 用量在子进程里累计，父进程按行加总后再打印
        if vlm is not None:
            for k in vlm.usage:
                vlm.usage[k] = sum(r.get("usage", {}).get(k, 0) for r in rows)
    else:
        rows = _batch_serial(pdfs, outdir, dpi, pages, force, vlm, use_ocr, want,
                             want_deep)

    md = ["# 批量提取汇总", "",
          f"- 来源目录: `{src}`", f"- 处理文件: {len(pdfs)} 个", ""]
    if vlm is not None:
        md += [f"- VLM 合计: {vlm.usage['calls']} 次调用（本地缓存命中 "
               f"{vlm.usage['cache_hits']} 次），输入 {vlm.usage['prompt_tokens']:,} / "
               f"输出 {vlm.usage['completion_tokens']:,} tokens", ""]
    extra_col = ["命中图"] if want else []
    md += ["| PDF | " + " | ".join(extra_col + ["面板", "跳过", "提取面板", "曲线",
                                                "抽查一致", "VLM调用", "输入tokens"]) + " |",
           "|---" * (7 + len(extra_col)) + "|"]
    for r in rows:
        u = r.get("usage") or {}
        name = r["pdf"] + (f" ⚠ {r['error']}" if r.get("error") else "")
        mid = f"{r.get('matched', '-')} | " if want else ""
        md.append(f"| {name} | {mid}{r['panels']} | {r['panels_skipped']} | "
                  f"{r.get('panels_extracted', 0)} | {r['curves']} | "
                  f"{r['spot_ok']}/{r['spot_n']} | {u.get('calls', '-')} | "
                  f"{u.get('prompt_tokens', '-')} |")
    tot_c = sum(r["curves"] for r in rows)
    tot_ok = sum(r["spot_ok"] for r in rows)
    tot_n = sum(r["spot_n"] for r in rows)
    md += ["", f"**合计**: {tot_c} 条曲线，抽查一致 {tot_ok}/{tot_n}"
           + (f"（{100 * tot_ok / tot_n:.0f}%）" if tot_n else "")]
    (outdir / "batch_summary.md").write_text("\n".join(md), encoding="utf-8")
    (outdir / "batch_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    log("\n" + "=" * 66)
    log(f"批量完成: {len(pdfs)} 个文件, 共 {tot_c} 条曲线, "
        f"抽查一致 {tot_ok}/{tot_n}，总耗时 {_elapsed(t_batch)}")
    log(f"汇总 -> {outdir / 'batch_summary.md'}")
    if tot_c == 0 and vlm is None and not force:
        log("提示：一条曲线都没提取。轴范围未经确认时 extract 会跳过所有面板——"
            "请加 --vlm（两路读数一致则自动确认）或 --force（直接信任 OCR 读数）。")
    elif tot_c == 0 and want:
        hit = [r["pdf"] for r in rows if r.get("matched")]
        log(f"提示：没有提取到曲线。需求「{want}」在"
            + (f" {len(hit)} 篇里命中（{', '.join(hit[:3])}…），但没追出曲线；"
               if hit else "任何一篇里都没有匹配的图；")
            + "各篇的候选清单见 <out>/<PDF>/report.md 的「本次需求」一节。")
    if vlm is not None:
        print_vlm_usage(vlm, "批量合计")


def confirm(cfg_path, panel_id, x, y, skip=False):
    cfg_path = Path(cfg_path).resolve()
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    hit = False
    for p in config["panels"]:
        if p["id"] != panel_id:
            continue
        hit = True
        if skip:
            p["skip"] = True
            p["skip_reason"] = "人工标记跳过"
        if x:
            p["axis"]["x"] = [float(v) for v in x.split(",")]
        if y:
            p["axis"]["y"] = [float(v) for v in y.split(",")]
        if p["axis"].get("x") and p["axis"].get("y"):
            p["axis"]["confirmed"] = True
            # an explicit confirmation overrides an automatic skip
            p.pop("skip", None)
            p.pop("skip_reason", None)
    if not hit:
        raise SystemExit(f"panel id not found: {panel_id}")
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"已更新 {panel_id}: x={x} y={y}" + ("  [已标记跳过]" if skip else ""))


def main():
    ap = argparse.ArgumentParser(description="PDF -> 图表数据 端到端流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="PDF -> 图片/面板/图例色/OCR 轴范围 + 配置")
    a.add_argument("pdf")
    a.add_argument("--out", required=True)
    a.add_argument("--dpi", type=int, default=300)
    a.add_argument("--pages", default=None, help="只处理指定页，如 7 或 3,5,7")
    a.add_argument("--vlm", action="store_true",
                   help="启用 DeepSeek VLM（分类/轴读数/命名），需配置 key")
    a.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    a.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    a.add_argument("--no-ocr", action="store_true",
                   help="不跑刻度 OCR（--vlm 时默认如此，比 OCR 快 5-10 倍）")
    a.add_argument("--ocr", action="store_true",
                   help="即使启用 VLM 也跑 OCR，用于两路交叉核对")
    a.add_argument("--want", default=None,
                   help="用一句话说要哪些图，如 --want \"800 bar 下速度随时间的折线图\"")
    a.add_argument("--want-deep", action="store_true",
                   help="再用正文里提到图的段落判定一次（更准，每篇约 ¥0.02）")

    e = sub.add_parser("extract", help="按配置提取数据（只跑已确认的面板）")
    e.add_argument("config")
    e.add_argument("--force", action="store_true", help="不检查 confirmed")
    e.add_argument("--only", nargs="*", default=None, help="只跑指定面板 id")
    e.add_argument("--vlm", action="store_true", help="提取后用 VLM 抽查校验")
    e.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    e.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)

    c = sub.add_parser("confirm", help="确认/修正某个面板的轴范围")
    c.add_argument("config")
    c.add_argument("--id", required=True)
    c.add_argument("--x", default=None)
    c.add_argument("--y", default=None)
    c.add_argument("--skip", action="store_true", help="标记该面板为跳过")

    al = sub.add_parser("all", help="analyze + extract（--force 时直接信任 OCR 数值）")
    al.add_argument("pdf")
    al.add_argument("--out", required=True)
    al.add_argument("--dpi", type=int, default=300)
    al.add_argument("--pages", default=None)
    al.add_argument("--force", action="store_true")
    al.add_argument("--vlm", action="store_true")
    al.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    al.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    al.add_argument("--no-ocr", action="store_true", help="不跑刻度 OCR")
    al.add_argument("--ocr", action="store_true", help="即使启用 VLM 也跑 OCR 交叉核对")
    al.add_argument("--want", default=None, help="用一句话说要哪些图（见 README）")
    al.add_argument("--want-deep", action="store_true", help="再用正文段落判定一次")

    b = sub.add_parser("batch", help="批量处理一个目录下的所有 PDF，并输出汇总")
    b.add_argument("dir")
    b.add_argument("--out", required=True)
    b.add_argument("--dpi", type=int, default=300)
    b.add_argument("--pages", default=None, help="只处理每篇的这些页（调试用）")
    b.add_argument("--pattern", default="*.pdf", help="文件名通配，默认 *.pdf")
    b.add_argument("--recursive", action="store_true", help="递归子目录")
    b.add_argument("--force", action="store_true")
    b.add_argument("--vlm", action="store_true")
    b.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    b.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    b.add_argument("--no-ocr", action="store_true", help="不跑刻度 OCR")
    b.add_argument("--ocr", action="store_true", help="即使启用 VLM 也跑 OCR 交叉核对")
    b.add_argument("--jobs", type=int, default=1,
                   help="并行进程数，每篇一个进程（默认 1=串行）")
    b.add_argument("--want", default=None,
                   help="用一句话说要哪些图；每篇都按同一句需求筛")
    b.add_argument("--want-deep", action="store_true", help="再用正文段落判定一次")

    sub.add_parser("doctor", help="环境自检：解释器、依赖、API key")

    args = ap.parse_args()

    def make_vlm():
        if not getattr(args, "vlm", False):
            return None
        client = vlmc.DeepSeekVLM(model=args.vlm_model, base_url=args.vlm_base_url,
                                  verbose=True)
        if not client.api_key:
            log("⚠ 指定了 --vlm 但没找到 API key："
                "请把 key 写入 fig-extract/deepseek_key.txt 或设置 DEEPSEEK_API_KEY。"
                "本次将只走 OCR / 图像处理路线。")
            return None
        log(f"VLM 已启用：model={client.model}  base_url={client.base_url}")
        return client

    def use_ocr():
        """OCR 是 VLM 的交叉核对；开了 VLM 就默认不跑（慢 5-10 倍，收益很小）。"""
        if getattr(args, "ocr", False):
            return True
        if getattr(args, "no_ocr", False):
            return False
        return not getattr(args, "vlm", False)

    if getattr(args, "no_ocr", False) and not getattr(args, "vlm", False):
        log("⚠ 指定了 --no-ocr 但没启用 --vlm：轴范围将没有任何自动读数来源，"
            "面板会因缺少轴范围被跳过（只能靠 run.py confirm 手填）。")
    if getattr(args, "want", None) and not getattr(args, "vlm", False):
        log("⚠ 指定了 --want 但没启用 --vlm：只能按图号匹配（--want \"Figure 5\"）；"
            "按图的内容找图需要模型（--vlm）。")

    if args.cmd == "analyze":
        analyze(args.pdf, args.out, args.dpi, vlm=make_vlm(), pages=args.pages,
                use_ocr=use_ocr(), want=args.want, want_deep=args.want_deep)
    elif args.cmd == "extract":
        extract(args.config, force=args.force,
                only=set(args.only) if args.only else None, vlm=make_vlm())
    elif args.cmd == "confirm":
        confirm(args.config, args.id, args.x, args.y, args.skip)
    elif args.cmd == "all":
        client = make_vlm()
        cfg = analyze(args.pdf, args.out, args.dpi, vlm=client, pages=args.pages,
                      use_ocr=use_ocr(), want=args.want, want_deep=args.want_deep)
        extract(cfg, force=args.force, vlm=client)
        if client is not None:
            print_vlm_usage(client, "本次合计")
    elif args.cmd == "batch":
        batch(args.dir, args.out, dpi=args.dpi, vlm=make_vlm(), pages=args.pages,
              force=args.force, pattern=args.pattern, recursive=args.recursive,
              use_ocr=use_ocr(), jobs=args.jobs, want=args.want,
              want_deep=args.want_deep)
    elif args.cmd == "doctor":
        doctor()


if __name__ == "__main__":
    main()
