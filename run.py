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

# 库模块都在 dig/ 里，先把它加进搜索路径。要在下面的 import 之前做，
# 否则 imgio/numpy 的"环境不对"误报会先跳出来（模块之间仍是扁平 import）。
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dig"))

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
import template_out as tpl  # noqa: E402

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
    # 轮廓框 + L 形坐标轴框：只画了左轴/下轴的子图（很常见）没有闭合轮廓，
    # 单靠轮廓法会退回"曲线围出的空洞"，把整幅子图的数据丢掉。
    boxes = sp.order_row_major(sp.merge_panel_boxes(sp.find_panel_boxes(gray),
                                                    sp.find_spine_boxes(gray)))
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


def _dark_line_wanted(panel):
    """黑线提取默认关闭，只有面板里显式写了 allow_dark=true 才做。

    实测代价：黑线通道在这些论文里会把 "EOI"、"11 MPa"、"S ∝ t^0.5" 这些黑字串成
    一条曲线输出（2014-01-9079 / 2014-01-1413 几乎每个面板都多出 dark/dark_2 两条
    假曲线），比"黑线提不出来"更糟。所以改成显式开启：分析阶段若发现深色数据线会在
    日志里提示，用户要把该面板的 allow_dark 改成 true 才会提取。
    """
    if not panel.get("allow_dark"):
        return False
    info = panel.get("vlm_dark_line") or {}
    if not info.get("has_dark_line"):
        return False
    kind = str(info.get("kind") or "").strip().lower()
    return kind in ("", "data", "model", "data_line", "model_line", "experiment")


def _extract_panel(panel, panel_path, rng, csv_dir, vlm=None):
    """两种方式：模型给锚点 + 代码精确追踪（优先），或按颜色全局追踪（回退）。

    锚点路线的分工：模型说"有哪几条曲线、大概在哪"，代码在锚点附近取色、从锚点向
    两端连续性追踪、换算成数值。模型不再评价候选清单，所以不会出现"数据被跳过"；
    追不到的会作为 failed 明确报告，而不是静默消失。
    """
    spec = panel.get("series_spec") or []
    if spec:
        try:
            res = _extract_seeded(panel, panel_path, rng, csv_dir, spec, vlm=vlm)
            if res is not None:
                return res
        except Exception as exc:  # noqa: BLE001
            log(f"      （锚点路线失败，回退颜色追踪：{type(exc).__name__}: {exc}）")
    try:
        return be.process_panel(
            panel_path, rng, csv_dir, sat_min=70, val_min=40, hue_tol=16,
            legend_colors=panel["legend_colors"] or None,
            name_map=panel.get("series_names") or None,
            series_y=series_y_from(panel),
            roles=panel.get("series_roles") or None,
            allow_dark=_dark_line_wanted(panel),
            frame_hint=panel.get("frame"))
    except Exception as exc:  # noqa: BLE001
        log(f"{panel['id']}: ✗ 提取失败 {type(exc).__name__}: {exc}")
        return None


def _gap_crop(img, trace, x0, x1, others, color, out_path):
    """裁出缺口那一小块，并把"已识别部分"画上去，给模型看走势。"""
    h, w = img.shape[:2]
    pad_x = max(12, int(0.03 * w))
    cx0, cx1 = max(0, int(x0 - pad_x)), min(w, int(x1 + pad_x))
    ys = [y for x, y in trace if cx0 <= x <= cx1]
    for _n, t in others:
        ys += [y for x, y in t if cx0 <= x <= cx1]
    if not ys:
        ys = [y for _x, y in trace]
    pad_y = max(14, int(0.25 * (max(ys) - min(ys) + 1)))
    cy0, cy1 = max(0, int(min(ys) - pad_y)), min(h, int(max(ys) + pad_y))
    if cx1 - cx0 < 12 or cy1 - cy0 < 12:
        return None
    crop = img[cy0:cy1, cx0:cx1].copy()
    for _n, t in others:                       # 别的曲线：灰
        pts = [(int(x - cx0), int(y - cy0)) for x, y in t if cx0 <= x < cx1]
        if len(pts) > 1:
            cv2.polylines(crop, [np.array(pts, np.int32)], False, (150, 150, 150), 1)
    pts = [(int(x - cx0), int(y - cy0)) for x, y in trace if cx0 <= x < cx1]
    if len(pts) > 1:                           # 已识别的部分：本色
        cv2.polylines(crop, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
    iio.imwrite(out_path, crop)
    return (cx0, cy0, cx1, cy1)


def _repair_gaps(vlm, panel, panel_path, picks, names, outdir, log, max_calls=4):
    """重合处只显示一种颜色 -> 追断的那几段：让模型给锚点，代码插回去。

    只在"这条曲线明显比别人短 / 中间有长断口"时才问，每次只裁那一小块。模型读图、说锚点；
    锚点只往已有缺口里插，已经追到的部分不动（想编也编不进去）。
    """
    import series_seed as ss
    usable = [p for p in picks if p["trace"] and p["kind"] != "loop"]
    if vlm is None or len(usable) < 2:
        return 0
    img = iio.imread(panel_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    allx = [x for p in usable for x, _y in p["trace"]]
    lo, hi = min(allx), max(allx)
    if hi - lo < 0.25 * w:
        return 0
    calls, added = 0, 0
    shot_dir = Path(outdir) / "gaps"
    shot_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(usable, key=lambda q: -len(q["trace"])):
        if calls >= max_calls:
            break
        xs = [x for x, _y in p["trace"]]
        windows = []
        if min(xs) - lo > 0.05 * w:
            windows.append((lo, min(xs)))
        if hi - max(xs) > 0.05 * w:
            windows.append((max(xs), hi))
        for gap in ((p["completion"] or {}).get("open_gaps") or []):
            if gap["cols"] > 0.04 * w:
                windows.append((gap["x0"], gap["x1"]))
        for x0, x1 in windows[:2]:
            if calls >= max_calls or x1 - x0 < 0.04 * w:
                continue
            label = str(p["s"].get("label") or "曲线")
            color = (p["color"] or (0, 0, 0)) if not p["dark"] else (60, 60, 60)
            others = [(names[j], q["trace"]) for j, q in enumerate(picks) if q is not p
                      and q["trace"] and q["kind"] != "loop"]
            safe = ss.safe_tag(label)
            crop_path = shot_dir / f"{panel['id']}_{safe}_{int(x0)}_{int(x1)}.png"
            box = _gap_crop(img, p["trace"], x0, x1, others, color, crop_path)
            if box is None:
                continue
            calls += 1
            try:
                data = vt.gap_anchors(
                    vlm, crop_path, label,
                    "#%02x%02x%02x" % (color[2], color[1], color[0]),
                    info=(f"缺口在整张图的 x={int(x0)}~{int(x1)} 像素（图宽 {w}）"))
            except Exception as exc:  # noqa: BLE001
                log(f"      （补缺口调用失败：{type(exc).__name__}: {exc}）")
                continue
            cx0, cy0, cx1, cy1 = box
            got = []
            for a in (data.get("anchors") or []):
                try:
                    ax, ay = float(a[0]), float(a[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not (0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0):
                    continue
                px = cx0 + ax * (cx1 - cx0)
                py = cy0 + ay * (cy1 - cy0)
                if x0 - 4 <= px <= x1 + 4:
                    got.append((px, py))
            if not got:
                continue
            merged = ss.bridge_through_anchors(p["trace"], got)
            n_add = len(merged) - len(p["trace"])
            if n_add > 0:
                added += n_add
                p["completion"] = dict(p["completion"] or {})
                p["completion"]["model_anchors"] = {
                    "cols": n_add,
                    "by": f"模型读图补的锚点（{len(got)} 个，x={int(x0)}~{int(x1)}）"}
                p["trace"] = merged
                log(f"      模型补缺口：{label} x={int(x0)}~{int(x1)}，"
                    f"插了 {len(got)} 个锚点（+{n_add} 列）")
    return added


def _spec_dark(s, color):
    """模型说这条线是黑色/深灰（或它给的颜色本身就是深灰）。

    黑白论文图（实测 2006-01-1387 整本都是）里模型会说 "color=#000000"，旧实现
    直接判"颜色定不下来"把整本图的曲线全丢了——其实是代码不认黑白，不是图上没有
    数据。模型给了 dark=true，代码就改用"深色笔画 + 锚点"这条路线。
    """
    flag = s.get("dark")
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "1", "yes", "y")
    if flag:
        return True
    # 模型给的颜色本身是黑/深灰（黑白论文图整本都这样）。注意 resolve_color 会拒绝
    # 灰白色，所以这里必须看**模型给的原始十六进制**，不能只看解析结果。
    for candidate in (color, s.get("color")):
        if candidate is None:
            continue
        try:
            bgr = (el.hex_to_bgr(str(candidate)) if isinstance(candidate, str)
                   else candidate)
            _h, sat, val = el.bgr_to_hsv(bgr)
        except Exception:  # noqa: BLE001
            continue
        if val < 110 and sat < 70:
            return True
    return False


def _spec_want(s):
    """draw -> 要出什么。一个图例条目 = 一条曲线：

    markers_connected（散点用折线连起来）只出**点**——线就是这些点的连线，
    不再另出一个 `_line.csv`，否则用户看到的是"一条曲线变成了两条"。
    markers_plus_line（散点 + 独立的拟合/模型线）才是真的两条。
    """
    draw = str(s.get("draw") or "").strip().lower()
    if draw in ("markers_only", "markers_connected"):
        return "points"
    if draw == "line_only":
        return "line"
    if draw == "markers_plus_line":
        return "both"
    return str(s.get("output") or "line").strip().lower()


def _spec_has_markers(s):
    if s.get("has_markers") is True:
        return True
    if str(s.get("draw") or "").lower().startswith("markers"):
        return True
    ls = str(s.get("linestyle") or "").lower()
    return ls in ("marker", "markers", "scatter")


def _markers_cover_curve(markers, trace, min_n=5, cover=0.6):
    """这条曲线是不是"就是这些标记点连起来的"——标记够多、且铺满曲线的横向范围。

    实测（2023-01-1635 图 4 那种散点图）：每个圆点宽十几像素，逐列追踪会在它上面走出
    一条水平段，画出来就是明显的**阶梯**。标记中心才是真正的数据点，一个点一个值。
    """
    if not markers or len(markers) < min_n or len(trace) < 20:
        return False
    mx = [float(m[0]) for m in markers]
    tx = [float(q[0]) for q in trace]
    span_t, span_m = max(tx) - min(tx), max(mx) - min(mx)
    return span_t > 0 and span_m >= cover * span_t


REVIEW_COLORS = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 150, 255),
                 (255, 0, 190), (110, 60, 0), (0, 90, 255), (170, 0, 170)]


def revise_panel(vlm, panel_path, panel, spec, entries, failed, aliases, csv_dir=None):
    """复盘：把追出来的轨迹**编号画回原图**，让模型核对数量与归属。

    只做三件事的判定：哪两个编号是同一条（merge）、哪条追错了（drop）、漏了哪条
    （add）。模型像素上不准，但"这两条是不是同一条"这种判断是它的强项——这正是
    "算法在模型指导下收尾"。只在有问题的面板上调用，结果落盘可审计。
    """
    img = iio.imread(panel_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    rows = []
    for i, e in enumerate(entries, start=1):
        p, s = e["p"], e["p"]["s"]
        color = REVIEW_COLORS[(i - 1) % len(REVIEW_COLORS)]
        pts = [(int(round(x)), int(round(y))) for x, y in p["trace"]]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 3, cv2.LINE_AA)
        if pts:
            mx, my = pts[len(pts) // 2]
            mx, my = min(max(mx + 4, 4), w - 60), min(max(my - 6, 16), h - 6)
            cv2.putText(img, f"#{i}", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(img, f"#{i}", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        color, 2, cv2.LINE_AA)
        data = e["line_data"] or e["point_data"]
        if data:
            xs = [q[0] for q in data]
            ys = [q[1] for q in data]
            rows.append(f"#{i}: {s.get('label')} | {len(data)} 点 | "
                        f"x=[{min(xs):.3g},{max(xs):.3g}] y=[{min(ys):.3g},{max(ys):.3g}] | "
                        f"颜色={'黑/灰' if p['dark'] else s.get('color')} | "
                        f"线型={s.get('linestyle')}")
        else:
            rows.append(f"#{i}: {s.get('label')} | （没有可用点）")
    for f in failed or []:
        a = next((q for q in (f.get("anchors") or []) if len(q) == 2), None)
        if not a:
            continue
        ax, ay = int(round(float(a[0]) * w)), int(round(float(a[1]) * h))
        if not (0 <= ax < w and 0 <= ay < h):
            continue
        cv2.drawMarker(img, (ax, ay), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
        cv2.putText(img, f"✗{str(f.get('label'))[:12]}", (ax + 10, ay + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    spec_text = "；".join(
        f"{s.get('label')}(draw={s.get('draw') or s.get('output')}"
        f"{', 黑线' if s.get('dark') else ''}{', 闭合' if s.get('closed') else ''})"
        for s in spec) or "（无）"
    base = Path(csv_dir).parent if csv_dir else panel_path.parent.parent
    outdir = base / "review"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{panel['id']}_review.png"
    iio.imwrite(png, img)
    ax = panel.get("axis") or {}
    axis_text = (f"x={ax.get('x')} y={ax.get('y')}"
                 + (f" 数量级乘数={ax.get('multiplier')}" if ax.get("multiplier") else "")
                 + (f"（来源 {panel.get('axis_src_x')}/{panel.get('axis_src_y')}）"
                    if panel.get("axis_src_x") else ""))
    try:
        data = vt.revise_traces(vlm, png, spec_text, "\n".join(rows), axis_text=axis_text)
    except Exception as exc:  # noqa: BLE001
        log(f"      （复盘调用失败，按原样输出：{type(exc).__name__}: {exc}）")
        return None
    (outdir / f"{panel['id']}_review.json").write_text(
        json.dumps({"spec": spec_text, "found": rows, "answer": data},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"      复盘：merge={data.get('merge') or []} drop={data.get('drop') or []} "
        f"add={len(data.get('add') or [])} axis_fix={'有' if data.get('axis_fix') else '无'}"
        f" —— {str(data.get('note'))[:60]}")
    return data


def _apply_review(review, entries, failed, aliases):
    """按复盘结论合并/丢弃（add 由调用方重追）。编号都是 1 起。"""
    n = len(entries)
    drop = {int(i) for i in (review.get("drop") or [])
            if str(i).strip().lstrip("-").isdigit() and 1 <= int(i) <= n}
    merged_into = {}
    for pair in (review.get("merge") or []):
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 1 <= a <= n and 1 <= b <= n and a != b:
            merged_into[max(a, b)] = min(a, b)
    out = []
    for idx, e in enumerate(entries, start=1):
        if idx in drop or idx in merged_into:
            keep = merged_into.get(idx)
            aliases.append({
                "label": e["p"]["s"].get("label") or f"series{idx}",
                "alias_of": (entries[keep - 1]["p"]["s"].get("label") if keep else None),
                "overlap": 1.0, "by": "模型复盘",
                "why": "drop" if idx in drop else "同一条曲线被重复提取"})
            continue
        out.append(e)
    return out, failed, aliases


def _extract_seeded(panel, panel_path, rng, csv_dir, spec, vlm=None):
    """"模型指路 + 代码追踪 + 模型复盘"；返回 None 表示这条路线不可用（回退）。

    分工（这一版把模型的理解用足，代码只做像素）：
      * 模型说：有哪几条曲线、一条还是两条（draw）、是不是闭合回线（closed）、
        是不是黑线（dark）、跟谁重合（overlaps）、哪几段看不见（occluded）+
        每条线的颜色与锚点；
      * 代码做：按模型给的颜色/深色掩膜做精确追踪、遮挡补全、坐标换算；
      * 追完之后再问一次模型（见 revise_panel）：把追出来的轨迹**编号画回原图**
        让它核对——哪两个编号是同一条、哪条追错了、漏了哪条。只在有问题的面板
        上触发，正常面板不多花 token。
    """
    import series_seed as ss
    img = iio.imread(panel_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = be._frame_of(gray, panel.get("frame"))
    exclude = be.inner_boxes(gray, frame, img=img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    xmin, xmax, ymin, ymax = rng
    palette = list(panel.get("legend_colors") or []) + \
        list((panel.get("series_names") or {}).keys())
    series, failed = [], []
    used_keys, accepted = set(), []
    width, height = img.shape[1], img.shape[0]

    def trace_one(s, hue_tol):
        """一条 spec -> (pick, 失败说明)。颜色/黑线两条路线在这里合流。"""
        anchors = ss.anchors_to_pixels(s.get("anchors"), frame, (width, height))
        color, how = ss.resolve_color(img, anchors, s.get("color"), palette)
        dark = _spec_dark(s, color)
        # closed 字段由模型给；老配置没有这个字段时退回几何判据（见 candidates_in_mask）
        allow_loop = bool(s["closed"]) if "closed" in s else True
        if dark:
            dmask = el.dark_mask(gray, frame, exclude_boxes=exclude)
            trace, kind, info = ss.trace_in_mask(dmask, frame, anchors,
                                                 avoid=used_keys, allow_loop=allow_loop,
                                                 reject_rules=True)
            info["dark_px"] = int((dmask > 0).sum())
            how = "dark"
        elif color is not None:
            trace, kind, info = ss.trace_series(hsv, frame, color, anchors, exclude,
                                                hue_tol=hue_tol, avoid=used_keys,
                                                allow_loop=allow_loop)
        else:
            return None, {"label": s.get("label"), "anchors": s.get("anchors"),
                          "reason": ("颜色定不下来（模型没给颜色、锚点也没落在线上、"
                                     "图例里也没有）")}
        info["hue_tol"] = hue_tol
        span = ((max(p[0] for p in trace) - min(p[0] for p in trace))
                / max(1.0, frame[2] - frame[0])) if trace else 0.0
        pick = {"s": s, "anchors": anchors, "color": color, "how": how,
                "hue_tol": hue_tol, "trace": trace, "kind": kind, "info": info,
                "span": span, "completion": {}, "dark": dark}
        if trace and info.get("key") is not None:
            used_keys.add(info["key"])       # 同色多条时，后面那条去认领别的实例
        return pick, None

    # 先给每条曲线定颜色，再按"颜色之间离得多远"给每条独立的色相容差：橙和金
    # 只差 12 个色相时，统一的容差会让两条线互相串色（实测 E30 因此追成了 E00）。
    resolved = []
    for s in spec:
        anchors = ss.anchors_to_pixels(s.get("anchors"), frame, (width, height))
        color, how = ss.resolve_color(img, anchors, s.get("color"), palette)
        resolved.append((s, anchors, color, how))
    tols = ss.hue_tolerances([c for _s, _a, c, _h in resolved])
    # ---- 第一遍：每条曲线按自己的颜色精确追踪 ----
    picks = []
    for (s, anchors, color, how), hue_tol in zip(resolved, tols):
        # 太短的先留着：它可能正是"大半段被压在另一条线下面"的那条，补全后再定去留。
        pick, why = trace_one(s, hue_tol)
        if pick is None:
            failed.append(why)
            continue
        picks.append(pick)

    # ---- 第二遍：遮挡补全（"重合部分颜色只显示一个"）----
    # 两条线画在一起时，压在下面的那条在重合段里一个像素都没有：不是追踪失败，是
    # 图上没有东西可追。这里拿"另一条线真的画在那里"当证据，把断掉的那几段沿它补
    # 回来；没有证据的长断口一律不补（曲线真到头了的情况必须保持原样）。
    names = {i: str(p["s"].get("label") or f"series{i + 1}")
             for i, p in enumerate(picks)}
    occl = {i: ss.dense_trace(p["trace"]) for i, p in enumerate(picks)
            if p["kind"] != "loop" and p["trace"]}
    for idx, p in sorted(enumerate(picks), key=lambda t: -len(t[1]["trace"])):
        if p["kind"] == "loop":
            # 闭合回线的补全要先把散成十几块的弧串起来（见 series_seed 说明），
            # 这一版还没做，先不在这里假装补上。
            continue
        others = [(names[j], d) for j, d in occl.items() if j != idx]
        if not others:
            continue
        # 只补"本来就追到了大半条"的曲线。碎片（几像素的杂点、网格线的残段）不补：
        # 补全是"把被压住的一段接回来"，不是"凭一小段造出一条曲线"。
        if p["span"] < 0.08 or len(p["trace"]) < 30:
            continue
        mask = (el.dark_mask(gray, frame, exclude_boxes=exclude) if p["dark"]
                else el.color_mask(hsv, frame, p["color"], p["hue_tol"],
                                   exclude_boxes=exclude))
        filled, cinfo = ss.complete_trace(p["trace"], mask, frame, others)
        # 补出来的部分不能超过整条曲线的 70%：否则说明原始轨迹只是个碎片，
        # 那更可能是网格线/杂点（实测被补成一条横线报了出去）。
        if filled and len(filled) > len(p["trace"]) \
                and len(p["trace"]) >= 0.3 * len(filled):
            p["raw_trace"], p["trace"], p["completion"] = p["trace"], filled, cinfo
        elif cinfo:
            p["completion"] = cinfo

    # 还有没补上的缺口（重合处只显示一种颜色、压在下面的追断了）-> 让模型读图给锚点
    if vlm is not None:
        _repair_gaps(vlm, panel, panel_path, picks, names, Path(csv_dir).parent, log)

    def build_entry(p):
        """一条 pick -> 待落盘的 entry（数据先在内存里，复盘后统一写文件）。"""
        s, trace, kind, info = p["s"], p["trace"], p["kind"], p["info"]
        span = ((max(q[0] for q in trace) - min(q[0] for q in trace))
                / max(1.0, frame[2] - frame[0])) if trace else 0.0
        if len(trace) < 20 or span < 0.12:
            return None, {"label": s.get("label"), "anchors": s.get("anchors"),
                          "reason": (f"没追上这条线（颜色取自{p['how']}，图上候选 "
                                     f"{info.get('candidates', 0)} 条，最长只追到 "
                                     f"{len(trace)} 点；补全后仍不足）")}
        want = _spec_want(s)
        markers = []
        if _spec_has_markers(s):
            mmask = (el.dark_mask(gray, frame, exclude_boxes=exclude) if p["dark"]
                     else el.color_mask(hsv, (0, 0, width - 1, height - 1), p["color"], 12))
            markers = ss.markers_in_mask(mmask, trace)
        note_draw = None
        marker_line = False
        if markers:
            trace = ss.snap_line_to_markers(trace, markers)      # 标记处别走平台
        if _markers_cover_curve(markers, trace):
            # 散点图：标记中心就是数据。用"标记中心连线"替掉逐列追出来的线，
            # 阶梯消失，每个点对应一个值（线是点连出来的，不额外损失信息）
            centers = ss.line_through_markers(markers)
            if len(centers) >= 3:
                # 标记范围**之外**的点要留着——那正是模型读图补的锚点（被压住的那一段
                # 没有标记）。上一版直接用 centers 覆盖，把刚补好的缺口又抹掉了
                # （实测 2024-01-3408 Diesel：补了锚点，CSV 的 x 范围却没变）。
                mx0 = min(c[0] for c in centers)
                mx1 = max(c[0] for c in centers)
                kept = [(x, y) for x, y in trace if x < mx0 - 1.0 or x > mx1 + 1.0]
                trace = sorted(kept + centers)
                # 一个标记一个点：散点图上十几个点就是完整数据，不能再按"线要 20 点"砍
                marker_line = True
                note_draw = f"按标记中心出数据（{len(centers)} 个点），消除标记造成的阶梯"
        elif _spec_has_markers(s):
            # 标记检测不全（点太密/连成串）时退一步：把逐列追踪留下的"平台"压成一个点，
            # 平台正是每个标记的宽度造成的阶梯
            flat = ss.collapse_plateaus(trace)
            if len(flat) < len(trace):
                note_draw = f"标记处的平台已压缩（{len(trace)} -> {len(flat)} 点），消除阶梯"
                trace = flat
        if str(s.get("draw") or "").lower() == "markers_connected" and len(markers) >= 3:
            trace = ss.line_through_markers(markers)             # 点即曲线
            want = "points"
            note_draw = note_draw or "散点用折线连起来：点即曲线（不再另出线文件）"
        elif want == "points" and len(markers) < 3:
            want = "line"
            note_draw = "模型说这条带标记，但没检出标记符号，改出线轨迹"
        line_data = el.to_data(trace, frame, xmin, xmax, ymin, ymax)
        point_data = el.to_data(markers, frame, xmin, xmax, ymin, ymax)
        return {"p": p, "want": want, "markers": markers, "line_data": line_data,
                "point_data": point_data, "note_dup": None, "note_draw": note_draw,
                "marker_line": marker_line, "span": span}, None

    # ---- 第三遍：定去留 + 合并重复轨迹 ----
    # 同一根线被模型拆成两条（甚至四条）上报是实测最常见的错法：63 个面板里出现
    # 30 条完全相同的轨迹。≥0.97 直接合并；0.85~0.97 保留但标注，交给复盘轮定夺。
    entries, aliases = [], []
    for pidx, p in enumerate(picks, start=1):
        entry, why = build_entry(p)
        if entry is None:
            failed.append(why)
            continue
        trace = p["trace"]
        dup_of, ov, same_inst = None, 0.0, False
        for name, other, other_key in accepted:
            o = ss.trace_overlap(trace, other)
            k = p["info"].get("key")
            if k is not None and other_key is not None and k == other_key and o >= 0.8:
                dup_of, ov, same_inst = name, max(o, 0.97), True
                continue
            if o > ov:
                dup_of, ov = name, o
        # 模型明说"这两条有重合"时不能合并：那是两条不同的曲线画在一起（图 10 那种
        # 两条相距几像素的虚线，中位差只有 4.5px，只按像素去重会把它们合掉）。
        overlaps_hi = {str(x).strip().lower() for x in (p["s"].get("overlaps") or [])}
        veto = str(dup_of).strip().lower() in overlaps_hi if dup_of else False
        if (same_inst or ov >= 0.97) and not veto:
            aliases.append({"label": p["s"].get("label") or f"series{pidx}",
                            "alias_of": dup_of,
                            "overlap": round(ov, 2)})
            continue
        if ov >= 0.80:
            entry["note_dup"] = (f"与「{dup_of}」重合 {int(round(ov * 100))}%"
                                 "（可能是同一条，已请模型复核）")
        accepted.append((p["s"].get("label") or f"series{pidx}", trace,
                         p["info"].get("key")))
        entries.append(entry)

    # ---- 第四遍：模型复盘（把追出来的轨迹编号画回原图，让模型核对）----
    review = None
    if vlm is not None and (failed or aliases
                            or any(e.get("note_dup") for e in entries)
                            or len(entries) != len([s for s in spec])):
        review = revise_panel(vlm, panel_path, panel, spec, entries, failed, aliases,
                              csv_dir=csv_dir)
        if review:
            entries, failed, aliases = _apply_review(review, entries, failed, aliases)
            for s in (review.get("add") or []):
                pick, why = trace_one(s, 8)
                if pick is None or len(pick["trace"]) < 20:
                    failed.append({"label": s.get("label"), "anchors": s.get("anchors"),
                                   "reason": f"模型复盘补的这条没追上（{why or '点太少'}）"})
                    continue
                others = [(str(e["p"]["s"].get("label")), ss.dense_trace(e["p"]["trace"]))
                          for e in entries if e["p"]["kind"] != "loop" and e["p"]["trace"]]
                mmask = (el.dark_mask(gray, frame, exclude_boxes=exclude) if pick["dark"]
                         else el.color_mask(hsv, frame, pick["color"], pick["hue_tol"],
                                            exclude_boxes=exclude))
                if others:
                    filled, cinfo = ss.complete_trace(pick["trace"], mmask, frame, others)
                    if filled and len(filled) > len(pick["trace"]):
                        pick["raw_trace"], pick["trace"] = pick["trace"], filled
                        pick["completion"] = cinfo
                entry, why2 = build_entry(pick)
                if entry is None:
                    failed.append(why2)
                    continue
                entry["from_review"] = True
                entries.append(entry)

    # ---- 第五遍：落盘 ----
    bridged_px = []
    for entry in entries:
        p = entry["p"]; s = p["s"]
        trace = p["trace"]
        label = ss.safe_tag(s.get("label") or f"series{len(series) + 1}")
        want = entry["want"]
        emitted = []
        if want in ("points", "both") and len(entry["point_data"]) >= 3:
            name = ss.unique_path(csv_dir, f"{panel_path.stem}_{label}.csv")
            ss.write_csv(name, entry["point_data"])
            emitted.append(("points", name))
        min_line = 3 if entry.get("marker_line") else 20   # 散点图：一个点一个值
        if want in ("line", "both") and len(entry["line_data"]) >= min_line:
            suffix = "_line" if emitted else ""
            name = ss.unique_path(csv_dir, f"{panel_path.stem}_{label}{suffix}.csv")
            ss.write_csv(name, entry["line_data"])
            emitted.append(("line", name))
        if not emitted:
            failed.append({"label": s.get("label"), "anchors": s.get("anchors"),
                           "reason": "追踪出的点太少，未写出"})
            continue
        color = p["color"]
        color_hex = ("#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0]))
                     if color is not None else str(s.get("color") or "#000000"))
        cinfo = p["completion"] or {}
        raw = p.get("raw_trace")
        if raw and cinfo:
            rx = {int(round(x)) for x, _y in raw}
            bridged_px.extend((x, y) for x, y in trace if int(round(x)) not in rx)
        completion = jsonable_completion(cinfo)
        for kind_out, name in emitted:
            data = entry["point_data"] if kind_out == "points" else entry["line_data"]
            series.append({
                "file": name.name, "source": "seeded", "kind": kind_out,
                "geometry": p["kind"], "series_name": s.get("label"), "output": want,
                "draw": s.get("draw"), "dark": bool(p["dark"]),
                "color_hex": color_hex, "color_src": p["how"], "y_axis": s.get("y_axis"),
                "anchors": s.get("anchors"), "note": s.get("note"),
                "anchors_hit": p["info"].get("hits"),
                "anchor_med_dist": (round(p["info"]["med_dist"], 1)
                                    if p["info"].get("med_dist") is not None else None),
                "coverage": round(entry["span"], 3), "points": len(data),
                "duplicate_note": entry.get("note_dup"),
                "draw_note": entry.get("note_draw"),
                "completion": completion,
                "x_range": [round(min(q[0] for q in data), 4), round(max(q[0] for q in data), 4)],
                "y_range": [round(min(q[1] for q in data), 4), round(max(q[1] for q in data), 4)],
            })
    if not series and not failed:
        return None
    traces_px = [(str(e["p"]["s"].get("label") or f"series{i + 1}"), e["p"]["trace"])
                 for i, e in enumerate(entries)]
    return {"panel": panel_path.name, "frame": list(frame), "series": series,
            "failed": failed, "aliases": aliases, "review": review,
            "_traces": traces_px, "_bridged_px": bridged_px, "method": "seeded"}


def jsonable_completion(cinfo):
    """补全信息进 config/report 前瘦身（只留计数与前几段，避免 JSON 膨胀）。"""
    if not cinfo:
        return None
    out = {"bridged_cols": int(cinfo.get("bridged_cols") or 0),
            "filled": int(cinfo.get("filled") or 0),
            "spans": [{"x0": int(s["x0"]), "x1": int(s["x1"]), "cols": int(s["cols"]),
                       "by": str(s.get("by"))} for s in (cinfo.get("spans") or [])[:8]],
            "open_gaps": [{"x0": int(s["x0"]), "x1": int(s["x1"]), "cols": int(s["cols"])}
                          for s in (cinfo.get("open_gaps") or [])[:8]]}
    if cinfo.get("model_anchors"):
        ma = cinfo["model_anchors"]
        out["model_anchors"] = {"cols": int(ma.get("cols") or 0), "by": str(ma.get("by"))}
    return out


def series_y_from(panel):
    """{颜色: 纵轴范围} —— 双 Y 轴图里每条曲线读自己那条轴。"""
    out = {}
    for col, side in (panel.get("series_axes") or {}).items():
        detail = (panel.get("y_axes") or {}).get(side) or {}
        if detail.get("range"):
            out[col.upper()] = detail["range"]
    return out or None


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


def _as_float(value, default=1.0):
    """宽容地把模型给的数字（可能是 "1e-4" / "×10^-4" / None）变成 float。"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace("×", "").replace("x", "").replace("*", "")
    text = text.replace("^", "e").replace("10e", "1e")
    try:
        return float(text)
    except ValueError:
        return default


def _axis_hint(want):
    """把用户需求交给轴读数那一步顺带核对——不额外调用，也不靠"模型觉得像"。

    实测：`--want "贯穿距随时间变化"` 会把"贯穿距 vs 曲轴转角"也算命中。这里让模型在
    读刻度时就回答"这张图的横轴/纵轴是不是用户要的"，命中靠标题核对。
    """
    if not want:
        return None
    return (f"用户要的是：{want}。请额外读出 x 轴与各纵轴的标题/单位，并判断这张图的"
            f"横轴与纵轴是否就是用户要的那两个量：符合填 matches_request=true，不符合填 "
            f"false（例如用户要时间轴而这里是曲轴转角），看不清填 null。")


def _record_axis_check(entry, va, want, log):
    """把轴标题与"是否合需求"记进 config；明确不符合的图直接标跳过（报告里给理由）。"""
    x = va.get("x") if isinstance(va.get("x"), dict) else {}
    ylist = [d for d in (va.get("y_axes") or []) if isinstance(d, dict)]
    y0 = ylist[0] if ylist else {}
    entry["axis"]["x_title"] = x.get("title")
    entry["axis"]["x_unit"] = x.get("unit")
    entry["axis"]["y_title"] = y0.get("title")
    titles = (f"x={x.get('title')}"
              + (f" [{x.get('unit')}]" if x.get("unit") else "")
              + f" / y={y0.get('title')}"
              + (f" [{y0.get('unit')}]" if y0.get("unit") else ""))
    if not want:
        return
    match, note = va.get("matches_request"), va.get("match_note")
    note = str(note or "")
    # "读不出标题"不等于"不符合"：模型有时因为图里没有轴标题就填 false，上一版据此把
    # 面板整个跳过（实测 2026-01-0340 图 10 左图：横轴标题被裁掉，但确实是时间轴，
    # 结果整张图一条数据都没出）。只有明确"轴是别的东西"才跳过。
    if match is False and any(w in note for w in
                              ("无法确认", "无法判断", "无法判定", "看不清", "不确定",
                               "无标题", "没有标题", "未标")):
        match = None
        note += "（标题读不出，按「保留待复核」处理）"
    entry["axis_check"] = {"want": want, "matches": match, "note": note,
                           "x_title": x.get("title"), "y_title": y0.get("title")}
    verdict = "符合" if match is True else ("不符合" if match is False else "看不清")
    log(f"      轴标题核对（{verdict}）: {titles}"
        + (f" —— {str(note)[:60]}" if note else ""))
    if match is False:
        entry["skip"] = True
        entry["skip_reason"] = (f"轴标题与需求不符（x={x.get('title')} / "
                                f"y={y0.get('title')}）：{str(note or '')[:80]}")
        log("      ⏭ 按需求跳过这张图（轴标题核对没通过）")


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
            out[side] = {"range": rng, "labels": item.get("labels"),
                         "multiplier": _as_float(item.get("multiplier"), 1.0),
                         "unit": item.get("unit"), "title": item.get("title")}
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
            want_deep=False, find_series=True):
    """use_ocr=None means "auto": with a VLM the model is the primary axis reader and
    OCR (5-10x slower, same job) stays off; without one OCR is the only reader left."""
    if use_ocr is None:
        use_ocr = vlm is None
    t0 = time.time()
    pdf_path = Path(pdf_path).resolve()
    outdir = Path(outdir).resolve()
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "panels").mkdir(parents=True, exist_ok=True)
    if find_series and vlm is not None:
        (outdir / "objects").mkdir(parents=True, exist_ok=True)
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
        series_roles = {}
        dark_info = {}
        if vlm is not None and legend_colors:
            try:
                nm = vt.name_series(vlm, fig_path, legend_colors)
                for s in nm.get("series", []):
                    col = (s.get("color") or "").upper()
                    if col and s.get("label"):
                        name_map[col] = s["label"]
                    if col and s.get("y_axis"):
                        series_sides[col] = str(s["y_axis"]).strip().lower()
                    if col:
                        # 模型说"这不是数据线"（切线/拟合线/标注/放大子图）就别提取
                        role = str(s.get("role") or "").strip().lower()
                        if s.get("is_data") is False and not role:
                            role = "not-data"
                        if role:
                            series_roles[col] = role
                log("      VLM 系列命名: " + ", ".join(f"{k}->{v}" for k, v in name_map.items()))
                if series_sides:
                    log("      VLM 轴归属: " + ", ".join(f"{k}->{v}轴" for k, v in series_sides.items()))
                drop = {k: v for k, v in series_roles.items()
                        if v in ("tangent", "fit", "annotation", "legend", "inset",
                                 "other", "not-data")}
                if drop:
                    log("      VLM 判定不提取: " + ", ".join(f"{k}({v})" for k, v in drop.items()))
                dark_info = nm.get("dark_series") or {}
                if dark_info.get("has_dark_line"):
                    log(f"      图中有深色线：{str(dark_info.get('kind') or '?')}"
                        f"（{str(dark_info.get('desc') or '')[:50]}）"
                        " → 默认不提取；要提就在 config 里把该面板的 allow_dark 设为 true")
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
                "series_roles": series_roles,
                "vlm_dark_line": dark_info,
                "vlm_classification": vlm_cls,
                "axis": {"x": None, "y": None, "confirmed": False},
            }
            # ---- 轴读数：VLM 先读（主），OCR 随后独立核对（辅）----
            va = None
            if vlm is not None:
                try:
                    va = vt.read_axis_ranges(vlm, panel_path, hint=_axis_hint(want))
                    y_axes = _norm_y_axes(va)
                    entry["vlm_axis"] = {"x": va.get("x"), "y_axes": va.get("y_axes"),
                                         "y": va.get("y"),
                                         "confidence": va.get("confidence"),
                                         "note": va.get("note")}
                    if y_axes:
                        entry["y_axes"] = y_axes
                    _record_axis_check(entry, va, want, log)
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
                    v_mult = 1.0
                    if isinstance(v_val, dict):          # 新格式：{"min","max","multiplier"}
                        v_mult = _as_float(v_val.get("multiplier"), 1.0)
                    elif isinstance(va, dict):
                        v_mult = _as_float(va.get("multiplier"), 1.0)
                else:
                    y_axes = (entry.get("y_axes") or {})
                    first = y_axes.get("left") or (list(y_axes.values())[0] if y_axes else None)
                    v_val = first["range"] if first else None
                    v_mult = (first or {}).get("multiplier", 1.0)
                o_val = (ocr_ranges.get(axis_key, {}) or {}).get("range") \
                    if axis_key in ocr_ranges else None
                vals, conf, src, note = merge_axis(v_val, o_val)
                if vals:
                    entry["axis"][axis_key] = vals
                if abs(_as_float(v_mult, 1.0) - 1.0) > 1e-12:
                    entry["axis"].setdefault("multiplier", {})[axis_key] = \
                        _as_float(v_mult, 1.0)
                    entry.setdefault("notes", []).append(
                        f"{axis_key}: 图上标注了 ×{_as_float(v_mult, 1.0):g}，"
                        f"数据已按此换算")
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

            # ---- 物件级判定：让模型看着编号裁图说清"哪根是哪根" ----
            # 让模型**自己指出**数据曲线和它们的大致位置（锚点），代码再从锚点精确追踪。
            # 关键在于模型输出的是"要提取什么"（正向清单），不再是"我给的候选里哪个不是"
            # ——后者必然出现"数据被 skip 掉、该 skip 的却留下"。
            if vlm is not None and find_series:
                try:
                    legend_text = "、".join(
                        f"{v}（{k}）" for k, v in (name_map or {}).items()
                    ) or "、".join(legend_colors)
                    found = vt.find_series(vlm, panel_path, legend_text)
                    spec = [s for s in (found.get("series") or []) if isinstance(s, dict)]
                    if spec:
                        entry["series_spec"] = spec
                        entry["ignore"] = found.get("ignore") or []
                        for s in spec:
                            a = s.get("anchors") or []
                            log(f"      模型指出曲线: {str(s.get('label'))[:26]:<26}"
                                f" 颜色={s.get('color')} 线型={s.get('linestyle')}"
                                f" 标记={s.get('has_markers')} 输出={s.get('output')}"
                                f" 锚点={len(a)}")
                        for ig in entry["ignore"]:
                            log(f"      模型指出忽略: {str(ig.get('what'))[:44]}"
                                f"（{str(ig.get('why'))[:40]}）")
                except vlmc.VLMError as exc:
                    log(f"      曲线定位失败: {exc}")
                except Exception as exc:  # noqa: BLE001
                    log(f"      曲线定位异常: {type(exc).__name__}: {exc}")

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
            f"- 轴范围: x={ax.get('x')}  y={ax.get('y')}"
            + ("  数量级标注: "
               + "、".join(f"{k}×{float(v):g}" for k, v in (ax.get("multiplier") or {}).items())
               + "（CSV 里的数值已按此换算）" if ax.get("multiplier") else ""),
        ]
        if ax.get("x_title") or ax.get("y_title"):
            chk = p.get("axis_check") or {}
            verdict = ("✅ 符合" if chk.get("matches") is True else
                       "❌ 不符合（已跳过）" if chk.get("matches") is False else
                       "⚠️ 看不清（保留）" if chk else "")
            lines.append(f"- 轴标题: x={ax.get('x_title') or '?'}"
                         + (f" [{ax.get('x_unit')}]" if ax.get("x_unit") else "")
                         + f"  y={ax.get('y_title') or '?'}"
                         + (f"  需求核对: {verdict} — {str(chk.get('note') or '')[:80]}"
                            if verdict else ""))
        if p.get("skip") and p.get("skip_reason"):
            lines.append(f"- ⏭ 跳过原因: {p['skip_reason']}")
        ocr = p.get("ocr", {})
        for axis in ("x", "y"):
            if axis in ocr:
                o = ocr[axis]
                lines.append(
                    f"  - {axis} 轴 OCR: 读到 {o['labels']} ({o['n_labels']} 个), "
                    f"R²={o['r2']}, 步长={o['step']}, 吸附={'是' if o['snapped'] else '否'}"
                    + (f", 离群={o['outliers']}" if o["outliers"] else "")
                    + (f", 已修复={o['repaired']}" if o["repaired"] else ""))
        spec = p.get("series_spec")
        if spec:
            lines.append("- 模型指出的曲线（锚点只是种子，数值由代码从锚点追踪得出）:")
            for s in spec:
                lines.append(f"  - {s.get('label') or '(无图例条目)'} ｜ 颜色={s.get('color')} "
                             f"｜ 线型={s.get('linestyle')} ｜ 标记={s.get('has_markers')} "
                             f"｜ 输出={s.get('output')} ｜ 锚点 {len(s.get('anchors') or [])} 个"
                             + (f" ｜ {s['note']}" if s.get("note") else ""))
        for ig in (p.get("ignore") or []):
            lines.append(f"  - 忽略：{ig.get('what')}（{ig.get('why')}）")
        if results and p["id"] in results:
            r = results[p["id"]]
            lines.append(f"- 提取结果: {len(r['series'])} 条曲线")
            for s in r["series"]:
                extra = []
                if s.get("geometry") == "loop":
                    extra.append("闭合回线（沿轮廓取点，不是按 x 排序）")
                if s.get("dark"):
                    extra.append("黑/灰线（按深色笔画追）")
                if s.get("color_src"):
                    extra.append(f"颜色取自{s['color_src']}")
                if s.get("anchors_hit") is not None:
                    extra.append(f"锚点命中 {s['anchors_hit']} 个"
                                 + (f"，中位偏差 {s['anchor_med_dist']}px"
                                    if s.get("anchor_med_dist") is not None else ""))
                if s.get("duplicate_note"):
                    extra.append("⚠️ " + s["duplicate_note"])
                if s.get("draw_note"):
                    extra.append(s["draw_note"])
                comp = s.get("completion") or {}
                if comp.get("bridged_cols"):
                    bys = sorted({str(x.get("by")) for x in (comp.get("spans") or [])})
                    extra.append(f"补全 {comp['bridged_cols']} 列"
                                 + (f"（{('、'.join(bys[:3]))}）" if bys else ""))
                if comp.get("open_gaps"):
                    cols = sum(int(x["cols"]) for x in comp["open_gaps"])
                    extra.append(f"⚠️ {len(comp['open_gaps'])} 段长断口未补"
                                 f"（共 {cols} 列，没有别的笔画经过，保持断开）")
                if comp.get("model_anchors"):
                    extra.append(f"🧠 {comp['model_anchors'].get('by')}"
                                 f"（+{comp['model_anchors'].get('cols')} 列）")
                lines.append(f"  - `{s['file']}`  {s['points']} 点, "
                             f"y=[{s['y_range'][0]}, {s['y_range'][1]}]"
                             + ("（" + "；".join(extra) + "）" if extra else ""))
            for f in (r.get("failed") or []):
                lines.append(f"  - ✗ 未追到 `{f.get('label')}`：{f.get('reason')}")
            for a in (r.get("aliases") or []):
                why = a.get("why") or "轨迹完全相同"
                lines.append(f"  - ⇄ 合并：`{a.get('label')}` 与 `{a.get('alias_of')}` "
                             f"是同一条曲线（{why}，重合度 {a.get('overlap')}）——只出一份数据")
            rv = r.get("review") or {}
            if rv:
                lines.append(f"  - 模型复盘：merge={rv.get('merge') or []} "
                             f"drop={rv.get('drop') or []} "
                             f"add={len(rv.get('add') or [])} ｜ {str(rv.get('note'))[:70]}")
            lines.append(f"- 质检图: `verify/{p['id']}_verify.png`")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def make_verify_image(panel_path, frame, axis, legend_colors, series_files, out_path,
                      failed=None, anchors=None, bridged=None):
    """Thin wrapper so callers do not have to build the series tuples themselves."""
    series = []
    for csv_path, color_hex in zip(series_files, legend_colors):
        xs, ys = vo.read_csv(csv_path)
        series.append((Path(csv_path).stem, color_hex, xs, ys))
    # 模型指出了但代码没追到的曲线：把它的锚点圈出来，肉眼即可判断是模型错还是代码错
    marks = []
    if failed:
        img = iio.imread(panel_path)
        if img is not None:
            w, h = img.shape[1], img.shape[0]
            for f in failed:
                for a in (f.get("anchors") or []):
                    try:
                        x, y = float(a[0]) * w, float(a[1]) * h
                    except (TypeError, ValueError, IndexError):
                        continue
                    marks.append(([int(x) - 10, int(y) - 10, int(x) + 10, int(y) + 10],
                                  f"missed:{str(f.get('label'))[:12]}"))
    vo.compose_overlay(panel_path, frame, tuple(axis["x"]), tuple(axis["y"]),
                       series, out_path, annotations=marks, anchors=anchors,
                       bridged=bridged)


def _write_template_output(fmt, res, panel, panel_path, csv_dir, outdir, vlm, log):
    """把这一面板的曲线按用户模板再输出一份（模板代码由模型写一次，之后复用）。"""
    # 同一个系列可能有两个文件（标记点 + 线）：模板里一条曲线只该出现一次，
    # 取**线**（它包含被遮挡段补出来的点）；只出点的那种取点。
    picked = {}
    for s in res["series"]:
        xs, ys = vo.read_csv(csv_dir / s["file"])
        if not xs:
            continue
        name = str(s.get("series_name") or Path(s["file"]).stem)
        rank = 1 if s.get("kind") == "line" else 0
        old = picked.get(name)
        if old is None or rank > old[0] or (rank == old[0] and len(xs) > len(old[1]["x"])):
            picked[name] = (rank, {"name": name, "file": s["file"], "x": xs, "y": ys,
                                   "params": {}})
    series = [v[1] for v in picked.values()]
    if not series:
        return
    context = {"paper": panel_path.parent.parent.name, "panel": panel["id"],
               "ext": fmt.ext, "template": fmt.name}
    # 模板声明的参数 + 代码实际用到的参数（模型有时只列了"按曲线"的那部分，
    # 模板里的工况块就被漏掉 -> 渲染 KeyError。让代码自己说它要什么）
    keys = list(dict.fromkeys(fmt.params + tpl.probe_params(fmt, series, context)))
    vals = {}
    if keys:
        vals = tpl.ask_params(vlm, panel_path, panel.get("caption"), series, keys,
                             hints=tpl.params_hints(fmt, keys))
    for item in series:
        item["params"] = {k: str((vals.get(item["name"]) or {}).get(k, "")) for k in keys}
    try:
        files = fmt.render(series, context) or {}
    except Exception as exc:  # noqa: BLE001 - 模板坏了不该拖垮提取
        # 模板里写的是 %.E 这类数值格式，而模型读回来的是字符串 -> 能转数字的转数字，
        # 没读到的填空串会报错，就退成 0，再试一次（绝不因为参数格式把整份输出丢掉）
        coerced, blanked = 0, 0
        for item in series:
            fixed = {}
            for k, v in item["params"].items():
                if v == "":
                    fixed[k], blanked = 0.0, blanked + 1
                    continue
                try:
                    fixed[k] = float(str(v).strip())
                    coerced += 1
                except (TypeError, ValueError):
                    fixed[k] = v
            item["params"] = fixed
        try:
            files = fmt.render(series, context) or {}
            log(f"      （模板要数值：{coerced} 个参数转成数字"
                + (f"，{blanked} 个没读到按 0 填" if blanked else "")
                + "，渲染继续）")
        except Exception as exc2:  # noqa: BLE001
            log(f"      （模板渲染失败，跳过：{type(exc2).__name__}: {exc2}）")
            return
    dest = Path(outdir) / "templates" / Path(str(fmt.name)).stem
    for fname, text in files.items():
        p = dest / Path(str(fname)).name               # 只取文件名，防目录穿越
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(text), encoding="utf-8")
    log(f"      模板输出 {len(files)} 个文件 -> {dest}")


def extract(cfg_path, force=False, only=None, vlm=None, template=None):
    """template: 文件路径 = 用这个模板；None = 沿用上次的模板；False = 不用模板（只出 CSV）。"""
    t0 = time.time()
    cfg_path = Path(cfg_path).resolve()
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    vlm_usage0 = dict(vlm.usage) if vlm is not None else None
    outdir = Path(config.get("outdir") or cfg_path.parent)
    csv_dir = outdir / "csv"
    verify_dir = outdir / "verify"
    csv_dir.mkdir(parents=True, exist_ok=True)
    verify_dir.mkdir(parents=True, exist_ok=True)
    fmt = None
    if template is not False:
        try:
            fmt = tpl.load(template, vlm=vlm, log=log)
        except Exception as exc:  # noqa: BLE001 - 模板不可用不该拖垮提取
            log(f"      （模板不可用，只出 CSV：{type(exc).__name__}: {exc}）")

    # 重跑时先清掉这次要覆盖的旧产物：否则新旧结果混在一个目录里，
    # 分不清哪个 CSV 是这一次的（实测把这批论文重跑时踩到过）
    if not only:
        for d in (csv_dir, verify_dir):
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
    else:
        for p in config["panels"]:
            if p["id"] not in only:
                continue
            stem = Path(p.get("panel_image") or "").stem
            for d in (csv_dir, verify_dir):
                for f in list(d.glob(f"{stem}*")) if stem else []:
                    f.unlink()

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
        # 数量级标注（轴旁边的 ×10^-4 之类）：模型读出来、代码乘上去。缩放两端点等于
        # 缩放整条映射，所以在这里做一次就够了（y = ymin + t*(ymax-ymin) 是线性的）。
        mult = ax.get("multiplier") or {}
        kx = _as_float(mult.get("x"), 1.0)
        ky = _as_float(mult.get("y"), 1.0)
        xr = [xr[0] * kx, xr[1] * kx]
        yr = [yr[0] * ky, yr[1] * ky]
        rng = [xr[0], xr[1], yr[0], yr[1]]
        # dual-axis figures: give each series the y range of the axis it reads
        series_y = {}
        tol_by_range = {}
        for col, side in (p.get("series_axes") or {}).items():
            detail = (p.get("y_axes") or {}).get(side) or {}
            r = detail.get("range")
            if r:
                k = _as_float(detail.get("multiplier"), 1.0)   # 该纵轴自己的数量级
                r = [r[0] * k, r[1] * k]
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
        res = _extract_panel(p, panel_path, rng, csv_dir, vlm=vlm)
        if res is None:
            results[p["id"]] = {"series": [], "error": "提取失败", "axis_range": rng}
            continue
        res["axis_range"] = rng
        # 复盘里模型如果指出"轴的映射不对"（曲线整体偏移/压平/漏了数量级），用它给的
        # 范围重跑这个面板一次：CSV 先清掉，避免留下上一轮的旧文件。
        fix = (res.get("review") or {}).get("axis_fix")
        if isinstance(fix, dict) and any(fix.get(k) for k in ("x", "y", "multiplier")):
            try:
                new_rng = list(rng)
                if isinstance(fix.get("x"), (list, tuple)) and len(fix["x"]) == 2:
                    new_rng[0], new_rng[1] = float(fix["x"][0]), float(fix["x"][1])
                if isinstance(fix.get("y"), (list, tuple)) and len(fix["y"]) == 2:
                    new_rng[2], new_rng[3] = float(fix["y"][0]), float(fix["y"][1])
                kfix = _as_float(fix.get("multiplier"), 1.0)
                if abs(kfix - 1.0) > 1e-12:
                    new_rng = [new_rng[0], new_rng[1], new_rng[2] * kfix, new_rng[3] * kfix]
                for f in csv_dir.glob(f"{panel_path.stem}*"):
                    f.unlink()
                res2 = _extract_panel(p, panel_path, new_rng, csv_dir)
                if res2:
                    log(f"      轴映射按模型复盘修正：{rng} -> {new_rng}，已重跑该面板")
                    res2["axis_range"] = new_rng
                    res2["axis_fix"] = fix
                    res = res2
            except Exception as exc:  # noqa: BLE001
                log(f"      （轴映射修正失败，保留原结果：{type(exc).__name__}: {exc}）")
        results[p["id"]] = res
        log(f"{p['id']}: {len(res.get('series', []))} 条曲线"
            + (f"  方式={'锚点+追踪' if res.get('method') == 'seeded' else '颜色追踪'}"
               if res.get("series") else ""))
        for f in res.get("failed") or []:
            log(f"      ✗ 未追到：{str(f.get('label'))[:30]}（{str(f.get('reason'))[:60]}）")
        if fmt and res.get("series"):
            _write_template_output(fmt, res, p, panel_path, csv_dir, outdir, vlm, log)
        if res.get("series"):
            files = [csv_dir / s["file"] for s in res["series"]]
            colors, anchors = [], []
            panel_img = iio.imread(panel_path)
            panel_h, panel_w = (panel_img.shape[0], panel_img.shape[1]) \
                if panel_img is not None else (0, 0)
            for s in res["series"]:
                hexs = s.get("color_hex")
                if not hexs and s.get("target_bgr"):
                    hexs = "#{:02x}{:02x}{:02x}".format(
                        s["target_bgr"][2], s["target_bgr"][1], s["target_bgr"][0])
                colors.append(hexs or "#ff00ff")
                # 模型给的锚点画成十字：锚点不在线上 = 模型没指准；锚点在线上而数据
                # 没追上来 = 追踪器的问题。QA 时一眼分得清是哪个环节坏了。
                if s.get("anchors") and panel_w:
                    for a in s["anchors"]:
                        try:
                            anchors.append((float(a[0]) * panel_w, float(a[1]) * panel_h,
                                            str(s.get("series_name"))[:12]))
                        except (TypeError, ValueError, IndexError):
                            continue
            make_verify_image(panel_path, res["frame"], ax, colors, files,
                              verify_dir / f"{p['id']}_verify.png",
                              failed=res.get("failed"), anchors=anchors,
                              bridged=res.get("_bridged_px"))

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

    for r in results.values():                 # 私有字段（像素轨迹）不进 JSON
        for k in [k for k in r if k.startswith("_")]:
            r.pop(k, None)
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


def _run_one(pdf, sub, row, dpi, pages, force, vlm, use_ocr, want=None, want_deep=False,
             find_series=True, template=None):
    """One PDF end-to-end (analyze + extract). Shared by serial and parallel batch."""
    cfg = analyze(pdf, sub, dpi=dpi, vlm=vlm, pages=pages, use_ocr=use_ocr, want=want,
                  want_deep=want_deep, find_series=find_series)
    config = json.loads(Path(cfg).read_text(encoding="utf-8"))
    if config.get("selection"):
        row["matched"] = len(config["selection"].get("ids") or [])
    row["panels"] = len([p for p in config["panels"] if not p.get("skip")])
    row["panels_skipped"] = len([p for p in config["panels"] if p.get("skip")])
    results = extract(cfg, force=force, vlm=vlm, template=template) or {}
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
               want_deep=False, find_series=True, template=None):
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
        _run_one(pdf, sub, row, dpi, pages, force, client, use_ocr, want, want_deep,
                 find_series, template)
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
                  want_deep=False, find_series=True, template=None):
    rows = []
    for i, pdf in enumerate(pdfs, start=1):
        t_file = time.time()
        sub = outdir / pdf.stem
        log(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        before = dict(vlm.usage) if vlm is not None else None
        row = new_row(pdf, sub)
        try:
            _run_one(pdf, sub, row, dpi, pages, force, vlm, use_ocr, want, want_deep,
                     find_series, template)
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
                    want_deep=False, find_series=True, template=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    spec = {"model": vlm.model, "base_url": vlm.base_url} if vlm is not None else None
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_batch_one, str(pdf), str(outdir / pdf.stem), dpi, pages,
                             force, spec, use_ocr, want, want_deep,
                             find_series, template): pdf for pdf in pdfs}
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
          want_deep=False, find_series=True, template=None):
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
                               want_deep, find_series, template)
        # 用量在子进程里累计，父进程按行加总后再打印
        if vlm is not None:
            for k in vlm.usage:
                vlm.usage[k] = sum(r.get("usage", {}).get(k, 0) for r in rows)
    else:
        rows = _batch_serial(pdfs, outdir, dpi, pages, force, vlm, use_ocr, want,
                             want_deep, find_series, template)

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
    a.add_argument("--no-find-series", action="store_true",
                   help="不让模型找曲线（省掉每面板一次调用，但就不知道该提哪几条线）")

    e = sub.add_parser("extract", help="按配置提取数据（只跑已确认的面板）")
    e.add_argument("config")
    e.add_argument("--force", action="store_true", help="不检查 confirmed")
    e.add_argument("--only", nargs="*", default=None, help="只跑指定面板 id")
    e.add_argument("--vlm", action="store_true", help="提取后用 VLM 抽查校验")
    e.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    e.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    e.add_argument("--template", default=None,
                   help="按这个模板文件（txt/dat 等）再输出一份；模型只读一次即缓存")
    e.add_argument("--no-template", action="store_true", help="不用模板（只出 CSV）")

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
    al.add_argument("--no-find-series", action="store_true", help="不让模型找曲线")
    al.add_argument("--template", default=None,
                    help="按这个模板文件（txt/dat 等）再输出一份；模型只读一次即缓存")
    al.add_argument("--no-template", action="store_true", help="不用模板（只出 CSV）")

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
    b.add_argument("--no-find-series", action="store_true", help="不让模型找曲线")
    b.add_argument("--template", default=None,
                   help="按这个模板文件（txt/dat 等）再输出一份；模型只读一次即缓存")
    b.add_argument("--no-template", action="store_true", help="不用模板（只出 CSV）")

    sub.add_parser("doctor", help="环境自检：解释器、依赖、API key")

    args = ap.parse_args()

    def make_vlm():
        if not getattr(args, "vlm", False):
            return None
        client = vlmc.DeepSeekVLM(model=args.vlm_model, base_url=args.vlm_base_url,
                                  verbose=True)
        if not client.api_key:
            log("⚠ 指定了 --vlm 但没找到 API key："
                "请把 key 写入项目根目录的 deepseek_key.txt 或设置 DEEPSEEK_API_KEY。"
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
                use_ocr=use_ocr(), want=args.want, want_deep=args.want_deep,
                find_series=not args.no_find_series)
    elif args.cmd == "extract":
        extract(args.config, force=args.force,
                only=set(args.only) if args.only else None, vlm=make_vlm(),
                template=False if args.no_template else args.template)
    elif args.cmd == "confirm":
        confirm(args.config, args.id, args.x, args.y, args.skip)
    elif args.cmd == "all":
        client = make_vlm()
        cfg = analyze(args.pdf, args.out, args.dpi, vlm=client, pages=args.pages,
                      use_ocr=use_ocr(), want=args.want, want_deep=args.want_deep,
                      find_series=not args.no_find_series)
        extract(cfg, force=args.force, vlm=client,
                template=False if args.no_template else args.template)
        if client is not None:
            print_vlm_usage(client, "本次合计")
    elif args.cmd == "batch":
        batch(args.dir, args.out, dpi=args.dpi, vlm=make_vlm(), pages=args.pages,
              force=args.force, pattern=args.pattern, recursive=args.recursive,
              use_ocr=use_ocr(), jobs=args.jobs, want=args.want,
              want_deep=args.want_deep, find_series=not args.no_find_series,
              template=False if args.no_template else args.template)
    elif args.cmd == "doctor":
        doctor()


if __name__ == "__main__":
    main()
