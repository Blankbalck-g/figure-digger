"""Batch line extraction over split panels, reusing the logic in extract_lines.py.

Usage:
  python batch_extract.py --panels dir/panels_fig6 --prefix fig6 --outdir out/csv_fig6 \
      --range 0,5,0,2.5 --range 0,11,0,3 --range 0,7,0,3 \
      --range 0,9,0,3 --range 0,3.5,0,2 --range 0,7,0,3

Ranges are given in panel order as xmin,xmax,ymin,ymax. If fewer ranges than panels
are supplied, the last one is reused.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import imgio as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_lines as el  # noqa: E402
import legend_colors as lc  # noqa: E402

# 可信度闸门：宁可报告"这条没提取出来"，也不要输出一条看起来像数据、实际是
# 标注文字/图例样本的曲线。阈值偏保守，可用下面的常量调。
MIN_SPAN_X = 0.30        # 曲线必须覆盖横轴的这么宽
MIN_POINTS = 25          # 至少这么多个采样点
MAX_BIG_JUMPS = 4        # 允许的"大跳变"次数（超过就判为不合理震荡）
BIG_JUMP_FRAC = 0.25     # 多大算大跳变（占纵轴量程的比例）
DARK_MIN_Y_SPAN = 0.12   # 黑线至少要在纵向上跨这么多，否则多半是轴线/文字行
# 模型判定为这些角色的颜色不提取（切线/拟合线/标注/图例样本/放大子图）
SKIP_ROLES = {"tangent", "fit", "annotation", "legend", "inset", "other", "not-data"}


def _quality(data, xr, yr):
    """Is this trace plausible as a single data series?"""
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    x_span = (max(xs) - min(xs)) / max(1e-9, xr[1] - xr[0])
    y_axis = max(1e-9, yr[1] - yr[0])
    y_span = (max(ys) - min(ys)) / y_axis
    jumps = 0
    if len(ys) > 1:
        dy = np.abs(np.diff(np.array(ys, dtype=float)))
        jumps = int((dy > BIG_JUMP_FRAC * y_axis).sum())
    reasons = []
    if len(data) < MIN_POINTS:
        reasons.append(f"点数只有 {len(data)}")
    if x_span < MIN_SPAN_X:
        reasons.append(f"只覆盖横轴的 {x_span * 100:.0f}%（疑似标注/图例样本）")
    if jumps > MAX_BIG_JUMPS:
        reasons.append(f"有 {jumps} 处不合理跳变（疑似标记点/误差棒被并入）")
    return {"ok": not reasons, "x_span": round(x_span, 3), "points": len(data),
            "y_span": round(y_span, 3), "big_jumps": jumps, "reasons": reasons}


def _same_line(a, b, tol=3.0, need=0.9):
    """True when two traces are the same curve (different legend colours overlap).

    Neighbouring hues share pixels - a cyan legend entry and a blue one can both
    trace the same line - so the survivors are compared once at the end.

    Compare in **pixel** space: a tolerance in data units means something different on
    every axis (1.2 units is 8px on a 0-80 axis and half the range on a 0-2.5 one),
    which silently merged distinct series. The tolerance is deliberately tight: three
    real curves of the same family (800K / 946K / 1150K) run a few pixels apart and
    must all survive.
    """
    by = {}
    for x, y in b:
        by.setdefault(x, []).append(y)
    lo = max(min(p[0] for p in a), min(p[0] for p in b))
    hi = min(max(p[0] for p in a), max(p[0] for p in b))
    close = total = 0
    for x, y in a:
        if not (lo <= x <= hi):
            continue
        ys = by.get(x)
        if not ys:
            continue
        total += 1
        if min(abs(y - yy) for yy in ys) <= tol:
            close += 1
    return total >= 20 and close / total >= need

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def process_panel(image_path, rng, outdir, sat_min, val_min, hue_tol,
                  legend_colors=None, legend_image=None, name_map=None, series_y=None,
                  frame_hint=None, roles=None, allow_dark=False):
    xmin, xmax, ymin, ymax = rng
    img = iio.imread(image_path)
    if img is None:
        return {"panel": image_path.name, "error": "unreadable"}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left, top, right, bottom = _frame_of(gray, frame_hint)
    frame = (left, top, right, bottom)
    legends = inner_boxes(gray, frame)

    series = []
    skipped = []
    kept_traces = []          # 整个面板共用：相邻色相的同一条线只保留一次
    if legend_colors:
        # 图例给的是颜色，不是"有几条线"：同色多条线要逐条追出来
        for spec in legend_colors:
            target = el.hex_to_bgr(spec) if isinstance(spec, str) else list(spec)
            hexs = "#{:02x}{:02x}{:02x}".format(int(target[2]), int(target[1]), int(target[0]))
            role = ((roles or {}).get(hexs) or (roles or {}).get(hexs.upper())
                    or (roles or {}).get(hexs.lower()) or "")
            if role and role.lower() in SKIP_ROLES:
                nice_early = (name_map or {}).get(hexs) or (name_map or {}).get(hexs.upper())
                skipped.append({"color": hexs, "series_name": nice_early,
                                "reason": f"模型判定这条不是数据线（{role}）"})
                continue
            # each series may read a different y axis (dual-axis figures)
            y_rng = ((series_y or {}).get(hexs) or (series_y or {}).get(hexs.upper())
                     or [ymin, ymax])
            h, s, v = el.bgr_to_hsv(target)
            label = el.hue_name(h)
            nice = (name_map or {}).get(hexs) or (name_map or {}).get(hexs.upper())
            tag = label
            if nice:
                import re as _re
                tag = _re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", nice).strip("_")[:40]
            instances = el.trace_series_instances(hsv, frame, target, hue_tol,
                                                  exclude_boxes=legends)
            if not instances:
                skipped.append({"color": hexs, "series_name": nice,
                                "reason": "这个颜色没连成可用的线"})
                continue
            for k, pts in enumerate(instances, start=1):
                data = el.to_data(pts, frame, xmin, xmax, y_rng[0], y_rng[1])
                q = _quality(data, (xmin, xmax), (y_rng[0], y_rng[1]))
                if not q["ok"]:
                    skipped.append({"color": hexs, "series_name": nice, "instance": k,
                                    "points": len(data), "reason": "；".join(q["reasons"])})
                    continue
                if any(_same_line(pts, prev) for prev in kept_traces):
                    skipped.append({"color": hexs, "series_name": nice, "instance": k,
                                    "points": len(data),
                                    "reason": "与已保留的曲线重合（相邻色相互相覆盖）"})
                    continue
                kept_traces.append(pts)
                name = f"{image_path.stem}_{tag}{'' if k == 1 else f'_{k}'}.csv"
                with (outdir / name).open("w", newline="", encoding="utf-8") as fh:
                    wr = csv.writer(fh)
                    wr.writerow(["x", "y"])
                    for xv, yv in data:
                        wr.writerow([f"{xv:.6g}", f"{yv:.6g}"])
                series.append({
                    "file": name,
                    "source": "legend",
                    "target_bgr": target,
                    "label": label,
                    "series_name": nice,
                    "instance": k,
                    "y_axis_range": list(y_rng),
                    "points": len(data),
                    "quality": q,
                    "x_range": [round(min(p[0] for p in data), 4), round(max(p[0] for p in data), 4)],
                    "y_range": [round(min(p[1] for p in data), 4), round(max(p[1] for p in data), 4)],
                })
        if allow_dark:                      # 黑色数据线：模型确认存在时才做
            for k, pts in enumerate(el.trace_dark_instances(gray, frame, legends,
                                                            max_instances=2), start=1):
                data = el.to_data(pts, frame, xmin, xmax, ymin, ymax)
                q = _quality(data, (xmin, xmax), (ymin, ymax))
                if not q["ok"]:
                    skipped.append({"color": "dark", "points": len(data),
                                    "reason": "黑线：" + "；".join(q["reasons"])})
                    continue
                if q["y_span"] < DARK_MIN_Y_SPAN:
                    # 贴着坐标轴的水平暗条多半是轴线/文字行，不是数据曲线
                    skipped.append({"color": "dark", "points": len(data),
                                    "reason": f"黑线：几乎水平（y 只跨 {q['y_span'] * 100:.0f}% 量程），"
                                              f"疑似轴线或文字行"})
                    continue
                if any(_same_line(pts, prev) for prev in kept_traces):
                    continue
                kept_traces.append(pts)
                name = f"{image_path.stem}_dark{'' if k == 1 else f'_{k}'}.csv"
                with (outdir / name).open("w", newline="", encoding="utf-8") as fh:
                    wr = csv.writer(fh)
                    wr.writerow(["x", "y"])
                    for xv, yv in data:
                        wr.writerow([f"{xv:.6g}", f"{yv:.6g}"])
                series.append({
                    "file": name, "source": "dark", "color": "dark", "label": "dark",
                    "instance": k, "points": len(data), "quality": q,
                    "x_range": [round(min(p[0] for p in data), 4), round(max(p[0] for p in data), 4)],
                    "y_range": [round(min(p[1] for p in data), 4), round(max(p[1] for p in data), 4)],
                })
    else:
        peaks = el.detect_series_colors(hsv, frame, legends, sat_min, val_min)
        for hue_c, count in peaks:
            color = el.hue_name(hue_c)
            kept = 0
            for k, pts in enumerate(
                    el.trace_hue_instances(hsv, frame, hue_c, hue_tol, sat_min, val_min,
                                           legends), start=1):
                data = el.to_data(pts, frame, xmin, xmax, ymin, ymax)
                q = _quality(data, (xmin, xmax), (ymin, ymax))
                if not q["ok"]:
                    if kept == 0:
                        skipped.append({"color": color, "points": len(data),
                                        "reason": "；".join(q["reasons"])})
                    break
                if any(_same_line(pts, prev) for prev in kept_traces):
                    continue
                kept += 1
                kept_traces.append(pts)
                name = f"{image_path.stem}_{color}{'' if k == 1 else f'_{k}'}.csv"
                with (outdir / name).open("w", newline="", encoding="utf-8") as fh:
                    wr = csv.writer(fh)
                    wr.writerow(["x", "y"])
                    for xv, yv in data:
                        wr.writerow([f"{xv:.6g}", f"{yv:.6g}"])
                series.append({
                    "file": name,
                    "source": "hue-cluster",
                    "color": color,
                    "hue": hue_c,
                    "hue_pixels": count,
                    "instance": k,
                    "points": len(data),
                    "quality": q,
                    "x_range": [round(min(p[0] for p in data), 4), round(max(p[0] for p in data), 4)],
                    "y_range": [round(min(p[1] for p in data), 4), round(max(p[1] for p in data), 4)],
                })
    return {
        "panel": image_path.name,
        "frame": [left, top, right, bottom],
        "legend_boxes": legends,
        "series": series,
        "skipped_series": skipped,
    }


def inner_boxes(gray, frame, max_frac=0.92):
    """Regions to mask out before tracing: legend frames and zoom insets.

    Two complementary sources: rectangles formed by long straight dark lines (the
    reliable one) and contour detection (fallback when a border is thin or partly
    covered). Both routinely return the axes box itself - anything that covers almost
    the whole frame is dropped, otherwise the entire plot would be masked away.
    """
    left, top, right, bottom = frame
    fw, fh = max(1, right - left), max(1, bottom - top)
    out = []
    for (l, t, r, b) in el.find_boxes(gray)[1:]:
        if l > left and t > top and r < right and b < bottom:
            out.append((l, t, r - l, b - t))
    for (x, y, w, h) in el.find_inner_boxes((gray < 120).astype(np.uint8) * 255):
        if (x > left and y > top and x + w < right and y + h < bottom
                and w < max_frac * fw and h < max_frac * fh):
            out.append((x, y, w, h))
    keep = []
    for b in out:
        if b[2] < 0.10 * fw or b[3] < 0.07 * fh:
            continue
        if any(abs(b[0] - k[0]) < 8 and abs(b[1] - k[1]) < 8
               and abs(b[2] - k[2]) < 8 and abs(b[3] - k[3]) < 8 for k in keep):
            continue
        keep.append(b)
    return keep


def _frame_of(gray, frame_hint=None):
    """Locate the axes box; fall back to the caller's hint when detection fails.

    Two failure modes showed up on real papers and both used to end with zero points
    extracted: (a) a tight crop puts the frame within the `edge_frac` border that is
    discarded by design, (b) a cropped vector chart has no long enough dark run for
    `find_frame`. When the caller already knows the box (from the PDF itself) there is
    no reason to give up on it.
    """
    try:
        return el.find_frame(gray)
    except RuntimeError:
        pass
    try:
        return el.find_frame(gray, edge_frac=0.0)     # (a)
    except RuntimeError:
        pass
    h, w = gray.shape[:2]
    if frame_hint and 0 <= frame_hint[0] < frame_hint[2] <= w \
            and 0 <= frame_hint[1] < frame_hint[3] <= h:
        return tuple(int(v) for v in frame_hint)      # (b)
    raise RuntimeError("plot frame not found - try lowering --dark-thresh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--range", action="append", required=True,
                    help="xmin,xmax,ymin,ymax (repeatable, in panel order)")
    ap.add_argument("--sat-min", type=int, default=70)
    ap.add_argument("--val-min", type=int, default=40)
    ap.add_argument("--hue-tol", type=int, default=8)
    ap.add_argument("--legend-image", default=None,
                    help="full figure containing the legend; its swatch colours are used")
    ap.add_argument("--legend-colors", default=None,
                    help="comma separated #rrggbb list (skips legend detection)")
    args = ap.parse_args()

    panels_dir = Path(args.panels).resolve()
    prefix = args.prefix or panels_dir.name
    images = sorted(p for p in panels_dir.glob(f"{prefix}_panel[0-9]*.png"))
    if not images:
        images = sorted(p for p in panels_dir.glob("*.png")
                        if "detected" not in p.name and "overlay" not in p.name)
    if not images:
        raise SystemExit(f"no panel images in {panels_dir}")

    ranges = [[float(v) for v in r.split(",")] for r in args.range]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    legend_colors = None
    if args.legend_colors:
        legend_colors = [c.strip() for c in args.legend_colors.split(",") if c.strip()]
        print(f"图例颜色（手动指定）: {legend_colors}")
    elif args.legend_image:
        limg = iio.imread(Path(args.legend_image).resolve())
        if limg is None:
            raise SystemExit(f"cannot read legend image {args.legend_image}")
        lgray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        best = None
        for b in lc.find_boxes(lgray):
            sw = lc.find_swatches(limg, b)
            if len(sw) > 1 and (best is None or len(sw) > len(best[1])):
                best = (b, sw)
        if best is None:
            raise SystemExit("no legend with multiple colour swatches found")
        box, sw = best
        legend_colors = [s["hex"] for s in sw]
        print(f"图例框 {box} -> {len(sw)} 个色块: {legend_colors}")

    results = []
    for i, img in enumerate(images):
        rng = ranges[i] if i < len(ranges) else ranges[-1]
        res = process_panel(img, rng, outdir, args.sat_min, args.val_min, args.hue_tol,
                            legend_colors=legend_colors, legend_image=args.legend_image)
        res["axis_range"] = rng
        results.append(res)
        tag = f"{img.name}  x=[{rng[0]},{rng[1]}] y=[{rng[2]},{rng[3]}]"
        if res.get("error"):
            print(f"  {tag}  ERROR {res['error']}")
            continue
        print(f"  {tag}  frame={res['frame']}  series={len(res['series'])}")
        for s in res["series"]:
            extra = s.get("target_bgr", s.get("hue"))
            print(f"      {s['file']}  n={s['points']}  y=[{s['y_range'][0]},{s['y_range'][1]}]  {extra}")

    report = {"panels_dir": str(panels_dir), "results": results}
    rp = outdir / f"{prefix}_extract.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(len(r.get("series", [])) for r in results)
    print(f"\n共 {len(images)} 个面板 / {total} 条曲线 -> {outdir}")
    print(f"报告 -> {rp}")


if __name__ == "__main__":
    main()
