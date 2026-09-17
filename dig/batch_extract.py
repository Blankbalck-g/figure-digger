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

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def process_panel(image_path, rng, outdir, sat_min, val_min, hue_tol,
                  legend_colors=None, legend_image=None, name_map=None, series_y=None,
                  frame_hint=None):
    xmin, xmax, ymin, ymax = rng
    img = iio.imread(image_path)
    if img is None:
        return {"panel": image_path.name, "error": "unreadable"}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left, top, right, bottom = _frame_of(gray, frame_hint)
    inner = el.find_inner_boxes((gray < 120).astype(np.uint8) * 255)
    legends = [b for b in inner if left < b[0] and b[1] > top
               and b[0] + b[2] < right and b[1] + b[3] < bottom
               and b[2] > 0.15 * (right - left)]

    series = []
    if legend_colors:
        # legend-driven: targets come from the figure legend, order is preserved
        for spec in legend_colors:
            target = el.hex_to_bgr(spec) if isinstance(spec, str) else list(spec)
            hexs = "#{:02x}{:02x}{:02x}".format(int(target[2]), int(target[1]), int(target[0]))
            # each series may read a different y axis (dual-axis figures)
            y_rng = ((series_y or {}).get(hexs) or (series_y or {}).get(hexs.upper())
                     or [ymin, ymax])
            pts = el.trace_series_by_target(hsv, (left, top, right, bottom), target,
                                            hue_tol, exclude_boxes=legends)
            data = el.to_data(pts, (left, top, right, bottom), xmin, xmax,
                              y_rng[0], y_rng[1])
            if len(data) < 20:
                continue
            h, s, v = el.bgr_to_hsv(target)
            label = el.hue_name(h)
            nice = (name_map or {}).get(hexs) or (name_map or {}).get(hexs.upper())
            tag = label
            if nice:
                import re as _re
                tag = _re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", nice).strip("_")[:40]
            name = f"{image_path.stem}_{tag}.csv"
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
                "y_axis_range": list(y_rng),
                "points": len(data),
                "x_range": [round(min(p[0] for p in data), 4), round(max(p[0] for p in data), 4)],
                "y_range": [round(min(p[1] for p in data), 4), round(max(p[1] for p in data), 4)],
            })
    else:
        peaks = el.detect_series_colors(hsv, (left, top, right, bottom), legends, sat_min, val_min)
        for hue_c, count in peaks:
            pts = el.trace_series(hsv, (left, top, right, bottom), hue_c, hue_tol, sat_min, val_min, legends)
            data = el.to_data(pts, (left, top, right, bottom), xmin, xmax, ymin, ymax)
            if len(data) < 20:
                continue
            color = el.hue_name(hue_c)
            name = f"{image_path.stem}_{color}.csv"
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
                "points": len(data),
                "x_range": [round(min(p[0] for p in data), 4), round(max(p[0] for p in data), 4)],
                "y_range": [round(min(p[1] for p in data), 4), round(max(p[1] for p in data), 4)],
            })
    return {
        "panel": image_path.name,
        "frame": [left, top, right, bottom],
        "legend_boxes": legends,
        "series": series,
    }


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
