"""Read the series colours out of a figure's legend.

Clustering hues inside a plot is unreliable for dark red / dark blue series (hue wraps
at 0/180 and anti-aliasing spreads the cluster). The legend already states the exact
colour of every series, so we detect the little colour swatches next to each label and
use those as targets.

Usage:
  python legend_colors.py figure.png                  # auto-detect the legend box
  python legend_colors.py figure.png --box 516,546,351,141
  python legend_colors.py figure.png --debug out/legend_debug.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import imgio as iio
import numpy as np

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def find_boxes(gray, dark_thresh=120):
    """Rectangular outlines in the image (candidate legend boxes)."""
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.08 * w or bh < 0.03 * h or bw * bh > 0.6 * w * h:
            continue
        sub = dark[y:y + bh, x:x + bw] > 0
        # single-pixel strips: journal figures often draw a 1px legend border, which a
        # 2px strip test rejects (coverage lands around 0.5)
        border = np.concatenate([sub[:1, :].ravel(), sub[-1:, :].ravel(),
                                 sub[:, :1].ravel(), sub[:, -1:].ravel()])
        if border.mean() < 0.7:
            continue
        boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: -b[2] * b[3])
    kept = []
    for b in boxes:
        if any(abs(b[0] - k[0]) < 12 and abs(b[1] - k[1]) < 12
               and abs(b[2] - k[2]) < 12 and abs(b[3] - k[3]) < 12 for k in kept):
            continue
        kept.append(b)
    return kept


def find_swatches(img, box, min_run=20, min_sat=40, merge_gap=3):
    """Locate the colourful line samples in a legend box.

    Legends come in two layouts - entries stacked vertically (swatches share an x) or
    side by side (swatches share a y) - so rather than assume a layout this keys off
    saturation: the swatches of *coloured* series are the only saturated elements in a
    legend. Grey/black entries (error bands, dash-dot reference curves) are skipped on
    purpose: matching those by colour would collide with the axes, grid and text.
    """
    x, y, w, h = box
    pad = 3
    crop = img[y + pad:y + h - pad, x + pad:x + w - pad]
    if crop.size == 0:
        return []
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    nonwhite = g < 245
    ch, cw = nonwhite.shape

    # every run in a row, not just the longest: a horizontal legend holds several
    # swatches side by side and the shorter ones would otherwise be swallowed
    groups = []
    for ry in range(ch):
        row = nonwhite[ry]
        run_start = None
        for i in range(cw + 1):
            v = row[i] if i < cw else False
            if v and run_start is None:
                run_start = i
            elif not v and run_start is not None:
                if i - run_start >= min_run:
                    x0, x1 = run_start, i - 1
                    for grp in groups:
                        gy, gx0, gx1 = grp[-1]
                        if ry - gy <= merge_gap and not (x1 < gx0 - 2 or x0 > gx1 + 2):
                            grp.append((ry, x0, x1))
                            break
                    else:
                        groups.append([(ry, x0, x1)])
                run_start = None

    swatches = []
    for g_rows in groups:
        pixels = []
        for ry, x0, x1 in g_rows:
            seg = crop[ry, x0:x1 + 1]
            pixels.extend(seg.reshape(-1, 3))
        if not pixels:
            continue
        arr = np.array(pixels)
        # drop anti-aliased edge pixels, which drag the median toward the background
        core = arr[arr.max(axis=1) < 200] if (arr.max(axis=1) < 200).any() else arr
        color = np.median(core, axis=0)
        sat = int(color.max()) - int(color.min())
        if sat < min_sat:
            continue
        x0 = min(r[1] for r in g_rows)
        x1 = max(r[2] for r in g_rows)
        y0 = min(r[0] for r in g_rows)
        y1 = max(r[0] for r in g_rows)
        swatches.append({
            "color_bgr": [int(c) for c in color],
            "hex": "#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0])),
            "sat": sat,
            "row_y": y + pad + (y0 + y1) // 2,
            "x0": x + pad + x0,
            "x1": x + pad + x1,
            "thickness_px": len(g_rows),
            "run_len": x1 - x0 + 1,
        })
    swatches.sort(key=lambda s: (s["row_y"], s["x0"]))
    swatches += find_marker_swatches(img, box, min_sat)
    return merge_similar(swatches)


def find_marker_swatches(img, box, min_sat=40, min_area=25, max_side=48, max_area=2500):
    """Legend entries drawn as coloured markers (circles / squares) instead of lines.

    Many journals key a scatter series with a dot; a dot is far too short to show up in
    the horizontal-run scan, so those legends need connected-component detection.
    """
    x, y, w, h = box
    pad = 3
    crop = img[y + pad:y + h - pad, x + pad:x + w - pad]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] >= min_sat) & (hsv[:, :, 2] >= 40)).astype(np.uint8) * 255
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < min_area or area > max_area or bw > max_side or bh > max_side:
            continue
        if bw < 4 or bh < 4:
            continue
        comp = crop[labels == i]
        core = comp[comp.max(axis=1) < 200] if (comp.max(axis=1) < 200).any() else comp
        color = np.median(core, axis=0)
        sat = int(color.max()) - int(color.min())
        if sat < min_sat:
            continue
        out.append({
            "color_bgr": [int(c) for c in color],
            "hex": "#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0])),
            "sat": sat,
            "row_y": int(y + pad + cents[i][1]),
            "x0": int(x + pad + bx),
            "x1": int(x + pad + bx + bw),
            "thickness_px": int(bh),
            "run_len": int(bw),
            "shape": "marker",
        })
    return out


def merge_similar(swatches, hue_tol=12, dist_tol=60):
    """Collapse swatches that are really the same colour sampled at different spots.

    The same series is often detected twice (a line sample plus its marker) and
    anti-aliasing shifts the shade slightly, so merge on hue *or* colour distance and
    keep the most saturated sample.
    """
    def hue_of(s):
        b, g, r = s["color_bgr"]
        h, _, _ = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        return int(h)

    kept = []
    for s in sorted(swatches, key=lambda x: (-x["sat"], -x["run_len"])):
        hs = hue_of(s)
        cs = np.array(s["color_bgr"], dtype=float)
        dup = False
        for k in kept:
            hk = hue_of(k)
            d = min(abs(hs - hk), 180 - abs(hs - hk))
            cdist = float(np.linalg.norm(cs - np.array(k["color_bgr"], dtype=float)))
            if d <= hue_tol or cdist <= dist_tol:
                dup = True
                break
        if not dup:
            s["hue"] = hs
            kept.append(s)
    kept.sort(key=lambda x: (x["row_y"], x["x0"]))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--box", default=None, help="x,y,w,h of the legend box")
    ap.add_argument("--debug", default=None)
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--min-sat", type=int, default=40)
    args = ap.parse_args()

    src = Path(args.image).resolve()
    img = iio.imread(src)
    if img is None:
        raise SystemExit("cannot read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    boxes = []
    if args.box:
        boxes = [tuple(int(float(v)) for v in args.box.split(","))]
    else:
        boxes = find_boxes(gray)
    if not boxes:
        raise SystemExit("no legend box found - pass --box")

    print(f"{src.name} {w}x{h}  候选图例框 {len(boxes)} 个")
    vis = img.copy()
    for b in boxes[:4]:
        sw = find_swatches(img, b, args.min_run, args.min_sat)
        print(f"  box {b}: {len(sw)} 个色块")
        cv2.rectangle(vis, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 160, 0), 2)
        for s in sw:
            print(f"     {s['hex']}  BGR={s['color_bgr']}  饱和度={s['sat']}  y={s['row_y']} "
                  f"x={s['x0']}-{s['x1']}  厚度={s['thickness_px']}px")
            cv2.rectangle(vis, (s["x0"], s["row_y"] - 3), (s["x1"], s["row_y"] + 3), (0, 0, 255), 1)

    if args.debug:
        d = Path(args.debug)
        d.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(d, vis)
        print(f"debug -> {d}")

    if boxes:
        sw = find_swatches(img, boxes[0], args.min_run, args.min_sat)
        print("\nlegend_colors 参数（供批处理使用）:")
        print("  " + " ".join(s["hex"] for s in sw))


if __name__ == "__main__":
    main()
