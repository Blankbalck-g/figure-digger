"""Turn the model's coarse "where the curves are" into precise numbers.

Division of labour (this replaced the older "propose candidate blobs, let the model
accept/reject them" design, which failed in both directions: real curves vanished and
junk survived):

  * the **model** reads the figure and says which data curves exist, roughly where
    they start, pass and end (normalised coordinates, accurate to a few percent), what
    they are called, whether they carry markers, and whether we want points, a line or
    both;
  * the **code** samples the colour at an anchor, follows the stroke from there to both
    sides (bridging dash gaps, ignoring everything that is not on that path) and maps
    the pixels to data values with the axis calibration.

The model's coordinates only have to be good enough to land somewhere on the curve -
a few dozen pixels. Precision comes from the tracer, and a failed trace is reported
instead of silently dropped.
"""

import cv2
import numpy as np
import csv
import re
from pathlib import Path

import extract_lines as el


def safe_tag(text, limit=40):
    t = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(text or "")).strip("_")
    return t[:limit] or "series"


def unique_path(outdir, name):
    """Never let two series write to the same file (the second silently won once)."""
    p = Path(outdir) / name
    if not p.exists():
        return p
    for k in range(2, 100):
        cand = Path(outdir) / f"{p.stem}_{k}{p.suffix}"
        if not cand.exists():
            return cand
    return p


def write_csv(path, data):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x", "y"])
        for xv, yv in data:
            wr.writerow([f"{xv:.6g}", f"{yv:.6g}"])
    return Path(path).name


def anchors_to_pixels(anchors, frame, size):
    """Normalised [x, y] from the model -> pixel coordinates inside the panel image."""
    w, h = size
    out = []
    for a in anchors or []:
        try:
            x, y = float(a[0]), float(a[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            out.append((x * w, y * h))
    return out


def sample_color(img, points, radius=7, sat_floor=35):
    """Colour of the curve near the anchors: the saturated part of a small window.

    The model's hex is a hint, not a measurement; sampling where it said the curve is
    gives the real stroke colour, which is what the mask needs.
    """
    h, w = img.shape[:2]
    patches = []
    for x, y in points:
        x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch):
            patches.append(patch)
    if not patches:
        return None
    px = np.vstack(patches)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[:, 1].astype(int)
    val = hsv[:, 2].astype(int)
    keep = (sat >= max(sat_floor, np.percentile(sat, 75))) & (val >= 40)
    use = px[keep] if keep.sum() >= 4 else px
    return tuple(int(v) for v in np.median(use, axis=0))


def _runs_by_column(mask, frame):
    left, top, right, bottom = frame
    y0, y1 = top + 2, bottom - 1
    return {x: el._column_runs(mask, x, y0, y1) for x in range(left + 1, right)
            if el._column_runs(mask, x, y0, y1)}


def trace_from_seed(hsv, frame, seed, target_bgr, exclude_boxes=(), hue_tol=14,
                    max_jump=28.0, max_gap=90):
    """Follow the stroke that passes near `seed` to both sides."""
    mask = el.color_mask(hsv, frame, target_bgr, hue_tol, exclude_boxes=exclude_boxes)
    runs = _runs_by_column(mask, frame)
    if not runs:
        return []
    sx, sy = float(seed[0]), float(seed[1])
    best = None
    for x, rs in runs.items():
        for cy, _ln in rs:
            if abs(cy - sy) > 45:                # 锚点只给到几十像素精度
                continue
            score = abs(cy - sy) + 0.3 * abs(x - sx)
            if best is None or score < best[0]:
                best = (score, x, cy)
    if best is None:
        return []
    _, x0, y0 = best
    xs = sorted(runs)
    right = el._walk(runs, [x for x in xs if x > x0], y0, max_jump, max_gap)
    left = el._walk(runs, [x for x in xs if x < x0][::-1], y0, max_jump, max_gap)
    pts = sorted(left + [(x0, y0)] + right)
    return el._smooth_despike(el._despeckle(pts))


def trace_best(img, hsv, frame, anchors, color_bgr, exclude_boxes=()):
    """Try every anchor as a seed; keep the longest trace. Returns (trace, info)."""
    traces = []
    for a in anchors:
        t = trace_from_seed(hsv, frame, a, color_bgr, exclude_boxes)
        if t:
            traces.append(t)
    if not traces:
        return [], {"anchors": len(anchors), "seeded": 0}
    best = max(traces, key=len)
    return best, {"anchors": len(anchors), "seeded": len(traces),
                  "coverage": round((max(p[0] for p in best) - min(p[0] for p in best))
                                    / max(1.0, frame[2] - frame[0]), 3)}


def markers_in_corridor(img, hsv, trace, color_bgr, half=22, min_markers=3):
    """Marker glyph centres that sit **on the traced path**.

    Erosion keeps blobs and removes strokes (the tracer's own trick); restricting the
    search to the corridor means text and the other curves' markers can never appear.
    """
    h, w = img.shape[:2]
    corridor = np.zeros((h, w), np.uint8)
    for x, y in trace:
        cv2.circle(corridor, (int(x), int(y)), half, 255, -1)
    mask = el.color_mask(hsv, (0, 0, w - 1, h - 1), color_bgr, 12)
    m = cv2.bitwise_and(mask, corridor)
    if int((m > 0).sum()) < 60:
        return []
    eroded = cv2.erode(m, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(eroded)
    blobs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        blobs.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                      "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                      "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                      "area": int(stats[i, cv2.CC_STAT_AREA])})
    if len(blobs) < min_markers:
        return []
    areas = np.array([b["area"] for b in blobs], float)
    med = float(np.median(areas))
    keep = [b for b in blobs if 0.35 * med <= b["area"] <= 3.0 * med]
    keep.sort(key=lambda b: b["cx"])
    return [(b["cx"], b["cy"]) for b in keep]
