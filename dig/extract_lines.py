"""Semi-automatic line-chart extraction from a figure image.

Pipeline:
  1. locate the plot frame (longest contiguous dark run -> top/bottom/left/right edges)
  2. find and exclude legend boxes (dark rectangles inside the frame)
  3. auto-detect data-series colours by hue histogram of saturated pixels
  4. trace each series column by column (median y per x)
  5. write CSV + a verification overlay PNG

Usage:
  python extract_lines.py fig.png --xmin 0 --xmax 5 --ymin 0 --ymax 70 \
         --xlabel "Time (ms)" --ylabel "Axial penetration (mm)" [--out out/fig5] [--debug]
"""

import argparse
import csv
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


def longest_run(mask_1d):
    """Length and start index of the longest True run."""
    best_len, best_start, cur_start = 0, -1, None
    for i, v in enumerate(mask_1d):
        if v:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                if i - cur_start > best_len:
                    best_len, best_start = i - cur_start, cur_start
                cur_start = None
    if cur_start is not None and len(mask_1d) - cur_start > best_len:
        best_len, best_start = len(mask_1d) - cur_start, cur_start
    return best_len, best_start


def _line_candidates(runs, length, thresh, group_gap=3):
    """Indices whose run exceeds thresh, grouped into distinct lines."""
    ok = [i for i, r in enumerate(runs) if r > thresh]
    groups, cur = [], []
    for i in ok:
        if cur and i - cur[-1] > group_gap:
            groups.append(int(np.mean(cur)))
            cur = []
        cur.append(i)
    if cur:
        groups.append(int(np.mean(cur)))
    return groups


def _box_edges_dark(dark, box, min_cover=0.6):
    """True when all four edges of the box are really drawn."""
    left, top, right, bottom = box
    h, w = dark.shape
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        return False
    cover = min(dark[top, left:right].mean(), dark[bottom, left:right].mean(),
                dark[top:bottom, left].mean(), dark[top:bottom, right].mean())
    return cover >= min_cover


def find_boxes(gray, dark_thresh=120, min_run_frac=0.4, min_frac=0.22):
    """Every rectangle whose four edges are long straight dark lines, largest first.

    The axes box, a legend frame and a zoom inset are all built from the same kind of
    long straight lines. The old code took the outermost pair of long lines and threw
    away candidates close to the crop edge - on a tight crop that deleted the *real*
    top edge and returned a 105px-tall strip spanning the legend box instead of the
    plot (measured on 2014-01-1413 p14, and the same thing happens with an inset in
    the top-left corner). So: enumerate boxes, keep the ones whose edges are genuinely
    drawn, and let the largest win.
    """
    dark = gray < dark_thresh
    h, w = dark.shape
    row_runs = [longest_run(dark[y, :])[0] for y in range(h)]
    col_runs = [longest_run(dark[:, x])[0] for x in range(w)]
    rows = _line_candidates(row_runs, h, min_run_frac * w)
    cols = _line_candidates(col_runs, w, min_run_frac * h)
    if len(rows) < 2 or len(cols) < 2:
        return []
    boxes = []
    for i, top in enumerate(rows):
        for bottom in rows[i + 1:]:
            if bottom - top < min_frac * h:
                continue
            for k, left in enumerate(cols):
                for right in cols[k + 1:]:
                    if right - left < min_frac * w:
                        continue
                    if _box_edges_dark(dark, (left, top, right, bottom)):
                        boxes.append((left, top, right, bottom))
    boxes.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    # 只保留最大的那些：被别的框包住的（嵌套边框、图例框）丢掉
    out = []
    for b in boxes:
        if any(b[0] >= o[0] and b[1] >= o[1] and b[2] <= o[2] and b[3] <= o[3] for o in out):
            continue
        out.append(b)
    # 整幅图的外边框不是坐标框（它只是裁剪痕迹），排到后面
    def is_image_border(b):
        return (b[0] <= 3 and b[1] <= 3 and b[2] >= w - 4 and b[3] >= h - 4)
    out.sort(key=lambda b: (is_image_border(b), -((b[2] - b[0]) * (b[3] - b[1]))))
    return out


def find_frame(gray, dark_thresh=120, min_run_frac=0.4, edge_frac=0.03):
    """Locate the axes box (see find_boxes). `edge_frac` is kept for API compatibility."""
    boxes = find_boxes(gray, dark_thresh, min_run_frac)
    if boxes:
        return boxes[0]
    raise RuntimeError("plot frame not found - try lowering --dark-thresh")


def find_inner_boxes(binary_dark):
    """Rectangular outlines inside the figure (legend / parameter boxes).

    A candidate must actually look like a rectangle border: its bounding-box edge must
    be almost fully dark and its interior must not be dense. Without this test, a region
    merely *bounded* by curves is mistaken for a legend and the curves inside it get
    masked away.
    """
    contours, _ = cv2.findContours(binary_dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 30:
            continue
        sub = binary_dark[y:y + h, x:x + w] > 0
        if sub.size == 0:
            continue
        border = np.concatenate([sub[:2, :].ravel(), sub[-2:, :].ravel(),
                                 sub[:, :2].ravel(), sub[:, -2:].ravel()])
        if border.mean() < 0.75:
            continue
        if w > 10 and h > 10:
            inner = sub[3:-3, 3:-3]
            if inner.size and inner.mean() > 0.5:
                continue
        boxes.append((x, y, w, h))
    # drop boxes that contain another candidate (nested inner/outer contours)
    unique = []
    for b in sorted(boxes, key=lambda t: t[2] * t[3]):
        if any(b[0] >= u[0] and b[1] >= u[1] and b[0] + b[2] <= u[0] + u[2]
               and b[1] + b[3] <= u[1] + u[3] for u in unique):
            continue
        unique.append(b)
    return unique


def hue_name(hue_center):
    """Rough colour name for a hue in OpenCV's 0..179 scale."""
    h = hue_center % 180
    if h <= 12 or h >= 165:
        return "red"
    if 13 <= h <= 35:
        return "orange"
    if 36 <= h <= 75:
        return "green"
    if 76 <= h <= 105:
        return "teal"
    if 106 <= h <= 140:
        return "blue"
    return "violet"


def hex_to_bgr(text):
    t = text.strip().lstrip("#")
    r, g, b = int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
    return [b, g, r]


def bgr_to_hsv(color):
    px = np.uint8([[color]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
    return int(h), int(s), int(v)


def _column_runs(mask, x, y0, y1):
    """Contiguous vertical runs of set pixels in one column -> [(center_y, length)]."""
    col = mask[y0:y1, x]
    idx = np.nonzero(col)[0]
    runs = []
    if idx.size == 0:
        return runs
    start = prev = idx[0]
    for v in idx[1:]:
        if v == prev + 1:
            prev = v
        else:
            runs.append(((start + prev) / 2.0 + y0, prev - start + 1))
            start = prev = v
    runs.append(((start + prev) / 2.0 + y0, prev - start + 1))
    return runs


def _walk(runs_by_col, order, prev_y, max_jump, max_gap, slope=0.0):
    """Follow one line column by column, always taking the run closest to the last y.

    Taking the median of everything that matches in a column breaks as soon as
    something else shares the colour: a marker, an error bar or a text annotation in
    the same column pulls the median into the empty gap between them, which shows up
    as spurious oscillation in the extracted data. Continuity plus a limited gap
    tolerance also lets dashed / dash-dot lines through (the gaps are just skipped).

    The next position is *predicted* from the recent slope, so the tolerance stays
    tight for adjacent columns (no hopping onto a nearby text annotation) while a dash
    gap of 20 columns is still bridged on a steeply rising line.
    """
    pts = []
    gap = 0
    last_x = None
    for x in order:
        cands = runs_by_col.get(x)
        best = None
        if cands:
            step = 1 if last_x is None else max(1, abs(x - last_x))
            predict = prev_y + slope * (0 if last_x is None else (x - last_x))
            tol = max_jump + (step - 1) * abs(slope) * 1.6
            for cy, _ln in cands:
                d = abs(cy - predict)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, cy)
        if best is None:
            gap += 1
            if gap > max_gap:
                break
            continue
        gap = 0
        prev_y = best[1]
        last_x = x
        pts.append((x, prev_y))
        if len(pts) >= 4:                      # 用最近几列估斜率，抗单点抖动
            x0, y0 = pts[-4]
            if x - x0 >= 2:
                slope = float(np.clip((prev_y - y0) / (x - x0), -3.0, 3.0))
    return pts


def _trace_score(pts, frame_w):
    """Prefer the trajectory that covers the most columns and wobbles the least."""
    if len(pts) < 8:
        return -1e9
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    span = (xs.max() - xs.min()) / max(1.0, float(frame_w))
    rough = float(np.median(np.abs(np.diff(ys)))) if ys.size > 1 else 0.0
    return span * 1000.0 - rough * 8.0


def _despeckle(pts, max_dev=25.0, window=2):
    """Drop single-column outliers (arrow tips, glyphs crossing the line)."""
    if len(pts) < 5:
        return pts
    ys = np.array([p[1] for p in pts], dtype=float)
    keep = []
    for i, (x, y) in enumerate(pts):
        lo, hi = max(0, i - window), min(len(pts), i + window + 1)
        neigh = np.delete(ys[lo:hi], i - lo)
        if neigh.size and abs(y - np.median(neigh)) > max_dev:
            continue
        keep.append((x, y))
    return keep


def _running_median(values, win):
    """Median over a sliding window, edges clipped (no scipy dependency)."""
    n = len(values)
    half = max(1, win // 2)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def _smooth_despike(pts, win=25, min_dev=9.0, mad_k=4.0):
    """Remove short excursions away from the line's own trend.

    A greedy walk can hop onto a text annotation that happens to share the colour and
    come back a few columns later (that is the remaining ±28px jitter). Comparing each
    point with a running median of the trace separates such excursions from real
    steepness, which the running median follows.
    """
    if len(pts) < 12:
        return pts
    ys = np.array([p[1] for p in pts], dtype=float)
    base = _running_median(ys, win)
    resid = ys - base
    mad = float(np.median(np.abs(resid - np.median(resid))))
    thr = max(min_dev, mad_k * mad)
    keep = [(x, y) for (x, y), r in zip(pts, resid) if abs(r) <= thr]
    return keep if len(keep) >= 8 else pts


def _trace_mask(mask, frame, max_jump=28.0, max_gap=45, seed_step=24):
    """Trace whatever line the mask contains: multi-seed walk, keep the best one."""
    left, top, right, bottom = frame
    y0, y1 = top + 2, bottom - 1
    runs_by_col = {}
    for x in range(left + 1, right):
        r = _column_runs(mask, x, y0, y1)
        if r:
            runs_by_col[x] = r
    if not runs_by_col:
        return []
    xs_sorted = sorted(runs_by_col)
    frame_w = max(1, right - left)

    seeds = []
    for i in range(0, len(xs_sorted), seed_step):
        x = xs_sorted[i]
        for cy, _ln in runs_by_col[x]:
            seeds.append((x, cy))

    best, best_score = [], -1e9
    for sx, sy in seeds:
        right_pts = _walk(runs_by_col, [x for x in xs_sorted if x > sx], sy,
                          max_jump, max_gap, 0.0)
        left_pts = _walk(runs_by_col, [x for x in xs_sorted if x < sx][::-1], sy,
                         max_jump, max_gap, 0.0)
        pts = sorted(left_pts + [(sx, sy)] + right_pts)
        pts = _smooth_despike(_despeckle(pts))
        sc = _trace_score(pts, frame_w)
        if sc > best_score:
            best, best_score = pts, sc
    return best


def trace_series_by_target(hsv, frame, target_bgr, hue_tol=14, sat_frac=0.45,
                           min_sat=45, val_frac=0.25, exclude_boxes=()):
    """Trace a series whose exact colour is known from the legend.

    Matching on hue + saturation around the legend colour is far steadier than
    clustering hues in the plot: black text and grey grid lines have low saturation and
    drop out, and near-hue series cannot merge because the centres are fixed by the
    legend rather than guessed.
    """
    th, ts, tv = bgr_to_hsv(target_bgr)
    hue = hsv[:, :, 0].astype(int)
    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)

    dh = np.abs(hue - th)
    dh = np.minimum(dh, 180 - dh)
    mask = ((dh <= hue_tol) & (sat >= max(min_sat, sat_frac * ts)) & (val >= val_frac * tv))
    mask = mask.astype(np.uint8) * 255

    left, top, right, bottom = frame
    mask[:top + 2, :] = 0
    mask[bottom:, :] = 0
    mask[:, :left + 2] = 0
    mask[:, right:] = 0
    for (x, y, w, h) in exclude_boxes:
        if left < x and y > top and x + w < right and y + h < bottom:
            mask[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] = 0

    return _trace_mask(mask, frame)


def trace_series_instances(hsv, frame, target_bgr, hue_tol=14, sat_frac=0.45,
                           min_sat=45, val_frac=0.25, exclude_boxes=(),
                           max_instances=5, min_points=25, min_span=0.28,
                           erase_width=45, strict_tol=8):
    """Trace every line of this colour, not just the best one.

    Figures routinely draw two or three series in the same colour (different
    conditions, experiment vs model), so tracing once loses data. After a trace, its
    *connected components* are erased and the search repeats on what is left; a trace
    that no longer covers a decent share of the plot is the signal to stop.

    The erasure uses a wide corridor (~one marker radius) rather than the bare path:
    a series is drawn as a line **plus its markers**, and tracing the markers' outlines
    produced a second "series" 14-17px away that no distance test can separate from a
    genuinely parallel curve (measured: real neighbouring curves sit 8.5-19px apart).
    Within one colour, anything that close is the same series, so it goes with it.

    Two passes on purpose: a tight hue tolerance first, so each legend colour traces
    *its own* line (two neighbouring legend colours like cyan #26c6bc and blue #0478c5
    share a wide band when the tolerance is loose, and each would trace the same, more
    prominent line). Only if that finds nothing does it fall back to the loose one, for
    legends whose swatch colour differs a little from the drawn line.
    """
    tols = [strict_tol, hue_tol] if strict_tol and strict_tol < hue_tol else [hue_tol]
    for tol in tols:
        mask = color_mask(hsv, frame, target_bgr, tol, sat_frac, min_sat, val_frac,
                          exclude_boxes)
        inst = _instances_from_mask(mask, frame, max_instances, min_points, min_span,
                                    erase_width)
        if inst:
            return inst
    return []


def color_mask(hsv, frame, target_bgr, hue_tol=14, sat_frac=0.45, min_sat=45,
               val_frac=0.25, exclude_boxes=()):
    """Pixels belonging to a legend colour, clipped to the plot area."""
    th, ts, tv = bgr_to_hsv(target_bgr)
    hue = hsv[:, :, 0].astype(int)
    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)
    dh = np.abs(hue - th)
    dh = np.minimum(dh, 180 - dh)
    mask = ((dh <= hue_tol) & (sat >= max(min_sat, sat_frac * ts)) & (val >= val_frac * tv))
    return _clip_and_mask(mask.astype(np.uint8) * 255, frame, exclude_boxes)


def trace_hue_instances(hsv, frame, hue_center, hue_tol=14, sat_min=70, val_min=40,
                        legend_boxes=(), max_instances=3, min_points=25,
                        min_span=0.28, erase_width=9):
    """Same as trace_series_instances but for a hue cluster (no legend colour known)."""
    hue = hsv[:, :, 0].astype(int)
    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)
    dh = np.abs(hue - hue_center)
    dh = np.minimum(dh, 180 - dh)
    mask = ((dh <= hue_tol) & (sat >= sat_min) & (val >= val_min)).astype(np.uint8) * 255
    mask = _clip_and_mask(mask, frame, legend_boxes)
    return _instances_from_mask(mask, frame, max_instances, min_points, min_span,
                                erase_width)


def dark_mask(gray, frame, exclude_boxes=(), dark_thresh=100, line_frac=0.55):
    """Dark pixels that are line-like: no plot frame, no grid, no tick labels.

    Black data lines were skipped by design because "black" also means the axes, the
    dashed grid and every text annotation. Removing the long straight rules (a row or
    column that is mostly dark inside the frame is a grid/frame line) plus the caller's
    excluded boxes leaves the curves, the dashes and the text; the text is then dropped
    by the continuity tracker and the credibility gate. Callers should only ask for
    this when the model has confirmed the figure really has a dark data line.
    """
    left, top, right, bottom = frame
    mask = (gray < dark_thresh).astype(np.uint8) * 255
    mask = _clip_and_mask(mask, frame, exclude_boxes)
    if right - left < 10 or bottom - top < 10:
        return mask
    inside = mask[top + 2:bottom - 1, left + 2:right - 1]
    row_cov = (inside > 0).mean(axis=1)
    col_cov = (inside > 0).mean(axis=0)
    for i, cov in enumerate(row_cov):
        if cov > line_frac:
            mask[top + 2 + i, :] = 0
    for i, cov in enumerate(col_cov):
        if cov > line_frac:
            mask[:, left + 2 + i] = 0
    return mask


def trace_dark_instances(gray, frame, exclude_boxes=(), dark_thresh=100,
                         max_instances=3, min_points=25, min_span=0.28, erase_width=45,
                         max_run=9.0):
    mask = dark_mask(gray, frame, exclude_boxes, dark_thresh)
    return _instances_from_mask(mask, frame, max_instances, min_points, min_span,
                                erase_width,
                                validator=lambda pts, m: _median_run_len(m, pts) <= max_run)


def _median_run_len(mask, pts):
    """Median vertical extent of the mask where the trace runs.

    A dashed line gives short runs (its stroke thickness); a chain of error bars gives
    long ones. This is what stops error-bar caps from being reported as a black curve.
    """
    h = mask.shape[0]
    lens = []
    for x, y in pts:
        yy = int(round(y))
        if not (0 <= yy < h):
            continue
        col = mask[:, x] > 0
        a = b = yy
        while a > 0 and col[a - 1]:
            a -= 1
        while b + 1 < h and col[b + 1]:
            b += 1
        lens.append(b - a + 1)
    return float(np.median(lens)) if lens else 0.0


def _clip_and_mask(mask, frame, exclude_boxes):
    left, top, right, bottom = frame
    mask[:top + 2, :] = 0
    mask[bottom:, :] = 0
    mask[:, :left + 2] = 0
    mask[:, right:] = 0
    fw, fh = max(1, right - left), max(1, bottom - top)
    for (x, y, w, h) in exclude_boxes:
        mask[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] = 0
        # 图例框 / 放大子图的**刻度文字**常画在框外一圈（inset 的 70/60/50 就标在
        # 框左边约 40px 处），只清框内救不了它们。所以再清掉框周围一圈里的
        # 小碎块——只删小块（文字/刻度），细长的线段一律保留。
        m = 46
        x0, y0 = max(left, x - m), max(top, y - m)
        x1, y1 = min(right, x + w + m), min(bottom, y + h + m)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        ring = np.zeros_like(mask)
        ring[y0:y1, x0:x1] = 255
        ring[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] = 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.bitwise_and(mask, ring))
        for i in range(1, n):
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            cx = stats[i, cv2.CC_STAT_LEFT]
            cy = stats[i, cv2.CC_STAT_TOP]
            fully_in_ring = (cx >= x0 and cy >= y0 and cx + cw <= x1 and cy + ch <= y1)
            if fully_in_ring and cw < 0.06 * fw and ch < 0.06 * fh:
                mask[labels == i] = 0
    return mask


def _instances_from_mask(mask, frame, max_instances, min_points, min_span, erase_width,
                         validator=None):
    left, _, right, _ = frame
    frame_w = max(1, right - left)
    out = []
    for _ in range(max(1, max_instances)):
        pts = _trace_mask(mask, frame)
        if len(pts) < min_points:
            break
        xs = [p[0] for p in pts]
        if (max(xs) - min(xs)) < min_span * frame_w:
            break
        ok = True if validator is None else validator(pts, mask)
        _erase_trace(mask, pts, erase_width)
        if ok:
            out.append(pts)
        elif out:
            break
    return out


def _erase_trace(mask, pts, width=9):
    """Delete every connected component the traced path runs through."""
    if len(pts) < 2:
        return
    corridor = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i in range(len(pts) - 1):
        cv2.line(corridor, (int(pts[i][0]), int(pts[i][1])),
                 (int(pts[i + 1][0]), int(pts[i + 1][1])), 255, max(1, width))
    n, labels = cv2.connectedComponents(mask)
    hit = np.unique(labels[corridor > 0])
    for lab in hit:
        if lab:
            mask[labels == lab] = 0


def _circular_mean(hues):
    ang = np.radians(np.asarray(hues, dtype=float) * 2.0)
    return float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180.0


def detect_series_colors(hsv, frame, legend_boxes, sat_min=70, val_min=40,
                         min_share=0.10, merge_dist=26):
    """Dominant hues among saturated pixels inside the frame.

    Red straddles the 0/180 hue boundary, so peaks are clustered circularly - without
    this a single red curve shows up as two separate series.
    """
    left, top, right, bottom = frame
    region_mask = np.zeros(hsv.shape[:2], np.uint8)
    region_mask[top + 2:bottom - 1, left + 2:right - 1] = 255
    for (x, y, w, h) in legend_boxes:
        if left < x and y > top and x + w < right and y + h < bottom:
            region_mask[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = 0

    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)
    hue = hsv[:, :, 0].astype(int)
    sel = (region_mask > 0) & (sat >= sat_min) & (val >= val_min)
    hues = hue[sel]
    if hues.size == 0:
        return []

    hist = np.bincount(hues, minlength=180).astype(float)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    padded = np.concatenate([hist[-3:], hist, hist[:3]])
    smooth = np.convolve(padded, kernel, mode="same")[3:-3]

    peaks = []
    for h in range(180):
        neighbours = smooth[[(h + d) % 180 for d in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)]]
        if smooth[h] >= neighbours.max() and smooth[h] > 0:
            peaks.append(h)
    peaks = [p for p in peaks if smooth[p] >= min_share * smooth.max()]
    if not peaks:
        peaks = [int(np.argmax(smooth))]

    clusters = []
    for p in sorted(peaks, key=lambda q: -smooth[q]):
        for c in clusters:
            d = min(abs(p - c[0]), 180 - abs(p - c[0]))
            if d <= merge_dist:
                c[1].append(p)
                break
        else:
            clusters.append([p, [p]])

    out = []
    for _, members in clusters:
        center = int(round(_circular_mean(members)))
        count = int(sum(hist[h] for h in range(180)
                        if min(abs(h - center), 180 - abs(h - center)) <= merge_dist))
        out.append((center, count))
    out.sort(key=lambda t: -t[1])
    return out


def trace_series(hsv, frame, hue_center, hue_tol, sat_min=70, val_min=40, legend_boxes=()):
    left, top, right, bottom = frame
    mask = np.zeros(hsv.shape[:2], np.uint8)
    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)
    hue = hsv[:, :, 0].astype(int)
    dh = np.abs(hue - hue_center)
    dh = np.minimum(dh, 180 - dh)
    mask[(dh <= hue_tol) & (sat >= sat_min) & (val >= val_min)] = 255
    mask[:top + 2, :] = 0
    mask[bottom:, :] = 0
    mask[:, :left + 2] = 0
    mask[:, right:] = 0
    for (x, y, w, h) in legend_boxes:
        if left < x and y > top and x + w < right and y + h < bottom:
            mask[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] = 0

    pts = []
    for x in range(left + 1, right):
        col = np.nonzero(mask[:, x])[0]
        if col.size == 0:
            continue
        pts.append((x, float(np.median(col))))
    return pts


def to_data(pts, frame, xmin, xmax, ymin, ymax):
    left, top, right, bottom = frame
    out = []
    for px, py in pts:
        xv = xmin + (px - left) * (xmax - xmin) / (right - left)
        yv = ymax - (py - top) * (ymax - ymin) / (bottom - top)
        out.append((xv, yv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--xmin", type=float, required=True)
    ap.add_argument("--xmax", type=float, required=True)
    ap.add_argument("--ymin", type=float, required=True)
    ap.add_argument("--ymax", type=float, required=True)
    ap.add_argument("--xlabel", default="x")
    ap.add_argument("--ylabel", default="y")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sat-min", type=int, default=70)
    ap.add_argument("--val-min", type=int, default=40)
    ap.add_argument("--hue-tol", type=int, default=8)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    src = Path(args.image).resolve()
    img = iio.imread(src)
    if img is None:
        raise SystemExit(f"cannot read image: {src}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = find_frame(gray)
    left, top, right, bottom = frame
    print(f"image {img.shape[1]}x{img.shape[0]}  frame=({left},{top})-({right},{bottom})")

    inner = find_inner_boxes((gray < 120).astype(np.uint8) * 255)
    legends = [b for b in inner if left < b[0] and b[1] > top
               and b[0] + b[2] < right and b[1] + b[3] < bottom
               and b[2] > 0.15 * (right - left)]
    print(f"legend boxes excluded: {legends}")

    peaks = detect_series_colors(hsv, frame, legends, args.sat_min, args.val_min)
    print(f"detected hue peaks: {peaks}")

    outbase = Path(args.out).resolve() if args.out else src.with_suffix("")
    outbase.parent.mkdir(parents=True, exist_ok=True)
    overlay = img.copy()

    summary = []
    for idx, (hue_c, count) in enumerate(peaks, start=1):
        pts = trace_series(hsv, frame, hue_c, args.hue_tol, args.sat_min, args.val_min, legends)
        data = to_data(pts, frame, args.xmin, args.xmax, args.ymin, args.ymax)
        if len(data) < 20:
            continue
        name = f"series{idx}"
        csv_path = Path(f"{outbase}_{name}.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow([args.xlabel, args.ylabel])
            for xv, yv in data:
                wr.writerow([f"{xv:.6g}", f"{yv:.6g}"])
        color = cv2.cvtColor(np.uint8([[[hue_c, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        for px, py in pts:
            cv2.circle(overlay, (px, int(py)), 1, (0, 0, 0), -1)
        summary.append((name, f"hue={hue_c}", len(data), str(csv_path)))
        print(f"  {name}: hue={hue_c} pixels={count} points={len(data)} -> {csv_path.name}")

    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 200, 0), 2)
    for (x, y, w, h) in legends:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 140, 255), 2)
    overlay_path = Path(f"{outbase}_overlay.png")
    iio.imwrite(overlay_path, overlay)
    print(f"overlay -> {overlay_path}")
    if not summary:
        print("no series extracted - try tuning --sat-min / --hue-tol")


if __name__ == "__main__":
    main()
