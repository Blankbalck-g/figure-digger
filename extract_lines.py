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


def find_frame(gray, dark_thresh=120, min_run_frac=0.4, edge_frac=0.03):
    """Locate the axes box.

    Exported figure crops usually carry an outer border rectangle as well, so a
    detected pair sitting within `edge_frac` of the image boundary is discarded
    and the next pair inwards is used.
    """
    dark = gray < dark_thresh
    h, w = dark.shape
    row_runs = [longest_run(dark[y, :])[0] for y in range(h)]
    col_runs = [longest_run(dark[:, x])[0] for x in range(w)]
    rows = _line_candidates(row_runs, h, min_run_frac * w)
    cols = _line_candidates(col_runs, w, min_run_frac * h)
    if len(rows) < 2 or len(cols) < 2:
        raise RuntimeError("plot frame not found - try lowering --dark-thresh")

    # A panel crop made with a generous margin can still contain the *figure's* outer
    # border line, which spans the whole crop. The axes box is never that close to the
    # crop edge when a margin is present, so drop boundary-adjacent candidates first.
    rows = [r for r in rows if edge_frac * h <= r <= (1 - edge_frac) * h]
    cols = [c for c in cols if edge_frac * w <= c <= (1 - edge_frac) * w]
    if len(rows) < 2 or len(cols) < 2:
        raise RuntimeError("plot frame not found after dropping border-adjacent lines")

    def is_border_pair(first, last, size):
        return first < edge_frac * size and last > (1 - edge_frac) * size

    if len(rows) >= 4 and is_border_pair(rows[0], rows[-1], h):
        top, bottom = rows[1], rows[-2]
    else:
        top, bottom = rows[0], rows[-1]
    if len(cols) >= 4 and is_border_pair(cols[0], cols[-1], w):
        left, right = cols[1], cols[-2]
    else:
        left, right = cols[0], cols[-1]
    return left, top, right, bottom


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

    pts = []
    for x in range(left + 1, right):
        col = np.nonzero(mask[:, x])[0]
        if col.size == 0:
            continue
        pts.append((x, float(np.median(col))))
    return pts


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
