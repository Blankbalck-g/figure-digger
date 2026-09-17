"""Split a multi-panel figure (e.g. 2x3 sub-plots) into individual panel images.

Finds the plot frames by contour analysis, drops the outer figure border and nested
contours, orders the panels row-major and crops each one (plus a small margin for the
tick labels).

Usage:
  python split_panels.py figure.png --outdir out/panels [--prefix fig6] [--verify]
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


def find_panel_boxes(gray, dark_thresh=120, min_frac=0.12, max_area_frac=0.75):
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    h, w = gray.shape
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_frac * w or bh < min_frac * h:
            continue
        if bw * bh > max_area_frac * w * h:
            continue
        boxes.append((x, y, bw, bh))

    # keep the largest box of each nested group (the outermost frame)
    boxes.sort(key=lambda b: -b[2] * b[3])
    kept = []
    for b in boxes:
        if any(b[0] >= k[0] - 3 and b[1] >= k[1] - 3
               and b[0] + b[2] <= k[0] + k[2] + 3 and b[1] + b[3] <= k[1] + k[3] + 3
               for k in kept):
            continue
        kept.append(b)

    # drop boxes that are nearly identical (same frame detected twice)
    unique = []
    for b in kept:
        if any(abs(b[0] - u[0]) < 12 and abs(b[1] - u[1]) < 12
               and abs(b[2] - u[2]) < 12 and abs(b[3] - u[3]) < 12 for u in unique):
            continue
        unique.append(b)
    return unique


def order_row_major(boxes, row_tol_frac=0.35):
    if not boxes:
        return []
    med_h = float(np.median([b[3] for b in boxes]))
    boxes = sorted(boxes, key=lambda b: b[1])
    rows, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - cur[0][1]) <= row_tol_frac * med_h:
            cur.append(b)
        else:
            rows.append(sorted(cur, key=lambda t: t[0]))
            cur = [b]
    rows.append(sorted(cur, key=lambda t: t[0]))
    ordered = []
    for r in rows:
        ordered.extend(r)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--margin", type=int, default=14, help="extra pixels around each panel")
    ap.add_argument("--verify", action="store_true", help="write an image with the detected boxes")
    args = ap.parse_args()

    src = Path(args.image).resolve()
    img = iio.imread(src)
    if img is None:
        raise SystemExit(f"cannot read {src}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    boxes = order_row_major(find_panel_boxes(gray))
    print(f"{src.name}: {w}x{h}, 检测到 {len(boxes)} 个面板")
    if not boxes:
        raise SystemExit("没有检测到面板框")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or src.stem

    for i, (x, y, bw, bh) in enumerate(boxes):
        x0 = max(0, x - args.margin)
        y0 = max(0, y - args.margin)
        x1 = min(w, x + bw + args.margin)
        y1 = min(h, y + bh + args.margin)
        crop = img[y0:y1, x0:x1]
        name = f"{prefix}_panel{i + 1:02d}.png"
        iio.imwrite(outdir / name, crop)
        print(f"  {name}: frame=({x},{y},{bw}x{bh}) crop=({x0},{y0})-({x1},{y1})")

    if args.verify:
        vis = img.copy()
        for i, (x, y, bw, bh) in enumerate(boxes):
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 180, 0), 2)
            cv2.putText(vis, f"P{i + 1}", (x + 6, y + 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)
        vp = outdir / f"{prefix}_panels_detected.png"
        iio.imwrite(vp, vis)
        print(f"verification image -> {vp}")

    # emit the frame list so downstream steps can reuse the exact geometry
    frame_txt = outdir / f"{prefix}_frames.txt"
    with frame_txt.open("w", encoding="utf-8") as fh:
        for i, (x, y, bw, bh) in enumerate(boxes):
            fh.write(f"{i + 1}\t{x}\t{y}\t{x + bw}\t{y + bh}\n")
    print(f"frame list -> {frame_txt}")


if __name__ == "__main__":
    main()
