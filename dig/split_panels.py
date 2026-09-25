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
import extract_lines as el
import numpy as np

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _edge_cover(dark, box):
    """面板框"左边 + 下边"两条轴线的实画比例。

    轮廓法找出来的框里混着大量"曲线自己围出来的空洞"（斜线的 boundingRect 四条边
    基本没画东西）。真正的坐标框一定画了左轴和下轴，所以用这两条边来验真；
    实测 2026-01-0340 图 2(a) 的假框左边 0.99、下边 0.02，一眼就能分辨。
    """
    x, y, w, h = box
    if w < 4 or h < 4:
        return 0.0
    return float(min(dark[y:y + h, x].mean(), dark[y + h - 1, x:x + w].mean()))


def find_panel_boxes(gray, dark_thresh=120, min_frac=0.12, max_area_frac=0.75,
                     edge_thresh=160, edge_cover=0.5):
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    h, w = gray.shape
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edges = gray < edge_thresh
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_frac * w or bh < min_frac * h:
            continue
        if bw * bh > max_area_frac * w * h:
            continue
        if _edge_cover(edges, (x, y, bw, bh)) < edge_cover:
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


def model_plot_boxes(data, image_shape, expected=None):
    """Validate model-supplied normalised plot rectangles.

    The model only supplies semantic geometry. Pixel tracing still uses these boxes
    as numeric frames, so malformed, duplicated or strongly overlapping boxes are
    rejected rather than silently producing plausible-looking wrong coordinates.
    """
    h, w = image_shape[:2]
    rows = data.get("panels") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    if expected is not None and len(rows) != int(expected):
        return []
    out = []
    for item in rows:
        raw = item.get("plot_box") if isinstance(item, dict) else item
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return []
        try:
            vals = [float(v) for v in raw]
        except (TypeError, ValueError):
            return []
        if not all(np.isfinite(v) for v in vals):
            return []
        x0, y0, x1, y1 = vals
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            return []
        if x1 - x0 < 0.08 or y1 - y0 < 0.08:
            return []
        box = (int(round(x0 * w)), int(round(y0 * h)),
               int(round(x1 * w)), int(round(y1 * h)))
        box = (max(0, min(w - 2, box[0])), max(0, min(h - 2, box[1])),
               max(1, min(w - 1, box[2])), max(1, min(h - 1, box[3])))
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            return []
        out.append(box)

    # Independent panels may touch at their borders, but their interiors cannot
    # substantially overlap. A bad box here would mix two coordinate systems.
    xywh = [(a, b, c - a, d - b) for a, b, c, d in out]
    for i, a in enumerate(xywh):
        for b in xywh[i + 1:]:
            if _inter(a, b) > 0.2 * min(_area(a), _area(b)):
                return []
    return out


def _longest_bounds(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0, -1, -1
    best, start, prev = (1, int(idx[0]), int(idx[0])), int(idx[0]), int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        if prev - start + 1 > best[0]:
            best = (prev - start + 1, start, prev)
        start = prev = value
    if prev - start + 1 > best[0]:
        best = (prev - start + 1, start, prev)
    return best


def refine_plot_box(gray, box, edge_thresh=245, search_frac=0.08):
    """Snap an approximate model box to nearby long pixel axis/grid lines.

    This is deliberately conservative: each edge may move by only a small fraction
    of the hinted box. Missing/open axes simply retain the model coordinate.
    """
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    sx, sy = max(5, int(search_frac * bw)), max(5, int(search_frac * bh))
    wx0, wx1 = max(0, x0 - sx), min(w, x1 + sx + 1)
    wy0, wy1 = max(0, y0 - sy), min(h, y1 + sy + 1)

    def horizontal(target):
        best = None
        for yy in range(max(0, target - sy), min(h, target + sy + 1)):
            length, lo, hi = _longest_bounds(gray[yy, wx0:wx1] < edge_thresh)
            if length < 0.55 * bw:
                continue
            gx0, gx1 = wx0 + lo, wx0 + hi
            centre_penalty = abs((gx0 + gx1) / 2.0 - (x0 + x1) / 2.0)
            score = length - 2.0 * abs(yy - target) - centre_penalty
            if best is None or score > best[0]:
                best = (score, yy, gx0, gx1)
        return best

    def vertical(target):
        best = None
        for xx in range(max(0, target - sx), min(w, target + sx + 1)):
            length, lo, hi = _longest_bounds(gray[wy0:wy1, xx] < edge_thresh)
            if length < 0.55 * bh:
                continue
            gy0, gy1 = wy0 + lo, wy0 + hi
            centre_penalty = abs((gy0 + gy1) / 2.0 - (y0 + y1) / 2.0)
            score = length - 2.0 * abs(xx - target) - centre_penalty
            if best is None or score > best[0]:
                best = (score, xx, gy0, gy1)
        return best

    top, bottom = horizontal(y0), horizontal(y1)
    left, right = vertical(x0), vertical(x1)
    if top:
        y0 = top[1]
    if bottom:
        y1 = bottom[1]
    if left:
        x0 = left[1]
    if right:
        x1 = right[1]

    # Horizontal borders/grid lines often survive anti-aliasing better than vertical
    # axes. Their run endpoints are useful x-edge evidence when both agree.
    hlines = [v for v in (top, bottom) if v]
    if hlines:
        hx0 = int(round(np.median([v[2] for v in hlines])))
        hx1 = int(round(np.median([v[3] for v in hlines])))
        if abs(hx0 - box[0]) <= 1.5 * sx:
            x0 = hx0
        if abs(hx1 - box[2]) <= 1.5 * sx:
            x1 = hx1
    vlines = [v for v in (left, right) if v]
    if vlines:
        vy0 = int(round(np.median([v[2] for v in vlines])))
        vy1 = int(round(np.median([v[3] for v in vlines])))
        if abs(vy0 - box[1]) <= 1.5 * sy:
            y0 = vy0
        if abs(vy1 - box[3]) <= 1.5 * sy:
            y1 = vy1

    if x1 - x0 < 0.7 * bw or y1 - y0 < 0.7 * bh:
        return tuple(int(v) for v in box)
    return (int(x0), int(y0), int(x1), int(y1))


def _span_overlap(a, b):
    """两条线段在长度方向上的重叠比例（a/b 都是 (idx, start, end, length)）。"""
    inter = max(0, min(a[2], b[2]) - max(a[1], b[1]) + 1)
    return inter / float(max(1, min(a[2] - a[1] + 1, b[2] - b[1] + 1)))


def _group_lines(items, gap=3, min_overlap=0.5):
    """把相邻几行/几列的同一根线合成一根，取最长的那个。

    必须要求两段的跨度也重叠：图 2 的面板 (a) 左轴在 x=99、面板 (b) 左轴在
    x=100，只按"列号相邻"合并会把上面那根丢掉，整个面板消失。
    """
    if not items:
        return []
    items = sorted(items, key=lambda t: t[0])
    groups, cur = [], [items[0]]
    for it in items[1:]:
        if it[0] - cur[-1][0] <= gap and _span_overlap(cur[-1], it) >= min_overlap:
            cur.append(it)
        else:
            groups.append(max(cur, key=lambda t: t[3]))
            cur = [it]
    groups.append(max(cur, key=lambda t: t[3]))
    return groups


def _span_with_gaps(vec, thresh, max_gap=25, min_seg=30):
    """最长的一段"基本连续"的线：允许被曲线压断 max_gap 像素。

    坐标轴被数据曲线横穿时会断成几截（实测 2026-01-0340 图 2(a) 的左轴断成
    19-538 / 540-617 / 634-687 三段），只取最长的一截会严重低估轴线长度，
    于是 L 形配对失败、面板框整体偏小。这里把 ≤max_gap 的断口接起来。

    但是坐标轴旁边的刻度数字（"700"、"0"）离轴线只有几个像素，也会被接进来，
    让轴线凭空长出一截（面板框向下多 125px，标定全错）。所以只接长度 ≥min_seg
    的线段——文字笔画是十几像素，轴线断片是几十像素。
    """
    d = vec < thresh
    idx = np.nonzero(d)[0]
    if idx.size == 0:
        return 0, -1, -1
    segs, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        segs.append((start, prev))
        start = prev = i
    segs.append((start, prev))
    segs = [(s, e) for s, e in segs if e - s + 1 >= min_seg]
    if not segs:
        return 0, -1, -1
    best = (0, -1, -1)
    start, prev = segs[0][0], segs[0][1]
    for s, e in segs[1:]:
        if s - prev <= max_gap:
            prev = e
            continue
        if prev - start + 1 > best[0]:
            best = (prev - start + 1, start, prev)
        start, prev = s, e
    if prev - start + 1 > best[0]:
        best = (prev - start + 1, start, prev)
    return best


def find_spine_boxes(gray, dark_thresh=160, min_frac=0.3, corner_frac=0.06,
                     max_area_frac=0.75):
    """只有 L 形坐标轴的面板框（很多论文图只画左轴 + 下轴）。

    `find_panel_boxes` 找的是"四条边都画了"的闭合矩形。可大量期刊插图只画左轴和
    下轴，闭合轮廓根本不存在，于是轮廓法会退回"曲线自己围出来的空洞"——实测
    2026-01-0340 图 2(b) 因此只抠到右上角一小块，整幅子图的数据全丢。这里改用
    长横线 + 长竖线，取"拐角对得上"的一对围出面板框。
    """
    h, w = gray.shape
    hlines, vlines = [], []
    for y in range(h):
        ln, x0, _x1 = _span_with_gaps(gray[y, :], dark_thresh)
        if ln >= min_frac * w:
            hlines.append((y, x0, _x1, ln))
    for x in range(w):
        ln, y0, _y1 = _span_with_gaps(gray[:, x], dark_thresh)
        if ln >= min_frac * h:
            vlines.append((x, y0, _y1, ln))
    hlines = _group_lines(hlines)
    vlines = _group_lines(vlines)
    tol = max(8, int(corner_frac * min(h, w)))
    boxes = []
    for (y, hx0, hx1, _ln) in hlines:
        for (x, vy0, vy1, _ln2) in vlines:
            if abs(x - hx0) <= tol and abs(y - vy1) <= tol:
                # 左下角 L：竖轴 + 下轴
                box = (min(x, hx0), min(vy0, y), max(hx1, x), max(vy1, y))
            elif abs(x - hx1) <= tol and abs(y - vy0) <= tol:
                # 右上角 L：坐标轴画在右侧/顶部
                box = (min(hx0, x), min(y, vy0), max(x, hx1), max(vy1, y))
            else:
                continue
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 0.2 * w or bh < 0.2 * h:
                continue
            # 整幅图的边框也是一横一竖，它不是面板（跟 find_panel_boxes 同一条界线）
            if bw * bh > max_area_frac * w * h:
                continue
            boxes.append((box[0], box[1], bw, bh))
    unique = []
    for b in sorted(boxes, key=lambda t: -t[2] * t[3]):
        if any(abs(b[0] - u[0]) < 12 and abs(b[1] - u[1]) < 12
               and abs(b[2] - u[2]) < 12 and abs(b[3] - u[3]) < 12 for u in unique):
            continue
        unique.append(b)
    return unique


def _area(box):
    return box[2] * box[3]


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1 = min(a[0] + a[2], b[0] + b[2])
    y1 = min(a[1] + a[3], b[1] + b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def merge_panel_boxes(contour_boxes, spine_boxes):
    """轮廓框为主，L 形轴框补"只画了左轴+下轴"的面板。

    规则（都是踩过坑之后定的）：
      * L 形框已经落在某个轮廓框里 → 不要（同一个面板，轮廓框更准）；
      * L 形框正好罩住**一个**轮廓框、且大得多 → 那个轮廓框是"曲线围出的空洞"，
        换成 L 形框（2026-01-0340 图 2(b) 就是这么救回来的）；
      * L 形框罩住**多个**轮廓框 → 那是"整幅图/半幅图的外边框"，一横一竖的长线
        在多面板图里到处都是，不能当面板（实测会把 2023-01-1635 的 2x3 图并成
        一个大框）；
      * 没罩住任何轮廓框 → 补成一个新面板（轮廓法对 L 形轴的图完全找不到框）。
    """
    keep = list(contour_boxes)
    for s in spine_boxes:
        if any(_inter(s, c) >= 0.8 * _area(s) for c in keep):
            continue
        inside = [c for c in keep if _inter(s, c) >= 0.8 * _area(c)
                  and _area(s) >= 1.5 * _area(c)]
        if len(inside) > 1:
            continue
        # 新补的框不能和已有面板大面积重叠——一横一竖两条长线在多面板图里到处都是，
        # 不加这条会把相邻面板并成一个大框（实测 2014 那批 2x2 图）。
        if not inside and any(_inter(s, c) > 0.3 * _area(s) for c in keep):
            continue
        for c in inside:
            keep.remove(c)
        keep.append(s)
    return keep


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
