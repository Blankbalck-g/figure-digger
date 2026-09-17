"""Read axis tick labels with RapidOCR and derive the axis calibration.

Instead of assuming the data spans the plot frame, every tick label is OCR'd and a
least-squares line is fitted from label position -> label value. The fitted line then
gives the exact value at the frame edges, which is what the extractor needs.

Usage:
  python ocr_ticks.py panel.png --frame 14,14,516,294
  python ocr_ticks.py panel.png --frame 14,14,516,294 --debug-crops out/dbg
"""

import argparse
import itertools
import re
import sys

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

NUMBER_RE = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")


def parse_number(text):
    t = text.strip().replace(" ", "").replace("−", "-").replace("–", "-")
    t = t.replace(",", "") if t.count(",") and "." in t else t.replace(",", ".")
    if NUMBER_RE.match(t):
        try:
            return float(t)
        except ValueError:
            return None
    return None


def run_ocr(engine, img):
    result, _ = engine(img)
    out = []
    for box, text, score in (result or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({
            "text": text,
            "score": float(score),
            "cx": float(sum(xs) / 4.0),
            "cy": float(sum(ys) / 4.0),
            "box": box,
        })
    return out


def nice_step(step):
    """True if step looks like a human axis step (1, 2, 2.5, 5, 10 x 10^k)."""
    if not step:
        return False
    s = abs(step)
    while s < 1:
        s *= 10
    while s >= 10:
        s /= 10
    return any(abs(s - t) <= 0.02 * t for t in (1.0, 2.0, 2.5, 5.0))


def snap_to_step(value, step, frac=0.3):
    """Pull an axis endpoint onto the tick grid when it is already close to it."""
    if not step or step == 0:
        return value, False
    cand = round(value / step) * step
    if abs(cand - value) <= frac * abs(step):
        return cand, abs(cand - value) > 1e-9
    return value, False


def linfit(xs, ys):
    """Closed-form least-squares line, computed with plain arithmetic.

    np.polyfit / np.linalg.* go through LAPACK, which crashes hard (0xC06D007F) in some
    conda environments. A straight-line fit needs no LAPACK.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None, None, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def fit_axis(items, coord_getter, label):
    """items: [(coord, value, text, score)] -> dict with fit and edge values."""
    pts = sorted(items, key=lambda t: t[0])
    if len(pts) < 2:
        return None
    coords = [float(p[0]) for p in pts]
    raw_values = [float(p[1]) for p in pts]
    if max(coords) - min(coords) == 0:
        return None

    # --- global search over per-label scale candidates -------------------------
    # OCR commonly drops the decimal point ("0.5" -> "05"), which still yields a
    # perfectly linear fit, so R^2 cannot detect it. Trying v, v/10, v/100 for every
    # label and scoring the whole assignment catches it.
    candidates = [[v] + ([v / 10.0, v / 100.0] if v != 0 else []) for v in raw_values]
    best = None
    for combo in itertools.product(*candidates):
        slope, intercept, r2 = linfit(coords, list(combo))
        if slope is None:
            continue
        span = (max(combo) - min(combo)) or 1.0
        res = [abs(v - (slope * c + intercept)) for c, v in zip(coords, combo)]
        changes = sum(1 for a, b in zip(combo, raw_values) if abs(a - b) > 1e-12)
        step_guess = (max(combo) - min(combo)) / max(1, len(combo) - 1)
        score = (round(max(res) / span, 6), changes, 0 if nice_step(step_guess) else 1)
        if best is None or score < best[0]:
            best = (score, list(combo), slope, intercept, r2)
    if best is None:
        return None
    _, values, slope, intercept, r2 = best
    repaired = [(pts[k][2], raw_values[k], values[k])
                for k in range(len(pts)) if abs(values[k] - raw_values[k]) > 1e-12]

    # --- drop residual outliers with a robust pairwise fit ---
    span = (max(values) - min(values)) or 1.0
    tol = 0.05 * span
    robust = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            s, b, _ = linfit([coords[i], coords[j]], [values[i], values[j]])
            if s is None:
                continue
            res = [abs(v - (s * c + b)) for c, v in zip(coords, values)]
            inl = sum(1 for r in res if r <= tol)
            if robust is None or (inl, -sum(res)) > (robust[0], -robust[1]):
                robust = (inl, sum(res), s, b)
    if robust:
        slope, intercept = robust[2], robust[3]
        _, _, r2 = linfit(coords, values)
    res = [abs(v - (slope * c + intercept)) for c, v in zip(coords, values)]
    tol2 = max(tol, 0.02 * span)
    outliers = [pts[k][2] for k, r in enumerate(res) if r > tol2]

    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    is_log = False
    if len(values) >= 3 and all(v > 0 for v in values):
        ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
        is_log = all(abs(r - ratios[0]) <= 0.02 * abs(ratios[0]) for r in ratios)
    return {
        "axis": label,
        "labels": [{"text": t[2], "value": t[1], "pos": round(t[0], 1), "score": round(t[3], 2)}
                   for t in pts],
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "inliers": sum(1 for r in res if r <= tol2),
        "outliers": outliers,
        "repaired": repaired,
        "step": float(sorted(diffs)[len(diffs) // 2]) if diffs else None,
        "looks_log": is_log,
        "n_labels": len(pts),
    }


def main():
    """CLI only. The reading logic lives in axis_ranges.py (single implementation)."""
    import axis_ranges as ar

    ap = argparse.ArgumentParser(description="读取单张图的坐标轴刻度范围")
    ap.add_argument("image")
    ap.add_argument("--frame", required=True, help="left,top,right,bottom of the plot box")
    ap.add_argument("--x-band", type=int, default=0, help="x 标签带高度（0=自动）")
    ap.add_argument("--y-band", type=int, default=0, help="y 标签带宽度（0=自动）")
    ap.add_argument("--debug-crops", default=None)
    args = ap.parse_args()
    frame = tuple(int(float(v)) for v in args.frame.split(","))

    res = ar.read_axis_ranges(args.image, frame, x_band=args.x_band or None,
                              y_band=args.y_band or None, log=print,
                              debug_crops=args.debug_crops)
    print()
    for axis in ("x", "y"):
        if axis not in res:
            print(f"  {axis} 轴: 未读到足够的数字标签")
            continue
        d = res[axis]
        warn = ""
        if d["outliers"]:
            warn += f"  ⚠ 离群标签 {d['outliers']}"
        if d["repaired"]:
            warn += "  已修复 " + ",".join(d["repaired"])
        if d["snapped"]:
            warn += "  端点已吸附到刻度网格"
        print(f"  {axis} 轴: {d['n_labels']} 个刻度 {d['labels']}  "
              f"R²={d['r2']:.5f}  步长≈{d['step']}  范围={d['range']}{warn}")
    if "x" in res and "y" in res:
        xl, xh = res["x"]["range"]
        yl, yh = res["y"]["range"]
        print(f"\n--range 参数: {xl:.6g},{xh:.6g},{yl:.6g},{yh:.6g}")


if __name__ == "__main__":
    main()
