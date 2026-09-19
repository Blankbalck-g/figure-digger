"""Turn the model's coarse "where the curves are" into precise numbers.

分工（这一版把"锚点优先"改成"颜色优先、锚点消歧"）：

  * **模型负责"懂"**：这张图里有哪几条数据曲线、对应图例里哪个条目、什么颜色、
    什么线型、要不要出标记点，以及作者画的辅助线/示意图要忽略；
  * **代码负责"准"**：颜色由"图例色块"和"锚点处的真实像素"确定，再从该颜色的
    连通实例里挑出这条曲线、逐列追踪；锚点只在**同色多实例**时用来消歧。

为什么不再拿锚点当主键：实测模型给的归一化锚点经常偏 40–300 px（提示词里要求
±3%）。旧实现把锚点同时当作"种子"和"取色点"，锚点落到白底时 sample_color 返回
(254,254,254)，bgr_to_hsv 得到 hue=0 / sat=0 的"万能掩膜"，结果是**同一张图里
几条不同颜色的曲线最后追出了完全相同的一条轨迹**（实测 5 个 CSV 一模一样、
质检图上是同一条线画了五遍）。所以现在的三条硬规则：

  1. 取色绝不返回近白/近灰——找不到饱和像素就交还给图例色 / 模型给的十六进制；
  2. 颜色以图例色块为准（它是从这张图上量出来的），锚点实测色只在与图例同色系时采信；
  3. 同色多实例按"离锚点多近"排序，而不是"谁能追得更长"（最长的那条往往是别的线）。

颜色也说了不算的情况只有一种：同一张图里同一种颜色画了多条曲线（同色虚线、
放大子图里的复制品）。那才轮到锚点投票，并且会把"锚点与曲线对不上"如实写进报告。
"""

import csv
import re
from pathlib import Path

import cv2
import numpy as np

import extract_lines as el

# 追踪参数：横向最大跳变 / 允许的连续空列数（虚线断口靠它跨过去）
MAX_JUMP = 28.0
MAX_GAP = 90

# 颜色判定的两道阈值，都比掩膜的 hue_tol(14) 紧：
#   * 模型的十六进制与图例色块相差 ≤PALETTE_SNAP 才算同一个颜色，这时用图例色
#     （它是在本图上量出来的，比模型猜的十六进制准）；
#   * 锚点处实测到的颜色与"模型说它是这个颜色"相差 ≤ANCHOR_AGREE 才采信锚点。
# 实测教训：2026-01-0340 图 2 的 E30 是金色（hue 21），图例里只有橙（hue 9）
# 和绿（hue 30），阈值放宽到 25 时它被判成橙色，于是 E00 和 E30 追上同一条线。
PALETTE_SNAP = 7
ANCHOR_AGREE = 10


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


# --------------------------------------------------------------------- 颜色

def _hsv_of(bgr):
    px = np.uint8([[[int(v) for v in bgr]]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return int(h), int(s), int(v)


def _hue_dist(a, b):
    d = abs(int(a) - int(b)) % 180
    return min(d, 180 - d)


def _median_bgr(colors):
    arr = np.array(colors, dtype=float)
    return tuple(int(v) for v in np.median(arr, axis=0))


def _hex_to_bgr(text):
    try:
        return tuple(int(v) for v in el.hex_to_bgr(str(text)))
    except Exception:  # noqa: BLE001
        return None


def anchor_colors(img, points, radius=9, min_sat=55, min_px=4):
    """每个锚点窗口里"真正的笔画色"（取最饱和的那一小撮像素的中位数）。

    窗口里一个饱和像素都没有（锚点落在白底/网格上）时，这个锚点直接跳过——
    旧实现会退回"整窗像素的中位数"，也就是白色，然后变成万能掩膜。这是本模块
    最要命的那个 bug，所以这里宁可返回 None 也不返回灰白色。
    """
    h, w = img.shape[:2]
    out = []
    for x, y in points or []:
        x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch) < 12:
            continue
        hsv = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        sat = hsv[:, 1].astype(int)
        val = hsv[:, 2].astype(int)
        keep = (sat >= max(min_sat, int(np.percentile(sat, 90)))) & (val >= 40)
        if int(keep.sum()) < min_px:
            continue
        out.append(_median_bgr(patch[keep]))
    return out


def resolve_color(img, anchors, hint_hex=None, palette=()):
    """这条线到底什么颜色 -> (bgr, 来源)。

    来源取值：anchor（锚点处的真实像素）/ palette（图例色块，从本图量出）/
    hint（模型给的十六进制）/ none。绝不返回近白近灰。
    """
    hint = _hex_to_bgr(hint_hex) if hint_hex else None
    if hint is not None and _hsv_of(hint)[1] < 40:
        hint = None                       # 模型给了个灰白色，等于没给
    pal = [c for c in (_hex_to_bgr(p) for p in (palette or [])) if c is not None]
    pal = [c for c in pal if _hsv_of(c)[1] >= 40]

    reference = hint if hint is not None else (pal[0] if pal else None)
    ref_hue = _hsv_of(reference)[0] if reference is not None else None

    # 图例色块优先：它是从这张图上量出来的，比模型报的十六进制更准
    if pal and ref_hue is not None:
        best = min(pal, key=lambda c: _hue_dist(_hsv_of(c)[0], ref_hue))
        if _hue_dist(_hsv_of(best)[0], ref_hue) <= PALETTE_SNAP:
            return best, "palette"

    measured = anchor_colors(img, anchors)
    if measured:
        if ref_hue is None:
            return _median_bgr(measured), "anchor"
        close = [c for c in measured
                 if _hue_dist(_hsv_of(c)[0], ref_hue) <= ANCHOR_AGREE]
        if close:
            return _median_bgr(close), "anchor"
        # 锚点上量到的颜色和"模型说它是这个颜色"对不上 → 锚点大概落在别的线上，
        # 颜色听模型的，位置后面再靠实例消歧。
    if hint is not None:
        return hint, "hint"
    if pal:
        return pal[0], "palette"
    return None, "none"


def hue_tolerances(colors, base=14, floor=4):
    """每条曲线自己的色相容差：颜色相近时收紧，免得两条线互相串色。

    实测 2026-01-0340 图 2(a)：E00 是橙（hue 9）、E30 是金（hue 21），只差 12。
    用统一的 hue_tol=14 时两条线互为"同色"，掩膜里混着对方的像素，金线被当成
    橙线的一部分，两条曲线最后追成同一条。这里按"离最近的其他颜色有多远"取
    0.45 倍，落在 4~14 之间。
    """
    out = []
    for i, c in enumerate(colors):
        if c is None:
            out.append(base)
            continue
        hi = _hsv_of(c)[0]
        dists = [_hue_dist(hi, _hsv_of(o)[0])
                 for j, o in enumerate(colors) if o is not None and j != i]
        out.append(int(max(floor, min(base, 0.45 * min(dists)))) if dists else base)
    return out


# --------------------------------------------------------------------- 实例

def _runs_by_column(mask, frame):
    left, top, right, bottom = frame
    y0, y1 = top + 2, bottom - 1
    out = {}
    for x in range(left + 1, right):
        rs = el._column_runs(mask, x, y0, y1)
        if rs:
            out[x] = rs
    return out


def color_instances(hsv, frame, color, exclude_boxes=(), hue_tol=14, sat_floor=None,
                    bridge=31, min_px=25, min_span=0.005):
    """该颜色的连通实例。

    先把掩膜横向膨胀一点再连通域标记，这样虚线/点画线的断口不会把一条线拆成
    十几段（追踪器自己也会跨断口，但"实例"要能表达整条线才好排序）。
    膨胀只用于标记，每个实例里保留的仍是真实像素。

    阈值放得比较松（25 像素、2% 帧宽）：陡峭虚线在图上是一串彼此错开的小段
    （实测 2026-01-0340 图 10 的红色虚线初始段被拆成 10 段，每段只有几十像素），
    门槛稍微高一点它们就整段消失，曲线的前 20% 直接丢失。它们随后会被
    `_merge_groups` 按"端点相邻"接回主线，孤立的碎片则在排序里垫底。
    """
    kwargs = {} if sat_floor is None else {"min_sat": sat_floor}
    mask = el.color_mask(hsv, frame, color, hue_tol, exclude_boxes=exclude_boxes,
                         **kwargs)
    return mask_instances(mask, frame, bridge=bridge, min_px=min_px, min_span=min_span,
                          key_prefix=(round(_hsv_of(color)[0]),))


def mask_instances(mask, frame, bridge=31, min_px=25, min_span=0.005, key_prefix=()):
    """任意二值掩膜的连通实例（颜色掩膜、深色线掩膜都用它）。

    黑线路线（模型说 dark=true 时）复用同一套"实例 -> 候选 -> 追踪"逻辑：颜色是
    一个掩膜，深色笔画也是一个掩膜，后面的排序（锚点命中）完全一样。
    """
    if int((mask > 0).sum()) == 0:
        return []
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(bridge)), 3))
    joined = cv2.dilate(mask, ker)
    n, labels = cv2.connectedComponents(joined)
    frame_w = max(1, frame[2] - frame[0])
    out = []
    for i in range(1, n):
        comp = np.where(labels == i, mask, 0).astype(np.uint8)
        npx = int((comp > 0).sum())
        if npx < min_px:
            continue
        ys, xs = np.nonzero(comp)
        span = float(xs.max() - xs.min()) / frame_w
        if span < min_span:
            continue
        out.append({"mask": comp, "key": tuple(key_prefix) + (i,), "npix": npx,
                    "span": round(span, 3), "x0": int(xs.min()), "x1": int(xs.max()),
                    "y0": int(ys.min()), "y1": int(ys.max())})
    out.sort(key=lambda d: -d["npix"])
    return out


def _merge_groups(insts, gap_x=70, gap_y=55):
    """把"同一根线被超长断口拆开"的实例并成一组（长虚线、被遮挡的线段）。

    只在水平方向相邻、纵向范围也叠得上时才合并——两条颜色相同的不同曲线通常
    纵向差得远，不会被误并。
    """
    parent = list(range(len(insts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(insts)):
        for j in range(i + 1, len(insts)):
            a, b = insts[i], insts[j]
            dx = max(a["x0"] - b["x1"], b["x0"] - a["x1"])
            dy = max(a["y0"] - b["y1"], b["y0"] - a["y1"])
            if dx <= gap_x and dy <= gap_y:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    groups = {}
    for i, inst in enumerate(insts):
        groups.setdefault(find(i), []).append(inst)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        merged = members[0]["mask"].copy()
        for m in members[1:]:
            merged = cv2.bitwise_or(merged, m["mask"])
        ys, xs = np.nonzero(merged)
        frame_w = max(1, int(xs.max()) - int(xs.min()))
        out.append({"mask": merged, "key": tuple(m["key"] for m in members),
                    "npix": int((merged > 0).sum()),
                    "span": round((int(xs.max()) - int(xs.min())) / frame_w, 3),
                    "x0": int(xs.min()), "x1": int(xs.max()),
                    "y0": int(ys.min()), "y1": int(ys.max())})
    out.sort(key=lambda d: -d["npix"])
    return out


# --------------------------------------------------------------------- 追踪

def _nearest_run(runs, seed):
    sx, sy = float(seed[0]), float(seed[1])
    best = None
    for x, rs in runs.items():
        for cy, _ln in rs:
            score = abs(cy - sy) + 0.15 * abs(x - sx)
            if best is None or score < best[0]:
                best = (score, x, cy)
    return (best[1], best[2]) if best else None


def _tallest_run(runs):
    x = max(runs, key=lambda k: max(ln for _cy, ln in runs[k]))
    cy = max(runs[x], key=lambda r: r[1])[0]
    return x, cy


def _trace_in_mask(mask, frame, seed=None, max_jump=MAX_JUMP, max_gap=MAX_GAP):
    runs = _runs_by_column(mask, frame)
    if not runs:
        return []
    start = _nearest_run(runs, seed) if seed is not None else None
    if start is None:
        start = _tallest_run(runs)
    x0, y0 = start
    xs = sorted(runs)
    right = el._walk(runs, [x for x in xs if x > x0], y0, max_jump, max_gap)
    left = el._walk(runs, [x for x in xs if x < x0][::-1], y0, max_jump, max_gap)
    pts = sorted(left + [(x0, y0)] + right)
    return el._smooth_despike(el._despeckle(pts))


def _looks_like_loop(mask, frame, min_ratio=0.45, min_span=0.3):
    """闭曲线（P-V 图那种一圈的）在逐列追踪里只走得到半边。

    判据：多数列里有两条以上的竖切（说明曲线在同一列上去了又回来），
    且横向跨度够大。
    """
    runs = _runs_by_column(mask, frame)
    if len(runs) < 20:
        return False
    multi = sum(1 for rs in runs.values() if len(rs) >= 2)
    span = (max(runs) - min(runs)) / max(1.0, frame[2] - frame[0])
    return multi / len(runs) >= min_ratio and span >= min_span


def _single_blob(mask, min_share=0.9):
    """这个掩膜是不是连成整块。碎片化的回线（虚线、被遮挡、压着别的线）不能用轮廓法：
    最大那块只是一个弧段，拿它的轮廓当整圈会得到一条来回折返的假轨迹。"""
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if n <= 2:
        return True
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    return areas[0] >= min_share * max(1, sum(areas))


def _encloses_area(mask, contour, min_ratio=2.0):
    """这一圈是不是真的**围出了一块面积**（而不是一条线出去又回来）。

    实测教训：散点+折线的曲线，标记符号让很多列出现两条以上竖切，`_looks_like_loop`
    会误判成闭合回线，于是 CSV 沿轮廓走一圈——画出来就是一个"回路"。
    真回线：轮廓面积 ≫ 笔画像素数（实测细笔画围的圈在 10 左右）；出去又回来的带子、
    几条线挤成的胖块都只有 1 上下。
    """
    n = int((mask > 0).sum())
    if n <= 0 or len(contour) < 20:
        return False
    area = abs(cv2.contourArea(np.array(contour, dtype=np.float32).reshape(-1, 1, 2)))
    return area >= min_ratio * n


def _contour_points(mask, frame, max_points=900):
    """沿闭合轮廓取一圈点（顺序沿曲线走，不是按 x 排序）。"""
    left, top, right, bottom = frame
    m = mask.copy()
    m[:top + 2, :] = 0
    m[bottom:, :] = 0
    m[:, :left + 2] = 0
    m[:, right:] = 0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if len(c) >= 20]
    if not cnts:
        return []
    c = max(cnts, key=lambda cc: cv2.arcLength(cc, True))
    pts = [(float(p[0][0]), float(p[0][1])) for p in c]
    pts.append(pts[0])
    if len(pts) > max_points:
        step = len(pts) / float(max_points)
        idx = sorted({min(len(pts) - 1, int(i * step)) for i in range(max_points)})
        pts = [pts[i] for i in idx]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
    return pts


def _anchor_stats(trace, anchors):
    """(命中锚点数, 中位距离) —— 锚点只是粗位置，这里只用来排序。"""
    if not anchors or not trace:
        return 0, None
    arr = np.array(trace, dtype=float)
    hits, dists = 0, []
    for a in anchors:
        d = float(np.min(np.hypot(arr[:, 0] - a[0], arr[:, 1] - a[1])))
        dists.append(d)
        if d <= 45.0:
            hits += 1
    return hits, float(np.median(dists))


def _dedupe(traces, min_overlap=0.8, tol=3.0):
    """丢掉"同一条轨迹"的重复候选（不同种子起出来的同一条线）。"""
    kept = []
    for t in sorted(traces, key=len, reverse=True):
        dup = False
        for k in kept:
            common = set(x for x, _ in k) & set(x for x, _ in t)
            if len(common) < max(10, 0.5 * min(len(k), len(t))):
                continue
            kd, td = dict(k), dict(t)
            same = sum(1 for x in common if abs(kd[x] - td[x]) <= tol)
            if same >= min_overlap * len(common):
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept


def _dominant_hues(hsv, frame, sat_min=70, val_min=40, min_share=0.05):
    """图上真实存在的色相簇（按像素数从多到少），用于"图例色≠线色"的兜底。"""
    left, top, right, bottom = frame
    sub = hsv[top + 2:bottom - 1, left + 2:right - 1]
    if sub.size == 0:
        return []
    hue = sub[:, :, 0].astype(int)
    sat = sub[:, :, 1].astype(int)
    val = sub[:, :, 2].astype(int)
    sel = (sat >= sat_min) & (val >= val_min)
    n = int(sel.sum())
    if n < 200:
        return []
    hist = np.bincount(hue[sel], minlength=180).astype(float)
    ker = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    sm = np.convolve(np.r_[hist[-2:], hist, hist[:2]], ker / ker.sum(),
                     mode="same")[2:-2]
    peaks = [(float(sm[h]), h) for h in range(180)
             if sm[h] >= min_share * n
             and sm[h] >= sm[h - 1] and sm[h] >= sm[(h + 1) % 180]]
    peaks.sort(reverse=True)
    return [h for _v, h in peaks]


def _hue_color(hsv, frame, hue, tol=8, sat_min=60, val_min=40):
    """某个色相簇的代表色（中位 HSV -> BGR）。"""
    left, top, right, bottom = frame
    sub = hsv[top + 2:bottom - 1, left + 2:right - 1]
    if sub.size == 0:
        return None
    h = sub[:, :, 0].astype(int)
    s = sub[:, :, 1].astype(int)
    v = sub[:, :, 2].astype(int)
    dh = np.abs(h - hue)
    dh = np.minimum(dh, 180 - dh)
    sel = (dh <= tol) & (s >= sat_min) & (v >= val_min)
    if int(sel.sum()) < 50:
        return None
    hh = int(np.median(h[sel]))
    ss_ = int(np.median(s[sel]))
    vv = int(np.median(v[sel]))
    bgr = cv2.cvtColor(np.uint8([[[hh, ss_, vv]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(t) for t in bgr)


def is_rule(trace, frame, flat_px=4.0, span_frac=0.6):
    """这条轨迹是不是"一条几乎水平/垂直、横跨大半张图的直线"——网格线或坐标框。

    实测：黑线路线（模型说 dark=true）里，背景的**点状网格线**是深灰的、又横跨整幅，
    很容易被当成数据曲线追出来（用户看到的"把背景网格都识别上了"）。网格线/坐标框
    的形态特征就是"几乎没有起伏 + 跨度极大"，真正的数据曲线极少同时满足这两条。
    """
    if not trace or len(trace) < 20:
        return False
    xs = [p[0] for p in trace]
    ys = [p[1] for p in trace]
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    fw = max(1.0, frame[2] - frame[0])
    fh = max(1.0, frame[3] - frame[1])
    return ((dy <= flat_px and dx >= span_frac * fw)
            or (dx <= flat_px and dy >= span_frac * fh))


def is_rule_line(mask, trace, frame, cover_need=0.4, band=3, beyond=2.5):
    """这条轨迹所在的高度上，整幅图是不是都有同一条线（网格线/坐标框的碎片）。

    扫描版老图里网格线是断的，追踪器只抓到其中一小段（实测 960773 图 8 里只有
    13% 宽），光看轨迹自己"平不平"不够。判据：它所在的那条水平线上，掩膜横跨整幅
    的比例远大于它自己占的宽度——真实数据线不会在别处也有同样的线。
    """
    if not trace or len(trace) < 12:
        return False
    xs = [p[0] for p in trace]
    ys = [p[1] for p in trace]
    fw = max(1.0, frame[2] - frame[0])
    fh = max(1.0, frame[3] - frame[1])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if y1 - y0 <= band:
        y = int(round((y0 + y1) / 2))
        strip = mask[max(0, y - band):y + band + 1, frame[0] + 2:frame[2] - 1]
        if strip.size == 0:
            return False
        cover = float((strip > 0).any(axis=0).mean())
        return cover >= cover_need and cover >= beyond * (x1 - x0) / fw
    if x1 - x0 <= band:
        x = int(round((x0 + x1) / 2))
        strip = mask[frame[1] + 2:frame[3] - 1, max(0, x - band):x + band + 1]
        if strip.size == 0:
            return False
        cover = float((strip > 0).any(axis=1).mean())
        return cover >= cover_need and cover >= beyond * (y1 - y0) / fh
    return False


def flat_run_frac(trace, tol=1.0):
    """最长"几乎完全水平"的连续段占整条轨迹的比例。

    实测：黑线路线有时会顺着一条点状网格线走很远——整条轨迹 86% 的点都在同一行
    （真实曲线即使有平台段，这个比例也只有 3~8%，因为还有标记符号造成的微小起伏）。
    """
    pts = sorted(trace, key=lambda q: q[0])
    if len(pts) < 12:
        return 1.0
    best = cur = 1
    for i in range(1, len(pts)):
        if abs(pts[i][1] - pts[i - 1][1]) <= tol:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best / float(len(pts))


def candidates_in_mask(mask, frame, anchors, allow_loop=False, reject_rules=False):
    """掩膜里所有候选轨迹，按"锚点命中数 -> 锚点中位距离 -> 覆盖长度"排好序。

    `allow_loop` 由**模型**说了算（FIND_SERIES_PROMPT 里的 closed 字段）：闭合回线
    要走轮廓法，普通曲线绝不能走——否则会得到一条来回折返的假轨迹（实测 2024-01-3408
    的三条温度曲线被判成回线，画出来就是一个圈）。
    """
    insts = mask_instances(mask, frame)
    groups = _merge_groups(insts)
    cands = []
    for g in groups:
        span_px = g["x1"] - g["x0"]
        # 图例里的一小截色条：又短又薄的水平/竖直小条。真正的数据曲线很少这么短，
        # 而它一旦被当成曲线就会在图上凭空多出一条"数据"。
        if (span_px < 0.25 * max(1, frame[2] - frame[0])
                and min(g["y1"] - g["y0"], 99) <= 8):
            continue
        traces = []
        for a in (anchors or []):
            t = _trace_in_mask(g["mask"], frame, a)
            if t:
                traces.append(t)
        traces.append(_trace_in_mask(g["mask"], frame, None))
        traces = [t for t in _dedupe(traces) if t]
        if not traces:
            continue
        trace = max(traces, key=len)
        if reject_rules and (is_rule(trace, frame)
                             or is_rule_line(mask, trace, frame)
                             or flat_run_frac(trace) >= 0.4):
            continue                       # 网格线/坐标框：不是数据
        kind = "line"
        if allow_loop and _looks_like_loop(g["mask"], frame) and _single_blob(g["mask"]):
            loop = _contour_points(g["mask"], frame)
            if len(loop) >= 20 and _encloses_area(g["mask"], loop):
                trace, kind = loop, "loop"
        hits, med = _anchor_stats(trace, anchors)
        span = (max(p[0] for p in trace) - min(p[0] for p in trace)) / \
            max(1.0, frame[2] - frame[0])
        cands.append({"trace": trace, "kind": kind, "key": g["key"], "npix": g["npix"],
                      "hits": hits, "med_dist": med, "span": round(span, 3)})
    if not cands:
        return []
    cands.sort(key=lambda c: (-c["hits"],
                              c["med_dist"] if c["med_dist"] is not None else 1e9,
                              -c["span"], -len(c["trace"])))
    return cands


def _candidates_for_color(hsv, frame, color, anchors, exclude_boxes=(), hue_tol=14,
                          allow_loop=False):
    """该颜色的所有候选轨迹（颜色偏淡时放宽饱和下限再试一次）。"""
    mask = el.color_mask(hsv, frame, color, hue_tol, exclude_boxes=exclude_boxes)
    cands = candidates_in_mask(mask, frame, anchors, allow_loop=allow_loop)
    if not cands:
        mask = el.color_mask(hsv, frame, color, hue_tol, exclude_boxes=exclude_boxes,
                             min_sat=40)
        cands = candidates_in_mask(mask, frame, anchors, allow_loop=allow_loop)
    return cands


def _pick_candidate(cands, avoid=()):
    """从排好序的候选里挑一条：跳过前面曲线已经认领过的实例。"""
    info = {"candidates": len(cands), "avoided": 0}
    if not cands:
        return None, info
    avoid = set(avoid)
    pick = next((c for c in cands if c["key"] not in avoid), None)
    if pick is None:                      # 全被前面的曲线认领了
        pick = cands[0]
        info["reused"] = True
    else:
        info["avoided"] = sum(1 for c in cands if c["key"] in avoid)
    info.update({"hits": pick["hits"], "med_dist": pick["med_dist"],
                 "span": pick["span"], "npix": pick["npix"], "key": pick["key"]})
    return pick, info


def trace_in_mask(mask, frame, anchors, avoid=(), allow_loop=False, reject_rules=False):
    """在任意掩膜上按锚点挑一条轨迹（黑线路线用它）。返回 (trace, kind, info)。"""
    info = {"mask_px": int((mask > 0).sum())}
    cands = candidates_in_mask(mask, frame, anchors, allow_loop=allow_loop,
                               reject_rules=reject_rules)
    pick, pinfo = _pick_candidate(cands, avoid)
    info.update(pinfo)
    if pick is None:
        info["reason"] = "掩膜里没有可用的线段（深色笔画都被判成坐标轴/文字了？）"
        return [], "none", info
    return pick["trace"], pick["kind"], info


def trace_series(hsv, frame, color, anchors, exclude_boxes=(), hue_tol=14, avoid=(),
                 allow_loop=False):
    """挑出这条曲线的轨迹。返回 (trace, kind, info)。

    排序原则：先看有几个锚点落在候选上（同色多实例时这是唯一可靠的信号），
    再看锚点到它的中位距离，最后才比覆盖长度。`avoid` 是已经被前面几条曲线
    认领过的实例，命中就顺延到下一个候选——两条线追成同一条是实测踩过的坑。

    按模型给的颜色一个实例都找不到时，会退回"这张图真实的色相簇"里最近的一个
    （±25 色相内）重试：图例色块和实际画的线不一定同色——实测 2026-01-0340
    图 2(a) 的 MeOH 图例是 hue 43，画出来的线是 hue 33，差 10 就让整条线消失。
    """
    info = {"color": [int(v) for v in color], "candidates": 0, "avoided": 0}
    cands = _candidates_for_color(hsv, frame, color, anchors, exclude_boxes, hue_tol,
                                  allow_loop=allow_loop)
    if not cands:
        base_hue = _hsv_of(color)[0]
        for h in sorted(_dominant_hues(hsv, frame),
                        key=lambda x: _hue_dist(x, base_hue)):
            near = _hue_dist(h, base_hue)
            if near > 25 or near < 2:
                continue
            alt_color = _hue_color(hsv, frame, h)
            if alt_color is None:
                continue
            alt = _candidates_for_color(hsv, frame, alt_color, anchors, exclude_boxes,
                                        hue_tol, allow_loop=allow_loop)
            if alt:
                cands = alt
                info["color"] = [int(v) for v in alt_color]
                info["color_shift"] = h
                break
    pick, pinfo = _pick_candidate(cands, avoid)
    info.update(pinfo)
    if pick is None:
        info["reason"] = "这个颜色在图上找不到线段（掩膜为空）"
        return [], "none", info
    return pick["trace"], pick["kind"], info


def trace_overlap(t1, t2, tol=3.0):
    """两条轨迹的重合度（共同列里 y 差 ≤ tol 的占比）。抓"两条线追成一条"。"""
    if not t1 or not t2:
        return 0.0
    d1, d2 = dict(t1), dict(t2)
    common = set(d1) & set(d2)
    if len(common) < 10:
        return 0.0
    same = sum(1 for x in common if abs(d1[x] - d2[x]) <= tol)
    return same / float(len(common))


# ----------------------------------------------------------------- 遮挡补全
#
# 这是"重合部分颜色往往只显示一个"的正面对策。两条线画在一起时，压在下面的那条
# **在重合段里根本没有像素**——不是追踪器追丢了，而是图上没有东西可追。旧实现只能
# 如实断在那里：曲线少一截，或者两条线被追成同一条。
#
# 判据是证据，不是猜：
#   * 本颜色在 x1 消失、在 x2 重新出现，而这两点之间**另一条曲线真的画在那里**
#     （桥接路径上大部分列都有别的笔画压着）-> 判为遮挡，沿那根笔画补过去；
#   * 断口很短（正常虚线断口）-> 直接线性补；
#   * 断口长、又没有任何笔画经过 -> 不补（曲线本来就在那里结束，不能凭空造数据）。
#
# 补出来的点与实测点分开计数（写进报告、画在原图上），所以"哪段是量的、哪段是补的"
# 永远查得出来。


def is_x_monotone(trace, tol=2.0):
    """这条轨迹能不能按列读（普通曲线）。闭合回线不能，禁止对它做按列补全。"""
    if len(trace) < 3:
        return True
    xs = [p[0] for p in trace]
    back = sum(1 for a, b in zip(xs, xs[1:]) if b < a - tol)
    return back <= max(1, int(0.02 * len(xs)))


def dense_trace(trace):
    """{列: y} —— 每列一个值（实测点之间线性插值），供"沿另一条线补全"用。"""
    out = {}
    if not trace:
        return out
    pts = sorted(trace)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ix0, ix1 = int(round(x0)), int(round(x1))
        if ix1 <= ix0:
            out.setdefault(ix0, float(y0))
            continue
        for x in range(ix0, ix1 + 1):
            out[x] = float(y0 + (y1 - y0) * (x - ix0) / float(ix1 - ix0))
    out[int(round(pts[0][0]))] = float(pts[0][1])
    out[int(round(pts[-1][0]))] = float(pts[-1][1])
    return out


def _slope_of(pts, k=9):
    """轨迹末端的局部斜率（每列多少像素）。k 取大一点，抗单点抖动。"""
    if len(pts) < 2:
        return 0.0
    (x0, y0), (x1, y1) = pts[max(0, len(pts) - k)], pts[-1]
    if abs(x1 - x0) < 1e-6:
        return 0.0
    return float(np.clip((y1 - y0) / (x1 - x0), -5.0, 5.0))


def _run_at(runs, y, tol):
    """这一列里离 y 最近的同色竖切（超出容差就是没有）。"""
    best = None
    for cy, _ln in runs or ():
        d = abs(cy - y)
        if d <= tol and (best is None or d < best[0]):
            best = (d, float(cy))
    return None if best is None else best[1]


def _interp_cols(x0, y0, x1, y1):
    """两列之间的插值点（不含两端）。"""
    step = 1 if x1 >= x0 else -1
    return [(float(x), float(y0 + (y1 - y0) * (x - x0) / float(x1 - x0)))
            for x in range(int(x0) + step, int(x1), step)]


def _corridor_cover(occluders, x0, y0, x1, y1, near=8.0):
    """"这条桥上有多少比例的列被某一根别的笔画压着" -> (占比, 是谁)。

    这就是"被遮挡"与"曲线到头了"的分界证据：真被压着，桥的每一列附近都该有那根
    笔画的像素；曲线到头的地方，桥上是空白。
    """
    step = 1 if x1 >= x0 else -1
    xs = list(range(int(x0) + step, int(x1), step))
    if not xs:
        return 1.0, None
    best_n, best_label = 0, None
    for label, dense in occluders or ():
        hit = 0
        for x in xs:
            cy = dense.get(x)
            if cy is None:
                continue
            t = (x - x0) / float(x1 - x0) if x1 != x0 else 0.0
            if abs(cy - (y0 + (y1 - y0) * t)) <= near:
                hit += 1
        if hit > best_n:
            best_n, best_label = hit, label
    return best_n / float(len(xs)), best_label


def _bridge(runs, occluders, last_x, last_y, slope, direction, max_bridge, cover_need,
            evidence_near=8.0):
    """本颜色消失之后，在 max_bridge 列内找它"重新出现"的位置 -> (x, y, 补点, 说明)。"""
    step = 1 if direction > 0 else -1
    for d in range(2, int(max_bridge) + 1):
        x = int(round(last_x)) + step * d
        y = _run_at(runs.get(x), last_y + slope * d, 9.0 + 0.12 * d)
        if y is None:
            continue
        cover, by = _corridor_cover(occluders, last_x, last_y, x, y, evidence_near)
        if cover >= cover_need:
            return x, y, _interp_cols(last_x, last_y, x, y), f"被「{by or '其它笔画'}」遮挡"
        if d <= 4:                       # 短断口：正常虚线断口，不需要证据
            return x, y, _interp_cols(last_x, last_y, x, y), "短断口"
    return None


def _walk_extend(runs, frame, x_end, y_end, slope, direction, occ, max_jump, max_gap,
                 max_bridge, cover_need):
    """从端点朝一个方向续追（返回的点按行走顺序，允许跨遮挡桥接）。"""
    step = 1 if direction > 0 else -1
    left, _top, right, _bottom = frame
    lo, hi = left + 1, right
    pts, spans = [], []
    prev_y, last_x = float(y_end), float(x_end)
    gap = 0
    x = int(round(x_end)) + step
    while lo < x < hi:
        predict = prev_y + slope * (x - last_x)
        tol = max_jump + max(0.0, abs(x - last_x) - 1) * abs(slope) * 1.6
        y = _run_at(runs.get(x), predict, tol)
        if y is None:
            gap += 1
            if gap > max_gap:
                br = _bridge(runs, occ, last_x, prev_y, slope, direction, max_bridge,
                             cover_need)
                if br is None:
                    break
                bx, by, filler, how = br
                pts.extend(filler)
                pts.append((float(bx), float(by)))
                spans.append({"x0": int(round(last_x)), "x1": int(round(bx)),
                              "cols": abs(int(round(bx)) - int(round(last_x))),
                              "by": how})
                prev_y, last_x = by, float(bx)
                gap = 0
                x = int(round(bx)) + step
                continue
            x += step
            continue
        gap = 0
        pts.append((float(x), y))
        last_x, prev_y = float(x), y
        if len(pts) >= 4:
            slope = _slope_of(pts)
        x += step
    return pts, spans


def _follow_occluder(runs, occ, pts, direction, join_near=6.0, rejoin_near=6.0,
                     min_len=6, max_len=500, look_back=14):
    """曲线贴到另一条线上、然后一直被压着走到头 -> 沿那条线补到它的末端。

    与 _bridge 的分工：_bridge 要求本颜色**重新出现**（有落点）；这里处理它再也不
    出现的收尾段——判据是消失点已经贴在另一条线上（≤join_near 像素）。两条曲线在
    末端完全重合的图（p14 那种）只能这样补。
    """
    step = 1 if direction > 0 else -1
    # 末端几个点里找一个"确实还贴在另一条线上"的落点：续追时偶尔会跳到一个杂色
    # 像素上（实测 p14 在 x=680 有个孤立像素把终点带偏了 9px），从那里起跳会接不
    # 上。往回找最近一个贴着遮挡线的点，用它当起点，并让调用方丢掉后面的杂点。
    tail = pts[-look_back:] if direction > 0 else pts[:look_back]
    tail = list(reversed(tail)) if direction > 0 else tail
    for label, dense in occ or ():
        if not dense:
            continue
        start = None
        for x, y in tail:
            at = dense.get(int(round(x)))
            if at is not None and abs(at - y) <= join_near:
                start = (float(x), float(y))
                break
        if start is None:
            continue
        x_end, y_end = start
        xs = sorted(dense)
        reach = (max(xs) - x_end) if direction > 0 else (x_end - min(xs))
        if reach < min_len:
            continue
        filler, rejoin = [], None
        for d in range(1, int(min(reach, max_len)) + 1):
            x = int(round(x_end)) + step * d
            cy = dense.get(x)
            if cy is None:
                break
            filler.append((float(x), float(cy)))
            hit = _run_at(runs.get(x), cy, rejoin_near)
            if hit is not None:
                rejoin = (x, hit)
                break
        if rejoin is not None:
            if abs(rejoin[0] - x_end) < min_len:
                continue
            return start, rejoin, filler, f"沿「{label}」回到可见段"
        if len(filler) >= min_len:
            return start, filler[-1], filler, f"末端与「{label}」重合"
    return None


def complete_trace(trace, mask, frame, occluders=(), max_jump=MAX_JUMP, max_gap=MAX_GAP,
                   cover_need=0.6, max_bridge=None, max_total=None):
    """把"压在别的曲线下面"的那几段补上 -> (新轨迹, 补全信息)。

    occluders 是 [(名字, dense_trace(另一条轨迹))]。补全信息会写进报告与质检图，
    所以实测点/补出来的点分得清。闭合回线（kind=loop）不做按列补全。
    """
    info = {"filled": 0, "bridged_cols": 0, "spans": [],
            "occluders": [l for l, _ in (occluders or [])]}
    if not trace or not is_x_monotone(trace):
        return list(trace), info
    runs = _runs_by_column(mask, frame)
    if not runs:
        return list(trace), info
    fw = max(1, frame[2] - frame[0])
    if max_bridge is None:
        max_bridge = int(0.45 * fw)
    if max_total is None:
        max_total = int(0.70 * fw)
    pts = sorted((float(x), float(y)) for x, y in trace)
    total = 0
    # ---- 两端续追：交替走"可见段"与"贴着另一条线补" ----
    # （一段可见的虚线之后往往又是遮挡，所以这里要来回换手，不是一次定输赢）
    for direction in (1, -1):
        for _round in range(4):
            if total >= max_total:
                break
            if direction > 0:
                x_end, y_end, slope = pts[-1][0], pts[-1][1], _slope_of(pts)
            else:
                x_end, y_end, slope = pts[0][0], pts[0][1], _slope_of(pts[::-1])
            ext, spans = _walk_extend(runs, frame, x_end, y_end, slope, direction,
                                      occluders, min(max_jump, 10.0), max_gap,
                                      min(max_bridge, max_total - total), cover_need)
            if not ext:
                fol = _follow_occluder(runs, occluders, pts, direction)
                if fol is None:
                    break
                (sx, _sy), (fx, fy), filler, how = fol
                # 丢掉末端被杂色像素带偏的那几个点，从"确实贴在遮挡线上"的位置重接
                pts = [q for q in pts if (q[0] <= sx + 1e-6 if direction > 0
                                          else q[0] >= sx - 1e-6)] or pts
                ext = list(filler)
                spans = [{"x0": int(round(sx)), "x1": int(round(fx)),
                          "cols": abs(int(round(fx)) - int(round(sx))), "by": how}]
            add = sum(s["cols"] for s in spans)
            if total + add > max_total:
                break
            total += add
            info["bridged_cols"] += add
            info["spans"].extend(spans)
            pts = pts + ext if direction > 0 else ext[::-1] + pts
    # ---- 中间断口：短断口线性补；被遮挡沿那条线补；长断口无证据就不补 ----
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        out.append((x0, y0))
        n = int(round(x1)) - int(round(x0))
        if n <= 1:
            continue
        filler, occluded = None, False
        for label, dense in occluders or ():
            a, b = dense.get(int(round(x0))), dense.get(int(round(x1)))
            if a is None or b is None or abs(a - y0) > 6.0 or abs(b - y1) > 6.0:
                continue
            cand = [dense.get(x) for x in range(int(round(x0)) + 1, int(round(x1)))]
            if any(c is None for c in cand):
                continue
            filler = [(float(int(round(x0)) + 1 + i), float(c))
                      for i, c in enumerate(cand)]
            info["spans"].append({"x0": int(round(x0)), "x1": int(round(x1)),
                                  "cols": len(filler), "by": f"被「{label}」遮挡"})
            occluded = True
            break
        if filler is None:
            if n > max_gap:                 # 长断口又没有证据：保持断开，报告里点名
                info.setdefault("open_gaps", []).append(
                    {"x0": int(round(x0)), "x1": int(round(x1)), "cols": n})
                continue
            filler = _interp_cols(x0, y0, x1, y1)
        if total + len(filler) > max_total:
            continue
        total += len(filler)
        info["filled"] += len(filler)
        if occluded:
            info["bridged_cols"] += len(filler)
        out.extend(filler)
    out.append(pts[-1])
    return out, info


def markers_in_mask(mask, trace, half=22, min_markers=3):
    """标记符号的中心点（落在追踪走廊内的那些）。

    腐蚀留住团块、去掉细笔画（追踪器自己的老办法）；限制在走廊内，文字和别的曲线
    的标记就进不来。颜色路线和深色路线共用它——模型说这条曲线带标记时，标记中心
    比线本身更适合当数据（标记处线会被自己的符号撑出一个平台）。
    """
    h, w = mask.shape[:2]
    corridor = np.zeros((h, w), np.uint8)
    for x, y in trace:
        cv2.circle(corridor, (int(x), int(y)), half, 255, -1)
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


def markers_in_corridor(img, hsv, trace, color_bgr, half=22, min_markers=3):
    """Marker glyph centres that sit **on the traced path**（颜色路线的入口）。"""
    h, w = img.shape[:2]
    mask = el.color_mask(hsv, (0, 0, w - 1, h - 1), color_bgr, 12)
    return markers_in_mask(mask, trace, half=half, min_markers=min_markers)


def line_through_markers(markers, min_dx=2.0):
    """把标记中心连成曲线（散点+折线那种图，点即曲线）。

    比沿画出来的线追踪更准：线会被标记符号撑出平台、被别的曲线盖住，而标记中心
    是画图时真正的数据点位置。
    """
    out = []
    for x, y in sorted(markers):
        if out and abs(x - out[-1][0]) < min_dx:
            continue
        out.append((float(x), float(y)))
    return out


def snap_line_to_markers(trace, markers, half_w=7.0):
    """线轨迹穿过标记符号时会被符号的边缘带成平台：在标记的横向范围内改用标记中心。"""
    if not trace or not markers:
        return list(trace)
    ys = sorted((float(m[0]), float(m[1])) for m in markers)
    out = []
    for x, y in trace:
        x = float(x)
        # 找最近的标记中心（按 x）
        best = None
        for mx, my_ in ys:
            d = abs(mx - x)
            if d <= half_w and (best is None or d < best[0]):
                best = (d, my_)
        out.append((x, best[1] if best else float(y)))
    return out


def collapse_plateaus(trace, tol=1.0, min_run=3, min_keep=20):
    """把标记符号撑出来的"平台"压成一个点：连续同高(±tol)的一段只留中间那个。

    散点图的每个标记宽十几像素，逐列追踪在它上面走平，画出来就是**阶梯**（实测
    2023-01-1635 图 4：一条曲线 683 行里有 38 段平台、190 个点落在平台上）。
    只在模型说"这条带标记"时用——实心数据线的水平段不能压。
    压完太短（说明整条线基本是平的）就保持原样，交回调用方。
    """
    pts = sorted((float(x), float(y)) for x, y in trace)
    out, run = [], []

    def flush():
        if not run:
            return
        if len(run) >= min_run:
            mid = run[len(run) // 2]
            out.append(mid)
        else:
            out.extend(run)

    for x, y in pts:
        if run and abs(y - run[-1][1]) <= tol:
            run.append((x, y))
        else:
            flush()
            run = [(x, y)]
    flush()
    return out if len(out) >= min(min_keep, len(pts)) else list(trace)
