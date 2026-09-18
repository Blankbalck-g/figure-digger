"""Propose the *objects* a figure is made of, so a model can say which is which.

The old pipeline went colour -> series: every colour became one curve. That cannot
work on real figures, because the meaningful unit is not a colour but an object:

  * one colour can host a data series *and* an author's slope guide ("S ∝ t^0.5"),
  * one series can be drawn as markers *and* a connecting/model line of the same
    colour, and those two must not be merged into one wobbling trace,
  * a legend entry can be a marker swatch while the line next to it belongs to a
    different entry ("▽ Iso-octane" vs "—·— 7 MPa").

So this module only *proposes* objects - line strokes, marker series, text blocks,
shaded bands - each with a crop and a short factual description. Deciding what each
one means is the model's job (see vlm_tasks.judge_objects); extracting values from an
accepted object is the code's job (marker centres for markers, a continuity trace for
lines).
"""

import cv2
import numpy as np

import imgio as iio

MAX_OBJECTS = 12          # 一张面板最多送几个物件去做判定（再多缩略图就看不清了）
MIN_MARKERS = 4           # 少于这么多重复小图形，不足以称为"标记系列"
STROKE_THICK = 7.0        # 笔画厚度上限：面积/长轴，超过它算"块"（标记），否则算"线"
MARKER_MIN_SIDE = 7       # 标记符号的最小边长（比这更小的是虚线的"点"或文字笔画）
FILL_RATIO = 0.5          # 实心度超过它 + 面积够大 => 误差带/底纹
STRICT_SAT = 85           # 不透明笔画的最低饱和度（半透明误差带低于它，会被排除）
STRICT_DARK = 110         # 深色笔画的上限（文字、黑线）
MAX_INSTANCES = 3         # 同一颜色族里最多认几条线


def _clip(mask, frame, exclude_boxes):
    left, top, right, bottom = frame
    out = mask.copy()
    out[:top + 2, :] = 0
    out[bottom:, :] = 0
    out[:, :left + 2] = 0
    out[:, right:] = 0
    for (x, y, w, h) in exclude_boxes:
        out[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] = 0
    return out


def ink_mask(hsv, gray, frame, exclude_boxes=()):
    """Opaque strokes only: markers, lines and text - not the shaded error bands.

    The bands have to be excluded *before* connected components: a dashed line drawn
    on top of its band touches it, so the two merge into one component and the whole
    panel becomes a single blob (measured: one 14.6k-pixel component was the band plus
    both curves, and nothing else was found in that panel). Bands are semi-transparent
    - high value, moderate saturation - while stroke colours are saturated or dark.
    """
    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)
    mask = ((sat >= STRICT_SAT) | (gray < STRICT_DARK)).astype(np.uint8) * 255
    return _clip(mask, frame, exclude_boxes)


def _components(mask, min_area=30):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or max(w, h) < 6:      # 网格残渣、抗锯齿碎点
            continue
        out.append({"label": i, "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "area": int(area), "cx": float(cents[i][0]), "cy": float(cents[i][1])})
    return out


def _median_color(img, mask, comp):
    sel = (mask[comp["y"]:comp["y"] + comp["h"], comp["x"]:comp["x"] + comp["w"]] > 0)
    patch = img[comp["y"]:comp["y"] + comp["h"], comp["x"]:comp["x"] + comp["w"]]
    if not sel.any():
        return (0, 0, 0)
    return tuple(int(v) for v in np.median(patch[sel], axis=0))


def _hex(bgr):
    return "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _hue(bgr):
    px = np.uint8([[list(bgr)]])
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0][0])


def _family(comp):
    """Colour family: neighbouring hues in one bucket, dark grey/black in its own.

    Grouping by exact hue does not work here: one red series shows up as a range of
    reds (marker fill, dash core, anti-aliased edges), and bucketing those separately
    split a single series into five "series" (measured on 2014-01-1413 p14).
    """
    if comp["sat"] < 60 and max(comp["color"]) < 130:
        return "dark"
    return f"h{int(comp['hue'] // 30)}"


def _cluster_along_x(comps, min_dx_frac=0.4, max_dx_frac=2.2, max_dy=30.0, slope=0.9):
    """Chain markers that lie along the same curve.

    Plain proximity clustering fails on these figures: markers of one series sit
    ~60px apart along x, while the two curves of the figure have pairs of markers
    stacked 18-40px apart at the *same* x. A radius big enough to chain a series
    therefore swallows the neighbouring curve (measured: radius=70 gave pairs,
    radius=130 merged the curves).

    So a link needs both: a real horizontal step (>= 0.4 x the median spacing) and a
    plausible rise. Vertically stacked markers at the same x - the other curve's -
    fail the first test.
    """
    if not comps:
        return []
    xs = sorted({round(c["cx"], 1) for c in comps})     # 去重：同 x 的成对标记不算间距
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    median_dx = float(np.median(gaps)) if gaps else 0.0
    limit_dx = max(20.0, max_dx_frac * median_dx)
    need_dx = min_dx_frac * median_dx
    n = len(comps)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(comps[i]["cx"] - comps[j]["cx"])
            if dx > limit_dx or dx < need_dx:
                continue
            dy = abs(comps[i]["cy"] - comps[j]["cy"])
            if dy <= max(max_dy, slope * dx):
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(comps[i])
    return list(groups.values())


def _varies_along_curve(members):
    """A data series rises or falls; a text line stays on one baseline.

    Used to reject "marker series" that are really the words of an annotation
    (36 letters of a label grouped into words, all at the same height).
    """
    ys = [c["cy"] for c in members]
    heights = [c["h"] for c in members]
    return (max(ys) - min(ys)) >= 3.0 * max(1.0, float(np.median(heights)))


def _chain_markers(points, max_step_frac=3.0, min_tol=25.0, tol_frac=0.6):
    """Group marker centres into curves by *following* them, one curve at a time.

    Symmetric "are these two linked" rules fail here: when two curves run close, a
    marker of one curve sits within the vertical tolerance of the other curve's next
    marker, and the whole panel collapses into a single chain (measured on
    2014-01-1413 p14: 14 markers of two curves merged into one series). Following a
    chain with a predicted position - the same trick the line tracer uses - keeps the
    two curves apart, because the step to the other curve never matches the predicted
    rise.
    """
    pts = sorted((float(x), float(y)) for x, y in points)
    if not pts:
        return []
    xs = sorted({round(x, 1) for x, _ in pts})
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    median_dx = float(np.median(gaps)) if gaps else 0.0
    max_step = max(25.0, max_step_frac * median_dx)
    groups = []
    left = list(pts)
    while left:
        cur = left.pop(0)
        chain = [cur]
        slope = 0.0
        while True:
            best = None
            for k, (x, y) in enumerate(left):
                dx = x - cur[0]
                if dx <= 0 or dx > max_step:
                    continue
                tol = max(min_tol, tol_frac * dx)
                d = abs(y - (cur[1] + slope * dx))
                if d <= tol and (best is None or d < best[0]):
                    best = (d, k, x, y)
            if best is None:
                break
            _, k, x, y = best
            slope = max(-2.0, min(2.0, (y - cur[1]) / max(1e-6, x - cur[0])))
            cur = (x, y)
            chain.append(cur)
            left.pop(k)
        groups.append(chain)
    return groups


def _line_y_at(trace, x, reach=12.0):
    """y of the traced line at this x (nearest traced column within `reach`)."""
    best = None
    for tx, ty in trace:
        d = abs(tx - x)
        if d <= reach and (best is None or d < best[0]):
            best = (d, ty)
    return best[1] if best else None


def _drop_fragments(objects, tol=8.0, need=0.75):
    """A shorter trace lying on top of a longer one of the same colour is its fragment.

    Sending every fragment to the model blurred the picture: a clean two-curve panel
    produced nine numbered objects, and the model (correctly, given what it saw) called
    the long one "a frame line". Fragments stay in the report but are not offered.
    """
    lines = [o for o in objects if o["kind"] == "line" and o.get("trace")]
    lines.sort(key=lambda o: -len(o["trace"]))
    keep, dropped = [], 0
    for o in lines:
        by = {}
        for k in keep:
            for x, y in k["trace"]:
                by.setdefault(x, []).append(y)
        inside = total = 0
        for x, y in o["trace"]:
            ys = by.get(x)
            if not ys:
                continue
            total += 1
            if min(abs(y - yy) for yy in ys) <= tol:
                inside += 1
        if total >= 15 and inside / total >= need:
            o["fragment_of"] = True
            dropped += 1
            continue
        keep.append(o)
    return [o for o in objects if not o.get("fragment_of")]


def _heal_small(comps, max_gap=6.0, max_major=80):
    """Merge a marker that an error bar cut into left/right halves.

    A black error bar drawn through a coloured triangle splits it into two thin
    components, each of which looks like a stroke (measured on 2014-01-9079 p15: the
    red markers vanished from the palette while the blue ones survived). Halves are
    horizontally adjacent, of similar size and share most of their y range - unlike
    the dashes of a line, which are separated along the stroke direction.
    """
    out = [c for c in comps]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if max(a["w"], a["h"]) > max_major or max(b["w"], b["h"]) > max_major:
                    continue
                gap = max(a["x"], b["x"]) - min(a["x"] + a["w"], b["x"] + b["w"])
                y0 = max(a["y"], b["y"])
                y1 = min(a["y"] + a["h"], b["y"] + b["h"])
                overlap = max(0, y1 - y0) / max(1.0, float(min(a["h"], b["h"])))
                ratio = max(a["area"], b["area"]) / max(1.0, float(min(a["area"], b["area"])))
                # 同色、横向紧邻、纵向大体对齐 —— 被误差棒切开的两半
                if _family(a) != _family(b):
                    continue
                if gap <= max_gap and overlap >= 0.5 and ratio <= 2.5:
                    x0, y0b = min(a["x"], b["x"]), min(a["y"], b["y"])
                    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
                    y1b = max(a["y"] + a["h"], b["y"] + b["h"])
                    merged = {"label": a["label"], "x": x0, "y": y0b, "w": x1 - x0,
                              "h": y1b - y0b, "area": a["area"] + b["area"],
                              "cx": (x0 + x1) / 2.0, "cy": (y0b + y1b) / 2.0,
                              "color": a["color"], "hue": a["hue"], "sat": a["sat"],
                              "thickness": (a["area"] + b["area"]) / max(1.0, float(x1 - x0)),
                              "fill": 0.5, "healed": True}
                    out = [c for k, c in enumerate(out) if k not in (i, j)] + [merged]
                    changed = True
                    break
            if changed:
                break
    return out


def _trace_in_region(line_mask, region_mask, frame):
    """Follow the stroke inside a region; returns the traced pixel path."""
    import extract_lines as el
    sub = np.where(region_mask > 0, line_mask, 0).astype(np.uint8)
    return el._trace_mask(sub, frame)


def detect_objects(img, frame, exclude_boxes=(), max_objects=MAX_OBJECTS):
    """-> [object dict]; each has kind, colour, bbox, count/points and a crop.

    Markers are separated from lines by *erosion*, not by shape ratios: erode the
    family mask with a kernel a little larger than the stroke width and only the thick
    blobs (the markers) survive, while dashes, dots, arrows and text strokes vanish.
    Shape-ratio rules kept failing in both directions on real figures - a dash dot is
    as round as a marker, and a marker that an error bar cuts in half is as thin as a
    line - so the decision is made by thickness, measured locally.
    """
    import extract_lines as el

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = ink_mask(hsv, gray, frame, exclude_boxes)
    left, top, right, bottom = frame
    comps = _components(mask)
    if not comps:
        return []
    for c in comps:
        c["color"] = _median_color(img, mask, c)
        c["hue"] = _hue(c["color"])
        c["sat"] = int(hsv[int(c["cy"]), int(c["cx"]), 1])
        c["thickness"] = c["area"] / max(1.0, float(max(c["w"], c["h"])))
    comps = _heal_small(comps)            # 被误差棒切开的标记先缝上

    fam_strokes = {}
    for c in comps:
        fam_strokes.setdefault(_family(c), []).append(c["thickness"])
    # 用原始 label 图把每个家族的像素单独取出来
    _n_lab, labels = cv2.connectedComponents(mask)
    label_of = {}
    for c in comps:
        label_of.setdefault(_family(c), set()).add(c["label"])
    fam_mask = {}
    for fam, labs in label_of.items():
        m = np.zeros(mask.shape[:2], np.uint8)
        m[np.isin(labels, list(labs))] = 255
        fam_mask[fam] = m

    objects = []
    texts = []                  # 标记分组的落选者，最后和文字块一起处理
    for fam, m in fam_mask.items():
        if int((m > 0).sum()) < 120:
            continue
        # 笔画粗细：取该家族"细笔画"的众数（多数组件是虚线段/文字笔画）
        thick = sorted(fam_strokes.get(fam, [4.0]))
        stroke_w = thick[len(thick) // 4] if len(thick) > 3 else 4.0
        k = max(3, int(round(stroke_w)) + 2)
        core = cv2.erode(m, np.ones((k, k), np.uint8))
        n_core, cores = cv2.connectedComponents(core)
        marker_px = np.zeros(mask.shape[:2], np.uint8)
        cand = []
        for i in range(1, n_core):
            ys, xs = np.nonzero(cores == i)
            if len(xs) < 3:
                continue
            w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            area = len(xs)
            if max(w, h) > 60:          # 太长的多半是一条粗线，不是标记
                continue
            cand.append({"x": int(xs.min()), "y": int(ys.min()), "w": int(w), "h": int(h),
                         "area": int(area), "cx": float(xs.mean()), "cy": float(ys.mean()),
                         "label": i, "color": (0, 0, 0), "hue": 0, "sat": 0,
                         "thickness": area / max(1.0, float(max(w, h)))})
            marker_px[ys, xs] = 255
        # 线：把标记的像素减掉，剩下的细笔画交给连续性追踪器（它自己跨虚线断口）
        line_mask = cv2.bitwise_and(m, cv2.bitwise_not(marker_px))
        if int((line_mask > 0).sum()) >= 80:
            # 断口容忍放到 90 列：这些图的虚线间隔比追踪器默认的 45 大，否则一条
            # 虚线会被切成好几段，模型看到的就是一堆碎片而不是"一条线"
            for pts in el._instances_from_mask(line_mask, frame, MAX_INSTANCES, 20, 0.15,
                                               40, max_gap=90):
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cols = [img[int(y), int(x)] for x, y in pts[::max(1, len(pts) // 20)]]
                col = tuple(int(v) for v in np.median(np.array(cols, float), axis=0))
                objects.append({
                    "kind": "line", "family": fam, "hex": _hex(col), "color_bgr": col,
                    "hue": _hue(col), "nx": len(pts), "trace": pts,
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                })

        marker_px = cv2.dilate(marker_px, np.ones((k + 2, k + 2), np.uint8))

        # 标记系列：同尺寸、沿曲线成串、纵向有起伏
        classes = []
        for c in sorted(cand, key=lambda c: c["area"]):
            for cl in classes:
                a, b = cl[-1]["area"], c["area"]
                if max(a, b) <= 1.8 * max(1, min(a, b)):
                    cl.append(c)
                    break
            else:
                classes.append([c])
        # 分组用"挂在哪条线上"：同一根线上的标记才是一系列。纯几何串接在两条曲线靠近时
        # 会连错（实测两条红虚线的标记被并成一组：上曲线在 x=281 的点与下曲线在 x=342
        # 的点纵向只差 16px），而"离哪条线最近"对这几个图是稳的。
        fam_traces = [(o.get("id", id(o)), o["trace"]) for o in objects
                      if o["kind"] == "line" and o["family"] == fam and o.get("trace")]
        for cl in classes:
            if len(cl) < MIN_MARKERS:
                continue
            buckets = {}
            for c in cl:
                host = None
                for key, trace in fam_traces:
                    yl = _line_y_at(trace, c["cx"])
                    if yl is None:
                        continue
                    d = abs(yl - c["cy"])
                    if d <= 20 and (host is None or d < host[0]):
                        host = (d, key)
                buckets.setdefault(host[1] if host else None, []).append(c)
            for key, members in buckets.items():
                members.sort(key=lambda c: c["cx"])
                if len(members) < MIN_MARKERS:
                    texts.extend(members)
                    continue
                if not _varies_along_curve(members):
                    texts.extend(members)     # 同一水平线上的一串 = 文字
                    continue
                xs = [c["cx"] for c in members]
                if (max(xs) - min(xs)) < 0.2 * (right - left):
                    texts.extend(members)
                    continue
                px = np.zeros(mask.shape[:2], np.uint8)
                for c in members:
                    px[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] = 255
                col = tuple(int(v) for v in np.median(img[px > 0], axis=0))
                objects.append({
                    "kind": "markers", "family": fam, "hex": _hex(col),
                    "color_bgr": col, "hue": _hue(col),
                    "n_markers": len(members), "pixels": px, "on_line": key,
                    "centroids": [(c["cx"], c["cy"]) for c in members],
                    "bbox": [int(min(c["x"] for c in members)),
                             int(min(c["y"] for c in members)),
                             int(max(c["x"] + c["w"] for c in members)),
                             int(max(c["y"] + c["h"] for c in members))],
                })

    # 已被"标记系列/线"认领的像素减掉，剩下的才是标注文字；再把碎字聚成几块，
    # 否则几十个碎片会把送去做判定的名额吃光（一张面板最多 12 个物件）
    claimed = np.zeros(mask.shape[:2], np.uint8)
    for o in objects:
        if o["kind"] == "markers":
            claimed = cv2.bitwise_or(claimed, o["pixels"])
        elif o.get("trace"):
            for x, y in o["trace"]:
                cv2.circle(claimed, (int(x), int(y)), 6, 255, -1)
    rest = cv2.bitwise_and(mask, cv2.bitwise_not(claimed))
    rest = cv2.morphologyEx(rest, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11)))
    n_t, treg = cv2.connectedComponents(cv2.dilate(rest, np.ones((3, 3), np.uint8), 2))
    texts = []
    for i in range(1, n_t):
        ys, xs = np.nonzero((treg == i) & (rest > 0))
        if len(xs) < 60:
            continue
        texts.append({"kind": "text", "hex": _hex(tuple(int(v) for v in
                                                        np.median(img[ys, xs], axis=0))),
                      "nx": len(xs),
                      "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    texts.sort(key=lambda o: -o["nx"])
    objects += texts[:4]                 # 标注只留最大的几块，够模型判断即可

    objects = _drop_fragments(objects)
    # 线和标记优先占名额，剩下的名额才给标注文字
    objects.sort(key=lambda o: (o["kind"] == "text", -(o.get("nx") or 0),
                                -(o.get("n_markers") or 0)))
    for i, o in enumerate(objects[:max_objects], start=1):
        o["id"] = i
    # on_line 里存的是运行期 id，换成最终编号，模型才好引用（"挂在 #5 上"）
    idmap = {id(o): o.get("id") for o in objects[:max_objects]}
    for o in objects[:max_objects]:
        if o.get("on_line") in idmap:
            o["on_line"] = idmap[o["on_line"]]
    return objects[:max_objects]


def describe(obj, frame):
    """A short factual line about an object - the text half of the model's evidence."""
    left, top, right, bottom = frame
    x0, y0, x1, y1 = obj["bbox"]
    pos = (f"横向占图宽 {(x1 - x0) / max(1, right - left) * 100:.0f}%，"
           f"纵向占图高 {(y1 - y0) / max(1, bottom - top) * 100:.0f}%")
    if obj["kind"] == "line":
        return f"线状物件（{obj['nx']} 像素，{pos}），颜色 {obj['hex']}"
    if obj["kind"] == "markers":
        on_line = f"，挂在线 #{obj['on_line']} 上" if obj.get("on_line") else "，附近没有同色的线"
        return (f"{obj['n_markers']} 个同色同尺寸的标记符号（{pos}），"
                f"颜色 {obj['hex']}{on_line}")
    if obj["kind"] == "text":
        return f"文字/标注块（{obj['nx']} 像素，{pos}），颜色 {obj['hex']}"
    return f"大面积色块（{obj['nx']} 像素，{pos}），颜色 {obj['hex']}，疑似误差带/底纹"


def evidence_text(objects, frame):
    """The factual list handed to the model next to the numbered crops."""
    lines = []
    for o in objects:
        kind = {"line": "线", "markers": "标记系列", "text": "文字/标注",
                "band": "大面积色块"}.get(o["kind"], o["kind"])
        lines.append(f"#{o['id']}: {kind} —— {describe(o, frame)}")
    return "\n".join(lines)


def render_sheet(img, objects, frame, out_path, tile=260, cols=4):
    """Left: the figure with objects outlined and numbered. Right: one crop each."""
    left, top, right, bottom = frame
    vis = img.copy()
    for o in objects:
        x0, y0, x1, y1 = o["bbox"]
        color = (0, 0, 255) if o["kind"] != "text" else (255, 0, 0)
        if o["kind"] == "markers":
            for cx, cy in o["centroids"]:
                cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        elif o["kind"] == "line" and o.get("trace"):
            # 画稀疏的点而不是连续折线：连续线会把"虚线"画成"实线"，
            # 模型看到实线就会说"图例里是虚线，这根对不上"→ 判成边框（实测踩到）
            for x, y in o["trace"][::6]:
                cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(vis, (x0, max(0, y0 - 22)), (x0 + 34, y0), (255, 255, 255), -1)
        cv2.putText(vis, str(o["id"]), (x0 + 4, max(16, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    pad = 14
    tiles = []
    for o in objects:
        x0, y0, x1, y1 = o["bbox"]
        m = 18
        cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
        cx1, cy1 = min(img.shape[1], x1 + m), min(img.shape[0], y1 + m)
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        scale = min(tile / max(1, w), tile / max(1, h))
        rs = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA)
        canvas = np.full((tile, tile, 3), 255, np.uint8)
        oy, ox = (tile - rs.shape[0]) // 2, (tile - rs.shape[1]) // 2
        canvas[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs
        cv2.rectangle(canvas, (0, 0), (tile - 1, tile - 1), (200, 200, 200), 1)
        cv2.rectangle(canvas, (0, 0), (52, 30), (255, 255, 255), -1)
        cv2.putText(canvas, f"#{o['id']}", (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(canvas)

    if tiles:
        rows = []
        for i in range(0, len(tiles), cols):
            chunk = tiles[i:i + cols]
            while len(chunk) < cols:
                chunk.append(np.full((tile, tile, 3), 255, np.uint8))
            rows.append(cv2.hconcat(chunk))
        grid = cv2.vconcat(rows)
    else:
        grid = np.full((tile, tile, 3), 255, np.uint8)

    h = max(vis.shape[0], grid.shape[0]) + 2 * pad
    w = vis.shape[1] + grid.shape[1] + 3 * pad
    canvas = np.full((h, w, 3), 245, np.uint8)
    canvas[pad:pad + vis.shape[0], pad:pad + vis.shape[1]] = vis
    gy, gx = pad, pad * 2 + vis.shape[1]
    canvas[gy:gy + grid.shape[0], gx:gx + grid.shape[1]] = grid
    iio.imwrite(out_path, canvas)
    return out_path
