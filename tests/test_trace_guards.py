"""回归：什么样的掩膜才允许走"闭合回线（轮廓）"这条路线。

实测踩过的坑（2024-01-3408 图 4）：散点+折线画的曲线，标记符号让很多列出现两条
以上竖切，`_looks_like_loop` 判成闭合回线，CSV 沿轮廓绕一圈 —— 在用户那里画出来
就是一个莫名其妙的"回路"。这里用合成图把三类形状钉死：

  1. 真正的闭合回线（细笔画围成一圈）-> 允许走轮廓法
  2. 散点+折线的曲线（标记符号造成多竖切）-> 必须按普通曲线追
  3. 几条曲线挤成一片的胖块 -> 必须按普通曲线追

不需要 key、不联网。用法：python tests/test_trace_guards.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dig"))

import series_seed as ss  # noqa: E402
import batch_extract as be  # noqa: E402
import run as app  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


FRAME = (0, 0, 640, 480)


def _loop_mask():
    m = np.zeros((480, 640), np.uint8)
    cv2.ellipse(m, (320, 240), (200, 150), 0, 0, 360, 255, 5)
    return m


def _marker_line_mask():
    """散点 + 折线 + 误差棒：标记 12px、连线 3px、每点一根 26px 误差棒。

    误差棒正是实测那张图让"每列竖切数"飙升的东西：没有它，普通曲线不会被误判成
    回线；加上它，必须靠"围没围出面积"这条判据才分得开。
    """
    m = np.zeros((480, 640), np.uint8)
    pts = [(60 + i * 52, 400 - i * 30) for i in range(10)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(m, a, b, 255, 3)
    for p in pts:
        cv2.line(m, (p[0], p[1] - 13), (p[0], p[1] + 13), 255, 2)
        cv2.rectangle(m, (p[0] - 6, p[1] - 6), (p[0] + 6, p[1] + 6), 255, -1)
    return m


def _legend_image():
    img = np.full((480, 640, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (620, 460), (0, 0, 0), 2)
    cv2.rectangle(img, (430, 320), (610, 445), (0, 0, 0), 2)
    for y, color in ((345, (255, 0, 0)), (380, (0, 255, 0)), (415, (0, 0, 255))):
        cv2.line(img, (445, y), (500, y), color, 3)
    cv2.putText(img, "Series", (510, 352), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _fat_blob_mask():
    """三条交叉曲线挤成一片。"""
    m = np.zeros((480, 640), np.uint8)
    for k in range(3):
        pts = np.array([[60 + i * 52, 240 + k * 14 + int(90 * np.sin(i / 2.0))]
                        for i in range(11)], np.int32)
        cv2.polylines(m, [pts], False, 255, 9)
    return m


def _dot_grid_trace():
    """点状网格线：横跨整幅、几乎没有起伏（用户在图上看到被当成曲线的那种）。"""
    return [(float(x), 240.0 + (x % 3) * 0.4) for x in range(20, 620)]


def _flat_data_trace():
    """真正的水平数据线（只走一半宽度）——不能被当成网格线丢掉。"""
    return [(float(x), 300.0) for x in range(100, 400)]


def _verdict(mask):
    if not ss._looks_like_loop(mask, FRAME) or not ss._single_blob(mask):
        return "line"
    loop = ss._contour_points(mask, FRAME)
    if len(loop) < 20:
        return "line"
    return "loop" if ss._encloses_area(mask, loop) else "line"


def main():
    cases = [("闭合回线", _loop_mask(), "loop"),
             ("散点+折线", _marker_line_mask(), "line"),
             ("多条曲线挤在一起", _fat_blob_mask(), "line")]
    bad = 0
    for name, mask, want in cases:
        got = _verdict(mask)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:16s} -> {got:5s}（期望 {want}）"
              f"  looks_like_loop={ss._looks_like_loop(mask, FRAME)}"
              f" single_blob={ss._single_blob(mask)}")
    for name, trace, want in [("点状网格线", _dot_grid_trace(), True),
                              ("水平数据线", _flat_data_trace(), False)]:
        got = ss.is_rule(trace, FRAME)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:16s} -> is_rule={got}（期望 {want}）")
    # 顺着网格线走了很久的轨迹（86% 的点在同一行）vs 正常的上升曲线
    riding = [(float(x), 157.5) for x in range(120, 400)] + \
             [(float(x), 157.5 - (x - 400) * 0.5) for x in range(400, 440)]
    rising = [(float(x), 300 - x * 0.4 + (x % 7)) for x in range(120, 440)]
    for name, trace, want in [("顺着网格线走", riding, True), ("正常上升曲线", rising, False)]:
        frac = ss.flat_run_frac(trace)
        got = frac >= 0.4
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:16s} -> 水平段占比 {frac:.2f}"
              f"（判为网格={got}，期望 {want}）")
    # 散点图的"阶梯"：标记符号宽十几像素，逐列追踪在它上面走平 —— 压成一个点
    step = []
    for i, (mx, my) in enumerate(zip(range(100, 400, 12), range(300, 200, -5))):
        step += [(float(mx + dx), float(my)) for dx in range(12)]
    collapsed = ss.collapse_plateaus(step)
    ok = len(collapsed) <= len(step) // 8
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 标记阶梯：{len(step)} 点 -> {len(collapsed)} 点"
          f"（每个标记一个值）")
    line = [(float(x), 300 - x * 0.3) for x in range(100, 400)]
    ok = len(ss.collapse_plateaus(line)) == len(line)
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 普通斜线不受影响：{len(line)} 点保持 {len(line)} 点")
    global_markers = ss.marker_centers(_marker_line_mask())
    ok = len(global_markers) == 10
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 全绘图区 marker：检测到 {len(global_markers)}/10 个")
    # 模型补锚点：缺口在两头（重合处最常见）和中间都要能插进去
    trace = [(float(x), 200.0 - x * 0.1) for x in range(200, 400, 5)]
    head = [(float(x), 215.0 - x * 0.05) for x in (120, 150, 180)]
    inner = [(310.0, 165.0), (320.0, 163.0)]
    tail = [(420.0, 155.0), (450.0, 152.0)]
    got = ss.bridge_through_anchors(trace, head + inner + tail)
    xs = [x for x, _ in got]
    ok = (min(xs) == 120 and max(xs) == 450
          and all(any(abs(x - a) < 0.5 for x, _ in got) for a in (310.0, 320.0)))
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 模型锚点插缺口：{len(trace)} -> {len(got)} 点，"
          f"x=[{min(xs):.0f},{max(xs):.0f}]（两头+中间都插上了）")
    # Agent coordinate contract: planner uses plot-normalised coordinates, critic
    # uses actual data coordinates.  Both must land in the same plotting rectangle,
    # never in the surrounding tick/label margins.
    frame = (100, 50, 500, 450)
    planner_px = ss.anchors_to_pixels([[0.25, 0.25]], frame, (800, 600))
    critic_px = ss.data_anchors_to_pixels([[2.5, 25]], frame, [0, 10, 0, 100])
    coord_ok = planner_px == [(200.0, 350.0)] and critic_px == [(200.0, 350.0)]
    bad += 0 if coord_ok else 1
    print(f"  [{'OK ' if coord_ok else 'FAIL'}] Agent 坐标协议："
          f"plot_norm={planner_px} data={critic_px}")
    # VLM gap/critic anchors are suggestions, so they must stay strictly inside the
    # plotting rectangle.  A measured line may touch an axis; a model point on that
    # axis is ambiguous and previously created a false vertical segment at x=0.
    guarded = app._model_anchors_in_frame(
        [(98, 200), (100, 200), (101.9, 200), (102, 200),
         (250, 49), (250, 52), (498, 448), (499, 449)], frame)
    guard_ok = guarded == [(102.0, 200.0), (250.0, 52.0), (498.0, 448.0)]
    bad += 0 if guard_ok else 1
    print(f"  [{'OK ' if guard_ok else 'FAIL'}] Agent 补点坐标框约束：{guarded}")
    continuous = [(float(x), 200.0) for x in range(100, 201)]
    gapped = [(float(x), 200.0) for x in list(range(100, 141)) + list(range(170, 201))]
    no_hole = app._missing_trace_spans(continuous, 100, 200, 6)
    one_hole = app._missing_trace_spans(gapped, 100, 200, 6)
    gap_ok = no_hole == [] and one_hole == [(140.0, 170.0)]
    bad += 0 if gap_ok else 1
    print(f"  [{'OK ' if gap_ok else 'FAIL'}] Agent 仅补真实缺口："
          f"连续={no_hole} 断口={one_hole}")
    # The whole legend is forbidden evidence. Masking only its coloured swatches lets
    # legend text/markers become fake data or a false continuation of a real curve.
    legend_img = _legend_image()
    legend_gray = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
    excluded = be.inner_boxes(legend_gray, (20, 20, 620, 460), img=legend_img)
    full_boxes = [b for b in excluded if len(b) == 4 and b[2] > 150 and b[3] > 100]
    legend_ok = len(full_boxes) == 1 and full_boxes[0][0] <= 430 \
        and full_boxes[0][1] <= 320 and full_boxes[0][0] + full_boxes[0][2] >= 610 \
        and full_boxes[0][1] + full_boxes[0][3] >= 445
    bad += 0 if legend_ok else 1
    print(f"  [{'OK ' if legend_ok else 'FAIL'}] 图例硬禁区：{full_boxes}")
    semantic = app._planner_exclude_boxes(
        {"extraction_plan": {"exclude_regions": [
            {"role": "legend", "box": [0.7, 0.05, 0.95, 0.2]},
            {"role": "annotation", "box": [0.1, 0.1, 0.2, 0.2]},
            {"role": "legend", "box": [0.0, 0.0, 1.0, 1.0]},
        ]}}, (100, 50, 500, 450))
    semantic_ok = semantic == [(377, 367, 106, 66)]
    bad += 0 if semantic_ok else 1
    print(f"  [{'OK ' if semantic_ok else 'FAIL'}] Agent 图例禁区：{semantic}")
    print("通过" if not bad else f"{bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
