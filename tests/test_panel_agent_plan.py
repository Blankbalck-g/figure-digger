"""Regression: a model-confirmed six-panel figure must never become one panel.

No API key or network is used. Usage: python tests/test_panel_agent_plan.py <figure.png>
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dig"))

import run as app  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("用法：python tests/test_panel_agent_plan.py <六子图图片.png>")
        return 2
    image = Path(sys.argv[1])
    if not image.exists():
        print(f"找不到回归图片：{image}")
        return 2

    plan = {"panels": [
        {"label": "150 MPa", "plot_box": [.078, .013, .511, .293]},
        {"label": "120 MPa", "plot_box": [.564, .013, .969, .293]},
        {"label": "100 MPa", "plot_box": [.078, .337, .511, .617]},
        {"label": "80 MPa", "plot_box": [.564, .337, .969, .617]},
        {"label": "60 MPa", "plot_box": [.078, .662, .511, .943]},
        {"label": "40 MPa", "plot_box": [.564, .662, .969, .943]},
    ]}

    bad = 0
    with tempfile.TemporaryDirectory(prefix="panel_agent_test_") as tmp:
        meta = {}
        panels = app.split_figure(image, tmp, expected_panels=6,
                                  panel_geometry=plan, split_meta=meta)
        six_ok = len(panels) == 6 and meta.get("source") == "agent-plan+pixel-refine"
        bad += 0 if six_ok else 1
        print(f"  [{'OK ' if six_ok else 'FAIL'}] Agent 六面板计划被执行："
              f"panels={len(panels)}, source={meta.get('source')}")

        expected = [
            (71, 16, 476, 385), (522, 16, 901, 388),
            (71, 443, 476, 813), (522, 443, 901, 816),
            (73, 871, 475, 1239), (525, 871, 901, 1243),
        ]
        got = [tuple(x) for x in meta.get("plot_boxes") or []]
        snap_ok = len(got) == 6 and all(
            max(abs(a - b) for a, b in zip(actual, target)) <= 4
            for actual, target in zip(got, expected))
        bad += 0 if snap_ok else 1
        print(f"  [{'OK ' if snap_ok else 'FAIL'}] 模型粗框已由像素校准到坐标边界：{got}")

        # The dangerous old behaviour was flattening the whole 930x1317 image into
        # one coordinate system. A confirmed multi-panel count without valid geometry
        # must now fail explicitly instead.
        meta2 = {}
        unsafe = app.split_figure(image, tmp, expected_panels=6,
                                  panel_geometry={}, split_meta=meta2)
        guard_ok = unsafe == [] and meta2.get("source") == "count-mismatch"
        bad += 0 if guard_ok else 1
        print(f"  [{'OK ' if guard_ok else 'FAIL'}] 缺少有效计划时禁止整图降级："
              f"panels={len(unsafe)}, source={meta2.get('source')}")

    entries = [
        {"p": {"s": {"label": "Exp 100MPa", "role": "data",
                      "draw": "markers_only"}}},
        {"p": {"s": {"label": "Sim 100MPa", "role": "model",
                      "draw": "line_only"}}},
    ]
    aliases = []
    kept, _failed, aliases = app._apply_review(
        {"merge": [[1, 2]], "drop": []}, entries, [], aliases)
    semantic_ok = len(kept) == 2 and any(
        a.get("by") == "语义计划保护" for a in aliases)
    bad += 0 if semantic_ok else 1
    print(f"  [{'OK ' if semantic_ok else 'FAIL'}] 局部复盘不能合并实验与模拟系列："
          f"kept={len(kept)}, audit={aliases}")

    print("通过" if not bad else f"{bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
