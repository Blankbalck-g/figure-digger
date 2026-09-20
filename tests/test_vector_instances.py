"""Regression: vector series are physical instances, not one bucket per colour.

The original implementation concatenated every blue path into one CSV.  A chart
with five blue experimental trajectories therefore jumped back to x=0 four times.
This synthetic page pins the intended behaviour without depending on user PDFs.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dig"))

import vector_extract as vx  # noqa: E402


AXES = (0.0, 0.0, 200.0, 120.0)


def _path(y, color=(0.0, 0.0, 1.0)):
    pts = [(10.0 + i * 15.0, y - i * 4.0) for i in range(10)]
    return {"pts": pts, "x0": 10.0, "x1": 145.0, "top": min(p[1] for p in pts),
            "bottom": max(p[1] for p in pts), "stroke": True, "fill": False,
            "stroking_color": color, "non_stroking_color": None, "path": []}


def _rect(x, y, color=(1.0, 0.0, 0.0)):
    return {"x0": x - 1.5, "x1": x + 1.5, "top": y - 1.5, "bottom": y + 1.5,
            "stroke": True, "fill": True, "stroking_color": color,
            "non_stroking_color": color}


def main():
    blue = [_path(105.0 - j * 10.0) for j in range(5)]
    red_squares = [_rect(15.0 + i * 18.0, 100.0 - i * 6.0) for i in range(8)]
    page = SimpleNamespace(curves=blue, lines=[], rects=red_squares)
    got = vx._vector_candidates(page, AXES)
    blue_paths = [c for c in got if c["color_key"] == (0.0, 0.0, 1.0)]
    red_tracks = [c for c in got if c["color_key"] == (1.0, 0.0, 0.0)]
    ok = len(blue_paths) == 5 and len(red_tracks) == 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 同色物理路径保持独立："
          f"蓝线={len(blue_paths)}（期望5），红色标记轨迹={len(red_tracks)}（期望1）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
