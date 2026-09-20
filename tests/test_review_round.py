"""回归：复盘轮（模型核对代码追出来的曲线）的接线与合并/丢弃语义。

复盘的用途：编号画回原图 -> 模型说"哪两个编号是同一条""哪条追错了""漏了哪条"。
这里用 mock VLM 验证整条链路（画图 -> 提问 -> 解析），并单独钉住
merge/drop 的编号语义（编号是 1 起的，合并保留编号小的那个）。

不需要 key、不联网。用法：python tests/test_review_round.py
"""

import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dig"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as R  # noqa: E402
import vlm_client as vc  # noqa: E402
from test_vlm_mock import Handler  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _entry(label, trace):
    return {"p": {"s": {"label": label}, "trace": trace, "kind": "line",
                  "info": {}, "dark": False, "color": (0, 0, 255), "how": "palette"},
            "want": "line", "markers": [], "line_data": [(1.0, 2.0), (2.0, 3.0)],
            "point_data": [], "note_dup": None, "note_draw": None, "span": 1.0}


def _trace(y0):
    return [(x, y0 + x * 0.01) for x in range(100, 300)]


def main():
    bad = 0
    # ---- 1) merge / drop 的编号语义 ----
    entries = [_entry("A", _trace(100)), _entry("B", _trace(200)),
               _entry("C", _trace(100))]
    out, failed, aliases = R._apply_review(
        {"merge": [[1, 3]], "drop": [2]}, entries, [], [])
    got = [e["p"]["s"]["label"] for e in out]
    ok = got == ["A"] and len(aliases) == 2
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] merge=[[1,3]] drop=[2] -> 保留 {got}"
          f"，合并/丢弃 {[a['label'] for a in aliases]}")

    # ---- 2) 复盘整链路（mock VLM）：默认走临时图，不污染正式输出 ----
    tmp = Path(tempfile.mkdtemp(prefix="dig_reviewtest_"))
    vc.CACHE_DIR = tmp / ".c"
    vc.LOG_PATH = tmp / "l.jsonl"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    vlm = vc.DeepSeekVLM(api_key="sk-mock",
                         base_url=f"http://127.0.0.1:{srv.server_address[1]}")
    import imgio as iio  # noqa: E402  （测试用：自造一张面板图，不依赖任何运行产物）
    panel_img = tmp / "panel.png"
    iio.imwrite(panel_img, np.full((320, 480, 3), 255, np.uint8))
    panel = {"id": "test_panel", "frame": [0, 0, 100, 100]}
    answer = R.revise_panel(vlm, panel_img, panel, [{"label": "A"}],
                            [_entry("A", _trace(100)), _entry("B", _trace(300))],
                            [{"label": "C", "anchors": [[0.5, 0.5]]}], [],
                            csv_dir=tmp / "out" / "csv")
    srv.shutdown()
    png = tmp / "out" / "review" / "test_panel_review.png"
    js = tmp / "out" / "review" / "test_panel_review.json"
    ok = bool(answer) and not png.exists() and not js.exists() and "merge" in answer
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 复盘链路：answer={answer} "
          f"默认留图={png.exists()} JSON={js.exists()}")

    # 明确要求诊断产物时仍可审计落盘。
    R._KEEP_INTERMEDIATES = True
    R.revise_panel(vlm, panel_img, panel, [{"label": "A"}],
                   [_entry("A", _trace(100)), _entry("B", _trace(300))],
                   [{"label": "C", "anchors": [[0.5, 0.5]]}], [],
                   csv_dir=tmp / "out" / "csv")
    R._KEEP_INTERMEDIATES = False
    keep_ok = png.exists() and js.exists()
    bad += 0 if keep_ok else 1
    print(f"  [{'OK ' if keep_ok else 'FAIL'}] --keep-intermediates："
          f"图={png.exists()} JSON={js.exists()}")
    if js.exists():
        saved = json.loads(js.read_text(encoding="utf-8"))
        print(f"        留档键: {sorted(saved)}（找到了 {len(saved.get('found') or [])} 条编号）")

    # 默认运行应清掉旧调试目录，但不能碰 csv 等持久成果。
    durable = tmp / "out" / "csv" / "keep.csv"
    durable.parent.mkdir(parents=True, exist_ok=True)
    durable.write_text("x,y\n0,0\n", encoding="utf-8")
    for name in ("gaps", "debug", "select", "objects"):
        d = tmp / "out" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "old.txt").write_text("stale", encoding="utf-8")
    R._clean_stale_intermediates(tmp / "out")
    clean_ok = durable.exists() and not any(
        (tmp / "out" / name).exists()
        for name in ("gaps", "review", "debug", "select", "objects"))
    bad += 0 if clean_ok else 1
    print(f"  [{'OK ' if clean_ok else 'FAIL'}] 默认清理旧中间目录且保留 CSV")
    print("通过" if not bad else f"{bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
