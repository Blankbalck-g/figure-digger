"""回归：模板输出。

三条要求：①模板只让模型读一次，同样的模板再来必须复用缓存（不再调模型）
②生成的代码能把曲线渲染成模板文件（参数由模型给值、代码填空，读不到留空）
③不给模板时沿用上一次的；从来没有过就什么都不做（调用方只出 CSV）。

不需要 key、不联网。用法：python tests/test_template_out.py
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

import template_out as tpl  # noqa: E402
import run as R  # noqa: E402
import vlm_client as vc  # noqa: E402
from test_vlm_mock import Handler  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TEMPLATE = """### DIG export v1
paper={paper} panel={panel}
curve=fuel
time[ms] value
"""


def main():
    bad = 0
    tmp = Path(tempfile.mkdtemp(prefix="dig_tpl_"))
    tfile = tmp / "export_template.dat"
    tfile.write_text(TEMPLATE, encoding="utf-8")
    cache = tmp / ".templates"
    vc.CACHE_DIR = tmp / ".c"
    vc.LOG_PATH = tmp / "l.jsonl"

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    vlm = vc.DeepSeekVLM(api_key="sk-mock",
                         base_url=f"http://127.0.0.1:{srv.server_address[1]}")

    fmt = tpl.load(tfile, vlm=vlm, cache_dir=cache, log=print)
    calls1 = vlm.usage.get("calls", 0)
    ok = fmt is not None and fmt.ext == ".dat" and fmt.params == ["fuel"]
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 编译模板：ext={fmt.ext} params={fmt.params} 调用={calls1}")

    again = tpl.load(tfile, vlm=vlm, cache_dir=cache, log=print)     # 同一模板 -> 不再调模型
    calls2 = vlm.usage.get("calls", 0)
    ok = again is not None and again.key == fmt.key and calls2 == calls1
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 同一模板复用：调用 {calls1} -> {calls2}（应不变）")

    series = [{"name": "Heatedtip", "x": [0.1, 0.2], "y": [11, 22],
               "params": {"fuel": "MeOH"}},
              {"name": "Normaltip", "x": [0.1], "y": [9], "params": {"fuel": ""}}]
    files = fmt.render(series, {"paper": "P", "panel": "p1", "ext": ".dat"})
    text = files.get("Heatedtip.dat", "")
    ok = len(files) == 2 and "# fuel=MeOH" in text and "0.1\t11" in text
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 渲染：{sorted(files)} 首行={text.splitlines()[:3]}")

    fake_panel = tmp / "panel.png"
    import imgio as iio  # noqa: E402  （测试用：造一张小图当面板）
    iio.imwrite(fake_panel, np.full((40, 60, 3), 255, np.uint8))
    vals = tpl.ask_params(vlm, fake_panel, "cap", series, ["fuel", "pressure"])
    filled = {k: str((vals.get("Heatedtip") or {}).get(k, "")) for k in ("fuel", "pressure")}
    ok = filled == {"fuel": "MeOH", "pressure": ""}
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 参数：模型只给读到的 {vals.get('Heatedtip')}，"
          f"代码补齐没读到的键 -> {filled}")

    last = tpl.load(None, vlm=vlm, cache_dir=cache, log=print)       # 不给模板 -> 用上次的
    ok = last is not None and last.key == fmt.key
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 不传模板：沿用上次 {last.name if last else None}")

    empty = tpl.load(None, vlm=vlm, cache_dir=tmp / "none", log=None)
    ok = empty is None
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 从没用过模板：返回 None（只出 CSV）")

    # ---- 接进提取流程：面板的 CSV -> 模板文件（run._write_template_output）----
    csv_dir = tmp / "out" / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / "p01_Heatedtip.csv").write_text("x,y\n0.1,11\n0.2,22\n", encoding="utf-8")
    res = {"series": [{"file": "p01_Heatedtip.csv", "series_name": "Heatedtip"}]}
    R._write_template_output(fmt, res, {"id": "p01", "caption": "cap"},
                             fake_panel, csv_dir, tmp / "out", vlm, print)
    made = tmp / "out" / "templates" / "export_template" / "Heatedtip.dat"
    ok = made.exists() and "# fuel=MeOH" in made.read_text(encoding="utf-8")
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 接进提取：{made.exists()} "
          f"{made.read_text(encoding='utf-8').splitlines()[:3] if made.exists() else ''}")

    srv.shutdown()
    print("通过" if not bad else f"{bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
