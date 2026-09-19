"""回归：--want 的轴标题核对（命中要过标题这一关，不靠"模型觉得像"）。

规则：模型在轴读数那一步顺带回答 matches_request——
  true  -> 保留（报告里写"✅ 符合"）
  false -> 标 skip 并写明理由（这张图不提取）
  null  -> 看不清，保留但标注（宁可让人复核，也不静默丢数据）

不需要 key、不联网。用法：python tests/test_axis_check.py
"""

import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dig"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as R  # noqa: E402
import vlm_client as vc  # noqa: E402
import vlm_tasks as vt  # noqa: E402
from test_vlm_mock import Handler  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

VA_TIME = {"x": {"title": "Time after start of injection", "unit": "ms"},
           "y_axes": [{"side": "left", "title": "Spray tip penetration", "unit": "mm"}]}


def main():
    bad = 0
    ok = R._axis_hint(None) is None and "贯穿距" in R._axis_hint("贯穿距随时间")
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 需求文本进入轴读数提示：{R._axis_hint('贯穿距随时间')!r}")

    for match, want_skip, label in ((True, False, "符合"), (False, True, "不符合"),
                                    (None, False, "看不清")):
        entry = {"axis": {}}
        log = []
        R._record_axis_check(entry, {**VA_TIME, "matches_request": match,
                                     "match_note": f"mock-{label}"},
                             "喷雾贯穿距与时间的关系图象", log.append)
        got_skip = bool(entry.get("skip"))
        ok = got_skip == want_skip and entry["axis"]["x_title"].startswith("Time")
        bad += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] matches={match}（{label}）-> "
              f"skip={got_skip}（期望 {want_skip}）"
              + (f" 理由={entry.get('skip_reason')}" if got_skip else ""))

    tmp = Path(tempfile.mkdtemp(prefix="dig_axischk_"))
    vc.CACHE_DIR = tmp / ".c"
    vc.LOG_PATH = tmp / "l.jsonl"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    vlm = vc.DeepSeekVLM(api_key="sk-mock",
                         base_url=f"http://127.0.0.1:{srv.server_address[1]}")
    va = vt.read_axis_ranges(vlm, None, hint=R._axis_hint("喷雾贯穿距与时间的关系图象"))
    srv.shutdown()
    ok = (va.get("x") or {}).get("title") and va.get("matches_request") is True
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 轴读数返回值带标题与判定："
          f"x.title={((va.get('x') or {}).get('title'))!r} "
          f"matches={va.get('matches_request')}")

    prompt = vt.AXIS_PROMPT.format(hint=R._axis_hint("贯穿距随时间"))
    ok = "matches_request" in prompt and "贯穿距随时间" in prompt
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 提示词含核对要求与需求原文")

    print("通过" if not bad else f"{bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
