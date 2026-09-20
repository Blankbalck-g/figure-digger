"""回归：模板输出。

三条要求：①模板只让模型读一次，同样的模板再来必须复用缓存（不再调模型）
②生成的代码能把曲线渲染成模板文件（参数由模型给值、代码填空，读不到留空）
③不给模板时沿用上一次的；从来没有过就什么都不做（调用方只出 CSV）。

不需要 key、不联网。用法：python tests/test_template_out.py
"""

import json
import re
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
    ok = set(filled) == {"fuel", "pressure"} and all(v for v in filled.values())
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 参数：mock 按问到的键回值 -> {filled}")

    # 证据包必须给图号、条件表、字段检索各自配额，不能让当前页重复文字挤掉前面的表。
    fake_pdf = tmp / "paper.pdf"
    fake_pdf.write_bytes(b"stub")
    read_pages = tpl._paper_pages
    tpl._paper_pages = lambda _path: (
        "TABLE 2 Experimental conditions. Injection pressure was set to 35 MPa. "
        "The injector has a 100 um hole diameter.",
        "Figure 10 shows the cold-start early injection case at 273 K and 0.1 MPa.")
    packet = tpl.paper_param_context(
        fake_pdf, page=2, caption="Figure 10 cold-start early injection",
        keys=["injection_pressure", "nozzle_radius", "ambient_temperature"])
    tpl._paper_pages = read_pages
    ok = all(x in packet for x in ("当前图直接证据", "实验条件表", "字段专项检索",
                                    "35 MPa", "100 um", "273 K"))
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 参数证据分栏：{len(packet)} 字符")

    # 双栏论文的默认 reading order 会把右栏文字插进左栏句子；参数证据读取必须
    # 同时保留左右栏顺序，才能恢复完整的 "injection pressure was set"。
    import pdfplumber  # noqa: E402

    class FakeCrop:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakePage:
        width, height = 600, 800

        def extract_text(self):
            return ("The injection motion blur and camera exposure pressure was set "
                    "to 35 MPa using a pump.")

        def crop(self, box):
            if box[0] == 0:
                return FakeCrop(("Experimental apparatus and operating conditions. " * 3)
                                + "The injection\npressure was set to 35 MPa using a pump.")
            return FakeCrop("Motion blur was reduced by synchronized illumination. " * 3)

    class FakeDoc:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    real_pdf_open = pdfplumber.open
    try:
        pdfplumber.open = lambda _path: FakeDoc()
        tpl._paper_pages.cache_clear()
        page_text = tpl._paper_pages(str(tmp / "two-column.pdf"))[0]
    finally:
        pdfplumber.open = real_pdf_open
        tpl._paper_pages.cache_clear()
    ok = "The injection pressure was set to 35 MPa" in page_text
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 双栏阅读序恢复实验句子")

    class RetryVLM:
        def __init__(self):
            self.calls = 0

        def ask_json(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ({"params": {"global": {"ambient_temperature": "273"}},
                         "evidence": {"global": {"ambient_temperature": "p2: 273 K"}}}, "")
            return ({"params": {"global": {"injection_pressure": "35",
                                               "nozzle_radius": "0.005"}},
                     "basis": {"global": {"injection_pressure": "explicit",
                                             "nozzle_radius": "derived"}}}, "")

    retry_vlm = RetryVLM()
    retry_vals, retry_meta = tpl.ask_params(
        retry_vlm, fake_panel, "Figure 10", [series[0]],
        ["ambient_temperature", "injection_pressure", "nozzle_radius"],
        evidence=packet, return_meta=True)
    got = retry_vals["Heatedtip"]
    ok = retry_vlm.calls == 2 and got == {
        "ambient_temperature": "273", "injection_pressure": "35", "nozzle_radius": "0.005"}
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 缺项定向复核：calls={retry_meta['calls']} {got}")

    class KnowledgeVLM:
        def __init__(self):
            self.calls = 0

        def ask_json(self, _image, prompt, **_kwargs):
            self.calls += 1
            if "论文级实验工况知识库" in prompt:
                return ({
                    "facts": [
                        {"id": "f_tamb", "parameter": "ambient_temperature",
                         "raw_value": "273", "raw_unit": "K", "relation": "direct",
                         "evidence": "p4 cold-start condition: 273 K"},
                        {"id": "f_pamb", "parameter": "ambient_pressure",
                         "raw_value": "0.1", "raw_unit": "MPa", "relation": "direct",
                         "evidence": "p4 early injection: 0.1 MPa"},
                        {"id": "f_pinj", "parameter": "injection_pressure",
                         "raw_value": "350", "raw_unit": "bar", "relation": "direct",
                         "evidence": "p4 injection pressure 350 bar"},
                        {"id": "f_nozzle", "parameter": "nozzle_radius",
                         "raw_value": "100", "raw_unit": "um",
                         "relation": "diameter_to_radius",
                         "evidence": "p3 hole diameter 100 um"},
                        {"id": "f_hot", "parameter": "fuel_temperature",
                         "raw_value": "90", "raw_unit": "C", "relation": "direct",
                         "evidence": "p4 heated tip 90 C"},
                        {"id": "f_normal", "parameter": "fuel_temperature",
                         "raw_value": "", "raw_unit": "", "relation": "same_as",
                         "source_parameter": "ambient_temperature",
                         "evidence": "p4 normal tip same as ambient"},
                    ],
                    "conditions": [
                        {"id": "cold", "labels": ["cold-start"],
                         "fact_ids": ["f_tamb"]},
                        {"id": "early", "labels": ["early injection"],
                         "fact_ids": ["f_pamb"]},
                        {"id": "heated", "labels": ["heated tip"],
                         "fact_ids": ["f_hot"]},
                        {"id": "normal", "labels": ["normal tip"],
                         "fact_ids": ["f_normal"]},
                    ],
                    "global_fact_ids": ["f_pinj", "f_nozzle"], "notes": []}, "")
            return ({
                "status": "bound", "global_condition_ids": ["cold", "early"],
                "global_fact_ids": [],
                "global_parameter_fact_ids": {
                    "ambient_temperature": ["f_tamb"],
                    "ambient_pressure": ["f_pamb"],
                    "injection_pressure": ["f_pinj"],
                    "nozzle_radius": ["f_nozzle"],
                },
                "series": {
                    "Heatedtip": {"condition_ids": ["heated"], "fact_ids": [],
                                   "parameter_fact_ids": {"fuel_temperature": ["f_hot"]}},
                    "Normaltip": {"condition_ids": ["normal"], "fact_ids": [],
                                   "parameter_fact_ids": {"fuel_temperature": ["f_normal"]}},
                },
                "evidence": "Figure 10 cold-start early injection", "unresolved": []}, "")

    knowledge_vlm = KnowledgeVLM()
    knowledge_state = {}
    knowledge_keys = ["ambient_temperature", "ambient_pressure", "injection_pressure",
                      "nozzle_radius", "fuel_temperature"]
    knowledge_hints = {
        "ambient_temperature": "ambient_temperature (K)",
        "ambient_pressure": "ambient_pressure (dyn/cm2)",
        "injection_pressure": "injection_pressure (MPa)",
        "nozzle_radius": "nozzle_radius (cm)",
        "fuel_temperature": "fuel_temperature (K)",
    }
    kb_path = tmp / "condition_knowledge.json"
    knowledge_vals, knowledge_meta = tpl.ask_params_with_knowledge(
        knowledge_vlm, fake_panel, "Figure 10 cold-start early injection", series,
        knowledge_keys, hints=knowledge_hints, evidence=packet, paper_path=fake_pdf,
        knowledge_cache=kb_path, state=knowledge_state, return_meta=True)
    hot, normal = knowledge_vals["Heatedtip"], knowledge_vals["Normaltip"]
    ok = (knowledge_vlm.calls == 2 and knowledge_meta["mode"] == "paper_knowledge"
          and kb_path.exists()
          and hot == {"ambient_temperature": "273", "ambient_pressure": "1000000",
                      "injection_pressure": "35", "nozzle_radius": "0.005",
                      "fuel_temperature": "363.15"}
          and normal == {"ambient_temperature": "273", "ambient_pressure": "1000000",
                         "injection_pressure": "35", "nozzle_radius": "0.005",
                         "fuel_temperature": "273"})
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 知识库→图级绑定→代码换算：{hot} / {normal}")

    _again_vals, again_meta = tpl.ask_params_with_knowledge(
        knowledge_vlm, fake_panel, "Figure 10 cold-start early injection", series,
        knowledge_keys, hints=knowledge_hints, evidence=packet, paper_path=fake_pdf,
        knowledge_cache=kb_path, state=knowledge_state, return_meta=True)
    ok = knowledge_vlm.calls == 3 and again_meta["calls"] == 1 \
        and again_meta["knowledge_source"] == "memory"
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 同论文知识库复用：总调用={knowledge_vlm.calls} "
          f"本面板调用={again_meta['calls']}")

    # 有些旧论文的 PDF 文本层没有保留图内标题。第一张图可以把明写在图上的
    # 原始事实受控写回知识库；后续图只绑定稳定 ID，不应重复造事实。
    class PanelFactsVLM:
        def __init__(self):
            self.calls = 0

        def ask_json(self, _image, prompt, **_kwargs):
            self.calls += 1
            if "论文级实验工况知识库" in prompt:
                return ({
                    "facts": [], "conditions": [], "global_fact_ids": [],
                    "notes": ["PDF text layer contains no operating-condition values"]}, "")
            stable_ids = dict(re.findall(
                r'"id":"(pf_[0-9a-f_]+)"[^{}]*?"parameter":"([^"]+)"', prompt))
            by_parameter = {parameter: fact_id for fact_id, parameter in stable_ids.items()}
            if all(key in by_parameter for key in (
                    "ambient_density", "injection_pressure", "nozzle_radius")):
                selected = {key: [by_parameter[key]] for key in by_parameter}
                return ({"status": "bound", "global_condition_ids": [],
                         "global_fact_ids": [], "global_parameter_fact_ids": selected,
                         "panel_facts": [],
                         "series": {"spray": {"condition_ids": [], "fact_ids": [],
                                               "parameter_fact_ids": {}}},
                         "evidence": "same figure family", "unresolved": []}, "")
            return ({
                "status": "bound", "global_condition_ids": [], "global_fact_ids": [],
                "global_parameter_fact_ids": {
                    "ambient_density": ["pf1"], "injection_pressure": ["pf2"],
                    "nozzle_radius": ["pf3"]},
                "panel_facts": [
                    {"id": "pf1", "parameter": "ambient_density", "raw_value": "21",
                     "raw_unit": "g/L", "relation": "direct",
                     "evidence": "figure title: 21 g/L CHAMBER DENSITY", "confidence": 0.99},
                    {"id": "pf2", "parameter": "injection_pressure", "raw_value": "700",
                     "raw_unit": "bar", "relation": "direct",
                     "evidence": "figure title: 700 BAR RAIL PRESSURE", "confidence": 0.99},
                    {"id": "pf3", "parameter": "nozzle_radius", "raw_value": "0.18",
                     "raw_unit": "mm", "relation": "diameter_to_radius",
                     "evidence": "figure title: 5x0.18x150 TIP", "confidence": 0.99}],
                "series": {"spray": {"condition_ids": [], "fact_ids": [],
                                      "parameter_fact_ids": {}}},
                "evidence": "values printed in figure title", "unresolved": []}, "")

    panel_vlm = PanelFactsVLM()
    panel_state, panel_cache = {}, tmp / "panel_condition_knowledge.json"
    panel_series = [{"name": "spray", "x": [0.1], "y": [1], "params": {}}]
    panel_keys = ["ambient_density", "injection_pressure", "nozzle_radius"]
    panel_hints = {
        "ambient_density": "ambient_density (g/cm3)",
        "injection_pressure": "injection_pressure (MPa)",
        "nozzle_radius": "nozzle_radius (cm)",
    }
    panel_vals, panel_meta = tpl.ask_params_with_knowledge(
        panel_vlm, fake_panel, "Figures 9-11", panel_series, panel_keys,
        hints=panel_hints, evidence="figure title is visible", paper_path=fake_pdf,
        knowledge_cache=panel_cache, state=panel_state, return_meta=True)
    saved = json.loads(panel_cache.read_text(encoding="utf-8"))["knowledge"]["facts"]
    expected = {"ambient_density": "0.021", "injection_pressure": "70",
                "nozzle_radius": "0.009"}
    ok = (panel_vals["spray"] == expected and len(panel_meta["panel_facts_added"]) == 3
          and len(saved) == 3 and all(fid.startswith("pf_")
                                     for fid in panel_meta["panel_facts_added"]))
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 图内事实入库+换算：{panel_vals['spray']} 新增="
          f"{panel_meta['panel_facts_added']}")

    panel_vals2, panel_meta2 = tpl.ask_params_with_knowledge(
        panel_vlm, fake_panel, "Figures 9-11", panel_series, panel_keys,
        hints=panel_hints, evidence="same figure family", paper_path=fake_pdf,
        knowledge_cache=panel_cache, state={}, return_meta=True)
    saved2 = json.loads(panel_cache.read_text(encoding="utf-8"))["knowledge"]["facts"]
    ok = (panel_vals2["spray"] == expected and panel_meta2["calls"] == 1
          and panel_meta2["knowledge_source"] == "disk"
          and panel_meta2["panel_facts_added"] == [] and len(saved2) == 3)
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 后续图复用入库事实：调用={panel_meta2['calls']} "
          f"事实数={len(saved2)}")

    class ExpandedKnowledgeVLM:
        def __init__(self):
            self.calls = 0

        def ask_json(self, *_args, **_kwargs):
            self.calls += 1
            return ({"facts": [{"id": "f_temp", "parameter": "ambient_temperature",
                                 "raw_value": "300", "raw_unit": "K",
                                 "relation": "direct", "evidence": "p2 chamber at 300 K"}],
                     "conditions": [], "global_fact_ids": ["f_temp"], "notes": []}, "")

    expanded_vlm = ExpandedKnowledgeVLM()
    expanded, expanded_meta = tpl.ckb.build_knowledge(
        expanded_vlm, fake_pdf, panel_keys + ["ambient_temperature"],
        "- ambient_temperature: chamber temperature", "paper evidence",
        cache_path=panel_cache, state={})
    expanded_saved = json.loads(panel_cache.read_text(encoding="utf-8"))
    expanded_ids = {fact["id"] for fact in expanded["facts"]}
    ok = (expanded_vlm.calls == 1 and expanded_meta["source"] == "model"
          and set(panel_meta["panel_facts_added"]).issubset(expanded_ids)
          and len(expanded["facts"]) == 4
          and set(expanded_saved["keys"]) == set(panel_keys + ["ambient_temperature"]))
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 扩展知识库保留图内事实："
          f"facts={len(expanded['facts'])} keys={len(expanded_saved['keys'])}")

    # 多组候选值、范围或“未报告，但物性在 298/303 K 测量”不是标量。
    # 代码必须拒绝它们，绝不能默认取第一个数字。
    ambiguous_knowledge = tpl.ckb.normalize_knowledge({
        "facts": [
            {"id": "a1", "parameter": "ambient_temperature", "raw_value": "298",
             "raw_unit": "K", "relation": "direct", "evidence": "chamber at 298 K"},
            {"id": "a2", "parameter": "injection_pressure",
             "raw_value": "400, 800 and 1600", "raw_unit": "bar",
             "relation": "direct", "evidence": "three candidate conditions"},
            {"id": "a3", "parameter": "ambient_pressure",
             "raw_value": "2.5-20", "raw_unit": "bar",
             "relation": "direct", "evidence": "reported test range"},
            {"id": "a4", "parameter": "fuel_temperature",
             "raw_value": "not reported; density at 298 K and viscosity at 303 K",
             "raw_unit": "K", "relation": "direct", "evidence": "property methods"},
        ],
        "conditions": [], "global_fact_ids": ["a1", "a2", "a3", "a4"]},
        ["ambient_temperature", "injection_pressure", "ambient_pressure",
         "fuel_temperature"])
    ambiguous_binding = tpl.ckb.normalize_binding(
        {"status": "bound", "global_fact_ids": ["a1", "a2", "a3", "a4"],
         "series": {"spray": {"fact_ids": ["a1", "a2", "a3", "a4"]}}},
        ambiguous_knowledge, panel_series)
    ambiguous_vals, _ambiguous_meta = tpl.ckb.resolve_params(
        ambiguous_knowledge, ambiguous_binding, panel_series,
        ["ambient_temperature", "injection_pressure", "ambient_pressure",
         "fuel_temperature"], hints={"ambient_temperature": "(K)",
                                    "injection_pressure": "(MPa)",
                                    "ambient_pressure": "(dyn/cm2)",
                                    "fuel_temperature": "(K)"})
    ok = ambiguous_vals["spray"] == {"ambient_temperature": "298"}
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 多候选数值拒绝默认取首值："
          f"{ambiguous_vals['spray']}")
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
    ok = made.exists() and "# fuel=" in made.read_text(encoding="utf-8")
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 接进提取：{made.exists()} "
          f"{made.read_text(encoding='utf-8').splitlines()[:3] if made.exists() else ''}")

    # 模型一个参数都没读到（用户那份模板踩过的坑）：仍然要出文件，不能整份丢掉
    tpl_ask = tpl.ask_params
    tpl.ask_params = lambda *a, **k: {}
    R._write_template_output(fmt, res, {"id": "p01", "caption": "cap"},
                             fake_panel, csv_dir, tmp / "out2", vlm, print)
    tpl.ask_params = tpl_ask
    made2 = list((tmp / "out2" / "templates").glob("*/*"))
    ok = bool(made2)
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'FAIL'}] 参数一个都没读到也要出文件：{bool(made2)}")

    srv.shutdown()
    print("通过" if not bad else f"{bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
