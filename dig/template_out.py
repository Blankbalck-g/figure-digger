"""把提取结果按用户给的模板文件输出。

模板只让模型读**一次**：它读模板结构 + 下面这份数据契约，写出一段最简代码；代码按
模板内容的 sha1 缓存，同一个模板再来就直接复用（不花 token）。不指定模板时用上一次
的（cache/last.txt），没有就返回 None —— 调用方照旧只出 CSV。

契约（生成代码必须定义 render）：
    def render(series, context) -> dict[str, str]
      series  = [{"name": 曲线名, "x": [...], "y": [...], "params": {参数名: 值或""}}]
      context = {"paper": 论文名, "panel": 面板 id, "ext": 模板扩展名, "template": 模板名}
      返回    = {输出文件名: 文件内容}

模板里需要按曲线标注参数时，模型只负责**读**出参数值（图例/标题/标注/图注），填空由
代码做；读不到的留空字符串。
"""

import builtins
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

try:
    import condition_kb as ckb
except ImportError:  # package-style imports used by diagnostics/tests outside run.py
    from . import condition_kb as ckb

CACHE_DIR = Path(__file__).resolve().parent.parent / ".templates"
SAFE_IMPORTS = ("json", "math", "re", "csv", "io", "datetime", "statistics", "textwrap")

TEMPLATE_SYSTEM = "你是数据格式工程师，只输出 json。"

TEMPLATE_PROMPT = """把一个数据模板编译成渲染代码，只输出 json：
{{"ext": ".dat", "params": ["参数名"], "code": "def render(series, context):\\n ..."}}
数据契约：
- series: 列表，每项 {{"name": 曲线名, "x": [数值...], "y": [数值...], "params": {{参数名: 值或""}}}}
- context: {{"paper": 论文名, "panel": 面板号, "ext": 扩展名, "template": 模板名}}
- code 必须定义 render(series, context)，返回 {{输出文件名: 文件内容}}
要求：
- 代码最短、**不要注释、不要 print、不要多余 import**（标准库够用）
- 模板里的固定文字、分隔符、换行原样保留
- **params 必须列出模板里所有"要填值"的字段名**（不只是每条曲线的：工况、压力、温度、
  燃料、文件名、时间…凡是模型要读出来填进去的都算）。列不全，渲染时会 KeyError
- 代码里用 s['params']['字段名'] 取值；读不到的字段由调用方填空/0
- 模板里的重复段按模板出现的形式渲染每条曲线
- 只输出 json，不要解释
模板内容：
---
{template}
---"""

PARAMS_SYSTEM = "你是科研论文实验条件分析员。依据图、图注和正文证据做语义归属，只输出 json。"

PARAMS_PROMPT = """目标图是「{caption}」，图里的曲线有：{names}。
模板要求每条曲线标注这些参数：{keys}。
字段语义（用于区分环境/燃料/喷嘴等同名量）：
{field_guide}
下面是代码从论文中组装的分栏证据包。它可能包含其它工况，必须利用图号、图注、图内标题、
系列名以及“early/late、cold/room、heated/normal”等语义做归属，不能看到数字就照抄：
---
{evidence}
---
请像实验条件分析员一样完成：
1. 先判断整张图对应哪一个实验工况；2. 再区分全图共享参数与随系列变化参数；
3. 优先使用“当前图直接证据/实验条件表/装置与喷嘴”证据，普通背景描述只能辅助；
4. 若字段可由明确证据唯一换算或推导，可填写并在 basis 标为 derived，例如直径→半径、
   bar→MPa/dyn·cm⁻²、kg/m³→g/cm³；理想气体密度仅在气体种类、压力、温度都明确时推导；
5. 同一物理工况适用于全部曲线时放在 global，系列名本身表示温度/燃料/加热状态时逐系列覆盖。
只输出 json：
{{"params": {{"global": {{"参数名": "目标单位数值"}},
              "曲线名": {{"参数名": "目标单位数值"}}}},
  "evidence": {{"global": {{"参数名": "页码 + 支持该值的原文短句"}},
                 "曲线名": {{"参数名": "页码 + 支持该值的原文短句"}}}},
  "basis": {{"global": {{"参数名": "explicit 或 derived"}},
              "曲线名": {{"参数名": "explicit 或 derived"}}}}}}
要求：
- 参数名后面若附了"模板里的写法"（通常带单位），请按**模板要的单位**给数值
- 图上标的是整张图的工况（不是某条曲线的），每条曲线都填同一个值
- 参数来自直径而模板要半径时除以 2；密度、压力、温度必须辨认对象和单位后再换算
- 论文列出一组可选条件时，必须由当前图或正文对该图的描述选中其中一个，不能任选
- 读不到就填空字符串 ""，**不要猜、不要编造**"""

PARAMS_RETRY_PROMPT = """上一次已经确定的参数如下：
{known}
仍缺少：{missing}。

目标图：「{caption}」；曲线：{names}。
字段语义：
{field_guide}
论文证据包：
---
{evidence}
---
请只复核缺失字段。重点检查实验条件表、装置/喷嘴段落、正文中直接解释该图的句子，以及系列名
表达的加热/温度/燃料差异。允许有明确前提的单位换算或物理推导，但必须给证据和 explicit/derived。
不要修改上一次已有值。输出与上一轮相同结构的 json；仍无法唯一确定就留空。"""


PARAM_ALIASES = {
    "ambient_density": ["ambient density", "gas density", "charge gas density",
                        "core density", "rho amb", "ρamb"],
    "ambient_temperature": ["ambient temperature", "gas temperature", "core temperature",
                            "charge gas temperature", "tamb", "t amb"],
    "fuel_temperature": ["fuel temperature", "fuel rail temperature", "injector tip temperature",
                          "heated tip temperature", "tip temperature", "fuel was heated"],
    "injection_pressure": ["injection pressure", "rail pressure", "pinj", "p inj"],
    "nozzle_radius": ["nozzle radius", "nozzle diameter", "hole diameter", "orifice diameter",
                      "nozzle hole", "injector has a"],
    "discharge_coefficient": ["discharge coefficient", "nozzle cd", "coefficient cd"],
    "fuel_mass": ["fuel mass", "injected mass", "injection quantity", "mass per injection",
                  "total mass injected"],
    "ambient_pressure": ["ambient pressure", "chamber pressure", "back pressure", "pamb", "p amb"],
}

PARAM_MEANINGS = {
    "ambient_density": "环境/定容室气体密度，不是燃料液体密度",
    "ambient_temperature": "环境/定容室气体温度，不是燃料或喷嘴温度",
    "fuel_temperature": "喷射前燃料温度；仅有喷嘴尖端温度时要明确二者是否等同",
    "injection_pressure": "燃油喷射/轨压，模板单位通常为 MPa",
    "nozzle_radius": "喷孔半径；论文常给直径，半径=直径/2，模板单位 cm",
    "discharge_coefficient": "喷孔流量系数 Cd，无量纲；模型公式中的其它系数不能代替",
    "fuel_mass": "当前工况单次喷射燃料质量，不是累计燃烧质量或公式变量",
    "ambient_pressure": "环境/背压/定容室压力，不是喷射压力；模板单位 dyn/cm2",
}


class Format:
    """一份模板的实现（代码 + 参数名 + 扩展名）。"""

    def __init__(self, key, name, data):
        self.key, self.name = key, name
        self.ext = str(data.get("ext") or ".txt")
        self.params = [str(k) for k in (data.get("params") or [])]
        self.code = str(data.get("code") or "")
        self.text = str(data.get("text") or "")
        self.render = _compile(self.code)


def probe_params(fmt, series, context):
    """空跑一遍 render，返回代码实际用到的参数键（模板声明的 params 可能不全）。

    实测：模板里明明有 `Prediction Input:` 那 8 个字段，模型给的 params 却是空的
    （它以为只要列"按曲线"的参数），于是渲染 `KeyError` 直接跳过、用户什么都拿不到。
    这里先让它要什么自己记下来，再去问模型要值。
    """
    missing = []

    class Probe(dict):
        def __missing__(self, key):
            missing.append(str(key))
            return 0.0

    probe = [{"name": s["name"], "x": s.get("x") or [], "y": s.get("y") or [],
              "params": Probe()} for s in series]
    try:
        fmt.render(probe, context)
    except Exception:  # noqa: BLE001 - 探针失败不影响：能拿到多少键算多少
        pass
    return list(dict.fromkeys(missing))


def _compile(code):
    """在只给白名单 import 的环境里编译 render()。生成的代码是模型写的，不放开整个
    builtins；同时把源码留在缓存里，坏了可以手工改。"""
    allowed = ("len", "range", "enumerate", "zip", "sorted", "min", "max", "sum", "abs",
               "round", "str", "int", "float", "bool", "list", "dict", "tuple", "set",
               "map", "filter", "repr", "format", "isinstance", "Exception", "ValueError",
               "KeyError", "IndexError")
    safe = {n: getattr(builtins, n) for n in allowed}

    def _import(name, *a, **k):
        if name.split(".")[0] not in SAFE_IMPORTS:
            raise ImportError(f"模板代码不允许 import {name}")
        return __import__(name, *a, **k)

    safe["__import__"] = _import
    ns = {"__builtins__": safe}
    exec(compile(code, "<template>", "exec"), ns)          # noqa: S102 - 用户自己的模板
    fn = ns.get("render")
    if not callable(fn):
        raise ValueError("模板代码没有定义 render()")
    return fn


def load(template=None, vlm=None, cache_dir=None, log=None):
    """-> Format / None。template=None 时用上一次的模板；没有就 None（默认输出）。"""
    cache = Path(cache_dir or CACHE_DIR)
    if template is None:
        last = cache / "last.txt"
        if not last.exists():
            return None
        key = last.read_text(encoding="utf-8").strip()
        path = cache / f"{key}.json"
        if not key or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if log:
            log(f"      模板：沿用上次的 {data.get('template')}（{data.get('ext')}，0 token）")
        return Format(key, str(data.get("template") or key), data)

    text = Path(template).read_text(encoding="utf-8", errors="replace")
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    path = cache / f"{key}.json"
    if path.exists():                                      # 模板没变 -> 不写代码
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("text"):                           # 老缓存没存原文：补上（单位提示要用）
            try:
                data["text"] = Path(template).read_text(encoding="utf-8", errors="replace")
                path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            except OSError:
                pass
        if log:
            log(f"      模板未变，复用已有实现：{Path(template).name}（0 token）")
    else:
        if vlm is None:
            raise RuntimeError("需要 --vlm 才能把模板编译成代码")
        prompt = TEMPLATE_PROMPT.format(template=text[:12000])
        data, _raw = vlm.ask_json(None, prompt, system=TEMPLATE_SYSTEM, max_tokens=2000)
        data["template"] = Path(template).name
        data["text"] = text
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        if log:
            log(f"      模板：模型写出实现 -> {path.name}"
                f"（参数 {len(data.get('params') or [])} 个）")
    (cache / "last.txt").write_text(key, encoding="utf-8")
    return Format(key, Path(template).name, data)


def params_hints(fmt, keys):
    """从模板原文里给每个参数名找一行上下文（带单位），读数值时有用。

    例：模板写着 `ambient_pressure (dyn/cm2) : 0`，图上是 "P_amb = 20 bar"，
    模型看到单位才知道要换算。
    """
    lines = [ln.strip() for ln in (fmt.text or "").splitlines() if ln.strip()]
    out = {}
    for k in keys:
        name = str(k)
        for ln in lines:
            if name in ln or name.replace("_", " ") in ln.lower():
                # 只留"字段名 + 单位"，模板里自带的示例值不要给模型看（免得照抄）
                out[name] = ln.split(":")[0].split("=")[0].strip()[:60]
                break
    return out


@lru_cache(maxsize=12)
def _paper_pages(pdf_path):
    """Read PDF text once per process; several panels from one paper reuse it.

    Keep both pdfplumber's default reading order and a left-column/right-column
    view.  Scientific two-column PDFs often interleave the columns in the default
    text (for example ``The injection [right column] pressure was set ...``), which
    destroys the very phrases used to retrieve experimental conditions.
    """
    import pdfplumber
    with pdfplumber.open(pdf_path) as doc:
        pages = []
        for page in doc.pages:
            views = [page.extract_text() or ""]
            try:
                mid = float(page.width) / 2.0
                left = page.crop((0, 0, mid, page.height)).extract_text() or ""
                right = page.crop((mid, 0, page.width, page.height)).extract_text() or ""
                if len(left.strip()) >= 120 and len(right.strip()) >= 120:
                    views.extend((left, right))
            except Exception:  # noqa: BLE001 - default reading order remains usable
                pass
            pages.append(re.sub(r"\s+", " ", "\n".join(views)).strip())
        return tuple(pages)


def paper_param_context(pdf_path, page=None, caption="", keys=(), max_chars=18000):
    """Build a balanced evidence packet instead of one globally ranked snippet pile.

    Figure-local prose, condition tables and apparatus/nozzle facts get independent
    quotas.  This prevents repeated current-page prose from pushing an earlier table
    containing the actual pressure/temperature/nozzle values out of the token budget.
    """
    if not pdf_path or not Path(pdf_path).exists() or not keys:
        return "（没有可用的论文正文证据）"
    try:
        pages = _paper_pages(str(Path(pdf_path).resolve()))
    except Exception:  # noqa: BLE001 - 参数是附加输出，正文读取失败不影响曲线
        return "（论文正文读取失败，仅依据图片和图注）"

    buckets = {"当前图直接证据": [], "实验条件表": [], "字段专项检索": [], "相邻页补充": []}
    field_buckets = {}
    fig_ids = list(dict.fromkeys(re.findall(
        r"(?:figure|fig\.?)[\s_-]*(\d+[a-z]?)", caption or "", re.I)))
    cap_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", caption or "")}

    def add(bucket, score, pno, text, why, width=1500):
        clean = re.sub(r"\s+", " ", text or "").strip()
        if len(clean) >= 35:
            buckets[bucket].append((score, pno, clean[:width], why))

    def windows(text, term, before=500, after=850, limit=2):
        found = []
        # Short symbolic aliases such as "p inj" must be token-bounded.  Plain
        # substring matching also found the tail of "tip injector", which ranked
        # unrelated current-page prose above an explicit "injection pressure" value.
        left = r"(?<!\w)" if term and term[0].isalnum() else ""
        right = r"(?!\w)" if term and term[-1].isalnum() else ""
        pattern = re.compile(left + re.escape(term) + right, re.I)
        for match in pattern.finditer(text):
            pos = match.start()
            found.append(text[max(0, pos - before):min(len(text), pos + len(term) + after)])
            if len(found) >= limit:
                break
        return found

    # Exact figure references are strongest: captions often omit conditions while the
    # paragraph saying "Figure N shows ... at ..." supplies them.
    for pno, text in enumerate(pages, start=1):
        for fid in fig_ids:
            for spelling in (f"figure {fid}", f"fig. {fid}"):
                for chunk in windows(text, spelling, before=650, after=1100, limit=3):
                    score = 12.0 + (3.0 if page and pno == int(page) else 0.0)
                    add("当前图直接证据", score, pno, chunk, f"正文提到 Figure {fid}")

    if caption:
        add("当前图直接证据", 20.0, int(page or 0), caption, "配置中的图注", width=1800)

    condition_terms = [
        "experimental conditions", "test conditions", "operating conditions",
        "experimental apparatus", "table 1", "table 2", "table 3",
        "injection pressure was set", "ambient pressure was set",
        "ambient temperature", "injector tip temperature", "chamber density",
        "rail pressure", "injector has a", "hole diameter",
    ]
    for pno, text in enumerate(pages, start=1):
        for term in condition_terms:
            for chunk in windows(text, term, before=420, after=1050, limit=2):
                low = chunk.lower()
                score = 6.0 + (2.0 if "table" in term else 0.0)
                score += 0.2 * sum(1 for w in cap_words if w in low)
                if re.search(r"\d\s*(?:bar|mpa|kpa|pa|k|°c|mm|µm|um|kg/m|g/l|g/cm|mg|g)\b",
                             chunk, re.I):
                    score += 2.0
                add("实验条件表", score, pno, chunk, f"命中 {term}")

    # Guarantee every requested field some space in the packet.  Per-key quotas avoid
    # a common word such as temperature consuming all evidence for rarer nozzle fields.
    for key in keys:
        terms = list(dict.fromkeys(PARAM_ALIASES.get(str(key), [])
                                   + [str(key).replace("_", " ")]))
        field_hits = []
        canonical = str(key).replace("_", " ").lower()
        for pno, text in enumerate(pages, start=1):
            for term in terms:
                for chunk in windows(text, term, before=380, after=700, limit=2):
                    score = 5.0 + (2.0 if page and pno == int(page) else 0.0)
                    compact = re.sub(r"\W+", "", term)
                    score += (4.0 if term.lower() == canonical
                              else (2.0 if len(compact) >= 8 else 0.0))
                    low_chunk = chunk.lower()
                    if any(mark in low_chunk for mark in ("was set", "test condition",
                                                           "experimental condition", "apparatus")):
                        score += 2.0
                    if "doi.org" in low_chunk:
                        score -= 3.0
                    if any(ch.isdigit() for ch in chunk):
                        score += 1.0
                    field_hits.append((score, pno, chunk, f"{key}: {term}"))
        field_buckets[str(key)] = sorted(field_hits, reverse=True)[:4]

    if page:
        for pno in range(max(1, int(page) - 1), min(len(pages), int(page) + 1) + 1):
            text = pages[pno - 1]
            # One bounded numeric excerpt from each nearby page is useful when PDF text
            # order separates a table heading from its values.
            positions = [m.start() for m in re.finditer(r"\d", text)]
            if positions:
                pos = positions[len(positions) // 2]
                add("相邻页补充", 2.0 - abs(pno - int(page)), pno,
                    text[max(0, pos - 600):min(len(text), pos + 900)], "当前/相邻页")

    quotas = {"当前图直接证据": 4300, "实验条件表": 5200,
              "字段专项检索": 7000, "相邻页补充": 1800}
    # De-duplicate inside each evidence lane, not across lanes.  A compact paragraph
    # can legitimately be both the condition-table evidence and the only hit for a
    # rare field such as nozzle radius.  Cross-lane de-duplication used to erase the
    # field-specific lane entirely on short papers, defeating the balanced quotas.
    seen = {name: set() for name in buckets}
    sections, total = [], 0
    for bucket in ("当前图直接证据", "实验条件表", "字段专项检索", "相邻页补充"):
        if bucket == "字段专项检索":
            # Give every requested field one protected slot before considering a
            # second hit.  A global score sort let repeated current-page temperature
            # prose consume this whole lane and hide an explicit pressure/nozzle
            # value on an earlier apparatus page.  Short first-pass excerpts keep
            # eight fields within the same token budget.
            rows, used = [], 0
            field_seen = {str(k): set() for k in keys}
            for rank in range(4):
                for key in (str(k) for k in keys):
                    hits = field_buckets.get(key) or []
                    if rank >= len(hits):
                        continue
                    _score, pno, field_text, why = hits[rank]
                    limit = 760 if rank == 0 else 560
                    field_text = re.sub(r"\s+", " ", field_text).strip()[:limit]
                    sig = re.sub(r"\W+", "", field_text.lower())[:220]
                    if not sig or sig in field_seen[key]:
                        continue
                    block = f"[p{pno} · {why}] {field_text}"
                    if used + len(block) > quotas[bucket] or total + len(block) > max_chars:
                        continue
                    field_seen[key].add(sig)
                    rows.append(block)
                    used += len(block) + 2
                    total += len(block) + 2
            if rows:
                sections.append(f"## {bucket}\n" + "\n\n".join(rows))
            continue
        rows, used = [], 0
        for score, pno, text, why in sorted(buckets[bucket], key=lambda x: -x[0]):
            sig = re.sub(r"\W+", "", text.lower())[:220]
            if not sig or sig in seen[bucket]:
                continue
            block = f"[p{pno} · {why}] {text}"
            if used + len(block) > quotas[bucket] or total + len(block) > max_chars:
                continue
            seen[bucket].add(sig)
            rows.append(block)
            used += len(block) + 2
            total += len(block) + 2
        if rows:
            sections.append(f"## {bucket}\n" + "\n\n".join(rows))
    return "\n\n".join(sections) if sections else "（未检索到与这些字段直接相关的正文片段）"


def _merge_param_answer(data, out, meta, keys):
    """Merge one agent answer; tolerate old/simple response shapes."""
    if not isinstance(data, dict):
        return 0
    allowed = {str(k) for k in keys}
    norm_names = {re.sub(r"\W+", "", str(name)).lower(): name for name in out}

    def target_name(name):
        if str(name).lower() == "global":
            return "global"
        return name if name in out else norm_names.get(re.sub(r"\W+", "", str(name)).lower())

    added = 0
    vals = data.get("params") or {}
    if isinstance(vals, dict):
        for raw_name, kv in vals.items():
            target = target_name(raw_name)
            if not isinstance(kv, dict) or target is None:
                continue
            targets = list(out) if target == "global" else [target]
            for key, value in kv.items():
                key = str(key)
                if key not in allowed or value in (None, ""):
                    continue
                for name in targets:
                    if not out[name].get(key):
                        out[name][key] = str(value).strip()
                        added += 1

    for field in ("evidence", "basis"):
        src = data.get(field) or {}
        if not isinstance(src, dict):
            continue
        for raw_name, kv in src.items():
            target = target_name(raw_name)
            if not isinstance(kv, dict) or target is None:
                continue
            targets = list(out) if target == "global" else [target]
            for key, value in kv.items():
                key = str(key)
                if key not in allowed or value in (None, ""):
                    continue
                for name in targets:
                    meta[field].setdefault(name, {})[key] = str(value).strip()
    return added


def ask_params(vlm, image_path, caption, series, keys, hints=None, evidence="",
               return_meta=False):
    """Extract template parameters with one primary call and one conditional repair call."""
    out = {str(s["name"]): {} for s in series}
    meta = {"evidence": {name: {} for name in out},
            "basis": {name: {} for name in out}, "calls": 0}
    if not keys or vlm is None:
        return (out, meta) if return_meta else out
    names = "、".join(out)
    keys_text = "、".join(
        (f"{k}（模板里的写法：{hints[k]}）" if hints and k in hints else str(k))
        for k in keys)
    guide = "\n".join(f"- {k}: {PARAM_MEANINGS.get(str(k), '按字段名通常含义判断')}"
                      for k in keys)
    packet = (evidence or "（没有额外正文证据）")[:18000]
    prompt = PARAMS_PROMPT.format(caption=(caption or "（无图注）")[:1400], names=names,
                                  keys=keys_text, field_guide=guide, evidence=packet)
    try:
        data, _raw = vlm.ask_json(image_path, prompt, system=PARAMS_SYSTEM, max_tokens=1800)
        meta["calls"] += 1
        _merge_param_answer(data, out, meta, keys)
    except Exception:  # noqa: BLE001 - 参数读不到不影响数据本身
        pass

    missing = {name: [str(k) for k in keys if not out[name].get(str(k))] for name in out}
    missing_n = sum(len(v) for v in missing.values())
    total = max(1, len(out) * len(keys))
    # One focused retry only when the first pass left a material gap.  This spends
    # tokens on the actual missing fields instead of re-reading all eight fields again.
    if missing_n >= max(2, total // 3) and packet and packet[0] != "（":
        retry = PARAMS_RETRY_PROMPT.format(
            known=json.dumps(out, ensure_ascii=False, separators=(",", ":")),
            missing=json.dumps(missing, ensure_ascii=False, separators=(",", ":")),
            caption=(caption or "（无图注）")[:1400], names=names, field_guide=guide,
            evidence=packet)
        try:
            data, _raw = vlm.ask_json(image_path, retry, system=PARAMS_SYSTEM, max_tokens=1400)
            meta["calls"] += 1
            _merge_param_answer(data, out, meta, keys)
        except Exception:  # noqa: BLE001
            pass
    return (out, meta) if return_meta else out


def ask_params_with_knowledge(vlm, image_path, caption, series, keys, hints=None,
                              evidence="", paper_path=None, knowledge_cache=None,
                              state=None, return_meta=False):
    """Three-stage parameter agent: paper knowledge -> panel binding -> code conversion.

    The legacy direct reader remains a failure fallback so a malformed knowledge reply
    cannot suppress template output.  A valid ambiguous binding is *not* a failure and
    deliberately remains blank.
    """
    if not keys or vlm is None or not paper_path or not Path(paper_path).exists():
        return ask_params(vlm, image_path, caption, series, keys, hints=hints,
                          evidence=evidence, return_meta=return_meta)
    state = state if state is not None else {}
    requested = list(dict.fromkeys(
        [str(k) for k in (state.get("knowledge_keys") or [])]
        + [str(k) for k in keys]))
    all_hints = dict(state.get("knowledge_hints") or {})
    all_hints.update(hints or {})
    state["knowledge_hints"] = all_hints
    guide = "\n".join(
        f"- {key}: {PARAM_MEANINGS.get(key, '按字段名和模板上下文判断')}"
        + (f"；模板目标写法：{all_hints[key]}" if key in all_hints else "")
        for key in requested)
    try:
        paper_evidence = paper_param_context(
            paper_path, keys=requested, max_chars=22000)
        knowledge, knowledge_meta = ckb.build_knowledge(
            vlm, paper_path, requested, guide, paper_evidence,
            cache_path=knowledge_cache, state=state)
        binding, binding_meta = ckb.bind_panel(
            vlm, image_path, caption, series, keys, knowledge, evidence)
        if not binding.get("valid"):
            raise ValueError("图级工况绑定返回格式无效")
        added_facts = list(binding_meta.get("new_facts") or [])
        if added_facts:
            # bind_panel mutates the shared in-memory knowledge object.  Persist that
            # exact object so later panels and later runs can bind to the stable IDs
            # without asking the paper-level model to rediscover figure-only facts.
            cache_keys = list(dict.fromkeys(
                [str(k) for k in (state.get("knowledge_keys") or [])] + requested))
            ckb.save_knowledge(knowledge_cache, paper_path, cache_keys, knowledge)
        values, resolved = ckb.resolve_params(
            knowledge, binding, series, keys, hints=hints or {})
        meta = {
            "evidence": resolved.get("evidence") or {},
            "basis": resolved.get("basis") or {},
            "fact_ids": resolved.get("fact_ids") or {},
            "raw": resolved.get("raw") or {},
            "conflicts": resolved.get("conflicts") or {},
            "calls": int(knowledge_meta.get("calls") or 0)
            + int(binding_meta.get("calls") or 0),
            "mode": "paper_knowledge",
            "knowledge_source": knowledge_meta.get("source"),
            "panel_facts_added": added_facts,
            "binding": {key: binding.get(key) for key in (
                "status", "global_condition_ids", "global_fact_ids",
                "global_parameter_fact_ids", "series", "evidence", "unresolved")},
        }
        return (values, meta) if return_meta else values
    except Exception as exc:  # noqa: BLE001 - legacy path keeps templates available
        values, meta = ask_params(
            vlm, image_path, caption, series, keys, hints=hints,
            evidence=evidence, return_meta=True)
        meta["mode"] = "legacy_fallback"
        meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return (values, meta) if return_meta else values
