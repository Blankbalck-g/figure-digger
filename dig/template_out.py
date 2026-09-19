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
from pathlib import Path

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

PARAMS_SYSTEM = "你是科研论文图表的读数助手，只输出 json。"

PARAMS_PROMPT = """这张图是「{caption}」，图里的曲线有：{names}。
模板要求每条曲线标注这些参数：{keys}。
请从图上的图例、标题、工况标注（必要时参考文件名/图注）读出每条曲线对应的参数值，只输出 json：
{{"params": {{"曲线名": {{"参数名": "值"}}}}}}
要求：
- 参数名后面若附了"模板里的写法"（通常带单位），请按**模板要的单位**给数值，
  必要时换算（例如 1 bar = 1e6 dyn/cm^2，298 K 就写 298）
- 图上标的是整张图的工况（不是某条曲线的），每条曲线都填同一个值
- 读不到就填空字符串 ""，**不要猜、不要编造**"""


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


def ask_params(vlm, image_path, caption, series, keys, hints=None):
    """让模型读每条曲线的参数值。读不到填空字符串；没有 VLM 就全空。"""
    out = {s["name"]: {} for s in series}
    if not keys or vlm is None:
        return out
    names = "、".join(str(s["name"]) for s in series)
    keys_text = "、".join(
        (f"{k}（模板里的写法：{hints[k]}）" if hints and k in hints else str(k))
        for k in keys)
    prompt = PARAMS_PROMPT.format(caption=(caption or "（无图注）")[:200], names=names,
                                  keys=keys_text)
    try:
        data, _raw = vlm.ask_json(image_path, prompt, system=PARAMS_SYSTEM, max_tokens=600)
    except Exception:  # noqa: BLE001 - 参数读不到不影响数据本身
        return out
    vals = data.get("params") or {}
    for name, kv in vals.items():
        if name in out and isinstance(kv, dict):
            out[name] = {str(k): str(v) for k, v in kv.items() if v not in (None, "")}
    return out
