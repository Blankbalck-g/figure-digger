"""Paper-level experiment knowledge, panel binding, and deterministic conversion.

The model extracts raw facts once per paper and later binds a panel to fact/condition
IDs.  It never performs the final unit conversion: code owns arithmetic and rejects
conflicting facts instead of guessing.
"""

import hashlib
import json
import re
from pathlib import Path


SCHEMA_VERSION = 1

KNOWLEDGE_SYSTEM = (
    "你是科研论文实验条件知识库构建器。只抽取有原文证据的原始事实，不做目标单位换算，只输出 json。"
)

KNOWLEDGE_PROMPT = """请把下面论文证据整理成可复用的论文级实验工况知识库。
目标参数及语义：
{field_guide}

论文证据：
---
{evidence}
---

只输出以下结构的 json：
{{
  "facts": [
    {{"id":"f1", "parameter":"目标参数名", "raw_value":"原文数值或文本",
      "raw_unit":"原文单位", "relation":"direct|diameter_to_radius|same_as",
      "source_parameter":"same_as 时引用的参数名，否则为空",
      "tags":["cold-start","early injection"],
      "evidence":"页码 + 原文短句", "confidence":0.0}}
  ],
  "conditions": [
    {{"id":"简短稳定ID", "labels":["论文里的名称/同义说法"],
      "fact_ids":["f1"], "parent_ids":[], "evidence":"支持该工况关系的原文"}}
  ],
  "global_fact_ids":["真正适用于整篇当前实验的事实ID"],
  "notes":["歧义或适用范围说明"]
}}

规则：
- parameter 必须严格使用给定目标参数名；数值保留论文原单位，不换算成模板单位
- 喷孔给直径、目标参数是半径时 relation=diameter_to_radius，raw_value 仍写原始直径
- “normal tip temperature same as ambient”用 relation=same_as、source_parameter=ambient_temperature
- early/late、cold/room、heated/normal、不同燃料等分别建 condition；一个图可同时绑定多个 condition
- 论文列出多组候选值时全部保留在各自 condition，不能任选一个放进 global
- 引用文献中的条件、背景知识和模拟示例不得冒充当前实验事实
- evidence 必须能支持该事实；没有证据就不要创建 fact
- id 只需在这份知识库内唯一；不要解释，只输出 json"""

BINDING_SYSTEM = (
    "你是科研图表工况绑定器。优先选择给定知识库里的 condition/fact ID；"
    "只有图内或当前图直接证据明写、且知识库缺失的值才可补成 panel_facts。"
    "不得凭常识、候选工况或其它图编造数值，只输出 json。"
)

BINDING_PROMPT = """请把当前图和每条系列绑定到论文级工况知识库。
图注：{caption}
系列：{series_names}
模板需要：{keys}

论文级知识库：
---
{knowledge}
---

当前图直接证据：
---
{local_evidence}
---

只输出 json：
{{
  "status":"bound|ambiguous",
  "global_condition_ids":["适用于整张图的 condition ID"],
  "global_fact_ids":["不经 condition 也能确认适用于整张图的 fact ID"],
  "global_parameter_fact_ids":{{"参数名":["为该参数精确选中的 fact ID"]}},
  "panel_facts":[
    {{"id":"pf1", "parameter":"目标参数名", "raw_value":"图内原始值",
      "raw_unit":"图内单位", "relation":"direct|diameter_to_radius|same_as",
      "source_parameter":"", "tags":["figure-local"],
      "evidence":"图内标题/图注中的原文", "confidence":0.0}}
  ],
  "series":{{
    "原系列名":{{"condition_ids":["该系列额外条件"], "fact_ids":["该系列直接事实"],
                 "parameter_fact_ids":{{"参数名":["该系列该参数的精确 fact ID"]}}}}
  }},
  "evidence":"绑定理由，引用图注/图内标题/当前图正文",
  "unresolved":["无法唯一绑定的字段及原因"]
}}

规则：
- 除 panel_facts 这个受控例外外，只返回知识库中真实存在的 ID，不要输出新数值
- cold-start + early injection 可同时选择；heated/normal 通常按系列分别选择
- 若一个 condition 同时含 core/bulk、环境/燃料等同名参数，必须在 parameter_fact_ids 中
  只选择当前图真正使用的 fact；不要把互相竞争的 fact 一起选入
- 只有当值明确写在当前图、图内标题、图注或当前图直接证据中，而知识库确实缺少它时，
  才可写入 panel_facts；必须保留原始值和原单位，ID 用 pf1/pf2…，并在后面的绑定中引用它
- raw_value 含多个候选值或范围的 fact 不是可填模板的单值；当前图明确写出其中
  某一个值时，应将该单值作为 panel_fact，否则保持 unresolved
- 图注或系列名不能把候选工况唯一选中时 status=ambiguous，对应字段保持未绑定
- 知识库已有完全相同的 fact 时 panel_facts 必须为 []，直接引用已有 ID
- series 只列出相对全图有额外差异的系列；若全部系列共享同一工况，必须返回
  "series":{{}}，不得把全局 ID 重复到每条系列
- 不要把引用文献工况绑定到当前图"""


DEFAULT_TARGET_UNITS = {
    "ambient_density": "g/cm3",
    "ambient_temperature": "K",
    "fuel_temperature": "K",
    "injection_pressure": "MPa",
    "nozzle_radius": "cm",
    "discharge_coefficient": "-",
    "fuel_mass": "g",
    "ambient_pressure": "dyn/cm2",
}


def _source_signature(pdf_path):
    path = Path(pdf_path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [] if value in (None, "") else [value]


def _unique(values):
    out = []
    for value in values:
        value = str(value).strip()
        if value and value not in out:
            out.append(value)
    return out


def _normal_name(value):
    return re.sub(r"\W+", "", str(value)).lower()


def normalize_knowledge(data, keys):
    """Validate the model reply and keep only facts for requested parameters."""
    allowed = {str(k) for k in keys}
    data = data if isinstance(data, dict) else {}
    facts, used_ids = [], set()
    raw_facts = data.get("facts") or []
    if isinstance(raw_facts, dict):
        raw_facts = list(raw_facts.values())
    for index, raw in enumerate(raw_facts, start=1):
        if not isinstance(raw, dict):
            continue
        parameter = str(raw.get("parameter") or "").strip()
        if parameter not in allowed:
            continue
        fact_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw.get("id") or f"f{index}"))
        while not fact_id or fact_id in used_ids:
            fact_id = f"f{index}_{len(used_ids) + 1}"
        used_ids.add(fact_id)
        relation = str(raw.get("relation") or "direct").strip().lower()
        if relation not in ("direct", "diameter_to_radius", "same_as"):
            relation = "direct"
        facts.append({
            "id": fact_id,
            "parameter": parameter,
            "raw_value": raw.get("raw_value", ""),
            "raw_unit": str(raw.get("raw_unit") or "").strip(),
            "relation": relation,
            "source_parameter": str(raw.get("source_parameter") or "").strip(),
            "tags": _unique(_as_list(raw.get("tags"))),
            "evidence": str(raw.get("evidence") or "").strip(),
            "confidence": _confidence(raw.get("confidence")),
        })

    fact_ids = {fact["id"] for fact in facts}
    conditions, condition_ids = [], set()
    raw_conditions = data.get("conditions") or []
    if isinstance(raw_conditions, dict):
        raw_conditions = [dict(value, id=key) if isinstance(value, dict) else {"id": key}
                          for key, value in raw_conditions.items()]
    for index, raw in enumerate(raw_conditions, start=1):
        if not isinstance(raw, dict):
            continue
        condition_id = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(raw.get("id") or f"condition_{index}"))
        if not condition_id or condition_id in condition_ids:
            continue
        condition_ids.add(condition_id)
        conditions.append({
            "id": condition_id,
            "labels": _unique(_as_list(raw.get("labels"))),
            "fact_ids": [x for x in _unique(_as_list(raw.get("fact_ids"))) if x in fact_ids],
            "parent_ids": _unique(_as_list(raw.get("parent_ids"))),
            "evidence": str(raw.get("evidence") or "").strip(),
        })
    for condition in conditions:
        condition["parent_ids"] = [x for x in condition["parent_ids"] if x in condition_ids]

    return {
        "schema_version": SCHEMA_VERSION,
        "facts": facts,
        "conditions": conditions,
        "global_fact_ids": [
            x for x in _unique(_as_list(data.get("global_fact_ids"))) if x in fact_ids],
        "notes": _unique(_as_list(data.get("notes"))),
    }


def _confidence(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def build_knowledge(vlm, pdf_path, keys, field_guide, evidence, cache_path=None, state=None):
    """Build/load one paper-level knowledge base. Returns (knowledge, metadata)."""
    keys = _unique(keys)
    source = _source_signature(pdf_path)
    previous = None
    if state is not None:
        cached_keys = set(state.get("knowledge_keys") or [])
        if (state.get("knowledge_source") == source
                and isinstance(state.get("knowledge"), dict)):
            previous = state["knowledge"]
            if set(keys).issubset(cached_keys):
                return previous, {"calls": 0, "source": "memory"}

    path = Path(cache_path) if cache_path else None
    if path and path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("schema_version") == SCHEMA_VERSION
                    and saved.get("source") == source):
                saved_keys = saved.get("keys") or keys
                saved_knowledge = normalize_knowledge(saved.get("knowledge"), saved_keys)
                if previous is None:
                    previous = saved_knowledge
                if set(keys).issubset(set(saved_keys)):
                    _remember(state, source, saved_keys, saved_knowledge)
                    return saved_knowledge, {"calls": 0, "source": "disk"}
        except (OSError, ValueError):
            pass

    prompt = KNOWLEDGE_PROMPT.format(field_guide=field_guide, evidence=evidence[:24000])
    data, _raw = vlm.ask_json(None, prompt, system=KNOWLEDGE_SYSTEM, max_tokens=2800)
    knowledge = normalize_knowledge(data, keys)
    _carry_panel_facts(previous, knowledge, keys)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "keys": keys,
            "knowledge": knowledge,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    _remember(state, source, keys, knowledge)
    return knowledge, {"calls": 1, "source": "model"}


def _carry_panel_facts(previous, knowledge, keys):
    """Keep figure-observed facts when a larger parameter set rebuilds the paper KB."""
    if not isinstance(previous, dict):
        return
    allowed = set(str(key) for key in keys)
    existing = knowledge.setdefault("facts", [])
    signatures = {_fact_signature(fact) for fact in existing}
    used_ids = {str(fact.get("id") or "") for fact in existing}
    for old in previous.get("facts") or []:
        if (not isinstance(old, dict) or not str(old.get("id") or "").startswith("pf_")
                or str(old.get("parameter") or "") not in allowed):
            continue
        signature = _fact_signature(old)
        if signature in signatures:
            continue
        fact = dict(old)
        fact_id = str(fact.get("id") or "")
        suffix = 2
        while fact_id in used_ids:
            fact_id = f"{old['id']}_{suffix}"
            suffix += 1
        fact["id"] = fact_id
        existing.append(fact)
        signatures.add(signature)
        used_ids.add(fact_id)


def save_knowledge(cache_path, pdf_path, keys, knowledge):
    """Persist panel-observed facts added after the initial paper pass."""
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source": _source_signature(pdf_path),
        "keys": _unique(keys),
        "knowledge": knowledge,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _remember(state, source, keys, knowledge):
    if state is not None:
        state["knowledge_source"] = source
        state["knowledge_keys"] = list(keys)
        state["knowledge"] = knowledge


def bind_panel(vlm, image_path, caption, series, keys, knowledge, local_evidence):
    """Bind a panel/series and merge explicitly evidenced figure-local facts."""
    prompt = BINDING_PROMPT.format(
        caption=(caption or "（无图注）")[:1600],
        series_names="、".join(str(item.get("name") or "") for item in series),
        keys="、".join(str(k) for k in keys),
        knowledge=json.dumps(knowledge, ensure_ascii=False, separators=(",", ":"))[:18000],
        local_evidence=(local_evidence or "（无额外证据）")[:9000],
    )
    data, _raw = vlm.ask_json(
        image_path, prompt, system=BINDING_SYSTEM, max_tokens=2000)
    data, added = _merge_panel_facts(data, knowledge, keys)
    return normalize_binding(data, knowledge, series), {"calls": 1, "new_facts": added}


def _merge_panel_facts(data, knowledge, keys):
    """Accept raw figure-local facts, assign stable IDs, and remap binding references."""
    if not isinstance(data, dict) or not data.get("panel_facts"):
        return data, []
    normalized = normalize_knowledge({"facts": data.get("panel_facts")}, keys)
    existing = knowledge.setdefault("facts", [])
    used_ids = {fact.get("id") for fact in existing}
    remap, added = {}, []
    for fact in normalized.get("facts") or []:
        old_id = fact["id"]
        if not fact.get("evidence"):
            continue
        equivalent = next((item for item in existing if _fact_signature(item) ==
                           _fact_signature(fact)), None)
        if equivalent is not None:
            remap[old_id] = equivalent["id"]
            continue
        digest = hashlib.sha1(json.dumps(
            _fact_signature(fact), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()[:10]
        fact_id = f"pf_{digest}"
        suffix = 2
        while fact_id in used_ids:
            fact_id = f"pf_{digest}_{suffix}"
            suffix += 1
        fact["id"] = fact_id
        used_ids.add(fact_id)
        existing.append(fact)
        added.append(fact_id)
        remap[old_id] = fact_id
    if not remap:
        return data, []
    clean = json.loads(json.dumps(data, ensure_ascii=False))

    def remap_ids(value):
        return [remap.get(str(item), str(item)) for item in _as_list(value)]

    clean["global_fact_ids"] = remap_ids(clean.get("global_fact_ids"))
    clean["global_parameter_fact_ids"] = {
        key: remap_ids(ids)
        for key, ids in (clean.get("global_parameter_fact_ids") or {}).items()}
    for value in (clean.get("series") or {}).values():
        if not isinstance(value, dict):
            continue
        value["fact_ids"] = remap_ids(value.get("fact_ids"))
        value["parameter_fact_ids"] = {
            key: remap_ids(ids)
            for key, ids in (value.get("parameter_fact_ids") or {}).items()}
    clean.pop("panel_facts", None)
    return clean, added


def _fact_signature(fact):
    return (
        str(fact.get("parameter") or ""),
        str(fact.get("raw_value") or "").strip().casefold(),
        _normalize_unit(fact.get("raw_unit") or ""),
        str(fact.get("relation") or "direct").strip().lower(),
        str(fact.get("source_parameter") or "").strip(),
    )


def normalize_binding(data, knowledge, series):
    data = data if isinstance(data, dict) else {}
    valid_fact_ids = {fact["id"] for fact in knowledge.get("facts") or []}
    valid_condition_ids = {item["id"] for item in knowledge.get("conditions") or []}
    names = [str(item.get("name") or "") for item in series]
    name_map = {_normal_name(name): name for name in names}
    fact_parameters = {fact["id"]: fact["parameter"] for fact in knowledge.get("facts") or []}

    def parameter_map(value):
        out = {}
        if not isinstance(value, dict):
            return out
        for parameter, ids in value.items():
            chosen = [x for x in _unique(_as_list(ids))
                      if x in valid_fact_ids and fact_parameters.get(x) == str(parameter)]
            if chosen:
                out[str(parameter)] = chosen
        return out

    by_series = {name: {"condition_ids": [], "fact_ids": [],
                        "parameter_fact_ids": {}} for name in names}
    raw_series = data.get("series") or {}
    if isinstance(raw_series, dict):
        for raw_name, value in raw_series.items():
            name = raw_name if raw_name in by_series else name_map.get(_normal_name(raw_name))
            if name is None or not isinstance(value, dict):
                continue
            by_series[name] = {
                "condition_ids": [x for x in _unique(_as_list(value.get("condition_ids")))
                                  if x in valid_condition_ids],
                "fact_ids": [x for x in _unique(_as_list(value.get("fact_ids")))
                             if x in valid_fact_ids],
                "parameter_fact_ids": parameter_map(value.get("parameter_fact_ids")),
            }
    structurally_valid = any(key in data for key in (
        "status", "global_condition_ids", "global_fact_ids", "series"))
    return {
        "valid": structurally_valid,
        "status": "ambiguous" if str(data.get("status")).lower() == "ambiguous" else "bound",
        "global_condition_ids": [
            x for x in _unique(_as_list(data.get("global_condition_ids")))
            if x in valid_condition_ids],
        "global_fact_ids": [
            x for x in _unique(_as_list(data.get("global_fact_ids"))) if x in valid_fact_ids],
        "global_parameter_fact_ids": parameter_map(data.get("global_parameter_fact_ids")),
        "series": by_series,
        "evidence": str(data.get("evidence") or "").strip(),
        "unresolved": _unique(_as_list(data.get("unresolved"))),
    }


def resolve_params(knowledge, binding, series, keys, hints=None):
    """Resolve selected facts and convert them to template units deterministically."""
    names = [str(item.get("name") or "") for item in series]
    values = {name: {} for name in names}
    meta = {
        "evidence": {name: {} for name in names},
        "basis": {name: {} for name in names},
        "fact_ids": {name: {} for name in names},
        "raw": {name: {} for name in names},
        "conflicts": {name: {} for name in names},
    }
    facts = {fact["id"]: fact for fact in knowledge.get("facts") or []}
    conditions = {item["id"]: item for item in knowledge.get("conditions") or []}

    global_ids = _unique((knowledge.get("global_fact_ids") or [])
                         + (binding.get("global_fact_ids") or [])
                         + _condition_fact_ids(binding.get("global_condition_ids") or [], conditions))
    global_parameter_ids = binding.get("global_parameter_fact_ids") or {}
    for name in names:
        selected = (binding.get("series") or {}).get(name) or {}
        specific_ids = _unique((selected.get("fact_ids") or [])
                               + _condition_fact_ids(selected.get("condition_ids") or [], conditions))
        series_parameter_ids = selected.get("parameter_fact_ids") or {}
        preferred = dict(global_parameter_ids)
        preferred.update(series_parameter_ids)
        mapped_ids = [fid for ids in preferred.values() for fid in ids]
        all_ids = _unique(global_ids + specific_ids + mapped_ids)
        for key in (str(k) for k in keys):
            mapped = [facts[fid] for fid in preferred.get(key, []) if fid in facts]
            specific = [facts[fid] for fid in specific_ids
                        if fid in facts and facts[fid]["parameter"] == key]
            candidates = mapped or specific or [facts[fid] for fid in global_ids
                                                if fid in facts and facts[fid]["parameter"] == key]
            resolved = []
            for fact in candidates:
                item = _resolve_fact(
                    fact, key, all_ids, facts, hints or {}, set(), preferred)
                if item is not None:
                    resolved.append((fact, item))
            groups = {}
            for fact, item in resolved:
                groups.setdefault(_value_signature(item["value"]), []).append((fact, item))
            if len(groups) != 1:
                if len(groups) > 1:
                    meta["conflicts"][name][key] = [item["value"]
                                                     for rows in groups.values()
                                                     for _fact, item in rows]
                continue
            rows = next(iter(groups.values()))
            fact, item = max(rows, key=lambda row: row[0].get("confidence", 0.0))
            values[name][key] = item["value"]
            meta["evidence"][name][key] = fact.get("evidence") or ""
            meta["basis"][name][key] = item["basis"]
            meta["fact_ids"][name][key] = fact["id"]
            meta["raw"][name][key] = {
                "value": fact.get("raw_value", ""), "unit": fact.get("raw_unit", ""),
                "relation": fact.get("relation", "direct"),
            }
    return values, meta


def _condition_fact_ids(condition_ids, conditions):
    out, visiting = [], set()

    def visit(condition_id):
        if condition_id in visiting or condition_id not in conditions:
            return
        visiting.add(condition_id)
        condition = conditions[condition_id]
        for parent in condition.get("parent_ids") or []:
            visit(parent)
        out.extend(condition.get("fact_ids") or [])

    for condition_id in condition_ids:
        visit(condition_id)
    return _unique(out)


def _resolve_fact(fact, key, selected_ids, facts, hints, stack, preferred=None):
    fact_id = fact["id"]
    if fact_id in stack:
        return None
    stack = set(stack)
    stack.add(fact_id)
    relation = fact.get("relation") or "direct"
    if relation == "same_as":
        source = fact.get("source_parameter") or ""
        source_ids = (preferred or {}).get(source) or selected_ids
        candidates = [facts[fid] for fid in source_ids
                      if fid in facts and facts[fid]["parameter"] == source]
        resolved = [_resolve_fact(item, source, selected_ids, facts, hints, stack, preferred)
                    for item in candidates]
        resolved = [item for item in resolved if item is not None]
        signatures = {_value_signature(item["value"]) for item in resolved}
        if len(signatures) != 1:
            return None
        source_value = resolved[0]["value"]
        source_unit = _target_unit(source, hints)
        target_unit = _target_unit(key, hints)
        converted = _convert_value(source_value, source_unit, target_unit)
        return None if converted is None else {"value": converted, "basis": "derived"}

    raw = fact.get("raw_value", "")
    raw_unit = fact.get("raw_unit") or ""
    target_unit = _target_unit(key, hints)
    numeric = _parse_number(raw)
    if numeric is None:
        if target_unit:
            return None
        text = str(raw).strip()
        return {"value": text, "basis": "explicit"} if text else None
    if relation == "diameter_to_radius":
        numeric /= 2.0
    converted = _convert_value(numeric, raw_unit, target_unit)
    if converted is None:
        return None
    raw_norm, target_norm = _normalize_unit(raw_unit), _normalize_unit(target_unit)
    basis = "derived" if relation != "direct" or raw_norm != target_norm else "explicit"
    return {"value": converted, "basis": basis}


def _target_unit(key, hints):
    hint = str((hints or {}).get(key) or "")
    matches = re.findall(r"\(([^()]*)\)", hint)
    if matches:
        unit = matches[-1].strip()
        if unit:
            return unit
    return DEFAULT_TARGET_UNITS.get(key, "")


def _parse_number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace("\u2212", "-")
    # Commas inside a thousands-group are formatting, while commas followed by
    # whitespace usually separate candidate values ("400, 800 and 1600 bar").
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    # Template fields are scalar.  Silently taking the first number from a range,
    # candidate list, nozzle designation, or prose such as "density at 298 K and
    # viscosity at 303 K" creates plausible-looking but false experimental data.
    if len(matches) != 1:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _normalize_unit(unit):
    text = str(unit or "").strip().lower()
    text = text.replace("³", "3").replace("²", "2").replace("μ", "u").replace("µ", "u")
    text = text.replace("·", "").replace("^", "").replace(" ", "")
    aliases = {
        "degc": "c", "°c": "c", "celsius": "c", "kelvin": "k",
        "dyn/cm²": "dyn/cm2", "dyncm-2": "dyn/cm2", "dyncm2": "dyn/cm2",
        "kg/m^3": "kg/m3", "g/cm^3": "g/cm3", "g/litre": "g/l",
        "micron": "um", "microns": "um", "-": "1", "none": "1",
        "dimensionless": "1",
    }
    return aliases.get(text, text)


def _convert_value(value, raw_unit, target_unit):
    numeric = _parse_number(value)
    if numeric is None:
        return None
    raw, target = _normalize_unit(raw_unit), _normalize_unit(target_unit)
    if not target:
        return _format_number(numeric)
    if raw == target or (not raw and target in ("1", "")):
        return _format_number(numeric)
    if target == "1" and raw in ("", "1"):
        return _format_number(numeric)

    pressure = {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": 1e5,
                "dyn/cm2": 0.1}
    density = {"kg/m3": 1.0, "g/l": 1.0, "mg/cm3": 1.0, "g/cm3": 1000.0}
    length = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
    mass = {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "ug": 1e-9}
    for table in (pressure, density, length, mass):
        if raw in table and target in table:
            return _format_number(numeric * table[raw] / table[target])
    if raw in ("k", "c") and target in ("k", "c"):
        kelvin = numeric if raw == "k" else numeric + 273.15
        return _format_number(kelvin if target == "k" else kelvin - 273.15)
    return None


def _format_number(value):
    if abs(value) < 5e-15:
        value = 0.0
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.12g}"


def _value_signature(value):
    number = _parse_number(value)
    return ("number", round(number, 12)) if number is not None else (
        "text", str(value).strip().casefold())
