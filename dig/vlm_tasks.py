"""Four VLM jobs for the extraction pipeline.

  1. classify_chart     - is this a line chart? how many panels? dual Y axis?
  2. read_axis_ranges   - tick ranges for x and y (replaces / cross-checks OCR)
  3. name_series        - map detected colours to legend labels
  4. spot_check         - re-read a few y values independently, to audit our tracing

All prompts demand json (required by DeepSeek's JSON Output mode) and include an
example of the expected shape.
"""

CLASSIFY_SYSTEM = "你是科研论文图表的分析助手，只输出 json。"

CLASSIFY_PROMPT = """判断这张论文插图属于哪一类，只输出 json。
示例 json 输出：
{"is_line_chart": true, "chart_type": "line", "panel_count": 6, "dual_y_axis": false, "has_legend": true, "reason": "六联折线图"}
字段要求：
- chart_type 只能是 line|scatter|bar|pie|photo|schematic|table|mixed 之一
- is_line_chart 表示"是否是可以提取数据点的折线图/散点图"
- panel_count 是子图数量（(a)(b)(c) 这种分开计数）
- dual_y_axis 表示是否有左右两条 Y 轴
- reason 用一句话说明判断依据
注意：实验装置照片、示意图、表格都不是折线图（is_line_chart=false）。"""

AXIS_SYSTEM = "你是科研论文图表的读数助手，只输出 json。"

AXIS_PROMPT = """读出这张图表坐标轴的刻度范围和刻度值，只输出 json。
示例 json 输出：
{{"x": {{"min": 0, "max": 5, "labels": ["0", "1", "2", "3", "4", "5"], "scale": "linear"}},
 "y_axes": [{{"side": "left", "min": 0, "max": 70, "labels": ["0", "10", "20", "30", "40", "50", "60", "70"], "scale": "linear"}}],
 "confidence": 0.9, "note": ""}}
字段要求：
- min/max 是坐标框两端对应的数值（不是数据的最小最大值，是坐标轴的范围）
- labels 按顺序列出你能看清的刻度数字文本
- scale 只能是 linear 或 log
- **y_axes 是数组**：只有一个纵轴时放一个元素；若图中有左右两条纵轴，请放两个元素，
  分别用 "side": "left" / "right" 标明，并各自给出自己的 min/max/labels
{hint}
只依据图像内容填写；不确定时把 confidence 调低，不要编造数字。"""

NAME_SYSTEM = "你是科研论文图例解析助手，只输出 json。"

NAME_PROMPT = """这张图的图例把每个数据系列标了颜色。我按顺序给出颜色，请给出对应的图例文字，只输出 json。
示例 json 输出：
{{"series": [{{"color": "#053856", "label": "S_tip (model from [12])", "style": "solid",
             "y_axis": "left", "is_data": true, "role": "data",
             "reason": "图例里是这个颜色的实测曲线"}}],
 "dark_series": {{"has_dark_line": true, "kind": "model", "desc": "黑色虚线是模型预测"}}}}
字段要求：
- color 原样返回我给你的颜色
- label 是图例里对应的文字；如果这个颜色在图例里找不到，label 填 null
- style 是 solid|dashed|dashdot|marker|band 之一；无法判断填 null
- **y_axis**：这条曲线读的是哪一条纵轴，填 "left" 或 "right"；只有一个纵轴时填 "left"
- **is_data**：这个颜色的线是不是"要提取的实验数据曲线"。
- **role** 只能是 data|model|tangent|fit|annotation|legend|inset|other 之一：
  * data   = 实测/仿真得到的数据曲线（要提取）
  * model  = 模型预测曲线（通常也要，但请在 reason 里说明）
  * tangent/fit = 作者画的切线、拟合线、辅助斜线（形如 "S ∝ t^0.5"、"S ∝ t" 那种），**不是数据**
  * annotation = 标注文字、箭头、"EOI"、"11 MPa" 这类注释
  * inset = 左上/右上那种放大子图里的线
- reason 用一句话说明判断依据（例如"这条线旁边写着 S ∝ t^0.5，是作者的斜率分析线"）
- **dark_series**：图里有没有黑色/深灰色的线？它是数据、模型还是坐标轴辅助线？
  有就填 has_dark_line=true 并说明；没有填 false
颜色列表（按顺序）：{colors}"""

SPOT_SYSTEM = "你是科研论文图表的读数助手，只输出 json。"

SPOT_PROMPT = """这是一张折线图。请**只根据图像本身**（图上印着的刻度数字）回答下面的问题，只输出 json。
示例 json 输出：
{{"checks": [{{"id": 1, "read_x": 3.0, "read_y": 42.0, "note": ""}}]}}
要读的位置：
{queries}
要求：
- 位置用**视觉描述**给出（如"横轴最左端"、"横轴正中间"），请你自己在图上去定位
- read_y：该处曲线在**纵轴**上的数值，按图上刻度读数
- read_x：你定位到的**横轴**数值（按图上刻度读数），不要猜测题目中的数字
- 不要把任何外部信息当作已知条件；这一题没有任何提示答案
- 看不清就填 null 并在 note 里说明"""

SELECT_SYSTEM = "你是科研论文的图检索助手，只输出 json。"

# 论文图很多时先用纯文字过一遍，把候选缩到对照图放得下的数量
SHORTLIST_PROMPT = """用户要从一篇论文里找出他需要的数据图。下面是这篇论文的图清单。
用户需求：{want}
只输出 json：
{{"shortlist": [2, 5, 7], "reason": "…", "missing": false, "available": "…"}}
字段要求：
- shortlist：可能是目标图的编号（整数数组，按可能性从高到低，最多 12 个）
- 可能沾边的都要留下：这一步只看图注文字，宁可多留，不要漏
- 一个都不沾边就填空数组、missing=true，并在 available 里一句话说明这篇里实际有哪些数据图
图清单：
{index}"""

SELECT_PROMPT = """用户要从这篇论文里找出他需要的数据图。我给了两份材料：
（1）这篇论文的图清单；（2）一张**对照图**：把所有候选图的缩略图按编号排在一起，每张缩略图左上角的红字就是它的编号。
用户需求：{want}
只输出 json：
{{"selected": [2, 7], "reason": "…", "confidence": 0.8, "missing": false, "available": "…"}}
字段要求：
- selected：符合需求的编号（整数数组，按符合程度从高到低）
- confidence：你对自己选择的把握（0~1）。图注没写清、只能靠缩略图猜时给低分
- 判断依据可以是图注文字，也可以是你从缩略图里看到的坐标轴标题、图例、工况标注
- 只选**能取数据**的图（折线图、散点图）；照片、装置照片、示意图、二维云图、表格一律不选
- 用户提到了条件（某种燃料、压力、工况等），条件不符的**不要**选
- **这篇论文里可能根本没有**用户要的图：这时 selected 填空数组、missing=true，并在 available 里
  一句话说明这篇里实际有哪些数据图（列出图号+画的是什么）
- 不要为了给出答案而勉强选一张最像的——选错比说"没有"更糟
图清单：
{index}"""

# 第三步：正文里"Figure N shows …"这类句子，常常是唯一写清"哪张图对应哪个工况"的地方
BODY_SELECT_PROMPT = """用户要从这篇论文里找出他需要的数据图。我给了两份材料：
（1）这篇论文的图清单（编号 + 图注 + 图内文字）；
（2）论文正文里**所有提到图的段落**（只摘这些，不是全文）。
用户需求：{want}
只输出 json：
{{"selected": [2, 7], "reason": "…", "confidence": 0.8, "missing": false, "available": "…"}}
字段要求：
- selected：符合需求的编号（整数数组，按符合程度从高到低）
- 正文里常写 "Figure 5 presents the penetration at Pi = 800 bar" 这类句子：**优先用正文明确定位
  "哪个条件属于哪张图"**，图注和缩略图可能都没写
- reason 里**引用你依据的正文原句片段**（方便人工复核），并说明是哪张图
- 只选**能取数据**的图（折线图、散点图）；正文提到的照片、示意图、云图、表格都不算
- 条件不符的不要选；**真的没有**就填空数组、missing=true，并在 available 里说明这篇实际有哪些数据图
- 不要为了给出答案而勉强选：选错比说"没有"更糟
图清单：
{index}
正文中提到图的段落：
{body}"""


def classify_chart(vlm, image_path):
    data, raw = vlm.ask_json(image_path, CLASSIFY_PROMPT, system=CLASSIFY_SYSTEM)
    data["_raw"] = raw
    return data


def read_axis_ranges(vlm, image_path, hint=None):
    h = f"OCR 的初步读数是：{hint}。请核对并纠正。" if hint else "图中没有其他辅助信息。"
    data, raw = vlm.ask_json(image_path, AXIS_PROMPT.format(hint=h), system=AXIS_SYSTEM)
    data["_raw"] = raw
    return data


def name_series(vlm, image_path, colors):
    data, raw = vlm.ask_json(image_path, NAME_PROMPT.format(colors=", ".join(colors)),
                             system=NAME_SYSTEM)
    data["_raw"] = raw
    return data


def spot_check(vlm, image_path, queries):
    """Independent re-reading: the prompt carries NO axis range on purpose.

    Handing the model our own axis values would let it convert positions with our
    (possibly wrong) calibration, so its answer would agree with our data even when the
    calibration is off. By asking only "read the value at x on this chart" the answer
    comes from the chart's own printed ticks and is therefore a true cross-check.
    """
    lines = "\n".join(
        f"- id={q['id']}: 在{q.get('where') or ('横轴坐标约为 x=%g 处' % q['x'])}，"
        f"{q.get('series') or '图中对应曲线'} 的纵轴数值是多少？"
        + (f"（请按{'左' if q['axis'] == 'left' else '右'}侧纵轴的刻度读数）"
           if q.get("axis") else "")
        for q in queries)
    prompt = SPOT_PROMPT.format(queries=lines)
    data, raw = vlm.ask_json(image_path, prompt, system=SPOT_SYSTEM)
    data["_raw"] = raw
    return data


def shortlist_figures(vlm, want, index_text):
    """Text-only pass: narrow a long figure list down to what fits on a contact sheet."""
    data, raw = vlm.ask_json(
        None, SHORTLIST_PROMPT.format(want=want, index=index_text),
        system=SELECT_SYSTEM, retries=2)
    data["_raw"] = raw
    return data


def select_figures(vlm, want, index_text, image_path=None):
    """Pick the figures a natural-language request asks for.

    The contact sheet carries the part a caption cannot: what the axes actually plot.
    Users describe the chart they want ("velocity versus time at 800 bar"), which is
    usually written inside the figure, not in its caption.
    """
    prompt = SELECT_PROMPT.format(want=want, index=index_text)
    if image_path is None:                     # 没有缩略图可看时退化成纯文字
        prompt = prompt.replace("（2）一张**对照图**：把所有候选图的缩略图按编号排在一起，"
                                "每张缩略图左上角的红字就是它的编号。", "（2）没有缩略图，只能依据图注文字判断。")
    data, raw = vlm.ask_json(image_path, prompt, system=SELECT_SYSTEM, retries=2)
    data["_raw"] = raw
    return data


def select_figures_by_body(vlm, want, index_text, body_text):
    """Second opinion from the body text: only the paragraphs that mention a figure.

    A caption says what a figure *is*; the body says what it *shows* and under which
    conditions. That is the part a caption-only selection cannot resolve, and the
    relevant paragraphs are only a few percent of the paper.
    """
    data, raw = vlm.ask_json(
        None, BODY_SELECT_PROMPT.format(want=want, index=index_text, body=body_text),
        system=SELECT_SYSTEM, retries=2)
    data["_raw"] = raw
    return data


OBJECT_SYSTEM = "你是科研论文图表的读图助手，只输出 json。"

OBJECT_PROMPT = """这张论文插图的绘图区里，我检测出 {n} 个候选物件。
左边是原图（每个候选物件用红框标出并编号），右边是把每个物件单独放大的对照图（编号在左上角）。
图例里读到的条目：{legend}
每个物件的客观信息（代码量的，供你参考）：
{objects}

请逐个判断它们分别是什么，只输出 json：
{{"objects": [{{"id": 1, "what": "蓝色虚线，带三角标记", "is_data": true,
  "belongs_to": "Normaltip", "output": "line",
  "reason": "图例里 Normaltip 就是蓝色虚线"}}], "notes": ""}}
字段要求：
- what：一句话客观描述这个物件（颜色 + 线型 + 有没有标记符号 + 旁边有没有文字）
- is_data：它是不是**用户要提取的数据**。作者画的斜率参考线（如旁边写着 S ∝ t^0.5）、拟合线、
  示意线、坐标辅助线、纯文字/箭头标注 → false
- belongs_to：对应图例里的哪个条目；**对不上任何条目就填 null**
- output：points（这一系列要"标记点的坐标"）/ line（要"线的轨迹"）/
  both（点和线都要，分成两个产物）/ skip（不提取）
- reason：一句话依据。**判 false 或 skip 时一定要写清理由**（例如"图例只有 Heatedtip 与 Normaltip
  两个条目，这个物件没有对应条目，是作者的斜率分析线"）
注意：
- 同一根线被拆成几个编号时，只有最长的那段填 output，其余填 skip 并写明"是 #N 的一段"
- 图例条目是"标记符号"（如 ▽）的多半要 points；图例条目是"线型"（如 —·—）的多半要 line
- 不确定就填 skip，并在 reason 里说明，不要猜"""


def judge_objects(vlm, sheet_path, legend_text, objects_text):
    """Ask the model what each detected object is, and what to do with it."""
    n = objects_text.count("\n") + 1 if objects_text.strip() else 0
    prompt = OBJECT_PROMPT.format(n=n, legend=legend_text or "（没读到图例）",
                                  objects=objects_text or "（无）")
    data, raw = vlm.ask_json(sheet_path, prompt, system=OBJECT_SYSTEM, retries=2,
                             max_tokens=1600)
    data["_raw"] = raw
    return data


def compare_spot_check(checks, expected, tol_abs, tol_by_id=None):
    """Attach our own traced value and an agreement verdict to each VLM reading.

    `tol_abs` is an absolute tolerance, chosen by the caller from the chart's own tick
    spacing (half a tick is the best any reader - human or model - can be expected to
    achieve). A fixed percentage of the axis range is the wrong yardstick: 5% of a
    0-70 axis is 3.5 units, while 5% of a 0-2.5 axis is 0.125, which no eye can hit.
    """
    out = []
    for c in checks.get("checks", []):
        cid = c.get("id")
        ours = expected.get(cid)
        read = c.get("read_y")
        if ours is not None and isinstance(read, (int, float)):
            diff = abs(float(read) - float(ours))
            # a series reading a different y axis needs that axis' tolerance; the value
            # travels separately because the model's reply only echoes id/read_x/read_y
            tol = (tol_by_id or {}).get(cid) or c.get("tol") or tol_abs
            c["our_y"] = ours
            c["abs_diff"] = round(diff, 6)
            c["tol"] = tol
            c["agree"] = bool(diff <= tol)
        out.append(c)
    n = sum(1 for c in out if c.get("agree") is not None)
    ok = sum(1 for c in out if c.get("agree"))
    return {"checks": out, "n_compared": n, "n_agree": ok,
            "agreement": (ok / n) if n else None, "tol_abs": tol_abs}


def axis_tolerance(axis_range, tick_step, min_frac=0.02, tick_frac=0.6):
    """Absolute tolerance for a spot check: a fraction of a tick, never below 2% of range."""
    span = abs(axis_range[1] - axis_range[0]) if axis_range else 0.0
    tol = min_frac * span
    if tick_step and tick_step > 0:
        tol = max(tol, tick_frac * abs(tick_step))
    return round(tol, 6)
