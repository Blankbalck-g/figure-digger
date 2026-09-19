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
{{"x": {{"min": 0, "max": 5, "labels": ["0", "1", "2", "3", "4", "5"], "scale": "linear",
        "title": "Time after start of injection", "unit": "ms"}},
  "y_axes": [{{"side": "left", "min": 0, "max": 70, "labels": ["0", "10", "20", "30", "40", "50", "60", "70"],
              "scale": "linear", "multiplier": 1e-4, "unit": "mm", "title": "Spray tip penetration",
              "multiplier_note": "轴标题左上角写着 ×10^-4"}}],
  "matches_request": true, "match_note": "横轴是时间(ms)、纵轴是贯穿距(mm)，与需求一致",
  "confidence": 0.9, "note": ""}}
字段要求：
- min/max 是坐标框两端对应的数值（不是数据的最小最大值，是坐标轴的范围）
- labels 按顺序列出你能看清的刻度数字文本
- scale 只能是 linear 或 log
- **multiplier**：这张图有没有数量级/单位换算标注（轴旁边的 "×10^-4"、"×10^3"、
  "1e-3"、"(×10⁻⁴)" 这类，常在轴标题或图的左上/右上角）？有就填成**乘数**
  （例如 ×10^-4 填 0.0001），没有填 1。这条决定最终数据的数值大小，请务必看清
  ——**注意区分是乘在 x 轴还是 y 轴**，分别填到对应的轴里
- unit：轴标题里的单位文字（没有填 null）
- **title**：轴标题文字（例如 "Time after start of injection"、"Spray tip penetration"），
  **不带单位**；看不清填 null
- **y_axes 是数组**：只有一个纵轴时放一个元素；若图中有左右两条纵轴，请放两个元素，
  分别用 "side": "left" / "right" 标明，并各自给出自己的 min/max/labels
- **matches_request**：只有补充信息里给了"用户要的是什么图"时才判断——这张图的横轴/
  纵轴是否**就是**用户要的？是填 true，不是填 false，看不清填 null。
  要严格：用户要"贯穿距随时间变化"，则横轴必须是时间、纵轴必须是贯穿距；
  横轴是曲轴转角、或纵轴是喷雾锥角/动量通量的都算 false
- match_note：一句话说明依据（横轴是…、纵轴是…，所以符合/不符合）
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
    h = f"补充信息：{hint}" if hint else \
        "补充信息：没有（不用判断 matches_request，填 null）"
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


FIND_SERIES_SYSTEM = "你是科研论文图表的读图助手，只输出 json。"

# 让模型**自己找出**数据曲线，而不是评价代码给出的候选清单。这样它的输出是一份
# "要点提取的曲线"清单（正向），不会出现"数据被判成非数据而消失"的情况；漏了哪条
# 也只是报告里对账得到，不会静默丢。
FIND_SERIES_PROMPT = """这是一张论文插图。请找出图里**所有需要提取数据的曲线**，并给出每条曲线的大致位置。
图例里读到的条目（供参考，可能不全）：{legend}
只输出 json：
{{"series": [
  {{"label": "Normaltip", "y_axis": "left", "color": "#1816c0", "dark": false,
    "linestyle": "dashed", "draw": "markers_connected", "closed": false,
    "overlaps": ["Heatedtip"], "occluded": [[0.62, 0.72], [0.78, 0.60]],
    "anchors": [[0.06, 0.02], [0.5, 0.55], [0.95, 0.78]],
    "note": "两条曲线在前半段几乎重合，重合段压在下面"}}],
 "ignore": [{{"what": "蓝色实线，旁边写着 S ∝ t^0.5", "why": "作者的斜率参考线，不是数据"}}]}}
字段要求：
- label：这条曲线对应图例里的哪个条目；图例里没有就填 null 并在 note 说明
- y_axis：读的是左轴还是右轴（只有一个纵轴时填 left）
- color：这条线**实际**的颜色，十六进制；黑色/灰色线填 "#000000" 并把 dark 填 true
- **dark**：这条线是不是黑色/深灰色。黑白论文图里必须填 true——代码会改用"深色
  笔画 + 你给的锚点"来追，不再依赖颜色
- linestyle：solid|dashed|dashdot|marker|band
- **draw**：这张图里**这条曲线**是怎么画出来的，四选一：
  * markers_only        只有散点，没有连线
  * markers_connected   散点用折线连起来（点即曲线；**不要**报成"点+线"两条）
  * line_only           只有线，没有标记
  * markers_plus_line   散点是实验值，另有一条**独立的**拟合/模型曲线（这才是两条）
- **closed**：这条曲线是不是首尾相接的闭合回线（P-V 图边界、喷雾包络、等值线那种
  围成一圈的）。普通从左到右的曲线填 false。**这条决定代码用哪种追踪方式**，填错会让
  曲线被追成一圈来回走的假轨迹
- **overlaps**：这条曲线与哪几条曲线有重合（颜色互相盖住）？填那些曲线的 label，
  没有就填 []。重合处只显示上面那条的颜色是**正常的**——不要因此漏掉被压住的那条，
  也不要把同一根线报两遍
- **occluded**：这条曲线**自己颜色看不见**的段落，每段给一个点 [x, y]（归一化），
  例如被另一条曲线压住、被图例框或文字盖住的位置。代码会在这些地方沿另一条曲线补全
- **anchors**：这条曲线上三个点的**大致**位置，按 [x, y] 给，
  x、y 都是相对图片的归一化坐标（左上角 0,0，右下角 1,1），顺序为
  "从左数第一个可见点、中间一点、最右端一点"。**不要求精确**（±3% 就够），
  代码会以它们为种子去精确追踪。曲线被遮挡或断裂时，挑你确实看得见的位置。
- note：一句话说明（例如"与另一条曲线重叠"、"前半段被图例遮住"）
另外用 ignore 列出你**没有**当作数据的东西（作者画的斜率参考线、拟合辅助线、示意图、
标注文字等），各写一句理由。
注意：
1. 坐标轴、网格、图例框、纯标注文字都不是数据曲线。
2. **一个图例条目 = 一条曲线 = 一份数据**：不要把同一根曲线的"点"和"线"报成两条
   （散点连线的图用 draw=markers_connected）；也不要把同一根曲线因为被遮住而分成两条。
3. 颜色相同、靠得很近的两条线（例如同色虚线）请**分别**报出来，并在 note 里说明
   它们靠什么区分（上下位置 / 线型 / 哪一段分开）。"""


def find_series(vlm, image_path, legend_text):
    """Ask the model where the data curves are (coarse seeds), not what our blobs are."""
    prompt = FIND_SERIES_PROMPT.format(legend=legend_text or "（没读到图例）")
    data, raw = vlm.ask_json(image_path, prompt, system=FIND_SERIES_SYSTEM, retries=2,
                             max_tokens=2200)
    data["_raw"] = raw
    return data


REVISE_SYSTEM = "你是科研论文图表的校对助手，只输出 json。"

# 复盘：模型像素上不准（所以不让它画线），但"这两条是不是同一条""这条追到图例上了"
# 这类判断正是它的强项。把代码追出来的结果编号画回原图给它看，让它当裁判——
# 算法在模型指导下收尾，只在有问题的面板上多问一次。
REVISE_PROMPT = """图上是代码从这张图里追出来的曲线：彩色描边 + 编号 #1 #2 ...，
红色 ✗ 是代码没追到的。请核对它们对不对，只输出 json：
{{"merge": [[1, 3]], "drop": [4], "add": [], "ok": false, "note": "一句话"}}
字段要求：
- merge：哪两个编号其实是**同一条曲线**被重复提取了（配对给出，保留编号小的那个）。
  颜色相同、位置重合、明显是同一根线的都算。没有填 []
- drop：哪个编号追错了（不是数据 / 只是碎段 / 追到别的曲线或图例上 / 追到了坐标轴上 /
  **把背景网格线或坐标框当成了曲线**——那种轨迹横跨整幅、几乎没有起伏）。没有填 []
- add：图上有数据曲线、而代码**一条都没追上**的（注意看图例里列了、但编号里没有的）。
  每条给：{{"label": 图例名, "color": "#rrggbb", "dark": false, "closed": false,
  "draw": "line_only", "anchors": [[x, y], [x, y], [x, y]], "why": "哪条"}}
  锚点必须是你能看见的、落在那条线上的点（归一化坐标，左上 0,0）。没有填 []
- ok：上面三项都为空时填 true
- **axis_fix**：只在**坐标轴映射明显不对**时才给（代码用的轴范围见下面的"轴的现状"）。
  典型的错法：曲线数值整体偏移、被压成一条平线、超出轴范围、或者漏了 ×10^-4 这种
  数量级标注。给的话写成
  {{"x": [min, max], "y": [min, max], "multiplier": 0.0001}}
  （multiplier 是数据要乘的倍数；分不清就只给 x/y）。没问题就不要这个字段
- note：一句话总结（例如"#2 与 #4 是同一条，另外漏了图例里的 Ethanol"）
判断只看图上画的描边和原图，注意：
* **重合处只显示上面那条的颜色是正常的**，不要因为下面那条看不见就说它多余；
* 散点用折线连起来的那种图是**一条**曲线，不是"点加线"两条；
* 拿不准的不要动（宁可 ok=true）。
模型最初读到的曲线清单（供参考）：{spec}
代码实际追出来的：
{found}
轴的现状（代码正在用的映射）：{axis}"""


def revise_traces(vlm, image_path, spec_text, found_text, axis_text=""):
    """One review round: the model judges the traces the code produced."""
    prompt = REVISE_PROMPT.format(spec=spec_text or "（无）", found=found_text or "（无）",
                                  axis=axis_text or "（未记录）")
    data, raw = vlm.ask_json(image_path, prompt, system=REVISE_SYSTEM, retries=2,
                             max_tokens=900)
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
