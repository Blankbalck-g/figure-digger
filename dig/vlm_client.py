"""Minimal DeepSeek (OpenAI-compatible) client for chart reading tasks.

Key handling - no key is hard-coded; it is read from the first place that has it:
  1. environment variable DEEPSEEK_API_KEY
  2. <项目根目录>/deepseek_key.txt          (first non-empty line, plain text)
  3. <项目根目录>/secrets.json              ({"deepseek_api_key": "..."})

Every call is cached (sha256 of model+prompt+image) and appended to an audit log, so
runs are reproducible and reviewable.

CLI:
  python vlm_client.py --check                      # is a key configured / reachable
  python vlm_client.py --list-models                # discover exact model ids
  python vlm_client.py --ask chart.png --prompt "read the y axis"
  python vlm_client.py --selftest                   # offline plumbing test (no key needed)
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 项目根目录（run.py 所在的那一层）：密钥、缓存、审计日志都放在这里，
# 而不是模块所在目录——模块在 dig/ 里面，路径不能跟着它走。
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
CACHE_DIR = ROOT / ".vlm_cache"
LOG_PATH = ROOT / ".vlm_log.jsonl"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SIDE = 2000


class VLMError(RuntimeError):
    pass


def load_api_key(explicit=None):
    if explicit:
        return explicit.strip()
    env = os.environ.get("DEEPSEEK_API_KEY")
    if env and env.strip():
        return env.strip()
    f = ROOT / "deepseek_key.txt"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    f = ROOT / "secrets.json"
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            key = data.get("deepseek_api_key") or data.get("api_key")
            if key:
                return key.strip()
        except Exception:  # noqa: BLE001
            pass
    return None


def prepare_image(image_path):
    """Return (data_uri, sha256). PNG/JPEG in, downscaled only if oversized."""
    from PIL import Image

    raw = Path(image_path).read_bytes()
    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        img.save(buf := io.BytesIO(), format="JPEG", quality=88)
        data = buf.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii"), digest


class DeepSeekVLM:
    def __init__(self, api_key=None, base_url=None, model=None, timeout=120,
                 cache=True, log=True, verbose=False, thinking=False):
        self.api_key = load_api_key(api_key)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.use_cache = cache
        self.use_log = log
        self.verbose = verbose
        # DeepSeek enables thinking mode by default (effort=high). The chain-of-thought
        # consumes max_tokens before any answer is produced, which makes `content` come
        # back empty. These are structured extraction tasks, so thinking is off by default.
        self.thinking = thinking
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
                      "cache_hits": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0,
                      "reasoning_tokens": 0}
        self.last_usage = {}
        self.last_model = None

    # ---------- low level ----------
    def _post(self, path, payload):
        if not self.api_key:
            raise VLMError(
                "未配置 API key。请在项目根目录的 deepseek_key.txt 写入你的 key，"
                "或设置环境变量 DEEPSEEK_API_KEY。")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key,
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code == 401:
                raise VLMError(f"认证失败(401)：key 无效或过期。{detail}") from exc
            if exc.code == 402:
                raise VLMError(f"余额不足(402)：{detail}") from exc
            if exc.code == 429:
                raise VLMError(f"限流(429)：{detail}") from exc
            raise VLMError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise VLMError(f"网络不可达：{exc.reason}") from exc

    def list_models(self):
        req = urllib.request.Request(
            self.base_url + "/models",
            headers={"Authorization": "Bearer " + self.api_key,
                     "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id") for m in data.get("data", [])]

    # ---------- main entry ----------
    def _payload(self, messages, json_mode=False, max_tokens=1024, temperature=0.0):
        """Single place where the request body is built.

        DeepSeek enables thinking mode by default; the chain-of-thought consumes
        max_tokens before any answer is produced, so `content` comes back empty. Every
        request path must therefore carry the explicit thinking switch.
        """
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature,
                   "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def ask(self, image_path, prompt, system=None, json_mode=True, max_tokens=1024,
            retries=1, temperature=0.0):
        # image_path=None -> text-only request (used by the figure selector's
        # shortlist pass); everything else below - cache, log, usage - is shared.
        data_uri, digest = (None, "text-only") if image_path is None else prepare_image(image_path)
        cache_key = hashlib.sha256(
            f"{self.model}|{json_mode}|{prompt}|{system or ''}|{digest}".encode()).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if self.use_cache and cache_file.exists():
            self.usage["cache_hits"] += 1
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if self.use_log:
                with LOG_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "requested_model": self.model, "response_model": None,
                        "image": str(image_path), "image_sha256": digest,
                        "prompt": prompt[:500], "cached": True, "usage": {},
                    }, ensure_ascii=False) + "\n")
            if self.verbose:
                print(f"      [vlm] cache hit {cache_key[:10]}")
            return cached["content"]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        blocks = [{"type": "text", "text": prompt}]
        if data_uri:
            blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
        messages.append({"role": "user", "content": blocks})
        payload = self._payload(messages, json_mode=json_mode, max_tokens=max_tokens,
                                temperature=temperature)

        last_err = None
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                resp = self._post("/chat/completions", payload)
            except VLMError as exc:
                last_err = exc
                if "429" in str(exc) and attempt < retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            finish = (resp.get("choices") or [{}])[0].get("finish_reason")
            usage = resp.get("usage") or {}
            self.last_usage = usage
            self.last_model = resp.get("model") or self.model
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            # DeepSeek reports how much of the prompt came from its上下文 cache
            self.usage["cache_hit_tokens"] += int(usage.get("prompt_cache_hit_tokens") or 0)
            self.usage["cache_miss_tokens"] += int(usage.get("prompt_cache_miss_tokens") or 0)
            # reasoning tokens only appear if thinking mode leaked in - useful evidence
            self.usage["reasoning_tokens"] += int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            if self.use_log:
                with LOG_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "requested_model": self.model,
                        # the model that actually served the request
                        "response_model": resp.get("model") or self.model,
                        "response_id": resp.get("id"),
                        "image": str(image_path),
                        "image_sha256": digest, "prompt": prompt[:500],
                        "elapsed_s": round(time.time() - t0, 2),
                        "content": content[:4000], "usage": usage, "cached": False,
                        "finish_reason": finish, "thinking": self.thinking,
                    }, ensure_ascii=False) + "\n")
            if content.strip() or attempt >= retries:
                if self.use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"content": content}, ensure_ascii=False),
                                          encoding="utf-8")
                if not content.strip():
                    if finish == "length":
                        last_err = VLMError(
                            f"模型输出被 max_tokens({max_tokens}) 截断"
                            + ("（thinking 模式会先消耗推理 token）" if self.thinking else ""))
                    else:
                        last_err = VLMError("模型返回空内容（JSON 模式偶发，可重试）")
                    if attempt < retries:
                        continue
                    raise last_err
                return content
        raise last_err or VLMError("unknown failure")

    def ask_json(self, image_path, prompt, system=None, **kw):
        """ask() + robust JSON parsing (models sometimes wrap it in code fences)."""
        raw = self.ask(image_path, prompt, system=system, json_mode=True, **kw)
        return parse_json_loose(raw), raw

    def ask_text(self, prompt, max_tokens=16):
        """Text-only request; used by --probe to reveal the serving model and usage."""
        payload = self._payload([{"role": "user", "content": prompt}],
                                max_tokens=max_tokens)
        t0 = time.time()
        return self._post("/chat/completions", payload), time.time() - t0


def parse_json_loose(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise


# Official DeepSeek prices, USD per 1M tokens as (off_peak, peak).
# Source: https://api-docs.deepseek.com/quick_start/pricing
PRICES = {
    "deepseek-flash": {"in_hit": (0.003, 0.006), "in_miss": (0.15, 0.30), "out": (0.60, 1.20)},
    "deepseek-v4-pro": {"in_hit": (0.022, 0.044), "in_miss": (0.66, 1.32), "out": (1.98, 3.96)},
}

# Image billing rule (https://api-docs.deepseek.com/guides/vision):
#   images below ~544x544 are scaled up, larger ones are scaled down to a ~1300x1300
#   pixel budget, and there is an upper bound of 1024 tokens per image.
IMAGE_TOKEN_BOUND = 1024
IMAGE_BUDGET_PX = 1300 * 1300
IMAGE_UPSCALE_PX = 544 * 544


def estimate_image_tokens(image_path):
    """Return (tokens_upper_bound, note) for one image, following the documented rule."""
    from PIL import Image
    w, h = Image.open(image_path).size
    px = w * h
    if px < IMAGE_UPSCALE_PX:
        action = f"会被放大到约 1300x1300 的像素量（原图 {w}x{h} = {px/1e6:.2f} Mpx 偏小）"
    elif px > IMAGE_BUDGET_PX:
        action = f"会被缩小到约 1300x1300 的像素量（原图 {w}x{h} = {px/1e6:.2f} Mpx）"
    else:
        action = f"尺寸已在预算内（{w}x{h} = {px/1e6:.2f} Mpx）"
    return IMAGE_TOKEN_BOUND, action


def log_stats(path=LOG_PATH, model=None):
    """Aggregate the audit log: which models answered, how many tokens, estimated cost."""
    if not Path(path).exists():
        print(f"还没有调用记录（{path} 不存在）。先跑一次带 --vlm 的流程。")
        return
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    api = [r for r in rows if not r.get("cached")]
    cached = [r for r in rows if r.get("cached")]
    req_models = {}
    resp_models = {}
    tot = {"prompt": 0, "completion": 0, "hit": 0, "miss": 0}
    for r in api:
        # older log lines used "model"; keep them readable
        req = r.get("requested_model") or r.get("model") or "?"
        resp = r.get("response_model") or r.get("model") or "?"
        req_models[req] = req_models.get(req, 0) + 1
        resp_models[resp] = resp_models.get(resp, 0) + 1
        u = r.get("usage") or {}
        tot["prompt"] += int(u.get("prompt_tokens") or 0)
        tot["completion"] += int(u.get("completion_tokens") or 0)
        tot["hit"] += int(u.get("prompt_cache_hit_tokens") or 0)
        tot["miss"] += int(u.get("prompt_cache_miss_tokens") or 0)

    print(f"日志: {path}")
    print(f"  记录总数        : {len(rows)}  （真实请求 {len(api)}，本地缓存命中 {len(cached)}）")
    print(f"  请求的模型      : {req_models}")
    print(f"  实际应答的模型  : {resp_models}")
    print(f"  输入 tokens     : {tot['prompt']:,}"
          + (f"  （其中上下文缓存命中 {tot['hit']:,} / 未命中 {tot['prompt'] - tot['hit']:,}）"
             if tot['hit'] else ""))
    print(f"  输出 tokens     : {tot['completion']:,}")
    print(f"  合计 tokens     : {tot['prompt'] + tot['completion']:,}")

    if api:
        print(f"  平均每张图      : 输入 {tot['prompt']/len(api):,.0f} tokens / "
              f"输出 {tot['completion']/len(api):,.0f} tokens"
              f"（按 {len(api)} 次请求平均）")
    for name in (resp_models or req_models):
        p = PRICES.get(name)
        if not p:
            continue
        hit = tot["hit"]
        miss = max(0, tot["prompt"] - hit)
        lo = hit / 1e6 * p["in_hit"][0] + miss / 1e6 * p["in_miss"][0] \
            + tot["completion"] / 1e6 * p["out"][0]
        hi = hit / 1e6 * p["in_hit"][1] + miss / 1e6 * p["in_miss"][1] \
            + tot["completion"] / 1e6 * p["out"][1]
        print(f"  预估费用（{name}）: ${lo:.4f} ~ ${hi:.4f}（闲时~高峰，按官方价）")
    if rows:
        print(f"  时间范围        : {rows[0].get('ts', '?')[:19]} ~ {rows[-1].get('ts', '?')[:19]}")


def mock_response(prompt):
    """Canned replies so the plumbing can be tested without a key."""
    p = prompt.lower()
    # 模板编译（把用户给的格式编译成 render 代码；只调一次，之后走本地缓存）
    if "render(series, context)" in p or "数据契约" in p:
        code = ("def render(series, context):\n"
                "    out={}\n"
                "    for s in series:\n"
                "        lines=['# '+context['panel'], '# fuel='+s['params'].get('fuel','')]\n"
                "        lines+=[str(x)+'\\t'+str(y) for x,y in zip(s['x'],s['y'])]\n"
                "        out[s['name']+context['ext']]='\\n'.join(lines)+'\\n'\n"
                "    return out")
        return json.dumps({"ext": ".dat", "params": ["fuel"], "code": code})
    # 曲线的参数（模板要求标注参数时才问）：按提示词里问到的曲线名与参数名原样回一份
    if "params" in p and "曲线有" in p:
        m = re.search(r"曲线有：(.+?)。", prompt)
        names = [x.strip() for x in (m.group(1).split("、") if m else []) if x.strip()]
        m2 = re.search(r"标注这些参数：(.+?)。", prompt, re.S)
        keys = []
        if m2:
            for part in m2.group(1).split("、"):
                keys.append(re.split(r"[（(]", part.strip())[0].strip())
        return json.dumps({"params": {n: {k: "1.5e-2" for k in keys if k}
                                      for n in names}})
    # 补缺口：曲线被压住那一段的锚点（裁剪图里的归一化坐标）
    if "锚点" in p and "裁剪图" in p:
        return json.dumps({"anchors": [[0.3, 0.55], [0.5, 0.5], [0.7, 0.45]],
                           "note": "mock：沿两侧可见走向补的锚点"})
    # 复盘（识别出来的曲线编号画回图上，让模型核对数量/归属）
    if "merge" in p and "drop" in p:
        return json.dumps({"merge": [], "drop": [], "add": [], "ok": True,
                           "note": "mock：编号与图例一一对应，没有多余或遗漏"})
    # 曲线定位（模型自己找出数据曲线 + 锚点）
    if "anchors" in p:
        return json.dumps({
            "series": [
                {"label": "Heatedtip", "y_axis": "left", "color": "#ed464e",
                 "dark": False, "linestyle": "dashed", "draw": "line_only",
                 "closed": False, "overlaps": ["Normaltip"], "occluded": [[0.62, 0.72]],
                 "anchors": [[0.08, 0.95], [0.5, 0.55], [0.92, 0.22]],
                 "note": "mock：红色虚线带方块标记"},
                {"label": "Normaltip", "y_axis": "left", "color": "#1816c0",
                 "dark": False, "linestyle": "dashed", "draw": "line_only",
                 "closed": False, "overlaps": ["Heatedtip"], "occluded": [],
                 "anchors": [[0.08, 0.97], [0.5, 0.52], [0.92, 0.20]],
                 "note": "mock：蓝色虚线带方块标记"}],
            "ignore": [{"what": "蓝色实线，旁边写着 S ∝ t^0.5",
                        "why": "mock：作者的斜率参考线"}]})
    # 选图（三档里最先判，因为它的提示词里也含"图例"等词）
    if "shortlist" in p:
        if "不存在" in p:
            return json.dumps({"shortlist": [], "reason": "mock: 没有相关的图",
                               "missing": True, "available": "mock: 这篇只有贯穿距和排放两类曲线"})
        return json.dumps({"shortlist": [1, 2], "reason": "mock: 粗筛留下 1、2",
                           "missing": False, "available": ""})
    if "selected" in p:
        if "提到图的段落" in p:            # 正文通道：比看图那一步更"有据可依"
            if "不存在" in p:
                return json.dumps({"selected": [], "reason": "mock: 正文里也没有这张图",
                                   "confidence": 0.9, "missing": True,
                                   "available": "mock: 正文提到图 5 贯穿距、图 7 排放"})
            return json.dumps({"selected": [2],
                               "reason": "mock: 正文写 “as shown in Figure 5(a)” 处的工况是 800 bar",
                               "confidence": 0.85, "missing": False, "available": ""})
        if "不存在" in p:
            return json.dumps({"selected": [], "reason": "mock: 这篇里没有这张图",
                               "missing": True,
                               "available": "mock: 实际有图 5 贯穿距、图 7 排放"})
        return json.dumps({"selected": [1], "reason": "mock: 第 1 张是你要的（按轴标题判断）",
                           "missing": False, "available": ""})
    if "classify" in p or "chart type" in p or "哪一类" in p:
        return json.dumps({
            "is_line_chart": True, "chart_type": "line", "panel_count": 1,
            "dual_y_axis": False, "reason": "mock reply",
        })
    if "axis" in p or "tick" in p or "刻度范围" in p:
        return json.dumps({
            "x": {"min": 0, "max": 5, "labels": ["0", "1", "2", "3", "4", "5"],
                  "scale": "linear", "title": "Time after start of injection", "unit": "ms"},
            "y_axes": [{"side": "left", "min": 0, "max": 70,
                        "labels": ["0", "10", "20", "30", "40", "50", "60", "70"],
                        "scale": "linear", "title": "Spray tip penetration", "unit": "mm"}],
            "matches_request": True, "match_note": "mock：横轴是时间、纵轴是贯穿距",
            "confidence": 0.95,
        })
    if "legend" in p or "series name" in p or "图例" in p:
        return json.dumps({"series": [{"color": "#053856", "label": "S_tip (model from [12])"},
                                      {"color": "#8a532d", "label": "S_tail (model from [12])"}]})
    if "spot" in p or "check" in p or "要读的位置" in p:
        return json.dumps({"checks": [{"id": 1, "read_y": 42.0, "agree": True}]})
    return json.dumps({"ok": True, "echo": prompt[:60]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", metavar="IMAGE")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--check", action="store_true", help="检查 key 是否配置及可用")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--stats", action="store_true", help="统计日志里的 token 用量与费用")
    ap.add_argument("--probe", action="store_true",
                    help="发一个极小请求，确认实际应答的模型与 token 用量")
    ap.add_argument("--image-tokens", metavar="IMAGE", default=None,
                    help="按官方缩放规则估算单张图片的 token")
    ap.add_argument("--probe-image", metavar="IMAGE", default=None,
                    help="发一次真实图片请求，实测该图的 token 用量")
    ap.add_argument("--selftest", action="store_true", help="离线自检（不需要 key）")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--thinking", action="store_true",
                    help="启用 thinking 模式（默认关闭；开启会消耗推理 token，容易挤空输出）")
    args = ap.parse_args()

    if args.selftest:
        print("[selftest] 检查 JSON 解析、缓存键、图片编码（不联网）")
        for text in ['{"a": 1}', '```json\n{"a": 2}\n```', 'prefix {"a": 3} suffix']:
            print(f"   parse {text[:20]!r:26} -> {parse_json_loose(text)}")
        print(f"   默认 base_url = {DEFAULT_BASE_URL}")
        print(f"   默认 model    = {DEFAULT_MODEL}")
        key = load_api_key()
        print(f"   key 来源      = {'已配置' if key else '未配置（需写入 deepseek_key.txt）'}")
        return

    if args.stats:
        log_stats()
        return

    if args.image_tokens:
        tok, note = estimate_image_tokens(args.image_tokens)
        print(f"图片: {args.image_tokens}")
        print(f"  规则: {note}")
        print(f"  估算: 约 {tok} tokens（官方上界；实际以 API 返回的 usage 为准）")
        return

    vlm = DeepSeekVLM(api_key=args.api_key, base_url=args.base_url, model=args.model,
                      cache=not args.no_cache, verbose=True, thinking=args.thinking)
    if args.check:
        if not vlm.api_key:
            print("❌ 未找到 key。请把 key 写入项目根目录的 deepseek_key.txt（一行即可），"
                  "或设置环境变量 DEEPSEEK_API_KEY。")
            return
        print(f"✅ 已找到 key（长度 {len(vlm.api_key)}），base_url={vlm.base_url}")
        try:
            models = vlm.list_models()
            print(f"✅ 连接成功，可用模型: {models}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 连接失败: {exc}")
        return
    if args.probe:
        resp, dt = vlm.ask_text("请只回复两个字：收到")
        usage = resp.get("usage") or {}
        print(f"配置的模型   : {vlm.model}")
        print(f"实际应答模型 : {resp.get('model')}")
        print(f"响应 id      : {resp.get('id')}")
        print(f"耗时         : {dt:.2f}s")
        print(f"usage        : {json.dumps(usage, ensure_ascii=False)}")
        print(f"回答         : {(resp.get('choices') or [{}])[0].get('message', {}).get('content')!r}")
        return
    if args.probe_image:
        tok, note = estimate_image_tokens(args.probe_image)
        print(f"图片 {args.probe_image}: {note}")
        print(f"官方估算: 约 {tok} tokens —— 下面发一次真实请求实测")
        raw = vlm.ask(args.probe_image, "这张图有几个坐标轴？只回复 json：{\"axes\": 2}",
                      json_mode=True, max_tokens=64)
        print(f"实际应答模型: {vlm.last_model}")
        print(f"真实 usage   : {json.dumps(vlm.last_usage, ensure_ascii=False)}")
        print(f"回答         : {raw[:200]}")
        return
    if args.list_models:
        print(vlm.list_models())
        return
    if args.ask:
        out = vlm.ask(args.ask, args.prompt or "Describe this image.", json_mode=False)
        print(out)
        print(f"\n用量: {vlm.usage}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
