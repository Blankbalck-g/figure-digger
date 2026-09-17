"""Pair every figure with *its own* caption, then pick the figures a request asks for.

Two jobs:

1. Caption pairing (local, no API cost).
   A journal page often carries several figures and several "Figure N" text lines -
   real captions plus references in the body ("Figure 3 shows that …"). Taking "the
   first caption on the page" - the old behaviour - labels the wrong figure. Here each
   figure is paired with the caption that overlaps it horizontally and sits closest to
   its bottom (or top) edge; body references lose on both distance and phrasing.

2. Figure selection from one sentence.
   The user writes what they want ("某某条件下速度随时间的折线图"). A figure number in
   that sentence is matched locally for free; otherwise one VLM call gets the caption
   index *and* a numbered contact sheet of the candidates, because what the axes plot
   is written inside the figure, not in its caption. A paper may simply not contain the
   chart - the model is required to say so and list what is actually there.
"""

import re
from pathlib import Path

import cv2
import imgio as iio
import numpy as np

import vlm_tasks as vt
import vlm_client as vlmc

SHEET_MAX = 12          # 对照图最多放几张缩略图（再多就小到看不清坐标轴了）
TILE_W, TILE_H = 430, 300
BODY_TRIGGER = 0.6      # 选图置信度低于这个值，就去正文里再问一次

LABEL_NUM_RE = re.compile(r"(?:fig(?:ure)?s?\.?|图|圖)\s*(\d+)", re.IGNORECASE)
# 正文里的引用句：Figure 3 shows… / Figure 2) is used… / Figure 15 and Figure 16 gives…
BODY_REF_RE = re.compile(
    r"^\s*fig(?:ure)?s?\.?\s*\d+[a-z]?\s*"
    r"(?:and\s+fig(?:ure)?s?\.?\s*\d+[a-z]?\s*)?[)\]]?\s*"
    r"(?:shows?|compares?|presents?|depicts?|indicates?|illustrates?|gives?|"
    r"is|are|was|were|can|will|has|have|also|and)\b", re.IGNORECASE)
TITLE_RE = re.compile(r"^\s*fig(?:ure)?s?\.?\s*\d+[a-z]?\s*[.:\-–—)]?\s+\S", re.IGNORECASE)


def looks_like_title(text):
    """True for "Figure 5. Spray penetration …", False for "Figure 5 shows …"."""
    t = (text or "").strip()
    if not t or BODY_REF_RE.match(t):
        return False
    return bool(TITLE_RE.match(t))


def label_number(label):
    m = re.search(r"(\d+)", label or "")
    return int(m.group(1)) if m else None


def _overlap_x(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _touching(a, b, gap=8.0):
    """Two boxes that belong to one figure split into stacked/side-by-side pieces."""
    dx = max(a[0], b[0]) - min(a[2], b[2])      # >0 when separated horizontally
    dy = max(a[1], b[1]) - min(a[3], b[3])      # >0 when separated vertically
    overlap_x = _overlap_x(a, b)
    overlap_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    stacked = overlap_x >= 0.6 * min(a[2] - a[0], b[2] - b[0]) and dy <= gap
    side_by_side = overlap_y >= 0.6 * min(a[3] - a[1], b[3] - b[1]) and dx <= gap
    return stacked or side_by_side


def attach_captions(cards, captions_by_page, log=None):
    """Give each card the caption that belongs to it (in place).

    cards:     [{"id", "page", "bbox", ...}] - bbox in PDF points
    captions:  {page: [{"label", "text", "bbox"}, ...]} from triage_pdf
    """
    for card in cards:
        card.setdefault("caption", "")
        card.setdefault("label", "")
        card.pop("caption_kind", None)

    for page, group in _by_page(cards).items():
        caps = captions_by_page.get(page, []) or []
        if not caps:
            continue
        scored = []
        for card in group:
            fig = card["bbox"]
            for cap in caps:
                box = cap["bbox"]
                if _overlap_x(fig, box) < 0.5 * min(fig[2] - fig[0], box[2] - box[0]):
                    continue
                below, above = box[1] - fig[3], fig[1] - box[3]
                if 0 <= below <= 80:
                    gap, side = below, "below"
                elif 0 <= above <= 40:
                    gap, side = above, "above"
                else:
                    continue
                score = gap
                if not looks_like_title(cap["text"]):
                    score += 45          # 正文引用句，能不选就不选
                if side == "above":
                    score += 20
                scored.append((score, card["id"], cap))
        scored.sort(key=lambda t: t[0])
        taken, used_caps, leftovers, direct = {}, set(), set(c["id"] for c in group), set()
        for _, cid, cap in scored:      # 第一轮：一条图注只给一张图
            if cid not in leftovers or id(cap) in used_caps:
                continue
            taken[cid] = cap
            used_caps.add(id(cap))
            leftovers.discard(cid)
            direct.add(cid)
        # 第二轮：一张图常由好几块紧挨着的子图组成（上下拼接的照片阵列），图注只写在
        # 整块下面——离得远，但"贴在一起"这件事是可靠的，按这个把图注传过去。
        for card in group:
            if card["id"] not in leftovers:
                continue
            for other in group:
                if other["id"] == card["id"] or other["id"] not in taken:
                    continue
                if _touching(card["bbox"], other["bbox"]):
                    taken[card["id"]] = taken[other["id"]]
                    leftovers.discard(card["id"])
                    break
        for card in group:
            cap = taken.get(card["id"])
            if not cap:
                continue
            card["caption"] = cap["text"]
            card["label"] = cap["label"]
            card["caption_kind"] = "title" if looks_like_title(cap["text"]) else "body-ref"
            card["caption_shared"] = card["id"] not in direct
    if log:
        paired = sum(1 for c in cards if c["caption"])
        log(f"      图注配对: {paired}/{len(cards)} 张图找到文字")
    return cards


def _by_page(cards):
    out = {}
    for c in cards:
        out.setdefault(c["page"], []).append(c)
    return out


def index_text(cards, max_caption=200, numbers=None):
    """The plain-text figure list handed to the model (numbers = candidate numbers)."""
    numbers = numbers or list(range(1, len(cards) + 1))
    lines = []
    for i, c in zip(numbers, cards):
        parts = [f"{i}. {c.get('label') or '(无编号)'} (p{c['page']})"]
        if c.get("caption"):
            parts.append(f"图注: {c['caption'][:max_caption]}")
        if c.get("inner_text"):
            parts.append(f"图内文字: {c['inner_text'][:160]}")
        if not c.get("caption") and not c.get("inner_text"):
            parts.append("（没有图注文字，只能看缩略图）")
        lines.append(" ｜ ".join(parts))
    return "\n".join(lines)


def build_contact_sheet(cards, out_dir, numbers=None, cols=3,
                        tile_w=TILE_W, tile_h=TILE_H):
    """Thumbnails of the candidates in a numbered grid - one image for one API call."""
    tiles = []
    numbers = numbers or list(range(1, len(cards) + 1))
    for card, num in zip(cards, numbers):
        img = iio.imread(card["thumb"]) if card.get("thumb") else None
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(tile_w / float(w), tile_h / float(h))
        resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        tile = np.full((tile_h, tile_w, 3), 255, dtype="uint8")
        y0 = (tile_h - resized.shape[0]) // 2
        x0 = (tile_w - resized.shape[1]) // 2
        tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (180, 180, 180), 1)
        cv2.rectangle(tile, (2, 2), (78, 34), (255, 255, 255), -1)
        cv2.putText(tile, str(num), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = tiles[i:i + cols]
        while len(chunk) < cols:
            chunk.append(np.full((tile_h, tile_w, 3), 255, dtype="uint8"))
        rows.append(cv2.hconcat(chunk))
    sheet = cv2.vconcat(rows)
    out_path = Path(out_dir) / "contact_sheet.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, sheet)
    return out_path


def local_match(want, cards):
    """A figure number in the request ("Figure 5" / "图5") -> no model call at all."""
    wanted = {int(n) for n in LABEL_NUM_RE.findall(want or "")}
    if not wanted:
        return [], ""
    ids = [c["id"] for c in cards if label_number(c.get("label")) in wanted]
    if ids:
        return ids, f"需求里的编号直接命中图 {sorted(wanted)}"
    return [], ""


def _candidates(cards, selected=(), screened=None):
    """One row per candidate figure. `screened` marks the ones the model actually saw."""
    out = []
    for n, card in enumerate(cards, start=1):
        out.append({"n": n, "id": card["id"], "label": card.get("label") or "",
                    "page": card["page"], "caption": (card.get("caption") or "")[:200],
                    "selected": card["id"] in selected,
                    "screened": (n in screened) if screened else True})
    return out


def _pick(data, cards):
    """Model reply -> list of card ids (numbers are 1-based positions in the index)."""
    picked = []
    for n in (data.get("selected") or []):
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(cards) and cards[n - 1]["id"] not in picked:
            picked.append(cards[n - 1]["id"])
    return picked


def select(want, cards, vlm=None, out_dir=None, log=None, body_text=None, deep=False):
    """Decide which figures the request is about. Returns a decision record.

    body_text: callable returning the paper's figure-mentioning paragraphs, or None.
               It is only called when the caption/thumbnail pass does not settle the
               question, so most papers never pay for it.
    deep:      always ask the body-text pass, even when the captions already matched.
    """
    if not cards:
        return {"want": want, "method": "none", "ids": [], "missing": True,
                "reason": "这篇里没找到任何候选图", "available": "", "candidates": []}

    ids, how = local_match(want, cards)
    if ids:
        if log:
            log(f"      需求命中（本地编号匹配，未调用模型）：{how}")
        return {"want": want, "method": "label", "ids": ids, "missing": False,
                "reason": how, "available": "", "body_used": False, "body_chars": 0,
                "body_reason": "", "candidates": _candidates(cards, selected=ids)}

    if vlm is None:
        reason = "未启用 VLM：只能按图号找图（如 --want 'Figure 5'），无法按图的内容找"
        if log:
            log(f"      ✗ {reason}")
        return {"want": want, "method": "none", "ids": [], "missing": True,
                "reason": reason, "available": "", "body_used": False, "body_chars": 0,
                "body_reason": "", "candidates": _candidates(cards)}

    index = index_text(cards)
    numbers = list(range(1, len(cards) + 1))
    sheet_cards = cards
    if len(cards) > SHEET_MAX:                  # 先纯文字粗筛，对照图只放得下 12 张
        try:
            pre = vt.shortlist_figures(vlm, want, index)
            keep = [int(n) for n in (pre.get("shortlist") or [])
                    if str(n).isdigit() and 1 <= int(n) <= len(cards)]
        except Exception:  # noqa: BLE001
            keep = []
        if keep:
            numbers = keep[:SHEET_MAX]
            sheet_cards = [cards[n - 1] for n in numbers]
            if log:
                log(f"      文字粗筛 -> 候选 {len(cards)} 张中的 {len(numbers)} 张进对照图")
        else:
            numbers = numbers[:SHEET_MAX]
            sheet_cards = cards[:SHEET_MAX]

    sheet = None
    screened = None
    if out_dir is not None:
        sheet = build_contact_sheet(sheet_cards, out_dir, numbers=numbers)
    if len(cards) > SHEET_MAX:
        screened = set(numbers)          # 粗筛后进对照图的那几张，其余只是没进画面
        index = index_text(sheet_cards, numbers=numbers)   # 看图这一步只带候选的图注
    try:
        data = vt.select_figures(vlm, want, index, image_path=sheet)
    except Exception as exc:  # noqa: BLE001
        return {"want": want, "method": "error", "ids": [], "missing": None,
                "reason": f"{type(exc).__name__}: {exc}", "available": "",
                "candidates": _candidates(cards, screened=screened)}

    picked = _pick(data, cards)
    conf = data.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else None
    reason = str(data.get("reason") or "")[:400]
    available = str(data.get("available") or "")[:400]
    body_note, body_chars, body_tokens = "", 0, 0

    # ---- 第二步（按需）：正文里提到图的段落。图注说不清"哪张图对应哪个工况"时，
    #      只有正文会写 "Figure 5 presents the penetration at Pi = 800 bar"。----
    need_body = body_text is not None and (deep or not picked
                                           or (conf is not None and conf < BODY_TRIGGER))
    if need_body:
        try:
            digest = body_text() or ""
        except Exception as exc:  # noqa: BLE001
            digest = ""
            log and log(f"      正文摘要提取失败：{type(exc).__name__}: {exc}")
        body_chars = len(digest)
        if digest:
            if log:
                log(f"      再问一次正文（提到图的段落 {body_chars} 字符）")
            usage0 = dict(vlm.usage)
            try:
                bdata = vt.select_figures_by_body(vlm, want, index, digest)
                bpick = _pick(bdata, cards)
                breason = str(bdata.get("reason") or "")[:400]
                conf2 = bdata.get("confidence")
                conf2 = float(conf2) if isinstance(conf2, (int, float)) else None
                body_note = breason
                if bpick:
                    if log:
                        log(f"      正文判定的结果：{len(bpick)} 张 -> "
                            f"{', '.join(bpick)}（依据：{breason[:60]}）")
                    picked, reason = bpick, breason      # 正文证据更具体，采用它
                    conf = conf2 if conf2 is not None else conf
                    available = str(bdata.get("available") or "")[:400]
                elif log:
                    log(f"      正文也没有找到（依据：{breason[:60]}）")
            except Exception as exc:  # noqa: BLE001
                body_note = f"{type(exc).__name__}: {exc}"
                if log:
                    log(f"      正文通道失败：{body_note}")
            du = {k: vlm.usage.get(k, 0) - usage0.get(k, 0) for k in vlm.usage}
            body_tokens = du["prompt_tokens"] + du["completion_tokens"]
            pr = vlmc.PRICES.get(getattr(vlm, "model", ""))
            if pr and body_tokens:
                lo = (du["prompt_tokens"] / 1e6 * pr["in_miss"][0]
                      + du["completion_tokens"] / 1e6 * pr["out"][0])
                hi = (du["prompt_tokens"] / 1e6 * pr["in_miss"][1]
                      + du["completion_tokens"] / 1e6 * pr["out"][1])
                if log:
                    log(f"      正文通道花费（实测）：{body_tokens:,} tokens "
                        f"≈ ¥{lo * 7.2:.3f}~¥{hi * 7.2:.3f}")
    if log:
        if picked:
            log(f"      模型选中 {len(picked)} 张：{', '.join(picked)}")
        else:
            log("      模型判定：这篇里没有符合需求的图")
    return {"want": want, "method": "vlm", "ids": picked,
            "missing": not picked,
            "reason": reason, "confidence": conf,
            "body_used": bool(body_chars), "body_chars": body_chars, "body_reason": body_note,
            "body_tokens": body_tokens,
            "available": available,
            "screened_n": len(numbers) if screened else len(cards),
            "candidates": _candidates(cards, selected=set(picked), screened=screened)}
