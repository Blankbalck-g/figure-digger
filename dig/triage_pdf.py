"""Triage a scientific PDF: locate figures, captions, and decide raster vs vector.

Usage:
    python dig/triage_pdf.py <input.pdf> [--outdir output] [--dpi 300] [--no-export]

Outputs:
    <outdir>/<pdf-stem>_triage.json   machine readable report
    <outdir>/figures/*.png            cropped figure regions (unless --no-export)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber

# console may be GBK on Windows; force UTF-8 so captions with special chars don't crash
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*[0-9]+(?:[.\-][0-9]+)?)\s*[.:]?\s*(.*)$", re.IGNORECASE)
TABLE_RE = re.compile(r"^\s*(table\s*[0-9]+)", re.IGNORECASE)
# 一行标题式的小标题（RESULTS / FIGURE 5 …）不该被当成图注的续行
HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,&\-]{2,40}$")
# 正文里提到图的句子（含 "Fig. 3"/"Figure 10"）
FIG_MENTION_RE = re.compile(r"\bfig(?:ure)?s?\.?\s*\d+", re.IGNORECASE)


def figure_mentions(page, context=1, min_chars=25):
    """Body lines that talk about a figure, with the next line for context.

    The interesting sentences are the ones that say what a figure shows and under
    which conditions ("Figure 10 shows the spray penetration of methanol…"), which is
    exactly what a caption usually leaves out. Lines come from page_text_lines(), so
    the two columns stay apart.
    """
    lines = page_text_lines(page)
    out = []
    for i, ln in enumerate(lines):
        text = (ln.get("text") or "").strip()
        if len(text) < min_chars or not FIG_MENTION_RE.search(text):
            continue
        if CAPTION_RE.match(text) and len(text) < 140:
            continue                      # 图注本身已经有清单了，不必重复
        piece = text
        tail = lines[i + 1:i + 1 + context]
        if tail:
            nxt = (tail[0].get("text") or "").strip()
            if nxt and not CAPTION_RE.match(nxt) and nxt[0].islower():
                piece += " " + nxt        # 句子被换行截断时接上后半句
        if piece not in out:
            out.append(piece)
    return out


def page_text_lines(page, col_gap=18.0):
    """Group words into lines ourselves, keeping the two columns apart.

    pdfplumber's extract_text_lines() merges text that shares a baseline across the
    two columns of a journal page, so a caption comes back glued to an unrelated body
    sentence ("FIGURE 2 (a) … (b) Vapor Experimental Apparatus and"). Grouping the
    words with an x-gap break keeps a caption line a caption line.

    Words belong to the same line when their vertical boxes overlap (not when their
    `top` values are within a tolerance): a stray superscript or a font change inside
    one line shifts `top` by several points, while two columns often differ by less.
    """
    try:
        words = page.extract_words()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if rows and _same_line(rows[-1], w):
            rows[-1].append(w)
        else:
            rows.append([w])
    out = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        seg = [row[0]]
        for w in row[1:]:
            if w["x0"] - seg[-1]["x1"] > col_gap:
                out.append(_line_from(seg))
                seg = []
            seg.append(w)
        out.append(_line_from(seg))
    return out


def _same_line(row, w):
    top = min(x["top"] for x in row)
    bottom = max(x["bottom"] for x in row)
    overlap = min(bottom, w["bottom"]) - max(top, w["top"])
    height = min(bottom - top, w["bottom"] - w["top"])
    return height > 0 and overlap >= 0.5 * height


def _line_from(words):
    return {
        "text": " ".join(w["text"] for w in words),
        "x0": min(w["x0"] for w in words), "x1": max(w["x1"] for w in words),
        "top": min(w["top"] for w in words), "bottom": max(w["bottom"] for w in words),
    }


def join_caption_block(lines, i, max_lines=6, x_tol=18.0, max_chars=240):
    """A caption usually runs over several lines; return the whole block.

    Only the first line starts with "Figure N", so without this the index used for
    text search would carry half a sentence ("FIGURE 5 Validation of the spray").
    The next line is treated as a continuation when it sits right below (gap < 1.8
    line heights) and starts at the same x - i.e. it is the same paragraph, not the
    next column or a heading.
    """
    first = lines[i]
    parts = [(first.get("text") or "").strip()]
    x0 = first["x0"]
    x1, bottom = first["x1"], first["bottom"]
    height = max(1.0, first["bottom"] - first["top"])
    for nxt in lines[i + 1:i + 1 + max_lines]:
        text = (nxt.get("text") or "").strip()
        if not text:
            continue
        # 续行可以缩进，但不能跑到另一栏去（起点必须落在本段已覆盖的横向范围内）
        if not (x0 - x_tol <= nxt["x0"] <= max(x1, x0 + 60) - 5):
            continue
        if CAPTION_RE.match(text) or TABLE_RE.match(text):
            break                       # 本栏的下一条图注 = 本段到此为止
        if len(text) < 25 and HEADING_RE.match(text):
            break
        gap = nxt["top"] - bottom
        if gap > 1.8 * height:
            break                       # 下方已经隔开段落了
        if gap < -0.5 * height:
            continue                    # 与本段同高但不在本段之下（多为另一栏）
        parts.append(text)
        x1 = max(x1, nxt["x1"])
        bottom = nxt["bottom"]
        if sum(len(p) for p in parts) >= max_chars:
            break                       # 越过正文就别再往下接了
    return " ".join(parts), [round(x0, 1), round(first["top"], 1),
                             round(x1, 1), round(bottom, 1)], len(parts)


def fmt_bbox(bbox):
    x0, top, x1, bottom = bbox
    return f"({x0:.0f},{top:.0f})-({x1:.0f},{bottom:.0f})"


def analyse_page(page, page_no):
    images = []
    for img in page.images:
        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
        disp_w_pt = bbox[2] - bbox[0]
        disp_h_pt = bbox[3] - bbox[1]
        native_w = int(img.get("srcsize", (0, 0))[0]) if img.get("srcsize") else int(img.get("width", 0))
        native_h = int(img.get("srcsize", (0, 0))[1]) if img.get("srcsize") else int(img.get("height", 0))
        dpi = (native_w / (disp_w_pt / 72.0)) if disp_w_pt > 0 and native_w else 0.0
        # vector objects that fall inside this image's bbox
        inner_vec = 0
        for obj in list(page.lines) + list(page.curves) + list(page.rects):
            if (obj["x0"] >= bbox[0] - 2 and obj["x1"] <= bbox[2] + 2
                    and obj["top"] >= bbox[1] - 2 and obj["bottom"] <= bbox[3] + 2):
                inner_vec += 1
        images.append({
            "bbox": [round(v, 1) for v in bbox],
            "display_pt": [round(disp_w_pt, 1), round(disp_h_pt, 1)],
            "native_px": [native_w, native_h],
            "effective_dpi": round(dpi, 1),
            "vector_objects_inside": inner_vec,
            "kind": "raster" if inner_vec < 5 else "raster+overlay",
        })

    captions = []
    lines = page_text_lines(page)
    for i, ln in enumerate(lines):
        text = (ln.get("text") or "").strip()
        m = CAPTION_RE.match(text)
        if m:
            full, bbox, n_lines = join_caption_block(lines, i)
            captions.append({
                "label": m.group(1),
                "text": full,
                "n_lines": n_lines,
                "bbox": bbox,
            })
        elif TABLE_RE.match(text):
            captions.append({"label": TABLE_RE.match(text).group(1), "text": text, "kind": "table",
                             "bbox": [round(ln["x0"], 1), round(ln["top"], 1), round(ln["x1"], 1), round(ln["bottom"], 1)]})

    return {
        "page": page_no,
        "size_pt": [round(page.width, 1), round(page.height, 1)],
        "vector": {"lines": len(page.lines), "rects": len(page.rects), "curves": len(page.curves)},
        "images": images,
        "captions": captions,
    }


def export_figures(pdf, report, outdir, dpi):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    for pageinfo in report["pages"]:
        page = pdf.pages[pageinfo["page"] - 1]
        for idx, img in enumerate(pageinfo["images"]):
            if img["display_pt"][0] < 60 or img["display_pt"][1] < 60:
                continue  # skip logos / small decorations
            bbox = tuple(img["bbox"])
            crop = page.crop(bbox, strict=False)
            name = f"{report['pdf_stem']}_p{pageinfo['page']}_img{idx + 1}.png"
            path = figdir / name
            try:
                crop.to_image(resolution=dpi).save(path)
                written.append({"file": str(path), "page": pageinfo["page"], "bbox": img["bbox"]})
            except Exception as exc:  # pragma: no cover
                written.append({"file": None, "page": pageinfo["page"], "bbox": img["bbox"], "error": str(exc)})
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    src = Path(args.pdf).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(src) as pdf:
        report = {
            "pdf": str(src),
            "pdf_stem": src.stem,
            "pages": [analyse_page(p, i + 1) for i, p in enumerate(pdf.pages)],
        }
        report["exported"] = [] if args.no_export else export_figures(pdf, report, outdir, args.dpi)

    json_path = outdir / f"{src.stem}_triage.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PDF: {src.name}  pages={len(report['pages'])}")
    for p in report["pages"]:
        vec = p["vector"]
        print(f"  p{p['page']:>2}: lines={vec['lines']:>3} rects={vec['rects']:>3} curves={vec['curves']:>3} "
              f"images={len(p['images'])} captions={len(p['captions'])}")
        for img in p["images"]:
            print(f"        image {fmt_bbox(img['bbox'])} disp={img['display_pt']}pt "
                  f"native={img['native_px']}px dpi={img['effective_dpi']} kind={img['kind']}")
        for cap in p["captions"]:
            print(f"        caption [{cap['label']}] {cap['text'][:110]}")
    if report["exported"]:
        ok = [e for e in report["exported"] if e.get("file")]
        print(f"exported {len(ok)} crops -> {outdir / 'figures'}")
    print(f"report -> {json_path}")


if __name__ == "__main__":
    main()
