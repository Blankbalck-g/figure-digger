"""Reusable axis-range reading: OCR the tick labels, fit, snap to the tick grid.

Wraps the pure helpers in ocr_ticks (linfit / fit_axis / snap_to_step / parse_number /
run_ocr) with the band-cropping and multi-band orchestration, so run.py can call it
without shelling out to the ocr_ticks CLI.
"""

from pathlib import Path

import cv2
import imgio as iio

import ocr_ticks as ot

_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _attempt(img, frame, engine, x_band, y_band):
    left, top, right, bottom = frame
    h, w = img.shape[:2]
    x_pad = 25
    x_x0, x_x1 = max(0, left - x_pad), min(w, right + x_pad)
    x_y0, x_y1 = min(h, bottom + 2), min(h, bottom + 2 + x_band)
    y_x0, y_x1 = max(0, left - y_band), max(1, left - 2)
    x_crop = img[x_y0:x_y1, x_x0:x_x1]
    y_crop = img[top:bottom, y_x0:y_x1]

    x_items = []
    for it in ot.run_ocr(engine, x_crop):
        val = ot.parse_number(it["text"])
        if val is not None:
            x_items.append((x_x0 + it["cx"], val, it["text"], it["score"]))
    y_items = []
    for it in ot.run_ocr(engine, y_crop):
        val = ot.parse_number(it["text"])
        if val is not None:
            y_items.append((top + it["cy"], val, it["text"], it["score"]))

    fx, fy = ot.fit_axis(x_items, None, "x"), ot.fit_axis(y_items, None, "y")
    score = ((fx["n_labels"] if fx else 0) + (fy["n_labels"] if fy else 0),
             -sum((1.0 - min(1.0, f["r2"])) + 0.02 * len(f["outliers"])
                  for f in (fx, fy) if f))
    return score, fx, fy, (x_band, y_band), (x_crop, y_crop)


def _edges(fit, lo_px, hi_px):
    raw_lo = fit["slope"] * lo_px + fit["intercept"]
    raw_hi = fit["slope"] * hi_px + fit["intercept"]
    v_lo, s_lo = ot.snap_to_step(raw_lo, fit["step"])
    v_hi, s_hi = ot.snap_to_step(raw_hi, fit["step"])
    return {
        "range": [min(v_lo, v_hi), max(v_lo, v_hi)],
        "raw_range": [min(raw_lo, raw_hi), max(raw_lo, raw_hi)],
        "labels": [l["text"] for l in fit["labels"]],
        "n_labels": fit["n_labels"],
        "r2": round(fit["r2"], 6),
        "step": fit["step"],
        "outliers": fit["outliers"],
        "repaired": [f"{t}->{c:g}" for t, _, c in fit["repaired"]],
        "snapped": bool(s_lo or s_hi),
        "looks_log": fit["looks_log"],
    }


def read_axis_ranges(image_path, frame, x_band=None, y_band=None, engine=None,
                     log=None, debug_crops=None, min_r2=0.999):
    """OCR both tick-label bands and return the axis ranges implied by the frame edges.

    Returns {'x': {...}, 'y': {...}, 'bands': (x_band, y_band)} where each axis dict has
    'range', the OCR evidence and quality flags. A missing axis means OCR could not read
    enough labels - the caller should ask a human rather than guess.
    """
    img = iio.imread(Path(image_path).resolve())
    if img is None:
        raise RuntimeError(f"cannot read {image_path}")
    engine = engine or get_engine()
    left, top, right, bottom = frame
    h, w = img.shape[:2]

    base_x = x_band or max(30, int(0.10 * (bottom - top)))
    base_y = y_band or max(40, int(0.12 * (right - left)))
    # The y band can never be wider than the space left of the axes box; a too-narrow
    # band cuts each label in half ("2.5" becomes "5" or splits into "2" and "5").
    y_max = max(30, left - 2)
    trials = [(base_x, base_y)]
    if not x_band and not y_band:
        trials += [(int(base_x * 1.6), int(min(y_max, base_y * 1.6))),
                   (int(base_x * 2.2), int(min(y_max, base_y * 2.2))),
                   (int(base_x * 1.6), y_max)]

    # Per axis, keep only fits that are essentially perfectly linear: real tick labels
    # are collinear by construction, so a low R^2 means OCR mixed up values.
    best_x = best_y = None
    best_bands = best_crops = None
    for xb, yb in trials:
        _, fx, fy, bands, crops = _attempt(img, frame, engine, xb, yb)
        if log:
            log(f"      x_band={xb} y_band={yb}: "
                f"x {fx['n_labels'] if fx else 0} 标签"
                + (f" R²={fx['r2']:.5f}" if fx else "")
                + f", y {fy['n_labels'] if fy else 0} 标签"
                + (f" R²={fy['r2']:.5f}" if fy else ""))
        for axis, fit in (("x", fx), ("y", fy)):
            if fit is None:
                continue
            cand = (fit["r2"] >= min_r2, fit["n_labels"], fit["r2"], fit, bands, crops)
            if axis == "x":
                if best_x is None or cand[:3] > best_x[:3]:
                    best_x = cand
            else:
                if best_y is None or cand[:3] > best_y[:3]:
                    best_y = cand

    fx = best_x[3] if best_x and best_x[0] else None
    fy = best_y[3] if best_y and best_y[0] else None
    bands = (best_x or best_y)[4] if (best_x or best_y) else (base_x, base_y)
    crops = (best_x or best_y)[5] if (best_x or best_y) else (None, None)
    if log and best_y and not best_y[0]:
        log(f"      y 轴拟合质量不足（最佳 R²={best_y[2]:.5f} < {min_r2}），"
            f"不输出轴范围，请人工确认")
    out = {"bands": list(bands), "image": str(image_path)}
    if fx:
        out["x"] = _edges(fx, left, right)
    if fy:
        out["y"] = _edges(fy, bottom, top)
    if debug_crops and crops[0] is not None:
        d = Path(debug_crops)
        d.mkdir(parents=True, exist_ok=True)
        iio.imwrite(d / f"{Path(image_path).stem}_xlabels.png", crops[0])
        iio.imwrite(d / f"{Path(image_path).stem}_ylabels.png", crops[1])
    return out
