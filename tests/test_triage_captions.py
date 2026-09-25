"""Regression checks for two-column figure-caption extraction.

No API key and no network are needed. Usage:
    python tests/test_triage_captions.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dig"))

from triage_pdf import analyse_page, figure_mentions, page_text_lines  # noqa: E402


def _word(text, x0, top, x1, bottom):
    return {"text": text, "x0": float(x0), "top": float(top),
            "x1": float(x1), "bottom": float(bottom)}


class FakePage:
    width = 612.0
    height = 792.0
    images = []
    lines = []
    curves = []
    rects = []

    def __init__(self, words):
        self._words = words

    def extract_words(self):
        return list(self._words)


def _add_line(words, text, x0, top, char_w=5.0, height=10.0):
    x = float(x0)
    for token in text.split():
        width = max(char_w, len(token) * char_w)
        words.append(_word(token, x, top, x + width, top + height))
        x += width + 3.0


def main():
    bad = 0

    # Reproduce the 2019 paper layout: the 16 pt heading in the left column overlaps
    # two 9 pt caption rows in the right column. It must not bridge those rows.
    words = []
    _add_line(words, "The comparison of spray penetration length for", 370, 69.3,
              char_w=4.0, height=9.0)
    _add_line(words, "FIGURE 3", 319, 70.7, char_w=5.5, height=8.0)
    _add_line(words, "4.2 Numerical Set-Up", 54, 70.2, char_w=7.0, height=16.0)
    _add_line(words, "experimental dot and simulated solid line cases", 315, 81.3,
              char_w=4.0, height=9.0)
    page = FakePage(words)
    lines = [x["text"] for x in page_text_lines(page)]
    caption_lines = [x for x in lines if "FIGURE" in x]
    line_ok = len(caption_lines) == 1 and caption_lines[0].startswith("FIGURE 3 The") \
        and not caption_lines[0].startswith("experimental")
    bad += 0 if line_ok else 1
    print(f"  [{'OK ' if line_ok else 'FAIL'}] 双栏高标题不再桥接图注：{caption_lines}")

    # A true caption belongs in the caption index; prose beginning "Figure 3 shows"
    # belongs in the body-evidence digest and must not be mistaken for a caption.
    words = []
    _add_line(words, "FIGURE 3 The comparison of spray penetration length", 315, 70)
    _add_line(words, "Figure 3 shows the comparison between measured spray penetration", 54, 300)
    _add_line(words, "and calculated spray penetration as a function of time", 54, 312)
    page = FakePage(words)
    analysed = analyse_page(page, 5)
    captions = [c["text"] for c in analysed["captions"]]
    mentions = figure_mentions(page)
    class_ok = (len(captions) == 1 and captions[0].startswith("FIGURE 3")
                and len(mentions) == 1 and mentions[0].startswith("Figure 3 shows")
                and "function of time" in mentions[0])
    bad += 0 if class_ok else 1
    print(f"  [{'OK ' if class_ok else 'FAIL'}] 图注与正文引用分流："
          f"captions={captions}, mentions={mentions}")

    # If the real paper is present, pin the exact failure that prompted this test.
    pdf = ROOT / "papers" / "2019-01-0060.pdf"
    if pdf.exists():
        import pdfplumber

        with pdfplumber.open(pdf) as doc:
            page5 = doc.pages[4]
            actual = analyse_page(page5, 5)
            actual_caps = [c["text"] for c in actual["captions"]]
            actual_mentions = figure_mentions(page5)
        image = actual["images"][0]
        fig3 = next((c for c in actual["captions"]
                     if c["text"].startswith("FIGURE 3")), None)
        # Mirror the caption-pairer's geometry without importing OpenCV: Figure 3's
        # caption is immediately above and horizontally overlaps the raster figure.
        overlap = 0.0
        above_gap = float("inf")
        if fig3:
            fbox, cbox = image["bbox"], fig3["bbox"]
            overlap = max(0.0, min(fbox[2], cbox[2]) - max(fbox[0], cbox[0]))
            above_gap = fbox[1] - cbox[3]
        pairable = (fig3 is not None
                    and overlap >= 0.5 * min(image["bbox"][2] - image["bbox"][0],
                                             fig3["bbox"][2] - fig3["bbox"][0])
                    and 0 <= above_gap <= 40)
        real_ok = (pairable
                   and any(c.startswith("FIGURE 3") and "penetration" in c.lower()
                           for c in actual_caps)
                   and any(m.lower().startswith("figure 3 shows")
                           and "function of" in m.lower() for m in actual_mentions))
        bad += 0 if real_ok else 1
        print(f"  [{'OK ' if real_ok else 'FAIL'}] 2019 论文 Figure 3："
              f"captions={actual_caps[:2]}, mentions={actual_mentions[:2]}")

    print("通过" if not bad else f"{bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
