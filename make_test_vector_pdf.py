"""Build a small vector PDF with a chart whose data we know exactly.

Ground truth (so the vector extractor can be verified point by point):
  axes box  x: 0..5  y: 0..10
  series A (red)  y = 2x              at x = 0,1,2,3,4,5
  series B (blue) y = 0.4*x^2         at x = 0,0.5,1,...,5

Written by hand because this environment has no reportlab/fpdf and its matplotlib is
broken (crashes at first draw).
"""

import sys
from pathlib import Path

PAGE_W, PAGE_H = 595, 842
BOX = (100.0, 500.0, 400.0, 700.0)      # x0, y0, x1, y1 in PDF user space
X_RANGE = (0.0, 5.0)
Y_RANGE = (0.0, 10.0)

SERIES_A = [(x, 2 * x) for x in range(0, 6)]
SERIES_B = [(x / 2.0, 0.4 * (x / 2.0) ** 2) for x in range(0, 11)]


def data_to_pdf(x, y):
    x0, y0, x1, y1 = BOX
    px = x0 + (x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * (x1 - x0)
    py = y0 + (y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * (y1 - y0)
    return px, py


def build_content():
    x0, y0, x1, y1 = BOX
    L = []
    L.append("0 0 0 RG 1 w")
    L.append(f"{x0} {y0} m {x1} {y0} l {x1} {y1} l {x0} {y1} l h S")

    # x ticks + labels (0..5)
    for i, xv in enumerate([0, 1, 2, 3, 4, 5]):
        px, _ = data_to_pdf(xv, 0)
        L.append(f"{px} {y0} m {px} {y0 - 8} l S")
        L.append(f"BT /F1 9 Tf 0 0 0 rg 1 0 0 1 {px - 2.5:.2f} {y0 - 20} Tm ({xv}) Tj ET")
    # y ticks + labels (0,2,..,10)
    for yv in [0, 2, 4, 6, 8, 10]:
        _, py = data_to_pdf(0, yv)
        L.append(f"{x0} {py} m {x0 - 8} {py} l S")
        w = 5.0 * len(str(yv))
        L.append(f"BT /F1 9 Tf 0 0 0 rg 1 0 0 1 {x0 - 12 - w:.2f} {py - 3:.2f} Tm ({yv}) Tj ET")

    # series A (red)
    L.append("1 0 0 RG 1.5 w")
    pts = [data_to_pdf(x, y) for x, y in SERIES_A]
    L.append(f"{pts[0][0]:.2f} {pts[0][1]:.2f} m "
             + " ".join(f"{p[0]:.2f} {p[1]:.2f} l" for p in pts[1:]) + " S")
    # series B (blue)
    L.append("0 0 1 RG 1.5 w")
    pts = [data_to_pdf(x, y) for x, y in SERIES_B]
    L.append(f"{pts[0][0]:.2f} {pts[0][1]:.2f} m "
             + " ".join(f"{p[0]:.2f} {p[1]:.2f} l" for p in pts[1:]) + " S")

    # a legend-ish swatch pair, to prove it is not mistaken for a series
    L.append("1 0 0 RG 1.5 w")
    L.append(f"{x0 + 20} {y1 - 30} m {x0 + 50} {y1 - 30} l S")
    L.append("0 0 1 RG 1.5 w")
    L.append(f"{x0 + 20} {y1 - 45} m {x0 + 50} {y1 - 45} l S")
    L.append("BT /F1 8 Tf 0 0 0 rg 1 0 0 1 "
             f"{x0 + 56:.2f} {y1 - 33:.2f} Tm (series A) Tj ET")
    L.append("BT /F1 8 Tf 0 0 0 rg 1 0 0 1 "
             f"{x0 + 56:.2f} {y1 - 48:.2f} Tm (series B) Tj ET")
    return "\n".join(L) + "\n"


def build_pdf(path):
    content = build_content().encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
         f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>").encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size " + str(len(objs) + 1).encode()
            + b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n")
    Path(path).write_bytes(bytes(out))
    print(f"wrote {path}  ({len(out)} bytes)")
    print("ground truth: A(red)=y=2x at x=0..5 ; B(blue)=y=0.4x^2 at x=0,0.5,..,5")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "fig-extract/test_vector.pdf"
    build_pdf(target)
