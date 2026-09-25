"""Small, dependency-free helpers for the Streamlit interface.

Keeping command construction and artifact discovery outside ``app.py`` makes the
UI thin and lets us test the potentially error-prone filesystem/CLI boundary.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_UNSAFE_FILENAME = re.compile(r"[^\w.\-()\u4e00-\u9fff]+", re.UNICODE)
_PAGES = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")


@dataclass(frozen=True)
class RunOptions:
    """Options shared by the one-shot and batch commands."""

    dpi: int = 300
    pages: str = ""
    force: bool = True
    use_vlm: bool = False
    use_ocr: bool = False
    want: str = ""
    want_deep: bool = False
    find_series: bool = True
    verbose: bool = False
    keep_intermediates: bool = False
    jobs: int = 1
    model: str = ""
    base_url: str = ""
    template: Path | None = None


def safe_filename(name: str, fallback: str = "document.pdf") -> str:
    """Return a portable basename and discard any user-supplied path."""

    cleaned = _UNSAFE_FILENAME.sub("_", Path(name).name).strip(" ._")
    return cleaned or fallback


def valid_pages(value: str) -> bool:
    """Accept the page-list syntax understood by the existing CLI."""

    return not value.strip() or bool(_PAGES.fullmatch(value))


def build_command(
    run_py: Path,
    inputs: Iterable[Path],
    output_dir: Path,
    options: RunOptions,
    *,
    analyze_only: bool = False,
) -> list[str]:
    """Translate UI state to the existing CLI without invoking a shell."""

    pdfs = list(inputs)
    if not pdfs:
        raise ValueError("至少需要一份 PDF")
    if analyze_only and len(pdfs) != 1:
        raise ValueError("仅分析模式一次只能处理一份 PDF")
    if not valid_pages(options.pages):
        raise ValueError("页码格式应为 7 或 3,5,7")

    if len(pdfs) == 1:
        command = [sys.executable, str(run_py), "analyze" if analyze_only else "all",
                   str(pdfs[0]), "--out", str(output_dir)]
    else:
        command = [sys.executable, str(run_py), "batch", str(pdfs[0].parent),
                   "--out", str(output_dir), "--jobs", str(max(1, options.jobs))]

    command.extend(["--dpi", str(options.dpi)])
    if options.pages.strip():
        command.extend(["--pages", options.pages.replace(" ", "")])
    if options.force and not analyze_only:
        command.append("--force")
    if options.use_vlm:
        command.append("--vlm")
        if options.model.strip():
            command.extend(["--vlm-model", options.model.strip()])
        if options.base_url.strip():
            command.extend(["--vlm-base-url", options.base_url.strip()])
    if options.use_ocr:
        command.append("--ocr")
    elif options.use_vlm:
        command.append("--no-ocr")
    if options.want.strip():
        command.extend(["--want", options.want.strip()])
    if options.want_deep:
        command.append("--want-deep")
    if not options.find_series:
        command.append("--no-find-series")
    if options.template is not None and not analyze_only:
        command.extend(["--template", str(options.template)])
    if options.verbose:
        command.append("--verbose")
    if options.keep_intermediates:
        command.append("--keep-intermediates")
    return command


def collect_artifacts(output_dir: Path) -> dict[str, list[Path]]:
    """Find the durable outputs that are useful in the result dashboard."""

    found: dict[str, list[Path]] = {
        "csv": [],
        "reports": [],
        "images": [],
        "configs": [],
    }
    if not output_dir.exists():
        return found
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            found["csv"].append(path)
        elif path.name in {"report.md", "batch_summary.md"}:
            found["reports"].append(path)
        elif suffix in {".png", ".jpg", ".jpeg"} and (
            "verify" in path.parts or "select" in path.parts
        ):
            found["images"].append(path)
        elif path.name.endswith("_config.json") or path.name == "paper_config.json":
            found["configs"].append(path)
    for values in found.values():
        values.sort(key=lambda item: item.as_posix().lower())
    return found


def make_archive(output_dir: Path) -> bytes:
    """Create an in-memory ZIP with paths relative to the selected run."""

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if output_dir.exists():
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output_dir))
    return data.getvalue()
