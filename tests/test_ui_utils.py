import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui_utils import RunOptions, build_command, collect_artifacts, make_archive, safe_filename, valid_pages


class UiUtilsTests(unittest.TestCase):
    def test_safe_filename_removes_paths_and_shell_punctuation(self):
        self.assertEqual(safe_filename("../../paper; rm.pdf"), "paper_rm.pdf")
        self.assertEqual(safe_filename("图 1.pdf"), "图_1.pdf")

    def test_pages_validation(self):
        self.assertTrue(valid_pages(""))
        self.assertTrue(valid_pages("3, 5,7"))
        self.assertFalse(valid_pages("2-4"))
        self.assertFalse(valid_pages("1;whoami"))

    def test_single_pdf_command_covers_ui_options(self):
        opts = RunOptions(dpi=400, pages="3, 5", force=True, use_vlm=True,
                          use_ocr=True, want="Figure 5", want_deep=True,
                          keep_intermediates=True, template=Path("format.dat"))
        command = build_command(Path("run.py"), [Path("paper.pdf")], Path("out"), opts)
        self.assertIn("all", command)
        self.assertIn("--force", command)
        self.assertIn("--vlm", command)
        self.assertIn("--ocr", command)
        self.assertEqual(command[command.index("--pages") + 1], "3,5")
        self.assertEqual(command[command.index("--want") + 1], "Figure 5")
        self.assertIn("--template", command)

    def test_multiple_pdfs_use_batch(self):
        inputs = [Path("papers/a.pdf"), Path("papers/b.pdf")]
        command = build_command(Path("run.py"), inputs, Path("out"), RunOptions(jobs=4))
        self.assertIn("batch", command)
        self.assertEqual(command[command.index("--jobs") + 1], "4")

    def test_collect_and_archive_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "csv").mkdir()
            (root / "verify").mkdir()
            (root / "csv" / "curve.csv").write_text("x,y\n1,2", encoding="utf-8")
            (root / "verify" / "curve.png").write_bytes(b"png")
            (root / "report.md").write_text("# Report", encoding="utf-8")
            artifacts = collect_artifacts(root)
            self.assertEqual(len(artifacts["csv"]), 1)
            self.assertEqual(len(artifacts["images"]), 1)
            with zipfile.ZipFile(BytesIO(make_archive(root))) as archive:
                self.assertIn("csv/curve.csv", [name.replace("\\", "/") for name in archive.namelist()])


if __name__ == "__main__":
    unittest.main()
