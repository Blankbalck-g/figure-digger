"""Integration test: run the real pipeline with the VLM pointed at a local mock.

Proves the wiring (classification gate, axis cross-check, series naming, spot check)
without a key and without network. Usage: python test_vlm_pipeline.py <pdf> [page]
"""

import json
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as wpd_run  # noqa: E402
import vlm_client as vc  # noqa: E402
from test_vlm_mock import Handler  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main():
    here = Path(__file__).resolve().parent
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else (here.parent / "2023-01-1635.pdf")
    page = sys.argv[2] if len(sys.argv) > 2 else "7"
    out = here / "vlm_test_out"

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"mock VLM: http://127.0.0.1:{port}   测试: {pdf.name} 第 {page} 页\n")

    client = vc.DeepSeekVLM(api_key="sk-mock", base_url=f"http://127.0.0.1:{port}",
                            model="deepseek-flash", verbose=True)
    print("=" * 70)
    print("阶段 1: analyze（VLM 分类 + 轴读数交叉核对 + 系列命名）")
    print("=" * 70)
    cfg = wpd_run.analyze(pdf, out, dpi=300, vlm=client, pages=page)

    config = json.loads(Path(cfg).read_text(encoding="utf-8"))
    print("\n配置摘要:")
    for p in config["panels"]:
        print(f"  {p['id']}: 方法={p.get('method', 'raster')} "
              f"轴x={p['axis'].get('x')} y={p['axis'].get('y')} "
              f"confirmed={p['axis'].get('confirmed')} "
              f"来源={p.get('axis_src_x')}/{p.get('axis_src_y')}")
        if p.get("series_names"):
            print(f"      系列命名: {p['series_names']}")
        if p.get("notes"):
            print(f"      备注: {p['notes']}")

    print("\n" + "=" * 70)
    print("阶段 2: extract（提取 + VLM 抽查校验）")
    print("=" * 70)
    wpd_run.extract(cfg, force=True, vlm=client)

    results = json.loads((out / "extract_results.json").read_text(encoding="utf-8"))
    print("\n结果摘要:")
    for pid, r in results.items():
        sc = r.get("spot_check")
        print(f"  {pid}: {len(r['series'])} 条曲线"
              + (f"  抽查 {sc['n_agree']}/{sc['n_compared']} 一致" if sc else "  （无抽查）"))
    print(f"\nVLM 调用统计: {client.usage}")
    print(f"缓存目录: {vc.CACHE_DIR.name}   审计日志: {vc.LOG_PATH.name}")
    srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
