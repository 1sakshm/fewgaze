"""Build a static folder for Cloudflare Pages (or any static host): site/ plus the compiled paper.

Usage: python scripts/build_pages.py --out <dir>
       npx wrangler pages deploy <dir> --project-name fewgaze --branch main
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = """/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(self), microphone=(), geolocation=()
/demo/vendor/*
  Cache-Control: public, max-age=604800
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    # demo/parity holds Columbia Gaze images for the local parity test: the dataset licence forbids redistribution.
    shutil.copytree(ROOT / "site", out, ignore=shutil.ignore_patterns("README.md", "NOTICE.md", "parity", "parity.html"))
    pdf = ROOT / "report" / "main.pdf"
    if not pdf.exists():
        raise SystemExit("report/main.pdf missing: compile the paper first")
    shutil.copy2(pdf, out / "paper.pdf")
    cfg = (out / "config.js").read_text(encoding="utf-8")
    (out / "config.js").write_text(cfg.replace('"../report/main.pdf"', '"paper.pdf"'), encoding="utf-8")
    shutil.copy2(ROOT / "site" / "demo" / "vendor" / "NOTICE.md", out / "demo" / "vendor" / "NOTICE.txt")
    (out / "_headers").write_text(HEADERS, encoding="utf-8")
    images = [p for p in out.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"} or "parity" in p.parts]
    if images:
        raise SystemExit(f"possible dataset images in the build: {images[:3]}")
    big = [p for p in out.rglob("*") if p.is_file() and p.stat().st_size > 25 * 1024 * 1024]
    if big:
        raise SystemExit(f"files over the 25 MiB Pages limit: {big}")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
