"""Package site/ for publishing as a claude.ai Artifact (a hosted single page with supporting files).

Differences from the GitHub Pages build, all forced by the artifact host:
* the page is written without <!doctype>/<html>/<head>/<body> (the host adds its own skeleton);
* config.js is inlined with no repository URL; the compiled paper ships as paper.pdf next to the page;
* face_landmarker.task is shipped as base64 text (the host does not serve .task files; demo.js decodes it).

Usage: python scripts/build_artifact.py --out <dir>   (prints the file list for the publish call)
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
COPY = [
    "styles.css",
    "app.js",
    "data/results.json",
    "data/extras.json",
    "assets/face_mesh.json",
    "demo/demo.js",
    "demo/fewgaze-core.js",
    "demo/model.json",
    "demo/vendor/vision_bundle.mjs",
    "demo/vendor/vision_wasm_internal.js",
    "demo/vendor/vision_wasm_internal.wasm",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    s = (SITE / "index.html").read_text(encoding="utf-8").replace("\r\n", "\n")
    for pat in [
        r"<!doctype html>\n",
        r"<html[^>]*>\n",
        r"<head>\n",
        r'<meta charset="utf-8">\n',
        r'<meta name="viewport"[^>]*>\n',
        r"</head>\n",
        r"<body>\n",
        r"</body>\n",
        r"</html>\n?",
    ]:
        s, n = re.subn(pat, "", s, count=1, flags=re.I)
        if n != 1:
            raise SystemExit(f"skeleton tag not found once: {pat}")
    s = s.replace('<script src="config.js"></script>', '<script>window.FEWGAZE_CONFIG = { repoUrl: "", paperUrl: "paper.pdf" };</script>')
    if not s.startswith("<title>") or re.search(r"<(html|head|body)\b", s, re.I):
        raise SystemExit("unexpected page structure after stripping")
    (out / "index.html").write_text(s, encoding="utf-8")
    files = {}
    for rel in COPY:
        src = SITE / rel
        if not src.exists():
            print(f"missing (skipped): {rel}", file=sys.stderr)
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = str(dst)
    pdf = SITE.parent / "report" / "main.pdf"
    if pdf.exists():
        shutil.copy2(pdf, out / "paper.pdf")
        files["paper.pdf"] = str(out / "paper.pdf")
    task = SITE / "demo/vendor/face_landmarker.task"
    b64 = out / "demo/vendor/face_landmarker.task.b64.txt"
    b64.write_text(base64.b64encode(task.read_bytes()).decode("ascii"), encoding="ascii")
    assert base64.b64decode(b64.read_text()) == task.read_bytes()
    files["demo/vendor/face_landmarker.task.b64.txt"] = str(b64)
    print(json.dumps({"page": str(out / "index.html"), "files": files}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
