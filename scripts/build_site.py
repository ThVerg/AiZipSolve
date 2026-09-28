"""Build the static (GitHub Pages) version of the game into dist/.

    python scripts/build_site.py [--out dist] [--bank PATH]

No Python server online: the pages run in "static mode" (window.ZIP_STATIC), where
zipsolve/app/static/play/static_api.js answers the game's /api calls from the precomputed
puzzle bank (zipsolve/app/static/bank/: puzzles with their unique solutions + recorded robot runs).

Output (works under any sub-path, e.g. https://<user>.github.io/AiZipSolve/):
    index.html   the game          lab.html   the AI show
    404.html     copy of the game  .nojekyll
    static/      the frontend (JS / CSS) + static/bank/
Local-only pages (editor, workbench, dashboard) are not included.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "zipsolve" / "app" / "static"
PAGES = {"index.html": "index.html", "lab.html": "lab.html"}
# the local-only pages' code (editor, workbench, dashboard) is left out; everything else is copied
LOCAL_ONLY = {"app.js", "editor.js", "editor.css", "dashboard.js", "dashboard.css", "play/ai.js"}
FLAG = ('<script>window.ZIP_STATIC = true; document.documentElement.classList.add("zip-static");</script>\n'
        '  <meta name="zip-static" content="1">\n')
NOT_FOUND = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zip</title>
<script>
  // the site lives under __BASE__ (GitHub project pages): map /lab to the AI show, anything else to the game
  (function () {
    var base = location.pathname.indexOf("__BASE__") === 0 ? "__BASE__" : "/";
    var rest = location.pathname.slice(base.length).replace(/\\/+$/, "");
    location.replace(base + (/^lab(\\.html)?$/.test(rest) ? "lab.html" : "") + location.search + location.hash);
  })();
</script>
</head><body><p><a href="./">Play Zip</a></p></body></html>
"""


def rewrite_html(html: str) -> str:
    """Absolute server paths -> relative ones, and switch the page into static mode."""
    html = html.replace('"/static/', '"static/')
    html = html.replace('href="/"', 'href="./"')
    html = html.replace('href="/lab"', 'href="lab.html"')
    # old-link redirects in the inline head scripts
    html = html.replace('location.replace("/lab"', 'location.replace("lab.html"')
    html = html.replace('location.replace("/workbench"', 'location.replace("./"')
    html = re.sub(r'<meta charset="utf-8">\n', lambda m: m.group(0) + "  " + FLAG, html, count=1)
    leftovers = re.findall(r'(?:src|href)="/(?!/)[^"]*"', html)
    leftovers = [x for x in leftovers if not re.search(r'"/(editor|dashboard|workbench)"', x)]
    if leftovers:
        raise SystemExit(f"absolute paths left in a page: {leftovers}")
    return html


def check_bank(bank: Path) -> dict:
    idx_file = bank / "index.json"
    if not idx_file.is_file():
        raise SystemExit(f"no puzzle bank at {bank} (expected index.json)")
    idx = json.loads(idx_file.read_text())
    missing = [d["file"] for m in idx.get("modes", {}).values() for d in m.values() if not (bank / d["file"]).is_file()]
    if missing:
        raise SystemExit(f"bank index lists missing pool files: {missing}")
    return idx


def build(out: Path, bank: Path, base: str = "/AiZipSolve/") -> None:
    idx = check_bank(bank)
    if out.exists():
        shutil.rmtree(out)
    (out / "static" / "play").mkdir(parents=True)
    for f in sorted([*STATIC.glob("*.js"), *STATIC.glob("*.css"), *(STATIC / "play").glob("*.js")]):
        rel = f.relative_to(STATIC).as_posix()
        if rel not in LOCAL_ONLY:
            shutil.copy2(f, out / "static" / rel)
    shutil.copytree(bank, out / "static" / "bank", ignore=shutil.ignore_patterns("*.py", "*.md", "__pycache__"))
    for src, dst in PAGES.items():
        (out / dst).write_text(rewrite_html((STATIC / src).read_text(encoding="utf-8")), encoding="utf-8")
    # GitHub Pages serves 404.html for unknown paths (/lab, /editor, old links): send them to the game or the show
    (out / "404.html").write_text(NOT_FOUND.replace("__BASE__", base), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    n = sum(d.get("count", 0) for m in idx["modes"].values() for d in m.values())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out} : {n} bank puzzles, {size / 1e6:.1f} MB")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--bank", type=Path, default=STATIC / "bank", help="puzzle bank directory")
    ap.add_argument("--base", default="/AiZipSolve/", help="URL path the site is served under (for 404.html)")
    a = ap.parse_args(argv)
    base = "/" + a.base.strip("/") + "/" if a.base.strip("/") else "/"
    build(a.out.resolve(), a.bank.resolve(), base)


if __name__ == "__main__":
    sys.exit(main())
