"""Screenshots of the web app (needs a running server + playwright chromium).

    python -m zipsolve.app --no-browser --port 8765 &
    python scripts/app_screenshots.py --url http://127.0.0.1:8765 --out outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--only", default=None, help="comma-separated shot names")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    errors = []

    shots = [
        # name, hash, action, viewport
        ("2d", "kind=grid2d&size=7&seed=11", "play", (1440, 900)),
        ("walls", "kind=walls&size=7&seed=5", "hint", (1440, 900)),
        ("islands", "kind=islands&size=3&seed=4", "exact", (1440, 900)),
        ("mask", "kind=mask&size=8&seed=2", "none", (1440, 900)),
        ("3d", "kind=grid3d&size=4&seed=3", "play3d", (1440, 1000)),
        ("4d", "kind=grid4d&size=3&seed=7", "play4d", (1440, 1000)),
        ("agent", "kind=grid2d&size=6&seed=21", "agent", (1440, 900)),
        ("agent_search", "kind=walls&size=7&seed=8", "agentsearch", (1440, 900)),
        ("agent_fail", "kind=walls&size=7&seed=8", "agentfail", (1440, 900)),
        ("agent3d", "kind=grid3d&size=3&seed=5", "agent", (1440, 1000)),
        ("mobile", "kind=grid2d&size=6&seed=11", "play", (390, 844)),
    ]
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        for name, h, action, vp in shots:
            if only and name not in only:
                continue
            ctx = browser.new_context(viewport={"width": vp[0], "height": vp[1]},
                                      color_scheme="dark" if args.dark else "light", device_scale_factor=1)
            ctx.add_init_script("try{localStorage.setItem('zip-onboarded','true')}catch(e){}")
            page = ctx.new_page()
            page.on("console", lambda m, n=name: m.type == "error" and errors.append(f"{n}: {m.text}"))
            page.on("pageerror", lambda e, n=name: errors.append(f"{n}: {e}"))
            page.goto(f"{args.url}/#{h}")
            page.wait_for_function("window.__zip && window.__zip.data && document.querySelector('#board svg')")
            page.wait_for_timeout(800)
            if action in ("play", "play3d", "play4d"):
                # play the first few moves of the solver's path by clicking cells
                page.evaluate("""async () => {
                  const S = window.__zip;
                  const r = await fetch('/api/solve/exact', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({puzzle: S.data, time_limit: 10})}).then(r => r.json());
                  const k = Math.floor(r.path.length * 0.45);
                  for (const v of r.path.slice(1, k)) window.__zipApi.push(v);
                }""")
            elif action == "hint":
                page.evaluate("""() => { const S = window.__zip; for (let i = 0; i < 6; i++) {
                   const head = S.path[S.path.length-1]; const nb = S.adj[head].filter(w => !S.visited[w] && !S.cpIndex.has(w));
                   if (nb.length) window.__zipApi.push(nb[nb.length-1]); } }""")
                page.click("#hintBtn")
                page.wait_for_timeout(2500)
            elif action == "exact":
                page.evaluate("window.__zipApi.selectTab('ai')")
                page.fill("#speed", "100")
                page.uncheck("#watchThink")
                page.click("#modeSeg button[data-mode=exact]")
                page.click("#runBtn")
                page.wait_for_function("window.__zip.path.length === window.__zip.n && !window.__zip.anim", timeout=30000)
            elif action in ("agent", "agentfail", "agentsearch"):
                page.evaluate("window.__zipApi.selectTab('ai')")
                page.fill("#speed", "100")
                page.uncheck("#watchThink")
                page.click(f"#modeSeg button[data-mode={'search' if action == 'agentsearch' else 'greedy'}]")
                page.click("#runBtn")
                page.wait_for_function("window.__zip.agent && !window.__zip.anim && !window.__zip.busy", timeout=60000)
                page.wait_for_timeout(1200)
            if action.endswith("3d") or name == "agent3d":
                page.wait_for_timeout(2500)
            page.screenshot(path=str(out / f"app_{name}{'_dark' if args.dark else ''}.png"), full_page=True)
            print("saved", name, page.evaluate("[window.__zip.path.length, window.__zip.n, document.getElementById('bannerMsg').innerText]"))
            ctx.close()
        browser.close()
    for e in errors:
        print("ERROR", e)


if __name__ == "__main__":
    main()
