"""Record the README demo GIF (needs playwright + chromium and Pillow).

    python scripts/build_site.py --out dist && python -m http.server -d dist 8780 &
    python scripts/record_demo.py --url http://127.0.0.1:8780/ --out docs/images/demo.gif

Works on the static build or the local app (python -m zipsolve.app). A scripted player draws
today's daily puzzle, wins, then the AI show's robot race plays. Frames come from the Chrome
DevTools screencast (with timestamps) and are assembled into an optimised GIF with Pillow.
"""
from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

CURSOR = """
addEventListener('DOMContentLoaded', () => {
  const d = document.createElement('div');
  d.id = '__demoCursor';
  d.style.cssText = 'position:fixed;left:0;top:0;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;' +
    'background:rgba(255,255,255,.55);border:2.5px solid rgba(30,30,40,.75);box-shadow:0 2px 8px rgba(0,0,0,.25);' +
    'pointer-events:none;z-index:99999;opacity:0;transition:opacity .2s,transform .12s';
  document.body.appendChild(d);
  addEventListener('pointermove', (e) => { d.style.left = e.clientX + 'px'; d.style.top = e.clientY + 'px'; d.style.opacity = 1; }, true);
  addEventListener('pointerdown', () => { d.style.transform = 'scale(.8)'; }, true);
  addEventListener('pointerup', () => { d.style.transform = ''; }, true);
});
try { localStorage.setItem('zip-tut-done', 'true'); localStorage.setItem('zip-first-hint', 'true'); } catch (e) {}
"""


PICK_RACE = """async (u) => {
    // a race worth watching: the Rookie gets stuck, the Scout backs out of dead ends and wins
    const { api } = await import(new URL(u, document.baseURI).href);
    const { MODES, genParams } = await import(new URL(u.replace('util.js', 'modes.js'), document.baseURI).href);
    let fallback = null;
    for (const [mode, diff] of [['islands', 'normal'], ['walls', 'normal'], ['classic', 'hard']]) {
        for (let seed = 0; seed < 12; seed++) {
            try {
                const g = await api('/api/generate', genParams(MODES[mode].diffs[diff], seed));
                if (g.seed !== seed && seed > 0) break;
                const run = (m) => api('/api/solve/rl', { puzzle: g.puzzle, mode: m, budget: 4000, time_limit: 20, compare: false, trace: m !== 'greedy' });
                const [r, s] = [await run('greedy'), await run('search')];
                const link = `#vs=rookie,scout&play=${mode}&diff=${diff}&seed=${seed}`;
                if (!r.solved && s.solved && (s.backtracks || 0) > 0) return link;
                if (!r.solved && s.solved && !fallback) fallback = link;
            } catch (e) { break; }
        }
    }
    return fallback;
}"""


GLIDE = """async ({ from, to, ms }) => {
    const t0 = performance.now();
    for (;;) {
        const f = Math.min(1, (performance.now() - t0) / ms), e = f * f * (3 - 2 * f);
        const x = from[0] + (to[0] - from[0]) * e, y = from[1] + (to[1] - from[1]) * e;
        document.dispatchEvent(new PointerEvent('pointermove', { clientX: x, clientY: y, bubbles: true }));
        if (f >= 1) break;
        await new Promise((r) => requestAnimationFrame(r));
    }
}"""
# drag along the solution: ~40 ms a cell, a little slower at turns, short pauses at some numbers
PLAYER = """async ({ path, seed }) => {
    let s = seed >>> 0;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const board = window.__zipGame.board(), svg = document.querySelector('#boardHost svg');
    const cps = new Set(window.__zipGame.G.data.checkpoints);
    const pts = path.map((v) => board.clientOf(v));
    const fire = (type, p, buttons) => svg.dispatchEvent(new PointerEvent(type, { clientX: p.x, clientY: p.y, pointerId: 1,
        pointerType: 'mouse', isPrimary: true, button: 0, buttons, bubbles: true, cancelable: true }));
    const frame = () => new Promise((r) => requestAnimationFrame(r));
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const glide = async (a, b, ms, buttons) => {
        const t0 = performance.now();
        for (;;) {
            const f = Math.min(1, (performance.now() - t0) / ms);
            fire('pointermove', { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f }, buttons);
            if (f >= 1) return;
            await frame();
        }
    };
    await glide({ x: pts[0].x + 60, y: pts[0].y + 80 }, pts[0], 350, 0);
    await sleep(120);
    fire('pointerdown', pts[0], 1);
    for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i], p = pts[i - 2];
        const turn = p && (b.x - a.x) * (a.y - p.y) !== (b.y - a.y) * (a.x - p.x);
        await glide(a, b, 28 + rnd() * 16 + (turn ? 26 : 0), 1);
        if (cps.has(path[i]) && rnd() < 0.35) await sleep(90 + rnd() * 70);
    }
    fire('pointerup', pts[pts.length - 1], 0);
    await glide(pts[pts.length - 1], { x: innerWidth - 50, y: innerHeight - 40 }, 400, 0);
}"""


def record(url: str, width: int, height: int, lab_hash: str | None) -> tuple[list[tuple[float, bytes]], list[tuple[float, float]]]:
    frames: list[tuple[float, bytes]] = []
    cuts: list[tuple[float, float]] = []
    marks: list[tuple[str, float]] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
        ctx.add_init_script(CURSOR)
        page = ctx.new_page()
        page.goto(url)
        page.wait_for_selector("#heroPlay")
        page.wait_for_timeout(600)
        util = "static/play/util.js" if page.evaluate("!!window.ZIP_STATIC") else "/static/play/util.js"
        if not lab_hash:
            lab_hash = page.evaluate(PICK_RACE, util) or "#vs=rookie,scout&play=islands&diff=normal&seed=0"
            print("race:", lab_hash)
        cdp = ctx.new_cdp_session(page)

        def on_frame(ev):
            frames.append((ev["metadata"]["timestamp"], base64.b64decode(ev["data"])))
            try:
                cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
            except Exception:  # noqa: BLE001
                pass
        cdp.on("Page.screencastFrame", on_frame)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "everyNthFrame": 1})
        page.wait_for_timeout(500)

        # 1) today's puzzle, drawn by a scripted player (in-page pointer events: smooth, exact pacing)
        hero = page.locator("#heroPlay").bounding_box()
        hx, hy = hero["x"] + hero["width"] / 2, hero["y"] + hero["height"] / 2
        page.evaluate(GLIDE, {"from": [width * 0.45, height * 0.6], "to": [hx, hy], "ms": 450})
        page.wait_for_timeout(150)
        page.mouse.click(hx, hy)
        page.wait_for_function("() => window.__zipGame.G.P && !window.__zipGame.G.busy && document.querySelector('#boardHost svg')")
        page.wait_for_timeout(500)
        sol = page.evaluate("""async (u) => {
            const { api } = await import(new URL(u, document.baseURI).href);
            const G = window.__zipGame.G;
            return (await api('/api/solve/exact', { puzzle: G.data, time_limit: 20 })).path;
        }""", util)
        marks.append(("draw", time.time()))
        page.evaluate(PLAYER, {"path": sol, "seed": 7})
        marks.append(("drawn", time.time()))
        page.wait_for_selector("#sheet:not([hidden])", timeout=15000)
        marks.append(("win", time.time()))
        page.wait_for_timeout(950)

        # 2) quick cut: the AI show's robot race
        t_nav = time.time()
        page.goto(url.rstrip("/") + ("/lab.html" if util.startswith("static") else "/lab") + lab_hash)
        page.wait_for_selector("#raceBtn:visible")
        page.wait_for_timeout(250)
        cuts.append((t_nav, time.time()))    # the page load: a hard cut
        page.wait_for_timeout(450)
        marks.append(("race", time.time()))
        page.click("#raceBtn")
        page.wait_for_selector("#fRun:not([hidden])")
        t_cd = page.evaluate("performance.timeOrigin + performance.now()") / 1000
        page.wait_for_timeout(2250)          # countdown: cut out of the GIF below, "3" and "Go!" stay
        cuts.append((t_cd + 0.6, t_cd + 1.95))
        page.click('#fRun [data-speed="2.5"]')
        page.wait_for_selector("#fDone:not([hidden])", timeout=40000)
        marks.append(("raced", time.time()))
        page.wait_for_timeout(1000)
        cdp.send("Page.stopScreencast")
        b.close()
    if frames:
        print("  ".join(f"{k} {t - frames[0][0]:.1f}s" for k, t in marks))
    return frames, cuts


def to_gif(frames: list[tuple[float, bytes]], out: Path, width: int, fps: float, colors: int,
           cuts: list[tuple[float, float]] = ()) -> None:
    t0, t_end = frames[0][0], frames[-1][0] + 0.5
    imgs = []
    # resample the (irregular) screencast onto a fixed frame rate, skipping the cut ranges
    k, t = 0, t0
    while t < t_end:
        if any(a <= t < b for a, b in cuts):
            t += 1.0 / fps
            continue
        while k + 1 < len(frames) and frames[k + 1][0] <= t:
            k += 1
        imgs.append(k)
        t += 1.0 / fps
    runs: list[list[int]] = []   # [frame index, count]
    for k in imgs:
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    decoded = {}
    for k, _ in runs:
        im = Image.open(io.BytesIO(frames[k][1])).convert("RGB")
        h = round(im.height * width / im.width)
        decoded[k] = im.resize((width, h), Image.LANCZOS)
    # a palette per frame: one shared palette posterised the dimmed win-screen backdrop
    out_frames = [decoded[k].quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                      dither=Image.Dither.NONE) for k, _ in runs]
    durations = [round(1000 * n / fps) for _, n in runs]
    out.parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=durations, loop=0,
                       optimize=True, disposal=1)
    print(f"{out}: {len(out_frames)} frames, {sum(durations) / 1000:.1f} s, {out.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8780/")
    ap.add_argument("--out", type=Path, default=Path("docs/images/demo.gif"))
    ap.add_argument("--width", type=int, default=720, help="GIF width")
    ap.add_argument("--viewport", default="960x720")
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--lab", default=None, help="AI show race link (default: search for a good one)")
    a = ap.parse_args()
    vw, vh = map(int, a.viewport.split("x"))
    frames, cuts = record(a.url, vw, vh, a.lab)
    to_gif(frames, a.out, a.width, a.fps, a.colors, cuts)


if __name__ == "__main__":
    main()
