"""Browser regressions (playwright + chromium; skipped when they are not installed).

* "New" after an AI run gives a fresh board: no robot starts on its own, no solver request fires
  (game page and AI show), and an interrupted "Show me" stops for good.
* Fog "Show me": numbers appear progressively as the robot's pen gets close (not all at once).
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

pw = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    import uvicorn

    from zipsolve.app.server import create_app
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
    th.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001 - no browser binary
            pytest.skip(f"chromium not available: {e}")
        yield b
        b.close()


def _page(browser):
    pg = browser.new_page(viewport={"width": 900, "height": 1000})
    pg.add_init_script("localStorage.setItem('zip-tut-done','true');localStorage.setItem('zip-first-hint','true');"
                       "localStorage.setItem('zip-tip-fog','true')")
    reqs: list[str] = []
    pg.on("request", lambda r: reqs.append(r.url) if "/api/solve" in r.url or "/api/policy" in r.url else None)
    errs: list[str] = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    return pg, reqs, errs


def test_lab_new_does_not_autorun(base_url, browser):
    pg, reqs, errs = _page(browser)
    pg.goto(base_url + "lab#play=classic&diff=easy&seed=1&bot=tortoise")
    pg.wait_for_selector("#watchBtn", state="visible", timeout=20000)
    pg.click("#watchBtn")
    time.sleep(1.5)                     # a watch is running (request made)
    assert reqs
    reqs.clear()
    pg.click("#newBtn")
    time.sleep(2.5)
    assert reqs == [], "pressing New started a robot"
    assert pg.is_visible("#watchBtn") and pg.is_visible("#dPre")
    assert not errs
    pg.close()


def test_game_new_cancels_show_me(base_url, browser):
    pg, reqs, errs = _page(browser)
    pg.goto(base_url + "#play=classic&diff=medium&seed=3")
    pg.wait_for_selector("#boardHost svg", timeout=20000)
    time.sleep(0.5)
    pg.click("#showMeBtn")
    time.sleep(1.2)
    reqs.clear()
    pg.evaluate("() => window.__zipGame.playMode('classic', 'medium', 5)")
    time.sleep(2.5)
    assert pg.evaluate("() => window.__zipGame.game().path.length") == 1
    assert not pg.evaluate("() => document.body.classList.contains('robot-on')")
    assert reqs == []
    assert not errs
    pg.close()


def test_fog_show_me_reveals_progressively(base_url, browser):
    pg, reqs, errs = _page(browser)
    pg.goto(base_url + "#play=fog&diff=normal&seed=0")
    pg.wait_for_selector("#boardHost svg .fog", timeout=20000)
    time.sleep(0.5)
    start = pg.eval_on_selector_all("#boardHost .cp.fogged", "e => e.length")
    assert start > 3
    pg.click("#showMeBtn")
    seen = set()
    t0 = time.time()
    while time.time() - t0 < 6:
        seen.add(pg.eval_on_selector_all("#boardHost .cp.fogged", "e => e.length"))
        time.sleep(0.1)
    assert any(0 < k < start for k in seen), f"fog did not lift progressively: {sorted(seen)}"
    assert any("/api/solve/exact" in u for u in reqs)
    assert not errs
    pg.close()
