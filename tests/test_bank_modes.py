"""The shipped "more modes" bank (scripts/build_bank_modes.py) and the weekly Architect endpoint."""
import json
from pathlib import Path

import pytest

from zipsolve.puzzle import Puzzle
from zipsolve.solver import count_solutions

BANK = Path(__file__).resolve().parents[1] / "zipsolve" / "app" / "static" / "bank"
NEW = ("portals", "torus", "hex", "tri", "oneway", "overpass", "keys", "cubesurf")
STRATS = ("detective", "mcts", "evolver", "gambler", "sat")


def _idx():
    return json.loads((BANK / "index.json").read_text())


pytestmark = pytest.mark.skipif(not (BANK / "index.json").exists() or "more_modes" not in _idx(),
                                reason="more-modes bank not built")


def _replay(trace):
    stack = []
    for e in trace:
        if e == "R":
            stack = []
        elif isinstance(e, int):
            if e >= 0:
                stack.append(e)
            else:
                del stack[len(stack) + e:]
        elif isinstance(e, dict) and e.get("t") == "path":
            stack = list(e["p"])
    return stack


def test_more_mode_pools():
    idx = _idx()
    ids = set()
    for mode in NEW:
        for diff in ("medium", "hard"):
            info = idx["modes"][mode][diff]
            lst = json.loads((BANK / info["file"]).read_text())
            assert len(lst) == info["count"] >= 30
            for e in lst:
                assert e["mode"] == mode and e["diff"] == diff and e["id"] not in ids and e["puzzle"]["kind"] == mode
                ids.add(e["id"])
            e = lst[0]
            p = Puzzle.from_dict(e["puzzle"])
            assert p.is_valid_solution(e["puzzle"]["solution"])
            assert count_solutions(p, 2, time_limit=30) == (1, "complete")
            runs = json.loads((BANK / "robots" / f"{e['id']}.json").read_text())
            assert set(runs) >= {"rookie", "scout", "grandmaster", "tortoise", *STRATS}
            for rb in STRATS:
                r = runs[rb]
                assert r["path"][0] == p.checkpoints[0]
                if r.get("trace"):
                    assert _replay(r["trace"]) == r["path"], (mode, rb)
                if r["solved"]:
                    assert p.is_valid_solution(r["path"])


def test_coop_pool():
    from zipsolve import coop
    info = _idx()["modes"]["coop"]["medium"]
    lst = json.loads((BANK / info["file"]).read_text())
    assert len(lst) == info["count"] >= 30
    for e in lst[:5]:
        p = coop.api_load(e["puzzle"])
        paths = e["puzzle"]["meta"]["coop"]["solution"]["paths"]
        assert coop.check(p, paths) is None


def test_bank_serves_coop_and_weekly():
    from fastapi.testclient import TestClient
    from zipsolve.app.server import create_app
    c = TestClient(create_app())
    r = c.post("/api/generate", json={"kind": "coop", "size": 6, "unique": True, "seed": 3, "bank": "coop-medium"}).json()
    assert r["source"] == "bank" and "solution" not in r["puzzle"]["meta"]["coop"]
    h = c.post("/api/hint", json={"puzzle": r["puzzle"], "paths": [[], []], "id": r["id"]}).json()
    assert h["status"] == "next"
    if not (BANK / "architect" / "weekly.json").exists():
        return
    w = c.get("/api/architect/weekly?week=2026-W40").json()
    assert w["weekly"]["week"] == "2026-W40" and w["puzzle"]["checkpoints"] and "solution" not in w["puzzle"]
    assert c.get("/api/architect/weekly?week=2031-W07").json()["weekly"]["bank_week"].startswith("202")
    assert c.get("/api/architect/weekly").status_code == 200
