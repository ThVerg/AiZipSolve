"""The fair fog-of-war planner (zipsolve.robots.fog) and its API wiring."""
from __future__ import annotations

import pytest

from zipsolve.generator import make_puzzle
from zipsolve.puzzle import Puzzle
from zipsolve.robots import fog, run_robot

CASES = [("grid2d", 6, 0), ("grid2d", 6, 1), ("grid2d", 6, 2), ("grid2d", 7, 0), ("grid2d", 7, 3),
         ("walls", 7, 1), ("walls", 7, 2), ("islands", 4, 1), ("islands", 4, 2),
         ("oneway", 6, 3), ("keys", 6, 3), ("hex", 3, 2), ("torus", 5, 0), ("portals", 6, 0)]

_cache: dict = {}


def fog_puzzle(kind, size, seed) -> Puzzle:
    key = (kind, size, seed)
    if key not in _cache:
        _cache[key] = make_puzzle(kind, size, rng=seed, unique=True, fog=True, time_limit=30)
    return _cache[key]


def replay(puzzle: Puzzle, trace: list):
    """Replay a fog trace; returns (final stack, list of (event index, revealed-before-push) checks)."""
    cps = set(puzzle.checkpoints)
    stack: list[int] = []
    revealed = {puzzle.checkpoints[0]}
    ui = {puzzle.checkpoints[0]}          # what the frontend's rule reveals (head within 2 steps)
    for ev in trace:
        if ev == "R":
            stack = []
            continue
        if isinstance(ev, dict):
            if ev["t"] == "reveal":
                revealed.update(ev["nodes"])
            elif ev["t"] == "path":
                stack = list(ev["p"])
            continue
        if ev >= 0:
            if ev in cps and stack:
                # stepping on a number: it must have been seen first (fairness)
                assert ev in revealed, f"stepped on hidden checkpoint {ev}"
            stack.append(ev)
            ui |= fog.near(puzzle.graph, ev) & cps
        else:
            del stack[len(stack) + ev:]
        # the run reveals exactly what the UI would (checked at the next event)
    return stack, revealed, ui


@pytest.mark.parametrize("kind,size,seed", CASES)
def test_fog_run_solves_fairly(kind, size, seed):
    p = fog_puzzle(kind, size, seed)
    assert fog.is_fog(p)
    r = fog.run(p, time_limit=20, seed=0)
    assert r["status"] == "solved" and r["solved"] and r["fog"]
    assert p.is_valid_solution(r["path"])
    assert not r["stats"]["fallback"]
    stack, revealed, ui = replay(p, r["trace"])
    assert stack == r["path"]
    assert revealed == ui                 # reveal events mirror the frontend's rule exactly
    kinds = {e.get("kind") for e in r["trace"] if isinstance(e, dict) and e["t"] == "note"}
    assert "intro" in kinds and "done" in kinds


def test_fog_backtracks_sometimes():
    total = 0
    for kind, size, seed in CASES:
        r = fog.run(fog_puzzle(kind, size, seed), time_limit=20, seed=0)
        total += r["stats"]["backtracks"]
        if r["stats"]["backtracks"]:
            notes = [e["msg"] for e in r["trace"] if isinstance(e, dict) and e.get("kind") == "backtrack"]
            assert notes and "backing up" in notes[0]
    assert total >= 2


def test_fog_deterministic_by_seed():
    p = fog_puzzle("grid2d", 7, 0)
    a = fog.run(p, seed=3)
    b = fog.run(p, seed=3)
    assert a["trace"] == b["trace"] and a["path"] == b["path"]


def _swap_hidden(p: Puzzle) -> tuple[Puzzle, set]:
    """Same board, two hidden (not initially visible, not the last) numbers swapped."""
    cps = list(p.checkpoints)
    near0 = fog.near(p.graph, cps[0])
    far = [i for i in range(1, len(cps) - 1) if cps[i] not in near0]
    i, j = far[0], far[-1]
    cps[i], cps[j] = cps[j], cps[i]
    return Puzzle(p.graph, cps), {cps[i], cps[j]}


@pytest.mark.parametrize("kind,size,seed", [("grid2d", 7, 0), ("walls", 7, 1), ("grid2d", 6, 2), ("islands", 4, 2)])
def test_decisions_only_use_revealed_numbers(kind, size, seed):
    """Metamorphic fairness: changing numbers the robot has not seen yet cannot change what it does
    until one of them is revealed."""
    p = fog_puzzle(kind, size, seed)
    q, swapped = _swap_hidden(p)
    a = fog.run(p, seed=1, time_limit=10)["trace"]
    b = fog.run(q, seed=1, time_limit=10, max_steps=4 * p.num_nodes)["trace"]
    k = 0
    while k < len(a):
        ev = a[k]
        assert k < len(b) and b[k] == ev, f"diverged at event {k} before the swapped numbers were seen"
        if isinstance(ev, dict) and ev["t"] == "reveal" and swapped & set(ev["nodes"]):
            break
        k += 1
    assert k < len(a)          # the swapped numbers do get revealed at some point


def test_knowledge_consistency_rules():
    p = fog_puzzle("grid2d", 6, 0)
    view = fog.FogView(p)
    view.show(p.checkpoints)
    know = view.knowledge()
    assert know.consistent(p.solution, p.num_nodes)
    assert know.end_known == p.checkpoints[-1]
    # nothing seen but the 1: every legal line through the clouds is still possible
    know0 = fog.FogView(p).knowledge()
    assert know0.end_known is None and know0.consistent(p.solution, p.num_nodes)


def test_fog_hint_uses_visible_info_only():
    p = fog_puzzle("grid2d", 6, 1)
    sol = p.solution
    # full information: on a unique puzzle the only consistent move is the solution's
    h = fog.fog_hint(p, sol[:5], seen=p.checkpoints)
    assert h["status"] == "next" and h["certain"] and h["next"] == sol[5]
    assert h["reason"].startswith("Based on what you can see")
    # at the start with the fog up, several moves are still possible: an honest "explore" hint
    h0 = fog.fog_hint(p, [sol[0]])
    assert h0["status"] == "next" and h0["fog"]
    if not h0["certain"]:
        assert "Not enough numbers visible yet" in h0["reason"]
    e = fog.fog_explain(p, [sol[0]])
    assert e["move"] == h0["next"] and e["technique"] == "fog"
    assert fog.fog_hint(p, sol)["status"] == "done"


def test_fog_hint_backtracks_on_impossible_line():
    p = fog_puzzle("grid2d", 6, 1)
    view_all = p.checkpoints
    # walk the solution, then take a wrong turn that full information rules out
    sol = p.solution
    from zipsolve.solver import legal_moves
    for k in range(3, len(sol) - 3):
        wrong = [m for m in legal_moves(p, sol[:k]) if m != sol[k]]
        if wrong:
            line = sol[:k] + [wrong[0]]
            h = fog.fog_hint(p, line, seen=view_all)
            assert h["status"] == "backtrack" and h["keep"] <= k and h["next"] == sol[h["keep"]]
            return
    pytest.skip("no wrong turn found")


def test_every_robot_is_fog_aware():
    p = fog_puzzle("grid2d", 6, 2)
    for bot in ("detective", "mcts", "gambler", "evolver"):
        r = run_robot(bot, p, time_limit=10, seed=0)
        assert r["fog"] and r["solved"] and r["robot"] == bot
        assert any(isinstance(e, dict) and e["t"] == "reveal" for e in r["trace"])


def test_api_routes_honour_fog():
    from fastapi.testclient import TestClient

    from zipsolve.app.server import create_app
    c = TestClient(create_app())
    p = fog_puzzle("grid2d", 6, 1)
    d = p.to_dict()
    d.pop("solution", None)
    assert d["meta"]["fog"] is True
    start = d["checkpoints"][0]
    for ep, body in [("/api/solve/exact", {"trace": True}), ("/api/solve/exact", {}),
                     ("/api/solve/rl", {"mode": "search", "trace": True}),
                     ("/api/solve/robot", {"robot": "detective"})]:
        r = c.post(ep, json={"puzzle": d, **body})
        assert r.status_code == 200, (ep, r.text)
        j = r.json()
        assert j["fog"] and j["solved"] and p.is_valid_solution(j["path"])
        assert any(isinstance(e, dict) and e.get("t") == "reveal" for e in j["trace"])
    # from the player's line, with what their board shows
    r = c.post("/api/solve/exact", json={"puzzle": d, "start_path": p.solution[:4], "trace": True,
                                          "fog_seen": [start]}).json()
    assert r["solved"] and r["start_len"] == 4
    h = c.post("/api/hint", json={"puzzle": d, "path": [start]}).json()
    assert h["fog"] and h["source"] == "fog" and h["status"] == "next"
    e = c.post("/api/hint/explain", json={"puzzle": d, "path": [start]}).json()
    assert e["fog"] and e["move"] == h["next"]
    # without fog nothing changes
    d2 = dict(d, meta={k: v for k, v in d["meta"].items() if k != "fog"})
    j = c.post("/api/solve/exact", json={"puzzle": d2}).json()
    assert j["solved"] and "fog" not in j
