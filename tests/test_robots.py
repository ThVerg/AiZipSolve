"""Strategy robots (zipsolve.robots): valid solutions, well-formed traces,
determinism, the /api/solve/robot endpoint and the Detective's teaching hints."""
from __future__ import annotations

import pytest

from zipsolve import solver
from zipsolve.generator import make_puzzle
from zipsolve.puzzle import Puzzle
from zipsolve.robots import ROBOTS, explain_next_move, listing, mathematician, resolve, run_robot

HAS_SAT = mathematician.available()
ALL = list(ROBOTS)


def _small():
    return [make_puzzle("grid2d", 5, rng=11, unique=True),
            make_puzzle("walls", 6, rng=12, unique=True),
            make_puzzle("islands", 2, rng=13, unique=True),
            make_puzzle("grid3d", 3, rng=14, unique=True)]


@pytest.fixture(scope="module")
def puzzles():
    return _small()


def _skip_sat(robot):
    if robot == "sat" and not HAS_SAT:
        pytest.skip("python-sat not installed")


def replay(puzzle: Puzzle, trace: list) -> list[int]:
    """Replay a robot trace; every displayed line must be a legal prefix."""
    assert trace and trace[0] == "R", trace[:3]
    stack: list[int] = []
    for ev in trace:
        if ev == "R":
            stack = []
        elif isinstance(ev, int) and not isinstance(ev, bool):
            if ev >= 0:
                stack.append(ev)
                assert solver.check_prefix(puzzle, stack) is None, (stack, solver.check_prefix(puzzle, stack))
            else:
                assert -ev <= len(stack)
                del stack[len(stack) + ev:]
        elif isinstance(ev, dict):
            t = ev.get("t")
            assert t in ("note", "path", "tree"), ev
            if t == "note":
                assert isinstance(ev["msg"], str) and ev["msg"].strip()
            elif t == "path":
                stack = [int(v) for v in ev["p"]]
                assert solver.check_prefix(puzzle, stack) is None
            else:
                assert isinstance(ev["nodes"], list)
                for nd in ev["nodes"]:
                    assert {"node", "n", "q", "p"} <= set(nd)
        else:
            raise AssertionError(f"bad event {ev!r}")
    return stack


@pytest.mark.parametrize("robot", ALL)
def test_robot_solves_small_puzzles(robot, puzzles):
    _skip_sat(robot)
    for p in puzzles:
        r = run_robot(robot, p, time_limit=30, trace=True, seed=3)
        assert r["robot"] == robot
        assert r["status"] == "solved", (robot, p.graph.kind, r["status"], r.get("stats"))
        assert r["solved"] is True
        assert p.check_solution(r["path"]) is None
        assert replay(p, r["trace"]) == r["path"]
        assert isinstance(r["steps"], list) and isinstance(r["seconds"], float)
        for key in ("nodes_expanded", "trace_truncated", "start_len"):
            assert key in r


@pytest.mark.parametrize("robot", ALL)
def test_robot_deterministic(robot, puzzles):
    _skip_sat(robot)
    p = puzzles[1]

    def strip(tr):   # captions of the SAT robot contain wall-clock times
        return [e for e in tr if not (isinstance(e, dict) and e.get("t") == "note")] if robot == "sat" else tr

    a = run_robot(robot, p, time_limit=30, seed=7)
    b = run_robot(robot, p, time_limit=30, seed=7)
    assert a["path"] == b["path"] and a["status"] == b["status"]
    assert strip(a["trace"]) == strip(b["trace"])


def test_trace_off(puzzles):
    r = run_robot("gambler", puzzles[0], time_limit=10, trace=False)
    assert r["trace"] is None and r["solved"]


def test_resolve_and_listing():
    assert resolve("Sage") == "mcts" and resolve("mathematician") == "sat"
    with pytest.raises(KeyError):
        resolve("nobody")
    ids = [r["id"] for r in listing()]
    assert ids == ["detective", "mcts", "evolver", "gambler", "sat"]


def test_sat_unavailable(monkeypatch, puzzles):
    monkeypatch.setattr(mathematician, "available", lambda: False)
    r = mathematician.run(puzzles[0], time_limit=5)
    assert r["status"] == "unavailable" and not r["solved"] and r["reason"]


@pytest.mark.skipif(not HAS_SAT, reason="python-sat not installed")
def test_sat_proves_unsat():
    from zipsolve.graph import grid
    # 3x3 grid, start and end on opposite colours of an odd board: impossible
    p = Puzzle(grid(3, 3), [0, 1])
    r = run_robot("sat", p, time_limit=10)
    assert r["status"] == "unsat" and not r["solved"]


def test_detective_notes_are_friendly(puzzles):
    r = run_robot("detective", puzzles[0], time_limit=20)
    notes = [e for e in r["trace"] if isinstance(e, dict) and e.get("t") == "note"]
    assert len(notes) >= 3
    for e in notes:
        assert isinstance(e["msg"], str) and len(e["msg"]) > 10
        assert "{" not in e["msg"] and "None" not in e["msg"]
    assert r["stats"]["techniques"]


def test_explain_next_move_follows_unique_solution(puzzles):
    for p in puzzles[:2]:
        assert solver.count_solutions(p, 2)[0] == 1
        sol = solver.solve(p).path
        for k in range(1, len(sol)):
            h = explain_next_move(p, sol[:k])
            assert h["status"] == "next", h
            assert h["move"] in solver.legal_moves(p, sol[:k])
            assert h["move"] == sol[k], (k, h)
            assert isinstance(h["reason"], str) and h["reason"].strip()
        assert explain_next_move(p, sol)["status"] == "done"


def test_explain_backtrack_and_invalid(puzzles):
    p = puzzles[0]
    sol = solver.solve(p).path
    # walk the greedy wrong way until the position is hopeless
    path = list(sol[:2])
    while True:
        ms = [m for m in solver.legal_moves(p, path) if m != sol[len(path)]] if len(path) < len(sol) else []
        if not ms:
            break
        path.append(ms[0])
        if solver.solve_from_prefix(p, path, time_limit=5).status == "unsat":
            break
    if solver.solve_from_prefix(p, path, time_limit=5).status == "unsat":
        h = explain_next_move(p, path)
        assert h["status"] == "backtrack", h
        assert 1 <= h["keep"] < len(path)
        assert h["move"] in solver.legal_moves(p, path[:h["keep"]])
        assert solver.solve_from_prefix(p, path[:h["keep"]] + [h["move"]], time_limit=5).status == "solved"
        assert h["reason"]
    bad = explain_next_move(p, [sol[1]])
    assert bad["status"] == "invalid"


# ---------------------------------------------------------------- endpoint
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from zipsolve.app.server import create_app
    return TestClient(create_app())


@pytest.mark.parametrize("robot", ALL)
def test_endpoint(client, robot, puzzles):
    _skip_sat(robot)
    p = puzzles[0]
    d = p.to_dict()
    r = client.post("/api/solve/robot", json={"puzzle": d, "robot": robot, "time_limit": 20, "trace": True})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["robot"] == robot and out["solved"] and p.check_solution(out["path"]) is None
    assert replay(p, out["trace"]) == out["path"]


def test_endpoint_errors_and_listing(client, puzzles):
    d = puzzles[0].to_dict()
    assert client.post("/api/solve/robot", json={"puzzle": d, "robot": "nobody"}).status_code == 400
    r = client.get("/api/robots")
    assert r.status_code == 200 and {x["id"] for x in r.json()["robots"]} == set(ROBOTS)


def test_explain_endpoint(client, puzzles):
    p = puzzles[0]
    sol = solver.solve(p).path
    r = client.post("/api/hint/explain", json={"puzzle": p.to_dict(), "path": sol[:3]})
    assert r.status_code == 200, r.text
    h = r.json()
    assert h["status"] == "next" and h["move"] == sol[3] and h["reason"]


# ---------------------------------------------------------------- new puzzle rules
def _new_kind(kind, size, seed):
    try:
        return make_puzzle(kind, size, rng=seed, unique=True, time_limit=30)
    except (ValueError, TypeError, KeyError) as e:   # kind not supported by this generator
        pytest.skip(f"{kind} puzzles unavailable: {e}")


@pytest.mark.parametrize("robot", ALL)
@pytest.mark.parametrize("kind,size", [("oneway", 5), ("keys", 6), ("hex", 3)])
def test_robots_follow_new_rules(robot, kind, size):
    _skip_sat(robot)
    p = _new_kind(kind, size, 21)
    r = run_robot(robot, p, time_limit=30, seed=1)
    assert r["solved"], (robot, kind, r["status"])
    assert p.check_solution(r["path"]) is None
    assert replay(p, r["trace"]) == r["path"]
