"""Web app API tests (FastAPI TestClient)."""
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from zipsolve.app.server import create_app  # noqa: E402
from zipsolve.puzzle import Puzzle  # noqa: E402

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
HAS_MODELS = any(CKPT_DIR.glob("*.pt"))

KINDS = [("grid2d", 5), ("walls", 5), ("islands", 2), ("mask", 5), ("grid3d", 3), ("grid4d", 2)]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def gen(client, kind, size, seed=1, **kw):
    r = client.post("/api/generate", json={"kind": kind, "size": size, "seed": seed, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "Zip" in r.text
    assert r.headers.get("cache-control") == "no-cache"
    assert "/static/game.js" in r.text          # the casual game
    for f in ("app.js", "game.js", "game.css", "style.css", "view3d.js"):
        assert client.get(f"/static/{f}").status_code == 200


def test_lab_page(client):
    # the power-user play page (AI vision, solver replays, compare, custom options) lives at /lab
    r = client.get("/lab")
    assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache"
    assert "/static/app.js" in r.text and 'id="pane-ai"' in r.text


@pytest.mark.parametrize("kind,size", KINDS)
def test_generate_each_kind(client, kind, size):
    res = gen(client, kind, size, seed=7, include_solution=True)
    assert "solution" not in res["puzzle"] or res["puzzle"].get("solution") is None
    p = Puzzle.from_dict(res["puzzle"])
    assert p.num_nodes == res["num_nodes"]
    assert p.is_valid_solution(res["solution"])
    # hidden by default
    res2 = gen(client, kind, size, seed=7)
    assert "solution" not in res2
    assert res2["puzzle"]["checkpoints"] == res["puzzle"]["checkpoints"]  # seeded => reproducible


def test_generate_errors(client):
    assert client.post("/api/generate", json={"kind": "nope", "size": 3}).status_code == 400


def test_check(client):
    res = gen(client, "grid2d", 4, include_solution=True)
    P, sol = res["puzzle"], res["solution"]
    ok = client.post("/api/check", json={"puzzle": P, "path": sol}).json()
    assert ok["valid"] and ok["legal_prefix"]
    part = client.post("/api/check", json={"puzzle": P, "path": sol[:5]}).json()
    assert not part["valid"] and part["legal_prefix"]
    bad = client.post("/api/check", json={"puzzle": P, "path": sol[::-1]}).json()
    assert not bad["valid"] and not bad["legal_prefix"]


@pytest.mark.parametrize("kind,size", KINDS)
def test_exact_solve(client, kind, size):
    P = gen(client, kind, size, seed=3)["puzzle"]
    r = client.post("/api/solve/exact", json={"puzzle": P, "time_limit": 20}).json()
    assert r["status"] == "solved", r
    assert Puzzle.from_dict(P).is_valid_solution(r["path"])


def test_exact_solve_from_prefix(client):
    res = gen(client, "grid2d", 5, include_solution=True)
    P, sol = res["puzzle"], res["solution"]
    r = client.post("/api/solve/exact", json={"puzzle": P, "start_path": sol[:6]}).json()
    assert r["status"] == "solved" and r["path"][:6] == sol[:6]
    assert Puzzle.from_dict(P).is_valid_solution(r["path"])


def test_hint_next_and_backtrack(client):
    res = gen(client, "grid2d", 5, seed=11, include_solution=True)
    P, sol, pid = res["puzzle"], res["solution"], res["id"]
    p = Puzzle.from_dict(P)
    # on the stored solution's prefix: suggests its next cell
    h = client.post("/api/hint", json={"puzzle": P, "path": sol[:4], "id": pid}).json()
    assert h["status"] == "next" and h["next"] == sol[4]
    # without the id the solver works it out; the hinted move must keep the path completable
    h = client.post("/api/hint", json={"puzzle": P, "path": sol[:4]}).json()
    assert h["status"] == "next" and p.graph.has_edge(sol[3], h["next"])
    # build a dead prefix: walk greedily until the solver says it can't be completed
    from zipsolve.app.engine import complete_prefix, validate_prefix
    dead = None
    path = [p.checkpoints[0]]
    for _ in range(p.num_nodes):
        head = path[-1]
        moves = [w for w in p.graph.neighbors[head] if w not in path
                 and validate_prefix(p, path + [w]) is None]
        if not moves:
            break
        # prefer a move that the solver rejects
        for w in moves:
            if complete_prefix(p, path + [w], 5).status == "unsat":
                dead = path + [w]
                break
        if dead:
            break
        path.append(moves[0])
    if dead is None:
        pytest.skip("could not construct a dead prefix on this puzzle")
    h = client.post("/api/hint", json={"puzzle": P, "path": dead, "id": pid}).json()
    assert h["status"] == "backtrack", h
    k = h["keep"]
    assert 1 <= k < len(dead)
    assert complete_prefix(p, dead[:k] + [h["next"]], 5).status == "solved"


def test_hint_complete(client):
    res = gen(client, "grid2d", 4, include_solution=True)
    h = client.post("/api/hint", json={"puzzle": res["puzzle"], "path": res["solution"]}).json()
    assert h["status"] == "done"


def test_models_endpoint(client):
    r = client.get("/api/models").json()
    assert isinstance(r["models"], list)
    if HAS_MODELS:
        assert sum(m["default"] for m in r["models"]) == 1


def test_no_models(tmp_path):
    c = TestClient(create_app(tmp_path))
    assert c.get("/api/models").json()["models"] == []
    P = gen(c, "grid2d", 4)["puzzle"]
    assert c.post("/api/solve/rl", json={"puzzle": P}).status_code == 404


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
@pytest.mark.parametrize("mode", ["greedy", "search"])
@pytest.mark.parametrize("kind,size", [("grid2d", 4), ("islands", 2), ("grid3d", 2)])
def test_rl_solve(client, mode, kind, size):
    P = gen(client, kind, size, seed=5)["puzzle"]
    r = client.post("/api/solve/rl", json={"puzzle": P, "mode": mode, "budget": 3000}).json()
    p = Puzzle.from_dict(P)
    assert r["path"][0] == p.checkpoints[0]
    assert len(r["steps"]) == len(r["path"]) - r["start_len"]
    for s, v in zip(r["steps"], r["path"][r["start_len"]:]):
        assert s["node"] == v and 0.0 <= s["p"] <= 1.0 and s["top"][0][1] >= s["p"] - 1e-3
    if r["solved"]:
        assert p.is_valid_solution(r["path"])
    else:
        assert r["stuck_at"] == len(r["path"])
    if mode == "search":
        assert r["nodes_expanded"] <= 3000


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
def test_rl_solve_from_prefix(client):
    res = gen(client, "grid2d", 4, include_solution=True)
    P, sol = res["puzzle"], res["solution"]
    r = client.post("/api/solve/rl", json={"puzzle": P, "mode": "search", "start_path": sol[:5]}).json()
    assert r["path"][:5] == sol[:5] and r["start_len"] == 5
    bad = client.post("/api/solve/rl", json={"puzzle": P, "start_path": sol[::-1]})
    assert bad.status_code == 400


# ---- malformed input: always a 4xx, never a server error --------------------
@pytest.fixture(scope="module")
def raising_client():
    # raise_server_exceptions=False turns an uncaught exception into a 500 we can assert on
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize("kind,options", [
    ("walls", {"walls_frac": "oops"}), ("walls", {"walls_frac": [1]}), ("walls", {"walls_frac": True}),
    ("islands", {"jumps": []}), ("islands", {"jumps": "3"}), ("islands", {"decoys": 1.5}),
    ("mask", {"fill": {"a": 1}}),
])
def test_generate_rejects_bad_options(raising_client, kind, options):
    r = raising_client.post("/api/generate", json={"kind": kind, "size": 3, "seed": 1, "options": options})
    assert r.status_code == 422, r.text


def test_generate_accepts_integral_float_option(client):
    gen(client, "islands", 3, options={"jumps": 5.0, "decoys": 0})


def _tiny():
    return {"kind": "grid", "coords": [[0, 0], [0, 1]], "edges": [[0, 1]], "meta": {},
            "checkpoints": [0, 1]}


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(checkpoints=[0, "1"]),
    lambda d: d.update(checkpoints=[0, 1.0]),
    lambda d: d.update(checkpoints=[0, 0]),
    lambda d: d.update(checkpoints=[0]),
    lambda d: d.update(checkpoints=[0, 5]),
    lambda d: d.update(checkpoints="01"),
    lambda d: d.update(edges=[[0, "1"]]),
    lambda d: d.update(edges=[[0, 7]]),
    lambda d: d.update(edges=[[0]]),
    lambda d: d.update(coords=[[0, 0], [0]]),
    lambda d: d.update(coords=[["a", 0], [0, 1]]),
    lambda d: d.update(coords=[]),
    lambda d: d.update(meta=[1]),
    lambda d: d.pop("edges"),
])
@pytest.mark.parametrize("endpoint,extra", [
    ("/api/check", {"path": [0, 1]}), ("/api/solve/exact", {}), ("/api/hint", {"path": [0]}),
])
def test_bad_puzzles_are_4xx(raising_client, mutate, endpoint, extra):
    d = _tiny()
    mutate(d)
    r = raising_client.post(endpoint, json={"puzzle": d, **extra})
    assert 400 <= r.status_code < 500, (r.status_code, r.text)


@pytest.mark.parametrize("start", [[99], [1], [-1], [0, 99], [0, 0]])
def test_exact_rejects_invalid_prefix(client, start):
    r = client.post("/api/solve/exact", json={"puzzle": _tiny(), "start_path": start})
    assert r.status_code == 400, r.text


def test_exact_accepts_single_node_start(client):
    r = client.post("/api/solve/exact", json={"puzzle": _tiny(), "start_path": [0]})
    assert r.status_code == 200 and r.json()["path"] == [0, 1]


def test_generate_timeout_is_clear_422(client):
    r = client.post("/api/generate", json={"kind": "grid3d", "size": 5, "seed": 1, "unique": True,
                                           "time_limit": 1.0})
    assert r.status_code == 422 and "time limit" in r.json()["detail"]


# ---- hybrid mode: exact solver + GNN move ordering --------------------------
@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
@pytest.mark.parametrize("kind,size", [("grid2d", 5), ("islands", 2), ("grid3d", 3), ("grid4d", 2)])
def test_rl_hybrid(client, kind, size):
    P = gen(client, kind, size, seed=5)["puzzle"]
    r = client.post("/api/solve/rl", json={"puzzle": P, "mode": "hybrid", "time_limit": 20})
    assert r.status_code == 200, r.text
    r = r.json()
    p = Puzzle.from_dict(P)
    assert r["mode"] == "hybrid" and r["solved"] and p.is_valid_solution(r["path"])
    assert len(r["steps"]) == len(r["path"]) - r["start_len"]
    for s, v in zip(r["steps"], r["path"][r["start_len"]:]):
        assert s["node"] == v and 0.0 <= s["p"] <= 1.0
    h = r["hybrid"]
    for k in ("inference_calls", "cache_hits", "hook_calls", "attempts", "inference_seconds", "search_seconds"):
        assert k in h
    assert h["inference_calls"] + h["cache_hits"] <= h["hook_calls"]
    assert r["exact"]["solved"] and p.is_valid_solution(r["exact"]["path"])
    assert r["exact"]["nodes_expanded"] >= 0 and r["exact"]["seconds"] >= 0


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
def test_rl_hybrid_from_prefix_and_no_compare(client):
    res = gen(client, "grid2d", 5, include_solution=True)
    P, sol = res["puzzle"], res["solution"]
    r = client.post("/api/solve/rl", json={"puzzle": P, "mode": "hybrid", "start_path": sol[:6],
                                           "compare": False}).json()
    assert r["solved"] and r["path"][:6] == sol[:6] and r["start_len"] == 6 and "exact" not in r
    assert len(r["steps"]) == len(r["path"]) - 6
    assert client.post("/api/solve/rl", json={"puzzle": P, "mode": "hybrid",
                                              "start_path": sol[::-1]}).status_code == 400
    assert client.post("/api/solve/rl", json={"puzzle": P, "mode": "nope"}).status_code == 400


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
def test_rl_hybrid_unsat(client):
    # 3x3 grid, corner -> edge-middle: impossible by parity; the hybrid proves it
    coords = [[r, c] for r in range(3) for c in range(3)]
    edges = [[r * 3 + c, r * 3 + c + 1] for r in range(3) for c in range(2)] + \
            [[r * 3 + c, (r + 1) * 3 + c] for r in range(2) for c in range(3)]
    P = {"kind": "grid2d", "coords": coords, "edges": edges, "checkpoints": [0, 1], "meta": {}}
    r = client.post("/api/solve/rl", json={"puzzle": P, "mode": "hybrid"}).json()
    assert r["status"] == "unsat" and not r["solved"] and r["exact"]["status"] == "unsat"


# ---- play page: pages, presets / daily, policy view, search traces, model names ----------
STATIC_DIR = Path(__file__).resolve().parents[1] / "zipsolve" / "app" / "static"


def test_pages_and_play_modules(client):
    for f in ("theme.js", "play/board.js", "play/ai.js", "play/util.js", "play/sound.js", "play/rules.js", "play/fx.js"):
        assert client.get(f"/static/{f}").status_code == 200, f
    for page in ("dashboard", "editor"):
        r = client.get(f"/{page}")
        if (STATIC_DIR / f"{page}.html").is_file():
            assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache"
        else:
            assert r.status_code == 404


def test_presets_and_daily(client):
    pr = client.get("/api/presets").json()
    ids = [p["id"] for p in pr["presets"]]
    assert ids == ["easy", "medium", "hard", "expert", "insane"]
    a = client.get("/api/daily", params={"difficulty": "easy", "date": "2026-03-01"}).json()
    b = client.get("/api/daily", params={"difficulty": "easy", "date": "2026-03-01"}).json()
    c = client.get("/api/daily", params={"difficulty": "easy", "date": "2026-03-02"}).json()
    assert a["puzzle"] == b["puzzle"] and a["seed"] == b["seed"] and "solution" not in a
    assert a["daily"] == {"date": "2026-03-01", "number": 60, "difficulty": "easy", "label": "Easy"}
    assert c["seed"] != a["seed"]
    # the daily id gives solution-backed hints
    P = a["puzzle"]
    h = client.post("/api/hint", json={"puzzle": P, "path": [P["checkpoints"][0]], "id": a["id"]}).json()
    assert h["status"] == "next" and h["source"] == "solution"
    assert client.get("/api/daily", params={"difficulty": "nope"}).status_code == 400
    assert client.get("/api/daily", params={"difficulty": "easy", "date": "yesterday"}).status_code == 422


def _replay(trace):
    """Client-side replay of a compressed trace: returns the final stack."""
    stack = []
    for e in trace:
        if e == "R":
            stack = []
        elif e >= 0:
            stack.append(e)
        else:
            del stack[len(stack) + e:]
    return stack


def test_exact_trace(client):
    P = gen(client, "grid2d", 5, seed=4)["puzzle"]
    r = client.post("/api/solve/exact", json={"puzzle": P, "trace": True}).json()
    assert r["solved"] and not r["trace_truncated"]
    assert r["trace"][0] == "R" and _replay(r["trace"]) == r["path"]
    assert Puzzle.from_dict(P).is_valid_solution(r["path"])
    # from a prefix: the replay still ends on the full path
    sol = r["path"]
    r2 = client.post("/api/solve/exact", json={"puzzle": P, "trace": True, "start_path": sol[:5]}).json()
    assert r2["solved"] and _replay(r2["trace"]) == r2["path"] and r2["path"][:5] == sol[:5]
    # unsat: parity-impossible 3x3 corner -> edge-middle
    coords = [[rr, cc] for rr in range(3) for cc in range(3)]
    edges = [[rr * 3 + cc, rr * 3 + cc + 1] for rr in range(3) for cc in range(2)] + \
            [[rr * 3 + cc, (rr + 1) * 3 + cc] for rr in range(2) for cc in range(3)]
    u = client.post("/api/solve/exact", json={"puzzle": {"kind": "grid2d", "coords": coords, "edges": edges,
                                                         "checkpoints": [0, 1], "meta": {}}, "trace": True}).json()
    assert u["status"] == "unsat" and not u["solved"]


def test_compress_trace():
    from zipsolve.app.engine import compress_trace
    assert compress_trace(["R", 0, 1, 2, -1, -1, 3, -1, "R", 0]) == ["R", 0, 1, 2, -2, 3, -1, "R", 0]


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
@pytest.mark.parametrize("mode", ["search", "hybrid"])
def test_rl_trace(client, mode):
    P = gen(client, "grid2d", 5, seed=9)["puzzle"]
    r = client.post("/api/solve/rl", json={"puzzle": P, "mode": mode, "trace": True, "compare": False}).json()
    assert isinstance(r["trace"], list) and r["trace"][0] == "R"
    if r["solved"] and not r["trace_truncated"]:
        assert _replay(r["trace"]) == r["path"]
    # no trace unless asked
    r = client.post("/api/solve/rl", json={"puzzle": P, "mode": mode, "compare": False}).json()
    assert "trace" not in r


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
def test_policy_view(client):
    res = gen(client, "grid2d", 5, seed=2, include_solution=True)
    P, sol = res["puzzle"], res["solution"]
    p = Puzzle.from_dict(P)
    r = client.post("/api/policy", json={"puzzle": P, "path": sol[:4]})
    assert r.status_code == 200, r.text
    r = r.json()
    assert r["model"] and r["head"] == sol[3] and not r["done"]
    legal = r["legal"]
    assert legal and abs(sum(pp for _, pp in legal) - 1) < 1e-2
    assert all(p.graph.has_edge(sol[3], v) and v not in sol[:4] for v, _ in legal)
    assert [x[1] for x in legal] == sorted((x[1] for x in legal), reverse=True)
    heat = {int(k): v for k, v in r["heat"].items()}
    assert heat and set(heat) <= set(range(p.num_nodes)) - set(sol[:4])
    assert max(heat.values()) == pytest.approx(1.0) and min(heat.values()) >= 0
    assert r["completable"] == "solved"          # a prefix of a solution
    assert r["greedy"]["status"] in ("solved", "stuck", "timeout")
    assert r["winnable"] is None or 0 <= r["winnable"] <= 1
    assert isinstance(r["value"], float)
    # no heat / checks when not asked
    r2 = client.post("/api/policy", json={"puzzle": P, "path": sol[:4], "full": False, "check": False}).json()
    assert r2["heat"] == {} and "completable" not in r2
    # finished board
    r3 = client.post("/api/policy", json={"puzzle": P, "path": sol}).json()
    assert r3["done"] and r3["legal"] == []
    # empty path = start position
    r4 = client.post("/api/policy", json={"puzzle": P, "path": []}).json()
    assert r4["head"] == P["checkpoints"][0]


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
def test_policy_errors(raising_client):
    res = gen(raising_client, "grid2d", 4, seed=1, include_solution=True)
    P, sol = res["puzzle"], res["solution"]
    assert raising_client.post("/api/policy", json={"puzzle": P, "path": sol[::-1]}).status_code == 400
    assert raising_client.post("/api/policy", json={"puzzle": P, "path": [0], "model": "nope.pt"}).status_code == 404
    bad = _tiny()
    bad["checkpoints"] = [0, 0]
    assert raising_client.post("/api/policy", json={"puzzle": bad, "path": [0]}).status_code == 422


@pytest.mark.parametrize("name", ["../x.pt", "/etc/passwd.pt", "a/../../b.pt", "a/./b.pt", "a\\b.pt", "C:/x.pt",
                                  "a//b.pt", "x.txt", "", "..", "remote/../../secret.pt"])
def test_model_name_rejects_traversal(name):
    from zipsolve.app.engine import ModelRegistry
    with pytest.raises(ValueError):
        ModelRegistry.check_name(name)


def test_model_traversal_is_404(tmp_path, raising_client):
    c = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    (tmp_path / "sub").mkdir()
    outside = tmp_path.parent / "outside_model.pt"
    outside.write_bytes(b"not a model")
    try:
        P = gen(c, "grid2d", 4)["puzzle"]
        for name in ("../outside_model.pt", str(outside), "sub/../../outside_model.pt"):
            for ep, extra in (("/api/solve/rl", {}), ("/api/policy", {"path": [P["checkpoints"][0]]})):
                r = c.post(ep, json={"puzzle": P, "model": name, **extra})
                assert r.status_code == 404, (ep, name, r.status_code, r.text)
    finally:
        outside.unlink()


@pytest.mark.skipif(not HAS_MODELS, reason="no checkpoints/*.pt")
def test_models_in_subfolders(tmp_path):
    import shutil
    src = sorted(CKPT_DIR.glob("*.pt"), key=lambda q: q.stat().st_size)[0]
    (tmp_path / "remote" / "host").mkdir(parents=True)
    shutil.copy(src, tmp_path / "remote" / "host" / "m_latest.pt")
    shutil.copy(src, tmp_path / "local_final.pt")
    c = TestClient(create_app(tmp_path))
    ms = {m["name"]: m for m in c.get("/api/models").json()["models"]}
    assert set(ms) == {"remote/host/m_latest.pt", "local_final.pt"}
    assert ms["remote/host/m_latest.pt"]["group"] == "remote/host" and ms["local_final.pt"]["default"]
    P = gen(c, "grid2d", 4)["puzzle"]
    r = c.post("/api/policy", json={"puzzle": P, "path": [P["checkpoints"][0]], "model": "remote/host/m_latest.pt"})
    assert r.status_code == 200 and r.json()["model"] == "remote/host/m_latest.pt"
    r = c.post("/api/solve/rl", json={"puzzle": P, "model": "remote/host/m_latest.pt"})
    assert r.status_code == 200 and r.json()["model"] == "remote/host/m_latest.pt"
