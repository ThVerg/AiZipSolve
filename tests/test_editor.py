"""Puzzle-editor API tests (save / load / list / delete, analyse, make unique, suggest)."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from zipsolve.app import editor_api  # noqa: E402
from zipsolve.graph import from_mask, grid, remove_edges  # noqa: E402
from zipsolve.puzzle import Puzzle  # noqa: E402
from zipsolve.solver import count_solutions  # noqa: E402

import numpy as np  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ZIPSOLVE_CUSTOM_DIR", str(tmp_path / "custom"))
    app = FastAPI()
    app.include_router(editor_api.router)
    return TestClient(app)


def pz(graph, cps) -> dict:
    d = Puzzle(graph, cps).to_dict()
    d.pop("solution")
    return d


def snake_puzzle(side=4):
    """side x side grid, checkpoints at the two ends of the boustrophedon path."""
    g = grid(side, side)
    return pz(g, [0, g.node_at((side - 1, 0 if side % 2 == 0 else side - 1))])


def test_save_load_list_delete(client):
    d = snake_puzzle(4)
    r = client.post("/api/custom", json={"name": "My first map!", "puzzle": d})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert pid.startswith("my-first-map-")
    lst = client.get("/api/custom").json()
    assert [x["id"] for x in lst] == [pid]
    assert lst[0]["num_nodes"] == 16 and lst[0]["name"] == "My first map!"
    got = client.get(f"/api/custom/{pid}").json()
    assert got["id"] == pid and got["puzzle"]["checkpoints"] == d["checkpoints"]
    assert "solution" not in got["puzzle"]
    assert sorted(map(tuple, got["puzzle"]["edges"])) == sorted(map(tuple, d["edges"]))
    # overwrite in place with the same id
    r2 = client.post("/api/custom", json={"name": "Renamed", "puzzle": d, "id": pid})
    assert r2.json()["id"] == pid
    assert client.get(f"/api/custom/{pid}").json()["name"] == "Renamed"
    assert len(client.get("/api/custom").json()) == 1
    assert client.delete(f"/api/custom/{pid}").status_code == 200
    assert client.get(f"/api/custom/{pid}").status_code == 404
    assert client.get("/api/custom").json() == []


@pytest.mark.parametrize("bad", ["../etc", "..", "a/b", "A", "x.json", "-x", "a" * 80, "%2e%2e", "a_b", "é"])
def test_bad_ids_rejected(client, bad):
    d = snake_puzzle(3)
    r = client.post("/api/custom", json={"name": "x", "puzzle": d, "id": bad})
    assert r.status_code in (400, 422), (bad, r.text)
    assert client.get(f"/api/custom/{bad}").status_code in (400, 404, 405)
    assert client.delete(f"/api/custom/{bad}").status_code in (400, 404, 405)


def test_traversal_paths_never_escape(client, tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text('{"puzzle": {"coords": [[0]]}}')
    for url in ("/api/custom/..%2Fsecret", "/api/custom/%2e%2e%2fsecret", "/api/custom/../secret"):
        r = client.get(url)
        assert r.status_code in (400, 404), url
    assert secret.exists()


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(coords="nope"),
    lambda d: d.update(coords=[[0, 0]] * 1501),
    lambda d: d.update(edges=[[0, 99]]),
    lambda d: d.update(edges=[[0, True]]),
    lambda d: d.update(checkpoints=[0]),
    lambda d: d.update(checkpoints=[0, 0]),
    lambda d: d.update(checkpoints=[0, 1.5]),
    lambda d: d.update(meta=[1]),
    lambda d: d.update(meta={"island": [0]}),
    lambda d: d.update(meta={"bridges": [[0, 1000]]}),
    lambda d: d.update(coords=[[0, 0, 0, 0, 0]] * 9),
])
def test_save_validation(client, mutate):
    d = snake_puzzle(3)
    mutate(d)
    r = client.post("/api/custom", json={"name": "bad", "puzzle": d})
    assert r.status_code in (400, 422), r.text


def test_save_strips_unknown_meta_and_solution(client):
    d = snake_puzzle(3)
    d["meta"] = {"shape": [3, 3], "evil": "x" * 1000, "editor": {"layers": 1, "nested": {"a": 1}}}
    d["solution"] = [0, 1, 2]
    pid = client.post("/api/custom", json={"name": "m", "puzzle": d}).json()["id"]
    got = client.get(f"/api/custom/{pid}").json()["puzzle"]
    assert got["meta"] == {"shape": [3, 3], "editor": {"layers": 1}}
    assert "solution" not in got


def test_analyze_solvable_and_unique(client):
    # 1 x 5 corridor: exactly one path
    g = grid(1, 5)
    r = client.post("/api/editor/analyze", json={"puzzle": pz(g, [0, 4]), "want_solution": True}).json()
    assert r["connected"] and r["parity_ok"] and r["solvable"] is True and r["unique"] is True
    assert r["solution"] == [0, 1, 2, 3, 4]


def test_analyze_not_unique(client):
    g = grid(4, 4)
    r = client.post("/api/editor/analyze", json={"puzzle": pz(g, [0, 3]), "want_solution": True}).json()
    assert r["solvable"] is True and r["unique"] is False
    assert r["solution"] != r["alternative"]
    p = Puzzle(g, [0, 3])
    assert p.is_valid_solution(r["solution"]) and p.is_valid_solution(r["alternative"])


def test_analyze_parity_unsat(client):
    g = grid(3, 3)   # 5 black / 4 white: both ends must be black (corners are black)
    r = client.post("/api/editor/analyze", json={"puzzle": pz(g, [0, 1])}).json()
    assert r["parity_ok"] is False and r["solvable"] is False and r["issues"]


def test_analyze_disconnected(client):
    mask = np.array([[1, 1, 0, 1, 1]], dtype=bool)
    g = from_mask(mask)
    r = client.post("/api/editor/analyze", json={"puzzle": pz(g, [0, 3])}).json()
    assert r["connected"] is False and r["solvable"] is False


def test_analyze_unsat_by_search(client):
    # parity is fine, so only the search decides; must agree with the exact solver
    g = grid(3, 4)
    p = Puzzle(g, [0, 5, 3])
    r = client.post("/api/editor/analyze", json={"puzzle": pz(g, p.checkpoints)}).json()
    exact = count_solutions(p, 1, 10)
    assert r["solvable"] is (exact[0] >= 1)


def test_analyze_without_checkpoints(client):
    g = remove_edges(grid(4, 4), [(0, 1)])
    d = pz(g, [0, 15])
    d["checkpoints"] = []
    r = client.post("/api/editor/analyze", json={"puzzle": d, "want_solution": True}).json()
    assert r["solvable"] is True and len(r["solution"]) == 16 and "note" in r


def test_make_unique(client):
    g = grid(5, 5)
    d = pz(g, [0, 24])
    r = client.post("/api/editor/make_unique", json={"puzzle": d, "time_limit": 20, "seed": 3})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["unique"] is True and j["added"] >= 1
    cps = j["puzzle"]["checkpoints"]
    assert cps[0] == 0 and cps[-1] == 24
    p = Puzzle(g, cps)
    assert count_solutions(p, 2, 20) == (1, "complete")
    assert p.is_valid_solution(j["solution"])


def test_make_unique_rejects_unsat(client):
    g = grid(3, 3)
    r = client.post("/api/editor/make_unique", json={"puzzle": pz(g, [0, 1])})
    assert r.status_code == 422


def test_suggest(client):
    g = grid(4, 5)
    d = pz(g, [0, 1])
    d["checkpoints"] = []
    r = client.post("/api/editor/suggest", json={"puzzle": d, "num_checkpoints": 5, "seed": 1})
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["puzzle"]["checkpoints"]) == 5
    assert Puzzle(g, j["puzzle"]["checkpoints"]).is_valid_solution(j["solution"])
    r = client.post("/api/editor/suggest", json={"puzzle": d, "unique": True, "seed": 2, "time_limit": 20}).json()
    assert r["unique"] is True
    assert count_solutions(Puzzle(g, r["puzzle"]["checkpoints"]), 2, 20) == (1, "complete")
