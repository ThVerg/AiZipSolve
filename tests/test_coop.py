"""Co-op (two paths) puzzles: solver vs brute force, generator, JSON, hints, API dispatch."""
import json
import random
import time

import pytest

from zipsolve.coop import (CoopPuzzle, brute_force, check, check_partial, count_solutions, find_solutions,
                           generate, hex_board, hint, legal_moves, make_coop, random_two_paths, rate, solve)
from zipsolve.graph import grid, random_mask
from zipsolve.puzzle import Puzzle


def _tiny_graph(rnd):
    k = rnd.choice("gghm")
    if k == "g":
        return grid(rnd.randint(2, 4), rnd.randint(3, 4))
    if k == "h":
        return hex_board(3, rnd.randint(2, 4))
    return random_mask((4, 4), fill=0.75, rng=rnd.randint(0, 10**6))


def test_bruteforce_agreement_random_checkpoints():
    rnd = random.Random(7)
    done = 0
    while done < 250:
        g = _tiny_graph(rnd)
        n = g.num_nodes
        if n < 5:
            continue
        k = rnd.randint(4, min(n, 6))
        nodes = rnd.sample(range(n), k)
        a = rnd.randint(2, k - 2)
        pz = CoopPuzzle(g, [nodes[:a], nodes[a:]])
        sols, status = find_solutions(pz, limit=10**6)
        assert status == "complete"
        assert sorted(map(str, sols)) == sorted(map(str, brute_force(pz)))
        done += 1


def test_bruteforce_agreement_generated_multi_solution():
    rnd = random.Random(3)
    multi = 0
    for _ in range(120):
        g = _tiny_graph(rnd)
        if g.num_nodes < 6:
            continue
        paths = random_two_paths(g, rnd.randint(0, 10**6), min_frac=0.25)
        if paths is None:
            continue
        pz = generate(g, paths, num_checkpoints=4, rng=rnd.randint(0, 10**6))
        bf = brute_force(pz)
        assert paths in bf
        sols, status = find_solutions(pz, limit=10**6)
        assert status == "complete"
        assert sorted(map(str, sols)) == sorted(map(str, bf))
        multi += len(bf) > 1
    assert multi > 10


@pytest.mark.parametrize("base", ["grid2d", "walls", "mask", "hex"])
def test_generate_unique_valid(base):
    for seed in range(3):
        p = make_coop(6, rng=seed, unique=True, base=base, time_limit=20)
        assert check(p, p.solution) is None
        assert count_solutions(p, limit=2, time_limit=20) == (1, "complete")
        n1, n2 = map(len, p.solution)
        assert n1 >= 2 and n2 >= 2 and n1 + n2 == p.num_nodes


@pytest.mark.parametrize("n", [6, 7, 8])
def test_generation_speed(n):
    for seed in range(3):
        t = time.perf_counter()
        p = make_coop(n, rng=100 + seed, unique=True, time_limit=20)
        assert time.perf_counter() - t < 5.0
        assert count_solutions(p, 2, 20)[0] == 1


def test_random_two_paths_cover():
    g = grid(7, 7)
    for s in range(5):
        a, b = random_two_paths(g, s)
        assert set(a).isdisjoint(b) and len(a) + len(b) == 49
        for p in (a, b):
            assert all(g.has_edge(u, v) for u, v in zip(p, p[1:]))


def test_json_round_trip_and_contract_shape():
    p = make_coop(5, rng=4)
    d = p.to_dict()
    assert d["kind"] == "coop"
    assert d["checkpoints"] == d["meta"]["coop"]["checkpoints"][0]
    assert d["meta"]["coop"]["solution"]["paths"] == p.solution
    d2 = json.loads(json.dumps(d))
    q = CoopPuzzle.from_dict(d2)
    assert q.checkpoints == p.checkpoints and q.solution == p.solution
    assert q.to_dict() == d
    assert "solution" not in p.to_dict(include_solution=False)["meta"]["coop"]
    Puzzle.from_dict(d2)   # single-path code must not crash on it


def test_check_reasons():
    p = make_coop(5, rng=5)
    a, b = p.solution
    assert check(p, [a, b]) is None
    assert "does not end" in check(p, [a[:-1], b])
    assert check(p, [b, a]) is not None
    assert check(p, [a]) == "need exactly two paths"
    assert check(p, [a, a]) is not None


def test_check_partial_and_moves():
    p = make_coop(5, rng=6)
    a, b = p.solution
    r = check_partial(p, [a[:3], []])
    assert r["legal"] and not r["valid"]
    assert r["paths"][0]["length"] == 3 and r["paths"][1]["length"] == 1
    assert a[3] in r["paths"][0]["moves"]
    assert b[1] in r["paths"][1]["moves"]
    assert legal_moves(p, [a[:3], []], 0) == r["paths"][0]["moves"]
    # stepping on the other path's checkpoint is illegal
    other = p.checkpoints[1][1]
    r = check_partial(p, [[p.checkpoints[0][0], other], []])
    assert not r["legal"] and not r["paths"][0]["legal"]
    # crossing
    r = check_partial(p, [a, b[:1] + [a[1]]])
    assert not r["legal"]
    r = check_partial(p, [a, b])
    assert r["valid"] and r["paths"][0]["complete"] and r["paths"][1]["complete"]


def test_hint():
    p = make_coop(6, rng=8)
    a, b = p.solution
    h = hint(p, [a[:4], b[:2]], p.solution, path_index=1)
    assert h["status"] == "next" and h["path_index"] == 1 and h["next"] == b[2]
    h = hint(p, [a[:4], b[:2]], None, path_index=0)
    assert h["status"] == "next" and h["path_index"] == 0 and h["next"] == a[4]
    assert hint(p, [a, b])["status"] == "done"
    # a wrong move -> backtrack
    wrong = [w for w in legal_moves(p, [a[:2], []], 0) if w != a[2]]
    if wrong:
        h = hint(p, [a[:2] + [wrong[0]], []], p.solution)
        assert h["status"] == "backtrack" and h["keep"][0] <= 2
    assert hint(p, [[a[1]], []])["status"] == "invalid"


def test_solve_from_prefix_and_rate():
    p = make_coop(6, rng=9)
    a, b = p.solution
    r = solve(p, time_limit=10, prefix=[a[:5], b[:3]])
    assert r.status == "solved" and r.paths == p.solution
    rt = rate(p)
    assert 0 <= rt["score"] <= 100 and rt["label"] in ("Easy", "Medium", "Hard", "Expert", "Insane")


# ---------------------------------------------------------------- API dispatch
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from zipsolve.app.server import create_app
    return TestClient(create_app())


def test_api_coop_flow(client):
    r = client.post("/api/generate", json={"kind": "coop", "size": 6, "seed": 3, "unique": True})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["kind"] == "coop" and g["id"] and g["rating"]["label"]
    pz = g["puzzle"]
    assert pz["kind"] == "coop" and "solution" not in pz["meta"]["coop"]
    r = client.post("/api/solve/exact", json={"puzzle": pz, "time_limit": 10})
    assert r.status_code == 200 and r.json()["solved"]
    a, b = r.json()["paths"]
    r = client.post("/api/check", json={"puzzle": pz, "paths": [a, b]})
    assert r.status_code == 200 and r.json()["valid"]
    r = client.post("/api/check", json={"puzzle": pz, "paths": [a[:3], []]})
    j = r.json()
    assert j["legal"] and not j["valid"] and a[3] in j["paths"][0]["moves"]
    r = client.post("/api/hint", json={"puzzle": pz, "id": g["id"], "paths": [a[:3], b[:2]], "path_index": 1})
    assert r.status_code == 200 and r.json()["next"] == b[2]
    r = client.post("/api/solve/exact", json={"puzzle": pz, "start_paths": [a[:4], []]})
    assert r.json()["paths"] == [a, b]
    r = client.post("/api/solve/exact", json={"puzzle": pz, "start_paths": [[b[0]], []]})
    assert r.status_code == 400


def test_api_coop_options(client):
    for base in ("hex", "walls", "mask"):
        r = client.post("/api/generate", json={"kind": "coop", "size": 5, "seed": 1, "unique": True,
                                               "include_solution": True, "options": {"base": base}})
        assert r.status_code == 200, r.text
        j = r.json()
        cp = CoopPuzzle.from_dict(j["puzzle"])
        assert check(cp, j["solution"]["paths"]) is None
    r = client.post("/api/generate", json={"kind": "coop", "options": {"base": "nope"}})
    assert r.status_code == 422
    r = client.post("/api/check", json={"puzzle": {"kind": "coop", "coords": [[0, 0]], "edges": []},
                                        "paths": [[], []]})
    assert r.status_code == 400


def test_api_single_path_unchanged(client):
    r = client.post("/api/generate", json={"kind": "grid2d", "size": 4, "seed": 1, "include_solution": True})
    j = r.json()
    r = client.post("/api/check", json={"puzzle": j["puzzle"], "path": j["solution"]})
    assert r.json()["valid"]
