"""New puzzle kinds: portals, torus, hex, tri, oneway, overpass, keys, cubesurf (+ fog).

Generation validity, unique mode, JSON round trip, rater, and brute-force agreement
of the exact solver (incl. one-way arcs and key/door precedence) on tiny boards.
"""
import json
import time

import numpy as np
import pytest

from zipsolve.difficulty import rate
from zipsolve.generator import make_puzzle
from zipsolve.graph import (cube_surface, from_edges, grid, hex_parallelogram, hexagon, overpass_grid,
                            torus, tri_hexagon, tri_rect, with_meta)
from zipsolve.puzzle import Puzzle
from zipsolve.solver import (_bipartite_colours, check_prefix, count_solutions, is_dead_end,
                             label_moves, legal_moves, solve, solve_from_prefix)

MEDIUM = [("portals", 7), ("torus", 6), ("hex", 4), ("tri", 3), ("oneway", 7), ("overpass", 7),
          ("keys", 7), ("cubesurf", 3)]


# ---------------------------------------------------------------------------
# brute force (honours arcs + precedence), independent of the solver
# ---------------------------------------------------------------------------
def brute_solutions(p: Puzzle, cap: int = 10**6) -> list[list[int]]:
    g = p.graph
    n = g.num_nodes
    cps = p.checkpoints
    cpi = {c: i for i, c in enumerate(cps)}
    arcs = {(u, v) for u, v in p.arcs}
    pre: dict[int, list[int]] = {}
    for a, b in p.precedence:
        pre.setdefault(b, []).append(a)
    out = []
    vis = [False] * n
    path = [cps[0]]
    vis[cps[0]] = True

    def rec(nxt):
        if len(out) >= cap:
            return
        if len(path) == n:
            if path[-1] == cps[-1]:
                out.append(list(path))
            return
        h = path[-1]
        for w in g.neighbors[h]:
            if vis[w] or (w, h) in arcs:
                continue
            if any(not vis[a] for a in pre.get(w, ())):
                continue
            k = cpi.get(w)
            if k is not None and k != nxt:
                continue
            vis[w] = True
            path.append(w)
            rec(nxt + (k is not None))
            path.pop()
            vis[w] = False

    if not pre.get(cps[0]):             # the start cannot be a locked door
        rec(1)
    return out


def _tiny_graphs():
    rng = np.random.default_rng(5)
    gs = [("torus33", torus(3, 3)), ("torus34", torus(3, 4)), ("hex2", hexagon(2)),
          ("hexrh", hex_parallelogram(3, 3)), ("trirect", tri_rect(3, 4)), ("trihex", tri_hexagon(1)),
          ("overpass", overpass_grid(4, 4, [(1, 1)])), ("overpass2", overpass_grid(4, 5, [(1, 1), (2, 3)])),
          ("cube1", cube_surface(1))]
    g = grid(4, 4)
    gs.append(("portals", from_edges(g.coords, g.edges() + [(0, 15), (3, 12)], "portals",
                                     {"portals": [[0, 15], [3, 12]]})))
    # random arcs / precedence on small grids (many instances are unsat: good for soundness)
    for i in range(6):
        g = grid(4, 4) if i % 2 == 0 else grid(3, 5)
        es = g.edges()
        idx = rng.choice(len(es), size=6, replace=False)
        arcs = [list(es[j]) if rng.random() < 0.5 else list(es[j][::-1]) for j in idx]
        gs.append((f"oneway{i}", with_meta(g, arcs=arcs, oneway=arcs)))
    for i in range(6):
        g = grid(4, 4) if i % 2 == 0 else grid(3, 5)
        nodes = rng.permutation(g.num_nodes)[:4].tolist()
        prec = [[nodes[0], nodes[1]], [nodes[2], nodes[3]]]
        gs.append((f"keys{i}", with_meta(g, precedence=prec)))
    return gs


@pytest.mark.parametrize("name,g", _tiny_graphs(), ids=[t[0] for t in _tiny_graphs()])
def test_bruteforce_agreement(name, g):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    n = g.num_nodes
    for trial in range(6):
        k = int(rng.integers(2, 5))
        cps = rng.permutation(n)[:k].tolist()
        p = Puzzle(g, cps)
        brute = brute_solutions(p)
        cnt, status = count_solutions(p, limit=10**6)
        assert status == "complete"
        assert cnt == len(brute), (name, cps, cnt, len(brute))
        for s in brute[:5]:
            assert p.check_solution(s) is None
        r = solve(p)
        assert (r.status == "solved") == bool(brute)
        if r.path:
            assert p.check_solution(r.path) is None
        # soundness of is_dead_end / legal_moves on prefixes of real solutions
        for s in brute[:3]:
            for i in range(1, n):
                pre = s[:i]
                assert check_prefix(p, pre) is None
                assert s[i] in legal_moves(p, pre)
                nxt = sum(1 for v in pre if v in set(cps))
                assert not is_dead_end(g, pre, pre[-1], nxt, cps)
            assert check_prefix(p, s) is None


def test_bruteforce_labels_and_prefix_agree():
    rng = np.random.default_rng(1)
    g = grid(4, 4)
    arcs = [[0, 1], [5, 6], [10, 9], [14, 13]]
    prec = [[3, 12], [15, 5]]
    g = with_meta(g, arcs=arcs, oneway=arcs, precedence=prec)
    checked = 0
    for _ in range(20):
        cps = rng.permutation(16)[:3].tolist()
        p = Puzzle(g, cps)
        brute = brute_solutions(p)
        if not brute:
            assert solve(p).status == "unsat"
            continue
        s = brute[0]
        pre = s[:4]
        labels = label_moves(p, pre, time_limit_per_move=5)
        for m, lab in labels.items():
            wins = any(b[:5] == pre + [m] for b in brute)
            assert lab == ("win" if wins else "lose")
        r = solve_from_prefix(p, pre)
        assert r.status == "solved" and r.path[:4] == pre and p.check_solution(r.path) is None
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
def test_check_solution_arcs_and_precedence():
    g = grid(1, 4)                       # 0-1-2-3
    p = Puzzle(with_meta(g, arcs=[[2, 1]]), [0, 3])
    assert p.check_solution([0, 1, 2, 3]) is not None
    p = Puzzle(with_meta(g, arcs=[[1, 2]]), [0, 3])
    assert p.check_solution([0, 1, 2, 3]) is None
    g2 = grid(2, 2)                      # 0 1 / 2 3
    p = Puzzle(with_meta(g2, precedence=[[3, 1]]), [0, 2])
    assert p.check_solution([0, 1, 3, 2]) is not None
    assert check_prefix(p, [0, 1]) is not None
    assert legal_moves(p, [0]) == []     # only neighbour 2 is the end, 1 is a locked door
    assert solve(p).status == "unsat"
    p = Puzzle(with_meta(g2, precedence=[[1, 3]]), [0, 2])
    assert p.check_solution([0, 1, 3, 2]) is None


def test_json_roundtrip_constraints():
    p = make_puzzle("oneway", 5, rng=1)
    d = json.loads(json.dumps(p.to_dict()))
    assert d["arcs"] and d["meta"]["oneway"] == d["arcs"] and "arcs" not in d["meta"]
    q = Puzzle.from_dict(d)
    assert q.arcs == p.arcs and q.check_solution(p.solution) is None
    # renderer copy alone is enough
    d2 = dict(d)
    d2.pop("arcs")
    assert Puzzle.from_dict(d2).arcs == p.arcs
    p = make_puzzle("keys", 6, rng=1)
    d = json.loads(json.dumps(p.to_dict()))
    assert d["precedence"] and len(d["meta"]["keys"]) == len(d["precedence"])
    assert Puzzle.from_dict(d).precedence == p.precedence
    d2 = dict(d)
    d2.pop("precedence")
    assert Puzzle.from_dict(d2).precedence == p.precedence
    # old puzzles unchanged: no new keys
    assert "arcs" not in make_puzzle("grid2d", 4, rng=0).to_dict()
    bad = dict(d)
    bad["arcs"] = [[0, 99]]
    with pytest.raises(ValueError):
        Puzzle.from_dict(bad)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,size", MEDIUM)
def test_generate_medium_fast_and_valid(kind, size):
    for seed in range(3):
        t = time.perf_counter()
        p = make_puzzle(kind, size, rng=seed)
        assert time.perf_counter() - t < 1.5
        assert p.graph.kind == kind
        assert p.check_solution(p.solution) is None
        q = Puzzle.from_dict(json.loads(json.dumps(p.to_dict())))
        assert q.check_solution(p.solution) is None
        assert q.graph.num_nodes == p.num_nodes and q.graph.edges() == p.graph.edges()
        assert q.arcs == p.arcs and q.precedence == p.precedence
        r = solve(q, time_limit=20)
        assert r.status == "solved" and q.check_solution(r.path) is None


@pytest.mark.parametrize("kind,size", MEDIUM)
def test_unique_mode(kind, size):
    t = time.perf_counter()
    p = make_puzzle(kind, size, rng=11, unique=True, time_limit=30)
    assert time.perf_counter() - t < 10
    assert p.check_solution(p.solution) is None
    assert count_solutions(p, 2, time_limit=30) == (1, "complete")
    r = rate(p)
    assert 0 <= r["score"] <= 100 and r["label"]


def test_kind_features():
    p = make_puzzle("portals", 7, rng=3, portals=3, decoys=1)
    pairs = [tuple(x) for x in p.graph.meta["portals"]]
    assert len(pairs) == 3
    used = {frozenset(e) for e in zip(p.solution, p.solution[1:])}
    assert sum(frozenset(x) in used for x in pairs) >= 1
    for a, b in pairs:
        assert p.graph.has_edge(a, b)

    p = make_puzzle("torus", (5, 6), rng=0)
    assert p.graph.meta["torus"] == [5, 6] and len(p.graph.meta["wrap_edges"]) == 11
    assert all(d == 4 for d in map(len, p.graph.neighbors))
    used = {frozenset(e) for e in zip(p.solution, p.solution[1:])}
    assert any(frozenset(e) in used for e in p.graph.meta["wrap_edges"])

    p = make_puzzle("hex", 4, rng=0)
    assert p.num_nodes == 37 and p.graph.meta["layout"] == "hex"
    assert _bipartite_colours(p.graph) is None          # parity pruning switches off
    assert make_puzzle("hex", (3, 4), rng=0).num_nodes == 12

    p = make_puzzle("tri", 3, rng=0)
    assert p.num_nodes == 54 and p.graph.meta["layout"] == "tri"
    assert max(map(len, p.graph.neighbors)) == 3
    assert make_puzzle("tri", (4, 6), rng=0).num_nodes == 24

    p = make_puzzle("oneway", 7, rng=2)
    arcs = p.arcs
    assert arcs and p.graph.meta["oneway"] == [list(a) for a in arcs]
    pos = {v: i for i, v in enumerate(p.solution)}
    on_path = [a for a in arcs if abs(pos[a[0]] - pos[a[1]]) == 1]
    assert on_path and all(pos[v] == pos[u] + 1 for u, v in on_path)
    assert len(on_path) < len(arcs)                     # decoys exist

    p = make_puzzle("overpass", 7, rng=4)
    ov = p.graph.meta["overpass"]
    assert ov
    for o in ov:
        hn, vn = o["h"], o["v"]
        assert hn != vn and list(p.graph.coords[hn]) == list(p.graph.coords[vn]) == o["cell"]
        assert len(p.graph.neighbors[hn]) == len(p.graph.neighbors[vn]) == 2
        assert all(p.graph.coords[x][0] == o["cell"][0] for x in p.graph.neighbors[hn])
        assert all(p.graph.coords[x][1] == o["cell"][1] for x in p.graph.neighbors[vn])
        assert hn not in p.checkpoints and vn not in p.checkpoints

    p = make_puzzle("keys", 7, rng=5)
    keys = p.graph.meta["keys"]
    assert keys and [[k["key"], k["door"]] for k in keys] == [list(x) for x in p.precedence]
    pos = {v: i for i, v in enumerate(p.solution)}
    for k in keys:
        assert pos[k["door"]] - pos[k["key"]] >= 3
        assert k["key"] not in p.checkpoints and k["door"] not in p.checkpoints

    p = make_puzzle("cubesurf", 3, rng=0)
    assert p.num_nodes == 54 and p.graph.meta["cubesurf"] == {"n": 3}
    assert all(len(ns) == 4 for ns in p.graph.neighbors)
    assert sorted(set(p.graph.meta["face"])) == list(range(6))
    assert p.graph.coords.shape[1] == 3


def test_fog_any_kind():
    for kind, size in [("grid2d", 5), ("hex", 3), ("keys", 5), ("islands", 2)]:
        p = make_puzzle(kind, size, rng=0, fog=True)
        assert p.to_dict()["meta"]["fog"] is True
    assert "fog" not in make_puzzle("grid2d", 5, rng=0).to_dict()["meta"]


def test_shapes():
    assert hexagon(3).num_nodes == 19
    assert tri_hexagon(2).num_nodes == 24
    assert _bipartite_colours(tri_hexagon(2)) is not None
    assert cube_surface(2).num_nodes == 24
    assert all(len(ns) == 4 for ns in cube_surface(2).neighbors)
    t = torus(3, 3)
    assert _bipartite_colours(t) is None and all(len(ns) == 4 for ns in t.neighbors)


def test_env_mask_respects_constraints():
    from zipsolve.rl.env import ZipEnv
    g = grid(2, 2)                                     # 0 1 / 2 3
    p = Puzzle(with_meta(g, arcs=[[1, 0]]), [0, 2])
    env = ZipEnv()
    env.reset(options={"puzzle": p})
    assert not env.action_masks()[1]                   # 0 -> 1 is against the arrow
    p = Puzzle(with_meta(g, precedence=[[3, 1]]), [0, 2])
    env.reset(options={"puzzle": p})
    assert not env.action_masks()[1]                   # door 1 locked until key 3


def test_generation_deterministic_new_kinds():
    for kind, size in MEDIUM:
        a = make_puzzle(kind, size, rng=7).to_dict()
        b = make_puzzle(kind, size, rng=7).to_dict()
        assert a == b, kind
