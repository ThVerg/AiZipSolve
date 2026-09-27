"""Tests for zipsolve.solver (correctness, soundness of pruning, counting)."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from zipsolve.graph import from_edges, from_mask, grid, islands, random_mask, remove_edges
from zipsolve.puzzle import Puzzle
from zipsolve.solver import SolveResult, count_solutions, is_dead_end, solve


# ---------------------------------------------------------------- helpers
def snake_coords(shape):
    """Boustrophedon order of all cells of an n-D box (consecutive cells adjacent)."""
    if len(shape) == 1:
        return [(i,) for i in range(shape[0])]
    sub = snake_coords(shape[1:])
    out = []
    for i in range(shape[0]):
        out.extend((i,) + c for c in (sub if i % 2 == 0 else sub[::-1]))
    return out


def snake_puzzle(shape, n_cp, rng=0):
    g = grid(*shape)
    path = [g.node_at(c) for c in snake_coords(shape)]
    rng = np.random.default_rng(rng)
    inner = sorted(rng.choice(np.arange(1, len(path) - 1), size=n_cp - 2, replace=False).tolist())
    cps = [path[0]] + [path[i] for i in inner] + [path[-1]]
    return Puzzle(g, cps, path)


def brute_force(puzzle, cap=10**6):
    """All solutions by plain DFS without pruning (tiny graphs only)."""
    g = puzzle.graph
    n = g.num_nodes
    cps = puzzle.checkpoints
    cpi = {c: i for i, c in enumerate(cps)}
    sols = []
    path = [cps[0]]
    vis = {cps[0]}

    def rec(nxt):
        if len(path) == n:
            if path[-1] == cps[-1]:
                sols.append(list(path))
            return
        for w in g.neighbors[path[-1]]:
            if w in vis:
                continue
            if w in cpi and cpi[w] != nxt:
                continue
            vis.add(w)
            path.append(w)
            rec(nxt + 1 if w in cpi else nxt)
            path.pop()
            vis.discard(w)

    rec(1)
    return sols


def random_small_puzzle(rng):
    kind = rng.integers(5)
    if kind == 0:
        g = grid(int(rng.integers(2, 5)), int(rng.integers(2, 5)))
    elif kind == 1:
        g = random_mask((4, 4), fill=0.8, rng=rng)
    elif kind == 2:
        g = grid(3, 4)
        edges = g.edges()
        k = int(rng.integers(1, 4))
        walls = [edges[i] for i in rng.choice(len(edges), size=k, replace=False)]
        g = remove_edges(g, walls)
    elif kind == 3:
        g = grid(2, 2, int(rng.integers(2, 4)))
    else:
        # non-bipartite: grid plus random diagonals
        base = grid(3, 3 + int(rng.integers(0, 2)))
        edges = base.edges()
        n = base.num_nodes
        for _ in range(3):
            u, v = rng.choice(n, size=2, replace=False)
            edges.append((int(u), int(v)))
        g = from_edges(base.coords, edges)
    n = g.num_nodes
    k = int(rng.integers(2, min(5, n) + 1))
    cps = [int(x) for x in rng.choice(n, size=k, replace=False)]
    return Puzzle(g, cps)


# ---------------------------------------------------------------- basics
def test_trivial_line():
    g = grid(1, 5)
    p = Puzzle(g, [0, 4])
    r = solve(p)
    assert isinstance(r, SolveResult)
    assert r.status == "solved" and r.path == [0, 1, 2, 3, 4]
    assert count_solutions(p, limit=5) == (1, "complete")
    assert solve(Puzzle(g, [1, 4])).status == "unsat"
    assert solve(Puzzle(g, [0, 3, 2, 4])).status == "unsat"   # wrong checkpoint order


def test_2x2_and_parity_unsat():
    g = grid(2, 2)          # 0 1 / 2 3
    assert solve(Puzzle(g, [0, 1])).status == "solved"
    assert solve(Puzzle(g, [0, 3])).status == "unsat"   # same colour, even n
    g = grid(3, 3)
    assert solve(Puzzle(g, [0, 1])).status == "unsat"   # corners must be endpoints colour
    assert solve(Puzzle(g, [0, 8])).status == "solved"


def test_disconnected_graph_unsat():
    g = from_edges(np.zeros((4, 2)), [(0, 1), (2, 3)])
    assert solve(Puzzle(g, [0, 3])).status == "unsat"
    assert count_solutions(Puzzle(g, [0, 3])) == (0, "complete")


@pytest.mark.parametrize("shape,ncp", [((5, 5), 5), ((6, 6), 8), ((7, 7), 10), ((4, 6), 4)])
def test_snake_2d(shape, ncp):
    p = snake_puzzle(shape, ncp, rng=1)
    r = solve(p, time_limit=30)
    assert r.status == "solved"
    assert p.check_solution(r.path) is None


def test_walls():
    g = grid(4, 4)
    # snake with walls forcing it: wall between rows except at the turns
    walls = [(g.node_at((0, c)), g.node_at((1, c))) for c in range(3)]
    walls += [(g.node_at((1, c)), g.node_at((2, c))) for c in range(1, 4)]
    walls += [(g.node_at((2, c)), g.node_at((3, c))) for c in range(3)]
    gw = remove_edges(g, walls)
    path = [gw.node_at(c) for c in snake_coords((4, 4))]
    p = Puzzle(gw, [path[0], path[-1]])
    assert count_solutions(p, limit=5) == (1, "complete")
    r = solve(p)
    assert r.path == path
    # with walls, the end at the other corner is impossible
    assert solve(Puzzle(gw, [path[0], gw.node_at((3, 3))])).status == "unsat"


def test_islands():
    g = islands([(3, 3), (2, 3), (3, 2)], rng=5)  # seed with solvable pairs
    br = g.meta["bridges"]
    # find a solution via exhaustive search for some endpoint pair and check it
    found = 0
    for s, e in itertools.permutations(range(g.num_nodes), 2):
        p = Puzzle(g, [s, e])
        r = solve(p, time_limit=5)
        assert r.status in ("solved", "unsat")
        if r.status == "solved":
            assert p.check_solution(r.path) is None
            found += 1
            # every solution crosses each bridge
            edges = {frozenset(x) for x in zip(r.path, r.path[1:])}
            assert all(frozenset(b) in edges for b in br)
        if found >= 5:
            break
    assert found > 0


def test_islands_count_vs_brute():
    rng = np.random.default_rng(5)
    for seed in range(6):
        g = islands([(2, 2), (2, 3), (2, 2)], rng=seed)
        n = g.num_nodes
        for _ in range(6):
            cps = [int(x) for x in rng.choice(n, size=int(rng.integers(2, 4)), replace=False)]
            p = Puzzle(g, cps)
            bf = brute_force(p)
            c, st = count_solutions(p, limit=10**6)
            assert st == "complete" and c == len(bf)


def test_3d():
    p = snake_puzzle((3, 3, 3), 6, rng=2)
    r = solve(p, time_limit=30)
    assert r.status == "solved" and p.check_solution(r.path) is None


@pytest.mark.parametrize("shape", [(2, 2, 2, 2), (3, 3, 3, 3)])
def test_4d(shape):
    p = snake_puzzle(shape, 6, rng=4)
    r = solve(p, time_limit=60)
    assert r.status == "solved" and p.check_solution(r.path) is None


def test_non_bipartite():
    # triangle-rich graph: complete graph K5
    g = from_edges(np.zeros((5, 1)), itertools.combinations(range(5), 2))
    p = Puzzle(g, [0, 2, 4])
    bf = brute_force(p)
    assert count_solutions(p, limit=1000) == (len(bf), "complete")
    assert len(bf) == 6   # {1,3} both before 2 (2 orders), both after (2), split (2)
    r = solve(p)
    assert p.check_solution(r.path) is None


# ---------------------------------------------------------------- soundness
def test_count_matches_brute_force_random():
    rng = np.random.default_rng(12345)
    n_sat = 0
    for _ in range(250):
        p = random_small_puzzle(rng)
        bf = brute_force(p)
        c, st = count_solutions(p, limit=10**6)
        assert st == "complete"
        assert c == len(bf), (p.to_dict(), c, len(bf))
        r = solve(p)
        if bf:
            n_sat += 1
            assert r.status == "solved" and p.check_solution(r.path) is None
        else:
            assert r.status == "unsat"
    assert n_sat > 20


def _backbite(graph, path, steps, rng):
    path = list(path)
    for _ in range(steps):
        if rng.random() < 0.5:
            path.reverse()
        j = path.index(int(rng.choice(graph.neighbors[path[-1]])))
        if j != len(path) - 2:
            path[j + 1:] = path[j + 1:][::-1]
    return path


@pytest.mark.parametrize("shape", [(4, 4), (3, 5), (2, 3, 3), (2, 2, 2, 2), (5, 4)])
def test_count_matches_brute_force_on_sat_puzzles(shape):
    rng = np.random.default_rng(sum(shape) * 31 + len(shape))
    g = grid(*shape)
    base = [g.node_at(c) for c in snake_coords(shape)]
    for _ in range(25):
        path = _backbite(g, base, 10 * g.num_nodes, rng)
        k = int(rng.integers(2, 6))
        inner = sorted(rng.choice(np.arange(1, len(path) - 1), size=k - 2, replace=False).tolist())
        p = Puzzle(g, [path[0]] + [path[i] for i in inner] + [path[-1]])
        bf = brute_force(p)
        assert bf
        assert count_solutions(p, limit=10**7) == (len(bf), "complete")
        r = solve(p)
        assert r.status == "solved" and p.check_solution(r.path) is None
        # also without the optional prunings / restarts
        r2 = solve(p, deep=False, restarts=False)
        assert r2.status == "solved" and p.check_solution(r2.path) is None


def test_count_limit_and_uniqueness():
    g = grid(4, 4)
    p = Puzzle(g, [0, 3])
    total = len(brute_force(p))
    assert total > 2
    assert count_solutions(p, limit=2) == (2, "limit")
    assert count_solutions(p, limit=total + 1) == (total, "complete")


def test_is_dead_end_sound_on_solution_prefixes():
    rng = np.random.default_rng(7)
    for shape in [(5, 5), (4, 6), (3, 3, 3)]:
        p = snake_puzzle(shape, 5, rng=int(rng.integers(100)))
        sol = p.solution
        cpi = {c: i for i, c in enumerate(p.checkpoints)}
        nxt = 0
        for k in range(1, len(sol)):
            prefix = sol[:k]
            nxt = sum(1 for v in prefix if v in cpi)
            assert not is_dead_end(p.graph, set(prefix), prefix[-1], nxt, p.checkpoints)
            # array form, and "not yet incremented" index form
            arr = np.zeros(p.num_nodes, dtype=bool)
            arr[prefix] = True
            assert not is_dead_end(p.graph, arr, prefix[-1], nxt, p.checkpoints)


def test_is_dead_end_detects():
    g = grid(3, 3)  # 0 1 2 / 3 4 5 / 6 7 8
    cps = [0, 8]
    # path 0 -> 1: corner 2 must still be passable (2 has nbrs 1(head), 5) fine
    assert not is_dead_end(g, {0, 1}, 1, 1, cps)
    # path 0,1,4: node 2 has only neighbour 5 free and 3 has only 6 free -> dead
    assert is_dead_end(g, {0, 1, 4}, 4, 1, cps)
    # path 0,3,4,1: node 2 has free nbr 5 + head -> fine? 2-5, then 6,7 cut? check connectivity
    assert is_dead_end(g, {0, 3, 4, 5}, 5, 1, cps)  # 3->4->5 splits {1,2} from {6,7,8}


def test_is_dead_end_random_consistency():
    """is_dead_end == True must imply no completion exists (brute-force check)."""
    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(150):
        p = random_small_puzzle(rng)
        g = p.graph
        cps = p.checkpoints
        cpi = {c: i for i, c in enumerate(cps)}
        # random legal walk prefix
        path = [cps[0]]
        vis = {cps[0]}
        nxt = 1
        for _ in range(int(rng.integers(1, g.num_nodes))):
            opts = [w for w in g.neighbors[path[-1]] if w not in vis and (w not in cpi or cpi[w] == nxt)]
            if not opts:
                break
            w = int(rng.choice(opts))
            path.append(w)
            vis.add(w)
            if w in cpi:
                nxt += 1
        if nxt >= len(cps):
            continue
        dead = is_dead_end(g, vis, path[-1], nxt, cps)
        # brute-force completion: puzzle on same graph with the prefix forced
        exists = _completion_exists(g, path, cps, nxt)
        if dead:
            assert not exists
        checked += 1
    assert checked > 50


def _completion_exists(g, path, cps, nxt):
    n = g.num_nodes
    cpi = {c: i for i, c in enumerate(cps)}
    vis = set(path)
    path = list(path)

    def rec(nxt):
        if len(path) == n:
            return path[-1] == cps[-1]
        for w in g.neighbors[path[-1]]:
            if w in vis or (w in cpi and cpi[w] != nxt):
                continue
            vis.add(w)
            path.append(w)
            ok = rec(nxt + 1 if w in cpi else nxt)
            path.pop()
            vis.discard(w)
            if ok:
                return True
        return False

    return rec(nxt)


# ---------------------------------------------------------------- hooks & limits
def test_move_order_hook():
    p = snake_puzzle((5, 5), 5, rng=0)
    calls = []

    def reverse_order(head, candidates, visited, next_cp_idx):
        assert isinstance(visited, np.ndarray) and visited.dtype == bool
        assert visited[head]
        assert 0 < next_cp_idx < len(p.checkpoints)
        calls.append(head)
        return list(reversed(candidates))

    r = solve(p, move_order=reverse_order)
    assert r.status == "solved" and p.check_solution(r.path) is None
    assert calls


def test_timeout():
    g = grid(9, 9)
    p = Puzzle(g, [0, 1, 80])
    r = solve(p, time_limit=0.05)
    assert r.status in ("timeout", "solved", "unsat")
    if r.status == "solved":
        assert p.check_solution(r.path) is None
    c, st = count_solutions(Puzzle(grid(8, 8), [0, 7]), limit=10**9, time_limit=0.05)
    assert st == "timeout" and c >= 0


# ---------------------------------------------------------------- benchmark (prints only)
def test_benchmark_print(capsys):
    cases = [("7x7 snake 10cp", snake_puzzle((7, 7), 10, rng=0)),
             ("5x5x5 snake 8cp", snake_puzzle((5, 5, 5), 8, rng=0)),
             ("3x3x3x3 snake 8cp", snake_puzzle((3, 3, 3, 3), 8, rng=0))]
    with capsys.disabled():
        print()
        for name, p in cases:
            t = time.perf_counter()
            r = solve(p, time_limit=30)
            print(f"  {name:22s} {r.status:8s} expanded={r.nodes_expanded:8d} "
                  f"{time.perf_counter() - t:7.3f}s")
