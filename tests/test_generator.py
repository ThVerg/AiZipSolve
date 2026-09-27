import time

import numpy as np
import pytest

from zipsolve.generator import (generate, make_puzzle, place_checkpoints,
                                random_hamiltonian_path)
from zipsolve.graph import from_edges, grid, islands
from zipsolve.puzzle import Puzzle

CASES = [
    ("grid2d", 5), ("grid2d", 7), ("grid2d", 8),
    ("walls", 7),
    ("islands", 3), ("islands", [(3, 3), (3, 3), (3, 3)]), ("islands", [(4, 4)] * 3),
    ("grid3d", 4), ("grid3d", 5),
    ("grid4d", 3),
    ("grid", (3, 4, 5)), ("grid", (2, 3, 2, 3)),
    ("mask", 7), ("mask", (5, 5, 5)),
]


def _is_ham_path(g, path):
    return (len(path) == g.num_nodes and len(set(path)) == len(path)
            and all(g.has_edge(a, b) for a, b in zip(path, path[1:])))


@pytest.mark.parametrize("kind,size", CASES, ids=lambda x: str(x))
def test_make_puzzle_valid(kind, size):
    times = []
    for seed in range(5):
        t = time.perf_counter()
        p = make_puzzle(kind, size, rng=seed)
        times.append(time.perf_counter() - t)
        assert p.solution is not None
        assert p.check_solution(p.solution) is None
        assert p.checkpoints[0] == p.solution[0] and p.checkpoints[-1] == p.solution[-1]
        assert len(p.checkpoints) >= 2
    print(f"{kind} {size}: n={p.num_nodes} cps={len(p.checkpoints)} "
          f"mean={np.mean(times) * 1e3:.1f}ms max={max(times) * 1e3:.1f}ms")


def test_default_checkpoints_like_linkedin():
    ks = [len(make_puzzle("grid2d", 7, rng=s).checkpoints) for s in range(20)]
    assert all(8 <= k <= 12 for k in ks), ks


@pytest.mark.parametrize("kind,size", [("grid2d", 7), ("grid3d", 4), ("grid4d", 3),
                                       ("islands", 3), ("mask", 7)])
def test_diversity(kind, size):
    sols = {tuple(make_puzzle(kind, size, rng=s).solution) for s in range(10)}
    assert len(sols) >= 9


def test_diversity_same_graph():
    g = grid(7, 7)
    paths = {tuple(random_hamiltonian_path(g, rng=s)) for s in range(30)}
    assert len(paths) == 30
    starts = {p[0] for p in paths}
    assert len(starts) > 5


def test_determinism():
    for kind, size in [("grid2d", 7), ("walls", 6), ("islands", 3), ("grid3d", 4), ("mask", 6)]:
        a, b = make_puzzle(kind, size, rng=123), make_puzzle(kind, size, rng=123)
        assert a.solution == b.solution and a.checkpoints == b.checkpoints
        assert a.graph.edges() == b.graph.edges()


def test_walls_do_not_cut_solution():
    for seed in range(5):
        p = make_puzzle("walls", 7, rng=seed, walls_frac=0.5)
        walls = p.graph.meta["walls"]
        assert len(walls) > 0
        assert p.graph.num_nodes == 49
        used = {frozenset(e) for e in zip(p.solution, p.solution[1:])}
        assert not any(frozenset(w) in used for w in walls)
        assert len(p.graph.edges()) == 84 - len(walls)
        assert p.check_solution(p.solution) is None


def test_start_and_parity():
    g = grid(5, 5)
    # odd 5x5: the start must be on the majority colour (corner colour)
    assert random_hamiltonian_path(g, rng=0, start=1) is None
    p = random_hamiltonian_path(g, rng=0, start=12)
    assert p[0] == 12 and _is_ham_path(g, p)
    g6 = grid(6, 6)
    for s in range(0, 36, 7):
        p = random_hamiltonian_path(g6, rng=s, start=s)
        assert p[0] == s and _is_ham_path(g6, p)


def test_no_path_graphs_terminate():
    star = from_edges(np.zeros((4, 2)), [(0, 1), (0, 2), (0, 3)])
    assert random_hamiltonian_path(star, rng=0) is None
    disconnected = from_edges(np.zeros((4, 2)), [(0, 1), (2, 3)])
    assert random_hamiltonian_path(disconnected, rng=0) is None
    # 3x3 + 3x3 + 3x3 islands: many bridge placements are infeasible; must answer fast
    t = time.perf_counter()
    for s in range(20):
        g = islands([(3, 3)] * 3, rng=s)
        p = random_hamiltonian_path(g, rng=s, time_limit=2.0)
        assert p is None or _is_ham_path(g, p)
    assert time.perf_counter() - t < 10


def test_extra_bridges_islands():
    p = make_puzzle("islands", 3, rng=0, extra_bridges=2)
    assert p.check_solution(p.solution) is None


def test_place_checkpoints():
    path = list(range(50))
    rng = np.random.default_rng(0)
    for k in [2, 3, 10, 25, 49, 50, 80]:
        cps = place_checkpoints(path, k, rng)
        assert cps[0] == 0 and cps[-1] == 49
        assert cps == sorted(cps) and len(set(cps)) == len(cps) == min(k, 50)
    assert place_checkpoints([5, 7], 5, rng) == [5, 7]


def test_generate_on_custom_graph():
    # Petersen graph (non-bipartite, non-grid) has Hamiltonian paths
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    g = from_edges(np.random.default_rng(0).random((10, 2)), outer + spokes + inner)
    p = generate(g, 4, rng=1)
    assert isinstance(p, Puzzle) and p.check_solution(p.solution) is None


def test_generate_raises_when_impossible():
    star = from_edges(np.zeros((4, 2)), [(0, 1), (0, 2), (0, 3)])
    with pytest.raises(ValueError):
        generate(star, 2, rng=0)


def test_unique():
    solver = pytest.importorskip("zipsolve.solver")
    for seed in range(3):
        p = make_puzzle("grid2d", 6, rng=seed, unique=True, time_limit=30)
        assert p.check_solution(p.solution) is None
        count, status = solver.count_solutions(p, limit=2)
        assert count == 1, (count, status)


# ---- archipelago: islands visited back and forth ---------------------------
from zipsolve.generator import archipelago  # noqa: E402


def _island_runs(p):
    isl = p.graph.meta["island"]
    seq = [isl[v] for v in p.solution]
    return [seq[0]] + [b for a, b in zip(seq, seq[1:]) if a != b]


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_archipelago_back_and_forth(k):
    for seed in range(3):
        p = make_puzzle("islands", k, rng=seed)
        assert p.check_solution(p.solution) is None
        runs = _island_runs(p)
        assert len(runs) - 1 == p.graph.meta["jumps"] == 2 * (k - 1) + 1
        assert len(set(runs)) == k
        assert len(runs) > len(set(runs))  # some island is revisited
        # every jump of the solution uses a bridge; decoys are bridges the solution skips
        isl = p.graph.meta["island"]
        bridges = {frozenset(e) for e in p.graph.meta["bridges"]}
        for a, b in zip(p.solution, p.solution[1:]):
            if isl[a] != isl[b]:
                assert frozenset((a, b)) in bridges
        # decoys: as many as requested fit on the free facing cells, never on the solution
        used = {frozenset(e) for e in zip(p.solution, p.solution[1:])}
        assert len(p.graph.meta["decoys"]) <= k - 1
        assert not used & {frozenset(e) for e in p.graph.meta["decoys"]}


def test_archipelago_params():
    g, path = archipelago(3, jumps=6, max_per_pair=4, decoys=0, rng=1)
    assert g.meta["jumps"] == 6 and g.meta["decoys"] == []
    for (a, b) in g.meta["bridges"]:  # bridges only go straight across a gap
        d = np.abs(g.coords[a] - g.coords[b])
        assert sorted(d.tolist()) == [0.0, 2.0]
    g2, _ = archipelago(4, layout=(1, 4), rng=2)
    assert g2.meta["layout"] == [1, 4]


def test_archipelago_deterministic_and_chain_kind():
    a = make_puzzle("islands", 4, rng=7)
    b = make_puzzle("islands", 4, rng=7)
    assert a.solution == b.solution and a.checkpoints == b.checkpoints
    c = make_puzzle("islands_chain", 3, rng=0)
    assert c.check_solution(c.solution) is None


# ---- time budget ------------------------------------------------------------
from zipsolve.generator import GenerationTimeout  # noqa: E402


@pytest.mark.parametrize("kind,size,limit", [
    ("grid3d", 5, 0.1), ("grid3d", 5, 1.0), ("grid4d", 3, 0.5), ("islands", 6, 0.05),
    ("mask", 9, 0.05), ("walls", 9, 0.05), ("islands_chain", 4, 0.05),
])
def test_generation_respects_time_limit(kind, size, limit):
    t0 = time.perf_counter()
    try:
        p = make_puzzle(kind, size, rng=1, unique=True, time_limit=limit)
    except GenerationTimeout:
        pass
    else:  # finished in time: then it must really be unique
        from zipsolve.solver import count_solutions
        assert count_solutions(p, 2, 30) == (1, "complete")
    assert time.perf_counter() - t0 < limit + 0.25


@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_organic_islands(k):
    for seed in range(4):
        p = make_puzzle("islands", k, rng=seed)
        assert p.check_solution(p.solution) is None
        isl = np.array(p.graph.meta["island"])
        full = sum(a * b for a, b in p.graph.meta["shapes"])
        assert p.num_nodes < full  # some coastline was carved
        for b in range(k):  # every island is still one connected landmass
            cells = set(np.flatnonzero(isl == b).tolist())
            start = next(iter(cells))
            seen, stack = {start}, [start]
            while stack:
                for w in p.graph.neighbors[stack.pop()]:
                    if w in cells and w not in seen:
                        seen.add(w)
                        stack.append(w)
            assert seen == cells and len(cells) >= 4


def test_rectangular_islands_still_available():
    p = make_puzzle("islands", 3, rng=0, organic=0)
    assert p.num_nodes == sum(a * b for a, b in p.graph.meta["shapes"])
