"""Render demo PNGs (and a GIF) of every Zip variant into outputs/.

Demo puzzles are built here directly (independent of the generator/solver):
a boustrophedon snake gives a Hamiltonian path on any n-D grid, "backbite"
moves randomise it, and checkpoints are sampled along the path.

    python scripts/viz_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zipsolve import graph as G  # noqa: E402
from zipsolve.puzzle import Puzzle  # noqa: E402
from zipsolve import viz  # noqa: E402

OUT = ROOT / "outputs"


# ------------------------------------------------------------ path builders
def snake(shape) -> list[tuple[int, ...]]:
    """Boustrophedon Hamiltonian path over an n-D box (list of index tuples)."""
    if len(shape) == 1:
        return [(i,) for i in range(shape[0])]
    sub = snake(shape[1:])
    out = []
    for i in range(shape[0]):
        out.extend((i,) + s for s in (sub if i % 2 == 0 else sub[::-1]))
    return out


def snake_path(g: G.ZipGraph, shape) -> list[int]:
    return [g.node_at(c) for c in snake(shape)]


def backbite(g: G.ZipGraph, path: list[int], steps: int, rng) -> list[int]:
    """Randomise a Hamiltonian path with backbite moves (valid on any graph)."""
    path = list(path)
    for _ in range(steps):
        if rng.random() < 0.5:
            path.reverse()
        end = path[-1]
        w = int(rng.choice(g.neighbors[end]))
        pos = path.index(w)
        if pos == len(path) - 2:
            continue
        path[pos + 1:] = path[pos + 1:][::-1]
    return path


def checkpoints_along(path, k, rng) -> list[int]:
    idx = sorted(rng.choice(np.arange(1, len(path) - 1), size=k - 2, replace=False).tolist())
    return [path[0]] + [path[i] for i in idx] + [path[-1]]


def ham_path_dfs(g: G.ZipGraph, start: int, limit: int = 200_000) -> list[int] | None:
    """Warnsdorff-ordered backtracking search for a Hamiltonian path."""
    n = g.num_nodes
    seen = [False] * n
    path = [start]
    seen[start] = True
    budget = [limit]

    def rec():
        if len(path) == n:
            return True
        budget[0] -= 1
        if budget[0] < 0:
            return False
        v = path[-1]
        cand = [w for w in g.neighbors[v] if not seen[w]]
        cand.sort(key=lambda w: sum(not seen[x] for x in g.neighbors[w]))
        for w in cand:
            seen[w] = True
            path.append(w)
            if rec():
                return True
            path.pop()
            seen[w] = False
        return False

    sys.setrecursionlimit(10_000)
    return path if rec() else None


# ------------------------------------------------------------ demo puzzles
def demo_grid(rng):
    g = G.grid(7, 7)
    p = backbite(g, snake_path(g, (7, 7)), 3000, rng)
    return Puzzle(g, checkpoints_along(p, 9, rng), p)


def demo_walls(rng):
    g = G.grid(7, 7)
    p = backbite(g, snake_path(g, (7, 7)), 3000, rng)
    on_path = {frozenset(e) for e in zip(p, p[1:])}
    free = [e for e in g.edges() if frozenset(e) not in on_path]
    walls = [free[i] for i in rng.choice(len(free), size=len(free) // 4, replace=False)]
    g2 = G.remove_edges(g, walls)
    return Puzzle(g2, checkpoints_along(p, 7, rng), p)


def demo_mask(rng):
    g = G.grid(9, 9)
    p = backbite(g, snake_path(g, (9, 9)), 6000, rng)
    keep = p[10:-12]
    mask = np.zeros((9, 9), bool)
    for v in keep:
        mask[tuple(g.coords[v].astype(int))] = True
    g2 = G.from_mask(mask)
    p2 = [g2.node_at(g.coords[v]) for v in keep]
    return Puzzle(g2, checkpoints_along(p2, 8, rng), p2)


def demo_islands(rng):
    for seed in range(200):
        g = G.islands([(4, 4), (5, 3), (3, 4)], extra_bridges=1, rng=seed)
        starts = [v for v in range(g.num_nodes) if g.meta["island"][v] == 0]
        for s in starts:
            p = ham_path_dfs(g, s, limit=20_000)
            if p:
                return Puzzle(g, checkpoints_along(p, 8, rng), p)
    raise RuntimeError("no islands demo found")


def demo_3d(rng, n=4):
    g = G.grid(n, n, n)
    p = backbite(g, snake_path(g, (n, n, n)), 8000, rng)
    return Puzzle(g, checkpoints_along(p, 10, rng), p)


def demo_4d(rng):
    g = G.grid(3, 3, 3, 3)
    p = backbite(g, snake_path(g, (3, 3, 3, 3)), 8000, rng)
    return Puzzle(g, checkpoints_along(p, 10, rng), p)


def demo_generic(rng):
    g0 = G.grid(2, 2, 2, 2, 2)  # 5D hypercube -> generic spring layout
    g = G.from_edges(g0.coords, g0.edges(), kind="custom")
    p = snake_path(g, (2, 2, 2, 2, 2))
    return Puzzle(g, checkpoints_along(p, 6, rng), p)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    made = []

    def out(name):
        f = OUT / name
        made.append(f)
        return f

    pz = demo_grid(rng)
    assert pz.is_valid_solution(pz.solution)
    viz.save(pz, None, out("viz_grid_puzzle.png"), title="7×7 grid")
    viz.save(pz, pz.solution, out("viz_grid_solved.png"), title="7×7 grid — solved")
    viz.save(pz, pz.solution[:30], out("viz_grid_partial.png"), title="partial path (30/49)")
    print(viz.to_ascii(pz, pz.solution[:30]))

    pz = demo_walls(rng)
    assert pz.is_valid_solution(pz.solution)
    viz.save(pz, None, out("viz_walls_puzzle.png"), title="7×7 with walls")
    viz.save(pz, pz.solution, out("viz_walls_solved.png"), title="7×7 with walls — solved")
    print(viz.to_ascii(pz, pz.solution))

    pz = demo_mask(rng)
    assert pz.is_valid_solution(pz.solution)
    viz.save(pz, pz.solution, out("viz_mask_solved.png"), title="irregular mask — solved")

    pz = demo_islands(rng)
    assert pz.is_valid_solution(pz.solution)
    viz.save(pz, None, out("viz_islands_puzzle.png"), title="islands + bridges")
    viz.save(pz, pz.solution, out("viz_islands_solved.png"), title="islands + bridges — solved")
    print(viz.to_ascii(pz, pz.solution))

    pz = demo_3d(rng)
    assert pz.is_valid_solution(pz.solution)
    viz.save(pz, pz.solution, out("viz_3d_solved.png"), title="4×4×4 — solved")
    viz.save(pz, pz.solution, out("viz_3d_slices.png"), slices=True,
             title="4×4×4 — layers")
    viz.save(pz, pz.solution[:37], out("viz_3d_partial.png"), title="4×4×4 — partial (37/64)")

    pz = demo_4d(rng)
    assert pz.is_valid_solution(pz.solution)
    viz.save(pz, pz.solution, out("viz_4d_solved.png"), title="3×3×3×3 — solved")

    pz = demo_generic(rng)
    assert pz.is_valid_solution(pz.solution)
    viz.save(pz, pz.solution, out("viz_generic_solved.png"), title="5D hypercube (generic)")

    pz = demo_grid(np.random.default_rng(1))
    viz.animate(pz, pz.solution, out("viz_grid_anim.gif"), fps=12)

    for f in made:
        print("wrote", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
