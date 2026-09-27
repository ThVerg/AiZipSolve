"""Timing table for zipsolve.solver.

Puzzles are built without the generator: start from a boustrophedon (snake)
Hamiltonian path, randomise it with backbite moves (a random walk on the set
of Hamiltonian paths, valid on any graph), then pick checkpoints along it.

    python scripts/bench_solver.py [--seeds 5] [--limit 30] [--no-deep]
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from zipsolve.graph import grid
from zipsolve.puzzle import Puzzle
from zipsolve.solver import count_solutions, solve


def snake_coords(shape):
    if len(shape) == 1:
        return [(i,) for i in range(shape[0])]
    sub = snake_coords(shape[1:])
    out = []
    for i in range(shape[0]):
        out.extend((i,) + c for c in (sub if i % 2 == 0 else sub[::-1]))
    return out


def backbite(graph, path, steps, rng):
    """Randomise a Hamiltonian path with backbite moves."""
    path = list(path)
    n = len(path)
    for _ in range(steps):
        if rng.random() < 0.5:
            path.reverse()
        pos = {v: i for i, v in enumerate(path)}
        end = path[-1]
        w = int(rng.choice(graph.neighbors[end]))
        j = pos[w]
        if j == n - 2:
            continue
        # edge end-w added, edge w-path[j+1] removed: reverse the tail after j
        path[j + 1:] = path[j + 1:][::-1]
    return path


def random_puzzle(shape, n_cp, rng):
    g = grid(*shape)
    path = [g.node_at(c) for c in snake_coords(shape)]
    path = backbite(g, path, 20 * g.num_nodes, rng)
    assert Puzzle(g, [path[0], path[-1]]).check_solution(path) is None
    inner = sorted(rng.choice(np.arange(1, len(path) - 1), size=n_cp - 2, replace=False).tolist())
    cps = [path[0]] + [path[i] for i in inner] + [path[-1]]
    return Puzzle(g, cps, path)


CASES = [
    ("7x7", (7, 7), 8), ("7x7", (7, 7), 12),
    ("8x8", (8, 8), 8), ("8x8", (8, 8), 12),
    ("5x5x5", (5, 5, 5), 10),
    ("6x6x6", (6, 6, 6), 12),
    ("3x3x3x3", (3, 3, 3, 3), 10),
    ("4x4x4x4", (4, 4, 4, 4), 12),
    ("10x10", (10, 10), 15),
]
COUNT_CASES = [("6x6", (6, 6), 8), ("7x7", (7, 7), 12), ("8x8", (8, 8), 16),
               ("4x4x4", (4, 4, 4), 10), ("3x3x3x3", (3, 3, 3, 3), 12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--limit", type=float, default=30.0)
    ap.add_argument("--no-deep", action="store_true", help="disable articulation-point pruning")
    a = ap.parse_args()
    print(f"{'case':10s} {'cps':>3s} {'n':>4s} | {'solved':>6s} {'timeout':>7s} | "
          f"{'median s':>9s} {'max s':>8s} | {'median exp':>10s} {'max exp':>9s}")
    print("-" * 82)
    for name, shape, ncp in CASES:
        times, exps, solved, to = [], [], 0, 0
        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 * seed + ncp)
            p = random_puzzle(shape, ncp, rng)
            t = time.perf_counter()
            r = solve(p, time_limit=a.limit, deep=not a.no_deep)
            times.append(time.perf_counter() - t)
            exps.append(r.nodes_expanded)
            if r.status == "solved":
                assert p.check_solution(r.path) is None
                solved += 1
            elif r.status == "timeout":
                to += 1
            else:
                raise AssertionError("solver reported unsat on a solvable puzzle")
        print(f"{name:10s} {ncp:3d} {p.num_nodes:4d} | {solved:6d} {to:7d} | "
              f"{statistics.median(times):9.4f} {max(times):8.3f} | "
              f"{int(statistics.median(exps)):10d} {max(exps):9d}")

    print()
    print("count_solutions(limit=2)  (uniqueness check, as a generator would use it)")
    print(f"{'case':10s} {'cps':>3s} | {'unique':>6s} {'multi':>5s} {'timeout':>7s} | "
          f"{'median s':>9s} {'max s':>8s}")
    print("-" * 58)
    for name, shape, ncp in COUNT_CASES:
        times, res = [], {"complete": 0, "limit": 0, "timeout": 0}
        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 * seed + ncp)
            p = random_puzzle(shape, ncp, rng)
            t = time.perf_counter()
            c, st = count_solutions(p, limit=2, time_limit=a.limit)
            times.append(time.perf_counter() - t)
            res[st] += 1
        print(f"{name:10s} {ncp:3d} | {res['complete']:6d} {res['limit']:5d} {res['timeout']:7d} | "
              f"{statistics.median(times):9.4f} {max(times):8.3f}")


if __name__ == "__main__":
    main()
