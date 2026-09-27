"""Write a few sample puzzles of every kind as JSON into samples/.

Usage: python scripts/gen_samples.py [--out samples] [--per-kind 3] [--seed 0] [--unique]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zipsolve.generator import make_puzzle

SPECS = [
    ("grid2d", 7, {}),
    ("grid2d", 6, {}),
    ("walls", 7, {"walls_frac": 0.3}),
    ("islands", 3, {}),
    ("islands", [(4, 4), (3, 4), (4, 3)], {}),
    ("grid3d", 4, {}),
    ("grid3d", 5, {}),
    ("grid4d", 3, {}),
    ("mask", 8, {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "samples"))
    ap.add_argument("--per-kind", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unique", action="store_true", help="require unique solutions (needs solver)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for kind, size, kw in SPECS:
        tag = size if isinstance(size, int) else f"{len(size)}isl"
        for i in range(args.per_kind):
            t = time.perf_counter()
            p = make_puzzle(kind, size, rng=rng, unique=args.unique, **kw)
            assert p.check_solution(p.solution) is None
            name = out / f"{kind}_{tag}_{i}.json"
            p.save(name)
            print(f"{name.name:28s} n={p.num_nodes:4d} checkpoints={len(p.checkpoints):3d} "
                  f"{(time.perf_counter() - t) * 1e3:7.1f} ms")


if __name__ == "__main__":
    main()
