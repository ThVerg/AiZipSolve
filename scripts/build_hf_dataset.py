"""Build the Zip puzzles dataset (Parquet shards for the Hugging Face Hub).

Every row is one solvable puzzle with a known solution, across all families:

    grid2d 4-12, walls 5-12, mask 6-12, islands 2-8 (organic, back-and-forth),
    islands_chain 2-4, grid3d 3-6, grid4d 2-4

with a checkpoint density drawn from {sparse, default, dense}. Each puzzle is also
run through the exact solver (default settings, --solver-time limit) to record a
difficulty signal. Rows are deduplicated by content and puzzles that appear in the
frozen benchmark sets (benchmarks/val.json, test.json) are excluded.

Usage:
    python scripts/build_hf_dataset.py --n 1000000 --workers 48 --out hf_dataset
Extend an existing build to 10M unique puzzles (fresh seeds, dedup against it,
sizes weighted by cell count so small boards don't run out of distinct puzzles):
    python scripts/build_hf_dataset.py --n 9500000 --weight nodes --index-offset 10000000 \
        --existing hf_dataset/data --target-total 10000000 --tag x1 --out hf_dataset_x1
Layout:
    <out>/data/<split>/<family>-<shard>.parquet   (split = train / validation / test)
    <out>/stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

FAMILIES = {  # family -> (sizes, share of the dataset)
    "grid2d": (list(range(4, 13)), 0.30),
    "walls": (list(range(5, 13)), 0.15),
    "mask": (list(range(6, 13)), 0.12),
    "islands": (list(range(2, 9)), 0.15),
    "islands_chain": (list(range(2, 5)), 0.04),
    "grid3d": (list(range(3, 7)), 0.15),
    "grid4d": (list(range(2, 5)), 0.09),
}
DENSITY = {"sparse": 0.6, "default": 1.0, "dense": 1.6}
SEED_TAG = 7777  # keeps the dataset's random streams apart from training / benchmark seeds


def canonical_hash(coords, edges, checkpoints) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(coords, dtype=np.int32).tobytes())
    h.update(np.asarray(sorted(tuple(sorted(e)) for e in edges), dtype=np.int32).tobytes())
    h.update(np.asarray(checkpoints, dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


def approx_nodes(family: str, size: int) -> float:
    return {"grid2d": size ** 2, "walls": size ** 2, "mask": 0.75 * size ** 2, "islands": 14.0 * size,
            "islands_chain": 12.0 * size, "grid3d": size ** 3, "grid4d": size ** 4}[family]


def size_label(family: str, size: int) -> str:
    if family in ("islands", "islands_chain"):
        return f"{size} islands"
    dim = {"grid3d": 3, "grid4d": 4}.get(family, 2)
    return "x".join([str(size)] * dim)


def make_row(family: str, size: int, index: int, solver_time: float) -> dict | None:
    from zipsolve.generator import GenerationTimeout, default_num_checkpoints, make_puzzle
    from zipsolve.solver import solve

    fam_id = list(FAMILIES).index(family)
    seed = int(np.random.SeedSequence([SEED_TAG, fam_id, size, index]).generate_state(1, np.uint32)[0])
    rng = np.random.default_rng(seed)
    density = ["sparse", "default", "dense"][index % 3]
    try:
        # draw the graph first at the generator's default, then re-place checkpoints for the density
        p = make_puzzle(family, size, rng=rng, time_limit=20.0)
    except (GenerationTimeout, ValueError):
        return None
    k = default_num_checkpoints(p.num_nodes, rng)
    k = int(min(p.num_nodes, max(2, round(k * DENSITY[density]))))
    from zipsolve.generator import place_checkpoints
    from zipsolve.puzzle import Puzzle
    p = Puzzle(p.graph, place_checkpoints(p.solution, k, rng), p.solution)
    assert p.check_solution(p.solution) is None

    g = p.graph
    coords = g.coords.astype(int).tolist()
    edges = [list(e) for e in g.edges()]
    t0 = time.perf_counter()
    r = solve(p, time_limit=solver_time)
    solver_ms = (time.perf_counter() - t0) * 1000
    meta = g.meta
    return {
        "id": canonical_hash(coords, edges, p.checkpoints),
        "family": family,
        "size": size_label(family, size),
        "dim": int(g.dim),
        "num_nodes": int(g.num_nodes),
        "num_edges": len(edges),
        "num_checkpoints": len(p.checkpoints),
        "density": density,
        "coords": coords,
        "edges": edges,
        "checkpoints": [int(c) for c in p.checkpoints],
        "solution": [int(v) for v in p.solution],
        "walls": [list(map(int, w)) for w in meta.get("walls", [])],
        "bridges": [list(map(int, b)) for b in meta.get("bridges", [])],
        "island": [int(i) for i in meta["island"]] if "island" in meta else None,
        "solver_status": r.status,
        "solver_nodes": int(r.nodes_expanded),
        "solver_ms": round(solver_ms, 3),
        "seed": seed,
    }


def work(task):
    family, size, start, count, solver_time = task[:5]
    rows = []
    for i in range(start, start + count):
        row = make_row(family, size, i, solver_time)
        if row is not None:
            rows.append(row)
    return family, rows


def split_of(pid: str) -> str:
    x = int(pid[:8], 16) % 100
    return "test" if x < 5 else "validation" if x < 10 else "train"


def benchmark_hashes(root: Path) -> set[str]:
    out = set()
    for name in ("val.json", "test.json"):
        f = root / "benchmarks" / name
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        items = data["records"] if isinstance(data, dict) else data
        for it in items:
            d = it.get("puzzle", it)
            out.add(canonical_hash(np.asarray(d["coords"]).astype(int).tolist(), d["edges"], d["checkpoints"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1_000_000, help="target number of puzzles")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="hf_dataset")
    ap.add_argument("--chunk", type=int, default=250, help="puzzles per worker task")
    ap.add_argument("--shard-rows", type=int, default=50_000)
    ap.add_argument("--max-buffered", type=int, default=150_000,
                    help="rows held in memory across all shards before the largest is written early")
    ap.add_argument("--solver-time", type=float, default=1.0)
    ap.add_argument("--weight", choices=["uniform", "nodes"], default="uniform",
                    help="split each family's share over sizes uniformly or by cell count")
    ap.add_argument("--index-offset", type=int, default=0, help="first puzzle index (new seeds)")
    ap.add_argument("--existing", nargs="*", default=[], help="parquet dirs of earlier builds to dedup against")
    ap.add_argument("--target-total", type=int, default=0,
                    help="stop once existing + new unique rows reach this (0 = run all tasks)")
    ap.add_argument("--tag", default="", help="shard name tag, e.g. x1 -> grid2d-x1-00000.parquet")
    args = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq

    i32, pairs = pa.int32(), pa.list_(pa.list_(pa.int32()))
    schema = pa.schema([
        ("id", pa.string()), ("family", pa.string()), ("size", pa.string()), ("dim", pa.int8()),
        ("num_nodes", pa.int32()), ("num_edges", pa.int32()), ("num_checkpoints", pa.int32()),
        ("density", pa.string()), ("coords", pa.list_(pa.list_(pa.int16()))), ("edges", pairs),
        ("checkpoints", pa.list_(i32)), ("solution", pa.list_(i32)), ("walls", pairs),
        ("bridges", pairs), ("island", pa.list_(pa.int16())), ("solver_status", pa.string()),
        ("solver_nodes", pa.int64()), ("solver_ms", pa.float32()), ("seed", pa.int64()),
        ("split", pa.string()),
    ])

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    exclude = benchmark_hashes(root)
    print(f"excluding {len(exclude)} benchmark puzzles", flush=True)

    existing: set[str] = set()
    for d in args.existing:
        for f in sorted(Path(d).rglob("*.parquet")):
            existing.update(pq.read_table(f, columns=["id"]).column("id").to_pylist())
    if existing:
        print(f"deduplicating against {len(existing)} existing puzzles", flush=True)

    tasks = []
    for fam, (sizes, share) in FAMILIES.items():
        w = [approx_nodes(fam, s) if args.weight == "nodes" else 1.0 for s in sizes]
        for s, ws in zip(sizes, w):
            per_size = math.ceil(args.n * share * ws / sum(w))
            for start in range(0, per_size, args.chunk):
                tasks.append((fam, s, args.index_offset + start, min(args.chunk, per_size - start),
                              args.solver_time))
    # random order, so stopping at --target-total keeps the family / size mix balanced
    np.random.default_rng(SEED_TAG).shuffle(tasks)
    total = sum(t[3] for t in tasks)
    print(f"{len(tasks)} tasks, {total} puzzles requested, {args.workers} workers", flush=True)

    seen: set[str] = set()
    buffers: dict[tuple[str, str], list] = defaultdict(list)
    shard_no: Counter = Counter()
    stats = {"rows": Counter(), "dupes": 0, "excluded": 0, "failed": 0, "solver_status": Counter()}

    def flush(key, force=False):
        rows = buffers[key]
        while rows and (force or len(rows) >= args.shard_rows):
            part, rows[:] = rows[:args.shard_rows], rows[args.shard_rows:]
            split, fam = key
            d = out / "data" / split
            d.mkdir(parents=True, exist_ok=True)
            tag = f"{args.tag}-" if args.tag else ""
            pq.write_table(pa.Table.from_pylist(part, schema=schema), d / f"{fam}-{tag}{shard_no[key]:05d}.parquet",
                           compression="zstd")
            shard_no[key] += 1
            if not force:
                buffered[0] -= len(part)
                break

    t0 = time.time()
    done = 0
    buffered = [0]

    def handle(rows):
        """Dedup + buffer one finished batch. Returns False once the target is reached."""
        fam = rows[0]["family"] if rows else None
        for row in rows:
            if args.target_total and len(existing) + len(seen) >= args.target_total:
                return False  # exact stop: never write more than the target
            if row["id"] in exclude:
                stats["excluded"] += 1
                continue
            if row["id"] in seen or row["id"] in existing:
                stats["dupes"] += 1
                continue
            seen.add(row["id"])
            split = split_of(row["id"])
            row["split"] = split
            stats["rows"][f"{split}/{fam}"] += 1
            stats["solver_status"][row["solver_status"]] += 1
            key = (split, fam)
            buffers[key].append(row)
            flush(key)
            buffered[0] += 1
            if buffered[0] >= args.max_buffered:
                # slow-filling shards (validation/test, rare families) are written early so
                # memory stays bounded; this crashed a 10M build once
                big = max(buffers, key=lambda k: len(buffers[k]))
                flush(big, force=True)
                buffered[0] = sum(len(v) for v in buffers.values())
        return not (args.target_total and len(existing) + len(seen) >= args.target_total)

    # bounded submission: finished futures are dropped right away, so memory stays
    # flat no matter how many puzzles are generated (holding every Future's result
    # list is what OOM-killed the first 10M attempt)
    with ProcessPoolExecutor(args.workers) as ex:
        pending, it, running = set(), iter(tasks), True
        while running:
            while len(pending) < args.workers * 4:
                t = next(it, None)
                if t is None:
                    break
                pending.add(ex.submit(work, t))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                _, rows = f.result()
                done += 1
                if not handle(rows):
                    running = False
                    break
                if done % 200 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(tasks)} tasks, {len(seen)} rows, {el:.0f}s", flush=True)
            del finished
        if not running:
            print(f"reached target {args.target_total}; cancelling remaining tasks", flush=True)
            for f in pending:
                f.cancel()
    for key in list(buffers):
        flush(key, force=True)

    stats["total"] = len(seen)
    stats["existing"] = len(existing)
    stats["requested"] = total
    stats["failed"] = total - len(seen) - stats["dupes"] - stats["excluded"]
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=dict))
    print(json.dumps(stats, indent=2, default=dict))


if __name__ == "__main__":
    main()
