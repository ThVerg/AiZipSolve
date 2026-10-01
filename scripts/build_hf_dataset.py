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
The newer puzzle types (portals, torus, hex, tri, oneway, overpass, keys, cubesurf,
coop) are built with --set more; --set unique builds a subset of puzzles proven to
have exactly one solution, across all families, with a human difficulty rating:
    python scripts/build_hf_dataset.py --set more --n 10500000 --weight nodes \
        --existing hf_dataset_10m/data --target-total 20000000 --out hf_dataset_more
    python scripts/build_hf_dataset.py --set unique --n 210000 --target-total 200000 --out hf_dataset_unique
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
# newer puzzle types (see generator.make_puzzle / zipsolve.coop)
FAMILIES_MORE = {
    "portals": (list(range(5, 13)), 0.12),
    "torus": (list(range(4, 11)), 0.11),
    "hex": (list(range(3, 8)), 0.11),
    "tri": (list(range(2, 7)), 0.11),
    "oneway": (list(range(5, 13)), 0.11),
    "overpass": (list(range(5, 13)), 0.11),
    "keys": (list(range(5, 13)), 0.11),
    "cubesurf": (list(range(2, 6)), 0.11),
    "coop": (list(range(4, 11)), 0.11),
}
# unique-solution subset: sizes kept moderate so uniqueness can be proven quickly
FAMILIES_UNIQUE = {
    "grid2d": (list(range(5, 10)), 0.10), "walls": (list(range(5, 10)), 0.07),
    "mask": (list(range(6, 10)), 0.06), "islands": (list(range(2, 7)), 0.07),
    "islands_chain": ([2, 3], 0.03), "grid3d": ([3], 0.05), "grid4d": ([2, 3], 0.04),
    "portals": (list(range(5, 10)), 0.06), "torus": (list(range(4, 8)), 0.06),
    "hex": (list(range(3, 6)), 0.06), "tri": (list(range(2, 5)), 0.06),
    "oneway": (list(range(5, 10)), 0.06), "overpass": (list(range(5, 10)), 0.06),
    "keys": (list(range(5, 10)), 0.06), "cubesurf": ([2, 3], 0.05), "coop": (list(range(4, 8)), 0.05),
}
ALL_FAMILIES = list(FAMILIES) + list(FAMILIES_MORE)   # index = seed stream (main indices unchanged)
SETS = {"main": FAMILIES, "more": FAMILIES_MORE, "unique": FAMILIES_UNIQUE}
SEED_TAGS = {"main": 7777, "more": 7777, "unique": 8888}
DENSITY = {"sparse": 0.6, "default": 1.0, "dense": 1.6}
SEED_TAG = 7777  # keeps the dataset's random streams apart from training / benchmark seeds


def canonical_hash(coords, edges, checkpoints, extra=None) -> str:
    """Content id. `extra` (arcs, precedence, second path's checkpoints, overpass nodes …)
    is hashed too so puzzles that differ only in those rules get different ids; without
    it the id is identical to the original 10M build's."""
    h = hashlib.sha1()
    h.update(np.asarray(coords, dtype=np.int32).tobytes())
    h.update(np.asarray(sorted(tuple(sorted(e)) for e in edges), dtype=np.int32).tobytes())
    h.update(np.asarray(checkpoints, dtype=np.int32).tobytes())
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()[:16]


def approx_nodes(family: str, size: int) -> float:
    return {"grid2d": size ** 2, "walls": size ** 2, "mask": 0.75 * size ** 2, "islands": 14.0 * size,
            "islands_chain": 12.0 * size, "grid3d": size ** 3, "grid4d": size ** 4,
            "portals": size ** 2, "torus": size ** 2, "hex": 3 * size * size - 3 * size + 1,
            "tri": 6 * size * size, "oneway": size ** 2, "overpass": size ** 2 + size, "keys": size ** 2,
            "cubesurf": 6 * size * size, "coop": size ** 2}[family]


def size_label(family: str, size: int) -> str:
    if family in ("islands", "islands_chain"):
        return f"{size} islands"
    if family == "hex":
        return f"hexagon side {size}"
    if family == "tri":
        return f"triangles side {size}"
    if family == "cubesurf":
        return f"cube surface {size}x{size}x{size}"
    dim = {"grid3d": 3, "grid4d": 4}.get(family, 2)
    return "x".join([str(size)] * dim)


_EXTRA_META = ("portals", "torus", "wrap_edges", "overpass", "keys", "face", "cubesurf")


def _row_common(family, size, index, seed, density, g, checkpoints, solution, status, nodes, ms,
                extra=None, checkpoints2=None, solution2=None, rating=None, unique=None):
    coords = g.coords.astype(int).tolist()
    edges = [list(e) for e in g.edges()]
    meta = g.meta
    arcs = [list(map(int, a)) for a in meta.get("arcs", [])]
    prec = [list(map(int, a)) for a in meta.get("precedence", [])]
    ex = {k: meta[k] for k in _EXTRA_META if k in meta}
    lay = meta.get("layout")
    if lay is not None and not isinstance(lay, str):   # islands use meta.layout for their island grid
        ex["island_layout"] = list(lay)
        lay = None
    hash_extra = {k: v for k, v in (("arcs", arcs), ("precedence", prec), ("cp2", checkpoints2),
                                     ("overpass", ex.get("overpass"))) if v}
    return {
        "id": canonical_hash(coords, edges, checkpoints, hash_extra or None),
        "family": family,
        "size": size_label(family, size),
        "dim": int(g.dim),
        "num_nodes": int(g.num_nodes),
        "num_edges": len(edges),
        "num_checkpoints": len(checkpoints) + len(checkpoints2 or []),
        "density": density,
        "coords": coords,
        "edges": edges,
        "checkpoints": [int(c) for c in checkpoints],
        "solution": [int(v) for v in solution],
        "walls": [list(map(int, w)) for w in meta.get("walls", [])],
        "bridges": [list(map(int, b)) for b in meta.get("bridges", [])],
        "island": [int(i) for i in meta["island"]] if "island" in meta else None,
        "solver_status": status,
        "solver_nodes": int(nodes),
        "solver_ms": round(ms, 3),
        "seed": seed,
        "layout": lay or "square",
        "arcs": arcs,
        "precedence": prec,
        "extras": json.dumps(ex, default=int, separators=(",", ":")) if ex else None,
        "checkpoints2": [int(c) for c in checkpoints2] if checkpoints2 else None,
        "solution2": [int(v) for v in solution2] if solution2 else None,
        "unique": unique,
        "rating_score": None if rating is None else float(rating["score"]),
        "rating_label": None if rating is None else rating["label"],
    }


def make_row(family: str, size: int, index: int, solver_time: float, which: str = "main") -> dict | None:
    from zipsolve.generator import GenerationTimeout, default_num_checkpoints, make_puzzle
    from zipsolve.solver import solve

    fam_id = ALL_FAMILIES.index(family)
    seed = int(np.random.SeedSequence([SEED_TAGS[which], fam_id, size, index]).generate_state(1, np.uint32)[0])
    if which != "main" or family in FAMILIES_MORE:
        return _make_row_v2(family, size, index, seed, solver_time, unique=(which == "unique"))
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


def _make_row_v2(family, size, index, seed, solver_time, unique):
    """Rows for the newer kinds, and for the unique subset (any kind)."""
    from zipsolve.generator import GenerationTimeout, default_num_checkpoints, make_puzzle
    rng = np.random.default_rng(seed)
    density = "default" if unique else ["sparse", "default", "dense"][index % 3]
    k = default_num_checkpoints(int(approx_nodes(family, size)), rng)
    k = int(max(2, round(k * DENSITY[density])))
    tl = 30.0 if unique else 20.0
    if family == "coop":
        from zipsolve import coop
        base = coop.BASES[index % len(coop.BASES)]
        try:
            cp = coop.make_coop(size, num_checkpoints=None if unique else max(4, k), rng=rng, unique=unique,
                                base=base, time_limit=tl)
        except (GenerationTimeout, ValueError):
            return None
        sol = cp.solution
        if sol is None or coop.check(cp, sol) is not None:
            return None
        t0 = time.perf_counter()
        r = coop.solve(cp, time_limit=solver_time)
        ms = (time.perf_counter() - t0) * 1000
        rating, is_unique = None, None
        if unique:
            if coop.count_solutions(cp, 2, time_limit=30.0) != (1, "complete"):
                return None
            rating, is_unique = coop.rate(cp), True
        g = cp.graph
        g.meta.setdefault("layout", "hex" if base == "hex" else "square")
        return _row_common(family, size, index, seed, density, g, cp.checkpoints[0], sol[0], r.status,
                           r.nodes_expanded, ms, checkpoints2=cp.checkpoints[1], solution2=sol[1],
                           rating=rating, unique=is_unique)
    from zipsolve.solver import count_solutions, solve
    try:
        p = make_puzzle(family, size, num_checkpoints=None if unique else k, rng=rng, unique=unique,
                        time_limit=tl)
    except (GenerationTimeout, ValueError):
        return None
    if p.solution is None or p.check_solution(p.solution) is not None:
        return None
    t0 = time.perf_counter()
    r = solve(p, time_limit=solver_time)
    ms = (time.perf_counter() - t0) * 1000
    rating, is_unique = None, None
    if unique:
        if count_solutions(p, 2, time_limit=30.0) != (1, "complete"):
            return None
        from zipsolve.difficulty import rate
        try:
            rating = rate(p, p.solution, time_limit=10.0)
        except Exception:  # noqa: BLE001 - a rating failure should not drop a proven-unique puzzle
            rating = None
        is_unique = True
    return _row_common(family, size, index, seed, density, p.graph, p.checkpoints, p.solution,
                       r.status, r.nodes_expanded, ms, rating=rating, unique=is_unique)


def dataset_schema():
    """Parquet schema (v2). Columns after `split` are null for kinds that don't use them."""
    import pyarrow as pa
    i32, pairs = pa.int32(), pa.list_(pa.list_(pa.int32()))
    return pa.schema([
        ("id", pa.string()), ("family", pa.string()), ("size", pa.string()), ("dim", pa.int8()),
        ("num_nodes", pa.int32()), ("num_edges", pa.int32()), ("num_checkpoints", pa.int32()),
        ("density", pa.string()), ("coords", pa.list_(pa.list_(pa.int16()))), ("edges", pairs),
        ("checkpoints", pa.list_(i32)), ("solution", pa.list_(i32)), ("walls", pairs),
        ("bridges", pairs), ("island", pa.list_(pa.int16())), ("solver_status", pa.string()),
        ("solver_nodes", pa.int64()), ("solver_ms", pa.float32()), ("seed", pa.int64()),
        ("split", pa.string()),
        # v2: newer puzzle types, co-op and the unique subset
        ("layout", pa.string()), ("arcs", pairs), ("precedence", pairs), ("extras", pa.string()),
        ("checkpoints2", pa.list_(i32)), ("solution2", pa.list_(i32)), ("unique", pa.bool_()),
        ("rating_score", pa.float32()), ("rating_label", pa.string()),
    ])


def work(task):
    family, size, start, count, solver_time = task[:5]
    which = task[5] if len(task) > 5 else "main"
    rows = []
    for i in range(start, start + count):
        row = make_row(family, size, i, solver_time, which)
        if row is not None:
            rows.append(row)
    return family, rows


def split_of(pid: str) -> str:
    """Split from bytes 8..16 of sha1(id): independent of the id's own digits and of the
    thinning key (bytes 0..8), so the split stays an unbiased ~90/5/5 even when a build
    lost rows (crashed or stopped builds lose rows still buffered in memory)."""
    import hashlib
    x = int.from_bytes(hashlib.sha1(pid.encode()).digest()[8:16], "big") % 100
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
    ap.add_argument("--set", choices=list(SETS), default="main",
                    help="main: original families; more: newer puzzle types; unique: proven-unique subset")
    args = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = dataset_schema()

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
    for fam, (sizes, share) in SETS[args.set].items():
        w = [approx_nodes(fam, s) if args.weight == "nodes" else 1.0 for s in sizes]
        for s, ws in zip(sizes, w):
            per_size = math.ceil(args.n * share * ws / sum(w))
            for start in range(0, per_size, args.chunk):
                tasks.append((fam, s, args.index_offset + start, min(args.chunk, per_size - start),
                              args.solver_time, args.set))
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
