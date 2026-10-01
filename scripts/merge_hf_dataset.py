"""Merge dataset build parts into one publishable layout with unbiased splits.

Splits are a pure function of the puzzle id: x = int(sha1(id)[8:16]) % 100 →
test (x < 5), validation (x < 10), train (rest). That hash played no part in any
earlier selection, so the splits are an unbiased ~90/5/5 even when the parts were
produced by interrupted runs (which lose rows still buffered in memory).

    python scripts/merge_hf_dataset.py --parts hf_dataset/data hf_dataset_x1/data ... --out hf_dataset_10m
Output: <out>/data/<split>/<family>-<nnnnn>.parquet (zstd, ≤ --shard-rows rows each)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def to_schema(t: pa.Table, schema: pa.Schema) -> pa.Table:
    """Add any missing (v2) columns as nulls and order/cast to `schema`."""
    cols = []
    for field in schema:
        if field.name in t.column_names:
            cols.append(t.column(field.name).cast(field.type))
        else:
            cols.append(pa.nulls(t.num_rows, type=field.type))
    return pa.Table.from_arrays(cols, schema=schema)


def _thin_key(pid: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha1(pid.encode()).digest()[:8], "big")


def _thin_set(parts, keep_parts: int, limit: int) -> set[str]:
    """Ids to drop so exactly `limit` unique rows remain: the thinnable ids with the smallest
    hash keys (a uniform random sample, independent of split, family and file order)."""
    fixed: set[str] = set()
    for part in parts[:keep_parts]:
        for f in Path(part).rglob("*.parquet"):
            fixed.update(pq.read_table(f, columns=["id"]).column("id").to_pylist())
    thin: set[str] = set()
    for part in parts[keep_parts:]:
        for f in Path(part).rglob("*.parquet"):
            thin.update(i for i in pq.read_table(f, columns=["id"]).column("id").to_pylist() if i not in fixed)
    excess = len(fixed) + len(thin) - limit
    if excess <= 0:
        return set()
    return set(sorted(thin, key=_thin_key)[:excess])


def split_of(pid: str) -> str:
    """Split from bytes 8..16 of sha1(id): independent of the id's own digits and of the
    thinning key (bytes 0..8), so the split stays an unbiased ~90/5/5 even when a build
    lost rows (crashed or stopped builds lose rows still buffered in memory)."""
    import hashlib
    x = int.from_bytes(hashlib.sha1(pid.encode()).digest()[8:16], "big") % 100
    return "test" if x < 5 else "validation" if x < 10 else "train"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-rows", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the total unique rows; the surplus is dropped uniformly at random (by a hash "
                         "of the id) from the parts after the first --keep-parts, so splits and the family "
                         "mix stay unbiased (0 = keep all)")
    ap.add_argument("--keep-parts", type=int, default=1, help="leading parts that are never thinned")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_hf_dataset import dataset_schema
    schema = dataset_schema()
    out = Path(args.out) / "data"
    writers: dict = {}
    counts: dict = defaultdict(int)
    shard: dict = defaultdict(int)
    seen = 0
    ids: set[str] = set()   # parts built concurrently may overlap: keep the first copy of each id
    dupes = 0
    drop: set[str] = set()
    if args.limit:
        drop = _thin_set(args.parts, args.keep_parts, args.limit)
        print(f"thinning: dropping {len(drop)} rows uniformly from parts {args.keep_parts}+", flush=True)

    def writer(split, fam, schema):
        key = (split, fam)
        if key in writers and counts[key] >= args.shard_rows:
            writers.pop(key).close()
            shard[key] += 1
            counts[key] = 0
        if key not in writers:
            (out / split).mkdir(parents=True, exist_ok=True)
            writers[key] = pq.ParquetWriter(out / split / f"{fam}-{shard[key]:05d}.parquet", schema,
                                            compression="zstd")
        return writers[key]

    for part in args.parts:
        for f in sorted(Path(part).rglob("*.parquet")):
            t = to_schema(pq.read_table(f), schema)
            keep = []
            for k, i in enumerate(t.column("id").to_pylist()):
                if i in ids or i in drop:
                    dupes += i in ids
                    continue
                ids.add(i)
                keep.append(k)
            if len(keep) < t.num_rows:
                t = t.take(pa.array(keep, type=pa.int64()))
            if t.num_rows == 0:
                continue
            splits = pa.array([split_of(i) for i in t.column("id").to_pylist()])
            t = t.set_column(t.schema.get_field_index("split"), "split", splits)
            for fam in pc.unique(t.column("family")).to_pylist():
                tf = t.filter(pc.equal(t.column("family"), fam))
                for sp in ("train", "validation", "test"):
                    ts = tf.filter(pc.equal(tf.column("split"), sp))
                    off = 0
                    while off < ts.num_rows:  # respect the shard size inside one input file too
                        w = writer(sp, fam, ts.schema)
                        room = args.shard_rows - counts[(sp, fam)]
                        chunk = ts.slice(off, room)
                        w.write_table(chunk)
                        counts[(sp, fam)] += chunk.num_rows
                        off += chunk.num_rows
            seen += t.num_rows
        print(f"{part}: done, {seen} rows so far", flush=True)
    for w in writers.values():
        w.close()
    print("total", seen, "duplicates dropped", dupes)


if __name__ == "__main__":
    main()
