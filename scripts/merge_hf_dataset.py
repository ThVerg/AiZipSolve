"""Merge dataset build parts into one publishable layout with unbiased splits.

Splits are a pure function of the puzzle id: x = int(id[8:16], 16) % 100 →
test (x < 5), validation (x < 10), train (rest). Using hash digits that played
no part in any earlier selection keeps the splits an unbiased ~90/5/5 even when
the parts were produced by interrupted runs.

    python scripts/merge_hf_dataset.py --parts hf_dataset/data hf_dataset_x1/data ... --out hf_dataset_10m
Output: <out>/data/<split>/<family>-<nnnnn>.parquet (zstd, ≤ --shard-rows rows each)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def split_of(pid: str) -> str:
    x = int(pid[8:16], 16) % 100
    return "test" if x < 5 else "validation" if x < 10 else "train"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-rows", type=int, default=200_000)
    args = ap.parse_args(argv)
    out = Path(args.out) / "data"
    writers: dict = {}
    counts: dict = defaultdict(int)
    shard: dict = defaultdict(int)
    seen = 0

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
            t = pq.read_table(f)
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
    print("total", seen)


if __name__ == "__main__":
    main()
