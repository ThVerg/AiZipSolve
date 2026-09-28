"""Build the curated puzzle bank (zipsolve/app/static/bank/) used by the game
(local app and the static site): difficulty pools per mode, the daily schedule
2026-01-01 .. 2027-12-31 and precomputed robot runs.

    python scripts/build_bank.py                 # everything (resumable)
    python scripts/build_bank.py --stage candidates|select|robots
    python scripts/build_bank.py --workers 12 --pool 80

Pipeline
  1. candidates: generate unique-solution puzzles (generator unique=True, then
     re-verified with count_solutions(p, 2) == (1, "complete")) for every
     source configuration below and rate them with zipsolve.difficulty.
     Cached in <cache>/candidates.jsonl (a rerun only generates what is missing).
  2. select: fill pools (mode x easy/medium/hard) and the daily schedule from
     the candidates (see BUCKETS) - all bank puzzles are distinct; dailies
     are not boring, have a good checkpoint spread and follow a weekly ramp
     (Monday easiest, Sunday hardest within each difficulty).
  3. robots: precompute the four robots of the AI show for every bank puzzle
     (rookie = greedy, scout = policy DFS, grandmaster = solver + GNN hybrid,
     tortoise = exact solver), in the app's /api/solve/* response shape with
     traces capped at TRACE_CAP raw events. One file per puzzle.
The format is documented in zipsolve/app/static/bank/README.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "zipsolve" / "app" / "static" / "bank"
CACHE = Path(os.environ.get("ZIP_BANK_CACHE") or Path.home() / ".cache" / "zip-bank")
MODEL = ROOT / "checkpoints" / "zip_gnn.pt"
DAILY_START = dt.date(2026, 1, 1)
DAILY_END = dt.date(2027, 12, 31)
TRACE_CAP = 4000
MODES = ("classic", "walls", "islands", "cube")
DIFFS = ("easy", "medium", "hard")
MODE_TAG = {"classic": "c", "walls": "w", "islands": "i", "cube": "q"}

# ---------------------------------------------------------------------------
# candidate sources: name -> (mode, kind, size, extra kwargs, checkpoint counts, how many)
# num_checkpoints is the *initial* count; unique=True adds more until unique,
# so small values give sparse (harder) puzzles and None the generator default.
# ---------------------------------------------------------------------------
SOURCES = {
    "c5": ("classic", "grid2d", 5, {}, [None, None, 4, 5, 6], 1000),
    "c6": ("classic", "grid2d", 6, {}, [None, None, 3, 4, 5, 6], 1300),
    "c7": ("classic", "grid2d", 7, {}, [None, None, 3, 4, 5, 6], 1700),
    "c8": ("classic", "grid2d", 8, {}, [None, 3, 4, 5, 6, 7], 1100),
    "w6": ("walls", "walls", 6, {"walls_frac": 0.3}, [None, 3, 4, 5], 300),
    "w7": ("walls", "walls", 7, {"walls_frac": 0.3}, [None, 3, 4, 5], 400),
    "w8": ("walls", "walls", 8, {"walls_frac": 0.3}, [None, 3, 4, 5], 400),
    "i3": ("islands", "islands", 3, {}, [None, None, 4, 5], 300),
    "i4": ("islands", "islands", 4, {}, [None, 3, 4, 5], 400),
    "i5": ("islands", "islands", 5, {}, [None, 3, 4, 5], 500),
    "i6": ("islands", "islands", 6, {}, [None, 3, 4, 5], 300),
    "q3": ("cube", "grid3d", 3, {}, [None, None, 4, 5, 6], 450),
    "q334": ("cube", "grid", (3, 3, 4), {}, [None, 4, 5, 6], 350),
    "q344": ("cube", "grid", (4, 4, 3), {}, [None, 4, 5, 6], 300),
    "q4": ("cube", "grid3d", 4, {}, [None, 8, 10], 150),
}

# bucket -> (sources, accepted labels, extra score window)
BUCKETS = {
    ("classic", "easy"): (["c5", "c6"], {"Easy"}, (5, 100)),
    ("classic", "medium"): (["c6", "c7"], {"Medium"}, (0, 100)),
    ("classic", "hard"): (["c7", "c8"], {"Hard", "Expert"}, (0, 100)),
    ("walls", "easy"): (["w6"], {"Easy"}, (5, 100)),
    ("walls", "medium"): (["w7", "w6"], {"Medium"}, (0, 100)),
    ("walls", "hard"): (["w8", "w7"], {"Hard", "Expert"}, (0, 100)),
    ("islands", "easy"): (["i3", "i4"], {"Easy"}, (5, 100)),
    ("islands", "medium"): (["i4", "i5"], {"Medium"}, (0, 100)),
    ("islands", "hard"): (["i5", "i6"], {"Hard", "Expert", "Insane"}, (0, 100)),
    ("cube", "easy"): (["q3"], {"Easy", "Medium", "Hard"}, (0, 45)),
    ("cube", "medium"): (["q3", "q334"], {"Hard", "Expert"}, (45, 72)),
    ("cube", "hard"): (["q344", "q4"], {"Expert", "Insane"}, (72, 101)),
}
SPECIAL_ROTATION = ("islands", "walls", "cube")     # Sunday specials cycle through these
SPECIAL_DIFF = {"islands": "hard", "walls": "hard", "cube": "medium"}
MIN_SPREAD = 0.8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def dumps(o) -> str:
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


def compact_puzzle(p) -> dict:
    d = p.to_dict()
    d["coords"] = [[int(x) if float(x).is_integer() else x for x in c] for c in d["coords"]]
    return d


def puzzle_id(mode: str, d: dict) -> str:
    h = hashlib.sha1(dumps([d["coords"], d["edges"], d["checkpoints"]]).encode()).hexdigest()
    return f"{MODE_TAG[mode]}{h[:9]}"


def size_text(mode: str, kind: str, size) -> str:
    if mode == "islands":
        return f"{size} islands"
    if isinstance(size, (tuple, list)):
        return "x".join(map(str, size))
    return "x".join([str(size)] * (3 if kind == "grid3d" else 2))


# ---------------------------------------------------------------------------
# stage 1: candidates
# ---------------------------------------------------------------------------
def _cand_job(job):
    src, i = job
    mode, kind, size, kw, ncps, _ = SOURCES[src]
    from zipsolve.difficulty import rate
    from zipsolve.generator import make_puzzle
    from zipsolve.solver import count_solutions
    seed = int.from_bytes(hashlib.sha256(f"zip-bank:{src}:{i}".encode()).digest()[:6], "big")
    ncp = ncps[i % len(ncps)]
    t0 = time.perf_counter()
    try:
        p = make_puzzle(kind, size, num_checkpoints=ncp, rng=seed, unique=True, time_limit=90, **kw)
    except ValueError:
        return {"src": src, "i": i, "fail": True}
    if count_solutions(p, 2, time_limit=120) != (1, "complete"):
        return {"src": src, "i": i, "fail": True}
    r = rate(p)
    d = compact_puzzle(p)
    return {"src": src, "i": i, "mode": mode, "id": puzzle_id(mode, d), "size": size_text(mode, kind, size),
            "score": r["score"], "label": r["label"], "boring": r["boring"],
            "features": {k: r["features"][k] for k in ("nodes", "checkpoints", "forced_frac", "max_lookahead",
                                                       "branching_points", "guesses", "spread", "effort")},
            "seconds": round(time.perf_counter() - t0, 2), "puzzle": d}


def stage_candidates(workers: int) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "candidates.jsonl"
    have = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                have[(c["src"], c["i"])] = c
    jobs = [(src, i) for src, s in SOURCES.items() for i in range(s[5]) if (src, i) not in have]
    # slow sources first so the pool stays busy
    jobs.sort(key=lambda j: j[0] not in ("q4", "q344", "i6"))
    t0 = time.perf_counter()
    if jobs:
        print(f"[candidates] generating {len(jobs)} ({len(have)} cached) with {workers} workers", flush=True)
        with Pool(workers) as pool, f.open("a") as out:
            for k, c in enumerate(pool.imap_unordered(_cand_job, jobs, chunksize=4)):
                have[(c["src"], c["i"])] = c
                out.write(dumps(c) + "\n")
                if (k + 1) % 500 == 0:
                    print(f"  {k + 1}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[candidates] {len(have)} total, {time.perf_counter() - t0:.0f}s", flush=True)
    return [c for c in have.values() if not c.get("fail")]


def report_candidates(cands: list[dict]) -> None:
    import numpy as np
    by = defaultdict(list)
    for c in cands:
        by[c["src"]].append(c)
    print(f"{'src':6} {'n':>5} {'med':>5} {'p10':>5} {'p90':>5}  labels  boring  sec")
    for src in SOURCES:
        cs = by.get(src, [])
        if not cs:
            continue
        sc = [c["score"] for c in cs]
        labs = defaultdict(int)
        for c in cs:
            labs[c["label"][:2]] += 1
        print(f"{src:6} {len(cs):>5} {np.median(sc):5.1f} {np.percentile(sc, 10):5.1f} {np.percentile(sc, 90):5.1f}  "
              + " ".join(f"{k}{v}" for k, v in sorted(labs.items()))
              + f"  {sum(c['boring'] for c in cs):>4}  {np.mean([c['seconds'] for c in cs]):.2f}")


# ---------------------------------------------------------------------------
# stage 2: select
# ---------------------------------------------------------------------------
def entry(c: dict, diff: str) -> dict:
    return {"id": c["id"], "mode": c["mode"], "diff": diff, "label": c["label"], "score": c["score"],
            "size": c["size"], "puzzle": c["puzzle"]}


def good(c: dict, bucket) -> bool:
    srcs, labels, (lo, hi) = BUCKETS[bucket]
    return (c["src"] in srcs and c["label"] in labels and lo <= c["score"] < hi and not c["boring"]
            and c["features"]["spread"] >= MIN_SPREAD)


def daily_dates():
    d = DAILY_START
    while d <= DAILY_END:
        yield d
        d += dt.timedelta(days=1)


def stage_select(cands: list[dict], pool_size: int) -> dict:
    rnd = random.Random(20260101)
    cands = sorted({c["id"]: c for c in cands}.values(), key=lambda c: c["id"])   # distinct, stable order
    used: set[str] = set()
    avail: dict = {}
    for b in BUCKETS:
        lst = [c for c in cands if good(c, b)]
        rnd.shuffle(lst)
        # prefer the size listed first in the bucket for about 60 % of the picks
        avail[b] = lst

    def take(b, k):
        out = []
        for c in avail[b]:
            if len(out) >= k:
                break
            if c["id"] in used:
                continue
            used.add(c["id"])
            out.append(c)
        if len(out) < k:
            print(f"  ! bucket {b}: only {len(out)} of {k}", flush=True)
        return out

    dates = list(daily_dates())
    sundays = [d for d in dates if d.weekday() == 6]
    # dailies first for classic (they need the most), then specials, then pools
    daily = {}
    for diff in DIFFS:
        picks = take(("classic", diff), len(dates))
        weeks = defaultdict(list)
        for d in dates:
            weeks[d.isocalendar()[:2]].append(d)
        it = iter(picks)
        for wk in sorted(weeks):
            ds = weeks[wk]
            chunk = [c for _, c in zip(ds, it)]
            chunk.sort(key=lambda c: c["score"])     # weekly ramp: Monday easiest
            for d, c in zip(ds, chunk):
                daily.setdefault(d, {})[diff] = entry(c, diff)
    for k, d in enumerate(sundays):
        mode = SPECIAL_ROTATION[k % len(SPECIAL_ROTATION)]
        got = take((mode, SPECIAL_DIFF[mode]), 1)
        daily.setdefault(d, {})["special"] = entry(got[0], SPECIAL_DIFF[mode]) if got else None
    pools = {b: [entry(c, b[1]) for c in take(b, pool_size)] for b in BUCKETS}
    for b in pools:
        pools[b].sort(key=lambda e: e["id"])
    return {"pools": pools, "daily": daily}


def write_bank(sel: dict) -> list[dict]:
    import shutil
    # additive: pools / index keys of other builders (scripts/architect_designs.py, build_bank_modes.py) are kept
    try:
        old = json.loads((OUT / "index.json").read_text())
    except (OSError, ValueError):
        old = {}
    if (OUT / "daily").exists():
        shutil.rmtree(OUT / "daily")
    (OUT / "daily").mkdir(parents=True)
    (OUT / "pools").mkdir(parents=True, exist_ok=True)
    for m in MODES:
        for f in (OUT / "pools").glob(f"{m}-*.json"):
            f.unlink()
    index = {"version": 1, "generated": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
             "daily_start": DAILY_START.isoformat(), "daily_end": DAILY_END.isoformat(),
             "daily_epoch": DAILY_START.isoformat(),
             "labels": ["Easy", "Medium", "Hard", "Expert", "Insane"],
             "robots": "robots/{id}.json", "modes": {}}
    allents = {}
    for (mode, diff), ents in sel["pools"].items():
        fn = f"pools/{mode}-{diff}.json"
        (OUT / fn).write_text(dumps(ents))
        index["modes"].setdefault(mode, {})[diff] = {"file": fn, "count": len(ents)}
        for e in ents:
            allents[e["id"]] = e
    months = defaultdict(dict)
    for d, row in sorted(sel["daily"].items()):
        rec = {"number": (d - DAILY_START).days + 1}
        for k in (*DIFFS, "special"):
            rec[k] = row.get(k)
            if row.get(k):
                allents[row[k]["id"]] = row[k]
        months[f"{d.year:04d}-{d.month:02d}"][d.isoformat()] = rec
    for m, recs in months.items():
        (OUT / "daily" / f"{m}.json").write_text(dumps(recs))
    index["daily_files"] = "daily/{yyyy}-{mm}.json"
    index["puzzles"] = len(allents)
    for m, ds in (old.get("modes") or {}).items():
        if m not in MODES:
            index["modes"][m] = ds
    for k, v in old.items():
        if k not in index:
            index[k] = v
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"[select] {len(allents)} bank puzzles; pools "
          + ", ".join(f"{m}-{d}:{len(v)}" for (m, d), v in sel["pools"].items()), flush=True)
    return list(allents.values())


# ---------------------------------------------------------------------------
# stage 3: robots
# ---------------------------------------------------------------------------
_W = {}


def _robot_init():
    import torch
    torch.set_num_threads(1)
    from zipsolve.app import engine
    from zipsolve.rl.gnn import load_model
    engine.TRACE_CAP = TRACE_CAP
    model, ck = load_model(MODEL)
    _W["model"], _W["meta"] = model, {k: v for k, v in ck.items() if k != "state_dict"}


def _slim(r: dict) -> dict:
    keep = ("mode", "status", "solved", "path", "start_len", "nodes_expanded", "backtracks", "seconds",
            "attempts", "stuck_at", "dead_from", "mean_confidence", "min_confidence", "trace", "trace_truncated")
    out = {k: r[k] for k in keep if k in r}
    for k in ("seconds", "mean_confidence", "min_confidence"):
        if isinstance(out.get(k), float):
            out[k] = round(out[k], 3)
    if "steps" in r:
        out["steps"] = [{"p": round(s["p"], 2)} for s in r["steps"]]
    if "trace" in out and out.get("trace") is not None:
        out["trace_truncated"] = bool(out.get("trace_truncated"))
    return out


def _robot_job(e: dict):
    from zipsolve.app import engine
    from zipsolve.puzzle import Puzzle
    d = dict(e["puzzle"])
    d["solution"] = None
    p = Puzzle.from_dict(d)
    m, meta = _W["model"], _W["meta"]
    out = {"model": MODEL.name}
    try:
        out["rookie"] = _slim(engine.rl_solve(m, p, "greedy", 4000, None, 20.0, meta, False, False))
        out["scout"] = _slim(engine.rl_solve(m, p, "search", 4000, None, 20.0, meta, False, True))
        out["grandmaster"] = _slim(engine.rl_solve(m, p, "hybrid", 4000, None, 20.0, meta, False, True))
    except Exception as ex:  # noqa: BLE001
        out["error"] = str(ex)
    t = engine.traced_exact(p, None, 20.0, cap=TRACE_CAP)
    t["solved"] = t["status"] == "solved"
    out["tortoise"] = _slim(t)
    (OUT / "robots" / f"{e['id']}.json").write_text(dumps(out))
    return e["id"], {k: out[k].get("status") for k in ("rookie", "scout", "grandmaster", "tortoise") if k in out}


def stage_robots(ents: list[dict], workers: int, force: bool = False) -> None:
    rdir = OUT / "robots"
    rdir.mkdir(parents=True, exist_ok=True)
    keep = {e["id"] for e in ents}
    for f in [*(OUT / "pools").glob("*.json"), *(OUT / "architect").glob("*.json")]:   # other builders' puzzles
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        keep |= {e["id"] for e in (data if isinstance(data, list) else data.values()) if isinstance(e, dict) and "id" in e}
    for f in rdir.glob("*.json"):          # drop runs of puzzles no longer in the bank
        if f.stem not in keep:
            f.unlink()
    todo = [e for e in ents if force or not (rdir / f"{e['id']}.json").exists()]
    todo.sort(key=lambda e: -len(e["puzzle"]["coords"]))
    t0 = time.perf_counter()
    stats = defaultdict(lambda: defaultdict(int))
    print(f"[robots] {len(todo)} to run ({len(ents) - len(todo)} cached)", flush=True)
    if todo:
        with Pool(workers, initializer=_robot_init) as pool:
            for k, (pid, st) in enumerate(pool.imap_unordered(_robot_job, todo, chunksize=2)):
                for b, s in st.items():
                    stats[b][s] += 1
                if (k + 1) % 250 == 0:
                    print(f"  {k + 1}/{len(todo)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[robots] {time.perf_counter() - t0:.0f}s; " +
          "; ".join(f"{b}: {dict(v)}" for b, v in stats.items()), flush=True)


def sizes() -> None:
    tot = 0
    for sub in ("index.json", "pools", "daily", "robots"):
        p = OUT / sub
        files = [p] if p.is_file() else list(p.rglob("*.json"))
        s = sum(f.stat().st_size for f in files)
        tot += s
        print(f"  {sub:10} {len(files):5} files {s / 1e6:7.2f} MB")
    print(f"  total                  {tot / 1e6:7.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "candidates", "select", "robots"], default="all")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--pool", type=int, default=80, help="puzzles per mode x difficulty pool")
    ap.add_argument("--force-robots", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cands = stage_candidates(a.workers)
    report_candidates(cands)
    if a.stage == "candidates":
        return
    if a.stage in ("all", "select"):
        ents = write_bank(stage_select(cands, a.pool))
    else:
        ents = {}
        for f in (OUT / "pools").glob("*.json"):
            for e in json.loads(f.read_text()):
                ents[e["id"]] = e
        for f in (OUT / "daily").glob("*.json"):
            for rec in json.loads(f.read_text()).values():
                for k in (*DIFFS, "special"):
                    if rec.get(k):
                        ents[rec[k]["id"]] = rec[k]
        ents = list(ents.values())
    if a.stage in ("all", "robots"):
        stage_robots(ents, a.workers, a.force_robots)
    sizes()
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
