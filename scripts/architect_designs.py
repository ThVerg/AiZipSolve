"""Build the Architect's puzzles into the bank (zipsolve/app/static/bank/).

    python scripts/architect_designs.py                     # everything (resumable)
    python scripts/architect_designs.py --stage designs|write|robots|compare
    python scripts/architect_designs.py --workers 12

Stages
  1. designs: run zipsolve.architect.design() for every job in JOBS (expert / insane
     targets on classic, walls and islands boards), cached in
     <cache>/architect.jsonl so a rerun only designs what is missing.  Every design
     is unique (exact solver certificate) and fair (rater guesses == 0).
  2. write: pick the pools and the weekly schedule from the designs and write
       pools/architect-expert.json   ~60 Expert puzzles  (classic 8-9, walls 8-9, islands 6-7)
       pools/architect-insane.json   ~40 Insane puzzles  (classic 9-10, walls 10, islands 8)
       architect/weekly.json         {"2026-W01": entry, ..., "2027-W52": entry}: the weekly
                                     "AI-designed challenge" (hardest designs, kinds rotating
                                     classic -> walls -> islands; Insane when available, else the kind's hardest
                                     Expert >= 70)
     and register them in index.json (additive keys only: modes.architect.{expert,insane}
     and a top-level "architect" block).  Entries have the bank entry shape plus
     "pool" and an "architect" block (design summary).
  3. robots: precompute the four robot runs per puzzle in robots/<id>.json with
     scripts/build_bank.py's own worker functions (same shape as the main bank).
  compare: rate baseline generator puzzles of the same boards (make_puzzle unique=True)
     and print the score distributions next to the Architect's.

NOTE: scripts/build_bank.py now keeps other builders' pools, index keys and robot files
(architect-*, the "more modes"); if an older bank build dropped them, rerun `--stage write`
and `--stage robots` here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "zipsolve" / "app" / "static" / "bank"
CACHE = Path(os.environ.get("ZIP_BANK_CACHE") or Path.home() / ".cache" / "zip-bank")
KIND_MODE = {"grid2d": "classic", "walls": "walls", "islands": "islands"}
WEEK_START, WEEK_END = (2026, 1), (2027, 52)
POOL_SIZES = {"expert": 60, "insane": 40}

# (target, kind, size, time budget s, how many)  -- insane jobs also feed the weekly schedule
JOBS = [
    ("expert", "grid2d", 8, 40, 22), ("expert", "grid2d", 9, 40, 22),
    ("expert", "walls", 8, 40, 10), ("expert", "walls", 9, 40, 10),
    ("expert", "islands", 6, 40, 12), ("expert", "islands", 7, 40, 8),
    ("insane", "grid2d", 10, 90, 70), ("insane", "grid2d", 9, 90, 20),
    ("insane", "walls", 10, 90, 45), ("insane", "islands", 8, 90, 45),
    ("insane", "grid2d", 10, 120, 60), ("insane", "walls", 10, 120, 30),
]


def dumps(o) -> str:
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


def _build_bank():
    spec = importlib.util.spec_from_file_location("build_bank", ROOT / "scripts" / "build_bank.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def size_text(kind: str, size) -> str:
    return f"{size} islands" if kind == "islands" else f"{size}x{size}"


def job_key(j) -> str:
    return f"{j[0]}:{j[1]}:{j[2]}:{j[3]}:{j[4]}"


def all_jobs():
    return [(t, k, s, b, i) for t, k, s, b, cnt in JOBS for i in range(cnt)]


# ---------------------------------------------------------------------------
# stage 1: designs
# ---------------------------------------------------------------------------
def _design_job(j):
    target, kind, size, budget, i = j
    from zipsolve.architect import design
    seed = int.from_bytes(hashlib.sha256(f"architect:{target}:{kind}:{size}:{i}".encode()).digest()[:4], "big")
    t0 = time.perf_counter()
    try:
        r = design(kind, size, target, time_budget=budget, seed=seed)
    except ValueError as e:
        return {"key": job_key(j), "fail": str(e)}
    if not r.ok:
        return {"key": job_key(j), "fail": "no certified design"}
    b = _build_bank()
    d = b.compact_puzzle(r.puzzle)
    f = r.rating["features"]
    h = hashlib.sha1(dumps([d["coords"], d["edges"], d["checkpoints"]]).encode()).hexdigest()
    return {"key": job_key(j), "target": target, "kind": kind, "size": size, "seed": seed,
            "id": f"a{h[:9]}", "score": r.rating["score"], "label": r.rating["label"],
            "boring": r.rating["boring"], "spread": f["spread"], "guesses": f["guesses"],
            "clues": f["checkpoints"], "max_lookahead": f["max_lookahead"], "counts": f["counts"],
            "evaluations": r.stats["evaluations"], "seconds": round(time.perf_counter() - t0, 1),
            "puzzle": d}


def stage_designs(workers: int) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "architect.jsonl"
    have = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                have[c["key"]] = c
    jobs = [j for j in all_jobs() if job_key(j) not in have]
    jobs.sort(key=lambda j: -j[3])          # long budgets first
    t0 = time.perf_counter()
    if jobs:
        print(f"[designs] {len(jobs)} to design ({len(have)} cached), {workers} workers", flush=True)
        with Pool(workers, maxtasksperchild=8) as pool, f.open("a") as out:
            for k, c in enumerate(pool.imap_unordered(_design_job, jobs)):
                have[c["key"]] = c
                out.write(dumps(c) + "\n")
                out.flush()
                if (k + 1) % 10 == 0:
                    print(f"  {k + 1}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[designs] {len(have)} total, {time.perf_counter() - t0:.0f}s", flush=True)
    keys = {job_key(j) for j in all_jobs()}
    return [c for c in have.values() if not c.get("fail") and c["key"] in keys]


def report(designs: list[dict]) -> None:
    import numpy as np
    by = defaultdict(list)
    for c in designs:
        by[(c["target"], c["kind"], c["size"])].append(c)
    print(f"{'target':7} {'kind':8} {'size':>4} {'n':>4} {'med':>5} {'p10':>5} {'p90':>5}  labels  sec")
    for key, cs in sorted(by.items()):
        sc = [c["score"] for c in cs]
        labs = Counter(c["label"][:2] for c in cs)
        print(f"{key[0]:7} {key[1]:8} {key[2]:>4} {len(cs):>4} {np.median(sc):5.1f} {np.percentile(sc, 10):5.1f} "
              f"{np.percentile(sc, 90):5.1f}  " + " ".join(f"{k}{v}" for k, v in sorted(labs.items()))
              + f"  {np.mean([c['seconds'] for c in cs]):.0f}")


# ---------------------------------------------------------------------------
# stage 2: write pools, weekly schedule, index
# ---------------------------------------------------------------------------
def entry(c: dict, diff: str, pool: str) -> dict:
    return {"id": c["id"], "mode": KIND_MODE.get(c["kind"], c["kind"]), "diff": diff, "label": c["label"],
            "score": c["score"], "size": size_text(c["kind"], c["size"]), "puzzle": c["puzzle"], "pool": pool,
            "architect": {"target": c["target"], "seed": c["seed"], "clues": c["clues"], "guesses": c["guesses"],
                          "max_lookahead": c["max_lookahead"], "counts": c["counts"], "spread": c["spread"],
                          "evaluations": c["evaluations"], "seconds": c["seconds"]}}


def iso_weeks():
    y, w = WEEK_START
    while (y, w) <= WEEK_END:
        yield f"{y:04d}-W{w:02d}"
        last = dt.date(y, 12, 28).isocalendar()[1]
        w += 1
        if w > last:
            y, w = y + 1, 1


def good(c) -> bool:
    return c["guesses"] == 0 and not c["boring"] and c["spread"] >= 0.8


def select(designs: list[dict]) -> dict:
    rnd = random.Random(20260101)
    ds = sorted({c["id"]: c for c in designs if good(c)}.values(), key=lambda c: c["id"])
    rnd.shuffle(ds)
    used: set[str] = set()

    def take(pred, k, key=None, mix=True):
        lst = [c for c in ds if c["id"] not in used and pred(c)]
        if key:
            lst.sort(key=key)
        if mix:   # round-robin over kinds so every pool mixes classic / walls / islands
            groups = defaultdict(list)
            for c in lst:
                groups[c["kind"]].append(c)
            lst = []
            its = [iter(v) for _, v in sorted(groups.items())]
            while its:
                for it in list(its):
                    x = next(it, None)
                    if x is None:
                        its.remove(it)
                    else:
                        lst.append(x)
        out = lst[:k]
        used.update(c["id"] for c in out)
        return out

    insane = take(lambda c: c["label"] == "Insane", POOL_SIZES["insane"], key=lambda c: -c["score"])
    expert = take(lambda c: c["label"] == "Expert" and c["target"] == "expert", POOL_SIZES["expert"])
    if len(expert) < POOL_SIZES["expert"]:
        expert += take(lambda c: c["label"] == "Expert", POOL_SIZES["expert"] - len(expert))
    weeks = list(iso_weeks())
    rot = ("grid2d", "walls", "islands")
    weekly = {}
    for k, wk in enumerate(weeks):
        want = rot[k % 3]
        pick = take(lambda c, w=want: c["kind"] == w and c["label"] == "Insane", 1, mix=False) or \
            take(lambda c, w=want: c["kind"] == w and c["score"] >= 70, 1, key=lambda c: -c["score"],
                 mix=False) or \
            take(lambda c: c["label"] == "Insane", 1, mix=False) or \
            take(lambda c: c["label"] == "Expert", 1, key=lambda c: -c["score"], mix=False)
        weekly[wk] = pick[0] if pick else None
    print(f"[write] expert {len(expert)}, insane {len(insane)}, weekly {sum(v is not None for v in weekly.values())}"
          f"/{len(weeks)} (" + ", ".join(f"{k}:{v}" for k, v in Counter(
              c["label"] for c in weekly.values() if c).items()) + ")", flush=True)
    return {"expert": expert, "insane": insane, "weekly": weekly}


def write(sel: dict) -> list[dict]:
    (OUT / "pools").mkdir(parents=True, exist_ok=True)
    (OUT / "architect").mkdir(parents=True, exist_ok=True)
    ents = []
    info = {}
    for diff in ("expert", "insane"):
        lst = sorted((entry(c, diff, f"architect-{diff}") for c in sel[diff]), key=lambda e: e["id"])
        fn = f"pools/architect-{diff}.json"
        (OUT / fn).write_text(dumps(lst))
        info[diff] = {"file": fn, "count": len(lst)}
        ents += lst
    weekly = {}
    for k, (wk, c) in enumerate(sel["weekly"].items()):
        if c is None:
            weekly[wk] = None
            continue
        e = entry(c, "insane" if c["label"] == "Insane" else "expert", "architect-weekly")
        e["week"], e["number"] = wk, k + 1
        weekly[wk] = e
        ents.append(e)
    (OUT / "architect" / "weekly.json").write_text(dumps(weekly))
    # register in index.json (re-read right before editing; additive keys only)
    idx_f = OUT / "index.json"
    idx = json.loads(idx_f.read_text()) if idx_f.exists() else {"version": 1, "modes": {}}
    idx.setdefault("modes", {})["architect"] = info
    wk = list(weekly)
    idx["architect"] = {"pools": info, "weekly": {"file": "architect/weekly.json", "count": len(wk),
                                                  "first": wk[0], "last": wk[-1], "key": "ISO week YYYY-Www"},
                        "robots": "robots/{id}.json", "puzzles": len(ents),
                        "generated": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}
    idx_f.write_text(json.dumps(idx, indent=1))
    print(f"[write] {len(ents)} architect puzzles -> pools/architect-*.json, architect/weekly.json, index.json",
          flush=True)
    return ents


def load_written() -> list[dict]:
    ents = {}
    for diff in ("expert", "insane"):
        f = OUT / "pools" / f"architect-{diff}.json"
        if f.exists():
            for e in json.loads(f.read_text()):
                ents[e["id"]] = e
    f = OUT / "architect" / "weekly.json"
    if f.exists():
        for e in json.loads(f.read_text()).values():
            if e:
                ents[e["id"]] = e
    return list(ents.values())


# ---------------------------------------------------------------------------
# stage 3: robots (build_bank's worker functions)
# ---------------------------------------------------------------------------
_B = {}


def _robot_init():
    _B["b"] = _build_bank()
    _B["b"]._robot_init()


def _robot_job(e):
    return _B["b"]._robot_job(e)


def stage_robots(ents: list[dict], workers: int, force: bool = False) -> None:
    rdir = OUT / "robots"
    rdir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in ents if force or not (rdir / f"{e['id']}.json").exists()]
    t0 = time.perf_counter()
    stats = defaultdict(Counter)
    print(f"[robots] {len(todo)} to run ({len(ents) - len(todo)} cached)", flush=True)
    if todo:
        with Pool(workers, initializer=_robot_init) as pool:
            for k, (pid, st) in enumerate(pool.imap_unordered(_robot_job, todo)):
                for b, s in st.items():
                    stats[b][s] += 1
                if (k + 1) % 25 == 0:
                    print(f"  {k + 1}/{len(todo)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[robots] {time.perf_counter() - t0:.0f}s; " + "; ".join(f"{b}: {dict(v)}" for b, v in stats.items()),
          flush=True)


# ---------------------------------------------------------------------------
# compare: baseline generator on the same boards
# ---------------------------------------------------------------------------
def _base_job(a):
    kind, size, i = a
    from zipsolve.difficulty import rate
    from zipsolve.generator import make_puzzle
    try:
        p = make_puzzle(kind, size, rng=50_000 + i, unique=True, time_limit=60)
    except ValueError:
        return None
    r = rate(p)
    return kind, size, r["score"], r["label"], r["features"]["guesses"]


def stage_compare(designs: list[dict], workers: int, n: int = 40) -> None:
    import numpy as np
    boards = sorted({(c["kind"], c["size"]) for c in designs})
    jobs = [(k, s, i) for k, s in boards for i in range(n)]
    with Pool(workers) as pool:
        base = [r for r in pool.imap_unordered(_base_job, jobs) if r]
    print(f"{'board':14} {'who':9} {'n':>4} {'p10':>5} {'med':>5} {'p90':>5} {'>=Exp':>6} {'>=Ins':>6} {'fair':>5}"
          f" {'fair&>=Exp':>10}")
    for k, s in boards:
        rows = {"baseline": [(r[2], r[4]) for r in base if r[0] == k and r[1] == s],
                "architect": [(c["score"], c["guesses"]) for c in designs if c["kind"] == k and c["size"] == s]}
        for who, xs in rows.items():
            if not xs:
                continue
            sc = np.array([x[0] for x in xs])
            print(f"{k + ' ' + str(s):14} {who:9} {len(sc):>4} {np.percentile(sc, 10):5.1f} {np.median(sc):5.1f} "
                  f"{np.percentile(sc, 90):5.1f} {np.mean(sc >= 62):6.0%} {np.mean(sc >= 80):6.0%} "
                  f"{np.mean([x[1] == 0 for x in xs]):5.0%} {np.mean([x[1] == 0 and x[0] >= 62 for x in xs]):10.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "designs", "write", "robots", "compare"], default="all")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--force-robots", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    designs = stage_designs(a.workers)
    report(designs)
    if a.stage == "compare":
        stage_compare(designs, a.workers)
        return
    if a.stage == "designs":
        return
    if a.stage in ("all", "write"):
        ents = write(select(designs))
    else:
        ents = load_written()
    if a.stage in ("all", "robots"):
        stage_robots(ents, a.workers, a.force_robots)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
