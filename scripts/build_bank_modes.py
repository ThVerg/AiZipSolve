"""Add the "more modes" to the curated puzzle bank (additive: never touches the classic
pools, the daily schedule or the Architect files).

    python scripts/build_bank_modes.py                  # everything (resumable)
    python scripts/build_bank_modes.py --stage candidates|write|robots|lab|compact
    python scripts/build_bank_modes.py --workers 8

Pools (zipsolve/app/static/bank/pools/<mode>-<diff>.json, same entry shape as the main bank):
    portals / torus / hex / tri / oneway / overpass / keys / cubesurf : medium + hard
    coop : medium  (entry puzzle = CoopPuzzle.to_dict(include_solution=True): the two unique
                    paths live under meta.coop.solution.paths; "solution" stays null)
The game's mode buttons ask for "<mode>-medium" (play/modes.js); every puzzle has exactly one
solution (generator unique=True, re-verified by count_solutions(p, 2) == (1, "complete")).

Stages
  1. candidates: generate + verify + rate (cached in <cache>/modes_candidates.jsonl).
  2. write: fill the pools and merge them into index.json (other modes / keys are kept).
  3. robots: robots/<id>.json for every new-mode puzzle (not co-op: no robot plays it):
     the four robots of the main bank (rookie, scout, grandmaster, tortoise) plus the
     strategy robots of zipsolve.robots (detective, mcts, evolver, gambler, sat), compacted
     (see compact_robot) so the whole bank stays small.
  4. lab: add the strategy robots to the existing robot files of the pools the AI show
     picks from most (LAB_POOLS: classic-medium/hard, walls-medium, islands-medium,
     architect-expert), so the new robots can be watched online too. Existing keys are kept.
     (Not every pool: the bank should stay around 40 MB.)

NOTE: scripts/build_bank.py rewrites index.json (it keeps the modes it does not own, see
write_bank) - rerun `--stage write` here after it if a mode went missing.
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
sys.path.insert(0, str(ROOT / "scripts"))

import build_bank as B  # noqa: E402

OUT = B.OUT
CACHE = B.CACHE
STRATS = ("detective", "mcts", "evolver", "gambler", "sat")
ROBOT_TL = 8.0
MODE_TAG = {"portals": "p", "torus": "t", "hex": "h", "tri": "r", "oneway": "o", "overpass": "v",
            "keys": "k", "cubesurf": "s", "coop": "x"}
# source -> (mode, kind, size, extra kwargs, initial checkpoint counts, how many)
SOURCES = {
    "p6": ("portals", "portals", 6, {}, [None, None, 3, 4], 110),
    "p8": ("portals", "portals", 8, {}, [None, 3, 4, 5], 90),
    "t5": ("torus", "torus", 5, {}, [None, None, 3, 4], 110),
    "t6": ("torus", "torus", 6, {}, [None, 3, 4, 5], 90),
    "h4": ("hex", "hex", 4, {}, [None, None, 3, 4], 110),
    "h5": ("hex", "hex", 5, {}, [None, 3, 4, 5], 90),
    "r3": ("tri", "tri", 3, {}, [None, None, 3, 4], 110),
    "r4": ("tri", "tri", 4, {}, [None, 3, 4, 5], 90),
    "o6": ("oneway", "oneway", 6, {}, [None, None, 3, 4], 110),
    "o8": ("oneway", "oneway", 8, {}, [None, 3, 4, 5], 90),
    "v6": ("overpass", "overpass", 6, {}, [None, None, 3, 4], 110),
    "v8": ("overpass", "overpass", 8, {}, [None, 3, 4, 5], 90),
    "k6": ("keys", "keys", 6, {}, [None, None, 3, 4], 110),
    "k8": ("keys", "keys", 8, {}, [None, 3, 4, 5], 90),
    "s3": ("cubesurf", "cubesurf", 3, {}, [None, None, 4, 5], 110),
    "s4": ("cubesurf", "cubesurf", 4, {}, [None, 5, 6], 70),
    "x6": ("coop", "coop", 6, {}, [None, None, 5, 6], 110),
}
# (mode, diff) -> (sources, preferred labels, pool size)
BUCKETS = {}
for _m, (_med, _hard) in {"portals": ("p6", "p8"), "torus": ("t5", "t6"), "hex": ("h4", "h5"),
                          "tri": ("r3", "r4"), "oneway": ("o6", "o8"), "overpass": ("v6", "v8"),
                          "keys": ("k6", "k8"), "cubesurf": ("s3", "s4")}.items():
    BUCKETS[(_m, "medium")] = ([_med], {"Medium", "Hard"}, 50)
    BUCKETS[(_m, "hard")] = ([_hard], {"Hard", "Expert", "Insane"}, 40)
BUCKETS[("coop", "medium")] = (["x6"], {"Medium", "Hard"}, 40)
LAB_POOLS = ("classic-medium", "classic-hard", "walls-medium", "islands-medium", "architect-expert")

dumps = B.dumps


def size_text(kind: str, size) -> str:
    if kind == "hex":
        return f"hexagon {size}"
    if kind == "tri":
        return f"{6 * size * size} triangles"
    if kind == "cubesurf":
        return f"{size}x{size}x{size} cube"
    return f"{size}x{size}"


# ---------------------------------------------------------------------------
# stage 1: candidates
# ---------------------------------------------------------------------------
def _cand_job(job):
    src, i = job
    mode, kind, size, kw, ncps, _ = SOURCES[src]
    seed = int.from_bytes(hashlib.sha256(f"zip-bank-modes:{src}:{i}".encode()).digest()[:6], "big")
    ncp = ncps[i % len(ncps)]
    t0 = time.perf_counter()
    try:
        if kind == "coop":
            from zipsolve import coop
            p = coop.make_coop(size, ncp, rng=seed, unique=True, time_limit=60, **kw)
            if coop.count_solutions(p, 2, time_limit=60)[0] != 1:
                return {"src": src, "i": i, "fail": True}
            r = coop.rate(p, time_limit=3.0)
            d = p.to_dict(include_solution=True)
            d["coords"] = [[int(x) for x in c] for c in d["coords"]]
            feats = {}
        else:
            from zipsolve.difficulty import rate
            from zipsolve.generator import make_puzzle
            from zipsolve.solver import count_solutions
            p = make_puzzle(kind, size, num_checkpoints=ncp, rng=seed, unique=True, time_limit=60, **kw)
            if count_solutions(p, 2, time_limit=90) != (1, "complete"):
                return {"src": src, "i": i, "fail": True}
            r = rate(p)
            d = B.compact_puzzle(p)
            feats = {k: r["features"].get(k) for k in ("nodes", "checkpoints", "max_lookahead", "guesses", "spread")}
    except Exception as e:  # noqa: BLE001  (generation timeout / no board): skip this seed
        return {"src": src, "i": i, "fail": True, "error": str(e)[:200]}
    h = hashlib.sha1(dumps([d["coords"], d["edges"], d["checkpoints"], d.get("meta", {}).get("coop")]).encode())
    return {"src": src, "i": i, "mode": mode, "id": f"{MODE_TAG[mode]}{h.hexdigest()[:9]}",
            "size": size_text(kind, size), "score": r["score"], "label": r["label"],
            "boring": bool(r.get("boring", False)), "features": feats,
            "seconds": round(time.perf_counter() - t0, 2), "puzzle": d}


def stage_candidates(workers: int) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "modes_candidates.jsonl"
    have = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                have[(c["src"], c["i"])] = c
    jobs = [(src, i) for src, s in SOURCES.items() for i in range(s[5]) if (src, i) not in have]
    jobs.sort(key=lambda j: j[0] not in ("s4", "h5", "r4"))
    t0 = time.perf_counter()
    if jobs:
        print(f"[candidates] generating {len(jobs)} ({len(have)} cached) with {workers} workers", flush=True)
        with Pool(workers, maxtasksperchild=50) as pool, f.open("a") as out:
            for k, c in enumerate(pool.imap_unordered(_cand_job, jobs, chunksize=2)):
                have[(c["src"], c["i"])] = c
                out.write(dumps(c) + "\n")
                out.flush()
                if (k + 1) % 200 == 0:
                    print(f"  {k + 1}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    good = [c for c in have.values() if not c.get("fail")]
    by = defaultdict(lambda: defaultdict(int))
    for c in good:
        by[c["src"]][c["label"]] += 1
    for src in SOURCES:
        print(f"  {src:4} " + " ".join(f"{k}:{v}" for k, v in sorted(by[src].items())), flush=True)
    print(f"[candidates] {len(good)} ok of {len(have)}, {time.perf_counter() - t0:.0f}s", flush=True)
    return good


# ---------------------------------------------------------------------------
# stage 2: write pools + merge index.json
# ---------------------------------------------------------------------------
def stage_write(cands: list[dict]) -> list[dict]:
    rnd = random.Random(20260928)
    cands = sorted({c["id"]: c for c in cands}.values(), key=lambda c: c["id"])
    used: set[str] = set()
    pools = {}
    for (mode, diff), (srcs, labels, k) in BUCKETS.items():
        lst = [c for c in cands if c["src"] in srcs and c["id"] not in used]
        rnd.shuffle(lst)
        # preferred labels and not boring first, then the rest (the label shown is always the true rating)
        # (outside the preferred labels the nearest scores come first: medium = lowest, hard = highest)
        lst.sort(key=lambda c: (c["label"] not in labels, c["boring"],
                                0 if c["label"] in labels else (c["score"] if diff == "medium" else -c["score"])))
        pick = lst[:k]
        used.update(c["id"] for c in pick)
        if len(pick) < k:
            print(f"  ! {mode}-{diff}: only {len(pick)} of {k}", flush=True)
        pools[(mode, diff)] = sorted((B.entry(c, diff) for c in pick), key=lambda e: e["id"])
    (OUT / "pools").mkdir(parents=True, exist_ok=True)
    idx = json.loads((OUT / "index.json").read_text())
    ents = []
    for (mode, diff), lst in pools.items():
        fn = f"pools/{mode}-{diff}.json"
        (OUT / fn).write_text(dumps(lst))
        idx.setdefault("modes", {}).setdefault(mode, {})[diff] = {"file": fn, "count": len(lst)}
        ents += lst
    idx["more_modes"] = {"modes": sorted({m for m, _ in pools}), "puzzles": len(ents),
                         "robots": ["rookie", "scout", "grandmaster", "tortoise", *STRATS],
                         "generated": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}
    (OUT / "index.json").write_text(json.dumps(idx, indent=1))
    labs = defaultdict(lambda: defaultdict(int))
    for (m, d), lst in pools.items():
        for e in lst:
            labs[f"{m}-{d}"][e["label"]] += 1
    print("[write] " + "; ".join(f"{k}: {len(pools[tuple(k.split('-'))])} " + ",".join(f"{a[:2]}{b}" for a, b in sorted(v.items()))
                                 for k, v in labs.items()), flush=True)
    return ents


def load_pool_entries(names) -> list[dict]:
    idx = json.loads((OUT / "index.json").read_text())
    out = []
    for name in names:
        mode, _, diff = name.partition("-")
        d = (idx.get("modes", {}).get(mode) or {}).get(diff)
        if d:
            out += json.loads((OUT / d["file"]).read_text())
    return out


# ---------------------------------------------------------------------------
# stage 3/4: robots
# ---------------------------------------------------------------------------
def _thin(items: list, k: int) -> list:
    """k items spread evenly over the list (always the first and the last)."""
    if len(items) <= k:
        return items
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


# notes per kind a replay keeps at most (captions flash by: hundreds of them only cost bytes). The
# Detective's notes that explain the unique solution's moves are always kept (the online teaching hints).
NOTE_CAP = {"bet": 12, "backtrack": 12, "guess": 15, "deduce": 20, "forced": 0, "think": 12}


def thin_trace(tr: list, sol: list | None = None) -> list:
    """Evolver: keep ~12 generation (path + its note) pairs; Sage: ~12 tree snapshots (top 3 children,
    2 kids each); notes capped per kind (NOTE_CAP). Pushes / pops are never dropped, so the replay is
    unchanged. Idempotent (also used by --stage compact on stored runs)."""
    gens = [i for i, e in enumerate(tr) if isinstance(e, dict) and e.get("t") == "path"]
    trees = [i for i, e in enumerate(tr) if isinstance(e, dict) and e.get("t") == "tree"]
    protect = set()
    if sol:
        stack, seen = [], set()
        for i, e in enumerate(tr):
            if e == "R":
                stack = []
            elif isinstance(e, int):
                if e >= 0:
                    stack.append(e)
                else:
                    del stack[len(stack) + e:]
            elif isinstance(e, dict) and e.get("t") == "path":
                stack = list(e.get("p") or [])
            elif isinstance(e, dict) and e.get("t") == "note" and e.get("move") is not None:
                k = len(stack)
                if k not in seen and k < len(sol) and sol[k] == e["move"] and stack == sol[:k] \
                        and e.get("kind") in ("deduce", "forced"):
                    seen.add(k)
                    protect.add(i)
    drop = set()
    if len(gens) > 12:
        keep = set(_thin(gens[:-1], 11)) | {gens[-1]}
        for i in gens:
            if i not in keep:
                drop.add(i)
                nx = i + 1
                if nx < len(tr) and isinstance(tr[nx], dict) and tr[nx].get("kind") == "generation":
                    drop.add(nx)
    if len(trees) > 12:
        drop |= set(trees) - set(_thin(trees, 12))
    for kind, cap in NOTE_CAP.items():
        idx = [i for i, e in enumerate(tr) if isinstance(e, dict) and e.get("t") == "note" and e.get("kind") == kind
               and i not in protect]
        room = max(0, cap - sum(1 for i in protect if tr[i].get("kind") == kind))
        if len(idx) > room:
            drop |= set(idx) - (set(_thin(idx, room)) if room >= 2 else set(idx[:room]))
    out = []
    for i, e in enumerate(tr):
        if i in drop:
            continue
        if isinstance(e, dict) and e.get("t") == "tree":
            e = {**e, "nodes": [{**n, "kids": (n.get("kids") or [])[:2]} if n.get("kids") else n
                                for n in (e.get("nodes") or [])[:3]]}
        out.append(e)
    return out


def stage_compact() -> None:
    sols = {}
    for f in (OUT / "pools").glob("*.json"):
        for e in json.loads(f.read_text()):
            sols[e["id"]] = e["puzzle"].get("solution")
    n = before = after = 0
    for f in (OUT / "robots").glob("*.json"):
        d = json.loads(f.read_text())
        sol = sols.get(f.stem)
        changed = False
        for rb in STRATS:
            r = d.get(rb)
            if isinstance(r, dict) and r.get("trace"):
                t = thin_trace(r["trace"], sol)
                if len(t) != len(r["trace"]) or t != r["trace"]:
                    r["trace"] = t
                    changed = True
        if changed:
            before += f.stat().st_size
            f.write_text(dumps(d))
            after += f.stat().st_size
            n += 1
    print(f"[compact] {n} files: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB", flush=True)


def compact_robot(r: dict, sol: list | None = None) -> dict:
    """A zipsolve.robots run -> compact bank form. steps become parallel lists ("sp": p, "sh": how;
    the node is path[i + 1]); Evolver generations and Sage trees are thinned; the trace is capped."""
    out = {k: r[k] for k in ("status", "solved", "path", "start_len", "nodes_expanded", "stuck_at", "backtracks",
                             "reason") if k in r and r[k] is not None}
    out["seconds"] = round(float(r.get("seconds") or 0), 2)
    steps = r.get("steps") or []
    if steps:
        ps = [None if s.get("p") is None else round(float(s["p"]), 2) for s in steps]
        if any(p is not None for p in ps):
            out["sp"] = ps
        hows = [s.get("how") for s in steps]
        if any(hows):
            out["sh"] = hows
    st = dict(r.get("stats") or {})
    if isinstance(st.get("fitness_history"), list):
        st["fitness_history"] = _thin(st["fitness_history"], 40)
    out["stats"] = st
    tr = r.get("trace")
    trunc = bool(r.get("trace_truncated"))
    if tr:
        tr = thin_trace(tr, sol)
        if len(tr) > B.TRACE_CAP:
            tr = tr[:B.TRACE_CAP]
            trunc = True
        # the replayed line must end on "path"
        if trunc or r.get("solved") is not None:
            last = tr[-1] if tr else None
            if not (isinstance(last, dict) and last.get("t") == "path" and last.get("p") == r.get("path")):
                stack = []
                for e in tr:
                    if e == "R":
                        stack = []
                    elif isinstance(e, int):
                        if e >= 0:
                            stack.append(e)
                        else:
                            del stack[len(stack) + e:]
                    elif isinstance(e, dict) and e.get("t") == "path":
                        stack = list(e.get("p") or [])
                if stack != r.get("path"):
                    tr.append({"t": "path", "p": r.get("path")})
        out["trace"] = tr
        out["trace_truncated"] = trunc
    return out


_W = {}


def _init(classic: bool):
    if classic:
        B._robot_init()
        _W["classic"] = True


def _robot_job(job):
    e, classic, force = job
    from zipsolve.puzzle import Puzzle
    from zipsolve.robots import run_robot
    f = OUT / "robots" / f"{e['id']}.json"
    try:
        out = json.loads(f.read_text()) if f.exists() else {}
    except ValueError:
        out = {}
    d = dict(e["puzzle"])
    d["solution"] = None
    p = Puzzle.from_dict(d)
    if classic and (force or "tortoise" not in out):
        from zipsolve.app import engine
        m, meta = B._W["model"], B._W["meta"]
        out["model"] = B.MODEL.name
        try:
            out["rookie"] = B._slim(engine.rl_solve(m, p, "greedy", 4000, None, 20.0, meta, False, False))
            out["scout"] = B._slim(engine.rl_solve(m, p, "search", 4000, None, 20.0, meta, False, True))
            out["grandmaster"] = B._slim(engine.rl_solve(m, p, "hybrid", 4000, None, 20.0, meta, False, True))
        except Exception as ex:  # noqa: BLE001
            out["error"] = str(ex)
        t = engine.traced_exact(p, None, 20.0, cap=B.TRACE_CAP)
        t["solved"] = t["status"] == "solved"
        out["tortoise"] = B._slim(t)
    for rb in STRATS:
        if rb in out and not force:
            continue
        seed = int(e["id"][1:], 16) % 1000
        try:
            r = run_robot(rb, p, ROBOT_TL, True, seed)
        except Exception as ex:  # noqa: BLE001
            continue
        if r.get("status") in ("unavailable", "unsupported"):
            continue
        out[rb] = compact_robot(r, e["puzzle"].get("solution"))
    f.write_text(dumps(out))
    return e["id"], {k: out[k].get("status") for k in ("rookie", "scout", "grandmaster", "tortoise", *STRATS)
                     if isinstance(out.get(k), dict)}, f.stat().st_size


def stage_robots(ents: list[dict], workers: int, classic: bool, force: bool = False, label: str = "robots"):
    (OUT / "robots").mkdir(parents=True, exist_ok=True)
    todo = []
    for e in ents:
        if e["puzzle"].get("kind") == "coop":
            continue
        f = OUT / "robots" / f"{e['id']}.json"
        have = json.loads(f.read_text()) if f.exists() else {}
        if force or any(k not in have for k in STRATS) or (classic and "tortoise" not in have):
            todo.append((e, classic, force))
    todo.sort(key=lambda j: -len(j[0]["puzzle"]["coords"]))
    print(f"[{label}] {len(todo)} to run ({len(ents) - len(todo)} done)", flush=True)
    t0 = time.perf_counter()
    stats = defaultdict(lambda: defaultdict(int))
    tot = 0
    if todo:
        with Pool(workers, initializer=_init, initargs=(classic,), maxtasksperchild=40) as pool:
            for k, (pid, st, sz) in enumerate(pool.imap_unordered(_robot_job, todo, chunksize=1)):
                tot += sz
                for b, s in st.items():
                    stats[b][s] += 1
                if (k + 1) % 100 == 0:
                    print(f"  {k + 1}/{len(todo)}  {time.perf_counter() - t0:.0f}s  avg {tot / (k + 1) / 1e3:.1f} KB",
                          flush=True)
    print(f"[{label}] {time.perf_counter() - t0:.0f}s; " + "; ".join(f"{b}: {dict(v)}" for b, v in stats.items()),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "candidates", "write", "robots", "lab", "compact"], default="all")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--force-robots", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.stage == "compact":
        stage_compact()
        B.sizes()
        return
    cands = stage_candidates(a.workers) if a.stage in ("all", "candidates", "write") else []
    if a.stage == "candidates":
        return
    if a.stage in ("all", "write"):
        ents = stage_write(cands)
    else:
        ents = load_pool_entries([f"{m}-{d}" for m, d in BUCKETS])
    if a.stage in ("all", "robots"):
        stage_robots(ents, a.workers, True, a.force_robots)
    if a.stage in ("all", "lab"):
        stage_robots(load_pool_entries(LAB_POOLS), a.workers, False, a.force_robots, "lab")
    B.sizes()
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
