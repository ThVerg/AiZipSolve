"""Frozen Zip benchmark: deterministic validation / test sets + evaluation report.

    python -m zipsolve.rl.benchmark build                   # benchmarks/val.json + test.json
    python -m zipsolve.rl.benchmark run --ckpt checkpoints/curric_4to6_final.pt --set val --threads 8
    python -m zipsolve.rl.benchmark run --ckpt A=runs/a_s0.pt --ckpt A=runs/a_s1.pt --ckpt B=b.pt

Sets
----
Families (``kind:size``): grid2d 5-8, walls 6-7, mask 7-8, islands (archipelago)
3-5, islands_chain 3, grid3d 3-4, grid4d 2-3. Puzzle i of a family uses
checkpoint density ``DENSITIES[i % 3]`` (sparse 0.6x / default 1x / dense 1.6x
the generator's default count) and is generated from its own random stream
``(DOMAIN_VAL|DOMAIN_TEST, family_id, i)`` (see zipsolve.rl.sampling): training
samplers use ``DOMAIN_TRAIN`` and never draw these streams, and training can
additionally exclude these puzzles by content hash. val = 15 per family
(240 puzzles, ~1-2 min on 8 threads), test = 60 per family. Files are plain
JSON (``Puzzle.to_dict`` per record) and are regenerated bit-identically by
``build``; the file header stores a fingerprint of the records.

Methods (per puzzle)
--------------------
Per checkpoint:
  greedy      most probable legal move, no backtracking
  sampled@K   K independent rollouts at temperature T (solved if any is)
  search      exact solver + policy move ordering (evaluate.hybrid_solve, or
              solver.solve(move_order=policy_move_order)) under --budget seconds
Baselines (once):
  solver      exact solver, default heuristic (+ restarts), --budget seconds
  heur_greedy Warnsdorff policy (next checkpoint if legal, else fewest free
              neighbours), no backtracking
  heur_search the same solver search as ``search`` with the Warnsdorff order
Reported per family / density / overall: solve rate, median and p90 latency,
mean expansions (moves tried; solver nodes for searches), mean neural
inference calls (forward passes). Several checkpoints can share a label
(``LABEL=path``, e.g. different training seeds): the table then shows the mean
+- std over them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..puzzle import Puzzle
from .sampling import (DOMAIN_TEST, DOMAIN_VAL, family_id, generate_puzzle, parse_spec, puzzle_key,
                       puzzle_rng)

FAMILIES = ["grid2d:5", "grid2d:6", "grid2d:7", "grid2d:8", "walls:6", "walls:7", "mask:7", "mask:8",
            "islands:3", "islands:4", "islands:5", "islands_chain:3", "grid3d:3", "grid3d:4",
            "grid4d:2", "grid4d:3"]
DENSITIES = [("sparse", 0.6), ("default", 1.0), ("dense", 1.6)]
SPLITS = {"val": (DOMAIN_VAL, 15), "test": (DOMAIN_TEST, 60)}
DEFAULT_DIR = "benchmarks"
FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Building / loading sets
# --------------------------------------------------------------------------- #
def make_record(family: str, split: str, i: int) -> dict:
    """Puzzle i of `family` in `split` (deterministic)."""
    domain = SPLITS[split][0] if split in SPLITS else int(split)
    (kind, size, ncp), = parse_spec(family)
    dname, factor = DENSITIES[i % len(DENSITIES)]
    key = [domain, family_id(family), i]
    p = generate_puzzle(kind, size, puzzle_rng(*key), num_checkpoints=ncp, density=factor)
    return {"id": f"{split}/{family}/{i}", "family": family, "kind": kind, "density": dname,
            "density_factor": factor, "seed_key": key, "n": p.num_nodes,
            "num_checkpoints": len(p.checkpoints), "puzzle": p.to_dict()}


def _record_task(args):
    return make_record(*args)


def fingerprint(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r["puzzle"], sort_keys=True).encode())
    return h.hexdigest()[:16]


def build_set(split: str, families=None, per_family: int | None = None, workers: int = 1) -> dict:
    families = list(families or FAMILIES)
    per_family = per_family or SPLITS[split][1]
    tasks = [(f, split, i) for f in families for i in range(per_family)]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            records = pool.map(_record_task, tasks, chunksize=4)
    else:
        records = [_record_task(t) for t in tasks]
    return {"format": FORMAT_VERSION, "split": split, "seed_domain": SPLITS[split][0],
            "families": families, "densities": DENSITIES, "per_family": per_family,
            "fingerprint": fingerprint(records), "records": records}


def save_set(data: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def set_path(split: str, root: str = DEFAULT_DIR) -> Path:
    return Path(root) / f"{split}.json"


def load_set(path_or_split, root: str = DEFAULT_DIR, build_missing: bool = True) -> dict:
    """Load a benchmark file (adds ``rec["_puzzle"]``); builds it if missing."""
    p = Path(path_or_split)
    if not p.suffix:
        p = set_path(str(path_or_split), root)
    if not p.exists():
        if not build_missing or p.stem not in SPLITS:
            raise FileNotFoundError(p)
        save_set(build_set(p.stem, workers=min(8, os.cpu_count() or 1)), p)
    data = json.loads(p.read_text())
    for r in data["records"]:
        r["_puzzle"] = Puzzle.from_dict(r["puzzle"])
    return data


def benchmark_keys(root: str = DEFAULT_DIR) -> set[str]:
    """Content keys of every saved benchmark puzzle (to exclude them from training)."""
    keys = set()
    for split in SPLITS:
        p = set_path(split, root)
        if p.exists():
            for r in json.loads(p.read_text())["records"]:
                keys.add(puzzle_key(Puzzle.from_dict(r["puzzle"])))
    return keys


def subset(records: list[dict], per_family: int | None = None, families=None) -> list[dict]:
    out, cnt = [], {}
    for r in records:
        if families is not None and r["family"] not in families:
            continue
        if per_family is not None and cnt.get(r["family"], 0) >= per_family:
            continue
        cnt[r["family"]] = cnt.get(r["family"], 0) + 1
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Policy rollouts
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollouts(model, puzzles: list[Puzzle], sample: bool = False, temperature: float = 1.0,
             seed: int = 0, meta: dict | None = None, max_batch: int = 512) -> tuple[list[bool], list[int], int]:
    """Batched greedy (or sampled) rollouts. Returns (solved, steps, forward_calls)."""
    from .gnn import collate, dense_logits
    from .ppo import make_env

    solved, steps, calls = [], [], 0
    gen = torch.Generator().manual_seed(int(seed))
    for s in range(0, len(puzzles), max_batch):
        chunk = puzzles[s:s + max_batch]
        envs = [make_env(model, meta, early_termination=True) for _ in chunk]
        obs = [e.reset(options={"puzzle": p})[0] for e, p in zip(envs, chunk)]
        sol = [bool(e.done) for e in envs]   # 1-node puzzles are solved at reset
        stp = [0] * len(chunk)
        active = [i for i, e in enumerate(envs) if not e.done]
        while active:
            gb = collate([obs[i] for i in active])
            logits, _ = model(gb)
            calls += 1
            d = dense_logits(logits, gb)
            if sample:
                u = torch.rand(d.shape, generator=gen).clamp_(1e-10, 1 - 1e-10)
                d = d / max(temperature, 1e-6) - torch.log(-torch.log(u))   # Gumbel-max
            acts = d.argmax(1).numpy()
            nxt = []
            for k, i in enumerate(active):
                o, _, done, _, info = envs[i].step(int(acts[k]))
                obs[i] = o
                stp[i] += 1
                if done:
                    sol[i] = bool(info["solved"])
                else:
                    nxt.append(i)
            active = nxt
        solved += sol
        steps += stp
    return solved, steps, calls


def quick_eval(model, records: list[dict], k: int = 0, temperature: float = 1.0, seed: int = 0,
               meta: dict | None = None) -> dict:
    """Fast batched validation: greedy (+ sampled@k) solve rate per family.

    Returns {"families": {fam: {"n", "greedy", "sampled"}}, "greedy": macro mean,
    "sampled": macro mean, "seconds"}.
    """
    t0 = time.time()
    puzzles = [r["_puzzle"] for r in records]
    g, _, _ = rollouts(model, puzzles, meta=meta)
    fams: dict[str, dict] = {}
    for r, ok in zip(records, g):
        f = fams.setdefault(r["family"], {"n": 0, "greedy": 0.0, "sampled": None})
        f["n"] += 1
        f["greedy"] += ok
    if k > 0:
        s, _, _ = rollouts(model, [p for p in puzzles for _ in range(k)], sample=True,
                           temperature=temperature, seed=seed, meta=meta)
        for j, r in enumerate(records):
            f = fams[r["family"]]
            f["sampled"] = (f["sampled"] or 0.0) + any(s[j * k:(j + 1) * k])
    for f in fams.values():
        f["greedy"] /= f["n"]
        if f["sampled"] is not None:
            f["sampled"] /= f["n"]
    out = {"families": fams, "greedy": float(np.mean([f["greedy"] for f in fams.values()])) if fams else 0.0,
           "sampled": float(np.mean([f["sampled"] for f in fams.values()])) if k > 0 and fams else None,
           "seconds": round(time.time() - t0, 2)}
    return out


# --------------------------------------------------------------------------- #
# Heuristic baseline (Warnsdorff)
# --------------------------------------------------------------------------- #
def warnsdorff_rank(nbrs, visited, target: int | None, cands, usable=None) -> list[int]:
    """Next checkpoint first, then fewest free onward neighbours (Warnsdorff), then node id.

    ``usable(w, z)`` (optional) says whether z would be a legal move after
    stepping to w; moves with no usable onward neighbour are tried last.
    """
    def key(w):
        if usable is None:
            free = sum(1 for z in nbrs[w] if not visited[z])
        else:
            free = sum(1 for z in nbrs[w] if not visited[z] and z != w and usable(w, z))
        return (w != target, free == 0, free, w)
    return sorted(cands, key=key)


def heuristic_greedy(puzzle: Puzzle) -> tuple[bool, int]:
    """Warnsdorff rollout (no backtracking); onward counts respect checkpoint order."""
    from .env import ZipEnv
    env = ZipEnv(early_termination=True, compute_distances=False)
    env.reset(options={"puzzle": puzzle})
    steps, info = 0, {"solved": env.done}
    while not env.done:
        legal = np.flatnonzero(env.action_masks()).tolist()
        tgt = env.cps[env.next_cp] if env.next_cp < len(env.cps) else None
        last_move = len(env.path) + 2 == env.n   # after w only the final cell is left

        def usable(w, z, _nc=env.next_cp, _t=tgt):
            nc = _nc + (w == _t)
            if env.cp_index[z] > nc:
                return False
            return z != env.end or last_move
        a = warnsdorff_rank(env.nbrs, env.visited, tgt, legal, usable)[0] if legal else 0
        _, _, _, _, info = env.step(a)
        steps += 1
    return bool(info["solved"]), steps


def warnsdorff_order(puzzle: Puzzle):
    """move_order hook for zipsolve.solver.solve."""
    nbrs, cps = puzzle.graph.neighbors, puzzle.checkpoints

    def order(head, cands, visited, next_cp):
        tgt = cps[next_cp] if next_cp < len(cps) else None
        return warnsdorff_rank(nbrs, visited, tgt, cands)
    return order


# --------------------------------------------------------------------------- #
# Per-puzzle evaluation (runs in worker processes)
# --------------------------------------------------------------------------- #
_W: dict = {}


def _init_worker(ckpts: list[tuple[str, str]], records: list[dict], opts: dict, threads: int = 1):
    from .gnn import load_model
    torch.set_num_threads(threads)
    _W["opts"] = opts
    _W["records"] = records
    _W["puzzles"] = [Puzzle.from_dict(r["puzzle"]) for r in records]
    _W["models"] = []
    _W["calls"] = [0]
    for _, path in ckpts:
        model, meta = load_model(path)
        model.eval()
        model.register_forward_hook(lambda *a: _W["calls"].__setitem__(0, _W["calls"][0] + 1))
        _W["models"].append((model, meta))


def _row(method, rec, solved, seconds, expansions, nn_calls, ckpt=None, status=None):
    return {"ckpt": ckpt, "method": method, "id": rec["id"], "family": rec["family"], "density": rec["density"],
            "n": rec["n"], "solved": bool(solved), "seconds": float(seconds), "expansions": int(expansions),
            "nn_calls": int(nn_calls), **({"status": status} if status else {})}


def _guided_search(model, meta, puzzle, budget):
    try:
        from .evaluate import hybrid_solve
    except Exception:  # noqa: BLE001
        hybrid_solve = None
    if hybrid_solve is not None:
        r = hybrid_solve(model, puzzle, time_limit=budget, meta=meta)
        return r["status"], r.get("path"), r["nodes_expanded"]
    from ..solver import solve
    from .evaluate import policy_move_order
    r = solve(puzzle, time_limit=budget, move_order=policy_move_order(model, puzzle))
    return r.status, r.path, r.nodes_expanded


def _task(task) -> list[dict]:
    kind, ci, ri = task
    opts, rec, p = _W["opts"], _W["records"][ri], _W["puzzles"][ri]
    budget = opts["budget"]
    rows = []
    if kind == "baseline":
        from ..solver import solve
        r = solve(p, time_limit=budget)
        ok = r.status == "solved" and p.is_valid_solution(r.path)
        rows.append(_row("solver", rec, ok, r.seconds, r.nodes_expanded, 0, status=r.status))
        t0 = time.perf_counter()
        ok, steps = heuristic_greedy(p)
        rows.append(_row("heur_greedy", rec, ok, time.perf_counter() - t0, steps, 0))
        t0 = time.perf_counter()
        r = solve(p, time_limit=budget, move_order=warnsdorff_order(p), restarts=False)
        ok = r.status == "solved" and p.is_valid_solution(r.path)
        rows.append(_row("heur_search", rec, ok, time.perf_counter() - t0, r.nodes_expanded, 0, status=r.status))
        return rows
    model, meta = _W["models"][ci]
    calls = _W["calls"]
    c0, t0 = calls[0], time.perf_counter()
    (ok,), (steps,), _ = rollouts(model, [p], meta=meta)
    rows.append(_row("greedy", rec, ok, time.perf_counter() - t0, steps, calls[0] - c0, ci))
    K = opts["k"]
    if K > 0:
        c0, t0 = calls[0], time.perf_counter()
        sol, stp, _ = rollouts(model, [p] * K, sample=True, temperature=opts["temperature"],
                               seed=ri * 7919 + 17, meta=meta)
        rows.append(_row(f"sampled@{K}", rec, any(sol), time.perf_counter() - t0, sum(stp), calls[0] - c0, ci))
    if opts["search"]:
        c0, t0 = calls[0], time.perf_counter()
        status, path, nodes = _guided_search(model, meta, p, budget)
        ok = status == "solved" and path is not None and p.is_valid_solution(path)
        rows.append(_row("search", rec, ok, time.perf_counter() - t0, nodes, calls[0] - c0, ci, status))
    return rows


def evaluate_records(ckpts: list[tuple[str, str]], records: list[dict], budget: float = 1.0, k: int = 8,
                     temperature: float = 1.0, search: bool = True, baselines: bool = True,
                     threads: int = 1, progress: bool = False) -> list[dict]:
    """Run every method on every record. ckpts = [(label, path)]. Returns flat rows."""
    opts = {"budget": budget, "k": k, "temperature": temperature, "search": search}
    tasks = []
    # interleave so slow families are spread over the workers
    for ri in range(len(records)):
        if baselines:
            tasks.append(("baseline", None, ri))
        tasks += [("ckpt", ci, ri) for ci in range(len(ckpts))]
    rows: list[dict] = []
    t0 = time.time()
    if threads <= 1:
        _init_worker(ckpts, records, opts, 1)
        it = map(_task, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(threads, initializer=_init_worker,
                                            initargs=(ckpts, records, opts, 1))
        it = pool.imap_unordered(_task, tasks, chunksize=1)
    try:
        for j, r in enumerate(it):
            rows += r
            if progress and (j + 1) % max(1, len(tasks) // 10) == 0:
                print(f"  {j + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    for r in rows:
        if r["ckpt"] is not None:
            r["label"] = ckpts[r["ckpt"]][0]
            r["ckpt_path"] = ckpts[r["ckpt"]][1]
        else:
            r["label"] = "baseline"
    return rows


# --------------------------------------------------------------------------- #
# Aggregation / report
# --------------------------------------------------------------------------- #
def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    sec = np.array([r["seconds"] for r in rows])
    return {"count": len(rows), "solve_rate": float(np.mean([r["solved"] for r in rows])),
            "median_ms": float(np.median(sec) * 1e3), "p90_ms": float(np.percentile(sec, 90) * 1e3),
            "mean_expansions": float(np.mean([r["expansions"] for r in rows])),
            "mean_nn_calls": float(np.mean([r["nn_calls"] for r in rows]))}


def _method_keys(rows) -> list[tuple[str, str]]:
    """Ordered (label, method) pairs; baselines first."""
    order = {"solver": 0, "heur_greedy": 1, "heur_search": 2, "greedy": 3, "search": 5}
    keys = sorted({(r["label"], r["method"]) for r in rows},
                  key=lambda k: (k[0] != "baseline", k[0], order.get(k[1], 4)))
    return keys


def aggregate(rows: list[dict]) -> dict:
    """{label/method: {"overall", "by_family", "by_density", "instances"}} (mean over instances)."""
    out = {}
    for label, method in _method_keys(rows):
        sel = [r for r in rows if r["label"] == label and r["method"] == method]
        insts = sorted({r["ckpt"] for r in sel}, key=lambda x: -1 if x is None else x)

        def agg(filt):
            per = [_metrics([r for r in sel if r["ckpt"] == c and filt(r)]) for c in insts]
            per = [m for m in per if m]
            if not per:
                return {}
            res = {k: float(np.mean([m[k] for m in per])) for k in per[0]}
            if len(per) > 1:
                res["solve_rate_std"] = float(np.std([m["solve_rate"] for m in per]))
            return res

        fams = list(dict.fromkeys(r["family"] for r in sel))
        dens = [d for d, _ in DENSITIES if any(r["density"] == d for r in sel)]
        out[f"{label}/{method}"] = {
            "label": label, "method": method, "instances": len(insts),
            "ckpt_paths": sorted({r.get("ckpt_path") for r in sel if r.get("ckpt_path")}),
            "overall": agg(lambda r: True),
            "by_family": {f: agg(lambda r, f=f: r["family"] == f) for f in fams},
            "by_density": {d: agg(lambda r, d=d: r["density"] == d) for d in dens},
        }
    return out


def _fmt_rate(m: dict) -> str:
    if not m:
        return "-"
    s = f"{100 * m['solve_rate']:.0f}"
    if "solve_rate_std" in m:
        s += f"±{100 * m['solve_rate_std']:.0f}"
    return s


def markdown_report(summary: dict, meta: dict) -> str:
    keys = list(summary)
    L = [f"# Zip benchmark: {meta['set']} ({meta['num_puzzles']} puzzles)", "",
         f"- budget {meta['budget']}s/puzzle for searches, sampled K={meta['k']} at T={meta['temperature']}, "
         f"{meta['threads']} worker(s), wall {meta['wall_seconds']:.0f}s",
         f"- set fingerprint `{meta.get('fingerprint')}`", ""]
    for key in keys:
        s = summary[key]
        if s["ckpt_paths"]:
            L.append(f"- `{key}`: {', '.join(s['ckpt_paths'])}")
    L += ["", "## Overall", "",
          "| method | solve % | median ms | p90 ms | mean expansions | mean NN calls |",
          "|---|---:|---:|---:|---:|---:|"]
    for key in keys:
        m = summary[key]["overall"]
        L.append(f"| {key} | {_fmt_rate(m)} | {m['median_ms']:.1f} | {m['p90_ms']:.1f} | "
                 f"{m['mean_expansions']:.0f} | {m['mean_nn_calls']:.0f} |")
    fams = list(dict.fromkeys(f for k in keys for f in summary[k]["by_family"]))
    hdr = "| family | " + " | ".join(keys) + " |"
    sep = "|---|" + "---:|" * len(keys)
    L += ["", "## Solve rate (%) by family", "", hdr, sep]
    for f in fams:
        L.append(f"| {f} | " + " | ".join(_fmt_rate(summary[k]["by_family"].get(f, {})) for k in keys) + " |")
    L += ["", "## Solve rate (%) by checkpoint density", "", hdr.replace("family", "density"), sep]
    for d, _ in DENSITIES:
        L.append(f"| {d} | " + " | ".join(_fmt_rate(summary[k]["by_density"].get(d, {})) for k in keys) + " |")
    L += ["", "## Median / p90 latency (ms) by family", "", hdr, sep]
    for f in fams:
        cells = []
        for k in keys:
            m = summary[k]["by_family"].get(f, {})
            cells.append(f"{m['median_ms']:.0f} / {m['p90_ms']:.0f}" if m else "-")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L += ["", "## Mean expansions / NN calls by family", "", hdr, sep]
    for f in fams:
        cells = []
        for k in keys:
            m = summary[k]["by_family"].get(f, {})
            cells.append(f"{m['mean_expansions']:.0f} / {m['mean_nn_calls']:.0f}" if m else "-")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def parse_ckpt_arg(s: str) -> tuple[str, str]:
    if "=" in s:
        label, path = s.split("=", 1)
        return label, path
    return Path(s).stem, s


def run(ckpts: list[str], split: str = "val", threads: int = 1, budget: float = 1.0, k: int = 8,
        temperature: float = 1.0, search: bool = True, baselines: bool = True, out_dir: str = "runs/bench",
        name: str | None = None, per_family: int | None = None, families=None, root: str = DEFAULT_DIR,
        progress: bool = True) -> dict:
    data = load_set(split, root=root)
    records = subset(data["records"], per_family, families)
    for r in records:
        r.pop("_puzzle", None)
    ck = [parse_ckpt_arg(c) for c in ckpts]
    t0 = time.time()
    rows = evaluate_records(ck, records, budget, k, temperature, search, baselines, threads, progress)
    meta = {"set": data["split"], "fingerprint": data.get("fingerprint"), "num_puzzles": len(records),
            "budget": budget, "k": k, "temperature": temperature, "threads": threads,
            "wall_seconds": time.time() - t0, "ckpts": ck, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    summary = aggregate(rows)
    if name is None:
        name = f"{data['split']}_" + ("_".join(dict.fromkeys(lbl for lbl, _ in ck)) or "baselines")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"meta": meta, "summary": summary, "rows": rows}
    (out / f"{name}.json").write_text(json.dumps(report, indent=1))
    md = markdown_report(summary, meta)
    (out / f"{name}.md").write_text(md)
    report["markdown"] = md
    report["paths"] = [str(out / f"{name}.json"), str(out / f"{name}.md")]
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate the frozen sets")
    b.add_argument("--set", default="all", choices=["all", *SPLITS])
    b.add_argument("--dir", default=DEFAULT_DIR)
    b.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    b.add_argument("--check", action="store_true", help="rebuild and verify existing files are identical")
    r = sub.add_parser("run", help="evaluate checkpoints on a set")
    r.add_argument("--ckpt", action="append", default=[], help="path or LABEL=path (repeatable)")
    r.add_argument("--set", default="val", choices=list(SPLITS))
    r.add_argument("--dir", default=DEFAULT_DIR)
    r.add_argument("--threads", type=int, default=8, help="worker processes (1 torch thread each)")
    r.add_argument("--budget", type=float, default=1.0, help="seconds per puzzle for searches")
    r.add_argument("--k", type=int, default=8, help="sampled attempts per puzzle (0 = skip)")
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--no-search", action="store_true")
    r.add_argument("--no-baselines", action="store_true")
    r.add_argument("--per-family", type=int, default=None, help="only the first N puzzles per family")
    r.add_argument("--families", default=None, help="comma-separated subset of families")
    r.add_argument("--out", default="runs/bench")
    r.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        for split in (SPLITS if args.set == "all" else [args.set]):
            t0 = time.time()
            data = build_set(split, workers=args.workers)
            path = set_path(split, args.dir)
            if args.check and path.exists():
                old = json.loads(path.read_text())
                same = old.get("fingerprint") == data["fingerprint"]
                print(f"{path}: {'identical' if same else 'DIFFERENT'} ({data['fingerprint']})")
                continue
            save_set(data, path)
            print(f"{path}: {len(data['records'])} puzzles, fingerprint {data['fingerprint']}, "
                  f"{time.time() - t0:.1f}s")
        return
    torch.set_num_threads(1)
    fams = args.families.split(",") if args.families else None
    rep = run(args.ckpt, args.set, args.threads, args.budget, args.k, args.temperature, not args.no_search,
              not args.no_baselines, args.out, args.name, args.per_family, fams, args.dir)
    print(rep["markdown"])
    print("wrote", ", ".join(rep["paths"]))


if __name__ == "__main__":
    main()
