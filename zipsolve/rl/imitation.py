"""Supervised pretraining (imitation of solver solutions) and DAgger for the Zip GNN.

Subcommands::

    # 1. dataset: N puzzles over all families, split train/val BY PUZZLE
    python -m zipsolve.rl.imitation generate --n 10000 --workers 32 --out data/imit/ds.pkl --label-all
    # 2. supervised pretraining (best checkpoint by val greedy solve rate)
    python -m zipsolve.rl.imitation train --data data/imit/ds.pkl --epochs 20 --workers 32 \\
        --out checkpoints/imit.pt
    # 3. DAgger: label the policy's own mistakes with the exact solver, retrain, repeat
    python -m zipsolve.rl.imitation dagger --ckpt checkpoints/imit.pt --data data/imit/ds.pkl \\
        --rounds 3 --puzzles-per-round 2000 --workers 32 --out checkpoints/imit_dagger
    # compare checkpoints on the held-out val puzzles (greedy rollouts, per family)
    python -m zipsolve.rl.imitation eval --data data/imit/ds.pkl --ckpt a.pt b.pt

Data
----
A dataset is a pickle with one record per puzzle (``Puzzle.to_dict()`` incl. the
generator's solution, family, size, split) and compact *examples*
``(prefix, pos, excl)``: the state is the partial path ``prefix``, ``pos`` the
verified-winning next moves, ``excl`` legal moves that are *excluded* from the
loss (unknown status).  Observations are never stored: they are built through
``ZipEnv.reset(options={"puzzle": p})`` + ``ZipEnv.set_state`` (so they always
use the current env features) in parallel worker processes once per training
run and cached in RAM.

Train/val are split by puzzle index *before* any prefix is extracted; val
puzzles whose content (edges + checkpoints) equals a train puzzle are dropped.

Targets and loss
----------------
Forced states (exactly one legal move) carry no signal and get weight
``--forced-weight`` (default 0 = skipped).  States where every counted move is
positive (nothing to push down) are skipped too.

A known solution gives *one* correct move, not the only one, so targets are
sets.  The loss for a state with logits ``z`` is::

    loss = -log( sum_{m in POS} p(m) / sum_{m in DEN} p(m) )
         = logsumexp(z[DEN]) - logsumexp(z[POS]),     DEN = legal \\ EXCL  (POS subset of DEN)

* default (solution labels only): POS = {solution move}, EXCL = {} -> ordinary
  cross-entropy, the other legal moves are implicitly pushed down (they may in
  fact be winning; that is the known bias of single-solution imitation).
* ``--label-all``: every legal move of a branching state is labelled with
  ``solver.label_moves`` (``--label-time`` s per move).  POS = "win" moves,
  "lose" moves are negatives, and "unknown" (timeout) moves go to EXCL: they
  are removed from numerator *and* denominator, i.e. the loss is the
  conditional log-likelihood of picking a winning move among the moves whose
  status is known.  An unknown move is therefore neither rewarded nor
  penalised (sound: a timeout is never treated as a loss), and states with
  unknowns are still used (dropping them would bias the data towards easy
  states).  Without unknowns this is exactly -log p(POS).

Validation (each epoch): on branching solution-prefix states of the val
puzzles, top-1 accuracy (argmax == the generator's solution move) and
set-accuracy (argmax is a verified winning move; val states are always
labelled with ``label_moves``, ``--val-label-time``), plus the greedy solve
rate per family on the val puzzles (ZipEnv rollouts, argmax policy, no
backtracking).  The checkpoint with the best overall val greedy solve rate is
saved with ``gnn.save_model`` (meta ``pretrained=True``), so ``train.py
--init`` and the app can load it.

DAgger
------
Each round runs the current policy (greedy, plus ``--samples`` sampled
rollouts at ``--temperature``) on random train puzzles.  Along every *failed*
trajectory, branching states are labelled with ``label_moves`` in order until
the first state where the move the policy took is proven "lose" (its
mistake; the states after it are all unwinnable and carry no positive).
States with >= 1 winning move and >= 1 counted non-winning move become
examples (same set loss; timeouts -> EXCL, never negatives).  The examples are
aggregated over rounds, the model is fine-tuned on base + DAgger data, and the
val greedy solve rate is reported per round.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..puzzle import Puzzle
from ..solver import label_moves, legal_moves
from .env import ZipEnv
from .gnn import ZipGNN, collate, dense_logits, load_model, save_model

# --------------------------------------------------------------------------- #
# puzzle families
# --------------------------------------------------------------------------- #
FAMILIES: dict[str, tuple[int, int]] = {
    "grid2d": (4, 8),
    "walls": (5, 8),
    "mask": (6, 9),
    "islands": (2, 5),          # archipelago: number of islands
    "islands_chain": (2, 3),    # single-bridge chain: number of islands
    "grid3d": (3, 4),
    "grid4d": (2, 3),
}


def parse_families(s: str | None) -> dict[str, tuple[int, int]]:
    """"grid2d:4-8,walls:5-8,grid3d:3" -> {name: (lo, hi)}; None -> FAMILIES."""
    if not s:
        return dict(FAMILIES)
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, rng_ = part.partition(":")
        if not rng_:
            out[name] = FAMILIES[name]
            continue
        lo, _, hi = rng_.partition("-")
        out[name] = (int(lo), int(hi or lo))
    return out


def sample_checkpoints(rng: np.random.Generator, n_hint: int | None = None):
    """Varying checkpoint density: 30% generator default (~1.4 sqrt(n)), else a
    density in [0.05, 0.35] of the nodes (returned as a fraction, resolved later)."""
    if rng.random() < 0.3:
        return None
    return float(rng.uniform(0.05, 0.35))


def puzzle_signature(d: dict) -> str:
    key = json.dumps([d["kind"], sorted(map(tuple, d["edges"])), d["checkpoints"]])
    return hashlib.md5(key.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    puzzle: int                 # index into Dataset.puzzles
    prefix: tuple               # partial path (tuple of ints), head = prefix[-1]
    pos: tuple                  # verified winning next moves (non-empty)
    excl: tuple = ()            # legal moves excluded from the loss (unknown status)
    sol: int = -1               # the generator solution's next node (-1: n/a, DAgger)
    nlegal: int = 0
    weight: float = 1.0
    src: str = "sol"            # "sol" | "dagger"


def solution_examples(p: Puzzle, pid: int, label_time: float | None, forced: bool = False,
                      deep: bool = True) -> tuple[list[Example], dict]:
    """Examples from every prefix of the known solution.

    label_time None: POS = {solution move}. Otherwise branching states get all
    legal moves labelled by label_moves (unknown -> excl).
    """
    sol = [int(v) for v in p.solution]
    out: list[Example] = []
    st = {"states": 0, "forced": 0, "unknown_moves": 0, "labelled_moves": 0}
    for k in range(1, len(sol)):
        prefix = sol[:k]
        legal = legal_moves(p, prefix)
        st["states"] += 1
        if len(legal) < 2:
            st["forced"] += 1
            if forced:
                out.append(Example(pid, tuple(prefix), (sol[k],), (), sol[k], len(legal)))
            continue
        if label_time is None:
            out.append(Example(pid, tuple(prefix), (sol[k],), (), sol[k], len(legal)))
            continue
        lab = label_moves(p, prefix, label_time, known_solution=sol, deep=deep)
        pos = tuple(m for m in legal if lab[m] == "win")
        unk = tuple(m for m in legal if lab[m] == "unknown")
        st["unknown_moves"] += len(unk)
        st["labelled_moves"] += len(legal)
        out.append(Example(pid, tuple(prefix), pos, unk, sol[k], len(legal)))
    return out, st


def has_signal(e: Example) -> bool:
    """True if the loss can be non-zero (some counted move is not positive)."""
    return len(e.pos) > 0 and e.nlegal - len(e.excl) > len(e.pos)


# --------------------------------------------------------------------------- #
# dataset generation (multiprocessing)
# --------------------------------------------------------------------------- #
def _gen_task(task) -> dict | None:
    from ..generator import make_puzzle
    idx, seed, split, fam, size, cp, label_time, val_label_time, gen_time = task
    for attempt in range(6):
        rng = np.random.default_rng([seed, idx, attempt])
        try:
            if cp is not None:
                # density -> count once the graph size is known: re-place checkpoints on the path
                from ..generator import place_checkpoints
                p0 = make_puzzle(fam, size, num_checkpoints=2, rng=rng, time_limit=gen_time)
                ncp = int(max(2, min(p0.num_nodes, round(cp * p0.num_nodes))))
                cps = place_checkpoints(p0.solution, ncp, rng)
                p = Puzzle(p0.graph, cps, p0.solution)
            else:
                p = make_puzzle(fam, size, rng=rng, time_limit=gen_time)
            if p.solution is None or not p.is_valid_solution(p.solution):
                continue
            break
        except Exception:  # noqa: BLE001 - generator timeouts etc.: retry with a new seed
            continue
    else:
        return None
    t0 = time.perf_counter()
    lt = label_time if split == "train" else val_label_time
    ex, st = solution_examples(p, idx, lt)
    return {"id": idx, "family": fam, "size": size, "split": split, "puzzle": p.to_dict(),
            "examples": [(e.prefix, e.pos, e.excl, e.sol, e.nlegal) for e in ex],
            "stats": st, "label_seconds": time.perf_counter() - t0}


_POOLS: dict = {}


def _pool(workers: int):
    """Process pool, created once per worker count and reused (closed at exit)."""
    pool = _POOLS.get(workers)
    if pool is None:
        import atexit
        # forkserver: the parent holds torch threads, so plain fork() may deadlock the children
        default = "forkserver" if sys.platform.startswith("linux") else "spawn"
        pool = mp.get_context(os.environ.get("ZIPSOLVE_MP_START", default)).Pool(workers)
        _POOLS[workers] = pool
        atexit.register(pool.terminate)
    return pool


def _pmap(fn, tasks: list, workers: int, chunksize: int = 1, progress: str | None = None):
    """Ordered parallel map (sequential if workers <= 1 or there is a single task)."""
    out = []
    t0 = time.time()
    every = max(1, len(tasks) // 10)
    if workers <= 1 or len(tasks) <= 1:
        it = map(fn, tasks)
    else:
        it = _pool(workers).imap(fn, tasks, chunksize=chunksize)
    for i, r in enumerate(it):
        out.append(r)
        if progress and (i + 1) % every == 0:
            print(f"  {progress}: {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def generate_dataset(n: int, workers: int = 1, seed: int = 0, val_frac: float = 0.1,
                     families: dict | None = None, label_time: float | None = None,
                     val_label_time: float | None = 0.2, gen_time: float = 10.0,
                     verbose: bool = True) -> dict:
    """Generate n puzzles (families uniformly, sizes uniform in range) and their examples.

    The train/val split is decided per puzzle index before anything else.
    """
    families = families or dict(FAMILIES)
    rng = np.random.default_rng(seed)
    names = sorted(families)
    n_val = int(round(val_frac * n))
    split = np.array(["train"] * n, dtype=object)
    split[rng.permutation(n)[:n_val]] = "val"
    tasks = []
    for i in range(n):
        fam = names[int(rng.integers(len(names)))]
        lo, hi = families[fam]
        size = int(rng.integers(lo, hi + 1))
        cp = sample_checkpoints(rng)
        tasks.append((i, seed, str(split[i]), fam, size, cp, label_time, val_label_time, gen_time))
    t0 = time.time()
    recs = _pmap(_gen_task, tasks, workers, chunksize=4, progress="generate" if verbose else None)
    recs = [r for r in recs if r is not None]
    train_sigs = {puzzle_signature(r["puzzle"]) for r in recs if r["split"] == "train"}
    dropped = [r["id"] for r in recs if r["split"] == "val" and puzzle_signature(r["puzzle"]) in train_sigs]
    recs = [r for r in recs if not (r["split"] == "val" and r["id"] in set(dropped))]
    meta = {"n": n, "seed": seed, "val_frac": val_frac, "families": families,
            "label_time": label_time, "val_label_time": val_label_time,
            "generated": len(recs), "failed": n - len(recs) - len(dropped),
            "val_duplicates_dropped": len(dropped), "seconds": round(time.time() - t0, 1)}
    ds = {"version": 1, "meta": meta, "records": recs}
    if verbose:
        s = dataset_summary(ds)
        print(json.dumps(s, indent=1), flush=True)
    return ds


def dataset_summary(ds: dict) -> dict:
    out: dict = {"meta": ds["meta"]}
    for split in ("train", "val"):
        rs = [r for r in ds["records"] if r["split"] == split]
        ex = [e for r in rs for e in r["examples"]]
        out[split] = {"puzzles": len(rs), "examples": len(ex),
                      "branching": sum(1 for e in ex if e[4] >= 2),
                      "unknown_moves": sum(len(e[2]) for e in ex),
                      "per_family": {f: sum(1 for r in rs if r["family"] == f)
                                     for f in sorted({r["family"] for r in rs})}}
    return out


def save_dataset(ds: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


class Dataset:
    """In-memory view: puzzles by id plus Example lists for train / val."""

    def __init__(self, ds: dict, forced_weight: float = 0.0):
        self.raw = ds
        self.puzzles: dict[int, Puzzle] = {}
        self.pdict: dict[int, dict] = {}
        self.family: dict[int, str] = {}
        self.split: dict[int, str] = {}
        self.train: list[Example] = []
        self.val: list[Example] = []
        for r in ds["records"]:
            pid = r["id"]
            self.pdict[pid] = r["puzzle"]
            self.puzzles[pid] = Puzzle.from_dict(r["puzzle"])
            self.family[pid] = r["family"]
            self.split[pid] = r["split"]
            for prefix, pos, excl, sol, nl in r["examples"]:
                e = Example(pid, tuple(prefix), tuple(pos), tuple(excl), sol, nl)
                if r["split"] == "train":
                    if nl < 2:
                        if forced_weight <= 0:
                            continue
                        e.weight = forced_weight
                    elif not has_signal(e):
                        continue
                    self.train.append(e)
                else:
                    if nl >= 2:
                        self.val.append(e)
        self.train_ids = sorted(p for p, s in self.split.items() if s == "train")
        self.val_ids = sorted(p for p, s in self.split.items() if s == "val")


# --------------------------------------------------------------------------- #
# observations (always through ZipEnv)
# --------------------------------------------------------------------------- #
def make_env() -> ZipEnv:
    return ZipEnv(early_termination=True)


def state_obs(env: ZipEnv, puzzle: Puzzle, prefix: Sequence[int]) -> dict:
    """Observation of the state after `prefix` (env must be reset on `puzzle`)."""
    prefix = [int(v) for v in prefix]
    vis = np.zeros(puzzle.num_nodes, dtype=bool)
    vis[prefix] = True
    cpset = set(puzzle.checkpoints)
    next_cp = sum(1 for v in prefix if v in cpset)
    return env.set_state(vis, prefix[-1], next_cp, path=prefix)


def _obs_task(task) -> list[dict]:
    pdict, prefixes = task
    p = Puzzle.from_dict(pdict)
    env = make_env()
    env.reset(options={"puzzle": p})
    return [state_obs(env, p, pre) for pre in prefixes]


def build_obs(ds: Dataset, examples: list[Example], workers: int = 1, verbose: bool = True) -> list[dict]:
    """Observations aligned with `examples` (grouped per puzzle, built in workers)."""
    groups: dict[int, list[int]] = {}
    for i, e in enumerate(examples):
        groups.setdefault(e.puzzle, []).append(i)
    keys = list(groups)
    tasks = [(ds.pdict[k], [examples[i].prefix for i in groups[k]]) for k in keys]
    res = _pmap(_obs_task, tasks, workers, chunksize=8,
                progress=f"obs ({len(examples)} states)" if verbose and len(tasks) > 200 else None)
    out: list[dict | None] = [None] * len(examples)
    for k, obs_list in zip(keys, res):
        for i, o in zip(groups[k], obs_list):
            out[i] = o
    for e, o in zip(examples, out):  # sanity: targets must be legal in the env too
        m = o["action_mask"]
        if not all(m[v] for v in e.pos) or int(m.sum()) != e.nlegal and e.nlegal:
            raise RuntimeError("env action mask disagrees with solver.legal_moves")
    return out  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# loss / batching
# --------------------------------------------------------------------------- #
def set_loss(dlogits: torch.Tensor, pos: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Per-state loss logsumexp(z[den]) - logsumexp(z[pos]) = -log(P(pos) / P(den)).

    dlogits (B, M) dense logits (-inf padding / illegal); pos, den (B, M) bool,
    pos subset of den, pos non-empty.
    """
    ninf = torch.tensor(float("-inf"), dtype=dlogits.dtype)
    lp = torch.logsumexp(torch.where(pos, dlogits, ninf), 1)
    ld = torch.logsumexp(torch.where(den, dlogits, ninf), 1)
    return ld - lp


def target_masks(examples: Sequence[Example], obs: Sequence[dict], max_nodes: int):
    B = len(examples)
    pos = torch.zeros(B, max_nodes, dtype=torch.bool)
    den = torch.zeros(B, max_nodes, dtype=torch.bool)
    for b, (e, o) in enumerate(zip(examples, obs)):
        m = o["action_mask"]
        den[b, :len(m)] = torch.from_numpy(np.asarray(m, dtype=bool))
        for v in e.excl:
            den[b, v] = False
        for v in e.pos:
            pos[b, v] = True
            den[b, v] = True
    return pos, den


# --------------------------------------------------------------------------- #
# rollouts and evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(model: ZipGNN, puzzles: Sequence[Puzzle], sample: bool = False, temperature: float = 1.0,
            seed: int = 0, batch: int = 512) -> list[tuple[bool, list[int]]]:
    """Batched policy rollouts (ZipEnv, early termination at provably dead states).

    Greedy (argmax) or sampled at `temperature`. Returns [(solved, path)].
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    envs = [make_env() for _ in puzzles]
    obs, res = [], [None] * len(puzzles)
    active = []
    for i, (e, p) in enumerate(zip(envs, puzzles)):
        o, info = e.reset(options={"puzzle": p})
        obs.append(o)
        if e.done:
            res[i] = (bool(info["solved"]), list(e.path))
        else:
            active.append(i)
    while active:
        nxt = []
        for c in range(0, len(active), batch):
            chunk = active[c:c + batch]
            gb = collate([obs[i] for i in chunk])
            logits, _ = model(gb)
            dl = dense_logits(logits, gb)
            if sample:
                pr = torch.softmax(dl / max(temperature, 1e-6), 1)
                acts = torch.multinomial(pr, 1, generator=gen).squeeze(1).numpy()
            else:
                acts = dl.argmax(1).numpy()
            for k, i in enumerate(chunk):
                o, _, done, _, info = envs[i].step(int(acts[k]))
                obs[i] = o
                if done:
                    res[i] = (bool(info["solved"]), list(info["path"]))
                else:
                    nxt.append(i)
        active = nxt
    return res  # type: ignore[return-value]


def greedy_by_family(model: ZipGNN, ds: Dataset, ids: Sequence[int] | None = None) -> dict:
    ids = list(ds.val_ids if ids is None else ids)
    out = rollout(model, [ds.puzzles[i] for i in ids])
    fam: dict[str, list[bool]] = {}
    for i, (ok, path) in zip(ids, out):
        if ok:
            assert ds.puzzles[i].is_valid_solution(path)
        fam.setdefault(ds.family[i], []).append(ok)
    res = {f: round(float(np.mean(v)), 4) for f, v in sorted(fam.items())}
    res["all"] = round(float(np.mean([ok for ok, _ in out])), 4) if out else 0.0
    res["n"] = len(ids)
    return res


@torch.no_grad()
def state_metrics(model: ZipGNN, examples: Sequence[Example], obs: Sequence[dict], batch: int = 512) -> dict:
    """top1: argmax == solution move; set_acc: argmax is a verified win; loss."""
    model.eval()
    if not examples:
        return {}
    top1 = setacc = 0
    tot_loss = 0.0
    for c in range(0, len(examples), batch):
        ex, ob = examples[c:c + batch], obs[c:c + batch]
        gb = collate(list(ob))
        logits, _ = model(gb)
        dl = dense_logits(logits, gb)
        pos, den = target_masks(ex, ob, gb.max_nodes)
        tot_loss += float(set_loss(dl, pos, den).sum())
        am = dl.argmax(1)
        for b, e in enumerate(ex):
            a = int(am[b])
            top1 += a == e.sol
            setacc += a in e.pos
    n = len(examples)
    return {"top1": round(top1 / n, 4), "set_acc": round(setacc / n, 4), "loss": round(tot_loss / n, 4),
            "states": n}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    epochs: int = 10
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: float = 0.03
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 1
    minutes: float = 0.0        # 0 = no wall-clock limit (else stop after the epoch that crosses it)


def train_loop(model: ZipGNN, examples: list[Example], obs: list[dict], ds: Dataset,
               val_obs: list[dict], cfg: TrainConfig, save_path: str | None = None,
               meta: dict | None = None, log: list | None = None, tag: str = "") -> dict:
    """AdamW + warmup/cosine LR over `examples`; per-epoch val metrics; saves best by val greedy."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(examples) / cfg.batch))
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(cfg.warmup * total))

    def lr_at(step):
        if step < warm:
            return cfg.lr * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return cfg.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    weights = np.array([e.weight for e in examples], dtype=np.float32)
    best = {"val_greedy": -1.0}
    log = log if log is not None else []
    step = 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(len(examples))
        el, en = 0.0, 0
        for c in range(0, len(perm), cfg.batch):
            idx = perm[c:c + cfg.batch]
            ex = [examples[i] for i in idx]
            ob = [obs[i] for i in idx]
            gb = collate(ob)
            logits, _ = model(gb)
            dl = dense_logits(logits, gb)
            pos, den = target_masks(ex, ob, gb.max_nodes)
            w = torch.from_numpy(weights[idx])
            loss = (set_loss(dl, pos, den) * w).sum() / w.sum().clamp(min=1e-8)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            el += loss.item() * len(idx)
            en += len(idx)
        row = {"tag": tag, "epoch": ep, "train_loss": round(el / max(1, en), 4),
               "lr": round(lr_at(step - 1), 6), "time": round(time.time() - t0, 1)}
        last = ep == cfg.epochs or (cfg.minutes and time.time() - t0 > cfg.minutes * 60)
        if ep % cfg.eval_every == 0 or last:
            vm = state_metrics(model, ds.val, val_obs)
            vg = greedy_by_family(model, ds)
            row.update({"val_" + k: v for k, v in vm.items()})
            row["val_greedy"] = vg
            if vg["all"] > best["val_greedy"]:
                best = {"val_greedy": vg["all"], "epoch": ep, "per_family": vg, "state": vm}
                if save_path:
                    save_imitation(model, save_path, meta, best)
                    row["saved"] = save_path
        log.append(row)
        print(json.dumps(row), flush=True)
        if last:
            break
    return best


def save_imitation(model: ZipGNN, path, meta: dict | None, val: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    from . import env as env_mod
    save_model(model, path, pretrained=True, imitation=meta or {}, val=val,
               feature_names=list(getattr(env_mod, "FEATURES", [])))


def _load_model(path: str) -> ZipGNN:
    return load_model(path)[0]


def _model_meta(args, ds: Dataset) -> dict:
    return {"args": {k: v for k, v in vars(args).items() if k != "func"},
            "dataset": ds.raw.get("meta", {}), "train_states": len(ds.train)}


# --------------------------------------------------------------------------- #
# DAgger
# --------------------------------------------------------------------------- #
def _dagger_task(task) -> tuple[list[tuple], dict]:
    """Label branching states along one policy trajectory.

    Stops after the first state whose taken move is proven "lose" (the
    mistake). Returns ([(prefix, pos, excl, nlegal)], stats).
    """
    pdict, path, label_time, all_states = task
    p = Puzzle.from_dict(pdict)
    out = []
    st = {"labelled": 0, "unknown": 0, "mistake_at": -1}
    for k in range(1, len(path)):
        prefix = path[:k]
        legal = legal_moves(p, prefix)
        if len(legal) < 2:
            continue
        lab = label_moves(p, prefix, label_time)
        st["labelled"] += 1
        pos = tuple(m for m in legal if lab[m] == "win")
        unk = tuple(m for m in legal if lab[m] == "unknown")
        st["unknown"] += len(unk)
        if pos and len(legal) - len(unk) > len(pos):
            out.append((tuple(prefix), pos, unk, len(legal)))
        if lab.get(path[k]) == "lose" and not all_states:
            st["mistake_at"] = k
            break
    return out, st


def dagger_collect(model: ZipGNN, ds: Dataset, pids: Sequence[int], label_time: float,
                   workers: int, samples: int = 0, temperature: float = 1.0, seed: int = 0,
                   include_solved: bool = False) -> tuple[list[Example], dict]:
    puzzles = [ds.puzzles[i] for i in pids]
    trajs = [(pid, ok, path) for pid, (ok, path) in zip(pids, rollout(model, puzzles))]
    greedy_rate = float(np.mean([ok for _, ok, _ in trajs])) if trajs else 0.0
    for s in range(samples):
        rs = rollout(model, puzzles, sample=True, temperature=temperature, seed=seed * 1000 + s)
        trajs += [(pid, ok, path) for pid, (ok, path) in zip(pids, rs)]
    tasks, owners = [], []
    seen = set()
    for pid, ok, path in trajs:
        if ok and not include_solved:
            continue
        key = (pid, tuple(path))
        if key in seen:
            continue
        seen.add(key)
        tasks.append((ds.pdict[pid], path, label_time, ok))
        owners.append(pid)
    res = _pmap(_dagger_task, tasks, workers, chunksize=2)
    ex: list[Example] = []
    stats = {"trajectories": len(trajs), "failed": sum(1 for _, ok, _ in trajs if not ok),
             "labelled_trajectories": len(tasks), "train_greedy": round(greedy_rate, 4),
             "labelled_states": 0, "unknown_moves": 0, "mistakes_found": 0}
    for pid, (items, st) in zip(owners, res):
        stats["labelled_states"] += st["labelled"]
        stats["unknown_moves"] += st["unknown"]
        stats["mistakes_found"] += st["mistake_at"] >= 0
        for prefix, pos, unk, nl in items:
            ex.append(Example(pid, prefix, pos, unk, -1, nl, 1.0, "dagger"))
    stats["new_examples"] = len(ex)
    return ex, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_generate(args) -> dict:
    fams = parse_families(args.families)
    ds = generate_dataset(args.n, args.workers, args.seed, args.val_frac, fams,
                          args.label_time if args.label_all else None,
                          args.val_label_time if args.val_label_time > 0 else None,
                          args.gen_time)
    save_dataset(ds, args.out)
    print(f"saved {args.out}")
    return ds


def _prepare(args) -> tuple[Dataset, list[dict], list[dict]]:
    t0 = time.time()
    ds = Dataset(load_dataset(args.data), forced_weight=args.forced_weight)
    if args.max_train_states and len(ds.train) > args.max_train_states:
        r = np.random.default_rng(args.seed)
        ds.train = [ds.train[i] for i in sorted(r.choice(len(ds.train), args.max_train_states, replace=False))]
    obs = build_obs(ds, ds.train, args.workers)
    vobs = build_obs(ds, ds.val, args.workers)
    print(f"dataset: {len(ds.train_ids)} train / {len(ds.val_ids)} val puzzles, "
          f"{len(ds.train)} train states, {len(ds.val)} val branching states "
          f"(obs built in {time.time() - t0:.0f}s)", flush=True)
    return ds, obs, vobs


def cmd_train(args) -> dict:
    torch.set_num_threads(args.threads)
    ds, obs, vobs = _prepare(args)
    if args.init:
        model = _load_model(args.init)
    else:
        kw = {}
        if args.head_version:
            kw["head_version"] = args.head_version
        if args.global_attn:
            kw["global_attn"] = True
        model = ZipGNN(hidden=args.hidden, layers=args.layers, **kw)
    print(f"model: {getattr(model, 'config', {})}", flush=True)
    cfg = TrainConfig(epochs=args.epochs, batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                      seed=args.seed, minutes=args.minutes)
    log: list = []
    best = train_loop(model, ds.train, obs, ds, vobs, cfg, args.out, _model_meta(args, ds), log, "pretrain")
    _write_log(args.out, log)
    print("best:", json.dumps(best))
    return best


def cmd_dagger(args) -> dict:
    torch.set_num_threads(args.threads)
    ds, obs, vobs = _prepare(args)
    model = _load_model(args.ckpt)
    base = greedy_by_family(model, ds)
    print("round 0 (input checkpoint) val greedy:", json.dumps(base), flush=True)
    history = [{"round": 0, "val_greedy": base}]
    agg_ex: list[Example] = []
    agg_obs: list[dict] = []
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    best_all = base["all"]
    log: list = []
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        k = min(args.puzzles_per_round, len(ds.train_ids))
        pids = sorted(rng.choice(ds.train_ids, size=k, replace=False).tolist())
        new_ex, st = dagger_collect(model, ds, pids, args.label_time, args.workers, args.samples,
                                    args.temperature, args.seed + rnd, args.include_solved)
        new_obs = build_obs(ds, new_ex, args.workers, verbose=False)
        agg_ex += new_ex
        agg_obs += new_obs
        st["collect_seconds"] = round(time.time() - t0, 1)
        print(f"round {rnd} collect: {json.dumps(st)}", flush=True)
        ex = ds.train + agg_ex
        for e in agg_ex:
            e.weight = args.dagger_weight
        ob = obs + agg_obs
        cfg = TrainConfig(epochs=args.epochs, batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                          seed=args.seed + rnd, minutes=args.minutes)
        path = str(out) + f"_r{rnd}.pt"
        meta = _model_meta(args, ds)
        meta.update({"dagger_round": rnd, "dagger_examples": len(agg_ex), "init": args.ckpt})
        best = train_loop(model, ex, ob, ds, vobs, cfg, path, meta, log, f"dagger{rnd}")
        model = _load_model(path)  # continue from the round's best
        history.append({"round": rnd, "val_greedy": best["per_family"], "collect": st,
                        "dagger_examples": len(agg_ex), "ckpt": path})
        if best["val_greedy"] > best_all:
            best_all = best["val_greedy"]
            save_imitation(model, str(out) + "_best.pt", meta, best)
        print(f"round {rnd}: val greedy {json.dumps(best['per_family'])} ({time.time() - t0:.0f}s)", flush=True)
    _write_log(str(out) + ".pt", log, extra={"rounds": history})
    print(json.dumps(history, indent=1))
    return {"rounds": history}


def cmd_eval(args) -> dict:
    torch.set_num_threads(args.threads)
    ds = Dataset(load_dataset(args.data))
    res = {}
    vobs = build_obs(ds, ds.val, args.workers, verbose=False) if args.states else None
    for ck in args.ckpt:
        model = _load_model(ck)
        r = {"val_greedy": greedy_by_family(model, ds)}
        if vobs is not None:
            r["val_states"] = state_metrics(model, ds.val, vobs)
        res[ck] = r
        print(ck, json.dumps(r), flush=True)
    return res


def _write_log(ckpt_path: str | None, log: list, extra: dict | None = None) -> None:
    if not ckpt_path:
        return
    p = Path(str(ckpt_path)).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"log": log, **(extra or {})}, indent=1, default=str))


def _add_train_args(p: argparse.ArgumentParser, defaults_epochs: int = 10) -> None:
    p.add_argument("--data", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--threads", type=int, default=4, help="torch intra-op threads")
    p.add_argument("--epochs", type=int, default=defaults_epochs)
    p.add_argument("--batch", type=int, default=128, help="states per batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--forced-weight", type=float, default=0.0,
                   help="weight of forced (single legal move) states; 0 = skip them")
    p.add_argument("--max-train-states", type=int, default=0, help="subsample train states (0 = all)")
    p.add_argument("--minutes", type=float, default=0.0, help="stop after the epoch crossing this (0 = off)")
    p.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a puzzle/example dataset")
    g.add_argument("--n", type=int, default=10_000, help="number of puzzles")
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    g.add_argument("--out", default="data/imitation/dataset.pkl")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--val-frac", type=float, default=0.1)
    g.add_argument("--families", default=None,
                   help="e.g. 'grid2d:4-8,walls:5-8,mask:6-9,islands:2-5,islands_chain:2-3,grid3d:3-4,grid4d:2-3'")
    g.add_argument("--label-all", action="store_true",
                   help="label every legal move of branching train states with solver.label_moves")
    g.add_argument("--label-time", type=float, default=0.2, help="seconds per move for --label-all")
    g.add_argument("--val-label-time", type=float, default=0.2,
                   help="seconds per move for labelling val states (set-accuracy); 0 = solution only")
    g.add_argument("--gen-time", type=float, default=10.0, help="generator time limit per puzzle")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="supervised pretraining")
    _add_train_args(t)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--head-version", type=int, default=0, help="ZipGNN head_version (0 = gnn default)")
    t.add_argument("--global-attn", action="store_true", help="ZipGNN global attention blocks")
    t.add_argument("--init", default=None, help="start from this checkpoint")
    t.add_argument("--out", default="checkpoints/imit.pt")
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("dagger", help="DAgger rounds on top of a checkpoint")
    _add_train_args(d, defaults_epochs=3)
    d.add_argument("--ckpt", required=True)
    d.add_argument("--rounds", type=int, default=3)
    d.add_argument("--puzzles-per-round", type=int, default=1000)
    d.add_argument("--samples", type=int, default=1, help="sampled rollouts per puzzle (besides greedy)")
    d.add_argument("--temperature", type=float, default=1.0)
    d.add_argument("--label-time", type=float, default=0.2, help="label_moves seconds per move")
    d.add_argument("--include-solved", action="store_true", help="also label states of solved trajectories")
    d.add_argument("--dagger-weight", type=float, default=1.0, help="loss weight of DAgger examples")
    d.add_argument("--out", default="checkpoints/imit_dagger", help="prefix: <out>_r{k}.pt, <out>_best.pt")
    d.set_defaults(func=cmd_dagger, lr=3e-4)

    e = sub.add_parser("eval", help="per-family greedy solve rate on the val puzzles")
    e.add_argument("--data", required=True)
    e.add_argument("--ckpt", nargs="+", required=True)
    e.add_argument("--states", action="store_true", help="also top-1 / set-accuracy on val states")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--threads", type=int, default=4)
    e.set_defaults(func=cmd_eval)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
