"""Puzzle sampling for RL (training curriculum, replay, seed domains).

All puzzles come from :func:`zipsolve.generator.make_puzzle` (no silent
fallback any more): a generator error is retried a few times with a fresh
random stream, each retry is logged as a warning and counted in
``GEN_STATS``, and after ``retries`` failures the error is raised.

Stage specs are strings such as ``"grid2d:5"``, ``"walls:6"``, ``"grid3d:3"``,
``"grid:3x4x2"``, ``"islands:4"`` (archipelago with 4 islands),
``"islands_chain:3"`` or ``"mask:6"``; several specs joined by ``+`` are mixed
uniformly (``"grid2d:5+walls:5"``). Optional fixed checkpoint count after
``@``: ``"grid2d:5@6"``. A *family* is one ``kind:size`` item.

Seed domains
------------
Every generated puzzle gets its own random stream
``SeedSequence(ZIP_ENTROPY, spawn_key=(domain, *key))``. The first spawn-key
word is the *domain*: ``DOMAIN_TRAIN`` for training samplers,
``DOMAIN_VAL`` / ``DOMAIN_TEST`` for the frozen benchmark
(:mod:`zipsolve.rl.benchmark`), ``DOMAIN_EVAL`` for ad-hoc evaluation. Two
streams from different domains are built from different seed-sequence inputs,
so training never re-draws a benchmark stream. In addition, training can
exclude benchmark puzzles by content (:func:`puzzle_key`), which makes the
disjointness exact rather than probabilistic.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import random as _pyrandom
import time
import zlib
from collections import Counter, deque
from typing import Callable, Iterable

import numpy as np

from ..graph import grid
from ..puzzle import Puzzle

log = logging.getLogger(__name__)

ZIP_ENTROPY = 0x5A1F_2026
DOMAIN_EVAL, DOMAIN_TRAIN, DOMAIN_VAL, DOMAIN_TEST = 0, 1, 2, 3
DOMAIN_NAMES = {DOMAIN_EVAL: "eval", DOMAIN_TRAIN: "train", DOMAIN_VAL: "val", DOMAIN_TEST: "test"}

# counters of generator trouble, visible in training logs
GEN_STATS: Counter = Counter()


def puzzle_rng(domain: int, *key: int) -> np.random.Generator:
    """Independent random stream for one puzzle, identified by (domain, *key)."""
    ss = np.random.SeedSequence(ZIP_ENTROPY, spawn_key=(int(domain),) + tuple(int(k) for k in key))
    return np.random.default_rng(ss)


def family_id(family: str) -> int:
    """Stable 32-bit id of a family string such as 'grid2d:5' (for seed keys)."""
    return zlib.crc32(family.encode())


def puzzle_key(p: Puzzle) -> str:
    """Content hash of a puzzle (graph edges + checkpoints), for de-duplication."""
    h = zlib.crc32(repr((p.graph.num_nodes, p.graph.edges(), list(map(int, p.checkpoints)))).encode())
    h2 = zlib.adler32(repr((p.graph.coords.round(3).tolist(), list(map(int, p.checkpoints)))).encode())
    return f"{h:08x}{h2:08x}"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def make_puzzle(kind: str, size, num_checkpoints=None, rng=None, retries: int = 3, **kw) -> Puzzle:
    """zipsolve.generator.make_puzzle with loud, counted retries (no fallback)."""
    from .. import generator as gen

    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    last = None
    for attempt in range(retries + 1):
        try:
            return gen.make_puzzle(kind, size, num_checkpoints=num_checkpoints, rng=rng, **dict(kw))
        except Exception as e:  # noqa: BLE001 - counted, logged, re-raised below
            last = e
            GEN_STATS["retries"] += 1
            GEN_STATS[f"retries[{kind}:{size}]"] += 1
            log.warning("generator failed for %s:%s (attempt %d/%d): %s: %s",
                        kind, size, attempt + 1, retries + 1, type(e).__name__, e)
    GEN_STATS["failures"] += 1
    raise RuntimeError(f"puzzle generation failed for {kind}:{size} after {retries + 1} attempts") from last


def using_real_generator() -> bool:
    """Kept for backwards compatibility: the real generator is always used now."""
    return True


def reseat_checkpoints(p: Puzzle, num_checkpoints: int, rng: np.random.Generator) -> Puzzle:
    """Same graph and solution path, `num_checkpoints` checkpoints re-placed along it."""
    from ..generator import place_checkpoints
    if p.solution is None:
        raise ValueError("reseat_checkpoints needs a puzzle with a known solution")
    k = int(np.clip(num_checkpoints, 2, p.num_nodes))
    return Puzzle(p.graph, [int(c) for c in place_checkpoints(list(p.solution), k, rng)], list(p.solution))


def density_checkpoints(n: int, factor: float, rng: np.random.Generator) -> int:
    """Checkpoint count = generator default for n nodes x `factor` (clipped to [2, n])."""
    from ..generator import default_num_checkpoints
    return int(np.clip(round(default_num_checkpoints(n, rng) * factor), 2, n))


def generate_puzzle(kind: str, size, rng: np.random.Generator, num_checkpoints: int | None = None,
                    density: float | tuple[float, float] | None = None) -> Puzzle:
    """One puzzle. `density` (a factor, or a (lo, hi) range sampled uniformly) scales
    the generator's default checkpoint count; ignored if `num_checkpoints` is set."""
    p = make_puzzle(kind, size, num_checkpoints=num_checkpoints, rng=rng)
    if num_checkpoints is None and density is not None:
        f = float(rng.uniform(*density)) if isinstance(density, (tuple, list)) else float(density)
        if f != 1.0:
            p = reseat_checkpoints(p, density_checkpoints(p.num_nodes, f, rng), rng)
        p._density = f  # type: ignore[attr-defined]
    return p


def _gen_task(kind, size, ncp, key, density):
    """Deterministic generation task (top level so worker processes can pickle it)."""
    t0 = time.perf_counter()
    p = generate_puzzle(kind, size, puzzle_rng(*key), num_checkpoints=ncp, density=density)
    return p, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Fixed test puzzles
# --------------------------------------------------------------------------- #
def puzzle_from_path(graph, path: list[int], num_checkpoints: int, rng: np.random.Generator) -> Puzzle:
    n = len(path)
    k = int(np.clip(num_checkpoints, 2, n))
    inner = sorted(rng.choice(np.arange(1, n - 1), size=k - 2, replace=False).tolist()) if k > 2 else []
    idx = [0] + inner + [n - 1]
    return Puzzle(graph, [int(path[i]) for i in idx], list(map(int, path)))


def snake_puzzle(rows: int, cols: int, num_checkpoints: int = 3,
                 rng: np.random.Generator | int | None = None) -> Puzzle:
    """Boustrophedon Hamiltonian path on a rows x cols grid, checkpoints along it."""
    rng = np.random.default_rng(rng)
    g = grid(rows, cols)
    path = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        path.extend(r * cols + c for c in cs)
    return puzzle_from_path(g, path, num_checkpoints, rng)


# --------------------------------------------------------------------------- #
# Stage specs
# --------------------------------------------------------------------------- #
def parse_spec(spec: str) -> list[tuple[str, object, int | None]]:
    """"grid2d:5+walls:5@4" -> [("grid2d", 5, None), ("walls", 5, 4)]."""
    out = []
    for part in spec.split("+"):
        part = part.strip()
        if not part:
            continue
        ncp = None
        if "@" in part:
            part, c = part.split("@")
            ncp = int(c)
        kind, _, size = part.partition(":")
        if "x" in size:
            sz: object = tuple(int(s) for s in size.split("x"))
        else:
            sz = int(size) if size else 5
        out.append((kind.strip(), sz, ncp))
    return out


def family_name(kind: str, size, ncp: int | None = None) -> str:
    s = "x".join(map(str, size)) if isinstance(size, tuple) else str(size)
    return f"{kind}:{s}" + (f"@{ncp}" if ncp is not None else "")


def spec_families(spec: str) -> list[str]:
    return [family_name(*it) for it in parse_spec(spec)]


def make_sampler(spec: str, seed: int | None = None, density=None,
                 domain: int = DOMAIN_EVAL) -> Callable[[], Puzzle]:
    """Simple uniform sampler over the families of `spec` (evaluation / tests).

    Puzzle i is generated from its own stream (domain, seed, family_id, i), so results do
    not depend on how many random numbers earlier puzzles consumed.
    """
    items = parse_spec(spec)
    rng = np.random.default_rng(seed)
    base = 0 if seed is None else int(seed) & 0xFFFFFFFF
    counter = [0]

    def sample() -> Puzzle:
        kind, size, ncp = items[int(rng.integers(len(items)))]
        fam = family_name(kind, size, ncp)
        p = generate_puzzle(kind, size, puzzle_rng(domain, base, family_id(fam), counter[0]), ncp, density)
        counter[0] += 1
        p._family = fam  # type: ignore[attr-defined]
        return p

    sample.spec = spec  # type: ignore[attr-defined]
    return sample


# --------------------------------------------------------------------------- #
# Background generation pool
# --------------------------------------------------------------------------- #
class PuzzlePool:
    """Prefetches deterministic generation tasks in worker processes.

    ``get(task)`` returns exactly what ``_gen_task(*task)`` would return in
    process; tasks are keyed, so a prefetched puzzle is identical to one
    generated on demand. ``plan(tasks)`` submits upcoming tasks ahead of time.
    """

    def __init__(self, workers: int = 2, max_pending: int = 256):
        ctx = mp.get_context("spawn")  # no torch state is forked into the workers
        self.pool = ctx.Pool(workers)
        self.pending: dict[tuple, object] = {}
        self.max_pending = max_pending
        self.hits = 0
        self.misses = 0

    def plan(self, tasks: Iterable[tuple]):
        for t in tasks:
            if t not in self.pending and len(self.pending) < self.max_pending:
                self.pending[t] = self.pool.apply_async(_gen_task, t)

    def get(self, task: tuple):
        res = self.pending.pop(task, None)
        if res is None:
            self.misses += 1
            return _gen_task(*task)
        self.hits += 1
        return res.get()

    def close(self):
        self.pool.terminate()
        self.pool.join()


# --------------------------------------------------------------------------- #
# Curriculum mixture sampler with replay of failed puzzles
# --------------------------------------------------------------------------- #
class MixtureSampler:
    """Training sampler: current stage / earlier stages / replay of failures.

    Each call draws a source with probabilities (``p_current``, ``p_earlier``,
    ``p_replay``), renormalised over the sources that are available (no earlier
    stages in stage 0, empty replay buffer at the start). Fresh puzzles pick a
    family uniformly from the chosen stage(s) and get a random checkpoint
    density factor from ``density`` (a (lo, hi) range; None = generator
    default). Puzzle j of family f is generated from stream
    (DOMAIN_TRAIN, seed, family_id(f), j): deterministic and resumable.

    The replay buffer holds (bounded, FIFO) training puzzles whose episode was
    not solved; it only provides *start states* for fresh on-policy rollouts.
    A replayed puzzle that gets solved is removed from the buffer.

    Returned puzzles carry ``_family`` and ``_source`` attributes.
    """

    def __init__(self, stages: list[str], seed: int = 0, p_current: float = 0.6,
                 p_earlier: float = 0.2, p_replay: float = 0.2, replay_size: int = 256,
                 density: tuple[float, float] | None = (0.6, 1.6), exclude: set[str] | None = None,
                 pool: PuzzlePool | None = None, prefetch: int = 4):
        self.stages = list(stages)
        self.stage_items = [parse_spec(s) for s in self.stages]
        self.stage = 0
        self.seed = int(seed)
        self.p = (float(p_current), float(p_earlier), float(p_replay))
        self.density = tuple(density) if density is not None else None
        self.replay: deque = deque(maxlen=max(1, int(replay_size)))
        self.use_replay = replay_size > 0 and p_replay > 0
        self.rng = np.random.default_rng(np.random.SeedSequence(ZIP_ENTROPY, spawn_key=(DOMAIN_TRAIN, self.seed, 0)))
        self.counters: Counter = Counter()
        self.exclude = set(exclude or ())
        self.pool = pool
        self.prefetch = prefetch
        self.stats: Counter = Counter()   # sources, exclusions, replay adds/removals
        self.gen_seconds = 0.0            # time spent waiting for / generating puzzles

    # -------------------------------------------------------------- stages
    def set_stage(self, i: int):
        self.stage = int(np.clip(i, 0, len(self.stages) - 1))

    def source_probs(self) -> dict[str, float]:
        pc, pe, pr = self.p
        if self.stage == 0:
            pe = 0.0
        if not (self.use_replay and len(self.replay)):
            pr = 0.0
        tot = pc + pe + pr
        if tot <= 0:
            return {"current": 1.0, "earlier": 0.0, "replay": 0.0}
        return {"current": pc / tot, "earlier": pe / tot, "replay": pr / tot}

    def families(self, stages: Iterable[int]) -> list[tuple[str, object, int | None]]:
        out = []
        for s in stages:
            for it in self.stage_items[s]:
                if it not in out:
                    out.append(it)
        return out

    # -------------------------------------------------------------- sampling
    def _task(self, item, j: int) -> tuple:
        kind, size, ncp = item
        fam = family_name(kind, size, ncp)
        return (kind, size, ncp, (DOMAIN_TRAIN, self.seed, family_id(fam), j), self.density)

    def _fresh(self, item) -> Puzzle:
        fam = family_name(*item)
        while True:
            j = self.counters[fam]
            self.counters[fam] += 1
            t0 = time.perf_counter()
            if self.pool is not None:
                p, _ = self.pool.get(self._task(item, j))
                self.pool.plan(self._task(item, j + d) for d in range(1, self.prefetch + 1))
            else:
                p, _ = _gen_task(*self._task(item, j))
            self.gen_seconds += time.perf_counter() - t0
            if self.exclude and puzzle_key(p) in self.exclude:
                self.stats["excluded"] += 1
                continue
            p._family = fam  # type: ignore[attr-defined]
            return p

    def __call__(self) -> Puzzle:
        probs = self.source_probs()
        src = ["current", "earlier", "replay"][int(self.rng.choice(3, p=[probs["current"], probs["earlier"],
                                                                        probs["replay"]]))]
        self.stats[src] += 1
        if src == "replay":
            p = self.replay[int(self.rng.integers(len(self.replay)))]
            p._source = "replay"  # type: ignore[attr-defined]
            return p
        fams = self.families([self.stage] if src == "current" else range(self.stage))
        item = fams[int(self.rng.integers(len(fams)))]
        p = self._fresh(item)
        p._source = src  # type: ignore[attr-defined]
        return p

    def episode_end(self, puzzle: Puzzle, solved: bool):
        """Feed back an episode outcome: failures enter the replay buffer."""
        if not self.use_replay:
            return
        if getattr(puzzle, "_source", None) == "replay":
            if solved:
                idx = next((i for i, q in enumerate(self.replay) if q is puzzle), None)
                if idx is None:   # restored copies (after resume): match by content
                    key = puzzle_key(puzzle)
                    idx = next((i for i, q in enumerate(self.replay) if puzzle_key(q) == key), None)
                if idx is not None:
                    del self.replay[idx]
                    self.stats["replay_removed"] += 1
        elif not solved:
            self.replay.append(puzzle)
            self.stats["replay_added"] += 1

    # -------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {
            "stage": self.stage, "seed": self.seed, "rng": self.rng.bit_generator.state,
            "counters": dict(self.counters), "stats": dict(self.stats),
            "replay": [{"family": getattr(p, "_family", "?"), "puzzle": p.to_dict()} for p in self.replay],
        }

    def load_state_dict(self, st: dict):
        self.stage = int(st["stage"])
        self.rng.bit_generator.state = st["rng"]
        self.counters = Counter(st.get("counters", {}))
        self.stats = Counter(st.get("stats", {}))
        self.replay.clear()
        for r in st.get("replay", []):
            p = Puzzle.from_dict(r["puzzle"])
            p._family = r["family"]  # type: ignore[attr-defined]
            self.replay.append(p)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None


def rng_state() -> dict:
    """Global RNG states (python, numpy, torch)."""
    import torch
    return {"python": _pyrandom.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(st: dict):
    import torch
    _pyrandom.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])


__all__ = ["make_puzzle", "make_sampler", "parse_spec", "spec_families", "family_name", "snake_puzzle",
           "puzzle_from_path", "generate_puzzle", "reseat_checkpoints", "density_checkpoints",
           "MixtureSampler", "PuzzlePool", "puzzle_rng", "puzzle_key", "family_id", "GEN_STATS",
           "DOMAIN_TRAIN", "DOMAIN_VAL", "DOMAIN_TEST", "DOMAIN_EVAL", "using_real_generator",
           "rng_state", "set_rng_state"]
