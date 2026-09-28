"""The Architect: an AI that *designs* hard but fair Zip puzzles.

The random generator (``zipsolve.generator``) draws a solution path, sprinkles
checkpoints along it and adds more until the solution is unique.  Its puzzles
are unique, but how hard they feel is luck, and the hard ones are usually hard
because the player has to *guess*.  The Architect instead treats a puzzle as a
design to optimise against the human-difficulty rater
(``zipsolve.difficulty.rate``): it wants puzzles that are

* **unique**  - exactly one solution;
* **fair**    - solvable by deduction alone: ``features["guesses"] == 0``;
* **hard**    - rated Expert (62-80) or Insane (80+);
* **pretty**  - not boring, numbers spread over the board (Clark-Evans >= 0.8).

How it works
------------
1. *Design space.*  A board (grid, walls grid, archipelago, or any graph
   ``generator.make_puzzle`` can build) and a random Hamiltonian path on it are
   fixed; the design is the **set of path positions that carry a checkpoint**
   (+ for the walls kind, the set of **walls on edges the path does not use**).
   Every design is automatically consistent with the path, so the path is
   always a solution.

2. *Fairness implies uniqueness.*  The rater replays the path and, for every
   wrong move at every step, looks for a *sound* refutation (dead ends, region
   splits, parity / articulation rules, one- and two-level forced lookahead).
   ``guesses == 0`` means every deviation from the path was proven dead, so no
   second solution can exist.  The rater costs ~5-15 ms on a 9x9, orders of
   magnitude cheaper than counting solutions, which is what makes search
   affordable.  The final design is still certified with the exact solver
   (``count_solutions(p, 2) == (1, "complete")``, time-limited; a timeout
   counts as *not proven* and the next-best design is tried).

3. *Search.*  Start dense (a checkpoint on every other cell: trivially fair),
   then the classic minimal-clue trick: **remove clues while fairness holds**
   (minimal-clue puzzles are the hardest).  Then **simulated annealing** over
   moves *shift a checkpoint along the path*, *remove*, *add*, *relocate*,
   *add/remove a wall*; unfair designs are rejected outright, fair ones are
   accepted by the Metropolis rule on the objective

       J = min(score, sweet) - 2 * max(0, score - ceiling)       (target band)
           - 20 * boring - 40 * max(0, 0.8 - spread)

   (expert: sweet 75, ceiling 79.5 so designs stay in the Expert band;
   insane: sweet 100, no ceiling).  Several random paths are explored as a
   **beam with successive halving**: every round the worse half of the paths
   is dropped and its budget goes to the survivors.

4. *Learning (the surrogate).*  A small neural *move model* (MLP,
   [f(candidate), f(candidate) - f(current)] = 2 x 27 hand features -> 64 -> 64
   -> [score change, fairness logit]) is trained on the moves the search itself
   rated with the exact rater (``collect_samples`` + ``train_surrogate``;
   64 800 moves from 216 surrogate-free runs over classic 7-10, walls 8-9,
   islands 5-7).  Features are cheap (~1 ms): checkpoint density / spread /
   path-gap statistics, board statistics and an *ambiguity profile* of the
   solution walk (how many wrong moves each step offers and how many of them a
   one-cell pocket test does not already kill).  During annealing the
   Architect draws ``M`` random moves, ranks them by expected improvement
   ``P(fair) * softplus(predicted gain)`` (an unfair design is only rejected,
   so only the upside counts) and spends an exact rating on the best one
   (20 % of the time on a random one, to keep exploring).
   ``python -m zipsolve.architect --train`` rebuilds it and prints held-out
   (by search run) metrics; shipped model: fairness AUC 0.86 (accuracy 0.77
   vs 0.51 base rate), Spearman 0.59 on the score change and 0.96 on the
   implied score of fair candidates.  Honest ablation: at an equal number of
   exact ratings, surrogate-ranked and random proposals end at the same scores
   within noise (e.g. 9x9 insane 73.0 vs 73.9, walls 9x9 70.1 vs 70.6,
   islands 6 58.0 vs 57.4, 12 seeds each) - the rater costs only 5-15 ms, so
   there is little to save; the model earns its keep when ratings get
   expensive (bigger boards, slower raters).  ``surrogate=None`` switches it off.

Budget and determinism
----------------------
``design(kind, size, target, time_budget, seed)`` converts ``time_budget``
into a fixed number of exact evaluations (a per-board-size cost estimate), so
the result depends only on the arguments: same seed -> same puzzle.  The time
budget is also a hard wall-clock cap; if a slow machine hits it, the search
stops early (and only then can results differ between machines).

Public API
----------
``design(kind, size, target="expert", time_budget=30, seed=0) -> DesignResult``
(``.puzzle``, ``.rating``, ``.log``, ``.stats``, ``.ok``, ``.to_dict()``),
``design_features(puzzle, path)``, ``Surrogate``, ``train_surrogate``,
``collect_samples``, ``load_surrogate``.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .difficulty import rate
from .graph import ZipGraph, grid, remove_edges
from .puzzle import Puzzle

__all__ = ["design", "DesignResult", "TARGETS", "design_features", "FEATURE_NAMES", "Surrogate",
           "train_surrogate", "collect_samples", "load_surrogate", "spearman", "SURROGATE_PATH"]

SURROGATE_PATH = Path(__file__).with_name("architect_surrogate.npz")
MIN_SPREAD = 0.8
# target -> (accept threshold, sweet spot, ceiling or None)
TARGETS = {"hard": (42.0, 58.0, 61.5), "expert": (62.0, 75.0, 79.5), "insane": (80.0, 100.0, None)}
KIND_ALIASES = {"classic": "grid2d", "grid": "grid2d", "cube": "grid3d"}
LOG_CAP = 160
EI_SCALE = 2.0            # surrogate ranking: softplus temperature of the predicted score change
EXPLORE = 0.2             # ... and the chance of rating a random proposal instead of the best-ranked


# ---------------------------------------------------------------------------
# board + path
# ---------------------------------------------------------------------------
@dataclass
class Board:
    kind: str                       # canonical kind (grid2d, walls, islands, ...)
    base: ZipGraph                  # graph without design walls
    path: list[int]
    wall_edges: list[tuple[int, int]] = field(default_factory=list)   # edges walls may go on (walls kind)
    _cache: dict = field(default_factory=dict)

    def graph(self, walls: frozenset) -> ZipGraph:
        if not walls:
            return self.base
        g = self._cache.get(walls)
        if g is None:
            if len(self._cache) > 256:
                self._cache.clear()
            g = remove_edges(self.base, [self.wall_edges[i] for i in sorted(walls)])
            self._cache[walls] = g
        return g


def _size_tuple(size) -> tuple[int, int]:
    if isinstance(size, (list, tuple)):
        return int(size[0]), int(size[-1])
    return int(size), int(size)


def make_board(kind: str, size, rng: np.random.Generator, time_limit: float = 20.0) -> Board:
    """A board and a random Hamiltonian path on it (the solution the design is built around)."""
    from .generator import make_puzzle, random_hamiltonian_path
    kind = KIND_ALIASES.get(kind, kind)
    if kind in ("grid2d", "walls"):
        g = grid(*_size_tuple(size))
        path = random_hamiltonian_path(g, rng, time_limit=time_limit)
        if path is None:
            raise ValueError("no Hamiltonian path found")
        path = [int(v) for v in path]
        if kind == "grid2d":
            return Board(kind, g, path)
        used = {frozenset(e) for e in zip(path, path[1:])}
        free = [e for e in g.edges() if frozenset(e) not in used]
        return Board(kind, g, path, free)
    # any other kind: let the generator build the board, keep its solution path
    p = make_puzzle(kind, size, num_checkpoints=2, rng=rng, unique=False, time_limit=time_limit)
    if p.solution is None:
        raise ValueError(f"generator returned no solution for {kind}")
    return Board(kind, p.graph, [int(v) for v in p.solution])


# ---------------------------------------------------------------------------
# features (shared by the surrogate and the stats)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "n_scaled", "log_n", "cp_density", "cp_count_scaled", "spread",
    "gap_mean", "gap_max", "gap_std", "gap_ge4", "gap_ge8", "gap_sq",
    "amb_any", "amb_two", "amb_mean", "hard_frac", "hard_per_node", "hard_run", "hard_in_gap",
    "walls_per_node", "deg_mean", "deg_le2", "cp_border", "turn_frac",
    "kind_grid", "kind_walls", "kind_islands", "kind_other",
]


def _spread(coords: np.ndarray, cps: Sequence[int], n: int) -> float:
    k = len(cps)
    if k < 3:
        return 1.0
    P = coords[list(cps)]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1).mean()
    dim = coords.shape[1]
    expected = 0.5 * math.sqrt(n / k) if dim == 2 else 0.554 * (n / k) ** (1 / 3) if dim == 3 \
        else 0.6 * (n / k) ** (1 / dim)
    return float(nn / max(expected, 1e-9))


def design_features(puzzle: Puzzle, path: Sequence[int], kind: str | None = None) -> np.ndarray:
    """Cheap numeric description of a design (puzzle + its solution path)."""
    g = puzzle.graph
    nbrs = g.neighbors
    n = g.num_nodes
    cps = puzzle.checkpoints
    k = len(cps)
    pos = {v: i for i, v in enumerate(path)}
    cpi = {v: i for i, v in enumerate(cps)}
    idx = sorted(pos[c] for c in cps)
    gaps = np.diff(idx).astype(float) if k > 1 else np.array([float(n)])
    # ambiguity profile of the solution walk
    visited = bytearray(n)
    free_deg = [len(nbrs[v]) for v in range(n)]
    end = cps[-1]

    def visit(v):
        visited[v] = 1
        for w in nbrs[v]:
            free_deg[w] -= 1

    visit(path[0])
    nxt = 1
    amb_any = amb_two = 0
    wrong_total = 0
    hard_steps = 0
    hard_total = 0
    run = best_run = 0
    hard_in_gap = 0
    for i in range(n - 1):
        h = path[i]
        right = path[i + 1]
        wrong = []
        for w in nbrs[h]:
            if visited[w] or w == right:
                continue
            c = cpi.get(w, -1)
            if c >= 0 and c != nxt:
                continue
            if w == end and i < n - 2:
                continue
            wrong.append(w)
        if wrong:
            amb_any += 1
            amb_two += len(wrong) >= 2
            wrong_total += len(wrong)
            hard = 0
            # pocket test: a neighbour of the head left with < 2 free cells kills every wrong move
            pocket = any(not visited[x] and x != end and free_deg[x] < 2 for x in nbrs[h])
            for w in wrong:
                dead = pocket or (free_deg[w] == 0 and w != end)
                if not dead:
                    hard += 1
            if hard:
                hard_steps += 1
                hard_total += hard
                run += 1
                best_run = max(best_run, run)
                if nxt < k and pos[cps[nxt]] - i >= 6:
                    hard_in_gap += 1
            else:
                run = 0
        else:
            run = 0
        visit(right)
        if nxt < k and right == cps[nxt]:
            nxt += 1
    steps = max(1, n - 1)
    coords = g.coords
    # border checkpoints (bounding box of the coordinates) and path turns
    lo, hi = coords.min(0), coords.max(0)
    border = sum(1 for c in cps if np.any(coords[c] == lo) or np.any(coords[c] == hi)) / k
    turns = 0
    if n > 2:
        d = np.diff(coords[list(path)], axis=0)
        turns = int(np.sum(np.any(d[1:] != d[:-1], axis=1)))
    degs = np.array([len(x) for x in nbrs], dtype=float)
    walls = len(g.meta.get("walls", ()) or ())
    kind = kind or g.kind
    f = [
        n / 100.0, math.log(n), k / n, k / 30.0, _spread(coords, cps, n),
        gaps.mean() / n * 10, gaps.max() / n, gaps.std() / n * 10,
        float(np.mean(gaps >= 4)), float(np.mean(gaps >= 8)), float(np.mean(gaps ** 2)) / n,
        amb_any / steps, amb_two / steps, wrong_total / steps, hard_steps / steps, hard_total / n,
        best_run / 10.0, hard_in_gap / steps,
        walls / n, degs.mean() / 4.0, float(np.mean(degs <= 2)), border, turns / steps,
        float(kind in ("grid2d", "grid")), float(kind == "walls"), float(kind == "islands"),
        float(kind not in ("grid2d", "grid", "walls", "islands")),
    ]
    return np.asarray(f, dtype=np.float32)


# ---------------------------------------------------------------------------
# surrogate: a small MLP (trained with torch, run with numpy)
# ---------------------------------------------------------------------------
class Surrogate:
    """Move model: predicts (score change, P(fair)) of a candidate design from its features and the
    change against the current design: input = [f(candidate), f(candidate) - f(current)]."""

    def __init__(self, params: dict):
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}

    def predict(self, F: np.ndarray, F_cur: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(predicted score change, P(fair)) for candidate features F (rows) vs the current design."""
        p = self.p
        F = np.atleast_2d(F).astype(np.float32)
        X = pair_input(F, F if F_cur is None else F_cur)
        X = (X - p["mu"]) / p["sd"]
        h = np.maximum(X @ p["W1"] + p["b1"], 0)
        h = np.maximum(h @ p["W2"] + p["b2"], 0)
        out = h @ p["W3"] + p["b3"]
        delta = out[:, 0] * 100.0
        fair = 1.0 / (1.0 + np.exp(-out[:, 1]))
        return delta, fair

    def save(self, path: str | Path = SURROGATE_PATH, **meta) -> None:
        extra = {f"meta_{k}": np.asarray(v) for k, v in meta.items()}
        np.savez_compressed(path, **self.p, **extra)

    @classmethod
    def load(cls, path: str | Path = SURROGATE_PATH) -> "Surrogate":
        z = np.load(path)
        s = cls({k: z[k] for k in ("mu", "sd", "W1", "b1", "W2", "b2", "W3", "b3")})
        s.meta = {k[5:]: z[k].tolist() for k in z.files if k.startswith("meta_")}
        return s


_SURROGATE: list = []


def pair_input(F: np.ndarray, F_cur: np.ndarray) -> np.ndarray:
    F = np.atleast_2d(F).astype(np.float32)
    return np.concatenate([F, F - np.atleast_2d(F_cur).astype(np.float32)], axis=1)


def load_surrogate(path: str | Path | None = None) -> Surrogate | None:
    """The shipped surrogate (cached), or None when it has not been trained."""
    if path is not None:
        return Surrogate.load(path) if Path(path).exists() else None
    if not _SURROGATE:
        _SURROGATE.append(Surrogate.load(SURROGATE_PATH) if SURROGATE_PATH.exists() else None)
    return _SURROGATE[0]


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def train_surrogate(X: np.ndarray, score: np.ndarray, fair: np.ndarray, groups: np.ndarray | None = None,
                    base: np.ndarray | None = None, epochs: int = 200, hidden: int = 64, seed: int = 0,
                    holdout: float = 0.2, verbose: bool = False) -> tuple[Surrogate, dict]:
    """Fit the MLP; returns (surrogate, report) with held-out Spearman / accuracy.

    X = ``pair_input(f_candidate, f_current)`` rows, score = the candidate's score
    change, fair = candidate fairness, base = the current design's score (only
    used to report the Spearman of the implied absolute score).
    ``groups`` (e.g. the design run each sample came from) makes the split
    honest: samples of one search trajectory are all train or all test.
    """
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    score = np.asarray(score, np.float32)
    fair = np.asarray(fair, np.float32)
    n = len(X)
    if groups is None:
        groups = np.arange(n)
    ug = np.unique(groups)
    test_g = set(rng.choice(ug, size=max(1, int(round(holdout * len(ug)))), replace=False).tolist()) \
        if holdout > 0 and len(ug) > 1 else set()
    te = np.array([g in test_g for g in groups])
    tr = ~te
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd)
    ys = torch.tensor(score / 100.0)
    yf = torch.tensor(fair)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    itr = torch.tensor(np.flatnonzero(tr))
    bs = 256
    for ep in range(epochs):
        perm = itr[torch.randperm(len(itr))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            out = net(Xt[b])
            # score is only meaningful where the design is fair (unfair ones are rejected anyway),
            # but it still carries signal; weight fair samples more
            w = 0.3 + 0.7 * yf[b]
            loss = (w * (out[:, 0] - ys[b]) ** 2).mean() * 20 + \
                torch.nn.functional.binary_cross_entropy_with_logits(out[:, 1], yf[b]) * 0.5
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and ep % 50 == 0:
            print(f"  epoch {ep} loss {loss.item():.4f}", flush=True)
    W = [m for m in net if isinstance(m, torch.nn.Linear)]
    params = {"mu": mu, "sd": sd}
    for i, m in enumerate(W, 1):
        params[f"W{i}"] = m.weight.detach().numpy().T.copy()
        params[f"b{i}"] = m.bias.detach().numpy().copy()
    s = Surrogate(params)
    rep = {"train": int(tr.sum()), "test": int(te.sum()), "features": len(FEATURE_NAMES)}
    if te.any():
        nf = len(FEATURE_NAMES)
        pd, pf = s.predict(X[te, :nf], X[te, :nf] - X[te, nf:])
        fm = fair[te] > 0.5
        rep["spearman_delta"] = round(spearman(pd[fm], score[te][fm]), 3) if fm.sum() > 2 else None
        rep["mae_delta"] = round(float(np.abs(pd - score[te])[fm].mean()), 2) if fm.any() else None
        if base is not None:
            b0 = np.asarray(base, np.float32)[te]
            rep["spearman_score"] = round(spearman((b0 + pd)[fm], (b0 + score[te])[fm]), 3)
        rep["fair_accuracy"] = round(float(((pf > 0.5) == fm).mean()), 3)
        rep["fair_base_rate"] = round(float(fm.mean()), 3)
        rep["fair_auc"] = round(_auc(pf, fm), 3)
    return s, rep


def _auc(p, y) -> float:
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(p)).astype(float) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------
@dataclass
class Eval:
    score: float
    fair: bool
    boring: bool
    spread: float
    J: float
    rating: dict | None


def objective(r: dict, target: str) -> float:
    _, sweet, ceil = TARGETS[target]
    s = r["score"]
    J = min(s, sweet)
    if ceil is not None and s > ceil:
        J -= 2.0 * (s - ceil) + 1.0
    J -= 20.0 * bool(r["boring"])
    J -= 40.0 * max(0.0, MIN_SPREAD - r["features"]["spread"])
    return J


def _est_eval_seconds(n: int) -> float:
    """Rough cost of one exact rating (used to turn a time budget into an evaluation budget)."""
    return 0.0008 + 3.5e-7 * n * n


class _PathSearch:
    """Annealing state for one board/path of the beam."""

    def __init__(self, board: Board, target: str, rng: random.Random, pid: int):
        self.b = board
        self.target = target
        self.rng = rng
        self.pid = pid
        self.n = len(board.path)
        self.idx: frozenset = frozenset()
        self.walls: frozenset = frozenset()
        self.cur: Eval | None = None
        self.best: tuple | None = None           # (J, idx, walls, Eval)
        self.archive: list = []                  # top distinct fair designs
        self.evals = 0
        self.T = 3.0
        self.cur_f = None                        # features of the current design (surrogate / recorder)
        self.track = False

    # ---- helpers
    def puzzle(self, idx, walls) -> Puzzle:
        path = self.b.path
        return Puzzle(self.b.graph(walls), [path[i] for i in sorted(idx)], list(path))

    def evaluate(self, idx, walls, rec: Callable | None = None, feats=None) -> tuple[Eval, np.ndarray | None]:
        self.evals += 1
        p = self.puzzle(idx, walls)
        if feats is None and self.track:
            feats = design_features(p, self.b.path, self.b.kind)
        try:
            r = rate(p, solution=self.b.path)
        except Exception:  # noqa: BLE001
            return Eval(0.0, False, True, 0.0, -1e9, None), feats
        fair = r["features"]["guesses"] == 0
        J = objective(r, self.target) if fair else -1e9
        if rec is not None and self.cur_f is not None:
            rec(feats, self.cur_f, r["score"], self.cur.score, fair)
        return Eval(r["score"], fair, r["boring"], r["features"]["spread"], J, r), feats

    def consider_best(self, e: Eval, idx, walls) -> bool:
        if not e.fair:
            return False
        key = (idx, walls)
        if all((a[1], a[2]) != key for a in self.archive):
            self.archive.append((e.J, idx, walls, e))
            self.archive.sort(key=lambda a: -a[0])
            del self.archive[4:]
        if self.best is None or e.J > self.best[0] + 1e-9:
            self.best = (e.J, idx, walls, e)
            return True
        return False

    # ---- moves
    def propose(self):
        rng, n = self.rng, self.n
        idx, walls = self.idx, self.walls
        inner = sorted(i for i in idx if 0 < i < n - 1)
        wall_kind = bool(self.b.wall_edges)
        r = rng.random()
        if wall_kind and r < 0.18:
            if walls and (rng.random() < 0.4 or len(walls) >= len(self.b.wall_edges)):
                w = rng.choice(sorted(walls))
                return f"remove wall {self.b.wall_edges[w]}", idx, walls - {w}
            free = [i for i in range(len(self.b.wall_edges)) if i not in walls]
            w = rng.choice(free)
            return f"add wall {self.b.wall_edges[w]}", idx, walls | {w}
        r = rng.random()
        if inner and r < 0.45:
            i = rng.choice(inner)
            j = i + rng.choice((-3, -2, -1, 1, 2, 3))
            if 0 < j < n - 1 and j not in idx:
                return f"shift #{i}->{j}", (idx - {i}) | {j}, walls
        elif inner and r < 0.65:
            i = rng.choice(inner)
            return f"remove #{i}", idx - {i}, walls
        elif r < 0.85:
            j = rng.randrange(1, n - 1)
            if j not in idx:
                return f"add #{j}", idx | {j}, walls
        elif inner:
            i = rng.choice(inner)
            j = rng.randrange(1, n - 1)
            if j not in idx:
                return f"relocate #{i}->{j}", (idx - {i}) | {j}, walls
        return None

    def feats(self, idx, walls):
        return design_features(self.puzzle(idx, walls), self.b.path, self.b.kind)

    def pick(self, sur: Surrogate | None, m: int, stats: dict):
        """One move.  Without a surrogate: a random proposal.  With one: m random proposals are
        ranked by their expected improvement P(fair) * softplus(predicted score change) (an
        unfair design is simply rejected, so only the upside counts) and the best is rated."""
        props = []
        for _ in range(m * 3):
            pr = self.propose()
            if pr is not None and (pr[1], pr[2]) != (self.idx, self.walls):
                props.append(pr)
                if sur is None or len(props) >= m:
                    break
        if not props:
            return None, None
        if sur is None or len(props) == 1:
            return props[0], None
        F = np.stack([self.feats(p[1], p[2]) for p in props])
        d, pf = sur.predict(F, self.cur_f)
        stats["surrogate_calls"] += len(props)
        _, sweet, ceil = TARGETS[self.target]
        s = self.cur.score + d
        gain = np.minimum(s, sweet) - (0 if ceil is None else 2.0 * np.maximum(0, s - ceil)) - \
            min(self.cur.score, sweet)
        util = pf * np.logaddexp(0.0, gain / EI_SCALE)
        k = int(np.argmax(util)) if self.rng.random() > EXPLORE else self.rng.randrange(len(props))
        return props[k], F[k]

    # ---- phases
    def init(self, sur, rec, stats, log):
        self.track = sur is not None or rec is not None
        n = self.n
        idx = frozenset(set(range(0, n, 2)) | {n - 1})
        walls = frozenset()
        if self.b.wall_edges:  # start with the generator's density of walls (30 % of unused edges)
            k = int(round(0.3 * len(self.b.wall_edges)))
            walls = frozenset(self.rng.sample(range(len(self.b.wall_edges)), k))
        e, f = self.evaluate(idx, walls, rec)
        if not e.fair:
            idx = frozenset(range(n))
            e, f = self.evaluate(idx, walls, rec)
        self.idx, self.walls, self.cur, self.cur_f = idx, walls, e, f
        self.consider_best(e, idx, walls)
        # classic trick: drop clues while the puzzle stays fair (hence unique)
        order = sorted(i for i in idx if 0 < i < n - 1)
        self.rng.shuffle(order)
        k0 = len(idx)
        for i in order:
            cand = self.idx - {i}
            e2, f2 = self.evaluate(cand, self.walls, rec)
            if e2.fair and not (e2.boring and not self.cur.boring and e2.score < self.cur.score):
                self.idx, self.cur, self.cur_f = cand, e2, f2
                self.consider_best(e2, cand, self.walls)
        log.append({"path": self.pid, "phase": "minimise",
                    "msg": f"path {self.pid}: removed {k0 - len(self.idx)} of {k0} clues while fair -> "
                           f"{len(self.idx)} clues, score {self.cur.score:.1f}"})

    def anneal(self, steps: int, sur, rec, stats, log, deadline: float, T_end: float = 0.3,
               m: int = 6):
        if steps <= 0:
            return
        decay = (T_end / max(self.T, T_end)) ** (1.0 / steps)
        for _ in range(steps):
            if time.perf_counter() > deadline:
                stats["deadline_hit"] = True
                break
            pr, f = self.pick(sur, m, stats)
            if pr is None:
                continue
            desc, idx, walls = pr
            e, f = self.evaluate(idx, walls, rec, f)
            stats["moves"] += 1
            if not e.fair:
                stats["rejected_unfair"] += 1
            elif e.J >= self.cur.J or self.rng.random() < math.exp((e.J - self.cur.J) / self.T):
                self.idx, self.walls, self.cur, self.cur_f = idx, walls, e, f
                stats["accepted"] += 1
                if self.consider_best(e, idx, walls) and len(log) < LOG_CAP:
                    log.append({"path": self.pid, "phase": "anneal", "move": desc,
                                "msg": f"path {self.pid}: {desc} -> new best {e.score:.1f} "
                                       f"({len(idx)} clues{', %d walls' % len(walls) if walls else ''})"})
            self.T = max(T_end, self.T * decay)


@dataclass
class DesignResult:
    puzzle: Puzzle | None
    rating: dict | None
    log: list
    stats: dict
    kind: str
    size: object
    target: str
    seed: int

    @property
    def ok(self) -> bool:
        return self.puzzle is not None

    @property
    def on_target(self) -> bool:
        return self.ok and self.rating["score"] >= TARGETS[self.target][0]

    def to_dict(self, include_solution: bool = True) -> dict:
        d = self.puzzle.to_dict() if self.puzzle is not None else None
        if d is not None and not include_solution:
            d["solution"] = None
        r = None
        if self.rating is not None:
            f = self.rating["features"]
            r = {"label": self.rating["label"], "score": self.rating["score"], "boring": self.rating["boring"],
                 "guesses": f["guesses"], "max_lookahead": f["max_lookahead"], "spread": f["spread"],
                 "checkpoints": f["checkpoints"], "counts": f["counts"]}
        return {"kind": self.kind, "size": self.size, "target": self.target, "seed": self.seed,
                "ok": self.ok, "on_target": self.on_target, "puzzle": d, "rating": r,
                "log": self.log, "stats": self.stats}


def design(kind: str = "grid2d", size=8, target: str = "expert", time_budget: float = 30.0, seed: int = 0,
           surrogate: Surrogate | None | str = "auto", paths: int | None = None, evals: int | None = None,
           verify_time: float | None = None, recorder: Callable | None = None, proposals: int = 6) -> DesignResult:
    """Design a unique, fair puzzle aimed at ``target`` ("hard" | "expert" | "insane").

    kind: "classic"/"grid2d", "walls", "islands" or any ``generator.make_puzzle`` kind;
    size as for make_puzzle (islands: number of islands).  ``surrogate``: "auto"
    (the shipped model if present), None (plain annealing) or a Surrogate.
    ``evals`` overrides the evaluation budget derived from ``time_budget``;
    ``recorder(f_candidate, f_current, score, current_score, fair)`` sees every exact
    evaluation (training data for the surrogate).
    """
    from .solver import count_solutions
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}")
    t0 = time.perf_counter()
    deadline = t0 + float(time_budget)
    kind = KIND_ALIASES.get(kind, kind)
    sur = load_surrogate() if surrogate == "auto" else surrogate
    rng = random.Random(int(seed) * 7919 + 17)
    nrng = np.random.default_rng(int(seed))
    stats = {"evaluations": 0, "surrogate_calls": 0, "moves": 0, "accepted": 0, "rejected_unfair": 0,
             "paths": 0, "deadline_hit": False, "surrogate": sur is not None}
    log: list = []
    # boards: a beam of random paths
    probe = make_board(kind, size, nrng, time_limit=max(2.0, 0.2 * time_budget))
    n = len(probe.path)
    per_eval = _est_eval_seconds(n) + (proposals * 0.9e-5 * n if sur is not None else 0.0)
    budget = int(evals if evals is not None else max(60, 0.55 * time_budget / per_eval))
    B = paths if paths is not None else (2 if budget < 400 else 4 if budget < 2500 else 6)
    boards = [probe]
    for _ in range(B - 1):
        try:
            boards.append(make_board(kind, size, nrng, time_limit=max(2.0, 0.1 * time_budget)))
        except ValueError:
            break
    B = len(boards)
    stats["paths"] = B
    log.append({"phase": "start", "msg": f"designing a {target} {kind} puzzle ({n} cells): {B} candidate "
                                         f"paths, budget {budget} exact ratings"
                                         f"{', surrogate-guided' if sur else ''}"})
    searches = [_PathSearch(b, target, random.Random(rng.random()), i) for i, b in enumerate(boards)]
    rounds = max(1, math.ceil(math.log2(B)) + 1)
    per_round = budget / rounds
    alive = list(searches)
    for s in alive:
        s.init(sur, recorder, stats, log)
    for rd in range(rounds):
        if time.perf_counter() > deadline:
            stats["deadline_hit"] = True
            break
        spent = sum(s.evals for s in searches)
        left = budget - spent
        if left <= 0:
            break
        share = int(min(per_round, left) / len(alive)) if rd < rounds - 1 else int(left / len(alive))
        for s in alive:
            s.anneal(share, sur, recorder, stats, log, deadline, m=proposals)
        alive.sort(key=lambda s: -(s.best[0] if s.best else -1e9))
        if len(alive) > 1 and rd < rounds - 1:
            keep = max(1, len(alive) // 2)
            dropped = alive[keep:]
            alive = alive[:keep]
            log.append({"phase": "prune", "msg": "kept paths " + ", ".join(
                f"{s.pid} ({s.best[3].score:.1f})" for s in alive if s.best) + "; dropped " + ", ".join(
                f"{s.pid}" for s in dropped)})
        _, sweet, _ = TARGETS[target]
        if alive[0].best and alive[0].best[0] >= sweet - 1e-9 and rd >= 1:
            log.append({"phase": "stop", "msg": f"reached the {target} sweet spot"})
            break
    stats["evaluations"] = sum(s.evals for s in searches)
    # certify with the exact solver, best first
    cands = sorted((a for s in searches for a in s.archive), key=lambda a: -a[0])
    vt = verify_time if verify_time is not None else max(3.0, min(60.0, 0.3 * time_budget))
    best_p = best_r = None
    for J, idx, walls, e in cands[:6]:
        s = next(x for x in searches if (J, idx, walls, e) in x.archive)
        p = s.puzzle(idx, walls)
        cnt = count_solutions(p, 2, time_limit=vt)
        if cnt == (1, "complete"):
            best_p, best_r = p, e.rating
            log.append({"phase": "verify", "msg": f"exact solver certifies a unique solution "
                                                  f"(path {s.pid}, score {e.score:.1f}, {len(idx)} clues)"})
            break
        log.append({"phase": "verify", "msg": f"certificate failed ({cnt[1]}): trying the next design"})
    stats["seconds"] = round(time.perf_counter() - t0, 2)
    if best_r is not None:
        stats["score"] = best_r["score"]
        log.append({"phase": "done", "msg": f"{best_r['label']} ({best_r['score']:.1f}), "
                                            f"{best_r['features']['checkpoints']} numbers, no guessing needed"})
    return DesignResult(best_p, best_r, log, stats, kind, size, target, int(seed))


# ---------------------------------------------------------------------------
# training data + CLI
# ---------------------------------------------------------------------------
TRAIN_CONFIGS = [("grid2d", 7, "expert"), ("grid2d", 8, "expert"), ("grid2d", 9, "insane"),
                 ("grid2d", 10, "insane"), ("walls", 8, "expert"), ("walls", 9, "insane"),
                 ("islands", 5, "expert"), ("islands", 6, "insane"), ("islands", 7, "insane")]


def _collect_job(args):
    kind, size, target, seed, evals = args
    X, D, F, B = [], [], [], []

    def rec(f, f_cur, score, cur_score, fair):
        X.append(pair_input(f, f_cur)[0])
        D.append(score - cur_score)
        F.append(fair)
        B.append(cur_score)
    try:
        design(kind, size, target, time_budget=600, seed=seed, surrogate=None, paths=2, evals=evals,
               verify_time=0.01, recorder=rec)
    except ValueError:
        pass
    return (np.array(X, np.float32).reshape(-1, 2 * len(FEATURE_NAMES)), np.array(D, np.float32),
            np.array(F, np.float32), np.array(B, np.float32))


def collect_samples(runs_per_config: int = 6, evals: int = 400, configs=None, workers: int | None = None,
                    seed: int = 1000, max_per_run: int = 300):
    """Rated moves from surrogate-free search runs: (X, score change, fair, group, current score)."""
    from multiprocessing import Pool
    configs = configs or TRAIN_CONFIGS
    jobs = [(k, s, t, seed + 101 * i + 7 * j, evals) for j, (k, s, t) in enumerate(configs)
            for i in range(runs_per_config)]
    rng = np.random.default_rng(seed)
    out = [[], [], [], [], []]
    pool = None
    if workers == 1:
        res = map(_collect_job, jobs)
    else:
        pool = Pool(workers)
        res = pool.imap(_collect_job, jobs)
    for g, (X, D, F, B) in enumerate(res):
        if len(X) > max_per_run:
            keep = np.sort(rng.choice(len(X), max_per_run, replace=False))
            X, D, F, B = X[keep], D[keep], F[keep], B[keep]
        for lst, v in zip(out, (X, D, F, np.full(len(X), g), B)):
            lst.append(v)
    if pool is not None:
        pool.close()
    return tuple(np.concatenate(v) for v in out)


def _main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="The Architect: design hard, fair Zip puzzles")
    ap.add_argument("--train", action="store_true", help="collect samples and train the surrogate")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--kind", default="grid2d")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--target", default="expert")
    ap.add_argument("--budget", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.train:
        t0 = time.perf_counter()
        X, D, F, G, B = collect_samples(a.runs, a.evals, workers=a.workers)
        print(f"{len(X)} samples ({F.mean():.0%} fair) from {len(np.unique(G))} runs "
              f"in {time.perf_counter() - t0:.0f}s", flush=True)
        np.savez_compressed(Path.home() / ".cache" / "architect_samples.npz", X=X, D=D, F=F, G=G, B=B)
        sur, rep = train_surrogate(X, D, F, G, base=B, verbose=True)
        print(json.dumps(rep), flush=True)
        sur2, _ = train_surrogate(X, D, F, G, holdout=0.0)
        sur2.save(SURROGATE_PATH, **{k: v for k, v in rep.items() if v is not None}, samples=len(X))
        print(f"saved {SURROGATE_PATH}")
        return
    r = design(a.kind, a.size, a.target, a.budget, a.seed)
    for e in r.log:
        print(" ", e["msg"])
    print(json.dumps({k: v for k, v in r.to_dict().items() if k not in ("puzzle", "log")}))


if __name__ == "__main__":
    _main()
