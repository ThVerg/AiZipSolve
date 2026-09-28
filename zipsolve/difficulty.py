"""Human difficulty rating for Zip puzzles.

The exact solver (``zipsolve.solver``) says nothing about how hard a puzzle
feels: it happily brute-forces a 9x9 that a person finds trivial, and a small
board can need a nasty "what if" for a human.  This module rates a puzzle by
replaying its (unique) solution with a *human-like deduction solver* and
measuring, at every step, the cheapest reasoning that proves the next move.

Model of the player
-------------------
The player extends the line from its head (as in the game).  At each step the
rule-legal moves are the unvisited neighbours of the head that are not a later
checkpoint (and the last checkpoint only as the very last cell), not against a
one-way arrow and not a door whose key is still missing.  All techniques are
sound on every board: the region / parity / articulation rules reason on the
undirected graph (a relaxation), parity switches itself off on non-bipartite
boards (hex, torus with odd sides, cube surface, portals), and L1 also knows
the arrow rule "a cell with no usable way in / out" and "a cell only enterable
from the head must be next".  Every
*wrong* legal move must be ruled out; the move's cost is the cheapest of the
following techniques, tried in order of how obvious they are to a person:

====  ===========================================================  =====
tier  technique                                                     cost
====  ===========================================================  =====
L1    local dead end: after the move some cell next to the old or    0.25
      new head is left with fewer than 2 free neighbours (corner /
      pocket), the head has nowhere to go, or two cells both need
      the head next (the classic "a cell with 2 free neighbours
      must use both" / "don't strand the corner")
L2    region reasoning: the free cells split into two regions, or    1.0
      the next number can no longer be reached without crossing a
      later number (checkpoint-order reachability)
LA1   one-line lookahead: "if I go there, then the next moves are    1.5 + 0.1 per
      all forced (by L1/L2) ... and I get stuck"                     forced step (cap 4)
L3    expert global rules: colour parity (chessboard counting),      4.0
      articulation points / bridges that must be crossed now,
      forced-edge chains (the solver's deep checks)
LA2   two-level lookahead: the forced line reaches a fork and        6.0
      *every* branch then dies by LA1
G     none of the above: the player has to guess (or backtrack)      10.0
====  ===========================================================  =====

A step's cost is the most expensive wrong move plus a quarter of the others
(ruling out three moves is more work than one).  A step with a single legal
move costs nothing ("forced by the rules").  The puzzle's *effort* is the sum
of step costs times a dimension factor (x1.2 for 3D, x1.5 for 4D: a human has
to picture the layers).  The score is ``100 * (1 - exp(-effort / 66))``, a
saturating curve: effort 3 -> 4, 10 -> 14, 20 -> 26, 40 -> 45, 70 -> 65,
150 -> 90.

Calibration
-----------
The constants were tuned on random unique puzzles with the generator's default
checkpoint counts; ``python -m zipsolve.difficulty --calibrate --count 30``
prints the distribution per mode/size.  Result used for the thresholds
(30 puzzles each; score quartiles):

    kind     size   min   p25   med   p75   max   labels
    grid2d      5   3.5   8.9  11.2  17.5  31.8   Easy 25, Medium 5
    grid2d      6  10.5  20.5  25.6  35.2  56.9   Easy 10, Medium 17, Hard 3
    grid2d      7  15.2  29.6  42.2  55.5  73.2   Ea 3, Me 12, Ha 12, Ex 3
    grid2d      8  33.9  49.8  54.2  59.6  80.8   Me 3, Ha 21, Ex 5, In 1
    walls       6   6.4  10.1  13.8  19.1  29.0   Easy 27, Medium 3
    walls       7  13.8  19.1  30.6  39.1  51.7   Ea 10, Me 15, Ha 5
    walls       8  28.1  39.9  47.1  53.8  70.0   Me 9, Ha 18, Ex 3
    islands     3   3.0   8.0  12.6  19.9  45.2   Easy 27, Medium 2, Hard 1
    islands     4   6.9  14.2  19.4  30.9  52.6   Ea 19, Me 7, Ha 4
    islands     5   5.3  19.8  29.4  43.3  63.5   Ea 8, Me 14, Ha 7, Ex 1
    grid3d      3  24.6  44.5  51.2  58.9  78.5   Me 7, Ha 17, Ex 6
    grid3d      4  79.3  89.0  93.6  97.1  98.7   Ex 1, In 29

Label thresholds:

    Easy < 22 <= Medium < 42 <= Hard < 62 <= Expert < 80 <= Insane

so a LinkedIn-style 5x5/6x6 lands on Easy, 6x6/7x7 on Medium, boards that
need a few one-line lookaheads or a two-level lookahead are Hard, several
guesses make Expert, and a 4x4x4 cube is Insane.  Fewer checkpoints (the
generator adds only as many as uniqueness needs) push a board up the scale.

Boring puzzles
--------------
``rate(...)["boring"]`` is True when the puzzle offers nothing to think about:
more than 95 % of the moves are forced by the rules or by local dead ends
(L0/L1) *and* no lookahead is ever needed, or the numbers are so dense
(more than 45 % of the cells) that at most one-line lookaheads are needed.
``spread`` is the Clark-Evans nearest-neighbour ratio of the checkpoints
(1 = random, < 0.6 clumped, > 1 evenly spread), used by the bank builder to
prefer puzzles whose numbers cover the board.

Public API
----------
``rate(puzzle, solution=None, time_limit=None) -> dict`` with keys
``score`` (0-100), ``label``, ``boring`` and ``features``.
``label_for(score) -> str``.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Sequence

import numpy as np

from .puzzle import Puzzle
from .solver import _Search, solve

__all__ = ["rate", "label_for", "LABELS", "THRESHOLDS", "COSTS"]

LABELS = ("Easy", "Medium", "Hard", "Expert", "Insane")
THRESHOLDS = (22.0, 42.0, 62.0, 80.0)     # score cut points between consecutive labels
COSTS = {"L1": 0.25, "L2": 1.0, "LA1": 1.5, "L3": 4.0, "LA2": 6.0, "G": 10.0}
LA1_PER_STEP, LA1_CAP = 0.1, 4.0
SCALE = 66.0
DIM_FACTOR = {1: 1.0, 2: 1.0, 3: 1.2, 4: 1.5}
MAX_CHAIN = 40            # longest forced line a player follows in their head
NODE_BUDGET = 2500        # pushes per refutation attempt (keeps rating fast)


def label_for(score: float) -> str:
    for lab, t in zip(LABELS, THRESHOLDS):
        if score < t:
            return lab
    return LABELS[-1]


class _Budget(Exception):
    pass


class _Human:
    """Deduction engine over the solver's search state (push/pop, degrees)."""

    def __init__(self, puzzle: Puzzle):
        self.s = _Search(puzzle.graph, puzzle.checkpoints, deep=True)
        self.n = puzzle.num_nodes
        self.pushes = 0
        self.budget = NODE_BUDGET

    # ---- rules ---------------------------------------------------------------
    def legal(self) -> list[int]:
        s = self.s
        h = s.path[-1]
        out = []
        for w in s.nbrs[h]:
            if s.visited[w]:
                continue
            c = s.cpi[w]
            if c >= 0 and c != s.nxt:
                continue
            if w == s.end and s.remaining > 1:
                continue
            if s.constrained and not s.ok_step(h, w):   # one-way arrows, keys before doors
                continue
            out.append(w)
        return out

    def push(self, w: int) -> None:
        self.pushes += 1
        if self.pushes > self.budget:
            raise _Budget
        self.s.push(w)

    def pop(self) -> None:
        self.s.pop()

    def local_dead(self, p: int) -> bool:
        """L1 after the move p -> head: pockets / corners / double forcing."""
        s = self.s
        h = s.path[-1]
        vis, deg, nbrs, end = s.visited, s.deg, s.nbrs, s.end
        forced = -1
        any_free = False
        hn = nbrs[h]
        for x in hn:
            if vis[x]:
                continue
            any_free = True
            d = deg[x]
            if x == end:
                if d == 0 and s.remaining > 1:
                    return True
            elif d <= 1:
                if d == 0 or forced >= 0:
                    return True
                forced = x
        if not any_free:
            return True
        if s.blocked is not None:   # arrows: a neighbour only enterable from the head, or with no way in/out
            for grp in (hn, nbrs[p]):
                for x in grp:
                    if vis[x]:
                        continue
                    st = s.dir_status(x, h)
                    if st == -2:
                        return True
                    if st == 1:
                        if forced >= 0 and forced != x:
                            return True
                        forced = x
        if forced >= 0:
            c = s.cpi[forced]
            if c >= 0 and c != s.nxt:
                return True
            if s.constrained and not s.ok_step(h, forced):
                return True
        for x in nbrs[p]:
            if vis[x] or x in hn:
                continue
            if deg[x] < (1 if x == end else 2):
                return True
        return False

    def region_dead(self) -> bool:
        """L2: free cells connected, next checkpoint reachable in order."""
        s = self.s
        vis, nbrs, cpi = s.visited, s.nbrs, s.cpi
        h = s.path[-1]
        free = [x for x in nbrs[h] if not vis[x]]
        if not free:
            return True
        start = free[0]
        seen = {start}
        dq = [start]
        while dq:
            v = dq.pop()
            for w in nbrs[v]:
                if not vis[w] and w not in seen:
                    seen.add(w)
                    dq.append(w)
        if len(seen) != s.remaining:
            return True
        # the next checkpoint must be reachable from the head without crossing a later one
        nxt = s.nxt
        if nxt >= len(s.cps):
            return False
        target = s.cps[nxt]
        if target in nbrs[h]:
            return False
        seen = {target}
        dq = deque([target])
        hn = set(nbrs[h])
        while dq:
            v = dq.popleft()
            for w in nbrs[v]:
                if vis[w] or w in seen or cpi[w] > nxt:
                    continue
                if w in hn:
                    return False
                seen.add(w)
                dq.append(w)
        return True

    def expert_dead(self) -> bool:
        """L3: the solver's full sound checks (parity, articulation points, chains)."""
        s = self.s
        ok = s.global_check()
        return not ok

    def static_level(self, p: int, upto: int = 2) -> int | None:
        """Cheapest static rule (1, 2 or 3) proving the current state dead, else None."""
        if self.s.remaining == 0:
            return None
        if self.local_dead(p):
            return 1
        if upto >= 2 and self.region_dead():
            return 2
        if upto >= 3 and self.expert_dead():
            return 3
        return None

    def viable(self) -> list[int]:
        """Legal moves not refuted by L1/L2 (what a player sees as 'possible')."""
        s = self.s
        h = s.path[-1]
        out = []
        for w in self.legal():
            self.push(w)
            try:
                if s.remaining == 0 or self.static_level(h, 2) is None:
                    out.append(w)
            finally:
                self.pop()
        return out

    def refute(self, depth: int) -> tuple[bool, int]:
        """Follow forced moves from the current state; True if it dies.

        depth 1: a single forced line; depth 2: at a fork every branch must die
        by a depth-1 line.  Returns (refuted, forced steps followed).
        """
        pushed = 0
        try:
            while True:
                if self.s.remaining == 0:
                    return False, pushed
                moves = self.viable()
                if not moves:
                    return True, pushed
                if len(moves) == 1:
                    if pushed >= MAX_CHAIN:
                        return False, pushed
                    self.push(moves[0])
                    pushed += 1
                    continue
                if depth <= 1:
                    return False, pushed
                for m in moves:
                    self.push(m)
                    try:
                        ok, _ = self.refute(depth - 1)
                    finally:
                        self.pop()
                    if not ok:
                        return False, pushed
                return True, pushed
        finally:
            for _ in range(pushed):
                self.pop()

    def refute_move(self, w: int) -> tuple[str, float]:
        """Cheapest technique ruling out the (wrong) move w from the current head."""
        s = self.s
        h = s.path[-1]
        s.push(w)
        try:
            lvl = self.static_level(h, 2)
            if lvl == 1:
                return "L1", COSTS["L1"]
            if lvl == 2:
                return "L2", COSTS["L2"]
            self.pushes, self.budget = 0, NODE_BUDGET
            try:
                ok, steps = self.refute(1)
            except _Budget:
                ok, steps = False, 0
            if ok:
                return "LA1", COSTS["LA1"] + min(LA1_CAP - COSTS["LA1"], LA1_PER_STEP * steps)
            if self.expert_dead():
                return "L3", COSTS["L3"]
            self.pushes = 0
            try:
                ok, _ = self.refute(2)
            except _Budget:
                ok = False
            if ok:
                return "LA2", COSTS["LA2"]
            return "G", COSTS["G"]
        finally:
            s.pop()


def _spread(puzzle: Puzzle) -> float:
    """Clark-Evans ratio of the checkpoint positions (see module doc)."""
    cps = puzzle.checkpoints
    X = puzzle.graph.coords
    k = len(cps)
    if k < 3:
        return 1.0
    P = X[cps]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1).mean()
    dim = X.shape[1]
    vol = puzzle.num_nodes   # unit cells
    expected = 0.5 * math.sqrt(vol / k) if dim == 2 else 0.554 * (vol / k) ** (1 / 3) if dim == 3 \
        else 0.6 * (vol / k) ** (1 / dim)
    return float(nn / max(expected, 1e-9))


def rate(puzzle: Puzzle, solution: Sequence[int] | None = None,
         time_limit: float | None = None) -> dict:
    """Human difficulty of ``puzzle`` (deterministic).

    ``solution``: the (unique) solution; defaults to ``puzzle.solution`` or is
    found by the exact solver.  For puzzles with several solutions the rating
    follows the given one (other solutions only make it easier).
    ``time_limit``: optional soft cap in seconds; when it is hit the remaining
    steps are extrapolated from the average so far (``features["partial"]``).
    """
    t0 = time.perf_counter()
    sol = list(solution) if solution is not None else (list(puzzle.solution) if puzzle.solution else None)
    if sol is None or not puzzle.is_valid_solution(sol):
        r = solve(puzzle, time_limit=60)
        if r.path is None:
            raise ValueError("puzzle has no solution")
        sol = r.path
    hm = _Human(puzzle)
    s = hm.s
    s.push(sol[0])
    n = puzzle.num_nodes
    counts = {k: 0 for k in ("rules", "L1", "L2", "LA1", "L3", "LA2", "G")}
    step_costs = []
    branching = 0
    max_depth = 0
    partial = False
    depth_of = {"L1": 0, "L2": 0, "L3": 0, "LA1": 1, "LA2": 2, "G": 3}
    for i in range(n - 1):
        legal = hm.legal()
        right = sol[i + 1]
        wrong = [w for w in legal if w != right]
        if not wrong:
            counts["rules"] += 1
            step_costs.append(0.0)
        else:
            res = [hm.refute_move(w) for w in wrong]
            costs = sorted((c for _, c in res), reverse=True)
            cost = costs[0] + 0.25 * sum(costs[1:])
            worst = max(res, key=lambda t: t[1])[0]
            counts[worst] += 1
            d = max(depth_of[k] for k, _ in res)
            max_depth = max(max_depth, d)
            if d >= 1:
                branching += 1
            step_costs.append(cost)
        s.push(right)
        if time_limit is not None and time.perf_counter() - t0 > time_limit and i < n - 2:
            partial = True
            break
    steps = len(step_costs)
    effort = float(sum(step_costs))
    if partial and steps:
        effort *= (n - 1) / steps
    dim = puzzle.graph.dim
    effort *= DIM_FACTOR.get(dim, 1.7)
    score = 100.0 * (1.0 - math.exp(-effort / SCALE))
    moves = max(1, steps)
    forced = (counts["rules"] + counts["L1"]) / moves
    density = len(puzzle.checkpoints) / n
    boring = bool((forced > 0.95 and max_depth == 0) or (density > 0.45 and max_depth <= 1))
    feats = {
        "nodes": n, "dim": dim, "checkpoints": len(puzzle.checkpoints),
        "cp_density": round(density, 3),
        "forced_frac": round(forced, 3),
        "rules_frac": round(counts["rules"] / moves, 3),
        "deduction_frac": round((counts["rules"] + counts["L1"] + counts["L2"] + counts["L3"]) / moves, 3),
        "max_lookahead": max_depth,        # 0 none, 1 one line, 2 two-level, 3 guess
        "branching_points": branching,
        "guesses": counts["G"],
        "counts": counts,
        "effort": round(effort, 2),
        "spread": round(_spread(puzzle), 3),
        "partial": partial,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    score = round(score, 1)
    return {"score": score, "label": label_for(score), "boring": boring, "features": feats}


# ---------------------------------------------------------------------------
# calibration CLI:  python -m zipsolve.difficulty --calibrate [--count 12]
# ---------------------------------------------------------------------------
def _cal_job(args):
    kind, size, seed, ncp = args
    from .generator import make_puzzle
    try:
        p = make_puzzle(kind, size, num_checkpoints=ncp, rng=seed, unique=True, time_limit=30)
    except ValueError:
        return None
    r = rate(p)
    return kind, size, ncp, r["score"], r["label"], r["boring"], r["features"]["seconds"]


def _calibrate(count: int = 12) -> None:
    from multiprocessing import Pool
    cfg = [("grid2d", 5), ("grid2d", 6), ("grid2d", 7), ("grid2d", 8), ("walls", 6), ("walls", 7),
           ("walls", 8), ("islands", 3), ("islands", 4), ("islands", 5), ("grid3d", 3), ("grid3d", 4)]
    jobs = [(k, s, 1000 + i, None) for k, s in cfg for i in range(count)]
    with Pool() as pool:
        rows = [r for r in pool.imap_unordered(_cal_job, jobs) if r]
    print(f"{'kind':8} {'size':>4} {'n':>3} {'min':>5} {'p25':>5} {'med':>5} {'p75':>5} {'max':>5}  labels (Ea/Me/Ha/Ex/In)  boring  sec")
    for k, s in cfg:
        sc = sorted(r[3] for r in rows if r[0] == k and r[1] == s)
        if not sc:
            continue
        labs = {}
        for r in rows:
            if r[0] == k and r[1] == s:
                labs[r[4][:2]] = labs.get(r[4][:2], 0) + 1
        bor = sum(1 for r in rows if r[0] == k and r[1] == s and r[5])
        sec = np.mean([r[6] for r in rows if r[0] == k and r[1] == s])
        q = np.percentile(sc, [0, 25, 50, 75, 100])
        print(f"{k:8} {s:>4} {len(sc):>3} " + " ".join(f"{x:5.1f}" for x in q)
              + "  " + "".join(f"{a}{b}" for a, b in sorted(labs.items())) + f"  {bor:>3}  {sec:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--count", type=int, default=12)
    a = ap.parse_args()
    if a.calibrate:
        _calibrate(a.count)
