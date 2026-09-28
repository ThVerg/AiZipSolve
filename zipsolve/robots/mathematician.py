"""🧮 Mathematician: the puzzle as a Boolean satisfiability problem.

Positional encoding: variable ``x[v, t]`` = "cell v is the t-th cell of the
line" (t = 0 .. n-1), created only where it is possible at all (distance from
number 1 and to the last number, chessboard colour parity on bipartite
graphs, the number order). Clauses:

* every step holds exactly one cell and every cell is used exactly once
  (sequential-counter at-most-one + at-least-one);
* ``x[v, t] -> OR x[u, t+1]`` over the cells u the line may step to from v
  (one-way arcs respected) and the mirrored predecessor clause;
* number 1 at step 0, the last number at step n-1;
* order: auxiliary ``r[a, t]`` = "a was visited at step <= t"; for each pair
  (a before b) - consecutive numbers and key/door precedence - ``x[b, t] -> r[a, t-1]``.

Solved with CaDiCaL (``python-sat``; optional dependency ``[robots]``). The
model is decoded and verified with :meth:`Puzzle.check_solution`; if a rule the
encoder does not know rejects it, that exact line is blocked and the solver asked
again. Without ``python-sat`` the robot reports ``"unavailable"``.

Trace: captions ("Encoded 2,304 variables and 18,112 clauses", "Solver found a
model in 0.21 s") and then the whole line at once (``{"t": "path"}``).
"""
from __future__ import annotations

import threading
import time
from collections import deque

from ..puzzle import Puzzle
from .base import Tracer, result, unavailable, unsupported_reason

ROBOT = "sat"


def available() -> bool:
    try:
        import pysat.solvers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _arcs(puzzle: Puzzle) -> list[tuple[int, int]]:
    try:
        from ..graph import arcs_of
        got = arcs_of(puzzle.graph)
        if got:
            return got
    except ImportError:
        pass
    for src in (getattr(puzzle, "arcs", None), (puzzle.graph.meta or {}).get("arcs"),
                (puzzle.graph.meta or {}).get("oneway")):
        if src:
            try:
                return [(int(a), int(b)) for a, b in src]
            except (TypeError, ValueError):
                continue
    return []


def _precedence(puzzle: Puzzle) -> list[tuple[int, int]]:
    try:
        from ..graph import precedence_of
        got = precedence_of(puzzle.graph)
        if got:
            return got
    except ImportError:
        pass
    meta = puzzle.graph.meta or {}
    for src in (getattr(puzzle, "precedence", None), meta.get("precedence")):
        if src:
            try:
                return [(int(a), int(b)) for a, b in src]
            except (TypeError, ValueError):
                continue
    keys = meta.get("keys")
    if keys:
        try:
            return [(int(k["key"]), int(k["door"])) for k in keys]
        except (TypeError, KeyError, ValueError):
            pass
    return []


def _bfs(nbrs, src: int, n: int) -> list[int]:
    d = [n] * n
    d[src] = 0
    q = deque([src])
    while q:
        v = q.popleft()
        for w in nbrs[v]:
            if d[w] > d[v] + 1:
                d[w] = d[v] + 1
                q.append(w)
    return d


def _colours(nbrs, n: int):
    col = [-1] * n
    for s in range(n):
        if col[s] >= 0:
            continue
        col[s] = 0
        st = [s]
        while st:
            v = st.pop()
            for w in nbrs[v]:
                if col[w] < 0:
                    col[w] = 1 - col[v]
                    st.append(w)
                elif col[w] == col[v]:
                    return None
    return col


class Encoding:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        g = puzzle.graph
        n = self.n = g.num_nodes
        nbrs = g.neighbors
        cps = puzzle.checkpoints
        s, e = cps[0], cps[-1]
        forbidden = set(_arcs(puzzle))   # arc (a, b): only a -> b; so b -> a is forbidden
        self.succ = [[u for u in nbrs[v] if (u, v) not in forbidden] for v in range(n)]
        self.pred = [[u for u in nbrs[v] if (v, u) not in forbidden] for v in range(n)]
        ds, de = _bfs(nbrs, s, n), _bfs(nbrs, e, n)
        col = _colours(nbrs, n)
        cpi = {c: i for i, c in enumerate(cps)}
        k = len(cps)
        self.nv = 0
        self.x: dict[tuple[int, int], int] = {}
        self.clauses: list[list[int]] = []
        self.dom: list[list[int]] = [[] for _ in range(n)]    # allowed t per v
        for v in range(n):
            for t in range(n):
                if v == s and t != 0 or v == e and t != n - 1:
                    continue
                if t == 0 and v != s or t == n - 1 and v != e:
                    continue
                if t < ds[v] or n - 1 - t < de[v]:
                    continue
                if col is not None and col[v] != col[s] ^ (t & 1):
                    continue
                i = cpi.get(v)
                if i is not None and (t < i or n - 1 - t < k - 1 - i):
                    continue
                self.nv += 1
                self.x[v, t] = self.nv
                self.dom[v].append(t)
        self.at: list[list[int]] = [[] for _ in range(n)]    # cells allowed at t
        for (v, t) in self.x:
            self.at[t].append(v)
        self.ok = all(self.dom[v] for v in range(n)) and all(self.at[t] for t in range(n))
        if not self.ok:
            return
        from pysat.card import CardEnc, EncType
        for v in range(n):
            lits = [self.x[v, t] for t in self.dom[v]]
            self._exactly_one(lits, CardEnc, EncType)
        for t in range(n):
            lits = [self.x[v, t] for v in self.at[t]]
            self._exactly_one(lits, CardEnc, EncType)
        X = self.x
        for (v, t), lit in X.items():
            if t < n - 1:
                self.clauses.append([-lit] + [X[u, t + 1] for u in self.succ[v] if (u, t + 1) in X])
            if t > 0:
                self.clauses.append([-lit] + [X[u, t - 1] for u in self.pred[v] if (u, t - 1) in X])
        pairs = [(cps[i], cps[i + 1]) for i in range(k - 1)] + _precedence(puzzle)
        self.r: dict[int, dict[int, int]] = {}
        for a, b in pairs:
            ra = self._reached(a)
            for t in self.dom[b]:
                if t == 0:
                    self.clauses.append([-X[b, t]])
                else:
                    self.clauses.append([-X[b, t], ra[t - 1]])
        self.pairs = pairs

    def _new(self) -> int:
        self.nv += 1
        return self.nv

    def _exactly_one(self, lits, CardEnc, EncType) -> None:
        self.clauses.append(list(lits))
        if len(lits) > 1:
            enc = CardEnc.atmost(lits=lits, bound=1, top_id=self.nv, encoding=EncType.seqcounter)
            self.nv = max(self.nv, enc.nv)
            self.clauses.extend(enc.clauses)

    def _reached(self, a: int) -> dict[int, int]:
        """r[t] <-> a is at some step <= t (t = 0 .. n-2)."""
        if a in self.r:
            return self.r[a]
        X = self.x
        r: dict[int, int] = {}
        prev = None
        for t in range(self.n - 1):
            lit = self._new()
            r[t] = lit
            here = X.get((a, t))
            # lit <-> prev OR here
            parts = [p for p in (prev, here) if p is not None]
            self.clauses.append([-lit] + parts)
            for p in parts:
                self.clauses.append([-p, lit])
            prev = lit
        self.r[a] = r
        return r

    def decode(self, model) -> list[int]:
        pos = set(v for v in model if v > 0)
        order = [None] * self.n
        for (v, t), lit in self.x.items():
            if lit in pos:
                order[t] = v
        return [int(v) for v in order]  # type: ignore[arg-type]


def run(puzzle: Puzzle, time_limit: float = 10.0, trace: bool = True, seed: int = 0) -> dict:
    bad = unsupported_reason(puzzle)
    if bad:
        return unavailable(ROBOT, puzzle, bad, "unsupported", trace)
    if not available():
        return unavailable(ROBOT, puzzle, "The Mathematician needs the optional 'python-sat' package "
                           "(pip install 'zipsolve[robots]').", "unavailable", trace)
    from pysat.solvers import Solver
    t0 = time.perf_counter()
    tr = Tracer(trace)
    start = puzzle.checkpoints[0]
    tr.restart([start])
    n = puzzle.num_nodes
    tr.note(f"Let x(cell, step) mean 'this cell is step number <step> of the line'. "
            f"{n} cells x {n} steps, minus everything that's impossible at a glance.", kind="intro")
    enc = Encoding(puzzle)
    t_enc = time.perf_counter() - t0
    if not enc.ok:
        tr.note("Some cell can't be placed at any step (distances / colours / order): no solution.",
                kind="stuck")
        return result(ROBOT, puzzle, [start], "unsat", t0, tr, None, 0,
                      stats={"variables": enc.nv, "clauses": len(enc.clauses), "encode_seconds": round(t_enc, 3)})
    nclauses = len(enc.clauses)
    tr.note(f"Encoded {enc.nv:,} variables and {nclauses:,} clauses in {t_enc:.2f} s "
            f"({len(enc.x):,} position variables survive the pruning).", kind="encode",
            variables=enc.nv, clauses=nclauses)
    status, path = "timeout", None
    tries = 0
    with Solver(name="cadical153", bootstrap_with=enc.clauses) as sv:
        while True:
            left = time_limit - (time.perf_counter() - t0)
            if left <= 0:
                break
            timer = threading.Timer(left, sv.interrupt)
            timer.start()
            ts = time.perf_counter()
            try:
                ok = sv.solve_limited(expect_interrupt=True)
            finally:
                timer.cancel()
            dt = time.perf_counter() - ts
            tries += 1
            if ok is None:
                tr.note(f"The solver gave up after {dt:.2f} s (time limit).", kind="stuck")
                break
            if ok is False:
                status = "unsat"
                tr.note(f"The solver proved in {dt:.2f} s that no line satisfies all the clauses: "
                        f"this puzzle has no solution.", kind="stuck")
                break
            cand = enc.decode(sv.get_model())
            why = puzzle.check_solution(cand) if None not in cand else "incomplete model"
            if why is None:
                status, path = "solved", cand
                tr.note(f"Solver found a model in {dt:.2f} s. Decoding it into a line...", kind="solved")
                break
            tr.note(f"The model breaks a rule I didn't encode ({why}); blocking it and asking again.",
                    kind="retry")
            sv.add_clause([-enc.x[v, t] for t, v in enumerate(cand) if (v, t) in enc.x])
            if tries > 50:
                break
    if path is not None:
        tr.path_event(path)
        tr.note("Q.E.D. Every clause is satisfied.", kind="done")
    return result(ROBOT, puzzle, path or [start], status, t0, tr, None, tries,
                  stats={"variables": enc.nv, "clauses": nclauses, "position_vars": len(enc.x),
                         "encode_seconds": round(t_enc, 3), "solver": "cadical153", "models": tries})


__all__ = ["run", "ROBOT", "available", "Encoding"]
