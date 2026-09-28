"""Shared plumbing for the robots: trace recording, the response shape, a fast
walker over the solver's search state and plain-English cell descriptions.

Legality always comes from the solver's public helpers
(:func:`zipsolve.solver.legal_moves`, :func:`~zipsolve.solver.check_prefix`,
:meth:`Puzzle.check_solution`), so every robot follows whatever rules the core
puzzle model supports (one-way arcs, key/door precedence, ...).  The solver's
search state (``_Search``: free degrees, incremental dead-end checks) is only
used for *heuristics and sound pruning*: its checks run on the undirected graph,
a relaxation of any extra rule, so a state it calls dead really is dead.
"""
from __future__ import annotations

import random
import time
from typing import Iterable, Sequence

import numpy as np

from .. import solver as _solver
from ..puzzle import Puzzle

TRACE_CAP = 40_000        # max recorded push/pop/path events per run
NOTE_CAP = 1_500          # max caption events per run (not counted in TRACE_CAP)

UNSUPPORTED_KINDS = {"coop"}  # two-path puzzles: none of the robots plays them (yet)


# ---------------------------------------------------------------------------
# rules (thin wrappers over the solver's public API)
# ---------------------------------------------------------------------------
def legal(puzzle: Puzzle, path: Sequence[int]) -> list[int]:
    """Legal next nodes after the legal prefix ``path`` (the solver's rules)."""
    return [int(v) for v in _solver.legal_moves(puzzle, path)]


def is_legal_prefix(puzzle: Puzzle, path: Sequence[int]) -> bool:
    return len(path) > 0 and _solver.check_prefix(puzzle, list(path)) is None


def is_solution(puzzle: Puzzle, path: Sequence[int]) -> bool:
    return len(path) == puzzle.num_nodes and puzzle.check_solution(list(path)) is None


def new_search(puzzle: Puzzle, deep: bool = True):
    """The solver's search state for ``puzzle`` (heuristics + sound pruning only)."""
    S = _solver._Search
    try:
        return S(puzzle.graph, puzzle.checkpoints, deep=deep, puzzle=puzzle)  # newer solvers
    except TypeError:
        return S(puzzle.graph, puzzle.checkpoints, deep=deep)


def unsupported_reason(puzzle: Puzzle) -> str | None:
    kind = getattr(puzzle.graph, "kind", "")
    if kind in UNSUPPORTED_KINDS or (puzzle.graph.meta or {}).get("coop"):
        return "two-path (co-op) puzzles are not supported by this robot"
    return None


class Walker:
    """A path plus the solver's incremental search state.

    ``moves()`` = legal moves (solver rules); ``push`` returns the solver's
    incremental verdict: -2 = provably dead, >= 0 = a node the line is forced
    to visit next (relaxed rules: only a hint), -1 = nothing special.
    """

    def __init__(self, puzzle: Puzzle, prefix: Sequence[int] | None = None, deep: bool = False):
        self.puzzle = puzzle
        self.n = puzzle.num_nodes
        self.s = new_search(puzzle, deep=deep)
        self.path = self.s.path
        self.verdicts: list[int] = []
        prefix = list(prefix or [puzzle.checkpoints[0]])
        for v in prefix:
            self.push(int(v))

    def moves(self) -> list[int]:
        if len(self.path) >= self.n:
            return []
        return legal(self.puzzle, self.path)

    def push(self, v: int) -> int:
        s = self.s
        p = s.path[-1] if s.path else -1
        s.push(v)
        if p < 0:
            r = -1
        else:
            r = s.after_move(p, v)
        self.verdicts.append(r)
        return r

    def pop(self) -> None:
        self.s.pop()
        self.verdicts.pop()

    def truncate(self, k: int) -> None:
        while len(self.path) > k:
            self.pop()

    @property
    def dead(self) -> bool:
        return bool(self.verdicts) and self.verdicts[-1] == -2

    @property
    def head(self) -> int:
        return self.path[-1]

    def free_degree(self, v: int) -> int:
        return self.s.deg[v]

    def dist_to_target(self, v: int) -> int:
        s = self.s
        k = min(s.nxt, len(s.cps) - 1)
        return s.dist[k][v]

    def complete(self) -> bool:
        return len(self.path) == self.n and is_solution(self.puzzle, self.path)


def heuristic_key(w: Walker, v: int) -> tuple:
    """Lower is better: the next checkpoint first, then Warnsdorff (fewest free
    neighbours), then closeness to the next checkpoint."""
    s = w.s
    is_next = s.cpi[v] >= 0 and s.cpi[v] == s.nxt
    return (0 if is_next else 1, s.deg[v], w.dist_to_target(v))


def rollout(w: Walker, rng: random.Random, eps: float = 0.15, max_steps: int | None = None,
            forced_first: bool = True) -> tuple[int, list[int] | None]:
    """Heuristic random playout from the walker's state (the walker is restored).

    Returns (nodes covered at the end, the full solution if the playout solved
    the puzzle else None). A playout stops at the first state the solver proves
    dead or with no legal move.
    """
    base = len(w.path)
    solved = False
    sol = None
    steps = 0
    try:
        while True:
            if len(w.path) == w.n:
                solved = w.complete()
                break
            if w.dead:
                break
            ms = w.moves()
            if not ms:
                break
            f = w.verdicts[-1] if forced_first else -1
            if f >= 0 and f in ms:
                v = f
            elif len(ms) == 1:
                v = ms[0]
            elif rng.random() < eps:
                v = ms[rng.randrange(len(ms))]
            else:
                best = min(heuristic_key(w, m) for m in ms)
                ties = [m for m in ms if heuristic_key(w, m) == best]
                v = ties[rng.randrange(len(ties))]
            w.push(v)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        covered = len(w.path)
        if w.dead and not solved:
            covered -= 1        # the last step was provably fatal
        if solved:
            sol = list(w.path)
        return covered, sol
    finally:
        w.truncate(base)


# ---------------------------------------------------------------------------
# trace recording
# ---------------------------------------------------------------------------
class Tracer:
    """Records the displayed line as push/pop events plus robot-specific
    events: ``{"t": "note", "msg": ...}``, ``{"t": "path", "p": [...]}`` and
    ``{"t": "tree", ...}``. Tracks the displayed stack so ``goto`` emits the
    minimal pops/pushes."""

    def __init__(self, enabled: bool = True, cap: int = TRACE_CAP, note_cap: int = NOTE_CAP):
        self.enabled = enabled
        self.cap = cap
        self.note_cap = note_cap
        self.events: list = []
        self.stack: list[int] = []
        self.count = 0
        self.notes = 0
        self.truncated = False

    def _room(self, k: int = 1) -> bool:
        if not self.enabled:
            return False
        if self.count + k > self.cap:
            self.truncated = True
            return False
        self.count += k
        return True

    def restart(self, path: Sequence[int]) -> None:
        self.stack = [int(v) for v in path]
        if self._room(1 + len(path)):
            self.events.append("R")
            self.events.extend(int(v) for v in path)

    def push(self, v: int) -> None:
        self.stack.append(int(v))
        if self._room():
            self.events.append(int(v))

    def pop(self, k: int = 1) -> None:
        k = min(k, len(self.stack))
        if k <= 0:
            return
        del self.stack[len(self.stack) - k:]
        if self._room():
            if self.events and isinstance(self.events[-1], int) and self.events[-1] < 0:
                self.events[-1] -= k
            else:
                self.events.append(-k)

    def goto(self, path: Sequence[int]) -> None:
        path = [int(v) for v in path]
        st = self.stack
        c = 0
        while c < len(st) and c < len(path) and st[c] == path[c]:
            c += 1
        if c == 0 and st:
            self.path_event(path)
            return
        self.pop(len(st) - c)
        for v in path[c:]:
            self.push(v)

    def path_event(self, path: Sequence[int]) -> None:
        self.stack = [int(v) for v in path]
        if self._room():
            self.events.append({"t": "path", "p": [int(v) for v in path]})

    def note(self, msg: str, **extra) -> None:
        if not self.enabled or self.notes >= self.note_cap:
            return
        self.notes += 1
        ev = {"t": "note", "msg": str(msg)}
        ev.update({k: v for k, v in extra.items() if v is not None})
        self.events.append(ev)

    def event(self, ev: dict) -> None:
        if self._room():
            self.events.append(ev)

    def finish(self, final_path: Sequence[int] | None) -> None:
        """Make sure replaying the trace ends on ``final_path`` (always recorded)."""
        if not self.enabled or final_path is None:
            return
        final_path = [int(v) for v in final_path]
        if self.stack != final_path:
            self.events.append({"t": "path", "p": final_path})
            self.stack = final_path


def result(robot: str, puzzle: Puzzle, path: Sequence[int] | None, status: str, t0: float,
           tracer: Tracer | None, steps: list[dict] | None = None, expanded: int = 0,
           **extra) -> dict:
    """The app's robot response shape (see CONTRACT.md "Clarifications (robots)")."""
    path = [int(v) for v in (path or [puzzle.checkpoints[0]])]
    solved = status == "solved" and is_solution(puzzle, path)
    if status == "solved" and not solved:
        status = "stuck"
    if tracer is not None:
        tracer.finish(path)
    out = {
        "robot": robot, "status": status, "solved": solved, "path": path, "start_len": 1,
        "steps": steps if steps is not None else [{"node": v, "p": None} for v in path[1:]],
        "nodes_expanded": int(expanded), "seconds": round(time.perf_counter() - t0, 4),
        "trace": tracer.events if (tracer is not None and tracer.enabled) else None,
        "trace_truncated": bool(tracer.truncated) if tracer is not None else False,
    }
    if not solved:
        out["stuck_at"] = len(path)
    out.update(extra)
    return out


def unavailable(robot: str, puzzle: Puzzle, reason: str, status: str = "unavailable",
                trace: bool = True) -> dict:
    t0 = time.perf_counter()
    tr = Tracer(trace)
    tr.restart([puzzle.checkpoints[0]])
    tr.note(reason, kind="error")
    return result(robot, puzzle, None, status, t0, tr, reason=reason)


# ---------------------------------------------------------------------------
# plain-English descriptions of cells, moves and regions
# ---------------------------------------------------------------------------
def _layout(puzzle: Puzzle) -> str:
    return str((puzzle.graph.meta or {}).get("layout", "square"))


def cell_name(puzzle: Puzzle, v: int) -> str:
    """'(2,3)' (1-based) for grid cells; checkpoints are named by their number."""
    lab = {c: i + 1 for i, c in enumerate(puzzle.checkpoints)}.get(int(v))
    if lab is not None:
        return f"number {lab}"
    return "cell " + coord_str(puzzle, v)


def coord_str(puzzle: Puzzle, v: int) -> str:
    c = puzzle.graph.coords[int(v)]
    if _layout(puzzle) == "hex" or np.any(c < 0) or np.any(c != np.round(c)):
        vals = [int(x) if float(x).is_integer() else round(float(x), 1) for x in c]
    else:
        vals = [int(x) + 1 for x in c]
    return "(" + ",".join(str(x) for x in vals) + ")"


_HEX_DIRS = {(1, 0): "right", (-1, 0): "left", (1, -1): "up-right", (0, -1): "up-left",
             (-1, 1): "down-left", (0, 1): "down-right"}


def direction(puzzle: Puzzle, a: int, b: int) -> str:
    """'up' / 'left' / ... for a move a -> b, or 'to (r,c)' when there is no simple word."""
    g = puzzle.graph
    meta = g.meta or {}
    pair = sorted((int(a), int(b)))
    for key, word in (("portals", "through the portal"), ("wrap_edges", "around the edge")):
        for e in meta.get(key) or []:
            try:
                if sorted(int(x) for x in e) == pair:
                    return f"{word} to {coord_str(puzzle, b)}"
            except (TypeError, ValueError):
                continue
    d = g.coords[int(b)] - g.coords[int(a)]
    lay = _layout(puzzle)
    if lay == "hex" and g.dim == 2:
        w = _HEX_DIRS.get((int(d[0]), int(d[1])))
        if w:
            return w
    if g.dim == 2 and lay in ("square", "tri"):
        dr, dc = float(d[0]), float(d[1])
        if (abs(dr), abs(dc)) == (1.0, 0.0):
            return "down" if dr > 0 else "up"
        if (abs(dr), abs(dc)) == (0.0, 1.0):
            return "right" if dc > 0 else "left"
    if g.dim == 3 and np.abs(d).sum() == 1 and g.kind != "cubesurf":
        ax = int(np.argmax(np.abs(d)))
        if ax == 0:
            return "to the next layer" if d[0] > 0 else "to the previous layer"
        if ax == 1:
            return "down" if d[1] > 0 else "up"
        return "right" if d[2] > 0 else "left"
    return "to " + coord_str(puzzle, b)


def going(puzzle: Puzzle, a: int, b: int) -> str:
    """'going up' / 'going to (3,4)' / 'going to number 5'."""
    lab = {c: i + 1 for i, c in enumerate(puzzle.checkpoints)}.get(int(b))
    if lab is not None:
        return f"going to number {lab}"
    return "going " + direction(puzzle, a, b)


def region_name(puzzle: Puzzle, nodes: Iterable[int]) -> str:
    """'the bottom-right corner', 'the top edge', 'the middle', ... (2D; else 'a group of k cells')."""
    nodes = list(nodes)
    g = puzzle.graph
    k = len(nodes)
    if g.dim != 2 or not k:
        return f"a group of {k} cell{'s' if k != 1 else ''}"
    X = g.coords
    lo, hi = X.min(0), X.max(0)
    span = np.maximum(hi - lo, 1e-9)
    c = (X[nodes].mean(0) - lo) / span
    if _layout(puzzle) == "hex":
        rows = ("top", "middle", "bottom")
        v = rows[min(2, int(c[1] * 3))]
        h = ("left", "centre", "right")[min(2, int(c[0] * 3))]
    else:
        v = ("top", "middle", "bottom")[min(2, int(c[0] * 3))]
        h = ("left", "centre", "right")[min(2, int(c[1] * 3))]
    if k == 1:
        return cell_name(puzzle, nodes[0])
    if v == "middle" and h == "centre":
        where = "the middle"
    elif v == "middle":
        where = f"the {h} side"
    elif h == "centre":
        where = f"the {v} edge"
    else:
        where = f"the {v}-{h} corner"
    return f"{where} ({k} cells)"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def rng_for(seed: int, salt: str) -> random.Random:
    return random.Random(f"{salt}:{int(seed)}")


__all__ = ["Walker", "Tracer", "result", "unavailable", "legal", "is_legal_prefix", "is_solution",
           "new_search", "rollout", "heuristic_key", "cell_name", "coord_str", "direction", "going",
           "region_name", "unsupported_reason", "pct", "rng_for", "TRACE_CAP"]
