"""🔦 Fair play in the fog: a partially-observable planner.

In fog mode (``meta.fog``) the UI hides the checkpoint *numbers* under little
"?" clouds until the path head comes within graph distance 2 of them.  The
clouds themselves are visible, so a player knows **where** the numbered cells
are and **how many** there are, and the "1" is always shown.  This module
plays by exactly the same information:

* :class:`FogView` holds the real puzzle privately and hands out only what the
  UI shows: the checkpoint cells, their count, the start and the labels
  revealed so far (``look(head)`` = the frontend's rule in
  ``play/board.js``: every checkpoint within 2 graph steps of the head,
  over all puzzle edges, becomes visible for good).
* :class:`Knowledge` is a snapshot of that view (the only thing the planner
  reads).  A completion is *consistent* with it when it is a legal Zip line
  (walls, one-way arcs, keys/doors) that meets the checkpoint cells so that
  every revealed number ``k`` is the k-th checkpoint met, hidden ones fill the
  remaining slots in any order, and the path ends on the checkpoint holding
  the largest number (which is some hidden cell while that number is unseen).
* :func:`run` plays: plan a consistent completion (a DFS with Warnsdorff move
  ordering and sound prunings: connectivity, degree / end rule, bipartite
  parity, plus a memo of proven-dead states; the exact solver on a relaxed
  puzzle - revealed numbers only + a virtual end node - is used as a fast
  second opinion), follow it one move at a time, reveal what comes into view,
  and when a newly seen number breaks the plan, re-plan; if no consistent
  completion exists any more, **back up** step by step until one does - like
  a person who discovers a surprise.  The true solution is always consistent
  with what is seen, so the planner converges; a step/time budget with a
  final fallback (the exact solver on the full puzzle, flagged in the trace
  and in ``stats.fallback``) guards the worst case.

Trace (standard robot trace, see CONTRACT.md) plus
``{"t": "reveal", "nodes": [cells]}`` whenever numbers come into view (so a
replay can lift the fog in sync) and ``{"t": "note", "msg", "kind"}`` captions
(kinds: intro, explore, spot, replan, backtrack, fallback, done).

:func:`fog_hint` / :func:`fog_explain` answer hints from the visible
information only ("Based on what you can see, ..."; "Not enough numbers
visible yet — explore toward the fog" when the view does not decide).
"""
from __future__ import annotations

import random
import time
import zlib
from collections import deque
from typing import Iterable, Sequence

import numpy as np

from .. import solver as _solver
from ..graph import blocked_steps, from_edges, prerequisites
from ..puzzle import Puzzle
from .base import Tracer, coord_str, direction, result, unavailable, unsupported_reason

ROBOT = "fog"
REVEAL_DIST = 2          # the UI's reveal radius (play/board.js updateFog -> near(P, head, 2))


def is_fog(puzzle: Puzzle) -> bool:
    return bool((puzzle.graph.meta or {}).get("fog"))


def near(graph, v: int, k: int = REVEAL_DIST) -> set[int]:
    """Nodes within graph distance ``k`` of ``v`` (all edges, like the frontend)."""
    seen = {int(v)}
    front = [int(v)]
    for _ in range(k):
        nx = []
        for u in front:
            for w in graph.neighbors[u]:
                if w not in seen:
                    seen.add(w)
                    nx.append(w)
        front = nx
    return seen


class FogView:
    """What a player sees on a fog board.  The true labels stay private."""

    def __init__(self, puzzle: Puzzle):
        cps = [int(c) for c in puzzle.checkpoints]
        self.graph = puzzle.graph
        self.cells = frozenset(cps)          # the "?" clouds (+ the 1)
        self.count = len(cps)
        self.start = cps[0]
        self.__label = {c: i for i, c in enumerate(cps)}
        self.revealed: dict[int, int] = {self.start: 0}   # cell -> 0-based label

    def look(self, head: int) -> list[int]:
        """Reveal every checkpoint within 2 steps of ``head``; returns the new ones."""
        new = sorted(c for c in near(self.graph, head) if c in self.cells and c not in self.revealed)
        for c in new:
            self.revealed[c] = self.__label[c]
        return new

    def show(self, cells: Iterable[int]) -> list[int]:
        """Mark checkpoints the player's board already shows (client-reported); others are ignored."""
        new = []
        for c in cells:
            try:
                c = int(c)
            except (TypeError, ValueError):
                continue
            if c in self.cells and c not in self.revealed:
                self.revealed[c] = self.__label[c]
                new.append(c)
        return new

    def knowledge(self) -> "Knowledge":
        return Knowledge(self.cells, self.count, self.start, dict(self.revealed))


class Knowledge:
    """A snapshot of the visible information (the only input of the planner)."""

    def __init__(self, cells, count: int, start: int, revealed: dict[int, int]):
        self.cells = frozenset(cells)
        self.K = int(count)
        self.start = int(start)
        self.lab = dict(revealed)                     # cell -> slot (0-based label)
        self.slot = {s: c for c, s in self.lab.items()}  # slot -> cell
        last = self.slot.get(self.K - 1)
        self.end_known = last

    def end_candidate(self, u: int) -> bool:
        if u not in self.cells:
            return False
        if self.end_known is not None:
            return u == self.end_known
        return u not in self.lab

    def consistent(self, path: Sequence[int], n: int) -> bool:
        """True if the (full or partial) line meets the checkpoints consistently."""
        j = 0
        L = len(path)
        for i, v in enumerate(path):
            if v in self.cells:
                s = self.lab.get(v)
                if s is None:
                    if j in self.slot:
                        return False
                elif s != j:
                    return False
                if j == self.K - 1 and i != n - 1:
                    return False
                j += 1
        if L == n and (j != self.K or path[-1] not in self.cells):
            return False
        return True


class _Budget(Exception):
    pass


class Planner:
    """Consistency search over completions of a line, given a :class:`Knowledge`.

    ``dead`` memoises proven-dead states ``(visited mask, head)``; since
    knowledge only grows during a run, a state dead once stays dead.
    """

    def __init__(self, puzzle: Puzzle, seed: int = 0):
        g = puzzle.graph
        self.puzzle = puzzle
        self.g = g
        self.n = g.num_nodes
        self.nbrs = [tuple(int(w) for w in ns) for ns in g.neighbors]
        self.blocked = blocked_steps(g) or {}
        self.pre = prerequisites(g) or {}
        self.rng = random.Random(seed * 7919 + 17)
        self.dead: set = set()
        self.expanded = 0
        self.colour = self._bipartite()
        self._dist: dict[int, list[int]] = {}
        # on a bipartite board the line alternates colours: its last cell has a known colour
        c0 = self.colour[int(puzzle.checkpoints[0])] if self.colour is not None else None
        self.end_colour = None if c0 is None else (c0 if self.n % 2 == 1 else 1 - c0)

    def end_ok(self, know: Knowledge, u: int) -> bool:
        return know.end_candidate(u) and (self.end_colour is None or self.colour[u] == self.end_colour)

    def _bipartite(self):
        col = [-1] * self.n
        for s in range(self.n):
            if col[s] >= 0:
                continue
            col[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for w in self.nbrs[u]:
                    if col[w] < 0:
                        col[w] = 1 - col[u]
                        q.append(w)
                    elif col[w] == col[u]:
                        return None
        return col

    def dist(self, src: int) -> list[int]:
        d = self._dist.get(src)
        if d is None:
            d = [1 << 20] * self.n
            d[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for w in self.nbrs[u]:
                    if d[w] > d[u] + 1:
                        d[w] = d[u] + 1
                        q.append(w)
            self._dist[src] = d
        return d

    # ---- one search -----------------------------------------------------
    def complete(self, path: Sequence[int], know: Knowledge, budget: int = 20000,
                 deadline: float | None = None):
        """Find a completion of ``path`` consistent with ``know``.

        Returns ``("found", full_path)``, ``("dead", None)`` (proof: none
        exists) or ``("unknown", None)`` (budget / time ran out).
        """
        n = self.n
        path = [int(v) for v in path]
        if not know.consistent(path, n):
            return "dead", None
        if len(path) == n:
            return ("found", path) if know.consistent(path, n) else ("dead", None)
        st = _State(self, know, path)
        if not st.feasible():
            return "dead", None
        self._left = budget
        self._deadline = deadline
        try:
            ok = self._dfs(st)
        except _Budget:
            return "unknown", None
        return ("found", list(st.path)) if ok else ("dead", None)

    def _dfs(self, st: "_State") -> bool:
        if len(st.path) == self.n:
            return True
        key = (st.mask, st.path[-1])
        if key in self.dead:
            return False
        self._left -= 1
        self.expanded += 1
        if self._left < 0:
            raise _Budget
        if self._deadline is not None and (self._left & 127) == 0 and time.perf_counter() > self._deadline:
            raise _Budget
        for w in st.ordered_moves():
            st.push(w)
            if st.feasible() and self._dfs(st):
                return True
            st.pop()
        self.dead.add(key)
        return False

    # ---- relaxed exact solver (second opinion) --------------------------
    def relaxed(self, path: Sequence[int], know: Knowledge, time_limit: float):
        """The exact solver on a relaxation: revealed numbers in order, then the
        end (the revealed largest number, else a virtual node joined to every
        hidden cell).  "unsat" is a proof; a solution is only a candidate."""
        g = self.g
        n = self.n
        order = [know.slot[s] for s in sorted(know.slot)]
        edges = [tuple(e) for e in g.edges()]
        coords = np.asarray(g.coords, dtype=float)
        meta = {k: v for k, v in (g.meta or {}).items() if not str(k).startswith("_")}
        if know.end_known is None:
            hidden = [c for c in know.cells if self.end_ok(know, c)]
            if not hidden:
                return "unsat", None
            coords = np.vstack([coords, coords[:1] * 0 - 7])
            edges += [(c, n) for c in hidden]
            order = order + [n]
        try:
            rp = Puzzle(from_edges(coords, edges, g.kind, meta), order)
            r = _solver.solve_from_prefix(rp, list(path), time_limit=time_limit)
        except (ValueError, KeyError):
            return "timeout", None
        if r.status != "solved":
            return r.status, None
        p = [int(v) for v in r.path if v < n]
        return "solved", p

    def hypothesis(self, path: Sequence[int], know: Knowledge, line: Sequence[int], time_limit: float):
        """The exact solver on one full guess of the hidden order (the order ``line`` meets the hidden
        cells); a solution is consistent with ``know``, "unsat" only rules out this guess."""
        order = _hypothesis_order(know, line)
        try:
            r = _solver.solve_from_prefix(Puzzle(self.g, order), list(path), time_limit=time_limit)
        except (ValueError, StopIteration):
            return "timeout", None
        return r.status, (None if r.path is None else [int(v) for v in r.path])


def _hypothesis_order(know: Knowledge, line: Sequence[int]) -> list[int]:
    """A full checkpoint order consistent with ``know``: hidden cells fill the free slots in the order
    ``line`` meets them."""
    hidden = [v for v in line if v in know.cells and v not in know.lab]
    it = iter(hidden)
    return [know.slot[s] if s in know.slot else next(it) for s in range(know.K)]


class _State:
    """Incremental line state for the consistency DFS."""

    def __init__(self, pl: Planner, know: Knowledge, path: list[int]):
        self.pl = pl
        self.k = know
        self.n = pl.n
        self.path = list(path)
        self.vis = bytearray(self.n)
        self.mask = 0
        self.j = 0
        self.cnt = [0, 0]
        for v in self.path:
            self.vis[v] = 1
            self.mask |= 1 << v
            if v in know.cells:
                self.j += 1
        col = pl.colour
        if col is not None:
            for v in range(self.n):
                if not self.vis[v]:
                    self.cnt[col[v]] += 1

    def allowed(self, w: int) -> bool:
        k = self.k
        if self.vis[w]:
            return False
        h = self.path[-1]
        if w in self.pl.blocked.get(h, ()):
            return False
        for a in self.pl.pre.get(w, ()):
            if not self.vis[a]:
                return False
        if w in k.cells:
            s = k.lab.get(w)
            if s is None:
                if self.j in k.slot:
                    return False
            elif s != self.j:
                return False
            if self.j == k.K - 1 and len(self.path) + 1 != self.n:
                return False
        return True

    def push(self, w: int) -> None:
        self.path.append(w)
        self.vis[w] = 1
        self.mask |= 1 << w
        if w in self.k.cells:
            self.j += 1
        if self.pl.colour is not None:
            self.cnt[self.pl.colour[w]] -= 1

    def pop(self) -> None:
        w = self.path.pop()
        self.vis[w] = 0
        self.mask &= ~(1 << w)
        if w in self.k.cells:
            self.j -= 1
        if self.pl.colour is not None:
            self.cnt[self.pl.colour[w]] += 1

    def ordered_moves(self) -> list[int]:
        h = self.path[-1]
        nb = self.pl.nbrs
        vis = self.vis
        k = self.k
        tgt = k.slot.get(self.j)
        dist = self.pl.dist(tgt) if tgt is not None else None
        rnd = self.pl.rng.random
        out = []
        for w in nb[h]:
            if not self.allowed(w):
                continue
            onward = sum(1 for x in nb[w] if not vis[x])
            # Warnsdorff first; then head for the next visible number; then a seeded coin
            out.append((onward, dist[w] if dist is not None else 0, rnd(), w))
        out.sort()
        return [w for *_, w in out]

    def feasible(self) -> bool:
        """Sound prunings: parity, connectivity, degree/end rule."""
        n = self.n
        r = n - len(self.path)
        if r == 0:
            return True
        h = self.path[-1]
        col = self.pl.colour
        if col is not None:
            x = 1 - col[h]
            need_x = (r + 1) // 2
            if self.cnt[x] != need_x or self.cnt[1 - x] != r - need_x:
                return False
        nb = self.pl.nbrs
        vis = self.vis
        k = self.k
        start = -1
        for w in nb[h]:
            if not vis[w]:
                start = w
                break
        if start < 0:
            return False
        hn = set(nb[h])
        # BFS over unvisited nodes: connectivity + degree rule in one pass
        seen = {start}
        q = [start]
        ends = 0
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            d = 0
            for w in nb[u]:
                if not vis[w]:
                    d += 1
                    if w not in seen:
                        seen.add(w)
                        q.append(w)
            a = 1 if u in hn else 0
            if d + a < 2:
                if d == 0 and r > 1:
                    return False
                if not self.pl.end_ok(k, u):
                    return False
                ends += 1
                if ends > 1:
                    return False
        return len(q) == r


# ---------------------------------------------------------------------------
# words (fog-safe: hidden numbers are never named)
# ---------------------------------------------------------------------------
def _num(know: Knowledge, v: int) -> str | None:
    s = know.lab.get(int(v))
    return None if s is None else str(s + 1)


def _cell(puzzle: Puzzle, know: Knowledge, v: int) -> str:
    s = _num(know, v)
    if s is not None:
        return f"the {s}"
    if int(v) in know.cells:
        return f"a hidden number at {coord_str(puzzle, v)}"
    return coord_str(puzzle, v)


def _labels(know: Knowledge, cells: Sequence[int]) -> str:
    nums = sorted(know.lab[c] + 1 for c in cells)
    words = [str(x) for x in nums]
    art = "An" if words[0].startswith("8") or words[0] in ("11", "18") else "A"
    if len(words) == 1:
        return f"{art} {words[0]}"
    return f"{art} " + ", ".join(words[:-1]) + " and " + words[-1]


# ---------------------------------------------------------------------------
# the fog run
# ---------------------------------------------------------------------------
def run(puzzle: Puzzle, time_limit: float = 10.0, trace: bool = True, seed: int = 0,
        start_path: Sequence[int] | None = None, seen: Iterable[int] | None = None,
        robot: str = ROBOT, max_steps: int | None = None) -> dict:
    """Play the fog puzzle fairly (see the module docstring).  Same response
    shape as the other robots, plus ``fog: true`` and fog ``stats``."""
    bad = unsupported_reason(puzzle)
    if bad:
        return unavailable(robot, puzzle, bad, "unsupported", trace)
    t0 = time.perf_counter()
    deadline = t0 + max(0.2, float(time_limit))
    n = puzzle.num_nodes
    view = FogView(puzzle)
    # each robot keeps its own (deterministic) tie-breaks, so two robots in the fog take different routes
    pl = Planner(puzzle, seed * 1009 + (zlib.crc32(robot.encode()) % 997 if robot != ROBOT else 0))
    tr = Tracer(trace)
    path = [int(v) for v in (start_path or [puzzle.checkpoints[0]])]
    if _solver.check_prefix(puzzle, path) is not None:
        path = [int(puzzle.checkpoints[0])]
    tr.restart(path)
    first = set(view.show(seen or []))
    for v in path:
        first.update(view.look(v))
    if first:
        tr.event({"t": "reveal", "nodes": sorted(first)})
    stats = {"reveals": len(first) + 1, "surprises": 0, "replans": 0, "backtracks": 0, "backed_up_steps": 0,
             "searches": 0, "fallback": False, "max_back": 0}
    know = view.knowledge()
    hidden = view.count - len(know.lab)
    tr.note(f"Fog! I can see {len(know.lab)} of the {view.count} numbers. The rest are hiding under the "
            f"clouds: I know where they are, not which is which, so I'll plan with what I see and "
            f"re-think when a surprise shows up." if hidden else
            "Fog, but every number is already in sight.", kind="intro")
    max_steps = max_steps or (30 * n + 300)
    steps = 0
    plan: list[int] | None = None
    exploring = None      # last "no target in sight" state we captioned

    def search(p):
        stats["searches"] += 1
        left = deadline - time.perf_counter()
        st, full = pl.complete(p, know, budget=6000 if n <= 64 else 3000,
                               deadline=time.perf_counter() + max(0.05, min(left * 0.25, 1.5)))
        if st != "unknown":
            return st, full
        rs, rp = pl.relaxed(p, know, max(0.05, min(left * 0.15, 0.8)))
        if rs == "unsat":
            return "dead", None
        if rs == "solved" and know.consistent(rp, n):
            return "found", rp
        # guesses at the hidden order, each solved exactly: the relaxed line's order, the order of the plan
        # that just failed, and nearest-first from the pen
        hidden = [c for c in know.cells if c not in know.lab and c not in p]
        d = pl.dist(p[-1])
        guesses = [g for g in (rp if rs == "solved" else None, plan, sorted(hidden, key=lambda c: (d[c], c)))
                   if g is not None]
        for g in guesses:
            hs, hp = pl.hypothesis(p, know, g, max(0.05, min((deadline - time.perf_counter()) * 0.1, 0.8)))
            if hs == "solved" and know.consistent(hp, n):
                return "found", hp
        st, full = pl.complete(p, know, budget=60000,
                               deadline=time.perf_counter() + max(0.05, min(left * 0.4, 4.0)))
        return st, full

    def fallback(why: str):
        nonlocal path
        stats["fallback"] = True
        sol = _solver.solve_from_prefix(puzzle, path, time_limit=max(1.0, deadline - time.perf_counter() + 5))
        if sol.status != "solved":
            sol = _solver.solve(puzzle, time_limit=30)
            if sol.status != "solved":
                return False
            c = 0
            while c < len(path) and path[c] == sol.path[c]:
                c += 1
            tr.pop(len(path) - max(1, c))
            path = path[:max(1, c)]
        tr.note(why + " Peeking at the whole map to finish (fallback).", kind="fallback")
        for v in sol.path[len(path):]:
            path.append(int(v))
            tr.push(v)
            new = view.look(v)
            if new:
                tr.event({"t": "reveal", "nodes": new})
        return True

    status = "stuck"
    while len(path) < n:
        steps += 1
        if steps > max_steps or time.perf_counter() > deadline:
            status = "solved" if fallback("That took too long in the fog.") else "stuck"
            break
        if plan is None or plan[:len(path)] != path or not know.consistent(plan, n):
            st, full = search(path)
            if st == "dead":
                back = 0
                while st == "dead" and len(path) > 1:
                    path.pop()
                    back += 1
                    st, full = search(path)
                if st == "dead":            # cannot happen: the true solution is always consistent
                    status = "solved" if fallback("Something doesn't add up.") else "stuck"
                    break
                tr.pop(back)
                stats["backtracks"] += 1
                stats["backed_up_steps"] += back
                stats["max_back"] = max(stats["max_back"], back)
            if st == "found":
                plan = full
            else:                            # unknown: take the most promising sensible move, re-plan next time
                plan = None
                s2 = _State(pl, know, path)
                mv = None
                for w in s2.ordered_moves():
                    s2.push(w)
                    ok = s2.feasible() and (s2.mask, w) not in pl.dead
                    s2.pop()
                    if ok:
                        mv = w
                        break
                if mv is None:
                    pl.dead.add((s2.mask, path[-1]))
                    path.pop()
                    tr.pop(1)
                    stats["backtracks"] += 1
                    stats["backed_up_steps"] += 1
                    if len(path) == 0:
                        path = [int(puzzle.checkpoints[0])]
                    continue
                v = mv
        if plan is not None:
            v = plan[len(path)]
        # "no numbers in sight" caption, once per stretch
        nxt_slot = sum(1 for x in path if x in know.cells)
        if nxt_slot not in know.slot and exploring != nxt_slot:
            exploring = nxt_slot
            tr.note(f"No {nxt_slot + 1} in sight yet: heading for unexplored cells.", kind="explore")
        path.append(int(v))
        tr.push(v)
        new = view.look(v)
        if new:
            tr.event({"t": "reveal", "nodes": new})
            stats["reveals"] += len(new)
            know = view.knowledge()
            if plan is not None and not know.consistent(plan, n):
                stats["surprises"] += 1
                st, full = search(path)
                what = _labels(know, new)
                if st == "found":
                    stats["replans"] += 1
                    plan = full
                    tr.note(f"{what} just appeared — that changes things. New plan from here.", kind="replan",
                            cells=list(new))
                elif st == "dead":
                    # count how far we have to back up (the actual pops happen at the top of the loop)
                    k = 0
                    probe = list(path)
                    while st == "dead" and len(probe) > 1:
                        probe.pop()
                        k += 1
                        st, full = search(probe)
                    tr.note(f"{what} just appeared — that changes things, backing up {k} "
                            f"step{'s' if k != 1 else ''}.", kind="backtrack", cells=list(new))
                    path = probe
                    tr.pop(k)
                    stats["backtracks"] += 1
                    stats["backed_up_steps"] += k
                    stats["max_back"] = max(stats["max_back"], k)
                    plan = full if st == "found" else None
                else:
                    plan = None
            else:
                cur = sum(1 for x in path if x in know.cells)
                tgt = know.slot.get(cur)
                if tgt is not None and tgt in new:
                    tr.note(f"There's the {cur + 1} — heading for it.", kind="spot", cells=[tgt])
                    exploring = None
    else:
        status = "solved"
    if len(path) == n and status != "solved":
        status = "solved"
    if status == "solved" and not puzzle.is_valid_solution(path):
        # should never happen (every consistent full line is a real solution once all numbers are seen)
        status = "solved" if fallback("Hmm, that line isn't right.") else "stuck"
    if status == "solved":
        tr.note(f"Solved in the fog: {stats['surprises']} surprise{'s' if stats['surprises'] != 1 else ''}, "
                f"backed up {stats['backtracks']} time{'s' if stats['backtracks'] != 1 else ''}"
                + (" (with a peek)." if stats["fallback"] else "."), kind="done")
    stats["nodes"] = pl.expanded
    return result(robot, puzzle, path, status, t0, tr, expanded=pl.expanded,
                  backtracks=stats["backtracks"], stats=stats, fog=True)


# ---------------------------------------------------------------------------
# hints from the visible information only
# ---------------------------------------------------------------------------
def _view_for(puzzle: Puzzle, path: Sequence[int], seen: Iterable[int] | None) -> FogView:
    view = FogView(puzzle)
    if seen is not None:
        view.show(seen)
        if path:
            view.look(path[-1])
    else:
        for v in path:
            view.look(v)
    return view


def _advice(puzzle: Puzzle, path: list[int], seen, time_limit: float):
    """(status, keep, move, certain, reason, cells, expanded)."""
    t_end = time.perf_counter() + time_limit
    n = puzzle.num_nodes
    view = _view_for(puzzle, path, seen)
    know = view.knowledge()
    pl = Planner(puzzle, 0)

    def search(p, frac=0.3):
        left = max(0.05, (t_end - time.perf_counter()) * frac)
        return pl.complete(p, know, budget=40000, deadline=time.perf_counter() + left)

    st, full = search(path)
    if st == "dead":
        lo, hi, best = 1, len(path) - 1, None     # prefix[:lo] completable, prefix[:hi+1] not
        s1, f1 = search(path[:1])
        best = f1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            s, f = search(path[:mid])
            if s == "dead":
                hi = mid - 1
            else:
                lo = mid
                best = f if s == "found" else best
        if best is None or best[:lo] != path[:lo]:
            s, f = search(path[:lo], 0.9)
            best = f
        mv = best[lo] if best and len(best) > lo else None
        if mv is None:
            s2 = _State(pl, know, path[:lo])
            mv = next(iter(s2.ordered_moves()), None)
        return ("backtrack", lo, mv, True,
                f"Based on what you can see, your line can't work from here: back up to step {lo}"
                + (f" and go {direction(puzzle, path[lo - 1], mv)}." if mv is not None else "."),
                [mv] if mv is not None else [], pl.expanded)
    s0 = _State(pl, know, path)
    moves = s0.ordered_moves()
    possible = []
    for m in moves:
        s, f = search(path + [m], 0.5 / max(1, len(moves)))
        if s != "dead":
            possible.append((m, s, f))
    if not possible:     # the planner's own line (search above said found / unknown)
        mv = full[len(path)] if full else (moves[0] if moves else None)
        possible = [(mv, st, full)] if mv is not None else []
    if len(possible) == 1:
        m = possible[0][0]
        return ("next", len(path), m, True,
                f"Based on what you can see, going {direction(puzzle, path[-1], m)} to "
                f"{_cell(puzzle, know, m)} is the only move that can still work.",
                [m], pl.expanded)
    # not decided by the visible numbers: suggest the move that heads toward the fog
    hidden = [c for c in know.cells if c not in know.lab]
    best = possible[0][0]
    if hidden:
        def fogd(m):
            return min(pl.dist(c)[m] for c in hidden)
        best = min((m for m, *_ in possible), key=lambda m: (fogd(m), [x for x, *_ in possible].index(m)))
    return ("next", len(path), best, False,
            f"Not enough numbers visible yet — explore toward the fog. {len(possible)} moves could still "
            f"work; going {direction(puzzle, path[-1], best)} gets you closer to the hidden numbers.",
            [best], pl.expanded)


def fog_hint(puzzle: Puzzle, path: Sequence[int], seen: Iterable[int] | None = None,
             time_limit: float = 3.0) -> dict:
    """``/api/hint`` in fog mode: {"status": "next"|"backtrack"|"done"|"invalid", "next", "keep",
    "reason", "certain", "source": "fog", "fog": true, ...}."""
    t0 = time.perf_counter()
    path = [int(v) for v in path] or [int(puzzle.checkpoints[0])]
    bad = _solver.check_prefix(puzzle, path)
    if bad is not None:
        return {"status": "invalid", "message": bad, "fog": True}
    if len(path) == puzzle.num_nodes:
        return {"status": "done", "message": "Already solved!", "fog": True}
    st, keep, mv, certain, reason, cells, exp = _advice(puzzle, path, seen, time_limit)
    if mv is None:
        return {"status": "timeout", "message": "No hint right now.", "fog": True}
    out = {"status": st, "next": int(mv), "keep": int(keep), "source": "fog", "reason": reason,
           "certain": bool(certain), "fog": True, "nodes_expanded": exp,
           "seconds": round(time.perf_counter() - t0, 4)}
    if st == "backtrack":
        out["message"] = reason
    return out


def fog_explain(puzzle: Puzzle, path: Sequence[int], seen: Iterable[int] | None = None,
                time_limit: float = 3.0) -> dict:
    """``/api/hint/explain`` in fog mode (same shape as the Detective's teaching hint)."""
    t0 = time.perf_counter()
    path = [int(v) for v in path] or [int(puzzle.checkpoints[0])]
    bad = _solver.check_prefix(puzzle, path)
    if bad is not None:
        return {"status": "invalid", "move": None, "reason": f"That line breaks a rule: {bad}.", "fog": True}
    if len(path) == puzzle.num_nodes:
        return {"status": "done", "move": None, "reason": "Already solved.", "fog": True}
    st, keep, mv, certain, reason, cells, _ = _advice(puzzle, path, seen, time_limit)
    out = {"status": st, "move": None if mv is None else int(mv), "reason": reason, "technique": "fog",
           "technique_name": "What you can see" if certain else "Explore the fog", "cells": cells,
           "certain": bool(certain), "fog": True, "seconds": round(time.perf_counter() - t0, 4)}
    if st == "backtrack":
        out["keep"] = keep
    return out


__all__ = ["FogView", "Knowledge", "Planner", "run", "fog_hint", "fog_explain", "is_fog", "near"]
