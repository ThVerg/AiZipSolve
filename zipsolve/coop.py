"""Co-op Zip ("two paths"): two vertex-disjoint paths that together cover every node.

Rules: path i (i = 0, 1) starts at ``checkpoints[i][0]``, ends at
``checkpoints[i][-1]`` and visits its own checkpoints in order.  A path never
touches the other path's checkpoints.  Together the paths cover every node once.
In the UI path 1 is drawn with a warm gradient, path 2 with a cool one (two
"pens": hot-seat on one device, or one player switching pens).

JSON (``CoopPuzzle.to_dict``)::

    {"kind": "coop", "coords": [...], "edges": [[u, v], ...],
     "checkpoints": [...path 1's checkpoints...],          # so single-path code does not crash
     "solution": null,
     "meta": {..., "coop": {"checkpoints": [[...], [...]], "base": "grid2d",
                            "solution": {"paths": [[...], [...]]}}}}  # solution only when known

Solver
------
Depth-first search over *both* path heads (iterative, no recursion).  At every
state the search picks ONE active head and branches over all of its legal
moves (complete: every solution fixes that head's next node, and each solution
is generated exactly once, so counting is exact).  A head with a forced move is
extended first, otherwise the head with the fewest legal moves.  Sound
prunings, all derived from "the unvisited nodes U must be covered by the
remainders of the active paths":

* never step on the other path's checkpoints, on an own checkpoint out of order,
  or on the own end before every own checkpoint was met (edges joining
  checkpoints of different paths are removed up front: no solution uses them);
* degree rule: a node of U that is not an active end needs two usable
  neighbours (unvisited ones plus heads allowed to step on it), an end needs one;
  a node whose only two usable neighbours are the two heads is impossible;
  a node with exactly two usable neighbours, one of them head i, forces head i;
* components: the remainder of path i is connected inside U, so it lies in the
  component of its end e_i; hence every component of U contains an active end,
  head i must be able to step into comp(e_i), and all remaining checkpoints of
  path i lie in comp(e_i);
* bipartite parity: on bipartite graphs a path from x to y has colour balance
  (s(x) + s(y)) / 2 with s = +1 / -1 per colour; each component's balance must
  equal the sum over the path remainders it hosts.

Difficulty: rated by solver effort (search nodes needed to solve *and* prove
uniqueness, per board node) plus checkpoint sparsity, mapped onto the same
labels as ``zipsolve.difficulty`` (see ``rate``).
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .graph import ZipGraph, from_edges, grid, random_mask, remove_edges

__all__ = ["CoopPuzzle", "CoopResult", "solve", "count_solutions", "find_solutions", "check",
           "check_partial", "legal_moves", "hint", "rate", "generate", "make_coop", "hex_board",
           "random_two_paths", "BASES"]

BASES = ("grid2d", "walls", "mask", "hex")
_TIME_CHECK_EVERY = 128


# ============================================================================
# representation
# ============================================================================
@dataclass
class CoopPuzzle:
    graph: ZipGraph
    checkpoints: list[list[int]]              # [path-1 checkpoints, path-2 checkpoints]
    solution: list[list[int]] | None = None   # [path 1, path 2] when known
    base: str = "grid2d"                      # the board type the puzzle was made on

    def __post_init__(self):
        self.checkpoints = [[int(v) for v in c] for c in self.checkpoints]
        if len(self.checkpoints) != 2:
            raise ValueError("a co-op puzzle has exactly two checkpoint lists")
        if any(len(c) < 2 for c in self.checkpoints):
            raise ValueError("each path needs at least a start and an end checkpoint")
        allc = self.checkpoints[0] + self.checkpoints[1]
        if len(set(allc)) != len(allc):
            raise ValueError("checkpoints must be distinct nodes (also across the two paths)")
        n = self.graph.num_nodes
        if any(not (0 <= v < n) for v in allc):
            raise ValueError("checkpoint out of range")
        if self.solution is not None:
            self.solution = [[int(v) for v in p] for p in self.solution]

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    def owner(self) -> list[int]:
        """node -> 0 / 1 (checkpoint of that path) or -1."""
        own = [-1] * self.num_nodes
        for i, cps in enumerate(self.checkpoints):
            for v in cps:
                own[v] = i
        return own

    def check_solution(self, paths) -> str | None:
        return check(self, paths)

    def is_valid_solution(self, paths) -> bool:
        return check(self, paths) is None

    # ---- serialisation -------------------------------------------------
    def to_dict(self, include_solution: bool = True) -> dict:
        from .puzzle import _jsonable
        meta = {k: v for k, v in self.graph.meta.items() if not k.startswith("_") and k != "coop"}
        meta = json.loads(json.dumps(meta, default=_jsonable))
        coop = {"checkpoints": [list(c) for c in self.checkpoints], "base": self.base}
        if include_solution and self.solution is not None:
            coop["solution"] = {"paths": [list(p) for p in self.solution]}
        meta["coop"] = coop
        return {
            "kind": "coop",
            "coords": self.graph.coords.tolist(),
            "edges": [list(e) for e in self.graph.edges()],
            "meta": meta,
            "checkpoints": list(self.checkpoints[0]),
            "solution": None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoopPuzzle":
        meta = dict(d.get("meta") or {})
        coop = meta.pop("coop", None)
        if not isinstance(coop, dict) or "checkpoints" not in coop:
            raise ValueError("co-op puzzle needs meta.coop.checkpoints")
        cps = coop["checkpoints"]
        if not isinstance(cps, list) or len(cps) != 2 or not all(isinstance(c, list) for c in cps):
            raise ValueError("meta.coop.checkpoints must be two lists")
        sol = coop.get("solution")
        paths = sol.get("paths") if isinstance(sol, dict) else None
        g = from_edges(np.array(d["coords"], dtype=float), [tuple(e) for e in d["edges"]],
                       "coop", meta)
        return cls(g, cps, paths, coop.get("base", "grid2d"))

    @staticmethod
    def is_coop(d) -> bool:
        return isinstance(d, dict) and d.get("kind") == "coop"


# ============================================================================
# checking
# ============================================================================
def check(puzzle: CoopPuzzle, paths) -> str | None:
    """None if `paths` = [path1, path2] is a full valid solution, else a reason."""
    if not isinstance(paths, (list, tuple)) or len(paths) != 2 or \
            not all(isinstance(p, (list, tuple)) for p in paths):
        return "need exactly two paths"
    n = puzzle.num_nodes
    for i, p in enumerate(paths):
        bad = _prefix_reason(puzzle, i, p, puzzle.owner())
        if bad is not None:
            return f"path {i + 1}: {bad}"
        if not p or p[-1] != puzzle.checkpoints[i][-1]:
            return f"path {i + 1} does not end at its last checkpoint"
    if set(paths[0]) & set(paths[1]):
        return "the two paths share a node"
    covered = len(paths[0]) + len(paths[1])
    if covered != n:
        return f"the paths cover {covered} of {n} nodes"
    return None


def _prefix_reason(puzzle: CoopPuzzle, i: int, p: Sequence[int], own: list[int]) -> str | None:
    """Why the partial path `p` of path i is illegal on its own (None if legal)."""
    g = puzzle.graph
    cps = puzzle.checkpoints[i]
    n = g.num_nodes
    if not p:
        return None
    if any(not isinstance(v, (int, np.integer)) or not (0 <= v < n) for v in p):
        return "contains an invalid node id"
    if p[0] != cps[0]:
        return f"must start at its checkpoint 1"
    if len(set(p)) != len(p):
        return "revisits a node"
    nxt = 0
    for k, v in enumerate(p):
        if k and not g.has_edge(p[k - 1], v):
            return f"nodes {p[k - 1]} and {v} are not adjacent"
        if own[v] == 1 - i:
            return f"steps on the other path's checkpoint (node {v})"
        if own[v] == i:
            idx = cps.index(v)
            if idx != nxt:
                return f"reaches checkpoint {idx + 1} before checkpoint {nxt + 1}"
            nxt += 1
            if idx == len(cps) - 1 and k != len(p) - 1:
                return "continues past its final checkpoint"
    return None


def check_partial(puzzle: CoopPuzzle, paths) -> dict:
    """Per-path validation of a partial two-path state (for the two-pen UI).

    Returns {"valid", "reason", "legal", "reason_partial", "paths": [per-path info], "covered",
    "num_nodes"}; per path: {"legal", "reason", "length", "complete", "next_checkpoint", "moves"}.
    "complete" = the path reached its final checkpoint legally; "moves" = legal next nodes for
    that path's head (empty when complete or illegal).
    """
    paths = _norm_paths(puzzle, paths)
    own = puzzle.owner()
    info = []
    reasons = []
    for i, p in enumerate(paths):
        r = _prefix_reason(puzzle, i, p, own)
        if r is None:
            other = set(paths[1 - i])
            clash = [v for v in p if v in other]
            if clash and i == 1:
                r = f"crosses path 1 at node {clash[0]}"
        cps = puzzle.checkpoints[i]
        done = r is None and bool(p) and p[-1] == cps[-1]
        nxt = sum(1 for v in p if own[v] == i) if r is None else None
        info.append({"legal": r is None, "reason": r, "length": len(p), "complete": done,
                     "next_checkpoint": None if (nxt is None or nxt >= len(cps)) else nxt,
                     "moves": []})
        if r is not None:
            reasons.append(f"path {i + 1} {r}")
    legal = not reasons
    if legal:
        for i in range(2):
            info[i]["moves"] = legal_moves(puzzle, paths, i)
    full = check(puzzle, paths) if legal else reasons[0]
    covered = len(set(paths[0]) | set(paths[1]))
    return {"valid": full is None, "reason": full, "legal": legal,
            "reason_partial": reasons[0] if reasons else None,
            "paths": info, "covered": covered, "num_nodes": puzzle.num_nodes}


def _norm_paths(puzzle: CoopPuzzle, paths) -> list[list[int]]:
    paths = [list(map(int, p)) for p in (paths or [])][:2]
    while len(paths) < 2:
        paths.append([])
    return [p if p else [puzzle.checkpoints[i][0]] for i, p in enumerate(paths)]


def legal_moves(puzzle: CoopPuzzle, paths, i: int) -> list[int]:
    """Rule-legal next nodes for path i's head (no look-ahead)."""
    paths = _norm_paths(puzzle, paths)
    own = puzzle.owner()
    cps = puzzle.checkpoints[i]
    p = paths[i]
    if p[-1] == cps[-1]:
        return []
    used = set(paths[0]) | set(paths[1])
    nxt = sum(1 for v in p if own[v] == i)
    out = []
    for w in puzzle.graph.neighbors[p[-1]]:
        if w in used or own[w] == 1 - i:
            continue
        if own[w] == i and cps.index(w) != nxt:
            continue
        out.append(int(w))
    return out


# ============================================================================
# solver
# ============================================================================
@dataclass
class CoopResult:
    status: str                       # "solved" | "unsat" | "timeout"
    paths: list[list[int]] | None
    nodes_expanded: int
    seconds: float


class _Timeout(Exception):
    pass


def _colours(nbrs) -> list[int] | None:
    n = len(nbrs)
    col = [-1] * n
    for s in range(n):
        if col[s] != -1:
            continue
        col[s] = 0
        st = [s]
        while st:
            v = st.pop()
            for w in nbrs[v]:
                if col[w] == -1:
                    col[w] = 1 - col[v]
                    st.append(w)
                elif col[w] == col[v]:
                    return None
    return col


class _Coop:
    def __init__(self, puzzle: CoopPuzzle):
        self.pz = puzzle
        g = puzzle.graph
        self.n = n = g.num_nodes
        self.own = own = puzzle.owner()
        # drop edges that join checkpoints of different paths (never usable)
        self.nb = [tuple(w for w in g.neighbors[v] if not (own[v] >= 0 and own[w] >= 0 and own[v] != own[w]))
                   for v in range(n)]
        self.cps = puzzle.checkpoints
        self.cpi = [-1] * n
        for cps in self.cps:
            for k, v in enumerate(cps):
                self.cpi[v] = k
        self.ends = [c[-1] for c in self.cps]
        self.col = _colours(g.neighbors)
        self.sg = None if self.col is None else [1 if c == 0 else -1 for c in self.col]
        self.vis = bytearray(n)
        self.free = [len(ns) for ns in self.nb]
        self.paths: list[list[int]] = [[], []]
        self.nxt = [0, 0]
        self.done = [False, False]
        self.remaining = n
        self.comp = [0] * n
        self.stamp = 0
        self.expanded = 0

    # ---- state ----------------------------------------------------------
    def push(self, i: int, v: int) -> None:
        self.vis[v] = 1
        self.remaining -= 1
        for w in self.nb[v]:
            self.free[w] -= 1
        self.paths[i].append(v)
        if self.own[v] == i:
            self.nxt[i] += 1
            if self.nxt[i] == len(self.cps[i]):
                self.done[i] = True

    def pop(self, i: int) -> None:
        v = self.paths[i].pop()
        self.vis[v] = 0
        self.remaining += 1
        for w in self.nb[v]:
            self.free[w] += 1
        if self.own[v] == i:
            if self.nxt[i] == len(self.cps[i]):
                self.done[i] = False
            self.nxt[i] -= 1

    def allowed(self, i: int, w: int) -> bool:
        o = self.own[w]
        if o == -1:
            return True
        return o == i and self.cpi[w] == self.nxt[i]

    def moves(self, i: int) -> list[int]:
        if self.done[i]:
            return []
        vis = self.vis
        return [w for w in self.nb[self.paths[i][-1]] if not vis[w] and self.allowed(i, w)]

    # ---- pruning --------------------------------------------------------
    def chains_ok(self, tight: list[int], A) -> bool:
        """Forced-edge reasoning.  A non-end node of U with exactly two usable neighbours uses
        both edges, an active end with one uses it.  These forced edges are pieces of the
        final paths: they may not give a node more edges than its path degree (heads 1, ends 1,
        others 2), may not close a cycle, and one piece may not join nodes that belong to
        different paths (checkpoints / heads of path 1 and of path 2)."""
        vis, own = self.vis, self.own
        heads = {}
        for i in (0, 1):
            if not self.done[i]:
                heads[self.paths[i][-1]] = i
        ends = set(self.ends)
        par: dict[int, int] = {}
        tag: dict[int, int] = {}
        cnt: dict[int, int] = {}
        seen_e = set()

        def find(x):
            r = x
            while par.get(r, r) != r:
                r = par[r]
            while par.get(x, x) != r:
                par[x], x = r, par[x]
            return r

        def ident(x):
            if x in heads:
                return heads[x]
            return own[x]

        for v in tight:
            us = [x for x in self.nb[v] if not vis[x]]
            for i in (0, 1):
                if v in A[i] and not self.done[i]:
                    us.append(self.paths[i][-1])
            for x in us:
                e = (v, x) if v < x else (x, v)
                if e in seen_e:
                    continue
                seen_e.add(e)
                for y in e:
                    c = cnt.get(y, 0) + 1
                    if c > (1 if (y in heads or y in ends) else 2):
                        return False
                    cnt[y] = c
                ra, rb = find(v), find(x)
                if ra == rb:
                    return False            # cycle
                ta = tag.get(ra, ident(v) if ra == v else -1)
                tb = tag.get(rb, ident(x) if rb == x else -1)
                if ta >= 0 and tb >= 0 and ta != tb:
                    return False            # would join path 1 and path 2
                par[ra] = rb
                tag[rb] = ta if ta >= 0 else tb
        return True

    def analyse(self):
        """None if the state is dead, else (head index, ordered move list)."""
        vis, free, own, ends, done = self.vis, self.free, self.own, self.ends, self.done
        if self.remaining == 0:
            return None
        A = [self.moves(0), self.moves(1)]
        extra = {}
        for i in (0, 1):
            for w in A[i]:
                extra[w] = extra.get(w, 0) + 1
        # degree rule + forced moves
        forced = [None, None]
        for i in (0, 1):
            if done[i]:
                continue
            for w in A[i]:
                d = free[w] + extra[w]
                if w == ends[i]:
                    if d == 1:
                        if forced[i] is not None and forced[i] != w:
                            return None
                        forced[i] = w
                elif d == 2:
                    if extra[w] == 2:          # only the two heads: would join the paths
                        return None
                    if forced[i] is not None and forced[i] != w:
                        return None
                    forced[i] = w
        active_ends = [ends[i] for i in (0, 1) if not done[i]]
        tight = []
        for v in range(self.n):
            if vis[v]:
                continue
            d = free[v] + extra.get(v, 0)
            if d > 2:
                continue
            if d == 0:
                return None
            if v in active_ends:
                if d == 1:
                    tight.append(v)
            elif d == 1:
                return None
            else:
                tight.append(v)
        if tight and not self.chains_ok(tight, A):
            return None
        # components of U, flood-filled from the active ends (labels are fresh per call)
        self.stamp += 2
        base = self.stamp
        comp = self.comp
        nb = self.nb
        sg = self.sg
        ncomp = 0
        seen_nodes = 0
        comp_bal = []
        for s in active_ends:
            if comp[s] >= base:
                continue
            cid = base + ncomp
            ncomp += 1
            comp[s] = cid
            stack = [s]
            b = 0
            while stack:
                u = stack.pop()
                seen_nodes += 1
                if sg is not None:
                    b += sg[u]
                for w in nb[u]:
                    if not vis[w] and comp[w] != cid:
                        comp[w] = cid
                        stack.append(w)
            comp_bal.append(b)
        if seen_nodes != self.remaining:
            return None            # a component without any active end
        # heads must reach the component of their end; remaining checkpoints must lie in it
        want = [0] * ncomp
        for i in (0, 1):
            if done[i]:
                continue
            ce = comp[ends[i]]
            if not any(comp[w] == ce for w in A[i]):
                return None
            cps = self.cps[i]
            for k in range(self.nxt[i], len(cps)):
                if comp[cps[k]] != ce:
                    return None
            if sg is not None:
                h = self.paths[i][-1]
                want[ce - base] += (-sg[h] + sg[ends[i]]) // 2
        if sg is not None and want != comp_bal:
            return None
        # choose the head to extend
        cand = [i for i in (0, 1) if not done[i]]
        for i in cand:
            if forced[i] is not None:
                return i, [forced[i]]
        i = min(cand, key=lambda k: len(A[k]))
        mv = A[i]
        if len(mv) > 1:
            nxt_cp = self.cps[i][self.nxt[i]]
            mv = sorted(mv, key=lambda w: (w != nxt_cp, free[w]))
        return i, mv

    # ---- search ---------------------------------------------------------
    def run(self, limit: int, time_limit: float | None, prefix=None):
        """Enumerate up to `limit` solutions. Returns (solutions, status)."""
        deadline = None if time_limit is None else time.perf_counter() + time_limit
        sols: list[list[list[int]]] = []
        prefix = prefix or [[self.cps[0][0]], [self.cps[1][0]]]
        for i in (0, 1):
            for v in prefix[i]:
                self.push(i, v)
        if self.remaining == 0:
            return ([[list(p) for p in self.paths]] if all(self.done) else []), "complete"
        stack = []        # frames: [head, moves, pos, pushed]
        first = self.analyse()
        if first is None:
            return sols, "complete"
        stack.append([first[0], first[1], 0, False])
        steps = 0
        while stack:
            fr = stack[-1]
            if fr[3]:
                self.pop(fr[0])
                fr[3] = False
            if fr[2] >= len(fr[1]):
                stack.pop()
                continue
            v = fr[1][fr[2]]
            fr[2] += 1
            self.push(fr[0], v)
            fr[3] = True
            self.expanded += 1
            steps += 1
            if deadline is not None and steps % _TIME_CHECK_EVERY == 0 and time.perf_counter() > deadline:
                return sols, "timeout"
            if self.remaining == 0:
                if all(self.done):
                    sols.append([list(p) for p in self.paths])
                    if len(sols) >= limit:
                        return sols, "limit"
                continue
            nx = self.analyse()
            if nx is not None:
                stack.append([nx[0], nx[1], 0, False])
        return sols, "complete"


def _prefix_ok(puzzle: CoopPuzzle, paths) -> str | None:
    own = puzzle.owner()
    for i, p in enumerate(paths):
        r = _prefix_reason(puzzle, i, p, own)
        if r is not None:
            return f"path {i + 1} {r}"
    if set(paths[0]) & set(paths[1]):
        return "the two paths share a node"
    return None


def find_solutions(puzzle: CoopPuzzle, limit: int = 2, time_limit: float | None = None,
                   prefix=None):
    """Up to `limit` solutions ([path1, path2] each) and a status:
    "complete" (all found), "limit" (stopped at `limit`) or "timeout"."""
    s = _Coop(puzzle)
    if prefix is not None:
        prefix = _norm_paths(puzzle, prefix)
        if _prefix_ok(puzzle, prefix) is not None:
            return [], "complete"
    return s.run(limit, time_limit, prefix)


def count_solutions(puzzle: CoopPuzzle, limit: int = 2, time_limit: float | None = None):
    sols, status = find_solutions(puzzle, limit, time_limit)
    return len(sols), status


def solve(puzzle: CoopPuzzle, time_limit: float | None = None, prefix=None) -> CoopResult:
    t0 = time.perf_counter()
    s = _Coop(puzzle)
    if prefix is not None:
        prefix = _norm_paths(puzzle, prefix)
        if _prefix_ok(puzzle, prefix) is not None:
            return CoopResult("unsat", None, 0, time.perf_counter() - t0)
    sols, status = s.run(1, time_limit, prefix)
    dt = time.perf_counter() - t0
    if sols:
        return CoopResult("solved", sols[0], s.expanded, dt)
    return CoopResult("timeout" if status == "timeout" else "unsat", None, s.expanded, dt)


def brute_force(puzzle: CoopPuzzle, limit: int = 10**9) -> list[list[list[int]]]:
    """Every solution by plain enumeration (tests only; tiny boards)."""
    g = puzzle.graph
    n = g.num_nodes
    out = []

    def paths_from(i, used):
        cps = puzzle.checkpoints[i]
        res = []

        def rec(p, seen):
            if p[-1] == cps[-1]:
                res.append(list(p))
                return
            for w in g.neighbors[p[-1]]:
                if w not in seen and w not in used:
                    p.append(w)
                    seen.add(w)
                    rec(p, seen)
                    seen.discard(w)
                    p.pop()
        s = cps[0]
        if s not in used:
            rec([s], {s})
        return res

    for p1 in paths_from(0, set()):
        for p2 in paths_from(1, set(p1)):
            if check(puzzle, [p1, p2]) is None:
                out.append([p1, p2])
                if len(out) >= limit:
                    return out
    return out


# ============================================================================
# hints
# ============================================================================
def hint(puzzle: CoopPuzzle, paths, solution=None, time_limit: float = 5.0,
         path_index: int | None = None) -> dict:
    """Next move for a two-path partial state.

    Returns {"status": "next", "path_index": i, "next": v, "keep": [k1, k2]} when the state can
    be completed (keep = the current lengths), {"status": "backtrack", "path_index": i,
    "keep": [k1, k2], "next": v} when the paths must first be cut back to lengths keep,
    {"status": "done"} when solved, or {"status": "invalid" | "timeout" | "unsat", "message"}.
    `path_index` (0/1): the pen the player is holding; the hint is for that path when it can
    move, else for the other one.
    """
    t0 = time.perf_counter()
    paths = _norm_paths(puzzle, paths)
    bad = _prefix_ok(puzzle, paths)
    if bad is not None:
        return {"status": "invalid", "message": bad}
    if check(puzzle, paths) is None:
        return {"status": "done", "message": "Already solved!"}
    keep = [len(paths[0]), len(paths[1])]

    def pick(sol, keep):
        order = [path_index, 1 - path_index] if path_index in (0, 1) else \
            sorted((0, 1), key=lambda i: keep[i] / max(1, len(sol[i])))
        for i in order:
            if keep[i] < len(sol[i]):
                return i, sol[i][keep[i]]
        return None, None

    def out(status, sol, keep, source, nodes=0):
        i, v = pick(sol, keep)
        return {"status": status, "path_index": i, "next": v, "keep": list(keep), "source": source,
                "nodes_expanded": nodes, "seconds": time.perf_counter() - t0}

    if solution is not None and check(puzzle, solution) is None:
        lcp = [_lcp(paths[i], solution[i]) for i in (0, 1)]
        if lcp == keep:
            return out("next", solution, keep, "solution")
    r = solve(puzzle, time_limit=max(0.05, time_limit - (time.perf_counter() - t0)), prefix=paths)
    if r.status == "solved":
        return out("next", r.paths, keep, "solver", r.nodes_expanded)
    if r.status == "timeout" and solution is None:
        return {"status": "timeout", "message": "The solver ran out of time on this position.",
                "nodes_expanded": r.nodes_expanded, "seconds": time.perf_counter() - t0}
    sol = solution if (solution is not None and check(puzzle, solution) is None) else None
    if sol is None:
        r2 = solve(puzzle, time_limit=max(0.05, time_limit - (time.perf_counter() - t0)))
        if r2.status != "solved":
            return {"status": "unsat" if r2.status == "unsat" else "timeout",
                    "message": "No solution found." if r2.status == "unsat" else
                    "The solver ran out of time on this position.",
                    "seconds": time.perf_counter() - t0}
        sol = r2.paths
    k = [max(1, _lcp(paths[i], sol[i])) for i in (0, 1)]
    return out("backtrack", sol, k, "solution" if solution is not None else "solver")


def _lcp(a, b) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


# ============================================================================
# difficulty
# ============================================================================
def rate(puzzle: CoopPuzzle, time_limit: float = 2.0) -> dict:
    """Difficulty from solver effort.

    effort = log2(1 + search nodes to find all (<= 2) solutions) relative to the board size,
    plus a bonus for sparse checkpoints and larger boards.  The score is on the 0..100 scale of
    ``zipsolve.difficulty`` and uses its labels.  The exact search count is deterministic.
    """
    from .difficulty import label_for
    s = _Coop(puzzle)
    sols, status = s.run(2, time_limit)
    n = puzzle.num_nodes
    nodes = s.expanded
    # search overhead beyond just walking the solution (n - 2 pushes)
    overhead = max(0.0, nodes - (n - 2)) / max(1, n)
    density = (len(puzzle.checkpoints[0]) + len(puzzle.checkpoints[1])) / n
    effort = 12.0 * math.log2(1.0 + overhead) + 30.0 * max(0.0, 0.45 - density) + 0.15 * n
    if status == "timeout":
        effort += 20.0
    score = round(100.0 * (1.0 - math.exp(-effort / 48.0)), 1)
    return {"score": score, "label": label_for(score),
            "features": {"nodes": n, "search_nodes": nodes, "cp_density": round(density, 3),
                         "solutions": len(sols), "status": status}}


# ============================================================================
# generation
# ============================================================================
def hex_board(h: int, w: int) -> ZipGraph:
    """Roughly rectangular hex board in axial (q, r) coords (meta.layout = "hex")."""
    cells = [(q - r // 2, r) for r in range(h) for q in range(w)]
    ids = {c: k for k, c in enumerate(cells)}
    edges = []
    for (q, r), k in ids.items():
        for dq, dr in ((1, 0), (0, 1), (-1, 1)):
            j = ids.get((q + dq, r + dr))
            if j is not None:
                edges.append((k, j))
    return from_edges(np.array(cells, dtype=float), edges, "hex", {"layout": "hex", "shape": (h, w)})


def _board(base: str, shape, rng: np.random.Generator, walls_frac: float, fill: float) -> ZipGraph:
    h, w = shape
    if base in ("grid2d", "grid", "walls"):
        return grid(h, w)          # walls are added later (on edges the solution does not use)
    if base == "mask":
        for _ in range(50):
            g = random_mask((h, w), fill=fill, rng=rng)
            if g.num_nodes >= 6:
                return g
        raise ValueError("could not build a mask board")
    if base == "hex":
        from . import graph as _g
        for name in ("hex_grid", "hex_board", "hexgrid"):
            f = getattr(_g, name, None)
            if f is not None:
                try:
                    return f(h, w)
                except Exception:  # noqa: BLE001  (unknown signature: use our own builder)
                    break
        return hex_board(h, w)
    raise ValueError(f"unknown co-op base {base!r}; expected one of {BASES}")


def random_two_paths(graph: ZipGraph, rng=None, time_limit: float = 5.0, min_frac: float = 0.3,
                     moves: int | None = None) -> list[list[int]] | None:
    """A random pair of vertex-disjoint paths covering `graph`.

    Starts from a random Hamiltonian path split in two, then mixes with "backbite" moves that
    may also cut into the other path (endpoint t of path A links to x in path B: A absorbs
    one side of B), so the two paths need not be the halves of one Hamiltonian path.
    Every path keeps at least `min_frac` of the nodes.
    """
    from .generator import random_hamiltonian_path
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    prng = random.Random(int(rng.integers(2**62)))
    n = graph.num_nodes
    if n < 4:
        return None
    ham = random_hamiltonian_path(graph, rng, time_limit=time_limit)
    if ham is None:
        return None
    lo = max(2, int(math.ceil(min_frac * n)))
    if n - lo < lo:
        lo = n // 2
    k = prng.randint(lo, n - lo)
    P = [ham[:k], ham[k:]]
    nbrs = graph.neighbors
    moves = moves if moves is not None else int(20 * n * math.sqrt(n))
    where = [0] * n
    for i in (0, 1):
        for v in P[i]:
            where[v] = i
    for _ in range(moves):
        a = prng.randrange(2)
        A = P[a]
        flip = prng.random() < 0.5
        if flip:
            A.reverse()
        t = A[-1]
        x = nbrs[t][prng.randrange(len(nbrs[t]))]
        if where[x] == a:
            if len(A) >= 2 and x != A[-2]:
                i = A.index(x)
                A[i + 1:] = A[:i:-1]
        else:
            B = P[1 - a]
            i = B.index(x)
            if prng.random() < 0.5:
                gain, rest = B[i:], B[:i]
            else:
                gain, rest = B[:i + 1][::-1], B[i + 1:]
            if len(rest) >= lo and len(A) + len(gain) >= lo:
                A.extend(gain)
                for v in gain:
                    where[v] = a
                P[1 - a] = rest
        if flip:
            P[a].reverse()
    return [list(map(int, P[0])), list(map(int, P[1]))]


def _num_cps(n: int, rng: np.random.Generator) -> int:
    from .generator import default_num_checkpoints
    return max(4, default_num_checkpoints(n, rng))


def generate(graph: ZipGraph, paths: list[list[int]], num_checkpoints: int | None = None, rng=None,
             unique: bool = False, time_limit: float = 10.0, base: str = "grid2d") -> CoopPuzzle:
    """Place checkpoints on the given two-path solution (adding clues until unique if asked)."""
    from .generator import GenerationTimeout, place_checkpoints
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    deadline = time.perf_counter() + time_limit
    n = graph.num_nodes
    k = num_checkpoints or _num_cps(n, rng)
    if num_checkpoints is None and _colours(graph.neighbors) is None:
        k = max(k, int(round(0.35 * n)))   # non-bipartite boards (hex) are much looser
    k = max(4, k)
    k0 = max(2, min(len(paths[0]), int(round(k * len(paths[0]) / n))))
    k1 = max(2, min(len(paths[1]), k - k0))
    cps = [place_checkpoints(paths[0], k0, rng), place_checkpoints(paths[1], k1, rng)]
    pz = CoopPuzzle(graph, cps, [list(p) for p in paths], base)
    if not unique:
        return pz
    pos = [{v: j for j, v in enumerate(p)} for p in paths]
    # short checks: a timeout adds a clue, which makes the next check easier
    per_check = max(0.2, min(0.4, time_limit / 20))
    while True:
        left = deadline - time.perf_counter()
        if left <= 0:
            raise GenerationTimeout("proving the co-op puzzle unique exceeded its time limit")
        sols, status = find_solutions(pz, 2, min(left, per_check))
        if len(sols) <= 1 and status == "complete":
            return pz
        others = [s for s in sols if s != paths]
        cand = []  # (path index, position)
        for i in (0, 1):
            have = set(pz.checkpoints[i])
            cand += [(i, j) for j in range(1, len(paths[i]) - 1) if paths[i][j] not in have]
        if not cand:
            return pz
        chosen = None
        if others:
            alt = others[0]
            killers = []
            for i, j in cand:
                trial = _with_cp(pz, i, j, pos, paths)
                if check(trial, alt) is not None:
                    killers.append((i, j))
            if killers:
                def spread(c):
                    i, j = c
                    return min(abs(j - pos[i][v]) for v in pz.checkpoints[i])
                d = np.array([spread(c) for c in killers], dtype=float)
                top = np.flatnonzero(d >= np.quantile(d, 0.75))
                chosen = killers[int(top[int(rng.integers(len(top)))])]
        if chosen is None:
            best = None
            for i in (0, 1):
                idx = sorted(pos[i][v] for v in pz.checkpoints[i])
                for p_, q_ in zip(idx, idx[1:]):
                    if q_ - p_ > 1 and (best is None or q_ - p_ > best[0]):
                        best = (q_ - p_, i, p_, q_)
            _, i, p_, q_ = best
            chosen = (i, int(rng.integers(p_ + 1, q_)))
        pz = _with_cp(pz, chosen[0], chosen[1], pos, paths)


def _with_cp(pz: CoopPuzzle, i: int, j: int, pos, paths) -> CoopPuzzle:
    cps = [list(c) for c in pz.checkpoints]
    idx = sorted([pos[i][v] for v in cps[i]] + [j])
    cps[i] = [paths[i][x] for x in idx]
    return CoopPuzzle(pz.graph, cps, pz.solution, pz.base)


def make_coop(size=6, num_checkpoints: int | None = None, rng=None, unique: bool = True,
              base: str = "grid2d", time_limit: float = 10.0, walls_frac: float = 0.25,
              fill: float = 0.8, max_tries: int = 30) -> CoopPuzzle:
    """Random co-op puzzle. size = n (n x n) or (h, w); base in BASES."""
    from .generator import GenerationTimeout
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    shape = (int(size), int(size)) if isinstance(size, (int, np.integer)) else tuple(int(s) for s in size)
    if len(shape) != 2:
        raise ValueError("co-op boards are 2D: size = n or (h, w)")
    deadline = time.perf_counter() + time_limit
    last = None
    for _ in range(max_tries):
        left = deadline - time.perf_counter()
        if left <= 0:
            break
        g = _board(base, shape, rng, walls_frac, fill)
        paths = random_two_paths(g, rng, time_limit=min(left, max(0.5, time_limit / 4)))
        if paths is None:
            continue
        if base == "walls":
            used = {frozenset(e) for p in paths for e in zip(p, p[1:])}
            free = [e for e in g.edges() if frozenset(e) not in used]
            m = int(round(walls_frac * len(free)))
            if m:
                pick = rng.choice(len(free), size=m, replace=False)
                g = remove_edges(g, [free[x] for x in sorted(pick)])
        try:
            return generate(g, paths, num_checkpoints, rng, unique,
                            max(0.1, deadline - time.perf_counter()), base)
        except GenerationTimeout as e:
            last = e
            continue
    if last is not None:
        raise last
    raise GenerationTimeout("co-op generation exceeded its time limit")


# ============================================================================
# web API helpers (used by zipsolve/app/server.py for kind "coop"; raise ValueError on bad input)
# ============================================================================
COOP_SIZE = (3, 10, 6)          # min, max, default board side (hex: max 8)
COOP_MAX_NODES = 150


def api_load(d: dict) -> CoopPuzzle:
    """Client puzzle dict -> CoopPuzzle (any embedded solution is ignored)."""
    if not isinstance(d, dict):
        raise ValueError("puzzle must be an object")
    coords, edges = d.get("coords"), d.get("edges")
    if not isinstance(coords, list) or not coords or not all(isinstance(c, list) for c in coords):
        raise ValueError("puzzle.coords must be a non-empty list of lists")
    if len(coords) > COOP_MAX_NODES:
        raise ValueError(f"co-op puzzle too large ({len(coords)} > {COOP_MAX_NODES} nodes)")
    n = len(coords)
    if not isinstance(edges, list) or any(
            not isinstance(e, list) or len(e) != 2 or not all(isinstance(v, int) and 0 <= v < n for v in e)
            for e in edges):
        raise ValueError("puzzle.edges must be [u, v] pairs of valid node ids")
    meta = d.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("coop"), dict):
        raise ValueError("co-op puzzle needs meta.coop")
    cps = meta["coop"].get("checkpoints")
    if not isinstance(cps, list) or len(cps) != 2 or not all(
            isinstance(c, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in c)
            for c in cps):
        raise ValueError("meta.coop.checkpoints must be two lists of node ids")
    d = dict(d)
    d["meta"] = dict(meta)
    d["meta"]["coop"] = {k: v for k, v in meta["coop"].items() if k != "solution"}
    return CoopPuzzle.from_dict(d)


def api_generate(size=None, num_checkpoints=None, seed=None, unique=True, options=None,
                 time_limit: float = 15.0) -> tuple[dict, list]:
    """/api/generate for kind "coop" -> (response without "id", solution paths).

    options: {"base": "grid2d"|"walls"|"mask"|"hex", "walls_frac": float, "fill": float}.
    """
    options = dict(options or {})
    base = options.get("base") or "grid2d"
    if base not in BASES:
        raise ValueError(f"unknown co-op base {base!r}; expected one of {BASES}")
    lo, hi, dflt = COOP_SIZE
    if base == "hex":
        hi = 8
    if size is None:
        size = dflt
    if isinstance(size, (list, tuple)):
        if len(size) != 2:
            raise ValueError("co-op size must be n or [h, w]")
        size = tuple(max(lo, min(hi, int(s))) for s in size)
    else:
        size = max(lo, min(hi, int(size)))
    kw = {}
    if options.get("walls_frac") is not None:
        kw["walls_frac"] = max(0.0, min(1.0, float(options["walls_frac"])))
    if options.get("fill") is not None:
        kw["fill"] = max(0.5, min(1.0, float(options["fill"])))
    ncp = None if not num_checkpoints else max(4, int(num_checkpoints))
    t0 = time.perf_counter()
    p = make_coop(size, ncp, rng=seed, unique=bool(unique), base=base, time_limit=time_limit, **kw)
    secs = time.perf_counter() - t0
    rating = None
    if unique:
        try:
            r = rate(p, time_limit=1.0)
            rating = {"label": r["label"], "score": r["score"]}
        except Exception:  # noqa: BLE001  (decoration only)
            rating = None
    d = p.to_dict(include_solution=False)
    out = {"seed": seed, "kind": "coop", "base": base, "size": size, "unique": bool(unique),
           "num_nodes": p.num_nodes, "seconds": secs, "puzzle": d, "rating": rating,
           "source": "generator"}
    return out, p.solution


def api_solve(puzzle: CoopPuzzle, time_limit: float = 10.0, start_paths=None) -> dict:
    """/api/solve/exact for kind "coop"."""
    if start_paths:
        bad = _prefix_ok(puzzle, _norm_paths(puzzle, start_paths))
        if bad is not None:
            raise ValueError(f"start_paths: {bad}")
    r = solve(puzzle, time_limit=time_limit, prefix=start_paths or None)
    return {"status": r.status, "paths": r.paths, "path": None, "nodes_expanded": r.nodes_expanded,
            "seconds": r.seconds, "solved": r.status == "solved", "kind": "coop"}
