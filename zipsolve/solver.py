"""Exact backtracking solver for Zip puzzles on arbitrary graphs.

A Zip puzzle asks for a Hamiltonian path that starts at ``checkpoints[0]``,
ends at ``checkpoints[-1]`` and meets the checkpoints in order.  The solver is
an iterative depth-first search (no Python recursion, so graphs with many
hundreds of nodes are fine) over the path head, with the following *sound*
prunings (none of them can discard a real solution):

1. **Checkpoint order** – never step onto a checkpoint other than the next
   required one; never step onto the final checkpoint before every other node
   has been visited.
2. **Degree / dead-end rule** – let ``deg(u)`` be the number of unvisited
   neighbours of an unvisited node ``u``.  A node that is not the final
   checkpoint must be entered and left, so it needs two usable neighbours
   (unvisited ones, plus the head if adjacent).  The final checkpoint needs
   one.  A non-final node adjacent to the head with ``deg == 1`` *must* be the
   next move ("forced move"); two different forced moves prune the branch.
   Only nodes around the old and new head change status, so this is checked
   incrementally in O(degree) per move.
3. **Connectivity** – the unvisited nodes (excluding the head) must form one
   connected component (the path leaves the head once and never returns).
   Since the previous unvisited set was connected, it is enough to check that
   all unvisited neighbours of the new head are still mutually reachable; a
   BFS with early termination does this, usually touching only a few nodes.
   This also covers checkpoint reachability.
4. **Bipartite parity** – on bipartite graphs (all grid variants) a path
   alternates colours, so the colour balance of the whole graph fixes whether
   a solution can exist at all (checked once, it is invariant during search).
5. **Articulation-point reasoning** (Tarjan, every move by default) on the
   graph induced by the unvisited nodes plus the head.  For every cut vertex
   ``a``: removing it may leave at most two parts; if two, the head must be on
   one side and the final checkpoint on the other (and ``a`` is not the end);
   the remaining checkpoints must be ordered head-side < ``a`` < end-side; and
   on bipartite graphs ``a`` must have the colour implied by its forced path
   position, and both sides must have a feasible colour balance.  This is very
   effective on island graphs (bridges) and on narrow corridors.
6. **Forced-edge chains** – an unvisited non-final node with exactly two
   usable neighbours must use both edges (the final checkpoint with one must
   use it).  These forced edges are fragments of the final path: no node may
   get more forced edges than its path degree, they may not close a cycle, a
   fragment joining head and end must cover everything, and checkpoints along
   a fragment must carry consecutive labels (increasing away from the head).
   A forced edge at the head is a forced move.

7. **One-way arcs and precedence** (``graph.meta["arcs"]`` /
   ``graph.meta["precedence"]``, see ``zipsolve.graph``).  Moves honour them
   (an arc {u, v} is usable only u -> v; a node with prerequisites can only be
   entered once they are all visited).  Every pruning above reasons on the
   *undirected* graph without precedence, which is a relaxation (a solution of
   the constrained puzzle is also one of the relaxed puzzle), so it stays
   sound; forced moves are dropped when the constraints forbid them (which
   prunes the branch).  On top of that, with arcs every unvisited node needs a
   usable in-neighbour (unvisited or the head) and a usable out-neighbour
   (unvisited, a different one), the final checkpoint an in-neighbour, and a
   node whose only in-neighbour is the head is a forced move.  With
   precedence the articulation-point reasoning also demands key side <= door
   side (head side < cut vertex < end side).

Move ordering (default): the next checkpoint first, otherwise lowest
``free_degree + distance_to_next_checkpoint`` (Warnsdorff plus a pull towards
the next checkpoint; the distance term matters a lot on 3D/4D grids).
``solve`` additionally uses randomised restarts with growing node budgets,
which tames the heavy-tailed running time of DFS while staying complete.

Public API
----------
``solve(puzzle, time_limit=None, move_order=None) -> SolveResult``
``count_solutions(puzzle, limit=2, time_limit=None) -> (count, status)``
``is_dead_end(graph, visited, head, next_checkpoint_idx, checkpoints) -> bool``
``solve_from_prefix(puzzle, prefix, time_limit=None, move_order=None) -> SolveResult``
``label_moves(puzzle, prefix, time_limit_per_move) -> {move: "win"|"lose"|"unknown"}``
``check_prefix(puzzle, prefix) -> str | None`` / ``legal_moves(puzzle, prefix)``

Move-order hook
---------------
``move_order(head, candidates, visited, next_cp_idx) -> Sequence[int]``

* ``head``: current path head (node id).
* ``candidates``: list of legal next nodes (already filtered by all prunings,
  already sorted by the default heuristic: next checkpoint first, then
  fewest free neighbours – Warnsdorff).  If a move is forced this list has a
  single element and the hook is *not* called.
* ``visited``: read-only ``np.ndarray`` of bool, shape ``(n,)`` – a live
  view of the solver's internal state (copy it if you keep it).
* ``next_cp_idx``: index into ``puzzle.checkpoints`` of the next checkpoint
  still to be reached.

It returns the candidates in the order they should be tried.  Returning a
subset makes the search incomplete (fine for a heuristic policy, but then an
``"unsat"`` result only means "not found in that subset").  Anything that is
not one of ``candidates`` is ignored.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .graph import ZipGraph, blocked_steps, precedence_of, prerequisites
from .puzzle import Puzzle

MoveOrder = Callable[[int, list, np.ndarray, int], Sequence[int]]

_TIME_CHECK_EVERY = 256
_RESTART_GROWTH = 1.5
_RESTART_BASE = 200


@dataclass
class SolveResult:
    status: str                 # "solved" | "unsat" | "timeout"
    path: list[int] | None
    nodes_expanded: int
    seconds: float


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bipartite_colours(graph: ZipGraph) -> list[int] | None:
    """0/1 colouring if the graph is bipartite, else None."""
    n = graph.num_nodes
    nbrs = graph.neighbors
    col = [-1] * n
    for s in range(n):
        if col[s] != -1:
            continue
        col[s] = 0
        stack = [s]
        while stack:
            v = stack.pop()
            cv = col[v]
            for w in nbrs[v]:
                if col[w] == -1:
                    col[w] = 1 - cv
                    stack.append(w)
                elif col[w] == cv:
                    return None
    return col


def _bfs_dist(graph: ZipGraph, src: int) -> list[int]:
    n = graph.num_nodes
    nbrs = graph.neighbors
    d = [n] * n
    d[src] = 0
    q = deque([src])
    while q:
        v = q.popleft()
        dv = d[v] + 1
        for w in nbrs[v]:
            if d[w] > dv:
                d[w] = dv
                q.append(w)
    return d


class _Search:
    """Mutable search state + pruning checks.  One instance per solve call."""

    def __init__(self, graph: ZipGraph, checkpoints: Sequence[int],
                 deep: bool = True):
        self.graph = graph
        self.n = n = graph.num_nodes
        self.nbrs = graph.neighbors
        self.cps = [int(c) for c in checkpoints]
        self.end = self.cps[-1]
        self.cpi = [-1] * n
        for i, c in enumerate(self.cps):
            self.cpi[c] = i
        self.col = _bipartite_colours(graph)
        # one-way arcs (node -> forbidden next nodes) and precedence (node -> prerequisites)
        self.blocked = blocked_steps(graph)
        self.pre = prerequisites(graph)
        self.prec = precedence_of(graph)
        self.constrained = self.blocked is not None or self.pre is not None
        self.deep = deep
        self.chains = True
        self.dyn = False        # BFS from next checkpoint (sound, but did not pay off in benchmarks)
        self.dd = None
        self.dyn_order = False  # order moves by the dynamic (not static) distance
        # unvisited nodes with at most 2 unvisited neighbours (chain reasoning)
        self.low: set[int] = {v for v in range(n) if len(self.nbrs[v]) <= 2}
        self.cp_pref = True     # try the next checkpoint first
        # static BFS distances to every checkpoint (move-ordering heuristic)
        self.dist = [_bfs_dist(graph, c) for c in self.cps]
        self.dist_weight = 1.0
        self.deg_weight = 1.0
        self.rng = None         # random tie-breaking (used by restarts)
        self.visited = bytearray(n)
        self.deg = [len(ns) for ns in self.nbrs]
        self.path: list[int] = []
        self.nxt = 0
        self.remaining = n
        # scratch arrays for BFS with stamps (avoid O(n) resets)
        self.mark = [0] * n
        self.stamp = 0

    def rebuild_low(self) -> None:
        vis, deg = self.visited, self.deg
        self.low = {v for v in range(self.n) if not vis[v] and deg[v] <= 2}

    # ---- state updates ----------------------------------------------------
    def push(self, h: int) -> None:
        self.visited[h] = 1
        self.path.append(h)
        deg = self.deg
        low = self.low
        low.discard(h)
        vis = self.visited
        for x in self.nbrs[h]:
            deg[x] -= 1
            if deg[x] == 2 and not vis[x]:
                low.add(x)
        self.remaining -= 1
        if self.cpi[h] == self.nxt:
            self.nxt += 1

    def pop(self) -> None:
        h = self.path.pop()
        self.visited[h] = 0
        deg = self.deg
        low = self.low
        for x in self.nbrs[h]:
            deg[x] += 1
            if deg[x] == 3:
                low.discard(x)
        if deg[h] <= 2:
            low.add(h)
        self.remaining += 1
        if self.cpi[h] >= 0 and self.cpi[h] == self.nxt - 1:
            self.nxt -= 1

    # ---- constraints (arcs / precedence) -----------------------------------
    def ok_step(self, h: int, w: int) -> bool:
        """Move h -> w allowed by the arcs and by precedence (adjacency not checked)."""
        bl = self.blocked
        if bl is not None:
            b = bl.get(h)
            if b is not None and w in b:
                return False
        pre = self.pre
        if pre is not None:
            ks = pre.get(w)
            if ks is not None:
                vis = self.visited
                for k in ks:
                    if not vis[k]:
                        return False
        return True

    def dir_status(self, x: int, h: int) -> int:
        """Directed degree rule for an unvisited node x (only with arcs).

        -2: dead; 1: x must be the next move (its only usable in-neighbour is
        the head); 0: fine.
        """
        bl, vis = self.blocked, self.visited
        empty = ()
        bx = bl.get(x, empty)
        ins = []
        outs = []
        for y in self.nbrs[x]:
            if vis[y] and y != h:
                continue
            if x not in bl.get(y, empty):
                ins.append(y)
            if y != h and y not in bx:
                outs.append(y)
        if x == self.end:
            if not ins:
                return -2
            if len(ins) == 1 and ins[0] == h:
                return -2 if self.remaining > 1 else 1
            return 0
        if not ins or not outs:
            return -2
        if len(ins) == 1:
            if len(outs) == 1 and ins[0] == outs[0]:
                return -2
            if ins[0] == h:
                return 1
        return 0

    # ---- checks -----------------------------------------------------------
    def global_check(self) -> bool:
        """Full (non-incremental) check of the current state. True = viable."""
        n, nbrs, vis, deg = self.n, self.nbrs, self.visited, self.deg
        if not self.path:
            return False
        h = self.path[-1]
        # checkpoints before nxt visited, from nxt on unvisited
        for i, c in enumerate(self.cps):
            if (i < self.nxt) != bool(vis[c]):
                return False
        for a, b in self.prec:          # a door already visited without its key
            if vis[b] and not vis[a]:
                return False
        if self.remaining == 0:
            return h == self.end
        if vis[self.end]:
            return False
        hn = set(nbrs[h])
        forced = -1
        for u in range(n):
            if vis[u]:
                continue
            d = deg[u]
            adj = u in hn
            if u == self.end:
                if d + adj < 1 or (d == 0 and self.remaining > 1):
                    return False
            else:
                if d + adj < 2:
                    return False
                if adj and d == 1:
                    if forced >= 0:
                        return False
                    forced = u
        if self.blocked is not None:
            for u in range(n):
                if vis[u]:
                    continue
                st = self.dir_status(u, h)
                if st == -2:
                    return False
                if st == 1:
                    if forced >= 0 and forced != u:
                        return False
                    forced = u
        if forced >= 0:
            c = self.cpi[forced]
            if c >= 0 and c != self.nxt:
                return False
            if not self.ok_step(h, forced):
                return False
        # connectivity of the unvisited set, head adjacent to it
        if not any(not vis[x] for x in nbrs[h]):
            return False
        start = next(u for u in range(n) if not vis[u])
        seen = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for w in nbrs[v]:
                if not vis[w] and w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(seen) != self.remaining:
            return False
        # bipartite colour balance of head + unvisited
        col = self.col
        if col is not None:
            ch = col[h]
            total = self.remaining + 1
            if col[self.end] != ch ^ ((total - 1) & 1):
                return False
            s = 1 + sum(1 if col[u] == ch else -1 for u in range(n) if not vis[u])
            if s != (total & 1):
                return False
        self.rebuild_low()
        if self.chains:
            f2 = self.chain_check(h)
            if f2 == -2:
                return False
            if f2 >= 0:
                if forced >= 0 and forced != f2:
                    return False
                c = self.cpi[f2]
                if c >= 0 and c != self.nxt:
                    return False
        if self.dyn and not self.segment_check(h):
            return False
        if self.deep and not self.deep_check():
            return False
        return True

    def after_move(self, p: int, h: int) -> int:
        """Incremental check after moving the head from p to h.

        Returns -2 if the state is dead, the forced next node if there is
        one, else -1.
        """
        if self.remaining == 0:
            return -1
        nbrs, vis, deg, end = self.nbrs, self.visited, self.deg, self.end
        self.stamp += 1
        st = self.stamp
        mark = self.mark
        forced = -1
        A = []
        for x in nbrs[h]:
            if vis[x]:
                continue
            mark[x] = st
            A.append(x)
            d = deg[x]
            if x == end:
                if d == 0 and self.remaining > 1:
                    return -2
            elif d <= 1:
                if d == 0 or forced >= 0:
                    return -2
                forced = x
        if not A:
            return -2
        # neighbours of the old head that lost head-adjacency
        for x in nbrs[p]:
            if vis[x] or mark[x] == st:
                continue
            if deg[x] < (1 if x == end else 2):
                return -2
        if self.blocked is not None:
            for grp in (nbrs[h], nbrs[p]):
                for x in grp:
                    if vis[x]:
                        continue
                    stx = self.dir_status(x, h)
                    if stx == -2:
                        return -2
                    if stx == 1:
                        if forced >= 0 and forced != x:
                            return -2
                        forced = x
        if forced >= 0:
            c = self.cpi[forced]
            if c >= 0 and c != self.nxt:
                return -2
            if self.constrained and not self.ok_step(h, forced):
                return -2
        # connectivity: all unvisited neighbours of h mutually reachable
        if len(A) > 1:
            self.stamp += 1
            st2 = self.stamp
            need = len(A) - 1
            src = A[0]
            mark[src] = st2 + 0  # visited in this BFS
            # targets carry stamp st; BFS marks with st2
            q = deque([src])
            found = 0
            while q and found < need:
                v = q.popleft()
                for w in nbrs[v]:
                    if vis[w]:
                        continue
                    mw = mark[w]
                    if mw == st2:
                        continue
                    if mw == st:
                        found += 1
                    mark[w] = st2
                    q.append(w)
            if found < need:
                return -2
        if self.chains:
            f2 = self.chain_check(h)
            if f2 == -2:
                return -2
            if f2 >= 0:
                if forced >= 0 and forced != f2:
                    return -2
                forced = f2
                c = self.cpi[forced]
                if c >= 0 and c != self.nxt:
                    return -2
        if self.dyn and not self.segment_check(h):
            return -2
        if self.deep and not self.deep_check():
            return -2
        return forced

    def segment_check(self, h: int) -> bool:
        """BFS from the next checkpoint through unvisited nodes, never passing
        a later checkpoint (the path cannot cross one before reaching the next
        checkpoint).  The head must be reached.  The distances are kept in
        ``self.dd`` for move ordering."""
        n, nbrs, vis, cpi = self.n, self.nbrs, self.visited, self.cpi
        nxt = self.nxt
        target = self.cps[nxt]
        dd = [n] * n
        dd[target] = 0
        q = [target]
        hn = nbrs[h]
        reached = target in hn
        i = 0
        while i < len(q):
            v = q[i]
            i += 1
            dv = dd[v] + 1
            for w in nbrs[v]:
                if vis[w] or dd[w] <= dv:
                    continue
                if cpi[w] > nxt:
                    continue
                dd[w] = dv
                q.append(w)
        if not reached:
            for w in hn:
                if not vis[w] and dd[w] < n:
                    reached = True
                    break
        self.dd = dd
        return reached

    def chain_check(self, h: int) -> int:
        """Forced-edge reasoning.  Returns -2 (dead), forced move, or -1.

        An unvisited non-final node with exactly two usable neighbours
        (unvisited ones + head) must use both edges; the final checkpoint with
        one usable neighbour must use it.  Forced edges form path fragments
        ("chains") of the final path, so: no node may get more forced edges
        than its path degree (2, or 1 for head / final checkpoint), chains
        must not close cycles, a chain joining head and end must cover
        everything, and checkpoints along a chain must be consecutive labels
        (increasing away from the head if the chain contains the head).
        """
        if not self.low:
            return -1
        vis, deg, nbrs, end, cpi = self.visited, self.deg, self.nbrs, self.end, self.cpi
        hset = nbrs[h]
        fadj: dict[int, list[int]] = {}
        for v in self.low:
            use = deg[v] + (1 if v in hset else 0)
            req = 1 if v == end else 2
            if use > req:
                continue
            if use < req:
                return -2
            for w in nbrs[v]:
                if vis[w] and w != h:
                    continue
                lv = fadj.get(v)
                if lv is None:
                    fadj[v] = [w]
                elif w not in lv:
                    lv.append(w)
                lw = fadj.get(w)
                if lw is None:
                    fadj[w] = [v]
                elif v not in lw:
                    lw.append(v)
        if not fadj:
            return -1
        for x, lx in fadj.items():
            if len(lx) > (1 if (x == h or x == end) else 2):
                return -2
        seen = set()
        nxt = self.nxt
        for x, lx in fadj.items():
            if len(lx) != 1 or x in seen:
                continue
            # walk the chain starting at endpoint x
            seq = [x]
            seen.add(x)
            prev, cur = x, lx[0]
            while True:
                seq.append(cur)
                seen.add(cur)
                lc = fadj[cur]
                if len(lc) == 1:
                    break
                nxt_node = lc[0] if lc[0] != prev else lc[1]
                prev, cur = cur, nxt_node
            has_h = seq[-1] == h or seq[0] == h
            if has_h:
                if seq[-1] == h:
                    seq.reverse()
                if seq[-1] == end and len(seq) != self.remaining + 1:
                    return -2
                expect = nxt
                for v in seq[1:]:
                    c = cpi[v]
                    if c >= 0:
                        if c != expect:
                            return -2
                        expect += 1
            else:
                last = -1
                step = 0
                for v in seq:
                    c = cpi[v]
                    if c >= 0:
                        if last >= 0:
                            d = c - last
                            if d != 1 and d != -1:
                                return -2
                            if step and d != step:
                                return -2
                            step = d
                        last = c
        if len(seen) != len(fadj):
            return -2           # forced edges close a cycle
        lh = fadj.get(h)
        return lh[0] if lh else -1

    def deep_check(self) -> bool:
        """Articulation-point reasoning on G[unvisited + head] (see module doc)."""
        n, nbrs, vis = self.n, self.nbrs, self.visited
        h = self.path[-1]
        end = self.end
        col = self.col
        disc = [-1] * n
        low = [0] * n
        sz = [1] * n
        ss = [0] * n
        disc[h] = 0
        t = 1
        stack = [h]
        its = [0]
        root_children = 0
        arts = []           # (a, separated child c)
        sep = {}
        ch = col[h] if col is not None else 0
        while stack:
            v = stack[-1]
            i = its[-1]
            nb = nbrs[v]
            ln = len(nb)
            advanced = False
            while i < ln:
                w = nb[i]
                i += 1
                if vis[w] and w != h:
                    continue
                dw = disc[w]
                if dw == -1:
                    its[-1] = i
                    disc[w] = low[w] = t
                    t += 1
                    stack.append(w)
                    its.append(0)
                    advanced = True
                    break
                if dw < low[v]:
                    low[v] = dw
            if advanced:
                continue
            stack.pop()
            its.pop()
            if col is not None:
                ss[v] += 1 if col[v] == ch else -1
            if v in sep:
                if sep[v] >= 2 or v == end:
                    return False
            if stack:
                a = stack[-1]
                if low[v] < low[a]:
                    low[a] = low[v]
                sz[a] += sz[v]
                ss[a] += ss[v]
                if a == h:
                    root_children += 1
                elif low[v] >= disc[a]:
                    k = sep.get(a, 0) + 1
                    sep[a] = k
                    if k >= 2:
                        return False
                    arts.append((a, v))
        total = t
        if total != self.remaining + 1 or root_children > 1:
            return False
        if not arts:
            return True
        de = disc[end]
        rem_cps = self.cps[self.nxt:]
        total_sum = ss[h]
        for a, c in arts:
            if a == end:
                return False
            lo = disc[c]
            hi = lo + sz[c]
            if not (lo <= de < hi):
                return False
            # checkpoint order: head side (0) < a (1) < end side (2)
            prev = 0
            for cp in rem_cps:
                if cp == a:
                    k = 1
                elif lo <= disc[cp] < hi:
                    k = 2
                else:
                    k = 0
                if k < prev:
                    return False
                prev = k
            # precedence: a key may not lie on a later side than its door
            for kk, dd in self.prec:
                if vis[kk]:
                    continue
                sk = 1 if kk == a else (2 if lo <= disc[kk] < hi else 0)
                sd = 1 if dd == a else (2 if lo <= disc[dd] < hi else 0)
                if sd < sk:
                    return False
            if col is not None:
                R = total - sz[c] - 1
                if col[a] != ch ^ (R & 1):
                    return False
                sa = 1 if col[a] == ch else -1
                if total_sum - sa - ss[c] != (R & 1):
                    return False
                S = sz[c]
                first = 1 if ((R + 1) & 1) == 0 else -1
                if ss[c] != (first if S & 1 else 0):
                    return False
        return True

    # ---- move generation --------------------------------------------------
    def candidates(self, h: int, forced: int) -> list[int]:
        """Legal next moves, best first (default heuristic).

        Score = deg_weight * free_degree + dist_weight * distance to the next
        checkpoint (lower is better); the next checkpoint itself goes first.
        """
        cpi, nxt, end = self.cpi, self.nxt, self.end
        if forced >= 0:
            if self.constrained and not self.ok_step(h, forced):
                return []
            return [forced]
        vis, deg, rem = self.visited, self.deg, self.remaining
        constrained = self.constrained
        rnd = self.rng.random if self.rng is not None else None
        cp_pref = self.cp_pref
        dist = self.dd if (self.dyn_order and self.dd is not None) else self.dist[nxt]
        a, b = self.deg_weight, self.dist_weight
        out = []
        for w in self.nbrs[h]:
            if vis[w]:
                continue
            if constrained and not self.ok_step(h, w):
                continue
            c = cpi[w]
            if c >= 0:
                if c != nxt:
                    continue
                if w == end and rem > 1:
                    continue
                key = -1e9 if cp_pref else a * deg[w]
            else:
                if deg[w] == 0 and rem > 1:
                    continue
                key = a * deg[w] + b * dist[w]
            out.append((key, rnd() if rnd else 0.0, w))
        out.sort()
        return [t[2] for t in out]

    # ---- main loop ----------------------------------------------------------
    def run(self, max_solutions: int, time_limit: float | None,
            move_order: MoveOrder | None, t0: float, max_expansions: int | None = None,
            prefix: Sequence[int] | None = None):
        """Returns (solutions, status, nodes_expanded).

        status: "complete" (search exhausted), "limit", "timeout" or
        "budget" (``max_expansions`` reached).  ``prefix`` (a legal partial
        path starting at checkpoint 1, validated by the caller) initialises
        the search state: only its completions are searched.
        """
        sols: list[list[int]] = []
        if prefix is None or len(prefix) <= 1:
            start = self.cps[0]
            self.push(start)
            self.nxt = 1
        else:
            for v in prefix:
                self.push(int(v))
            start = self.path[-1]
            if self.remaining == 0:     # complete path given (caller checked validity)
                sols.append(list(self.path))
                return sols, ("limit" if max_solutions <= 1 else "complete"), 0
        if self.n == 1:
            return sols, "complete", 0
        if not self.global_check():
            return sols, "complete", 0
        vis_view = np.frombuffer(self.visited, dtype=np.bool_) if move_order else None
        deadline = None if time_limit is None else t0 + time_limit
        expanded = 0

        def order(h, cands):
            if move_order is None or len(cands) <= 1:
                return cands
            pref = move_order(h, list(cands), vis_view, self.nxt)
            allowed = set(cands)
            res, seen = [], set()
            for w in pref:
                w = int(w)
                if w in allowed and w not in seen:
                    seen.add(w)
                    res.append(w)
            return res

        frames = [[order(start, self.candidates(start, -1)), 0]]
        path = self.path
        while frames:
            fr = frames[-1]
            cands = fr[0]
            if fr[1] >= len(cands):
                frames.pop()
                if frames:
                    self.pop()
                continue
            w = cands[fr[1]]
            fr[1] += 1
            expanded += 1
            if max_expansions is not None and expanded > max_expansions:
                return sols, "budget", expanded
            if deadline is not None and expanded % _TIME_CHECK_EVERY == 0:
                if time.perf_counter() > deadline:
                    return sols, "timeout", expanded
            p = path[-1]
            self.push(w)
            if self.remaining == 0:
                sols.append(list(path))
                self.pop()
                if len(sols) >= max_solutions:
                    return sols, "limit", expanded
                continue
            forced = self.after_move(p, w)
            if forced == -2:
                self.pop()
                continue
            nc = self.candidates(w, forced)
            if not nc:
                self.pop()
                continue
            frames.append([order(w, nc), 0])
        return sols, "complete", expanded


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def solve(puzzle: Puzzle, time_limit: float | None = None,
          move_order: MoveOrder | None = None, *, deep: bool = True,
          restarts: bool | None = None, seed: int = 0) -> SolveResult:
    """Find one solution.

    ``time_limit`` is in seconds (None = unlimited).  ``move_order`` is the
    optional policy hook (see module docstring).  Keyword-only extras:

    * ``deep``: articulation-point pruning (default on; big win on islands /
      2D, a bit slower per node).
    * ``restarts``: randomised restarts with geometrically growing node
      budgets (random tie-breaking among equally ranked
      moves; budget x1.5 per attempt).  DFS on high-dimensional grids is heavy-tailed; restarts make
      3D/4D puzzles orders of magnitude faster.  The search stays complete:
      the last attempt is simply the first one whose budget is not exhausted,
      so "unsat" is still a proof (costing ~3x a single run).  Defaults to on
      without ``move_order``, off with it (so a policy's order is followed
      exactly).
    * ``seed``: seed for the restart tie-breaking (results are deterministic).

    ``nodes_expanded`` counts expansions over all attempts.
    """
    return _solve(puzzle, None, time_limit, move_order, deep, restarts, seed)


def _solve(puzzle: Puzzle, prefix: Sequence[int] | None, time_limit: float | None,
           move_order: MoveOrder | None, deep: bool, restarts: bool | None,
           seed: int) -> SolveResult:
    import random

    t0 = time.perf_counter()
    if restarts is None:
        restarts = move_order is None
    budget = max(_RESTART_BASE, 2 * puzzle.num_nodes) if restarts else None
    total = 0
    attempt = 0
    while True:
        s = _Search(puzzle.graph, puzzle.checkpoints, deep=deep)
        if attempt > 0:
            s.rng = random.Random(seed * 1_000_003 + attempt)
        remaining_time = None
        if time_limit is not None:
            remaining_time = time_limit - (time.perf_counter() - t0)
            if remaining_time <= 0:
                return SolveResult("timeout", None, total, time.perf_counter() - t0)
        sols, status, expanded = s.run(1, remaining_time, move_order,
                                       time.perf_counter(), budget, prefix)
        total += expanded
        if sols:
            return SolveResult("solved", sols[0], total, time.perf_counter() - t0)
        if status == "timeout":
            return SolveResult("timeout", None, total, time.perf_counter() - t0)
        if status != "budget":
            return SolveResult("unsat", None, total, time.perf_counter() - t0)
        budget = int(budget * _RESTART_GROWTH)
        attempt += 1


def check_prefix(puzzle: Puzzle, prefix: Sequence[int]) -> str | None:
    """None if ``prefix`` is a legal partial path (possibly complete), else a reason.

    Legal: non-empty, starts at checkpoint 1, valid distinct node ids, consecutive
    nodes adjacent (one-way arcs used forwards only), checkpoints met in order,
    every node entered only after its prerequisites (keys before doors), and
    the final checkpoint only as the very last node of a complete path.
    """
    g = puzzle.graph
    bl = blocked_steps(g) or {}
    pre = prerequisites(g) or {}
    n = g.num_nodes
    cps = puzzle.checkpoints
    cpi = {c: i for i, c in enumerate(cps)}
    if len(prefix) == 0:
        return "empty prefix"
    seen: set[int] = set()
    nxt = 0
    for i, v in enumerate(prefix):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return f"invalid node id {v!r}"
        if not (0 <= v < n):
            return f"invalid node id {v}"
        if i == 0 and v != cps[0]:
            return "prefix must start at checkpoint 1"
        if v in seen:
            return f"node {v} visited twice"
        seen.add(v)
        if i > 0 and not g.has_edge(int(prefix[i - 1]), v):
            return f"nodes {int(prefix[i - 1])} and {v} are not adjacent"
        if i > 0 and v in bl.get(int(prefix[i - 1]), ()):
            return f"one-way edge {v} -> {int(prefix[i - 1])} used backwards"
        for a in pre.get(v, ()):
            if a not in seen:
                return f"node {v} entered before node {a} (door before its key)"
        k = cpi.get(v)
        if k is not None:
            if k != nxt:
                return f"checkpoint {k + 1} reached before checkpoint {nxt + 1}"
            nxt += 1
        if v == cps[-1] and i != n - 1:
            return "the last checkpoint must be the final node"
    return None


def solve_from_prefix(puzzle: Puzzle, prefix: Sequence[int], time_limit: float | None = None,
                      move_order: MoveOrder | None = None, *, deep: bool = True,
                      restarts: bool | None = None, seed: int = 0) -> SolveResult:
    """Complete a legal partial path; sound and complete like :func:`solve`.

    The search state (visited set, free degrees, next checkpoint, ...) is
    initialised from ``prefix`` and the same DFS / prunings / restarts as
    :func:`solve` run from its last node.  ``"solved"``: ``path`` is a full
    solution starting with ``prefix``; ``"unsat"``: proof that no completion
    exists; ``"timeout"``: unknown.  ``solve_from_prefix(p, [start])`` is
    ``solve(p)``.  Raises ``ValueError`` for an illegal prefix (see
    :func:`check_prefix`).
    """
    prefix = [int(v) for v in prefix] if len(prefix) else []
    reason = check_prefix(puzzle, prefix)
    if reason is not None:
        raise ValueError(f"invalid prefix: {reason}")
    if len(prefix) == puzzle.num_nodes:
        ok = puzzle.is_valid_solution(prefix)
        return SolveResult("solved" if ok else "unsat", prefix if ok else None, 0, 0.0)
    return _solve(puzzle, prefix, time_limit, move_order, deep, restarts, seed)


def legal_moves(puzzle: Puzzle, prefix: Sequence[int]) -> list[int]:
    """Legal next nodes after a (legal) prefix, by the game rules only (same as
    ``ZipEnv``'s action mask): unvisited neighbours of the head that are not a
    checkpoint out of order, the final checkpoint only as the very last node,
    not against a one-way arc, not a door whose key is still unvisited."""
    g = puzzle.graph
    bl = blocked_steps(g) or {}
    pre = prerequisites(g) or {}
    cps = puzzle.checkpoints
    cpi = {c: i for i, c in enumerate(cps)}
    vis = set(int(v) for v in prefix)
    nxt = sum(1 for v in vis if v in cpi)
    n = g.num_nodes
    out = []
    h = int(prefix[-1])
    bh = bl.get(h, ())
    for w in g.neighbors[h]:
        if w in vis or w in bh:
            continue
        if any(a not in vis for a in pre.get(w, ())):
            continue
        k = cpi.get(w, -1)
        if k > nxt:
            continue
        if w == cps[-1] and len(vis) != n - 1:
            continue
        out.append(int(w))
    return out


def label_moves(puzzle: Puzzle, prefix: Sequence[int], time_limit_per_move: float | None,
                return_witnesses: bool = False, known_solution: Sequence[int] | None = None,
                deep: bool = True):
    """Label every legal next move after ``prefix``.

    Returns ``{move: "win" | "lose" | "unknown"}`` (moves in ``legal_moves``
    order): ``"win"`` = a completion through the move exists (witness found),
    ``"lose"`` = proven impossible (exhaustive search or a sound pruning such
    as :func:`is_dead_end`), ``"unknown"`` = the per-move time limit ran out.
    A timeout is never reported as ``"lose"``.  With ``return_witnesses`` the
    result is ``(labels, witnesses)`` where ``witnesses[move]`` is a full
    solution through ``prefix + [move]`` for every winning move.
    ``known_solution``: a known full solution; if it extends ``prefix`` its
    next node is labelled "win" without searching.
    """
    prefix = [int(v) for v in prefix]
    reason = check_prefix(puzzle, prefix)
    if reason is not None:
        raise ValueError(f"invalid prefix: {reason}")
    labels: dict[int, str] = {}
    wit: dict[int, list[int]] = {}
    k = len(prefix)
    known_next = None
    if known_solution is not None and len(known_solution) > k \
            and [int(v) for v in known_solution[:k]] == prefix \
            and puzzle.is_valid_solution(known_solution):
        known_next = int(known_solution[k])
    for m in legal_moves(puzzle, prefix):
        if m == known_next:
            labels[m] = "win"
            wit[m] = [int(v) for v in known_solution]
            continue
        res = solve_from_prefix(puzzle, prefix + [m], time_limit=time_limit_per_move, deep=deep)
        if res.status == "solved":
            labels[m] = "win"
            wit[m] = res.path
        elif res.status == "unsat":
            labels[m] = "lose"
        else:
            labels[m] = "unknown"
    return (labels, wit) if return_witnesses else labels


def count_solutions(puzzle: Puzzle, limit: int = 2,
                    time_limit: float | None = None) -> tuple[int, str]:
    """Count solutions up to ``limit``.

    Returns ``(count, status)`` with status ``"complete"`` (count is exact),
    ``"limit"`` (at least ``limit`` solutions exist) or ``"timeout"`` (count is
    a lower bound).  ``count_solutions(p, 2)[0] == 1`` tests uniqueness.
    """
    t0 = time.perf_counter()
    if limit <= 0:
        return 0, "limit"
    s = _Search(puzzle.graph, puzzle.checkpoints)
    sols, status, _ = s.run(limit, time_limit, None, t0)
    return len(sols), status


def find_solutions(puzzle: Puzzle, limit: int = 2,
                   time_limit: float | None = None) -> tuple[list[list[int]], str]:
    """Like :func:`count_solutions` but returns the solutions themselves.

    Returns ``(solutions, status)``; the generator uses the second solution to
    pick a checkpoint that rules it out.
    """
    t0 = time.perf_counter()
    if limit <= 0:
        return [], "limit"
    sols, status, _ = _Search(puzzle.graph, puzzle.checkpoints).run(limit, time_limit, None, t0)
    return [list(x) for x in sols], status


def is_dead_end(graph: ZipGraph, visited, head: int, next_checkpoint_idx: int,
                checkpoints: Sequence[int]) -> bool:
    """True if the partial state provably cannot be completed.

    ``visited``: the nodes already on the path *including* ``head`` (a set /
    iterable of node ids, or a bool array of shape (n,)).
    ``next_checkpoint_idx``: index of the next checkpoint still to be reached.
    Runs the same sound checks as the solver (degree rule, connectivity,
    parity, articulation points) in O(n + m); False does not guarantee that
    a solution exists.
    """
    n = graph.num_nodes
    s = _Search(graph, checkpoints)
    if isinstance(visited, np.ndarray) and visited.dtype == np.bool_ and visited.shape == (n,):
        vset = np.flatnonzero(visited).tolist()
    else:
        vset = [int(v) for v in visited]
    vset = set(vset)
    head = int(head)
    if head not in vset:
        vset.add(head)
    for v in vset:
        s.visited[v] = 1
        for x in s.nbrs[v]:
            s.deg[x] -= 1
    s.remaining = n - len(vset)
    s.path = [head]
    s.nxt = int(next_checkpoint_idx)
    # a checkpoint just reached by the head counts as visited
    if s.nxt < len(s.cps) and s.cps[s.nxt] == head:
        s.nxt += 1
    if s.nxt > len(s.cps):
        return True
    return not s.global_check()


__all__ = ["SolveResult", "solve", "count_solutions", "find_solutions", "is_dead_end",
           "solve_from_prefix", "label_moves", "check_prefix", "legal_moves"]
