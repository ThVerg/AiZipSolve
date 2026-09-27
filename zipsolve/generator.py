"""Random puzzle generation for every Zip variant.

Pipeline: build a graph -> find a *random* Hamiltonian path -> place ordered
checkpoints along it (optionally adding more until the solution is unique).

Hamiltonian path search (``random_hamiltonian_path``) works on any graph:

1. Cheap necessary conditions reject hopeless graphs instantly: connectivity,
   at most two degree-1 nodes, bipartite colour balance (|A|-|B| <= 1), and
   the "bridge tree" (2-edge-connected components joined by cut edges) must
   be a simple chain, because a Hamiltonian path crosses every cut edge
   exactly once. This is what makes chains of islands tractable.
2. The graph is split at its bridges; each component gets its own path with
   fixed entry/exit cells (the bridge endpoints), found by randomised DFS
   with the Warnsdorff heuristic, random tie-breaking, dead-end/connectivity
   pruning and restarts. A fully exhausted search proves infeasibility.
3. The path is randomised with "backbite" moves (the standard Markov chain on
   Hamiltonian paths: link an endpoint to one of its neighbours and reverse
   the tail segment). For components whose endpoints are pinned we run the
   chain on the free end and keep the last state whose end is back on the
   required cell.
"""
from __future__ import annotations

import math
import random
import time
from typing import Sequence

import numpy as np

from .graph import ZipGraph, grid, islands, random_mask, remove_edges
from .puzzle import Puzzle

__all__ = ["random_hamiltonian_path", "place_checkpoints", "generate", "make_puzzle",
           "default_num_checkpoints", "KINDS"]

class GenerationTimeout(ValueError):
    """Generation ran out of its time budget (a ValueError so older callers still catch it)."""


def _left(deadline: float, what: str = "puzzle generation") -> float:
    """Seconds left before `deadline`; raises GenerationTimeout once it has passed."""
    left = deadline - time.perf_counter()
    if left <= 0:
        raise GenerationTimeout(f"{what} exceeded its time limit")
    return left


KINDS = ("grid2d", "walls", "islands", "islands_chain", "grid3d", "grid4d", "mask", "grid")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _np_rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _py_rng(rng: np.random.Generator) -> random.Random:
    return random.Random(int(rng.integers(2**62)))


def _two_colour(nbrs: Sequence[Sequence[int]], nodes: Sequence[int]):
    """Return {node: 0/1} if the induced subgraph on `nodes` is bipartite, else None."""
    nodeset = set(nodes)
    col: dict[int, int] = {}
    for s in nodes:
        if s in col:
            continue
        col[s] = 0
        stack = [s]
        while stack:
            u = stack.pop()
            for w in nbrs[u]:
                if w not in nodeset:
                    continue
                if w not in col:
                    col[w] = col[u] ^ 1
                    stack.append(w)
                elif col[w] == col[u]:
                    return None
    return col


def _parity_ok(col, nodes, start, end) -> bool:
    """Bipartite parity condition for a Hamiltonian path on `nodes`."""
    if col is None:
        return True
    a = sum(1 for v in nodes if col[v] == 0)
    b = len(nodes) - a
    if abs(a - b) > 1:
        return False
    if a == b:
        if start is not None and end is not None and col[start] == col[end]:
            return False
        return True
    major = 0 if a > b else 1
    return all(x is None or col[x] == major for x in (start, end))


def _bridges(nbrs: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Cut edges of the graph (iterative Tarjan)."""
    n = len(nbrs)
    disc = [-1] * n
    low = [0] * n
    out = []
    t = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = t
        t += 1
        stack = [(root, -1, iter(nbrs[root]))]
        while stack:
            u, parent, it = stack[-1]
            advanced = False
            for w in it:
                if w == parent:
                    continue
                if disc[w] == -1:
                    disc[w] = low[w] = t
                    t += 1
                    stack.append((w, u, iter(nbrs[w])))
                    advanced = True
                    break
                low[u] = min(low[u], disc[w])
            if not advanced:
                stack.pop()
                if parent != -1:
                    low[parent] = min(low[parent], low[u])
                    if low[u] > disc[parent]:
                        out.append((parent, u))
    return out


# ----------------------------------------------------------------------------
# randomised Warnsdorff DFS with pruning
# ----------------------------------------------------------------------------
_EXHAUSTED = "exhausted"   # search space fully explored: no path exists
_GAVE_UP = "gave_up"       # budget / time ran out


def _dfs(nbrs, nodes, start, end, prng: random.Random, deadline: float, budget: int):
    """Hamiltonian path on induced subgraph `nodes` from `start` (to `end` if given).

    Returns a list of nodes, _EXHAUSTED or _GAVE_UP.
    """
    n = len(nodes)
    inset = set(nodes)
    if n == 1:
        return [start] if end in (None, start) else _EXHAUSTED
    if end == start:
        return _EXHAUSTED
    visited = {v: False for v in nodes}
    loc = {v: [w for w in nbrs[v] if w in inset] for v in nodes}
    avail = {v: len(loc[v]) for v in nodes}   # number of unvisited neighbours

    def visit(v):
        visited[v] = True
        for w in loc[v]:
            avail[w] -= 1

    def unvisit(v):
        visited[v] = False
        for w in loc[v]:
            avail[w] += 1

    path = [start]
    visit(start)

    def feasible(h) -> bool:
        remaining = n - len(path)
        if remaining == 0:
            return True
        hn = set(loc[h])
        ones = 0
        first = None
        for v in nodes:
            if visited[v]:
                continue
            if first is None:
                first = v
            eff = avail[v] + (1 if v in hn else 0)
            if eff == 0:
                return False
            if eff == 1 and remaining > 1:
                if end is not None:
                    if v != end:
                        return False
                else:
                    ones += 1
                    if ones > 1:
                        return False
        # remaining unvisited nodes must be connected
        seen = {first}
        stack = [first]
        while stack:
            u = stack.pop()
            for w in loc[u]:
                if not visited[w] and w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == remaining

    def candidates(h):
        cs = [w for w in loc[h] if not visited[w] and (w != end or len(path) == n - 1)]
        # Warnsdorff: fewest onward moves first; random tie-break. Stored so pop() gives best.
        cs.sort(key=lambda w: (avail[w], prng.random()), reverse=True)
        return cs

    stack = [candidates(start)]
    steps = 0
    while stack:
        if len(path) == n:
            return path
        steps += 1
        if steps > budget or ((steps & 255) == 0 and time.perf_counter() > deadline):
            return _GAVE_UP
        top = stack[-1]
        if not top:
            stack.pop()
            if len(path) > 1:
                unvisit(path.pop())
            continue
        w = top.pop()
        path.append(w)
        visit(w)
        if not feasible(w):
            unvisit(path.pop())
            continue
        stack.append(candidates(w))
    return path if len(path) == n else _EXHAUSTED


def _search(nbrs, nodes, start, end, prng, deadline, col=None):
    """Randomised DFS with restarts. `start` may be None (free), `end` may be None."""
    nodes = list(nodes)
    if col is None:
        col = _two_colour(nbrs, nodes)
    if not _parity_ok(col, nodes, start, end):
        return None
    inset = set(nodes)
    n = len(nodes)
    if start is None and end is not None:
        p = _search(nbrs, nodes, end, None, prng, deadline, col)
        return None if p is None else p[::-1]
    if start is None:
        deg1 = [v for v in nodes if sum(1 for w in nbrs[v] if w in inset) <= 1]
        if len(deg1) > 2:
            return None
        pool = deg1
        if not pool and col is not None:
            a = sum(1 for v in nodes if col[v] == 0)
            if 2 * a != n:
                major = 0 if 2 * a > n else 1
                pool = [v for v in nodes if col[v] == major]
        if not pool:
            pool = nodes
    else:
        pool = [start]
    budget = max(2000, 30 * n)
    tried_exhaustive = set()
    while True:
        s = pool[prng.randrange(len(pool))]
        if s in tried_exhaustive:
            if len(tried_exhaustive) >= len(set(pool)):
                return None
            continue
        res = _dfs(nbrs, nodes, s, end, prng, deadline, budget)
        if isinstance(res, list):
            return res
        if res == _EXHAUSTED:
            tried_exhaustive.add(s)
            if len(tried_exhaustive) >= len(set(pool)):
                return None
        if time.perf_counter() > deadline:
            return None
        budget = int(budget * 1.5)


# ----------------------------------------------------------------------------
# backbite mixing
# ----------------------------------------------------------------------------
def _backbite(path: list[int], nbrs, prng: random.Random, moves: int,
              fix_start: bool = False, target_end: int | None = None,
              allowed: set | None = None, deadline: float = math.inf) -> list[int]:
    """Apply random backbite moves. If target_end is given, return the last visited
    state whose end equals target_end (the input path must already end there)."""
    path = list(path)
    n = len(path)
    if n < 3:
        return path
    best = list(path) if target_end is not None else None
    for k in range(moves):
        if (k & 1023) == 0 and time.perf_counter() > deadline:
            break
        flip = (not fix_start) and prng.random() < 0.5
        if flip:
            path.reverse()
        t = path[-1]
        ns = nbrs[t] if allowed is None else [w for w in nbrs[t] if w in allowed]
        x = ns[prng.randrange(len(ns))]
        if x != path[-2]:
            i = path.index(x)
            path[i + 1:] = path[:i:-1]
        if flip:
            path.reverse()
        if target_end is not None and path[-1] == target_end:
            best = list(path)
    return best if target_end is not None else path


def _default_moves(n: int) -> int:
    return int(max(1000, 30 * n * math.sqrt(n)))


# ----------------------------------------------------------------------------
# public: random Hamiltonian path
# ----------------------------------------------------------------------------
def random_hamiltonian_path(graph: ZipGraph, rng=None, time_limit: float = 5.0,
                            start: int | None = None) -> list[int] | None:
    """A random Hamiltonian path of `graph` (optionally starting at `start`).

    Returns None if none exists (proven by the necessary-condition checks or an
    exhaustive search) or none was found within `time_limit` seconds.
    """
    rng = _np_rng(rng)
    prng = _py_rng(rng)
    deadline = time.perf_counter() + time_limit
    nbrs = graph.neighbors
    n = graph.num_nodes
    if n == 0:
        return None
    if n == 1:
        return [0] if start in (None, 0) else None
    if not graph.is_connected():
        return None
    deg1 = [v for v in range(n) if len(nbrs[v]) == 1]
    if len(deg1) > 2 or (start is not None and deg1 and start not in deg1 and len(deg1) == 2):
        return None
    col = _two_colour(nbrs, range(n))
    if not _parity_ok(col, list(range(n)), start, None):
        return None

    # ---- decompose at bridges into a chain of 2-edge-connected components
    br = _bridges(nbrs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    brset = {frozenset(e) for e in br}
    for u, v in graph.edges():
        if frozenset((u, v)) not in brset:
            parent[find(u)] = find(v)
    comp_of = [find(v) for v in range(n)]
    comps: dict[int, list[int]] = {}
    for v in range(n):
        comps.setdefault(comp_of[v], []).append(v)
    tree: dict[int, list[tuple[int, int, int]]] = {c: [] for c in comps}  # c -> (other, my_end, their_end)
    for u, v in br:
        cu, cv = comp_of[u], comp_of[v]
        tree[cu].append((cv, u, v))
        tree[cv].append((cu, v, u))
    if any(len(t) > 2 for t in tree.values()):
        return None
    leaves = [c for c, t in tree.items() if len(t) <= 1]
    if start is not None:
        if comp_of[start] not in leaves:
            return None
        first = comp_of[start]
    else:
        first = leaves[prng.randrange(len(leaves))]
    order = [first]
    links = []  # (exit node of order[i], entry node of order[i+1])
    prev = None
    while True:
        nxt = [t for t in tree[order[-1]] if t[0] != prev]
        if not nxt:
            break
        other, mine, theirs = nxt[0]
        links.append((mine, theirs))
        prev = order[-1]
        order.append(other)

    moves = _default_moves(n)
    full: list[int] = []
    for i, c in enumerate(order):
        nodes = comps[c]
        entry = start if i == 0 else links[i - 1][1]
        exit_ = links[i][0] if i < len(order) - 1 else None
        p = _search(nbrs, nodes, entry, exit_, prng, deadline)
        if p is None:
            return None
        if len(nodes) >= 3:
            allowed = set(nodes) if len(order) > 1 else None
            share = max(200, moves * len(nodes) // n)
            if entry is None and exit_ is None:
                p = _backbite(p, nbrs, prng, share, allowed=allowed, deadline=deadline)
            elif exit_ is None:
                p = _backbite(p, nbrs, prng, share, fix_start=True, allowed=allowed,
                              deadline=deadline)
            elif entry is None:
                p = _backbite(p[::-1], nbrs, prng, share, fix_start=True, allowed=allowed,
                              deadline=deadline)[::-1]
            else:
                p = _backbite(p, nbrs, prng, 2 * share, fix_start=True, target_end=exit_,
                              allowed=allowed, deadline=deadline)
        full.extend(p)
    return full


# ----------------------------------------------------------------------------
# checkpoints and puzzle generation
# ----------------------------------------------------------------------------
def default_num_checkpoints(num_nodes: int, rng=None) -> int:
    """Roughly LinkedIn-like: ~8-12 for a 7x7 board, grows with sqrt(n)."""
    rng = _np_rng(rng)
    k = int(round(1.4 * math.sqrt(num_nodes))) + int(rng.integers(-1, 2))
    return int(min(max(2, k), num_nodes))


def place_checkpoints(path: list[int], num_checkpoints: int, rng=None) -> list[int]:
    """Pick `num_checkpoints` nodes of `path` in path order, always including both ends.

    Interior checkpoints are stratified (one random position per equal slice of
    the path) so they do not clump together.
    """
    rng = _np_rng(rng)
    n = len(path)
    k = int(min(max(2, num_checkpoints), n))
    inner = k - 2
    interior = n - 2
    if inner == 0:
        idx = []
    else:
        bounds = np.linspace(1, n - 1, inner + 1)
        idx = []
        lo_prev = 0
        for j in range(inner):
            lo = max(int(math.ceil(bounds[j])), lo_prev + 1)
            hi = max(int(math.ceil(bounds[j + 1])) - 1, lo)
            # leave room for the remaining checkpoints
            hi = min(hi, n - 2 - (inner - 1 - j))
            lo = min(lo, hi)
            x = int(rng.integers(lo, hi + 1))
            idx.append(x)
            lo_prev = x
        assert len(set(idx)) == inner and interior >= inner
    return [path[0]] + [path[i] for i in idx] + [path[-1]]


def generate(graph: ZipGraph, num_checkpoints: int, rng=None, unique: bool = False,
             time_limit: float = 10.0) -> Puzzle:
    """Random puzzle on `graph` with `num_checkpoints` checkpoints (Puzzle.solution set).

    With unique=True, checkpoints taken from the solution are added (in the
    largest gap between consecutive checkpoints) until the exact solver proves
    the solution unique. Raises ValueError if no Hamiltonian path is found.
    """
    rng = _np_rng(rng)
    t0 = time.perf_counter()
    path = random_hamiltonian_path(graph, rng, time_limit=time_limit)
    if path is None:
        if time.perf_counter() - t0 >= time_limit:
            raise GenerationTimeout("no Hamiltonian path found within the time limit")
        raise ValueError("no Hamiltonian path found for this graph")
    return _puzzle_from_path(graph, path, num_checkpoints, rng, unique,
                             time_limit - (time.perf_counter() - t0))


def _puzzle_from_path(graph, path, num_checkpoints, rng, unique, time_limit) -> Puzzle:
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    cps = place_checkpoints(path, num_checkpoints, rng)
    puzzle = Puzzle(graph, cps, list(path))
    if unique:
        pos = {v: i for i, v in enumerate(path)}
        while True:
            # hard stop: never return a puzzle whose uniqueness is not proven
            remaining = _left(deadline, "proving the puzzle unique")
            # short per-check budget: on timeout we add a clue, which makes the next check easier
            per_check = min(remaining, max(0.5, time_limit / 20))
            sols, status = _two_solutions(puzzle, per_check)
            # status: "complete" (exact), "limit" (>= 2 found) or "timeout" (lower bound).
            if len(sols) <= 1 and status == "complete":
                break
            cps = set(puzzle.checkpoints)
            free = [i for i in range(1, len(path) - 1) if path[i] not in cps]
            if not free:
                break
            others = [s for s in sols if s is not None and list(s) != list(path)]
            killers = []
            if others:
                # checkpoints (from our path) that invalidate the alternative solution
                alt = others[0]
                for i in free:
                    trial = Puzzle(graph, [path[j] for j in sorted(
                        [pos[c] for c in puzzle.checkpoints] + [i])])
                    if trial.check_solution(alt) is not None:
                        killers.append(i)
            if killers:
                # prefer killers far from existing checkpoints (spread clues out)
                idxs = [pos[c] for c in puzzle.checkpoints]
                dist = np.array([min(abs(i - j) for j in idxs) for i in killers], dtype=float)
                top = np.flatnonzero(dist >= np.quantile(dist, 0.75))
                new = killers[int(top[int(rng.integers(len(top)))])]
            else:  # no alternative known (timeout / fallback): split the largest gap
                idxs = sorted(pos[c] for c in puzzle.checkpoints)
                gaps = [(q - p, p, q) for p, q in zip(idxs, idxs[1:]) if q - p > 1]
                big = max(g[0] for g in gaps)
                cands = [g for g in gaps if g[0] == big]
                _, p, q = cands[int(rng.integers(len(cands)))]
                lo, hi = p + max(1, (q - p) // 4), q - max(1, (q - p) // 4)
                new = int(rng.integers(lo, max(lo, hi) + 1))
            idxs = sorted([pos[c] for c in puzzle.checkpoints] + [new])
            puzzle = Puzzle(graph, [path[i] for i in idxs], list(path))
    return puzzle


def _two_solutions(puzzle: Puzzle, time_limit: float):
    """Up to two solutions of `puzzle` plus the solver status."""
    from .solver import find_solutions  # lazy: keeps the generator importable on its own
    return find_solutions(puzzle, limit=2, time_limit=time_limit)


# ----------------------------------------------------------------------------
# archipelago: islands the path visits back and forth
# ----------------------------------------------------------------------------
_LAYOUTS = {2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3)}


def _pick_blocks(layout, k, rng) -> list[tuple[int, int]]:
    """k blocks of the R x C layout that form an edge-connected set (random removal)."""
    R, C = layout
    blocks = [(r, c) for r in range(R) for c in range(C)]
    while len(blocks) > k:
        order = rng.permutation(len(blocks))
        for i in order:
            rest = blocks[:i] + blocks[i + 1:]
            seen, stack = {rest[0]}, [rest[0]]
            while stack:
                r, c = stack.pop()
                for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if nb in rest and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if len(seen) == len(rest):
                blocks = rest
                break
        else:
            break
    return blocks


def _carve_along_path(mask, bid, path, frac, prng, deadline):
    """Carve organic coastlines while keeping `path` (a list of (r, c)) Hamiltonian.

    Where the path makes a U-turn a -> x -> y -> b with a and b adjacent, the domino
    (x, y) can be removed and the path spliced to a -> b. Only rim dominoes of one
    island are carved (bays and capes, not holes), islands stay connected, and a
    domino has one cell of each chessboard colour, so parity is preserved for free.
    Random backbite moves between carvings create fresh U-turns.
    Returns the new path; mask and bid are updated in place.
    """
    H, W = mask.shape
    sizes = {int(b): int((bid == b).sum()) for b in np.unique(bid) if b >= 0}
    target = {b: int(round(frac * n / 2)) * 2 for b, n in sizes.items()}
    removed = {b: 0 for b in sizes}

    def nbrs(cell):
        r, c = cell
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                yield (rr, cc)

    def on_rim(cell):
        r, c = cell
        return any(not (0 <= rr < H and 0 <= cc < W) or bid[rr, cc] != bid[r, c]
                   for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))

    def island_connected(b):
        cells = list(zip(*np.nonzero(bid == b)))
        seen, stack = {cells[0]}, [cells[0]]
        while stack:
            for nb in nbrs(stack.pop()):
                if bid[nb] == b and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(cells)

    path = [tuple(map(int, x)) for x in path]
    stall = 0
    while any(removed[b] < target[b] for b in sizes) and stall < 400:
        if time.perf_counter() > deadline:
            break
        cands = []
        for i in range(len(path) - 3):
            a, x, y, b = path[i:i + 4]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                continue
            isl = bid[x]
            if bid[y] != isl or removed[isl] + 2 > target[isl] or sizes[isl] - removed[isl] - 2 < 4:
                continue
            if on_rim(x) or on_rim(y):
                cands.append(i)
        prng.shuffle(cands)
        done = False
        for i in cands[:8]:
            x, y = path[i + 1], path[i + 2]
            isl = int(bid[x])
            for cell in (x, y):
                mask[cell] = False
                bid[cell] = -1
            if island_connected(isl):
                path = path[:i + 1] + path[i + 3:]
                removed[isl] += 2
                done = True
                break
            for cell in (x, y):
                mask[cell] = True
                bid[cell] = isl
        if done:
            stall = 0
            continue
        stall += 1
        for _ in range(60):  # backbite on the current shape to create new U-turns
            flip = prng.random() < 0.5
            if flip:
                path.reverse()
            ns = list(nbrs(path[-1]))
            nb = ns[prng.randrange(len(ns))]
            if nb != path[-2]:
                j = path.index(nb)
                path[j + 1:] = path[:j:-1]
            if flip:
                path.reverse()
    return path


def _tune_crossings(path, nbrs, block, target, max_pair, prng, steps, deadline):
    """Backbite MCMC steering the number of island-to-island jumps of `path`.

    Energy = |jumps - target| + sum over island pairs of max(0, jumps_pair - max_pair).
    A backbite move removes one path edge and adds one, so the energy change is O(1).
    Returns (path, energy) for the lowest-energy path seen (the last one among ties,
    so mixing continues at zero energy).
    """
    def key(u, v):
        a, b = block[u], block[v]
        return None if a == b else (min(a, b), max(a, b))

    pair: dict = {}
    for u, v in zip(path, path[1:]):
        k = key(u, v)
        if k is not None:
            pair[k] = pair.get(k, 0) + 1
    total = sum(pair.values())

    def energy(tot, pr):
        return abs(tot - target) + sum(max(0, c - max_pair) for c in pr.values())

    E = energy(total, pair)
    best, bestE = list(path), E
    path = list(path)
    for step in range(steps):
        if (step & 511) == 0 and time.perf_counter() > deadline:
            break
        flip = prng.random() < 0.5
        if flip:
            path.reverse()
        t = path[-1]
        ns = nbrs[t]
        x = ns[prng.randrange(len(ns))]
        if x != path[-2]:
            i = path.index(x)
            k_add, k_rem = key(t, x), key(x, path[i + 1])
            if k_add != k_rem:
                new_pair = dict(pair)
                new_total = total
                if k_add is not None:
                    new_pair[k_add] = new_pair.get(k_add, 0) + 1
                    new_total += 1
                if k_rem is not None:
                    new_pair[k_rem] -= 1
                    new_total -= 1
                newE = energy(new_total, new_pair)
                dE = newE - E
                # strict once at the target, Metropolis before
                ok = dE <= 0 or (E > 0 and prng.random() < math.exp(-1.5 * dE))
            else:
                ok, newE, new_pair, new_total = True, E, pair, total
            if ok:
                path[i + 1:] = path[:i:-1]
                E, pair, total = newE, new_pair, new_total
        if flip:
            path.reverse()
        if E <= bestE:
            best, bestE = list(path), E
    return best, bestE


def archipelago(num_islands: int = 3, layout: tuple[int, int] | None = None,
                min_side: int = 3, max_side: int = 4, jumps: int | None = None,
                max_per_pair: int | None = None, decoys: int | None = None, gap: int = 1,
                organic: float = 0.2, rng=None, time_limit: float = 10.0) -> tuple[ZipGraph, list[int]]:
    """Rectangular islands the solution path crosses between several times.

    Construction (path first, then the map):
      1. Lay out `num_islands` rectangular blocks (random sides in [min_side, max_side])
         on an R x C layout, touching each other, and take a random Hamiltonian path
         of the union as one big grid.
      2. Steer the path with backbite MCMC so it crosses island boundaries exactly
         `jumps` times in total, at most `max_per_pair` times between any two islands.
         With organic > 0, about that fraction of each island's rim cells is carved
         away first (bays and capes instead of perfect rectangles).
         jumps > num_islands - 1 forces the path to leave and come back to islands,
         so checkpoint numbers interleave between islands.
      3. Pull the islands apart by `gap` empty cells. Every boundary crossing of the
         path becomes a bridge; `decoys` extra bridges (unused by the solution) are
         added between other facing cells.
    Returns (graph, solution_path). meta: island, bridges, decoys, shapes, layout.
    """
    rng = _np_rng(rng)
    prng = _py_rng(rng)
    t_end = time.perf_counter() + time_limit
    k = int(num_islands)
    if k < 2:
        raise ValueError("need at least 2 islands")
    layout = tuple(layout) if layout else _LAYOUTS.get(k, (math.ceil(k / 3), 3))
    if layout[0] * layout[1] < k:
        raise ValueError(f"layout {layout} too small for {k} islands")
    jumps = int(jumps) if jumps is not None else 2 * (k - 1) + 1
    jumps = max(jumps, k - 1)
    max_per_pair = int(max_per_pair) if max_per_pair is not None else 3
    decoys = int(decoys) if decoys is not None else k - 1

    for _attempt in range(50):
        _left(t_end, "building the island map")
        blocks = _pick_blocks(layout, k, rng)
        heights = [int(x) for x in rng.integers(min_side, max_side + 1, size=layout[0])]
        widths = [int(x) for x in rng.integers(min_side, max_side + 1, size=layout[1])]
        r0 = np.concatenate([[0], np.cumsum(heights)]).astype(int)
        c0 = np.concatenate([[0], np.cumsum(widths)]).astype(int)
        mask = np.zeros((r0[-1], c0[-1]), dtype=bool)
        bid = -np.ones(mask.shape, dtype=int)
        for b, (br, bc) in enumerate(blocks):
            mask[r0[br]:r0[br + 1], c0[bc]:c0[bc + 1]] = True
            bid[r0[br]:r0[br + 1], c0[bc]:c0[bc + 1]] = b
        from .graph import from_mask, from_edges
        base = from_mask(mask)
        path = random_hamiltonian_path(base, rng, time_limit=min(max(0.5, time_limit / 4), _left(t_end, "building the island map")))
        if path is None:
            continue
        if organic > 0:
            cells_path = _carve_along_path(mask, bid, [tuple(base.coords[v].astype(int)) for v in path],
                                           organic, prng, t_end)
            base = from_mask(mask)
            path = [base.node_at(cell) for cell in cells_path]
        cells = base.coords.astype(int)
        block = [int(bid[r, c]) for r, c in cells]
        n = base.num_nodes
        path, E = _tune_crossings(path, base.neighbors, block, jumps, max_per_pair, prng,
                                  steps=int(200 * n), deadline=t_end)
        if E != 0:
            continue

        # separate the islands and keep only in-island edges + bridges
        row_blk = np.searchsorted(r0, np.arange(r0[-1]), side="right") - 1
        col_blk = np.searchsorted(c0, np.arange(c0[-1]), side="right") - 1
        coords = np.array([(r + gap * row_blk[r], c + gap * col_blk[c]) for r, c in cells], float)
        on_path = {frozenset(e) for e in zip(path, path[1:])}
        inner, bridges, spare = [], [], []
        for u, v in base.edges():
            if block[u] == block[v]:
                inner.append((u, v))
            elif frozenset((u, v)) in on_path:
                bridges.append((u, v))
            else:
                spare.append((u, v))
        # decoys: prefer spots not touching an existing bridge (keeps the map readable)
        used_nodes = {x for e in bridges for x in e}
        order = sorted(rng.permutation(len(spare)).tolist(),
                       key=lambda i: spare[i][0] in used_nodes or spare[i][1] in used_nodes)
        fake = []
        for i in order[:decoys]:
            fake.append(spare[i])
            used_nodes.update(spare[i])
        g = from_edges(coords, inner + bridges + fake, "islands", {
            "island": block,
            "bridges": [tuple(map(int, e)) for e in bridges + fake],
            "decoys": [tuple(map(int, e)) for e in fake],
            "shapes": [(heights[br], widths[bc]) for br, bc in blocks],
            "sizes": [int((np.array(block) == b).sum()) for b in range(len(blocks))],
            "layout": list(layout),
            "jumps": len(bridges),
        })
        return g, list(path)
    raise ValueError(f"could not build a {k}-island map with {jumps} jumps "
                     f"(max {max_per_pair} per pair) in time")


# ----------------------------------------------------------------------------
# make_puzzle: one entry point for every variant
# ----------------------------------------------------------------------------
def _as_shape(size, dim: int | None) -> tuple[int, ...]:
    if isinstance(size, (int, np.integer)):
        return (int(size),) * (dim or 2)
    return tuple(int(s) for s in size)


def make_puzzle(kind: str, size, num_checkpoints: int | None = None, rng=None, **kw) -> Puzzle:
    """Generate a random puzzle of a given kind.

    kind:
      "grid2d"  size=int n (n x n) or (h, w)
      "walls"   2D grid + walls on edges unused by the solution; kw walls_frac (0.25)
      "islands" size=int number of islands -> archipelago(): the path jumps between
                islands several times (back and forth); kw jumps, max_per_pair,
                decoys, layout, min_side (3), max_side (4), gap (1). See archipelago().
      "islands_chain" size=list of shapes or int: the old single-bridge chain of islands;
                kw extra_bridges (0), gap (2), min_side (3), max_side (4)
      "grid3d"  size=int n -> n^3 ;  "grid4d" size=int n -> n^4
      "grid"    size=tuple of any dimensionality
      "mask"    irregular blob via graph.random_mask; size=int (n x n) or shape; kw fill (0.75)
    Other kw: unique (False), time_limit (10.0), max_tries (100).
    """
    rng = _np_rng(rng)
    unique = kw.pop("unique", False)
    time_limit = kw.pop("time_limit", 10.0)
    max_tries = kw.pop("max_tries", 100)
    deadline = time.perf_counter() + time_limit

    def left() -> float:
        return _left(deadline)

    def finish(graph, path=None):
        k = num_checkpoints if num_checkpoints is not None else \
            default_num_checkpoints(graph.num_nodes, rng)
        if path is None:
            return generate(graph, k, rng, unique=unique, time_limit=left())
        return _puzzle_from_path(graph, path, k, rng, unique, left())

    if kind in ("grid2d", "grid3d", "grid4d", "grid"):
        dim = {"grid2d": 2, "grid3d": 3, "grid4d": 4, "grid": None}[kind]
        if kind == "grid" and isinstance(size, (int, np.integer)):
            dim = 2
        return finish(grid(*_as_shape(size, dim)))

    if kind == "walls":
        frac = kw.pop("walls_frac", 0.25)
        g = grid(*_as_shape(size, 2))
        path = random_hamiltonian_path(g, rng, time_limit=left())
        if path is None:
            left()
            raise ValueError("no Hamiltonian path on base grid")
        used = {frozenset(e) for e in zip(path, path[1:])}
        free = [e for e in g.edges() if frozenset(e) not in used]
        m = int(round(frac * len(free)))
        chosen = rng.choice(len(free), size=m, replace=False) if m else []
        walls = [free[i] for i in sorted(chosen)]
        return finish(remove_edges(g, walls), path)

    if kind == "islands" and isinstance(size, (int, np.integer)):
        g, path = archipelago(
            int(size), layout=kw.pop("layout", None),
            min_side=kw.pop("min_side", 3), max_side=kw.pop("max_side", 5),
            organic=kw.pop("organic", 0.2),
            jumps=kw.pop("jumps", None), max_per_pair=kw.pop("max_per_pair", None),
            decoys=kw.pop("decoys", kw.pop("extra_bridges", None)), gap=kw.pop("gap", 1),
            rng=rng, time_limit=left())
        return finish(g, path)

    if kind in ("islands", "islands_chain"):
        extra = kw.pop("extra_bridges", 0)
        gap = kw.pop("gap", 2)
        lo, hi = kw.pop("min_side", 3), kw.pop("max_side", 4)
        for _ in range(max_tries):
            per_try = min(max(0.2, time_limit / 10), left())
            if isinstance(size, (int, np.integer)):
                shapes = [tuple(int(x) for x in rng.integers(lo, hi + 1, size=2))
                          for _ in range(int(size))]
            else:
                shapes = [tuple(s) for s in size]
            g = islands(shapes, extra_bridges=extra, gap=gap, rng=rng)
            path = random_hamiltonian_path(g, rng, time_limit=per_try)
            if path is not None:
                return finish(g, path)
        raise ValueError(f"no Hamiltonian islands graph found in {max_tries} tries")

    if kind == "mask":
        fill = kw.pop("fill", 0.75)
        shape = _as_shape(size, kw.pop("dim", 2))
        for _ in range(max_tries):
            per_try = min(max(0.2, time_limit / 10), left())
            g = random_mask(shape, fill=fill, rng=rng)
            if g.num_nodes < 2:
                continue
            path = random_hamiltonian_path(g, rng, time_limit=per_try)
            if path is not None:
                return finish(g, path)
        raise ValueError(f"no Hamiltonian mask found in {max_tries} tries")

    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
