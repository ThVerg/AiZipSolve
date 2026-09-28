"""Graph representation shared by every Zip variant.

A Zip puzzle is a Hamiltonian path problem with ordered checkpoints on an
arbitrary undirected graph. Every variant (2D grid, walls, islands, 3D, 4D,
irregular masks) is just a different graph built by one of the functions below.

Nodes are integers 0..n-1. Each node has a coordinate vector (used only for
visualisation and optional positional features) and an adjacency list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ZipGraph:
    coords: np.ndarray                      # (n, d) float array, d = spatial dimension
    neighbors: tuple[tuple[int, ...], ...]  # neighbors[v] = sorted adjacent node ids
    kind: str = "custom"                    # "grid", "islands", "mask", ...
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.neighbors)

    @property
    def dim(self) -> int:
        return self.coords.shape[1]

    def edges(self) -> list[tuple[int, int]]:
        """Each undirected edge once, as (u, v) with u < v."""
        return [(u, v) for u, ns in enumerate(self.neighbors) for v in ns if u < v]

    def edge_index(self) -> np.ndarray:
        """(2, 2E) int64 array with both directions of every edge (GNN format)."""
        src = [u for u, ns in enumerate(self.neighbors) for _ in ns]
        dst = [v for ns in self.neighbors for v in ns]
        return np.array([src, dst], dtype=np.int64).reshape(2, -1)

    def degree(self, v: int) -> int:
        return len(self.neighbors[v])

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.neighbors[u]

    def node_at(self, coord: Sequence[float]) -> int:
        """Node id at an exact coordinate (raises KeyError if none)."""
        lookup = self.meta.get("_lookup")
        if lookup is None:
            lookup = {tuple(c): i for i, c in enumerate(self.coords.tolist())}
            self.meta["_lookup"] = lookup
        return lookup[tuple(float(x) for x in coord)]

    def is_connected(self) -> bool:
        if self.num_nodes == 0:
            return True
        seen = {0}
        stack = [0]
        while stack:
            for w in self.neighbors[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == self.num_nodes


def from_edges(coords, edges: Iterable[tuple[int, int]], kind="custom", meta=None) -> ZipGraph:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    adj: list[set[int]] = [set() for _ in range(len(coords))]
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    meta = {k: v for k, v in dict(meta or {}).items() if not str(k).startswith("_")}  # drop caches
    return ZipGraph(coords, tuple(tuple(sorted(a)) for a in adj), kind, meta)


def from_mask(mask: np.ndarray, kind="mask", meta=None) -> ZipGraph:
    """Grid graph of any dimensionality restricted to the True cells of `mask`.

    Cells are numbered in row-major (C) order of the mask. Two cells are
    adjacent when their index vectors differ by exactly 1 along one axis.
    """
    mask = np.asarray(mask, dtype=bool)
    cells = [idx for idx in np.ndindex(mask.shape) if mask[idx]]
    ids = {c: i for i, c in enumerate(cells)}
    edges = []
    for c, i in ids.items():
        for axis in range(mask.ndim):
            nb = c[:axis] + (c[axis] + 1,) + c[axis + 1:]
            j = ids.get(nb)
            if j is not None:
                edges.append((i, j))
    m = {"shape": tuple(mask.shape)}
    m.update(meta or {})
    return from_edges(np.array(cells, dtype=float).reshape(len(cells), mask.ndim), edges, kind, m)


def grid(*shape: int) -> ZipGraph:
    """Full grid of any dimension: grid(7, 7), grid(4, 4, 4), grid(3, 3, 3, 3)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return from_mask(np.ones(shape, dtype=bool), kind="grid")


def remove_edges(graph: ZipGraph, walls: Iterable[tuple[int, int]]) -> ZipGraph:
    """Copy of `graph` with the given edges (walls) removed."""
    walls = {frozenset(w) for w in walls}
    edges = [e for e in graph.edges() if frozenset(e) not in walls]
    meta = dict(graph.meta)
    meta.pop("_lookup", None)
    meta["walls"] = sorted(tuple(sorted(w)) for w in walls)
    return from_edges(graph.coords, edges, graph.kind, meta)


def islands(
    shapes: Sequence[Sequence[int]],
    extra_bridges: int = 0,
    gap: int = 2,
    rng: np.random.Generator | int | None = None,
) -> ZipGraph:
    """Several grid islands laid out side by side along axis 1, joined by bridges.

    Islands i and i+1 are always joined by one bridge between random cells on
    their facing borders (so the whole graph is connected); `extra_bridges`
    adds further random bridges between random island pairs. All islands must
    have the same dimensionality. meta["island"] maps node -> island index and
    meta["bridges"] lists the bridge edges.
    """
    rng = np.random.default_rng(rng)
    dim = len(shapes[0])
    assert all(len(s) == dim for s in shapes), "islands must share dimensionality"
    coords: list[tuple[float, ...]] = []
    edges: list[tuple[int, int]] = []
    island_of: list[int] = []
    members: list[list[int]] = []
    offset = 0
    for k, shape in enumerate(shapes):
        sub = grid(*shape)
        base = len(coords)
        shift = np.zeros(dim)
        if dim > 1:
            shift[1] = offset
        else:
            shift[0] = offset
        coords.extend(tuple(c) for c in (sub.coords + shift).tolist())
        edges.extend((base + u, base + v) for u, v in sub.edges())
        island_of.extend([k] * sub.num_nodes)
        members.append(list(range(base, base + sub.num_nodes)))
        offset += (shape[1] if dim > 1 else shape[0]) + gap

    coords_arr = np.array(coords)
    lay_axis = 1 if dim > 1 else 0

    def border(k: int, side: str) -> list[int]:
        vals = coords_arr[members[k], lay_axis]
        target = vals.max() if side == "right" else vals.min()
        return [v for v in members[k] if coords_arr[v, lay_axis] == target]

    bridges = []
    for k in range(len(shapes) - 1):
        u = int(rng.choice(border(k, "right")))
        v = int(rng.choice(border(k + 1, "left")))
        bridges.append((u, v))
    for _ in range(extra_bridges):
        a, b = sorted(rng.choice(len(shapes), size=2, replace=False).tolist())
        bridges.append((int(rng.choice(members[a])), int(rng.choice(members[b]))))
    edges.extend(bridges)
    return from_edges(
        coords_arr, edges, "islands",
        {"island": island_of, "bridges": bridges, "shapes": [tuple(s) for s in shapes]},
    )


def random_mask(shape: Sequence[int], fill: float = 0.8,
                rng: np.random.Generator | int | None = None) -> ZipGraph:
    """Irregular connected blob: grow a random region covering ~fill of the box."""
    rng = np.random.default_rng(rng)
    shape = tuple(shape)
    total = int(np.prod(shape))
    target = max(2, int(round(fill * total)))
    mask = np.zeros(shape, dtype=bool)
    start = tuple(int(rng.integers(s)) for s in shape)
    mask[start] = True
    frontier = [start]
    count = 1
    while count < target and frontier:
        c = frontier[int(rng.integers(len(frontier)))]
        opts = []
        for axis, s in enumerate(shape):
            for d in (-1, 1):
                x = c[axis] + d
                if 0 <= x < s:
                    nb = c[:axis] + (x,) + c[axis + 1:]
                    if not mask[nb]:
                        opts.append(nb)
        if not opts:
            frontier.remove(c)
            continue
        nb = opts[int(rng.integers(len(opts)))]
        mask[nb] = True
        frontier.append(nb)
        count += 1
    return from_mask(mask, kind="mask")




# ----------------------------------------------------------------------------
# directed restrictions (one-way arrows) and precedence (keys before doors)
# ----------------------------------------------------------------------------
# Both live in graph.meta so that everything that only sees the graph (the
# solver's is_dead_end, the RL env, the rater) honours them:
#   meta["arcs"]       [[u, v], ...]  edge {u, v} may only be traversed u -> v
#   meta["precedence"] [[a, b], ...]  node a must be visited before node b
# Puzzle.to_dict writes them as top-level "arcs" / "precedence" (contract).

def arcs_of(graph: ZipGraph) -> list[tuple[int, int]]:
    return [(int(u), int(v)) for u, v in (graph.meta.get("arcs") or [])]


def precedence_of(graph: ZipGraph) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in (graph.meta.get("precedence") or [])]


def blocked_steps(graph: ZipGraph) -> dict[int, frozenset] | None:
    """node -> neighbours it may NOT step to (reverse direction of an arc); None if no arcs."""
    cache = graph.meta.get("_blocked", False)
    if cache is not False:
        return cache
    arcs = arcs_of(graph)
    out: dict[int, set] | None = None
    if arcs:
        out = {}
        for u, v in arcs:
            out.setdefault(v, set()).add(u)
        out = {k: frozenset(x) for k, x in out.items()}
    graph.meta["_blocked"] = out
    return out


def prerequisites(graph: ZipGraph) -> dict[int, tuple[int, ...]] | None:
    """node -> nodes that must be visited before it; None if no precedence constraints."""
    cache = graph.meta.get("_prereq", False)
    if cache is not False:
        return cache
    pre = precedence_of(graph)
    out = None
    if pre:
        d: dict[int, list] = {}
        for a, b in pre:
            d.setdefault(b, []).append(a)
        out = {k: tuple(x) for k, x in d.items()}
    graph.meta["_prereq"] = out
    return out


def step_allowed(graph: ZipGraph, u: int, w: int, visited=None) -> bool:
    """True if the move u -> w is allowed by arcs and (given `visited`, a container /
    bool array of visited nodes) precedence. Adjacency is NOT checked."""
    bl = blocked_steps(graph)
    if bl is not None and w in bl.get(u, ()):
        return False
    pre = prerequisites(graph)
    if pre is not None and visited is not None:
        is_set = isinstance(visited, (set, frozenset, dict))
        for a in pre.get(w, ()):
            if (a not in visited) if is_set else (not visited[a]):
                return False
    return True


def with_meta(graph: ZipGraph, **updates) -> ZipGraph:
    """Same graph (shared coords/adjacency) with updated meta (caches dropped)."""
    meta = {k: v for k, v in graph.meta.items() if not k.startswith("_")}
    meta.update(updates)
    return ZipGraph(graph.coords, graph.neighbors, graph.kind, meta)


# ----------------------------------------------------------------------------
# new board shapes
# ----------------------------------------------------------------------------
def torus(h: int, w: int) -> ZipGraph:
    """h x w grid with wrap-around edges on all four borders (h, w >= 3).

    meta: torus=[h, w], wrap_edges=[[u, v], ...] (the wrapping edges, also in edges()).
    """
    h, w = int(h), int(w)
    if h < 3 or w < 3:
        raise ValueError("torus needs h, w >= 3")
    g = grid(h, w)
    wraps = [(r * w + (w - 1), r * w) for r in range(h)] + [((h - 1) * w + c, c) for c in range(w)]
    return from_edges(g.coords, g.edges() + wraps, "torus",
                      {"torus": [h, w], "wrap_edges": [list(e) for e in wraps]})


_HEX_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


def hex_board(cells: Sequence[tuple[int, int]], kind: str = "hex", meta=None) -> ZipGraph:
    """Hexagonal cells in axial coordinates (q, r); pointy-top rendering."""
    cells = [tuple(map(int, c)) for c in cells]
    ids = {c: i for i, c in enumerate(cells)}
    edges = []
    for (q, r), i in ids.items():
        for dq, dr in _HEX_DIRS:
            j = ids.get((q + dq, r + dr))
            if j is not None and i < j:
                edges.append((i, j))
    m = {"layout": "hex"}
    m.update(meta or {})
    return from_edges(np.array(cells, dtype=float).reshape(len(cells), 2), edges, kind, m)


def hexagon(side: int) -> ZipGraph:
    """Hexagon of hex cells with `side` cells per side: 3*s*(s-1)+1 cells."""
    s = int(side)
    if s < 1:
        raise ValueError("side must be >= 1")
    cells = [(q, r) for r in range(-(s - 1), s) for q in range(-(s - 1), s) if abs(q + r) <= s - 1]
    return hex_board(cells, meta={"hex_shape": "hexagon", "side": s})


def hex_parallelogram(h: int, w: int) -> ZipGraph:
    """Rhombus of hex cells: q in 0..w-1, r in 0..h-1."""
    cells = [(q, r) for r in range(int(h)) for q in range(int(w))]
    return hex_board(cells, meta={"hex_shape": "rhombus", "shape": [int(h), int(w)]})


def tri_board(cells: Sequence[tuple[int, int]], kind: str = "tri", meta=None) -> ZipGraph:
    """Triangle cells (row, col); UP if (row+col) even (vertical neighbour row+1), else
    DOWN (vertical neighbour row-1); left/right neighbours in the same row.
    The graph is bipartite (UP cells only touch DOWN cells) with max degree 3."""
    cells = [tuple(map(int, c)) for c in cells]
    ids = {c: i for i, c in enumerate(cells)}
    edges = []
    for (r, c), i in ids.items():
        up = (r + c) % 2 == 0
        for nb in ((r, c + 1), ((r + 1, c) if up else (r - 1, c))):
            j = ids.get(nb)
            if j is not None:
                edges.append((i, j))
    m = {"layout": "tri"}
    m.update(meta or {})
    return from_edges(np.array(cells, dtype=float).reshape(len(cells), 2), edges, kind, m)


def tri_rect(rows: int, cols: int) -> ZipGraph:
    """rows x cols triangles (zig-zag sided band); (0, 0) points up."""
    return tri_board([(r, c) for r in range(int(rows)) for c in range(int(cols))],
                     meta={"tri_shape": "rect", "shape": [int(rows), int(cols)]})


def tri_hexagon(side: int) -> ZipGraph:
    """Regular hexagon made of 6*s^2 triangles (s triangles per side)."""
    s = int(side)
    if s < 1:
        raise ValueError("side must be >= 1")
    base = s - 1 if (s - 1) % 2 == 0 else s      # first cell of each top row points up
    cells = []
    for r in range(s):                            # top half: rows widen
        cells += [(r, c) for c in range(base - r, base + 2 * s + r + 1)]
    for j in range(s):                            # bottom half: rows narrow
        k = s - 1 - j
        cells += [(s + j, c) for c in range(base - k, base + 2 * s + k + 1)]
    return tri_board(cells, meta={"tri_shape": "hexagon", "side": s})


_CUBE_FACES = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))   # (axis, 0=low / 1=high)


def cube_surface(n: int) -> ZipGraph:
    """Surface of an n x n x n cube: 6*n^2 cells, 4 neighbours each (across cube edges too).

    Coordinates are the face-cell centres scaled by 2 (integers in 0..2n): the fixed
    axis is 0 or 2n, the other two are odd. meta: cubesurf={"n": n}, face=[0..5 per node]
    (face 2*axis + side: 0 x=0, 1 x=2n, 2 y=0, 3 y=2n, 4 z=0, 5 z=2n).
    Two cells are adjacent iff their centres are at squared distance 4 (same face)
    or 2 (neighbours across a cube edge).
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    coords, face = [], []
    for f, (axis, side) in enumerate(_CUBE_FACES):
        for a in range(n):
            for b in range(n):
                p = [0, 0, 0]
                others = [x for x in range(3) if x != axis]
                p[axis] = 2 * n * side
                p[others[0]] = 2 * a + 1
                p[others[1]] = 2 * b + 1
                coords.append(tuple(p))
                face.append(f)
    ids = {c: i for i, c in enumerate(coords)}
    edges = set()
    offs = [d for d in product((-2, -1, 0, 1, 2), repeat=3)
            if sum(x * x for x in d) in (2, 4) and max(abs(x) for x in d) <= 2]
    for c, i in ids.items():
        for d in offs:
            j = ids.get((c[0] + d[0], c[1] + d[1], c[2] + d[2]))
            if j is not None and i < j:
                if sum(x * x for x in d) == 4 and face[i] != face[j]:
                    continue
                edges.add((i, j))
    return from_edges(np.array(coords, dtype=float), sorted(edges), "cubesurf",
                      {"cubesurf": {"n": n}, "face": face})


def overpass_grid(h: int, w: int, cells: Iterable[tuple[int, int]],
                  walls: Iterable[tuple[tuple[int, int], tuple[int, int]]] = ()) -> ZipGraph:
    """h x w grid where each cell in `cells` is an overpass: two nodes with the same
    coordinates, an H node linked only to its left/right neighbours and a V node
    linked only to its up/down neighbours (so the path crosses it twice).

    meta.overpass = [{"cell": [r, c], "h": nodeH, "v": nodeV}, ...].
    `walls`: optional pairs of adjacent cells ((r, c), (r2, c2)) with no edge.
    """
    h, w = int(h), int(w)
    ov = {tuple(map(int, c)) for c in cells}
    coords, hid, vid = [], {}, {}
    for r in range(h):
        for c in range(w):
            hid[(r, c)] = len(coords)
            coords.append((r, c))
            if (r, c) in ov:
                vid[(r, c)] = len(coords)
                coords.append((r, c))
            else:
                vid[(r, c)] = hid[(r, c)]
    wallset = {frozenset((tuple(a), tuple(b))) for a, b in walls}
    edges = []
    for r in range(h):
        for c in range(w):
            if c + 1 < w and frozenset(((r, c), (r, c + 1))) not in wallset:
                edges.append((hid[(r, c)], hid[(r, c + 1)]))
            if r + 1 < h and frozenset(((r, c), (r + 1, c))) not in wallset:
                edges.append((vid[(r, c)], vid[(r + 1, c)]))
    meta = {"shape": (h, w), "overpass": [{"cell": [r, c], "h": hid[(r, c)], "v": vid[(r, c)]}
                                          for r, c in sorted(ov)]}
    return from_edges(np.array(coords, dtype=float).reshape(len(coords), 2), edges, "overpass", meta)


__all__ = ["ZipGraph", "from_edges", "from_mask", "grid", "remove_edges", "islands", "random_mask",
           "arcs_of", "precedence_of", "blocked_steps", "prerequisites", "step_allowed", "with_meta",
           "torus", "hex_board", "hexagon", "hex_parallelogram", "tri_board", "tri_rect",
           "tri_hexagon", "cube_surface", "overpass_grid"]
