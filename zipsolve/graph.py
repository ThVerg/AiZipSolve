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
    return ZipGraph(coords, tuple(tuple(sorted(a)) for a in adj), kind, dict(meta or {}))


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


__all__ = ["ZipGraph", "from_edges", "from_mask", "grid", "remove_edges", "islands", "random_mask"]
