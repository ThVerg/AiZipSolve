"""Visualisation of Zip puzzles on arbitrary graphs.

Public API
----------
draw(puzzle, path=None, ax=None, title=None, partial=False, view="auto") -> Figure
draw_slices(puzzle, path=None, axis=2, ax=None, title=None, partial=False) -> Figure
save(puzzle, path, filename, dpi=150, **kw)
animate(puzzle, path, filename, fps=10, max_frames=150)
to_ascii(puzzle, path=None, mode="arrows") -> str

Rendering is chosen from the graph:

* 2D grid-like graphs (grid / walls / mask / islands): LinkedIn-style board with
  rounded cells, walls as thick bars, bridges as dashed arcs, islands tinted.
* 3D grid-like graphs: an mplot3d view (``view="slices"`` or ``draw_slices``
  gives a row of 2D layer panels instead).
* 4D grid-like graphs: a grid of 2D panels (panel row = coord 2, col = coord 3).
* anything else: spring-layout node/edge drawing.

A ``path`` may be a prefix of a solution (e.g. an RL agent that got stuck); the
head of a partial path is highlighted with a ring.

All sizes of cells, path and walls are in data units, so the pictures scale
cleanly; only font sizes are derived from the axes size.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from .puzzle import Puzzle

# ---------------------------------------------------------------- palette
PATH_CMAP = LinearSegmentedColormap.from_list(
    "zip_path", ["#ff9f1c", "#ff5a4e", "#d63a8f", "#7b3fd6", "#2f6fe0"])
BG = "#ffffff"
CELL = "#f2eee8"
CELL_EDGE = "#dcd4c8"
WALL = "#25252b"
CHECKPOINT = "#141414"
BRIDGE = "#8b909a"
EDGE_GENERIC = "#c9cdd4"
ISLAND_TINTS = ["#e3eefc", "#fde8dc", "#e2f3e2", "#f1e6fa", "#fbf3d2", "#dff3f3",
                "#fbe3ec", "#ececec"]
AXIS_NAME = {0: "y", 1: "x", 2: "z", 3: "w", 4: "v", 5: "u"}

CELL_SIZE = 0.92
PATH_WIDTH = 0.34
CP_RADIUS = 0.30


# ================================================================ helpers
def _is_partial(puzzle: Puzzle, path, partial: bool) -> bool:
    return bool(path) and (partial or len(path) < puzzle.num_nodes)


def _path_color(t: float):
    return PATH_CMAP(min(max(t, 0.0), 1.0))


def _grid_like(puzzle: Puzzle) -> bool:
    """Integer coordinates and every edge unit length or a declared bridge."""
    g = puzzle.graph
    c = g.coords
    if g.num_nodes == 0 or not np.allclose(c, np.round(c)):
        return False
    bridges = {frozenset(map(int, b)) for b in g.meta.get("bridges", [])}
    for u, v in g.edges():
        if abs(np.abs(c[u] - c[v]).sum() - 1) > 1e-9 and frozenset((u, v)) not in bridges:
            return False
    return True


def _cell_pts(ax, xspan: float, yspan: float) -> float:
    """Size of one data unit in points (for fonts), given the axes' data span."""
    fig = ax.figure
    bb = ax.get_position()
    w_in = bb.width * fig.get_figwidth()
    h_in = bb.height * fig.get_figheight()
    return 72.0 * min(w_in / max(xspan, 1e-9), h_in / max(yspan, 1e-9))


def _arc(p, q, bend: float = 0.22, n: int = 32) -> np.ndarray:
    """Quadratic Bezier arc from p to q bulging to one (deterministic) side."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.hypot(*d)
    perp = np.array([-d[1], d[0]]) / max(L, 1e-9)
    if perp[1] < 0 or (abs(perp[1]) < 1e-9 and perp[0] < 0):
        perp = -perp
    if (abs(d[0]) < 1e-9 or abs(d[1]) < 1e-9) and L <= 3.01:
        bend = 0.0  # short straight hop across the gap between two islands: draw a plank
    ctrl = (p + q) / 2 + perp * min(bend * L, 1.6)
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * ctrl + t ** 2 * q


def _thick_polyline(pts: np.ndarray, t0: float, t1: float, width: float,
                    patches: list, colors: list, sub: int = 4) -> None:
    """Append rectangle/circle patches forming a thick gradient polyline."""
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    total = lens.sum()
    if total <= 0:
        return
    cum = np.concatenate([[0], np.cumsum(lens)])
    hw = width / 2
    pieces = max(1, int(round(sub / max(len(seg), 1)))) if len(seg) > 1 else sub
    for k in range(len(seg)):
        a, b = pts[k], pts[k + 1]
        if lens[k] <= 0:
            continue
        u = seg[k] / lens[k]
        nrm = np.array([-u[1], u[0]]) * hw
        for j in range(pieces):
            s0, s1 = j / pieces, (j + 1) / pieces
            pa = a + (b - a) * s0 - u * 0.004
            pb = a + (b - a) * s1 + u * 0.004
            frac = (cum[k] + lens[k] * (s0 + s1) / 2) / total
            patches.append(Polygon([pa + nrm, pb + nrm, pb - nrm, pa - nrm], closed=True))
            colors.append(_path_color(t0 + (t1 - t0) * frac))
        if k > 0:  # joint
            patches.append(Circle(a, hw))
            colors.append(_path_color(t0 + (t1 - t0) * cum[k] / total))


def _node_joints(path, pos_of, nodeset, width, N, patches, colors):
    """Round joints at path ends and turns so corners/caps look smooth."""
    for i, v in enumerate(path):
        if v not in nodeset:
            continue
        p = np.asarray(pos_of(v), float)
        prev = path[i - 1] if i > 0 and path[i - 1] in nodeset else None
        nxt = path[i + 1] if i + 1 < len(path) and path[i + 1] in nodeset else None
        if prev is not None and nxt is not None:
            a = p - np.asarray(pos_of(prev), float)
            b = np.asarray(pos_of(nxt), float) - p
            if abs(a[0] * b[1] - a[1] * b[0]) < 1e-9 and a @ b > 0:
                continue  # straight through: no joint needed
        patches.append(Circle(p, width / 2))
        colors.append(_path_color(i / max(N - 1, 1)))


# ================================================================ planar core
def _draw_planar(ax, puzzle: Puzzle, nodes: Sequence[int], pos: dict, path,
                 *, generic: bool = False, head: int | None = None,
                 badges: Sequence[tuple] = (), extent=None, show_title=None) -> None:
    """Draw the sub-board `nodes` (positions `pos`) of `puzzle` on `ax`.

    badges: (node, text, color, filled, corner) with corner in {"out", "in"}.
    """
    g = puzzle.graph
    N = g.num_nodes
    nodeset = set(nodes)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    if extent is None:
        xy = np.array([pos[v] for v in nodes]) if nodes else np.zeros((1, 2))
        extent = _extent(g, nodes, pos, generic)
    pad = 0.62
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    unit = _cell_pts(ax, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad)
    if show_title:
        ax.set_title(show_title, fontsize=max(8, min(13, unit * 0.45)), color="#333", pad=4)

    island = g.meta.get("island")
    cells_w = PATH_WIDTH if not generic else 0.2
    cp_r = CP_RADIUS if not generic else 0.27

    # ---- cells / nodes
    if generic:
        segs = [(pos[u], pos[v]) for u, v in g.edges() if u in nodeset and v in nodeset]
        ax.add_collection(LineCollection(segs, colors=EDGE_GENERIC, linewidths=max(0.5, unit * 0.025),
                                         zorder=1))
        for v in nodes:
            ax.add_patch(Circle(pos[v], 0.18, fc=CELL, ec=CELL_EDGE, lw=0.8, zorder=2))
    else:
        r = 0.14
        s = CELL_SIZE - 2 * r
        for v in nodes:
            x, y = pos[v]
            fc = ISLAND_TINTS[int(island[v]) % len(ISLAND_TINTS)] if island is not None else CELL
            ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s,
                                        boxstyle=f"round,pad={r}", fc=fc, ec=CELL_EDGE,
                                        lw=max(0.4, unit * 0.012), zorder=1))
        # walls: present, unit-distance cells without an edge
        lookup = {(round(pos[v][0]), round(pos[v][1])): v for v in nodes}
        for (x, y), u in lookup.items():
            for dx, dy in ((1, 0), (0, -1)):
                v = lookup.get((x + dx, y + dy))
                if v is not None and not g.has_edge(u, v):
                    mx, my = x + dx / 2, y + dy / 2
                    t = 0.13
                    if dx:  # vertical bar
                        rect = (mx - t / 2, my - 0.5, t, 1.0)
                    else:
                        rect = (mx - 0.5, my - t / 2, 1.0, t)
                    ax.add_patch(FancyBboxPatch(rect[:2], rect[2], rect[3],
                                                boxstyle="round,pad=0.02", fc=WALL, ec="none",
                                                zorder=3))
        # bridges / non-unit edges drawn as dashed arcs
        for u, v in g.edges():
            if u in nodeset and v in nodeset:
                p, q = np.array(pos[u]), np.array(pos[v])
                if abs(np.abs(p - q).sum() - 1) > 1e-9:
                    a, b = (u, v) if u < v else (v, u)
                    pts = _arc(pos[a], pos[b])
                    ax.plot(pts[:, 0], pts[:, 1], ls=(0, (4, 3)), color=BRIDGE,
                            lw=max(1.0, unit * 0.06), zorder=2, solid_capstyle="round")
                    for e in (p, q):
                        ax.add_patch(Circle(e, 0.09, fc=BRIDGE, ec="none", zorder=2))

    # ---- path
    if path:
        patches, colors = [], []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u not in nodeset or v not in nodeset:
                continue
            p, q = np.array(pos[u]), np.array(pos[v])
            if not generic and abs(np.abs(p - q).sum() - 1) > 1e-9:
                a, b = (u, v) if u < v else (v, u)
                pts = _arc(pos[a], pos[b])
                if a != u:
                    pts = pts[::-1]
            else:
                pts = np.array([p, q])
            _thick_polyline(pts, i / max(N - 1, 1), (i + 1) / max(N - 1, 1), cells_w,
                            patches, colors)
        _node_joints(path, lambda v: pos[v], nodeset, cells_w, N, patches, colors)
        if patches:
            # matching thin edges hide antialiasing seams between pieces
            ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors=colors,
                                              linewidths=max(0.3, unit * 0.01), zorder=4))

    # ---- head of a partial path
    if head is not None and head in nodeset:
        i = len(path) - 1
        col = _path_color(i / max(N - 1, 1))
        rr = 0.44 if not generic else 0.3
        ax.add_patch(Circle(pos[head], rr, fc="none", ec=col, lw=max(1.5, unit * 0.09), zorder=6))
        ax.add_patch(Circle(pos[head], rr * 0.28, fc="white", ec=col, lw=max(1, unit * 0.04),
                            zorder=7))

    # ---- checkpoints
    labels = puzzle.checkpoint_label()
    fs = max(4.5, min(22, unit * 0.30))
    for v, lab in labels.items():
        if v not in nodeset:
            continue
        ax.add_patch(Circle(pos[v], cp_r, fc=CHECKPOINT, ec="white", lw=max(0.5, unit * 0.02),
                            zorder=8))
        txt = str(lab)
        ax.text(pos[v][0], pos[v][1] - 0.01, txt, ha="center", va="center", color="white",
                fontsize=fs * (0.8 if len(txt) > 2 else 1.0), fontweight="bold", zorder=9)

    # ---- badges for moves leaving the panel
    bfs = max(4, min(14, unit * 0.2))
    for v, text, col, filled, corner in badges:
        if v not in nodeset:
            continue
        dx, dy = (0.3, 0.3) if corner == "out" else (-0.3, -0.3)
        x, y = pos[v][0] + dx, pos[v][1] + dy
        ax.text(x, y, text, ha="center", va="center", fontsize=bfs, fontweight="bold",
                color="white" if filled else col, zorder=10,
                bbox=dict(boxstyle="round,pad=0.18", fc=col if filled else "white",
                          ec=col, lw=max(0.6, unit * 0.03)))


def _extent(g, nodes, pos, generic=False):
    """Bounding box of node positions plus any bridge arcs between them."""
    nodeset = set(nodes)
    pts = [np.array([pos[v] for v in nodes]) if nodes else np.zeros((1, 2))]
    if not generic:
        for u, v in g.edges():
            if u in nodeset and v in nodeset:
                if abs(np.abs(np.subtract(pos[u], pos[v])).sum() - 1) > 1e-9:
                    pts.append(_arc(pos[min(u, v)], pos[max(u, v)]))
    xy = np.vstack(pts)
    return (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())


def _pos2d(coords: np.ndarray, i: int = 0, j: int = 1) -> dict:
    """Map node -> (x, y) with coord j as x and coord i as downward y."""
    return {v: (float(c[j]), -float(c[i])) for v, c in enumerate(coords)}


def _new_fig(w: float, h: float):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig


def _sub_gridspec(fig, ax, nrows, ncols, **kw):
    """Grid of panels filling the area of an existing axes (which is removed)."""
    spec = ax.get_subplotspec()
    if spec is not None:
        ax.remove()
        return spec.subgridspec(nrows, ncols, **kw)
    bb = ax.get_position()
    ax.remove()
    return fig.add_gridspec(nrows, ncols, left=bb.x0, right=bb.x1, bottom=bb.y0, top=bb.y1, **kw)


# ================================================================ 2D
def _draw_2d(puzzle, path, ax, title, partial):
    g = puzzle.graph
    pos = _pos2d(g.coords)
    x0, x1, y0, y1 = _extent(g, list(range(g.num_nodes)), pos)
    w = x1 - x0 + 1.24
    h = y1 - y0 + 1.24
    if ax is None:
        s = float(np.clip(7.0 / max(w, h), 0.28, 0.62))
        fig = _new_fig(w * s + 0.2, h * s + (0.55 if title else 0.2))
        top = 1 - (0.5 / fig.get_figheight() if title else 0.1 / fig.get_figheight())
        ax = fig.add_axes([0.1 / fig.get_figwidth(), 0.1 / fig.get_figheight(),
                           1 - 0.2 / fig.get_figwidth(),
                           top - 0.1 / fig.get_figheight()])
    head = path[-1] if _is_partial(puzzle, path, partial) else None
    _draw_planar(ax, puzzle, range(g.num_nodes), pos, path, head=head)
    if title:
        ax.set_title(title, fontsize=12, color="#222", pad=6, fontweight="bold")
    return ax.figure


def _draw_generic(puzzle, path, ax, title, partial):
    g = puzzle.graph
    if g.dim == 2:
        xy = g.coords.astype(float).copy()
    else:
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(g.num_nodes))
            G.add_edges_from(g.edges())
            lay = nx.spring_layout(G, seed=0, iterations=200)
            xy = np.array([lay[v] for v in range(g.num_nodes)])
        except ImportError:  # circle layout fallback
            ang = np.linspace(0, 2 * np.pi, g.num_nodes, endpoint=False)
            xy = np.c_[np.cos(ang), np.sin(ang)]
    if g.num_nodes > 1:  # scale so the typical nearest-neighbour distance is 1
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        nn = np.median(d.min(axis=1))
        if np.isfinite(nn) and nn > 0:
            xy = xy / nn
    pos = {v: (float(xy[v, 0]), float(xy[v, 1])) for v in range(g.num_nodes)}
    if ax is None:
        w = np.ptp(xy[:, 0]) + 1.2
        h = np.ptp(xy[:, 1]) + 1.2
        s = 7.0 / max(w, h)
        fig = _new_fig(w * s, h * s + (0.4 if title else 0))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.92 if title else 0.98)
    head = path[-1] if _is_partial(puzzle, path, partial) else None
    _draw_planar(ax, puzzle, range(g.num_nodes), pos, path, generic=True, head=head)
    if title:
        ax.set_title(title, fontsize=12, color="#222", fontweight="bold")
    return ax.figure


# ================================================================ panels (3D slices, 4D)
def _draw_panels(puzzle: Puzzle, path, ax, title, partial, *, panel_axes: Sequence[int],
                 plane: tuple[int, int], ncols_wrap: int | None, legend: str,
                 cross_text: Callable[[np.ndarray], str]):
    """Split nodes by the coordinates in `panel_axes` and draw one 2D panel each.

    With one panel axis the panels form a (wrapped) row; with two, panel row =
    panel_axes[0] value and panel column = panel_axes[1] value.
    """
    g = puzzle.graph
    c = g.coords
    i, j = plane
    pos = _pos2d(c, i, j)
    vals = [sorted(set(c[:, a].tolist())) for a in panel_axes]
    if len(panel_axes) == 1:
        L = len(vals[0])
        ncols = min(L, ncols_wrap or L)
        nrows = math.ceil(L / ncols)

        def cell_of(v):
            k = vals[0].index(c[v, panel_axes[0]])
            return divmod(k, ncols)

        def ptitle(r, cc):
            k = r * ncols + cc
            return f"{AXIS_NAME.get(panel_axes[0], panel_axes[0])} = {int(vals[0][k])}" \
                if k < L else None
    else:
        nrows, ncols = len(vals[0]), len(vals[1])

        def cell_of(v):
            return vals[0].index(c[v, panel_axes[0]]), vals[1].index(c[v, panel_axes[1]])

        def ptitle(r, cc):
            return (f"{AXIS_NAME.get(panel_axes[0])}={int(vals[0][r])}, "
                    f"{AXIS_NAME.get(panel_axes[1])}={int(vals[1][cc])}")

    groups: dict[tuple[int, int], list[int]] = {}
    for v in range(g.num_nodes):
        groups.setdefault(cell_of(v), []).append(v)
    extent = _extent(g, list(range(g.num_nodes)), pos)
    pw = extent[1] - extent[0] + 1.24
    ph = extent[3] - extent[2] + 1.24

    # badges for cross-panel moves
    N = g.num_nodes
    badges = []
    if path:
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            if cell_of(u) != cell_of(v):
                col = _path_color((k + 0.5) / max(N - 1, 1))
                badges.append((u, cross_text(c[v] - c[u]), col, True, "out"))
                badges.append((v, cross_text(c[u] - c[v]), col, False, "in"))

    title_h = 0.5 if title else 0.0
    legend_h = 0.2 + 0.17 * (legend.count("\n") + 1) if legend else 0.0
    if ax is None:
        s = float(np.clip(10.5 / max(pw * ncols, ph * nrows * 1.15), 0.22, 0.6))
        fig = _new_fig(pw * s * ncols + 0.3, (ph * s + 0.3) * nrows + title_h + legend_h + 0.1)
        H = fig.get_figheight()
        gs = fig.add_gridspec(nrows, ncols, left=0.01, right=0.99,
                              top=1 - (title_h + 0.05) / H, bottom=(legend_h + 0.05) / H,
                              wspace=0.06, hspace=0.25)
    else:
        fig = ax.figure
        gs = _sub_gridspec(fig, ax, nrows, ncols, wspace=0.06, hspace=0.25)
    head = path[-1] if _is_partial(puzzle, path, partial) else None
    for r in range(nrows):
        for cc in range(ncols):
            pt = ptitle(r, cc)
            if pt is None:
                continue
            pax = fig.add_subplot(gs[r, cc])
            _draw_planar(pax, puzzle, groups.get((r, cc), []), pos, path, head=head,
                         badges=badges, extent=extent, show_title=pt)
    if title:
        fig.suptitle(title, fontsize=13, color="#222", fontweight="bold",
                     y=1 - 0.08 / fig.get_figheight(), va="top")
    if legend:
        fig.text(0.5, 0.12 / fig.get_figheight(), legend, ha="center", va="bottom",
                 fontsize=9, color="#555")
    return fig


def draw_slices(puzzle: Puzzle, path: list[int] | None = None, axis: int = 2, ax=None,
                title: str | None = None, partial: bool = False) -> Figure:
    """3D graph as a row of 2D layer panels along `axis`.

    A solid badge marks a node whose next step leaves the layer (glyph = where
    it goes: ▲ higher layer, ▼ lower layer); a hollow badge marks a node entered
    from another layer (glyph = where it came from).
    """
    g = puzzle.graph
    assert g.dim == 3, "draw_slices needs a 3D graph"
    plane = tuple(a for a in range(3) if a != axis)
    name = AXIS_NAME.get(axis, str(axis))

    def cross(d):
        return "▲" if d[axis] > 0 else "▼"

    legend = (f"solid badge: next step moves to layer {name}+1 (▲) / {name}−1 (▼)\n"
              f"hollow badge: arrived from layer {name}+1 (▲) / {name}−1 (▼)")
    return _draw_panels(puzzle, path, ax, title, partial, panel_axes=(axis,), plane=plane,
                        ncols_wrap=5, legend=legend if path else "", cross_text=cross)


def _draw_4d(puzzle, path, ax, title, partial):
    def cross(d):
        parts = []
        for a in (2, 3):
            if d[a]:
                parts.append(f"{AXIS_NAME[a]}{'+' if d[a] > 0 else '−'}")
        return "".join(parts) or "·"

    legend = ("panel rows: z, columns: w\nsolid badge: next step goes to slice z±1 / w±1\n"
              "hollow badge: arrived from slice z±1 / w±1")
    return _draw_panels(puzzle, path, ax, title, partial, panel_axes=(2, 3), plane=(0, 1),
                        ncols_wrap=None, legend=legend if path else "", cross_text=cross)


# ================================================================ 3D
def _draw_3d(puzzle, path, ax, title, partial):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    g = puzzle.graph
    c = g.coords
    X, Y, Z = c[:, 1], c[:, 0], c[:, 2]
    if ax is None:
        fig = _new_fig(7.5, 7.5)
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        fig.subplots_adjust(0, 0, 1, 0.95 if title else 1)
    elif getattr(ax, "name", "") != "3d":
        fig = ax.figure
        spec = ax.get_subplotspec()
        bb = ax.get_position()
        ax.remove()
        if spec is not None:
            ax = fig.add_subplot(spec, projection="3d", computed_zorder=False)
        else:
            ax = fig.add_axes(bb, projection="3d", computed_zorder=False)
    fig = ax.figure
    ax.set_facecolor(BG)
    N = g.num_nodes
    span = max(np.ptp(X), np.ptp(Y), np.ptp(Z), 1)

    # lattice edges (missing edges reveal walls)
    segs = [[(X[u], Y[u], Z[u]), (X[v], Y[v], Z[v])] for u, v in g.edges()]
    if segs:
        ax.add_collection3d(Line3DCollection(segs, colors="#b9bec7", linewidths=0.7, alpha=0.45,
                                            zorder=1))
    size = float(np.clip(2600 / (span + 1) ** 2, 12, 260))
    ax.scatter(X, Y, Z, s=size, c="#dcd6cc", edgecolors="#a8a196", linewidths=0.5, alpha=0.35,
               depthshade=False, zorder=2)

    if path:
        pts, cols = [], []
        sub = 3
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            p = np.array([X[u], Y[u], Z[u]])
            q = np.array([X[v], Y[v], Z[v]])
            for s_ in range(sub):
                a = p + (q - p) * s_ / sub
                b = p + (q - p) * (s_ + 1) / sub
                pts.append([a, b])
                cols.append(_path_color((k + (s_ + 0.5) / sub) / max(N - 1, 1)))
        lw = float(np.clip(40 / (span + 1), 2.5, 10))
        if pts:
            ax.add_collection3d(Line3DCollection(pts, colors=cols, linewidths=lw,
                                                 capstyle="round", zorder=3))
        if _is_partial(puzzle, path, partial):
            h = path[-1]
            ax.scatter([X[h]], [Y[h]], [Z[h]], s=size * 3.2, facecolors="none",
                       edgecolors=[_path_color((len(path) - 1) / max(N - 1, 1))],
                       linewidths=2.5, depthshade=False, zorder=4)

    labels = puzzle.checkpoint_label()
    cps = list(labels)
    ax.scatter(X[cps], Y[cps], Z[cps], s=size * 3.2, c=CHECKPOINT, edgecolors="white",
               linewidths=1.0, depthshade=False, alpha=1.0, zorder=5)
    fs = float(np.clip(80 / (span + 1), 6, 14))
    for v, lab in labels.items():
        ax.text(X[v], Y[v], Z[v], str(lab), color="white", fontsize=fs, fontweight="bold",
                ha="center", va="center", zorder=6)

    ax.set_box_aspect((max(np.ptp(X), 1), max(np.ptp(Y), 1), max(np.ptp(Z), 1)))
    ax.invert_yaxis()
    ax.view_init(elev=24, azim=-60)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13, color="#222", fontweight="bold")
    return fig


# ================================================================ public API
def draw(puzzle: Puzzle, path: list[int] | None = None, ax=None, title: str | None = None,
         partial: bool = False, view: str = "auto") -> Figure:
    """Draw a puzzle (and optionally a full or partial path); returns the Figure.

    view: "auto" (by dimension), "2d", "3d", "slices" (3D as layer panels),
    "panels" (4D slice grid) or "generic" (spring layout).
    """
    path = list(path) if path is not None else None
    g = puzzle.graph
    if view == "auto":
        if not _grid_like(puzzle):
            view = "generic"
        else:
            view = {1: "2d", 2: "2d", 3: "3d", 4: "panels"}.get(g.dim, "generic")
    if view == "2d":
        if g.dim == 1:  # pad 1D graphs to a single row
            p1 = Puzzle(type(g)(np.c_[np.zeros(g.num_nodes), g.coords], g.neighbors, g.kind,
                                dict(g.meta)), puzzle.checkpoints, puzzle.solution)
            return _draw_2d(p1, path, ax, title, partial)
        return _draw_2d(puzzle, path, ax, title, partial)
    if view == "3d":
        return _draw_3d(puzzle, path, ax, title, partial)
    if view == "slices":
        return draw_slices(puzzle, path, 2, ax, title, partial)
    if view == "panels":
        return _draw_4d(puzzle, path, ax, title, partial)
    if view == "generic":
        return _draw_generic(puzzle, path, ax, title, partial)
    raise ValueError(f"unknown view {view!r}")


def save(puzzle: Puzzle, path, filename, dpi: int = 150, slices: bool = False, **kw) -> None:
    """Draw and save to `filename` (format from extension). kw go to draw()."""
    fig = draw_slices(puzzle, path, **kw) if slices else draw(puzzle, path, **kw)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def animate(puzzle: Puzzle, path, filename, fps: int = 10, max_frames: int = 150,
            view: str = "auto", dpi: int = 90, hold: float = 1.5) -> None:
    """GIF of the path being drawn step by step (matplotlib PillowWriter)."""
    from matplotlib.animation import PillowWriter

    path = list(path)
    n = len(path)
    ks = sorted(set(np.linspace(1, n, min(n, max_frames)).round().astype(int).tolist()))
    ks += [n] * int(round(hold * fps))
    fig = draw(puzzle, path, view=view, title=f"step {n}/{puzzle.num_nodes}")
    boxes = [a.get_position() for a in fig.axes]
    rect = [min(b.x0 for b in boxes), min(b.y0 for b in boxes)]
    rect += [max(b.x1 for b in boxes) - rect[0], max(b.y1 for b in boxes) - rect[1]]
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(filename), dpi=dpi):
        for k in ks:
            fig.clear()
            ax = fig.add_axes(rect)
            draw(puzzle, path[:k], ax=ax, view=view, title=f"step {k}/{puzzle.num_nodes}")
            writer.grab_frame(facecolor=BG)
    plt.close(fig)


def to_ascii(puzzle: Puzzle, path: Sequence[int] | None = None, mode: str = "arrows") -> str:
    """Text rendering of a 2D puzzle.

    Cells show the checkpoint number, or (with a path) an arrow to the next
    cell (> < ^ v, ~ for a bridge), '@' for the head of a partial path, '.' for
    unvisited cells; absent cells are blank. mode="order" shows the 1-based
    position of each cell along the path instead. Walls are drawn as '|' and
    '---'; the board is framed with '+', '-' and '|'.
    """
    g = puzzle.graph
    if g.dim != 2:
        raise ValueError("to_ascii supports 2D graphs only")
    c = np.round(g.coords).astype(int)
    r0, q0 = c.min(axis=0)
    R, C = c.max(axis=0) - (r0, q0) + 1
    at = {(int(r - r0), int(q - q0)): v for v, (r, q) in enumerate(c)}
    labels = puzzle.checkpoint_label()
    path = list(path) if path else []
    order = {v: i for i, v in enumerate(path)}
    partial = bool(path) and len(path) < g.num_nodes

    def content(v):
        if mode == "order" and v in order:
            return str(order[v] + 1)
        if v in labels:
            return str(labels[v])
        if v in order:
            i = order[v]
            if i == len(path) - 1:
                return "@" if partial else "*"
            w = path[i + 1]
            d = c[w] - c[v]
            return {(0, 1): ">", (0, -1): "<", (-1, 0): "^", (1, 0): "v"}.get(tuple(d), "~")
        return "."

    width = max(3, max(len(content(v)) for v in range(g.num_nodes)) + 2)

    def wall(u, v):
        return u is not None and v is not None and not g.has_edge(u, v)

    lines = ["+" + "-" * (C * (width + 1) - 1) + "+"]
    for r in range(R):
        row = "|"
        for q in range(C):
            v = at.get((r, q))
            row += (content(v) if v is not None else "").center(width)
            if q < C - 1:
                row += "|" if wall(v, at.get((r, q + 1))) else " "
        lines.append(row + "|")
        if r < R - 1:
            sep = "|"
            for q in range(C):
                sep += ("-" * width) if wall(at.get((r, q)), at.get((r + 1, q))) else " " * width
                if q < C - 1:
                    sep += " "
            lines.append(sep + "|")
    lines.append(lines[0])
    return "\n".join(lines)


__all__ = ["draw", "draw_slices", "save", "animate", "to_ascii", "PATH_CMAP"]
