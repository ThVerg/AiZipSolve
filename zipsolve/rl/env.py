"""Gymnasium environment for Zip puzzles on arbitrary graphs.

State: visited set, head node (current end of the path), index of the next
checkpoint to reach. An action is the id of the node to move to; legal actions
are the unvisited neighbours of the head that are not a later checkpoint out of
order (and the final checkpoint only as the very last move).

Observations are dicts of variable size (the graph changes between episodes):
    x           (n, F) float32  dimension-agnostic node features (see FEATURES)
    edge_index  (2, 2E) int64   both directions of every edge
    action_mask (n,)   bool
    head        ()     int64
    neighbors   (n, D) int64   adjacency table padded with -1 (fast aggregation)
No raw coordinates are used, so a policy trained on 2D grids can be run on 3D,
4D, islands, masks, ...

Feature versions (``ZipEnv(feature_version=...)``, default = latest):
    1  the original 17 features (FEATURES_V1). Checkpoints saved without a
       ``feature_version`` key were trained on these.
    2  FEATURES_V1 followed by 12 structural features (FEATURES_V2). The first
       17 columns are bit-for-bit identical to version 1, so a version-1 model
       can consume version-2 observations by slicing (ZipGNN does this).
Use ``zipsolve.rl.gnn.make_env_for_model`` / ``env_kwargs_from_meta`` to build
the env matching a checkpoint.

Version-2 features (R = remaining graph = unvisited nodes + head; m = #unvisited):
    dist_next_cp_ord  BFS distance from the next checkpoint through unvisited
                      nodes where *later* checkpoints (index > next) are leaves:
                      they get a distance but are never expanded, so no
                      order-violating shortcut is counted. The head is also a
                      leaf target, so its value is the order-respecting
                      distance head -> next checkpoint. Unreachable = 1; / m.
    dist_head_ord     same rule, BFS from the head (the next checkpoint is
                      expandable, later checkpoints are leaves).
    is_art            articulation point of R (iterative Tarjan rooted at head;
                      head uses the root rule: >= 2 DFS children).
    bridge_frac       number of incident bridges of R / max degree.
    cut_beyond        for an articulation point v != head: #nodes that removing
                      v cuts off from the head, / m. The path must enter that
                      region through v and can never leave it, so it must hold
                      the end (else the state is dead).
    cut_min           for an articulation point: size of the smallest component
                      of R - v, / m.
    cut_has_end       1 if the end lies in the region cut off from the head by v.
    bipartite         global: the whole graph is bipartite (else the 4 parity
                      features below are 0).
    colour            node colour (0/1) of a fixed 2-colouring.
    same_colour_head  node has the head's colour.
    parity_excess     global: (#unvisited nodes with the head's colour - m//2) / m.
                      A Hamiltonian path head -> end over the m remaining nodes
                      needs exactly m//2 of them with the head's colour.
    parity_ok         global: counts match and the end's colour is consistent
                      (end has the head's colour iff m is even).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..puzzle import Puzzle

FEATURES = [
    "visited",            # 0 node already on the path
    "is_head",            # 1 current end of the path
    "is_next_cp",         # 2 next checkpoint to reach
    "is_future_cp",       # 3 unvisited checkpoint after the next one
    "cp_rank",            # 4 for unvisited checkpoints: 1 for next, decreasing to ~0 for last
    "is_final",           # 5 the final checkpoint (path must end here)
    "free_deg",           # 6 unvisited neighbours / max degree
    "deg",                # 7 degree / max degree
    "reachable",          # 8 reachable from head through unvisited nodes
    "dist_head",          # 9 BFS distance from head through unvisited / #unvisited
    "dist_next_cp",       # 10 BFS distance from next checkpoint through unvisited / #unvisited
    "legal",              # 11 legal action
    "forced",             # 12 unvisited, not final, with <=1 free neighbour besides head (dead end)
    "adj_head",           # 13 adjacent to head
    "frac_done",          # 14 global: fraction of nodes visited
    "frac_cp_done",       # 15 global: fraction of checkpoints reached
    "is_start",           # 16 first node of the path (checkpoint 1)
]
FEATURES_V1 = list(FEATURES)
FEATURES_V2 = FEATURES_V1 + [
    "dist_next_cp_ord",   # 17
    "dist_head_ord",      # 18
    "is_art",             # 19
    "bridge_frac",        # 20
    "cut_beyond",         # 21
    "cut_min",            # 22
    "cut_has_end",        # 23
    "bipartite",          # 24 global
    "colour",             # 25
    "same_colour_head",   # 26
    "parity_excess",      # 27 global
    "parity_ok",          # 28 global
]
FEATURE_SETS = {1: FEATURES_V1, 2: FEATURES_V2}
LATEST_FEATURE_VERSION = max(FEATURE_SETS)
FEATURES = FEATURE_SETS[LATEST_FEATURE_VERSION]
NUM_FEATURES = len(FEATURES)  # of the latest version
F = {name: i for i, name in enumerate(FEATURES)}  # indices are stable across versions (prefix)


def num_features(feature_version: int = LATEST_FEATURE_VERSION) -> int:
    return len(FEATURE_SETS[feature_version])


def feature_version_for_dim(in_dim: int) -> int:
    """Feature version whose observation width is `in_dim` (for old checkpoints)."""
    for v, feats in FEATURE_SETS.items():
        if len(feats) == in_dim:
            return v
    raise ValueError(f"no feature version with {in_dim} features")


@dataclass
class RewardConfig:
    solve: float = 1.0
    fail: float = -1.0
    checkpoint: float = 0.1
    step: float = 0.0          # per-move reward (in addition to progress)
    progress: float = 0.5      # total spread over a full path: each move gives progress/(n-1)


class ZipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, puzzle_sampler: Callable[[], Puzzle] | None = None,
                 reward: RewardConfig | None = None, early_termination: bool = True,
                 compute_distances: bool = True, dead_check: str = "basic",
                 feature_version: int = LATEST_FEATURE_VERSION):
        """dead_check: "basic" (connectivity + degree rule, built in) or "solver"
        (zipsolve.solver.is_dead_end: adds parity and articulation-point rules).
        feature_version: observation feature set (see module docstring)."""
        super().__init__()
        if feature_version not in FEATURE_SETS:
            raise ValueError(f"unknown feature_version {feature_version}; known {sorted(FEATURE_SETS)}")
        self.feature_version = int(feature_version)
        self.num_features = len(FEATURE_SETS[self.feature_version])
        self.dead_check = dead_check
        self.puzzle_sampler = puzzle_sampler
        self.reward_cfg = reward or RewardConfig()
        self.early_termination = early_termination
        self.compute_distances = compute_distances
        self.puzzle: Puzzle | None = None
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    # ------------------------------------------------------------------ setup
    def _load(self, puzzle: Puzzle):
        self.puzzle = puzzle
        g = puzzle.graph
        n = g.num_nodes
        self.n = n
        self.nbrs = g.neighbors
        maxd = max(1, max(len(ns) for ns in g.neighbors))
        self.max_deg = maxd
        pad = np.full((n, maxd), n, dtype=np.int64)
        for v, ns in enumerate(g.neighbors):
            pad[v, :len(ns)] = ns
        self.nbr_pad = pad
        self.nbr_obs = np.where(pad == n, -1, pad)  # (n, maxdeg) padded with -1
        self.deg = np.array([len(ns) for ns in g.neighbors], dtype=np.float32)
        self.edge_index = g.edge_index()
        self.cps = list(puzzle.checkpoints)
        self.cp_index = np.full(n, -1, dtype=np.int64)
        for k, c in enumerate(self.cps):
            self.cp_index[c] = k
        self.end = self.cps[-1]
        if self.feature_version >= 2:
            self.nbr_lists = [list(ns) for ns in g.neighbors]
            self.colour, self.bipartite = _two_colouring(self.nbr_lists)
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict({
            "x": spaces.Box(-1.0, 1.0, (n, self.num_features), np.float32),
            "edge_index": spaces.Box(0, max(n - 1, 0), self.edge_index.shape, np.int64),
            "action_mask": spaces.MultiBinary(n),
            "head": spaces.Discrete(n),
            "neighbors": spaces.Box(-1, max(n - 1, 0), pad.shape, np.int64),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        puzzle = (options or {}).get("puzzle")
        if puzzle is None:
            if self.puzzle_sampler is None:
                if self.puzzle is None:
                    raise ValueError("no puzzle given and no puzzle_sampler")
                puzzle = self.puzzle
            else:
                puzzle = self.puzzle_sampler()
        self._load(puzzle)
        n = self.n
        self.visited = np.zeros(n + 1, dtype=bool)  # last slot = padding sentinel (always "visited")
        self.visited[n] = True
        self.head = self.cps[0]
        self.visited[self.head] = True
        self.path = [self.head]
        self.next_cp = 1
        self.done = False
        self._mask = self._compute_mask()
        info = {"solved": False, "path": list(self.path)}
        if self.n == 1:
            self.done = True
            info["solved"] = True
        return self._obs(), info

    # ------------------------------------------------------------------ rules
    def _allowed(self, v: int) -> bool:
        if self.visited[v]:
            return False
        k = self.cp_index[v]
        if k > self.next_cp:
            return False
        if v == self.end and len(self.path) != self.n - 1:
            return False
        return True

    def _compute_mask(self) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        for w in self.nbrs[self.head]:
            if self._allowed(w):
                m[w] = True
        return m

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def dead_reason(self) -> str | None:
        """Cheap necessary conditions for the remaining puzzle to be solvable."""
        n = self.n
        remaining = n - len(self.path)
        if remaining == 0:
            return None
        if not self._mask.any():
            return "no legal move"
        if not self.early_termination:
            return None
        reason = dead_state(self.nbrs, self.visited[:n], self.head, self.end, remaining)
        if reason is None and self.dead_check == "solver":
            from ..solver import is_dead_end
            if is_dead_end(self.puzzle.graph, self.visited[:n], self.head, self.next_cp, self.cps):
                reason = "solver dead end"
        return reason

    def set_state(self, visited, head: int, next_cp: int, path=None):
        """Jump to an arbitrary search state (used by solver move-order hooks).

        `visited` is a bool array (n,) including head; `path` is optional (only
        its length matters for the rules, so a placeholder is built if absent).
        """
        n = self.n
        self.visited = np.ones(n + 1, dtype=bool)
        self.visited[:n] = np.asarray(visited, dtype=bool)
        self.head = int(head)
        self.next_cp = int(next_cp)
        if path is None:
            others = [v for v in np.flatnonzero(self.visited[:n]).tolist() if v != self.head]
            path = others + [self.head]
        self.path = list(path)
        self.done = False
        self._mask = self._compute_mask()
        return self._obs()

    # ------------------------------------------------------------------ step
    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on finished episode; call reset()")
        a = int(action)
        rc = self.reward_cfg
        if not (0 <= a < self.n) or not self._mask[a]:
            self.done = True
            return self._obs(), rc.fail, True, False, {"solved": False, "path": list(self.path),
                                                        "reason": "illegal action"}
        self.visited[a] = True
        self.path.append(a)
        self.head = a
        reward = rc.step + rc.progress / max(1, self.n - 1)
        if self.cp_index[a] == self.next_cp:
            self.next_cp += 1
            reward += rc.checkpoint
        info = {"solved": False}
        terminated = False
        if len(self.path) == self.n:
            solved = self.puzzle.is_valid_solution(self.path)
            info["solved"] = solved
            reward += rc.solve if solved else rc.fail
            terminated = True
            self._mask = np.zeros(self.n, dtype=bool)
        else:
            self._mask = self._compute_mask()
            reason = self.dead_reason()
            if reason is not None:
                reward += rc.fail
                terminated = True
                info["reason"] = reason
        self.done = terminated
        info["path"] = list(self.path)
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ features
    def _bfs(self, src: int) -> np.ndarray:
        """Distances from src through unvisited nodes (src itself may be visited)."""
        dist = np.full(self.n, -1, dtype=np.int64)
        dist[src] = 0
        q = deque([src])
        vis = self.visited
        nbrs = self.nbrs
        while q:
            u = q.popleft()
            d = dist[u] + 1
            for w in nbrs[u]:
                if dist[w] < 0 and not vis[w]:
                    dist[w] = d
                    q.append(w)
        return dist

    def _obs(self) -> dict:
        n = self.n
        x = np.zeros((n, self.num_features), dtype=np.float32)
        vis = self.visited[:n]
        unv = ~vis
        n_unv = max(1, int(unv.sum()))
        x[:, F["visited"]] = vis
        x[self.head, F["is_head"]] = 1.0
        ncp = len(self.cps)
        if self.next_cp < ncp:
            x[self.cps[self.next_cp], F["is_next_cp"]] = 1.0
            rem = ncp - self.next_cp
            for k in range(self.next_cp, ncp):
                c = self.cps[k]
                if k > self.next_cp:
                    x[c, F["is_future_cp"]] = 1.0
                x[c, F["cp_rank"]] = 1.0 - (k - self.next_cp) / rem
        x[self.end, F["is_final"]] = 1.0
        free = (~self.visited[self.nbr_pad]).sum(1).astype(np.float32)
        x[:, F["free_deg"]] = free / self.max_deg
        x[:, F["deg"]] = self.deg / self.max_deg
        adj_head = np.zeros(n, dtype=bool)
        adj_head[list(self.nbrs[self.head])] = True
        x[:, F["adj_head"]] = adj_head
        if self.compute_distances:
            dh = self._bfs(self.head)
            reach = dh >= 0
            x[:, F["reachable"]] = reach & (unv | (np.arange(n) == self.head))
            x[:, F["dist_head"]] = np.where(reach, dh, n_unv) / n_unv
            if self.next_cp < ncp:
                dc = self._bfs(self.cps[self.next_cp])
                x[:, F["dist_next_cp"]] = np.where(dc >= 0, dc, n_unv) / n_unv
        x[:, F["legal"]] = self._mask
        forced = unv & ((free + adj_head) <= 1)
        forced[self.end] = False
        x[:, F["forced"]] = forced
        x[:, F["frac_done"]] = len(self.path) / n
        x[:, F["frac_cp_done"]] = self.next_cp / ncp
        x[self.cps[0], F["is_start"]] = 1.0
        if self.feature_version >= 2:
            self._features_v2(x, n_unv)
        return {"x": x, "edge_index": self.edge_index, "action_mask": self._mask.copy(),
                "head": np.int64(self.head), "neighbors": self.nbr_obs}

    # ------------------------------------------------------------------ v2 features
    def _bfs_ordered(self, src: int, vis: list, leaf: list) -> list:
        """BFS from src through unvisited nodes; `leaf` nodes (later checkpoints)
        and the head get a distance but are not expanded."""
        n = self.n
        nb = self.nbr_lists
        head = self.head
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        for u in q:  # q grows while iterating
            d = dist[u] + 1
            for w in nb[u]:
                if dist[w] < 0:
                    if not vis[w]:
                        dist[w] = d
                        if not leaf[w]:
                            q.append(w)
                    elif w == head:
                        dist[w] = d
        return dist

    def _features_v2(self, x: np.ndarray, n_unv: int):
        n = self.n
        head = self.head
        vis = self.visited[:n].tolist()
        ncp = len(self.cps)
        nxt = self.next_cp
        inv = 1.0 / n_unv
        # --- order-respecting distances
        if self.compute_distances and nxt < ncp:
            leaf = [False] * n
            for c in self.cps[nxt + 1:]:
                leaf[c] = True
            for src, col in ((self.cps[nxt], 17), (head, 18)):
                d = np.asarray(self._bfs_ordered(src, vis, leaf), dtype=np.float32)
                x[:, col] = np.where(d >= 0, d * inv, 1.0)
        # --- articulation points / bridges of R (iterative Tarjan rooted at head)
        nb = self.nbr_lists
        end = self.end
        disc = [-1] * n
        low = [0] * n
        size = [1] * n
        parent = [-1] * n
        ptr = [0] * n
        nbridge = [0] * n
        sep_sum = [0] * n     # sum of subtree sizes cut off from the head by v
        sep_min = [0] * n     # smallest separated child subtree
        sep_end = [False] * n
        root_children = 0
        disc[head] = 0
        low[head] = 0
        t = 1
        stack = [head]
        while stack:
            u = stack[-1]
            lst = nb[u]
            i = ptr[u]
            if i < len(lst):
                ptr[u] = i + 1
                w = lst[i]
                dw = disc[w]
                if dw < 0:
                    if vis[w]:  # visited and not head (head has disc 0)
                        continue
                    parent[w] = u
                    disc[w] = low[w] = t
                    t += 1
                    stack.append(w)
                elif w != parent[u] and dw < low[u]:
                    low[u] = dw
            else:
                stack.pop()
                p = parent[u]
                if p < 0:
                    continue
                su = size[u]
                size[p] += su
                lu = low[u]
                if lu < low[p]:
                    low[p] = lu
                if p == head:
                    root_children += 1
                    if sep_min[p] == 0 or su < sep_min[p]:
                        sep_min[p] = su
                elif lu >= disc[p]:
                    sep_sum[p] += su
                    if sep_min[p] == 0 or su < sep_min[p]:
                        sep_min[p] = su
                    de = disc[end]
                    if de >= 0 and disc[u] <= de < disc[u] + su:
                        sep_end[p] = True
                if lu > disc[p]:
                    nbridge[p] += 1
                    nbridge[u] += 1
        total = t  # nodes of R reachable from head (incl. head)
        nbridge_a = np.asarray(nbridge, dtype=np.float32)
        x[:, 20] = nbridge_a * (1.0 / self.max_deg)
        sep = np.asarray(sep_sum, dtype=np.float32)
        art = sep > 0  # non-root articulation points
        if art.any():
            smin = np.asarray(sep_min, dtype=np.float32)
            rest = (total - 1) - sep
            cmin = np.where(rest > 0, np.minimum(smin, rest), smin)
            x[:, 19] = art
            x[:, 21] = sep * inv
            x[:, 22] = np.where(art, cmin * inv, 0.0)
            x[:, 23] = np.asarray(sep_end, dtype=np.float32)
        if root_children >= 2:
            x[head, 19] = 1.0
            x[head, 22] = sep_min[head] * inv
        # --- bipartite parity
        if self.bipartite:
            col = self.colour
            ch = col[head]
            same = col == ch
            x[:, 24] = 1.0
            x[:, 25] = col
            x[:, 26] = same
            m = n - len(self.path)
            n_same = int(np.count_nonzero(same & ~self.visited[:n]))
            x[:, 27] = (n_same - m // 2) / max(1, m)
            ok = n_same == m // 2 and (bool(col[end] == ch) == (m % 2 == 0))
            x[:, 28] = float(ok)


def _two_colouring(nbrs) -> tuple[np.ndarray, bool]:
    """2-colouring by BFS per component; (colours float32 0/1, is_bipartite)."""
    n = len(nbrs)
    col = [-1] * n
    ok = True
    for s in range(n):
        if col[s] >= 0:
            continue
        col[s] = 0
        q = [s]
        for u in q:
            cu = col[u]
            for w in nbrs[u]:
                if col[w] < 0:
                    col[w] = 1 - cu
                    q.append(w)
                elif col[w] == cu:
                    ok = False
    if not ok:
        return np.zeros(n, dtype=np.float32), False
    return np.asarray(col, dtype=np.float32), True


def dead_state(nbrs, visited: np.ndarray, head: int, end: int, remaining: int) -> str | None:
    """Necessary-condition dead-state test shared by env and search.

    `visited` is a bool array of length n (head counts as visited), `remaining`
    the number of unvisited nodes. Returns a reason string if the state cannot
    be completed, else None.
      * every unvisited node must be reachable from head through unvisited nodes;
      * every unvisited node other than `end` needs >=2 neighbours among
        (unvisited nodes + head); `end` needs >=1.
    """
    if remaining == 0:
        return None
    head_nb = set(nbrs[head])
    seen = {head}
    stack = [head]
    count = 0
    while stack:
        u = stack.pop()
        for w in nbrs[u]:
            if not visited[w] and w not in seen:
                seen.add(w)
                count += 1
                stack.append(w)
                c = 0
                for z in nbrs[w]:
                    if not visited[z] or z == head:
                        c += 1
                if c < (1 if w == end else 2):
                    return "dead end"
    if count != remaining:
        return "disconnected"
    return None


__all__ = ["ZipEnv", "RewardConfig", "FEATURES", "NUM_FEATURES", "F", "dead_state", "FEATURES_V1",
           "FEATURES_V2", "FEATURE_SETS", "LATEST_FEATURE_VERSION", "num_features",
           "feature_version_for_dim"]
