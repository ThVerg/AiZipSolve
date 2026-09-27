"""Solver / agent adapters used by the web app.

Everything here builds on the public APIs of ``zipsolve.solver`` and
``zipsolve.rl`` without modifying them:

* ``validate_prefix``      - is a partial path legal (Zip rules)?
* ``complete_prefix``      - exact solver continuing a given prefix (it solves the
                             residual puzzle on the unvisited nodes + head).
* ``hint``                 - next move for the user, or how far to backtrack.
* ``rl_solve``             - greedy rollout or policy-guided DFS of a GNN policy,
                             optionally continuing a prefix, recording per-step
                             top probabilities (the agent's "confidence").
* ``ModelRegistry``        - lists/loads checkpoints (cached by mtime).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np

from ..graph import from_edges
from ..puzzle import Puzzle
from .. import solver as _solver


# ---------------------------------------------------------------------------
# prefixes
# ---------------------------------------------------------------------------
def validate_prefix(puzzle: Puzzle, path: Sequence[int]) -> str | None:
    """None if `path` is a legal partial path (possibly complete), else a reason."""
    g = puzzle.graph
    n = g.num_nodes
    cps = puzzle.checkpoints
    cpi = {c: i for i, c in enumerate(cps)}
    if not path:
        return "empty path"
    if path[0] != cps[0]:
        return "path must start at checkpoint 1"
    seen = set()
    nxt = 0
    for i, v in enumerate(path):
        if not (0 <= v < n):
            return f"invalid node id {v}"
        if v in seen:
            return f"node {v} visited twice"
        seen.add(v)
        if i > 0 and not g.has_edge(path[i - 1], v):
            return f"nodes {path[i - 1]} and {v} are not adjacent"
        k = cpi.get(v)
        if k is not None:
            if k != nxt:
                return f"checkpoint {k + 1} reached before checkpoint {nxt + 1}"
            nxt += 1
        if v == cps[-1] and i != n - 1:
            return "the last checkpoint must be the final cell"
    return None


def _next_cp(puzzle: Puzzle, path: Sequence[int]) -> int:
    cps = set(puzzle.checkpoints)
    return sum(1 for v in path if v in cps)


@dataclass
class Completion:
    status: str               # "solved" | "unsat" | "timeout" | "invalid"
    path: list[int] | None    # full path (prefix + continuation) if solved
    nodes_expanded: int
    seconds: float
    reason: str | None = None


def complete_prefix(puzzle: Puzzle, prefix: Sequence[int], time_limit: float | None = 5.0) -> Completion:
    """Exact solver continuing `prefix` (adapter: residual puzzle on unvisited nodes + head)."""
    t0 = time.perf_counter()
    prefix = [int(v) for v in prefix]
    reason = validate_prefix(puzzle, prefix)
    if reason is not None:
        return Completion("invalid", None, 0, 0.0, reason)
    g = puzzle.graph
    n = g.num_nodes
    if len(prefix) == n:
        ok = puzzle.is_valid_solution(prefix)
        return Completion("solved" if ok else "unsat", prefix if ok else None, 0, 0.0)
    head = prefix[-1]
    nxt = _next_cp(puzzle, prefix)
    cps_left = puzzle.checkpoints[nxt:]
    if not cps_left:
        return Completion("unsat", None, 0, 0.0, "final checkpoint already reached")
    if len(prefix) == 1:
        res = _solver.solve(puzzle, time_limit=time_limit)
        return Completion(res.status, res.path, res.nodes_expanded, res.seconds)
    sfp = getattr(_solver, "solve_from_prefix", None)
    if sfp is not None:  # native: search state initialised from the prefix (with restarts)
        res = sfp(puzzle, prefix, time_limit)
        if res.status == "solved" and (res.path is None or list(res.path[:len(prefix)]) != prefix
                                       or not puzzle.is_valid_solution(res.path)):
            return Completion("unsat", None, res.nodes_expanded, time.perf_counter() - t0,
                              "internal: continuation invalid")
        return Completion(res.status, list(res.path) if res.path else None, res.nodes_expanded,
                          time.perf_counter() - t0)
    visited = set(prefix)
    keep = [v for v in range(n) if v not in visited or v == head]
    new_id = {v: i for i, v in enumerate(keep)}
    edges = [(new_id[u], new_id[v]) for u, v in g.edges() if u in new_id and v in new_id]
    sub = from_edges(g.coords[keep], edges, g.kind, {})
    sub_cps = [new_id[head]] + [new_id[c] for c in cps_left]
    sub_puzzle = Puzzle(sub, sub_cps)
    res = _solver.solve(sub_puzzle, time_limit=time_limit)
    full = None
    if res.status == "solved" and res.path is not None:
        full = prefix[:-1] + [keep[i] for i in res.path]
        if not puzzle.is_valid_solution(full):  # paranoia
            return Completion("unsat", None, res.nodes_expanded, time.perf_counter() - t0,
                              "internal: continuation invalid")
    return Completion(res.status, full, res.nodes_expanded, time.perf_counter() - t0)


def hint(puzzle: Puzzle, path: Sequence[int], solution: Sequence[int] | None = None,
         time_limit: float = 5.0) -> dict:
    """Suggest the user's next move.

    Returns {"status": "next", "next": v, "keep": len(path)} when the prefix can be
    completed, {"status": "backtrack", "keep": k, "next": v} when only its first k
    nodes can (then v is the move after truncating), {"status": "done"} if solved,
    or {"status": "timeout"/"unsat"/"invalid", "message"}.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    path = [int(v) for v in path] or [puzzle.checkpoints[0]]
    reason = validate_prefix(puzzle, path)
    if reason is not None:
        return {"status": "invalid", "message": reason}
    n = puzzle.num_nodes
    if len(path) == n:
        return {"status": "done", "message": "Already solved!"}
    nodes = 0

    # fast path: the stored solution extends this prefix
    known_ok = 1
    if solution is not None:
        solution = list(solution)
        lcp = 0
        while lcp < len(path) and lcp < len(solution) and path[lcp] == solution[lcp]:
            lcp += 1
        known_ok = max(1, lcp)
        if lcp == len(path):
            return {"status": "next", "next": solution[len(path)], "keep": len(path),
                    "source": "solution", "nodes_expanded": 0,
                    "seconds": time.perf_counter() - t0}

    def attempt(k):
        nonlocal nodes
        remaining = max(0.05, deadline - time.perf_counter())
        if _solver.is_dead_end(puzzle.graph, set(path[:k]), path[k - 1],
                               _next_cp(puzzle, path[:k]), puzzle.checkpoints):
            return Completion("unsat", None, 0, 0.0)
        c = complete_prefix(puzzle, path[:k], time_limit=remaining)
        nodes += c.nodes_expanded
        return c

    c = attempt(len(path))
    if c.status == "solved":
        return {"status": "next", "next": c.path[len(path)], "keep": len(path), "source": "solver",
                "nodes_expanded": nodes, "seconds": time.perf_counter() - t0}
    if c.status == "timeout":
        return {"status": "timeout", "message": "The solver ran out of time on this position.",
                "nodes_expanded": nodes, "seconds": time.perf_counter() - t0}
    # binary search the longest completable prefix (completability is monotone in k)
    lo, hi = known_ok, len(path) - 1      # prefix[:lo] known completable, prefix[:hi+1] not
    best = None
    if solution is not None and lo >= 1:
        best = list(solution)
    while lo < hi:
        if time.perf_counter() > deadline:
            break
        mid = (lo + hi + 1) // 2
        c = attempt(mid)
        if c.status == "solved":
            lo, best = mid, c.path
        elif c.status == "unsat":
            hi = mid - 1
        else:
            break  # timeout: settle for lo
    if best is None or best[:lo] != path[:lo]:
        c = complete_prefix(puzzle, path[:lo], time_limit=max(0.5, deadline - time.perf_counter()))
        nodes += c.nodes_expanded
        if c.status != "solved":
            return {"status": "unsat" if c.status == "unsat" else "timeout",
                    "message": "Could not find a completion (puzzle may have no solution).",
                    "nodes_expanded": nodes, "seconds": time.perf_counter() - t0}
        best = c.path
    return {"status": "backtrack", "keep": lo, "next": best[lo], "source": "solver",
            "message": f"Your path can't be completed - backtrack to step {lo}.",
            "nodes_expanded": nodes, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# RL agent
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Lists checkpoints under a directory (recursively: e.g. ``remote/host/x.pt``
    mirrored from a training server) and caches loaded models (keyed by mtime).

    Model names are POSIX paths relative to the directory; names containing
    ``..``, backslashes or absolute paths are rejected (no traversal)."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._cache: dict[str, tuple[float, object, dict]] = {}
        self._lock = Lock()

    @staticmethod
    def check_name(name: str) -> str:
        """Validate a model name (relative POSIX path ending in .pt); raises ValueError."""
        from pathlib import PurePosixPath
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("invalid model name")
        p = PurePosixPath(name)
        if p.is_absolute() or any(part in ("..", "", ".") for part in name.split("/")) \
                or ":" in name or p.suffix != ".pt":
            raise ValueError(f"invalid model name {name!r}")
        return p.as_posix()

    def _files(self) -> list[tuple[str, Path]]:
        root = self.directory
        out = []
        for p in root.rglob("*.pt"):
            try:
                if not p.is_file():
                    continue
                rel = p.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            if any(part.startswith(".") for part in rel.split("/")):
                continue
            out.append((rel, p))
        return out

    def list(self) -> list[dict]:
        if not self.directory.is_dir():
            return []
        out = []
        entries = []
        for rel, p in self._files():
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, rel, st))
        entries.sort(key=lambda e: e[0], reverse=True)
        for mtime, rel, st in entries:
            group = rel.rsplit("/", 1)[0] if "/" in rel else ""
            info = {"name": rel, "file": rel.rsplit("/", 1)[-1], "group": group,
                    "size": st.st_size, "mtime": mtime}
            cached = self._cache.get(rel)
            if cached and cached[0] == mtime:
                meta = cached[2]
                info.update({k: meta.get(k) for k in ("stage", "iteration", "global_step")})
            out.append(info)
        # preferred default: a local (top-level) *final* / *latest* / *best*, else the newest local file
        local = [m for m in out if not m["group"]] or out
        pref = [m for m in local if any(t in m["file"] for t in ("final", "latest", "best"))]
        default = (pref or local or [None])[0]
        for m in out:
            m["default"] = m is default
        return out

    def resolve(self, name: str | None) -> Path:
        if name:
            try:
                name = self.check_name(name)
            except ValueError as e:
                raise FileNotFoundError(str(e))
        models = self.list()
        if not models:
            raise FileNotFoundError("no checkpoints found in " + str(self.directory))
        if not name:
            name = next(m["name"] for m in models if m["default"])
        names = {m["name"] for m in models}
        if name not in names:
            raise FileNotFoundError(f"unknown model {name!r}")
        path = (self.directory / name).resolve()
        root = self.directory.resolve()
        if root != path and root not in path.parents:
            raise FileNotFoundError(f"invalid model name {name!r}")
        return self.directory / name

    def load(self, name: str | None):
        path = self.resolve(name)
        rel = path.relative_to(self.directory).as_posix()
        from ..rl.gnn import load_model
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(rel)
            if cached and cached[0] == mtime:
                return cached[1], cached[2], rel
            model, ck = load_model(path)
            meta = {k: v for k, v in ck.items() if k != "state_dict"}
            self._cache[rel] = (mtime, model, meta)
            return model, meta, rel


def _probs(model, obs) -> np.ndarray:
    from ..rl.evaluate import policy_probs
    return policy_probs(model, obs)


def _record(p: np.ndarray, legal: np.ndarray, chosen: int, k: int = 3) -> dict:
    order = legal[np.argsort(-p[legal])][:k]
    return {"node": int(chosen), "p": float(p[chosen]),
            "top": [[int(v), round(float(p[v]), 4)] for v in order],
            "n_legal": int(len(legal))}


def _first_dead_step(puzzle: Puzzle, path: list[int]) -> int | None:
    """Smallest prefix length k whose state is provably dead (solver's sound checks)."""
    cps = puzzle.checkpoints
    for k in range(2, len(path) + 1):
        if _solver.is_dead_end(puzzle.graph, set(path[:k]), path[k - 1],
                               _next_cp(puzzle, path[:k]), cps):
            return k
    return None


RL_MODES = ("greedy", "search", "hybrid")


def _annotate_path(model, env, puzzle: Puzzle, path: list[int], start_len: int,
                   chunk: int = 64) -> list[dict]:
    """Per-step policy records (p(chosen), top-3) along `path` after the first
    `start_len` nodes, evaluated in batched forward passes."""
    import torch
    from ..rl.gnn import collate
    cps = puzzle.checkpoints
    cpi = {c: i for i, c in enumerate(cps)}
    n = puzzle.num_nodes
    env.reset(options={"puzzle": puzzle})
    states = []
    vis = np.zeros(n, dtype=bool)
    nxt = 0
    for i, v in enumerate(path[:-1]):
        vis[v] = True
        if cpi.get(v) == nxt:
            nxt += 1
        if i + 1 >= start_len:
            obs = env.set_state(vis, v, nxt, path[:i + 1])
            states.append((obs, path[i + 1]))
    out = []
    with torch.no_grad():
        for k in range(0, len(states), chunk):
            part = states[k:k + chunk]
            obs_list = [o for o, _ in part]
            gb = collate(obs_list)
            logits, _ = model(gb)
            for j, (o, chosen) in enumerate(part):
                lg = logits[gb.ptr[j]:gb.ptr[j + 1]]
                p = torch.softmax(lg, 0).numpy()
                legal = np.flatnonzero(o["action_mask"])
                if chosen not in set(legal.tolist()):   # env/solver disagree: show p = 0
                    out.append({"node": int(chosen), "p": 0.0, "top": [], "n_legal": int(len(legal))})
                else:
                    out.append(_record(p, legal, int(chosen)))
    return out


def rl_solve(model, puzzle: Puzzle, mode: str = "greedy", budget: int = 5000,
             start_path: Sequence[int] | None = None, time_limit: float = 20.0,
             meta: dict | None = None, compare: bool = True, trace: bool = False) -> dict:
    """Run the policy. Adapter over zipsolve.rl.env.ZipEnv (greedy_rollout /
    policy_search re-implemented here to record per-step probabilities and to
    support starting from a user prefix).

    Modes: "greedy" (argmax rollout, no backtracking), "search" (policy-ordered
    DFS with the env's basic pruning, node budget) and "hybrid" (the exact
    solver with the GNN ordering its candidates: complete, see
    zipsolve.rl.evaluate.hybrid_solve). For "hybrid" with ``compare`` the plain
    exact solver is also run from the same start, under ``out["exact"]``.

    ``trace``: for "search" / "hybrid", also return the search's push/pop
    events (see :func:`compress_trace`) under ``out["trace"]``.
    """
    import torch
    from ..rl.evaluate import make_env_for

    t0 = time.perf_counter()
    deadline = t0 + time_limit
    start = [int(v) for v in (start_path or [puzzle.checkpoints[0]])]
    reason = validate_prefix(puzzle, start)
    if reason is not None:
        raise ValueError("start_path: " + reason)
    if mode not in RL_MODES:
        raise ValueError("mode must be one of " + ", ".join(RL_MODES))
    if mode == "hybrid":
        return _hybrid(model, puzzle, start, time_limit, meta, compare, trace)
    env = make_env_for(model, meta, early_termination=False)
    obs, info = env.reset(options={"puzzle": puzzle})
    for v in start[1:]:
        obs, _, done, _, info = env.step(v)
        if done and not info.get("solved"):
            raise ValueError("start_path leaves no legal move")
    n = puzzle.num_nodes
    steps: list[dict] = []
    events: list | None = [] if (trace and mode == "search") else None
    status = "stuck"
    expanded = 0
    backtracks = 0

    with torch.no_grad():
        if mode == "greedy":
            while not env.done and len(env.path) < n:
                legal = np.flatnonzero(obs["action_mask"])
                if len(legal) == 0:
                    break
                p = _probs(model, obs)
                a = int(legal[np.argmax(p[legal])])
                steps.append(_record(p, legal, a))
                expanded += 1
                obs, _, done, _, info = env.step(a)
                if time.perf_counter() > deadline:
                    status = "timeout"
                    break
            path = list(env.path)
            if len(path) == n and puzzle.is_valid_solution(path):
                status = "solved"
        elif mode == "search":
            env.early_termination = True
            from ..rl.env import dead_state
            path_steps: list[dict] = []

            class _Stop(Exception):
                pass

            def snapshot():
                return (env.head, env.next_cp, env._mask.copy(), env.done, len(env.path))

            def restore(s):
                head, ncp, mask, done, plen = s
                for v in env.path[plen:]:
                    env.visited[v] = False
                if events is not None and len(events) < TRACE_CAP:
                    events.extend([-1] * (len(env.path) - plen))
                del env.path[plen:]
                env.head, env.next_cp, env._mask, env.done = head, ncp, mask, done

            def dfs(o) -> bool:
                nonlocal expanded, backtracks
                p = _probs(model, o)
                legal = np.flatnonzero(o["action_mask"])
                for a in legal[np.argsort(-p[legal])]:
                    expanded += 1
                    if expanded > budget:
                        raise _Stop("budget")
                    if time.perf_counter() > deadline:
                        raise _Stop("timeout")
                    s = snapshot()
                    path_steps.append(_record(p, legal, int(a)))
                    o2, _, done, _, inf = env.step(int(a))
                    if events is not None and len(events) < TRACE_CAP:
                        events.append(int(a))
                    if len(env.path) > len(deepest[0]):
                        deepest[0], deepest[1] = list(env.path), list(path_steps)
                    if done and inf.get("solved"):
                        return True
                    if not done and dfs(o2):
                        return True
                    restore(s)
                    path_steps.pop()
                    backtracks += 1
                return False

            deepest = [list(env.path), []]
            if len(start) == n:
                status = "solved" if puzzle.is_valid_solution(start) else "unsat"
            elif dead_state(env.nbrs, env.visited[:n], env.head, env.end, n - len(env.path)):
                status = "unsat"
            else:
                try:
                    status = "solved" if dfs(obs) else "unsat"
                except _Stop as e:
                    status = str(e)
            if status == "solved":
                path, steps = list(env.path), path_steps
            else:  # show the deepest partial path the search reached
                path, steps = deepest

    out = {
        "mode": mode, "status": status, "solved": status == "solved", "path": path,
        "start_len": len(start), "steps": steps,
        "nodes_expanded": expanded, "backtracks": backtracks,
        "seconds": time.perf_counter() - t0,
    }
    if status != "solved":
        out["stuck_at"] = len(path)
        out["dead_from"] = _first_dead_step(puzzle, path)
    ps = [s["p"] for s in steps]
    out["mean_confidence"] = float(np.mean(ps)) if ps else None
    out["min_confidence"] = float(np.min(ps)) if ps else None
    if events is not None:
        out["trace"] = compress_trace(["R"] + list(start) + events)
        out["trace_truncated"] = len(events) >= TRACE_CAP
    return out


def _hybrid(model, puzzle: Puzzle, start: list[int], time_limit: float, meta: dict | None,
            compare: bool, trace: bool = False) -> dict:
    from ..rl.evaluate import hybrid_solve, make_env_for
    events: list | None = [] if trace else None
    kw = {"trace": events, "trace_cap": TRACE_CAP} if trace else {}
    try:
        h = hybrid_solve(model, puzzle, time_limit, start_path=start, meta=meta, **kw)
    except TypeError:  # older hybrid_solve without trace support
        events = None
        h = hybrid_solve(model, puzzle, time_limit, start_path=start, meta=meta)
    if h["status"] == "invalid":
        raise ValueError("start_path: " + h.get("reason", "invalid"))
    solved = h["status"] == "solved"
    path = h["path"] if solved else list(start)
    t1 = time.perf_counter()
    steps = _annotate_path(model, make_env_for(model, meta), puzzle, path, len(start)) if solved else []
    out = {
        "mode": "hybrid", "status": h["status"], "solved": solved, "path": path,
        "start_len": len(start), "steps": steps,
        "nodes_expanded": h["nodes_expanded"], "backtracks": None,
        "seconds": h["seconds"], "annotate_seconds": time.perf_counter() - t1,
        "hybrid": {k: h[k] for k in ("attempts", "hook_calls", "inference_calls", "cache_hits",
                                     "skipped_single", "deadline_skips", "inference_seconds",
                                     "obs_seconds", "search_seconds")},
    }
    if not solved:
        out["stuck_at"] = len(path)
        out["dead_from"] = _first_dead_step(puzzle, path) if h["status"] == "unsat" else None
    ps = [s["p"] for s in steps]
    out["mean_confidence"] = float(np.mean(ps)) if ps else None
    out["min_confidence"] = float(np.min(ps)) if ps else None
    if events is not None:
        _trim_solution_pop(events, h["status"], TRACE_CAP)
        out["trace"] = compress_trace(events)
        out["trace_truncated"] = len(events) >= TRACE_CAP
    if compare:  # the plain exact solver from the same start, same time limit
        c = complete_prefix(puzzle, start, time_limit=time_limit)
        out["exact"] = {"status": c.status, "solved": c.status == "solved", "path": c.path,
                        "nodes_expanded": c.nodes_expanded, "seconds": c.seconds}
    return out


# ---------------------------------------------------------------------------
# search traces ("watch it think")
# ---------------------------------------------------------------------------
TRACE_CAP = 60_000   # max raw events recorded per run (the search itself is not limited)


def compress_trace(events: Sequence) -> list:
    """Compact push/pop event list for the client.

    Raw events: node id (>= 0) = push, -1 = pop one node, "R" = a new search
    attempt starts (the client clears its stack). Output: same, with runs of
    pops merged into one negative count (-k = pop k nodes).
    """
    out: list = []
    for e in events:
        if isinstance(e, str):
            out.append(e)
        elif e < 0:
            if out and not isinstance(out[-1], str) and out[-1] < 0:
                out[-1] += e
            else:
                out.append(int(e))
        else:
            out.append(int(e))
    return out


def _trim_solution_pop(events: list, status: str, cap: int) -> None:
    """The solver records a solution and then pops its last node before returning:
    drop that final pop so replaying the trace ends on the solution."""
    if status == "solved" and events and len(events) < cap and events[-1] == -1:
        events.pop()


def traced_search_class(Search, events: list, cap: int = TRACE_CAP):
    """Subclass of the solver's internal ``_Search`` that records push/pop
    events (read-only instrumentation; the search itself is unchanged)."""

    class Traced(Search):  # type: ignore[misc, valid-type]
        def push(self, h):
            super().push(h)
            if len(events) < cap:
                events.append(int(h))

        def pop(self):
            super().pop()
            if len(events) < cap:
                events.append(-1)

    return Traced


def traced_exact(puzzle: Puzzle, prefix: Sequence[int] | None = None, time_limit: float = 10.0,
                 cap: int = TRACE_CAP) -> dict:
    """Exact solver (same search, prunings and restarts as ``solver.solve``)
    with its push/pop events recorded. Falls back to the untraced solver if the
    solver's internals are not available (then ``trace`` is None)."""
    t0 = time.perf_counter()
    prefix = [int(v) for v in (prefix or [puzzle.checkpoints[0]])]
    Search = getattr(_solver, "_Search", None)
    events: list = []
    base = getattr(_solver, "_RESTART_BASE", 200)
    growth = getattr(_solver, "_RESTART_GROWTH", 1.5)
    if Search is not None:
        import random
        try:
            Traced = traced_search_class(Search, events, cap)
            budget = max(base, 2 * puzzle.num_nodes)
            total, attempt = 0, 0
            while True:
                s = Traced(puzzle.graph, puzzle.checkpoints, deep=True)
                if attempt > 0:
                    s.rng = random.Random(attempt)
                left = time_limit - (time.perf_counter() - t0)
                if left <= 0:
                    status, path = "timeout", None
                    break
                if len(events) < cap:
                    events.append("R")
                sols, st, expanded = s.run(1, left, None, time.perf_counter(), budget,
                                           prefix if len(prefix) > 1 else None)
                total += expanded
                attempt += 1
                if sols:
                    status, path = "solved", list(sols[0])
                    break
                if st == "timeout":
                    status, path = "timeout", None
                    break
                if st != "budget":
                    status, path = "unsat", None
                    break
                budget = int(budget * growth)
            if path is not None and not puzzle.is_valid_solution(path):
                status, path = "unsat", None
            _trim_solution_pop(events, status, cap)
            return {"status": status, "path": path, "nodes_expanded": int(total),
                    "seconds": time.perf_counter() - t0, "attempts": attempt,
                    "trace": compress_trace(events), "trace_truncated": len(events) >= cap}
        except (TypeError, AttributeError):   # solver internals changed: untraced fallback
            pass
    c = complete_prefix(puzzle, prefix, time_limit)
    return {"status": c.status, "path": c.path, "nodes_expanded": c.nodes_expanded,
            "seconds": c.seconds, "trace": None, "trace_truncated": False}


# ---------------------------------------------------------------------------
# policy view (heatmap / value estimate for any position)
# ---------------------------------------------------------------------------
def value_is_trained(meta: dict | None) -> bool:
    """PPO-trained checkpoints have a meaningful value head; pure imitation ones do not."""
    meta = meta or {}
    return bool(meta.get("ppo_config") or meta.get("iteration") or meta.get("global_step"))


def _winnable(meta: dict | None, value: float, puzzle: Puzzle, path: Sequence[int]) -> float | None:
    """Rough P(solve) from the value estimate. Assumes the episode runs to the end:
    V ~ (discounted shaping still to come: progress + checkpoints) +
    gamma^remaining * (p * solve + (1 - p) * fail); solved for p, clipped to [0, 1].
    Only a heuristic reading of an (often poorly calibrated) value head."""
    if not value_is_trained(meta):
        return None
    cfg = (meta or {}).get("ppo_config") or {}
    rw = cfg.get("reward") or {}
    gamma = float(cfg.get("gamma", 0.99) or 0.99)
    solve, fail = float(rw.get("solve", 1.0)), float(rw.get("fail", -1.0))
    progress, cpr = float(rw.get("progress", 0.0)), float(rw.get("checkpoint", 0.0))
    if solve == fail:
        return None
    n = puzzle.num_nodes
    rem = n - len(path)
    disc = gamma ** np.arange(rem)
    shaping = progress / max(1, n - 1) * float(disc.sum())
    cps_left = len(puzzle.checkpoints) - _next_cp(puzzle, path)
    if cps_left > 0:   # assume the remaining checkpoints are spread evenly over the remaining moves
        at = np.linspace(rem / cps_left, rem, cps_left) - 1
        shaping += cpr * float((gamma ** np.clip(at, 0, None)).sum())
    term = gamma ** max(0, rem - 1)
    return float(np.clip(((value - shaping) / max(term, 1e-6) - fail) / (solve - fail), 0.0, 1.0))


def policy_view(model, meta: dict | None, puzzle: Puzzle, path: Sequence[int], full: bool = True,
                check: bool = True, check_time: float = 0.5, rollout: bool = True,
                rollout_time: float = 0.8) -> dict:
    """The GNN's view of a position: move probabilities over the legal moves,
    value estimate (+ a rough "winnable" probability for PPO-trained models) and,
    with ``full``, a board-wide score heat: the policy head evaluated on *every*
    unvisited node (mask lifted), rescaled to [0, 1] (10th percentile .. max,
    gamma 1.6), i.e. where the network "looks". ``check`` adds the exact solver's verdict on whether
    the position can still be completed (short time limit); ``rollout`` a greedy
    rollout of the policy from here (does the agent finish from this position?)."""
    import dataclasses
    import torch
    from ..rl.evaluate import make_env_for
    from ..rl.gnn import collate

    t0 = time.perf_counter()
    path = [int(v) for v in (path or [puzzle.checkpoints[0]])]
    reason = validate_prefix(puzzle, path)
    if reason is not None:
        raise ValueError("path: " + reason)
    n = puzzle.num_nodes
    out: dict = {"head": path[-1], "length": len(path), "num_nodes": n,
                 "value_trained": value_is_trained(meta)}
    if len(path) == n:
        out.update({"done": True, "legal": [], "value": None, "winnable": 1.0, "heat": {},
                    "entropy": None, "completable": "solved", "seconds": time.perf_counter() - t0})
        return out
    env = make_env_for(model, meta, early_termination=False)
    env.reset(options={"puzzle": puzzle})
    vis = np.zeros(n, dtype=bool)
    vis[path] = True
    obs = env.set_state(vis, path[-1], _next_cp(puzzle, path), path)
    legal = np.flatnonzero(obs["action_mask"])
    gb = collate([obs])
    with torch.no_grad():
        logits, values = model(gb)
        heat: dict[int, float] = {}
        if full:
            free = ~vis
            gb_all = dataclasses.replace(gb, mask=torch.from_numpy(free.copy()))
            lg_all, _ = model(gb_all)
            la = lg_all.numpy().astype(np.float64)
            idx = np.flatnonzero(free)
            if len(idx):
                # relative score of every free node: logits rescaled to [0, 1] between a low
                # percentile and the max (softmax over the whole board is too peaked to show
                # anything but the top move), then gamma > 1 to emphasise the high end
                z = la[idx]
                lo, hi = np.percentile(z, 10), z.max()
                s = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** 1.6
                heat = {int(v): round(float(x), 3) for v, x in zip(idx, s)}
    value = float(values[0])
    probs = []
    ent = None
    if len(legal):
        lg = logits.numpy().astype(np.float64)[legal]
        z = np.exp(lg - lg.max())
        pr = z / z.sum()
        order = np.argsort(-pr)
        probs = [[int(legal[i]), round(float(pr[i]), 4)] for i in order]
        ent = float(-(pr * np.log(np.clip(pr, 1e-12, 1))).sum())
    out.update({"done": False, "legal": probs, "value": value,
                "winnable": _winnable(meta, value, puzzle, path), "heat": heat, "entropy": ent})
    if rollout and len(legal):
        dl = time.perf_counter() + rollout_time
        obs2, info = obs, {}
        moves, logp, status = 0, 0.0, "stuck"
        with torch.no_grad():
            while True:
                lg2 = np.flatnonzero(obs2["action_mask"])
                if not len(lg2):
                    break
                pp = _probs(model, obs2)
                a = int(lg2[np.argmax(pp[lg2])])
                logp += float(np.log(max(pp[a], 1e-12)))
                obs2, _, done, _, info = env.step(a)
                moves += 1
                if done or len(env.path) == n:
                    break
                if time.perf_counter() > dl:
                    status = "timeout"
                    break
        if len(env.path) == n and puzzle.is_valid_solution(list(env.path)):
            status = "solved"
        out["greedy"] = {"status": status, "moves": moves, "remaining": n - len(path),
                         "path_prob": float(np.exp(logp)), "path": list(env.path)}
    if check:
        if not len(legal):
            out["completable"] = "unsat"
        elif _solver.is_dead_end(puzzle.graph, set(path), path[-1], _next_cp(puzzle, path),
                                 puzzle.checkpoints):
            out["completable"] = "unsat"
        else:
            c = complete_prefix(puzzle, path, time_limit=check_time)
            out["completable"] = c.status
    out["seconds"] = time.perf_counter() - t0
    return out
