"""Evaluate a trained Zip policy: greedy solve rate and policy-guided search.

    python -m zipsolve.rl.evaluate --ckpt checkpoints/zip_ppo_final.pt --kind grid3d --size 3 --episodes 100
    python -m zipsolve.rl.evaluate --ckpt ... --spec "islands:4+mask:6" --search --budget 2000 --solver

Greedy: always take the most probable legal move (no backtracking).
Search: depth-first search that orders moves by policy probability, with the
same dead-state pruning as the environment, up to --budget node expansions.
If zipsolve.solver is available, --solver runs the exact solver on the same
puzzles and reports its nodes expanded / time for comparison, both with its
default move order ("solver_*") and as the hybrid ("hybrid_*", see
hybrid_solve: the solver's pruning + GNN move ordering, complete search).

Public API: greedy_rollout, greedy_batch, policy_search, policy_move_order,
PolicyScorer, hybrid_solve, make_env_for.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import torch

from .env import ZipEnv
from .gnn import ZipGNN, collate, dense_logits, load_model
from .sampling import make_sampler, using_real_generator


@torch.no_grad()
def policy_probs(model: ZipGNN, obs: dict) -> np.ndarray:
    logits, _ = model(collate([obs]))
    return torch.softmax(logits, 0).numpy()


@torch.no_grad()
def greedy_rollout(model: ZipGNN, puzzle, early_termination: bool = False) -> tuple[bool, list[int]]:
    env = make_env_for(model, early_termination=early_termination)
    obs, info = env.reset(options={"puzzle": puzzle})
    done = env.done
    while not done:
        a = int(np.argmax(policy_probs(model, obs)))
        obs, _, done, _, info = env.step(a)
    return bool(info["solved"]), info["path"]


@torch.no_grad()
def greedy_batch(model: ZipGNN, puzzles) -> list[bool]:
    """Greedy rollouts for many puzzles at once (batched forward passes)."""
    envs = [make_env_for(model, early_termination=True) for _ in puzzles]
    obs = [e.reset(options={"puzzle": p})[0] for e, p in zip(envs, puzzles)]
    active = list(range(len(envs)))
    solved = [False] * len(envs)
    while active:
        gb = collate([obs[i] for i in active])
        logits, _ = model(gb)
        acts = dense_logits(logits, gb).argmax(1).numpy()
        nxt = []
        for k, i in enumerate(active):
            o, _, done, _, info = envs[i].step(int(acts[k]))
            obs[i] = o
            if done:
                solved[i] = bool(info["solved"])
            else:
                nxt.append(i)
        active = nxt
    return solved


class _Budget(Exception):
    pass


@torch.no_grad()
def policy_search(model: ZipGNN, puzzle, budget: int = 5000) -> dict:
    """Policy-ordered DFS with dead-state pruning. Returns status/path/nodes_expanded.
    (Weaker than :func:`hybrid_solve`, which uses the exact solver's pruning.)"""
    env = make_env_for(model, early_termination=True)
    obs, _ = env.reset(options={"puzzle": puzzle})
    count = [0]
    t0 = time.time()

    def snapshot():
        return (env.head, env.next_cp, env._mask, env.done, len(env.path))

    def restore(s):
        head, ncp, mask, done, plen = s
        for v in env.path[plen:]:
            env.visited[v] = False
        del env.path[plen:]
        env.head, env.next_cp, env._mask, env.done = head, ncp, mask, done

    def dfs(o) -> bool:
        p = policy_probs(model, o)
        legal = np.flatnonzero(o["action_mask"])
        for a in legal[np.argsort(-p[legal])]:
            count[0] += 1  # moves tried (same unit as the solver's nodes_expanded)
            if count[0] > budget:
                raise _Budget
            s = snapshot()
            o2, _, done, _, info = env.step(int(a))
            if done and info["solved"]:
                return True
            if not done and dfs(o2):
                return True
            restore(s)
        return False

    try:
        ok = dfs(obs)
        status = "solved" if ok else "unsat"
    except _Budget:
        status = "budget"
    return {"status": status, "path": list(env.path) if status == "solved" else None,
            "nodes_expanded": count[0], "seconds": time.time() - t0}


def policy_move_order(model: ZipGNN, puzzle, meta: dict | None = None):
    """move_order hook for zipsolve.solver.solve: rank candidates by policy score.

    Returns a :class:`PolicyScorer` (callable; keeps all candidates, skips
    inference for single candidates, caches per state, counts calls)."""
    return PolicyScorer(model, puzzle, meta)


# ---------------------------------------------------------------------------
# Hybrid: exact solver (legality + pruning) + GNN move ordering
# ---------------------------------------------------------------------------
def make_env_for(model=None, meta: dict | None = None, **overrides) -> ZipEnv:
    """A ZipEnv whose observation features match `model` / checkpoint `meta`
    (via zipsolve.rl.gnn.make_env_for_model when available; otherwise the
    feature version is inferred from the model's input width). ``overrides``
    go to ZipEnv (default early_termination=False)."""
    from . import env as _env_mod
    from . import gnn as _gnn_mod
    overrides.setdefault("early_termination", False)
    fn = getattr(_gnn_mod, "make_env_for_model", None)
    if fn is not None:
        return fn(model, meta, **overrides)
    vfd = getattr(_env_mod, "feature_version_for_dim", None)
    in_dim = (getattr(model, "config", None) or {}).get("in_dim")
    if vfd is not None and in_dim is not None:
        overrides.setdefault("feature_version", vfd(int(in_dim)))
    return ZipEnv(**overrides)


_STD_OBS_KEYS = {"x", "edge_index", "action_mask", "head", "neighbors"}


class PolicyScorer:
    """Scores candidate moves at arbitrary solver states with the GNN.

    One ZipEnv is reused (``set_state``) and, for the standard observation
    layout, the static part of the GraphBatch (edges, neighbour table, batch
    vector) is built once, so each evaluation only rebuilds node features.
    Evaluations are cached by (visited bytes, head, next checkpoint).
    """

    def __init__(self, model: ZipGNN, puzzle, meta: dict | None = None, env: ZipEnv | None = None,
                 cache: dict | None = None, max_cache: int = 200_000, deadline: float | None = None):
        self.model = model
        self.puzzle = puzzle
        self.env = env or make_env_for(model, meta)
        self.env.early_termination = False
        obs, _ = self.env.reset(options={"puzzle": puzzle})
        self.n = puzzle.num_nodes
        self._gb = collate([obs]) if set(obs) <= _STD_OBS_KEYS else None
        self.cache = {} if cache is None else cache
        self.max_cache = max_cache
        self.deadline = deadline
        self.hook_calls = 0        # times the hook was asked to order moves
        self.skipped = 0           # single-candidate calls answered without inference
        self.cache_hits = 0
        self.inference_calls = 0   # forward passes actually run
        self.obs_seconds = 0.0     # building observations
        self.inference_seconds = 0.0   # forward passes
        self.deadline_skips = 0
        self._noise_rng = None
        self._temperature = 1.0

    def set_noise(self, rng, temperature: float = 1.0) -> None:
        """Order by logit + temperature * Gumbel noise (sampling moves from the
        policy without replacement) instead of argmax order; None disables."""
        self._noise_rng = rng
        self._temperature = float(temperature)

    @torch.no_grad()
    def logits(self, obs: dict) -> np.ndarray:
        if self._gb is not None:
            gb = dataclasses.replace(self._gb, x=torch.from_numpy(obs["x"]),
                                     mask=torch.from_numpy(np.asarray(obs["action_mask"], dtype=bool)),
                                     head=torch.tensor([int(obs["head"])], dtype=torch.long))
        else:
            gb = collate([obs])
        lg, _ = self.model(gb)
        return lg.numpy()

    def scores(self, visited, head: int, next_cp: int, cands) -> dict:
        """{candidate: logit} at this state (cached)."""
        vis = np.asarray(visited, dtype=bool)
        key = (vis.tobytes(), int(head), int(next_cp))
        hit = self.cache.get(key)
        if hit is not None and all(c in hit for c in cands):
            self.cache_hits += 1
            return hit
        if self.deadline is not None and time.perf_counter() > self.deadline:
            self.deadline_skips += 1   # out of time: let the solver hit its own limit quickly
            return {}
        t0 = time.perf_counter()
        obs = self.env.set_state(vis, head, next_cp)
        t1 = time.perf_counter()
        lg = self.logits(obs)
        t2 = time.perf_counter()
        self.obs_seconds += t1 - t0
        self.inference_seconds += t2 - t1
        self.inference_calls += 1
        legal = np.flatnonzero(obs["action_mask"])
        out = {int(v): float(lg[v]) for v in legal}
        for c in cands:  # never drop a candidate, even if the env disagrees on legality
            out.setdefault(int(c), float("-inf"))
        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[key] = out
        return out

    def order(self, head, cands, visited, next_cp):
        """move_order hook for zipsolve.solver: all candidates, best policy score first."""
        self.hook_calls += 1
        cands = list(cands)
        if len(cands) <= 1:
            self.skipped += 1
            return cands
        s = self.scores(visited, head, next_cp, cands)
        key = np.array([s.get(int(w), float("-inf")) for w in cands], dtype=np.float64)
        if self._noise_rng is not None:
            key = key + self._temperature * self._noise_rng.gumbel(size=len(cands))
        # stable sort: ties (and unscored candidates) keep the solver's heuristic order
        return [cands[i] for i in np.argsort(-key, kind="stable")]

    __call__ = order

    def stats(self) -> dict:
        return {"hook_calls": self.hook_calls, "skipped_single": self.skipped,
                "cache_hits": self.cache_hits, "inference_calls": self.inference_calls,
                "obs_seconds": self.obs_seconds, "inference_seconds": self.inference_seconds,
                "deadline_skips": self.deadline_skips}


def _prefix_next_cp(puzzle, prefix) -> int:
    cps = set(puzzle.checkpoints)
    return sum(1 for v in prefix if v in cps)


def _check_prefix(puzzle, prefix) -> str | None:
    g, cps = puzzle.graph, puzzle.checkpoints
    cpi = {c: i for i, c in enumerate(cps)}
    if not prefix or prefix[0] != cps[0]:
        return "path must start at checkpoint 1"
    seen, nxt = set(), 0
    for i, v in enumerate(prefix):
        if not 0 <= v < g.num_nodes or v in seen:
            return f"invalid or repeated node {v}"
        seen.add(v)
        if i and not g.has_edge(prefix[i - 1], v):
            return f"nodes {prefix[i - 1]} and {v} are not adjacent"
        if v in cpi:
            if cpi[v] != nxt:
                return f"checkpoint {cpi[v] + 1} reached before checkpoint {nxt + 1}"
            nxt += 1
        if v == cps[-1] and i != g.num_nodes - 1:
            return "the last checkpoint must be the final cell"
    return None


def _traced(Search, trace: list, cap: int):
    """``Search`` subclass appending push (node id) / pop (-1) events to ``trace``."""
    class _TracedSearch(Search):  # type: ignore[misc, valid-type]
        def push(self, h):
            super().push(h)
            if len(trace) < cap:
                trace.append(int(h))

        def pop(self):
            super().pop()
            if len(trace) < cap:
                trace.append(-1)
    return _TracedSearch


def _search_once(solver_mod, puzzle, order, time_left, budget, deep, prefix=None, trace=None,
                 trace_cap: int = 60_000):
    """One complete DFS attempt of the exact solver with a node budget.
    Returns (status, path, expanded); status in solved|unsat|timeout|budget.
    ``trace``: optional list receiving the attempt's push/pop events ("R" first)."""
    Search = getattr(solver_mod, "_Search", None)
    if Search is not None and trace is not None:
        if len(trace) < trace_cap:
            trace.append("R")
        Search = _traced(Search, trace, trace_cap)
    if Search is not None:
        try:
            srch = Search(puzzle.graph, puzzle.checkpoints, deep=deep)
            kw = {"prefix": prefix} if prefix is not None and len(prefix) > 1 else {}
            sols, status, expanded = srch.run(1, time_left, order, time.perf_counter(), budget, **kw)
        except TypeError:  # internal API changed: fall back to the public one (no budget)
            pass
        else:
            if sols:
                return "solved", list(sols[0]), expanded
            return {"timeout": "timeout", "budget": "budget"}.get(status, "unsat"), None, expanded
    if prefix is not None and len(prefix) > 1:
        r = solver_mod.solve_from_prefix(puzzle, prefix, time_left, order, deep=deep, restarts=False)
    else:
        r = solver_mod.solve(puzzle, time_limit=time_left, move_order=order, deep=deep, restarts=False)
    return r.status, r.path, r.nodes_expanded


def _native_prefix_support(solver_mod) -> bool:
    import inspect
    Search = getattr(solver_mod, "_Search", None)
    try:
        return Search is not None and "prefix" in inspect.signature(Search.run).parameters
    except (TypeError, ValueError):
        return False


def hybrid_solve(model: ZipGNN, puzzle, time_limit: float | None = 10.0, start_path=None,
                 meta: dict | None = None, *, restarts: bool = True, temperature: float = 1.0,
                 seed: int = 0, cache: dict | None = None, deep: bool = True,
                 trace: list | None = None, trace_cap: int = 60_000) -> dict:
    """Exact solver + GNN move ordering (complete search).

    The exact solver generates the legal candidates and prunes provably dead
    branches (degree / connectivity / parity / articulation points); the
    policy only *orders* the surviving candidates and never drops one, so
    "unsat" is a proof. Inference is skipped when a single candidate is left,
    cached per state (visited bytes, head, next checkpoint) and uses one
    reusable env / GraphBatch.

    ``restarts``: like the plain solver, run attempts with geometrically
    growing node budgets (x1.5). Attempt 0 follows the policy's argmax order;
    later attempts sample orders from the policy (Gumbel-perturbed logits at
    ``temperature``; cached logits are reused). Each attempt is a complete
    DFS, so an exhausted attempt still proves "unsat".

    ``start_path``: continue from this prefix (a legal partial path); the
    solver's state is initialised from it (or, with an older solver, it runs on
    the residual puzzle) while the policy always sees the full board.
    The returned path includes the prefix.

    ``trace``: optional list that receives the search's events for
    visualisation: "R" at the start of each attempt, then node id = push,
    -1 = pop (at most ``trace_cap`` events; node ids are the target puzzle's,
    i.e. the full puzzle's unless an old solver forces the residual fallback).

    Returns dict(status, path, nodes_expanded, attempts, inference_calls,
    cache_hits, skipped_single, hook_calls, deadline_skips, inference_seconds,
    obs_seconds, search_seconds, seconds); status in
    solved | unsat | timeout | invalid.
    """
    from .. import solver as _solver

    t0 = time.perf_counter()
    deadline = None if time_limit is None else t0 + time_limit
    n = puzzle.num_nodes
    prefix = [int(v) for v in (start_path or [puzzle.checkpoints[0]])]
    total = 0
    attempts = 0

    def result(status, path, scorer=None, reason=None):
        secs = time.perf_counter() - t0
        st = scorer.stats() if scorer else dict.fromkeys(
            ("hook_calls", "skipped_single", "cache_hits", "inference_calls", "deadline_skips"), 0) | {
            "obs_seconds": 0.0, "inference_seconds": 0.0}
        out = {"status": status, "path": path, "nodes_expanded": int(total), "attempts": attempts,
               **st, "seconds": secs,
               "search_seconds": max(0.0, secs - st["obs_seconds"] - st["inference_seconds"])}
        if reason:
            out["reason"] = reason
        return out

    bad = (getattr(_solver, "check_prefix", None) or _check_prefix)(puzzle, prefix)
    if bad is not None:
        return result("invalid", None, reason=bad)
    if len(prefix) == n:
        ok = puzzle.is_valid_solution(prefix)
        return result("solved" if ok else "unsat", prefix if ok else None)

    scorer = PolicyScorer(model, puzzle, meta, cache=cache, deadline=deadline)
    run_prefix = None
    if len(prefix) == 1:
        target, order, to_full = puzzle, scorer.order, (lambda path: path)
    elif _native_prefix_support(_solver):
        # the solver initialises its state from the prefix (full-graph node ids)
        target, order, to_full, run_prefix = puzzle, scorer.order, (lambda path: path), prefix
    else:
        # residual puzzle: unvisited nodes + head, checkpoints [head] + remaining ones
        from ..graph import from_edges
        from ..puzzle import Puzzle
        g = puzzle.graph
        head, nxt = prefix[-1], _prefix_next_cp(puzzle, prefix)
        cps_left = puzzle.checkpoints[nxt:]
        if not cps_left:
            return result("unsat", None, scorer, "final checkpoint already reached")
        pset = set(prefix)
        keep = [v for v in range(n) if v not in pset or v == head]
        new_id = {v: i for i, v in enumerate(keep)}
        edges = [(new_id[u], new_id[v]) for u, v in g.edges() if u in new_id and v in new_id]
        target = Puzzle(from_edges(g.coords[keep], edges, g.kind, {}),
                        [new_id[head]] + [new_id[c] for c in cps_left])
        keep_arr = np.asarray(keep)
        base_vis = np.zeros(n, dtype=bool)
        base_vis[list(pset)] = True

        def order(h, cands, visited, sub_nxt):
            if len(cands) <= 1:
                return scorer.order(h, cands, None, 0)
            full_vis = base_vis.copy()
            full_vis[keep_arr[np.asarray(visited, dtype=bool)]] = True
            ordered = scorer.order(keep[h], [keep[c] for c in cands], full_vis, nxt + sub_nxt - 1)
            return [new_id[v] for v in ordered]

        def to_full(path):
            return prefix[:-1] + [keep[i] for i in path]

    budget = max(200, 2 * n) if restarts else None
    rng = np.random.default_rng(seed)
    while True:
        time_left = None if deadline is None else deadline - time.perf_counter()
        if time_left is not None and time_left <= 0:
            return result("timeout", None, scorer)
        scorer.set_noise(rng if attempts > 0 else None, temperature)
        status, path, expanded = _search_once(_solver, target, order, time_left, budget, deep, run_prefix,
                                              trace if target is puzzle else None, trace_cap)
        total += expanded
        attempts += 1
        if status == "budget":
            budget = int(budget * 1.5)
            continue
        if status == "solved":
            full = to_full(path)
            if not puzzle.is_valid_solution(full):  # paranoia
                return result("unsat", None, scorer, "internal: continuation invalid")
            return result("solved", full, scorer)
        return result(status, None, scorer)


def _get_solver():
    try:
        from ..solver import solve  # type: ignore
        return solve
    except Exception:  # noqa: BLE001
        return None


def evaluate(model: ZipGNN, spec: str, episodes: int = 100, seed: int = 12345, search: bool = False,
             budget: int = 5000, solver: bool = False, solver_time: float = 10.0) -> dict:
    sampler = make_sampler(spec, seed=seed)
    puzzles = [sampler() for _ in range(episodes)]
    t0 = time.time()
    greedy = greedy_batch(model, puzzles)
    res = {"spec": spec, "episodes": episodes, "avg_nodes": float(np.mean([p.num_nodes for p in puzzles])),
           "greedy_solve_rate": float(np.mean(greedy)), "greedy_seconds": round(time.time() - t0, 2)}
    if search:
        out = [policy_search(model, p, budget) for p in puzzles]
        for o, p in zip(out, puzzles):
            if o["status"] == "solved":
                assert p.is_valid_solution(o["path"])
        res["search_solve_rate"] = float(np.mean([o["status"] == "solved" for o in out]))
        res["search_nodes_mean"] = float(np.mean([o["nodes_expanded"] for o in out]))
        res["search_nodes_median"] = float(np.median([o["nodes_expanded"] for o in out]))
        res["search_seconds_mean"] = float(np.mean([o["seconds"] for o in out]))
    if solver:
        solve = _get_solver()
        if solve is None:
            res["solver"] = "unavailable"
        else:
            def summarise(prefix, out):
                res[f"{prefix}_solve_rate"] = float(np.mean([o.status == "solved" for o in out]))
                res[f"{prefix}_nodes_mean"] = float(np.mean([o.nodes_expanded for o in out]))
                res[f"{prefix}_nodes_median"] = float(np.median([o.nodes_expanded for o in out]))
                res[f"{prefix}_seconds_mean"] = float(np.mean([o.seconds for o in out]))
            # exact solver with its default (checkpoint-first, Warnsdorff) move order
            summarise("solver", [solve(p, time_limit=solver_time) for p in puzzles])
            # hybrid: same exact solver, moves ordered by the policy (complete search, GNN ordering)
            hy = [hybrid_solve(model, p, solver_time) for p in puzzles]
            res["hybrid_solve_rate"] = float(np.mean([o["status"] == "solved" for o in hy]))
            res["hybrid_nodes_median"] = float(np.median([o["nodes_expanded"] for o in hy]))
            res["hybrid_seconds_mean"] = float(np.mean([o["seconds"] for o in hy]))
            res["hybrid_inference_calls_mean"] = float(np.mean([o["inference_calls"] for o in hy]))
            res["hybrid_inference_frac"] = float(np.sum([o["inference_seconds"] + o["obs_seconds"] for o in hy])
                                                 / max(1e-9, np.sum([o["seconds"] for o in hy])))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--kind", default="grid2d")
    p.add_argument("--size", default="5", help="int, or AxBxC for kind=grid")
    p.add_argument("--spec", default=None, help="full spec, e.g. 'grid3d:3+islands:4' (overrides kind/size)")
    p.add_argument("--checkpoints", type=int, default=None, help="number of checkpoints")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--search", action="store_true", help="also run policy-guided DFS")
    p.add_argument("--budget", type=int, default=5000, help="DFS node budget per puzzle")
    p.add_argument("--solver", action="store_true", help="compare with zipsolve.solver.solve")
    p.add_argument("--solver-time", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.ckpt)
    spec = args.spec or f"{args.kind}:{args.size}" + (f"@{args.checkpoints}" if args.checkpoints else "")
    print(f"generator: {'zipsolve.generator' if using_real_generator() else 'built-in fallback'}")
    res = evaluate(model, spec, args.episodes, args.seed, args.search, args.budget, args.solver,
                   args.solver_time)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
