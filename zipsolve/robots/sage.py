"""🌳 Sage: AlphaZero-style Monte-Carlo tree search.

PUCT selection ``Q + c * P * sqrt(N_parent) / (1 + N_child)`` with priors ``P``
from the shipped GNN policy (``checkpoints/zip_gnn.pt``; uniform priors if torch
or the checkpoint is missing). Leaf value:

* the value head of a PPO fine-tuned checkpoint when one exists
  (``checkpoints/remote/**/D_ppo_ft_best.pt`` / ``D_ppo_ft_final.pt`` whose
  value head is trained), else
* a cheap rollout value: a heuristic playout (next number first, fewest free
  neighbours, a little randomness) scored ``(covered / n) ** 3``; ``0`` when
  the solver proves the position dead. A playout that finishes the puzzle ends
  the search at once.

MCTS-solver style proofs: a node whose state is provably dead (or all of whose
children are) is marked dead and never selected again, so the Sage backs out of
lost branches. After ``sims_per_move`` simulations it commits to the most
visited child (the tree below is kept). Deterministic for a given seed as long
as the time limit is not hit.

Trace: push/pop events for the committed line, ``{"t": "tree", ...}`` snapshots
every ``snap_every`` simulations and ``{"t": "note"}`` captions.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..puzzle import Puzzle
from .base import (Tracer, Walker, direction, result, rng_for, rollout, unavailable, unsupported_reason)

ROBOT = "mcts"
ROOT = Path(__file__).resolve().parents[2]
C_PUCT = 1.4
_BRAINS: dict = {}


def find_value_checkpoint(ckdir: Path | None = None) -> Path | None:
    """A PPO fine-tuned checkpoint (trained value head), if one is present."""
    ckdir = Path(ckdir or ROOT / "checkpoints")
    for name in ("D_ppo_ft_best.pt", "D_ppo_ft_final.pt"):
        hits = sorted((ckdir / "remote").glob(f"**/{name}")) + sorted(ckdir.glob(name))
        if hits:
            return hits[0]
    return None


def _load(path: Path):
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _BRAINS.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        from ..rl.gnn import load_model
        model, ck = load_model(path)
        meta = {k: v for k, v in ck.items() if k != "state_dict"}
    except Exception:  # noqa: BLE001  (no torch / broken checkpoint: the Sage plays without a brain)
        return None
    _BRAINS[key] = (mtime, (model, meta))
    return model, meta


def load_brains(ckdir: Path | None = None) -> dict:
    """{"policy": (model, meta) | None, "value": (model, meta) | None, names}."""
    ckdir = Path(ckdir or ROOT / "checkpoints")
    pol = _load(ckdir / "zip_gnn.pt")
    val = None
    vp = find_value_checkpoint(ckdir)
    if vp is not None:
        got = _load(vp)
        if got is not None:
            from ..app.engine import value_is_trained
            if value_is_trained(got[1]):
                val = got
    return {"policy": pol, "value": val, "policy_name": "zip_gnn.pt" if pol else None,
            "value_name": (str(vp.relative_to(ckdir)) if (val and vp) else None)}


class _Node:
    __slots__ = ("move", "parent", "children", "N", "W", "P", "dead", "expanded")

    def __init__(self, move: int, parent: "_Node | None", prior: float):
        self.move = move
        self.parent = parent
        self.children: list[_Node] = []
        self.N = 0
        self.W = 0.0
        self.P = prior
        self.dead = False
        self.expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


class _Evaluator:
    def __init__(self, puzzle: Puzzle, brains: dict, deadline: float):
        self.puzzle = puzzle
        self.policy = None
        self.value = None
        if brains.get("policy") is not None:
            try:
                from ..rl.evaluate import PolicyScorer
                m, meta = brains["policy"]
                self.policy = PolicyScorer(m, puzzle, meta, deadline=deadline)
            except Exception:  # noqa: BLE001
                self.policy = None
        if brains.get("value") is not None:
            try:
                from ..rl.evaluate import PolicyScorer
                m, meta = brains["value"]
                self.value = (PolicyScorer(m, puzzle, meta, deadline=deadline), meta)
            except Exception:  # noqa: BLE001
                self.value = None
        self.calls = 0

    def priors(self, w: Walker, moves: list[int]) -> list[float]:
        if self.policy is None or len(moves) == 1:
            return [1.0 / len(moves)] * len(moves)
        s = w.s
        vis = np.frombuffer(bytes(s.visited), dtype=np.uint8).astype(bool)
        try:
            lg = self.policy.scores(vis, w.head, s.nxt, moves)
            self.calls += 1
        except Exception:  # noqa: BLE001
            lg = {}
        x = np.array([lg.get(m, float("-inf")) for m in moves], dtype=np.float64)
        if not np.isfinite(x).any():
            return [1.0 / len(moves)] * len(moves)
        x = np.where(np.isfinite(x), x, x[np.isfinite(x)].min() - 4.0)
        e = np.exp(x - x.max())
        p = e / e.sum()
        p = 0.9 * p + 0.1 / len(moves)       # never starve a legal move
        return p.tolist()

    def leaf_value(self, w: Walker, rng) -> tuple[float, list[int] | None]:
        covered, sol = rollout(w, rng)
        if sol is not None:
            return 1.0, sol
        v = (covered / w.n) ** 3
        if self.value is not None:
            scorer, meta = self.value
            try:
                import torch
                from ..app.engine import _winnable
                s = w.s
                vis = np.frombuffer(bytes(s.visited), dtype=np.uint8).astype(bool)
                obs = scorer.env.set_state(vis, w.head, s.nxt)
                from ..rl.gnn import collate
                with torch.no_grad():
                    _, val = scorer.model(collate([obs]))
                pv = _winnable(meta, float(val[0]), w.puzzle, list(w.path))
                if pv is not None:
                    v = 0.5 * v + 0.5 * pv
            except Exception:  # noqa: BLE001
                pass
        return v, None


def _tree_event(root: _Node, root_len: int, head: int, sims: int, k: int = 4) -> dict:
    kids = sorted(root.children, key=lambda c: -c.N)[:k]

    def one(c: _Node, depth: int) -> dict:
        d = {"node": int(c.move), "n": c.N, "q": round(c.Q, 3), "p": round(c.P, 3)}
        if c.dead:
            d["dead"] = True
        if depth > 0 and c.children:
            d["kids"] = [one(g, depth - 1) for g in sorted(c.children, key=lambda g: -g.N)[:3] if g.N]
        return d

    return {"t": "tree", "at": int(root_len), "head": int(head), "sims": int(sims),
            "n": root.N, "nodes": [one(c, 1) for c in kids]}


def run(puzzle: Puzzle, time_limit: float = 10.0, trace: bool = True, seed: int = 0,
        sims_per_move: int | None = None, snap_every: int = 16, brains: dict | None = None) -> dict:
    bad = unsupported_reason(puzzle)
    if bad:
        return unavailable(ROBOT, puzzle, bad, "unsupported", trace)
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    n = puzzle.num_nodes
    rng = rng_for(seed, "sage")
    brains = load_brains() if brains is None else brains
    ev = _Evaluator(puzzle, brains, deadline)
    sims_per_move = sims_per_move or max(24, min(96, 2 * int(math.sqrt(n)) * 4))
    tr = Tracer(trace)
    start = puzzle.checkpoints[0]
    w = Walker(puzzle, [start])
    tr.restart([start])
    brain = ("my policy network" if ev.policy is not None else "no neural network (uniform priors)")
    valtxt = ("a trained value head" if ev.value is not None else "quick simulated futures")
    tr.note(f"I grow a search tree: {brain} suggests moves, {valtxt} judge them. "
            f"{sims_per_move} simulations per move.", kind="intro")

    root = _Node(start, None, 1.0)
    root_path = [start]
    total_sims = 0
    status = "stuck"
    solution = None
    steps: list[dict] = []
    backtracks = 0

    def expand(node: _Node) -> None:
        moves = w.moves()
        node.expanded = True
        if not moves:
            node.dead = len(w.path) < n
            return
        pri = ev.priors(w, moves)
        node.children = [_Node(m, node, p) for m, p in zip(moves, pri)]

    def simulate() -> list[int] | None:
        """One MCTS simulation from the root; returns a full solution if found."""
        node = root
        base = len(w.path)
        value = 0.0
        found = None
        try:
            while True:
                if len(w.path) == n:
                    if w.complete():
                        found = list(w.path)
                        value = 1.0
                    else:
                        node.dead = True
                    break
                if node.dead:
                    break
                if not node.expanded:
                    expand(node)
                    if node.dead:
                        break
                    value, found = ev.leaf_value(w, rng)
                    break
                live = [c for c in node.children if not c.dead]
                if not live:
                    node.dead = True
                    break
                sq = math.sqrt(max(1, node.N))
                fpu = node.Q - 0.1 if node.N else 0.5
                best = max(live, key=lambda c: (c.Q if c.N else fpu) + C_PUCT * c.P * sq / (1 + c.N))
                verdict = w.push(best.move)
                node = best
                if verdict == -2:
                    node.dead = True
                    node.expanded = True
                    break
            # backpropagate (dead leaves count as value 0)
            x = node
            while x is not None:
                x.N += 1
                x.W += value
                if x is not root and x.parent is not None and x.parent.children \
                        and all(c.dead for c in x.parent.children):
                    x.parent.dead = True
                if x is root:
                    break
                x = x.parent
            return found
        finally:
            w.truncate(base)

    while True:
        if len(w.path) == n:
            if w.complete():
                status, solution = "solved", list(w.path)
            break
        if time.perf_counter() > deadline:
            status = "timeout"
            tr.note("Out of thinking time!", kind="stuck")
            break
        if not root.expanded:
            expand(root)
        live = [c for c in root.children if not c.dead]
        if root.dead or not live:
            root.dead = True
            if root.parent is None:
                status = "unsat"
                tr.note("Every branch of my tree is dead: this puzzle has no solution.", kind="stuck")
                break
            backtracks += 1
            parent = root.parent
            tr.note(f"This branch is hopeless; stepping back from {direction(puzzle, parent.move, root.move)}.",
                    kind="backtrack")
            w.pop()
            root_path.pop()
            tr.pop(1)
            steps.pop()
            root = parent
            continue
        if len(live) == 1 and root.children and len(root.children) == 1:
            choice = live[0]
            share = 1.0
        else:
            sims = 0
            while sims < sims_per_move and not root.dead:
                if time.perf_counter() > deadline:
                    break
                found = simulate()
                sims += 1
                total_sims += 1
                if found is not None:
                    solution = found
                    break
                if trace and total_sims % snap_every == 0:
                    tr.event(_tree_event(root, len(root_path), root_path[-1], total_sims))
            if solution is not None:
                tr.event(_tree_event(root, len(root_path), root_path[-1], total_sims))
                tr.note(f"One of my simulated futures reached the end after {total_sims} simulations. "
                        f"Following it!", kind="done")
                for v in solution[len(root_path):]:
                    tr.push(v)
                    steps.append({"node": int(v), "p": None})
                status = "solved"
                break
            live = [c for c in root.children if not c.dead]
            if not live:
                continue
            choice = max(live, key=lambda c: (c.N, c.Q))
            tot = sum(c.N for c in live) or 1
            share = choice.N / tot
            if len(live) > 1 and trace:
                ranked = sorted(live, key=lambda c: -c.N)[:3]
                desc = ", ".join(f"{direction(puzzle, root_path[-1], c.move)} {c.N}x (value {c.Q:.2f})"
                                 for c in ranked)
                tr.note(f"{sims} simulations: {desc}. Going {direction(puzzle, root_path[-1], choice.move)}.",
                        kind="think", move=choice.move)
        verdict = w.push(choice.move)
        root_path.append(choice.move)
        tr.push(choice.move)
        steps.append({"node": int(choice.move), "p": round(share, 3)})
        root = choice
        if verdict == -2:
            root.dead = True
    path = solution if solution is not None else list(w.path)
    return result(ROBOT, puzzle, path, status, t0, tr, steps,
                  total_sims, backtracks=backtracks,
                  stats={"simulations": total_sims, "sims_per_move": sims_per_move,
                         "policy": brains.get("policy_name"), "value": brains.get("value_name") or "rollout",
                         "inference_calls": ev.calls, "backtracks": backtracks})


__all__ = ["run", "ROBOT", "load_brains", "find_value_checkpoint"]
