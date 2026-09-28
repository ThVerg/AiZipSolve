"""🎲 Gambler: flat Monte Carlo.

At every step the Gambler plays ``M`` random-but-sensible futures (heuristic
playouts with a pinch of randomness, see :func:`zipsolve.robots.base.rollout`)
after each legal move and picks the move whose futures *survive* best:

* a future **survives** when it finishes the puzzle or gets at least
  ``horizon`` cells further without the solver proving it dead;
* ties are broken by the average share of the board the futures cover.

A future that finishes the puzzle is a jackpot: the Gambler follows it at once.
If every move busts (or the line is stuck), it backs up one step and bans the
move that got it there (a small, capped number of times: it is a gambler, not a
solver). Captions: "Tried 40 futures: 83% survive going up (best covers 97%)".
Deterministic for a given seed (unless the time limit is hit).
"""
from __future__ import annotations

import time

from ..puzzle import Puzzle
from .base import (Tracer, Walker, direction, pct, result, rng_for, rollout, unavailable,
                   unsupported_reason)

ROBOT = "gambler"


def run(puzzle: Puzzle, time_limit: float = 10.0, trace: bool = True, seed: int = 0,
        futures: int | None = None, max_backtracks: int | None = None) -> dict:
    bad = unsupported_reason(puzzle)
    if bad:
        return unavailable(ROBOT, puzzle, bad, "unsupported", trace)
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    n = puzzle.num_nodes
    rng = rng_for(seed, "gambler")
    M = futures or (24 if n <= 64 else 16)
    max_bt = max_backtracks if max_backtracks is not None else 4 * n
    horizon = max(4, n // 6)
    tr = Tracer(trace)
    start = puzzle.checkpoints[0]
    w = Walker(puzzle, [start])
    tr.restart([start])
    tr.note(f"Feeling lucky! Before every move I'll play {M} random futures and bet on the move "
            f"that survives most often.", kind="intro")
    banned: dict[int, set[int]] = {}      # path length -> moves proven/judged bad there
    steps: list[dict] = []
    status = "stuck"
    solution = None
    rollouts = 0
    backtracks = 0

    def back_up(msg: str) -> bool:
        nonlocal backtracks
        if len(w.path) <= 1 or backtracks >= max_bt:
            return False
        backtracks += 1
        bad_move = w.path[-1]
        w.pop()
        tr.pop(1)
        steps.pop()
        for k in [k for k in banned if k > len(w.path)]:
            del banned[k]
        banned.setdefault(len(w.path), set()).add(bad_move)
        tr.note(msg, kind="backtrack")
        return True

    while True:
        if len(w.path) == n:
            if w.complete():
                status, solution = "solved", list(w.path)
            break
        if time.perf_counter() > deadline:
            status = "timeout"
            tr.note("Out of time. The house wins this round.", kind="stuck")
            break
        head = w.head
        moves = [m for m in w.moves() if m not in banned.get(len(w.path), ())]
        if not moves:
            if back_up("Bust! No way forward from here, so I back up one step and never bet on that move again."):
                continue
            tr.note("Out of chips: every bet from here went bust." if backtracks < max_bt
                    else "Too many busts, I fold.", kind="stuck")
            break
        stats = {}
        for m in moves:
            verdict = w.push(m)
            surv, cov, best = 0, 0, len(w.path)
            if verdict == -2:
                stats[m] = (0.0, 0.0, len(w.path) - 1)
                w.pop()
                continue
            base = len(w.path)
            for _ in range(M):
                covered, sol = rollout(w, rng)
                rollouts += 1
                if sol is not None:
                    solution = sol
                    break
                cov += covered
                best = max(best, covered)
                if covered - base >= horizon:
                    surv += 1
            w.pop()
            if solution is not None:
                break
            stats[m] = (surv / M, cov / (M * n), best)
            if time.perf_counter() > deadline:
                break
        if solution is not None:
            tr.note(f"Jackpot! A future {'after going ' + direction(puzzle, head, m) if len(moves) > 1 else ''} "
                    f"covers the whole board. Cashing in!".replace("  ", " "), kind="done")
            for v in solution[len(w.path):]:
                tr.push(v)
                steps.append({"node": int(v), "p": None})
            status = "solved"
            break
        if not stats:
            continue
        m_best = max(stats, key=lambda m: (stats[m][0], stats[m][1], -m))
        s_best = stats[m_best]
        if s_best[0] == 0 and s_best[1] == 0:
            if back_up(f"Every future from here is a bust ({len(moves)} option{'s' if len(moves) > 1 else ''}"
                       f" tried {M} times each). Backing up one step."):
                continue
            tr.note("Every future is a bust. I fold.", kind="stuck")
            break
        if len(moves) > 1 and trace:
            others = sorted((m for m in stats if m != m_best), key=lambda m: -stats[m][0])[:2]
            extra = "; ".join(f"{direction(puzzle, head, m)} {pct(stats[m][0])}" for m in others)
            tr.note(f"Tried {M * len(stats)} futures: {pct(s_best[0])} survive going "
                    f"{direction(puzzle, head, m_best)} (best covers {pct(s_best[2] / n)})"
                    + (f"; {extra}" if extra else "") + ".", kind="bet", move=m_best,
                    odds={str(m): round(stats[m][0], 3) for m in stats})
        w.push(m_best)
        tr.push(m_best)
        steps.append({"node": int(m_best), "p": round(s_best[0], 3)})
    path = solution if solution is not None else list(w.path)
    return result(ROBOT, puzzle, path, status, t0, tr, steps, rollouts, backtracks=backtracks,
                  stats={"rollouts": rollouts, "futures_per_move": M, "horizon": horizon,
                         "backtracks": backtracks})


__all__ = ["run", "ROBOT"]
