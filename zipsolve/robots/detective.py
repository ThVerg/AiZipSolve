"""🕵️ Detective: solves like a person, one justified move at a time.

At every step the Detective lists the legal moves (solver rules) and tries to
rule out each wrong one with the human techniques of
:mod:`zipsolve.difficulty`, cheapest first:

* **L1** local dead ends (a cell left with one exit, a corner stranded, two
  cells both needing the line next),
* **L2** regions (the free cells split in two, the next number cut off),
* **LA1** one-line lookahead ("then the next moves are forced ... stuck"),
* **L3** expert rules (chessboard colour count, forced two-exit chains,
  bottlenecks),
* **LA2** two-level lookahead ("whatever I do next, I get stuck").

If exactly one move survives it plays it and explains why in plain English
(``{"t": "note", "msg": ...}`` events). If several survive it makes an
educated guess (and says so); a contradiction later sends it back to its most
recent guess. Every refutation is sound, so on a puzzle with a unique solution
every *deduced* move is the solution's move.

``explain_next_move(puzzle, path)`` gives the teaching hint for the game's Hint
button: ``{"status", "move", "reason", "technique", "cells", ...}``.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Sequence

from .. import solver as _solver
from ..difficulty import MAX_CHAIN, NODE_BUDGET, _Budget, _Human
from ..puzzle import Puzzle
from .base import (Tracer, cell_name, direction, going, heuristic_key, is_solution, legal,
                   new_search, region_name, result, unavailable, unsupported_reason)

ROBOT = "detective"
TECH_NAMES = {"rules": "the rules", "L1": "a local dead end", "L2": "region reasoning",
              "LA1": "a short lookahead", "L3": "an expert rule", "LA2": "a deeper lookahead",
              "G": "a guess", "search": "a search"}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


class _Deducer(_Human):
    """difficulty._Human with the solver's public legality and explained refutations."""

    def __init__(self, puzzle: Puzzle):
        super().__init__(puzzle)
        self.puzzle = puzzle
        self.s = new_search(puzzle, deep=True)
        self.label = {c: i + 1 for i, c in enumerate(puzzle.checkpoints)}

    def legal(self) -> list[int]:
        s = self.s
        if s.remaining == 0:
            return []
        return legal(self.puzzle, s.path)

    # ---- explained static checks (after the head moved p -> h) -------------
    def local_detail(self, p: int) -> dict | None:
        s = self.s
        h = s.path[-1]
        vis, deg, nbrs, end = s.visited, s.deg, s.nbrs, s.end
        forced = -1
        any_free = False
        hn = nbrs[h]
        for x in hn:
            if vis[x]:
                continue
            any_free = True
            d = deg[x]
            if x == end:
                if d == 0 and s.remaining > 1:
                    return {"why": "end_trapped", "cell": x}
            elif d <= 1:
                if d == 0:
                    return {"why": "pocket", "cell": x}
                if forced >= 0:
                    return {"why": "two_forced", "cells": [forced, x]}
                forced = x
        if not any_free:
            return {"why": "no_exit", "cell": h}
        if forced >= 0:
            c = s.cpi[forced]
            if c >= 0 and c != s.nxt:
                return {"why": "forced_cp", "cell": forced}
        for x in nbrs[p]:
            if vis[x] or x in hn:
                continue
            if deg[x] < (1 if x == end else 2):
                return {"why": "stranded", "cell": x, "left": deg[x]}
        return None

    def region_detail(self) -> dict | None:
        s = self.s
        vis, nbrs, cpi = s.visited, s.nbrs, s.cpi
        h = s.path[-1]
        free = [x for x in nbrs[h] if not vis[x]]
        if not free:
            return {"why": "no_exit", "cell": h}
        comps = []
        seen_all: set[int] = set()
        for st in range(s.n):
            if vis[st] or st in seen_all:
                continue
            comp = {st}
            dq = [st]
            while dq:
                v = dq.pop()
                for w in nbrs[v]:
                    if not vis[w] and w not in comp:
                        comp.add(w)
                        dq.append(w)
            seen_all |= comp
            comps.append(comp)
        if len(comps) > 1:
            comps.sort(key=len)
            return {"why": "split", "cells": sorted(comps[0]), "parts": len(comps)}
        nxt = s.nxt
        if nxt >= len(s.cps):
            return None
        target = s.cps[nxt]
        if target in nbrs[h]:
            return None
        seen = {target}
        dq2 = deque([target])
        hn = set(nbrs[h])
        while dq2:
            v = dq2.popleft()
            for w in nbrs[v]:
                if vis[w] or w in seen or cpi[w] > nxt:
                    continue
                if w in hn:
                    return None
                seen.add(w)
                dq2.append(w)
        return {"why": "cp_unreachable", "cell": target}

    def expert_detail(self) -> dict | None:
        s = self.s
        if s.global_check():
            return None
        h = s.path[-1]
        col = s.col
        if col is not None and s.remaining > 0:
            ch = col[h]
            total = s.remaining + 1
            bal = 1 + sum(1 if col[u] == ch else -1 for u in range(s.n) if not s.visited[u])
            if col[s.end] != ch ^ ((total - 1) & 1) or bal != (total & 1):
                return {"why": "parity"}
        s.rebuild_low()
        if s.chains and s.chain_check(h) == -2:
            return {"why": "chain"}
        if s.deep and not s.deep_check():
            return {"why": "bottleneck"}
        return {"why": "expert"}

    def forced_line(self) -> tuple[bool, list[int], dict | None]:
        """Follow the moves that are forced (only one move survives L1/L2)."""
        line: list[int] = []
        try:
            while True:
                if self.s.remaining == 0:
                    return False, list(line), None
                moves = self.viable()
                if not moves:
                    return True, list(line), {"cell": self.s.path[-1]}
                if len(moves) != 1 or len(line) >= MAX_CHAIN:
                    return False, list(line), None
                self.push(moves[0])
                line.append(moves[0])
        finally:
            for _ in line:
                self.pop()

    def refute_explained(self, w: int, max_tier: str = "LA2") -> tuple[str, dict] | None:
        """Cheapest technique proving the move w (from the current head) loses."""
        s = self.s
        h = s.path[-1]
        s.push(w)
        try:
            if s.remaining == 0:
                return None
            d = self.local_detail(h)
            if d:
                return "L1", d
            d = self.region_detail()
            if d:
                return "L2", d
            if max_tier == "L2":
                return None
            self.pushes, self.budget = 0, NODE_BUDGET
            try:
                ok, line, stuck = self.forced_line()
            except _Budget:
                ok, line, stuck = False, [], None
            if ok:
                return "LA1", {"why": "line", "line": line, "cell": (stuck or {}).get("cell")}
            d = self.expert_detail()
            if d:
                return "L3", d
            if max_tier == "L3":
                return None
            self.pushes, self.budget = 0, NODE_BUDGET
            try:
                ok, _ = self.refute(2)
            except _Budget:
                ok = False
            if ok:
                return "LA2", {"why": "fork"}
            return None
        finally:
            s.pop()


# ---------------------------------------------------------------------------
# plain-English reasons
# ---------------------------------------------------------------------------
def _refutation_text(p: Puzzle, head: int, w: int, tech: str, d: dict) -> str:
    g = going(p, head, w)
    why = d.get("why")
    cell = d.get("cell")
    cn = cell_name(p, cell) if cell is not None else ""
    if why == "pocket":
        return f"{g} would turn {cn} into a dead-end pocket with no way out"
    if why == "two_forced":
        a, b = d["cells"]
        return f"{g} would leave both {cell_name(p, a)} and {cell_name(p, b)} needing the line next, and it can't do both"
    if why == "no_exit":
        return f"{g} walks into a dead end with no free cell next to it"
    if why == "forced_cp":
        return f"{g} would force the line onto {cn} before its turn"
    if why == "stranded":
        left = d.get("left", 1)
        what = "no free neighbours" if left == 0 else "only one free neighbour"
        return f"{g} would strand {cn}: it would be left with {what}"
    if why == "end_trapped":
        return f"{g} would wall in {cn}, the last number"
    if why == "split":
        return f"{g} would cut off {region_name(p, d['cells'])}"
    if why == "cp_unreachable":
        return f"{g} would leave {cn} unreachable without crossing a higher number"
    if why == "line":
        k = len(d.get("line") or [])
        at = cell_name(p, cell) if cell is not None else "a dead end"
        if k == 0:
            return f"{g} leaves no safe move afterwards"
        if k == 1:
            return f"{g}, the next move is forced and the line gets stuck at {at}"
        return f"{g}, the next {k} moves are forced and the line gets stuck at {at}"
    if why == "parity":
        return f"{g} breaks the chessboard count: the cells left can't be covered by alternating colours"
    if why == "chain":
        return f"{g} would force the two-exit cells to link up into a loop (or put the numbers out of order)"
    if why == "bottleneck":
        return f"{g} would create a bottleneck the line can't pass through in the right order"
    if why == "fork":
        return f"{g}, every way of continuing runs into a dead end within a few moves"
    return f"{g} leads to a position that can't be finished"


def _forced_by_rules_text(p: Puzzle, head: int, m: int, dz: _Deducer) -> str:
    s = dz.s
    free = [x for x in s.nbrs[head] if not s.visited[x]]
    lab = dz.label.get(m)
    if len(free) <= 1:
        return f"Only one free cell next to the line, so it goes {direction(p, head, m)}"
    if lab is not None:
        return f"Number {lab} is next and right here, so the line takes it"
    why = _blocked_reasons(p, dz, head, [x for x in free if x != m])
    if why:
        return f"The line must go {direction(p, head, m)}: {why}"
    return (f"The only move that doesn't jump ahead to a later number (or break a rule) is "
            f"{direction(p, head, m)}")


def _blocked_reasons(p: Puzzle, dz: _Deducer, head: int, cells: list[int]) -> str:
    """Why free neighbours are illegal (descriptive only: legality is the solver's)."""
    try:
        from ..graph import blocked_steps, prerequisites
        bl = blocked_steps(p.graph) or {}
        pre = prerequisites(p.graph) or {}
    except ImportError:
        bl, pre = {}, {}
    s = dz.s
    out = []
    for x in cells:
        d = direction(p, head, x)
        if x in bl.get(head, ()):
            out.append(f"{d} is against a one-way arrow")
        elif any(not s.visited[a] for a in pre.get(x, ())):
            out.append(f"the door {d} is locked until its key is collected")
        elif s.cpi[x] > s.nxt:
            out.append(f"{d} is number {s.cpi[x] + 1}, which comes later")
    return "; ".join(out)


def _deduced_text(p: Puzzle, head: int, m: int, refs: dict, dz: _Deducer) -> str:
    s = dz.s
    wrong = [(w, r) for w, r in refs.items() if w != m and r is not None]
    # positive phrasings first
    if m != s.end and s.deg[m] == 1 and not dz.label.get(m):
        d = direction(p, head, m)
        tail = "" if d.startswith("to ") else f": it goes {d}"
        return f"{_cap(cell_name(p, m))} has only two exits, so the line must pass through it now{tail}"
    lab = dz.label.get(m)
    if lab is not None and wrong and all(r[1].get("why") == "cp_unreachable" for _, r in wrong):
        return f"Number {lab} is only reachable this way"
    order = {"L1": 0, "L2": 1, "LA1": 2, "L3": 3, "LA2": 4}
    wrong.sort(key=lambda t: order.get(t[1][0], 9))
    parts = [_refutation_text(p, head, w, r[0], r[1]) for w, r in wrong[:2]]
    if not parts:
        return f"The line goes {direction(p, head, m)}"
    txt = "; ".join([_cap(parts[0])] + parts[1:])
    if len(wrong) > 2:
        txt += f" (and {len(wrong) - 2} more option{'s fail' if len(wrong) > 3 else ' fails'} too)"
    return f"{txt}. So the line goes {direction(p, head, m)}"


def _cells_of(ref: tuple[str, dict] | None) -> list[int]:
    if not ref:
        return []
    d = ref[1]
    out = []
    if d.get("cell") is not None:
        out.append(int(d["cell"]))
    out.extend(int(c) for c in (d.get("cells") or [])[:12])
    return out


# ---------------------------------------------------------------------------
# one step of reasoning
# ---------------------------------------------------------------------------
def _analyse(dz: _Deducer, max_tier: str = "LA2") -> dict:
    """Decision at the deducer's current state:
    {"kind": "done"|"forced"|"deduced"|"guess"|"dead", "move", "options", "refs", "reason", "tech"}."""
    s = dz.s
    p = dz.puzzle
    h = s.path[-1]
    if s.remaining == 0:
        return {"kind": "done"}
    moves = dz.legal()
    if not moves:
        return {"kind": "dead", "reason": f"The line at {cell_name(p, h)} has no legal move left",
                "tech": "rules", "cells": [h]}
    if len(moves) == 1:
        m = moves[0]
        ref = dz.refute_explained(m, max_tier)
        if ref is not None:
            return {"kind": "dead", "move": m, "refs": {m: ref}, "tech": ref[0], "cells": _cells_of(ref),
                    "reason": _cap(_refutation_text(p, h, m, *ref)) + ", and that's the only legal move"}
        return {"kind": "forced", "move": m, "tech": "rules", "cells": [m],
                "reason": _forced_by_rules_text(p, h, m, dz)}
    refs = {w: dz.refute_explained(w, max_tier) for w in moves}
    alive = [w for w in moves if refs[w] is None]
    if not alive:
        w0 = min(moves, key=lambda w: {"L1": 0, "L2": 1, "LA1": 2, "L3": 3, "LA2": 4}[refs[w][0]])
        texts = [_refutation_text(p, h, w, *refs[w]) for w in moves[:3]]
        return {"kind": "dead", "refs": refs, "tech": refs[w0][0], "cells": _cells_of(refs[w0]),
                "reason": "Every way forward fails: " + "; ".join(texts)}
    if len(alive) == 1:
        m = alive[0]
        worst = max((refs[w][0] for w in moves if w != m),
                    key=lambda t: {"L1": 0, "L2": 1, "LA1": 2, "L3": 3, "LA2": 4}[t])
        cells = [m] + [c for w in moves if w != m for c in _cells_of(refs[w])]
        return {"kind": "deduced", "move": m, "refs": refs, "tech": worst, "cells": cells[:12],
                "reason": _deduced_text(p, h, m, refs, dz)}
    key = lambda w: heuristic_key(_WalkerView(dz), w)   # noqa: E731
    alive.sort(key=key)
    names = [direction(p, h, w) for w in alive]
    opts = ", ".join(names[:-1]) + " or " + names[-1]
    return {"kind": "guess", "move": alive[0], "options": alive, "refs": refs, "tech": "G",
            "cells": list(alive),
            "reason": f"I can't rule out {opts} yet, so I'll make an educated guess: "
                      f"{direction(p, h, alive[0])}"}


class _WalkerView:
    """Adapter so base.heuristic_key can score moves on a deducer's search state."""

    def __init__(self, dz: _Deducer):
        self.s = dz.s

    def dist_to_target(self, v: int) -> int:
        s = self.s
        return s.dist[min(s.nxt, len(s.cps) - 1)][v]


# ---------------------------------------------------------------------------
# the robot
# ---------------------------------------------------------------------------
def run(puzzle: Puzzle, time_limit: float = 10.0, trace: bool = True, seed: int = 0) -> dict:
    """Deduction-first DFS with explained moves. Deterministic (seed unused:
    the Detective never rolls dice)."""
    bad = unsupported_reason(puzzle)
    if bad:
        return unavailable(ROBOT, puzzle, bad, "unsupported", trace)
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    tr = Tracer(trace)
    n = puzzle.num_nodes
    start = puzzle.checkpoints[0]
    dz = _Deducer(puzzle)
    dz.s.push(start)
    tr.restart([start])
    tr.note("Let's look at this puzzle carefully. I'll only move when I can explain why.", kind="intro")
    how: list[str] = []                 # technique per path step (after the start)
    conf: list[float] = []
    guesses: list[tuple[int, list[int]]] = []   # (path length at the guess, untried options)
    counts: dict[str, int] = {}
    pushes = 0
    wrong_guesses = 0
    status = "stuck"
    prev_forced = False

    def play(m: int, tech: str, c: float):
        nonlocal pushes
        dz.s.push(m)
        tr.push(m)
        how.append(tech)
        conf.append(c)
        counts[tech] = counts.get(tech, 0) + 1
        pushes += 1

    def unwind(at: int):
        while len(dz.s.path) > at:
            dz.s.pop()
            how.pop()
            conf.pop()
        tr.pop(len(tr.stack) - at)

    def backtrack(reason: str) -> bool:
        """Undo the most recent guess that still has untried options."""
        nonlocal wrong_guesses
        first = True
        while guesses:
            at, rest = guesses.pop()
            wrong = dz.s.path[at]
            wrong_guesses += 1
            unwind(at)
            gh = dz.s.path[-1]
            what = f"{direction(puzzle, gh, wrong)} from {cell_name(puzzle, gh)}"
            msg = (f"Contradiction! {reason}. So my guess ({what}) was wrong" if first
                   else f"That means my earlier guess ({what}) was wrong too")
            first = False
            if not rest:
                tr.note(msg + "; backing up further.", kind="backtrack")
                continue
            if len(rest) == 1:
                tr.note(msg + f", so the line must go {direction(puzzle, gh, rest[0])}.",
                        kind="backtrack", move=rest[0], cells=[rest[0]], technique="G")
                play(rest[0], "G", 1.0)
            else:
                tr.note(msg + f". Next I'll try {direction(puzzle, gh, rest[0])}.", kind="backtrack",
                        move=rest[0], cells=list(rest), technique="G")
                guesses.append((at, list(rest[1:])))
                play(rest[0], "G", 1.0 / len(rest))
            return True
        return False

    while True:
        if len(dz.s.path) == n:
            if is_solution(puzzle, dz.s.path):
                status = "solved"
                tr.note("Every cell is covered and the numbers are in order. Case closed!", kind="done")
                break
            dec = {"kind": "dead", "reason": "The line is complete but breaks a rule", "cells": []}
        elif time.perf_counter() > deadline:
            status = "timeout"
            tr.note("I've run out of time on this case.", kind="stuck")
            break
        else:
            dec = _analyse(dz)
        k = dec["kind"]
        head = dz.s.path[-1]
        if k == "forced":
            if not prev_forced:
                tr.note(dec["reason"] + ".", kind="forced", move=dec["move"], cells=dec.get("cells"),
                        technique="rules")
            prev_forced = True
            play(dec["move"], "rules", 1.0)
            continue
        prev_forced = False
        if k == "deduced":
            tr.note(dec["reason"] + ".", kind="deduce", move=dec["move"], cells=dec.get("cells"),
                    technique=dec["tech"])
            play(dec["move"], dec["tech"], 1.0)
            continue
        if k == "guess":
            opts = dec["options"]
            tr.note(dec["reason"] + ".", kind="guess", move=opts[0], cells=opts, technique="G")
            guesses.append((len(dz.s.path), list(opts[1:])))
            play(opts[0], "G", 1.0 / len(opts))
            continue
        # dead: back to the most recent guess
        dead_reason = dec.get("reason") or "Dead end"
        if not backtrack(dead_reason):
            status = "unsat"
            tr.note(f"{dead_reason}. Every possibility fails, so this puzzle has no solution!",
                    kind="stuck", cells=dec.get("cells"))
            break
    path = list(dz.s.path)
    steps = [{"node": int(v), "p": round(c, 3), "how": t} for v, c, t in zip(path[1:], conf, how)]
    return result(ROBOT, puzzle, path, status, t0, tr, steps, pushes,
                  stats={"techniques": counts, "guesses": counts.get("G", 0), "wrong_guesses": wrong_guesses,
                         "deduced_frac": round(1 - counts.get("G", 0) / max(1, len(how)), 3)})


# ---------------------------------------------------------------------------
# teaching hints
# ---------------------------------------------------------------------------
def explain_next_move(puzzle: Puzzle, path: Sequence[int] | None, time_limit: float = 3.0,
                      solution: Sequence[int] | None = None) -> dict:
    """Teaching hint for the position ``path``.

    Returns ``{"status": "next", "move": v, "reason": str, "technique": t, "cells": [...]}``
    (``technique`` one of rules / L1 / L2 / LA1 / L3 / LA2 / search), or
    ``{"status": "backtrack", "keep": k, "move": v, "reason": ...}`` when the line
    can't be finished (keep the first k nodes, then play v), ``"done"`` when solved,
    ``"invalid"`` for an illegal path, ``"timeout"`` when even the solver gives up.
    """
    t0 = time.perf_counter()
    path = [int(v) for v in (path or [puzzle.checkpoints[0]])]
    bad = _solver.check_prefix(puzzle, path)
    if bad is not None:
        return {"status": "invalid", "move": None, "reason": f"That line breaks a rule: {bad}."}
    if unsupported_reason(puzzle):
        return {"status": "invalid", "move": None, "reason": unsupported_reason(puzzle)}
    if len(path) == puzzle.num_nodes:
        ok = is_solution(puzzle, path)
        return {"status": "done" if ok else "invalid", "move": None,
                "reason": "Solved! Every cell is covered." if ok else "The line is complete but breaks a rule."}
    dz = _Deducer(puzzle)
    for v in path:
        dz.s.push(v)
    dec = _analyse(dz)
    head = path[-1]
    base = {"technique_name": TECH_NAMES.get(dec.get("tech", ""), "")}
    if dec["kind"] in ("forced", "deduced"):
        return {"status": "next", "move": int(dec["move"]), "reason": dec["reason"] + ".",
                "technique": dec["tech"], "cells": [int(c) for c in dec.get("cells") or []],
                "seconds": round(time.perf_counter() - t0, 3), **base}
    left = max(0.2, time_limit - (time.perf_counter() - t0))
    if dec["kind"] == "guess":
        opts = dec["options"]
        known = None
        if solution is not None and list(solution[:len(path)]) == path and is_solution(puzzle, solution):
            known = int(solution[len(path)])
        winner, losers = None, []
        if known in opts:
            winner = known
        else:
            for m in opts:
                r = _solver.solve_from_prefix(puzzle, path + [m], time_limit=left / len(opts))
                if r.status == "solved":
                    winner = m
                    break
                if r.status == "unsat":
                    losers.append(m)
        if winner is None:
            return {"status": "timeout", "move": None, "technique": "search",
                    "reason": "This one is tricky: no simple deduction applies and I couldn't settle it in time.",
                    "seconds": round(time.perf_counter() - t0, 3)}
        others = [going(puzzle, head, m) for m in opts if m != winner]
        why = (f"No quick deduction works here: {', '.join(others)} can't be ruled out at a glance. "
               f"Thinking further ahead, {going(puzzle, head, winner)} is the move that works")
        if losers:
            why += f" ({' and '.join(direction(puzzle, head, m) for m in losers)} eventually runs into a dead end)"
        return {"status": "next", "move": int(winner), "reason": why + ".", "technique": "search",
                "cells": [int(winner)], "technique_name": TECH_NAMES["search"],
                "seconds": round(time.perf_counter() - t0, 3)}
    # dead position: find how far to back up (longest completable prefix)
    reason = "This line can't be finished. " + (dec.get("reason") or "")
    lo, hi, best = 1, len(path) - 1, None   # path[:lo] assumed completable, path[:hi + 1] is not
    while lo < hi and time.perf_counter() - t0 < time_limit:
        mid = (lo + hi + 1) // 2
        r = _solver.solve_from_prefix(puzzle, path[:mid], time_limit=max(0.1, time_limit / 4))
        if r.status == "solved":
            lo, best = mid, (mid, r.path)
        elif r.status == "unsat":
            hi = mid - 1
        else:
            break
    if best is None or best[0] != lo:
        r = _solver.solve_from_prefix(puzzle, path[:lo], time_limit=max(0.2, time_limit / 2))
        if r.status != "solved":
            return {"status": "timeout" if r.status == "timeout" else "unsat", "move": None,
                    "reason": reason + ".", "technique": dec.get("tech"),
                    "seconds": round(time.perf_counter() - t0, 3)}
        best = (lo, r.path)
    keep, sol = best
    return {"status": "backtrack", "keep": int(keep), "move": int(sol[keep]),
            "reason": f"{reason}. Back up to step {keep} and go {direction(puzzle, path[keep - 1], sol[keep])} instead.",
            "technique": dec.get("tech"), "cells": [int(c) for c in dec.get("cells") or []],
            "seconds": round(time.perf_counter() - t0, 3), **base}


__all__ = ["run", "explain_next_move", "ROBOT"]
