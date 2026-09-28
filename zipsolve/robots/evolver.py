"""🧬 Evolver: a genetic algorithm over candidate lines.

Individuals are legal partial paths from number 1 (a complete one is a
solution); fitness is the number of cells covered (a legal prefix already
respects the number order and every other rule). Each generation:

* **elitism**: the best few survive unchanged;
* **crossover** (edge inheritance): the child copies a prefix of parent A, then
  keeps taking the move parent B makes from the same cell whenever it is legal,
  else a heuristic move;
* **mutation**: *regrow* (cut the tail at a random point and grow it again),
  *backbite* (the head links back to an earlier neighbour on its own line and
  the loop in between is reversed: the classic Hamiltonian-path move) or
  *segment reversal* (2-opt: reverse a stretch whose ends can be reconnected),
  each followed by regrowing the tail.

Growth is the same heuristic playout the other robots use (next number first,
fewest free neighbours, randomness, forced moves). Every edited line is checked
with the solver's :func:`~zipsolve.solver.check_prefix`, so the robot follows
every rule of the puzzle kind. It does not always solve large boards; the
result then reports the best line found.

Trace: ``{"t": "path", "p": best}`` whenever the champion changes (and every
few generations) plus a caption with the generation number and fitness.
"""
from __future__ import annotations

import time

from ..puzzle import Puzzle
from .base import (Tracer, Walker, heuristic_key, is_legal_prefix, pct, result, rng_for,
                   unavailable, unsupported_reason)

ROBOT = "evolver"


class Line(list):
    """A candidate line; ``alive`` = length of its longest prefix the solver can't
    yet prove hopeless (the fitness, ties broken by total length)."""
    alive: int = 0


def _grow(puzzle: Puzzle, prefix: list[int], rng, eps: float, guide: dict | None = None) -> Line:
    """Extend a legal prefix until stuck (guide: preferred successor per cell)."""
    w = Walker(puzzle, prefix)
    while len(w.path) < w.n:
        ms = w.moves()
        if not ms:
            break
        f = w.verdicts[-1]
        g = guide.get(w.head) if guide else None
        if g is not None and g in ms and rng.random() < 0.9:
            v = g
        elif f >= 0 and f in ms:
            v = f
        elif len(ms) == 1:
            v = ms[0]
        elif rng.random() < eps:
            v = ms[rng.randrange(len(ms))]
        else:
            best = min(heuristic_key(w, m) for m in ms)
            ties = [m for m in ms if heuristic_key(w, m) == best]
            v = ties[rng.randrange(len(ties))]
        w.push(v)
    line = Line(w.path)
    dead_at = next((i for i, r in enumerate(w.verdicts) if r == -2), None)
    line.alive = len(line) if dead_at is None else dead_at
    if len(line) == w.n and not w.complete():
        line.alive = min(line.alive, len(line) - 1)
    return line


def _cut_point(rng, line) -> int:
    """Random cut in [1, alive], biased towards the end of the living part."""
    L = min(len(line), getattr(line, "alive", len(line)) + 1)
    if L <= 2:
        return 1
    x = 1.0 - rng.random() ** 2        # more mass near 1
    return max(1, min(L - 1, int(x * (L - 1)) + (1 if rng.random() < 0.5 else 0)))


def _backbite(puzzle: Puzzle, path: list[int], rng) -> list[int] | None:
    nb = puzzle.graph.neighbors
    h = path[-1]
    pos = {v: i for i, v in enumerate(path)}
    opts = [pos[u] for u in nb[h] if u in pos and pos[u] < len(path) - 2]
    if not opts:
        return None
    i = opts[rng.randrange(len(opts))]
    new = path[:i + 1] + path[i + 1:][::-1]
    return new if is_legal_prefix(puzzle, new) else None


def _reverse(puzzle: Puzzle, path: list[int], rng, tries: int = 12) -> list[int] | None:
    g = puzzle.graph
    L = len(path)
    if L < 5:
        return None
    for _ in range(tries):
        i = rng.randrange(1, L - 2)
        j = rng.randrange(i + 1, L - 1)
        if g.has_edge(path[i - 1], path[j]) and g.has_edge(path[i], path[j + 1]):
            new = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
            if is_legal_prefix(puzzle, new):
                return new
    return None


def _crossover(puzzle: Puzzle, a: list[int], b: list[int], rng, eps: float) -> list[int]:
    k = _cut_point(rng, a)
    guide = {u: v for u, v in zip(b, b[1:])}
    return _grow(puzzle, a[:k], rng, eps, guide)


def run(puzzle: Puzzle, time_limit: float = 10.0, trace: bool = True, seed: int = 0,
        population: int | None = None, generations: int | None = None) -> dict:
    bad = unsupported_reason(puzzle)
    if bad:
        return unavailable(ROBOT, puzzle, bad, "unsupported", trace)
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    n = puzzle.num_nodes
    rng = rng_for(seed, "evolver")
    P = population or (24 if n <= 80 else 16)
    G = generations or 5000
    elite = 2
    tr = Tracer(trace)
    start = puzzle.checkpoints[0]
    tr.restart([start])
    tr.note(f"Let evolution do the work: {P} random lines compete, the fittest breed and mutate.",
            kind="intro")

    def fit(p: Line) -> tuple:
        return (p.alive, len(p))

    pop = [_grow(puzzle, [start], rng, eps) for eps in
           [0.05 + 0.6 * i / max(1, P - 1) for i in range(P)]]
    evaluated = P
    best = max(pop, key=fit)
    tr.path_event(best)
    tr.note(f"Generation 0: the best random line covers {len(best)}/{n} cells ({pct(len(best) / n)}).",
            kind="generation", gen=0, fitness=len(best))
    status = "stuck"
    gen = 0
    history = [best.alive]
    stall = 0
    while len(best) < n:
        if gen >= G:
            break
        if time.perf_counter() > deadline:
            status = "timeout"
            break
        gen += 1
        pop.sort(key=fit, reverse=True)
        nxt = list(pop[:elite])
        eps = 0.1 + min(0.4, 0.02 * stall)     # more randomness when evolution stalls

        def pick() -> list[int]:
            c = [pop[rng.randrange(len(pop))] for _ in range(3)]
            return max(c, key=fit)

        while len(nxt) < P:
            r = rng.random()
            a = pick()
            if r < 0.35:
                child = _crossover(puzzle, a, pick(), rng, eps)
            elif r < 0.6:
                bb = _backbite(puzzle, a, rng)
                child = _grow(puzzle, bb, rng, eps) if bb else _grow(puzzle, a[:_cut_point(rng, a)], rng, eps)
            elif r < 0.75:
                rv = _reverse(puzzle, a, rng)
                base = rv if rv else a
                child = _grow(puzzle, base[:_cut_point(rng, base)] if not rv else base, rng, eps)
            else:
                child = _grow(puzzle, a[:_cut_point(rng, a)], rng, eps)
            evaluated += 1
            nxt.append(child)
            if len(child) == n:
                break
        pop = nxt
        champ = max(pop, key=fit)
        improved = fit(champ) > fit(best)
        if improved or champ != best:
            stall = 0 if improved else stall + 1
            best = champ
        else:
            stall += 1
        history.append(best.alive)
        if improved or gen % 5 == 0 or len(best) == n:
            tr.path_event(best)
            tr.note(f"Generation {gen}: fitness {best.alive}/{n}. The best line covers {len(best)} cells"
                    + (f", the first {best.alive} of them still look solvable." if best.alive < len(best) else "."),
                    kind="generation", gen=gen, fitness=best.alive)
    if len(best) == n and puzzle.check_solution(best) is None:
        status = "solved"
        tr.note(f"Generation {gen}: a perfect line evolved! Every cell covered.", kind="done", gen=gen)
    elif status != "timeout":
        tr.note(f"Evolution stalled after {gen} generations: the best line covers {len(best)}/{n} cells.",
                kind="stuck", gen=gen)
    else:
        tr.note(f"Out of time after {gen} generations: the best line covers {len(best)}/{n} cells.",
                kind="stuck", gen=gen)
    steps = [{"node": int(v), "p": None} for v in best[1:]]
    return result(ROBOT, puzzle, best, status, t0, tr, steps, evaluated,
                  stats={"generations": gen, "population": P, "individuals": evaluated,
                         "best_fitness": best.alive, "best_length": len(best), "fitness_history": history[-200:]})


__all__ = ["run", "ROBOT"]
