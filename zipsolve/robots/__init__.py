"""AI robots with different strategies for Zip puzzles.

Every robot has the same entry point::

    run(puzzle, time_limit=10.0, trace=True, seed=0) -> dict

returning the app's robot response shape (``path``, ``solved``, ``status``,
``trace``, ``steps``, ``nodes_expanded``, ``seconds``, ``robot``, ``stats`` ...;
see CONTRACT.md "Clarifications (robots)"). Results are deterministic for a
given seed as long as the time limit is not reached.

=============  =========================================================
id             robot
=============  =========================================================
``detective``  🕵️ Detective: human deductions with explanations, guesses
               and backtracking (``explain_next_move`` for hints)
``mcts``       🌳 Sage: AlphaZero-style MCTS (GNN priors, rollout value)
``evolver``    🧬 Evolver: genetic algorithm over candidate lines
``gambler``    🎲 Gambler: flat Monte Carlo over random futures
``sat``        🧮 Mathematician: SAT encoding solved by CaDiCaL (optional
               dependency ``python-sat``)
=============  =========================================================
"""
from __future__ import annotations

from typing import Callable

from ..puzzle import Puzzle
from . import detective, evolver, gambler, mathematician, sage
from .detective import explain_next_move

ROBOTS: dict[str, dict] = {
    "detective": {"name": "Detective", "emoji": "🕵️", "line": "Explains every move, guesses only when stuck",
                  "run": detective.run},
    "mcts": {"name": "Sage", "emoji": "🌳", "line": "Grows a search tree of possible futures (AlphaZero-style)",
             "run": sage.run},
    "evolver": {"name": "Evolver", "emoji": "🧬", "line": "Breeds and mutates whole lines (genetic algorithm)",
                "run": evolver.run},
    "gambler": {"name": "Gambler", "emoji": "🎲", "line": "Plays random futures and bets on the best odds",
                "run": gambler.run},
    "sat": {"name": "Mathematician", "emoji": "🧮", "line": "Turns the puzzle into logic and calls a SAT solver",
            "run": mathematician.run},
}
ALIASES = {"sage": "mcts", "mathematician": "sat", "alphazero": "mcts", "ga": "evolver", "genetic": "evolver",
           "montecarlo": "gambler", "mc": "gambler"}


def resolve(name: str) -> str:
    key = (name or "").strip().lower()
    key = ALIASES.get(key, key)
    if key not in ROBOTS:
        raise KeyError(f"unknown robot {name!r}; one of: {', '.join(ROBOTS)}")
    return key


def available(name: str) -> bool:
    return resolve(name) != "sat" or mathematician.available()


def listing() -> list[dict]:
    return [{"id": k, "name": v["name"], "emoji": v["emoji"], "line": v["line"], "available": available(k)}
            for k, v in ROBOTS.items()]


def run_robot(name: str, puzzle: Puzzle, time_limit: float = 10.0, trace: bool = True, seed: int = 0) -> dict:
    key = resolve(name)
    fn: Callable = ROBOTS[key]["run"]
    out = fn(puzzle, time_limit=time_limit, trace=trace, seed=seed)
    out["robot"] = key
    out["robot_name"] = ROBOTS[key]["name"]
    out["emoji"] = ROBOTS[key]["emoji"]
    return out


__all__ = ["ROBOTS", "run_robot", "resolve", "available", "listing", "explain_next_move"]
