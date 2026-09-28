"""Human difficulty rater (zipsolve.difficulty)."""
import numpy as np

from zipsolve.difficulty import LABELS, label_for, rate
from zipsolve.generator import make_puzzle
from zipsolve.graph import grid
from zipsolve.puzzle import Puzzle
from zipsolve.solver import count_solutions


def _unique(kind, size, seed, **kw):
    p = make_puzzle(kind, size, rng=seed, unique=True, time_limit=30, **kw)
    assert count_solutions(p, 2) == (1, "complete")
    return p


def test_label_for_thresholds():
    assert label_for(0) == "Easy" and label_for(100) == "Insane"
    labs = [label_for(s) for s in range(0, 101)]
    # monotone: labels only ever go up with the score
    assert [LABELS.index(x) for x in labs] == sorted(LABELS.index(x) for x in labs)
    assert set(labs) == set(LABELS)


def test_deterministic():
    p = _unique("grid2d", 6, 3)
    a, b = rate(p), rate(p)
    a["features"].pop("seconds"), b["features"].pop("seconds")
    assert a == b
    assert 0 <= a["score"] <= 100 and a["label"] in LABELS
    for k in ("forced_frac", "max_lookahead", "branching_points", "nodes", "spread", "guesses"):
        assert k in a["features"]


def test_trivially_forced_puzzle_is_easy_and_boring():
    # 3x4 snake with a number on every other cell: every move is forced
    g = grid(3, 4)
    snake = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]
    p = Puzzle(g, snake[::2] + [11], snake)
    r = rate(p)
    assert r["score"] < 10 and r["label"] == "Easy"
    assert r["features"]["max_lookahead"] == 0
    assert r["boring"]


def test_lookahead_puzzle_scores_higher():
    # find a generated puzzle whose rating needs lookahead; it must beat the forced snake
    easy = rate(Puzzle(grid(3, 4), [0, 2, 6, 4, 10, 11], [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]))
    for seed in range(20):
        r = rate(_unique("grid2d", 6, seed, num_checkpoints=3))
        if r["features"]["max_lookahead"] >= 1:
            assert r["score"] > easy["score"] + 5
            assert r["features"]["branching_points"] >= 1
            assert not r["boring"]
            return
    raise AssertionError("no lookahead puzzle found")


def test_score_correlates_with_size():
    sizes, scores = [], []
    for size in (4, 5, 6, 7):
        for seed in range(8):
            sizes.append(size)
            scores.append(rate(_unique("grid2d", size, 100 + seed))["score"])
    rho = np.corrcoef(np.argsort(np.argsort(sizes)), np.argsort(np.argsort(scores)))[0, 1]
    assert rho > 0.4, rho
    means = [np.mean([s for z, s in zip(sizes, scores) if z == k]) for k in (4, 5, 6, 7)]
    assert means[0] < means[-1]


def test_rate_without_stored_solution():
    p = _unique("walls", 5, 2)
    q = Puzzle(p.graph, p.checkpoints)          # no solution: found by the solver
    a, b = rate(p), rate(q)
    assert a["score"] == b["score"]
