"""The Architect (zipsolve.architect): designed puzzles are unique, fair, hard, deterministic."""
import json
from pathlib import Path

import numpy as np
import pytest

from zipsolve.architect import (FEATURE_NAMES, TARGETS, collect_samples, design, design_features, spearman,
                                train_surrogate)
from zipsolve.difficulty import rate
from zipsolve.generator import make_puzzle
from zipsolve.puzzle import Puzzle
from zipsolve.solver import count_solutions

BANK = Path(__file__).resolve().parents[1] / "zipsolve" / "app" / "static" / "bank"


def _check(r):
    assert r.ok
    p = r.puzzle
    assert p.is_valid_solution(p.solution)
    assert count_solutions(p, 2, time_limit=30) == (1, "complete")
    again = rate(p)
    assert again["features"]["guesses"] == 0 and again["score"] == r.rating["score"]


@pytest.mark.parametrize("kind,size", [("classic", 7), ("walls", 7), ("islands", 4)])
def test_design_unique_and_fair(kind, size):
    r = design(kind, size, "expert", time_budget=60, seed=3, evals=250, surrogate=None)
    _check(r)
    assert r.log and r.log[-1]["phase"] == "done"
    d = r.to_dict()
    assert d["rating"]["guesses"] == 0 and d["puzzle"]["solution"] == r.puzzle.solution
    json.dumps(d)
    if kind == "walls":
        assert r.puzzle.graph.meta.get("walls") is not None


def test_design_deterministic():
    a = design("classic", 6, "expert", time_budget=60, seed=11, evals=150)
    b = design("classic", 6, "expert", time_budget=60, seed=11, evals=150)
    assert a.puzzle.checkpoints == b.puzzle.checkpoints and a.puzzle.solution == b.puzzle.solution
    assert a.rating["score"] == b.rating["score"]
    c = design("classic", 6, "expert", time_budget=60, seed=12, evals=150)
    assert (c.puzzle.solution, c.puzzle.checkpoints) != (a.puzzle.solution, a.puzzle.checkpoints)


def test_designed_harder_than_baseline():
    """Same board: fair (no-guess) puzzles at/above Hard are far more common from the Architect than from
    the random unique generator (whose hard puzzles are mostly hard because they need guessing)."""
    designed = [design("classic", 8, "expert", time_budget=120, seed=s, evals=800).rating for s in range(3)]
    base = [rate(make_puzzle("grid2d", 8, rng=900 + s, unique=True, time_limit=30)) for s in range(10)]
    assert all(r["features"]["guesses"] == 0 for r in designed)

    def fair_hard(rs):
        return np.mean([r["features"]["guesses"] == 0 and r["score"] >= TARGETS["hard"][0] for r in rs])
    assert fair_hard(designed) > fair_hard(base)
    fair_base = [r["score"] for r in base if r["features"]["guesses"] == 0] or [0.0]
    assert np.median([r["score"] for r in designed]) > np.median(fair_base)


def test_features_and_surrogate_train():
    p = make_puzzle("grid2d", 6, rng=1, unique=True)
    f = design_features(p, p.solution)
    assert f.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(f))
    X, D, F, G, B = collect_samples(2, 120, configs=[("grid2d", 6, "expert"), ("walls", 6, "expert")], workers=1)
    assert len(X) == len(D) == len(F) == len(G) == len(B) > 100 and F.min() == 0 and F.max() == 1
    assert X.shape[1] == 2 * len(FEATURE_NAMES)
    sur, rep = train_surrogate(X, D, F, G, base=B, epochs=40, holdout=0.25)
    nf = len(FEATURE_NAMES)
    s, pf = sur.predict(X[:5, :nf], X[:5, :nf] - X[:5, nf:])
    assert s.shape == (5,) and np.all((pf >= 0) & (pf <= 1))
    assert rep["test"] > 0 and np.isfinite(rep["spearman_score"]) and 0 <= rep["fair_auc"] <= 1
    r = design("classic", 6, "expert", time_budget=60, seed=5, evals=150, surrogate=sur)
    assert r.ok and r.stats["surrogate_calls"] > 0
    assert spearman([1, 2, 3, 4], [2, 4, 6, 9]) == pytest.approx(1.0)


def test_architect_endpoint():
    from fastapi.testclient import TestClient
    from zipsolve.app.server import create_app
    c = TestClient(create_app())
    r = c.post("/api/architect/design", json={"kind": "classic", "size": 6, "target": "hard", "time_budget": 4,
                                               "seed": 7, "include_solution": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["source"] == "architect" and d["rating"]["guesses"] == 0 and d["log"]
    assert "solution" not in d["puzzle"] and d["id"]
    p = Puzzle.from_dict({**d["puzzle"], "solution": None})
    assert p.is_valid_solution(d["solution"])
    assert c.post("/api/architect/design", json={"kind": "nope"}).status_code == 400
    assert c.post("/api/architect/design", json={"target": "trivial"}).status_code == 400


@pytest.mark.skipif(not (BANK / "pools" / "architect-expert.json").exists(), reason="architect pools not built")
def test_shipped_architect_bank():
    idx = json.loads((BANK / "index.json").read_text())
    ids = set()
    for diff, lo in (("expert", 62), ("insane", 80)):
        info = idx["modes"]["architect"][diff]
        assert idx["architect"]["pools"][diff] == info
        lst = json.loads((BANK / info["file"]).read_text())
        assert len(lst) == info["count"] > 0
        for e in lst:
            assert e["diff"] == diff and e["score"] >= lo and e["id"] not in ids and e["id"].startswith("a")
            assert e["architect"]["guesses"] == 0
            ids.add(e["id"])
        p = Puzzle.from_dict(lst[0]["puzzle"])
        assert p.is_valid_solution(lst[0]["puzzle"]["solution"])
        assert (BANK / "robots" / f"{lst[0]['id']}.json").exists()
    weekly = json.loads((BANK / idx["architect"]["weekly"]["file"]).read_text())
    assert "2026-W01" in weekly and "2027-W52" in weekly
    e = weekly["2026-W01"]
    assert e["id"] not in ids and e["week"] == "2026-W01"
    assert count_solutions(Puzzle.from_dict(e["puzzle"]), 2, time_limit=60) == (1, "complete")
