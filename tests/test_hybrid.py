"""Hybrid solver: exact solver pruning + GNN move ordering (zipsolve.rl.evaluate.hybrid_solve)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from zipsolve import solver
from zipsolve.generator import make_puzzle
from zipsolve.graph import grid
from zipsolve.puzzle import Puzzle
from zipsolve.rl.evaluate import PolicyScorer, hybrid_solve, make_env_for, policy_move_order
from zipsolve.rl.gnn import ZipGNN, load_model

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "curric_4to6_final.pt"
FAMILIES = [("grid2d", 5), ("walls", 5), ("islands", 2), ("mask", 6), ("grid3d", 3), ("grid4d", 2)]


def _random_model(seed=0):
    torch.manual_seed(seed)
    env = make_env_for(None)
    env.reset(options={"puzzle": make_puzzle("grid2d", 3, rng=0)})
    m = ZipGNN(in_dim=int(env._obs()["x"].shape[1]), hidden=16, layers=2)
    m.eval()
    return m


@pytest.fixture(scope="module")
def rand_model():
    return _random_model()


@pytest.fixture(scope="module")
def trained():
    if not CKPT.exists():
        pytest.skip("no trained checkpoint")
    m, ck = load_model(CKPT)
    return m, {k: v for k, v in ck.items() if k != "state_dict"}


@pytest.mark.parametrize("kind,size", FAMILIES)
def test_hybrid_solves_all_families(trained, kind, size):
    model, meta = trained
    for seed in range(3):
        p = make_puzzle(kind, size, rng=seed)
        r = hybrid_solve(model, p, 20.0, meta=meta)
        assert r["status"] == "solved", (kind, seed, r["status"])
        assert p.is_valid_solution(r["path"])
        assert r["inference_calls"] + r["cache_hits"] <= r["hook_calls"]
        assert r["seconds"] >= r["inference_seconds"]


def _random_cp_puzzle(rng, shape):
    g = grid(*shape)
    k = int(rng.integers(2, 5))
    cps = rng.choice(g.num_nodes, size=k, replace=False).tolist()
    return Puzzle(g, cps)


def test_untrained_model_is_complete(rand_model):
    """Any (even random) ordering keeps the search complete: same verdicts as the plain solver."""
    rng = np.random.default_rng(0)
    seen = {"solved": 0, "unsat": 0}
    for i in range(40):
        p = _random_cp_puzzle(rng, [(3, 3), (4, 4), (3, 4), (2, 2, 3)][i % 4])
        ref = solver.solve(p, time_limit=20)
        for restarts in (False, True):
            r = hybrid_solve(rand_model, p, 20.0, restarts=restarts)
            assert r["status"] == ref.status, (i, restarts, r["status"], ref.status)
            if r["status"] == "solved":
                assert p.is_valid_solution(r["path"])
        seen[ref.status] += 1
    assert seen["solved"] >= 3 and seen["unsat"] >= 3


def test_unsat_needs_search_still_unsat(rand_model):
    # parity: 3x3 grid, corner (majority colour) to edge-middle (minority) is impossible
    p = Puzzle(grid(3, 3), [0, 1])
    assert hybrid_solve(rand_model, p, 10.0)["status"] == "unsat"
    # a family puzzle with a wrong final checkpoint order is unsat too
    q = make_puzzle("grid2d", 4, rng=1)
    cps = list(q.checkpoints)
    if len(cps) >= 4:
        cps[1], cps[2] = cps[2], cps[1]
        bad = Puzzle(q.graph, cps)
        assert hybrid_solve(rand_model, bad, 10.0)["status"] == solver.solve(bad, 10.0).status


def test_scorer_cache_and_skip(rand_model):
    p = make_puzzle("grid2d", 5, rng=3)
    sc = PolicyScorer(rand_model, p)
    head = p.checkpoints[0]
    vis = np.zeros(p.num_nodes, dtype=bool)
    vis[head] = True
    cands = list(p.graph.neighbors[head])
    assert len(cands) >= 2
    o1 = sc.order(head, cands, vis, 1)
    o2 = sc.order(head, list(cands), vis.copy(), 1)
    assert sorted(o1) == sorted(cands) and o1 == o2           # all candidates kept, deterministic
    assert sc.inference_calls == 1 and sc.cache_hits == 1 and sc.hook_calls == 2
    assert sc.order(head, cands[:1], vis, 1) == cands[:1]      # single candidate: no inference
    assert sc.skipped == 1 and sc.inference_calls == 1
    # a subset returned by the hook would make search incomplete; ours never drops one
    assert set(sc.order(head, cands[::-1], vis, 1)) == set(cands)
    # legacy helper is the same scorer
    assert isinstance(policy_move_order(rand_model, p), PolicyScorer)


def test_scores_match_policy_logits(trained):
    """Fast path (reused GraphBatch) gives the same logits as a fresh collate."""
    from zipsolve.rl.gnn import collate
    model, meta = trained
    p = make_puzzle("islands", 3, rng=2)
    sc = PolicyScorer(model, p, meta)
    env = make_env_for(model, meta)
    env.reset(options={"puzzle": p})
    obs = env._obs()
    with torch.no_grad():
        ref = model(collate([obs]))[0].numpy()
    got = sc.logits(obs)
    legal = np.flatnonzero(obs["action_mask"])
    np.testing.assert_allclose(got[legal], ref[legal], rtol=1e-5, atol=1e-5)


def test_start_path(rand_model):
    p = make_puzzle("grid2d", 5, rng=4)
    sol = solver.solve(p, 10).path
    for k in (2, 7, len(sol) - 1, len(sol)):
        r = hybrid_solve(rand_model, p, 10.0, start_path=sol[:k])
        assert r["status"] == "solved" and r["path"][:k] == sol[:k] and p.is_valid_solution(r["path"])
    # a legal but dead prefix is proven unsat
    for k in range(3, len(sol)):
        pre = sol[:k - 1]
        for w in p.graph.neighbors[pre[-1]]:
            cand = pre + [w]
            if w in pre or w == sol[k - 1] or w in p.checkpoints:
                continue
            if solver.solve_from_prefix(p, cand, 10).status == "unsat" if hasattr(solver, "solve_from_prefix") \
                    else False:
                assert hybrid_solve(rand_model, p, 10.0, start_path=cand)["status"] == "unsat"
                break
        else:
            continue
        break
    assert hybrid_solve(rand_model, p, 10.0, start_path=sol[::-1])["status"] == "invalid"


def test_time_limit_respected(rand_model):
    p = make_puzzle("grid3d", 4, rng=0)
    r = hybrid_solve(rand_model, p, 0.2, restarts=False)
    assert r["status"] in ("solved", "timeout")
    assert r["seconds"] < 2.0
