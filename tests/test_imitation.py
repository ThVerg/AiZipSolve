"""Tests for solver.solve_from_prefix / label_moves and zipsolve.rl.imitation."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zipsolve.generator import make_puzzle
from zipsolve.graph import grid
from zipsolve.puzzle import Puzzle
from zipsolve.solver import check_prefix, label_moves, legal_moves, solve, solve_from_prefix


# --------------------------------------------------------------------------- helpers
def brute_count(p: Puzzle, prefix, cap: int = 10**6) -> int:
    """Number of valid completions of `prefix` (plain DFS, no pruning)."""
    n = p.num_nodes
    path = list(prefix)
    vis = set(path)
    count = 0

    def dfs():
        nonlocal count
        if count >= cap:
            return
        if len(path) == n:
            count += p.is_valid_solution(path)
            return
        for w in legal_moves(p, path):
            path.append(w)
            vis.add(w)
            dfs()
            path.pop()
            vis.discard(w)

    dfs()
    return count


def tiny_puzzles():
    rng = np.random.default_rng(7)
    out = []
    for kind, size, k in [("grid2d", 3, 2), ("grid2d", 4, 2), ("grid2d", 4, 4), ("walls", 4, 3),
                          ("mask", 5, 3), ("grid3d", 2, 2), ("grid4d", 2, 3), ("grid", (2, 3, 2), 3)]:
        out.append(make_puzzle(kind, size, num_checkpoints=k, rng=rng))
    return out


def random_legal_prefixes(p: Puzzle, rng, count=6):
    """Random walks of the game rules (may or may not be completable)."""
    out = []
    for _ in range(count):
        path = [p.checkpoints[0]]
        L = int(rng.integers(1, p.num_nodes))
        while len(path) < L:
            mv = legal_moves(p, path)
            if not mv:
                break
            path.append(int(rng.choice(mv)))
        out.append(path)
    return out


# --------------------------------------------------------------------------- solver API
def test_prefix_solve_on_solution_prefixes():
    rng = np.random.default_rng(0)
    for kind, size in [("grid2d", 6), ("walls", 6), ("islands", 3), ("grid3d", 3), ("mask", 7)]:
        p = make_puzzle(kind, size, rng=rng)
        full = solve(p, time_limit=20)
        assert full.status == "solved"
        assert solve_from_prefix(p, [p.checkpoints[0]]).path == full.path  # same search
        for k in [1, 2, len(p.solution) // 2, len(p.solution) - 2, len(p.solution) - 1, len(p.solution)]:
            r = solve_from_prefix(p, p.solution[:k], time_limit=20)
            assert r.status == "solved", (kind, k)
            assert r.path[:k] == p.solution[:k]
            assert p.is_valid_solution(r.path)


def test_prefix_solve_matches_brute_force():
    rng = np.random.default_rng(1)
    for p in tiny_puzzles():
        for pre in random_legal_prefixes(p, rng):
            r = solve_from_prefix(p, pre)
            bc = brute_count(p, pre, cap=1)
            assert (r.status == "solved") == (bc > 0), (p.checkpoints, pre)
            assert r.status in ("solved", "unsat")
            if r.path:
                assert r.path[:len(pre)] == pre and p.is_valid_solution(r.path)


def test_prefix_rejects_invalid():
    p = Puzzle(grid(3, 3), [0, 2, 4, 8])
    good = [0, 1, 2, 5, 4]
    assert check_prefix(p, good) is None
    bad = [[], [1], [0, 2], [0, 1, 0], [0, 3, 4], [0, 1, 99], [0, 1, 2, 5, 8],
           [0, 3, 6, 7, 8]]
    for pre in bad:
        assert check_prefix(p, pre) is not None, pre
        with pytest.raises(ValueError):
            solve_from_prefix(p, pre)
        with pytest.raises(ValueError):
            label_moves(p, pre, 1.0)


def test_label_moves_matches_brute_force():
    rng = np.random.default_rng(2)
    for p in tiny_puzzles():
        for pre in random_legal_prefixes(p, rng) + [p.solution[:3]]:
            if len(pre) == p.num_nodes:
                continue
            labels, wit = label_moves(p, pre, None, return_witnesses=True)
            assert list(labels) == legal_moves(p, pre)
            for m, lab in labels.items():
                truth = brute_count(p, pre + [m], cap=1) > 0
                assert lab == ("win" if truth else "lose"), (pre, m)
                if lab == "win":
                    assert wit[m][:len(pre) + 1] == pre + [m] and p.is_valid_solution(wit[m])


def test_label_moves_known_solution_and_unknown_on_timeout():
    p = make_puzzle("grid2d", 8, num_checkpoints=2, rng=np.random.default_rng(3))
    pre = p.solution[:5]
    labs = label_moves(p, pre, 0.0)          # no time at all: nothing can be proven
    assert labs and set(labs.values()) == {"unknown"}
    labs = label_moves(p, pre, 0.0, known_solution=p.solution)
    assert labs[p.solution[5]] == "win"
    assert all(v == "unknown" for m, v in labs.items() if m != p.solution[5])
    # a tiny (non-zero) limit on a harder puzzle: every "lose" must really be a proof
    q = make_puzzle("grid3d", 4, num_checkpoints=2, rng=np.random.default_rng(4))
    labs = label_moves(q, q.solution[:3], 1e-4)
    for m, v in labs.items():
        if v == "lose":
            assert solve_from_prefix(q, q.solution[:3] + [m], time_limit=30).status == "unsat"


# --------------------------------------------------------------------------- imitation
from zipsolve.rl import imitation as im  # noqa: E402
from zipsolve.rl.env import ZipEnv  # noqa: E402
from zipsolve.rl.gnn import load_model  # noqa: E402

SMALL = {"grid2d": (4, 5), "walls": (5, 5), "grid3d": (3, 3), "islands_chain": (2, 2)}


@pytest.fixture(scope="module")
def small_ds(tmp_path_factory):
    ds = im.generate_dataset(24, workers=2, seed=5, val_frac=0.25, families=SMALL,
                             label_time=0.2, val_label_time=0.2, verbose=False)
    path = tmp_path_factory.mktemp("imit") / "ds.pkl"
    im.save_dataset(ds, path)
    return ds, path


def test_dataset_split_has_no_puzzle_overlap(small_ds):
    ds, _ = small_ds
    tr = [r for r in ds["records"] if r["split"] == "train"]
    va = [r for r in ds["records"] if r["split"] == "val"]
    assert tr and va
    assert not ({r["id"] for r in tr} & {r["id"] for r in va})
    assert not ({im.puzzle_signature(r["puzzle"]) for r in tr} & {im.puzzle_signature(r["puzzle"]) for r in va})
    d = im.Dataset(ds)
    assert all(d.split[e.puzzle] == "train" for e in d.train)
    assert all(d.split[e.puzzle] == "val" for e in d.val)
    # branching only (forced states skipped), targets are verified wins and legal
    for e in d.train:
        assert e.nlegal >= 2 and e.pos and e.sol in e.pos
        assert set(e.pos) <= set(legal_moves(d.puzzles[e.puzzle], list(e.prefix)))
        assert not set(e.pos) & set(e.excl)


def test_set_loss_math():
    z = torch.tensor([[1.0, 2.0, 3.0, float("-inf")]])
    T, Fa = True, False
    lse = lambda *v: math.log(sum(math.exp(x) for x in v))  # noqa: E731
    # single positive, all legal in the denominator = cross-entropy
    l1 = im.set_loss(z, torch.tensor([[Fa, T, Fa, Fa]]), torch.tensor([[T, T, T, Fa]]))
    assert l1.item() == pytest.approx(lse(1, 2, 3) - 2, abs=1e-5)
    # set of two winners: -log(p1 + p2)
    l2 = im.set_loss(z, torch.tensor([[Fa, T, T, Fa]]), torch.tensor([[T, T, T, Fa]]))
    assert l2.item() == pytest.approx(-math.log((math.exp(2) + math.exp(3)) / (math.exp(1) + math.exp(2) + math.exp(3))), abs=1e-5)
    # move 0 unknown -> excluded from numerator and denominator
    l3 = im.set_loss(z, torch.tensor([[Fa, T, Fa, Fa]]), torch.tensor([[Fa, T, T, Fa]]))
    assert l3.item() == pytest.approx(lse(2, 3) - 2, abs=1e-5)
    # everything counted is positive -> zero loss
    l4 = im.set_loss(z, torch.tensor([[T, T, T, Fa]]), torch.tensor([[T, T, T, Fa]]))
    assert l4.item() == pytest.approx(0.0, abs=1e-6)


def test_tiny_pretrain_end_to_end(small_ds, tmp_path):
    _, path = small_ds
    out = tmp_path / "imit.pt"
    best = im.main(["train", "--data", str(path), "--epochs", "1", "--workers", "2", "--threads", "2",
                    "--hidden", "16", "--layers", "2", "--out", str(out)])
    assert out.exists() and "per_family" in best
    model, ck = load_model(out)
    assert ck["pretrained"] is True and ck["config"]["hidden"] == 16
    p = make_puzzle("grid2d", 5, rng=np.random.default_rng(9))
    env = ZipEnv(early_termination=True)
    obs, info = env.reset(options={"puzzle": p})
    done = False
    with torch.no_grad():
        from zipsolve.rl.gnn import collate
        while not done:
            logits, _ = model(collate([obs]))
            obs, _, done, _, info = env.step(int(torch.argmax(logits)))
    assert "solved" in info
    # one tiny DAgger round runs and writes a checkpoint
    res = im.main(["dagger", "--ckpt", str(out), "--data", str(path), "--rounds", "1",
                   "--puzzles-per-round", "6", "--epochs", "1", "--workers", "2", "--threads", "2",
                   "--samples", "1", "--label-time", "0.05", "--out", str(tmp_path / "dag")])
    assert len(res["rounds"]) == 2
    load_model(tmp_path / "dag_r1.pt")
