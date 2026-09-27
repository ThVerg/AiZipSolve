"""Tests for the RL package: environment rules, batching, PPO smoke run."""
import numpy as np
import pytest
import torch

from zipsolve.graph import grid
from zipsolve.puzzle import Puzzle
from zipsolve.rl.env import F, NUM_FEATURES, RewardConfig, ZipEnv, dead_state
from zipsolve.rl.gnn import ZipGNN, collate, dense_logits, load_model, masked_distribution, save_model
from zipsolve.rl.sampling import make_sampler, parse_spec, snake_puzzle


def snake(rows=4, cols=4):
    # 0 1 2 3 / 7 6 5 4 / 8 9 10 11 / 15 14 13 12 ; checkpoints at path positions 0, 5, 10, last
    p = snake_puzzle(rows, cols, 2, rng=0)
    path = p.solution
    cps = [path[0], path[5], path[10], path[-1]]
    return Puzzle(p.graph, cps, path)


def test_replay_solution_solves():
    p = snake()
    cfg = RewardConfig(solve=1.0, fail=-1.0, checkpoint=0.1, step=0.0, progress=0.5)
    env = ZipEnv(reward=cfg)
    obs, info = env.reset(options={"puzzle": p})
    assert obs["x"].shape == (16, NUM_FEATURES) and obs["x"].dtype == np.float32
    assert obs["edge_index"].shape == (2, 2 * len(p.graph.edges()))
    total = 0.0
    for a in p.solution[1:]:
        assert obs["action_mask"][a]
        obs, r, term, trunc, info = env.step(a)
        total += r
    assert term and not trunc and info["solved"]
    assert info["path"] == p.solution
    assert total == pytest.approx(1.0 + 0.5 + 0.1 * 3)


def test_mask_rules():
    g = grid(3, 3)
    # checkpoints: start 0, then 2, end 8
    p = Puzzle(g, [0, 2, 8])
    env = ZipEnv(early_termination=False)
    obs, _ = env.reset(options={"puzzle": p})
    m = env.action_masks()
    assert m[1] and m[3] and m.sum() == 2
    # future checkpoint out of order is masked: start at 4 with checkpoints 4, 1, 7, 8
    p2 = Puzzle(g, [4, 1, 7, 8])
    env.reset(options={"puzzle": p2})
    m = env.action_masks()
    assert m[1] and m[3] and m[5] and not m[7]
    # end node only as last move
    p3 = Puzzle(g, [4, 5])
    env.reset(options={"puzzle": p3})
    assert not env.action_masks()[5]


def test_illegal_action_terminates():
    env = ZipEnv()
    env.reset(options={"puzzle": snake()})
    _, r, term, _, info = env.step(15)
    assert term and r == -1.0 and not info["solved"]


def test_early_termination_dead_end():
    g = grid(3, 3)
    # 0 1 2 / 3 4 5 / 6 7 8 ; start 1, end 0. Moving 1->4 leaves node 2 with a single
    # usable neighbour (5), so it can only be a path end -> dead state.
    p = Puzzle(g, [1, 0])
    env = ZipEnv()
    env.reset(options={"puzzle": p})
    _, r, term, _, info = env.step(4)
    assert term and r < 0 and not info["solved"]
    env2 = ZipEnv(early_termination=False)
    env2.reset(options={"puzzle": p})
    _, r, term, _, _ = env2.step(4)
    assert not term


def test_dead_state_disconnected():
    g = grid(1, 5)
    vis = np.array([False, False, True, False, False])
    assert dead_state(g.neighbors, vis, 2, 4, 4) is not None


def test_features_are_dimension_agnostic():
    for spec in ["grid2d:4", "grid3d:3", "grid4d:2", "islands:2"]:
        env = ZipEnv(make_sampler(spec, 0))
        obs, _ = env.reset()
        x = obs["x"]
        assert x.shape[1] == NUM_FEATURES
        assert np.isfinite(x).all() and x.min() >= -1 and x.max() <= 1
        assert x[:, F["is_head"]].sum() == 1


def test_random_policy_episodes_terminate():
    env = ZipEnv(make_sampler("grid2d:5+walls:5+mask:5", 1))
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, _ = env.reset()
        for _ in range(env.n):
            a = rng.choice(np.flatnonzero(obs["action_mask"]))
            obs, r, term, trunc, info = env.step(a)
            if term:
                break
        assert term
        if info["solved"]:
            assert env.puzzle.is_valid_solution(info["path"])


def test_collate_and_model_shapes():
    envs = [ZipEnv(make_sampler(s, 0)) for s in ["grid2d:4", "grid3d:3", "islands:2", "grid2d:5"]]
    obs = [e.reset()[0] for e in envs]
    gb = collate(obs)
    sizes = [o["x"].shape[0] for o in obs]
    assert gb.x.shape == (sum(sizes), NUM_FEATURES)
    assert gb.ptr.tolist() == np.concatenate([[0], np.cumsum(sizes)]).tolist()
    assert gb.batch.shape[0] == sum(sizes) and gb.num_graphs == 4
    assert gb.edge_index.shape[1] == sum(o["edge_index"].shape[1] for o in obs)
    assert gb.nbr.shape[0] == sum(sizes)
    for k, o in enumerate(obs):
        assert gb.head[k] == int(o["head"]) + gb.ptr[k]
    model = ZipGNN(hidden=32, layers=2)
    logits, values = model(gb)
    assert logits.shape == (sum(sizes),) and values.shape == (4,)
    assert torch.isinf(logits[~gb.mask]).all() and torch.isfinite(logits[gb.mask]).all()
    dist = masked_distribution(logits, gb)
    a = dist.sample()
    for k, o in enumerate(obs):
        assert o["action_mask"][int(a[k])]
    assert dense_logits(logits, gb).shape == (4, max(sizes))
    # batching is equivalent to running graphs one at a time
    single = torch.cat([model(collate([o]))[0] for o in obs])
    assert torch.allclose(torch.nan_to_num(single, neginf=0), torch.nan_to_num(logits, neginf=0), atol=1e-5)


def test_save_load(tmp_path):
    m = ZipGNN(hidden=16, layers=1)
    save_model(m, tmp_path / "m.pt")
    m2, ck = load_model(tmp_path / "m.pt")
    assert ck["config"]["hidden"] == 16


def test_parse_spec():
    assert parse_spec("grid2d:5+grid:3x3x2@4") == [("grid2d", 5, None), ("grid", (3, 3, 2), 4)]


def test_ppo_smoke(tmp_path):
    from zipsolve.rl.ppo import PPOConfig, PPOTrainer
    cfg = PPOConfig(num_envs=4, rollout_steps=8, epochs=2, minibatch_size=16, hidden=16, layers=2)
    tr = PPOTrainer(make_sampler("grid2d:4", 0), cfg, run_name="t", run_dir=str(tmp_path / "runs"),
                    ckpt_dir=str(tmp_path / "ck"), verbose=False)
    for _ in range(3):
        row = tr.train_iteration()
    assert row["iter"] == 3 and np.isfinite(row["pg_loss"])
    path = tr.save()
    tr.close()
    assert path.exists() and (tmp_path / "runs" / "t.csv").exists()


def test_policy_search_finds_solution():
    from zipsolve.rl.evaluate import policy_search
    model = ZipGNN(hidden=16, layers=1)
    p = snake()
    out = policy_search(model, p, budget=20000)
    assert out["status"] == "solved" and p.is_valid_solution(out["path"])


def test_train_cli_help():
    from zipsolve.rl.train import build_parser
    assert "--stages" in build_parser().format_help()


@pytest.mark.parametrize("steps,mb", [(3, 2), (1, 1), (5, 4)])
def test_ppo_singleton_minibatch_stays_finite(tmp_path, steps, mb):
    """Regression: a 1-sample minibatch made a.std() NaN and wrote NaN weights."""
    from zipsolve.rl.ppo import PPOConfig, PPOTrainer
    torch.manual_seed(0)
    cfg = PPOConfig(num_envs=1, rollout_steps=steps, epochs=1, minibatch_size=mb, hidden=16, layers=2)
    tr = PPOTrainer(make_sampler("grid2d:4", 0), cfg, run_name="t1", run_dir=str(tmp_path / "runs"),
                    ckpt_dir=str(tmp_path / "ck"), verbose=False)
    for _ in range(3):
        row = tr.train_iteration()
        assert np.isfinite(row["pg_loss"]) and np.isfinite(row["v_loss"])
    tr.close()
    assert all(torch.isfinite(p).all() for p in tr.model.parameters())
