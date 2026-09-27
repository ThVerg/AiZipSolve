"""Tests for the frozen benchmark, training reliability features and sampling."""
import csv
import json

import numpy as np
import pytest
import torch

from zipsolve.puzzle import Puzzle
from zipsolve.rl import benchmark as B
from zipsolve.rl.env import RewardConfig, ZipEnv
from zipsolve.rl.gnn import ZipGNN, load_model, save_model
from zipsolve.rl.ppo import PPOConfig, PPOTrainer, checkpoint_bonus, gamma_for
from zipsolve.rl.sampling import (DOMAIN_TRAIN, GEN_STATS, MixtureSampler, PuzzlePool, _gen_task,
                                  make_puzzle, make_sampler, puzzle_key, snake_puzzle)

FAMS = ["grid2d:5", "grid3d:3"]


@pytest.fixture(scope="module")
def tiny_bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    B.save_set(B.build_set("val", families=FAMS, per_family=3), root / "val.json")
    return root


def tiny_ckpt(path, hidden=8, layers=1):
    torch.manual_seed(0)
    save_model(ZipGNN(hidden=hidden, layers=layers), path)
    return path


# --------------------------------------------------------------------------- benchmark sets
def test_build_is_deterministic_and_covers_densities():
    a = B.build_set("val", families=FAMS, per_family=3)
    b = B.build_set("val", families=FAMS, per_family=3)
    assert a["fingerprint"] == b["fingerprint"]
    assert [r["puzzle"] for r in a["records"]] == [r["puzzle"] for r in b["records"]]
    assert {r["density"] for r in a["records"]} == {"sparse", "default", "dense"}
    for r in a["records"]:
        p = Puzzle.from_dict(r["puzzle"])
        assert p.is_valid_solution(p.solution)
    # the test split is a different stream
    t = B.build_set("test", families=FAMS, per_family=3)
    assert t["fingerprint"] != a["fingerprint"]


def test_density_changes_checkpoint_count():
    recs = [B.make_record("grid2d:8", "val", i) for i in range(6)]
    by = {}
    for r in recs:
        by.setdefault(r["density"], []).append(r["num_checkpoints"])
    assert np.mean(by["sparse"]) < np.mean(by["default"]) < np.mean(by["dense"])


def test_benchmark_and_training_seeds_are_disjoint():
    val_keys = {tuple(B.make_record("grid2d:5", "val", i)["seed_key"]) for i in range(5)}
    test_keys = {tuple(B.make_record("grid2d:5", "test", i)["seed_key"]) for i in range(5)}
    assert all(k[0] == B.SPLITS["val"][0] for k in val_keys) and all(k[0] == B.SPLITS["test"][0] for k in test_keys)
    assert B.SPLITS["val"][0] != DOMAIN_TRAIN and B.SPLITS["test"][0] != DOMAIN_TRAIN
    s = MixtureSampler(["grid2d:5"], seed=B.SPLITS["val"][0], p_replay=0)   # adversarial seed choice
    task = s._task(s.stage_items[0][0], 0)
    assert task[3][0] == DOMAIN_TRAIN
    # content: training puzzles never equal benchmark puzzles
    bench = {puzzle_key(Puzzle.from_dict(B.make_record("grid2d:5", sp, i)["puzzle"]))
             for sp in ("val", "test") for i in range(6)}
    train = {puzzle_key(s()) for _ in range(30)}
    assert not (bench & train)


def test_exclusion_by_content():
    s0 = MixtureSampler(["grid2d:4"], seed=3, p_replay=0)
    first = s0()
    s1 = MixtureSampler(["grid2d:4"], seed=3, p_replay=0, exclude={puzzle_key(first)})
    p = s1()
    assert puzzle_key(p) != puzzle_key(first) and s1.stats["excluded"] == 1


def test_run_on_tiny_model(tmp_path, tiny_bench):
    ck = tiny_ckpt(tmp_path / "tiny.pt")
    rep = B.run([f"tiny={ck}", f"tiny={ck}"], "val", threads=1, budget=0.3, k=2, out_dir=str(tmp_path / "out"),
                per_family=2, root=str(tiny_bench), progress=False)
    rows = rep["rows"]
    methods = {r["method"] for r in rows}
    assert methods == {"solver", "heur_greedy", "heur_search", "greedy", "sampled@2", "search"}
    assert len(rows) == 4 * (3 + 2 * 3)
    s = rep["summary"]
    assert s["baseline/solver"]["overall"]["solve_rate"] == 1.0
    assert s["tiny/greedy"]["instances"] == 2 and "solve_rate_std" in s["tiny/greedy"]["overall"]
    for r in rows:
        if r["method"] == "greedy":
            assert r["nn_calls"] == r["expansions"] > 0
    md = (tmp_path / "out" / "val_tiny.md").read_text()
    assert "grid3d:3" in md and "sampled@2" in md
    assert json.loads((tmp_path / "out" / "val_tiny.json").read_text())["meta"]["num_puzzles"] == 4


def test_quick_eval_and_heuristics(tiny_bench):
    data = B.load_set(tiny_bench / "val.json")
    torch.manual_seed(0)
    m = ZipGNN(hidden=8, layers=1)
    res = B.quick_eval(m, data["records"], k=2)
    assert set(res["families"]) == set(FAMS) and 0 <= res["greedy"] <= 1 and res["sampled"] is not None
    p = snake_puzzle(4, 4, 2, rng=0)
    ok, steps = B.heuristic_greedy(p)
    assert steps >= 1
    from zipsolve.solver import solve
    r = solve(p, time_limit=2, move_order=B.warnsdorff_order(p))
    assert r.status == "solved" and p.is_valid_solution(r.path)


# --------------------------------------------------------------------------- sampling
def test_replay_mixture_proportions():
    s = MixtureSampler(["grid2d:4", "grid2d:5", "grid2d:6"], seed=0, p_current=0.6, p_earlier=0.2,
                       p_replay=0.2, replay_size=8)
    s.set_stage(2)
    assert s.source_probs() == pytest.approx({"current": 0.75, "earlier": 0.25, "replay": 0.0})  # empty replay
    fail = snake_puzzle(4, 4, 3, rng=0)
    fail._family = "grid2d:4"
    s.episode_end(fail, solved=False)
    assert len(s.replay) == 1
    probs = s.source_probs()
    assert probs == pytest.approx({"current": 0.6, "earlier": 0.2, "replay": 0.2})
    # sample sources without generating puzzles
    src = [["current", "earlier", "replay"][int(s.rng.choice(3, p=list(probs.values())))] for _ in range(4000)]
    frac = {k: src.count(k) / len(src) for k in probs}
    assert frac == pytest.approx(probs, abs=0.03)
    # real draws: earlier-stage puzzles come from stages 0-1 only, replay returns the stored puzzle
    for _ in range(40):
        p = s()
        if p._source == "earlier":
            assert p._family in ("grid2d:4", "grid2d:5")
        elif p._source == "current":
            assert p._family == "grid2d:6"
        else:
            assert p is fail
    # stage 0: no earlier stages -> renormalised
    s.set_stage(0)
    assert s.source_probs() == pytest.approx({"current": 0.75, "earlier": 0.0, "replay": 0.25})
    # solving a replayed puzzle removes it; buffer is bounded
    fail._source = "current"          # a fresh puzzle being solved does not touch the buffer
    s.episode_end(fail, solved=True)
    assert len(s.replay) == 1
    fail._source = "replay"
    s.episode_end(fail, solved=True)
    assert len(s.replay) == 0
    for i in range(20):
        q = snake_puzzle(4, 4, 3, rng=i)
        q._source = "current"
        s.episode_end(q, solved=False)
    assert len(s.replay) == 8


def test_sampler_state_roundtrip():
    s = MixtureSampler(["grid2d:4", "grid2d:5"], seed=1, replay_size=4)
    s.set_stage(1)
    for _ in range(5):
        s.episode_end(s(), solved=False)
    st = s.state_dict()
    nxt = [puzzle_key(s()) for _ in range(5)]
    s2 = MixtureSampler(["grid2d:4", "grid2d:5"], seed=1, replay_size=4)
    s2.load_state_dict(st)
    assert [puzzle_key(s2()) for _ in range(5)] == nxt


def test_generator_failure_is_loud(monkeypatch):
    from zipsolve import generator
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise ValueError("boom")
    monkeypatch.setattr(generator, "make_puzzle", boom)
    before = GEN_STATS["retries"]
    with pytest.raises(RuntimeError):
        make_puzzle("grid2d", 4, rng=0, retries=2)
    assert len(calls) == 3 and GEN_STATS["retries"] - before == 3


def test_puzzle_pool_matches_inprocess():
    task = ("grid2d", 5, None, (DOMAIN_TRAIN, 0, 1, 7), (0.6, 1.6))
    pool = PuzzlePool(1)
    try:
        pool.plan([task])
        p, _ = pool.get(task)
    finally:
        pool.close()
    q, _ = _gen_task(*task)
    assert p.checkpoints == q.checkpoints and p.solution == q.solution and pool.hits == 1


def test_make_sampler_family_streams_differ():
    a = make_sampler("grid2d:4", 0)()
    b = make_sampler("grid4d:2", 0)()
    assert a._family == "grid2d:4" and b._family == "grid4d:2"
    assert puzzle_key(a) != puzzle_key(b)


# --------------------------------------------------------------------------- rewards / gamma
@pytest.mark.parametrize("total", [0.5, 1.0])
def test_checkpoint_reward_normalisation_invariant(total):
    cfg = PPOConfig(cp_reward_total=total)
    sums = []
    for k in (2, 3, 6, 12):
        p = snake_puzzle(5, 5, k, rng=k)
        env = ZipEnv(reward=RewardConfig(checkpoint=0.0))
        env.reset(options={"puzzle": p})
        tot = 0.0
        for a in p.solution[1:]:
            before = env.next_cp
            env.step(a)
            tot += checkpoint_bonus(len(env.cps), env.next_cp - before, cfg)
        sums.append(tot)
    assert sums == pytest.approx([total] * 4)
    assert checkpoint_bonus(5, 1, PPOConfig(cp_reward_total=None)) == 0.0


def test_gamma_auto():
    assert gamma_for(25, PPOConfig(gamma=0.97)) == 0.97
    cfg = PPOConfig(gamma_auto=2.0)
    assert gamma_for(25, cfg) == pytest.approx(0.98)
    # terminal discount from the first move ~ exp(-1/C) at every size
    for n in (16, 49, 81):
        assert gamma_for(n, cfg) ** n == pytest.approx(np.exp(-0.5), abs=0.03)


def test_ppo_logs_new_stats(tmp_path):
    cfg = PPOConfig(num_envs=4, rollout_steps=8, epochs=1, minibatch_size=16, hidden=8, layers=1, gamma_auto=2.0)
    s = MixtureSampler(["grid2d:4"], seed=0)
    tr = PPOTrainer(s, cfg, run_name="st", run_dir=str(tmp_path), ckpt_dir=str(tmp_path), verbose=False)
    for _ in range(2):
        row = tr.train_iteration()
    tr.close()
    for k in ("grad_norm", "grad_norm_max", "explained_var", "entropy_branch", "frac_branch", "t_gen", "t_env",
              "t_forward", "t_collate", "t_update", "family_sr", "replay_size"):
        assert k in row, k
    assert np.isfinite(row["grad_norm"]) and "grid2d:4" in json.loads(row["family_sr"])


# --------------------------------------------------------------------------- checkpoints / resume
def test_resume_restores_state(tmp_path):
    cfg = PPOConfig(num_envs=3, rollout_steps=6, epochs=1, minibatch_size=8, hidden=8, layers=1)
    s = MixtureSampler(["grid2d:4", "grid2d:5"], seed=0, replay_size=16)
    tr = PPOTrainer(s, cfg, run_name="r", run_dir=str(tmp_path), ckpt_dir=str(tmp_path), verbose=False)
    s.set_stage(1)
    for _ in range(3):
        tr.train_iteration()
    path = tr.save("r_latest")
    ref_rand = (torch.rand(3), np.random.rand(3))
    ref_opt = tr.opt.state_dict()
    tr.close()
    model, ck = load_model(path)                     # still a plain load_model checkpoint
    assert "training_state" in ck and ck["iteration"] == 3
    s2 = MixtureSampler(["grid2d:4", "grid2d:5"], seed=0, replay_size=16)
    tr2 = PPOTrainer(s2, cfg, model=model, run_name="r", run_dir=str(tmp_path), ckpt_dir=str(tmp_path),
                     verbose=False, append_log=True)
    tr2.load_training_state(ck["training_state"])
    assert tr2.iteration == 3 and tr2.global_step == tr.global_step and s2.stage == 1
    assert len(s2.replay) == len(s.replay) and s2.counters == s.counters
    o1, o2 = ref_opt["state"], tr2.opt.state_dict()["state"]
    assert o1.keys() == o2.keys() and all(torch.equal(o1[k]["exp_avg"], o2[k]["exp_avg"]) for k in o1)
    assert torch.equal(torch.rand(3), ref_rand[0]) and np.allclose(np.random.rand(3), ref_rand[1])
    tr2.train_iteration()
    tr2.close()
    rows = list(csv.DictReader(open(tmp_path / "r.csv")))
    assert [int(r["iter"]) for r in rows] == [1, 2, 3, 4]


def _train_args(tmp_path, bench_root, *extra):
    return ["--stages", "grid2d:4,grid2d:5", "--num-envs", "3", "--rollout-steps", "6", "--minibatch", "8",
            "--epochs", "1", "--hidden", "8", "--layers", "1", "--val-every", "1", "--val-per-family", "1",
            "--val-k", "2", "--stage-val-per-family", "2", "--min-iters", "1", "--val-threshold", "0",
            "--val-set", str(bench_root / "val.json"), "--run-dir", str(tmp_path / "runs"),
            "--ckpt-dir", str(tmp_path / "ck"), "--threads", "1", "--save-every", "1", *extra]


def test_train_cli_best_and_resume(tmp_path, tiny_bench, capsys):
    from zipsolve.rl.train import main
    main(_train_args(tmp_path, tiny_bench, "--iters", "2", "--run-name", "cli"))
    ck = tmp_path / "ck"
    for name in ("cli_best", "cli_latest", "cli_final", "cli_stage0"):
        assert (ck / f"{name}.pt").exists(), name
    _, meta = load_model(ck / "cli_best.pt")
    assert "val" in meta and meta["training_state"]["iteration"] >= 1
    val_rows = list(csv.DictReader(open(tmp_path / "runs" / "cli_val.csv")))
    assert {r["scope"] for r in val_rows} == {"global", "stage"}
    # stage advanced on validation (threshold 0) and --resume continues the same run
    _, fin = load_model(ck / "cli_final.pt")
    assert fin["curriculum_state"]["stage_i"] == 1
    main(["--resume", str(ck / "cli_latest.pt"), "--iters", "3"])
    _, fin2 = load_model(ck / "cli_final.pt")
    assert fin2["iteration"] == 3 and fin2["curriculum_state"]["stage_i"] == 1
    rows = list(csv.DictReader(open(tmp_path / "runs" / "cli.csv")))
    assert [int(r["iter"]) for r in rows] == [1, 2, 3]
    assert "resumed" in capsys.readouterr().out


def test_train_init_from_pretrained(tmp_path, tiny_bench):
    """--init with a plain gnn.save_model checkpoint (e.g. imitation): hidden/layers come from it."""
    from zipsolve.rl.train import main
    init = tiny_ckpt(tmp_path / "pre.pt", hidden=12, layers=2)
    main(_train_args(tmp_path, tiny_bench, "--iters", "1", "--run-name", "ini", "--val-every", "0",
                     "--init", str(init)))
    m, meta = load_model(tmp_path / "ck" / "ini_final.pt")
    assert meta["config"]["hidden"] == 12 and meta["config"]["layers"] == 2
    assert meta["ppo_config"]["hidden"] == 12
