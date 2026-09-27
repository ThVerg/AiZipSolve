"""Quick A/B of observation feature versions / policy heads with a short PPO run.

    python scripts/feature_ab.py --feature-version 1 --minutes 6 --name ab_v1
    python scripts/feature_ab.py --feature-version 2 --minutes 6 --name ab_v2
    python scripts/feature_ab.py --feature-version 2 --global-attn --name ab_v2attn

Trains with PPOTrainer (env built to match the model via make_env_for_model),
then reports the greedy solve rate on a fixed held-out puzzle set.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from zipsolve.rl.gnn import collate, make_env_for_model, make_model
from zipsolve.rl.ppo import PPOConfig, PPOTrainer
from zipsolve.rl.sampling import make_sampler


@torch.no_grad()
def greedy_solve_rate(model, spec: str, episodes: int, seed: int) -> float:
    model.eval()
    samp = make_sampler(spec, seed)
    puzzles = [samp() for _ in range(episodes)]
    envs = [make_env_for_model(model, early_termination=True) for _ in puzzles]
    obs = [e.reset(options={"puzzle": p})[0] for e, p in zip(envs, puzzles)]
    alive = list(range(len(envs)))
    solved = [False] * len(envs)
    while alive:
        logits, _ = model(collate([obs[i] for i in alive]))
        gb_ptr = np.cumsum([0] + [obs[i]["x"].shape[0] for i in alive])
        nxt = []
        for k, i in enumerate(alive):
            a = int(torch.argmax(logits[gb_ptr[k]:gb_ptr[k + 1]]))
            obs[i], _, term, _, info = envs[i].step(a)
            if term:
                solved[i] = bool(info.get("solved"))
            else:
                nxt.append(i)
        alive = nxt
    return float(np.mean(solved))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-version", type=int, default=2)
    ap.add_argument("--head-version", type=int, default=None, help="default: 1 for v1, 2 for v2")
    ap.add_argument("--global-attn", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--spec", default="grid2d:5+islands:3")
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=300)
    ap.add_argument("--name", default="ab")
    ap.add_argument("--out-dir", default="runs/feature_ab")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    hv = args.head_version or (1 if args.feature_version == 1 else 2)
    model = make_model(hidden=args.hidden, layers=args.layers, feature_version=args.feature_version,
                       head_version=hv, global_attn=args.global_attn)
    cfg = PPOConfig(num_envs=32, rollout_steps=32, hidden=args.hidden, layers=args.layers, seed=args.seed)
    tr = PPOTrainer(make_sampler(args.spec, args.seed), cfg, model=model, run_name=args.name,
                    run_dir=args.out_dir, ckpt_dir=args.out_dir, verbose=False,
                    model_meta={"config": model.config})
    assert tr.envs[0].feature_version == args.feature_version
    t0 = time.time()
    rows = []
    while time.time() - t0 < args.minutes * 60:
        row = tr.train_iteration()
        rows.append(row)
        if row["iter"] % 10 == 0:
            print(f"[{args.name}] iter={row['iter']} step={row['step']} t={row['time']} "
                  f"solve_rate={row['solve_rate']}", flush=True)
    path = tr.save(f"{args.name}_final")
    tr.close()
    res = {"name": args.name, "feature_version": args.feature_version, "head_version": hv,
           "global_attn": args.global_attn, "hidden": args.hidden, "layers": args.layers,
           "iters": tr.iteration, "steps": tr.global_step, "train_minutes": round((time.time() - t0) / 60, 2),
           "train_solve_rate_last200": rows[-1]["solve_rate"] if rows else None,
           "greedy_eval": {s: greedy_solve_rate(tr.model, s, args.eval_episodes, 999)
                           for s in ["grid2d:5", "islands:3"]},
           "checkpoint": str(path)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
