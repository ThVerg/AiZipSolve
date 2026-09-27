"""Curriculum PPO training of the Zip GNN policy.

Example:
    python -m zipsolve.rl.train --minutes 60 --run-name curric --threads 8
    python -m zipsolve.rl.train --stages "grid2d:4,grid2d:5+walls:5" --minutes 5 --val-every 5
    python -m zipsolve.rl.train --resume checkpoints/curric_latest.pt --minutes 30   # continue
    python -m zipsolve.rl.train --init checkpoints/imitation.pt ...                  # fine-tune weights

Curriculum
    Each stage is a puzzle spec (see zipsolve.rl.sampling); by default the
    stages together cover every benchmark family. Every sampled training
    puzzle comes from: the current stage (--p-current, default 0.6), the
    earlier stages (--p-earlier, 0.2; stages are retained after they are
    passed) or a bounded replay buffer of previously *failed* training
    puzzles (--p-replay, 0.2; --replay-size). Replay only supplies start
    puzzles for fresh on-policy rollouts. Unavailable sources are
    renormalised away. Checkpoint density varies per puzzle: the generator's
    default count x U(--density-min, --density-max).

Advancing
    --advance-on val (default): after >= --min-iters iterations, when the
    greedy solve rate on held-out validation puzzles of the current stage's
    families (--stage-val-per-family each, from the frozen benchmark val set
    where it has the family, else generated from the validation seed domain)
    reaches --val-threshold. --advance-on train: rolling training solve rate
    (last --window episodes) >= --threshold. both: both. --max-iters-per-stage
    forces advancing. After the last stage training continues on the full mix.

Validation / checkpoints
    Every --val-every iterations: greedy (+ sampled@--val-k) solve rate per
    family on the first --val-per-family puzzles of each benchmark val family
    (runs/<run>_val.csv, long format). checkpoints/<run>_best.pt keeps the best
    macro-averaged validation score (--best-metric); <run>_latest.pt every
    --save-every iterations; <run>_final.pt at the end; <run>_stageK.pt when
    stage K is passed. All of them are gnn.load_model-compatible and hold the
    full training state, so --resume continues exactly (optimizer, iteration,
    stage, replay buffer, sampler + python/numpy/torch RNG states; in-flight
    episodes restart).

Rewards / discount
    --cp-reward-total T (default 0.5): reaching all checkpoints is worth T in
    total, whatever their number (--cp-reward-total 0 -> per-checkpoint
    --r-checkpoint instead). --gamma-auto C: gamma = 1 - 1/(C*n) per n-node
    puzzle (terminal reward discounted by ~exp(-1/C) from the first move at
    every size); default off (--gamma, 0.99).

Profiling
    Seconds per iteration spent in generation, env stepping/observation
    construction, collation, forward passes and the update are always logged
    (t_* columns); --profile prints the breakdown. --gen-workers N prefetches
    puzzles in N background processes (identical puzzles, just earlier).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .env import RewardConfig
from .gnn import load_model
from .ppo import PPOConfig, PPOTrainer
from .sampling import MixtureSampler, PuzzlePool, spec_families

DEFAULT_STAGES = ("grid2d:4+grid3d:2+grid4d:2,"
                  "grid2d:5+walls:5+mask:6+islands:2+grid3d:3,"
                  "grid2d:6+walls:6+mask:7+islands:3+islands_chain:3,"
                  "grid2d:7+walls:7+mask:8+islands:4+grid3d:4+grid4d:3,"
                  "grid2d:8+islands:5+islands_chain:3+grid4d:3")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--stages", default=DEFAULT_STAGES, help="comma-separated curriculum stage specs")
    a("--threshold", type=float, default=0.8, help="training solve rate to advance (--advance-on train/both)")
    a("--window", type=int, default=300, help="episodes in the rolling training solve rate")
    a("--min-iters", type=int, default=10, help="min iterations per stage")
    a("--max-iters-per-stage", type=int, default=0, help="0 = unlimited")
    a("--advance-on", choices=["val", "train", "both"], default="val")
    a("--minutes", type=float, default=15.0, help="wall-clock budget (this session)")
    a("--iters", type=int, default=0, help="max total iterations (0 = unlimited)")
    a("--run-name", default="zip_ppo")
    a("--run-dir", default="runs")
    a("--ckpt-dir", default="checkpoints")
    a("--init", default=None, help="checkpoint to take the weights from (e.g. supervised pretraining)")
    a("--resume", default=None, help="checkpoint to resume training from (full state)")
    a("--threads", type=int, default=4, help="torch intra-op threads")
    a("--seed", type=int, default=0)
    a("--save-every", type=int, default=20)
    # mixture / replay / density
    a("--p-current", type=float, default=0.6)
    a("--p-earlier", type=float, default=0.2)
    a("--p-replay", type=float, default=0.2)
    a("--replay-size", type=int, default=256, help="failed puzzles kept for replay (0 = off)")
    a("--density-min", type=float, default=0.6, help="checkpoint density factor range (x generator default)")
    a("--density-max", type=float, default=1.6)
    a("--no-density", action="store_true", default=False, help="generator default checkpoint counts only")
    a("--no-exclude-bench", action="store_true", default=False,
      help="do not filter benchmark puzzles out of training (by content)")
    a("--gen-workers", type=int, default=0, help="background puzzle-generation processes")
    # validation
    a("--val-every", type=int, default=10, help="validate every N iterations (0 = off)")
    a("--val-set", default="benchmarks/val.json")
    a("--val-per-family", type=int, default=3)
    a("--val-families", default=None, help="comma-separated subset of val families (default: all)")
    a("--val-k", type=int, default=4, help="sampled attempts in validation (0 = greedy only)")
    a("--stage-val-per-family", type=int, default=12)
    a("--val-threshold", type=float, default=None, help="default: --threshold")
    a("--best-metric", choices=["greedy", "sampled"], default="greedy")
    # PPO
    a("--num-envs", type=int, default=32)
    a("--rollout-steps", type=int, default=32)
    a("--epochs", type=int, default=4)
    a("--minibatch", type=int, default=256)
    a("--lr", type=float, default=3e-4)
    a("--gamma", type=float, default=0.99)
    a("--gamma-auto", type=float, default=0.0, help="C > 0: gamma = 1 - 1/(C*n) per puzzle")
    a("--gae-lambda", type=float, default=0.95)
    a("--ent-coef", type=float, default=0.01)
    a("--target-kl", type=float, default=0.03)
    a("--hidden", type=int, default=64)
    a("--layers", type=int, default=6)
    a("--model-kw", default="{}", help='extra ZipGNN kwargs as JSON, e.g. \'{"feature_version": 2}\'')
    a("--no-early-termination", action="store_true", default=False)
    # rewards
    a("--r-solve", type=float, default=1.0)
    a("--r-fail", type=float, default=-1.0)
    a("--r-checkpoint", type=float, default=0.1, help="per checkpoint (only with --cp-reward-total 0)")
    a("--cp-reward-total", type=float, default=0.5, help="total checkpoint reward (0 = per-checkpoint mode)")
    a("--r-step", type=float, default=0.0)
    a("--r-progress", type=float, default=0.5)
    a("--profile", action="store_true", default=False)
    return p


def parse_args(argv=None) -> tuple[argparse.Namespace, dict | None]:
    """Parse CLI; with --resume the saved run arguments are the defaults and only
    options given explicitly on the command line override them."""
    args = build_parser().parse_args(argv)
    if not args.resume:
        return args, None
    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    saved = ck.get("train_args")
    if saved is None or "training_state" not in ck:
        raise SystemExit(f"{args.resume} has no training state; use --init to load weights only")
    probe = build_parser()
    for act in probe._actions:
        act.default = argparse.SUPPRESS
    explicit = vars(probe.parse_args(argv))
    merged = dict(saved)
    merged.update(explicit)
    merged["resume"] = args.resume
    merged["init"] = None
    return argparse.Namespace(**{**vars(args), **merged}), ck


def make_config(args) -> PPOConfig:
    return PPOConfig(
        num_envs=args.num_envs, rollout_steps=args.rollout_steps, epochs=args.epochs,
        minibatch_size=args.minibatch, lr=args.lr, gamma=args.gamma, gamma_auto=args.gamma_auto,
        gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
        target_kl=args.target_kl if args.target_kl > 0 else None,
        hidden=args.hidden, layers=args.layers, early_termination=not args.no_early_termination,
        reward=RewardConfig(args.r_solve, args.r_fail, args.r_checkpoint, args.r_step, args.r_progress),
        cp_reward_total=args.cp_reward_total if args.cp_reward_total > 0 else None,
        seed=args.seed,
    )


class Validator:
    """Held-out validation puzzles (frozen benchmark val set + generated val-domain families)."""

    def __init__(self, args):
        from .benchmark import load_set, make_record, subset
        self._make_record = make_record
        try:   # built on the fly (deterministically) if the default file is missing
            self.file_records: list[dict] = load_set(Path(args.val_set))["records"]
        except FileNotFoundError:
            self.file_records = []
        fams = args.val_families.split(",") if args.val_families else None
        self.global_records = subset(self.file_records, args.val_per_family, fams)
        self.k = args.val_k
        self.stage_n = args.stage_val_per_family
        self._stage_cache: dict[str, list[dict]] = {}

    def stage_records(self, spec: str) -> list[dict]:
        out = []
        for fam in spec_families(spec):
            if fam not in self._stage_cache:
                recs = [r for r in self.file_records if r["family"] == fam][:self.stage_n]
                i = len(recs)
                from ..puzzle import Puzzle
                while len(recs) < self.stage_n:     # families not in the file: same val seed domain
                    r = self._make_record(fam, "val", i)
                    r["_puzzle"] = Puzzle.from_dict(r["puzzle"])
                    recs.append(r)
                    i += 1
                self._stage_cache[fam] = recs
            out += self._stage_cache[fam]
        return out

    def run(self, model, records, k=None, meta=None) -> dict:
        from .benchmark import quick_eval
        model.eval()
        return quick_eval(model, records, self.k if k is None else k, meta=meta)


def main(argv=None):
    args, resume_ck = parse_args(argv)
    torch.set_num_threads(args.threads)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    cfg = make_config(args)
    val_thr = args.val_threshold if args.val_threshold is not None else args.threshold
    # model: --resume > --init > fresh
    model, meta = None, None
    if resume_ck is not None:
        model, meta = load_model(args.resume)
    elif args.init:
        model, meta = load_model(args.init)
        print(f"init weights from {args.init} (config {meta.get('config')})")
    exclude = set()
    if not args.no_exclude_bench:
        from .benchmark import benchmark_keys
        exclude = benchmark_keys()
    pool = PuzzlePool(args.gen_workers) if args.gen_workers > 0 else None
    sampler = MixtureSampler(stages, seed=args.seed, p_current=args.p_current, p_earlier=args.p_earlier,
                             p_replay=args.p_replay, replay_size=args.replay_size,
                             density=None if args.no_density else (args.density_min, args.density_max),
                             exclude=exclude, pool=pool)
    trainer = PPOTrainer(sampler, cfg, model=model, run_name=args.run_name, run_dir=args.run_dir,
                         ckpt_dir=args.ckpt_dir, model_meta=meta, append_log=resume_ck is not None,
                         window=args.window, model_kw=json.loads(args.model_kw or "{}"))
    cur = {"stage_i": 0, "stage_iters": 0, "history": [], "best_score": float("-inf"), "best_iter": 0,
           "last_stage_val": None}
    if resume_ck is not None:
        trainer.load_training_state(resume_ck["training_state"])
        cur.update(resume_ck.get("curriculum_state", {}))
        print(f"resumed {args.resume}: iteration {trainer.iteration}, stage {cur['stage_i']} "
              f"({stages[cur['stage_i']]}), replay {len(sampler.replay)}")
    sampler.set_stage(cur["stage_i"])
    trainer.stage = stages[cur["stage_i"]]
    print(f"stages: {stages}")
    print(f"benchmark puzzles excluded from training: {len(exclude)}")

    validator = Validator(args) if args.val_every > 0 else None
    val_path = Path(args.run_dir) / f"{args.run_name}_val.csv"
    val_file = open(val_path, "a" if resume_ck is not None and val_path.exists() else "w", newline="")
    val_csv = csv.writer(val_file)
    if val_file.tell() == 0:
        val_csv.writerow(["iter", "step", "time", "stage", "scope", "family", "n", "greedy", "sampled"])

    def save(name, **extra):
        return trainer.save(name, train_args=vars(args), curriculum_state=dict(cur), stages=stages, **extra)

    def log_val(scope, res):
        for fam, f in res["families"].items():
            val_csv.writerow([trainer.iteration, trainer.global_step, round(trainer.elapsed(), 1),
                              trainer.stage, scope, fam, f["n"], round(f["greedy"], 4),
                              "" if f["sampled"] is None else round(f["sampled"], 4)])
        val_csv.writerow([trainer.iteration, trainer.global_step, round(trainer.elapsed(), 1), trainer.stage,
                          scope, "_macro", sum(f["n"] for f in res["families"].values()),
                          round(res["greedy"], 4), "" if res["sampled"] is None else round(res["sampled"], 4)])
        val_file.flush()

    prof_tot: dict[str, float] = {}
    deadline = time.time() + args.minutes * 60
    while time.time() < deadline and (args.iters == 0 or trainer.iteration < args.iters):
        row = trainer.train_iteration()
        cur["stage_iters"] += 1
        for k, v in row.items():
            if k.startswith("t_"):
                prof_tot[k] = prof_tot.get(k, 0.0) + v
        if args.profile and trainer.iteration % trainer.log_every == 0:
            tt = {k[2:]: v for k, v in row.items() if k.startswith("t_") and k != "t_update_collate"}
            tot = sum(tt.values()) or 1.0
            print("  profile: " + " ".join(f"{k}={v:.2f}s({100 * v / tot:.0f}%)" for k, v in tt.items())
                  + f" update_collate={row.get('t_update_collate', 0):.2f}s", flush=True)
        # ---- validation
        stage_val_ok = None
        if validator is not None and trainer.iteration % args.val_every == 0:
            if validator.global_records:
                res = validator.run(trainer.model, validator.global_records, meta=meta)
                log_val("global", res)
                score = res["sampled"] if args.best_metric == "sampled" and res["sampled"] is not None \
                    else res["greedy"]
                msg = (f"  val[{res['seconds']}s]: greedy={res['greedy']:.3f}"
                       + (f" sampled@{validator.k}={res['sampled']:.3f}" if res["sampled"] is not None else "")
                       + " " + " ".join(f"{f}={v['greedy']:.2f}" for f, v in res["families"].items()))
                if score > cur["best_score"]:
                    cur["best_score"], cur["best_iter"] = score, trainer.iteration
                    save(f"{args.run_name}_best", val=res)
                    msg += f"  -> new best {score:.3f}"
                print(msg, flush=True)
            if args.advance_on in ("val", "both"):
                sres = validator.run(trainer.model, validator.stage_records(stages[cur["stage_i"]]), k=0,
                                     meta=meta)
                log_val("stage", sres)
                cur["last_stage_val"] = sres["greedy"]
                stage_val_ok = sres["greedy"] >= val_thr
                print(f"  stage val ({stages[cur['stage_i']]}): greedy={sres['greedy']:.3f} "
                      f"(threshold {val_thr})", flush=True)
        # ---- stage advance
        full = len(trainer.recent) >= args.window
        rate = trainer.solve_rate()
        train_ok = full and rate >= args.threshold
        mode = args.advance_on if validator is not None else "train"
        if mode == "train":
            ok = train_ok
        elif mode == "val":
            ok = bool(stage_val_ok)
        else:
            ok = train_ok and bool(stage_val_ok)
        passed = ok and cur["stage_iters"] >= args.min_iters
        forced = bool(args.max_iters_per_stage and cur["stage_iters"] >= args.max_iters_per_stage)
        si = cur["stage_i"]
        if (passed or forced) and si < len(stages) - 1:
            cur["history"].append({"stage": stages[si], "iters": cur["stage_iters"], "solve_rate": rate,
                                   "stage_val": cur["last_stage_val"], "passed": bool(passed),
                                   "time": row["time"], "step": row["step"]})
            path = save(f"{args.run_name}_stage{si}")
            print(f"=== stage {si} ({stages[si]}) done: train solve_rate={rate:.3f} "
                  f"stage_val={cur['last_stage_val']} after {cur['stage_iters']} iters; saved {path}", flush=True)
            cur["stage_i"] = si + 1
            cur["stage_iters"] = 0
            cur["last_stage_val"] = None
            sampler.set_stage(cur["stage_i"])
            trainer.set_sampler(sampler, stages[cur["stage_i"]], reset_envs=False)
        if trainer.iteration % args.save_every == 0:
            save(f"{args.run_name}_latest")
    si = cur["stage_i"]
    final_hist = cur["history"] + [{"stage": stages[si], "iters": cur["stage_iters"],
                                    "solve_rate": trainer.solve_rate(), "stage_val": cur["last_stage_val"],
                                    "passed": None, "time": round(trainer.elapsed(), 1),
                                    "step": trainer.global_step}]
    path = save(f"{args.run_name}_final", curriculum=final_hist)
    save(f"{args.run_name}_latest", curriculum=final_hist)
    trainer.close()
    sampler.close()
    val_file.close()
    print("curriculum summary:")
    print(json.dumps(final_hist, indent=1))
    if args.profile and prof_tot:
        tot = sum(v for k, v in prof_tot.items() if k != "t_update_collate") or 1.0
        print("profile totals: " + " ".join(f"{k[2:]}={v:.1f}s({100 * v / tot:.0f}%)"
                                            for k, v in sorted(prof_tot.items()) if k != "t_update_collate")
              + f" (update_collate={prof_tot.get('t_update_collate', 0):.1f}s, part of update)")
    print(f"sampler: {dict(sampler.stats)}; best val {cur['best_score']:.3f} at iter {cur['best_iter']}")
    print(f"final checkpoint: {path}")
    print(f"log: {trainer.csv_path}  validation log: {val_path}")


if __name__ == "__main__":
    main()
