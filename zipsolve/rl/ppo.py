"""PPO (clipped, GAE) for variable-size graph observations.

Environments are a plain list of ZipEnv stepped in one process; each step all
envs' observations are collated into one disjoint-union graph for a single
batched forward pass. The action distribution is a masked categorical over the
nodes of each graph (padded to the largest graph in the batch).

Strictly on-policy: every iteration collects fresh transitions with the current
policy and discards them after the update. A sampler may hand out *puzzles*
seen before (replay of failed puzzles), but only as start states of new
rollouts; stale transitions are never reused.

Reward options (on top of env.RewardConfig):
  * ``cp_reward_total`` (default 0.5, None = off): checkpoint reward is
    normalised so that reaching all checkpoints is worth this total, whatever
    the number of checkpoints (each of the K-1 checkpoints after the start
    gives total/(K-1)); the env's own per-checkpoint reward is disabled. With
    None the env's constant ``RewardConfig.checkpoint`` per checkpoint is used.
  * ``gamma_auto`` (default 0 = off): per-puzzle discount
    ``gamma = 1 - 1/(gamma_auto * n)`` for an n-node puzzle, so the terminal
    reward seen from the first move is discounted by ~exp(-1/gamma_auto)
    whatever the puzzle size (e.g. 2.0 -> ~0.61; n=25: 0.98, n=81: 0.9938).
    Otherwise the fixed ``gamma`` is used.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .env import RewardConfig, ZipEnv
from .gnn import ZipGNN, collate, masked_distribution, save_model
from .sampling import GEN_STATS, rng_state, set_rng_state


@dataclass
class PPOConfig:
    num_envs: int = 32
    rollout_steps: int = 32          # steps per env per iteration
    epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gamma_auto: float = 0.0          # >0: gamma = 1 - 1/(gamma_auto * n) per puzzle
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    hidden: int = 64
    layers: int = 6
    early_termination: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    cp_reward_total: float | None = 0.5   # None: env's per-checkpoint reward
    seed: int = 0


def gamma_for(n: int, cfg: PPOConfig) -> float:
    """Discount for an n-node puzzle (see module docstring)."""
    if cfg.gamma_auto and cfg.gamma_auto > 0:
        return float(1.0 - 1.0 / (cfg.gamma_auto * max(2, n)))
    return float(cfg.gamma)


def checkpoint_bonus(num_checkpoints: int, reached: int, cfg: PPOConfig) -> float:
    """Normalised checkpoint reward for `reached` new checkpoints (0 if disabled)."""
    if cfg.cp_reward_total is None or reached <= 0:
        return 0.0
    return float(cfg.cp_reward_total) * reached / max(1, num_checkpoints - 1)


def make_env(model=None, meta: dict | None = None, sampler=None, reward: RewardConfig | None = None,
             early_termination: bool = True) -> ZipEnv:
    """A ZipEnv whose observations match `model` / checkpoint `meta` (feature
    version): gnn.make_env_for_model if available, else evaluate.make_env_for,
    else the default ZipEnv."""
    from . import gnn as _gnn
    env = None
    fn = getattr(_gnn, "make_env_for_model", None)
    if fn is not None:
        env = fn(model, meta)
    else:
        try:
            from .evaluate import make_env_for
            env = make_env_for(model, meta)
        except Exception:  # noqa: BLE001
            env = None
    if env is None:
        env = ZipEnv()
    env.puzzle_sampler = sampler
    env.reward_cfg = reward or RewardConfig()
    env.early_termination = early_termination
    return env


class Timer:
    """Accumulates wall time per named section."""

    def __init__(self):
        self.t: Counter = Counter()

    def add(self, k: str, dt: float):
        self.t[k] += dt

    def pop(self) -> dict:
        out, self.t = dict(self.t), Counter()
        return out


class PPOTrainer:
    def __init__(self, sampler: Callable, cfg: PPOConfig | None = None, model: ZipGNN | None = None,
                 run_name: str = "ppo", run_dir: str = "runs", ckpt_dir: str = "checkpoints",
                 log_every: int = 1, verbose: bool = True, model_meta: dict | None = None,
                 append_log: bool = False, window: int = 400, model_kw: dict | None = None):
        self.cfg = cfg = cfg or PPOConfig()
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.model = model or ZipGNN(hidden=cfg.hidden, layers=cfg.layers, **(model_kw or {}))
        mc = getattr(self.model, "config", {}) or {}
        cfg.hidden, cfg.layers = mc.get("hidden", cfg.hidden), mc.get("layers", cfg.layers)
        self.model_meta = model_meta
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        env_reward = cfg.reward if cfg.cp_reward_total is None else dataclasses.replace(cfg.reward, checkpoint=0.0)
        self.sampler = sampler
        self.envs = [make_env(self.model, model_meta, sampler, env_reward, cfg.early_termination)
                     for _ in range(cfg.num_envs)]
        self.timer = Timer()
        self.obs = [self._reset(e) for e in self.envs]
        self.ep_ret = np.zeros(cfg.num_envs)
        self.ep_len = np.zeros(cfg.num_envs, dtype=int)
        self.recent = deque(maxlen=window)   # (solved, return, length, frac_visited, family)
        self.global_step = 0
        self.iteration = 0
        self.stage = ""
        self.run_name = run_name
        self.verbose = verbose
        self.log_every = log_every
        self.skipped_updates = 0
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(run_dir) / f"{run_name}.csv"
        self._fieldnames = None
        if append_log and self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with open(self.csv_path, newline="") as f:
                self._fieldnames = next(csv.reader(f))
            self._csv_file = open(self.csv_path, "a", newline="")
        else:
            self._csv_file = open(self.csv_path, "w", newline="")
        self._csv = None
        self.t0 = time.time()
        self.elapsed_before = 0.0   # wall time of earlier (resumed) sessions

    # ------------------------------------------------------------------ utils
    def _gen_seconds(self) -> float:
        return float(getattr(self.sampler, "gen_seconds", 0.0))

    def _reset(self, env: ZipEnv) -> dict:
        t0, g0 = time.perf_counter(), self._gen_seconds()
        o = env.reset()[0]
        dg = self._gen_seconds() - g0
        self.timer.add("gen", dg)
        self.timer.add("env", time.perf_counter() - t0 - dg)
        return o

    def set_sampler(self, sampler: Callable, stage: str = "", reset_envs: bool = True):
        self.sampler = sampler
        for e in self.envs:
            e.puzzle_sampler = sampler
        self.stage = stage
        self.recent.clear()
        if reset_envs:
            self.reset_envs()

    def reset_envs(self):
        self.obs = [self._reset(e) for e in self.envs]
        self.ep_ret[:] = 0
        self.ep_len[:] = 0

    def solve_rate(self, last: int | None = None) -> float:
        items = list(self.recent)[-last:] if last else list(self.recent)
        return float(np.mean([it[0] for it in items])) if items else 0.0

    def family_solve_rates(self, last: int | None = None) -> dict[str, tuple[float, int]]:
        items = list(self.recent)[-last:] if last else list(self.recent)
        by: dict[str, list] = {}
        for it in items:
            by.setdefault(it[4], []).append(it[0])
        return {k: (float(np.mean(v)), len(v)) for k, v in sorted(by.items())}

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect(self):
        cfg = self.cfg
        T, N = cfg.rollout_steps, cfg.num_envs
        obs_buf, act_buf = [], np.zeros((T, N), dtype=np.int64)
        logp_buf = np.zeros((T, N), dtype=np.float32)
        val_buf = np.zeros((T + 1, N), dtype=np.float32)
        rew_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)
        gam_buf = np.zeros((T, N), dtype=np.float32)
        nlegal_buf = np.zeros((T, N), dtype=np.int32)
        self.model.eval()
        ep_end = getattr(self.sampler, "episode_end", None)
        for t in range(T):
            t0 = time.perf_counter()
            gb = collate(self.obs)
            t1 = time.perf_counter()
            logits, values = self.model(gb)
            dist = masked_distribution(logits, gb)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            t2 = time.perf_counter()
            self.timer.add("collate", t1 - t0)
            self.timer.add("forward", t2 - t1)
            obs_buf.append(self.obs)
            act_buf[t] = actions.numpy()
            logp_buf[t] = logp.numpy()
            val_buf[t] = values.numpy()
            new_obs = []
            for i, env in enumerate(self.envs):
                nlegal_buf[t, i] = int(np.count_nonzero(self.obs[i]["action_mask"]))
                gam_buf[t, i] = gamma_for(env.n, cfg)
                ts = time.perf_counter()
                cp_before = env.next_cp
                o, r, term, trunc, info = env.step(int(actions[i]))
                r += checkpoint_bonus(len(env.cps), env.next_cp - cp_before, cfg)
                self.timer.add("env", time.perf_counter() - ts)
                rew_buf[t, i] = r
                self.ep_ret[i] += r
                self.ep_len[i] += 1
                if term or trunc:
                    done_buf[t, i] = 1.0
                    solved = bool(info.get("solved"))
                    fam = getattr(env.puzzle, "_family", env.puzzle.graph.kind)
                    self.recent.append((solved, self.ep_ret[i], self.ep_len[i], len(info["path"]) / env.n, fam))
                    if ep_end is not None:
                        ep_end(env.puzzle, solved)
                    self.ep_ret[i] = 0
                    self.ep_len[i] = 0
                    o = self._reset(env)
                new_obs.append(o)
            self.obs = new_obs
            self.global_step += N
        t0 = time.perf_counter()
        gb = collate(self.obs)
        t1 = time.perf_counter()
        _, last_v = self.model(gb)
        self.timer.add("collate", t1 - t0)
        self.timer.add("forward", time.perf_counter() - t1)
        val_buf[T] = last_v.numpy()
        # GAE (episodes never truncate: they end solved or dead within n-1 moves);
        # the discount is per step (per puzzle with gamma_auto)
        adv = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - done_buf[t]
            g = gam_buf[t]
            delta = rew_buf[t] + g * val_buf[t + 1] * nonterm - val_buf[t]
            last = delta + g * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + val_buf[:T]
        flat_obs = [obs_buf[t][i] for t in range(T) for i in range(N)]
        return flat_obs, act_buf.reshape(-1), logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1), \
            val_buf[:T].reshape(-1), nlegal_buf.reshape(-1)

    # ------------------------------------------------------------------ update
    def update(self, flat_obs, actions, old_logp, adv, ret, old_v, nlegal=None):
        cfg = self.cfg
        self.model.train()
        M = len(flat_obs)
        actions = torch.from_numpy(actions)
        old_logp = torch.from_numpy(old_logp)
        adv_t = torch.from_numpy(adv)
        # normalise over the whole rollout once (population std: finite even for 1 sample),
        # so a small tail minibatch neither yields NaN nor gets its own noisy scale
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        ret_t = torch.from_numpy(ret)
        old_v_t = torch.from_numpy(old_v)
        branch = torch.from_numpy(np.asarray(nlegal) >= 2) if nlegal is not None else None
        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": [], "grad_norm": []}
        ent_branch_sum, ent_branch_cnt = 0.0, 0
        stop = False
        for _ in range(cfg.epochs):
            perm = np.random.permutation(M)
            for s in range(0, M, cfg.minibatch_size):
                idx = perm[s:s + cfg.minibatch_size]
                t0 = time.perf_counter()
                gb = collate([flat_obs[j] for j in idx])
                self.timer.add("update_collate", time.perf_counter() - t0)
                logits, values = self.model(gb)
                dist = masked_distribution(logits, gb)
                it = torch.from_numpy(idx)
                logp = dist.log_prob(actions[it])
                ent_all = dist.entropy()
                ent = ent_all.mean()
                a = adv_t[it]
                ratio = torch.exp(logp - old_logp[it])
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v_clip = old_v_t[it] + (values - old_v_t[it]).clamp(-cfg.clip, cfg.clip)
                v_loss = 0.5 * torch.max((values - ret_t[it]) ** 2, (v_clip - ret_t[it]) ** 2).mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                if not torch.isfinite(loss):
                    # never let a bad batch write non-finite weights
                    self.skipped_updates += 1
                    continue
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)  # pre-clip norm
                if not torch.isfinite(gn):
                    self.skipped_updates += 1
                    self.opt.zero_grad()
                    continue
                self.opt.step()
                with torch.no_grad():
                    lr_ = logp - old_logp[it]
                    kl = ((torch.exp(lr_) - 1) - lr_).mean().item()
                    if branch is not None:
                        bm = branch[it]
                        ent_branch_sum += float(ent_all[bm].sum())
                        ent_branch_cnt += int(bm.sum())
                stats["pg_loss"].append(pg.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(ent.item())
                stats["kl"].append(kl)
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                stats["grad_norm"].append(float(gn))
                if cfg.target_kl is not None and kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
        out["grad_norm_max"] = float(np.max(stats["grad_norm"])) if stats["grad_norm"] else float("nan")
        out["entropy_branch"] = ent_branch_sum / ent_branch_cnt if ent_branch_cnt else float("nan")
        if branch is not None:
            out["frac_branch"] = float(branch.float().mean())
        return out

    # ------------------------------------------------------------------ loop
    def train_iteration(self) -> dict:
        t0 = time.perf_counter()
        data = self.collect()
        t1 = time.perf_counter()
        stats = self.update(*data)
        t2 = time.perf_counter()
        ret, old_v = data[4], data[5]
        var_r = float(np.var(ret))
        ev = float(1.0 - np.var(ret - old_v) / var_r) if var_r > 1e-12 else float("nan")
        self.iteration += 1
        tm = self.timer.pop()
        rollout = t1 - t0
        tm["policy_other"] = max(0.0, rollout - sum(tm.get(k, 0.0) for k in ("gen", "env", "collate", "forward")))
        tm["update"] = t2 - t1
        rec = list(self.recent)[-200:]
        smp = self.sampler
        row = {
            "iter": self.iteration, "step": self.global_step,
            "time": round(self.elapsed(), 1), "stage": self.stage,
            "solve_rate": round(self.solve_rate(200), 4),
            "ep_return": round(float(np.mean([r[1] for r in rec])) if rec else 0.0, 4),
            "ep_len": round(float(np.mean([r[2] for r in rec])) if rec else 0.0, 2),
            "frac_visited": round(float(np.mean([r[3] for r in rec])) if rec else 0.0, 4),
            "episodes": len(self.recent),
            **{k: round(v, 5) for k, v in stats.items()},
            "explained_var": round(ev, 4),
            "skipped_updates": self.skipped_updates,
            "gen_retries": GEN_STATS.get("retries", 0),
            "replay_size": len(getattr(smp, "replay", ())),
            "src_counts": json.dumps(dict(getattr(smp, "stats", {})), sort_keys=True),
            "family_sr": json.dumps({k: [round(v, 3), c] for k, (v, c) in self.family_solve_rates(200).items()}),
            **{f"t_{k}": round(v, 3) for k, v in sorted(tm.items())},
        }
        self._log(row)
        return row

    def elapsed(self) -> float:
        return self.elapsed_before + (time.time() - self.t0)

    def _log(self, row: dict):
        if self._csv is None:
            fields = self._fieldnames or list(row.keys())
            self._csv = csv.DictWriter(self._csv_file, fieldnames=fields, extrasaction="ignore", restval="")
            if self._fieldnames is None:
                self._csv.writeheader()
        self._csv.writerow(row)
        self._csv_file.flush()
        if self.verbose and self.iteration % self.log_every == 0:
            skip = {"src_counts", "family_sr"}
            print(" ".join(f"{k}={v}" for k, v in row.items() if k not in skip and not k.startswith("t_")),
                  flush=True)

    # ------------------------------------------------------------------ checkpoints
    def training_state(self) -> dict:
        """Everything needed to resume (optimizer, counters, RNGs, sampler incl. replay)."""
        st = {
            "optimizer": self.opt.state_dict(), "iteration": self.iteration,
            "global_step": self.global_step, "stage": self.stage, "rng": rng_state(),
            "recent": list(self.recent), "elapsed": self.elapsed(),
            "skipped_updates": self.skipped_updates,
        }
        if hasattr(self.sampler, "state_dict"):
            st["sampler"] = self.sampler.state_dict()
        # in-flight puzzles: restarted from their first move on resume
        st["env_puzzles"] = [{"puzzle": e.puzzle.to_dict(), "family": getattr(e.puzzle, "_family", None),
                              "source": getattr(e.puzzle, "_source", None)} for e in self.envs]
        return st

    def load_training_state(self, st: dict, reset_envs: bool = True):
        self.opt.load_state_dict(st["optimizer"])
        self.iteration = int(st["iteration"])
        self.global_step = int(st["global_step"])
        self.stage = st.get("stage", "")
        self.skipped_updates = int(st.get("skipped_updates", 0))
        self.elapsed_before = float(st.get("elapsed", 0.0))
        self.t0 = time.time()
        if "sampler" in st and hasattr(self.sampler, "load_state_dict"):
            self.sampler.load_state_dict(st["sampler"])
        self.recent.clear()
        self.recent.extend(tuple(r) for r in st.get("recent", []))
        if reset_envs:
            saved = st.get("env_puzzles") or []
            if len(saved) == len(self.envs):   # restart the in-flight puzzles from their first move
                from ..puzzle import Puzzle
                for i, (env, d) in enumerate(zip(self.envs, saved)):
                    p = Puzzle.from_dict(d["puzzle"])
                    p._family, p._source = d.get("family") or p.graph.kind, d.get("source")
                    self.obs[i] = env.reset(options={"puzzle": p})[0]
                self.ep_ret[:] = 0
                self.ep_len[:] = 0
            else:
                self.reset_envs()
        set_rng_state(st["rng"])  # last, so env resets above do not shift the restored streams

    def save(self, name: str | None = None, full: bool = True, **extra) -> Path:
        """Checkpoint readable by gnn.load_model (config + state_dict + extra keys);
        with full=True it also holds ``training_state`` for --resume."""
        path = self.ckpt_dir / f"{name or self.run_name}.pt"
        if full:
            extra.setdefault("training_state", self.training_state())
        save_model(self.model, path, iteration=self.iteration, global_step=self.global_step,
                   stage=self.stage, ppo_config={k: v for k, v in asdict(self.cfg).items()}, **extra)
        return path

    def close(self):
        self._csv_file.close()


__all__ = ["PPOConfig", "PPOTrainer", "gamma_for", "checkpoint_bonus", "make_env"]
