"""FastAPI server for the Zip web app.

    python -m zipsolve.app [--port 8000] [--host 127.0.0.1] [--no-browser]

Endpoints (JSON):
    GET  /api/models
    POST /api/generate     {kind, size, num_checkpoints?, unique?, seed?, options?, bank?}
        bank = "<mode>-<diff>" (e.g. "classic-hard"): pick pool[seed % len] from the curated
        bank (static/bank/, scripts/build_bank.py) when it exists, else generate as usual.
        Responses carry "rating" {label, score} (the human difficulty, zipsolve.difficulty).
    POST /api/check        {puzzle, path}
    POST /api/solve/exact  {puzzle, time_limit?, start_path?}
    POST /api/solve/rl     {puzzle, model?, mode: greedy|search|hybrid, budget?, start_path?, time_limit?,
                            compare?}
        greedy = policy argmax rollout; search = policy-ordered DFS (env pruning, node
        budget); hybrid = exact solver + GNN move ordering (complete), with the plain
        exact solver's stats under "exact" when compare (default true).
    POST /api/hint         {puzzle, path, id?}
    POST /api/hint/explain {puzzle, path, id?, time_limit?}  the Detective's teaching hint:
                           {status: next|backtrack|done|invalid|timeout, move, reason, technique, cells, keep?}
    GET  /api/robots       the strategy robots (zipsolve.robots) and whether each is available
    POST /api/solve/robot  {puzzle, robot: detective|mcts|evolver|gambler|sat, time_limit?, trace?, seed?}
                           same shape as /api/solve/rl + "robot", "stats"; extra trace event kinds
                           {"t":"note"|"path"|"tree", ...} (see zipsolve.robots)
    POST /api/policy       {puzzle, path, model?, full?, check?}
        the GNN's view of a position: legal-move probabilities, value / "winnable"
        estimate, board-wide score heat, and the solver's completable verdict.
    GET  /api/presets      difficulty presets (easy .. insane)
    POST /api/architect/design {kind: classic|walls|islands, size, target: expert|insane|hard, time_budget?,
                           seed?, include_solution?}  😈 the Architect designs a unique, fair (no guessing)
                           puzzle aimed at the target: {puzzle, rating, log: [{phase, msg}], stats, id, ...}
    GET  /api/architect/weekly ?week=YYYY-Www  😈 the week's Architect challenge (bank/architect/weekly.json)
    GET  /api/daily        ?difficulty=&date=YYYY-MM-DD  deterministic daily puzzle: from the bank's
                           daily schedule (easy/medium/hard/special) when present, else generated
    ("trace": true on /api/solve/exact and /api/solve/rl search|hybrid returns the
    search's push/pop events for the "watch it think" visualisation.)
Pages: / (the casual game), /lab (the AI show: watch the robots think, robot vs robot),
/workbench (power-user AI page: AI vision, solver replays, model comparison, custom options), /dashboard and /editor (served when their files exist; their
APIs are optional routers from dashboard_api.py / editor_api.py).
CPU-heavy work runs in the threadpool; every call has a time limit.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import secrets
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..puzzle import Puzzle
from .. import solver as _solver
from .bank import Bank, seed_of
from .engine import (RL_MODES, ModelRegistry, complete_prefix, hint, policy_view, rl_solve, traced_exact,
                     validate_prefix)

STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parents[2]

# kind -> (min size, max size, default size)
SIZE_LIMITS = {
    "grid2d": (2, 12, 7), "walls": (3, 12, 7), "islands": (2, 8, 3), "mask": (4, 12, 8),
    "grid3d": (2, 6, 4), "grid4d": (2, 4, 3), "grid": (2, 12, 6),
    "portals": (4, 12, 7), "torus": (3, 8, 6), "hex": (2, 7, 4), "tri": (2, 5, 3),
    "oneway": (3, 12, 7), "overpass": (4, 12, 7), "keys": (4, 12, 7), "cubesurf": (2, 5, 3),
}
OPTION_KEYS = {"walls": {"walls_frac": float}, "islands": {"jumps": int, "decoys": int, "max_per_pair": int,
                                                            "extra_bridges": int, "min_side": int,
                                                            "max_side": int, "gap": int},
               "mask": {"fill": float},
               "portals": {"portals": int, "decoys": int, "min_dist": int, "walls_frac": float},
               "torus": {"wraps": int},
               "oneway": {"oneway_frac": float, "decoys": int, "walls_frac": float},
               "overpass": {"overpasses": int},
               "keys": {"keys": int, "min_gap": int, "walls_frac": float}}
MAX_NODES = 1500

# Difficulty presets (shared with the client via /api/presets; the daily puzzles use them).
PRESETS: dict[str, dict] = {
    "easy": {"label": "Easy", "kind": "grid2d", "size": 5, "unique": True, "options": {},
             "blurb": "5×5 grid"},
    "medium": {"label": "Medium", "kind": "grid2d", "size": 7, "unique": True, "options": {},
               "blurb": "7×7 grid"},
    "hard": {"label": "Hard", "kind": "walls", "size": 8, "unique": True, "options": {"walls_frac": 0.35},
             "blurb": "8×8 with walls"},
    "expert": {"label": "Expert", "kind": "islands", "size": 4, "unique": True, "options": {},
               "blurb": "4 islands + bridges"},
    "insane": {"label": "Insane", "kind": "grid3d", "size": 4, "unique": False, "options": {},
               "blurb": "4×4×4 cube"},
}
DAILY_EPOCH = _dt.date(2026, 1, 1)   # Daily #1
MODE_KIND = {"classic": "grid2d", "walls": "walls", "islands": "islands", "cube": "grid3d"}
RATE_MAX_NODES = 150                 # rate generated puzzles up to this size (bank ones are pre-rated)


class GenerateReq(BaseModel):
    kind: str = "grid2d"
    size: int | list[int] = 7
    num_checkpoints: int | None = None
    unique: bool = False
    seed: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    time_limit: float = 15.0
    include_solution: bool = False
    bank: str | None = None           # "<mode>-<diff>": serve from the curated bank when available
    fog: bool = False                 # fog of war (any kind): numbers stay hidden until the path is near


class PuzzleReq(BaseModel):
    puzzle: dict
    id: str | None = None


class CheckReq(PuzzleReq):
    path: list[int] = Field(default_factory=list)
    paths: list[list[int]] | None = None       # kind "coop": [path 1, path 2] (partial allowed)


class ExactReq(PuzzleReq):
    time_limit: float = 10.0
    start_path: list[int] | None = None
    start_paths: list[list[int]] | None = None   # kind "coop"
    trace: bool = False


class PolicyReq(PuzzleReq):
    path: list[int] = Field(default_factory=list)
    model: str | None = None
    full: bool = True
    check: bool = True


class RLReq(PuzzleReq):
    model: str | None = None
    mode: str = "greedy"
    budget: int = 5000
    start_path: list[int] | None = None
    time_limit: float = 20.0
    compare: bool = True
    trace: bool = False


class RobotReq(PuzzleReq):
    robot: str = "detective"       # detective | mcts | evolver | gambler | sat (zipsolve.robots)
    time_limit: float = 10.0
    trace: bool = True
    seed: int = 0


class ExplainReq(PuzzleReq):
    path: list[int] = Field(default_factory=list)
    time_limit: float = 3.0


class HintReq(PuzzleReq):
    path: list[int] = Field(default_factory=list)
    paths: list[list[int]] | None = None       # kind "coop"
    path_index: int | None = None              # kind "coop": the pen the player holds (0/1)
    time_limit: float = 5.0


class ArchitectReq(BaseModel):                 # POST /api/architect/design (zipsolve.architect)
    kind: str = "classic"
    size: int = 8
    target: str = "expert"
    time_budget: float = 20.0
    seed: int | None = None
    include_solution: bool = False


# kind -> (min size, max size) for the Architect endpoint (islands: number of islands)
ARCHITECT_SIZES = {"classic": (5, 10), "grid2d": (5, 10), "walls": (5, 10), "islands": (3, 8)}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) and x == x and abs(x) != float("inf")


def _option(name: str, value, typ):
    """Validate one generation option: ints must be whole numbers, floats finite numbers."""
    if typ is int:
        if _is_int(value):
            return value
        if isinstance(value, float) and _is_num(value) and value.is_integer():
            return int(value)
        raise HTTPException(422, f"option {name!r} must be an integer")
    if not _is_num(value):
        raise HTTPException(422, f"option {name!r} must be a number")
    return float(value)


def _check_puzzle_dict(d) -> None:
    """Structural validation of a client-supplied puzzle dict (before building a graph)."""
    if not isinstance(d, dict):
        raise HTTPException(422, "puzzle must be an object")
    coords, edges, cps = d.get("coords"), d.get("edges"), d.get("checkpoints")
    if not isinstance(coords, list) or not coords:
        raise HTTPException(422, "puzzle.coords must be a non-empty list")
    if len(coords) > MAX_NODES:
        raise HTTPException(400, f"puzzle too large ({len(coords)} > {MAX_NODES} nodes)")
    dim = len(coords[0]) if isinstance(coords[0], list) else 0
    if not 1 <= dim <= 4 or any(not isinstance(c, list) or len(c) != dim or not all(map(_is_num, c))
                                for c in coords):
        raise HTTPException(422, "puzzle.coords must be lists of 1-4 numbers, all the same length")
    n = len(coords)
    if not isinstance(edges, list) or any(
            not isinstance(e, list) or len(e) != 2 or not all(_is_int(v) and 0 <= v < n for v in e)
            for e in edges):
        raise HTTPException(422, "puzzle.edges must be [u, v] pairs of valid node ids")
    if not isinstance(cps, list) or len(cps) < 2 or not all(_is_int(c) and 0 <= c < n for c in cps):
        raise HTTPException(422, "puzzle.checkpoints must be a list of at least 2 valid node ids")
    if len(set(cps)) != len(cps):
        raise HTTPException(422, "puzzle.checkpoints must be distinct")
    if d.get("meta") is not None and not isinstance(d["meta"], dict):
        raise HTTPException(422, "puzzle.meta must be an object")


def _rating(p: Puzzle) -> dict | None:
    """Human difficulty of a generated puzzle (None when too big or unsolved)."""
    if p.num_nodes > RATE_MAX_NODES or p.solution is None:
        return None
    try:
        from ..difficulty import rate
        r = rate(p, time_limit=1.5)
    except Exception:  # noqa: BLE001  (the rating is decoration: never fail a request over it)
        return None
    return {"label": r["label"], "score": r["score"]}


def _bank_rating(e: dict) -> dict:
    return {"label": e.get("label"), "score": e.get("score"), "mode": e.get("mode"), "diff": e.get("diff")}


def create_app(checkpoint_dir: str | Path | None = None, bank_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Zip puzzles", version="0.1")
    bank = Bank(bank_dir or os.environ.get("ZIPSOLVE_BANK") or STATIC / "bank")
    ckdir = Path(checkpoint_dir or os.environ.get("ZIPSOLVE_CHECKPOINTS") or ROOT / "checkpoints")
    registry = ModelRegistry(ckdir)
    store: OrderedDict[str, list[int] | None] = OrderedDict()   # puzzle id -> solution
    store_lock = Lock()

    def remember(sol) -> str:
        pid = secrets.token_hex(6)
        with store_lock:
            store[pid] = sol
            while len(store) > 500:
                store.popitem(last=False)
        return pid

    def load_puzzle(d: dict) -> Puzzle:
        _check_puzzle_dict(d)
        try:
            d = dict(d)
            d["solution"] = None
            p = Puzzle.from_dict(d)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"bad puzzle: {e}")
        if p.num_nodes > MAX_NODES:
            raise HTTPException(400, f"puzzle too large ({p.num_nodes} > {MAX_NODES} nodes)")
        if any(not (0 <= c < p.num_nodes) for c in p.checkpoints):
            raise HTTPException(400, "checkpoint out of range")
        return p

    daily_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()

    def _generate(kind, size, ncp, seed, unique, kw, tl, rated=False):
        from ..generator import make_puzzle
        t0 = time.perf_counter()
        p = make_puzzle(kind, size, num_checkpoints=ncp, rng=seed, unique=unique, time_limit=tl, **kw)
        secs = time.perf_counter() - t0
        if rated:
            return p, secs, (_rating(p) if unique else None)
        return p, secs

    def from_bank(e: dict, seed: int) -> tuple[dict, list | None]:
        d = dict(e["puzzle"])
        sol = d.pop("solution", None)
        return {"id": remember(sol), "seed": seed, "kind": MODE_KIND.get(e.get("mode"), d.get("kind")),
                "size": e.get("size"), "unique": True, "num_nodes": len(d["coords"]), "seconds": 0.0,
                "puzzle": d, "rating": _bank_rating(e), "bank_id": e.get("id"), "source": "bank"}, sol

    def load_model_or_http(name):
        try:
            return registry.load(name)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:  # noqa: BLE001  (e.g. checkpoint being written by training)
            raise HTTPException(503, f"could not load model: {e}")

    # ------------------------------------------------------------------ API
    @app.get("/api/models")
    def models():
        return {"directory": str(ckdir), "models": registry.list()}

    @app.get("/api/bank")
    def bank_info():
        idx = bank.index
        if not idx:
            return {"available": False}
        return {"available": True, "generated": idx.get("generated"), "daily_start": idx.get("daily_start"),
                "daily_end": idx.get("daily_end"),
                "modes": {m: {d: v.get("count") for d, v in ds.items()} for m, ds in idx.get("modes", {}).items()}}

    @app.post("/api/generate")
    async def generate(req: GenerateReq):
        out = await _generate_request(req)
        if req.fog and isinstance(out.get("puzzle"), dict):
            out["puzzle"].setdefault("meta", {})["fog"] = True
        return out

    async def _generate_request(req: GenerateReq):
        from ..generator import GenerationTimeout, make_puzzle
        if req.kind == "coop":   # two-path co-op puzzles (zipsolve.coop)
            from .. import coop as _coop
            seed = req.seed if req.seed is not None else secrets.randbelow(10**9)
            e = bank.pick(*req.bank.partition("-")[::2], seed) if req.bank and "-" in req.bank else None
            if e is not None and e["puzzle"].get("kind") == "coop":   # curated co-op pool (same pick as online)
                d = json.loads(json.dumps(e["puzzle"]))
                sol = (d.get("meta", {}).get("coop", {}).pop("solution", None) or {}).get("paths")
                out = {"id": remember(sol), "seed": seed, "kind": "coop", "size": e.get("size"), "unique": True,
                       "num_nodes": len(d["coords"]), "seconds": 0.0, "puzzle": d, "rating": _bank_rating(e),
                       "bank_id": e.get("id"), "source": "bank"}
                if req.include_solution:
                    out["solution"] = {"paths": sol}
                return out
            try:
                out, sol = await run_in_threadpool(_coop.api_generate, req.size, req.num_checkpoints, seed,
                                                   req.unique, req.options, _clamp(req.time_limit, 1.0, 60.0))
            except ValueError as e:   # includes GenerationTimeout
                raise HTTPException(422, str(e))
            out["id"] = remember(sol)
            if req.include_solution:
                out["solution"] = {"paths": sol}
            return out
        if req.bank and "-" in req.bank:
            mode, _, diff = req.bank.partition("-")
            bseed = req.seed if req.seed is not None and req.seed >= 0 else secrets.randbelow(10**9)
            e = bank.pick(mode, diff, bseed)
            if e is not None:
                out, sol = from_bank(e, bseed)
                if req.include_solution:
                    out["solution"] = sol
                return out
        if req.kind not in SIZE_LIMITS:
            raise HTTPException(400, f"unknown kind {req.kind!r}")
        lo, hi, _ = SIZE_LIMITS[req.kind]
        if isinstance(req.size, list):
            if req.kind != "grid" or not req.size or len(req.size) > 4:
                raise HTTPException(400, "list size only for kind 'grid' (1..4 dims)")
            size: Any = tuple(_clamp(int(s), 1, hi) for s in req.size)
        else:
            size = _clamp(int(req.size), lo, hi)
        kw = {}
        for k, typ in OPTION_KEYS.get(req.kind, {}).items():
            if k in req.options and req.options[k] is not None:
                kw[k] = _option(k, req.options[k], typ)
        if "walls_frac" in kw:
            kw["walls_frac"] = _clamp(kw["walls_frac"], 0.0, 1.0)
        if "fill" in kw:
            kw["fill"] = _clamp(kw["fill"], 0.3, 1.0)
        if "extra_bridges" in kw:
            kw["extra_bridges"] = _clamp(kw["extra_bridges"], 0, 6)
        if "jumps" in kw:  # at least one jump per extra island, at most ~4 per island
            kw["jumps"] = _clamp(kw["jumps"], size - 1, 4 * size) if isinstance(size, int) else kw["jumps"]
        if "decoys" in kw:
            kw["decoys"] = _clamp(kw["decoys"], 0, 12)
        if "max_per_pair" in kw:
            kw["max_per_pair"] = _clamp(kw["max_per_pair"], 1, 6)
        if "min_side" in kw or "max_side" in kw:
            a = _clamp(kw.get("min_side", 3), 2, 6)
            b = _clamp(kw.get("max_side", 4), a, 7)
            kw["min_side"], kw["max_side"] = a, b
        seed = req.seed if req.seed is not None else secrets.randbelow(10**9)
        ncp = None if req.num_checkpoints in (None, 0) else _clamp(int(req.num_checkpoints), 2, 10**6)
        tl = _clamp(req.time_limit, 1.0, 60.0)

        try:
            p, secs, rating = await run_in_threadpool(_generate, req.kind, size, ncp, seed, req.unique, kw, tl, True)
        except GenerationTimeout as e:
            raise HTTPException(422, f"{e}. Try a longer time limit, fewer islands/cells, "
                                     f"or turn off 'unique solution'.")
        except ValueError as e:
            raise HTTPException(422, str(e))
        d = p.to_dict()
        sol = d.pop("solution")
        pid = remember(sol)
        out = {"id": pid, "seed": seed, "kind": req.kind, "size": size, "unique": req.unique,
               "num_nodes": p.num_nodes, "seconds": secs, "puzzle": d, "rating": rating, "source": "generator"}
        if req.include_solution:
            out["solution"] = sol
        return out

    @app.post("/api/check")
    async def check(req: CheckReq):
        if isinstance(req.puzzle, dict) and req.puzzle.get("kind") == "coop":
            from .. import coop as _coop
            try:
                cp = _coop.api_load(req.puzzle)
            except ValueError as e:
                raise HTTPException(400, f"bad puzzle: {e}")
            return _coop.check_partial(cp, req.paths if req.paths is not None else [req.path, []])
        p = load_puzzle(req.puzzle)
        full = p.check_solution(req.path)
        partial = validate_prefix(p, req.path) if req.path else "empty path"
        return {"valid": full is None, "reason": full, "legal_prefix": partial is None,
                "prefix_reason": partial, "length": len(req.path), "num_nodes": p.num_nodes}

    @app.post("/api/solve/exact")
    async def solve_exact(req: ExactReq):
        if isinstance(req.puzzle, dict) and req.puzzle.get("kind") == "coop":
            from .. import coop as _coop
            try:
                cp = _coop.api_load(req.puzzle)
                return await run_in_threadpool(_coop.api_solve, cp, _clamp(req.time_limit, 0.1, 60.0),
                                               req.start_paths)
            except ValueError as e:
                raise HTTPException(400, str(e))
        p = load_puzzle(req.puzzle)
        tl = _clamp(req.time_limit, 0.1, 60.0)
        if req.start_path:
            bad = validate_prefix(p, req.start_path)
            if bad is not None:
                raise HTTPException(400, f"start_path: {bad}")
        if req.trace:
            r = await run_in_threadpool(traced_exact, p, req.start_path, tl)
            r["solved"] = r["status"] == "solved"
            return r
        if req.start_path and len(req.start_path) > 1:
            c = await run_in_threadpool(complete_prefix, p, req.start_path, tl)
            if c.status == "invalid":
                raise HTTPException(400, c.reason)
            return {"status": c.status, "path": c.path, "nodes_expanded": c.nodes_expanded,
                    "seconds": c.seconds, "solved": c.status == "solved"}
        r = await run_in_threadpool(_solver.solve, p, tl)
        return {"status": r.status, "path": r.path, "nodes_expanded": r.nodes_expanded,
                "seconds": r.seconds, "solved": r.status == "solved"}

    @app.post("/api/solve/rl")
    async def solve_rl(req: RLReq):
        p = load_puzzle(req.puzzle)
        if req.mode not in RL_MODES:
            raise HTTPException(400, "mode must be one of " + ", ".join(RL_MODES))
        model, meta, name = await run_in_threadpool(load_model_or_http, req.model)
        budget = _clamp(int(req.budget), 1, 200_000)
        tl = _clamp(req.time_limit, 0.5, 60.0)
        try:
            out = await run_in_threadpool(rl_solve, model, p, req.mode, budget, req.start_path, tl, meta,
                                         req.compare, req.trace)
        except ValueError as e:
            raise HTTPException(400, str(e))
        out["model"] = name
        out["model_stage"] = meta.get("stage")
        return out

    @app.get("/api/robots")
    def robots_list():   # the strategy robots of zipsolve.robots (availability: optional deps)
        from .. import robots as _robots
        return {"robots": _robots.listing()}

    @app.post("/api/solve/robot")
    async def solve_robot(req: RobotReq):
        from .. import robots as _robots
        try:
            key = _robots.resolve(req.robot)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0]))
        p = load_puzzle(req.puzzle)
        tl = _clamp(req.time_limit, 0.2, 60.0)
        return await run_in_threadpool(_robots.run_robot, key, p, tl, req.trace, int(req.seed))

    @app.post("/api/hint/explain")
    async def explain_hint(req: ExplainReq):   # the Detective's teaching hint: {status, move, reason, ...}
        from .. import robots as _robots
        p = load_puzzle(req.puzzle)
        sol = store.get(req.id) if req.id else None
        if sol is not None and not p.is_valid_solution(sol):
            sol = None
        return await run_in_threadpool(_robots.explain_next_move, p, req.path, _clamp(req.time_limit, 0.5, 20.0),
                                       sol)

    @app.post("/api/hint")
    async def get_hint(req: HintReq):
        if isinstance(req.puzzle, dict) and req.puzzle.get("kind") == "coop":
            from .. import coop as _coop
            try:
                cp = _coop.api_load(req.puzzle)
            except ValueError as e:
                raise HTTPException(400, f"bad puzzle: {e}")
            sol = store.get(req.id) if req.id else None
            return await run_in_threadpool(_coop.hint, cp, req.paths if req.paths is not None else [req.path, []],
                                           sol, _clamp(req.time_limit, 0.5, 20.0), req.path_index)
        p = load_puzzle(req.puzzle)
        sol = store.get(req.id) if req.id else None
        if sol is not None and not p.is_valid_solution(sol):
            sol = None
        tl = _clamp(req.time_limit, 0.5, 20.0)
        return await run_in_threadpool(hint, p, req.path, sol, tl)

    @app.post("/api/policy")
    async def get_policy(req: PolicyReq):
        p = load_puzzle(req.puzzle)
        model, meta, name = await run_in_threadpool(load_model_or_http, req.model)
        try:
            out = await run_in_threadpool(policy_view, model, meta, p, req.path, req.full, req.check)
        except ValueError as e:
            raise HTTPException(400, str(e))
        out["model"] = name
        out["model_stage"] = meta.get("stage")
        return out

    @app.post("/api/architect/design")
    async def architect_design(req: ArchitectReq):
        """😈 Ask the Architect: design a unique, fair puzzle aimed at Expert / Insane."""
        from ..architect import TARGETS, design
        if req.kind not in ARCHITECT_SIZES:
            raise HTTPException(400, f"unknown kind {req.kind!r}; expected one of {sorted(ARCHITECT_SIZES)}")
        if req.target not in TARGETS:
            raise HTTPException(400, f"target must be one of {sorted(TARGETS)}")
        lo, hi = ARCHITECT_SIZES[req.kind]
        size = _clamp(int(req.size), lo, hi)
        budget = _clamp(float(req.time_budget), 2.0, 60.0)
        seed = req.seed if req.seed is not None and req.seed >= 0 else secrets.randbelow(10**9)
        try:
            r = await run_in_threadpool(design, req.kind, size, req.target, budget, seed)
        except ValueError as e:
            raise HTTPException(422, str(e))
        out = r.to_dict()
        if not r.ok:
            raise HTTPException(422, "the Architect could not certify a design in time; try a larger time_budget")
        sol = out["puzzle"].pop("solution")
        out["id"] = remember(sol)
        out["num_nodes"] = len(out["puzzle"]["coords"])
        out["seed"] = seed
        out["source"] = "architect"
        if req.include_solution:
            out["solution"] = sol
        return out

    @app.get("/api/architect/weekly")
    def architect_weekly(week: str | None = None):
        """😈 This week's Architect challenge (bank/architect/weekly.json, keyed by ISO week "YYYY-Www")."""
        if week is None:
            y, w, _ = _dt.date.today().isocalendar()
            week = f"{y:04d}-W{w:02d}"
        wk = bank.weekly(week)
        if wk is None:
            raise HTTPException(404, "no weekly Architect challenge in the bank")
        e, key = wk
        out, _ = from_bank(e, seed_of(e))
        out["weekly"] = {"week": week, "bank_week": key, "number": e.get("number"), "label": e.get("label"),
                         "clues": (e.get("architect") or {}).get("clues")}
        return out

    @app.get("/api/presets")
    def presets():
        return {"presets": [{"id": k, **v} for k, v in PRESETS.items()],
                "daily_epoch": DAILY_EPOCH.isoformat()}

    @app.get("/api/daily")
    async def daily(difficulty: str = "medium", date: str | None = None):
        if difficulty not in PRESETS and difficulty != "special":
            raise HTTPException(400, "difficulty must be one of " + ", ".join([*PRESETS, "special"]))
        try:
            day = _dt.date.fromisoformat(date) if date else _dt.date.today()
        except ValueError:
            raise HTTPException(422, "date must be YYYY-MM-DD")
        key = (day.isoformat(), difficulty)
        number = (day - DAILY_EPOCH).days + 1
        label = PRESETS[difficulty]["label"] if difficulty in PRESETS else "Special"
        got = bank.daily(day, difficulty) if difficulty in ("easy", "medium", "hard", "special") else None
        if got is not None:   # the curated schedule (same puzzle as the static site)
            e, _num = got
            out, _ = from_bank(e, seed_of(e))
            out["daily"] = {"date": key[0], "number": number, "difficulty": difficulty, "label": label}
            return out
        with store_lock:
            hit = daily_cache.get(key)
        if hit is None:
            pr = PRESETS[difficulty if difficulty in PRESETS else "expert"]
            seed = int.from_bytes(hashlib.sha256(f"zip-daily:{key[0]}:{difficulty}".encode()).digest()[:4],
                                  "big") % 10**9
            try:
                p, secs, rating = await run_in_threadpool(_generate, pr["kind"], pr["size"], None, seed,
                                                          pr["unique"], dict(pr["options"]), 30.0, True)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"could not generate the daily puzzle: {e}")
            d = p.to_dict()
            sol = d.pop("solution")
            hit = {"seed": seed, "kind": pr["kind"], "size": pr["size"], "unique": pr["unique"],
                   "num_nodes": p.num_nodes, "seconds": secs, "puzzle": d, "solution": sol,
                   "rating": rating, "source": "generator",
                   "daily": {"date": key[0], "number": number, "difficulty": difficulty, "label": label}}
            with store_lock:
                daily_cache[key] = hit
                while len(daily_cache) > 64:
                    daily_cache.popitem(last=False)
        out = {k: v for k, v in hit.items() if k != "solution"}
        out["id"] = remember(hit["solution"])
        return out
    # ------------------------------------------------------------------ static
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    def _page(name):
        f = STATIC / name
        if not f.is_file():
            raise HTTPException(404, f"{name} not available")
        return FileResponse(f, headers={"Cache-Control": "no-cache"})

    @app.get("/lab")
    def lab_page():   # the AI show (watch the robots think, robot vs robot)
        return _page("lab.html")

    @app.get("/workbench")
    def workbench_page():   # the power-user AI page (models, limits, compare, AI vision)
        return _page("workbench.html")

    @app.get("/dashboard")
    def dashboard_page():
        return _page("dashboard.html")

    @app.get("/editor")
    def editor_page():
        return _page("editor.html")

    # optional routers owned by the dashboard / editor modules (the app works without them)
    import importlib
    import logging
    for mod_name in ("dashboard_api", "editor_api"):
        try:
            mod = importlib.import_module(f"{__package__}.{mod_name}")
        except ImportError as e:
            if getattr(e, "name", None) not in (f"{__package__}.{mod_name}", mod_name):
                logging.getLogger(__name__).warning("could not import %s: %s", mod_name, e)
            continue
        except Exception as e:  # noqa: BLE001  (a broken optional module must not take the app down)
            logging.getLogger(__name__).warning("could not import %s: %s", mod_name, e)
            continue
        router = getattr(mod, "router", None)
        if router is not None:
            app.include_router(router)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
