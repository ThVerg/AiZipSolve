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
    POST /api/policy       {puzzle, path, model?, full?, check?}
        the GNN's view of a position: legal-move probabilities, value / "winnable"
        estimate, board-wide score heat, and the solver's completable verdict.
    GET  /api/presets      difficulty presets (easy .. insane)
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
}
OPTION_KEYS = {"walls": {"walls_frac": float}, "islands": {"jumps": int, "decoys": int, "max_per_pair": int,
                                                            "extra_bridges": int, "min_side": int,
                                                            "max_side": int, "gap": int},
               "mask": {"fill": float}}
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


class PuzzleReq(BaseModel):
    puzzle: dict
    id: str | None = None


class CheckReq(PuzzleReq):
    path: list[int]


class ExactReq(PuzzleReq):
    time_limit: float = 10.0
    start_path: list[int] | None = None
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


class HintReq(PuzzleReq):
    path: list[int]
    time_limit: float = 5.0


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
        from ..generator import GenerationTimeout, make_puzzle
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
        p = load_puzzle(req.puzzle)
        full = p.check_solution(req.path)
        partial = validate_prefix(p, req.path) if req.path else "empty path"
        return {"valid": full is None, "reason": full, "legal_prefix": partial is None,
                "prefix_reason": partial, "length": len(req.path), "num_nodes": p.num_nodes}

    @app.post("/api/solve/exact")
    async def solve_exact(req: ExactReq):
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

    @app.post("/api/hint")
    async def get_hint(req: HintReq):
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
