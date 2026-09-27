"""Puzzle-editor API: save / load custom puzzles and analyse drafts.

Mounted by the main app (``app.include_router(router)``):

    GET    /api/custom                 -> [{id, name, num_nodes, kind, dim, checkpoints, updated}]
    GET    /api/custom/{id}            -> {id, name, created, updated, puzzle}
    POST   /api/custom                 {name, puzzle, id?} -> {id, name, updated, ...}
    DELETE /api/custom/{id}
    POST   /api/editor/analyze         {puzzle, time_limit?, want_solution?, check_unique?}
               -> {connected, parity_ok, degree_ok, issues, solvable, unique, status, seconds,
                   solution?, alternative?}
    POST   /api/editor/make_unique     {puzzle, time_limit?, seed?} -> {puzzle, added, unique, seconds}
    POST   /api/editor/suggest         {puzzle, num_checkpoints?, unique?, time_limit?, seed?}
               -> {puzzle, unique, seconds}   (checkpoints from a random Hamiltonian path)

Custom puzzles live in ``puzzles/custom/<id>.json`` (override with $ZIPSOLVE_CUSTOM_DIR).
Ids are ``[a-z0-9-]`` only, so nothing can escape that directory. Solver work runs in the
threadpool with a hard time limit, at most two editor jobs at a time.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..graph import ZipGraph, from_edges
from ..puzzle import Puzzle

ROOT = Path(__file__).resolve().parents[2]
MAX_NODES = 1500
MAX_EDGES = 12 * MAX_NODES
MAX_NAME = 80
ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
META_KEYS = {"island", "bridges", "walls", "shape", "editor"}

router = APIRouter()
_jobs = threading.BoundedSemaphore(2)
_file_lock = threading.Lock()


def custom_dir() -> Path:
    d = Path(os.environ.get("ZIPSOLVE_CUSTOM_DIR") or ROOT / "puzzles" / "custom")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------- validation
def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _check_structure(d: Any, min_cps: int) -> None:
    """Strict structural validation (types, ranges, sizes) of a client puzzle dict."""
    if min_cps >= 2:
        try:  # the play server's validator, when available (same rules + messages)
            from .server import _check_puzzle_dict
            _check_puzzle_dict(d)
        except ImportError:
            pass
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
    if any(abs(x) > 10_000 for c in coords for x in c):
        raise HTTPException(422, "puzzle.coords out of range")
    n = len(coords)
    if not isinstance(edges, list) or len(edges) > MAX_EDGES or any(
            not isinstance(e, list) or len(e) != 2 or not all(_is_int(v) and 0 <= v < n for v in e)
            for e in edges):
        raise HTTPException(422, "puzzle.edges must be [u, v] pairs of valid node ids")
    if cps is None:
        cps = []
    if not isinstance(cps, list) or len(cps) < min_cps or not all(_is_int(c) and 0 <= c < n for c in cps):
        raise HTTPException(422, f"puzzle.checkpoints must be a list of at least {min_cps} valid node ids")
    if len(set(cps)) != len(cps):
        raise HTTPException(422, "puzzle.checkpoints must be distinct")
    meta = d.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise HTTPException(422, "puzzle.meta must be an object")
    kind = d.get("kind", "custom")
    if not isinstance(kind, str) or len(kind) > 32:
        raise HTTPException(422, "puzzle.kind must be a short string")


def _clean_meta(meta: dict | None, n: int) -> dict:
    """Keep only the meta keys the app understands, each strictly typed."""
    out: dict = {}
    meta = meta or {}

    def pairs(x):
        return isinstance(x, list) and len(x) <= MAX_EDGES and all(
            isinstance(e, list) and len(e) == 2 and all(_is_int(v) and 0 <= v < n for v in e) for e in x)

    isl = meta.get("island")
    if isl is not None:
        if not (isinstance(isl, list) and len(isl) == n and all(_is_int(v) and 0 <= v < 10_000 for v in isl)):
            raise HTTPException(422, "meta.island must list one island index per node")
        out["island"] = isl
    for k in ("bridges", "walls"):
        if meta.get(k) is not None:
            if not pairs(meta[k]):
                raise HTTPException(422, f"meta.{k} must be [u, v] pairs of valid node ids")
            out[k] = meta[k]
    shp = meta.get("shape")
    if shp is not None:
        if not (isinstance(shp, list) and 1 <= len(shp) <= 4 and all(_is_int(v) and 0 < v <= 10_000 for v in shp)):
            raise HTTPException(422, "meta.shape must be a list of 1-4 positive ints")
        out["shape"] = shp
    ed = meta.get("editor")
    if isinstance(ed, dict):
        out["editor"] = {k: v for k, v in ed.items()
                         if isinstance(k, str) and len(k) <= 32 and (_is_num(v) or isinstance(v, (bool, str)))
                         and (not isinstance(v, str) or len(v) <= 64)}
    return out


def _canonical(d: dict, min_cps: int) -> dict:
    """Validated, normalised puzzle dict (no solution)."""
    _check_structure(d, min_cps)
    n = len(d["coords"])
    return {
        "kind": d.get("kind") or "custom",
        "coords": [[float(x) for x in c] for c in d["coords"]],
        "edges": sorted({(min(u, v), max(u, v)) for u, v in d["edges"] if u != v}),
        "meta": _clean_meta(d.get("meta"), n),
        "checkpoints": list(d.get("checkpoints") or []),
    }


def _graph(d: dict) -> ZipGraph:
    return from_edges(np.array(d["coords"], dtype=float), [tuple(e) for e in d["edges"]],
                      d.get("kind", "custom"), d.get("meta") or {})


def _to_json_puzzle(g: ZipGraph, cps: list[int], kind: str, meta: dict) -> dict:
    return {"kind": kind, "coords": g.coords.tolist(), "edges": [list(e) for e in g.edges()],
            "meta": meta, "checkpoints": [int(c) for c in cps]}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------- analysis
def _structure_report(g: ZipGraph, cps: list[int]) -> dict:
    from ..generator import _parity_ok, _two_colour
    n = g.num_nodes
    issues: list[str] = []
    connected = g.is_connected()
    if not connected:
        seen, comps = set(), 0
        for s in range(n):
            if s in seen:
                continue
            comps += 1
            stack = [s]
            seen.add(s)
            while stack:
                for w in g.neighbors[stack.pop()]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
        issues.append(f"the cells form {comps} separate groups: add bridges or cells to join them")
    col = _two_colour(g.neighbors, range(n))
    start = cps[0] if cps else None
    end = cps[-1] if len(cps) >= 2 else None
    parity_ok = _parity_ok(col, list(range(n)), start, end) if col is not None else True
    if col is not None and not parity_ok:
        a = sum(1 for v in range(n) if col[v] == 0)
        b = n - a
        if abs(a - b) > 1:
            issues.append(f"chessboard colours are unbalanced ({a} vs {b}): a path alternates colours, "
                          f"so they may differ by at most 1")
        else:
            issues.append("start/end checkpoints are on the wrong chessboard colours for this cell count")
    dead = [v for v in range(n) if n > 1 and not g.neighbors[v]]
    leaves = [v for v in range(n) if len(g.neighbors[v]) == 1]
    ends = {start, end} - {None}
    bad_leaves = [v for v in leaves if v not in ends]
    degree_ok = not dead and len(leaves) <= 2 and len(bad_leaves) <= 2 - len(ends)
    if dead:
        issues.append(f"{len(dead)} cell(s) have no neighbours")
    elif not degree_ok:
        issues.append(f"{len(leaves)} dead-end cell(s): a path can only have its two ends there")
    return {"connected": connected, "bipartite": col is not None, "parity_ok": parity_ok,
            "degree_ok": degree_ok, "dead_cells": dead, "dead_ends": leaves, "issues": issues}


def _analyze(d: dict, tl: float, want_solution: bool, check_unique: bool) -> dict:
    from .. import solver
    from ..generator import random_hamiltonian_path
    t0 = time.perf_counter()
    g = _graph(d)
    cps = d["checkpoints"]
    rep = _structure_report(g, cps)
    out: dict[str, Any] = dict(rep, solvable=None, unique=None, status="", num_nodes=g.num_nodes)
    if not (rep["connected"] and rep["parity_ok"] and rep["degree_ok"]):
        out.update(solvable=False, unique=False, status="unsat")
    elif len(cps) < 2:
        # no clues yet: does the shape have any Hamiltonian path at all?
        path = random_hamiltonian_path(g, rng=0, time_limit=tl, start=cps[0] if cps else None)
        if path is not None:
            out.update(solvable=True, status="solved", unique=False)
            if want_solution:
                out["solution"] = path
        elif time.perf_counter() - t0 >= tl * 0.98:
            out.update(status="timeout")
        else:
            out.update(solvable=False, unique=False, status="unsat")
        out["note"] = "add at least 2 checkpoints (start and end)"
    else:
        p = Puzzle(g, list(cps))
        r = solver.solve(p, time_limit=tl)
        out["status"] = r.status
        out["nodes_expanded"] = r.nodes_expanded
        if r.status == "solved":
            out["solvable"] = True
            if want_solution:
                out["solution"] = r.path
            left = tl - (time.perf_counter() - t0)
            if check_unique and left > 0.05:
                sols, st = solver.find_solutions(p, limit=2, time_limit=left)
                if len(sols) >= 2:
                    out["unique"] = False
                    if want_solution:
                        out["solution"], out["alternative"] = sols[0], sols[1]
                elif st == "complete":
                    out["unique"] = True
                # timeout with <= 1 solution: unknown
            out["unique_status"] = "unknown" if out["unique"] is None else "proven"
        elif r.status == "unsat":
            out.update(solvable=False, unique=False)
    out["seconds"] = time.perf_counter() - t0
    return out


def _make_unique(g: ZipGraph, cps: list[int], path: list[int], deadline: float, rng) -> tuple[list[int], bool]:
    """Add checkpoints taken from `path` (a solution respecting `cps`) until it is the only one.

    Mirrors generator._puzzle_from_path: prefer a clue that rules out a known alternative
    solution, spread away from the existing clues; else split the largest gap.
    """
    from ..solver import find_solutions
    pos = {v: i for i, v in enumerate(path)}
    cps = sorted(cps, key=pos.__getitem__)
    total = max(0.5, deadline - time.perf_counter())
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.02:
            return cps, False
        per_check = min(remaining, max(0.5, total / 12))
        sols, status = find_solutions(Puzzle(g, cps), limit=2, time_limit=per_check)
        if len(sols) <= 1 and status == "complete":
            return cps, True
        taken = set(cps)
        free = [i for i in range(1, len(path) - 1) if path[i] not in taken]
        if not free:
            return cps, len(sols) <= 1 and status == "complete"
        others = [s for s in sols if list(s) != list(path)]
        idxs = sorted(pos[c] for c in cps)
        new = None
        if others:
            alt = others[0]
            apos = {v: i for i, v in enumerate(alt)}
            killers = []
            for i in free:
                trial = sorted(idxs + [i])
                order = [apos[path[j]] for j in trial]
                if order != sorted(order):   # alt no longer visits the clues in order
                    killers.append(i)
            if killers:
                dist = np.array([min(abs(i - j) for j in idxs) for i in killers], dtype=float)
                top = np.flatnonzero(dist >= np.quantile(dist, 0.75))
                new = killers[int(top[int(rng.integers(len(top)))])]
        if new is None:
            gaps = [(q - p, p, q) for p, q in zip(idxs, idxs[1:]) if q - p > 1]
            big = max(gp[0] for gp in gaps)
            _, p, q = [gp for gp in gaps if gp[0] == big][int(rng.integers(sum(1 for gp in gaps if gp[0] == big)))]
            lo, hi = p + max(1, (q - p) // 4), q - max(1, (q - p) // 4)
            new = int(rng.integers(lo, max(lo, hi) + 1))
        cps = [path[i] for i in sorted(idxs + [new])]


# ---------------------------------------------------------------------------- models
class SaveReq(BaseModel):
    name: str = "Untitled"
    puzzle: dict
    id: str | None = None


class AnalyzeReq(BaseModel):
    puzzle: dict
    time_limit: float = 5.0
    want_solution: bool = False
    check_unique: bool = True


class UniqueReq(BaseModel):
    puzzle: dict
    time_limit: float = 15.0
    seed: int | None = None


class SuggestReq(BaseModel):
    puzzle: dict
    num_checkpoints: int | None = None
    unique: bool = False
    time_limit: float = 10.0
    seed: int | None = None


async def _job(fn, *args):
    """Run CPU work in the threadpool, at most two editor jobs at once."""
    def run():
        if not _jobs.acquire(timeout=30):
            raise HTTPException(503, "the solver is busy, try again in a moment")
        try:
            return fn(*args)
        finally:
            _jobs.release()
    return await run_in_threadpool(run)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40].strip("-")
    return f"{s or 'puzzle'}-{secrets.token_hex(3)}"


def _path_for(pid: str) -> Path:
    if not isinstance(pid, str) or not ID_RE.match(pid):
        raise HTTPException(400, "invalid id (use a-z, 0-9 and '-')")
    d = custom_dir().resolve()
    p = (d / f"{pid}.json").resolve()
    if p.parent != d:
        raise HTTPException(400, "invalid id")
    return p


# ---------------------------------------------------------------------------- routes
@router.get("/api/custom")
def list_custom():
    out = []
    for f in custom_dir().glob("*.json"):
        if not ID_RE.match(f.stem):
            continue
        try:
            j = json.loads(f.read_text())
            pz = j["puzzle"]
            out.append({"id": f.stem, "name": j.get("name", f.stem), "num_nodes": len(pz["coords"]),
                        "kind": pz.get("kind", "custom"), "dim": len(pz["coords"][0]),
                        "checkpoints": len(pz.get("checkpoints", [])),
                        "created": j.get("created"), "updated": j.get("updated", f.stat().st_mtime)})
        except Exception:  # noqa: BLE001  (skip corrupt files)
            continue
    out.sort(key=lambda x: -(x["updated"] or 0))
    return out


@router.get("/api/custom/{pid}")
def get_custom(pid: str):
    p = _path_for(pid)
    if not p.is_file():
        raise HTTPException(404, "no such puzzle")
    j = json.loads(p.read_text())
    pz = dict(j["puzzle"])
    pz.pop("solution", None)
    return {"id": pid, "name": j.get("name", pid), "created": j.get("created"),
            "updated": j.get("updated"), "puzzle": pz}


@router.post("/api/custom")
def save_custom(req: SaveReq):
    name = (req.name or "").strip()[:MAX_NAME] or "Untitled"
    if any(ord(ch) < 32 for ch in name):
        raise HTTPException(422, "name contains control characters")
    pz = _canonical(req.puzzle, 2)
    try:
        Puzzle.from_dict(dict(pz, edges=[list(e) for e in pz["edges"]], solution=None))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"bad puzzle: {e}")
    pz["edges"] = [list(e) for e in pz["edges"]]
    pid = req.id or _slug(name)
    path = _path_for(pid)
    now = time.time()
    with _file_lock:
        created = now
        if path.is_file():
            try:
                created = json.loads(path.read_text()).get("created", now)
            except Exception:  # noqa: BLE001
                pass
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"id": pid, "name": name, "created": created, "updated": now, "puzzle": pz}))
        os.replace(tmp, path)
    return {"id": pid, "name": name, "created": created, "updated": now, "num_nodes": len(pz["coords"])}


@router.delete("/api/custom/{pid}")
def delete_custom(pid: str):
    p = _path_for(pid)
    if not p.is_file():
        raise HTTPException(404, "no such puzzle")
    p.unlink()
    return {"deleted": pid}


@router.post("/api/editor/analyze")
async def analyze(req: AnalyzeReq):
    d = _canonical(req.puzzle, 0)
    tl = _clamp(float(req.time_limit), 0.2, 30.0)
    return await _job(_analyze, d, tl, req.want_solution, req.check_unique)


@router.post("/api/editor/make_unique")
async def make_unique(req: UniqueReq):
    d = _canonical(req.puzzle, 2)
    tl = _clamp(float(req.time_limit), 0.5, 60.0)
    seed = req.seed if req.seed is not None else secrets.randbelow(10**9)

    def work():
        from .. import solver
        t0 = time.perf_counter()
        g = _graph(d)
        rep = _structure_report(g, d["checkpoints"])
        if not (rep["connected"] and rep["parity_ok"] and rep["degree_ok"]):
            raise HTTPException(422, "not solvable: " + "; ".join(rep["issues"]))
        r = solver.solve(Puzzle(g, d["checkpoints"]), time_limit=tl * 0.4)
        if r.status != "solved":
            raise HTTPException(422, "no solution found" if r.status == "unsat"
                                else "could not find a solution in time")
        cps, uniq = _make_unique(g, d["checkpoints"], r.path, t0 + tl, np.random.default_rng(seed))
        added = len(cps) - len(d["checkpoints"])
        return {"puzzle": _to_json_puzzle(g, cps, d["kind"], d["meta"]), "added": added, "unique": uniq,
                "solution": r.path, "seconds": time.perf_counter() - t0}
    return await _job(work)


@router.post("/api/editor/suggest")
async def suggest(req: SuggestReq):
    d = _canonical(req.puzzle, 0)
    tl = _clamp(float(req.time_limit), 0.5, 60.0)
    seed = req.seed if req.seed is not None else secrets.randbelow(10**9)

    def work():
        from ..generator import default_num_checkpoints, place_checkpoints, random_hamiltonian_path
        t0 = time.perf_counter()
        g = _graph(d)
        rng = np.random.default_rng(seed)
        rep = _structure_report(g, [])
        if not (rep["connected"] and rep["parity_ok"] and rep["degree_ok"]):
            raise HTTPException(422, "this shape has no path through every cell: " + "; ".join(rep["issues"]))
        path = random_hamiltonian_path(g, rng, time_limit=tl * (0.5 if req.unique else 0.95))
        if path is None:
            if time.perf_counter() - t0 >= tl * 0.45:
                raise HTTPException(422, "no path through every cell found in time")
            raise HTTPException(422, "this shape has no path through every cell")
        k = req.num_checkpoints or default_num_checkpoints(g.num_nodes, rng)
        k = _clamp(int(k), 2, g.num_nodes)
        cps = place_checkpoints(path, k, rng)
        uniq = None
        if req.unique:
            cps, uniq = _make_unique(g, cps, path, t0 + tl, rng)
        return {"puzzle": _to_json_puzzle(g, cps, d["kind"], d["meta"]), "unique": uniq, "solution": path,
                "seed": seed, "seconds": time.perf_counter() - t0}
    return await _job(work)


__all__ = ["router", "custom_dir"]
