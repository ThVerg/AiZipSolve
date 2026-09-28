"""Training dashboard API (mounted by server.create_app; page at /dashboard).

Data sources (all read-only, parsed lazily and cached by path + mtime + size):
    runs/<run>.csv            per-iteration PPO log (columns vary between versions;
                              JSON-valued cells such as family_sr / src_counts are
                              flattened to "family_sr.<family>" etc.)
    runs/<run>_val.csv        long-format validation log (iter, step, time, stage,
                              scope, family, n, greedy, sampled)
    runs/<run>.log|.out       stdout (planned stages, "final checkpoint:" marker, tail)
    runs/bench/*.json         benchmark reports (zipsolve.rl.benchmark: meta/summary/rows)
    runs/eval/*.json          evaluate.py results
    checkpoints/*.pt          torch checkpoints (meta loaded on demand)
and the remote mirrors made by scripts/sync_runs.py:
    runs/remote/<host>/...    checkpoints/remote/<host>/*.pt

Routes (prefix /api/dashboard):
    GET  /runs                          run cards (status, stage, sparkline, ...)
    GET  /runs/{source}/{name}          time series (+ validation, stage transitions)
    GET  /runs/{source}/{name}/log      stdout tail
    GET  /bench                         benchmark reports (summaries, no per-puzzle rows)
    GET  /eval                          evaluate.py results as rows
    GET  /checkpoints                   checkpoint files (+ meta when already cached)
    GET  /checkpoints/meta?name=...     load + cache one checkpoint's meta
    GET  /sync  POST /sync              sync status / start a sync (thread, debounced)
    POST /sync/auto {"minutes": N}      auto-sync every N minutes (0 = off)
``source`` is "local" or a remote host name.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import io
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
router = APIRouter()

X_COLS = ("iter", "step", "time")
RUNNING_WINDOW = 300.0  # a log touched within this many seconds counts as "running"
_SAFE = re.compile(r"^[A-Za-z0-9_.:+@=\-]+$")
_CKPT_KIND = re.compile(r"^(.*?)_(best|final|latest|stage\d+)$")


class _Cfg:
    root: Path = ROOT
    runs_dir: Path = ROOT / "runs"
    ckpt_dir: Path = Path(os.environ.get("ZIPSOLVE_CHECKPOINTS") or ROOT / "checkpoints")
    sync_host: str = os.environ.get("ZIPSOLVE_SYNC_HOST", "zipserver")
    sync_fn: Callable[..., dict] | None = None  # default: scripts/sync_runs.py:sync
    sync_timeout: float = 240.0
    sync_debounce: float = 15.0


CFG = _Cfg()


def configure(root=None, runs_dir=None, ckpt_dir=None, sync_host=None, sync_fn=None, sync_debounce=None):
    """Point the dashboard at other directories (tests, alternative layouts)."""
    if root is not None:
        CFG.root = Path(root)
        CFG.runs_dir = Path(root) / "runs"
        CFG.ckpt_dir = Path(root) / "checkpoints"
    if runs_dir is not None:
        CFG.runs_dir = Path(runs_dir)
    if ckpt_dir is not None:
        CFG.ckpt_dir = Path(ckpt_dir)
    if sync_host is not None:
        CFG.sync_host = sync_host
    if sync_fn is not None:
        CFG.sync_fn = sync_fn
    if sync_debounce is not None:
        CFG.sync_debounce = sync_debounce
    _CACHE.clear()


# --------------------------------------------------------------------------- helpers
_CACHE: dict[tuple, tuple[tuple, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _stat_key(path: Path):
    st = path.stat()
    return (st.st_mtime, st.st_size)


def _cached(kind: str, path: Path, fn):
    """fn(path) memoised on (kind, path) and invalidated when mtime/size change."""
    key = (kind, str(path))
    sk = _stat_key(path)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == sk:
            return hit[1]
    val = fn(path)
    with _CACHE_LOCK:
        _CACHE[key] = (sk, val)
    return val


def _safe(part: str) -> str:
    if not part or not _SAFE.match(part) or part.startswith("."):
        raise HTTPException(400, f"bad name {part!r}")
    return part


def _sources() -> dict[str, Path]:
    """source name -> runs directory ("local" + every mirrored host)."""
    out = {"local": CFG.runs_dir}
    rem = CFG.runs_dir / "remote"
    if rem.is_dir():
        for d in sorted(rem.iterdir()):
            if d.is_dir() and _SAFE.match(d.name) and d.name != "local":
                out[d.name] = d
    return out


def _ckpt_base(source: str) -> Path:
    return CFG.ckpt_dir if source == "local" else CFG.ckpt_dir / "remote" / source


def _sync_state(source: str) -> dict:
    if source == "local":
        return {}
    p = CFG.runs_dir / "remote" / source / ".sync.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _num(s: str):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# --------------------------------------------------------------------------- CSV parsing
def parse_run_csv(path: Path) -> dict:
    """Robust columnar parse of a training CSV.

    Unknown columns are kept generically: numeric -> ``metrics``; JSON objects are
    flattened ("family_sr.grid2d:5" from {"grid2d:5": [rate, n]}, "src_counts.replay",
    ...); other strings -> ``text``. Rows with a wrong field count are padded /
    truncated, repeated header rows skipped. Stage transitions come from "stage".
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        data = f.read()
    reader = csv.reader(io.StringIO(data.replace("\x00", "")))
    header = None
    num: dict[str, list] = {}
    text: dict[str, list] = {}
    n = 0

    def put(store, name, value):
        col = store.get(name)
        if col is None:
            col = store[name] = [None] * n
        col.append(value)

    for row in reader:
        if not row or all(not c.strip() for c in row):
            continue
        if header is None:
            header = [h.strip() for h in row]
            continue
        if row == header or [c.strip() for c in row] == header:
            continue
        before = n
        for name, cell in zip(header, row):
            if not name:
                continue
            cell = cell.strip()
            if cell == "":
                continue
            v = _num(cell)
            if v is not None or cell.lower() in ("nan", "inf", "-inf"):
                put(num, name, v)
            elif cell[:1] in "{[":
                try:
                    obj = json.loads(cell)
                except ValueError:
                    put(text, name, cell)
                    continue
                if isinstance(obj, dict):
                    for k, val in obj.items():
                        if isinstance(val, (list, tuple)) and val:
                            val = val[0]
                        if isinstance(val, bool):
                            val = float(val)
                        if isinstance(val, (int, float)):
                            put(num, f"{name}.{k}", _clean(float(val)))
                elif isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
                    put(num, name, _clean(float(obj[0])))
            else:
                put(text, name, cell)
        n = before + 1
        for store in (num, text):
            for col in store.values():
                if len(col) < n:
                    col.append(None)
    # a column that is mostly text but has a few numbers (e.g. stage "3") belongs to text
    for name in list(num):
        if name in text:
            t = sum(v is not None for v in text[name])
            k = sum(v is not None for v in num[name])
            if t >= k:
                merged = [text[name][i] if text[name][i] is not None else
                          (None if num[name][i] is None else f"{num[name][i]:g}") for i in range(n)]
                text[name] = merged
                del num[name]
            else:
                del text[name]
    xs = {}
    for c in X_COLS:
        if c in num:
            xs[c] = num.pop(c)
    if "iter" not in xs:
        xs["iter"] = [float(i + 1) for i in range(n)]
    stage_col = text.get("stage") or []
    transitions = []
    prev = object()
    for i, s in enumerate(stage_col):
        if s is None or s == prev:
            continue
        transitions.append({"index": i, "stage": s, **{c: xs[c][i] for c in xs}})
        prev = s
    # drop constant-None metrics
    num = {k: v for k, v in num.items() if any(x is not None for x in v)}
    return {"n": n, "columns": header or [], "x": xs, "metrics": num,
            "text": {k: v for k, v in text.items() if k != "stage"}, "stage": stage_col,
            "transitions": transitions}


def parse_val_csv(path: Path) -> dict:
    """Long-format validation CSV -> points (iter/step/time) + series keyed
    "<scope>/<family>/<metric>" aligned to the points. Unknown value columns are kept."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    id_cols = {"iter", "step", "time", "stage", "scope", "family", "n"}
    points: list[dict] = []
    index: dict[tuple, int] = {}
    series: dict[str, dict[int, float]] = {}
    counts: dict[str, int] = {}
    for r in rows:
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items() if k}
        it = _num(r.get("iter"))
        if it is None:
            continue
        key = (it, _num(r.get("step")))
        if key not in index:
            index[key] = len(points)
            points.append({"iter": it, "step": _num(r.get("step")), "time": _num(r.get("time")),
                           "stage": r.get("stage")})
        i = index[key]
        scope = r.get("scope") or "global"
        fam = r.get("family") or "_all"
        for col, cell in r.items():
            if col in id_cols:
                continue
            v = _num(cell)
            if v is None:
                continue
            name = f"{scope}/{fam}/{col}"
            series.setdefault(name, {})[i] = v
        nn = _num(r.get("n"))
        if nn is not None:
            counts[f"{scope}/{fam}"] = int(nn)
    m = len(points)
    return {"points": {c: [p[c] for p in points] for c in ("iter", "step", "time")},
            "stage": [p["stage"] for p in points],
            "series": {k: [d.get(i) for i in range(m)] for k, d in sorted(series.items())},
            "counts": counts}


def _downsample_idx(n: int, max_points: int) -> list[int] | None:
    if max_points <= 0 or n <= max_points:
        return None
    step = n / max_points
    idx = sorted({int(i * step) for i in range(max_points)} | {n - 1})
    return idx


def _pick(col, idx):
    return col if idx is None else [col[i] for i in idx]


# --------------------------------------------------------------------------- runs
def _log_path(base: Path, name: str) -> Path | None:
    for ext in (".log", ".out"):
        p = base / (name + ext)
        if p.exists():
            return p
    return None


def _parse_log(path: Path) -> dict:
    info = {"planned_stages": None, "finished": False, "final_checkpoint": None}
    try:
        with open(path, "rb") as f:
            head = f.read(64_000).decode("utf-8", "replace")
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64_000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return info
    m = re.search(r"^stages: (\[.*\])\s*$", head, re.M)
    if m:
        try:
            info["planned_stages"] = [str(s) for s in ast.literal_eval(m.group(1))]
        except (ValueError, SyntaxError):
            pass
    m = re.search(r"^final checkpoint: (.+)$", tail, re.M)
    if m:
        info["finished"] = True
        info["final_checkpoint"] = m.group(1).strip()
    if re.search(r"^Traceback \(most recent call last\)", tail, re.M) and not info["finished"]:
        info["crashed"] = True
    return info


def _recent_mean(col, k=10):
    vals = [v for v in col[-k * 3:] if v is not None][-k:]
    return sum(vals) / len(vals) if vals else None


def _spark(col, points=60):
    vals = [v for v in col if v is not None]
    if not vals:
        return []
    if len(vals) <= points:
        return [round(v, 4) for v in vals]
    b = len(vals) / points
    return [round(sum(vals[int(i * b):max(int(i * b) + 1, int((i + 1) * b))]) /
                  max(1, len(vals[int(i * b):max(int(i * b) + 1, int((i + 1) * b))])), 4) for i in range(points)]


def _run_files(base: Path):
    for p in sorted(base.glob("*.csv")):
        if p.name.endswith("_val.csv"):
            continue
        yield p


def run_summary(source: str, base: Path, csv_path: Path) -> dict:
    name = csv_path.stem
    d = _cached("run", csv_path, parse_run_csv)
    st = csv_path.stat()
    now = time.time()
    xs, mets = d["x"], d["metrics"]

    def last(col):
        for v in reversed(col or []):
            if v is not None:
                return v
        return None

    log = _log_path(base, name)
    loginfo = _cached("log", log, _parse_log) if log else {}
    finished = bool(loginfo.get("finished")) or (_ckpt_base(source) / f"{name}_final.pt").exists()
    sync = _sync_state(source)
    if source == "local":
        age = age_now = now - st.st_mtime
    else:  # judge against the remote clock at the last successful sync (no clock skew)
        ref = sync.get("last_ok_remote_now") or sync.get("last_ok") or now
        age = ref - st.st_mtime
        age_now = age + max(0.0, now - (sync.get("last_ok") or now))
    if finished:
        status = "finished"
    elif loginfo.get("crashed"):
        status = "crashed"
    elif age < RUNNING_WINDOW:
        status = "running"
    else:
        status = "idle"
    stage = last(d["stage"])
    planned = loginfo.get("planned_stages")
    seen = list(dict.fromkeys(t["stage"] for t in d["transitions"]))
    sr = mets.get("solve_rate") or []
    val_path = base / f"{name}_val.csv"
    val_last = None
    if val_path.exists():
        try:
            v = _cached("val", val_path, parse_val_csv)
            col = v["series"].get("global/_macro/greedy") or []
            val_last = last(col)
        except Exception:  # noqa: BLE001
            pass
    stage_idx = None
    if planned and stage in planned:
        stage_idx = planned.index(stage)
    elif stage in seen:
        stage_idx = seen.index(stage)
    return {
        "id": f"{source}/{name}", "source": source, "name": name, "status": status,
        "rows": d["n"], "iteration": last(xs.get("iter")), "step": last(xs.get("step")),
        "elapsed": last(xs.get("time")), "stage": stage, "stage_index": stage_idx,
        "stages_planned": planned, "stages_seen": seen,
        "solve_rate": _recent_mean(sr), "solve_rate_best": max((v for v in sr if v is not None), default=None),
        "val_macro": val_last, "has_val": val_path.exists(), "spark": _spark(sr),
        "mtime": st.st_mtime, "age": age_now, "age_at_sync": age, "size": st.st_size, "log": log.name if log else None,
        "synced_at": sync.get("last_ok") if source != "local" else None,
        "columns": sorted(mets),
    }


@router.get("/api/dashboard/runs")
def list_runs():
    runs = []
    for source, base in _sources().items():
        for p in _run_files(base):
            try:
                runs.append(run_summary(source, base, p))
            except Exception as e:  # noqa: BLE001  a half-written / foreign CSV must not break the list
                runs.append({"id": f"{source}/{p.stem}", "source": source, "name": p.stem, "status": "error",
                             "error": f"{type(e).__name__}: {e}", "mtime": p.stat().st_mtime, "spark": []})
    order = {"running": 0, "idle": 1, "crashed": 1, "finished": 2, "error": 3}
    runs.sort(key=lambda r: (order.get(r["status"], 3), -r.get("mtime", 0)))
    return {"runs": runs, "now": time.time(), "sync": SYNC.status()}


def _run_base(source: str, name: str) -> tuple[Path, Path]:
    _safe(source), _safe(name)
    srcs = _sources()
    if source not in srcs:
        raise HTTPException(404, f"unknown source {source!r}")
    p = srcs[source] / f"{name}.csv"
    if not p.is_file():
        raise HTTPException(404, f"no run {source}/{name}")
    return srcs[source], p


@router.get("/api/dashboard/runs/{source}/{name}")
def run_series(source: str, name: str, max_points: int = Query(2500, ge=0, le=100_000)):
    base, p = _run_base(source, name)
    d = _cached("run", p, parse_run_csv)
    idx = _downsample_idx(d["n"], max_points)
    out = {"id": f"{source}/{name}", "n": d["n"], "downsampled": idx is not None,
           "x": {k: _pick(v, idx) for k, v in d["x"].items()},
           "metrics": {k: _pick(v, idx) for k, v in d["metrics"].items()},
           "transitions": d["transitions"], "val": None}
    vp = base / f"{name}_val.csv"
    if vp.exists():
        try:
            out["val"] = _cached("val", vp, parse_val_csv)
        except Exception as e:  # noqa: BLE001
            out["val_error"] = str(e)
    return out


@router.get("/api/dashboard/runs/{source}/{name}/log")
def run_log(source: str, name: str, lines: int = Query(60, ge=1, le=2000)):
    base, _ = _run_base(source, name)
    log = _log_path(base, name)
    if not log:
        return {"name": None, "lines": []}
    with open(log, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400 * lines))
        txt = f.read().decode("utf-8", "replace")
    return {"name": log.name, "lines": txt.splitlines()[-lines:], "mtime": log.stat().st_mtime}


# --------------------------------------------------------------------------- benchmark / eval
_AGG_KEYS = ("solve_rate", "median_ms", "p90_ms", "mean_expansions", "mean_nn_calls")


def _aggregate_rows(rows: list) -> dict:
    """Fallback when a report has per-puzzle rows but no summary."""
    groups: dict[str, list] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = f"{r.get('label') or r.get('ckpt') or 'model'}/{r.get('method', '?')}"
        groups.setdefault(key, []).append(r)

    def agg(rs):
        if not rs:
            return {}
        secs = sorted(float(r.get("seconds") or 0) * 1000 for r in rs)
        return {"count": len(rs), "solve_rate": sum(bool(r.get("solved")) for r in rs) / len(rs),
                "median_ms": secs[len(secs) // 2], "p90_ms": secs[min(len(secs) - 1, int(0.9 * len(secs)))],
                "mean_expansions": sum(float(r.get("expansions") or 0) for r in rs) / len(rs),
                "mean_nn_calls": sum(float(r.get("nn_calls") or 0) for r in rs) / len(rs)}
    out = {}
    for key, rs in groups.items():
        fams = list(dict.fromkeys(r.get("family") for r in rs if r.get("family")))
        dens = list(dict.fromkeys(r.get("density") for r in rs if r.get("density")))
        out[key] = {"label": key.split("/")[0], "method": key.split("/", 1)[1], "overall": agg(rs),
                    "by_family": {f: agg([r for r in rs if r.get("family") == f]) for f in fams},
                    "by_density": {d: agg([r for r in rs if r.get("density") == d]) for d in dens}}
    return out


def _load_bench(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raw = {"rows": raw} if isinstance(raw, list) else {}
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    summary = raw.get("summary")
    if not isinstance(summary, dict) or not summary:
        summary = _aggregate_rows(raw.get("rows") or [])
    clean = {}
    for key, m in summary.items():
        if not isinstance(m, dict):
            continue
        entry = {k: v for k, v in m.items() if k not in ("by_family", "by_density", "overall")}
        for part in ("overall", "by_family", "by_density"):
            val = m.get(part)
            if isinstance(val, dict):
                entry[part] = val
        entry.setdefault("overall", {})
        entry.setdefault("by_family", {})
        entry.setdefault("by_density", {})
        clean[str(key)] = entry
    meta = {k: v for k, v in meta.items() if not isinstance(v, (dict,)) or len(json.dumps(v, default=str)) < 4000}
    return {"meta": json.loads(json.dumps(meta, default=str)), "summary": clean,
            "rows": len(raw.get("rows") or [])}


@router.get("/api/dashboard/bench")
def list_bench():
    reports = []
    for source, base in _sources().items():
        bdir = base / "bench"
        if not bdir.is_dir():
            continue
        for p in sorted(bdir.glob("*.json")):
            item = {"id": f"{source}/{p.stem}", "source": source, "name": p.stem, "mtime": p.stat().st_mtime}
            try:
                item.update(_cached("bench", p, _load_bench))
            except Exception as e:  # noqa: BLE001  (file being written, other schema)
                item["error"] = f"{type(e).__name__}: {e}"
            reports.append(item)
    reports.sort(key=lambda r: -r["mtime"])
    return {"reports": reports}


def _json_docs(text: str) -> list:
    """Every top-level JSON value in ``text`` (plain JSON, or captured stdout with
    other lines such as "generator: ..." around the documents)."""
    try:
        return [json.loads(text)]
    except ValueError:
        pass
    dec, docs, pos = json.JSONDecoder(), [], 0
    for m in re.finditer(r"^[\[{]", text, re.M):
        if m.start() < pos:
            continue
        try:
            obj, end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        docs.append(obj)
        pos = end
    return docs


def _load_eval(path: Path) -> list[dict]:
    items = []
    for doc in _json_docs(path.read_text(encoding="utf-8", errors="replace")):
        items.extend(doc if isinstance(doc, list) else [doc])
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        row = {}
        for k, v in it.items():
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, (int, float)) and math.isfinite(v):
                row[k] = v
            elif isinstance(v, str) and len(v) < 200:
                row[k] = v
        out.append(row)
    return out


@router.get("/api/dashboard/eval")
def list_eval():
    rows, cols = [], {}
    for source, base in _sources().items():
        edir = base / "eval"
        if not edir.is_dir():
            continue
        for p in sorted(edir.glob("*.json")):
            try:
                items = _cached("eval", p, _load_eval)
            except Exception:  # noqa: BLE001
                continue
            for it in items:
                row = {"source": source, "file": p.stem, "mtime": p.stat().st_mtime, **it}
                row.setdefault("spec", p.stem)
                rows.append(row)
                for k, v in it.items():
                    if isinstance(v, (int, float)):
                        cols[k] = True
    return {"rows": rows, "columns": list(cols)}


# --------------------------------------------------------------------------- checkpoints
_DROP = re.compile(r"state_dict|optimizer|training_state|rng|scaler|replay", re.I)


def _jsonable(v, depth=0):
    if depth > 4:
        return str(type(v).__name__)
    if v is None or isinstance(v, (bool, int, str)):
        return v[:400] if isinstance(v, str) else v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        out = {}
        for i, (k, x) in enumerate(v.items()):
            if i >= 80:
                out["..."] = f"{len(v) - 80} more"
                break
            if _DROP.search(str(k)):
                continue
            out[str(k)] = _jsonable(x, depth + 1)
        return out
    if isinstance(v, (list, tuple)):
        items = [_jsonable(x, depth + 1) for x in list(v)[:60]]
        return items
    shape = getattr(v, "shape", None)
    if shape is not None:
        try:
            if getattr(v, "numel", lambda: 2)() == 1:
                return float(v.item())
        except Exception:  # noqa: BLE001
            pass
        return f"tensor{list(shape)}"
    return str(v)[:200]


def _val_score(meta: dict):
    val = meta.get("val")
    if isinstance(val, dict):
        fams = val.get("families")
        if isinstance(fams, dict) and fams:
            g = [f.get("greedy") for f in fams.values() if isinstance(f, dict) and f.get("greedy") is not None]
            if g:
                return sum(g) / len(g)
        for k in ("val_greedy", "greedy", "score"):
            if isinstance(val.get(k), (int, float)):
                return float(val[k])
    cs = meta.get("curriculum_state")
    if isinstance(cs, dict) and isinstance(cs.get("best_score"), (int, float)) and cs["best_score"] >= 0:
        return float(cs["best_score"])
    imi = meta.get("imitation")
    if isinstance(imi, dict) and isinstance(imi.get("val"), dict):
        return imi["val"].get("val_greedy")
    return None


def _load_ckpt_meta(path: Path) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        return {"error": "not a dict checkpoint"}
    meta = {k: v for k, v in ck.items() if not _DROP.search(str(k))}
    out = _jsonable(meta)
    cfg = ck.get("config") if isinstance(ck.get("config"), dict) else {}
    params = None
    sd = ck.get("state_dict")
    if isinstance(sd, dict):
        try:
            params = int(sum(t.numel() for t in sd.values() if hasattr(t, "numel")))
        except Exception:  # noqa: BLE001
            params = None
    out["_summary"] = {
        "stage": ck.get("stage"), "iteration": ck.get("iteration"), "global_step": ck.get("global_step"),
        "hidden": cfg.get("hidden"), "layers": cfg.get("layers"),
        "feature_version": ck.get("feature_version", cfg.get("feature_version")),
        "head_version": ck.get("head_version", cfg.get("head_version")),
        "params": params, "val_score": _val_score(ck), "pretrained": bool(ck.get("pretrained")),
        "stages": ck.get("stages") if isinstance(ck.get("stages"), list) else None,
    }
    return out


def _ckpt_files():
    out = []
    base = CFG.ckpt_dir
    if base.is_dir():
        for p in base.glob("*.pt"):
            out.append(("local", p))
        rem = base / "remote"
        if rem.is_dir():
            for d in sorted(rem.iterdir()):
                if d.is_dir():
                    for p in d.glob("*.pt"):
                        out.append((d.name, p))
    return out


def _ckpt_path(name: str) -> Path:
    parts = name.split("/")
    if not name or any(not _SAFE.match(x) or x.startswith(".") for x in parts) or len(parts) not in (1, 3) \
            or (len(parts) == 3 and parts[0] != "remote") or not name.endswith(".pt"):
        raise HTTPException(400, f"bad checkpoint name {name!r}")
    p = CFG.ckpt_dir.joinpath(*parts)
    if not p.is_file():
        raise HTTPException(404, f"no checkpoint {name!r}")
    return p


@router.get("/api/dashboard/checkpoints")
def list_checkpoints():
    items = []
    for source, p in _ckpt_files():
        st = p.stat()
        m = _CKPT_KIND.match(p.stem)
        rel = p.relative_to(CFG.ckpt_dir).as_posix()
        item = {"name": rel, "file": p.name, "source": source, "run": m.group(1) if m else p.stem,
                "kind": m.group(2) if m else "other", "size": st.st_size, "mtime": st.st_mtime, "meta": None}
        hit = _CACHE.get(("ckpt", str(p)))
        if hit and hit[0] == (st.st_mtime, st.st_size):
            item["meta"] = hit[1].get("_summary")
            item["meta_error"] = hit[1].get("error")
        items.append(item)
    items.sort(key=lambda r: -r["mtime"])
    return {"checkpoints": items}


_CKPT_LOCK = threading.Lock()


@router.get("/api/dashboard/checkpoints/meta")
def checkpoint_meta(name: str, full: bool = False):
    p = _ckpt_path(name)
    with _CKPT_LOCK:  # torch.load is heavy; one at a time
        try:
            meta = _cached("ckpt", p, _load_ckpt_meta)
        except Exception as e:  # noqa: BLE001  (partially written checkpoint, foreign file)
            raise HTTPException(503, f"could not read checkpoint: {type(e).__name__}: {e}")
    return {"name": name, "summary": meta.get("_summary"), **({"meta": meta} if full else {})}


# --------------------------------------------------------------------------- sync
def _default_sync_fn():
    path = CFG.root / "scripts" / "sync_runs.py"
    if not path.exists():
        path = ROOT / "scripts" / "sync_runs.py"
    spec = importlib.util.spec_from_file_location("_zip_sync_runs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sync


class SyncManager:
    """One sync at a time, in a daemon thread; debounced; optional periodic auto-sync."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.last: dict | None = None
        self.started: float | None = None
        self.auto_minutes = 0.0
        self._auto_stop = threading.Event()
        self._auto_thread: threading.Thread | None = None

    def status(self) -> dict:
        running = bool(self.thread and self.thread.is_alive())
        last = self.last
        if last is None:
            st = _sync_state(CFG.sync_host)
            last = st or None
        return {"host": CFG.sync_host, "running": running, "started_at": self.started if running else None,
                "last": last, "auto_minutes": self.auto_minutes}

    def _work(self):
        try:
            fn = CFG.sync_fn or _default_sync_fn()
            res = fn(host=CFG.sync_host, root=CFG.root, timeout=CFG.sync_timeout)
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "host": CFG.sync_host}
        res.setdefault("finished", time.time())
        self.last = res

    def start(self, force: bool = False) -> dict:
        with self.lock:
            if self.thread and self.thread.is_alive():
                return {"started": False, "reason": "already running", **self.status()}
            if not force and self.last and self.last.get("ok") and \
                    time.time() - self.last.get("finished", 0) < CFG.sync_debounce:
                return {"started": False, "reason": "debounced", **self.status()}
            self.started = time.time()
            self.thread = threading.Thread(target=self._work, name="dashboard-sync", daemon=True)
            self.thread.start()
        return {"started": True, **self.status()}

    def wait(self, timeout=None):
        t = self.thread
        if t:
            t.join(timeout)

    def set_auto(self, minutes: float):
        self.auto_minutes = max(0.0, float(minutes))
        self._auto_stop.set()
        if self._auto_thread and self._auto_thread.is_alive():
            self._auto_thread.join(2)
        self._auto_stop = threading.Event()
        if self.auto_minutes > 0:
            stop, period = self._auto_stop, self.auto_minutes * 60

            def loop():
                while not stop.wait(period):
                    self.start()
            self._auto_thread = threading.Thread(target=loop, name="dashboard-autosync", daemon=True)
            self._auto_thread.start()
        return self.status()


SYNC = SyncManager()


class SyncReq(BaseModel):
    force: bool = False
    wait: float = 0.0  # seconds to wait for completion (0 = return immediately)


class AutoReq(BaseModel):
    minutes: float = 0.0


@router.get("/api/dashboard/sync")
def sync_status():
    return SYNC.status()


@router.post("/api/dashboard/sync")
def sync_start(req: SyncReq | None = None):
    req = req or SyncReq()
    out = SYNC.start(force=req.force)
    if req.wait > 0 and out.get("started"):
        SYNC.wait(min(req.wait, 60.0))
        out = {"started": True, **SYNC.status()}
    return out


@router.post("/api/dashboard/sync/auto")
def sync_auto(req: AutoReq):
    if req.minutes and req.minutes < 1:
        raise HTTPException(400, "minutes must be 0 (off) or >= 1")
    return SYNC.set_auto(min(req.minutes, 24 * 60))


__all__ = ["router", "configure", "parse_run_csv", "parse_val_csv", "SYNC", "CFG"]
