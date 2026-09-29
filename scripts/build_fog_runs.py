"""Record fair fog-of-war runs for the static site (GitHub Pages).

Fog mode (play/modes.js: bank "classic-medium" with meta.fog switched on) has no Python server online, so
"Show me" and the robots replay a run recorded here: the fog planner (zipsolve.robots.fog) playing from the
1 with only the numbers it has seen - reveals, captions and back-ups included.

    python scripts/build_fog_runs.py [--pools classic-medium] [--time 20]

Writes zipsolve/app/static/bank/fog/<entry id>.json ({path, status, solved, trace, backtracks, stats,
seconds}) and adds  "fog": {"pools": [...], "dir": "fog", "count": N}  to bank/index.json (additive:
nothing else in the index changes). Rerun after rebuilding those pools; deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zipsolve.puzzle import Puzzle  # noqa: E402
from zipsolve.robots import fog  # noqa: E402

BANK = ROOT / "zipsolve" / "app" / "static" / "bank"
KEEP = ("path", "status", "solved", "trace", "backtracks", "stats", "seconds")


def record(entry: dict, time_limit: float) -> dict:
    d = dict(entry["puzzle"])
    d["meta"] = dict(d.get("meta") or {}, fog=True)
    sol = d.pop("solution", None)
    p = Puzzle.from_dict(dict(d, solution=None))
    r = fog.run(p, time_limit=time_limit, seed=0)
    if not r["solved"] or (sol and r["path"] != sol):
        raise SystemExit(f"{entry['id']}: fog run did not reach the solution ({r['status']})")
    out = {k: r[k] for k in KEEP}
    out["seconds"] = round(out["seconds"], 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", default=["classic-medium"], help="<mode>-<diff> pools fog mode uses")
    ap.add_argument("--time", type=float, default=20.0, help="time limit per run (s)")
    ap.add_argument("--bank", type=Path, default=BANK)
    a = ap.parse_args()
    idx_file = a.bank / "index.json"
    idx = json.loads(idx_file.read_text())
    outdir = a.bank / "fog"
    outdir.mkdir(exist_ok=True)
    n = back = 0
    for key in a.pools:
        mode, diff = key.split("-", 1)
        entries = json.loads((a.bank / idx["modes"][mode][diff]["file"]).read_text())
        for e in entries:
            run = record(e, a.time)
            (outdir / f"{e['id']}.json").write_text(json.dumps(run, separators=(",", ":")))
            n += 1
            back += run["stats"]["backtracks"] > 0
    idx["fog"] = {"pools": list(a.pools), "dir": "fog", "count": n}
    idx_file.write_text(json.dumps(idx, indent=1))
    size = sum(f.stat().st_size for f in outdir.glob("*.json"))
    print(f"{n} fog runs ({back} with back-ups), {size / 1e3:.0f} KB in {outdir}")


if __name__ == "__main__":
    main()
