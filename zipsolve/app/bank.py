"""Read access to the curated puzzle bank (static/bank/, built by scripts/build_bank.py).

The local app and the static site serve the same puzzles from it: difficulty
picks are ``pool[seed % len(pool)]`` and the daily puzzle for a date is the
bank's daily record for that date.  Everything degrades gracefully: with no
bank (or a missing file) the lookups return None and the server falls back to
the random generator.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from threading import Lock

DEFAULT_DIR = Path(__file__).parent / "static" / "bank"
DIFFS = ("easy", "medium", "hard")
SLOTS = (*DIFFS, "special")


class Bank:
    def __init__(self, directory: str | Path = DEFAULT_DIR):
        self.dir = Path(directory)
        self._files: dict[str, object] = {}
        self._lock = Lock()

    def _load(self, rel: str):
        with self._lock:
            if rel in self._files:
                return self._files[rel]
        f = self.dir / rel
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            data = None
        with self._lock:
            if len(self._files) > 64:          # month files: keep the cache small
                for k in [k for k in self._files if k.startswith("daily/")][:16]:
                    self._files.pop(k, None)
            self._files[rel] = data
        return data

    @property
    def index(self) -> dict | None:
        idx = self._load("index.json")
        return idx if isinstance(idx, dict) and idx.get("version") == 1 else None

    def available(self) -> bool:
        return self.index is not None

    def pool(self, mode: str, diff: str) -> list | None:
        idx = self.index
        if not idx:
            return None
        m = (idx.get("modes") or {}).get(mode) or {}
        d = m.get(diff)
        if not d:
            return None
        lst = self._load(d["file"])
        return lst if isinstance(lst, list) and lst else None

    def pick(self, mode: str, diff: str, seed: int) -> dict | None:
        lst = self.pool(mode, diff)
        if not lst:
            return None
        return lst[int(seed) % len(lst)]

    def daily(self, day: _dt.date, slot: str) -> tuple[dict, int] | None:
        """(entry, number) for the date's slot (easy/medium/hard/special), or None."""
        idx = self.index
        if not idx or slot not in SLOTS:
            return None
        try:
            lo = _dt.date.fromisoformat(idx["daily_start"])
            hi = _dt.date.fromisoformat(idx["daily_end"])
        except (KeyError, ValueError):
            return None
        if not lo <= day <= hi:
            return None
        recs = self._load(f"daily/{day.year:04d}-{day.month:02d}.json")
        if not isinstance(recs, dict):
            return None
        rec = recs.get(day.isoformat())
        if not rec or not rec.get(slot):
            return None
        return rec[slot], int(rec.get("number", (day - lo).days + 1))


def seed_of(entry: dict) -> int:
    """A stable integer for a bank entry (its id's hex digits)."""
    try:
        return int(entry["id"][1:], 16) % 10**9
    except (KeyError, ValueError):
        return 0
