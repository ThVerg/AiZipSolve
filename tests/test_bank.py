"""Curated puzzle bank: server integration (with a tiny temporary bank) and the shipped bank's format."""
import json
from pathlib import Path

import pytest

from zipsolve.generator import make_puzzle
from zipsolve.puzzle import Puzzle
from zipsolve.solver import count_solutions

BANK = Path(__file__).resolve().parents[1] / "zipsolve" / "app" / "static" / "bank"


def _entry(mode, diff, seed, kind="grid2d", size=4):
    p = make_puzzle(kind, size, rng=seed, unique=True)
    d = p.to_dict()
    return {"id": f"{mode[0]}{seed:09x}", "mode": mode, "diff": diff, "label": "Easy", "score": 10.0 + seed,
            "size": f"{size}x{size}", "puzzle": d}


@pytest.fixture(scope="module")
def tiny_bank(tmp_path_factory):
    root = tmp_path_factory.mktemp("bank")
    (root / "pools").mkdir()
    (root / "daily").mkdir()
    pools = {("classic", "easy"): [_entry("classic", "easy", s) for s in (1, 2, 3)],
             ("walls", "medium"): [_entry("walls", "medium", 4, "walls", 5)]}
    idx = {"version": 1, "generated": "2026-01-01T00:00:00+00:00", "daily_start": "2026-01-01",
           "daily_end": "2026-01-31", "modes": {}}
    for (m, d), lst in pools.items():
        fn = f"pools/{m}-{d}.json"
        (root / fn).write_text(json.dumps(lst))
        idx["modes"].setdefault(m, {})[d] = {"file": fn, "count": len(lst)}
    day = {"number": 4, "easy": _entry("classic", "easy", 7), "medium": _entry("classic", "medium", 8),
           "hard": _entry("classic", "hard", 9), "special": _entry("walls", "hard", 10, "walls", 5)}
    (root / "daily" / "2026-01.json").write_text(json.dumps({"2026-01-04": day}))
    (root / "index.json").write_text(json.dumps(idx))
    return root, pools, day


@pytest.fixture(scope="module")
def client(tiny_bank):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from zipsolve.app.server import create_app
    return TestClient(create_app(bank_dir=tiny_bank[0]))


def test_generate_from_bank(client, tiny_bank):
    _, pools, _ = tiny_bank
    lst = pools[("classic", "easy")]
    r = client.post("/api/generate", json={"kind": "grid2d", "size": 5, "seed": 4, "bank": "classic-easy"}).json()
    e = lst[4 % len(lst)]
    assert r["source"] == "bank" and r["bank_id"] == e["id"] and r["seed"] == 4
    assert r["puzzle"]["checkpoints"] == e["puzzle"]["checkpoints"] and "solution" not in r["puzzle"]
    assert r["rating"]["label"] == "Easy" and r["rating"]["score"] == e["score"]
    # solution-backed hints for bank puzzles
    P = r["puzzle"]
    h = client.post("/api/hint", json={"puzzle": P, "path": [P["checkpoints"][0]], "id": r["id"]}).json()
    assert h["status"] == "next" and h["source"] == "solution" and h["next"] == e["puzzle"]["solution"][1]
    # unknown pool -> the generator, rated
    g = client.post("/api/generate", json={"kind": "grid2d", "size": 5, "seed": 4, "unique": True,
                                           "bank": "classic-insane"}).json()
    assert g["source"] == "generator" and g["rating"]["label"] in ("Easy", "Medium", "Hard", "Expert", "Insane")


def test_daily_from_bank(client, tiny_bank):
    _, _, day = tiny_bank
    for diff in ("easy", "medium", "hard", "special"):
        r = client.get("/api/daily", params={"difficulty": diff, "date": "2026-01-04"}).json()
        assert r["source"] == "bank" and r["bank_id"] == day[diff]["id"]
        assert r["daily"]["number"] == 4 and r["daily"]["difficulty"] == diff
        assert r["puzzle"]["edges"] == day[diff]["puzzle"]["edges"]
    # a date outside the bank -> generated as before (still deterministic)
    a = client.get("/api/daily", params={"difficulty": "easy", "date": "2026-03-01"}).json()
    b = client.get("/api/daily", params={"difficulty": "easy", "date": "2026-03-01"}).json()
    assert a["source"] == "generator" and a["puzzle"] == b["puzzle"] and a["daily"]["number"] == 60
    info = client.get("/api/bank").json()
    assert info["available"] and info["modes"]["classic"]["easy"] == 3


def test_no_bank_falls_back(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from zipsolve.app.server import create_app
    c = TestClient(create_app(bank_dir=tmp_path / "missing"))
    assert c.get("/api/bank").json() == {"available": False}
    r = c.post("/api/generate", json={"kind": "grid2d", "size": 4, "seed": 1, "bank": "classic-easy"}).json()
    assert r["source"] == "generator"
    d = c.get("/api/daily", params={"difficulty": "medium", "date": "2026-01-04"}).json()
    assert d["source"] == "generator" and d["daily"]["number"] == 4


# ---- the shipped bank ----------------------------------------------------------------------
shipped = pytest.mark.skipif(not (BANK / "index.json").exists(), reason="bank not built")


@shipped
def test_shipped_bank_format():
    idx = json.loads((BANK / "index.json").read_text())
    assert idx["version"] == 1 and idx["daily_start"] == "2026-01-01" and idx["daily_end"] == "2027-12-31"
    ids = set()
    for mode in ("classic", "walls", "islands", "cube"):
        for diff in ("easy", "medium", "hard"):
            info = idx["modes"][mode][diff]
            lst = json.loads((BANK / info["file"]).read_text())
            assert len(lst) == info["count"] > 0
            for e in lst:
                assert e["mode"] == mode and e["diff"] == diff and e["id"] not in ids
                ids.add(e["id"])
            e = lst[0]
            p = Puzzle.from_dict(e["puzzle"])
            assert p.is_valid_solution(e["puzzle"]["solution"])
            assert (BANK / "robots" / f"{e['id']}.json").exists()
    jan = json.loads((BANK / "daily" / "2026-01.json").read_text())
    assert len(jan) == 31 and jan["2026-01-01"]["number"] == 1 and jan["2026-01-31"]["number"] == 31
    assert jan["2026-01-04"]["special"] is not None and jan["2026-01-05"]["special"] is None   # Sunday
    dec = json.loads((BANK / "daily" / "2027-12.json").read_text())
    assert dec["2027-12-31"]["number"] == 730
    rec = jan["2026-01-01"]
    for diff in ("easy", "medium", "hard"):
        e = rec[diff]
        assert e["mode"] == "classic" and e["id"] not in ids
        p = Puzzle.from_dict(e["puzzle"])
        assert count_solutions(p, 2) == (1, "complete")
        runs = json.loads((BANK / "robots" / f"{e['id']}.json").read_text())
        assert set(runs) >= {"rookie", "scout", "grandmaster", "tortoise"}
        t = runs["tortoise"]
        assert t["solved"] and t["path"] == e["puzzle"]["solution"] and isinstance(t["trace"], list)
