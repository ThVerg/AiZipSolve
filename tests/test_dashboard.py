"""Training dashboard API (zipsolve/app/dashboard_api.py) + scripts/sync_runs.py."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zipsolve.app import dashboard_api as dash

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def env(tmp_path):
    saved = {k: getattr(dash.CFG, k) for k in ("root", "runs_dir", "ckpt_dir", "sync_host", "sync_fn", "sync_debounce")}
    calls = []

    def fake_sync(host, root, timeout):
        calls.append((host, Path(root)))
        time.sleep(0.05)
        return {"ok": True, "host": host, "method": "rsync", "changed_runs": ["x.csv"], "changed_checkpoints": [],
                "finished": time.time()}

    dash.configure(root=tmp_path, sync_host="testhost", sync_fn=fake_sync, sync_debounce=30)
    dash.SYNC.last = None
    (tmp_path / "runs").mkdir()
    (tmp_path / "checkpoints").mkdir()
    app = FastAPI()
    app.include_router(dash.router)
    yield TestClient(app), tmp_path, calls
    dash.SYNC.wait(5)
    dash.SYNC.set_auto(0)
    dash.SYNC.last = None
    for k, v in saved.items():
        setattr(dash.CFG, k, v)
    dash._CACHE.clear()


OLD_CSV = "iter,step,time,stage,solve_rate,ep_return,entropy\n" \
          "1,1024,3.0,grid2d:4,0.1,-0.5,0.4\n2,2048,6.0,grid2d:4,0.3,0.1,0.3\n3,3072,9.0,grid2d:5,0.2,0.0,0.3\n"

NEW_CSV = (
    'iter,step,time,stage,solve_rate,kl,grad_norm,family_sr,src_counts,note\n'
    '1,1024,46.3,grid2d:4,0.07,0.001,0.17,"{""grid2d:4"": [0.07, 200]}","{""current"": 175, ""replay"": 65}",warm\n'
    '2,2048,85.4,grid2d:4,nan,,0.2,"{""grid2d:4"": [0.1, 200]}","{""current"": 170}",\n'
    'iter,step,time,stage,solve_rate,kl,grad_norm,family_sr,src_counts,note\n'   # repeated header (appended log)
    '3,3072,120.0,grid2d:5+walls:5,0.2,0.01,0.3,"{""grid2d:5"": [0.2, 50], ""walls:5"": [0.3, 40]}",{},x,EXTRA\n'
    '4,4096,150.0,grid2d:5+walls:5\n'                                               # short row
)

VAL_CSV = ("iter,step,time,stage,scope,family,n,greedy,sampled\n"
           "5,5120,221.2,grid2d:4,global,grid2d:5,2,0.5,0.5\n"
           "5,5120,221.2,grid2d:4,global,_macro,4,0.25,0.5\n"
           "5,5120,221.2,grid2d:4,stage,grid2d:4,2,1.0,1.0\n"
           "10,10240,300.0,grid2d:4,global,grid2d:5,2,1.0,0.5\n"
           "10,10240,300.0,grid2d:4,global,_macro,4,0.75,0.5\n")


def test_parse_old_and_new_csv(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text(OLD_CSV)
    d = dash.parse_run_csv(p)
    assert d["n"] == 3 and d["x"]["iter"] == [1, 2, 3] and d["x"]["time"] == [3.0, 6.0, 9.0]
    assert set(d["metrics"]) == {"solve_rate", "ep_return", "entropy"}
    assert [t["stage"] for t in d["transitions"]] == ["grid2d:4", "grid2d:5"]
    assert d["transitions"][1]["iter"] == 3

    p = tmp_path / "new.csv"
    p.write_text(NEW_CSV)
    d = dash.parse_run_csv(p)
    assert d["n"] == 4
    m = d["metrics"]
    assert m["solve_rate"] == [0.07, None, 0.2, None]          # nan -> None, short row padded
    assert m["kl"] == [0.001, None, 0.01, None]
    assert m["family_sr.grid2d:4"] == [0.07, 0.1, None, None]  # JSON dicts flattened ([rate, n] -> rate)
    assert m["family_sr.walls:5"] == [None, None, 0.3, None]
    assert m["src_counts.replay"] == [65, None, None, None]
    assert d["text"]["note"] == ["warm", None, "x", None]
    assert all(len(v) == 4 for v in m.values())
    assert [t["stage"] for t in d["transitions"]] == ["grid2d:4", "grid2d:5+walls:5"]


def test_parse_val_csv(tmp_path):
    p = tmp_path / "r_val.csv"
    p.write_text(VAL_CSV)
    v = dash.parse_val_csv(p)
    assert v["points"]["iter"] == [5, 10]
    assert v["series"]["global/_macro/greedy"] == [0.25, 0.75]
    assert v["series"]["stage/grid2d:4/greedy"] == [1.0, None]
    assert v["counts"]["global/_macro"] == 4


def test_run_listing_and_series(env):
    client, root, _ = env
    runs = root / "runs"
    (runs / "done.csv").write_text(OLD_CSV)
    (runs / "done.log").write_text("stages: ['grid2d:4', 'grid2d:5', 'grid2d:6']\n...\nfinal checkpoint: checkpoints/done_final.pt\n")
    (runs / "live.csv").write_text(NEW_CSV)
    (runs / "live_val.csv").write_text(VAL_CSV)
    (runs / "broken.csv").write_bytes(b"")
    old = time.time() - 3600
    os.utime(runs / "done.csv", (old, old))
    rem = runs / "remote" / "testhost"
    rem.mkdir(parents=True)
    (rem / "far.csv").write_text(OLD_CSV)
    (rem / "far.out").write_text("stages: ['grid2d:4', 'grid2d:5']\niter=1 ...\n")
    now = time.time()
    (rem / ".sync.json").write_text(json.dumps({"ok": True, "last_ok": now, "last_ok_remote_now": now + 5}))

    r = client.get("/api/dashboard/runs").json()
    by = {x["id"]: x for x in r["runs"]}
    assert set(by) == {"local/done", "local/live", "local/broken", "testhost/far"}
    assert by["local/done"]["status"] == "finished"
    assert by["local/done"]["stage_index"] == 1 and by["local/done"]["stages_planned"][2] == "grid2d:6"
    assert by["local/live"]["status"] == "running"
    assert by["local/live"]["val_macro"] == 0.75 and by["local/live"]["has_val"]
    assert by["testhost/far"]["status"] == "running" and by["testhost/far"]["stage_index"] == 1
    assert by["local/done"]["spark"] == [0.1, 0.3, 0.2]
    assert r["runs"][0]["status"] == "running"  # running runs first

    s = client.get("/api/dashboard/runs/local/live").json()
    assert s["n"] == 4 and "family_sr.walls:5" in s["metrics"]
    assert s["val"]["series"]["global/_macro/greedy"] == [0.25, 0.75]
    s = client.get("/api/dashboard/runs/local/done", params={"max_points": 2}).json()
    assert s["downsampled"] and s["x"]["iter"][-1] == 3 and len(s["transitions"]) == 2
    assert client.get("/api/dashboard/runs/testhost/far/log").json()["lines"][0].startswith("stages")
    assert client.get("/api/dashboard/runs/local/nope").status_code == 404
    assert client.get("/api/dashboard/runs/local/..%2Fx").status_code in (400, 404)
    assert client.get("/api/dashboard/runs/elsewhere/done").status_code == 404


def test_bench_and_eval_tolerant(env):
    client, root, _ = env
    b = root / "runs" / "bench"
    b.mkdir()
    summary = {"m/greedy": {"label": "m", "method": "greedy", "overall": {"solve_rate": 0.5, "count": 2},
                            "by_family": {"grid2d:5": {"solve_rate": 0.5}}, "by_density": {}}}
    (b / "full.json").write_text(json.dumps({"meta": {"set": "val"}, "summary": summary, "rows": [{"x": 1}] * 5}))
    rows = [{"label": "m", "method": "search", "family": "grid2d:5", "density": "sparse", "solved": True, "seconds": 0.01},
            {"label": "m", "method": "search", "family": "walls:6", "density": "dense", "solved": False, "seconds": 0.03}]
    (b / "rowsonly.json").write_text(json.dumps({"rows": rows}))
    (b / "garbage.json").write_text("{not json")
    rep = {r["name"]: r for r in client.get("/api/dashboard/bench").json()["reports"]}
    assert rep["full"]["summary"]["m/greedy"]["overall"]["solve_rate"] == 0.5 and rep["full"]["rows"] == 5
    agg = rep["rowsonly"]["summary"]["m/search"]
    assert agg["overall"]["solve_rate"] == 0.5 and agg["by_family"]["walls:6"]["solve_rate"] == 0.0
    assert "error" in rep["garbage"]

    e = root / "runs" / "eval"
    e.mkdir()
    (e / "grid2d:4.json").write_text('generator: zipsolve.generator\n{\n "spec": "grid2d:4",\n "greedy_solve_rate": 0.9\n}\n')
    (e / "list.json").write_text(json.dumps([{"spec": "walls:6", "greedy_solve_rate": 0.5, "nested": {"a": 1}}]))
    ev = client.get("/api/dashboard/eval").json()
    assert {r["spec"] for r in ev["rows"]} == {"grid2d:4", "walls:6"}
    assert "greedy_solve_rate" in ev["columns"] and "nested" not in ev["columns"]


def test_checkpoint_meta_cache(env, monkeypatch):
    torch = pytest.importorskip("torch")
    client, root, _ = env
    ck = root / "checkpoints"
    torch.save({"config": {"hidden": 8, "layers": 2}, "stage": "grid2d:5", "iteration": 7, "global_step": 700,
                "state_dict": {"w": torch.zeros(3, 4)}, "training_state": {"optimizer": {"big": torch.zeros(10)}},
                "curriculum_state": {"best_score": 0.42}}, ck / "run1_best.pt")
    (ck / "remote" / "h").mkdir(parents=True)
    torch.save({"config": {}, "stage": "s"}, ck / "remote" / "h" / "run2_latest.pt")

    loads = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: loads.append(a[0]) or real(*a, **k))

    lst = client.get("/api/dashboard/checkpoints").json()["checkpoints"]
    assert {c["name"] for c in lst} == {"run1_best.pt", "remote/h/run2_latest.pt"}
    assert all(c["meta"] is None for c in lst) and not loads          # listing never loads torch files
    one = next(c for c in lst if c["name"] == "run1_best.pt")
    assert one["run"] == "run1" and one["kind"] == "best"

    m = client.get("/api/dashboard/checkpoints/meta", params={"name": "run1_best.pt"}).json()
    assert m["summary"]["stage"] == "grid2d:5" and m["summary"]["params"] == 12 and m["summary"]["val_score"] == 0.42
    client.get("/api/dashboard/checkpoints/meta", params={"name": "run1_best.pt"})
    assert len(loads) == 1                                               # cached by path + mtime + size
    full = client.get("/api/dashboard/checkpoints/meta", params={"name": "run1_best.pt", "full": True}).json()
    assert "state_dict" not in full["meta"] and "training_state" not in full["meta"]
    lst = client.get("/api/dashboard/checkpoints").json()["checkpoints"]
    assert next(c for c in lst if c["name"] == "run1_best.pt")["meta"]["iteration"] == 7

    t = time.time() + 5
    os.utime(ck / "run1_best.pt", (t, t))                               # changed file -> reloaded
    client.get("/api/dashboard/checkpoints/meta", params={"name": "run1_best.pt"})
    assert len(loads) == 2
    assert client.get("/api/dashboard/checkpoints/meta", params={"name": "remote/h/run2_latest.pt"}).status_code == 200
    for bad in ("../x.pt", "remote/../../x.pt", "/etc/passwd", "a/b.pt"):
        assert client.get("/api/dashboard/checkpoints/meta", params={"name": bad}).status_code == 400
    assert client.get("/api/dashboard/checkpoints/meta", params={"name": "nope.pt"}).status_code == 404


def test_sync_endpoint_debounce_and_auto(env):
    client, root, calls = env
    st = client.get("/api/dashboard/sync").json()
    assert st["host"] == "testhost" and not st["running"] and st["last"] is None
    r = client.post("/api/dashboard/sync", json={"wait": 5}).json()
    assert r["started"] and r["last"]["ok"] and calls == [("testhost", root)]
    r = client.post("/api/dashboard/sync", json={}).json()           # within the debounce window
    assert not r["started"] and r["reason"] == "debounced" and len(calls) == 1
    r = client.post("/api/dashboard/sync", json={"force": True, "wait": 5}).json()
    assert r["started"] and len(calls) == 2
    assert client.post("/api/dashboard/sync/auto", json={"minutes": 0.5}).status_code == 400
    assert client.post("/api/dashboard/sync/auto", json={"minutes": 5}).json()["auto_minutes"] == 5
    assert client.post("/api/dashboard/sync/auto", json={"minutes": 0}).json()["auto_minutes"] == 0


def test_sync_failure_is_reported(env):
    client, _, _ = env

    def boom(**kw):
        raise RuntimeError("no route to host")
    dash.configure(sync_fn=boom)
    r = client.post("/api/dashboard/sync", json={"wait": 5}).json()
    assert r["last"]["ok"] is False and "no route" in r["last"]["error"]


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_runs_t", ROOT / "scripts" / "sync_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sync_script_rsync_with_mocked_ssh(tmp_path, monkeypatch):
    mod = _load_sync_module()
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "ssh":
            return subprocess.CompletedProcess(cmd, 0, stdout="1790000000\nrsync\n", stderr="")
        assert cmd[0] == "rsync" and kw.get("timeout")
        assert any(c.startswith("--max-size=") for c in cmd)
        if cmd[-2].endswith("runs/"):
            assert "-r" in cmd and "--include=*.csv" in cmd and "--exclude=/remote/" in cmd
            out = ">f+++++++++ A.csv\n>f.st...... A.out\n.d..t...... ./\n"
        else:
            assert "-d" in cmd and "--include=*_best.pt" in cmd and "--include=*_latest.pt" in cmd
            out = ">f+++++++++ A_latest.pt\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(mod, "RUN", fake_run)
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/" + name)
    res = mod.sync("hostx", "Proj", tmp_path, timeout=30)
    assert res["ok"] and res["method"] == "rsync" and res["remote_now"] == 1790000000
    assert res["changed_runs"] == ["A.csv", "A.out"] and res["changed_checkpoints"] == ["A_latest.pt"]
    assert seen[1][-2:] == ["hostx:Proj/runs/", f"{tmp_path}/runs/remote/hostx/"]
    state = json.loads((tmp_path / "runs" / "remote" / "hostx" / ".sync.json").read_text())
    assert state["last_ok_remote_now"] == 1790000000

    def failing(cmd, **kw):
        if cmd[0] == "ssh":
            return subprocess.CompletedProcess(cmd, 255, stdout="", stderr="ssh: connect: Connection timed out")
        raise AssertionError("no rsync after failed probe")
    monkeypatch.setattr(mod, "RUN", failing)
    res = mod.sync("hostx", "Proj", tmp_path, timeout=30)
    assert not res["ok"] and "timed out" in res["error"]
    state = json.loads((tmp_path / "runs" / "remote" / "hostx" / ".sync.json").read_text())
    assert state["ok"] is False and state["last_ok_remote_now"] == 1790000000   # last good sync kept

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    monkeypatch.setattr(mod, "RUN", slow)
    res = mod.sync("hostx", "Proj", tmp_path, timeout=30)
    assert not res["ok"] and "timeout" in res["error"]


def test_sync_script_tar_fallback(tmp_path, monkeypatch):
    import io
    import tarfile
    mod = _load_sync_module()
    payload = b"iter,solve_rate\n1,0.5\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("A.csv")
        info.size = len(payload)
        info.mtime = 1700000000
        tf.addfile(info, io.BytesIO(payload))

    def fake_run(cmd, **kw):
        if cmd[0] == "ssh":
            remote = cmd[-1]
            if remote.startswith("date"):
                return subprocess.CompletedProcess(cmd, 0, stdout="1790000000\nno-rsync\n", stderr="")
            if "find" in remote:
                out = f"A.csv\t{len(payload)}\t1700000000.0\n" if "test -d runs" in remote else ""
                return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
            if "tar cf" in remote:
                kw["stdout"].write(buf.getvalue())
                return subprocess.CompletedProcess(cmd, 0, stdout=None, stderr=b"")
        return subprocess.run(cmd, **kw)  # local tar extract

    monkeypatch.setattr(mod, "RUN", fake_run)
    res = mod.sync("hostx", "Proj", tmp_path, timeout=30)
    assert res["ok"] and res["method"] == "tar" and res["changed_runs"] == ["A.csv"], res
    assert (tmp_path / "runs" / "remote" / "hostx" / "A.csv").read_bytes() == payload
    res = mod.sync("hostx", "Proj", tmp_path, timeout=30)          # unchanged (size + mtime) -> skipped
    assert res["ok"] and res["changed_runs"] == []
