"""Mirror training logs + selected checkpoints from a remote training host.

    python scripts/sync_runs.py                      # host faverg -> runs/remote/faverg, checkpoints/remote/faverg
    python scripts/sync_runs.py --host faverg --remote-dir FanisZipSolve --json

Read-only on the remote: only ``ssh <host> <read-only command>`` and rsync/tar
pulls FROM it. What is copied:

  runs/**/{*.csv,*.log,*.out,*.json,*.md}   (<= --max-run-mb each)  -> runs/remote/<host>/
  checkpoints/{*_best,*_final,*_latest}.pt  (<= --max-ckpt-mb each) -> checkpoints/remote/<host>/

rsync (both ends) is used when available: its quick-check (size + mtime) skips
unchanged files and remote mtimes are preserved, so the dashboard can tell
running from finished runs. Otherwise the fallback lists remote files with
``find -printf`` and pulls the changed ones through ``tar`` over ssh (mtimes
preserved by tar). A small ``.sync.json`` next to the mirrored runs records the
result (and the remote clock, used to judge "running" without clock-skew issues).

``sync(...)`` is importable (the dashboard API runs it in a thread); every
subprocess call goes through ``RUN`` (monkeypatched in tests) with a timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_PATTERNS = ["*.csv", "*.log", "*.out", "*.json", "*.md"]
CKPT_PATTERNS = ["*_best.pt", "*_final.pt", "*_latest.pt"]
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10"]
RUN = subprocess.run  # indirection for tests


class SyncError(RuntimeError):
    pass


def _run(cmd, timeout, **kw):
    try:
        return RUN(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired as e:
        raise SyncError(f"timeout after {timeout:.0f}s: {' '.join(map(str, cmd[:3]))} ...") from e
    except FileNotFoundError as e:
        raise SyncError(f"command not found: {cmd[0]}") from e


def _ssh(host, remote_cmd, timeout):
    r = _run(["ssh", *SSH_OPTS, host, remote_cmd], timeout)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip().splitlines()
        raise SyncError(f"ssh {host} failed ({r.returncode}): {msg[-1] if msg else ''}")
    return r.stdout


def _count_itemized(stdout: str) -> list[str]:
    """Files rsync actually transferred (``--itemize-changes`` lines starting with '>f')."""
    out = []
    for line in stdout.splitlines():
        if line.startswith(">f") and " " in line:
            out.append(line.split(" ", 1)[1].strip())
    return out


def _rsync(host, src, dst: Path, includes, max_mb, recursive, timeout):
    dst.mkdir(parents=True, exist_ok=True)
    ssh_e = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS)
    cmd = ["rsync", "-t", "-z", "--itemize-changes", "--timeout=30", f"--max-size={max_mb}m", "-e", ssh_e]
    if recursive:
        cmd += ["-r", "--prune-empty-dirs", "--exclude=/remote/", "--include=*/"]
    else:
        cmd += ["-d", "--exclude=*/"]
    cmd += [f"--include={p}" for p in includes] + ["--exclude=*", f"{host}:{src}/", f"{dst}/"]
    r = _run(cmd, timeout)
    # 23/24 = partial transfer / vanished source file (a checkpoint being rewritten): not fatal
    if r.returncode not in (0, 23, 24):
        msg = (r.stderr or "").strip().splitlines()
        raise SyncError(f"rsync {src} failed ({r.returncode}): {msg[-1] if msg else ''}")
    return _count_itemized(r.stdout)


def _remote_listing(host, remote_dir, sub, includes, max_mb, recursive, timeout):
    depth = "" if recursive else "-maxdepth 1"
    names = " -o ".join(f"-name {shlex.quote(p)}" for p in includes)
    cmd = (f"cd {remote_dir} 2>/dev/null && test -d {sub} && cd {sub} && "
           f"find . {depth} -path ./remote -prune -o -type f \\( {names} \\) -size -{max_mb * 1024}k "
           "-printf '%P\\t%s\\t%T@\\n' || true")
    out = {}
    for line in _ssh(host, cmd, timeout).splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            try:
                out[parts[0]] = (int(parts[1]), float(parts[2]))
            except ValueError:
                pass
    return out


def _tar_pull(host, remote_dir, sub, dst: Path, includes, max_mb, recursive, timeout):
    """Fallback without rsync: list remote files, pull the changed ones via tar over ssh."""
    listing = _remote_listing(host, remote_dir, sub, includes, max_mb, recursive, timeout)
    changed = []
    for rel, (size, mtime) in sorted(listing.items()):
        p = dst / rel
        try:
            st = p.stat()
            if st.st_size == size and abs(st.st_mtime - mtime) < 1.0:
                continue
        except OSError:
            pass
        changed.append(rel)
    if not changed:
        return []
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst) as tmp:
        files = " ".join(shlex.quote(c) for c in changed)
        remote = f"cd {remote_dir}/{sub} && tar cf - -- {files}"
        with open(Path(tmp) / "pull.tar", "wb") as fh:
            try:
                r = RUN(["ssh", *SSH_OPTS, host, remote], stdout=fh, stderr=subprocess.PIPE, timeout=timeout)
            except subprocess.TimeoutExpired as e:
                raise SyncError(f"timeout pulling {sub}") from e
        if r.returncode not in (0, 1):  # 1 = "file changed as we read it"
            raise SyncError(f"tar over ssh failed ({r.returncode})")
        r = _run(["tar", "xf", str(Path(tmp) / "pull.tar"), "-C", tmp], timeout)
        if r.returncode != 0:
            raise SyncError("tar extract failed: " + (r.stderr or "").strip()[-200:])
        for rel in changed:
            s = Path(tmp) / rel
            if s.exists():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                os.replace(s, dst / rel)
    return changed


def sync(host: str = "faverg", remote_dir: str = "FanisZipSolve", root: Path | str = ROOT,
         checkpoints: bool = True, max_run_mb: int = 20, max_ckpt_mb: int = 300,
         timeout: float = 240.0, method: str = "auto") -> dict:
    """Pull runs/ + selected checkpoints from ``host``. Returns a JSON-able result dict
    (``ok``, ``changed`` lists, ``method``, ``remote_now``, ``seconds``, ``error``)."""
    root = Path(root)
    runs_dst = root / "runs" / "remote" / host
    ckpt_dst = root / "checkpoints" / "remote" / host
    t0 = time.time()
    res = {"host": host, "ok": False, "method": None, "changed_runs": [], "changed_checkpoints": [],
           "started": t0, "remote_now": None, "error": None}
    try:
        probe = _ssh(host, "date +%s; command -v rsync >/dev/null 2>&1 && echo rsync || echo no-rsync",
                     min(timeout, 30)).split()
        res["remote_now"] = float(probe[0]) if probe and probe[0].isdigit() else None
        use_rsync = method == "rsync" or (method == "auto" and "rsync" in probe and shutil.which("rsync"))
        res["method"] = "rsync" if use_rsync else "tar"
        left = lambda: max(10.0, timeout - (time.time() - t0))  # noqa: E731
        if use_rsync:
            res["changed_runs"] = _rsync(host, f"{remote_dir}/runs", runs_dst, RUN_PATTERNS, max_run_mb,
                                         True, left())
            if checkpoints:
                res["changed_checkpoints"] = _rsync(host, f"{remote_dir}/checkpoints", ckpt_dst, CKPT_PATTERNS,
                                                    max_ckpt_mb, False, left())
        else:
            res["changed_runs"] = _tar_pull(host, remote_dir, "runs", runs_dst, RUN_PATTERNS, max_run_mb,
                                            True, left())
            if checkpoints:
                res["changed_checkpoints"] = _tar_pull(host, remote_dir, "checkpoints", ckpt_dst, CKPT_PATTERNS,
                                                       max_ckpt_mb, False, left())
        res["ok"] = True
    except SyncError as e:
        res["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    res["finished"] = time.time()
    res["seconds"] = round(res["finished"] - t0, 2)
    try:
        runs_dst.mkdir(parents=True, exist_ok=True)
        state_path = runs_dst / ".sync.json"
        prev = {}
        if state_path.exists():
            try:
                prev = json.loads(state_path.read_text())
            except (OSError, ValueError):
                prev = {}
        state = dict(res)
        if res["ok"]:
            state["last_ok"] = res["finished"]
            state["last_ok_remote_now"] = res["remote_now"]
        else:
            state["last_ok"] = prev.get("last_ok")
            state["last_ok_remote_now"] = prev.get("last_ok_remote_now")
        state_path.write_text(json.dumps(state, indent=1))
    except OSError:
        pass
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="faverg")
    ap.add_argument("--remote-dir", default="FanisZipSolve", help="project dir on the host (relative to ~)")
    ap.add_argument("--root", default=str(ROOT), help="local project root")
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--max-ckpt-mb", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--method", choices=["auto", "rsync", "tar"], default="auto")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args(argv)
    res = sync(a.host, a.remote_dir, a.root, not a.no_checkpoints, max_ckpt_mb=a.max_ckpt_mb,
               timeout=a.timeout, method=a.method)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        status = "ok" if res["ok"] else f"FAILED: {res['error']}"
        print(f"sync {a.host} via {res['method']}: {status} in {res['seconds']}s; "
              f"{len(res['changed_runs'])} run file(s), {len(res['changed_checkpoints'])} checkpoint(s) updated")
        for f in res["changed_runs"] + res["changed_checkpoints"]:
            print("  ", f)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
