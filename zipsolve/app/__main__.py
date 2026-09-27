"""python -m zipsolve.app [--port 8000] [--host 127.0.0.1] [--no-browser] [--checkpoints DIR]"""
from __future__ import annotations

import argparse
import os
import threading
import webbrowser


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zip puzzle web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--checkpoints", default=None, help="directory with *.pt models")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    args = ap.parse_args(argv)
    if args.checkpoints:
        os.environ["ZIPSOLVE_CHECKPOINTS"] = args.checkpoints
    try:
        import torch
        torch.set_num_threads(max(1, args.threads))
    except Exception:  # noqa: BLE001
        pass
    import uvicorn
    from .server import create_app

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Zip web app on {url}")
    uvicorn.run(create_app(args.checkpoints), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
