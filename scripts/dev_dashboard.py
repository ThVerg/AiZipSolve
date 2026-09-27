"""Stand-alone dev server for the training dashboard (the real app mounts the same router).

    python scripts/dev_dashboard.py [--port 8772]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from zipsolve.app.dashboard_api import router

STATIC = Path(__file__).resolve().parents[1] / "zipsolve" / "app" / "static"


def make_app() -> FastAPI:
    app = FastAPI(title="Zip dashboard (dev)")
    app.include_router(router)

    @app.get("/dashboard")
    def dashboard():
        return FileResponse(STATIC / "dashboard.html", headers={"Cache-Control": "no-cache"})

    @app.get("/")
    def index():
        return RedirectResponse("/dashboard")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port, log_level="warning")
