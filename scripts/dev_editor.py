"""Tiny dev server for the puzzle editor (editor API + static files), independent of server.py.

    python scripts/dev_editor.py --port 8773
Serves /editor, /static/*, /api/custom*, /api/editor/*, and (if importable) the main app's API.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from zipsolve.app.editor_api import router

STATIC = Path(__file__).resolve().parents[1] / "zipsolve" / "app" / "static"


def build() -> FastAPI:
    try:  # full app (generate / solve / play page) when available
        from zipsolve.app.server import create_app
        app = create_app()
        paths = {getattr(r, "path", None) for r in app.routes}
        if "/api/custom" not in paths:
            app.include_router(router)
        if "/editor" not in paths:
            app.add_api_route("/editor", lambda: FileResponse(STATIC / "editor.html"), methods=["GET"])
        return app
    except Exception as e:  # noqa: BLE001
        print("main app unavailable:", e)
    app = FastAPI()
    app.include_router(router)
    app.add_api_route("/editor", lambda: FileResponse(STATIC / "editor.html"), methods=["GET"])
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8773)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(build(), host=a.host, port=a.port, log_level="warning")
