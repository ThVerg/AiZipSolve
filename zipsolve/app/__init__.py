"""Local web app: generate Zip puzzles, play them, or watch the RL agent / exact solver.

Run with ``python -m zipsolve.app``. ``create_app`` builds the FastAPI app.
"""


def create_app(*args, **kwargs):
    from .server import create_app as _create
    return _create(*args, **kwargs)


__all__ = ["create_app"]
