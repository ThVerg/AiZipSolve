"""Puzzle definition, solution checking and JSON (de)serialisation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .graph import ZipGraph, from_edges


@dataclass
class Puzzle:
    """A Zip puzzle: visit every node once, passing checkpoints in order.

    checkpoints[0] is where the path starts ("1") and checkpoints[-1] is where
    it must end (the highest number). Checkpoint k is labelled k+1.
    `solution` is an optional known valid path (e.g. from the generator).
    """
    graph: ZipGraph
    checkpoints: list[int]
    solution: list[int] | None = None

    def __post_init__(self):
        if len(self.checkpoints) < 2:
            raise ValueError("need at least a start and an end checkpoint")
        if len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError("checkpoints must be distinct nodes")

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    def checkpoint_label(self) -> dict[int, int]:
        """node -> label (1-based)."""
        return {v: i + 1 for i, v in enumerate(self.checkpoints)}

    def is_valid_solution(self, path: Sequence[int]) -> bool:
        return self.check_solution(path) is None

    def check_solution(self, path: Sequence[int]) -> str | None:
        """None if `path` is a valid solution, else a human-readable reason."""
        g = self.graph
        path = list(path)
        if len(path) != g.num_nodes:
            return f"path has {len(path)} nodes, graph has {g.num_nodes}"
        if len(set(path)) != len(path):
            return "path revisits a node"
        if any(not (0 <= v < g.num_nodes) for v in path):
            return "path contains an invalid node id"
        for a, b in zip(path, path[1:]):
            if not g.has_edge(a, b):
                return f"nodes {a} and {b} are not adjacent"
        if path[0] != self.checkpoints[0]:
            return "path does not start at checkpoint 1"
        if path[-1] != self.checkpoints[-1]:
            return "path does not end at the last checkpoint"
        pos = {v: i for i, v in enumerate(path)}
        order = [pos[c] for c in self.checkpoints]
        if order != sorted(order):
            return "checkpoints are not visited in order"
        return None

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        meta = {k: v for k, v in self.graph.meta.items() if not k.startswith("_")}
        return {
            "kind": self.graph.kind,
            "coords": self.graph.coords.tolist(),
            "edges": [list(e) for e in self.graph.edges()],
            "meta": json.loads(json.dumps(meta, default=_jsonable)),
            "checkpoints": list(map(int, self.checkpoints)),
            "solution": None if self.solution is None else list(map(int, self.solution)),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Puzzle":
        g = from_edges(np.array(d["coords"], dtype=float), [tuple(e) for e in d["edges"]],
                       d.get("kind", "custom"), d.get("meta") or {})
        return cls(g, list(d["checkpoints"]), d.get("solution"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "Puzzle":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    raise TypeError(type(o))
