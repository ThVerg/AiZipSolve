"""AiZipSolve: Zip puzzles on arbitrary graphs (2D, walls, islands, 3D, 4D) + RL."""
from .graph import ZipGraph, from_edges, from_mask, grid, islands, random_mask, remove_edges
from .puzzle import Puzzle

__all__ = ["ZipGraph", "Puzzle", "from_edges", "from_mask", "grid", "islands",
           "random_mask", "remove_edges"]
