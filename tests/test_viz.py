import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import viz_demo as D  # noqa: E402
from zipsolve import graph as G  # noqa: E402
from zipsolve import viz  # noqa: E402
from zipsolve.puzzle import Puzzle  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _rng():
    return np.random.default_rng(0)


BUILDERS = {
    "grid": D.demo_grid,
    "walls": D.demo_walls,
    "mask": D.demo_mask,
    "islands": D.demo_islands,
    "3d": lambda r: D.demo_3d(r, 3),
    "4d": D.demo_4d,
    "generic": D.demo_generic,
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_draw_variants(name):
    pz = BUILDERS[name](_rng())
    assert pz.is_valid_solution(pz.solution)
    for path in (None, pz.solution, pz.solution[: len(pz.solution) // 2], pz.solution[:1]):
        fig = viz.draw(pz, path, title=name)
        assert isinstance(fig, matplotlib.figure.Figure)
        fig.canvas.draw()
        plt.close(fig)


def test_draw_into_existing_axes():
    pz2 = D.demo_grid(_rng())
    pz3 = D.demo_3d(_rng(), 3)
    pz4 = D.demo_4d(_rng())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    assert viz.draw(pz2, pz2.solution, ax=axes[0]) is fig
    assert viz.draw(pz3, pz3.solution, ax=axes[1]) is fig
    assert viz.draw(pz4, pz4.solution, ax=axes[2]) is fig
    fig.canvas.draw()


def test_draw_slices_and_views():
    pz = D.demo_3d(_rng(), 3)
    for axis in (0, 1, 2):
        viz.draw_slices(pz, pz.solution[:10], axis=axis).canvas.draw()
    viz.draw(pz, pz.solution, view="slices").canvas.draw()
    viz.draw(pz, pz.solution, view="generic").canvas.draw()


def test_save_and_animate(tmp_path):
    pz = D.demo_grid(_rng())
    viz.save(pz, pz.solution, tmp_path / "a.png")
    viz.save(D.demo_3d(_rng(), 3), None, tmp_path / "b.png", slices=True)
    assert (tmp_path / "a.png").stat().st_size > 1000
    assert (tmp_path / "b.png").stat().st_size > 1000
    small = Puzzle(G.grid(3, 3), [0, 8], D.snake_path(G.grid(3, 3), (3, 3)))
    viz.animate(small, small.solution, tmp_path / "c.gif", fps=5)
    from PIL import Image
    assert Image.open(tmp_path / "c.gif").n_frames > 1


def test_ascii_grid_no_path():
    g = G.grid(3, 4)
    pz = Puzzle(g, [0, 5, 11])
    s = viz.to_ascii(pz)
    lines = s.splitlines()
    assert lines[0].startswith("+") and lines[-1] == lines[0]
    assert len(lines) == 2 + 3 + 2  # frame, 3 rows, 2 separator lines
    assert len({len(line) for line in lines}) == 1
    body = s.replace("+", "").replace("-", "").replace("|", "")
    assert body.count(".") == 12 - 3
    for lab in "123":
        assert lab in body


def test_ascii_arrows_walls_and_partial():
    g = G.grid(2, 3)  # nodes: row-major, 0 1 2 / 3 4 5
    g = G.remove_edges(g, [(1, 4)])
    path = [0, 1, 2, 5, 4, 3]
    pz = Puzzle(g, [0, 3], path)
    s = viz.to_ascii(pz, path)
    assert pz.is_valid_solution(path)
    row0, sep, row1 = s.splitlines()[1:4]
    assert row0.split("|")[1].split() == ["1", ">", "v"]
    assert "---" in sep  # the wall between 1 and 4
    assert row1.split("|")[1].split() == ["2", "<", "<"]
    part = viz.to_ascii(pz, path[:3])
    assert "@" in part and "." in part
    order = viz.to_ascii(pz, path, mode="order")
    assert all(str(i) in order for i in range(1, 7))

    gw = G.remove_edges(G.grid(2, 2), [(0, 1)])
    s = viz.to_ascii(Puzzle(gw, [0, 1]))
    assert "|" in s.splitlines()[1][1:-1]


def test_ascii_mask_and_islands():
    pz = D.demo_islands(_rng())
    s = viz.to_ascii(pz, pz.solution)
    assert "~" in s  # bridge move
    pzm = D.demo_mask(_rng())
    assert viz.to_ascii(pzm).count(".") == pzm.num_nodes - len(pzm.checkpoints)
    with pytest.raises(ValueError):
        viz.to_ascii(D.demo_3d(_rng(), 3))
