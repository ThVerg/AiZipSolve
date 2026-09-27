"""Feature versions (v1 frozen, v2 structural features) and versioned GNN heads."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from zipsolve.graph import from_edges, grid
from zipsolve.puzzle import Puzzle
from zipsolve.rl.env import (F, FEATURES_V1, FEATURES_V2, LATEST_FEATURE_VERSION, ZipEnv,
                             num_features)
from zipsolve.rl.gnn import (ZipGNN, collate, env_kwargs_from_meta, load_model, make_env_for_model,
                             make_model, save_model)
from zipsolve.rl.sampling import make_sampler

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).parent.parent
OLD_CKPT = ROOT / "checkpoints" / "curric_4to6_final.pt"


def _snapshot():
    recs = json.loads((DATA / "features_v1_snapshot.json").read_text())
    arr = np.load(DATA / "features_v1_snapshot.npz")
    return recs, arr


def _replay(puzzle, actions, version):
    env = ZipEnv(early_termination=False, feature_version=version)
    obs, _ = env.reset(options={"puzzle": puzzle})
    out = [obs]
    for a in actions:
        obs, *_ = env.step(a)
        out.append(obs)
    return out


def _random_states(specs, seed=0, episodes=3, version=2):
    rng = np.random.default_rng(seed)
    for si, spec in enumerate(specs):
        samp = make_sampler(spec, seed + si)
        for _ in range(episodes):
            env = ZipEnv(early_termination=False, feature_version=version)
            obs, _ = env.reset(options={"puzzle": samp()})
            while True:
                yield env, obs
                legal = np.flatnonzero(obs["action_mask"])
                if len(legal) == 0:
                    break
                obs, _, term, _, _ = env.step(int(rng.choice(legal)))
                if term:
                    break


# ---------------------------------------------------------------- versioning
def test_feature_sets_are_prefix_compatible():
    assert FEATURES_V2[:len(FEATURES_V1)] == FEATURES_V1
    assert num_features(1) == 17 and num_features(2) == len(FEATURES_V2)
    assert LATEST_FEATURE_VERSION == 2
    assert ZipEnv().feature_version == LATEST_FEATURE_VERSION
    with pytest.raises(ValueError):
        ZipEnv(feature_version=99)


def test_v1_features_bit_for_bit_snapshot():
    """Snapshot taken from the implementation before feature versioning existed."""
    recs, arr = _snapshot()
    for k, r in enumerate(recs):
        p = Puzzle.from_dict(r["puzzle"])
        x1 = np.stack([o["x"] for o in _replay(p, r["actions"], 1)])
        assert x1.shape[2] == 17
        assert np.array_equal(x1, arr[f"x{k}"]), r["spec"]
        x2 = np.stack([o["x"] for o in _replay(p, r["actions"], 2)])
        assert x2.shape[2] == len(FEATURES_V2)
        assert np.array_equal(x2[:, :, :17], arr[f"x{k}"]), r["spec"]


@pytest.mark.skipif(not OLD_CKPT.exists(), reason="old checkpoint not present")
def test_old_checkpoint_identical_outputs():
    recs, arr = _snapshot()
    model, ck = load_model(OLD_CKPT)
    assert ck["feature_version"] == 1 and model.feature_version == 1 and model.head_version == 1
    assert env_kwargs_from_meta(ck) == {"feature_version": 1}
    assert make_env_for_model(model).feature_version == 1
    # Snapshot logits were produced before this change by the same code path; torch's
    # reduction order depends on the thread count (other tests change it), so compare
    # to 1e-5 against the snapshot and bit-for-bit between v1 and v2 observations.
    for k, r in enumerate(recs):
        p = Puzzle.from_dict(r["puzzle"])
        outs = []
        for version in (1, 2):  # a v1 model also runs on v2 observations (prefix slice)
            with torch.no_grad():
                lg, v = model(collate(_replay(p, r["actions"], version)))
            fin = np.isfinite(arr[f"logits{k}"])
            assert np.array_equal(np.isfinite(lg.numpy()), fin)
            assert np.allclose(lg.numpy()[fin], arr[f"logits{k}"][fin], atol=1e-5, rtol=0), r["spec"]
            assert np.allclose(v.numpy(), arr[f"values{k}"], atol=1e-5, rtol=0), r["spec"]
            outs.append((lg, v))
        assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_v2_model_rejects_v1_observations():
    p = Puzzle(grid(3, 3), [0, 8])
    obs, _ = ZipEnv(feature_version=1).reset(options={"puzzle": p})
    with pytest.raises(ValueError, match="feature"):
        make_model(hidden=16, layers=1)(collate([obs]))


def test_old_style_config_defaults(tmp_path):
    """A checkpoint written like before (config without version keys) loads as v1/head 1."""
    m = ZipGNN(in_dim=17, hidden=16, layers=1, head_version=1)
    cfg = {"in_dim": 17, "hidden": 16, "layers": 1}
    torch.save({"config": cfg, "state_dict": m.state_dict()}, tmp_path / "old.pt")
    m2, ck = load_model(tmp_path / "old.pt")
    assert m2.feature_version == 1 and m2.head_version == 1 and not m2.global_attn
    assert env_kwargs_from_meta(ck) == {"feature_version": 1}
    assert env_kwargs_from_meta({"in_dim": 17}) == {"feature_version": 1}


@pytest.mark.parametrize("attn", [False, True])
def test_v2_save_load_roundtrip(tmp_path, attn):
    torch.manual_seed(0)
    m = make_model(hidden=32, layers=2, global_attn=attn)
    assert m.config["feature_version"] == 2 and m.config["head_version"] == 2
    save_model(m, tmp_path / "m.pt", note="x")
    m2, ck = load_model(tmp_path / "m.pt")
    assert ck["feature_version"] == 2 and ck["note"] == "x" and m2.config == m.config
    env = make_env_for_model(m2, ck, make_sampler("grid2d:5", 0))
    assert env.feature_version == 2
    obs = [env.reset()[0] for _ in range(3)]
    m.eval()
    with torch.no_grad():
        a, va = m(collate(obs))
        b, vb = m2(collate(obs))
    assert torch.equal(a, b) and torch.equal(va, vb)


@pytest.mark.parametrize("head_version,attn", [(1, False), (2, False), (2, True), (1, True)])
def test_collate_v2_batched_matches_single(head_version, attn):
    torch.manual_seed(1)
    specs = ["grid2d:4", "grid3d:3", "islands:2", "grid2d:5", "mask:5"]
    obs = []
    for s in specs:
        env = ZipEnv(make_sampler(s, 0))
        o, _ = env.reset()
        for _ in range(2):  # a couple of moves so heads / targets differ
            legal = np.flatnonzero(o["action_mask"])
            if len(legal):
                o, *_ = env.step(int(legal[0]))
        obs.append(o)
    model = make_model(hidden=32, layers=3, head_version=head_version, global_attn=attn)
    model.eval()
    gb = collate(obs)
    assert gb.x.shape == (sum(o["x"].shape[0] for o in obs), len(FEATURES_V2))
    with torch.no_grad():
        logits, values = model(gb)
        single = [model(collate([o])) for o in obs]
    assert torch.isinf(logits[~gb.mask]).all() and torch.isfinite(logits[gb.mask]).all()
    sl = torch.cat([s[0] for s in single])
    sv = torch.cat([s[1] for s in single])
    assert torch.allclose(torch.nan_to_num(sl, neginf=0), torch.nan_to_num(logits, neginf=0), atol=1e-5)
    assert torch.allclose(sv, values, atol=1e-5)


def test_target_embedding_changes_policy():
    """head_version 2 sees the next checkpoint: moving the target changes logits."""
    torch.manual_seed(0)
    m = make_model(hidden=16, layers=1)
    m.eval()
    g = grid(3, 3)
    oa, _ = ZipEnv().reset(options={"puzzle": Puzzle(g, [4, 0, 8])})
    ob, _ = ZipEnv().reset(options={"puzzle": Puzzle(g, [4, 2, 8])})
    xa, xb = oa["x"].copy(), ob["x"].copy()
    # make the two observations identical except for the is_next_cp column
    xb[:, [c for c in range(xb.shape[1]) if c != F["is_next_cp"]]] = \
        xa[:, [c for c in range(xa.shape[1]) if c != F["is_next_cp"]]]
    ob["x"] = xb
    with torch.no_grad():
        la, _ = m(collate([oa]))
        lb, _ = m(collate([ob]))
    assert not torch.allclose(la[oa["action_mask"]], lb[oa["action_mask"]])


# ---------------------------------------------------------------- v2 features
def test_v2_features_finite_in_range():
    for spec in ["grid2d:5", "grid3d:3", "grid4d:2", "islands:3", "walls:5", "mask:6"]:
        env = ZipEnv(make_sampler(spec, 0))
        obs, _ = env.reset()
        x = obs["x"]
        assert x.shape[1] == len(FEATURES_V2)
        assert np.isfinite(x).all() and x.min() >= -1 and x.max() <= 1


def test_ordered_distance_blocks_later_checkpoints():
    # 0 1 2 3
    # 4 5 6 7     start 0, next 3, later checkpoint 1, end 7
    g = grid(2, 4)
    env = ZipEnv(early_termination=False)
    obs, _ = env.reset(options={"puzzle": Puzzle(g, [0, 3, 1, 7])})
    x = obs["x"]
    m = 7  # unvisited nodes
    # order-respecting route 3 -> head avoids 1 (and the end 7): 3-2-6-5-4-0
    assert x[0, F["dist_next_cp_ord"]] == pytest.approx(5 / m)
    # later checkpoints are leaves: reachable (distance) but not expanded
    assert x[1, F["dist_next_cp_ord"]] == pytest.approx(2 / m)
    assert x[7, F["dist_next_cp_ord"]] == pytest.approx(1 / m)
    assert x[3, F["dist_next_cp_ord"]] == 0.0
    # from the head: 1 is a leaf, so 2 is reached around the bottom row: 0-4-5-6-2
    assert x[2, F["dist_head_ord"]] == pytest.approx(4 / m)
    assert x[1, F["dist_head_ord"]] == pytest.approx(1 / m)
    assert x[0, F["dist_head_ord"]] == 0.0
    # the unordered v1 distance happily shortcuts through checkpoint 1 (3-2-1-5)
    assert x[5, F["dist_next_cp"]] == pytest.approx(3 / m)
    assert x[5, F["dist_next_cp_ord"]] == pytest.approx(3 / m)  # 3-2-6-5 also 3
    assert x[4, F["dist_next_cp_ord"]] == pytest.approx(4 / m)


def _ordered_ref(env, src):
    """Reference ordered BFS with networkx: interior nodes must be unvisited and
    not a later checkpoint; the target may be any unvisited node or the head."""
    n = env.n
    later = set(env.cps[env.next_cp + 1:])
    interior = {v for v in range(n) if not env.visited[v] and v not in later}
    out = np.full(n, -1)
    for v in range(n):
        if v != src and env.visited[v] and v != env.head:
            continue
        nodes = interior | {src, v}
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from((a, b) for a in nodes for b in env.nbrs[a] if b in nodes)
        # src / v can only be path endpoints in a simple shortest path
        try:
            out[v] = nx.shortest_path_length(G, src, v)
        except nx.NetworkXNoPath:
            pass
    return out


def test_ordered_distances_match_reference():
    count = 0
    for env, obs in _random_states(["grid2d:5", "islands:2", "walls:5"], seed=3, episodes=2):
        if env.next_cp >= len(env.cps) or count > 40:
            continue
        count += 1
        m = max(1, int((~env.visited[:env.n]).sum()))
        for src, col in ((env.cps[env.next_cp], "dist_next_cp_ord"), (env.head, "dist_head_ord")):
            ref = _ordered_ref(env, src)
            exp = np.where(ref >= 0, ref / m, 1.0)
            assert np.allclose(obs["x"][:, F[col]], exp), (col, env.path)
    assert count > 10


def test_articulation_and_bridges_match_networkx():
    checked = 0
    for env, obs in _random_states(["grid2d:5", "islands:3", "mask:6", "walls:5"], seed=7, episodes=2):
        x, n = obs["x"], env.n
        R = [v for v in range(n) if not env.visited[v] or v == env.head]
        Rs = set(R)
        G = nx.Graph()
        G.add_nodes_from(R)
        G.add_edges_from((u, w) for u in R for w in env.nbrs[u] if w in Rs)
        H = G.subgraph(nx.node_connected_component(G, env.head)).copy()
        ap = set(nx.articulation_points(H))
        assert set(np.flatnonzero(x[:, F["is_art"]]).tolist()) == ap
        nb = np.zeros(n)
        for u, w in nx.bridges(H):
            nb[u] += 1
            nb[w] += 1
        assert np.allclose(x[:, F["bridge_frac"]], nb / env.max_deg)
        m = max(1, n - len(env.path))
        for v in ap - {env.head}:
            H2 = H.copy()
            H2.remove_node(v)
            cc = list(nx.connected_components(H2))
            hc = next(c for c in cc if env.head in c)
            beyond = [c for c in cc if c is not hc]
            assert x[v, F["cut_beyond"]] == pytest.approx(sum(map(len, beyond)) / m)
            assert x[v, F["cut_min"]] == pytest.approx(min(map(len, cc)) / m)
            assert x[v, F["cut_has_end"]] == float(any(env.end in c for c in beyond))
        nonart = np.ones(n, bool)
        nonart[list(ap)] = False
        assert (x[nonart, F["cut_beyond"]] == 0).all()
        checked += 1
    assert checked > 50


def test_articulation_hand_made_path_graph():
    g = grid(1, 5)  # 0-1-2-3-4, start 0, end 4
    obs, _ = ZipEnv().reset(options={"puzzle": Puzzle(g, [0, 4])})
    x = obs["x"]
    assert x[:, F["is_art"]].tolist() == [0, 1, 1, 1, 0]
    assert x[:, F["bridge_frac"]].tolist() == [0.5, 1, 1, 1, 0.5]
    assert x[1, F["cut_beyond"]] == pytest.approx(3 / 4)
    assert x[3, F["cut_beyond"]] == pytest.approx(1 / 4)
    assert x[1, F["cut_min"]] == pytest.approx(1 / 4)
    assert x[2, F["cut_min"]] == pytest.approx(2 / 4)
    assert x[1:4, F["cut_has_end"]].tolist() == [1, 1, 1]


def test_articulation_region_without_end():
    # star-ish: head 0 - 1, and 1 branches to 2 and 3 (end); region {2} lacks the end
    g = from_edges(np.zeros((4, 1)), [(0, 1), (1, 2), (1, 3)])
    obs, _ = ZipEnv(early_termination=False).reset(options={"puzzle": Puzzle(g, [0, 3])})
    x = obs["x"]
    assert x[1, F["is_art"]] == 1 and x[1, F["cut_has_end"]] == 1
    assert x[1, F["cut_beyond"]] == pytest.approx(2 / 3)
    assert x[1, F["cut_min"]] == pytest.approx(1 / 3)
    # head as root articulation point: start in the middle of a path
    g2 = grid(1, 3)
    obs, _ = ZipEnv(early_termination=False).reset(options={"puzzle": Puzzle(g2, [1, 2])})
    assert obs["x"][1, F["is_art"]] == 1 and obs["x"][1, F["cut_beyond"]] == 0


def test_parity_features():
    # 3x3: colour 0 at even (r+c); start 0 (colour 0), end 8 (colour 0)
    obs, _ = ZipEnv().reset(options={"puzzle": Puzzle(grid(3, 3), [0, 8])})
    x = obs["x"]
    assert (x[:, F["bipartite"]] == 1).all()
    assert x[:, F["same_colour_head"]].tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert x[0, F["parity_excess"]] == 0 and x[0, F["parity_ok"]] == 1
    # end of the wrong colour: 8 remaining nodes (even) but end colour differs
    obs, _ = ZipEnv(early_termination=False).reset(options={"puzzle": Puzzle(grid(3, 3), [0, 1])})
    assert obs["x"][0, F["parity_ok"]] == 0
    # start on the minority colour: 4 same-colour of 8 remaining needed, 5 same-colour cells exist? no:
    # start 1 (colour 1): remaining colour-1 cells = 3 (3,5,7) but 8//2 = 4 needed
    obs, _ = ZipEnv(early_termination=False).reset(options={"puzzle": Puzzle(grid(3, 3), [1, 7])})
    x = obs["x"]
    assert x[0, F["parity_excess"]] == pytest.approx((3 - 4) / 8) and x[0, F["parity_ok"]] == 0
    # 2x3, start 0 end 5 (different colours, 5 remaining: odd) is consistent
    obs, _ = ZipEnv().reset(options={"puzzle": Puzzle(grid(2, 3), [0, 5])})
    assert obs["x"][0, F["parity_ok"]] == 1 and obs["x"][0, F["parity_excess"]] == 0
    # after a move the counts update
    env = ZipEnv(early_termination=False)
    env.reset(options={"puzzle": Puzzle(grid(3, 3), [0, 8])})
    o, *_ = env.step(1)
    assert o["x"][0, F["same_colour_head"]] == 0 and o["x"][1, F["same_colour_head"]] == 1
    assert o["x"][0, F["parity_ok"]] == 1  # 0-1 then 7 remaining, head colour 1: need 3 of colour 1


def test_parity_non_bipartite_zero():
    tri = from_edges(np.zeros((4, 1)), [(0, 1), (1, 2), (0, 2), (2, 3)])
    obs, _ = ZipEnv(early_termination=False).reset(options={"puzzle": Puzzle(tri, [0, 3])})
    x = obs["x"]
    for f in ["bipartite", "colour", "same_colour_head", "parity_excess", "parity_ok"]:
        assert (x[:, F[f]] == 0).all()


def test_parity_ok_holds_along_solutions():
    """A valid solution path never violates the parity condition."""
    for spec in ["grid2d:5", "walls:5", "grid3d:3", "islands:2"]:
        samp = make_sampler(spec, 11)
        for _ in range(3):
            p = samp()
            env = ZipEnv()
            obs, _ = env.reset(options={"puzzle": p})
            for a in p.solution[1:-1]:
                obs, *_ = env.step(a)
                x = obs["x"]
                if x[0, F["bipartite"]]:
                    assert x[0, F["parity_ok"]] == 1 and x[0, F["parity_excess"]] == 0
                # all articulation regions away from the head contain the end on a solvable state
                art = x[:, F["is_art"]] > 0
                art[env.head] = False
                assert (x[art, F["cut_has_end"]] == 1).all()


def test_make_env_for_model_call_forms():
    m1 = ZipGNN(in_dim=17, hidden=8, layers=1, head_version=1)
    ck1 = {"config": {"in_dim": 17, "hidden": 8, "layers": 1}}
    assert make_env_for_model(m1).feature_version == 1
    assert make_env_for_model(m1, ck1, early_termination=False).feature_version == 1
    assert make_env_for_model(meta=ck1).feature_version == 1
    assert make_env_for_model(None, None).feature_version == LATEST_FEATURE_VERSION
    env = make_env_for_model(make_model(hidden=8, layers=1), None, None, early_termination=False)
    assert env.feature_version == 2 and not env.early_termination and env.puzzle_sampler is None
