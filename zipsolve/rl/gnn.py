"""Graph neural network policy/value for Zip, in plain torch (no torch_geometric).

Many graphs of different sizes are processed as one disjoint-union graph:
`collate(obs_list)` concatenates node features, offsets edge indices and builds
a `batch` vector (graph id of every node) plus `ptr` offsets.

Architecture:
    encoder MLP (F -> H)
    L x message-passing block (residual + LayerNorm), neighbours gathered from a
    padded (N, D) adjacency table (much faster on CPU than scatter for the
    bounded-degree graphs used here):
        h <- LN(h + MLP([h, mean_{u~v} h_u, max_{u~v} h_u, sum_{u~v, u free} h_u / deg(v)]))
      "free" neighbours are unvisited nodes or the head, so one aggregation
      channel sees only the still-open part of the graph.
    head_version 1 (original):
      policy head: MLP([h_v, h_head(g), mean_g, max_g]) -> logit_v, masked softmax per graph
      value head:  MLP([mean_g, max_g, h_head(g)]) -> V(g)
    head_version 2: the heads also see the embedding of the current target
      (next checkpoint, from the is_next_cp feature) and of the end node:
      policy head: MLP([h_v, h_head, h_target, h_end, mean_g, max_g (, attn_g)])
      value head:  MLP([mean_g, max_g, h_head, h_target, h_end (, attn_g)])
    global_attn (optional): after every message-passing block a global
      attention-pooling step: queries from [h_head, h_target] per graph,
      multi-head attention over the *free* nodes (unvisited + head), pooled
      vector broadcast back to every node (h <- LN(h + MLP([h, pool_g]))). This
      is O(N * H), gives every node long-range context each round, and the
      final pooled vector attn_g is also fed to both heads.

Versioning / compatibility:
    ZipGNN.config holds in_dim, hidden, layers, feature_version, head_version,
    global_attn, attn_heads; save_model stores it (plus a top-level
    "feature_version"). load_model fills missing keys for old checkpoints
    (feature_version from in_dim -> 1, head_version 1, no attention), so old
    weights load unchanged. A model whose in_dim is smaller than the
    observation width uses the first in_dim columns (feature versions are
    prefix-compatible), so a v1 checkpoint also runs on a default (latest) env.
    To build the matching env: ``make_env_for_model(model, **env_kwargs)`` or
    ``ZipEnv(**env_kwargs_from_meta(ck), ...)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import (F, LATEST_FEATURE_VERSION, ZipEnv, feature_version_for_dim,
                  num_features)

LATEST_HEAD_VERSION = 2


@dataclass
class GraphBatch:
    x: torch.Tensor           # (N, F)
    edge_index: torch.Tensor  # (2, E) global node ids
    batch: torch.Tensor       # (N,) graph id
    ptr: torch.Tensor         # (B+1,) node offsets
    head: torch.Tensor        # (B,) global node id of each head
    mask: torch.Tensor        # (N,) bool legal actions
    num_graphs: int
    max_nodes: int
    nbr: torch.Tensor | None = None  # (N, D) neighbour table, padded with N


    def local_index(self) -> torch.Tensor:
        return torch.arange(self.x.shape[0]) - self.ptr[self.batch]


def _nbr_table(edge_index: np.ndarray, n: int) -> np.ndarray:
    deg = np.bincount(edge_index[1], minlength=n)
    D = max(1, int(deg.max()) if n else 1)
    tab = np.full((n, D), -1, dtype=np.int64)
    fill = np.zeros(n, dtype=np.int64)
    for u, v in edge_index.T:
        tab[v, fill[v]] = u
        fill[v] += 1
    return tab


def collate(obs_list: list[dict]) -> GraphBatch:
    xs, eis, masks, heads, sizes, nbrs = [], [], [], [], [], []
    off = 0
    total = sum(o["x"].shape[0] for o in obs_list)
    D = 1
    for o in obs_list:
        nb = o.get("neighbors")
        if nb is None:
            nb = _nbr_table(o["edge_index"], o["x"].shape[0])
        nbrs.append(nb)
        D = max(D, nb.shape[1])
    nbrs_out = []
    for o, nb in zip(obs_list, nbrs):
        n = o["x"].shape[0]
        xs.append(o["x"])
        eis.append(o["edge_index"] + off)
        masks.append(o["action_mask"])
        heads.append(int(o["head"]) + off)
        sizes.append(n)
        t = np.full((n, D), total, dtype=np.int64)
        t[:, :nb.shape[1]] = np.where(nb >= 0, nb + off, total)
        nbrs_out.append(t)
        off += n
    sizes_t = torch.tensor(sizes, dtype=torch.long)
    ptr = torch.zeros(len(sizes) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(sizes_t, 0)
    return GraphBatch(
        x=torch.from_numpy(np.concatenate(xs, 0)),
        edge_index=torch.from_numpy(np.concatenate(eis, 1)),
        batch=torch.repeat_interleave(torch.arange(len(sizes)), sizes_t),
        ptr=ptr,
        head=torch.tensor(heads, dtype=torch.long),
        mask=torch.from_numpy(np.concatenate(masks, 0)).bool(),
        num_graphs=len(sizes),
        max_nodes=int(max(sizes)),
        nbr=torch.from_numpy(np.concatenate(nbrs_out, 0)),
    )


def scatter_mean(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = src.new_zeros((size, src.shape[1])).index_add_(0, index, src)
    cnt = src.new_zeros(size).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


def scatter_max(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = src.new_zeros((size, src.shape[1]))
    return out.scatter_reduce(0, index.unsqueeze(1).expand_as(src), src, "amax", include_self=False)


def mlp(i, h, o, act=nn.SiLU):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, o))


class MPBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.mlp = mlp(4 * h, h, h)
        self.norm = nn.LayerNorm(h)

    def forward(self, h, nbr, valid, cnt, free_w):
        """nbr (N, D) neighbour ids padded with N; valid (N, D, 1) float; cnt (N, 1);
        free_w (N, D, 1) = valid * neighbour-is-free."""
        h_ext = torch.cat([h, h.new_zeros(1, h.shape[1])], 0)
        m = h_ext[nbr]                                    # (N, D, H)
        mean = (m * valid).sum(1) / cnt.clamp(min=1.0)
        mx = m.masked_fill(valid == 0, -1e4).max(1).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        fsum = (m * free_w).sum(1) / cnt.clamp(min=1.0)  # free-neighbour mass / degree
        return self.norm(h + self.mlp(torch.cat([h, mean, mx, fsum], 1)))


def scatter_sum(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    return src.new_zeros((size,) + tuple(src.shape[1:])).index_add_(0, index, src)


class GlobalAttnPool(nn.Module):
    """Multi-head attention pooling over the free nodes of each graph, with
    per-graph queries (built from head / target embeddings)."""

    def __init__(self, h: int, heads: int = 4, qdim: int | None = None):
        super().__init__()
        assert h % heads == 0, "hidden must be divisible by attn_heads"
        self.heads = heads
        self.q = nn.Linear(qdim or 2 * h, h)
        self.k = nn.Linear(h, h)
        self.v = nn.Linear(h, h)
        self.out = nn.Linear(h, h)

    def forward(self, h, qctx, batch, B, node_mask):
        N, H = h.shape
        nh, d = self.heads, H // self.heads
        q = self.q(qctx)[batch].view(N, nh, d)
        k = self.k(h).view(N, nh, d)
        v = self.v(h).view(N, nh, d)
        s = (q * k).sum(-1) / d ** 0.5                                   # (N, nh)
        s = s.masked_fill(~node_mask.unsqueeze(1), -1e4)
        smax = s.new_full((B, nh), -1e4).scatter_reduce(
            0, batch.unsqueeze(1).expand_as(s), s, "amax", include_self=True)
        e = torch.exp(s - smax[batch]) * node_mask.unsqueeze(1).to(s.dtype)
        den = scatter_sum(e, batch, B).clamp(min=1e-9)
        a = e / den[batch]
        pooled = scatter_sum(a.unsqueeze(2) * v, batch, B).reshape(B, H)
        return self.out(pooled)


class GlobalBlock(nn.Module):
    def __init__(self, h: int, heads: int):
        super().__init__()
        self.pool = GlobalAttnPool(h, heads)
        self.mlp = mlp(2 * h, h, h)
        self.norm = nn.LayerNorm(h)

    def forward(self, h, qctx, batch, B, node_mask):
        g = self.pool(h, qctx, batch, B, node_mask)
        return self.norm(h + self.mlp(torch.cat([h, g[batch]], 1))), g


class ZipGNN(nn.Module):
    def __init__(self, in_dim: int | None = None, hidden: int = 64, layers: int = 6,
                 feature_version: int | None = None, head_version: int = LATEST_HEAD_VERSION,
                 global_attn: bool = False, attn_heads: int = 4):
        """in_dim / feature_version: give either (the other is derived); both
        omitted -> latest feature version. head_version 1 = original heads,
        2 = heads see target (next checkpoint) and end embeddings."""
        super().__init__()
        if feature_version is None:
            feature_version = LATEST_FEATURE_VERSION if in_dim is None else feature_version_for_dim(in_dim)
        if in_dim is None:
            in_dim = num_features(feature_version)
        if in_dim != num_features(feature_version):
            raise ValueError(f"in_dim {in_dim} does not match feature_version {feature_version}")
        if head_version not in (1, 2):
            raise ValueError(f"unknown head_version {head_version}")
        self.config = {"in_dim": in_dim, "hidden": hidden, "layers": layers,
                       "feature_version": int(feature_version), "head_version": int(head_version),
                       "global_attn": bool(global_attn), "attn_heads": int(attn_heads)}
        self.in_dim = in_dim
        self.feature_version = int(feature_version)
        self.head_version = int(head_version)
        self.global_attn = bool(global_attn)
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList(MPBlock(hidden) for _ in range(layers))
        if self.global_attn:
            self.gblocks = nn.ModuleList(GlobalBlock(hidden, attn_heads) for _ in range(layers))
        if self.head_version == 1:
            pi_in, v_in = 4 * hidden, 3 * hidden
        else:
            extra = hidden if self.global_attn else 0
            pi_in, v_in = 6 * hidden + extra, 5 * hidden + extra
        self.pi = nn.Sequential(nn.Linear(pi_in, hidden), nn.SiLU(),
                                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.v = nn.Sequential(nn.Linear(v_in, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, gb: GraphBatch):
        """Returns (logits (N,), values (B,)); logits of illegal nodes are -inf."""
        x = gb.x
        if x.shape[1] != self.in_dim:
            if x.shape[1] < self.in_dim:
                raise ValueError(
                    f"observations have {x.shape[1]} features but the model needs {self.in_dim} "
                    f"(feature_version {self.feature_version}); build the env with "
                    f"make_env_for_model(model) / ZipEnv(feature_version={self.feature_version})")
            x = x[:, :self.in_dim]  # feature versions are prefix-compatible
        n = x.shape[0]
        nbr = gb.nbr
        if nbr is None:
            raise ValueError("GraphBatch.nbr missing; build batches with collate()")
        valid = (nbr < n).float().unsqueeze(2)            # (N, D, 1)
        cnt = valid.sum(1)                                # (N, 1)
        free = 1.0 - x[:, F["visited"]] + x[:, F["is_head"]]
        free_w = torch.cat([free, free.new_zeros(1)])[nbr].unsqueeze(2) * valid
        B = gb.num_graphs
        h = self.enc(x)
        if self.head_version == 1 and not self.global_attn:
            for blk in self.blocks:
                h = blk(h, nbr, valid, cnt, free_w)
            g_mean = scatter_mean(h, gb.batch, B)
            g_max = scatter_max(h, gb.batch, B)
            h_head = h[gb.head]
            per_node = torch.cat([h, h_head[gb.batch], g_mean[gb.batch], g_max[gb.batch]], 1)
            logits = self.pi(per_node).squeeze(1)
            logits = logits.masked_fill(~gb.mask, float("-inf"))
            values = self.v(torch.cat([g_mean, g_max, h_head], 1)).squeeze(1)
            return logits, values
        tgt_w = x[:, F["is_next_cp"]].unsqueeze(1)
        end_w = x[:, F["is_final"]].unsqueeze(1)
        node_free = free > 0.5
        g_att = None
        for i, blk in enumerate(self.blocks):
            h = blk(h, nbr, valid, cnt, free_w)
            if self.global_attn:
                qctx = torch.cat([h[gb.head], scatter_sum(h * tgt_w, gb.batch, B)], 1)
                h, g_att = self.gblocks[i](h, qctx, gb.batch, B, node_free)
        g_mean = scatter_mean(h, gb.batch, B)
        g_max = scatter_max(h, gb.batch, B)
        h_head = h[gb.head]
        if self.head_version == 1:
            ctx_pi = [h_head, g_mean, g_max]
            ctx_v = [g_mean, g_max, h_head]
        else:
            h_tgt = scatter_sum(h * tgt_w, gb.batch, B)   # zero if no next checkpoint
            h_end = scatter_sum(h * end_w, gb.batch, B)
            ctx_pi = [h_head, h_tgt, h_end, g_mean, g_max]
            ctx_v = [g_mean, g_max, h_head, h_tgt, h_end]
            if g_att is not None:
                ctx_pi.append(g_att)
                ctx_v.append(g_att)
        ctx = torch.cat(ctx_pi, 1)
        logits = self.pi(torch.cat([h, ctx[gb.batch]], 1)).squeeze(1)
        logits = logits.masked_fill(~gb.mask, float("-inf"))
        values = self.v(torch.cat(ctx_v, 1)).squeeze(1)
        return logits, values


def make_model(hidden: int = 64, layers: int = 6, feature_version: int = LATEST_FEATURE_VERSION,
               head_version: int = LATEST_HEAD_VERSION, global_attn: bool = False,
               attn_heads: int = 4) -> ZipGNN:
    return ZipGNN(hidden=hidden, layers=layers, feature_version=feature_version,
                  head_version=head_version, global_attn=global_attn, attn_heads=attn_heads)


def _meta_config(meta) -> dict:
    if isinstance(meta, ZipGNN):
        return meta.config
    if "config" in meta:
        cfg = dict(meta["config"])
        if "feature_version" not in cfg and "feature_version" in meta:
            cfg["feature_version"] = meta["feature_version"]
        return cfg
    return dict(meta)


def feature_version_of(meta) -> int:
    """Feature version a model / checkpoint dict / config needs (old checkpoints -> 1)."""
    cfg = _meta_config(meta)
    if cfg.get("feature_version") is not None:
        return int(cfg["feature_version"])
    return feature_version_for_dim(int(cfg.get("in_dim", num_features(1))))


def env_kwargs_from_meta(meta) -> dict:
    """ZipEnv kwargs matching a model, a checkpoint dict (as returned by
    load_model) or a config dict: ``ZipEnv(**env_kwargs_from_meta(ck), ...)``."""
    return {"feature_version": feature_version_of(meta)}


def make_env_for_model(model=None, meta=None, puzzle_sampler=None, **env_kwargs) -> ZipEnv:
    """ZipEnv whose observations match a model and/or checkpoint dict.

    ``make_env_for_model(model)``, ``make_env_for_model(model, ck)`` or
    ``make_env_for_model(meta=ck)``; the model's own config wins, then `meta`,
    else the latest feature version. Extra kwargs go to ZipEnv (e.g.
    early_termination=False, reward=...)."""
    src = model if model is not None else meta
    kw = env_kwargs_from_meta(src) if src is not None else {}
    kw.update(env_kwargs)
    return ZipEnv(puzzle_sampler, **kw)


def dense_logits(logits: torch.Tensor, gb: GraphBatch) -> torch.Tensor:
    """(N,) flat logits -> (B, max_nodes) padded with -inf; column = local node id."""
    out = logits.new_full((gb.num_graphs, gb.max_nodes), float("-inf"))
    out[gb.batch, gb.local_index()] = logits
    return out


def masked_distribution(logits: torch.Tensor, gb: GraphBatch) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(logits=dense_logits(logits, gb))


def save_model(model: ZipGNN, path, **extra):
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "feature_version": model.feature_version, "head_version": model.head_version,
                **extra}, path)


def load_model(path, map_location="cpu") -> tuple[ZipGNN, dict]:
    """Returns (model in eval mode, checkpoint dict). The dict's "config" and
    top-level "feature_version" are filled in for old checkpoints (-> 1), so
    ``env_kwargs_from_meta(ck)`` / ``make_env_for_model(model)`` always work."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ck["config"])
    cfg.setdefault("feature_version", feature_version_of(ck))
    cfg.setdefault("head_version", 1)
    cfg.setdefault("global_attn", False)
    cfg.setdefault("attn_heads", 4)
    model = ZipGNN(**cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ck["config"] = model.config
    ck["feature_version"] = model.feature_version
    ck["head_version"] = model.head_version
    return model, ck


__all__ = ["GraphBatch", "collate", "ZipGNN", "dense_logits", "masked_distribution",
           "save_model", "load_model", "scatter_mean", "scatter_max", "scatter_sum", "make_model",
           "make_env_for_model", "env_kwargs_from_meta", "feature_version_of", "LATEST_HEAD_VERSION"]
