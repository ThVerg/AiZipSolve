// SVG board: layout (2D grid / 3D layer panels / 4D panel grid), incremental path rendering with
// micro-animations, overlays (legal moves, hints, AI heat + move probabilities, search trace), pointer input.
import { svgEl as el, cssVar, pathColor, AXIS, clamp, reducedMotion } from "./util.js";

// ------------------------------------------------------------------ puzzle model
export function makeModel(d) {
  const coords = d.coords.map((c) => c.map((x) => Math.round(x)));
  const n = coords.length;
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of d.edges) { adj[u].push(v); adj[v].push(u); }
  const m = d.meta || {};
  const ek = (u, v) => (u < v ? `${u},${v}` : `${v},${u}`);
  // new puzzle types (see the shared contract): all optional, old puzzles get the defaults
  const arcs = d.arcs || m.oneway || [];
  const blocked = new Set(arcs.map(([u, v]) => `${v}>${u}`));   // "a>b" = a -> b is forbidden
  const prec = new Map();                                            // node -> nodes that must come first
  for (const [a, b] of d.precedence || []) { if (!prec.has(b)) prec.set(b, []); prec.get(b).push(a); }
  const twin = new Map(), overpass = m.overpass || [];
  for (const o of overpass) { twin.set(o.h, o.v); twin.set(o.v, o.h); }
  const coopCps = d.kind === "coop" && m.coop && Array.isArray(m.coop.checkpoints) ? m.coop.checkpoints : null;
  const cps = coopCps ? coopCps[0] : d.checkpoints;
  const layout = m.layout || (d.kind === "hex" ? "hex" : d.kind === "tri" ? "tri" : "square");
  return {
    data: d, n, dim: coords[0].length, coords, adj, adjSet: adj.map((a) => new Set(a)),
    cps, cpIndex: new Map(cps.map((c, i) => [c, i])),
    island: d.meta && d.meta.island ? d.meta.island : null,
    bridges: !!(d.meta && d.meta.bridges),
    layout, arcs, blocked, prec, twin, overpass,
    keys: m.keys || (d.precedence || []).map(([a, b], i) => ({ key: a, door: b, color: i })),
    portals: m.portals || [], portalSet: new Set((m.portals || []).map(([u, v]) => ek(u, v))),
    wraps: m.wrap_edges || [], wrapSet: new Set((m.wrap_edges || []).map(([u, v]) => ek(u, v))), torus: m.torus || null,
    fog: !!m.fog, cube: m.cubesurf ? { n: m.cubesurf.n, face: m.face || [] } : null,
    coop: coopCps ? { cps: coopCps, cpIndex: coopCps.map((l) => new Map(l.map((c, i) => [c, i]))) } : null,
    ek,
  };
}
// nodes within graph distance `k` of `v` (fog of war)
export function near(P, v, k = 2) {
  const seen = new Set([v]);
  let front = [v];
  for (let d = 0; d < k; d++) {
    const nx = [];
    for (const u of front) for (const w of P.adj[u]) if (!seen.has(w)) { seen.add(w); nx.push(w); }
    front = nx;
  }
  return seen;
}
// path colours: path 1 = the warm gradient (and the classic one), co-op path 2 = a cool gradient
const COOL = ["#1fb7a6", "#1d9bd6", "#2f6fe0", "#6d4ae0"].map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
const WARM = ["#ffb000", "#ff9f1c", "#ff5a4e", "#d63a8f"].map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
function grad(stops, t) {
  t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(Math.floor(t), stops.length - 2), f = t - i, a = stops[i], b = stops[i + 1];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
export const coolColor = (t) => grad(COOL, t);
export const warmColor = (t) => grad(WARM, t);
export const PORTAL_COLORS = ["#19b8ff", "#ff8a00", "#20c997", "#ff4fa3", "#b04dff", "#ffd23f"];
export const KEY_COLORS = ["#f5b700", "#2fb8ff", "#ff5a8a", "#35c46a", "#a070ff"];

function buildLayout(P) {
  const C = P.coords, dim = P.dim;
  const rc = C.map((x) => dim === 1 ? [0, x[0]] : [x[0], x[1]]);
  const rmin = Math.min(...rc.map((p) => p[0])), cmin = Math.min(...rc.map((p) => p[1]));
  const R = Math.max(...rc.map((p) => p[0])) - rmin + 1, Cc = Math.max(...rc.map((p) => p[1])) - cmin + 1;
  const L = { R, C: Cc, dim, panels: [], panelOf: [], cell: [], lookup: new Map() };
  if (dim <= 2) {
    L.panels = [{ key: "0", label: "" }];
    L.panelOf = C.map(() => 0);
  } else if (dim === 3) {
    const zs = [...new Set(C.map((x) => x[2]))].sort((a, b) => a - b);
    L.panels = zs.map((z, i) => ({ key: String(z), label: `z = ${z}`, z, idx: i }));
    L.panelOf = C.map((x) => zs.indexOf(x[2]));
    L.zs = zs;
  } else {
    const zs = [...new Set(C.map((x) => x[2]))].sort((a, b) => a - b);
    const ws = [...new Set(C.map((x) => x[3]))].sort((a, b) => a - b);
    for (const z of zs) for (const w of ws) L.panels.push({ key: `${z},${w}`, label: `z=${z}  w=${w}`, z, w, pr: zs.indexOf(z), pc: ws.indexOf(w) });
    L.panelOf = C.map((x) => zs.indexOf(x[2]) * ws.length + ws.indexOf(x[3]));
    L.zs = zs; L.ws = ws;
  }
  L.cell = rc.map(([r, c]) => [r - rmin, c - cmin]);
  C.forEach((x, v) => L.lookup.set(x.join(","), v));
  L.cellKey = new Map();
  L.cell.forEach(([r, c], v) => L.cellKey.set(`${L.panelOf[v]}|${r}|${c}`, v));
  // cell centres in cell units (panel-relative); hex / triangle boards are not square grids
  L.geo = P.layout === "hex" || P.layout === "tri" ? P.layout : "square";
  if (L.geo === "hex") {
    // axial (q, r), pointy-top, unit = the hex's width
    const s = 1 / Math.sqrt(3);
    const raw = C.map(([q, r]) => [q + r / 2, r * 1.5 * s]);
    const x0 = Math.min(...raw.map((p) => p[0])) - 0.5, y0 = Math.min(...raw.map((p) => p[1])) - s;
    L.pos = raw.map(([x, y]) => [x - x0, y - y0]);
    L.C = Math.max(...L.pos.map((p) => p[0])) + 0.5; L.R = Math.max(...L.pos.map((p) => p[1])) + s;
    L.hexR = s;
  } else if (L.geo === "tri") {
    // (row, col): UP triangle if (row + col) even; side a, row height h
    const a = 1.2, hh = a * Math.sqrt(3) / 2;
    const r0 = Math.min(...C.map((x) => x[0])), c0 = Math.min(...C.map((x) => x[1]));
    L.triA = a; L.triH = hh;
    L.up = C.map(([r, c]) => ((r + c) % 2 + 2) % 2 === 0);
    L.pos = C.map(([r, c], v) => [(c - c0) * a / 2 + a / 2, (r - r0) * hh + (L.up[v] ? 2 * hh / 3 : hh / 3)]);
    L.C = (Math.max(...C.map((x) => x[1])) - c0) * a / 2 + a; L.R = (Math.max(...C.map((x) => x[0])) - r0 + 1) * hh;
  } else L.pos = L.cell.map(([r, c]) => [c + 0.5, r + 0.5]);
  return L;
}

export class Board {
  /** opts: {interactive, mini, maxCell, onDown(v), onDrag(v), onUp(), label} */
  constructor(host, opts = {}) {
    this.host = host;
    this.opts = opts;
    this.P = null; this.L = null; this.svg = null;
  }

  setPuzzle(P) { this.P = P; this.L = buildLayout(P); this.svg = null; }
  // island puzzles on a flat board get the sea / beach / plank-bridge look
  isIslands() { return !!(this.P && this.P.island && this.L && this.L.dim <= 2 && this.L.panels.length === 1); }

  // ---------------------------------------------------------------- geometry
  place(availW, availH) {
    const L = this.L, np = L.panels.length, mini = !!this.opts.mini;
    const labelH = np > 1 ? (mini ? 0.6 : 0.8) : 0, gap = np > 1 ? (mini ? 0.7 : 1.1) : 0;
    const m = np > 1 ? 0.25 : this.isIslands() ? (mini ? 0.3 : 0.42) : this.P.torus ? (mini ? 0.3 : 0.62) : 0.06;   // islands: beaches + sea; torus: the wrap-around preview
    const axisX = L.dim === 4 && !mini ? 1.2 : 0, axisY = L.dim === 4 && !mini ? 0.5 : 0;
    let best = null;
    const maxCell = this.opts.maxCell || 76;
    const tryGrid = (nr, nc) => {
      const W = nc * L.C + (nc - 1) * gap + axisX + 2 * m;
      const H = nr * (L.R + labelH) + (nr - 1) * gap + axisY + 2 * m;
      const cell = Math.min(availW / W, availH / H, maxCell);
      if (!best || cell > best.cell + 1e-9) best = { nr, nc, cell, W, H };
    };
    if (L.dim === 4) tryGrid(L.zs.length, L.ws.length);
    else for (let nc = 1; nc <= np; nc++) tryGrid(Math.ceil(np / nc), nc);
    const cell = Math.max(best.cell, mini ? 6 : 14);
    const offX = axisX + m, offY = axisY + m;
    L.panels.forEach((p, i) => {
      const pr = L.dim === 4 ? p.pr : Math.floor(i / best.nc), pc = L.dim === 4 ? p.pc : i % best.nc;
      p.gr = pr; p.gc = pc;
      p.x = (offX + pc * (L.C + gap)) * cell;
      p.y = (offY + pr * (L.R + labelH + gap)) * cell;
      p.cy = p.y + labelH * cell;
      p.w = L.C * cell; p.h = L.R * cell;
    });
    L.cs = cell; L.gap = gap; L.labelH = labelH; L.nr = best.nr; L.nc = best.nc; L.axisX = axisX;
    L.W = Math.ceil(best.W * cell) + 2; L.H = Math.ceil(best.H * cell) + 2;
  }
  center(v) {
    const L = this.L, p = L.panels[L.panelOf[v]], [x, y] = L.pos[v];
    return [p.x + x * L.cs, p.cy + y * L.cs];
  }
  samePanel(u, v) { return this.L.panelOf[u] === this.L.panelOf[v]; }
  gridAdjacent(u, v) {
    if (!this.samePanel(u, v)) return false;
    const P = this.P;
    if (P.portalSet.size || P.wrapSet.size) { const k = P.ek(u, v); if (P.portalSet.has(k) || P.wrapSet.has(k)) return false; }
    if (this.L.geo !== "square") return P.adjSet[u].has(v);
    const [a, b] = this.L.cell[u], [c, d] = this.L.cell[v];
    return Math.abs(a - c) + Math.abs(b - d) === 1;
  }
  edgeKind(u, v) {
    const P = this.P, k = P.ek(u, v);
    return P.portalSet.has(k) ? "portal" : P.wrapSet.has(k) ? "wrap" : null;
  }
  // unit step u -> v on a torus (the direction you leave u in)
  wrapDir(u, v) {
    const [a, b] = this.L.cell[u], [c, d] = this.L.cell[v];
    let dr = c - a, dc = d - b;
    if (Math.abs(dr) > 1) dr = -Math.sign(dr);
    if (Math.abs(dc) > 1) dc = -Math.sign(dc);
    return [dr, dc];
  }
  // outline of cell v (inset in px), as an SVG path; null for plain squares (drawn as rects)
  shapeD(v, inset = 0) {
    const L = this.L, cs = L.cs, [x, y] = this.center(v);
    if (L.geo === "hex") {
      const R = L.hexR * cs - inset / Math.cos(Math.PI / 6);
      let d = "";
      for (let i = 0; i < 6; i++) { const a = Math.PI / 6 + (i * Math.PI) / 3; d += `${i ? "L" : "M"}${(x + R * Math.cos(a)).toFixed(1)},${(y + R * Math.sin(a)).toFixed(1)}`; }
      return d + "Z";
    }
    if (L.geo === "tri") {
      const a = L.triA * cs, hh = L.triH * cs, up = L.up[v];
      // centroid -> vertices, pulled in by `inset` (incircle scaling)
      const rIn = hh / 3, f = Math.max(0.1, (rIn - inset) / rIn);
      const vs = up ? [[0, -2 * hh / 3], [a / 2, hh / 3], [-a / 2, hh / 3]] : [[-a / 2, -hh / 3], [a / 2, -hh / 3], [0, 2 * hh / 3]];
      return vs.map(([dx, dy], i) => `${i ? "L" : "M"}${(x + dx * f).toFixed(1)},${(y + dy * f).toFixed(1)}`).join("") + "Z";
    }
    return null;
  }
  arcD(u, v) {
    const { x1, y1, mx, my, x2, y2 } = this.arcGeom(u, v);
    return `M${x1},${y1} Q${mx},${my} ${x2},${y2}`;
  }
  arcGeom(u, v) {
    const [x1, y1] = this.center(u), [x2, y2] = this.center(v);
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    let nx = -dy / len, ny = dx / len;
    if (ny > 0 || (Math.abs(ny) < 1e-6 && nx > 0)) { nx = -nx; ny = -ny; }
    const ca = this.P.coords[u], cb = this.P.coords[v];
    const straight = ca.length === 2 && (ca[0] === cb[0] || ca[1] === cb[1]) && Math.abs(ca[0] - cb[0]) + Math.abs(ca[1] - cb[1]) <= 3;
    const bend = straight ? 0 : Math.min(0.28 * len, 2.2 * this.L.cs);
    const mx = (x1 + x2) / 2 + nx * bend, my = (y1 + y2) / 2 + ny * bend;
    return { x1, y1, mx, my, x2, y2 };
  }
  crossLabel(u, v) {
    const a = this.P.coords[u], b = this.P.coords[v], parts = [];
    for (let k = 2; k < a.length; k++) if (a[k] !== b[k]) parts.push(`${AXIS[k] || "d" + k}${b[k] > a[k] ? "+" : "−"}`);
    return parts.join("") || "↔";
  }
  dirLabel(from, to) {
    const a = this.P.coords[from], b = this.P.coords[to];
    const d = a.map((x, k) => b[k] - x);
    if (!this.gridAdjacent(from, to) && this.samePanel(from, to)) return "bridge";
    const arrows = [];
    if (d[0]) arrows.push(d[0] > 0 ? "↓" : "↑");
    if (d.length > 1 && d[1]) arrows.push(d[1] > 0 ? "→" : "←");
    for (let k = 2; k < d.length; k++) if (d[k]) arrows.push(`${AXIS[k]}${d[k] > 0 ? "+" : "−"}`);
    return arrows.join("") || "·";
  }
  cellAt(x, y, inner = 1) {
    // node under board coordinates (x, y); `inner` < 1 only accepts the central part of a cell
    const L = this.L;
    if (L.geo !== "square") {
      // nearest centre (hex: inside the inscribed circle-ish; triangles: the incircle, a bit generous)
      let best = null, bd = Infinity;
      for (let v = 0; v < this.P.n; v++) { const [cx, cy] = this.center(v); const d = Math.hypot(cx - x, cy - y); if (d < bd) { bd = d; best = v; } }
      const lim = (L.geo === "hex" ? 0.5 : L.triH * 0.5) * L.cs;
      if (bd > lim * (inner < 1 ? inner : 1.15)) return inner < 1 && bd <= lim * 1.15 ? undefined : null;
      return best;
    }
    for (let i = 0; i < L.panels.length; i++) {
      const p = L.panels[i];
      if (x < p.x || x >= p.x + p.w || y < p.cy || y >= p.cy + p.h) continue;
      const fx = (x - p.x) / L.cs, fy = (y - p.cy) / L.cs;
      const c = Math.floor(fx), r = Math.floor(fy);
      if (inner < 1) {
        const ox = Math.abs(fx - c - 0.5), oy = Math.abs(fy - r - 0.5);
        if (ox > inner / 2 || oy > inner / 2) return undefined;   // dead zone between cells
      }
      const v = L.cellKey.get(`${i}|${r}|${c}`);
      return v === undefined ? null : v;
    }
    return null;
  }
  toBoard(cx, cy) {
    const L = this.L, r = this.svg.getBoundingClientRect();
    return [((cx - r.left) * L.W) / r.width, ((cy - r.top) * L.H) / r.height];
  }
  nodeAtClient(cx, cy) { const [x, y] = this.toBoard(cx, cy); return this.cellAt(x, y); }
  clientOf(v) {
    const [x, y] = this.center(v), r = this.svg.getBoundingClientRect();
    return { x: r.left + (x * r.width) / this.L.W, y: r.top + (y * r.height) / this.L.H };
  }

  // ---------------------------------------------------------------- static layer
  build(avW, avH) {
    const P = this.P;
    this.host.innerHTML = "";
    this.place(avW, avH);
    const L = this.L, cs = L.cs, mini = !!this.opts.mini;
    const col = this.col = {
      cell: cssVar("--cell"), edge: cssVar("--cell-edge"), wall: cssVar("--wall"), bridge: cssVar("--bridge"),
      cp: cssVar("--cp"), cpInk: cssVar("--cp-ink"), cpRing: cssVar("--cp-ring"), accent: cssVar("--accent"),
      legal: cssVar("--legal"), good: cssVar("--good"), bad: cssVar("--bad"), warn: cssVar("--warn"),
      panel: cssVar("--panel"), panel2: cssVar("--panel-2"), border: cssVar("--border"), ink3: cssVar("--ink-3"),
      heat: cssVar("--heat") || "#ff6a3d", ink: cssVar("--ink"),
    };
    const svg = this.svg = el("svg", { viewBox: `0 0 ${L.W} ${L.H}`, width: L.W, height: L.H, class: `zboard${mini ? " mini" : ""}`,
      role: this.opts.interactive ? "application" : "img", "aria-label": this.opts.label || "Zip puzzle board" }, this.host);
    if (this.opts.interactive) svg.setAttribute("tabindex", "0");
    const defs = el("defs", {}, svg);
    const glow = el("radialGradient", { id: `glow${this._uid = (Board._n = (Board._n || 0) + 1)}` }, defs);
    el("stop", { offset: "0", "stop-color": "#fff", "stop-opacity": 0.85 }, glow);
    this.glowStops = [
      el("stop", { offset: "0.4", "stop-color": pathColor(0), "stop-opacity": 0.5 }, glow),
      el("stop", { offset: "1", "stop-color": pathColor(0), "stop-opacity": 0 }, glow),
    ];

    const gS = el("g", { class: "g-static" }, svg);
    if (L.panels.length > 1) {
      for (const p of L.panels) {
        el("rect", { x: p.x - 0.18 * cs, y: p.cy - 0.18 * cs, width: p.w + 0.36 * cs, height: p.h + 0.36 * cs, rx: 0.3 * cs,
          fill: col.panel2, stroke: col.border, "stroke-width": 1.2, class: "pframe" }, gS);
        if (!mini || cs >= 9) {
          const t = el("text", { x: p.x, y: p.y + 0.5 * cs, "font-size": clamp(0.3 * cs, mini ? 8 : 10, 15), fill: col.ink3, "font-weight": 650 }, gS);
          t.textContent = p.label;
        }
      }
      if (!mini) {
        const chip = (x, y, text) => {
          const fs = clamp(0.26 * cs, 9, 13), w = fs * (text.length * 0.62 + 1.2), hh = fs * 1.5;
          el("rect", { x: x - w / 2, y: y - hh / 2, width: w, height: hh, rx: hh / 2, fill: col.panel, stroke: col.border }, gS);
          const t = el("text", { x, y: y + fs * 0.36, "font-size": fs, "text-anchor": "middle", fill: col.ink3, "font-weight": 650 }, gS);
          t.textContent = text;
        };
        const byGrid = new Map(L.panels.map((p) => [`${p.gr},${p.gc}`, p]));
        for (const p of L.panels) {
          const right = byGrid.get(`${p.gr},${p.gc + 1}`), down = byGrid.get(`${p.gr + 1},${p.gc}`);
          if (right && (L.dim === 4 || L.zs.indexOf(right.z) === L.zs.indexOf(p.z) + 1))
            chip(p.x + p.w + L.gap * cs / 2, p.cy + p.h / 2, L.dim === 4 ? "↔ w" : "↔ z");
          if (down && L.dim === 4) chip(p.x + p.w / 2, p.cy + p.h + (L.gap * cs) / 2 + 0.05 * cs, "↕ z");
        }
        if (L.dim === 4) {
          const fs = clamp(0.3 * cs, 10, 15);
          el("text", { x: 1.45 * cs, y: 0.5 * cs, "font-size": fs, fill: col.ink3, "font-weight": 700 }, gS).textContent = "w →";
          el("text", { x: 0.25 * cs, y: 1.6 * cs, "font-size": fs, fill: col.ink3, "font-weight": 700 }, gS).textContent = "z ↓";
        }
      }
    }
    const pad = Math.max(1, cs * 0.045), rx = cs * 0.2;
    const isl = this.isIslands();
    this.bglowId = null;
    if (isl) this.drawSea(gS, defs);
    this.cellEls = [];
    const tile = isl ? cssVar("--tile") : null, tileEdge = isl ? cssVar("--tile-edge") : null;
    if (P.torus) this.drawTorusGhost(gS, defs);
    for (let v = 0; v < P.n; v++) {
      const [x, y] = this.center(v);
      const fill = isl ? tile : P.island ? cssVar(`--island-${P.island[v] % 8}`) : col.cell;
      const sd = this.shapeD(v, pad * 1.4);
      if (P.twin.has(v) && P.twin.get(v) < v) { this.cellEls.push(this.cellEls[P.twin.get(v)]); continue; }   // overpass: one tile, two nodes
      this.cellEls.push(sd ? el("path", { d: sd, fill, stroke: col.edge, "stroke-width": mini ? 0.6 : 1, "stroke-linejoin": "round", class: "cell", "data-node": v }, gS)
        : el("rect", { x: x - cs / 2 + pad, y: y - cs / 2 + pad, width: cs - 2 * pad, height: cs - 2 * pad, rx, fill,
          stroke: isl ? tileEdge : col.edge, "stroke-width": mini ? 0.6 : 1, class: "cell", "data-node": v }, gS));
    }
    const wallW = Math.max(mini ? 1.5 : 3, cs * 0.12);
    for (let v = 0; v < P.n; v++) {
      if (L.geo !== "square") break;
      const pi = L.panelOf[v], [r, c] = L.cell[v];
      for (const [dr, dc] of [[0, 1], [1, 0]]) {
        const w = L.cellKey.get(`${pi}|${r + dr}|${c + dc}`);
        if (w === undefined || P.adjSet[v].has(w) || (P.twin.size && [v, P.twin.get(v)].some((a) => a !== undefined && [w, P.twin.get(w)].some((b) => b !== undefined && P.adjSet[a].has(b))))) continue;
        const [x1, y1] = this.center(v), [x2, y2] = this.center(w);
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, hh = cs / 2 - 1;
        el("line", { x1: dc ? mx : mx - hh, y1: dc ? my - hh : my, x2: dc ? mx : mx + hh, y2: dc ? my + hh : my,
          stroke: col.wall, "stroke-width": wallW, "stroke-linecap": "round" }, gS);
      }
    }
    for (const [u, v] of P.data.edges) {
      if (!this.samePanel(u, v) || this.gridAdjacent(u, v) || this.edgeKind(u, v)) continue;
      if (isl) { this.drawBridge(u, v, gS); continue; }
      el("path", { d: this.arcD(u, v), fill: "none", stroke: col.bridge, "stroke-width": Math.max(mini ? 1 : 2, cs * 0.06),
        "stroke-dasharray": `${cs * 0.18} ${cs * 0.14}`, "stroke-linecap": "round", opacity: 0.85, class: "bridge" }, gS);
    }
    this.drawExtras(gS, defs);
    this.gHeat = el("g", { class: "g-heat" }, svg);
    this.gTried = el("g", { class: "g-tried" }, svg);
    this.gTint = el("g", { class: "g-tint" }, svg);
    this.gPath = el("g", { class: "g-path" }, svg);
    // invisible anchor: keeps the group's bounding box non-degenerate, so filters on it (glow) never clip a straight line
    el("rect", { x: 0, y: 0, width: L.W, height: L.H, fill: "none", stroke: "none", class: "bbox-anchor" }, this.gPath);
    // overpass: the deck of each bridge + the path that runs over it (drawn above the path underneath)
    this.gOver = null;
    if (P.overpass.length) {
      this.gOver = el("g", { class: "g-over" }, svg);
      this.drawOverDecks(this.gOver);
      this.gOverPath = el("g", { class: "g-path g-overpath" }, this.gOver);
      el("rect", { x: 0, y: 0, width: L.W, height: L.H, fill: "none", stroke: "none", class: "bbox-anchor" }, this.gOverPath);
    }
    this.gHead = el("g", { class: "g-head" }, svg);
    this.gCp = el("g", { class: "g-cp" }, svg);
    this.gMarks = el("g", { class: "g-marks" }, svg);
    this.gFx = el("g", { class: "g-fx" }, svg);
    // per-node tints (visited fill), created once and toggled
    this.tints = [];
    const tp = pad + cs * 0.02;
    for (let v = 0; v < P.n; v++) {
      const [x, y] = this.center(v);
      const sd = this.shapeD(v, tp * 1.4);
      if (P.twin.has(v)) {   // overpass lanes: the H node tints a horizontal band, the V node a vertical one
        const o = P.overpass.find((q) => q.h === v || q.v === v), hz = o.h === v, w = cs * 0.46;
        this.tints.push(el("rect", hz ? { x: x - cs / 2 + tp, y: y - w / 2, width: cs - 2 * tp, height: w, rx: w * 0.3, class: "tint" }
          : { x: x - w / 2, y: y - cs / 2 + tp, width: w, height: cs - 2 * tp, rx: w * 0.3, class: "tint" }, this.gTint));
        continue;
      }
      this.tints.push(sd ? el("path", { d: sd, class: "tint" }, this.gTint)
        : el("rect", { x: x - cs / 2 + tp, y: y - cs / 2 + tp, width: cs - 2 * tp, height: cs - 2 * tp, rx, class: "tint" }, this.gTint));
    }
    this.triedEls = null;
    // checkpoints
    const cpR = cs * (L.geo === "tri" ? 0.24 : L.geo === "hex" ? 0.27 : 0.3);
    const mkCp = (v, k, fill, ring, cls = "") => {
      const [x, y] = this.center(v);
      const g = el("g", { class: `cp${cls}`, transform: `translate(${x} ${y})` }, this.gCp);
      const inner = el("g", { class: "cp-inner" }, g);
      el("circle", { r: cpR, fill, stroke: ring, "stroke-width": mini ? 1 : 1.5 }, inner);
      if (!mini || cs >= 12) {
        const t = el("text", { y: cpR * 0.37, "font-size": cpR * (k + 1 >= 100 ? 0.73 : k + 1 >= 10 ? 0.9 : 1.07), "text-anchor": "middle", "font-weight": 800, fill: col.cpInk, class: "cp-num" }, inner);
        t.textContent = k + 1;
      }
      return g;
    };
    const coop = P.coop;
    this.cpEls = P.cps.map((v, k) => mkCp(v, k, coop ? "#f26a1b" : col.cp, coop ? "#fff" : col.cpRing, coop ? " cp-warm" : ""));
    this.cpEls2 = coop ? coop.cps[1].map((v, k) => mkCp(v, k, "#2466d8", "#fff", " cp-cool")) : [];
    this.drawKeys();
    this.fogEls = null; this._fogShown = null;
    if (P.fog) this.drawFog();
    // head (co-op: one per path)
    const mkHead = () => {
      const hEl = el("g", { class: "head" }, this.gHead);
      hEl.style.color = pathColor(0);
      el("circle", { r: cs * 0.62, fill: `url(#glow${this._uid})`, class: "head-glow" }, hEl);
      const dot = el("circle", { r: cs * 0.19, fill: "currentColor", stroke: "#fff", "stroke-width": Math.max(mini ? 1 : 2, cs * 0.06), class: "head-dot" }, hEl);
      hEl.style.display = "none";
      return [hEl, dot];
    };
    [this.headEl, this.headDot] = mkHead();
    this.overV = new Set(P.overpass.map((o) => o.v));
    this.st2 = null;
    if (coop) { const [h2, d2] = mkHead(); this.st2 = { segs: [], rPath: [], headEl: h2, headDot: d2 }; h2.style.color = coolColor(0); h2.querySelector(".head-glow").setAttribute("fill", "none"); }
    this.segs = [];          // rendered segment elements, index i = path[i] -> path[i+1]
    this.rPath = [];         // path as rendered
    this._heatRef = undefined;
    this._keyState = null;
    if (this.opts.interactive) this.attachPointer();
    return svg;
  }

  // ---------------------------------------------------------------- dynamic layer
  // colour of path step i (co-op: path 1 warm, path 2 cool, each spread over about half the board)
  colorAt(i, which = 0) {
    const P = this.P;
    if (P.coop) { const t = Math.min(1, i / Math.max(1, P.n / 2 - 1)); return which ? coolColor(t) : warmColor(t); }
    return pathColor(P.n > 1 ? i / (P.n - 1) : 0);
  }

  makeSeg(i, u, v, which = 0) {
    const cs = this.L.cs, sw = cs * (this.L.geo === "tri" ? 0.3 : 0.36);
    const col = this.colorAt(i + 0.5, which);
    if (!this.samePanel(u, v)) return null;
    const kind = this.edgeKind(u, v);
    const layer = this.gOverPath && (this.overV.has(u) || this.overV.has(v)) ? this.gOverPath : this.gPath;
    let e;
    if (kind === "portal") {
      // teleport: a faint sparkly thread between the two rings (the line "jumps")
      e = el("path", { d: this.arcD(u, v), fill: "none", stroke: col, "stroke-width": Math.max(2, sw * 0.28), "stroke-linecap": "round",
        "stroke-dasharray": `0.1 ${(cs * 0.22).toFixed(1)}`, opacity: 0.75, class: "seg portal-seg" }, layer);
    } else if (kind === "wrap") {
      // torus: leave through one border, come back in through the opposite one
      const [dr, dc] = this.wrapDir(u, v), [x1, y1] = this.center(u), [x2, y2] = this.center(v), d = cs * 0.62;
      e = el("g", { class: "seg wrap-seg" }, layer);
      el("line", { x1, y1, x2: x1 + dc * d, y2: y1 + dr * d, stroke: col, "stroke-width": sw, "stroke-linecap": "round", pathLength: 1 }, e);
      el("line", { x1: x2 - dc * d, y1: y2 - dr * d, x2, y2, stroke: col, "stroke-width": sw, "stroke-linecap": "round", pathLength: 1 }, e);
    } else if (this.gridAdjacent(u, v)) {
      const [x1, y1] = this.center(u), [x2, y2] = this.center(v);
      e = el("line", { x1, y1, x2, y2, stroke: col, "stroke-width": sw, "stroke-linecap": "round", pathLength: 1, class: "seg" }, layer);
    } else {
      e = el("path", { d: this.arcD(u, v), fill: "none", stroke: col, "stroke-width": sw * 0.8, "stroke-linecap": "round", pathLength: 1, class: "seg bridge-seg" }, layer);
      if (this.bglowId) e.setAttribute("filter", `url(#${this.bglowId})`);
    }
    return e;
  }

  // incremental segments of one path layer (st = this for path 1, this.st2 for co-op path 2); returns the kept prefix
  syncPath(st, path, animate, which) {
    let k = 0;
    const old = st.rPath;
    while (k < old.length && k < path.length && old[k] === path[k]) k++;
    const keepSegs = Math.max(0, k - 1);
    const removed = st.segs.splice(keepSegs);
    const nRemoved = old.length - k;
    for (const s of removed) {
      if (!s) continue;
      if (animate && nRemoved <= 6) {
        s.classList.remove("seg-in"); s.classList.add("seg-out");
        setTimeout(() => s.remove(), 170);
      } else s.remove();
    }
    const added = path.length - Math.max(k, 1);
    for (let i = keepSegs; i + 1 < path.length; i++) {
      const s = this.makeSeg(i, path[i], path[i + 1], which);
      if (s && animate && added <= 4 && i + 1 >= k) {
        if (!s.classList.contains("portal-seg")) s.classList.add("seg-in");
        if (s.classList.contains("bridge-seg") && this.isIslands()) this.splash(path[i], path[i + 1]);
        if (s.classList.contains("portal-seg")) this.teleport(path[i], path[i + 1], this.colorAt(i + 1, which));
      }
      st.segs.push(s);
    }
    return k;
  }

  render(V) {
    if (!this.svg) return;
    const P = this.P, L = this.L, cs = L.cs, col = this.col, mini = !!this.opts.mini;
    const path = V.path || [];
    const path2 = (P.coop && V.path2) || [];
    const animate = V.animate !== false && !mini;
    const old = this.rPath, old2 = this.st2 ? this.st2.rPath : [];
    const k = this.syncPath(this, path, animate, 0);
    const k2 = this.st2 ? this.syncPath(this.st2, path2, animate, 1) : 0;
    // ---- tints (visited cells)
    const onPath = new Map(), onPath2 = new Map();
    path.forEach((v, i) => onPath.set(v, i));
    path2.forEach((v, i) => onPath2.set(v, i));
    const offTint = (v) => { const t = this.tints[v]; if (t && !onPath.has(v) && !onPath2.has(v)) t.classList.remove("on", "pop"); };
    for (let i = k; i < old.length; i++) offTint(old[i]);
    for (let i = k2; i < old2.length; i++) offTint(old2[i]);
    const onTint = (pp, kk, which) => {
      for (let i = Math.min(kk, pp.length); i < pp.length; i++) {
        const t = this.tints[pp[i]];
        t.setAttribute("fill", this.colorAt(i, which));
        t.classList.add("on");
        if (animate && pp.length - kk <= 4) { t.classList.remove("pop"); void t.getBBox(); t.classList.add("pop"); }
      }
      if (kk === 0 && pp.length) { const t = this.tints[pp[0]]; t.setAttribute("fill", this.colorAt(0, which)); t.classList.add("on"); }
    };
    onTint(path, k, 0);
    onTint(path2, k2, 1);
    // ---- checkpoints: reached state + pop when newly reached
    let next = 0, next2 = 0;
    P.cps.forEach((v, ci) => {
      const idx = onPath.get(v);
      const g = this.cpEls[ci];
      const reached = idx !== undefined;
      if (reached) next = Math.max(next, ci + 1);
      g.classList.toggle("reached", reached);
      if (reached && idx >= old.length && animate && ci > 0) this.popCp(ci);
    });
    if (P.coop) P.coop.cps[1].forEach((v, ci) => {
      const idx = onPath2.get(v), g = this.cpEls2[ci], reached = idx !== undefined;
      if (reached) next2 = Math.max(next2, ci + 1);
      g.classList.toggle("reached", reached);
      if (reached && idx >= old2.length && animate && ci > 0) this.popCp(ci, 1);
    });
    this.rPath = path.slice();
    if (this.st2) this.st2.rPath = path2.slice();
    // ---- keys & doors, fog
    if (P.keys.length) this.updateKeys(onPath, onPath2, animate);
    if (P.fog) this.updateFog(V.fogFrom || [path[path.length - 1], path2[path2.length - 1]].filter((x) => x !== undefined), animate);
    // ---- marks (rebuilt every render)
    const gm = this.gMarks;
    gm.textContent = "";
    const won = V.won, head = path[path.length - 1];
    // panel highlight (head panel + panels with legal moves)
    const legal = V.legal || [];
    if (L.panels.length > 1 && !won && path.length && !mini) {
      const hp = L.panelOf[head];
      const lp = new Set(legal.map((v) => L.panelOf[v]));
      L.panels.forEach((p, i) => {
        if (i !== hp && !lp.has(i)) return;
        el("rect", { x: p.x - 0.18 * cs, y: p.cy - 0.18 * cs, width: p.w + 0.36 * cs, height: p.h + 0.36 * cs, rx: 0.3 * cs,
          fill: "none", stroke: col.accent, "stroke-width": i === hp ? 2.4 : 1.4,
          "stroke-dasharray": i === hp ? null : "5 4", opacity: i === hp ? 0.9 : 0.7 }, gm);
      });
    }
    // start dot (path[0] is a checkpoint, drawn under it) + cross-panel badges
    if (!mini) {
      for (let i = 0; i + 1 < path.length; i++) {
        const u = path[i], v = path[i + 1];
        if (this.samePanel(u, v)) continue;
        this.badge(u, v, i, true); this.badge(v, u, i + 1, false);
      }
    }
    // agent confidence meters (only uncertain steps)
    if (V.conf && !mini) {
      for (const v of path) {
        const p = V.conf.get(v);
        if (p === undefined || p >= 0.9) continue;
        const [x, y] = this.center(v);
        const bw = cs * 0.34, bx = x - cs / 2 + 6, by = y - cs / 2 + 6;
        el("rect", { x: bx, y: by, width: bw, height: 4, rx: 2, fill: col.panel, opacity: 0.9 }, gm);
        el("rect", { x: bx, y: by, width: Math.max(2, bw * p), height: 4, rx: 2, fill: p < 0.5 ? col.warn : col.accent }, gm);
      }
    }
    // legal moves
    if (V.showLegal && !V.anim) {
      for (const v of legal) {
        const [x, y] = this.center(v);
        el("circle", { cx: x, cy: y, r: cs * 0.1, fill: col.legal, opacity: 0.8, class: "legal-dot" }, gm);
      }
    }
    // agent candidate rings during playback
    if (V.curStep && V.anim && !mini) {
      for (const [v, p] of V.curStep.top || []) {
        if (v === V.curStep.node) continue;
        const [x, y] = this.center(v);
        el("circle", { cx: x, cy: y, r: cs * 0.36, fill: "none", stroke: col.accent, "stroke-width": 2.5, opacity: Math.max(0.15, p) }, gm);
      }
    }
    // policy: move probabilities on candidate cells
    if (V.policy && V.policy.length && !won) {
      const best = V.policy[0][1];
      for (const [v, p] of V.policy) {
        const [x, y] = this.center(v);
        const top = p === best;
        el("circle", { cx: x, cy: y, r: cs * (0.16 + 0.26 * Math.sqrt(p)), fill: col.heat, opacity: 0.18 + 0.5 * p, class: "pol-blob" }, gm);
        if (top) el("circle", { cx: x, cy: y, r: cs * 0.43, fill: "none", stroke: col.heat, "stroke-width": Math.max(2, cs * 0.05), class: "pol-top" }, gm);
        if (!mini && cs >= 26 && !P.cpIndex.has(v)) {
          const fs = clamp(cs * 0.22, 9, 15), txt = p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`;
          const w = fs * (txt.length * 0.6 + 0.9), hh = fs * 1.35;
          el("rect", { x: x - w / 2, y: y + cs * 0.5 - hh - 3, width: w, height: hh, rx: hh / 2, fill: col.panel, stroke: col.heat, "stroke-width": 1.2, opacity: 0.95 }, gm);
          el("text", { x, y: y + cs * 0.5 - hh / 2 - 3 + fs * 0.36, "font-size": fs, "text-anchor": "middle", "font-weight": 750, fill: col.ink, class: "pol-txt" }, gm).textContent = txt;
        }
      }
    }
    // next checkpoint ring
    if (!won && next < P.cps.length && path.length && !(P.fog && this._fogShown && !this._fogShown.has(P.cps[next]))) {
      const [x, y] = this.center(P.cps[next]);
      el("circle", { cx: x, cy: y, r: cs * 0.38, fill: "none", stroke: P.coop ? "#f26a1b" : col.accent, "stroke-width": mini ? 1.2 : 2, opacity: 0.6, class: "next-ring" }, gm);
    }
    if (P.coop && !won && next2 < P.coop.cps[1].length && path2.length) {
      const [x, y] = this.center(P.coop.cps[1][next2]);
      el("circle", { cx: x, cy: y, r: cs * 0.38, fill: "none", stroke: "#2466d8", "stroke-width": mini ? 1.2 : 2, opacity: 0.6, class: "next-ring" }, gm);
    }
    // hint / stuck markers
    if (V.hint) {
      const [x, y] = this.center(V.hint.node);
      el("circle", { cx: x, cy: y, r: cs * 0.42, fill: "none", stroke: col.good, "stroke-width": 3.5, class: "pulse" }, gm);
      if (V.hint.keep && V.hint.keep < path.length) {
        const [bx, by] = this.center(path[V.hint.keep - 1]);
        el("circle", { cx: bx, cy: by, r: cs * 0.44, fill: "none", stroke: col.warn, "stroke-width": 3.5, "stroke-dasharray": "6 4" }, gm);
      }
    }
    if (V.stuck) {
      const [x, y] = this.center(V.stuck.node);
      el("circle", { cx: x, cy: y, r: cs * 0.42, fill: "none", stroke: col.bad, "stroke-width": mini ? 2 : 3.5, class: "pulse" }, gm);
      if (V.stuck.deadNode !== undefined && V.stuck.deadNode !== V.stuck.node) {
        const [bx, by] = this.center(V.stuck.deadNode);
        el("circle", { cx: bx, cy: by, r: cs * 0.44, fill: "none", stroke: col.warn, "stroke-width": mini ? 1.5 : 3, "stroke-dasharray": "5 4" }, gm);
      }
    }
    // ---- head
    const showHead = path.length > 1 && !won;
    this.headEl.style.display = showHead ? "" : "none";
    if (showHead) {
      const [x, y] = this.center(head);
      const moveAnim = animate && path.length - k <= 2 && old.length && Math.abs(path.length - old.length) <= 2;
      this.headEl.classList.toggle("glide", !!moveAnim);
      this.headEl.style.transform = `translate(${x}px, ${y}px)`;
      const hc = this.colorAt(path.length - 1);
      this.headEl.style.color = hc;
      if (this._glowCol !== hc) { this._glowCol = hc; for (const s of this.glowStops) s.setAttribute("stop-color", hc); }
      this.headDot.style.display = P.cpIndex.has(head) ? "none" : "";
    }
    if (this.st2) {
      const S2 = this.st2, h2 = path2[path2.length - 1], show2 = path2.length > 1 && !won;
      S2.headEl.style.display = show2 ? "" : "none";
      if (show2) {
        const [x, y] = this.center(h2);
        S2.headEl.classList.toggle("glide", animate);
        S2.headEl.style.transform = `translate(${x}px, ${y}px)`;
        S2.headEl.style.color = this.colorAt(path2.length - 1, 1);
        S2.headDot.style.display = P.coop.cpIndex[1].has(h2) ? "none" : "";
      }
      // the pen that is drawing right now
      this.headEl.classList.toggle("pen-off", V.active === 1);
      S2.headEl.classList.toggle("pen-off", V.active !== 1);
    }
    this.svg.classList.toggle("won", !!won);
    // ---- heat / tried layers
    if (V.heat !== this._heatRef) { this._heatRef = V.heat; this.drawHeat(V.heat); }
    if (V.tried) this.drawTried(V.tried);
    else if (this.triedEls) { this.gTried.textContent = ""; this.triedEls = null; }
  }

  // ---------------------------------------------------------------- new puzzle types: static decorations
  drawExtras(g) {
    const P = this.P, cs = this.L.cs, mini = !!this.opts.mini;
    // portals: paired swirling rings, one colour per pair
    this.portalEls = new Map();
    const pc = (this.P.data.meta && this.P.data.meta.portal_colors) || null;
    P.portals.forEach(([a, b], i) => {
      const c0 = pc ? pc[i] : i, color = typeof c0 === "string" ? c0 : PORTAL_COLORS[(Number(c0) || 0) % PORTAL_COLORS.length];
      for (const v of [a, b]) {
        const [x, y] = this.center(v);
        const pg = el("g", { class: "portal", transform: `translate(${x} ${y})`, style: `color:${color}` }, g);
        el("circle", { r: cs * 0.42, fill: "currentColor", opacity: 0.16 }, pg);
        const sw = el("g", { class: "portal-spin" }, pg);
        el("circle", { r: cs * 0.4, fill: "none", stroke: "currentColor", "stroke-width": Math.max(1.5, cs * 0.07), "stroke-dasharray": `${(cs * 0.5).toFixed(1)} ${(cs * 0.13).toFixed(1)}`, "stroke-linecap": "round" }, sw);
        if (!mini) {
          const sw2 = el("g", { class: "portal-spin rev" }, pg);
          el("circle", { r: cs * 0.3, fill: "none", stroke: "currentColor", "stroke-width": Math.max(1, cs * 0.04), opacity: 0.7, "stroke-dasharray": `${(cs * 0.22).toFixed(1)} ${(cs * 0.16).toFixed(1)}`, "stroke-linecap": "round" }, sw2);
        }
        this.portalEls.set(v, pg);
      }
    });
    // overpass: the left/right lane is a road that runs under the bridge deck (deck drawn above the path)
    for (const o of P.overpass) {
      const [x, y] = this.center(o.h), rw = cs * 0.5;
      el("rect", { x: x - cs / 2 + 1, y: y - rw / 2, width: cs - 2, height: rw, fill: this.col.edge, opacity: 0.7 }, g);
      el("line", { x1: x - cs / 2 + 3, y1: y, x2: x + cs / 2 - 3, y2: y, stroke: this.col.panel, "stroke-width": Math.max(1, cs * 0.03), "stroke-dasharray": `${cs * 0.08} ${cs * 0.07}` }, g);
    }
    // one-way edges: a chevron on the border between the two cells, pointing the only way you may cross
    this.arcEls = new Map();
    for (const [u, v] of P.arcs) {
      const [x1, y1] = this.center(u), [x2, y2] = this.center(v);
      const ang = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI, r = cs * 0.17;
      const ag = el("g", { class: "oneway", transform: `translate(${((x1 + x2) / 2).toFixed(1)} ${((y1 + y2) / 2).toFixed(1)}) rotate(${ang.toFixed(1)})` }, g);
      const inner = el("g", { class: "ow-inner" }, ag);
      el("circle", { r, fill: this.col.ink, opacity: 0.9 }, inner);
      el("path", { d: `M${-r * 0.35},${-r * 0.55} L${r * 0.3},0 L${-r * 0.35},${r * 0.55}`, fill: "none", stroke: this.col.panel, "stroke-width": Math.max(1.5, r * 0.34), "stroke-linecap": "round", "stroke-linejoin": "round" }, inner);
      this.arcEls.set(`${u}>${v}`, inner);
    }
  }
  drawTorusGhost(g, defs) {
    // a faint preview of the board's opposite edges just past each border: the board visibly continues
    const L = this.L, P = this.P, cs = L.cs, p = L.panels[0], m = cs * (this.opts.mini ? 0.3 : 0.62);
    const cid = `tc${this._uid}`;
    const cp = el("clipPath", { id: cid }, defs);
    el("rect", { x: p.x - m, y: p.cy - m, width: p.w + 2 * m, height: p.h + 2 * m, rx: m }, cp);
    const gg = el("g", { class: "torus-ghost", "clip-path": `url(#${cid})` }, g);
    const put = (v, dr, dc) => {
      const [x, y] = this.center(v), X = x + dc * p.w, Y = y + dr * p.h, pad = cs * 0.06;
      el("rect", { x: X - cs / 2 + pad, y: Y - cs / 2 + pad, width: cs - 2 * pad, height: cs - 2 * pad, rx: cs * 0.2, fill: this.col.cell, stroke: this.col.edge, opacity: 0.55 }, gg);
      const k = P.cpIndex.get(v);
      if (k !== undefined && !this.opts.mini) {
        el("circle", { cx: X, cy: Y, r: cs * 0.3, fill: this.col.cp, opacity: 0.3 }, gg);
      }
    };
    for (let v = 0; v < P.n; v++) {
      const [r, c] = L.cell[v];
      if (c === 0) put(v, 0, 1);
      if (c === L.C - 1) put(v, 0, -1);
      if (r === 0) put(v, 1, 0);
      if (r === L.R - 1) put(v, -1, 0);
    }
    // soft fade towards the outside + a gently flowing dashed border ("this edge wraps")
    el("rect", { x: p.x - 1, y: p.cy - 1, width: p.w + 2, height: p.h + 2, rx: cs * 0.22, fill: "none", stroke: this.col.accent, "stroke-width": Math.max(1.5, cs * 0.04),
      "stroke-dasharray": `${(cs * 0.18).toFixed(1)} ${(cs * 0.14).toFixed(1)}`, opacity: 0.55, class: "torus-edge" }, g);
  }
  drawOverDecks(g) {
    // overpass: the up/down lane is a little bridge deck with rails and a shadow, over the left/right lane
    const cs = this.L.cs;
    for (const o of this.P.overpass) {
      const [x, y] = this.center(o.v), w = cs * 0.52, hh = cs * 1.02;
      const dg = el("g", { class: "ov-deck", transform: `translate(${x} ${y})` }, g);
      el("rect", { x: -w / 2, y: -hh / 2, width: w, height: hh, rx: cs * 0.08, fill: this.col.cell, stroke: this.col.edge, "stroke-width": 1 }, dg);
      for (const sx of [-1, 1]) el("line", { x1: sx * (w / 2 - cs * 0.035), y1: -hh / 2 + cs * 0.04, x2: sx * (w / 2 - cs * 0.035), y2: hh / 2 - cs * 0.04, stroke: this.col.ink3, "stroke-width": Math.max(1.5, cs * 0.05), "stroke-linecap": "round" }, dg);
    }
  }
  // keys & doors: a coloured key; its door looks locked (hatched tile + padlock) until the key is on the path
  drawKeys() {
    const P = this.P, cs = this.L.cs;
    this.keyEls = [];
    if (!P.keys.length) return;
    const gDoor = el("g", { class: "g-doors" });
    this.svg.insertBefore(gDoor, this.gPath);
    const defs = this.svg.querySelector("defs");
    const hid = `hatch${this._uid}`;
    const pat = el("pattern", { id: hid, width: cs * 0.18, height: cs * 0.18, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    el("rect", { width: cs * 0.07, height: cs * 0.18, fill: "#000", opacity: 0.12 }, pat);
    P.keys.forEach((kp, i) => {
      const color = KEY_COLORS[(kp.color ?? i) % KEY_COLORS.length];
      const [kx, ky] = this.center(kp.key), [dx, dy] = this.center(kp.door), s = cs * 0.3;
      // key
      const kg = el("g", { class: "key", transform: `translate(${kx} ${ky})` }, this.gCp);
      const ki = el("g", { class: "key-inner" }, kg);
      el("circle", { r: s, fill: color, stroke: "#fff", "stroke-width": Math.max(1, cs * 0.04) }, ki);
      const kk = el("g", { transform: "rotate(-45)", stroke: "#fff", "stroke-width": Math.max(1.2, s * 0.17), fill: "none", "stroke-linecap": "round" }, ki);
      el("circle", { cx: -s * 0.36, cy: 0, r: s * 0.24 }, kk);
      el("path", { d: `M${-s * 0.12},0 H${s * 0.55} M${s * 0.3},0 V${s * 0.22} M${s * 0.5},0 V${s * 0.18}` }, kk);
      // door: tinted hatched tile + padlock
      const pad = cs * 0.06;
      const tile = el("g", { class: "door" }, gDoor);
      const sd = this.shapeD(kp.door, pad * 1.4);
      const shape = sd ? { d: sd } : { x: dx - cs / 2 + pad, y: dy - cs / 2 + pad, width: cs - 2 * pad, height: cs - 2 * pad, rx: cs * 0.2 };
      el(sd ? "path" : "rect", { ...shape, fill: color, opacity: 0.28 }, tile);
      el(sd ? "path" : "rect", { ...shape, fill: `url(#${hid})` }, tile);
      el(sd ? "path" : "rect", { ...shape, fill: "none", stroke: color, "stroke-width": Math.max(1.5, cs * 0.05) }, tile);
      const lg = el("g", { class: "lock", transform: `translate(${dx} ${dy})` }, this.gCp);
      const li = el("g", { class: "lock-inner" }, lg);
      const bw = s * 1.2, bh = s * 0.95;
      el("path", { class: "shackle", d: `M${-bw * 0.28},${-bh * 0.1} V${-bh * 0.55} a${bw * 0.28},${bw * 0.28} 0 0 1 ${bw * 0.56},0 V${-bh * 0.1}`, fill: "none", stroke: color, "stroke-width": Math.max(1.5, s * 0.2), "stroke-linecap": "round" }, li);
      el("rect", { x: -bw / 2, y: -bh * 0.2, width: bw, height: bh, rx: bw * 0.18, fill: color, stroke: "#fff", "stroke-width": Math.max(1, cs * 0.03) }, li);
      el("circle", { cx: 0, cy: bh * 0.22, r: bw * 0.09, fill: "#fff" }, li);
      this.keyEls.push({ kp, kg, tile, lg, color });
    });
  }
  updateKeys(onPath, onPath2, animate) {
    const st = this._keyState || (this._keyState = this.keyEls.map(() => ({ got: false, pass: false })));
    this.keyEls.forEach((k, i) => {
      const got = onPath.has(k.kp.key) || onPath2.has(k.kp.key), pass = onPath.has(k.kp.door) || onPath2.has(k.kp.door);
      const was = st[i];
      k.kg.classList.toggle("got", got);
      k.tile.classList.toggle("open", got);
      k.lg.classList.toggle("open", got);
      k.lg.classList.toggle("passed", pass);
      if (got && !was.got && animate && !reducedMotion()) {
        const li = k.lg.firstChild;
        li.classList.remove("unlock"); void li.getBBox(); li.classList.add("unlock");
        const [x, y] = this.center(k.kp.door);
        const ring = el("circle", { cx: x, cy: y, r: this.L.cs * 0.34, fill: "none", stroke: k.color, "stroke-width": 3, class: "burst" }, this.gFx);
        setTimeout(() => ring.remove(), 700);
      }
      st[i] = { got, pass };
    });
  }
  shakeLock(door) {
    const k = (this.keyEls || []).find((q) => q.kp.door === door);
    if (!k) return;
    const li = k.lg.firstChild;
    li.classList.remove("jiggle"); void li.getBBox(); li.classList.add("jiggle");
    const ki = k.kg.firstChild;
    ki.classList.remove("beckon"); void ki.getBBox(); ki.classList.add("beckon");
  }
  nudgeArc(u, v) {
    const e = this.arcEls && this.arcEls.get(`${v}>${u}`);
    if (!e) return;
    e.classList.remove("nudge"); void e.getBBox(); e.classList.add("nudge");
  }
  // fog of war: numbers hide under little "?" clouds until a path head comes within 2 steps
  drawFog() {
    const P = this.P, cs = this.L.cs;
    // revealed numbers stay revealed across re-builds (resize / theme change) of the same puzzle
    const prev = this._fogMem && this._fogMem.P === P ? this._fogMem.set : null;
    this._fogShown = prev || new Set([P.cps[0]]);
    if (P.coop) this._fogShown.add(P.coop.cps[1][0]);
    this._fogMem = { P, set: this._fogShown };
    this.fogEls = new Map();
    const all = P.coop ? [...this.cpEls.map((g, k) => [g, P.cps[k]]), ...this.cpEls2.map((g, k) => [g, P.coop.cps[1][k]])] : this.cpEls.map((g, k) => [g, P.cps[k]]);
    for (const [g, v] of all) {
      if (this._fogShown.has(v)) continue;
      g.classList.add("fogged");
      const [x, y] = this.center(v), r = cs * 0.24;
      const cg = el("g", { class: "fog", transform: `translate(${x} ${y})` }, this.gCp);
      const ci = el("g", { class: "fog-inner", style: `animation-delay:${(-(v % 5) * 0.6).toFixed(1)}s` }, cg);
      for (const [cx, cy, rr] of [[-r * 0.95, r * 0.3, r * 0.85], [r * 0.95, r * 0.3, r * 0.85], [-r * 0.2, -r * 0.35, r * 1.1], [r * 0.55, -r * 0.1, r * 0.8], [0, r * 0.45, r * 0.95]])
        el("circle", { cx, cy, r: rr, class: "fog-puff" }, ci);
      el("text", { y: r * 0.55, "font-size": r * 1.5, "text-anchor": "middle", "font-weight": 800, class: "fog-q" }, ci).textContent = "?";
      this.fogEls.set(v, { cg, g });
    }
  }
  updateFog(heads, animate) {
    if (!this.fogEls) return;
    const vis = new Set();
    for (const hd of heads) for (const v of near(this.P, hd, 2)) vis.add(v);
    for (const [v, f] of this.fogEls) {
      if (!vis.has(v)) continue;
      this._fogShown.add(v);
      this.fogEls.delete(v);
      f.g.classList.remove("fogged");
      if (animate && !reducedMotion()) {
        f.cg.classList.add("poof");
        setTimeout(() => f.cg.remove(), 650);
        const inner = f.g.firstChild;
        inner.classList.remove("pop"); void inner.getBBox(); inner.classList.add("pop");
      } else f.cg.remove();
    }
  }
  // portal hop: rings burst at both ends, in the path's colour
  teleport(u, v, color) {
    if (reducedMotion()) return;
    const cs = this.L.cs;
    for (const [w, d] of [[u, 0], [v, 90]]) {
      const [x, y] = this.center(w);
      const g = el("g", { transform: `translate(${x} ${y})` }, this.gFx);
      for (const [r, dd] of [[0.3, 0], [0.45, 80]]) el("circle", { r: cs * r, fill: "none", stroke: color, "stroke-width": 3, class: "burst", style: `animation-delay:${d + dd}ms` }, g);
      for (let i = 0; i < 6; i++) {
        const a = (i / 6) * Math.PI * 2;
        el("circle", { r: cs * 0.05, fill: color, class: "tp-spark", style: `--dx:${(Math.cos(a) * cs * 0.5).toFixed(1)}px;--dy:${(Math.sin(a) * cs * 0.5).toFixed(1)}px;animation-delay:${d}ms` }, g);
      }
      setTimeout(() => g.remove(), 900);
    }
    const pv = this.portalEls && this.portalEls.get(v);
    if (pv) { pv.classList.remove("zap"); void pv.getBBox(); pv.classList.add("zap"); }
  }

  // ---------------------------------------------------------------- islands: sea, beaches, plank bridges
  drawSea(g, defs) {
    const L = this.L, P = this.P, cs = L.cs, mini = !!this.opts.mini, uid = this._uid, p = L.panels[0];
    const C = (n) => cssVar(n);
    const W = L.W, H = L.H, rr = Math.min(cs * 0.6, 22);
    const wg = el("linearGradient", { id: `wg${uid}`, x1: 0, y1: 0, x2: 0.35, y2: 1 }, defs);
    el("stop", { offset: 0, "stop-color": C("--water-1") }, wg);
    el("stop", { offset: 1, "stop-color": C("--water-2") }, wg);
    const sea = el("g", { class: "sea" }, g);
    el("rect", { x: 1, y: 1, width: W - 2, height: H - 2, rx: rr, fill: `url(#wg${uid})` }, sea);
    if (!mini) {
      const cp = el("clipPath", { id: `wc${uid}` }, defs);
      el("rect", { x: 1, y: 1, width: W - 2, height: H - 2, rx: rr }, cp);
      const clip = el("g", { "clip-path": `url(#wc${uid})` }, sea);
      // caustics: soft light pools that breathe
      const rg = el("radialGradient", { id: `cg${uid}` }, defs);
      el("stop", { offset: 0, "stop-color": C("--foam"), "stop-opacity": 0.22 }, rg);
      el("stop", { offset: 1, "stop-color": C("--foam"), "stop-opacity": 0 }, rg);
      let seed = (P.n * 9301 + P.cps.length * 49297) % 233280;
      const rnd = () => (seed = (seed * 9301 + 49297) % 233280) / 233280;
      const nC = Math.round(clamp((W * H) / (cs * cs * 5), 4, 14));
      for (let i = 0; i < nC; i++) {
        el("circle", { cx: rnd() * W, cy: rnd() * H, r: cs * (0.7 + rnd() * 0.9), fill: `url(#cg${uid})`, class: "caustic", style: `animation-delay:${(-rnd() * 6).toFixed(2)}s` }, clip);
      }
      // drifting wave lines (two layers, seamless loop of one wavelength)
      const wl = cs * 1.7, amp = cs * 0.06;
      for (const [layer, y0, dy] of [["a", cs * 0.35, cs * 1.3], ["b", cs * 1.0, cs * 1.3]]) {
        const gm = el("g", { class: `waves w-${layer}`, style: `--wl:${wl.toFixed(1)}px` }, clip);
        for (let y = y0; y < H + cs; y += dy) {
          let d = `M${-wl * 2},${y.toFixed(1)}`;
          for (let x = -wl * 2; x < W + wl * 2; x += wl) d += ` q${(wl / 4).toFixed(1)},${-amp} ${(wl / 2).toFixed(1)},0 t${(wl / 2).toFixed(1)},0`;
          el("path", { d, fill: "none", stroke: C("--foam"), "stroke-width": Math.max(1, cs * 0.025), opacity: 0.22, "stroke-linecap": "round", "stroke-dasharray": `${(wl * 0.55).toFixed(1)} ${(wl * 0.95).toFixed(1)}` }, gm);
        }
      }
    }
    // glow filter for the path where it crosses a bridge
    const f = el("filter", { id: `bg${uid}`, filterUnits: "userSpaceOnUse", x: -cs, y: -cs, width: W + 2 * cs, height: H + 2 * cs }, defs);
    el("feGaussianBlur", { in: "SourceGraphic", stdDeviation: Math.max(1.5, cs * 0.08), result: "b" }, f);
    const fm = el("feMerge", {}, f);
    el("feMergeNode", { in: "b" }, fm); el("feMergeNode", { in: "SourceGraphic" }, fm);
    this.bglowId = `bg${uid}`;
    // islands: the union outline of each island's cells, offset + rounded (handles notches / concave shapes)
    const groups = new Map();
    for (let v = 0; v < P.n; v++) { const k = P.island[v]; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(L.cell[v]); }
    const X = (c) => p.x + c * cs, Y = (r) => p.cy + r * cs;
    const nLand = 6;
    let deco = 0;
    for (const [k, cells] of groups) {
      const loops = outlineLoops(cells);
      const shape = (off, rad) => loops.map((lp) => roundedPath(offsetLoop(lp, off).map(([r, c]) => [X(c), Y(r)]), rad * cs)).join(" ");
      const sand = shape(0.2, 0.3), land = shape(0.03, 0.26);
      if (!mini) el("path", { d: sand, fill: C("--water-deep"), opacity: 0.28, transform: `translate(0 ${(cs * 0.08).toFixed(1)})` }, g);
      if (!mini) el("path", { d: shape(0.3, 0.5), fill: "none", stroke: C("--foam"), "stroke-width": Math.max(1.2, cs * 0.05), opacity: 0.5, class: "foam" }, g);
      el("path", { d: sand, fill: C("--sand"), stroke: C("--sand-2"), "stroke-width": mini ? 0.6 : Math.max(1, cs * 0.02) }, g);
      el("path", { d: land, fill: C(`--land-${k % nLand}`) }, g);
      if (mini) continue;
      // tiny scenery at a couple of convex corners (on the beach, clear of cells and numbers)
      const corners = [];
      for (const lp of loops) {
        const m = lp.length;
        for (let i = 0; i < m; i++) {
          const a = lp[(i + m - 1) % m], b = lp[i], c = lp[(i + 1) % m];
          const d1 = [b[0] - a[0], b[1] - a[1]], d2 = [c[0] - b[0], c[1] - b[1]];
          if (d1[1] * d2[0] - d1[0] * d2[1] <= 0) continue;   // not convex (screen coords)
          const n1 = normL(d1), n2 = normL(d2);
          corners.push([b[0] + 0.04 * (n1[0] + n2[0]), b[1] + 0.04 * (n1[1] + n2[1]), n1[0] + n2[0], n1[1] + n2[1]]);
        }
      }
      const pick = corners.filter((_, i) => ((i * 7 + k * 5 + cells.length) % 5) === 0).slice(0, 2);
      for (const [r, c, dr, dc] of pick) this.scenery(g, X(c), Y(r), dr, dc, deco++ % 3);
    }
  }
  scenery(g, x, y, dr, dc, kind) {
    const cs = this.L.cs, s = cs * 0.34;
    const gg = el("g", { class: `scenery ${kind === 1 ? "rock" : "palm"}`, transform: `translate(${x.toFixed(1)} ${y.toFixed(1)})` }, g);
    if (kind === 1) {
      el("ellipse", { cx: 0, cy: s * 0.05, rx: s * 0.42, ry: s * 0.26, fill: cssVar("--rock") }, gg);
      el("ellipse", { cx: s * 0.28, cy: s * 0.15, rx: s * 0.24, ry: s * 0.16, fill: cssVar("--rock-2") }, gg);
      return;
    }
    const lean = dc >= 0 ? 1 : -1;
    const sway = el("g", { class: "sway" }, gg);
    el("path", { d: `M0,${s * 0.45} Q${lean * s * 0.05},${-s * 0.05} ${lean * s * 0.18},${-s * 0.42}`, fill: "none", stroke: cssVar("--wood-2"), "stroke-width": s * 0.14, "stroke-linecap": "round" }, sway);
    const tx = lean * s * 0.18, ty = -s * 0.42, leaf = cssVar("--leaf");
    for (const a of [-150, -105, -60, -15, 25]) {
      const rad = (a * Math.PI) / 180, lx = tx + Math.cos(rad) * s * 0.55, ly = ty + Math.sin(rad) * s * 0.55 + s * 0.12;
      el("path", { d: `M${tx},${ty} Q${(tx + lx) / 2},${(ty + ly) / 2 - s * 0.2} ${lx},${ly}`, fill: "none", stroke: leaf, "stroke-width": s * 0.16, "stroke-linecap": "round" }, sway);
    }
    el("circle", { cx: tx, cy: ty + s * 0.06, r: s * 0.07, fill: cssVar("--wood-2") }, sway);
    void dr;
  }
  bridgePoints(u, v, trim) {
    const { x1, y1, mx, my, x2, y2 } = this.arcGeom(u, v);
    const N = 32, pts = [];
    for (let i = 0; i <= N; i++) {
      const t = i / N, a = (1 - t) * (1 - t), b = 2 * (1 - t) * t, c = t * t;
      pts.push([a * x1 + b * mx + c * x2, a * y1 + b * my + c * y2]);
    }
    const len = [0];
    for (let i = 1; i < pts.length; i++) len.push(len[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const T = len[len.length - 1];
    const at = (d) => {   // point + unit tangent at arc length d
      let i = 1;
      while (i < len.length - 1 && len[i] < d) i++;
      const f = (d - len[i - 1]) / Math.max(1e-9, len[i] - len[i - 1]);
      const [ax, ay] = pts[i - 1], [bx, by] = pts[i];
      const tl = Math.hypot(bx - ax, by - ay) || 1;
      return [ax + (bx - ax) * f, ay + (by - ay) * f, (bx - ax) / tl, (by - ay) / tl];
    };
    return { at, from: Math.min(trim, T / 2), to: Math.max(T - trim, T / 2), T };
  }
  drawBridge(u, v, g) {
    const cs = this.L.cs, mini = !!this.opts.mini;
    const { at, from, to } = this.bridgePoints(u, v, cs * 0.34);
    const wood = cssVar("--wood"), wood2 = cssVar("--wood-2"), rope = cssVar("--rope");
    const bg = el("g", { class: "plank-bridge" }, g);
    const line = (off) => {
      let d = "";
      for (let s = from, i = 0; s <= to + 1e-6; s += (to - from) / 16, i++) {
        const [x, y, tx, ty] = at(s);
        d += `${i ? "L" : "M"}${(x - ty * off).toFixed(1)},${(y + tx * off).toFixed(1)}`;
      }
      return d;
    };
    const hw = cs * 0.23;
    if (mini) { el("path", { d: line(0), stroke: wood, "stroke-width": Math.max(2, cs * 0.4), fill: "none" }, bg); return; }
    el("path", { d: line(0), stroke: cssVar("--water-deep"), "stroke-opacity": 0.3, "stroke-width": hw * 2.3, fill: "none", transform: `translate(0 ${(cs * 0.07).toFixed(1)})` }, bg);
    const step = cs * 0.13;
    const n = Math.max(2, Math.round((to - from) / step));
    for (let i = 0; i <= n; i++) {
      const [x, y, tx, ty] = at(from + ((to - from) * i) / n);
      const ang = (Math.atan2(ty, tx) * 180) / Math.PI;
      el("rect", { x: -cs * 0.04, y: -hw, width: cs * 0.085, height: hw * 2, rx: cs * 0.02, fill: i % 3 === 1 ? wood2 : wood,
        transform: `translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${ang.toFixed(1)})` }, bg);
    }
    for (const off of [-hw * 0.92, hw * 0.92]) el("path", { d: line(off), stroke: rope, "stroke-width": Math.max(1.2, cs * 0.035), fill: "none", "stroke-linecap": "round" }, bg);
    for (const s of [from, to]) {
      const [x, y, tx, ty] = at(s);
      for (const off of [-hw, hw]) el("circle", { cx: x - ty * off, cy: y + tx * off, r: Math.max(1.5, cs * 0.05), fill: wood2 }, bg);
    }
  }
  splash(u, v) {
    const cs = this.L.cs;
    const { at, T } = this.bridgePoints(u, v, 0);
    const [x, y] = at(T / 2);
    const foam = cssVar("--foam");
    const g = el("g", { class: "splash-g", transform: `translate(${x.toFixed(1)} ${(y + cs * 0.1).toFixed(1)})` }, this.gFx);
    for (const [r, d] of [[0.35, 0], [0.55, 120]]) el("ellipse", { rx: cs * r, ry: cs * r * 0.45, fill: "none", stroke: foam, "stroke-width": 2, class: "splash", style: `animation-delay:${d}ms` }, g);
    for (let i = 0; i < 5; i++) el("circle", { r: cs * 0.04, fill: foam, class: "drop", style: `--dx:${((i - 2) * cs * 0.13).toFixed(1)}px;--dy:${(-cs * (0.3 + 0.12 * (i % 2))).toFixed(1)}px;animation-delay:${i * 25}ms` }, g);
    setTimeout(() => g.remove(), 900);
  }

  badge(a, b, idx, out) {
    const cs = this.L.cs, [x, y] = this.center(a);
    const text = this.crossLabel(a, b), fs = clamp(cs * 0.2, 8, 13);
    const w = fs * (text.length * 0.62 + 0.9), hh = fs * 1.35;
    const bx = out ? x + cs / 2 - w - 3 : x - cs / 2 + 3, by = out ? y - cs / 2 + 3 : y + cs / 2 - hh - 3;
    const c = this.colorAt(idx);
    el("rect", { x: bx, y: by, width: w, height: hh, rx: hh / 2, fill: out ? c : this.col.panel, stroke: c, "stroke-width": 1.4 }, this.gMarks);
    el("text", { x: bx + w / 2, y: by + hh / 2 + fs * 0.36, "font-size": fs, "text-anchor": "middle", "font-weight": 700, fill: out ? "#fff" : c }, this.gMarks).textContent = text;
  }

  popCp(ci, which = 0) {
    const g = (which ? this.cpEls2 : this.cpEls)[ci];
    const inner = g.firstChild;
    inner.classList.remove("pop"); void inner.getBBox(); inner.classList.add("pop");
    const [x, y] = this.center(which ? this.P.coop.cps[1][ci] : this.P.cps[ci]);
    const ring = el("circle", { cx: x, cy: y, r: this.L.cs * 0.32, fill: "none", stroke: this.col.accent, "stroke-width": 3, class: "burst" }, this.gFx);
    setTimeout(() => ring.remove(), 700);
  }

  drawHeat(heat) {
    const g = this.gHeat;
    g.textContent = "";
    if (!heat) return;
    const cs = this.L.cs, pad = cs * 0.08;
    for (const [key, val] of Object.entries(heat)) {
      const v = Number(key);
      if (!(val > 0.02) || v >= this.P.n) continue;
      const [x, y] = this.center(v);
      el("rect", { x: x - cs / 2 + pad, y: y - cs / 2 + pad, width: cs - 2 * pad, height: cs - 2 * pad, rx: cs * 0.2,
        fill: this.col.heat, "fill-opacity": (0.08 + 0.55 * val).toFixed(3), class: "heat" }, g);
    }
  }

  drawTried(tried) {
    const n = this.P.n, cs = this.L.cs;
    if (!this.triedEls) {
      this.triedEls = new Array(n).fill(null);
    }
    const pad = cs * 0.12;
    for (let v = 0; v < n; v++) {
      const t = tried[v];
      let e = this.triedEls[v];
      if (t > 0.03) {
        if (!e) {
          const [x, y] = this.center(v);
          e = this.triedEls[v] = el("rect", { x: x - cs / 2 + pad, y: y - cs / 2 + pad, width: cs - 2 * pad, height: cs - 2 * pad, rx: cs * 0.22, fill: this.col.bad, class: "tried" }, this.gTried);
        }
        e.setAttribute("fill-opacity", (0.55 * t).toFixed(3));
      } else if (e) { e.remove(); this.triedEls[v] = null; }
    }
  }

  flash(v) {
    const r = this.cellEls && this.cellEls[v];
    if (!r) return;
    r.classList.remove("shake"); void r.getBBox(); r.classList.add("shake");
    r.setAttribute("stroke", this.col.bad);
    setTimeout(() => r.setAttribute("stroke", this.col.edge), 380);
  }

  // ---------------------------------------------------------------- pointer input
  attachPointer() {
    const svg = this.svg, o = this.opts;
    let drag = null;
    svg.addEventListener("pointerdown", (ev) => {
      if (ev.button !== undefined && ev.button > 0) return;
      const [x, y] = this.toBoard(ev.clientX, ev.clientY);
      const v = this.cellAt(x, y);
      if (v == null) return;
      ev.preventDefault();
      drag = { last: [x, y], cell: v, id: ev.pointerId };
      try { svg.setPointerCapture(ev.pointerId); } catch { /* ignore */ }
      o.onDown && o.onDown(v);
    });
    svg.addEventListener("pointermove", (ev) => {
      if (!drag || ev.pointerId !== drag.id) return;
      const evs = ev.getCoalescedEvents ? ev.getCoalescedEvents() : [ev];
      for (const e of (evs.length ? evs : [ev])) {
        const [x, y] = this.toBoard(e.clientX, e.clientY);
        // sample the segment since the last event so fast drags never skip a cell
        const [x0, y0] = drag.last, d = Math.hypot(x - x0, y - y0);
        const steps = Math.max(1, Math.ceil(d / (this.L.cs * 0.2)));
        for (let s = 1; s <= steps; s++) {
          const px = x0 + ((x - x0) * s) / steps, py = y0 + ((y - y0) * s) / steps;
          const v = this.cellAt(px, py, 0.78);   // dead zone near cell borders: no accidental corner cuts
          if (v == null || v === drag.cell) continue;
          drag.cell = v;
          o.onDrag && o.onDrag(v);
        }
        drag.last = [x, y];
      }
    });
    const end = () => { if (drag) { drag = null; o.onUp && o.onUp(); } };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.addEventListener("lostpointercapture", end);
  }
}

// ------------------------------------------------------------------ island outline geometry
// Boundary loops of a set of grid cells ([r, c]), as lattice vertices [r, c], traversed with the island on the
// right-hand side in screen coordinates; collinear vertices are dropped. Works for notches / concave shapes / holes.
export function outlineLoops(cells) {
  const has = new Set(cells.map(([r, c]) => `${r},${c}`));
  const out = new Map();   // "r,c" -> [[r2, c2], ...]
  const add = (a, b) => { const k = `${a[0]},${a[1]}`; if (!out.has(k)) out.set(k, []); out.get(k).push(b); };
  for (const [r, c] of cells) {
    if (!has.has(`${r - 1},${c}`)) add([r, c], [r, c + 1]);
    if (!has.has(`${r},${c + 1}`)) add([r, c + 1], [r + 1, c + 1]);
    if (!has.has(`${r + 1},${c}`)) add([r + 1, c + 1], [r + 1, c]);
    if (!has.has(`${r},${c - 1}`)) add([r + 1, c], [r, c]);
  }
  const loops = [];
  for (;;) {
    let startK = null;
    for (const [k, l] of out) if (l.length) { startK = k; break; }
    if (startK === null) break;
    const start = startK.split(",").map(Number);
    const loop = [start];
    let cur = start, dir = null, guard = 0;
    for (;;) {
      const l = out.get(`${cur[0]},${cur[1]}`);
      if (!l || !l.length || guard++ > 100000) break;
      let j = 0;
      if (l.length > 1 && dir) {   // pinch point: prefer the right turn (keeps diagonal neighbours apart)
        const right = [dir[1], -dir[0]];
        const jj = l.findIndex((b) => b[0] - cur[0] === right[0] && b[1] - cur[1] === right[1]);
        if (jj >= 0) j = jj;
      }
      const nx = l.splice(j, 1)[0];
      dir = [nx[0] - cur[0], nx[1] - cur[1]];
      cur = nx;
      if (cur[0] === start[0] && cur[1] === start[1]) break;
      loop.push(cur);
    }
    // drop collinear vertices
    const m = loop.length, clean = [];
    for (let i = 0; i < m; i++) {
      const a = loop[(i + m - 1) % m], b = loop[i], c = loop[(i + 1) % m];
      if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) !== 0) clean.push(b);
    }
    if (clean.length >= 3) loops.push(clean);
  }
  return loops;
}
// outward (left-hand) unit normal of a lattice direction [dr, dc]
function normL([dr, dc]) { const l = Math.hypot(dr, dc) || 1; return [-dc / l, dr / l]; }
export function offsetLoop(loop, d) {
  const m = loop.length;
  return loop.map((b, i) => {
    const a = loop[(i + m - 1) % m], c = loop[(i + 1) % m];
    const n1 = normL([b[0] - a[0], b[1] - a[1]]), n2 = normL([c[0] - b[0], c[1] - b[1]]);
    return [b[0] + d * (n1[0] + n2[0]), b[1] + d * (n1[1] + n2[1])];
  });
}
// closed path through points [x, y] with every corner rounded (quadratic, radius <= half the shorter side)
export function roundedPath(pts, rad) {
  const m = pts.length;
  let d = "";
  for (let i = 0; i < m; i++) {
    const a = pts[(i + m - 1) % m], b = pts[i], c = pts[(i + 1) % m];
    const l1 = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, l2 = Math.hypot(c[0] - b[0], c[1] - b[1]) || 1;
    const r = Math.min(rad, l1 / 2, l2 / 2);
    const s = [b[0] + ((a[0] - b[0]) * r) / l1, b[1] + ((a[1] - b[1]) * r) / l1];
    const e = [b[0] + ((c[0] - b[0]) * r) / l2, b[1] + ((c[1] - b[1]) * r) / l2];
    d += `${i ? "L" : "M"}${s[0].toFixed(1)},${s[1].toFixed(1)} Q${b[0].toFixed(1)},${b[1].toFixed(1)} ${e[0].toFixed(1)},${e[1].toFixed(1)} `;
  }
  return d + "Z";
}
