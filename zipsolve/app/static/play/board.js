// SVG board: layout (2D grid / 3D layer panels / 4D panel grid), incremental path rendering with
// micro-animations, overlays (legal moves, hints, AI heat + move probabilities, search trace), pointer input.
import { svgEl as el, cssVar, pathColor, AXIS, clamp } from "./util.js";

// ------------------------------------------------------------------ puzzle model
export function makeModel(d) {
  const coords = d.coords.map((c) => c.map((x) => Math.round(x)));
  const n = coords.length;
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of d.edges) { adj[u].push(v); adj[v].push(u); }
  return {
    data: d, n, dim: coords[0].length, coords, adj, adjSet: adj.map((a) => new Set(a)),
    cps: d.checkpoints, cpIndex: new Map(d.checkpoints.map((c, i) => [c, i])),
    island: d.meta && d.meta.island ? d.meta.island : null,
    bridges: !!(d.meta && d.meta.bridges),
  };
}

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
    const m = np > 1 ? 0.25 : this.isIslands() ? (mini ? 0.3 : 0.42) : 0.06;   // islands: room for beaches + sea
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
    const L = this.L, p = L.panels[L.panelOf[v]], [r, c] = L.cell[v];
    return [p.x + (c + 0.5) * L.cs, p.cy + (r + 0.5) * L.cs];
  }
  samePanel(u, v) { return this.L.panelOf[u] === this.L.panelOf[v]; }
  gridAdjacent(u, v) {
    if (!this.samePanel(u, v)) return false;
    const [a, b] = this.L.cell[u], [c, d] = this.L.cell[v];
    return Math.abs(a - c) + Math.abs(b - d) === 1;
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
    for (let v = 0; v < P.n; v++) {
      const [x, y] = this.center(v);
      const fill = isl ? tile : P.island ? cssVar(`--island-${P.island[v] % 8}`) : col.cell;
      this.cellEls.push(el("rect", { x: x - cs / 2 + pad, y: y - cs / 2 + pad, width: cs - 2 * pad, height: cs - 2 * pad, rx, fill,
        stroke: isl ? tileEdge : col.edge, "stroke-width": mini ? 0.6 : 1, class: "cell", "data-node": v }, gS));
    }
    const wallW = Math.max(mini ? 1.5 : 3, cs * 0.12);
    for (let v = 0; v < P.n; v++) {
      const pi = L.panelOf[v], [r, c] = L.cell[v];
      for (const [dr, dc] of [[0, 1], [1, 0]]) {
        const w = L.cellKey.get(`${pi}|${r + dr}|${c + dc}`);
        if (w === undefined || P.adjSet[v].has(w)) continue;
        const [x1, y1] = this.center(v), [x2, y2] = this.center(w);
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, hh = cs / 2 - 1;
        el("line", { x1: dc ? mx : mx - hh, y1: dc ? my - hh : my, x2: dc ? mx : mx + hh, y2: dc ? my + hh : my,
          stroke: col.wall, "stroke-width": wallW, "stroke-linecap": "round" }, gS);
      }
    }
    for (const [u, v] of P.data.edges) {
      if (!this.samePanel(u, v) || this.gridAdjacent(u, v)) continue;
      if (isl) { this.drawBridge(u, v, gS); continue; }
      el("path", { d: this.arcD(u, v), fill: "none", stroke: col.bridge, "stroke-width": Math.max(mini ? 1 : 2, cs * 0.06),
        "stroke-dasharray": `${cs * 0.18} ${cs * 0.14}`, "stroke-linecap": "round", opacity: 0.85, class: "bridge" }, gS);
    }
    this.gHeat = el("g", { class: "g-heat" }, svg);
    this.gTried = el("g", { class: "g-tried" }, svg);
    this.gTint = el("g", { class: "g-tint" }, svg);
    this.gPath = el("g", { class: "g-path" }, svg);
    // invisible anchor: keeps the group's bounding box non-degenerate, so filters on it (glow) never clip a straight line
    el("rect", { x: 0, y: 0, width: L.W, height: L.H, fill: "none", stroke: "none", class: "bbox-anchor" }, this.gPath);
    this.gHead = el("g", { class: "g-head" }, svg);
    this.gCp = el("g", { class: "g-cp" }, svg);
    this.gMarks = el("g", { class: "g-marks" }, svg);
    this.gFx = el("g", { class: "g-fx" }, svg);
    // per-node tints (visited fill), created once and toggled
    this.tints = [];
    const tp = pad + cs * 0.02;
    for (let v = 0; v < P.n; v++) {
      const [x, y] = this.center(v);
      this.tints.push(el("rect", { x: x - cs / 2 + tp, y: y - cs / 2 + tp, width: cs - 2 * tp, height: cs - 2 * tp, rx, class: "tint" }, this.gTint));
    }
    this.triedEls = null;
    // checkpoints
    this.cpEls = P.cps.map((v, k) => {
      const [x, y] = this.center(v);
      const g = el("g", { class: "cp", transform: `translate(${x} ${y})` }, this.gCp);
      const inner = el("g", { class: "cp-inner" }, g);
      el("circle", { r: cs * 0.3, fill: col.cp, stroke: col.cpRing, "stroke-width": mini ? 1 : 1.5 }, inner);
      if (!mini || cs >= 12) {
        const t = el("text", { y: cs * 0.11, "font-size": cs * (k + 1 >= 100 ? 0.22 : k + 1 >= 10 ? 0.27 : 0.32), "text-anchor": "middle", "font-weight": 800, fill: col.cpInk }, inner);
        t.textContent = k + 1;
      }
      return g;
    });
    // head
    this.headEl = el("g", { class: "head" }, this.gHead);
    this.headEl.style.color = pathColor(0);
    el("circle", { r: cs * 0.62, fill: `url(#glow${this._uid})`, class: "head-glow" }, this.headEl);
    this.headDot = el("circle", { r: cs * 0.19, fill: "currentColor", stroke: "#fff", "stroke-width": Math.max(mini ? 1 : 2, cs * 0.06), class: "head-dot" }, this.headEl);
    this.headEl.style.display = "none";
    this.segs = [];          // rendered segment elements, index i = path[i] -> path[i+1]
    this.rPath = [];         // path as rendered
    this._heatRef = undefined;
    if (this.opts.interactive) this.attachPointer();
    return svg;
  }

  // ---------------------------------------------------------------- dynamic layer
  colorAt(i) { return pathColor(this.P.n > 1 ? i / (this.P.n - 1) : 0); }

  makeSeg(i, u, v) {
    const cs = this.L.cs, n = this.P.n, sw = cs * 0.36;
    const col = pathColor(n > 1 ? (i + 0.5) / (n - 1) : 0);
    if (!this.samePanel(u, v)) return null;
    let e;
    if (this.gridAdjacent(u, v)) {
      const [x1, y1] = this.center(u), [x2, y2] = this.center(v);
      e = el("line", { x1, y1, x2, y2, stroke: col, "stroke-width": sw, "stroke-linecap": "round", pathLength: 1, class: "seg" }, this.gPath);
    } else {
      e = el("path", { d: this.arcD(u, v), fill: "none", stroke: col, "stroke-width": sw * 0.8, "stroke-linecap": "round", pathLength: 1, class: "seg bridge-seg" }, this.gPath);
      if (this.bglowId) e.setAttribute("filter", `url(#${this.bglowId})`);
    }
    return e;
  }

  render(V) {
    if (!this.svg) return;
    const P = this.P, L = this.L, cs = L.cs, col = this.col, mini = !!this.opts.mini;
    const path = V.path || [];
    const animate = V.animate !== false && !mini;
    // ---- incremental path segments
    let k = 0;
    const old = this.rPath;
    while (k < old.length && k < path.length && old[k] === path[k]) k++;
    // remove segments that start at index >= k-1 (they end at a changed node)
    const keepSegs = Math.max(0, k - 1);
    const removed = this.segs.splice(keepSegs);
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
      const s = this.makeSeg(i, path[i], path[i + 1]);
      if (s && animate && added <= 4 && i + 1 >= k) {
        s.classList.add("seg-in");
        if (s.classList.contains("bridge-seg") && this.isIslands()) this.splash(path[i], path[i + 1]);
      }
      this.segs.push(s);
    }
    // ---- tints (visited cells)
    const onPath = new Map();
    path.forEach((v, i) => onPath.set(v, i));
    for (let i = k; i < old.length; i++) { const t = this.tints[old[i]]; if (t && !onPath.has(old[i])) t.classList.remove("on", "pop"); }
    for (let i = Math.min(k, path.length); i < path.length; i++) {
      const t = this.tints[path[i]];
      t.setAttribute("fill", this.colorAt(i));
      t.classList.add("on");
      if (animate && path.length - k <= 4) { t.classList.remove("pop"); void t.getBBox(); t.classList.add("pop"); }
    }
    if (k === 0 && path.length) { const t = this.tints[path[0]]; t.setAttribute("fill", this.colorAt(0)); t.classList.add("on"); }
    // ---- checkpoints: reached state + pop when newly reached
    let next = 0;
    P.cps.forEach((v, ci) => {
      const idx = onPath.get(v);
      const g = this.cpEls[ci];
      const reached = idx !== undefined;
      if (reached) next = Math.max(next, ci + 1);
      g.classList.toggle("reached", reached);
      if (reached && idx >= old.length && animate && ci > 0) this.popCp(ci);
    });
    this.rPath = path.slice();
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
    if (!won && next < P.cps.length && path.length) {
      const [x, y] = this.center(P.cps[next]);
      el("circle", { cx: x, cy: y, r: cs * 0.38, fill: "none", stroke: col.accent, "stroke-width": mini ? 1.2 : 2, opacity: 0.6, class: "next-ring" }, gm);
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
    this.svg.classList.toggle("won", !!won);
    // ---- heat / tried layers
    if (V.heat !== this._heatRef) { this._heatRef = V.heat; this.drawHeat(V.heat); }
    if (V.tried) this.drawTried(V.tried);
    else if (this.triedEls) { this.gTried.textContent = ""; this.triedEls = null; }
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

  popCp(ci) {
    const g = this.cpEls[ci];
    const inner = g.firstChild;
    inner.classList.remove("pop"); void inner.getBBox(); inner.classList.add("pop");
    const [x, y] = this.center(this.P.cps[ci]);
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
