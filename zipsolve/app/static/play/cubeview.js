// Cube-surface puzzles (kind "cubesurf"): a rotatable cube whose six faces are grids. Same interface as Board
// (board.js) for the game page: build / render / clientOf / colorAt / flash / pointer input. Drag a face to draw,
// drag the background to spin; the cube turns by itself to follow the line's head when it walks around an edge.
// A tiny immediate-mode 3D renderer on SVG (orthographic + a touch of perspective): no WebGL needed.
import { svgEl as el, cssVar, pathColor, reducedMotion } from "./util.js";

const TAU = Math.PI * 2;

export class CubeBoard {
  constructor(host, opts = {}) {
    this.host = host; this.opts = opts;
    this.P = null; this.svg = null;
    this.yaw = -0.62; this.pitch = 0.5; this.tYaw = null; this.tPitch = null;
    this.segs = []; this.tints = []; this.cpEls = []; this.cpEls2 = [];
    this.V = { path: [] };
    this.L = { cs: 40 };
  }
  setPuzzle(P) {
    this.P = P;
    const N = P.cube.n;
    this.N = N;
    // centred 3D points; each node's face normal = the axis sitting on the cube's skin
    this.X = P.coords.map((p) => p.map((x) => x - N));
    this.normal = this.X.map((x) => { const a = [0, 1, 2].find((k) => Math.abs(x[k]) === N); const n = [0, 0, 0]; n[a] = Math.sign(x[a]); return n; });
    this.axis = this.normal.map((n) => n.findIndex((x) => x !== 0));
    this.svg = null;
  }
  // board-compatible helpers used by PathGame
  samePanel() { return true; }
  gridAdjacent(u, v) { return this.P.adjSet[u].has(v); }
  edgeKind() { return null; }

  // ---------------------------------------------------------------- 3D
  rot([x, y, z]) {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
    return [x1, y * cp - z1 * sp, y * sp + z1 * cp];
  }
  proj(p) {
    const [x, y, z] = this.rot(p), D = this.N * 5, f = D / (D - z);
    return [this.cx + this.s * f * x, this.cy - this.s * f * y, z];
  }
  faceZ(v) { return this.rot(this.normal[v])[2]; }
  corners(v, k = 0.92) {
    const x = this.X[v], a = this.axis[v], o = [0, 1, 2].filter((k) => k !== a), out = [];
    for (const [d1, d2] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) { const p = x.slice(); p[o[0]] += d1 * k; p[o[1]] += d2 * k; out.push(p); }
    return out;
  }
  // route of one path step over the cube's skin: straight on a face, or via the shared edge
  stepPts(u, v) {
    const a = this.X[u], b = this.X[v];
    if (this.axis[u] === this.axis[v] && this.normal[u].every((x, k) => x === this.normal[v][k])) return [[a, b, u]];
    const e = a.slice(); const j = this.axis[v]; e[j] = b[j];
    return [[a, e, u], [e, b, v]];
  }

  // ---------------------------------------------------------------- build / render
  build(avW, avH) {
    const size = Math.max(160, Math.min(avW, avH));
    this.host.innerHTML = "";
    this.W = size; this.H = size;
    this.cx = size / 2; this.cy = size / 2;
    this.s = size / (3.7 * this.N);
    this.L.cs = this.s * 2;
    this.col = { cell: cssVar("--cell"), edge: cssVar("--cell-edge"), cp: cssVar("--cp"), cpInk: cssVar("--cp-ink"), accent: cssVar("--accent"),
      panel: cssVar("--panel"), ink: cssVar("--ink"), bad: cssVar("--bad"), good: cssVar("--good"), legal: cssVar("--legal") };
    const svg = this.svg = el("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size, class: "zboard zcube", role: "application", tabindex: 0,
      "aria-label": this.opts.label || "Cube puzzle: drag on the faces to draw, drag outside to spin" }, this.host);
    const defs = el("defs", {}, svg);
    const g = el("radialGradient", { id: "cubeShadow" }, defs);
    el("stop", { offset: 0, "stop-color": "#000", "stop-opacity": 0.28 }, g);
    el("stop", { offset: 1, "stop-color": "#000", "stop-opacity": 0 }, g);
    el("ellipse", { cx: this.cx, cy: size * 0.93, rx: size * 0.3, ry: size * 0.045, fill: "url(#cubeShadow)" }, svg);
    this.g = el("g", {}, svg);
    this.gFx = el("g", { class: "g-fx" }, svg);
    if (this.opts.interactive) this.attachPointer();
    this.draw();
    return svg;
  }
  colorAt(i, which = 0) { void which; return pathColor(this.P.n > 1 ? i / (this.P.n - 1) : 0); }
  render(V) {
    const prevHead = this._lastHead;   // (V.path is the game's live array: remember the head ourselves)
    this.V = V;
    const path = V.path || [], head = path[path.length - 1];
    // never turn the cube under a finger that is drawing: follow the head once the drag ends
    this._lastHead = head;
    if (head !== undefined && head !== prevHead && !V.won) { if (this._drawing) this._pending = head; else this.follow(head); }
    if (V.won && !this._spun) { this._spun = true; this.spin(); }
    if (!V.won) this._spun = false;
    this.draw();
  }
  // turn so that the head's face is comfortably in view (only when it has wandered to a side / the back)
  follow(v, force = false) {
    // turn when the head, or a free cell next to it, is (nearly) out of sight; judge where the camera is heading
    const seen = new Set(this.V.path || []);
    const fine = () => this.faceZ(v) > 0.45 && this.P.adj[v].every((w) => seen.has(w) || this.faceZ(w) > 0.2);
    if (!force) {
      let ok;
      if (this._to) { const y0 = this.yaw, p0 = this.pitch; [this.yaw, this.pitch] = this._to; ok = fine(); this.yaw = y0; this.pitch = p0; }
      else ok = fine();
      if (ok) return;
    }
    // look at the head's face and at every face a free neighbour sits on (so the way on is visible)
    const d = this.normal[v].slice(), faces = new Set([this.normal[v].join()]);
    for (const w of this.P.adj[v]) {
      const k = this.normal[w].join();
      if (seen.has(w) || faces.has(k)) continue;
      faces.add(k); for (let i = 0; i < 3; i++) d[i] += 0.9 * this.normal[w][i];
    }
    if (faces.size === 1) { const x = this.X[v].map((c) => c / this.N); for (let i = 0; i < 3; i++) d[i] += 0.5 * x[i]; }
    const r = Math.hypot(d[0], d[2]);
    const base = [Math.atan2(-d[0], d[2]), Math.atan2(d[1], r)];
    const y0 = this.yaw, p0 = this.pitch;
    let yaw = base[0], pitch = base[1];
    for (const [dy, dp] of [[-0.42, 0.34], [-0.2, 0.18], [0, 0]]) {   // a 3/4 view when it still shows everything
      this.yaw = base[0] + dy; this.pitch = Math.max(-1.2, Math.min(1.2, base[1] + dp));
      if (fine() || (dy === 0)) { yaw = this.yaw; pitch = this.pitch; break; }
    }
    this.yaw = y0; this.pitch = p0;
    while (yaw - this.yaw > Math.PI) yaw -= TAU;
    while (yaw - this.yaw < -Math.PI) yaw += TAU;
    this.animateTo(yaw, pitch);
  }
  animateTo(yaw, pitch, ms = 520) {
    if (reducedMotion()) { this.yaw = yaw; this.pitch = pitch; this.draw(); return; }
    const y0 = this.yaw, p0 = this.pitch, t0 = performance.now(), tok = (this._anim = (this._anim || 0) + 1);
    this._to = [yaw, pitch];
    const step = (t) => {
      if (tok !== this._anim) return;
      const f = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - f, 3);
      this.yaw = y0 + (yaw - y0) * e; this.pitch = p0 + (pitch - p0) * e;
      this.draw();
      if (f < 1) requestAnimationFrame(step); else this._to = null;
    };
    requestAnimationFrame(step);
  }
  spin() { this.animateTo(this.yaw + TAU, this.pitch, 1400); }
  draw() {
    if (!this.svg) return;
    const P = this.P, V = this.V, g = this.g, col = this.col;
    g.textContent = "";
    const path = V.path || [], onPath = new Map(path.map((v, i) => [v, i]));
    const legal = new Set(V.showLegal ? V.legal || [] : []);
    // faces back-to-front: cells of visible faces only
    const vis = [];
    for (let v = 0; v < P.n; v++) { const fz = this.faceZ(v); if (fz > 0.02) vis.push([v, fz]); }
    const cellG = el("g", {}, g);
    const faces = new Map();
    for (const [v] of vis) { const k = this.normal[v].join(","); if (!faces.has(k)) faces.set(k, v); }
    for (const [k, v] of faces) {
      const n = k.split(",").map(Number), a = n.findIndex((x) => x !== 0), o = [0, 1, 2].filter((i) => i !== a), N = this.N;
      const pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([s1, s2]) => { const p = [0, 0, 0]; p[a] = n[a] * N; p[o[0]] = s1 * N; p[o[1]] = s2 * N; return this.proj(p); });
      el("path", { d: pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("") + "Z", fill: col.edge, stroke: col.edge, "stroke-width": 2, "stroke-linejoin": "round" }, cellG);
      void v;
    }
    this.hit = [];
    for (const [v, fz] of vis) {
      const pts = this.corners(v).map((p) => this.proj(p));
      const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("") + "Z";
      el("path", { d, fill: col.cell, stroke: col.edge, "stroke-width": 1, "stroke-linejoin": "round", class: "cell" }, cellG);
      // simple light from the upper left: faces turned away get darker (reads as a solid box)
      const lit = this.rot(this.normal[v]), sh = Math.max(0, 0.34 - 0.3 * (0.2 * -lit[0] + 0.55 * lit[1] + 0.8 * lit[2]));
      if (sh > 0.01) el("path", { d, fill: "#000", opacity: sh.toFixed(3) }, cellG);
      if (onPath.has(v)) el("path", { d, fill: this.colorAt(onPath.get(v)), opacity: 0.22 }, cellG);
      this.hit.push([v, this.corners(v, 1).map((p) => this.proj(p))]);
    }
    // path
    const pg = el("g", { class: "g-path" }, g);
    const sw = this.s * 0.62;
    for (let i = 0; i + 1 < path.length; i++) {
      for (const [a, b, owner] of this.stepPts(path[i], path[i + 1])) {
        if (this.faceZ(owner) <= 0.02) continue;
        const [x1, y1] = this.proj(a), [x2, y2] = this.proj(b);
        el("line", { x1, y1, x2, y2, stroke: this.colorAt(i + 0.5), "stroke-width": sw, "stroke-linecap": "round" }, pg);
      }
    }
    // checkpoints, legal dots, hint, head
    const mk = el("g", {}, g);
    for (const [v, fz] of vis) {
      const [x, y] = this.proj(this.X[v]);
      const k = P.cpIndex.get(v);
      if (legal.has(v)) el("circle", { cx: x, cy: y, r: this.s * 0.16, fill: col.legal, opacity: 0.8 }, mk);
      if (k === undefined) continue;
      const r = this.s * 0.55 * (0.6 + 0.4 * fz);
      const cg = el("g", { class: `cp${onPath.has(v) ? " reached" : ""}`, transform: `translate(${x.toFixed(1)} ${y.toFixed(1)})` }, mk);
      el("circle", { r, fill: col.cp, stroke: "#fff", "stroke-width": 1.5 }, cg);
      el("text", { y: r * 0.37, "font-size": r * 1.05, "text-anchor": "middle", "font-weight": 800, fill: col.cpInk }, cg).textContent = k + 1;
    }
    if (V.hint && this.faceZ(V.hint.node) > 0.02) {
      const [x, y] = this.proj(this.X[V.hint.node]);
      el("circle", { cx: x, cy: y, r: this.s * 0.8, fill: "none", stroke: col.good, "stroke-width": 3.5, class: "pulse" }, mk);
    }
    const head = path[path.length - 1];
    if (path.length > 1 && !V.won && this.faceZ(head) > 0.02) {
      const [x, y] = this.proj(this.X[head]);
      const c = this.colorAt(path.length - 1);
      el("circle", { cx: x, cy: y, r: this.s * 0.75, fill: c, opacity: 0.25, class: "head-glow" }, mk);
      if (!P.cpIndex.has(head)) el("circle", { cx: x, cy: y, r: this.s * 0.34, fill: c, stroke: "#fff", "stroke-width": 2.5 }, mk);
    }
    this.svg.classList.toggle("won", !!V.won);
  }

  // ---------------------------------------------------------------- geometry for the page
  center(v) { const [x, y] = this.proj(this.X[v]); return [x, y]; }
  toBoard(cx, cy) { const r = this.svg.getBoundingClientRect(); return [((cx - r.left) * this.W) / r.width, ((cy - r.top) * this.H) / r.height]; }
  clientOf(v) {
    const [x, y] = this.center(v), r = this.svg.getBoundingClientRect();
    return { x: r.left + (x * r.width) / this.W, y: r.top + (y * r.height) / this.H };
  }
  cellAt(x, y) {
    for (const [v, pts] of this.hit || []) if (inPoly(x, y, pts)) return v;
    return null;
  }
  nodeAtClient(cx, cy) { const [x, y] = this.toBoard(cx, cy); return this.cellAt(x, y); }
  flash(v) {
    if (v == null || !this.svg || this.faceZ(v) <= 0.02) return;
    const [x, y] = this.proj(this.X[v]);
    const c = el("circle", { cx: x, cy: y, r: this.s * 0.8, fill: "none", stroke: this.col.bad, "stroke-width": 3, class: "burst" }, this.gFx);
    setTimeout(() => c.remove(), 650);
    this.svg.classList.remove("shake"); void this.svg.getBBox(); this.svg.classList.add("shake");
  }
  popCp() {}
  attachPointer() {
    const svg = this.svg, o = this.opts;
    let drag = null;
    svg.addEventListener("pointerdown", (ev) => {
      if (ev.button !== undefined && ev.button > 0) return;
      const [x, y] = this.toBoard(ev.clientX, ev.clientY);
      const v = this.cellAt(x, y);
      ev.preventDefault();
      try { svg.setPointerCapture(ev.pointerId); } catch { /* ignore */ }
      this._anim = (this._anim || 0) + 1; this._to = null;   // the player takes over the camera
      if (v == null) { drag = { spin: true, x: ev.clientX, y: ev.clientY, id: ev.pointerId }; svg.classList.add("spinning"); return; }
      drag = { cell: v, id: ev.pointerId };
      this._drawing = true;
      o.onDown && o.onDown(v);
    });
    svg.addEventListener("pointermove", (ev) => {
      if (!drag || ev.pointerId !== drag.id) return;
      if (drag.spin) {
        const k = 3.2 / this.W;
        this.yaw += (ev.clientX - drag.x) * k * 1.6;
        this.pitch = Math.max(-1.35, Math.min(1.35, this.pitch + (ev.clientY - drag.y) * k * 1.6));
        drag.x = ev.clientX; drag.y = ev.clientY;
        this.draw();
        return;
      }
      const [x, y] = this.toBoard(ev.clientX, ev.clientY);
      const v = this.cellAt(x, y);
      if (v == null || v === drag.cell) return;
      drag.cell = v;
      o.onDrag && o.onDrag(v);
    });
    const end = () => {
      if (!drag) return;
      svg.classList.remove("spinning"); drag = null; this._drawing = false;
      if (this._pending !== undefined && this._pending !== null) { const hd = this._pending; this._pending = null; this.follow(hd); }
      o.onUp && o.onUp();
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.addEventListener("lostpointercapture", end);
  }
}
function inPoly(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
