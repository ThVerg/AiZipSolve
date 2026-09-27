// Zip puzzle editor: draw cells, walls, islands + bridges, (3D) layers and numbered checkpoints;
// live validation (local + exact solver), make-unique / suggest numbers, save / load / play.
const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const MAX_NODES = 1500, MAX_SIDE = 30, MAX_Z = 6;
const STOPS = ["#ff9f1c", "#ff5a4e", "#d63a8f", "#7b3fd6", "#2f6fe0"].map((h) => { const x = parseInt(h.slice(1), 16); return [(x >> 16) & 255, (x >> 8) & 255, x & 255]; });
const DRAFT_KEY = "zip-editor-draft";

const TOOLS = [
  { id: "paint", key: "B", label: "Paint", hint: "Click or drag to add cells. Right-drag erases. Gaps between cells make islands.",
    icon: '<path d="M4 20c2.5 0 4-1.2 4-3.5a2.5 2.5 0 0 0-5 0"/><path d="m8.5 14 9.8-9.8a1.8 1.8 0 0 1 2.5 2.5L11 16.5"/>' },
  { id: "erase", key: "E", label: "Erase", hint: "Click or drag to remove cells (and their walls, bridges, numbers).",
    icon: '<path d="m7 21-4.3-4.3a1.6 1.6 0 0 1 0-2.3L13 4a1.6 1.6 0 0 1 2.3 0l5.6 5.6a1.6 1.6 0 0 1 0 2.3L12 21"/><path d="M7 21h14"/><path d="m5 12 7 7"/>' },
  { id: "rect", key: "R", label: "Rect", hint: "Drag a rectangle of cells. Hold Shift (or right-drag) to erase a rectangle.",
    icon: '<rect x="4" y="5" width="16" height="14" rx="2"/><path d="M4 10h16M10 5v14" opacity=".45"/>' },
  { id: "wall", key: "W", label: "Walls", hint: "Click near the border between two cells to add / remove a wall. Drag along borders to draw several.",
    icon: '<path d="M4 4v16M4 12h8M12 4v16" /><path d="M16 8h4M16 16h4" opacity=".45"/>' },
  { id: "bridge", key: "G", label: "Bridge", hint: "Click two cells to join them with a bridge (e.g. between islands). Click a bridge to remove it. Esc cancels.",
    icon: '<circle cx="5" cy="17" r="2"/><circle cx="19" cy="17" r="2"/><path d="M6.5 15.5C9 8 15 8 17.5 15.5" stroke-dasharray="2.5 2.5"/>' },
  { id: "number", key: "N", label: "Numbers", hint: "Click a cell to place the next number. Drag a number to move it (onto another number: swap). Click a number to delete it.",
    icon: '<circle cx="12" cy="12" r="8.5"/><path d="M10.5 9.5 12.5 8v8"/>' },
  { id: "zlink", key: "L", label: "Links", hint: "3D: click a cell to cut / restore its link to the cell one layer up (Shift: one layer down).", only3d: true,
    icon: '<path d="M12 3v7M12 14v7"/><path d="m8.5 6.5 3.5-3.5 3.5 3.5M8.5 17.5l3.5 3.5 3.5-3.5"/><path d="M7 12h10" stroke-dasharray="2 2"/>' },
];

// ------------------------------------------------------------------ state
const E = {
  R: 7, C: 7, Z: 1,
  on: new Uint8Array(49),
  walls: new Set(),          // "a-b" (a < b, cell indices): no edge between these grid neighbours
  bridges: [],               // [a, b] cell pairs joined by an extra edge
  cps: [],                   // cell indices, in order
  layer: 0,
  tool: "paint",
  pending: null,             // bridge start cell
  insertAt: -1,              // -1 = append; else index the next number is inserted at
  id: null, name: "",
  savedSig: null,
  analysis: null,            // server analysis for the current signature
  solution: null, alt: null, showAlt: false,
};
const undoStack = [], redoStack = [];
let view = null;             // {cs, pad, svg}
let drag = null;
let seq = 0, analyzeTimer = null;
let preview = null, previewTimer = null, previewOn = false;

// ------------------------------------------------------------------ helpers
const idx = (r, c, z) => (z * E.R + r) * E.C + c;
function rcz(i) { const c = i % E.C, t = (i - c) / E.C; return [t % E.R, c, Math.floor(t / E.R)]; }
const wkey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
const inside = (r, c, z) => r >= 0 && c >= 0 && z >= 0 && r < E.R && c < E.C && z < E.Z;
function pathColor(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(Math.floor(t), STOPS.length - 2), f = t - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]); }
async function api(url, body, method) {
  const opt = body === undefined ? { method: method || "GET" } : { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(url, opt);
  let j = null;
  try { j = await r.json(); } catch { /* ignore */ }
  if (!r.ok) throw new Error((j && (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail))) || `${r.status} ${r.statusText}`);
  return j;
}
let toastTimer = 0;
function toast(msg, tone = "") {
  const t = $("toast");
  t.textContent = msg;
  t.className = `ed-toast show ${tone}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = "ed-toast"; }, 3200);
}
function status(msg) { $("statusMsg").textContent = msg; }
function neighbours(i) {
  const [r, c, z] = rcz(i), out = [];
  for (const [dr, dc, dz] of [[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]) {
    if (inside(r + dr, c + dc, z + dz)) out.push(idx(r + dr, c + dc, z + dz));
  }
  return out;
}
function gridAdjacent(a, b) {
  const [r1, c1, z1] = rcz(a), [r2, c2, z2] = rcz(b);
  return Math.abs(r1 - r2) + Math.abs(c1 - c2) + Math.abs(z1 - z2) === 1;
}

// ------------------------------------------------------------------ snapshots / undo
function snap() {
  return { R: E.R, C: E.C, Z: E.Z, on: Array.from(E.on).join(""), walls: [...E.walls], bridges: E.bridges.map((b) => b.slice()), cps: E.cps.slice(), layer: E.layer };
}
function restore(s) {
  E.R = s.R; E.C = s.C; E.Z = s.Z;
  E.on = Uint8Array.from(String(s.on), (ch) => (ch === "1" ? 1 : 0));
  if (E.on.length !== E.R * E.C * E.Z) E.on = new Uint8Array(E.R * E.C * E.Z);
  E.walls = new Set(s.walls || []);
  E.bridges = (s.bridges || []).map((b) => b.slice());
  E.cps = (s.cps || []).slice();
  E.layer = Math.min(s.layer || 0, E.Z - 1);
  E.pending = null;
}
function commit() {
  undoStack.push(JSON.stringify(snap()));
  if (undoStack.length > 300) undoStack.shift();
  redoStack.length = 0;
}
function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.stringify(snap()));
  restore(JSON.parse(undoStack.pop()));
  changed(true);
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.stringify(snap()));
  restore(JSON.parse(redoStack.pop()));
  changed(true);
}

// ------------------------------------------------------------------ model edits
function purge(i) {
  E.cps = E.cps.filter((c) => c !== i);
  E.bridges = E.bridges.filter(([a, b]) => a !== i && b !== i);
  for (const k of [...E.walls]) { const [a, b] = k.split("-").map(Number); if (a === i || b === i) E.walls.delete(k); }
  if (E.pending === i) E.pending = null;
}
function setCell(i, v) {
  if (E.on[i] === v) return false;
  if (v && countOn() >= MAX_NODES) { toast(`At most ${MAX_NODES} cells`, "bad"); return false; }
  E.on[i] = v;
  if (!v) purge(i);
  return true;
}
function countOn() { let k = 0; for (const x of E.on) k += x; return k; }
function resize(R, C, Z, dr = 0, dc = 0) {
  R = Math.max(1, Math.min(MAX_SIDE, R | 0)); C = Math.max(1, Math.min(MAX_SIDE, C | 0)); Z = Math.max(1, Math.min(MAX_Z, Z | 0));
  const old = { R: E.R, C: E.C, Z: E.Z, on: E.on };
  const map = (i) => {
    const c = i % old.C, t = (i - c) / old.C, r = t % old.R, z = Math.floor(t / old.R);
    const nr = r + dr, nc = c + dc;
    return nr >= 0 && nc >= 0 && nr < R && nc < C && z < Z ? (z * R + nr) * C + nc : -1;
  };
  const on = new Uint8Array(R * C * Z);
  old.on.forEach((v, i) => { if (v) { const j = map(i); if (j >= 0) on[j] = 1; } });
  const walls = new Set();
  for (const k of E.walls) { const [a, b] = k.split("-").map(Number).map(map); if (a >= 0 && b >= 0) walls.add(wkey(a, b)); }
  const bridges = E.bridges.map(([a, b]) => [map(a), map(b)]).filter(([a, b]) => a >= 0 && b >= 0);
  const cps = E.cps.map(map).filter((x) => x >= 0);
  Object.assign(E, { R, C, Z, on, walls, bridges, cps });
  E.layer = Math.min(E.layer, Z - 1);
  E.pending = null;
}
function trim() {
  let r0 = Infinity, r1 = -1, c0 = Infinity, c1 = -1, z0 = Infinity, z1 = -1;
  E.on.forEach((v, i) => { if (!v) return; const [r, c, z] = rcz(i); r0 = Math.min(r0, r); r1 = Math.max(r1, r); c0 = Math.min(c0, c); c1 = Math.max(c1, c); z0 = Math.min(z0, z); z1 = Math.max(z1, z); });
  if (r1 < 0) return false;
  // shift layers down as well
  if (z0 > 0) {
    const R = E.R, C = E.C;
    const mapZ = (i) => { const [r, c, z] = rcz(i); return ((z - z0) * R + r) * C + c; };
    const on = new Uint8Array(E.on.length);
    E.on.forEach((v, i) => { if (v) on[mapZ(i)] = 1; });
    E.walls = new Set([...E.walls].map((k) => { const [a, b] = k.split("-").map(Number); return wkey(mapZ(a), mapZ(b)); }));
    E.bridges = E.bridges.map(([a, b]) => [mapZ(a), mapZ(b)]);
    E.cps = E.cps.map(mapZ);
    E.on = on;
  }
  resize(r1 - r0 + 1, c1 - c0 + 1, z1 - z0 + 1, -r0, -c0);
  return true;
}

// ------------------------------------------------------------------ model -> puzzle dict
function build() {
  const nodes = [], nodeOf = new Int32Array(E.on.length).fill(-1);
  let r0 = Infinity, c0 = Infinity;
  for (let r = 0; r < E.R; r++) for (let c = 0; c < E.C; c++) for (let z = 0; z < E.Z; z++) {
    const i = idx(r, c, z);
    if (E.on[i]) { nodeOf[i] = nodes.length; nodes.push(i); r0 = Math.min(r0, r); c0 = Math.min(c0, c); }
  }
  const coords = nodes.map((i) => { const [r, c, z] = rcz(i); return E.Z > 1 ? [r - r0, c - c0, z] : [r - r0, c - c0]; });
  const edges = [], walls = [], bridges = [];
  // islands: components of grid-adjacent cells (walls do not split islands; gaps do)
  const parent = nodes.map((_, k) => k);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (const i of nodes) {
    const [r, c, z] = rcz(i);
    for (const [dr, dc, dz] of [[0, 1, 0], [1, 0, 0], [0, 0, 1]]) {
      if (!inside(r + dr, c + dc, z + dz)) continue;
      const j = idx(r + dr, c + dc, z + dz);
      if (!E.on[j]) continue;
      parent[find(nodeOf[i])] = find(nodeOf[j]);
      if (E.walls.has(wkey(i, j))) walls.push([nodeOf[i], nodeOf[j]]);
      else edges.push([nodeOf[i], nodeOf[j]]);
    }
  }
  const seen = new Set();
  for (const [a, b] of E.bridges) {
    if (!E.on[a] || !E.on[b] || a === b) continue;
    const u = nodeOf[a], v = nodeOf[b], k = wkey(u, v);
    if (seen.has(k)) continue;
    seen.add(k);
    edges.push([u, v]); bridges.push([u, v]);
  }
  const rootId = new Map();
  const island = nodes.map((_, k) => { const r = find(k); if (!rootId.has(r)) rootId.set(r, rootId.size); return rootId.get(r); });
  const cps = E.cps.filter((i) => E.on[i]).map((i) => nodeOf[i]);
  let rMax = 0, cMax = 0;
  for (const c of coords) { rMax = Math.max(rMax, c[0]); cMax = Math.max(cMax, c[1]); }
  const meta = { shape: E.Z > 1 ? [rMax + 1, cMax + 1, E.Z] : [rMax + 1, cMax + 1], editor: { rows: E.R, cols: E.C, layers: E.Z } };
  if (walls.length) meta.walls = walls;
  if (bridges.length) meta.bridges = bridges;
  if (rootId.size > 1) meta.island = island;
  return { puzzle: { kind: "custom", coords, edges, meta, checkpoints: cps }, nodes, nodeOf, islands: rootId.size, islandOfCell: (i) => island[nodeOf[i]] };
}
function signature(pz) { return JSON.stringify([pz.coords, pz.edges, pz.checkpoints]); }

// ------------------------------------------------------------------ puzzle dict -> model
function importPuzzle(d, name) {
  if (d && d.puzzle && d.puzzle.coords) { name = name || d.name; d = d.puzzle; }
  if (!d || !Array.isArray(d.coords) || !d.coords.length) throw new Error("not a puzzle (no coords)");
  const dim = d.coords[0].length;
  if (dim > 3) throw new Error("the editor handles 2D and 3D (layered) puzzles; this one is 4D");
  const P = d.coords.map((c) => {
    const x = c.map(Number);
    if (x.some((v) => !Number.isFinite(v) || Math.abs(v - Math.round(v)) > 1e-6)) throw new Error("coords must be integer grid positions");
    return dim === 1 ? [0, Math.round(x[0]), 0] : [Math.round(x[0]), Math.round(x[1]), dim === 3 ? Math.round(x[2]) : 0];
  });
  const mins = [0, 1, 2].map((k) => Math.min(...P.map((p) => p[k])));
  const maxs = [0, 1, 2].map((k) => Math.max(...P.map((p) => p[k])));
  const R = maxs[0] - mins[0] + 1, C = maxs[1] - mins[1] + 1, Z = maxs[2] - mins[2] + 1;
  if (R > MAX_SIDE || C > MAX_SIDE || Z > MAX_Z) throw new Error(`too big for the editor (${R}×${C}×${Z}; max ${MAX_SIDE}×${MAX_SIDE}×${MAX_Z})`);
  commit();
  E.R = R; E.C = C; E.Z = Z;
  E.on = new Uint8Array(R * C * Z);
  const cellOf = P.map((p) => idx(p[0] - mins[0], p[1] - mins[1], p[2] - mins[2]));
  if (new Set(cellOf).size !== cellOf.length) throw new Error("two nodes share a grid position");
  for (const i of cellOf) E.on[i] = 1;
  const edgeSet = new Set((d.edges || []).map(([u, v]) => wkey(cellOf[u], cellOf[v])));
  E.walls = new Set();
  E.bridges = [];
  for (const i of cellOf) for (const j of neighbours(i)) if (j > i && E.on[j] && !edgeSet.has(wkey(i, j))) E.walls.add(wkey(i, j));
  for (const [u, v] of d.edges || []) { const a = cellOf[u], b = cellOf[v]; if (!gridAdjacent(a, b)) E.bridges.push([a, b]); }
  E.cps = (d.checkpoints || []).map((v) => cellOf[v]).filter((x) => x != null);
  E.layer = 0; E.pending = null; E.insertAt = -1;
  if (name != null) { E.name = name; $("pzName").value = name; }
}

// ------------------------------------------------------------------ local checks
function localChecks(b) {
  const pz = b.puzzle, n = pz.coords.length;
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of pz.edges) { adj[u].push(v); adj[v].push(u); }
  const out = { n, islands: b.islands, adj };
  // components
  const comp = new Int32Array(n).fill(-1);
  let nc = 0;
  for (let s = 0; s < n; s++) {
    if (comp[s] >= 0) continue;
    comp[s] = nc;
    const st = [s];
    while (st.length) { const u = st.pop(); for (const w of adj[u]) if (comp[w] < 0) { comp[w] = nc; st.push(w); } }
    nc++;
  }
  out.components = nc;
  out.connected = n > 0 && nc === 1;
  // bipartite parity
  const colr = new Int8Array(n).fill(-1);
  let bip = true;
  for (let s = 0; s < n && bip; s++) {
    if (colr[s] >= 0) continue;
    colr[s] = 0;
    const st = [s];
    while (st.length && bip) { const u = st.pop(); for (const w of adj[u]) { if (colr[w] < 0) { colr[w] = colr[u] ^ 1; st.push(w); } else if (colr[w] === colr[u]) { bip = false; break; } } }
  }
  out.bipartite = bip;
  const cps = pz.checkpoints, start = cps.length ? cps[0] : null, end = cps.length >= 2 ? cps[cps.length - 1] : null;
  out.parity = true; out.parityMsg = "";
  if (bip && n) {
    let a = 0; for (let v = 0; v < n; v++) if (colr[v] === 0) a++;
    const bb = n - a;
    out.colours = [a, bb];
    if (Math.abs(a - bb) > 1) { out.parity = false; out.parityMsg = `${Math.max(a, bb)} vs ${Math.min(a, bb)} chessboard colours (may differ by 1 at most)`; }
    else if (a === bb) { if (start != null && end != null && colr[start] === colr[end]) { out.parity = false; out.parityMsg = "even cell count: 1 and the last number need different colours"; } }
    else {
      const major = a > bb ? 0 : 1;
      const badEnds = [start, end].filter((x) => x != null && colr[x] !== major);
      if (badEnds.length) { out.parity = false; out.parityMsg = "odd cell count: the path must start and end on the majority colour"; }
    }
    out.colr = colr;
  }
  // degrees
  out.deadCells = []; out.deadEnds = [];
  for (let v = 0; v < n; v++) { if (n > 1 && adj[v].length === 0) out.deadCells.push(v); else if (adj[v].length === 1) out.deadEnds.push(v); }
  const ends = new Set([start, end].filter((x) => x != null));
  const badLeaves = out.deadEnds.filter((v) => !ends.has(v));
  out.degreeOk = !out.deadCells.length && out.deadEnds.length <= 2 && badLeaves.length <= 2 - ends.size;
  out.badLeaves = out.deadEnds.length > 2 ? out.deadEnds : badLeaves.length > 2 - ends.size ? badLeaves : [];
  out.numbers = cps.length;
  out.structOk = n > 1 && out.connected && out.parity && out.degreeOk;
  return out;
}

// ------------------------------------------------------------------ rendering
let lastBuild = null, lastLocal = null;
function layoutBoard() {
  const box = $("board");
  const stage = $("stage");
  const availW = Math.max(220, (box.clientWidth || stage.clientWidth || 600) - 4);
  const top = box.getBoundingClientRect().top;
  const availH = Math.max(260, window.innerHeight - Math.max(60, top) - 70);
  const cs = Math.max(16, Math.min(64, Math.floor(Math.min(availW / (E.C + 1.1), availH / (E.R + 1.1)))));
  return { cs, pad: Math.round(cs * 0.55) };
}
function cellXY(i) { const [r, c] = rcz(i); return [view.pad + (c + 0.5) * view.cs, view.pad + (r + 0.5) * view.cs]; }

function render() {
  const b = lastBuild;
  const L = lastLocal;
  const { cs, pad } = layoutBoard();
  const W = E.C * cs + 2 * pad, H = E.R * cs + 2 * pad;
  const box = $("board");
  box.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: `ed-svg tool-${E.tool}`, role: "img", "aria-label": "Editable puzzle grid" }, box);
  view = { cs, pad, svg, W, H };
  const z = E.layer;
  const gBase = el("g", {}, svg);
  const rad = cs * 0.2;
  const multi = b.islands > 1;
  // slots + cells
  for (let r = 0; r < E.R; r++) for (let c = 0; c < E.C; c++) {
    const i = idx(r, c, z);
    const x = pad + c * cs, y = pad + r * cs;
    if (!E.on[i]) {
      const below = z > 0 && E.on[idx(r, c, z - 1)];
      el("rect", { x: x + 3, y: y + 3, width: cs - 6, height: cs - 6, rx: rad, class: below ? "ed-slot ed-under" : "ed-slot" }, gBase);
      continue;
    }
    const cell = el("rect", { x: x + 1.5, y: y + 1.5, width: cs - 3, height: cs - 3, rx: rad, class: "ed-cell" }, gBase);
    if (multi) cell.style.fill = `var(--island-${b.islandOfCell(i) % 8})`;
  }
  // issue markers (dead ends / isolated)
  if (L && b.puzzle.coords.length > 1) {
    const mark = new Set([...L.deadCells, ...(L.badLeaves || [])]);
    for (const v of mark) {
      const i = b.nodes[v];
      const [r, c, zz] = rcz(i);
      if (zz !== z) continue;
      el("rect", { x: pad + c * cs + 4, y: pad + r * cs + 4, width: cs - 8, height: cs - 8, rx: rad * 0.8, class: "ed-issue" }, gBase);
    }
  }
  // vertical link markers (3D)
  if (E.Z > 1) {
    const g = el("g", { class: "ed-zmarks" }, svg);
    for (let r = 0; r < E.R; r++) for (let c = 0; c < E.C; c++) {
      const i = idx(r, c, z);
      if (!E.on[i]) continue;
      const x = pad + c * cs, y = pad + r * cs, s = Math.max(4, cs * 0.12);
      for (const [dz, cx, cy, dir] of [[1, x + cs - s * 1.6, y + s * 1.7, -1], [-1, x + s * 1.6, y + cs - s * 1.7, 1]]) {
        if (!inside(r, c, z + dz)) continue;
        const j = idx(r, c, z + dz);
        if (!E.on[j]) continue;
        const cut = E.walls.has(wkey(i, j));
        if (!cut && E.tool !== "zlink") continue;   // links are the default: only show cuts, unless editing links
        el("path", { d: `M${cx - s},${cy - dir * s * 0.55} L${cx},${cy + dir * s * 0.65} L${cx + s},${cy - dir * s * 0.55}`, class: cut ? "ed-zlink cut" : "ed-zlink" }, g).appendChild(document.createElementNS(SVGNS, "title")).textContent = cut ? `cut: no link to layer ${z + dz}` : `linked to layer ${z + dz}`;
        if (cut) el("line", { x1: cx - s * 0.9, y1: cy - s * 0.9, x2: cx + s * 0.9, y2: cy + s * 0.9, class: "ed-zcut" }, g);
      }
    }
  }
  // walls
  const gW = el("g", {}, svg);
  const ww = Math.max(3, cs * 0.12);
  for (const k of E.walls) {
    const [a, bb] = k.split("-").map(Number);
    if (!E.on[a] || !E.on[bb]) continue;
    const [r1, c1, z1] = rcz(a), [r2, c2, z2] = rcz(bb);
    if (z1 !== z || z2 !== z) continue;
    const [line] = wallLine(r1, c1, r2, c2);
    el("line", { ...line, class: "ed-wall", "stroke-width": ww }, gW);
  }
  // bridges
  const gB = el("g", {}, svg);
  E.bridges.forEach(([a, bb], k) => {
    if (!E.on[a] || !E.on[bb]) return;
    const za = rcz(a)[2], zb = rcz(bb)[2];
    if (za !== z && zb !== z) return;
    let d;
    if (za === z && zb === z) d = arcD(a, bb);
    else {
      // cross-layer bridge: stub towards the other end + layer tag
      const here = za === z ? a : bb, there = za === z ? bb : a;
      const [x, y] = cellXY(here), [tx, ty] = cellXY(there);
      const ang = Math.atan2(ty - y || -1, tx - x || 1), L2 = cs * 0.95;
      const ex = x + Math.cos(ang) * L2, ey = y + Math.sin(ang) * L2;
      d = `M${x},${y} Q${(x + ex) / 2 - Math.sin(ang) * cs * 0.3},${(y + ey) / 2 + Math.cos(ang) * cs * 0.3} ${ex},${ey}`;
      const tag = el("g", { class: "ed-btag" }, gB);
      const txt = `z${rcz(there)[2]}`;
      el("rect", { x: ex - cs * 0.3, y: ey - cs * 0.17, width: cs * 0.6, height: cs * 0.34, rx: cs * 0.17 }, tag);
      el("text", { x: ex, y: ey + cs * 0.08, "font-size": cs * 0.22, "text-anchor": "middle" }, tag).textContent = txt;
    }
    el("path", { d, class: "ed-bridge", "stroke-width": Math.max(2.5, cs * 0.075), "stroke-dasharray": `${cs * 0.16} ${cs * 0.12}` }, gB);
    el("path", { d, class: "ed-bridge-hit", "stroke-width": Math.max(10, cs * 0.3), "data-bridge": k }, gB);
  });
  // solution overlay
  const sol = E.showAlt && E.alt ? E.alt : E.solution;
  if ($("showSol").checked && sol && b) {
    const gS = el("g", { class: "ed-sol" }, svg);
    const n = sol.length;
    for (let k = 0; k + 1 < n; k++) {
      const a = b.nodes[sol[k]], c2 = b.nodes[sol[k + 1]];
      if (a == null || c2 == null) continue;
      const za = rcz(a)[2], zc = rcz(c2)[2];
      if (za !== z && zc !== z) continue;
      const [x1, y1] = cellXY(a), [x2, y2] = cellXY(c2);
      if (za !== zc) {
        const [x, y] = za === z ? [x1, y1] : [x2, y2];
        el("circle", { cx: x, cy: y, r: cs * 0.16, fill: pathColor(k / (n - 1)), opacity: 0.8 }, gS);
        continue;
      }
      const adjOk = gridAdjacent(a, c2);
      if (adjOk) el("line", { x1, y1, x2, y2, stroke: pathColor((k + 0.5) / (n - 1)), "stroke-width": cs * 0.3, "stroke-linecap": "round" }, gS);
      else el("path", { d: arcD(a, c2), stroke: pathColor((k + 0.5) / (n - 1)), "stroke-width": cs * 0.22, fill: "none", "stroke-linecap": "round" }, gS);
    }
  }
  // checkpoints
  const gC = el("g", {}, svg);
  const next = E.insertAt >= 0 ? E.insertAt : E.cps.length;
  E.cps.forEach((i, k) => {
    if (!E.on[i] || rcz(i)[2] !== z) return;
    const [x, y] = cellXY(i);
    const g = el("g", { class: `ed-cp${drag && drag.kind === "cp" && drag.k === k && drag.moved ? " dragging" : ""}`, "data-cp": k }, gC);
    el("circle", { cx: x, cy: y, r: cs * 0.31, class: "ed-cp-dot" }, g);
    const lbl = k + 1;
    el("text", { x, y: y + cs * 0.11, "font-size": cs * (lbl >= 100 ? 0.22 : lbl >= 10 ? 0.27 : 0.32), "text-anchor": "middle", class: "ed-cp-txt" }, g).textContent = lbl;
    if (k === 0 || k === E.cps.length - 1) el("circle", { cx: x, cy: y, r: cs * 0.38, class: "ed-cp-end" }, g);
  });
  void next;
  // pending bridge start
  if (E.pending != null && E.on[E.pending] && rcz(E.pending)[2] === z) {
    const [x, y] = cellXY(E.pending);
    el("circle", { cx: x, cy: y, r: cs * 0.4, class: "ed-pending" }, svg);
  }
  el("g", { id: "fx" }, svg);
  renderFx();
}
function wallLine(r1, c1, r2, c2) {
  const { cs, pad } = view;
  if (r1 === r2) { const x = pad + Math.max(c1, c2) * cs, y = pad + r1 * cs; return [{ x1: x, y1: y + 3, x2: x, y2: y + cs - 3 }]; }
  const y = pad + Math.max(r1, r2) * cs, x = pad + c1 * cs;
  return [{ x1: x + 3, y1: y, x2: x + cs - 3, y2: y }];
}
function arcD(a, b) {
  const [x1, y1] = cellXY(a), [x2, y2] = cellXY(b);
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
  let nx = -dy / len, ny = dx / len;
  if (ny > 0 || (Math.abs(ny) < 1e-6 && nx > 0)) { nx = -nx; ny = -ny; }
  const bend = Math.min(0.28 * len, 2.2 * view.cs);
  return `M${x1},${y1} Q${(x1 + x2) / 2 + nx * bend},${(y1 + y2) / 2 + ny * bend} ${x2},${y2}`;
}

// transient overlays: hover, wall preview, rect preview, number drag ghost
let hover = null;   // {r, c, i, edge?}
function renderFx() {
  const g = view && view.svg.querySelector("#fx");
  if (!g) return;
  g.innerHTML = "";
  const { cs, pad } = view;
  if (drag && drag.kind === "rect") {
    const [ra, ca] = drag.a, [rb, cb] = drag.b;
    const x = pad + Math.min(ca, cb) * cs, y = pad + Math.min(ra, rb) * cs;
    el("rect", { x, y, width: (Math.abs(cb - ca) + 1) * cs, height: (Math.abs(rb - ra) + 1) * cs, rx: cs * 0.2, class: drag.erase ? "ed-rectprev erase" : "ed-rectprev" }, g);
    const t = el("text", { x: x + 6, y: y - 6, class: "ed-dimtxt", "font-size": 12 }, g);
    t.textContent = `${Math.abs(rb - ra) + 1} × ${Math.abs(cb - ca) + 1}`;
    return;
  }
  if (!hover) return;
  const { r, c } = hover;
  if (E.tool === "wall" && hover.edge) {
    const [a, b] = hover.edge;
    const [r1, c1] = rcz(a), [r2, c2] = rcz(b);
    const [line] = wallLine(r1, c1, r2, c2);
    el("line", { ...line, class: E.walls.has(wkey(a, b)) ? "ed-wallprev remove" : "ed-wallprev", "stroke-width": Math.max(3, cs * 0.12) }, g);
    return;
  }
  if (!inside(r, c, E.layer)) return;
  el("rect", { x: pad + c * cs + 1.5, y: pad + r * cs + 1.5, width: cs - 3, height: cs - 3, rx: cs * 0.2, class: `ed-hover${E.tool === "erase" ? " erase" : ""}` }, g);
  if (E.tool === "bridge" && E.pending != null && E.on[hover.i] && hover.i !== E.pending && rcz(E.pending)[2] === E.layer) {
    el("path", { d: arcD(E.pending, hover.i), class: "ed-bridgeprev", "stroke-width": Math.max(2, cs * 0.06) }, g);
  }
  if (drag && drag.kind === "cp" && drag.moved) {
    const [x, y] = cellXY(hover.i);
    el("circle", { cx: x, cy: y, r: cs * 0.31, class: "ed-cp-ghost" }, g);
    const t = el("text", { x, y: y + cs * 0.11, "font-size": cs * 0.3, "text-anchor": "middle", class: "ed-cp-ghosttxt" }, g);
    t.textContent = drag.k + 1;
  }
  if (E.tool === "number" && !drag && E.on[hover.i] && !E.cps.includes(hover.i)) {
    const [x, y] = cellXY(hover.i);
    const t = el("text", { x, y: y + cs * 0.11, "font-size": cs * 0.3, "text-anchor": "middle", class: "ed-nextnum" }, g);
    t.textContent = (E.insertAt >= 0 ? E.insertAt : E.cps.length) + 1;
  }
}

// ------------------------------------------------------------------ side panels
function renderTools() {
  const box = $("tools");
  box.innerHTML = "";
  for (const t of TOOLS) {
    if (t.only3d && E.Z < 2) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.className = `ed-tool${E.tool === t.id ? " on" : ""}`;
    b.dataset.tool = t.id;
    b.title = `${t.label} (${t.key})`;
    b.setAttribute("aria-pressed", E.tool === t.id);
    b.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${t.icon}</svg><span>${t.label}</span><kbd>${t.key}</kbd>`;
    b.onclick = () => setTool(t.id);
    box.appendChild(b);
  }
  const t = TOOLS.find((x) => x.id === E.tool);
  $("toolHint").innerHTML = t ? `<kbd>${t.key}</kbd> ${escapeHtml(t.hint)}` : "";
}
function setTool(id) {
  if (id === "zlink" && E.Z < 2) return;
  E.tool = id;
  if (id !== "bridge") E.pending = null;
  renderTools();
  render();
}
function renderLayers() {
  const box = $("layerTabs");
  box.innerHTML = "";
  box.hidden = E.Z < 2;
  $("preview3dBtn").hidden = E.Z < 2;
  if (E.Z < 2) { if (previewOn) togglePreview(false); return; }
  const lbl = document.createElement("span");
  lbl.className = "muted"; lbl.textContent = "Layer";
  box.appendChild(lbl);
  for (let z = 0; z < E.Z; z++) {
    let k = 0; for (let r = 0; r < E.R; r++) for (let c = 0; c < E.C; c++) k += E.on[idx(r, c, z)];
    const b = document.createElement("button");
    b.type = "button"; b.role = "tab";
    b.className = `ed-ltab${z === E.layer ? " on" : ""}`;
    b.innerHTML = `z${z}<small>${k}</small>`;
    b.title = `Layer z = ${z} (${k} cells)`;
    b.setAttribute("aria-selected", z === E.layer);
    b.onclick = () => { E.layer = z; renderLayers(); render(); };
    box.appendChild(b);
  }
}
const CHECK_ICON = { ok: "✓", bad: "✕", warn: "!", unknown: "?", run: "", idle: "–" };
function renderChecks() {
  const L = lastLocal, b = lastBuild, A = E.analysis;
  const items = [];
  const n = L.n;
  items.push(["idle", "Cells", `${n}${b.islands > 1 ? ` in ${b.islands} islands` : ""}${E.Z > 1 ? ` · ${E.Z} layers` : ""}`]);
  if (n < 2) {
    items.push(["warn", "Draw some cells", "paint at least 2 cells"]);
  } else {
    items.push([L.connected ? "ok" : "bad", "Connected", L.connected ? (b.islands > 1 ? "islands joined by bridges" : "one piece") : `${L.components} separate groups: add bridges`]);
    items.push([L.parity ? "ok" : "bad", "Parity", L.parity ? (L.bipartite ? `${L.colours[0]} / ${L.colours[1]} cells per colour` : "not a chessboard graph (bridges)") : L.parityMsg]);
    items.push([L.degreeOk ? "ok" : "bad", "Dead ends", L.degreeOk ? (L.deadEnds.length ? `${L.deadEnds.length} end cell(s), fine` : "none") : L.deadCells.length ? `${L.deadCells.length} cell(s) with no neighbour` : `${L.badLeaves.length} dead-end cells (only the path ends may be)`]);
    items.push([L.numbers >= 2 ? "ok" : "warn", "Start & end", L.numbers >= 2 ? `${L.numbers} numbers` : "place at least 1 and a last number"]);
    const running = E.analysisPending;
    let sv, st, uv, ut;
    if (!L.structOk) { sv = "bad"; st = "fix the problems above"; uv = "bad"; ut = "-"; }
    else if (running) { sv = uv = "run"; st = ut = "checking..."; }
    else if (!A) { sv = uv = "idle"; st = ut = $("autoCheck").checked ? "waiting" : "press Deep check"; }
    else {
      if (A.solvable === true) { sv = "ok"; st = L.numbers >= 2 ? `yes (${fmtSec(A.seconds)})` : "the shape has a path through every cell"; }
      else if (A.solvable === false) { sv = "bad"; st = L.numbers >= 2 ? "no solution: the numbers cannot be visited in order" : "no path covers every cell"; }
      else { sv = "unknown"; st = `unknown (gave up after ${fmtSec(A.seconds)})`; }
      if (L.numbers < 2) { uv = "idle"; ut = "add numbers first"; }
      else if (A.unique === true) { uv = "ok"; ut = "exactly one solution"; }
      else if (A.unique === false && A.solvable) { uv = "bad"; ut = "2+ solutions: add numbers or walls"; }
      else if (A.solvable === false) { uv = "bad"; ut = "-"; }
      else { uv = "unknown"; ut = "not proven in time (Deep check)"; }
    }
    items.push([sv, "Solvable", st]);
    items.push([uv, "Unique", ut]);
  }
  $("checks").innerHTML = items.map(([s, k, d]) => `<li class="ck ${s}"><span class="ic">${s === "run" ? '<i class="ed-spin"></i>' : CHECK_ICON[s]}</span><span class="k">${k}</span><span class="d">${escapeHtml(d)}</span></li>`).join("");
  $("altBtn").hidden = !(E.alt && E.solution);
  $("altBtn").textContent = E.showAlt ? "Show 1st solution" : "Show 2nd solution";
  const ok = L.structOk && L.numbers >= 2;
  $("uniqueBtn").disabled = !ok || (A && A.unique === true);
  $("suggestBtn").disabled = !L.structOk;
  for (const id of ["saveBtn", "saveNewBtn", "playBtn", "aiBtn"]) $(id).disabled = n < 2 || L.numbers < 2;
}
function fmtSec(s) { return s == null ? "-" : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`; }
function renderCps() {
  const list = $("cpList");
  list.innerHTML = "";
  $("cpCount").textContent = E.cps.length ? `${E.cps.length}` : "";
  E.cps.forEach((i, k) => {
    const [r, c, z] = rcz(i);
    const li = document.createElement("li");
    li.draggable = true;
    li.dataset.k = k;
    li.className = "ed-cpitem";
    li.innerHTML = `<span class="num">${k + 1}</span><span class="pos">r${r} c${c}${E.Z > 1 ? ` z${z}` : ""}</span>
      <span class="tag">${k === 0 ? "start" : k === E.cps.length - 1 ? "end" : ""}</span>
      <button type="button" class="ed-x up" title="Move up" aria-label="Move number ${k + 1} up">↑</button>
      <button type="button" class="ed-x down" title="Move down" aria-label="Move number ${k + 1} down">↓</button>
      <button type="button" class="ed-x del" title="Delete" aria-label="Delete number ${k + 1}">✕</button>`;
    li.querySelector(".del").onclick = () => { commit(); E.cps.splice(k, 1); if (E.insertAt > E.cps.length) E.insertAt = -1; changed(); };
    li.querySelector(".up").onclick = () => { if (k > 0) { commit(); [E.cps[k - 1], E.cps[k]] = [E.cps[k], E.cps[k - 1]]; changed(); } };
    li.querySelector(".down").onclick = () => { if (k < E.cps.length - 1) { commit(); [E.cps[k + 1], E.cps[k]] = [E.cps[k], E.cps[k + 1]]; changed(); } };
    li.onmouseenter = () => { if (z !== E.layer) return; hover = { r, c, i }; renderFx(); };
    li.onclick = (e) => { if (e.target.closest("button")) return; if (z !== E.layer) { E.layer = z; renderLayers(); render(); } };
    li.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", String(k)); li.classList.add("dragging"); });
    li.addEventListener("dragend", () => li.classList.remove("dragging"));
    li.addEventListener("dragover", (e) => { e.preventDefault(); li.classList.add("over"); });
    li.addEventListener("dragleave", () => li.classList.remove("over"));
    li.addEventListener("drop", (e) => {
      e.preventDefault(); li.classList.remove("over");
      const from = Number(e.dataTransfer.getData("text/plain"));
      if (!Number.isInteger(from) || from === k) return;
      commit();
      const [m] = E.cps.splice(from, 1);
      E.cps.splice(k, 0, m);
      changed();
    });
    list.appendChild(li);
  });
  if (!E.cps.length) list.innerHTML = `<li class="ed-empty">No numbers yet. Use the Numbers tool (N) or “Suggest numbers”.</li>`;
  const sel = $("insertAt");
  const cur = E.insertAt;
  sel.innerHTML = `<option value="-1">the next number (${E.cps.length + 1})</option>` +
    E.cps.map((_, k) => `<option value="${k}">number ${k + 1} (insert before)</option>`).join("");
  sel.value = String(cur >= 0 && cur < E.cps.length ? cur : -1);
}

// ------------------------------------------------------------------ change pipeline
function changed(fromHistory = false) {
  void fromHistory;
  lastBuild = build();
  lastLocal = localChecks(lastBuild);
  const sig = signature(lastBuild.puzzle);
  if (!E.analysis || E.analysis.sig !== sig) {
    E.analysis = null; E.solution = null; E.alt = null; E.showAlt = false;
    scheduleAnalyze();
  }
  $("dimR").value = E.R; $("dimC").value = E.C; $("dimZ").value = E.Z;
  renderTools(); renderLayers(); render(); renderChecks(); renderCps();
  $("undoBtn").disabled = !undoStack.length;
  $("redoBtn").disabled = !redoStack.length;
  const dirty = E.savedSig !== sig || (E.name || "") !== ($("pzName").value || "");
  $("savedInfo").textContent = E.id ? `${dirty ? "Unsaved changes · " : "Saved · "}id ${E.id}` : dirty && lastLocal.n ? "Not saved yet" : "";
  saveDraft();
  schedulePreview();
}
function scheduleAnalyze() {
  clearTimeout(analyzeTimer);
  E.analysisPending = false;
  if (!$("autoCheck").checked || !lastLocal || !lastLocal.structOk) return;
  E.analysisPending = true;
  analyzeTimer = setTimeout(() => analyze(lastLocal.n > 400 ? 2 : 3, false), 450);
}
async function analyze(tl, manual) {
  if (!lastLocal.structOk) { if (manual) toast("Fix the structural problems first", "bad"); return; }
  const my = ++seq;
  const pz = lastBuild.puzzle, sig = signature(pz);
  E.analysisPending = true;
  renderChecks();
  if (manual) status("Running the exact solver...");
  try {
    const res = await api("/api/editor/analyze", { puzzle: pz, time_limit: tl, want_solution: true, check_unique: true });
    if (my !== seq || sig !== signature(lastBuild.puzzle)) return;
    E.analysis = { ...res, sig };
    E.solution = res.solution || null;
    E.alt = res.alternative || null;
    E.showAlt = false;
    if (manual) status(res.solvable ? (res.unique ? "Solvable with exactly one solution." : res.unique === false ? "Solvable, but not unique." : "Solvable (uniqueness not proven).") : res.solvable === false ? "No solution." : "The solver ran out of time.");
  } catch (e) {
    if (my !== seq) return;
    status(`Check failed: ${e.message}`);
  } finally {
    if (my === seq) { E.analysisPending = false; renderChecks(); render(); }
  }
}

// ------------------------------------------------------------------ pointer interaction
function eventCell(ev) {
  const pt = view.svg.createSVGPoint();
  pt.x = ev.clientX; pt.y = ev.clientY;
  const p = pt.matrixTransform(view.svg.getScreenCTM().inverse());
  const fx = (p.x - view.pad) / view.cs, fy = (p.y - view.pad) / view.cs;
  const c = Math.floor(fx), r = Math.floor(fy);
  const res = { r, c, fx: fx - c, fy: fy - r, i: inside(r, c, E.layer) ? idx(r, c, E.layer) : -1 };
  // nearest border (for walls)
  const cand = [];
  if (inside(r, c, E.layer)) {
    cand.push([res.fx, r, c - 1], [1 - res.fx, r, c + 1], [res.fy, r - 1, c], [1 - res.fy, r + 1, c]);
    cand.sort((a, b) => a[0] - b[0]);
    for (const [d, r2, c2] of cand) {
      if (d > 0.42) break;
      if (!inside(r2, c2, E.layer)) continue;
      const j = idx(r2, c2, E.layer);
      if (E.on[res.i] && E.on[j]) { res.edge = [res.i, j]; break; }
    }
  }
  return res;
}
function paintLine(a, b, v) {
  // Bresenham between two cells so fast drags leave no gaps
  let [r0, c0] = a; const [r1, c1] = b;
  const dr = Math.abs(r1 - r0), dc = Math.abs(c1 - c0), sr = r0 < r1 ? 1 : -1, sc = c0 < c1 ? 1 : -1;
  let err = dc - dr, any = false;
  for (let guard = 0; guard < 200; guard++) {
    if (inside(r0, c0, E.layer)) any = setCell(idx(r0, c0, E.layer), v) || any;
    if (r0 === r1 && c0 === c1) break;
    const e2 = 2 * err;
    if (e2 > -dr) { err -= dr; c0 += sc; }
    if (e2 < dc) { err += dc; r0 += sr; }
  }
  return any;
}
function onDown(ev) {
  if (!view || (ev.button !== 0 && ev.button !== 2)) return;
  const h = eventCell(ev);
  const right = ev.button === 2;
  const bridgeHit = ev.target.closest && ev.target.closest("[data-bridge]");
  ev.preventDefault();
  view.svg.setPointerCapture?.(ev.pointerId);
  if (E.tool === "bridge" && bridgeHit && !right) {
    commit();
    E.bridges.splice(Number(bridgeHit.dataset.bridge), 1);
    toast("Bridge removed");
    changed();
    return;
  }
  if (h.i < 0) return;
  const tool = right && (E.tool === "paint" || E.tool === "rect") ? (E.tool === "rect" ? "rect" : "erase") : E.tool;
  if (tool === "paint" || tool === "erase") {
    commit();
    const v = tool === "paint" ? 1 : 0;
    drag = { kind: "paint", v, last: [h.r, h.c] };
    if (!paintLine([h.r, h.c], [h.r, h.c], v)) undoStack.pop();
    else changed();
    drag.committed = true;
  } else if (tool === "rect") {
    drag = { kind: "rect", a: [h.r, h.c], b: [h.r, h.c], erase: right || ev.shiftKey };
    renderFx();
  } else if (tool === "wall") {
    if (!h.edge) { toast("Click on the border between two cells", ""); return; }
    commit();
    const k = wkey(...h.edge), v = !E.walls.has(k);
    if (v) E.walls.add(k); else E.walls.delete(k);
    drag = { kind: "wall", v };
    changed();
  } else if (tool === "bridge") {
    if (!E.on[h.i]) { toast("Bridges join two cells: paint cells first"); return; }
    if (E.pending == null) { E.pending = h.i; status("Now click the other end of the bridge (Esc to cancel)."); render(); return; }
    if (E.pending === h.i) { E.pending = null; status("Bridge cancelled."); render(); return; }
    const a = E.pending, b = h.i;
    if (gridAdjacent(a, b)) { toast("Neighbouring cells are already connected (use a wall to cut them)"); E.pending = null; render(); return; }
    commit();
    const k = E.bridges.findIndex(([x, y]) => (x === a && y === b) || (x === b && y === a));
    if (k >= 0) { E.bridges.splice(k, 1); toast("Bridge removed"); } else E.bridges.push([a, b]);
    E.pending = null;
    status("Bridge added. Click two more cells for another one.");
    changed();
  } else if (tool === "number") {
    if (!E.on[h.i]) { toast("Numbers go on cells: paint the cell first"); return; }
    const k = E.cps.indexOf(h.i);
    if (k >= 0) drag = { kind: "cp", k, from: h.i, moved: false };
    else {
      commit();
      if (E.insertAt >= 0 && E.insertAt <= E.cps.length) { E.cps.splice(E.insertAt, 0, h.i); E.insertAt++; if (E.insertAt >= E.cps.length) E.insertAt = -1; }
      else E.cps.push(h.i);
      changed();
    }
  } else if (tool === "zlink") {
    const dz = ev.shiftKey ? -1 : 1;
    const [r, c, z] = rcz(h.i);
    if (!inside(r, c, z + dz) || !E.on[idx(r, c, z + dz)] || !E.on[h.i]) { toast(`No cell ${dz > 0 ? "above" : "below"} to link to`); return; }
    commit();
    const k = wkey(h.i, idx(r, c, z + dz));
    if (E.walls.has(k)) E.walls.delete(k); else E.walls.add(k);
    changed();
  }
}
function onMove(ev) {
  if (!view) return;
  const h = eventCell(ev);
  const inCell = h.i >= 0;
  hover = inCell ? h : null;
  const [r, c] = [h.r, h.c];
  $("hoverInfo").textContent = inCell ? `r${r} c${c}${E.Z > 1 ? ` z${E.layer}` : ""}${E.on[h.i] ? "" : " (empty)"}${E.cps.includes(h.i) ? ` · number ${E.cps.indexOf(h.i) + 1}` : ""}` : "";
  if (drag) {
    if (drag.kind === "paint" && inCell) {
      if (drag.last[0] !== r || drag.last[1] !== c) { if (paintLine(drag.last, [r, c], drag.v)) changed(); drag.last = [r, c]; }
    } else if (drag.kind === "rect") {
      drag.b = [Math.max(0, Math.min(E.R - 1, r)), Math.max(0, Math.min(E.C - 1, c))];
    } else if (drag.kind === "wall" && h.edge) {
      const k = wkey(...h.edge);
      if (E.walls.has(k) !== drag.v) { if (drag.v) E.walls.add(k); else E.walls.delete(k); changed(); }
    } else if (drag.kind === "cp" && inCell && h.i !== drag.from) {
      if (!drag.moved) { drag.moved = true; render(); }
    }
  }
  renderFx();
}
function onUp(ev) {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (d.kind === "rect") {
    const [ra, ca] = d.a, [rb, cb] = d.b;
    commit();
    let any = false;
    for (let r = Math.min(ra, rb); r <= Math.max(ra, rb); r++) for (let c = Math.min(ca, cb); c <= Math.max(ca, cb); c++) any = setCell(idx(r, c, E.layer), d.erase ? 0 : 1) || any;
    if (!any) undoStack.pop();
    changed();
  } else if (d.kind === "cp") {
    const h = view ? eventCell(ev) : null;
    if (!d.moved || !h || h.i < 0 || h.i === d.from) {
      commit(); E.cps.splice(d.k, 1); toast(`Number ${d.k + 1} deleted${E.cps.length >= d.k + 1 ? " (later numbers shift down)" : ""}`);
      if (E.insertAt > E.cps.length) E.insertAt = -1;
    } else if (!E.on[h.i]) toast("Drop numbers on cells");
    else {
      commit();
      const j = E.cps.indexOf(h.i);
      if (j >= 0) { [E.cps[d.k], E.cps[j]] = [E.cps[j], E.cps[d.k]]; toast(`Swapped ${d.k + 1} and ${j + 1}`); }
      else E.cps[d.k] = h.i;
    }
    changed();
  } else render();
}

// ------------------------------------------------------------------ templates
function fromMask(rows, name, opts = {}) {
  commit();
  const R = rows.length, C = Math.max(...rows.map((r) => r.length)), Z = opts.Z || 1;
  E.R = R; E.C = C; E.Z = Z;
  E.on = new Uint8Array(R * C * Z);
  for (let z = 0; z < Z; z++) rows.forEach((row, r) => [...row].forEach((ch, c) => { if (ch !== "." && ch !== " ") E.on[idx(r, c, z)] = 1; }));
  E.walls = new Set(); E.bridges = []; E.cps = []; E.layer = 0; E.pending = null; E.insertAt = -1;
  (opts.walls || []).forEach(([r1, c1, r2, c2]) => E.walls.add(wkey(idx(r1, c1, 0), idx(r2, c2, 0))));
  (opts.bridges || []).forEach(([r1, c1, r2, c2]) => E.bridges.push([idx(r1, c1, 0), idx(r2, c2, 0)]));
  E.id = null; E.savedSig = null;
  E.name = name; $("pzName").value = name;
  changed();
  status(`Template “${name}”. Next: add numbers (N) or “Suggest numbers”.`);
}
const TEMPLATES = [
  { name: "Blank 7×7", fn: () => fromMask(Array(7).fill("#######"), "Blank 7×7") },
  { name: "Empty canvas", fn: () => { fromMask(Array(8).fill("........"), "Untitled"); setTool("paint"); } },
  { name: "Donut", fn: () => fromMask(["########", "########", "##....##", "##....##", "##....##", "##....##", "########", "########"], "Donut") },
  { name: "Cross", fn: () => fromMask(["...###...", "...###...", "...###...", "#########", "#########", "#########", "...###...", "...###...", "...###..."], "Cross") },
  { name: "Two islands", fn: () => fromMask(["####.####", "####.####", "####.####", "####.####"], "Two islands", { bridges: [[1, 3, 2, 5]] }) },
  { name: "Walled room", fn: () => fromMask(Array(6).fill("######"), "Walled room", { walls: [[1, 1, 1, 2], [2, 1, 2, 2], [1, 3, 2, 3], [1, 4, 2, 4], [3, 2, 4, 2], [3, 3, 4, 3], [4, 3, 4, 4], [3, 3, 3, 4]] }) },
  { name: "3D tower", fn: () => { fromMask(["###", "###", "###"], "3D tower", { Z: 3 }); } },
];
function renderTemplates() {
  const box = $("templates");
  box.innerHTML = "";
  for (const t of TEMPLATES) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "btn sm ed-tpl"; b.textContent = t.name;
    b.onclick = () => { t.fn(); };
    box.appendChild(b);
  }
}
async function randomTemplate() {
  const kind = $("randKind").value;
  const size = Number($("randSize").value) || 6;
  const btn = $("randBtn");
  btn.disabled = true;
  status("Generating...");
  try {
    const res = await api("/api/generate", { kind, size, include_solution: false, time_limit: 15 });
    importPuzzle(res.puzzle, `Random ${kind} ${res.seed}`);
    E.id = null; E.savedSig = null;
    changed();
    status(`Loaded a random ${kind} puzzle (seed ${res.seed}). Edit away!`);
  } catch (e) { toast(`Generate failed: ${e.message}`, "bad"); status("Ready."); }
  finally { btn.disabled = false; }
}

// ------------------------------------------------------------------ server actions
async function suggest() {
  const L = lastLocal;
  if (!L.structOk) return toast("Fix the structural problems first", "bad");
  const nWanted = Number($("suggestN").value) || null;
  const btn = $("suggestBtn");
  btn.disabled = true;
  status("Finding a random path through every cell...");
  try {
    const res = await api("/api/editor/suggest", { puzzle: { ...lastBuild.puzzle, checkpoints: [] }, num_checkpoints: nWanted, unique: false, time_limit: 10 });
    commit();
    E.cps = res.puzzle.checkpoints.map((v) => lastBuild.nodes[v]);
    E.insertAt = -1;
    changed();
    E.solution = res.solution;
    render();
    status(`Placed ${E.cps.length} numbers along a random path. “Make unique” adds more until only one solution remains.`);
  } catch (e) { toast(e.message, "bad"); status("Ready."); }
  finally { btn.disabled = false; renderChecks(); }
}
async function makeUnique() {
  const btn = $("uniqueBtn");
  btn.disabled = true;
  status("Adding numbers until the solution is unique (up to 25 s)...");
  const before = E.cps.length;
  try {
    const res = await api("/api/editor/make_unique", { puzzle: lastBuild.puzzle, time_limit: 25 });
    commit();
    E.cps = res.puzzle.checkpoints.map((v) => lastBuild.nodes[v]);
    E.insertAt = -1;
    changed();
    const added = E.cps.length - before;
    status(res.unique ? `Unique now: added ${added} number${added === 1 ? "" : "s"}.` : `Added ${added} numbers but ran out of time before proving uniqueness; try again.`);
    if (res.unique) toast(`Unique! (+${added})`, "good");
  } catch (e) { toast(e.message, "bad"); status("Ready."); }
  finally { renderChecks(); }
}
async function save(asNew) {
  const L = lastLocal;
  if (L.n < 2 || L.numbers < 2) { toast("Add at least 2 cells and 2 numbers before saving", "bad"); return null; }
  let name = ($("pzName").value || "").trim();
  if (!name) { name = `My puzzle ${new Date().toLocaleDateString()}`; $("pzName").value = name; }
  try {
    const res = await api("/api/custom", { name, puzzle: lastBuild.puzzle, id: asNew ? null : E.id });
    E.id = res.id; E.name = name;
    E.savedSig = signature(lastBuild.puzzle);
    history.replaceState(null, "", `#id=${encodeURIComponent(res.id)}`);
    toast(`Saved “${name}”`, "good");
    changed();
    loadLibrary();
    return res.id;
  } catch (e) { toast(`Save failed: ${e.message}`, "bad"); return null; }
}
async function playIt(ai) {
  const dirty = !E.id || E.savedSig !== signature(lastBuild.puzzle) || (E.name || "") !== ($("pzName").value || "").trim();
  let id = E.id;
  if (dirty) id = await save(false);
  if (!id) return;
  if (!lastLocal.structOk || (E.analysis && E.analysis.solvable === false)) toast("Heads up: this puzzle looks unsolvable", "bad");
  location.href = `/#custom=${encodeURIComponent(id)}${ai ? "&ai=1" : ""}`;
}
async function loadLibrary() {
  const ul = $("library");
  try {
    const list = await api("/api/custom");
    if (!list.length) { ul.innerHTML = `<li class="ed-empty">Nothing saved yet.</li>`; return; }
    ul.innerHTML = "";
    for (const it of list) {
      const li = document.createElement("li");
      li.className = `ed-libitem${it.id === E.id ? " cur" : ""}`;
      const when = it.updated ? relTime(it.updated * 1000) : "";
      li.innerHTML = `<button type="button" class="ed-libload" title="Load into the editor"><b>${escapeHtml(it.name)}</b>
        <span class="muted">${it.num_nodes} cells · ${it.checkpoints} numbers${it.dim === 3 ? " · 3D" : it.dim === 4 ? " · 4D" : ""}${when ? " · " + when : ""}</span></button>
        <a class="ed-x" href="/#custom=${encodeURIComponent(it.id)}" title="Play">▶</a>
        <button type="button" class="ed-x del" title="Delete" aria-label="Delete ${escapeHtml(it.name)}">✕</button>`;
      li.querySelector(".ed-libload").onclick = () => loadCustom(it.id);
      li.querySelector(".del").onclick = async () => {
        if (!confirm(`Delete “${it.name}”? This cannot be undone.`)) return;
        try { await api(`/api/custom/${encodeURIComponent(it.id)}`, undefined, "DELETE"); if (E.id === it.id) { E.id = null; E.savedSig = null; history.replaceState(null, "", "#"); changed(); } toast("Deleted"); loadLibrary(); }
        catch (e) { toast(e.message, "bad"); }
      };
      ul.appendChild(li);
    }
  } catch (e) { ul.innerHTML = `<li class="ed-empty">Could not load the list: ${escapeHtml(e.message)}</li>`; }
}
function relTime(ms) {
  const s = (Date.now() - ms) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ms).toLocaleDateString();
}
async function loadCustom(id) {
  try {
    const res = await api(`/api/custom/${encodeURIComponent(id)}`);
    importPuzzle(res.puzzle, res.name);
    E.id = res.id;
    lastBuild = build();
    E.savedSig = signature(lastBuild.puzzle);
    history.replaceState(null, "", `#id=${encodeURIComponent(res.id)}`);
    changed();
    loadLibrary();
    status(`Loaded “${res.name}”.`);
  } catch (e) { toast(`Load failed: ${e.message}`, "bad"); }
}
function exportJson() {
  const pz = lastBuild.puzzle;
  const name = ($("pzName").value || "puzzle").trim();
  const blob = new Blob([JSON.stringify({ name, puzzle: pz }, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "puzzle"}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
async function importFile(file) {
  try {
    const d = JSON.parse(await file.text());
    importPuzzle(d, d.name || file.name.replace(/\.json$/i, ""));
    E.id = null; E.savedSig = null;
    changed();
    toast("Imported", "good");
  } catch (e) { toast(`Import failed: ${e.message}`, "bad"); }
}

// ------------------------------------------------------------------ 3D preview
function togglePreview(on) {
  previewOn = on ?? !previewOn;
  $("preview3d").hidden = !previewOn;
  $("preview3dBtn").classList.toggle("on", previewOn);
  $("stage").classList.toggle("with-preview", previewOn);
  if (!previewOn && preview) { preview.dispose(); preview = null; }
  render();
  if (previewOn) mountPreview();
}
function schedulePreview() { if (!previewOn) return; clearTimeout(previewTimer); previewTimer = setTimeout(mountPreview, 450); }
async function mountPreview() {
  if (!previewOn) return;
  const b = lastBuild;
  const box = $("preview3d");
  if (preview) { preview.dispose(); preview = null; }
  if (b.puzzle.coords.length < 2) return;
  try {
    const mod = await import("./view3d.js");
    const pz = b.puzzle;
    preview = await mod.createView3D(box, {
      coords: pz.coords.map((c) => (c.length === 2 ? [c[0], c[1], 0] : c)), edges: pz.edges, cps: pz.checkpoints, n: pz.coords.length,
      onClick: (v) => { const i = b.nodes[v]; const z = rcz(i)[2]; if (z !== E.layer) { E.layer = z; renderLayers(); render(); } },
      theme: () => ({ bg: cssVar("--panel-2"), cell: cssVar("--cell-edge"), edge: cssVar("--border"), legal: cssVar("--legal"), good: cssVar("--good"), bad: cssVar("--bad"), plate: cssVar("--cell") }),
    });
    const sol = $("showSol").checked ? (E.showAlt && E.alt ? E.alt : E.solution) : null;
    const n = pz.coords.length;
    preview.update({ path: sol || [], legal: [], colorAt: (i) => pathColor(n > 1 ? i / (n - 1) : 0) });
  } catch (e) {
    box.innerHTML = `<div class="fallback">3D preview unavailable (${escapeHtml(e.message)})</div>`;
  }
}

// ------------------------------------------------------------------ drafts
function saveDraft() {
  try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ s: snap(), id: E.id, name: $("pzName").value, savedSig: E.savedSig })); } catch { /* ignore */ }
}
function loadDraft() {
  try {
    const d = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    if (!d || !d.s) return false;
    restore(d.s);
    E.id = d.id || null; E.savedSig = d.savedSig || null;
    E.name = d.name || ""; $("pzName").value = E.name;
    return true;
  } catch { return false; }
}

// ------------------------------------------------------------------ wiring
function init() {
  renderTemplates();
  const board = $("board");
  board.addEventListener("pointerdown", onDown);
  board.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  board.addEventListener("pointerleave", () => { hover = null; $("hoverInfo").textContent = ""; renderFx(); });
  board.addEventListener("contextmenu", (e) => e.preventDefault());
  $("undoBtn").onclick = undo; $("redoBtn").onclick = redo;
  const dimChange = () => {
    const R = Number($("dimR").value) || 1, C = Number($("dimC").value) || 1, Z = Number($("dimZ").value) || 1;
    if (R === E.R && C === E.C && Z === E.Z) return;
    commit();
    const grewZ = Z > E.Z;
    resize(R, C, Z);
    if (grewZ) toast("New layer added: paint it, it links to the layer below automatically");
    changed();
  };
  for (const id of ["dimR", "dimC", "dimZ"]) $(id).addEventListener("change", dimChange);
  $("fillAll").onclick = () => { commit(); let any = false; for (let r = 0; r < E.R; r++) for (let c = 0; c < E.C; c++) any = setCell(idx(r, c, E.layer), 1) || any; if (!any) undoStack.pop(); changed(); };
  $("clearAll").onclick = () => { commit(); E.on.fill(0); E.walls.clear(); E.bridges = []; E.cps = []; E.pending = null; changed(); };
  $("trimBtn").onclick = () => { commit(); if (!trim()) { undoStack.pop(); toast("Nothing to trim"); } changed(); };
  $("shiftBtn").onclick = () => { commit(); resize(E.R + 1, E.C + 1, E.Z, 1, 1); changed(); };
  $("randBtn").onclick = randomTemplate;
  $("exportBtn").onclick = exportJson;
  $("importFile").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) importFile(f); e.target.value = ""; });
  $("showSol").addEventListener("change", () => { render(); schedulePreview(); });
  $("altBtn").onclick = () => { E.showAlt = !E.showAlt; $("showSol").checked = true; render(); renderChecks(); schedulePreview(); };
  $("autoCheck").addEventListener("change", () => { if ($("autoCheck").checked) scheduleAnalyze(); renderChecks(); });
  $("analyzeBtn").onclick = () => analyze(20, true);
  $("suggestBtn").onclick = suggest;
  $("uniqueBtn").onclick = makeUnique;
  $("clearCps").onclick = () => { if (!E.cps.length) return; commit(); E.cps = []; E.insertAt = -1; changed(); };
  $("insertAt").addEventListener("change", (e) => { E.insertAt = Number(e.target.value); renderFx(); });
  $("saveBtn").onclick = () => save(false);
  $("saveNewBtn").onclick = () => save(true);
  $("playBtn").onclick = () => playIt(false);
  $("aiBtn").onclick = () => playIt(true);
  $("refreshLib").onclick = loadLibrary;
  $("pzName").addEventListener("input", () => { const sig = signature(lastBuild.puzzle); $("savedInfo").textContent = E.id ? `${E.savedSig !== sig || (E.name || "") !== $("pzName").value ? "Unsaved changes · " : "Saved · "}id ${E.id}` : ""; saveDraft(); });
  $("preview3dBtn").onclick = () => togglePreview();
  $("edKeysBtn").onclick = () => $("keysModal").classList.add("show");
  $("keysClose").onclick = () => $("keysModal").classList.remove("show");
  $("keysModal").addEventListener("click", (e) => { if (e.target === $("keysModal")) $("keysModal").classList.remove("show"); });
  document.addEventListener("keydown", (e) => {
    // letter shortcuts still work from number inputs / checkboxes (they cannot take letters anyway)
    const inField = e.target.closest && e.target.closest("select, textarea, input:not([type=number]):not([type=checkbox]):not([type=range])");
    const inNumber = e.target.closest && e.target.closest("input[type=number], input[type=range]");
    if (inNumber && !/^[a-zA-Z?]$/.test(e.key) && !(e.ctrlKey || e.metaKey)) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && (e.key === "s" || e.key === "S")) { e.preventDefault(); save(false); return; }
    if (inField) return;
    if (e.target.closest && e.target.closest(".zv-host")) return;   // the 3D preview has its own keys
    if (mod && (e.key === "z" || e.key === "Z")) { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
    if (mod && (e.key === "y" || e.key === "Y")) { e.preventDefault(); redo(); return; }
    if (mod || e.altKey) return;
    const t = TOOLS.find((x) => x.key.toLowerCase() === e.key.toLowerCase());
    if (t) { setTool(t.id); e.preventDefault(); return; }
    if ((e.key === "]" || e.key === "PageUp") && E.layer < E.Z - 1) { E.layer++; renderLayers(); render(); e.preventDefault(); }
    else if ((e.key === "[" || e.key === "PageDown") && E.layer > 0) { E.layer--; renderLayers(); render(); e.preventDefault(); }
    else if (e.key === "Escape") { E.pending = null; drag = null; $("keysModal").classList.remove("show"); render(); }
    else if (e.key === "?") $("keysModal").classList.add("show");
  });
  window.addEventListener("resize", () => { clearTimeout(init._rt); init._rt = setTimeout(render, 80); });
  window.addEventListener("zip-theme", () => { render(); if (preview) preview.refreshTheme(); });
  window.addEventListener("hashchange", () => { const m = location.hash.match(/id=([a-z0-9-]+)/); if (m && m[1] !== E.id) loadCustom(m[1]); });

  const m = location.hash.match(/id=([a-z0-9-]+)/);
  if (m) {
    E.on.fill(0);
    changed();
    loadCustom(m[1]);
  } else if (loadDraft()) {
    changed();
    status("Restored your last draft.");
  } else {
    TEMPLATES[0].fn();
    undoStack.length = 0;
    changed();
  }
  loadLibrary();
}
init();

// automation hooks (tests / screenshots)
window.__editor = {
  E, build: () => build(), setTool, changed, commit, analyze, save, loadCustom, importPuzzle,
  cellCenter(r, c, z = E.layer) {
    const i = idx(r, c, z), [x, y] = cellXY(i);
    const pt = view.svg.createSVGPoint(); pt.x = x; pt.y = y;
    const p = pt.matrixTransform(view.svg.getScreenCTM());
    return { x: p.x, y: p.y };
  },
  borderPoint(r, c, side) {   // client coords of a point on a cell border: side = right | bottom
    const i = idx(r, c, E.layer), [x, y] = cellXY(i), h = view.cs / 2;
    const pt = view.svg.createSVGPoint(); pt.x = side === "right" ? x + h - 2 : x; pt.y = side === "bottom" ? y + h - 2 : y;
    const p = pt.matrixTransform(view.svg.getScreenCTM());
    return { x: p.x, y: p.y };
  },
  get local() { return lastLocal; },
  get analysis() { return E.analysis; },
};
