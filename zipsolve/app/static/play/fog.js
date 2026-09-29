// Fair play in the fog: a JavaScript port of zipsolve/robots/fog.py for the static (GitHub Pages) site, where there
// is no Python server. The robot knows what a player sees on a fog board: where the numbered cells are (the "?"
// clouds), how many there are, the 1, and every number that has come within 2 steps of the pen (board.js rule).
// It plans a line consistent with that (revealed number k = the k-th checkpoint met, hidden ones fill the other
// slots, the line ends on the largest number), follows it, and backs up when a newly seen number spoils the plan.
//   fogAdvice(puzzle, path, seen, ms)  -> /api/hint shape  {status, next, keep, reason, certain, fog: true, ...}
//   fogExplain(puzzle, path, seen, ms) -> /api/hint/explain shape
//   fogRun(puzzle, startPath, seen, ms) -> robot run {path, solved, trace (+ {"t":"reveal"}), stats, fog: true}
// Recorded runs from the bank (fog/<id>.json) are used for fresh starts; this is for lines the player has begun.

function graphOf(d) {
  const n = d.coords.length, adj = Array.from({ length: n }, () => []);
  for (const [u, v] of d.edges) { if (!adj[u].includes(v)) adj[u].push(v); if (!adj[v].includes(u)) adj[v].push(u); }
  adj.forEach((a) => a.sort((x, y) => x - y));
  const blocked = new Map();                      // v -> nodes it may not step to
  for (const [u, v] of d.arcs || (d.meta && d.meta.oneway) || []) { if (!blocked.has(v)) blocked.set(v, new Set()); blocked.get(v).add(u); }
  const pre = new Map();
  const prec = d.precedence || ((d.meta && d.meta.keys) || []).map((k) => [k.key, k.door]);
  for (const [a, b] of prec) { if (!pre.has(b)) pre.set(b, []); pre.get(b).push(a); }
  return { n, adj, blocked, pre, start: d.checkpoints[0] };
}
export function near(adj, v, k = 2) {
  const seen = new Set([v]);
  let front = [v];
  for (let i = 0; i < k; i++) {
    const nx = [];
    for (const u of front) for (const w of adj[u]) if (!seen.has(w)) { seen.add(w); nx.push(w); }
    front = nx;
  }
  return seen;
}
function rng(seed) {   // mulberry32
  let a = (seed >>> 0) + 0x6d2b79f5;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

class View {
  constructor(d, G) {
    const cps = d.checkpoints;
    this.G = G; this.cells = new Set(cps); this.count = cps.length; this.start = cps[0];
    this._label = new Map(cps.map((c, i) => [c, i]));    // private: only look()/show() read it
    this.revealed = new Map([[this.start, 0]]);
  }
  look(head) {
    const out = [];
    for (const c of near(this.G.adj, head)) if (this.cells.has(c) && !this.revealed.has(c)) { this.revealed.set(c, this._label.get(c)); out.push(c); }
    return out.sort((a, b) => a - b);
  }
  show(cells) { for (const c of cells || []) if (this.cells.has(c) && !this.revealed.has(c)) this.revealed.set(c, this._label.get(c)); }
  know() { return new Know(this.cells, this.count, new Map(this.revealed)); }
}
class Know {
  constructor(cells, K, lab) {
    this.cells = cells; this.K = K; this.lab = lab;
    this.slot = new Map([...lab].map(([c, s]) => [s, c]));
    this.end = this.slot.has(K - 1) ? this.slot.get(K - 1) : null;
  }
  endCand(u) { return this.cells.has(u) && (this.end !== null ? u === this.end : !this.lab.has(u)); }
  consistent(path, n) {
    let j = 0;
    for (let i = 0; i < path.length; i++) {
      const v = path[i];
      if (!this.cells.has(v)) continue;
      const s = this.lab.get(v);
      if (s === undefined) { if (this.slot.has(j)) return false; } else if (s !== j) return false;
      if (j === this.K - 1 && i !== n - 1) return false;
      j++;
    }
    return !(path.length === n && (j !== this.K || !this.cells.has(path[n - 1])));
  }
}

class Budget extends Error {}
class Planner {
  constructor(G, seed = 0) {
    this.G = G; this.n = G.n; this.rnd = rng(seed * 7919 + 17); this.dead = new Set(); this.expanded = 0; this.dists = new Map();
    // bipartite colouring (parity pruning)
    const col = new Int8Array(G.n).fill(-1);
    let ok = true;
    for (let s = 0; s < G.n && ok; s++) {
      if (col[s] >= 0) continue;
      col[s] = 0; const q = [s];
      for (let i = 0; i < q.length && ok; i++) for (const w of G.adj[q[i]]) { if (col[w] < 0) { col[w] = 1 - col[q[i]]; q.push(w); } else if (col[w] === col[q[i]]) ok = false; }
    }
    this.col = ok ? col : null;
    // on a bipartite board the line alternates colours: its last cell has a known colour
    this.endCol = this.col ? (G.n % 2 === 1 ? this.col[G.start] : 1 - this.col[G.start]) : null;
  }
  endOk(know, u) { return know.endCand(u) && (this.endCol === null || this.col[u] === this.endCol); }
  dist(src) {
    if (!this.dists.has(src)) {
      const d = new Array(this.n).fill(1e9); d[src] = 0; const q = [src];
      for (let i = 0; i < q.length; i++) for (const w of this.G.adj[q[i]]) if (d[w] > d[q[i]] + 1) { d[w] = d[q[i]] + 1; q.push(w); }
      this.dists.set(src, d);
    }
    return this.dists.get(src);
  }
  state(know, path) { return new State(this, know, path); }
  // "found" + full path | "dead" (proof) | "unknown" (budget / time)
  complete(path, know, budget = 20000, deadline = Infinity) {
    const n = this.n;
    if (!know.consistent(path, n)) return ["dead", null];
    if (path.length === n) return ["found", path.slice()];
    const st = this.state(know, path);
    if (!st.feasible()) return ["dead", null];
    this.left = budget; this.deadline = deadline;
    try { return this.dfs(st) ? ["found", st.path.slice()] : ["dead", null]; }
    catch (e) { if (e instanceof Budget) return ["unknown", null]; throw e; }
  }
  dfs(st) {
    if (st.path.length === this.n) return true;
    const key = st.key();
    if (this.dead.has(key)) return false;
    this.expanded++;
    if (--this.left < 0 || ((this.left & 127) === 0 && performance.now() > this.deadline)) throw new Budget();
    for (const w of st.moves()) {
      st.push(w);
      if (st.feasible() && this.dfs(st)) return true;
      st.pop();
    }
    this.dead.add(key);
    return false;
  }
}
class State {
  constructor(pl, know, path) {
    this.pl = pl; this.k = know; this.n = pl.n; this.path = path.slice();
    this.vis = new Uint8Array(this.n); this.j = 0; this.cnt = [0, 0];
    for (const v of this.path) { this.vis[v] = 1; if (know.cells.has(v)) this.j++; }
    if (pl.col) for (let v = 0; v < this.n; v++) if (!this.vis[v]) this.cnt[pl.col[v]]++;
  }
  key() { let s = ""; for (let i = 0; i < this.n; i += 30) { let x = 0; for (let b = 0; b < 30 && i + b < this.n; b++) if (this.vis[i + b]) x |= 1 << b; s += x.toString(36) + "."; } return s + this.path[this.path.length - 1]; }
  allowed(w) {
    const k = this.k, h = this.path[this.path.length - 1], G = this.pl.G;
    if (this.vis[w]) return false;
    const bl = G.blocked.get(h); if (bl && bl.has(w)) return false;
    for (const a of G.pre.get(w) || []) if (!this.vis[a]) return false;
    if (k.cells.has(w)) {
      const s = k.lab.get(w);
      if (s === undefined) { if (k.slot.has(this.j)) return false; } else if (s !== this.j) return false;
      if (this.j === k.K - 1 && this.path.length + 1 !== this.n) return false;
    }
    return true;
  }
  push(w) { this.path.push(w); this.vis[w] = 1; if (this.k.cells.has(w)) this.j++; if (this.pl.col) this.cnt[this.pl.col[w]]--; }
  pop() { const w = this.path.pop(); this.vis[w] = 0; if (this.k.cells.has(w)) this.j--; if (this.pl.col) this.cnt[this.pl.col[w]]++; }
  moves() {
    const adj = this.pl.G.adj, h = this.path[this.path.length - 1], tgt = this.k.slot.get(this.j);
    const dist = tgt !== undefined ? this.pl.dist(tgt) : null, out = [];
    for (const w of adj[h]) {
      if (!this.allowed(w)) continue;
      let on = 0; for (const x of adj[w]) if (!this.vis[x]) on++;
      out.push([on, dist ? dist[w] : 0, this.pl.rnd(), w]);
    }
    out.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    return out.map((x) => x[3]);
  }
  feasible() {
    const n = this.n, r = n - this.path.length;
    if (r === 0) return true;
    const h = this.path[this.path.length - 1], col = this.pl.col, adj = this.pl.G.adj, vis = this.vis;
    if (col) { const x = 1 - col[h], need = (r + 1) >> 1; if (this.cnt[x] !== need || this.cnt[1 - x] !== r - need) return false; }
    let start = -1;
    for (const w of adj[h]) if (!vis[w]) { start = w; break; }
    if (start < 0) return false;
    const hn = new Set(adj[h]), seen = new Uint8Array(n), q = [start];
    seen[start] = 1;
    let ends = 0;
    for (let i = 0; i < q.length; i++) {
      const u = q[i];
      let d = 0;
      for (const w of adj[u]) if (!vis[w]) { d++; if (!seen[w]) { seen[w] = 1; q.push(w); } }
      if (d + (hn.has(u) ? 1 : 0) < 2) {
        if (d === 0 && r > 1) return false;
        if (!this.pl.endOk(this.k, u)) return false;
        if (++ends > 1) return false;
      }
    }
    return q.length === r;
  }
}

// ------------------------------------------------------------------ words
function coordStr(d, v) { const c = d.coords[v]; return `(${c.map((x) => (d.meta && d.meta.layout === "hex") || c.some((y) => y < 0) ? x : x + 1).join(",")})`; }
function dirWord(d, a, b) {
  const A = d.coords[a], B = d.coords[b];
  if (A.length === 2 && (!d.meta || !d.meta.layout || d.meta.layout === "square")) {
    const dr = B[0] - A[0], dc = B[1] - A[1];
    if (Math.abs(dr) + Math.abs(dc) === 1) return dr > 0 ? "down" : dr < 0 ? "up" : dc > 0 ? "right" : "left";
  }
  return `to ${coordStr(d, b)}`;
}
function cellWord(d, know, v) {
  const s = know.lab.get(v);
  if (s !== undefined) return `the ${s + 1}`;
  return know.cells.has(v) ? `a hidden number at ${coordStr(d, v)}` : coordStr(d, v);
}
function labelsWord(know, cells) {
  const w = cells.map((c) => know.lab.get(c) + 1).sort((a, b) => a - b).map(String);
  const art = /^8/.test(w[0]) || w[0] === "11" || w[0] === "18" ? "An" : "A";
  return w.length === 1 ? `${art} ${w[0]}` : `${art} ${w.slice(0, -1).join(", ")} and ${w[w.length - 1]}`;
}

// ------------------------------------------------------------------ hints
function viewFor(d, G, path, seen) {
  const view = new View(d, G);
  if (seen) { view.show(seen); if (path.length) view.look(path[path.length - 1]); }
  else for (const v of path) view.look(v);
  return view;
}
function advice(d, path, seen, ms) {
  const G = graphOf(d), n = G.n, tEnd = performance.now() + ms;
  const know = viewFor(d, G, path, seen).know(), pl = new Planner(G, 0);
  const search = (p, frac = 0.3) => pl.complete(p, know, 40000, performance.now() + Math.max(30, (tEnd - performance.now()) * frac));
  const [st, full] = search(path);
  if (st === "dead") {
    let lo = 1, hi = path.length - 1, best = search(path.slice(0, 1))[1];
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1, [s, f] = search(path.slice(0, mid));
      if (s === "dead") hi = mid - 1; else { lo = mid; if (s === "found") best = f; }
    }
    if (!best || best.slice(0, lo).some((v, i) => v !== path[i])) best = search(path.slice(0, lo), 0.9)[1];
    let mv = best && best.length > lo ? best[lo] : null;
    if (mv === null) mv = pl.state(know, path.slice(0, lo)).moves()[0] ?? null;
    return { st: "backtrack", keep: lo, mv, certain: true, cells: mv === null ? [] : [mv],
      reason: `Based on what you can see, your line can't work from here: back up to step ${lo}${mv !== null ? ` and go ${dirWord(d, path[lo - 1], mv)}.` : "."}` };
  }
  const moves = pl.state(know, path).moves();
  let possible = [];
  for (const m of moves) { const [s] = search([...path, m], 0.5 / Math.max(1, moves.length)); if (s !== "dead") possible.push(m); }
  if (!possible.length) { const mv = full ? full[path.length] : moves[0]; if (mv !== undefined) possible = [mv]; }
  if (!possible.length) return { st: "timeout" };
  if (possible.length === 1) {
    const m = possible[0];
    return { st: "next", keep: path.length, mv: m, certain: true, cells: [m],
      reason: `Based on what you can see, going ${dirWord(d, path[path.length - 1], m)} to ${cellWord(d, know, m)} is the only move that can still work.` };
  }
  const hidden = [...know.cells].filter((c) => !know.lab.has(c));
  let best = possible[0];
  if (hidden.length) {
    const fd = (m) => Math.min(...hidden.map((c) => pl.dist(c)[m]));
    best = possible.slice().sort((a, b) => fd(a) - fd(b) || possible.indexOf(a) - possible.indexOf(b))[0];
  }
  return { st: "next", keep: path.length, mv: best, certain: false, cells: [best],
    reason: `Not enough numbers visible yet — explore toward the fog. ${possible.length} moves could still work; going ${dirWord(d, path[path.length - 1], best)} gets you closer to the hidden numbers.` };
}
export function fogAdvice(d, path, seen = null, ms = 1500) {
  const t0 = performance.now();
  path = path && path.length ? path : [d.checkpoints[0]];
  if (path.length === d.coords.length) return { status: "done", message: "Already solved!", fog: true };
  const a = advice(d, path, seen, ms);
  if (a.st === "timeout" || a.mv === null) return { status: "timeout", message: "No hint right now.", fog: true };
  const out = { status: a.st, next: a.mv, keep: a.keep, source: "fog", reason: a.reason, certain: a.certain, fog: true, nodes_expanded: 0, seconds: (performance.now() - t0) / 1000 };
  if (a.st === "backtrack") out.message = a.reason;
  return out;
}
export function fogExplain(d, path, seen = null, ms = 1500) {
  path = path && path.length ? path : [d.checkpoints[0]];
  if (path.length === d.coords.length) return { status: "done", move: null, reason: "Already solved.", fog: true };
  const a = advice(d, path, seen, ms);
  if (a.st === "timeout") return { status: "timeout", move: null, reason: "No hint right now.", fog: true };
  const out = { status: a.st, move: a.mv, reason: a.reason, technique: "fog", technique_name: a.certain ? "What you can see" : "Explore the fog", cells: a.cells, certain: a.certain, fog: true, seconds: 0 };
  if (a.st === "backtrack") out.keep = a.keep;
  return out;
}

// ------------------------------------------------------------------ a whole run (from the player's line)
export function fogRun(d, startPath = null, seen = null, ms = 6000, solution = null) {
  const t0 = performance.now(), deadline = t0 + ms;
  const G = graphOf(d), n = G.n, view = new View(d, G), pl = new Planner(G, 0);
  let path = startPath && startPath.length ? startPath.slice() : [d.checkpoints[0]];
  const trace = ["R", ...path];
  const first = new Set();
  view.show(seen || []);
  for (const c of seen || []) if (view.cells.has(c)) first.add(c);
  for (const v of path) for (const c of view.look(v)) first.add(c);
  if (first.size) trace.push({ t: "reveal", nodes: [...first].sort((a, b) => a - b) });
  let know = view.know();
  const stats = { surprises: 0, replans: 0, backtracks: 0, fallback: false };
  trace.push({ t: "note", kind: "intro", msg: `Fog! I can see ${know.lab.size} of the ${view.count} numbers, so I'll plan with those and re-think when a surprise shows up.` });
  const search = (p) => {
    const left = deadline - performance.now();
    return pl.complete(p, know, n <= 64 ? 8000 : 4000, performance.now() + Math.max(30, Math.min(left * 0.3, 1500)));
  };
  const popTo = (len) => { const k = path.length - len; if (k > 0) { trace.push(-k); path = path.slice(0, len); } };
  let plan = null, steps = 0, exploring = -1, ok = true;
  while (path.length < n) {
    if (++steps > 30 * n + 300 || performance.now() > deadline) { ok = false; break; }
    if (!plan || plan.slice(0, path.length).some((v, i) => v !== path[i]) || !know.consistent(plan, n)) {
      let [st, full] = search(path), len = path.length;
      while (st === "dead" && len > 1) { len--; [st, full] = search(path.slice(0, len)); }
      if (st === "dead") { ok = false; break; }
      if (len < path.length) { stats.backtracks++; popTo(len); }
      plan = st === "found" ? full : null;
      if (!plan) {
        const s = pl.state(know, path);
        const mv = s.moves().find((w) => { s.push(w); const f = s.feasible(); s.pop(); return f; });
        if (mv === undefined) { if (path.length <= 1) { ok = false; break; } popTo(path.length - 1); stats.backtracks++; continue; }
        plan = [...path, mv];   // one step of "unknown": re-plan after it
      }
    }
    const v = plan[path.length];
    const slot = path.filter((x) => know.cells.has(x)).length;
    if (!know.slot.has(slot) && exploring !== slot) { exploring = slot; trace.push({ t: "note", kind: "explore", msg: `No ${slot + 1} in sight yet: heading for unexplored cells.` }); }
    path.push(v); trace.push(v);
    if (plan.length === path.length && path.length < n) plan = null;
    const nw = view.look(v);
    if (!nw.length) continue;
    trace.push({ t: "reveal", nodes: nw });
    know = view.know();
    if (plan && !know.consistent(plan, n)) {
      stats.surprises++;
      let [st, full] = search(path), len = path.length;
      while (st === "dead" && len > 1) { len--; [st, full] = search(path.slice(0, len)); }
      const what = labelsWord(know, nw);
      if (len === path.length) { stats.replans++; trace.push({ t: "note", kind: "replan", cells: nw, msg: `${what} just appeared — that changes things. New plan from here.` }); }
      else { const k = path.length - len; stats.backtracks++; trace.push({ t: "note", kind: "backtrack", cells: nw, msg: `${what} just appeared — that changes things, backing up ${k} step${k !== 1 ? "s" : ""}.` }); popTo(len); }
      plan = st === "found" ? full : null;
    } else {
      const cur = path.filter((x) => know.cells.has(x)).length, tgt = know.slot.get(cur);
      if (tgt !== undefined && nw.includes(tgt)) { exploring = -1; trace.push({ t: "note", kind: "spot", cells: [tgt], msg: `There's the ${cur + 1} — heading for it.` }); }
    }
  }
  if (!ok && solution) {   // out of patience: finish along the known solution (flagged)
    stats.fallback = true;
    let k = 0; while (k < path.length && path[k] === solution[k]) k++;
    popTo(Math.max(1, k));
    trace.push({ t: "note", kind: "fallback", msg: "That took too long in the fog. Peeking at the whole map to finish (fallback)." });
    for (const v of solution.slice(path.length)) { path.push(v); trace.push(v); const nw = view.look(v); if (nw.length) trace.push({ t: "reveal", nodes: nw }); }
  }
  const solved = path.length === n && view.know().consistent(path, n);
  if (solved) trace.push({ t: "note", kind: "done", msg: `Solved in the fog: ${stats.surprises} surprise${stats.surprises !== 1 ? "s" : ""}, backed up ${stats.backtracks} time${stats.backtracks !== 1 ? "s" : ""}.` });
  return { robot: "fog", status: solved ? "solved" : "stuck", solved, path, start_len: startPath && startPath.length ? startPath.length : 1,
    steps: path.slice(1).map((node) => ({ node, p: null })), nodes_expanded: pl.expanded, seconds: (performance.now() - t0) / 1000,
    trace, trace_truncated: false, backtracks: stats.backtracks, stats, fog: true };
}
