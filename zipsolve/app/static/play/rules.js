// Path rules + input for one Zip board (used by the casual game page and its tutorial).
// Owns the path; the Board (board.js) only draws. Callbacks:
//   onChange(info)  after every path mutation  info = {pushed: [v...], popped: n, cp: k|undefined}
//   onBad(v)        an illegal tap
//   onFull()        the path covers every cell (the caller checks / celebrates)
export class PathGame {
  constructor(board, cb = {}) {
    this.board = board;
    this.cb = cb;
    this.P = null;
    this.path = [];
    this.locked = false;      // no user input (animations, finished puzzles, countdowns)
  }

  setPuzzle(P) {
    this.P = P;
    this.reset();
  }
  reset() {
    const P = this.P;
    this.path = [P.cps[0]];
    this.visited = new Uint8Array(P.n);
    this.visited[P.cps[0]] = 1;
    this.rev = (this.rev || 0) + 1;
  }
  head() { return this.path[this.path.length - 1]; }
  full() { return this.path.length === this.P.n; }
  nextCp() { let k = 0; for (const v of this.path) if (this.P.cpIndex.has(v)) k++; return k; }
  isLegal(v) { return this.whyNot(v) === null; }
  // null if the head may step onto v, else why not ("visited", "far", "order", "early", "oneway", "locked")
  whyNot(v) {
    const P = this.P, hd = this.head();
    if (this.visited[v]) return "visited";
    if (!P.adjSet[hd].has(v)) return "far";
    if (P.blocked.size && P.blocked.has(`${hd}>${v}`)) return "oneway";
    const pre = P.prec.size ? P.prec.get(v) : null;
    if (pre && pre.some((a) => !this.visited[a])) return "locked";
    const k = P.cpIndex.get(v);
    if (k !== undefined && k !== this.nextCp()) return "order";
    if (v === P.cps[P.cps.length - 1] && this.path.length !== P.n - 1) return "early";
    return null;
  }
  // overpass cells are two nodes on one tile: pick the one the player means
  resolve(v) {
    const P = this.P;
    if (v == null || !P.twin.has(v)) return v;
    const w = P.twin.get(v), prev = this.path[this.path.length - 2];
    if (prev === v || prev === w) return prev;
    if (this.isLegal(v)) return v;
    if (this.isLegal(w)) return w;
    const iv = this.path.indexOf(v), iw = this.path.indexOf(w);
    if (iv >= 0 || iw >= 0) return iv > iw ? v : w;
    return P.adjSet[this.head()].has(w) ? w : v;
  }
  bad(v) {
    if (v == null) { this.cb.onBad && this.cb.onBad(this.head(), null, this.head()); return; }
    const why = this.whyNot(v);
    this.cb.onBad && this.cb.onBad(v, why, this.head());
  }
  legalMoves() { return this.full() ? [] : this.P.adj[this.head()].filter((v) => this.isLegal(v)); }

  // ---------------------------------------------------------------- mutations
  setPath(path) {
    this.path = path.slice();
    this.visited = new Uint8Array(this.P.n);
    for (const v of this.path) this.visited[v] = 1;
    this.rev++;
  }
  push(v, { silent = false } = {}) {
    if (!this.isLegal(v)) return false;
    this.path.push(v); this.visited[v] = 1; this.rev++;
    if (!silent) this._changed([v], 0);
    return true;
  }
  pushMany(vs) {
    const done = [];
    for (const v of vs) { if (!this.isLegal(v)) break; this.path.push(v); this.visited[v] = 1; done.push(v); }
    if (done.length) { this.rev++; this._changed(done, 0); }
  }
  truncate(len) {
    len = Math.max(1, len);
    if (len >= this.path.length) return;
    const n = this.path.length - len;
    for (const v of this.path.slice(len)) this.visited[v] = 0;
    this.path.length = len;
    this.rev++;
    this._changed([], n);
  }
  undo() { if (this.path.length > 1) this.truncate(this.path.length - 1); }
  _changed(pushed, popped) {
    const last = pushed[pushed.length - 1];
    const cp = pushed.map((v) => this.P.cpIndex.get(v)).filter((k) => k !== undefined && k > 0).pop();
    this.cb.onChange && this.cb.onChange({ pushed, popped, cp, last });
    if (this.full() && pushed.length) this.cb.onFull && this.cb.onFull();
  }

  // Route from the head to `target` through legal straight-ish moves (fast drags / clicks along a line).
  findRoute(target, maxLen) {
    const P = this.P, board = this.board, cpsLast = P.cps[P.cps.length - 1];
    if (!board.L || !board.L.cell || P.blocked.size || P.prec.size || P.twin.size) return null;
    const vis = this.visited.slice();
    let nxt = this.nextCp(), len = this.path.length;
    const tgt = board.L.cell[target];
    const same = (a, b) => board.samePanel(a, b);
    const legalFrom = (hd, v) => {
      if (vis[v] || !P.adjSet[hd].has(v)) return false;
      const k = P.cpIndex.get(v);
      if (k !== undefined && k !== nxt) return false;
      if (v === cpsLast && len !== P.n - 1) return false;
      return true;
    };
    const dist = (v) => { const [r, c] = board.L.cell[v]; return Math.abs(r - tgt[0]) + Math.abs(c - tgt[1]); };
    const out = [];
    const dfs = (hd, depth) => {
      if (hd === target) return true;
      if (depth === 0) return false;
      for (const w of P.adj[hd]) {
        if (!(same(w, target) && board.gridAdjacent(hd, w) && dist(w) < dist(hd) && legalFrom(hd, w))) continue;
        const k = P.cpIndex.get(w);
        vis[w] = 1; len++; if (k !== undefined) nxt++;
        out.push(w);
        if (dfs(w, depth - 1)) return true;
        out.pop(); vis[w] = 0; len--; if (k !== undefined) nxt--;
      }
      return false;
    };
    const hd = this.head();
    if (!same(hd, target)) return null;
    const d0 = dist(hd);
    if (d0 > maxLen) return null;
    return dfs(hd, d0) ? out : null;
  }

  // ---------------------------------------------------------------- pointer / keyboard input
  onDown(v) {
    if (this.locked || v == null || !this.P) return;
    v = this.resolve(v);
    const idx = this.path.indexOf(v);
    if (idx >= 0) { if (idx < this.path.length - 1) this.truncate(idx + 1); return; }
    if (this.isLegal(v)) { this.push(v); return; }
    const r = this.findRoute(v, Math.max(this.P.n, 12));
    if (r) {
      const [a, b] = this.board.L.cell[this.head()], [c, d] = this.board.L.cell[v];
      if (a === c || b === d) { this.pushMany(r); return; }
    }
    this.bad(v);
  }
  onDrag(v) {
    if (this.locked || v == null || !this.P) return;
    v = this.resolve(v);
    const idx = this.path.indexOf(v);
    if (idx >= 0) { if (idx < this.path.length - 1) this.truncate(idx + 1); return; }
    if (this.isLegal(v)) { this.push(v); return; }
    const r = this.findRoute(v, 4);
    if (r) { this.pushMany(r); return; }
    // tell the player about rules they can't see from the grid alone (one-way edges, locked doors)
    const why = this.P.adjSet[this.head()].has(v) ? this.whyNot(v) : null;
    if (why === "oneway" || why === "locked") this.bad(v);
  }
  moveDir(dr, dc) {
    if (this.locked) return;
    const board = this.board, hd = this.head(), prev = this.path[this.path.length - 2];
    const [hx, hy] = board.center(hd);
    let best = null, bestScore = 0.7;
    for (const w of this.P.adj[hd]) {
      if (!board.samePanel(hd, w)) continue;
      const kind = board.edgeKind ? board.edgeKind(hd, w) : null;
      if (kind === "portal") continue;
      const [x, y] = board.center(w);
      let dx = x - hx, dy = y - hy;
      if (kind === "wrap") { const [dr0, dc0] = board.wrapDir(hd, w); dx = dc0; dy = dr0; }
      const len = Math.hypot(dx, dy) || 1;
      const score = (dx * dc + dy * dr) / len;
      if (score > bestScore && (w === prev || this.isLegal(w))) { best = w; bestScore = score; }
    }
    if (best == null) { this.bad(null); return; }
    if (best === prev) this.undo(); else this.push(best);
  }
  moveAxis(axis, delta) {
    if (this.locked || this.P.dim <= axis) return;
    const hd = this.head(), prev = this.path[this.path.length - 2];
    const c = this.P.coords[hd].slice(); c[axis] += delta;
    const w = this.board.L.lookup.get(c.join(","));
    if (w === undefined || !this.P.adjSet[hd].has(w)) { this.cb.onBad && this.cb.onBad(hd); return; }
    if (w === prev) this.undo(); else if (!this.push(w)) this.cb.onBad && this.cb.onBad(hd);
  }
}

// Co-op: two lines at once (warm = path 0, cool = path 1). Vertex-disjoint, together they cover every cell;
// path i runs from its first to its last checkpoint through its own numbers in order. `active` is the pen:
// tapping either line (or a cell only the other pen may take) switches pens, so two players can share a screen.
export class CoopGame {
  constructor(board, cb = {}) {
    this.board = board; this.cb = cb; this.P = null;
    this.paths = [[], []]; this.active = 0; this.locked = false; this.rev = 0;
  }
  setPuzzle(P) { this.P = P; this.reset(); }
  reset() {
    const P = this.P, C = P.coop.cps;
    this.paths = [[C[0][0]], [C[1][0]]];
    this.visited = new Uint8Array(P.n);
    this.visited[C[0][0]] = 1; this.visited[C[1][0]] = 1;
    this.rev++;
  }
  get path() { return this.paths[this.active]; }
  get total() { return this.paths[0].length + this.paths[1].length; }
  head(i = this.active) { const p = this.paths[i]; return p[p.length - 1]; }
  cps(i) { return this.P.coop.cps[i]; }
  ended(i) { const c = this.cps(i); return this.head(i) === c[c.length - 1]; }
  full() { return this.total === this.P.n && this.ended(0) && this.ended(1); }
  nextCp(i = this.active) { const m = this.P.coop.cpIndex[i]; let k = 0; for (const v of this.paths[i]) if (m.has(v)) k++; return k; }
  whyNot(v, i = this.active) {
    const P = this.P, hd = this.head(i);
    if (this.visited[v]) return "visited";
    if (this.ended(i)) return "ended";
    if (!P.adjSet[hd].has(v)) return "far";
    if (P.blocked.size && P.blocked.has(`${hd}>${v}`)) return "oneway";
    const pre = P.prec.size ? P.prec.get(v) : null;
    if (pre && pre.some((a) => !this.visited[a])) return "locked";
    if (P.coop.cpIndex[1 - i].has(v)) return "theirs";
    const k = P.coop.cpIndex[i].get(v);
    if (k !== undefined && k !== this.nextCp(i)) return "order";
    const c = this.cps(i);
    if (v === c[c.length - 1] && this.ended(1 - i) && this.total !== P.n - 1) return "early";
    return null;
  }
  isLegal(v, i = this.active) { return this.whyNot(v, i) === null; }
  legalMoves(i = this.active) { return this.full() ? [] : this.P.adj[this.head(i)].filter((v) => this.isLegal(v, i)); }
  setActive(i) { if (i !== this.active) { this.active = i; this.cb.onPen && this.cb.onPen(i); } }
  setPaths(ps) {
    this.paths = ps.map((p) => p.slice());
    this.visited = new Uint8Array(this.P.n);
    for (const p of this.paths) for (const v of p) this.visited[v] = 1;
    this.rev++;
  }
  push(v, { silent = false, which = this.active } = {}) {
    if (!this.isLegal(v, which)) return false;
    this.paths[which].push(v); this.visited[v] = 1; this.rev++;
    if (!silent) this._changed([v], 0, which);
    return true;
  }
  truncate(len, which = this.active) {
    const p = this.paths[which];
    len = Math.max(1, len);
    if (len >= p.length) return;
    const n = p.length - len;
    for (const v of p.slice(len)) this.visited[v] = 0;
    p.length = len;
    this.rev++;
    this._changed([], n, which);
  }
  undo() { if (this.path.length > 1) this.truncate(this.path.length - 1); }
  _changed(pushed, popped, which) {
    const m = this.P.coop.cpIndex[which];
    const cp = pushed.map((v) => m.get(v)).filter((k) => k !== undefined && k > 0).pop();
    this.cb.onChange && this.cb.onChange({ pushed, popped, cp, which, last: pushed[pushed.length - 1] });
    if (this.full() && pushed.length) this.cb.onFull && this.cb.onFull();
  }
  owner(v) { return this.paths[0].includes(v) ? 0 : this.paths[1].includes(v) ? 1 : -1; }
  bad(v, i = this.active) { this.cb.onBad && this.cb.onBad(v == null ? this.head(i) : v, v == null ? null : this.whyNot(v, i), this.head(i)); }
  onDown(v) {
    if (this.locked || v == null || !this.P) return;
    const o = this.owner(v);
    if (o >= 0) {   // grab a line: switch pens and rewind that line to here
      this.setActive(o);
      const idx = this.paths[o].indexOf(v);
      if (idx < this.paths[o].length - 1) this.truncate(idx + 1, o); else this.cb.onChange && this.cb.onChange({ pushed: [], popped: 0, which: o });
      return;
    }
    if (this.isLegal(v)) { this.push(v); return; }
    if (this.isLegal(v, 1 - this.active)) { this.setActive(1 - this.active); this.push(v); return; }
    this.bad(v);
  }
  onDrag(v) {
    if (this.locked || v == null || !this.P) return;
    const p = this.path, idx = p.indexOf(v);
    if (idx >= 0) { if (idx < p.length - 1) this.truncate(idx + 1); return; }
    if (this.isLegal(v)) { this.push(v); return; }
    const why = this.P.adjSet[this.head()].has(v) ? this.whyNot(v) : null;
    if (why === "oneway" || why === "locked") this.bad(v);
  }
  moveDir(dr, dc) {
    if (this.locked) return;
    const board = this.board, hd = this.head(), prev = this.path[this.path.length - 2];
    const [hx, hy] = board.center(hd);
    let best = null, bestScore = 0.7;
    for (const w of this.P.adj[hd]) {
      const [x, y] = board.center(w);
      const dx = x - hx, dy = y - hy, len = Math.hypot(dx, dy) || 1, score = (dx * dc + dy * dr) / len;
      if (score > bestScore && (w === prev || this.isLegal(w))) { best = w; bestScore = score; }
    }
    if (best == null) { this.bad(null); return; }
    if (best === prev) this.undo(); else this.push(best);
  }
  moveAxis() { this.bad(null); }
}
