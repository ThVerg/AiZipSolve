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
  isLegal(v) {
    const P = this.P;
    if (this.visited[v]) return false;
    if (!P.adjSet[this.head()].has(v)) return false;
    const k = P.cpIndex.get(v);
    if (k !== undefined && k !== this.nextCp()) return false;
    if (v === P.cps[P.cps.length - 1] && this.path.length !== P.n - 1) return false;
    return true;
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
    const idx = this.path.indexOf(v);
    if (idx >= 0) { if (idx < this.path.length - 1) this.truncate(idx + 1); return; }
    if (this.isLegal(v)) { this.push(v); return; }
    const r = this.findRoute(v, Math.max(this.P.n, 12));
    const [a, b] = this.board.L.cell[this.head()], [c, d] = this.board.L.cell[v];
    if (r && (a === c || b === d)) { this.pushMany(r); return; }
    this.cb.onBad && this.cb.onBad(v);
  }
  onDrag(v) {
    if (this.locked || v == null || !this.P) return;
    const idx = this.path.indexOf(v);
    if (idx >= 0) { if (idx < this.path.length - 1) this.truncate(idx + 1); return; }
    if (this.isLegal(v)) { this.push(v); return; }
    const r = this.findRoute(v, 4);
    if (r) this.pushMany(r);
  }
  moveDir(dr, dc) {
    if (this.locked) return;
    const board = this.board, hd = this.head(), prev = this.path[this.path.length - 2];
    const [hx, hy] = board.center(hd);
    let best = null, bestScore = 0.7;
    for (const w of this.P.adj[hd]) {
      if (!board.samePanel(hd, w)) continue;
      const [x, y] = board.center(w);
      const dx = x - hx, dy = y - hy, len = Math.hypot(dx, dy) || 1;
      const score = (dx * dc + dy * dr) / len;
      if (score > bestScore && (w === prev || this.isLegal(w))) { best = w; bestScore = score; }
    }
    if (best == null) { this.cb.onBad && this.cb.onBad(hd); return; }
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
