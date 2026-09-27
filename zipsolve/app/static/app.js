// Zip play page: puzzle sources (daily / difficulty presets / custom / editor), drawing, win flow, records.
// AI features live in play/ai.js, the SVG board in play/board.js.
import { $, h, api, fmtTime, fmtSec, fmtInt, cssVar, store, icon, esc, localDateISO, addDays, clamp, reducedMotion } from "./play/util.js";
import { Board, makeModel } from "./play/board.js";
import { sound } from "./play/sound.js";
import { initAI } from "./play/ai.js";

const KIND_INFO = {
  grid2d: { label: "2D grid", size: [2, 12, 7], sizeLbl: "Side" },
  walls: { label: "Walls", size: [3, 12, 7], sizeLbl: "Side" },
  islands: { label: "Islands", size: [2, 8, 3], sizeLbl: "Islands" },
  mask: { label: "Irregular shape", size: [4, 12, 8], sizeLbl: "Box side" },
  grid3d: { label: "3D cube", size: [2, 6, 4], sizeLbl: "Side" },
  grid4d: { label: "4D hypercube", size: [2, 4, 3], sizeLbl: "Side" },
  grid: { label: "Grid", size: [2, 12, 6], sizeLbl: "Side" },
};
const FALLBACK_PRESETS = [
  { id: "easy", label: "Easy", kind: "grid2d", size: 5, unique: true, options: {}, blurb: "5×5 grid" },
  { id: "medium", label: "Medium", kind: "grid2d", size: 7, unique: true, options: {}, blurb: "7×7 grid" },
  { id: "hard", label: "Hard", kind: "walls", size: 8, unique: true, options: { walls_frac: 0.35 }, blurb: "8×8 with walls" },
  { id: "expert", label: "Expert", kind: "islands", size: 4, unique: true, options: {}, blurb: "4 islands + bridges" },
  { id: "insane", label: "Insane", kind: "grid3d", size: 4, unique: false, options: {}, blurb: "4×4×4 cube" },
];

// ------------------------------------------------------------------ state
const S = {
  data: null, P: null,  // puzzle dict from the server / derived model
  id: null, seed: null, kind: null,
  n: 0, dim: 2, cps: [], cpIndex: new Map(), adj: [], adjSet: [], coords: [],
  source: null,         // {type: daily|preset|custom|editor, difficulty?, date?, number?, name?}
  path: [], visited: null,
  rev: 0,               // path revision: bumped on every path mutation and on every new puzzle
  owner: "user",        // who drew the path: user | agent | exact
  undos: 0, hints: 0, t0: null, tEnd: null, won: false, assisted: false,
  hint: null, agent: null, exact: null, conf: new Map(), curStep: null, stuck: null,
  anim: null,           // {cancel(finish)} while a path / trace is being animated
  busy: false,
  tried: null,          // Float32Array: search-trace fading overlay
  presets: FALLBACK_PRESETS, difficulty: "medium",
  animateNext: true,
};
let board = null, view3d = null, view4d = null, viewMode = "panels";
let AI = null;

// ------------------------------------------------------------------ helpers
function bumpRev() { S.rev++; }
function pathToken() { return { data: S.data, id: S.id, rev: S.rev }; }
function tokenValid(t) { return t.data === S.data && t.id === S.id && t.rev === S.rev; }
function colorAt(i) { return board ? board.colorAt(i) : "#888"; }
function presetById(id) { return S.presets.find((p) => p.id === id); }
function head() { return S.path[S.path.length - 1]; }

function banner(msg, tone = "", actions = []) {
  const b = $("banner");
  if (!msg) { b.className = "banner"; $("bannerMsg").innerHTML = ""; $("bannerActions").innerHTML = ""; return; }
  b.className = `banner show ${tone}`;
  $("bannerMsg").innerHTML = msg;
  const box = $("bannerActions");
  box.innerHTML = "";
  for (const [label, fn, cls] of actions) box.appendChild(h("button", { class: `btn sm ${cls || ""}`, type: "button", onclick: fn, html: label }));
}
let toastT = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2200);
}
function setBusy(on, msg = "Working…", stoppable = false) {
  S.busy = on;
  $("busy").classList.toggle("show", on);
  $("busyMsg").textContent = msg;
  $("stopBtn").hidden = !stoppable;
  for (const id of ["newBtn", "hintBtn", "runBtn", "raceBtn", "customBtn", "cmpBtn"]) if ($(id)) $(id).disabled = on;
  if (!on && AI) AI.syncButtons();
}

// ------------------------------------------------------------------ puzzle loading
function loadPuzzle(res, source) {
  cancelAnim();
  if (AI) AI.onPuzzleWillChange();
  const d = res.puzzle;
  S.data = d; S.id = res.id || null; S.seed = res.seed ?? null; S.kind = res.kind || d.kind || "grid";
  S.P = makeModel(d);
  Object.assign(S, { n: S.P.n, dim: S.P.dim, cps: S.P.cps, cpIndex: S.P.cpIndex, adj: S.P.adj, adjSet: S.P.adjSet, coords: S.P.coords });
  S.source = source;
  S.agent = null; S.exact = null; S.hints = 0;
  board.setPuzzle(S.P);
  resetPath(true);
  // titles
  const kindLbl = KIND_INFO[S.kind]?.label || S.kind;
  const shape = d.meta && d.meta.shape ? d.meta.shape.join("×") : "";
  let kicker = "", title = "";
  if (source.type === "daily") { kicker = `Daily #${source.number} · ${source.label}`; title = `${kindLbl}${shape ? " " + shape : ""}`; }
  else if (source.type === "preset") { const p = presetById(source.difficulty); kicker = p ? p.label : "Puzzle"; title = `${kindLbl}${shape ? " " + shape : ""}`; }
  else if (source.type === "editor") { kicker = "From the editor"; title = source.name || "Custom puzzle"; }
  else { kicker = "Custom"; title = `${kindLbl}${shape ? " " + shape : ""}`; }
  $("boardKicker").textContent = kicker;
  $("boardTitle").textContent = title;
  $("boardSub").textContent = `${S.n} cells · ${S.cps.length} checkpoints` + (res.seed != null ? ` · seed ${res.seed}` : "") +
    (res.unique ? " · unique solution" : "") + (res.seconds != null ? ` · made in ${fmtSec(res.seconds)}` : "");
  if (res.seed != null) $("seed").placeholder = `random (last: ${res.seed})`;
  $("showLegal").checked = store.get("zip-showLegal", null) ?? S.dim >= 3;
  if (S.dim >= 3 && store.get("zip-showLegal", null) === null) $("showLegal").checked = true;
  highlightChips();
  AI.onPuzzleLoaded();   // before the layout: it may show the AI-vision strip
  layoutBoard();
  setupViews();
  renderLegend();
  render();
  document.title = `Zip · ${kicker || "Play"}`;
}

function resetPath(full = false) {
  cancelAnim();
  bumpRev();
  S.path = [S.cps[0]];
  S.visited = new Uint8Array(S.n); S.visited[S.cps[0]] = 1;
  S.owner = "user"; S.hint = null; S.conf = new Map(); S.curStep = null; S.stuck = null; S.tried = null;
  S.undos = 0; S.t0 = null; S.tEnd = null; S.won = false;
  if (full) { S.hints = 0; S.assisted = false; }
  S.animateNext = false;
  banner("");
  if (AI) AI.updateConfPanel();
}

function layoutBoard() {
  if (!S.P) return;
  const host = $("board");
  const views = $("views");
  const w3 = S.dim === 3, race = !$("raceSide").hidden;
  views.classList.toggle("with3d", w3);
  $("view3d").hidden = !(w3 || (S.dim === 4 && viewMode === "4d"));
  host.hidden = S.dim === 4 && viewMode === "4d";
  $("arena").classList.toggle("racing", race);
  const avW = Math.max(220, host.clientWidth || views.clientWidth || 600);
  const top = host.getBoundingClientRect().top + window.scrollY;
  const mobile = window.innerWidth <= 720;
  let avH = Math.max(mobile ? 260 : 320, window.innerHeight - Math.max(90, Math.min(top, 260)) - 110);
  if (mobile) avH = Math.max(avH, avW);
  if (w3 && window.innerWidth <= 1250) avH = Math.max(300, avH - 200);
  if (!mobile && !$("insight").hidden) avH = Math.max(300, avH - 118);   // keep the AI-vision strip on screen
  board.build(avW, avH);
  if (AI) AI.layoutMini();
}

// ------------------------------------------------------------------ rules
function nextCp() { let k = 0; for (const v of S.path) if (S.cpIndex.has(v)) k++; return k; }
function isLegal(v) {
  if (S.visited[v]) return false;
  if (!S.adjSet[head()].has(v)) return false;
  const k = S.cpIndex.get(v);
  if (k !== undefined && k !== nextCp()) return false;
  if (v === S.cps[S.cps.length - 1] && S.path.length !== S.n - 1) return false;
  return true;
}
function legalMoves() {
  if (S.won || S.path.length >= S.n) return [];
  return S.adj[head()].filter(isLegal);
}

// ------------------------------------------------------------------ render
function viewState() {
  const ai = AI ? AI.overlay() : {};
  return {
    path: S.path, won: S.won, legal: legalMoves(), showLegal: $("showLegal").checked, conf: S.conf, curStep: S.curStep,
    anim: !!S.anim, hint: S.hint, stuck: S.stuck, heat: ai.heat || null, policy: ai.policy || null, tried: S.tried,
    animate: S.animateNext && !reducedMotion(),
  };
}
function render() {
  if (!board || !S.P) return;
  const V = viewState();
  board.render(V);
  S.animateNext = true;
  updateHud();
  const v = activeView();
  if (v) {
    v.update({ path: S.path, legal: V.showLegal && !S.anim ? V.legal : [], hint: S.hint && S.hint.node, stuck: S.stuck && S.stuck.node, colorAt });
    const hm = V.heat || null;
    if (typeof v.setHeat === "function" && v.__zipHeat !== hm) { v.__zipHeat = hm; try { v.setHeat(hm); } catch { /* optional */ } }
  }
  if (AI) AI.afterRender();
}
function activeView() { return S.dim === 3 ? view3d : S.dim === 4 && viewMode === "4d" ? view4d : null; }

function updateHud() {
  $("stMoves").textContent = `${S.path.length - 1}/${S.n - 1}`;
  $("stHints").textContent = S.hints;
  $("hudHints").classList.toggle("dim", !S.hints);
  const t = S.t0 == null ? 0 : (S.tEnd || performance.now()) - S.t0;
  $("stTime").textContent = fmtTime(t);
  $("progress").style.width = `${(100 * (S.path.length - 1)) / Math.max(1, S.n - 1)}%`;
  $("undoBtn").disabled = S.path.length <= 1 || !!S.anim;
  $("resetBtn").disabled = S.path.length <= 1 && !S.anim;
}
setInterval(() => { if (S.t0 != null && !S.tEnd) updateHud(); }, 250);

// ------------------------------------------------------------------ moves
function takeOwnership() {
  if (S.owner !== "user") { S.owner = "user"; S.stuck = null; S.conf = new Map(); S.tried = null; }
  if (S.t0 == null) S.t0 = performance.now();
  if (AI) AI.raceUserStarted();
}
function push(v, { quiet = false } = {}) {
  if (!isLegal(v)) return false;
  takeOwnership();
  S.path.push(v); S.visited[v] = 1;
  bumpRev();
  S.hint = null;
  if (!quiet) {
    const k = S.cpIndex.get(v);
    if (k !== undefined && k > 0) sound.checkpoint(k); else sound.move(S.path.length / S.n);
  }
  afterMove();
  return true;
}
function truncate(len, { countUndo = true } = {}) {
  len = Math.max(1, len);
  if (len >= S.path.length) return;
  takeOwnership();
  for (const v of S.path.slice(len)) { S.visited[v] = 0; S.conf.delete(v); }
  S.path.length = len;
  bumpRev();
  if (countUndo) S.undos++;
  S.hint = null; S.won = false; S.tEnd = null;
  sound.undo();
  afterMove();
}
function undo() { if (!S.anim && S.path.length > 1) truncate(S.path.length - 1); }
function afterMove() {
  render();
  if (AI) AI.onPathChanged();
  if (S.path.length === S.n && S.owner === "user" && !S.won) checkWin();
}

// Route from the head to `target` through legal moves (fast drags / diagonal skips / clicks along a line).
function findRoute(target, maxLen) {
  const cpsLast = S.cps[S.cps.length - 1];
  const vis = S.visited.slice();
  let nxt = nextCp(), len = S.path.length;
  const tgtCell = board.L.cell[target];
  const same = (a, b) => board.samePanel(a, b);
  const legalFrom = (hd, v) => {
    if (vis[v] || !S.adjSet[hd].has(v)) return false;
    const k = S.cpIndex.get(v);
    if (k !== undefined && k !== nxt) return false;
    if (v === cpsLast && len !== S.n - 1) return false;
    return true;
  };
  const dist = (v) => { const [r, c] = board.L.cell[v]; return Math.abs(r - tgtCell[0]) + Math.abs(c - tgtCell[1]); };
  const out = [];
  const dfs = (hd, depth) => {
    if (hd === target) return true;
    if (depth === 0) return false;
    const cands = S.adj[hd].filter((w) => same(w, target) && board.gridAdjacent(hd, w) && dist(w) < dist(hd) && legalFrom(hd, w));
    for (const w of cands) {
      const k = S.cpIndex.get(w);
      vis[w] = 1; len++; if (k !== undefined) nxt++;
      out.push(w);
      if (dfs(w, depth - 1)) return true;
      out.pop(); vis[w] = 0; len--; if (k !== undefined) nxt--;
    }
    return false;
  };
  if (!same(head(), target)) return null;
  const d0 = dist(head());
  if (d0 > maxLen) return null;
  return dfs(head(), d0) ? out : null;
}
function pushRoute(route) {
  takeOwnership();
  for (const w of route) {
    if (!isLegal(w)) break;
    S.path.push(w); S.visited[w] = 1; bumpRev();
    const k = S.cpIndex.get(w);
    if (k !== undefined && k > 0) sound.checkpoint(k);
  }
  sound.move(S.path.length / S.n);
  S.hint = null;
  afterMove();
}

let dragMoved = false;
function onDown(v) {
  if (S.anim || S.busy || v == null || !S.P) return;
  dragMoved = false;
  const idx = S.path.indexOf(v);
  if (idx >= 0) { if (idx < S.path.length - 1) truncate(idx + 1); return; }
  if (isLegal(v)) { push(v); return; }
  const r = findRoute(v, Math.max(S.P.coords.length, 12));
  const [a, b] = board.L.cell[head()], [c, d] = board.L.cell[v];
  if (r && (a === c || b === d)) { pushRoute(r); return; }   // click along a straight line: fill it
  board.flash(v); sound.bad();
}
function onDrag(v) {
  if (S.anim || S.busy || v == null || !S.P) return;
  dragMoved = true;
  const idx = S.path.indexOf(v);
  if (idx >= 0) { if (idx < S.path.length - 1) truncate(idx + 1); return; }   // drag back over the line: rewind
  if (isLegal(v)) { push(v); return; }
  const r = findRoute(v, 4);    // fast drags / diagonal corner cuts: fill the (shortest, legal) gap
  if (r) pushRoute(r);
}

// keyboard movement
function moveDir(dr, dc) {
  const hd = head(), prev = S.path[S.path.length - 2];
  const [hx, hy] = board.center(hd);
  let best = null, bestScore = 0.7;
  for (const w of S.adj[hd]) {
    if (!board.samePanel(hd, w)) continue;
    const [x, y] = board.center(w);
    const dx = x - hx, dy = y - hy, len = Math.hypot(dx, dy) || 1;
    const score = (dx * dc + dy * dr) / len;
    if (score > bestScore && (w === prev || isLegal(w))) { best = w; bestScore = score; }
  }
  if (best == null) { flashHead(); return; }
  if (best === prev) undo(); else push(best);
}
function moveAxis(axis, delta) {
  if (S.dim <= axis) return;
  const hd = head(), prev = S.path[S.path.length - 2];
  const c = S.coords[hd].slice(); c[axis] += delta;
  const w = board.L.lookup.get(c.join(","));
  if (w === undefined || !S.adjSet[hd].has(w)) { flashHead(); return; }
  if (w === prev) undo(); else if (!push(w)) flashHead();
}
function flashHead() { board.flash(head()); sound.bad(); }

// ------------------------------------------------------------------ win + records
const REC_KEY = "zip-records";
function records() {
  const r = store.get(REC_KEY, null) || {};
  r.best = r.best || {}; r.solved = r.solved || {}; r.daily = r.daily || {}; r.race = r.race || { w: 0, l: 0 };
  return r;
}
function saveRecords(r) { store.set(REC_KEY, r); }
function recordKey() {
  const s = S.source;
  if (!s) return null;
  if (s.type === "daily") return `daily-${s.difficulty}`;
  if (s.type === "preset") return s.difficulty;
  return null;
}
function streakInfo(r = records()) {
  const today = localDateISO();
  const solvedOn = (d) => r.daily[d] && Object.keys(r.daily[d]).length > 0;
  let day = solvedOn(today) ? today : addDays(today, -1);
  let n = 0;
  while (solvedOn(day)) { n++; day = addDays(day, -1); }
  let best = r.bestStreak || 0;
  if (n > best) { best = n; }
  return { current: n, best, today: solvedOn(today) };
}

async function checkWin() {
  S.tEnd = performance.now();
  S.won = true;
  render();
  const tok = pathToken();
  let valid = true;
  try { valid = (await api("/api/check", { puzzle: S.data, path: S.path })).valid; } catch { /* offline: trust local rules */ }
  if (!tokenValid(tok)) return;
  if (!valid) { S.won = false; S.tEnd = null; banner("That path isn't valid.", "bad"); return; }
  const ms = S.tEnd - S.t0;
  if (AI && AI.raceUserFinished(ms)) return;   // race mode shows its own result
  sound.win();
  celebrate();
  // records
  const r = records(), key = recordKey();
  let pb = null;
  if (key) {
    r.solved[key] = (r.solved[key] || 0) + 1;
    if (!S.hints && !S.assisted) {
      const prev = r.best[key];
      if (!prev || ms < prev.ms) { pb = { prev: prev ? prev.ms : null }; r.best[key] = { ms, date: localDateISO() }; }
    }
  }
  let firstDaily = false;
  if (S.source && S.source.type === "daily") {
    const d = S.source.date;
    r.daily[d] = r.daily[d] || {};
    if (!r.daily[d][S.source.difficulty]) { firstDaily = true; r.daily[d][S.source.difficulty] = { ms, hints: S.hints, undos: S.undos }; }
  }
  const st = streakInfo(r);
  r.bestStreak = Math.max(r.bestStreak || 0, st.current);
  saveRecords(r);
  renderRecords();
  showWin(ms, pb, st, firstDaily);
  banner(`Solved in <b>${fmtTime(ms)}</b>.`, "good", [["New puzzle", () => newPuzzle(), "primary"]]);
}

function shareText(ms) {
  const s = S.source || {};
  const kindLbl = KIND_INFO[S.kind]?.label || S.kind;
  let head1 = s.type === "daily" ? `Zip #${s.number} · ${s.label}` : s.type === "preset" ? `Zip · ${presetById(s.difficulty)?.label || ""} (${kindLbl})` : s.type === "editor" ? `Zip · ${s.name || "custom puzzle"}` : `Zip · ${kindLbl}`;
  const lines = [head1, `⏱ ${fmtTime(ms)} · ${S.n - 1} moves · ${S.hints ? `${S.hints} hint${S.hints > 1 ? "s" : ""}` : "no hints"}${S.undos ? ` · ${S.undos} undo${S.undos > 1 ? "s" : ""}` : ""}`];
  if (s.type === "daily") { const st = streakInfo(); if (st.current > 1) lines.push(`🔥 ${st.current}-day streak`); }
  return lines.join("\n");
}
function showWin(ms, pb, st, firstDaily) {
  const s = S.source || {};
  $("winBadge").innerHTML = icon(pb ? "trophy" : "check");
  $("winBadge").className = `win-badge ${pb ? "gold" : ""}`;
  $("winKicker").textContent = s.type === "daily" ? `Daily #${s.number} · ${s.label}` : `${KIND_INFO[S.kind]?.label || S.kind} · ${S.n} cells`;
  $("winTitle").textContent = S.hints ? "Solved!" : S.undos === 0 ? "Flawless!" : ["Brilliant!", "Nice one!", "Great solve!"][Math.floor(Math.random() * 3)];
  $("winTime").textContent = fmtTime(ms);
  const pbEl = $("winPb");
  if (pb) { pbEl.hidden = false; pbEl.innerHTML = `${icon("trophy")} New personal best${pb.prev ? ` <span class="muted">(was ${fmtTime(pb.prev)})</span>` : ""}`; }
  else pbEl.hidden = true;
  const stats = [["Moves", S.n - 1], ["Undos", S.undos], ["Hints", S.hints]];
  if (s.type === "daily") stats.push(["Streak", `${st.current} day${st.current === 1 ? "" : "s"}`]);
  $("winStats").innerHTML = stats.map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
  $("shareText").textContent = shareText(ms);
  openModal("winModal");
  $("winNew").focus();
}

// ------------------------------------------------------------------ modals
let lastFocus = null;
function openModal(id) {
  lastFocus = document.activeElement;
  document.querySelectorAll(".modal.show").forEach((m) => m.classList.remove("show"));
  $(id).classList.add("show");
}
function closeModal(id) {
  $(id).classList.remove("show");
  if (lastFocus && lastFocus.focus) try { lastFocus.focus({ preventScroll: true }); } catch { /* ignore */ }
}
function anyModal() { return document.querySelector(".modal.show"); }

// ------------------------------------------------------------------ confetti
function celebrate(count = 170) {
  if (reducedMotion()) return;
  const cv = $("confetti"), ctx = cv.getContext("2d");
  cv.width = innerWidth * devicePixelRatio; cv.height = innerHeight * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  const cols = ["#ff9f1c", "#ff5a4e", "#d63a8f", "#7b3fd6", "#2f6fe0", "#1f8a4c"];
  const r = board && board.svg ? board.svg.getBoundingClientRect() : { left: innerWidth / 2, top: innerHeight / 3, width: 0, height: 0 };
  const parts = Array.from({ length: count }, () => ({
    x: r.left + r.width / 2 + (Math.random() - 0.5) * r.width * 0.6, y: r.top + r.height * 0.4, vx: (Math.random() - 0.5) * 15, vy: -Math.random() * 13 - 4,
    s: Math.random() * 6 + 4, c: cols[Math.floor(Math.random() * cols.length)], a: Math.random() * 6, va: (Math.random() - 0.5) * 0.3,
  }));
  const t0 = performance.now();
  const frame = (t) => {
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    for (const p of parts) {
      p.vy += 0.35; p.vx *= 0.99; p.x += p.vx; p.y += p.vy; p.a += p.va;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.a); ctx.fillStyle = p.c; ctx.fillRect(-p.s / 2, -p.s / 4, p.s, p.s / 2); ctx.restore();
    }
    if (t - t0 < 2800) requestAnimationFrame(frame); else ctx.clearRect(0, 0, innerWidth, innerHeight);
  };
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ agent / solver playback
function animDelay() { const v = Number($("speed").value) / 100; return 8 + 650 * Math.pow(1 - v, 2.2); }
function cancelAnim(finish = false) {
  if (!S.anim) return;
  const a = S.anim; S.anim = null;
  a.cancel(finish);
}
function setPathDirect(path, owner) {
  bumpRev();
  S.path = path.slice(); S.visited = new Uint8Array(S.n); for (const v of S.path) S.visited[v] = 1;
  if (owner) S.owner = owner;
}
function playPath(full, startLen, steps, owner, onDone) {
  cancelAnim();
  S.won = false; S.tEnd = null; S.hint = null; S.stuck = null; S.tried = null;
  S.assisted = true;
  setPathDirect(full.slice(0, startLen), owner);
  S.conf = new Map(); S.curStep = null;
  let i = startLen, timer = null, stopped = false;
  const finishAll = () => {
    for (; i < full.length; i++) {
      const v = full[i]; S.path.push(v); S.visited[v] = 1; bumpRev();
      if (steps && steps[i - startLen]) S.conf.set(v, steps[i - startLen].p);
    }
  };
  const done = () => { S.anim = null; S.curStep = steps && steps.length ? steps[steps.length - 1] : null; render(); AI.updateConfPanel(); AI.onPathChanged(); onDone && onDone(); };
  S.anim = { cancel(finish) { stopped = true; clearTimeout(timer); if (finish) finishAll(); done(); } };
  const tick = () => {
    if (stopped) return;
    if (i >= full.length) { S.anim = null; done(); return; }
    const v = full[i], st = steps && steps[i - startLen];
    S.path.push(v); S.visited[v] = 1; bumpRev();
    if (st) { S.conf.set(v, st.p); S.curStep = st; }
    i++;
    const k = S.cpIndex.get(v);
    if (k !== undefined && k > 0) sound.checkpoint(k); else if (animDelay() > 60) sound.move(i / S.n);
    render();
    if (st) AI.updateConfPanel();
    timer = setTimeout(tick, animDelay());
  };
  render();
  banner(`Animating the ${owner === "agent" ? "agent's" : "solver's"} path…`, "", [["Skip to end", () => cancelAnim(true)], ["Stop", () => cancelAnim(false)]]);
  timer = setTimeout(tick, animDelay());
}

// ------------------------------------------------------------------ hints
async function getHint() {
  if (S.busy || S.anim || !S.data || S.won) return;
  if (S.path.length === S.n) return;
  setBusy(true, "Thinking…");
  const tok = pathToken();
  try {
    const hres = await api("/api/hint", { puzzle: S.data, path: S.path, id: S.id });
    if (!tokenValid(tok)) return;
    if (hres.status === "next") {
      S.hints++; S.hint = { node: hres.next }; sound.hint();
      banner(`Hint: move to the highlighted cell${S.dim >= 3 && !board.samePanel(head(), hres.next) ? ` (${board.crossLabel(head(), hres.next)})` : ""}.`, "good");
    } else if (hres.status === "backtrack") {
      S.hints++; S.hint = { node: hres.next, keep: hres.keep }; sound.hint();
      const where = hres.keep <= 1 ? "all the way to checkpoint 1" : `to move <b>${hres.keep - 1}</b> (orange ring)`;
      banner(`Your path can't be completed. Backtrack ${where}, then go to the green cell.`, "warn",
        [[hres.keep <= 1 ? "Back to start" : `Backtrack to move ${hres.keep - 1}`, () => {
          if (!tokenValid(tok)) { banner("The path changed since this hint; ask for a new one.", "warn"); return; }
          const keep = hres.keep, nx = hres.next; truncate(keep); S.hint = { node: nx }; render(); banner("Now move to the highlighted cell.", "good");
        }, "primary"]]);
    } else if (hres.status === "done") banner("Already solved!", "good");
    else banner(esc(hres.message || `Hint unavailable (${hres.status}).`), "warn");
    render();
  } catch (e) { if (tokenValid(tok)) banner(`Hint failed: ${esc(e.message)}`, "bad"); }
  finally { setBusy(false); }
}

// ------------------------------------------------------------------ puzzle sources
function kindOptValues() { return [...document.querySelectorAll("#kindOpts [data-opt]")].map((i) => [i.dataset.opt, i.value]); }
function customParams() {
  const kind = $("kind").value;
  const opts = {};
  document.querySelectorAll("#kindOpts [data-opt]").forEach((i) => { if (i.value !== "") opts[i.dataset.opt] = Number(i.value); });
  const seedTxt = $("seed").value.trim(), ncp = $("ncp").value.trim();
  return { kind, size: Number($("size").value), num_checkpoints: ncp ? Number(ncp) : null,
    unique: $("unique").checked, seed: seedTxt === "" ? null : Number(seedTxt), options: opts,
    time_limit: $("unique").checked && (kind === "grid3d" || kind === "grid4d") ? 40 : 20 };
}
function customHash(p, res, rawOpts) {
  const hh = new URLSearchParams();
  hh.set("kind", res.kind); hh.set("size", p.size); hh.set("seed", res.seed);
  if (p.num_checkpoints) hh.set("cp", p.num_checkpoints);
  if (p.unique) hh.set("unique", "1");
  for (const [k, v] of rawOpts) hh.set(`opt.${k}`, v.trim());
  return "#" + hh.toString();
}
function setHash(hash) { try { history.replaceState(null, "", hash); } catch { /* ignore */ } }

async function generate(params, source, hashFn, busyMsg) {
  if (S.busy) return false;
  cancelAnim();
  closeModal("winModal");
  setBusy(true, busyMsg || (params.unique ? "Generating a puzzle with a unique solution…" : "Generating…"));
  try {
    const res = await api("/api/generate", params);
    loadPuzzle(res, source);
    if (hashFn) setHash(hashFn(res));
    return true;
  } catch (e) { banner(`Could not generate: ${esc(e.message)}`, "bad"); return false; }
  finally { setBusy(false); }
}
function playPreset(id, seed = null) {
  const p = presetById(id);
  if (!p) return;
  S.difficulty = id; store.set("zip-difficulty", id);
  return generate({ kind: p.kind, size: p.size, unique: p.unique, options: p.options, seed, time_limit: 25 },
    { type: "preset", difficulty: id }, (res) => `#d=${id}&seed=${res.seed}`);
}
function playCustom() {
  const p = customParams(), raw = kindOptValues();
  return generate(p, { type: "custom" }, (res) => customHash(p, res, raw));
}
async function playDaily(diff, date = localDateISO()) {
  if (S.busy) return false;
  cancelAnim(); closeModal("winModal");
  setBusy(true, "Fetching the daily puzzle…");
  try {
    const res = await api(`/api/daily?difficulty=${encodeURIComponent(diff)}&date=${encodeURIComponent(date)}`);
    loadPuzzle(res, { type: "daily", ...res.daily });
    store.set("zip-daily-diff", diff);
    setHash(`#daily=${diff}&date=${res.daily.date}`);
    const done = records().daily[res.daily.date]?.[diff];
    if (done) banner(`You already solved today's ${res.daily.label} puzzle in <b>${fmtTime(done.ms)}</b>. Replays don't change your record.`, "good");
    renderDaily();
    return true;
  } catch (e) { banner(`Could not load the daily puzzle: ${esc(e.message)}`, "bad"); return false; }
  finally { setBusy(false); }
}
async function playEditor(id) {
  setBusy(true, "Loading the custom puzzle…");
  try {
    const res = await api(`/api/custom/${encodeURIComponent(id)}`);
    loadPuzzle({ id: null, seed: null, kind: res.puzzle.kind, puzzle: res.puzzle }, { type: "editor", id, name: res.name });
    return true;
  } catch (e) {
    banner(`Could not load custom puzzle <b>${esc(id)}</b>: ${esc(e.message)}. ${e.status === 404 ? "It may have been deleted, or the editor is not installed." : ""}`, "bad", [["Play a random puzzle", () => playPreset(S.difficulty), "primary"]]);
    return false;
  } finally { setBusy(false); }
}
// "New puzzle" = another one of the same kind as the current one
function newPuzzle() {
  const s = S.source;
  if (s && s.type === "custom") return playCustom();
  if (s && s.type === "daily") return playPreset(s.difficulty);
  return playPreset(S.difficulty || "medium");
}
function replay() { if (S.P) { resetPath(true); render(); AI.onPathChanged(); } }

// ------------------------------------------------------------------ UI: chips, daily, records
function highlightChips() {
  const s = S.source || {};
  document.querySelectorAll("#diffChips .chip").forEach((c) => {
    const on = (s.type === "preset" && c.dataset.diff === s.difficulty) || (s.type === "custom" && c.dataset.diff === "custom");
    c.classList.toggle("on", on); c.setAttribute("aria-pressed", on);
  });
  const dc = $("dailyChip");
  dc.classList.toggle("on", s.type === "daily");
  dc.setAttribute("aria-pressed", s.type === "daily");
}
function renderChips() {
  const box = $("diffChips");
  box.innerHTML = "";
  for (const p of S.presets) {
    box.appendChild(h("button", { class: "chip", type: "button", "data-diff": p.id, title: p.blurb, "aria-pressed": "false",
      onclick: () => playPreset(p.id) }, h("span", { class: `dot d-${p.id}` }), p.label));
  }
  box.appendChild(h("button", { class: "chip", type: "button", "data-diff": "custom", "aria-pressed": "false", title: "Your own settings",
    onclick: () => { selectTab("game"); $("advanced").open = true; $("advanced").scrollIntoView({ behavior: "smooth", block: "nearest" }); $("kind").focus(); } }, "Custom…"));
}
function renderDaily() {
  const r = records(), today = localDateISO(), st = streakInfo(r);
  const epoch = new Date(2026, 0, 1), now = new Date();
  const num = Math.round((new Date(now.getFullYear(), now.getMonth(), now.getDate()) - epoch) / 86400000) + 1;
  $("dailyChip").innerHTML = `${icon("calendar")}<span>Daily <b>#${num}</b></span>${st.today ? `<span class="tick">${icon("check")}</span>` : ""}`;
  $("dailyDate").textContent = new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" }) + ` · puzzle #${num}`;
  $("streak").innerHTML = st.current ? `${icon("flame")} ${st.current}-day streak` : `<span class="muted">no streak yet</span>`;
  $("streak").title = `Best streak: ${Math.max(st.best, r.bestStreak || 0)} days`;
  const grid = $("dailyGrid");
  grid.innerHTML = "";
  for (const p of S.presets) {
    const done = r.daily[today] && r.daily[today][p.id];
    const cur = S.source && S.source.type === "daily" && S.source.difficulty === p.id;
    grid.appendChild(h("button", { class: `daily-tile ${done ? "done" : ""} ${cur ? "on" : ""}`, type: "button", onclick: () => playDaily(p.id),
      "aria-label": `Daily ${p.label}${done ? `, solved in ${fmtTime(done.ms)}` : ""}` },
      h("span", { class: "dl" }, h("span", { class: `dot d-${p.id}` }), p.label),
      h("span", { class: "dv mono", html: done ? `${icon("check")} ${fmtTime(done.ms)}` : `<span class="muted">${esc(p.blurb)}</span>` })));
  }
  $("dailyChip").onclick = () => {
    const pref = S.source && S.source.type === "daily" ? S.source.difficulty : (store.get("zip-daily-diff", null) || "medium");
    playDaily(pref);
  };
}
function renderRecords() {
  const r = records();
  const rows = S.presets.map((p) => {
    const b = r.best[p.id], bd = r.best[`daily-${p.id}`];
    const n = (r.solved[p.id] || 0) + (r.solved[`daily-${p.id}`] || 0);
    return `<tr><td><span class="dot d-${p.id}"></span>${p.label}</td><td class="mono">${b ? fmtTime(b.ms) : "-"}</td><td class="mono">${bd ? fmtTime(bd.ms) : "-"}</td><td class="mono muted">${n || ""}</td></tr>`;
  });
  $("records").innerHTML = `<thead><tr><th></th><th>Practice</th><th>Daily</th><th title="puzzles solved">#</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  const total = Object.values(r.solved).reduce((a, b) => a + b, 0);
  $("solvedCount").textContent = total ? `${total} solved` : "";
  renderDaily();
}

// ------------------------------------------------------------------ legend, shortcuts, onboarding
function renderLegend() {
  const items = [];
  if (S.dim >= 3) items.push(`Badges: <b>solid</b> = the line continues to another ${S.dim === 3 ? "layer" : "panel"}, <b>hollow</b> = arrived from one.`);
  if (S.P && S.P.bridges) items.push("Wooden bridges are the only way between islands. Some are decoys the solution never uses.");
  if (S.agent) items.push("Small bars in a cell: an agent step taken with < 90% confidence (orange: < 50%).");
  items.push(`Drag from <b>1</b> through every cell · drag back to undo · <kbd>?</kbd> shortcuts`);
  $("legend").innerHTML = items.map((x) => `<span>${x}</span>`).join("");
}
function shortcutRows() {
  const rows = [["← ↑ → ↓", "move"], ["Backspace / Ctrl+Z", "undo"], ["H", "hint"], ["R", "reset"], ["N", "new puzzle"],
    ["V", "AI vision on/off"], ["M", "sound on/off"], ["?", "this sheet"], ["Esc", "close / stop animation"]];
  if (S.dim === 3) rows.splice(1, 0, ["Q / E  (PgUp / PgDn)", "layer z − / +"]);
  if (S.dim === 4) rows.splice(1, 0, ["Q / E", "z − / + (panel row)"], ["A / D", "w − / + (panel column)"]);
  return rows;
}
function showKeys() {
  $("keysTable").innerHTML = shortcutRows().map(([k, d]) => `<tr><td>${k.split(" / ").map((x) => x.split(" ").map((y) => /^[()]/.test(y) ? y : `<kbd>${y}</kbd>`).join(" ")).join(" / ")}</td><td>${d}</td></tr>`).join("");
  openModal("keysModal");
}
function buildDemo() {
  const cells = document.querySelector(".demo-cells"), cps = document.querySelector(".demo-cps");
  if (!cells || cells.childNodes.length) return;
  const NS = "http://www.w3.org/2000/svg";
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const e = document.createElementNS(NS, "rect");
    Object.entries({ x: 4 + c * 50, y: 4 + r * 50, width: 42, height: 42, rx: 9 }).forEach(([k, v]) => e.setAttribute(k, v));
    cells.appendChild(e);
  }
  [[25, 25, 1], [125, 75, 2], [125, 125, 3]].forEach(([x, y, k]) => {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("transform", `translate(${x} ${y})`);
    g.innerHTML = `<circle r="15"/><text y="6" text-anchor="middle">${k}</text>`;
    cps.appendChild(g);
  });
}

// ------------------------------------------------------------------ 3D / 4D views
function viewTheme() { return () => ({ bg: cssVar("--panel-2"), cell: cssVar("--cell-edge"), edge: cssVar("--border"), legal: cssVar("--legal"), good: cssVar("--good"), bad: cssVar("--bad"), plate: cssVar("--cell"), heat: cssVar("--heat"), accent: cssVar("--accent") }); }
let viewGen = 0;
async function setupViews() {
  const gen = ++viewGen;
  if (view3d) { try { view3d.dispose(); } catch { /* ignore */ } view3d = null; }
  if (view4d) { try { view4d.dispose(); } catch { /* ignore */ } view4d = null; }
  const box = $("view3d");
  box.querySelectorAll(".fallback").forEach((x) => x.remove());
  $("viewToggle").hidden = true;
  if (S.dim === 4) {
    try {
      await import("./view4d.js");   // only offer the toggle when the module exists and loads
      if (gen === viewGen && S.dim === 4) $("viewToggle").hidden = false;
    } catch { $("viewToggle").hidden = true; if (viewMode === "4d") setViewMode("panels"); }
    if (viewMode === "4d") mount4D(gen);
    return;
  }
  if (S.dim !== 3) return;
  $("view3dLabel").textContent = "Drag to orbit · scroll to zoom · click a node to move";
  try {
    const mod = await import("./view3d.js");
    const v = await mod.createView3D(box, { coords: S.coords, edges: S.data.edges, cps: S.cps, n: S.n,
      onClick: (node) => onDown(node), theme: viewTheme() });
    if (gen !== viewGen || S.dim !== 3) { v.dispose(); return; }
    view3d = v;
    render();
  } catch (e) {
    box.appendChild(h("div", { class: "fallback", text: `3D view unavailable (${e.message}). The layer panels still work.` }));
  }
}
async function mount4D(gen = viewGen) {
  const box = $("view3d");
  $("view3dLabel").textContent = "Drag to rotate · click a node to move";
  try {
    const mod = await import("./view4d.js");
    const v = await mod.createView4D(box, { coords: S.coords, edges: S.data.edges, cps: S.cps, n: S.n,
      onClick: (node) => onDown(node), theme: viewTheme() });
    if (gen !== viewGen || S.dim !== 4 || viewMode !== "4d") { v.dispose(); return; }
    view4d = v;
    render();
  } catch (e) {
    box.appendChild(h("div", { class: "fallback", text: `4D view unavailable (${e.message}).` }));
  }
}
function setViewMode(m) {
  viewMode = m;
  document.querySelectorAll("#viewToggle button").forEach((b) => b.classList.toggle("on", b.dataset.v === m));
  if (view4d && m !== "4d") { try { view4d.dispose(); } catch { /* ignore */ } view4d = null; }
  layoutBoard();
  render();
  if (m === "4d" && !view4d) mount4D();
}

// ------------------------------------------------------------------ custom (advanced) controls
function syncKindUI() {
  const kind = $("kind").value, info = KIND_INFO[kind];
  const size = $("size");
  size.min = info.size[0]; size.max = info.size[1];
  if (!size.value || Number(size.value) < info.size[0] || Number(size.value) > info.size[1] || size.dataset.kind !== kind) size.value = info.size[2];
  size.dataset.kind = kind;
  $("sizeLbl").textContent = info.sizeLbl;
  const box = $("kindOpts");
  const keep = box.dataset.kind === kind ? new Map([...box.querySelectorAll("[data-opt]")].map((i) => [i.dataset.opt, i.value])) : new Map();
  box.dataset.kind = kind;
  box.innerHTML = "";
  const add = (label, opt, attrs) => {
    const id = `opt-${opt}`;
    const row = h("div", { class: "row", html: `<label class="lbl" for="${id}">${label}</label><input id="${id}" class="grow" data-opt="${opt}" ${attrs}><span class="muted opt-val"></span>` });
    const inp = row.querySelector("input"), out = row.querySelector("span");
    if (keep.has(opt)) inp.value = keep.get(opt);
    const show = () => { out.textContent = inp.type === "range" ? `${Math.round(inp.value * 100)}%` : ""; };
    inp.addEventListener("input", show); show();
    box.appendChild(row);
  };
  if (kind === "walls") add("Walls", "walls_frac", 'type="range" min="0.05" max="0.8" step="0.05" value="0.3"');
  if (kind === "mask") add("Fill", "fill", 'type="range" min="0.4" max="1" step="0.05" value="0.75"');
  if (kind === "islands") {
    add("Jumps", "jumps", 'type="number" min="1" max="32" placeholder="auto" title="How many times the solution crosses between islands. Blank = auto"');
    add("Decoy bridges", "decoys", 'type="number" min="0" max="12" placeholder="auto" title="Extra bridges the solution does not use. Blank = islands−1"');
    add("Max per pair", "max_per_pair", 'type="number" min="1" max="6" value="3" title="Most crossings allowed between the same two islands"');
  }
  $("uniqueNote").hidden = !($("unique").checked && (kind === "grid3d" || kind === "grid4d"));
}
function restoreKindOpts(hh) {
  document.querySelectorAll("#kindOpts [data-opt]").forEach((i) => {
    const v = hh.get(`opt.${i.dataset.opt}`);
    if (v === null) return;
    i.value = v;
    i.dispatchEvent(new Event("input"));
  });
}

// ------------------------------------------------------------------ tabs
function selectTab(name) {
  document.querySelectorAll(".tabs [role=tab]").forEach((t) => {
    const on = t.id === `tab-${name}`;
    t.classList.toggle("on", on); t.setAttribute("aria-selected", on); t.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll(".pane").forEach((p) => { p.hidden = p.id !== `pane-${name}`; });
  store.set("zip-tab", name);
  if (AI) AI.onTab(name);
}

// ------------------------------------------------------------------ keyboard
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest && ev.target.closest("input, select, textarea, [contenteditable]")) return;
  const m = anyModal();
  if (m) {
    if (ev.key === "Escape") { closeModal(m.id); ev.preventDefault(); }
    return;
  }
  if (ev.key === "?" ) { showKeys(); ev.preventDefault(); return; }
  if (ev.key === "m" || ev.key === "M") { toggleSound(); ev.preventDefault(); return; }
  if (!S.data) return;
  if (S.anim) { if (ev.key === "Escape") { cancelAnim(true); ev.preventDefault(); } return; }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) { if ((ev.key === "z" || ev.key === "Z") && (ev.ctrlKey || ev.metaKey)) { undo(); ev.preventDefault(); } return; }
  const k = ev.key;
  let handled = true;
  if (k === "ArrowUp") moveDir(-1, 0);
  else if (k === "ArrowDown") moveDir(1, 0);
  else if (k === "ArrowLeft") moveDir(0, -1);
  else if (k === "ArrowRight") moveDir(0, 1);
  else if (k === "PageUp" || k === "q" || k === "Q") moveAxis(2, -1);
  else if (k === "PageDown" || k === "e" || k === "E") moveAxis(2, 1);
  else if (k === "a" || k === "A") moveAxis(3, -1);
  else if (k === "d" || k === "D") moveAxis(3, 1);
  else if (k === "Backspace") undo();
  else if (k === "h" || k === "H") getHint();
  else if (k === "r" || k === "R") { if (!S.busy) { resetPath(); render(); AI.onPathChanged(); } }
  else if (k === "n" || k === "N") newPuzzle();
  else if (k === "v" || k === "V") AI.toggleVision();
  else handled = false;
  if (handled) ev.preventDefault();
});

// ------------------------------------------------------------------ sound + theme
function paintSound() {
  $("soundBtn").innerHTML = icon(sound.muted ? "mute" : "sound");
  $("soundBtn").setAttribute("aria-label", sound.muted ? "Unmute sounds" : "Mute sounds");
  $("soundBtn").setAttribute("aria-pressed", String(!sound.muted));
}
function toggleSound() { sound.setMuted(!sound.muted); paintSound(); toast(sound.muted ? "Sound off" : "Sound on"); }
function onTheme() {
  if (S.data) { layoutBoard(); render(); AI.drawSpark(); }
  for (const v of [view3d, view4d]) if (v) try { v.refreshTheme(); } catch { /* ignore */ }
}

// ------------------------------------------------------------------ init
async function init() {
  board = new Board($("board"), { interactive: true, onDown, onDrag, label: "Zip puzzle board: draw the path with the mouse, touch or arrow keys" });
  // static button labels (icons)
  $("undoBtn").innerHTML = `${icon("undo")}<span>Undo</span>`;
  $("hintBtn").innerHTML = `${icon("bulb")}<span>Hint</span>`;
  $("resetBtn").innerHTML = `${icon("reset")}<span>Reset</span>`;
  $("visionBtn").innerHTML = `${icon("eye")}<span>AI vision</span>`;
  $("newBtn").innerHTML = `${icon("plus")}<span>New</span>`;
  $("helpBtn").innerHTML = icon("help");
  $("keysBtn").innerHTML = icon("keyboard");
  $("winX").innerHTML = icon("x");
  document.querySelectorAll("[data-close].close").forEach((b) => { b.innerHTML = icon("x"); });
  $("shareBtn").innerHTML = `${icon("copy")}<span>Copy result</span>`;
  $("winWatch").innerHTML = `${icon("cpu")}<span>Watch the AI</span>`;
  $("winNew").innerHTML = `${icon("plus")}<span>Next puzzle</span>`;
  paintSound();

  AI = initAI({
    S, $, board: () => board, render, banner, setBusy, pathToken, tokenValid, bumpRev, setPathDirect, playPath, cancelAnim,
    truncate, legalMoves, newPuzzle, playPreset, replay, celebrate, toast, layoutBoard, records, saveRecords, renderLegend,
    openModal, closeModal, animDelay, sound, presetById, head,
  });

  $("undoBtn").onclick = undo;
  $("hintBtn").onclick = getHint;
  $("resetBtn").onclick = () => { if (S.anim) cancelAnim(false); if (S.data) { resetPath(); render(); AI.onPathChanged(); } };
  $("newBtn").onclick = () => newPuzzle();
  $("customBtn").onclick = () => playCustom();
  $("showLegal").onchange = () => { store.set("zip-showLegal", $("showLegal").checked); render(); };
  $("showTimer").checked = store.get("zip-showTimer", true);
  document.body.classList.toggle("no-timer", !$("showTimer").checked);
  $("showTimer").onchange = () => { store.set("zip-showTimer", $("showTimer").checked); document.body.classList.toggle("no-timer", !$("showTimer").checked); };
  $("soundBtn").onclick = toggleSound;
  $("helpBtn").onclick = () => { buildDemo(); openModal("helpModal"); };
  $("keysBtn").onclick = showKeys;
  $("winX").onclick = () => closeModal("winModal");
  $("winNew").onclick = () => { closeModal("winModal"); newPuzzle(); };
  $("winWatch").onclick = () => { closeModal("winModal"); selectTab("ai"); AI.runSolve("hybrid", { fromStart: true }); };
  $("shareBtn").onclick = async () => {
    const txt = $("shareText").textContent;
    try { await navigator.clipboard.writeText(txt); toast("Result copied to the clipboard"); }
    catch {
      const r = document.createRange(); r.selectNodeContents($("shareText"));
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      toast("Select + copy the text");
    }
  };
  document.querySelectorAll(".modal").forEach((m) => {
    m.addEventListener("click", (e) => { if (e.target === m || e.target.closest("[data-close]")) closeModal(m.id); });
  });
  document.querySelectorAll(".tabs [role=tab]").forEach((t) => {
    t.onclick = () => selectTab(t.id.replace("tab-", ""));
    t.onkeydown = (e) => {
      const tabs = [...document.querySelectorAll(".tabs [role=tab]")], i = tabs.indexOf(t);
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const nx = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        nx.focus(); nx.click(); e.preventDefault();
      }
    };
  });
  document.querySelectorAll("#viewToggle button").forEach((b) => { b.onclick = () => setViewMode(b.dataset.v); });
  $("kind").onchange = () => { syncKindUI(); $("seed").value = ""; };
  $("unique").onchange = syncKindUI;
  window.addEventListener("zip-theme", onTheme);
  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (S.data) { layoutBoard(); render(); AI.drawSpark(); } }, 120); });

  selectTab(store.get("zip-tab", "game") === "race" ? "game" : store.get("zip-tab", "game"));
  S.difficulty = store.get("zip-difficulty", "medium");
  try { const pr = await api("/api/presets"); if (pr && pr.presets && pr.presets.length) S.presets = pr.presets; } catch { /* fallback list */ }
  if (!presetById(S.difficulty)) S.difficulty = "medium";
  renderChips();
  renderRecords();

  // restore from the URL hash
  const hh = new URLSearchParams(location.hash.slice(1));
  if (hh.get("kind") && KIND_INFO[hh.get("kind")]) $("kind").value = hh.get("kind");
  syncKindUI();
  if (hh.get("size")) $("size").value = hh.get("size");
  if (hh.get("seed") && hh.get("kind")) $("seed").value = hh.get("seed");
  if (hh.get("cp")) $("ncp").value = hh.get("cp");
  if (hh.get("unique")) $("unique").checked = true;
  syncKindUI();
  if (hh.get("kind")) restoreKindOpts(hh);

  AI.loadModels();
  let ok = false;
  if (hh.get("custom")) ok = await playEditor(hh.get("custom"));
  else if (hh.get("daily")) ok = await playDaily(presetById(hh.get("daily")) ? hh.get("daily") : "medium", hh.get("date") || localDateISO());
  else if (hh.get("d") && presetById(hh.get("d"))) ok = await playPreset(hh.get("d"), hh.get("seed") ? Number(hh.get("seed")) : null);
  else if (hh.get("kind")) { $("advanced").open = true; ok = await playCustom(); }
  if (!ok && !S.data) {
    // first load: today's daily at the preferred difficulty, unless already solved -> a fresh practice puzzle
    const diff = store.get("zip-daily-diff", null) || S.difficulty || "medium";
    const solvedToday = records().daily[localDateISO()]?.[diff];
    if (solvedToday) await playPreset(S.difficulty); else await playDaily(diff);
  }
  if (!store.get("zip-onboarded", false)) { buildDemo(); openModal("helpModal"); store.set("zip-onboarded", true); }
  // links / manual edits of the hash (our own updates use replaceState and do not fire this)
  window.addEventListener("hashchange", () => {
    const q = new URLSearchParams(location.hash.slice(1));
    if (S.busy) return;
    if (q.get("custom")) { if (!(S.source && S.source.type === "editor" && S.source.id === q.get("custom"))) playEditor(q.get("custom")); }
    else if (q.get("daily") && presetById(q.get("daily"))) playDaily(q.get("daily"), q.get("date") || localDateISO());
    else if (q.get("d") && presetById(q.get("d"))) playPreset(q.get("d"), q.get("seed") ? Number(q.get("seed")) : null);
    else if (q.get("kind") && KIND_INFO[q.get("kind")]) {
      $("kind").value = q.get("kind"); syncKindUI();
      if (q.get("size")) $("size").value = q.get("size");
      $("seed").value = q.get("seed") || ""; $("ncp").value = q.get("cp") || ""; $("unique").checked = !!q.get("unique");
      syncKindUI(); restoreKindOpts(q); playCustom();
    }
  });
}

window.__zip = S;  // for debugging / automated screenshots
window.__zipApi = {
  newPuzzle, getHint, push, cancelAnim, playDaily, playPreset, playCustom, playEditor, onDown, onDrag, undo,
  runRL: (m) => AI.runSolve(m || "greedy"), runExact: () => AI.runSolve("exact"), ai: () => AI,
  board: () => board, view3d: () => view3d, view4d: () => view4d, selectTab, setViewMode,
};
init();
