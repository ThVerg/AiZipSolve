// Zip — the casual game page (/). Home (daily + modes), a big-board game shell, "Show me" robot solves,
// Race the robot, and a first-visit tutorial. The power-user page lives at /lab (app.js).
import { $, h, api, fmtTime, store, icon, localDateISO, addDays, clamp, sleep, reducedMotion, cssVar } from "./play/util.js";
import { Board, makeModel, outlineLoops, offsetLoop, roundedPath } from "./play/board.js";
import { sound } from "./play/sound.js";
import { PathGame } from "./play/rules.js";
import { createFx } from "./play/fx.js";

// ------------------------------------------------------------------ content
const MODES = {
  classic: { label: "Classic", dflt: "medium", diffs: {
    easy: { label: "Easy", kind: "grid2d", size: 5, unique: true },
    medium: { label: "Medium", kind: "grid2d", size: 7, unique: true },
    hard: { label: "Hard", kind: "grid2d", size: 9, unique: true } } },
  walls: { label: "Walls", dflt: "normal", diffs: { normal: { label: "Walls", kind: "walls", size: 7, unique: true, options: { walls_frac: 0.35 } } } },
  islands: { label: "Islands", dflt: "normal", diffs: { normal: { label: "Islands", kind: "islands", size: 4, unique: true } } },
  cube: { label: "3D Cube", dflt: "normal", diffs: { normal: { label: "4×4×4", kind: "grid3d", size: 4, unique: false } } },
};
// lab difficulty presets (old #d= / #daily= links) -> the nearest mode (for "Next puzzle")
const PRESET_MODE = { easy: ["classic", "easy"], medium: ["classic", "medium"], hard: ["walls", "normal"], expert: ["islands", "normal"], insane: ["cube", "normal"] };
const PRESETS = {
  easy: { label: "Easy", kind: "grid2d", size: 5, unique: true, options: {} },
  medium: { label: "Medium", kind: "grid2d", size: 7, unique: true, options: {} },
  hard: { label: "Hard", kind: "walls", size: 8, unique: true, options: { walls_frac: 0.35 } },
  expert: { label: "Expert", kind: "islands", size: 4, unique: true, options: {} },
  insane: { label: "Insane", kind: "grid3d", size: 4, unique: false, options: {} },
};
const KIND_MODE = { grid2d: "classic", walls: "walls", islands: "islands", grid3d: "cube" };
const RACE_PUZZLE = { kind: "grid2d", size: 6, unique: true };
const ROBOTS = [
  { id: "rookie", name: "Rookie", emoji: "🐣", line: "Fast but reckless", mode: "greedy", pace: 430,
    say: { go: ["zoom!", "wheee!", "easy peasy", "go go go"], win: "Done! 🎉", lose: "no fair! 😤", stuck: "uh-oh… stuck 😵" } },
  { id: "scout", name: "Scout", emoji: "🦊", line: "Careful. Backs up when stuck", mode: "search", pace: 560,
    say: { go: ["sniff sniff", "this way…", "hmm, yes", "tracking…"], win: "Found it! 🦊", lose: "well played!", stuck: "lost the trail…", back: "backing up…" } },
  { id: "grandmaster", name: "Grandmaster", emoji: "🦉", line: "Never loses. Just slow", mode: "hybrid", pace: 980,
    say: { go: ["indeed.", "as foreseen", "patience…", "hoo hoo"], win: "Inevitable. 🦉", lose: "…impressive.", stuck: "curious…" } },
];
const WIN_TITLES = ["Nice!", "Brilliant!", "Smooth!", "Nailed it!", "Zipped!", "Beautiful!"];
const EPOCH = new Date(2026, 0, 1);

const IC = {
  back: '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 18l-6-6 6-6"/></svg>',
  menu: '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
};

// ------------------------------------------------------------------ state
const G = {
  screen: "home", src: null, data: null, id: null, P: null,
  t0: null, tEnd: null, hints: 0, undos: 0, assisted: false, won: false, busy: false,
  hint: null, animating: false, animTok: 0, combo: 0, lastPush: 0, seed: 0, bad: 0,
  race: null, view: "flat", has3d: false,
};
let board = null, game = null, fx = null, view = null, viewGen = 0, miniBoard = null;

// ------------------------------------------------------------------ small helpers
const today = () => localDateISO();
function dailyNumber(iso = today()) {
  const [y, m, d] = iso.split("-").map(Number);
  return Math.round((new Date(y, m - 1, d) - EPOCH) / 86400000) + 1;
}
let toastT = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2000);
}
let coachT = null;
function coach(msg, ms = 0, tone = "") {
  const c = $("coach");
  clearTimeout(coachT);
  if (!msg) { c.classList.remove("show"); return; }
  c.innerHTML = msg; c.className = `coach show ${tone}`;
  if (ms) coachT = setTimeout(() => c.classList.remove("show"), ms);
}
function setHash(hash, push = false) {
  try { history[push ? "pushState" : "replaceState"]({ zip: 1 }, "", hash || location.pathname); } catch { /* ignore */ }
}

// ------------------------------------------------------------------ records (shared with /lab: "zip-records")
const REC = "zip-records";
function records() {
  const r = store.get(REC, null) || {};
  r.best = r.best || {}; r.solved = r.solved || {}; r.daily = r.daily || {}; r.race = r.race || { w: 0, l: 0 };
  return r;
}
function streakInfo(r = records()) {
  const t = today();
  const on = (d) => r.daily[d] && Object.keys(r.daily[d]).length > 0;
  let day = on(t) ? t : addDays(t, -1), n = 0;
  while (on(day)) { n++; day = addDays(day, -1); }
  return { current: n, today: on(t) };
}

// ------------------------------------------------------------------ screens
function showScreen(name) {
  G.screen = name;
  document.body.classList.remove("screen-home", "screen-game", "screen-rivals");
  document.body.classList.add(`screen-${name}`);
  $("home").hidden = name !== "home";
  $("game").hidden = name !== "game";
  $("rivals").hidden = name !== "rivals";
  closeMenu();
  window.scrollTo(0, 0);
}
function stopEverything() {
  G.animTok++; G.animating = false;
  if (game) game.locked = false;
  endRace();
  closeSheet();
  coach("");
  $("countdown").className = "countdown g-count";
}
function goHome(push = false) {
  stopEverything();
  disposeView();
  showScreen("home");
  renderHome();
  document.title = "Zip";
  if (push !== null) setHash("", push);
}
function back() {
  if (history.state && history.state.zip && history.length > 1 && G.cameFromHome) { G.cameFromHome = false; history.back(); }
  else goHome(false);
}

// ------------------------------------------------------------------ home
function renderHome() {
  const hr = new Date().getHours();
  const st = streakInfo();
  $("hello").textContent = hr < 5 ? "Up late? 🌙" : hr < 12 ? "Good morning ☀️" : hr < 18 ? "Good afternoon 👋" : "Good evening 🌙";
  const num = dailyNumber();
  $("heroTitle").textContent = `#${num}`;
  const done = records().daily[today()]?.medium;
  $("heroSub").innerHTML = done ? `Solved in <b>${fmtTime(done.ms)}</b>. See you tomorrow!` : "One line. Every square. Go!";
  $("heroPlayTxt").textContent = done ? "Play again" : "Play";
  $("hero").classList.toggle("done", !!done);
  const s = $("streak");
  s.hidden = !st.current;
  s.innerHTML = `<span class="flame" aria-hidden="true">🔥</span>${st.current}`;
  s.title = `${st.current}-day streak`;
  s.setAttribute("aria-label", `${st.current}-day streak`);
  const cd = store.get("zip-classic-diff", "medium");
  document.querySelectorAll(".m-classic .diff").forEach((b) => b.classList.toggle("on", b.dataset.diff === cd));
}
function buildArt() {
  const cells = (n, x0, y0, s, gap, cls = "a-cell") => {
    let out = "";
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) out += `<rect class="${cls}" x="${x0 + c * s + gap}" y="${y0 + r * s + gap}" width="${s - 2 * gap}" height="${s - 2 * gap}" rx="${s * 0.22}"/>`;
    return out;
  };
  const dot = (x, y, k, r = 8) => `<g class="a-dot" transform="translate(${x} ${y})"><circle r="${r}"/><text y="${r * 0.36}" font-size="${r * 1.05}">${k}</text></g>`;
  const grad = (id) => `<defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff9f1c"/><stop offset=".35" stop-color="#ff5a4e"/><stop offset=".65" stop-color="#d63a8f"/><stop offset="1" stop-color="#6d4ae0"/></linearGradient></defs>`;
  const A = {};
  // classic: 4x4 snake
  A.classic = `<svg viewBox="0 0 100 100">${grad("ag1")}${cells(4, 6, 6, 22, 1.6)}
    <path class="a-path" stroke="url(#ag1)" d="M17 17 H83 V39 H17 V61 H83 V83 H17" pathLength="1"/>
    ${dot(17, 17, 1)}${dot(83, 39, 2)}${dot(17, 61, 3)}${dot(17, 83, 4)}</svg>`;
  // walls
  A.walls = `<svg viewBox="0 0 100 100">${grad("ag2")}${cells(4, 6, 6, 22, 1.6)}
    <path class="a-path" stroke="url(#ag2)" d="M17 17 V83 H39 V17 H61 V83 H83 V17" pathLength="1"/>
    <path class="a-wall" d="M28 6 V72 M50 28 V94 M72 6 V72"/>
    ${dot(17, 17, 1)}${dot(39, 17, 2)}${dot(83, 17, 3)}</svg>`;
  // islands: sea, organic islands (same outline code as the board), a plank bridge, a palm
  A.islands = (() => {
    const S = 14;
    const isl = (cells, ox, oy, land) => {
      const loops = outlineLoops(cells);
      const sh = (off, rad) => loops.map((lp) => roundedPath(offsetLoop(lp, off).map(([r, c]) => [ox + c * S, oy + r * S]), rad * S)).join(" ");
      return `<path d="${sh(0.2, 0.3)}" fill="var(--water-deep)" opacity=".25" transform="translate(0 1.4)"/>
        <path d="${sh(0.34, 0.5)}" fill="none" stroke="var(--foam)" stroke-width="1" opacity=".55"/>
        <path d="${sh(0.2, 0.3)}" fill="var(--sand)" stroke="var(--sand-2)" stroke-width=".6"/><path d="${sh(0.03, 0.26)}" fill="var(${land})"/>
        ${cells.map(([r, c]) => `<rect x="${ox + c * S + 0.8}" y="${oy + r * S + 0.8}" width="${S - 1.6}" height="${S - 1.6}" rx="3" fill="var(--tile)" stroke="var(--tile-edge)" stroke-width=".5"/>`).join("")}`;
    };
    const planks = [];
    for (let x = 36; x <= 64; x += 3.2) planks.push(`<rect x="${x.toFixed(1)}" y="52.6" width="2" height="8.8" rx=".6" fill="var(--wood)"/>`);
    const waves = [18, 44, 84].map((y, i) => `<path class="a-sea-wave" style="animation-delay:${-i}s" d="M-10 ${y} q6 -2.5 12 0 t12 0 t12 0 t12 0 t12 0 t12 0 t12 0 t12 0 t12 0 t12 0"/>`).join("");
    return `<svg viewBox="2 4 96 66">${grad("ag3")}${waves}
      ${isl([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]], 6, 22, "--land-0")}
      ${isl([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]], 66, 22, "--land-2")}
      <rect x="33" y="52" width="36" height="10" fill="var(--water-deep)" opacity=".2" transform="translate(0 1.5)"/>
      ${planks.join("")}<path d="M34 52.8 H68 M34 61.2 H68" stroke="var(--rope)" stroke-width=".9"/>
      <g class="a-palm"><path d="M5 22 Q4 15 7.5 10" stroke="var(--wood-2)" stroke-width="1.8" fill="none" stroke-linecap="round"/>
        <path d="M7.5 10 q-5 -1 -8 3 M7.5 10 q-2 -4 -6 -4.5 M7.5 10 q2 -4.5 7 -4 M7.5 10 q5 0 7 4.5" stroke="var(--leaf)" stroke-width="2" fill="none" stroke-linecap="round"/></g>
      <path class="a-path" stroke="url(#ag3)" stroke-width="6" d="M27 29 H13 V43 H27 V57 H73 V43 V29 H87 V43" pathLength="1"/>
      ${dot(27, 29, 1, 6)}${dot(27, 57, 2, 6)}${dot(87, 43, 3, 6)}</svg>`;
  })();
  // cube: CSS 3D
  const face = (cls, d = "") => `<div class="cf ${cls}"><svg viewBox="0 0 60 60">${cells(3, 0, 0, 20, 1.4, "a-cell")}${d ? `<path class="a-path solid" stroke="url(#ag1)" d="${d}"/>` : ""}</svg></div>`;
  A.cube = `<div class="cube-scene"><div class="cube3">${face("f1", "M10 10 H50 V30 H10 V50 H60")}${face("f2", "M0 50 H10 V10 H30 V50 H50 V30")}${face("f3")}${face("f4")}${face("f5", "M10 50 V30 H50 V10")}${face("f6")}</div></div>`;
  A.race = `<div class="race-art">${ROBOTS.map((b, i) => `<span class="ra-av" style="--i:${i}">${b.emoji}</span>`).join("")}<span class="ra-vs">VS</span><span class="ra-you">🙂</span></div>`;
  document.querySelectorAll("[data-art]").forEach((el) => { el.innerHTML = A[el.dataset.art] || ""; });
  // hero art: 5x5 translucent grid
  const hc = document.querySelector(".hero-cells"), hd = document.querySelector(".hero-dots");
  let cellsHtml = "";
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) cellsHtml += `<rect x="${7 + c * 28}" y="${7 + r * 28}" width="22" height="22" rx="6"/>`;
  hc.innerHTML = cellsHtml;
  hd.innerHTML = [[18, 18, 1], [102, 46, 2], [18, 74, 3], [18, 102, 4]].map(([x, y, k]) => `<g transform="translate(${x} ${y})"><circle r="8.5"/><text y="3.6" text-anchor="middle">${k}</text></g>`).join("");
}

// ------------------------------------------------------------------ puzzle loading
function titleFor(src, d) {
  const shape = d.meta && d.meta.shape ? d.meta.shape.join("×") : "";
  if (src.type === "daily") return [`Zip #${src.number}`, `Daily · ${src.label}`];
  if (src.type === "mode") {
    const m = MODES[src.mode], df = m.diffs[src.diff];
    const isl = d.meta && d.meta.island ? new Set(d.meta.island).size : 0;
    const sub = src.mode === "classic" ? `${df.label} · ${shape}` : isl ? `${isl} islands` : src.mode === "walls" ? `${shape} · mind the walls` : shape ? shape.replace(/×/g, " × ") : df.label;
    return [m.label, sub];
  }
  if (src.type === "race") return [`vs ${src.bot.name}`, `Race · ${shape}`];
  if (src.type === "custom") return [src.name || "Custom puzzle", "Custom puzzle"];
  return ["Zip", shape];
}
function loadPuzzle(res, src) {
  G.animTok++; G.animating = false;
  const d = res.puzzle;
  G.data = d; G.id = res.id || null; G.src = src;
  G.P = makeModel(d);
  G.seed = res.seed ?? (d.checkpoints.reduce((a, b) => a * 31 + b, 7) >>> 0);
  G.t0 = null; G.tEnd = null; G.hints = 0; G.undos = 0; G.assisted = false; G.won = false; G.hint = null; G.combo = 0; G.bad = 0;
  board.setPuzzle(G.P);
  game.setPuzzle(G.P);
  game.locked = false;
  const [t, sub] = titleFor(src, d);
  $("gTitle").textContent = t; $("gSub").textContent = sub;
  document.title = t.startsWith("Zip") ? t : `${t} · Zip`;
  document.body.classList.toggle("racing", src.type === "race");
  document.body.classList.toggle("dim3", G.P.dim >= 3);
  $("raceStrip").hidden = src.type !== "race";
  closeSheet();
  showScreen("game");
  G.view = "flat";
  paintViewSeg();
  coach("");
  layout();
  setupView();
  if (src.type !== "race" && !store.get("zip-first-hint", false)) { coach("Start at <b>1</b> and drag ✍️", 3500); store.set("zip-first-hint", true); }
}

let loaderT = null;
function loading(on) {
  G.busy = on;
  clearTimeout(loaderT);
  if (on) loaderT = setTimeout(() => document.body.classList.add("loading"), 220);
  else { document.body.classList.remove("loading"); if (G.P) updateHud(); }
}
async function fetchAndLoad(fn, src, hash, push) {
  if (G.busy) return false;
  stopEverything();
  loading(true);
  try {
    const res = await fn();
    const s = typeof src === "function" ? src(res) : src;
    loadPuzzle(res, s);
    if (hash) setHash(typeof hash === "function" ? hash(res) : hash, push);
    if (push) G.cameFromHome = true;
    return true;
  } catch (e) {
    toast(`Couldn't load a puzzle: ${e.message}`);
    if (G.screen !== "game") goHome(null);
    return false;
  } finally { loading(false); }
}
function genParams(p, seed) { return { kind: p.kind, size: p.size, unique: !!p.unique, options: p.options || {}, seed: seed ?? null, time_limit: 20 }; }

function playMode(mode, diff, seed = null, push = false) {
  const m = MODES[mode];
  if (!m) return goHome();
  if (!m.diffs[diff]) diff = mode === "classic" ? store.get("zip-classic-diff", m.dflt) : m.dflt;
  if (!m.diffs[diff]) diff = m.dflt;
  if (mode === "classic") store.set("zip-classic-diff", diff);
  return fetchAndLoad(() => api("/api/generate", genParams(m.diffs[diff], seed)), { type: "mode", mode, diff },
    (res) => `#play=${mode}&diff=${diff}&seed=${res.seed}`, push);
}
function playDaily(diff = "medium", date = today(), push = false) {
  if (!PRESETS[diff]) diff = "medium";
  return fetchAndLoad(() => api(`/api/daily?difficulty=${encodeURIComponent(diff)}&date=${encodeURIComponent(date)}`),
    (res) => ({ type: "daily", ...res.daily }), (res) => `#daily=${diff}&date=${res.daily.date}`, push);
}
async function playCustom(id, ai = false, push = false) {
  const ok = await fetchAndLoad(async () => {
    const res = await api(`/api/custom/${encodeURIComponent(id)}`);
    return { id: null, seed: null, puzzle: res.puzzle, name: res.name };
  }, (res) => ({ type: "custom", id, name: res.name }), `#custom=${encodeURIComponent(id)}`, push);
  if (ok && ai) setTimeout(() => showMe(), 650);
  return ok;
}
function playParams(p, mode, seed, hash) {
  // old links: a preset or a basic #kind= link -> same puzzle, shown in the nearest mode's shell
  const diff = mode === "classic" ? (p.size <= 5 ? "easy" : p.size >= 9 ? "hard" : "medium") : "normal";
  return fetchAndLoad(() => api("/api/generate", genParams(p, seed)), { type: "mode", mode, diff }, hash);
}

// ------------------------------------------------------------------ layout + render
function layout() {
  if (!G.P || G.screen !== "game") return;
  const wrap = $("boardWrap");
  const dual = G.P.dim >= 3 && G.has3d && innerWidth >= 1000;
  const show3d = G.P.dim >= 3 && G.has3d && (dual || G.view === "3d");
  $("v3Host").hidden = !show3d;
  $("boardHost").hidden = show3d && !dual;
  wrap.classList.toggle("dual", dual);
  $("viewSeg").hidden = !(G.P.dim >= 3 && G.has3d && !dual);
  wrap.style.flex = ""; wrap.style.height = "";
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (!$("boardHost").hidden) board.build(dual ? (W - 24) / 2 : W, H);
  else if (!board.svg) board.build(W, H);
  // flat boards: hug the board so the buttons sit right under it (the whole group is centred)
  if (!show3d && board.svg) {
    const bh = board.svg.getBoundingClientRect().height;
    if (bh > 0 && bh < H - 8) { wrap.style.flex = "0 0 auto"; wrap.style.height = `${Math.ceil(bh) + 8}px`; }
  }
  if (G.race) layoutMini();
  render(false);
}
function render(animate = true) {
  if (!G.P || !board.svg) return;
  const legal = G.won ? [] : game.legalMoves();
  board.render({ path: game.path, won: G.won, legal, showLegal: G.P.dim >= 3 && !G.animating, hint: G.hint, animate: animate && !reducedMotion() });
  if (view) {
    try { view.update({ path: game.path, legal: G.animating ? [] : legal, hint: G.hint ? G.hint.node : null, stuck: null, colorAt: (i) => board.colorAt(i) }); } catch { /* ignore */ }
  }
  updateHud();
}
function updateHud() {
  const t = G.t0 == null ? 0 : (G.tEnd || performance.now()) - G.t0;
  $("tTime").textContent = fmtTime(t);
  const n = G.P ? G.P.n : 1;
  $("gProg").style.width = `${(100 * (game.path.length - 1)) / Math.max(1, n - 1)}%`;
  const lock = G.animating || G.won || (G.race && G.race.phase === "countdown");
  $("undoBtn").disabled = lock || game.path.length <= 1;
  $("restartBtn").disabled = lock || game.path.length <= 1;
  $("hintBtn").disabled = lock || G.busy;
  $("showMeBtn").disabled = G.animating || G.won;
}
setInterval(() => { if (G.screen === "game" && G.t0 != null && !G.tEnd) updateHud(); }, 250);

// ------------------------------------------------------------------ path events -> juice
function headClient() {
  const v = game.path[game.path.length - 1];
  if (!board.svg || $("boardHost").hidden) return null;
  return board.clientOf(v);
}
function onChange({ pushed, popped, cp }) {
  const now = performance.now();
  if (pushed.length) {
    if (!G.animating) {
      if (G.t0 == null && !(G.race && G.race.phase !== "running")) G.t0 = now;
      G.combo = now - G.lastPush < 260 ? G.combo + pushed.length : 0;
      G.lastPush = now;
      G.bad = 0;
    }
    const idx = game.path.length - 1;
    const col = board.colorAt(idx);
    const p = headClient();
    if (cp !== undefined) {
      sound.levelUp(cp);
      if (p) { const q = board.clientOf(G.P.cps[cp]); fx.burst(q.x, q.y, col); }
    } else {
      sound.note(idx - 1, { seed: G.seed, combo: G.combo });
      if (p) fx.sparks(p.x, p.y, col, 2 + Math.round(Math.min(G.combo, 8) / 2), 1 + Math.min(G.combo, 8) / 14);
    }
    if (G.hint && !G.animating) G.hint = null;
  }
  if (popped) {
    if (!G.animating) { G.undos++; sound.undo(); }
    G.won = false;
  }
  render(true);
}
function onBad(v) {
  board.flash(v);
  sound.bad();
  if (++G.bad >= 3) { coach("Keep going from the <b>end of your line</b>, in number order 🙂", 3200); G.bad = 0; }
}

// ------------------------------------------------------------------ win
async function onFull() {
  G.tEnd = performance.now();
  G.won = true;
  game.locked = true;
  render(false);
  if (G.src && G.src.type === "tutorial") return;
  let ok = true;
  try { ok = (await api("/api/check", { puzzle: G.data, path: game.path })).valid; } catch { /* offline: trust local rules */ }
  if (!ok) { G.won = false; G.tEnd = null; game.locked = false; coach("Hmm, that's not quite it. Try again!", 2500); render(false); return; }
  const tok = G.animTok;
  const ms = G.t0 == null ? 0 : G.tEnd - G.t0;
  if (G.race && raceUserFinished(ms)) return;
  await cinematic();
  if (tok !== G.animTok) return;
  showWin(ms);
}
async function cinematic() {
  const path = game.path, n = path.length;
  sound.win();
  if (reducedMotion()) { await sleep(250); return; }
  const step = Math.min(28, 700 / n);
  (board.segs || []).forEach((s, i) => { if (!s) return; s.style.animationDelay = `${i * step}ms`; s.classList.remove("lit"); void s.getBBox(); s.classList.add("lit"); });
  path.forEach((v, i) => { const t = board.tints[v]; if (!t) return; t.style.animationDelay = `${i * step}ms`; t.classList.add("ripple"); });
  (board.cpEls || []).forEach((g, k) => { const inner = g.firstChild; const i = path.indexOf(G.P.cps[k]); inner.style.animationDelay = `${i * step}ms`; inner.classList.remove("pop"); void inner.getBBox(); inner.classList.add("cheer"); });
  $("boardWrap").classList.add("victory");
  const r = board.svg && !$("boardHost").hidden ? board.svg.getBoundingClientRect() : $("boardWrap").getBoundingClientRect();
  setTimeout(() => fx.confetti(r), n * step * 0.6);
  await sleep(n * step + 650);
  $("boardWrap").classList.remove("victory");
}
function showWin(ms) {
  const src = G.src || {}, r = records();
  const rows = [];
  let badge = "🎉", title = WIN_TITLES[Math.floor(Math.random() * WIN_TITLES.length)], kick = "";
  let streak = 0;
  if (G.assisted) {
    badge = "🤖"; title = "Robot to the rescue!";
    rows.push(`<span class="pill-s">Robot solves don't count. Next one's yours 💪</span>`);
  } else {
    if (!G.hints && !G.undos) { title = "Flawless!"; badge = "⚡\uFE0F"; }
    let key = null;
    if (src.type === "daily") {
      key = `daily-${src.difficulty}`;
      r.daily[src.date] = r.daily[src.date] || {};
      if (!r.daily[src.date][src.difficulty]) r.daily[src.date][src.difficulty] = { ms, hints: G.hints, undos: G.undos };
    } else if (src.type === "mode") key = `${src.mode}-${src.diff}`;
    if (key) {
      r.solved[key] = (r.solved[key] || 0) + 1;
      if (!G.hints) {
        const prev = r.best[key];
        if (!prev || ms < prev.ms) { r.best[key] = { ms, date: today() }; if (prev) { badge = "🏆"; rows.push(`<span class="pill-s gold">🏆 New best! (was ${fmtTime(prev.ms)})</span>`); } }
      }
    }
    const st = streakInfo(r);
    r.bestStreak = Math.max(r.bestStreak || 0, st.current);
    store.set(REC, r);
    if (src.type === "daily") { streak = st.current; rows.unshift(`<span class="pill-s fire"><span class="flame">🔥</span> ${st.current}-day streak</span>`); }
    if (G.hints) rows.push(`<span class="pill-s">💡 ${G.hints} hint${G.hints > 1 ? "s" : ""}</span>`);
  }
  kick = src.type === "daily" ? `Zip #${src.number} · ${src.label}` : $("gTitle").textContent + ($("gSub").textContent ? ` · ${$("gSub").textContent}` : "");
  const share = G.assisted ? null : shareText(ms, streak);
  openSheet({
    badge, kick, title, time: G.assisted ? "" : fmtTime(ms), rows,
    primary: ["Next puzzle", nextPuzzle],
    alt: G.assisted ? ["Try it myself", restart] : null,
    share,
  });
}
function shareText(ms, streak) {
  const s = G.src || {};
  const head = s.type === "daily" ? `Zip #${s.number} · ${s.label}` : `Zip · ${$("gTitle").textContent}${s.type === "mode" && s.mode === "classic" ? " " + MODES.classic.diffs[s.diff].label : ""}`;
  const bar = "🟧🟥🟪🟦".repeat(2);
  return `${head}\n${bar}\n⚡ ${fmtTime(ms)}${G.hints ? ` · 💡${G.hints}` : " · no hints"}${streak > 1 ? ` · 🔥${streak}` : ""}`;
}
function nextPuzzle() {
  const s = G.src || {};
  closeSheet();
  if (s.type === "mode") return playMode(s.mode, s.diff);
  if (s.type === "daily") { const [m, d] = PRESET_MODE[s.difficulty] || ["classic", "medium"]; return playMode(m, d); }
  if (s.type === "race") return startRace(s.bot.id);
  return playMode("classic", store.get("zip-classic-diff", "medium"));
}
function restart() {
  closeSheet();
  if (!G.P) return;
  G.animTok++; G.animating = false;
  game.reset(); game.locked = false;
  G.won = false; G.tEnd = null; G.t0 = null; G.hint = null; G.assisted = false; G.undos = 0;
  coach("");
  render(false);
}

// ------------------------------------------------------------------ sheet
let sheetShare = null, lastFocus = null;
function openSheet({ badge, kick, title, time, rows = [], primary, alt, share, menu = true }) {
  $("shBadge").textContent = badge;
  $("shKick").textContent = kick || "";
  $("shTitle").textContent = title;
  $("shTime").textContent = time || "";
  $("shTime").hidden = !time;
  $("shRow").innerHTML = rows.join("");
  $("shPrimary").textContent = primary[0];
  $("shPrimary").onclick = primary[1];
  $("shAlt").hidden = !alt;
  if (alt) { $("shAlt").textContent = alt[0]; $("shAlt").onclick = alt[1]; }
  sheetShare = share;
  $("shShare").hidden = !share;
  $("shMenu").hidden = !menu;
  lastFocus = document.activeElement;
  $("sheet").hidden = false;
  requestAnimationFrame(() => $("sheet").classList.add("show"));
  setTimeout(() => $("shPrimary").focus({ preventScroll: true }), 60);
}
function closeSheet() {
  const s = $("sheet");
  if (s.hidden) return;
  s.classList.remove("show");
  s.hidden = true;
}

// ------------------------------------------------------------------ hint / undo / show me
async function getHint() {
  if (!G.data || G.won || G.animating || G.busy) return;
  if (G.race && G.race.phase === "countdown") return;
  G.busy = true; updateHud();
  const tok = G.animTok, rev = game.rev;
  try {
    const r = await api("/api/hint", { puzzle: G.data, path: game.path, id: G.id });
    if (tok !== G.animTok || rev !== game.rev) return;
    if (r.status === "next") {
      G.hints++; G.hint = { node: r.next }; sound.hint();
      coach("Try the glowing square ✨", 2600);
    } else if (r.status === "backtrack") {
      G.hints++;
      G.animating = true;
      while (game.path.length > Math.max(1, r.keep)) { game.truncate(game.path.length - 1); sound.undo(); await sleep(45); if (tok !== G.animTok) return; }
      G.animating = false;
      G.hint = { node: r.next }; sound.hint();
      coach("Dead end! I rewound you a bit. Try the glow ✨", 3200, "warn");
    } else if (r.status === "done") coach("Already solved! 🎉", 2000);
    else coach("Hmm, no hint right now. Try Undo 🙂", 2500);
    render(false);
  } catch { coach("Couldn't get a hint 😕", 2200); }
  finally { G.busy = false; G.animating = false; updateHud(); }
}
function undo() { if (!G.animating && !G.won) game.undo(); }

async function solveFull() {
  const path = game.path;
  if (path.length > 1) {
    try { const r = await api("/api/solve/exact", { puzzle: G.data, start_path: path, time_limit: 6 }); if (r.solved && r.path) return r.path; } catch { /* fall through */ }
  }
  for (const tl of [20, 45]) {
    try { const r = await api("/api/solve/exact", { puzzle: G.data, time_limit: tl }); if (r.solved && r.path) return r.path; } catch { /* retry */ }
  }
  return null;
}
async function showMe() {
  if (!G.data || G.won || G.animating) return;
  if (G.race && G.race.phase !== "done") return;
  const tok = ++G.animTok;
  G.animating = true; game.locked = true; G.hint = null;
  document.body.classList.add("robot-on");
  coach(`<span class="bot-ic">🤖</span> Let me show you…`);
  updateHud();
  const full = await solveFull();
  if (tok !== G.animTok) return;
  if (!full) {
    G.animating = false; game.locked = false; document.body.classList.remove("robot-on");
    coach("Even the robot is stumped 🤔", 2600); updateHud(); return;
  }
  G.assisted = true;
  sound.robot();
  // rewind to the part of your line that is on the solution, then draw the rest
  let k = 0;
  while (k < game.path.length && k < full.length && game.path[k] === full[k]) k++;
  while (game.path.length > Math.max(1, k)) { game.truncate(game.path.length - 1); sound.undo(); await sleep(30); if (tok !== G.animTok) return; }
  const rest = full.length - game.path.length;
  const dt = clamp(2600 / Math.max(1, rest), 35, 150);
  coach(`<span class="bot-ic">🤖</span> Beep boop…`);
  for (let i = game.path.length; i < full.length; i++) {
    if (tok !== G.animTok) return;
    if (i === full.length - 1) { G.animating = false; coach(""); document.body.classList.remove("robot-on"); }
    game.push(full[i]);
    await sleep(dt);
  }
  G.animating = false;
  document.body.classList.remove("robot-on");
}

// ------------------------------------------------------------------ 3D / 4D views
function viewTheme() { return { bg: cssVar("--panel-2"), cell: cssVar("--cell-edge"), edge: cssVar("--border"), legal: cssVar("--legal"), good: cssVar("--good"), bad: cssVar("--bad"), plate: cssVar("--cell"), heat: cssVar("--heat"), accent: cssVar("--accent") }; }
function disposeView() {
  viewGen++;
  if (view) { try { view.dispose(); } catch { /* ignore */ } view = null; }
  $("v3Host").innerHTML = "";
  $("v3Host").className = "v3-host";
  G.has3d = false;
}
async function setupView() {
  disposeView();
  const gen = viewGen;
  if (!G.P || G.P.dim < 3) return;
  G.has3d = true;
  layout();
  try {
    const mod = await import(G.P.dim === 3 ? "./view3d.js" : "./view4d.js");
    if (gen !== viewGen) return;
    const make = G.P.dim === 3 ? mod.createView3D : mod.createView4D;
    const host = $("v3Host");
    host.classList.add("zv-compact");
    const v = await make(host, { coords: G.P.coords, edges: G.data.edges, cps: G.P.cps, n: G.P.n, onClick: (node) => game.onDown(node), theme: viewTheme });
    if (gen !== viewGen) { v.dispose(); return; }
    view = v;
    render(false);
  } catch (e) {
    if (gen !== viewGen) return;
    G.has3d = false; G.view = "flat";
    layout();
  }
}
function paintViewSeg() {
  document.querySelectorAll("#viewSeg button").forEach((b) => { const on = b.dataset.v === G.view; b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on)); });
  const b3 = document.querySelector('#viewSeg [data-v="3d"]');
  if (b3 && G.P) b3.textContent = G.P.dim === 4 ? "4D" : "3D";
}

// ------------------------------------------------------------------ race
function botSay(txt, mood = "") {
  const b = $("bot");
  $("botSay").textContent = txt;
  b.className = `bot ${mood}`;
  $("botSay").classList.remove("pop"); void $("botSay").offsetWidth; $("botSay").classList.add("pop");
}
function showRivals(push = false) {
  stopEverything();
  disposeView();
  showScreen("rivals");
  document.title = "Race the robot · Zip";
  const r = records();
  const vs = r.raceBots || {};
  $("rivalList").innerHTML = "";
  ROBOTS.forEach((b, i) => {
    const rec = vs[b.id];
    $("rivalList").appendChild(h("button", { class: `rival r-${b.id}`, type: "button", style: `--i:${i}`, onclick: () => startRace(b.id, true) },
      h("span", { class: "rival-av", "aria-hidden": "true" }, b.emoji),
      h("span", { class: "rival-txt" }, h("b", {}, b.name), h("span", {}, b.line),
        rec && (rec.w || rec.l) ? h("span", { class: "rival-rec" }, `You ${rec.w} – ${rec.l} ${b.name}`) : null),
      h("span", { class: "rival-go", "aria-hidden": "true" }, "›")));
  });
  setHash("#race", push);
  if (push) G.cameFromHome = true;
}
function layoutMini() {
  if (!G.race || !G.P) return;
  if (!miniBoard) miniBoard = new Board($("botBoard"), { mini: true, maxCell: 14, label: "The robot's board" });
  miniBoard.setPuzzle(G.P);
  const s = innerWidth < 600 ? 64 : 84;
  miniBoard.build(s, s);
  renderMini();
}
function renderMini() {
  const R = G.race;
  if (!R || !miniBoard || !miniBoard.svg) return;
  miniBoard.render({ path: R.aiPath, won: R.aiSolved, legal: [], tried: R.tried, animate: false });
  $("botProg").style.width = `${(100 * (R.aiPath.length - 1)) / Math.max(1, G.P.n - 1)}%`;
}
function endRace() {
  const R = G.race;
  if (!R) return;
  R.phase = "done";
  R.timers.forEach(clearTimeout); R.timers = [];
  G.race = null;
  document.body.classList.remove("racing");
}
async function fetchPlan(bot, data) {
  try {
    return await api("/api/solve/rl", { puzzle: data, mode: bot.mode, budget: 4000, time_limit: 20, compare: false, trace: bot.mode === "search" });
  } catch {
    // no trained model available: the exact solver plays, and the Rookie keeps its reckless streak
    const r = await api("/api/solve/exact", { puzzle: data, time_limit: 20 });
    if (bot.mode === "greedy" && r.path && Math.random() < 0.45) {
      const cut = Math.floor(r.path.length * (0.5 + Math.random() * 0.35));
      return { solved: false, path: r.path.slice(0, cut), seconds: r.seconds };
    }
    return { solved: !!r.solved, path: r.path || [data.checkpoints[0]], seconds: r.seconds };
  }
}
async function startRace(botId, push = false) {
  const bot = ROBOTS.find((b) => b.id === botId) || ROBOTS[0];
  stopEverything();
  store.set("zip-opp2", bot.id);
  const ok = await fetchAndLoad(() => api("/api/generate", genParams(RACE_PUZZLE)), { type: "race", bot }, `#race=${bot.id}`, push);
  if (!ok) return;
  const R = G.race = { bot, phase: "countdown", aiPath: [G.P.cps[0]], aiSolved: false, aiStuck: false, timers: [], tried: null, data: G.data };
  document.body.classList.add("racing");
  $("raceStrip").hidden = false;
  $("botAv").textContent = bot.emoji;
  botSay("ready?", "idle");
  game.locked = true;
  layout();
  const plan = fetchPlan(bot, G.data).catch((e) => ({ error: e.message }));
  const cd = $("countdown");
  for (const k of ["3", "2", "1"]) {
    if (G.race !== R) return;
    cd.textContent = k; cd.className = "countdown g-count show"; void cd.offsetWidth; sound.count(false);
    await sleep(reducedMotion() ? 350 : 700);
    cd.className = "countdown g-count";
  }
  if (G.race !== R) return;
  cd.textContent = "Go!"; cd.className = "countdown g-count show go"; sound.count(true);
  setTimeout(() => { cd.className = "countdown g-count"; }, 500);
  R.phase = "running"; R.t0 = performance.now();
  G.t0 = R.t0; G.tEnd = null;
  game.locked = false;
  updateHud();
  botSay("hmm…", "thinking");
  const res = await plan;
  if (G.race !== R || R.phase !== "running") return;
  if (res.error || !res.path) { botSay("my circuits! 🔌", "sad"); R.aiStuck = true; return; }
  const events = [];
  if (bot.mode === "search" && res.trace) {
    let stack = [];
    for (const ev of res.trace) {
      if (ev === "R") { stack = []; continue; }
      if (ev >= 0) { stack.push(ev); if (stack.length > 1) events.push({ path: stack.slice(), dt: bot.pace }); }
      else { const popped = stack.splice(stack.length + ev); events.push({ path: stack.slice(), dt: bot.pace * 0.6, popped }); }
    }
  } else for (let i = 1; i < res.path.length; i++) events.push({ path: res.path.slice(0, i + 1), dt: bot.pace });
  const total = events.reduce((a, e) => a + e.dt, 0), cap = 2.5 * G.P.n * bot.pace;
  if (total > cap) { const f = cap / total; for (const e of events) e.dt *= f; }
  const think = bot.mode === "hybrid" ? Math.max(900, (res.seconds || 0) * 1000) : 350;
  let k = 0;
  const step = () => {
    if (G.race !== R || R.phase !== "running") return;
    const e = events[k++];
    if (e) {
      R.aiPath = e.path;
      if (e.popped) {
        R.tried = R.tried || new Float32Array(G.P.n); for (const v of e.popped) R.tried[v] = 1;
        if (!R.backing) botSay(bot.say.back || "oops…", "confused");
        R.backing = true;
      } else {
        if (R.backing || k === 1 || k % 9 === 0) botSay(bot.say.go[Math.floor(Math.random() * bot.say.go.length)], "going");
        R.backing = false;
      }
      if (R.tried) for (let v = 0; v < G.P.n; v++) R.tried[v] *= 0.8;
      if (k % 3 === 0) sound.robot();
      renderMini();
      R.timers.push(setTimeout(step, e.dt));
      return;
    }
    R.aiPath = res.path; R.tried = null;
    if (res.solved) {
      R.aiSolved = true; R.aiMs = performance.now() - R.t0;
      renderMini();
      botSay(bot.say.win, "win");
      robotWins();
    } else {
      R.aiStuck = true;
      renderMini();
      botSay(bot.say.stuck, "sad");
      coach(`${bot.emoji} ${bot.name} got stuck! Finish to win 🏁`, 3000);
    }
  };
  R.timers.push(setTimeout(step, think));
}
function raceRecord(bot, won) {
  const r = records();
  r.race.w = (r.race.w || 0) + (won ? 1 : 0); r.race.l = (r.race.l || 0) + (won ? 0 : 1);
  r.raceBots = r.raceBots || {};
  const b = r.raceBots[bot.id] = r.raceBots[bot.id] || { w: 0, l: 0 };
  if (won) b.w++; else b.l++;
  store.set(REC, r);
  return b;
}
function robotWins() {
  const R = G.race, bot = R.bot;
  R.phase = "done"; R.timers.forEach(clearTimeout);
  sound.lose();
  const rec = raceRecord(bot, false);
  setTimeout(() => {
    if (G.screen !== "game") return;
    openSheet({
      badge: bot.emoji, kick: `Race vs ${bot.name}`, title: `${bot.name} wins!`, time: fmtTime(R.aiMs),
      rows: [`<span class="pill-s">“${bot.say.win}”</span>`, `<span class="pill-s">You ${rec.w} – ${rec.l} ${bot.name}</span>`],
      primary: ["Rematch", () => startRace(bot.id)],
      alt: ["Keep solving", () => { closeSheet(); endRace(); $("raceStrip").hidden = false; }],
      share: null,
    });
  }, 700);
}
function raceUserFinished(ms) {
  const R = G.race;
  if (!R || R.phase !== "running" || R.data !== G.data) return false;
  const bot = R.bot;
  R.phase = "done"; R.timers.forEach(clearTimeout);
  botSay(bot.say.lose, "sad");
  const rec = raceRecord(bot, true);
  cinematic().then(() => {
    openSheet({
      badge: "🏆", kick: `Race vs ${bot.emoji} ${bot.name}`, title: R.aiStuck ? `${bot.name} got stuck. You win!` : `You beat ${bot.name}!`, time: fmtTime(ms),
      rows: [`<span class="pill-s">${bot.emoji} “${bot.say.lose}”</span>`, `<span class="pill-s gold">You ${rec.w} – ${rec.l} ${bot.name}</span>`],
      primary: ["Rematch", () => startRace(bot.id)],
      alt: ["New rival", () => showRivals()],
      share: `Zip race · I beat ${bot.name} ${bot.emoji}\n⚡ ${fmtTime(ms)}${R.aiStuck ? " · it got stuck 😵" : ""}`,
    });
  });
  return true;
}

// ------------------------------------------------------------------ tutorial
const TUT = (() => {
  const coords = [], edges = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) coords.push([r, c]);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) { if (c < 2) edges.push([r * 3 + c, r * 3 + c + 1]); if (r < 2) edges.push([r * 3 + c, r * 3 + c + 3]); }
  return { kind: "grid2d", coords, edges, checkpoints: [0, 2, 3, 8], meta: { shape: [3, 3] } };
})();
const TUT_SOL = [0, 1, 2, 5, 4, 3, 6, 7, 8];
let tut = null;
function openTutorial() {
  closeMenu();
  const T = tut = { step: 0, tok: 0 };
  $("tut").hidden = false;
  requestAnimationFrame(() => $("tut").classList.add("show"));
  const tb = new Board($("tutHost"), { interactive: true, maxCell: 72, label: "Practice board", onDown: (v) => T.game.onDown(v), onDrag: (v) => T.game.onDrag(v) });
  const P = makeModel(TUT);
  tb.setPuzzle(P);
  T.board = tb;
  T.game = new PathGame(tb, {
    onChange: ({ pushed, popped, cp }) => {
      if (pushed.length) {
        const i = T.game.path.length - 1, p = tb.clientOf(T.game.path[i]), col = tb.colorAt(i);
        if (cp !== undefined) { sound.levelUp(cp); fx.burst(p.x, p.y, col, 18); } else { sound.note(i - 1, { seed: 3 }); fx.sparks(p.x, p.y, col, 4); }
        $("tutFinger").classList.remove("show", "nudge");
      }
      if (popped && T.step === 1) sound.undo();
      tb.render({ path: T.game.path, legal: T.game.legalMoves(), won: T.game.full(), showLegal: T.step === 1 && T.game.path.length === 1 });
    },
    onBad: (v) => { tb.flash(v); sound.bad(); },
    onFull: () => {
      if (T.step !== 1) return;
      T.step = 2;
      sound.win();
      fx.confetti($("tutHost").getBoundingClientRect(), 90);
      tutStep();
    },
  });
  T.game.setPuzzle(P);
  const w = Math.min(230, innerWidth - 110);
  tb.build(w, w);
  tb.render({ path: T.game.path, legal: [], animate: false });
  tutStep();
}
async function tutStep() {
  const T = tut;
  if (!T) return;
  const tok = ++T.tok;
  document.querySelectorAll(".tut-dots i").forEach((d, i) => d.classList.toggle("on", i === T.step));
  const btn = $("tutNext");
  if (T.step === 0) {
    $("tutTitle").textContent = "Welcome to Zip!";
    $("tutTxt").innerHTML = "Draw <b>one line</b> through the numbers, <b>in order</b>…";
    btn.textContent = "Your turn"; btn.onclick = () => { T.step = 1; tutStep(); };
    T.game.locked = true;
    // demo: the ghost finger draws the solution
    T.game.reset(); T.board.render({ path: T.game.path, legal: [], animate: false });
    await sleep(700);
    const finger = $("tutFinger");
    for (let i = 1; i < TUT_SOL.length; i++) {
      if (tok !== T.tok) return;
      if (i === 3) { $("tutTxt").innerHTML = "…and fill <b>every square</b>. That's it!"; }
      T.game.push(TUT_SOL[i]);
      moveFinger(finger, TUT_SOL[i]);
      await sleep(i === 2 ? 700 : 380);
    }
    await sleep(900);
    if (tok !== T.tok) return;
    finger.classList.remove("show");
  } else if (T.step === 1) {
    $("tutTitle").textContent = "Your turn!";
    $("tutTxt").innerHTML = "Drag from <b>1</b> and fill every square, ending on <b>4</b>.";
    btn.textContent = "Show me again"; btn.onclick = () => { T.step = 0; tutStep(); };
    T.game.reset(); T.game.locked = false;
    T.board.render({ path: T.game.path, legal: T.game.legalMoves(), showLegal: true, animate: false });
    await sleep(500);
    if (tok !== T.tok) return;
    moveFinger($("tutFinger"), TUT_SOL[0]);
    $("tutFinger").classList.add("nudge");
  } else {
    $("tutTitle").textContent = "You're a natural! 🎉";
    $("tutTxt").innerHTML = "Stuck later? Tap <b>Hint</b>, or let the robot <b>Show you</b> 🤖";
    btn.textContent = "Let's play"; btn.onclick = closeTutorial;
    btn.focus({ preventScroll: true });
  }
}
function moveFinger(f, v) {
  const T = tut;
  const host = $("tutBoard").getBoundingClientRect(), p = T.board.clientOf(v);
  f.style.transform = `translate(${p.x - host.left - 6}px, ${p.y - host.top - 4}px)`;
  f.classList.add("show");
  f.classList.remove("nudge");
}
function closeTutorial() {
  if (!tut) return;
  tut.tok++; tut = null;
  store.set("zip-tut-done", true);
  $("tut").classList.remove("show");
  $("tut").hidden = true;
  $("tutHost").innerHTML = "";
}

// ------------------------------------------------------------------ menu, sound, theme
function paintSound() {
  $("soundBtn").innerHTML = icon(sound.muted ? "mute" : "sound");
  $("soundBtn").setAttribute("aria-label", sound.muted ? "Turn sound on" : "Turn sound off");
  $("miSoundIc").textContent = sound.muted ? "🔇" : "🔊";
  $("miSoundTxt").textContent = sound.muted ? "Sound off" : "Sound on";
}
function toggleSound() { sound.setMuted(!sound.muted); paintSound(); toast(sound.muted ? "Sound off 🔇" : "Sound on 🔊"); }
function openMenu() { $("menu").hidden = false; $("menuBtn").setAttribute("aria-expanded", "true"); requestAnimationFrame(() => $("menu").classList.add("show")); const f = $("menu").querySelector(".mi"); if (f) f.focus({ preventScroll: true }); }
function closeMenu() { if ($("menu").hidden) return; $("menu").classList.remove("show"); $("menu").hidden = true; $("menuBtn").setAttribute("aria-expanded", "false"); }

// ------------------------------------------------------------------ keyboard
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest && ev.target.closest("input, select, textarea, [contenteditable]")) return;
  if (ev.key === "Escape") {
    if (!$("menu").hidden) { closeMenu(); $("menuBtn").focus(); ev.preventDefault(); return; }
    if (tut) { closeTutorial(); ev.preventDefault(); return; }
    if (!$("sheet").hidden) { closeSheet(); ev.preventDefault(); return; }
    if (G.screen !== "home") { back(); ev.preventDefault(); }
    return;
  }
  if (tut || !$("sheet").hidden || !$("menu").hidden) return;
  if (ev.key === "m" || ev.key === "M") { toggleSound(); ev.preventDefault(); return; }
  if (G.screen !== "game" || !G.P) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) { if ((ev.key === "z" || ev.key === "Z") && (ev.ctrlKey || ev.metaKey)) { undo(); ev.preventDefault(); } return; }
  if (ev.target.closest && ev.target.closest(".zv-host")) return;   // the 3D view has its own keys
  const k = ev.key;
  let handled = true;
  if (k === "ArrowUp") game.moveDir(-1, 0);
  else if (k === "ArrowDown") game.moveDir(1, 0);
  else if (k === "ArrowLeft") game.moveDir(0, -1);
  else if (k === "ArrowRight") game.moveDir(0, 1);
  else if (k === "q" || k === "Q" || k === "PageUp") game.moveAxis(2, -1);
  else if (k === "e" || k === "E" || k === "PageDown") game.moveAxis(2, 1);
  else if ((k === "a" || k === "A") && G.P.dim === 4) game.moveAxis(3, -1);
  else if ((k === "d" || k === "D") && G.P.dim === 4) game.moveAxis(3, 1);
  else if (k === "z" || k === "Z" || k === "Backspace") undo();
  else if (k === "h" || k === "H") getHint();
  else if (k === "r" || k === "R") { if (!G.animating) restart(); }
  else handled = false;
  if (handled) ev.preventDefault();
});

// ------------------------------------------------------------------ routing
async function route(hash, push = false) {
  const q = new URLSearchParams((hash || "").replace(/^#/, ""));
  const seed = q.get("seed") && /^\d+$/.test(q.get("seed")) ? Number(q.get("seed")) : null;
  if (q.get("custom")) return playCustom(q.get("custom"), q.get("ai") === "1", push);
  if (q.get("daily")) return playDaily(q.get("daily"), q.get("date") || today(), push);
  if (q.get("play") && MODES[q.get("play")]) return playMode(q.get("play"), q.get("diff"), seed, push);
  if (q.get("d") && PRESETS[q.get("d")]) {
    const [m] = PRESET_MODE[q.get("d")];
    return playParams(PRESETS[q.get("d")], m, seed, null);
  }
  if (q.get("kind") && KIND_MODE[q.get("kind")]) {
    const size = Number(q.get("size")) || undefined;
    const kind = q.get("kind");
    const base = kind === "grid2d" ? MODES.classic.diffs.medium : MODES[KIND_MODE[kind]].diffs.normal;
    return playParams({ ...base, size: size || base.size, unique: q.has("unique") ? true : base.unique }, KIND_MODE[kind], seed, null);
  }
  if (q.has("race")) {
    const b = q.get("race");
    if (b && ROBOTS.some((x) => x.id === b)) return showRivals(push);
    return showRivals(push);
  }
  goHome(null);
}

// ------------------------------------------------------------------ init
function init() {
  fx = createFx($("fx"));
  board = new Board($("boardHost"), { interactive: true, maxCell: 84, label: "Zip board: drag from 1 through every square. Arrow keys work too.",
    onDown: (v) => game.onDown(v), onDrag: (v) => game.onDrag(v) });
  game = new PathGame(board, { onChange, onBad, onFull });

  $("backBtn").innerHTML = IC.back;
  $("menuBtn").innerHTML = IC.menu;
  document.querySelector("#undoBtn .cb-ic").innerHTML = icon("undo");
  document.querySelector("#hintBtn .cb-ic").innerHTML = icon("bulb");
  document.querySelector("#restartBtn .cb-ic").innerHTML = icon("reset");
  paintSound();
  buildArt();
  renderHome();

  $("backBtn").onclick = back;
  $("soundBtn").onclick = toggleSound;
  $("menuBtn").onclick = (e) => { e.stopPropagation(); if ($("menu").hidden) openMenu(); else closeMenu(); };
  document.addEventListener("click", (e) => { if (!$("menu").hidden && !e.target.closest("#menu")) closeMenu(); });
  $("menu").addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    const a = b.dataset.act;
    if (a === "howto") openTutorial();
    else if (a === "sound") toggleSound();
    else if (a === "theme") window.zipTheme && window.zipTheme.toggle();
    if (a !== "sound" && a !== "theme") closeMenu();
  });
  $("labLink").addEventListener("click", () => {
    // carry the current daily / custom puzzle into the lab
    const hh = location.hash;
    if (/^#(daily|custom)=/.test(hh)) $("labLink").href = "/lab" + hh;
    else $("labLink").href = "/lab";
  });

  $("heroPlay").onclick = () => playDaily("medium", today(), true);
  $("hero").addEventListener("click", (e) => { if (!e.target.closest("button")) playDaily("medium", today(), true); });
  document.querySelectorAll("[data-play]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      const m = b.dataset.play;
      if (m === "race") return showRivals(true);
      playMode(m, b.dataset.diff || null, null, true);
    });
  });

  $("undoBtn").onclick = undo;
  $("hintBtn").onclick = getHint;
  $("restartBtn").onclick = restart;
  $("showMeBtn").onclick = showMe;
  document.querySelectorAll("#viewSeg button").forEach((b) => { b.onclick = () => { G.view = b.dataset.v; paintViewSeg(); layout(); }; });

  $("shShare").onclick = async () => {
    if (!sheetShare) return;
    try { await navigator.clipboard.writeText(sheetShare); toast("Copied! Paste it anywhere 📋"); }
    catch { toast("Couldn't copy 😕"); }
  };
  $("shMenu").onclick = () => { closeSheet(); goHome(false); };
  $("sheet").addEventListener("click", (e) => { if (e.target === $("sheet")) closeSheet(); });
  $("tutSkip").onclick = closeTutorial;

  window.addEventListener("zip-theme", () => { if (G.screen === "game") { layout(); if (view) try { view.refreshTheme(); } catch { /* ignore */ } } });
  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(layout, 120); });
  window.addEventListener("popstate", () => { route(location.hash); });

  const first = !store.get("zip-tut-done", false);
  route(location.hash).then(() => { if (first && G.screen === "home") openTutorial(); });
}

window.__zipGame = { G, game: () => game, board: () => board, tut: () => tut, showMe, getHint, route, startRace, openTutorial, closeTutorial, playMode, playDaily };
init();
