// Zip — the AI show (/lab). Pick a puzzle, watch a robot "look" at the board (the network's heat glow),
// try branches, hit dead ends and back up; or race two robots on the same puzzle. Plain words, no tables:
// the numbers live behind "Nerd mode" (and in the workbench at /workbench).
import { $, h, api, store, icon, clamp, sleep, reducedMotion, fmtInt, svgEl as el } from "./play/util.js";
import { Board, makeModel } from "./play/board.js";
import { sound } from "./play/sound.js";
import { createFx } from "./play/fx.js";
import { MODES, ROBOTS, genParams, modeArt } from "./play/modes.js";

// ------------------------------------------------------------------ the cast
// Core robots share the default trained brain and differ in how they use it; the Tortoise has no brain at all.
const R = Object.fromEntries(ROBOTS.map((b) => [b.id, b]));
const CORE = [
  { ...R.rookie, line: "Trusts its gut. Never looks back", brain: true },
  { ...R.scout, line: "Follows its brain, backs up when stuck", brain: true },
  { ...R.grandmaster, line: "Brain + solver. Never loses", brain: true },
  { id: "tortoise", name: "Tortoise", emoji: "🐢", line: "No AI at all. Tries everything", mode: "exact", brain: false,
    say: { go: ["plod plod", "next one…", "slow & steady", "hmm hm hm"], win: "Slow and steady! 🐢", lose: "I'll get there…", stuck: "…zzz", back: "nope, back…" } },
];
// other trained checkpoints become challengers (Scout-style search with a different brain); newest / best first
const ANIMALS = [["🐙", "Octo"], ["🐼", "Panda"], ["🐯", "Tiger"], ["🐸", "Ribbit"], ["🦄", "Sparkle"], ["🐨", "Koala"], ["🐧", "Waddles"], ["🐝", "Buzz"]];
const CH_LINES = ["Fresh from training", "A different brain, same tricks", "An earlier brain. Still sharp?"];
let CAST = CORE.slice();
const botById = (id) => CAST.find((b) => b.id === id) || null;

// ------------------------------------------------------------------ narration
const LINES = {
  look: ["Let me look at this one… 👀", "Ooh, a new puzzle. Scanning… 👀", "Hmm, let me see… 👀"],
  corners: ["Checking the tight corners first…", "Corners are tricky. Looking there first…"],
  target: (k) => [`Eyeing number ${k}…`, `Where's ${k}? Ah, there.`],
  vague: ["Hmm, which way…?", "So many choices…"],
  go: ["Here goes!", "Off we go! ✏️", "Let's draw!"],
  cp: (k) => [`Reached ${k}! ✔`, `${k}, check! ✔`, `Hello, ${k} 👋`],
  dead: ["Dead end! Backing up ↩", "Nope, that traps a square. Undo! ↩", "Hmm… dead end!", "Oops, no way out there ↩"],
  reset: ["Fresh start. Trying another way 🔄", "Starting over with a new idea 🔄"],
  smooth: ["Smooth sailing… ⛵", "In the zone 😎", "This part is easy 👉"],
  tough: ["Tough call here… 🤔", "Two good options… 🤔"],
  many: (n) => [`${fmtInt(n)} tries and counting…`, `Still digging… ${fmtInt(n)} tries`],
  nobrain: ["No AI brain here. I just try everything! 🐢"],
};
const pickLine = (a) => a[Math.floor(Math.random() * a.length)];

// ------------------------------------------------------------------ settings (Nerd mode; per viewer)
const NERD_KEY = "zip-show-nerd";
const NERD = Object.assign({ model: "", time: 20, budget: 4000, odds: false }, store.get(NERD_KEY, {}) || {});
function saveNerd() { store.set(NERD_KEY, NERD); }

// ------------------------------------------------------------------ state
const S = {
  screen: "pick", puzzle: null, P: null, bot: "scout", botChosen: false, watch: null, race: null,
  vs: ["rookie", "grandmaster"], models: [], dflt: null, speed: 1, cache: new Map(),
};
let fx = null, board = null;

// ------------------------------------------------------------------ small helpers
let toastT = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2200);
}
function setHash(hash, push = false) {
  try { history[push ? "pushState" : "replaceState"]({ show: 1 }, "", (hash || location.pathname) + ""); } catch { /* ignore */ }
}
const glowColor = () => getComputedStyle(document.body).getPropertyValue("--glow").trim() || "#19b8ff";

// ------------------------------------------------------------------ screens
function showScreen(name) {
  S.screen = name;
  document.body.classList.remove("screen-pick", "screen-stage", "screen-faceoff");
  document.body.classList.add(`screen-${name}`);
  $("pick").hidden = name !== "pick";
  $("stage").hidden = name !== "stage";
  $("faceoff").hidden = name !== "faceoff";
  closeMenu();
  window.scrollTo(0, 0);
  paintBack();
}
function paintBack() {
  const b = $("backBtn");
  b.innerHTML = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 18l-6-6 6-6"/></svg>';
  b.setAttribute("aria-label", S.screen === "pick" ? "Back to the game" : "Back to the AI show");
  b.href = S.screen === "pick" ? "/" : "/lab";
}
function stopAll() {
  if (S.watch) { S.watch.stop(); S.watch = null; }
  if (S.race) { S.race.stop(); S.race = null; }
  $("countdown").className = "countdown g-count";
  closeSheet("picker");
}
function goPick(push = false) {
  stopAll();
  showScreen("pick");
  $("sTitle").textContent = "AI show"; $("sSub").textContent = "Watch the robots think";
  document.title = "AI show · Zip";
  if (push !== null) setHash("", push);
}

// ------------------------------------------------------------------ models -> characters
async function loadCast() {
  try {
    const r = await api("/api/models");
    S.models = r.models || [];
  } catch { S.models = []; }
  const dflt = S.models.find((m) => m.default);
  S.dflt = dflt ? dflt.name : null;
  const brainName = NERD.model && S.models.some((m) => m.name === NERD.model) ? NERD.model : S.dflt;
  // challengers: real training runs only (no smoke tests / intermediate stages), one per file name, best tag then newest
  const tag = (f) => /best/.test(f) ? 3 : /final/.test(f) ? 2 : /latest/.test(f) ? 1 : 0;
  const seen = new Set();
  const others = S.models.filter((m) => m.name !== brainName && !/smoke|stage\d|_r\d/.test(m.file))
    .sort((a, b) => b.mtime - a.mtime)
    .filter((m) => (seen.has(m.file) ? false : (seen.add(m.file), true)))
    .sort((a, b) => tag(b.file) - tag(a.file) || b.mtime - a.mtime).slice(0, 3)
    .sort((a, b) => b.mtime - a.mtime);
  const used = new Set();
  const hash = (s) => { let x = 7; for (const ch of s) x = (x * 31 + ch.charCodeAt(0)) >>> 0; return x; };
  const ch = others.map((m, i) => {
    let k = hash(m.name) % ANIMALS.length;
    while (used.has(k)) k = (k + 1) % ANIMALS.length;
    used.add(k);
    const [emoji, name] = ANIMALS[k];
    return { id: `ch-${k}`, name, emoji, line: CH_LINES[Math.min(i, CH_LINES.length - 1)], mode: "search", brain: true, model: m.name, challenger: true,
      say: { go: ["on it!", "this way…", "hmm, yes", "zip zip"], win: `${name} wins! ${emoji}`, lose: "next time!", stuck: "out of ideas…", back: "backing up…" } };
  });
  const noBrain = !S.models.length;
  CAST = CORE.map((b) => ({ ...b, model: b.brain ? brainName : null, off: b.brain && noBrain })).concat(ch);
  if (noBrain) { S.bot = "tortoise"; S.vs = ["tortoise", "tortoise"]; }
  paintNerd();
}

// ------------------------------------------------------------------ puzzles
function randomPick() {
  const opts = [["classic", "medium"], ["classic", "hard"], ["walls", "normal"], ["islands", "normal"], ["islands", "normal"], ["cube", "normal"], ["classic", "easy"]];
  return opts[Math.floor(Math.random() * opts.length)];
}
function titleFor(src, d) {
  const shape = d.meta && d.meta.shape ? d.meta.shape.join("×") : "";
  if (src.type === "daily") return [`Zip #${src.number}`, `Daily · ${src.label}`];
  if (src.type === "custom") return [src.name || "Custom puzzle", "From the editor"];
  const m = MODES[src.mode], df = m.diffs[src.diff];
  const isl = d.meta && d.meta.island ? new Set(d.meta.island).size : 0;
  if (src.mode === "classic") return [`Classic · ${df.label}`, shape];
  if (isl) return [m.label, `${isl} islands`];
  return [m.label, shape.replace(/×/g, " × ")];
}
function beatLink(src) {
  if (src.type === "daily") return `/#daily=${encodeURIComponent(src.difficulty)}&date=${src.date}`;
  if (src.type === "custom") return `/#custom=${encodeURIComponent(src.id)}`;
  return `/#play=${src.mode}&diff=${src.diff}&seed=${src.seed}`;
}
function srcHash(src) {
  if (src.type === "daily") return `daily=${encodeURIComponent(src.difficulty)}&date=${src.date}`;
  if (src.type === "custom") return `custom=${encodeURIComponent(src.id)}`;
  return `play=${src.mode}&diff=${src.diff}&seed=${src.seed}`;
}
let loaderT = null;
function loading(on) {
  clearTimeout(loaderT);
  if (on) loaderT = setTimeout(() => document.body.classList.add("loading"), 200);
  else document.body.classList.remove("loading");
}
async function fetchPuzzle(req) {
  loading(true);
  try {
    if (req.type === "daily") {
      const diff = req.difficulty || "medium";
      const r = await api(`/api/daily?difficulty=${encodeURIComponent(diff)}&date=${encodeURIComponent(req.date || "")}`);
      return { data: r.puzzle, id: r.id, src: { type: "daily", ...r.daily } };
    }
    if (req.type === "custom") {
      const r = await api(`/api/custom/${encodeURIComponent(req.id)}`);
      return { data: r.puzzle, id: null, src: { type: "custom", id: req.id, name: r.name } };
    }
    let { mode, diff } = req;
    if (!MODES[mode]) [mode, diff] = randomPick();
    if (!MODES[mode].diffs[diff]) diff = MODES[mode].dflt;
    const r = await api("/api/generate", genParams(MODES[mode].diffs[diff], req.seed ?? null));
    return { data: r.puzzle, id: r.id, src: { type: "mode", mode, diff, seed: r.seed } };
  } finally { loading(false); }
}
function setPuzzle(pz) {
  S.puzzle = pz;
  S.P = makeModel(pz.data);
  S.cache = new Map();
  const [t, sub] = titleFor(pz.src, pz.data);
  S.title = [t, sub];
}

// ------------------------------------------------------------------ robot runs (cached per puzzle + robot + settings)
function solveKey(bot) { return `${bot.id}|${bot.model || ""}|${NERD.time}|${NERD.budget}`; }
function solveFor(bot) {
  const key = solveKey(bot);
  if (S.cache.has(key)) return S.cache.get(key);
  const data = S.puzzle.data;
  const p = (async () => {
    const t0 = performance.now();
    let r;
    if (bot.mode === "exact") r = await api("/api/solve/exact", { puzzle: data, time_limit: NERD.time, trace: true });
    else r = await api("/api/solve/rl", { puzzle: data, mode: bot.mode, model: bot.model || null, budget: NERD.budget, time_limit: NERD.time, compare: false, trace: bot.mode !== "greedy" });
    r.wall = (performance.now() - t0) / 1000;
    return r;
  })();
  p.catch(() => S.cache.delete(key));
  S.cache.set(key, p);
  return p;
}
async function policyAt(bot, path) {
  return api("/api/policy", { puzzle: S.puzzle.data, path, model: bot.model || null, full: true, check: false });
}

// ------------------------------------------------------------------ replay lane (one robot on one board)
// ticks: push v | pop k | reset. One tick = one "try" step of the search, so two lanes replayed at the same
// tick rate race fairly: the robot that needs fewer steps finishes first.
function buildTicks(res, P) {
  const ticks = [];
  const start = P.cps[0];
  if (Array.isArray(res.trace) && res.trace.length) {
    let first = true;
    for (const e of res.trace) {
      if (e === "R") { if (!first) ticks.push({ t: "reset" }); first = false; continue; }
      first = false;
      if (e >= 0) { if (e === start && (!ticks.length || ticks[ticks.length - 1].t === "reset")) continue; ticks.push({ t: "push", v: e }); }
      else ticks.push({ t: "pop", k: -e });
    }
  } else if (res.path) {
    for (let i = 1; i < res.path.length; i++) ticks.push({ t: "push", v: res.path[i] });
  }
  return ticks;
}
class Lane {
  constructor({ board, P, bot, res, holds = false, onEvent = () => {} }) {
    Object.assign(this, { board, P, bot, res, holds, onEvent });
    this.ticks = buildTicks(res, P);
    this.solved = !!res.solved;
    this.final = res.path || null;
    this.i = 0; this.acc = 0; this.holdUntil = 0; this.holdBudget = 3800; this.held = false;
    this.stack = [P.cps[0]];
    this.onStack = new Uint8Array(P.n); this.onStack[P.cps[0]] = 1;
    this.tried = new Float32Array(P.n);
    this.tries = 0; this.deadEnds = 0; this.resets = 0; this.sincePop = 0;
    this.done = false; this.dirty = true; this.cand = null;
    this.reached = 1;
  }
  get total() { return this.ticks.length; }
  nextCp() { return this.reached; }
  apply(t, quiet) {
    const P = this.P;
    if (t.t === "push") {
      this.stack.push(t.v); this.onStack[t.v] = 1; this.tries++; this.sincePop++;
      const k = P.cpIndex.get(t.v);
      if (k !== undefined && k === this.reached) { this.reached++; if (!quiet) this.onEvent("cp", { k, v: t.v }); }
      else if (!quiet) this.onEvent("push", { v: t.v });
    } else if (t.t === "pop") {
      const k = Math.min(t.k, this.stack.length - 1);
      const gone = this.stack.splice(this.stack.length - k, k);
      for (const v of gone) { this.onStack[v] = 0; this.tried[v] = 1; if (P.cpIndex.has(v)) this.reached = Math.min(this.reached, P.cpIndex.get(v)); }
      this.deadEnds++; this.sincePop = 0;
      if (!quiet) this.onEvent("pop", { k: gone.length });
    } else if (t.t === "reset") {
      for (const v of this.stack.slice(1)) { this.onStack[v] = 0; this.tried[v] = 0.7; }
      this.stack = [P.cps[0]]; this.reached = 1; this.resets++; this.sincePop = 0;
      if (!quiet) this.onEvent("reset", {});
    }
    this.dirty = true;
  }
  // apply up to n ticks; with `holds`, pause before a dead end so it can be seen (the budget keeps it short)
  advance(n, now, speed) {
    let applied = 0;
    while (applied < n && this.i < this.ticks.length) {
      const t = this.ticks[this.i];
      if (this.holds && t.t === "pop" && !this.held && this.holdBudget > 0 && speed < 5) {
        this.held = true;
        const d = 520 / speed;
        this.holdBudget -= 520;
        this.holdUntil = now + d;
        this.onEvent("deadend", { head: this.stack[this.stack.length - 1] });
        break;
      }
      this.held = false;
      this.i++; applied++;
      this.apply(t, applied > 3);
    }
    if (this.i >= this.ticks.length && !this.done) this.finish();
    return applied;
  }
  finish() {
    if (this.solved && this.final && this.final.length === this.P.n) {
      // traces can be capped: land on the real solution
      if (this.stack.length !== this.final.length || this.stack.some((v, i) => v !== this.final[i])) {
        this.stack = this.final.slice(); this.onStack.fill(0); for (const v of this.stack) this.onStack[v] = 1;
      }
    }
    this.done = true; this.dirty = true;
    this.onEvent("done", { solved: this.solved });
  }
  skip() { while (this.i < this.ticks.length) { this.apply(this.ticks[this.i++], true); } if (!this.done) this.finish(); }
  render(animate) {
    if (!this.board.svg) return;
    const head = this.stack[this.stack.length - 1];
    const V = { path: this.stack, won: this.done && this.solved, legal: [], tried: this.tried, animate };
    const c = this.cand;
    if (c && c.head === head && !this.done) {
      if (NERD.odds) V.policy = c.legal;
      else { V.curStep = { node: -1, top: c.legal.slice(0, 3) }; V.anim = true; }
    }
    this.board.render(V);
    this.dirty = false;
  }
}

// drive lanes with requestAnimationFrame; msPerTick per lane (same for a fair race)
function drive(lanes, { onFrame = () => {}, getSpeed = () => S.speed } = {}) {
  let raf = 0, last = performance.now(), stopped = false, resolve;
  const done = new Promise((r) => { resolve = r; });
  const frame = (ts) => {
    if (stopped) return;
    const dt = Math.min(100, ts - last); last = ts;
    const speed = getSpeed();
    for (const L of lanes) {
      if (L.done) continue;
      if (L.holdUntil > ts) continue;
      L.acc += (dt * speed) / L.mpt;
      const n = Math.floor(L.acc);
      L.acc -= n;
      if (n > 0) L.lastN = L.advance(n, ts, speed);
    }
    const decay = Math.exp(-dt / 1100);
    for (const L of lanes) {
      let any = false;
      for (let v = 0; v < L.tried.length; v++) if (L.tried[v] > 0) { L.tried[v] = L.tried[v] * decay; if (L.tried[v] < 0.03) L.tried[v] = 0; any = true; }
      if (L.dirty || any) L.render((L.lastN || 0) <= 2 && !reducedMotion());
      L.lastN = 0;
    }
    onFrame(ts);
    if (lanes.every((L) => L.done)) { stopped = true; resolve(true); return; }
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
  return {
    done,
    stop() { stopped = true; cancelAnimationFrame(raf); resolve(false); },
    skip() { for (const L of lanes) { L.skip(); L.render(false); } },
  };
}

// ------------------------------------------------------------------ glow ("where the robot looks")
function glowDefs(b) {
  if (!b.svg) return;
  const defs = b.svg.querySelector("defs");
  const id = `sg${b._uid}`;
  if (b.svg.querySelector(`#${id}`)) return id;
  const g = el("radialGradient", { id }, defs);
  const col = glowColor();
  el("stop", { offset: "0", "stop-color": col, "stop-opacity": 0.95 }, g);
  el("stop", { offset: ".55", "stop-color": col, "stop-opacity": 0.45 }, g);
  el("stop", { offset: "1", "stop-color": col, "stop-opacity": 0 }, g);
  return id;
}
function drawGlow(b, heat, onStack) {
  if (!b.svg || !b.gHeat) return;
  const g = b.gHeat;
  g.textContent = "";
  if (!heat) return;
  const id = glowDefs(b), cs = b.L.cs;
  const ent = Object.entries(heat).map(([k, v]) => [Number(k), v]).filter(([v, x]) => x > 0.2 && v < b.P.n && !(onStack && onStack[v])).sort((a, c) => c[1] - a[1]).slice(0, Math.max(6, Math.round(b.P.n / 6)));
  ent.forEach(([v, x], i) => {
    const [cx, cy] = b.center(v);
    const r = cs * (0.5 + 0.5 * x);
    el("circle", { cx, cy, r, fill: `url(#${id})`, opacity: (0.35 + 0.65 * x).toFixed(3), class: `glow${i < 3 ? " hot" : ""}`, style: `animation-delay:${(-(v % 7) * 0.23).toFixed(2)}s` }, g);
    if (i < 3) el("circle", { cx, cy, r: cs * 0.1, fill: "#fff", opacity: 0.9, class: "glow-core" }, g);
  });
}
function clearGlow(b) { if (b && b.gHeat) b.gHeat.textContent = ""; }
function heatCaption(P, heat, path) {
  const top = Object.entries(heat || {}).map(([k, v]) => [Number(k), v]).sort((a, b) => b[1] - a[1]).slice(0, 5);
  if (!top.length) return pickLine(LINES.vague);
  const vis = new Set(path);
  const tight = top.filter(([v]) => P.adj[v].filter((u) => !vis.has(u)).length <= 2).length;
  if (tight >= 3) return pickLine(LINES.corners);
  const nxt = P.cps[1];
  if (nxt !== undefined && top.some(([v]) => v === nxt || P.adjSet[v].has(nxt))) return pickLine(LINES.target(2));
  return pickLine(LINES.vague);
}

// ------------------------------------------------------------------ captions + robot moods
let capLock = 0, capPrio = 0;
function caption(txt, { prio = 1, mood = null, ms = 1200, force = false } = {}) {
  const now = performance.now();
  if (!force && now < capLock && prio <= capPrio) return false;
  capLock = now + ms / Math.max(1, S.speed); capPrio = prio;
  const c = $("capTxt");
  c.textContent = txt;
  const cap = $("cap");
  cap.classList.remove("pop"); void cap.offsetWidth; cap.classList.add("pop");
  if (mood) setMood($("sBot"), mood);
  return true;
}
function setMood(elm, mood) { elm.className = `bot${elm.classList.contains("star") ? " star" : ""} ${mood || ""}`; }
function avatarClass(bot) { return bot.challenger ? "av-ch" : `av-${bot.id}`; }

// ------------------------------------------------------------------ WATCH
async function openStage(req, { push = false, autostart = false, bot = null } = {}) {
  stopAll();
  let pz;
  try { pz = await fetchPuzzle(req); }
  catch (e) { toast(`Couldn't load a puzzle: ${e.message}`); return; }
  setPuzzle(pz);
  // default star: the Scout (it backs out of dead ends: the best show); on 3D the Grandmaster (never gives up)
  if (bot && botById(bot)) { S.bot = bot; S.botChosen = true; }
  else if (!S.botChosen) S.bot = S.P.dim >= 3 ? "grandmaster" : "scout";
  if (!botById(S.bot) || botById(S.bot).off) S.bot = CAST.find((b) => !b.off).id;
  showScreen("stage");
  $("sTitle").textContent = S.title[0]; $("sSub").textContent = S.title[1];
  document.title = `${S.title[0]} · AI show · Zip`;
  setHash(`#watch&${srcHash(pz.src)}&bot=${S.bot}`, push);
  $("beatBtn").href = beatLink(pz.src);
  board = new Board($("boardHost"), { maxCell: 80, label: "The robot's board" });
  board.setPuzzle(S.P);
  S.lastLane = null; S.deckH = 0;
  setDeck("pre");
  paintBot();
  layoutStage();
  solveFor(botById(S.bot)).catch(() => {});   // think ahead: the replay starts at once
  if (autostart) setTimeout(() => watch(), 350);
}
function paintBot() {
  const bot = botById(S.bot);
  $("sBotAv").textContent = bot.emoji;
  $("sBotAv").className = `bot-av ${avatarClass(bot)}`;
  setMood($("sBot"), "idle");
  caption(`${bot.name}: ${bot.line.charAt(0).toLowerCase() + bot.line.slice(1)}.`, { force: true, prio: 0, ms: 0 });
  $("capTxt").innerHTML = `<span><b>${bot.name}</b>: ${bot.line.charAt(0).toLowerCase() + bot.line.slice(1)}.</span>`;
  $("who").innerHTML = "";
  for (const b of CAST) {
    if (b.off) continue;
    $("who").appendChild(h("button", { type: "button", class: `who-b ${avatarClass(b)}${b.id === S.bot ? " on" : ""}`, "aria-pressed": String(b.id === S.bot),
      "aria-label": `${b.name}: ${b.line}`, title: b.name, onclick: () => { if (S.watch) return; S.bot = b.id; S.botChosen = true; paintBot(); setHash(`#watch&${srcHash(S.puzzle.src)}&bot=${S.bot}`); resetBoard(); solveFor(b).catch(() => {}); } },
      h("span", { class: "who-av", "aria-hidden": "true" }, b.emoji), h("span", { class: "who-n" }, b.name)));
  }
  $("watchBtn").innerHTML = `Watch ${bot.name} <span aria-hidden="true">${bot.emoji}</span>`;
}
function setDeck(state) {
  $("dPre").hidden = state !== "pre";
  $("dRun").hidden = state !== "run";
  $("dDone").hidden = state !== "done";
  $("who").hidden = state !== "pre";
  document.body.classList.toggle("running", state === "run");
  // the deck can change height (stats wrap): keep the board clear of it
  const dh = $("deck").getBoundingClientRect().height;
  if (S.deckH && Math.abs(dh - S.deckH) > 2) layoutStage();
  S.deckH = dh;
}
function resetBoard() {
  S.lastLane = null;
  if (!board || !board.svg) return;
  clearGlow(board);
  board.render({ path: [S.P.cps[0]], legal: [], animate: false });
}
function layoutStage() {
  if (S.screen !== "stage" || !board) return;
  const wrap = $("boardWrap");
  wrap.style.flex = ""; wrap.style.height = "";
  const W = wrap.clientWidth, H = wrap.clientHeight;
  board.build(Math.max(120, W), Math.max(120, H));
  glowDefs(board);
  // hug the board so the robot, board and buttons sit together (the group is centred)
  const bh = board.svg.getBoundingClientRect().height;
  if (bh > 0 && bh < H - 8) { wrap.style.flex = "0 0 auto"; wrap.style.height = `${Math.ceil(bh) + 6}px`; }
  const w = S.watch, lane = (w && w.lane) || S.lastLane;
  if (lane) { lane.board = board; lane.render(false); if (w && w.heat) drawGlow(board, w.heat, lane.onStack); }
  else resetBoard();
  placeScan();
}
function placeScan() {
  const s = $("scan"), svg = board && board.svg;
  if (!svg) return;
  const wr = $("boardWrap").getBoundingClientRect(), r = svg.getBoundingClientRect();
  Object.assign(s.style, { left: `${r.left - wr.left}px`, top: `${r.top - wr.top}px`, width: `${r.width}px`, height: `${r.height}px` });
}
function setSpeedUI(v) {
  document.querySelectorAll(".speed [data-speed]").forEach((b) => { if (b.dataset.speed !== "skip") b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === v)); });
}

async function watch() {
  if (S.watch || !S.puzzle) return;
  const bot = botById(S.bot);
  const P = S.P, pz = S.puzzle;
  const W = S.watch = { stopped: false, lane: null, heat: null, drv: null, stop() { this.stopped = true; if (this.drv) this.drv.stop(); } };
  S.speed = 1; setSpeedUI(1);
  setDeck("run");
  $("triesN").textContent = "0"; $("triesL").textContent = "tries";
  resetBoard();
  sound.robot();
  const alive = () => !W.stopped && S.watch === W && S.puzzle === pz;
  // 1) look at the board
  caption(pickLine(LINES.look), { force: true, prio: 3, mood: "thinking" });
  const scan = $("scan");
  placeScan();
  if (!reducedMotion()) { scan.classList.remove("on"); void scan.offsetWidth; scan.classList.add("on"); }
  const resP = solveFor(bot);
  const t0 = performance.now();
  if (bot.brain) {
    try {
      const pol = await policyAt(bot, [P.cps[0]]);
      if (!alive()) return;
      W.heat = pol.heat; drawGlow(board, pol.heat, null);
      await sleep(Math.max(0, 900 - (performance.now() - t0)));
      if (!alive()) return;
      caption(heatCaption(P, pol.heat, [P.cps[0]]), { force: true, prio: 3, mood: "thinking" });
      await sleep(1500);
    } catch { await sleep(700); }
  } else {
    await sleep(700);
    if (!alive()) return;
    caption(pickLine(LINES.nobrain), { force: true, prio: 3, mood: "thinking" });
    await sleep(1400);
  }
  if (!alive()) return;
  // 2) get the robot's actual run (usually ready by now)
  let res;
  const slow = setTimeout(() => { if (alive()) caption("Thinking really hard… 🤯", { force: true, mood: "thinking" }); }, 400);
  try { res = await resP; }
  catch (e) {
    clearTimeout(slow);
    if (!alive()) return;
    const noModel = e.status === 404 || e.status === 503;
    caption(noModel ? "My brain is missing! 🔌 Try the Tortoise." : `Short circuit! 🔌 ${e.message}`, { force: true, mood: "sad" });
    setDeck("pre"); S.watch = null; return;
  }
  clearTimeout(slow);
  if (!alive()) return;
  // 3) replay the search
  const lane = S.lastLane = W.lane = new Lane({ board, P, bot, res, holds: true, onEvent: (type, info) => onWatchEvent(W, type, info) });
  const T = Math.max(1, lane.total);
  lane.mpt = clamp(clamp(T * 115, 3800, 16000) / T, 2, 140);
  caption(pickLine(LINES.go), { force: true, mood: "going", prio: 2 });
  if (bot.brain) lookLoop(W, bot);
  W.drv = drive([lane], { onFrame: () => { $("triesN").textContent = fmtInt(lane.tries); $("triesL").textContent = lane.tries === 1 ? "try" : "tries"; } });
  const finished = await W.drv.done;
  if (!alive() || !finished) return;
  $("triesN").textContent = fmtInt(lane.tries);
  W.heat = null; clearGlow(board);
  await celebrate(W, lane, res, bot);
}
async function lookLoop(W, bot) {
  while (!W.stopped && W.lane && !W.lane.done) {
    const t0 = performance.now();
    const path = W.lane.stack.slice();
    if (path.length >= S.P.n) break;
    try {
      const pol = await policyAt(bot, path);
      if (W.stopped || !W.lane || W.lane.done) break;
      W.heat = pol.heat;
      drawGlow(board, pol.heat, W.lane.onStack);
      W.lane.cand = { head: path[path.length - 1], legal: pol.legal || [] };
      W.lane.dirty = true;
      const top = (pol.legal || [])[0];
      if (top && pol.legal.length > 1 && top[1] < 0.55 && Math.random() < 0.5) caption(pickLine(LINES.tough), { mood: "thinking" });
    } catch { break; }
    await sleep(Math.max(60, 380 - (performance.now() - t0)));
  }
}
function onWatchEvent(W, type, info) {
  const L = W.lane, b = board;
  if (!b || !b.svg) return;
  if (type === "push" || type === "cp") {
    const idx = L.stack.length - 1;
    const col = b.colorAt(idx);
    let p = null;
    try { p = b.clientOf(info.v); } catch { /* ignore */ }
    if (type === "cp") {
      sound.levelUp(info.k);
      if (p) fx.burst(p.x, p.y, col, 18);
      caption(pickLine(LINES.cp(info.k + 1)), { prio: 2, mood: "going" });
    } else {
      sound.note(idx - 1, { seed: S.P.n });
      if (p && S.speed < 5) fx.sparks(p.x, p.y, col, 2);
      if (L.sincePop === 14) caption(pickLine(LINES.smooth), { mood: "going" });
      if ([100, 500, 1000, 5000, 20000].includes(L.tries)) caption(pickLine(LINES.many(L.tries)), { prio: 2 });
    }
  } else if (type === "deadend") {
    b.flash(info.head);
    sound.bad();
    caption(pickLine(LINES.dead), { prio: 3, mood: "confused", ms: 1300 });
  } else if (type === "pop") {
    sound.undo();
    if (!L.holds || L.holdBudget <= 0) caption(pickLine(LINES.dead), { prio: 2, mood: "confused" });
  } else if (type === "reset") {
    sound.whoosh();
    caption(pickLine(LINES.reset), { prio: 3, mood: "confused", ms: 1500 });
  }
}
async function celebrate(W, lane, res, bot) {
  const P = S.P;
  const sure = (res.steps || []).filter((s) => s.p >= 0.9).length, steps = (res.steps || []).length;
  const secs = res.seconds != null ? res.seconds : res.wall;
  const tm = secs < 1 ? `${Math.max(0.01, secs).toFixed(secs < 0.1 ? 2 : 1)} s` : `${secs.toFixed(1)} s`;
  const stats = [];
  if (lane.solved) {
    sound.win();
    setMood($("sBot"), "win");
    const clean = lane.deadEnds === 0 && lane.resets === 0;
    caption(clean ? "Got it. No wrong turns! 🎉" : `Got it in ${fmtInt(lane.tries)} tries! 🎉`, { force: true, prio: 5, ms: 99999 });
    await cinematic(board, lane.stack);
    stats.push(`⚡ ${tm} of thinking`);
    if (!clean) stats.push(`↩ ${fmtInt(lane.deadEnds)} dead end${lane.deadEnds === 1 ? "" : "s"} escaped`);
    else if (steps && bot.brain) stats.push(`🎯 Sure about ${sure} of ${steps} moves`);
    else stats.push(`🧠 ${fmtInt(lane.tries)} tries`);
  } else {
    sound.lose();
    setMood($("sBot"), "sad");
    const why = res.status === "budget" || res.status === "timeout" ? "I give up. Too many dead ends 😵" : bot.mode === "greedy" ? "Uh-oh… I'm stuck 😵" : "No way through… 😵";
    caption(why, { force: true, prio: 5, ms: 99999 });
    stats.push(`🧩 Filled ${lane.stack.length} of ${P.n} squares`);
    stats.push(bot.mode === "greedy" ? "🐣 Rookie never backs up. Try 🦉!" : `🧠 ${fmtInt(lane.tries)} tries in ${tm}`);
  }
  $("stats").innerHTML = stats.map((s) => `<span class="pill-s">${s}</span>`).join("");
  S.watch = null;
  setDeck("done");
}
async function cinematic(b, path) {
  if (!b.svg || reducedMotion()) { await sleep(300); return; }
  const n = path.length, step = Math.min(28, 700 / n);
  (b.segs || []).forEach((s, i) => { if (!s) return; s.style.animationDelay = `${i * step}ms`; s.classList.remove("lit"); void s.getBBox(); s.classList.add("lit"); });
  path.forEach((v, i) => { const t = b.tints[v]; if (!t) return; t.style.animationDelay = `${i * step}ms`; t.classList.add("ripple"); });
  (b.cpEls || []).forEach((g, k) => { const inner = g.firstChild; const i = path.indexOf(S.P.cps[k]); inner.style.animationDelay = `${i * step}ms`; inner.classList.remove("pop"); void inner.getBBox(); inner.classList.add("cheer"); });
  const r = b.svg.getBoundingClientRect();
  setTimeout(() => fx.confetti(r, 120), n * step * 0.5);
  await sleep(n * step + 500);
}

// ------------------------------------------------------------------ FACE-OFF
const lanesUI = { A: null, B: null };
async function openFaceoff(req, { push = false, pair = null } = {}) {
  stopAll();
  if (req) {
    let pz;
    try { pz = await fetchPuzzle(req); }
    catch (e) { toast(`Couldn't load a puzzle: ${e.message}`); return; }
    setPuzzle(pz);
  }
  if (!S.puzzle) return openFaceoff({ type: "mode", mode: "islands", diff: "normal" }, { push, pair });
  if (pair) S.vs = pair.map((id) => (botById(id) && !botById(id).off ? id : "tortoise"));
  showScreen("faceoff");
  $("sTitle").textContent = "Robot vs robot"; $("sSub").textContent = S.title.join(" · ");
  document.title = "Robot vs robot · AI show · Zip";
  setHash(`#vs=${S.vs.join(",")}&${srcHash(S.puzzle.src)}`, push);
  $("fBeatBtn").href = beatLink(S.puzzle.src);
  buildLanes();
  fDeck("pre");
  for (const id of S.vs) solveFor(botById(id)).catch(() => {});
}
function fDeck(state) {
  $("fPre").hidden = state !== "pre";
  $("fRun").hidden = state !== "run";
  $("fDone").hidden = state !== "done";
  document.body.classList.toggle("running", state === "run");
  document.querySelectorAll(".lane .bot").forEach((b) => b.classList.toggle("locked", state !== "pre"));
}
function buildLanes() {
  ["A", "B"].forEach((side, i) => {
    const bot = botById(S.vs[i]);
    const host = $(`lane${side}`);
    host.className = "lane";
    host.innerHTML = "";
    const av = h("button", { type: "button", class: "bot", "aria-label": `${bot.name}. Tap to swap robot`, onclick: () => { if (!S.race) openPicker(i); } },
      h("span", { class: `bot-av ${avatarClass(bot)}`, "aria-hidden": "true" }, bot.emoji),
      h("span", { class: "crown", "aria-hidden": "true" }, "👑"));
    const name = h("div", { class: "l-name" }, h("b", {}, bot.name), h("span", {}, bot.line));
    const say = h("div", { class: "bot-say", role: "status" }, "ready!");
    const bh = h("div", { class: "l-board" });
    const prog = h("div", { class: "l-prog", "aria-hidden": "true" }, h("i"));
    const tries = h("div", { class: "l-tries" }, h("span", { class: "mono" }, "0"), " tries");
    host.append(h("div", { class: "l-head" }, av, name), say, bh, prog, tries);
    const b = new Board(bh, { maxCell: 60, label: `${bot.name}'s board` });
    b.setPuzzle(S.P);
    lanesUI[side] = { host, bot, say, bh, prog: prog.firstChild, tries: tries.firstChild, board: b, av };
  });
  layoutLanes();
}
function layoutLanes() {
  if (S.screen !== "faceoff") return;
  for (const side of ["A", "B"]) {
    const u = lanesUI[side];
    if (!u) continue;
    // boards fill the lane width, and leave room below for progress, tries and the buttons
    const W = u.bh.clientWidth;
    const deckH = Math.max(120, document.querySelector("#faceoff .deck").getBoundingClientRect().height);
    const H = innerHeight - u.bh.getBoundingClientRect().top - deckH - 64;
    u.board.build(Math.max(80, W), Math.max(80, Math.min(W * 1.4, H)));
    if (u.lane) u.lane.render(false);
    else u.board.render({ path: [S.P.cps[0]], legal: [], animate: false });
  }
}
function laneSay(u, txt, mood = "", force = false) {
  const now = performance.now();
  if (!force && (u.say.textContent === txt || now < (u.sayUntil || 0))) return;
  u.sayUntil = now + 900 / Math.max(1, S.speed);
  u.say.textContent = txt;
  u.say.classList.remove("pop"); void u.say.offsetWidth; u.say.classList.add("pop");
  const b = u.av;
  b.className = `bot ${mood} locked`;
}
async function race() {
  if (S.race) return;
  const pz = S.puzzle, P = S.P;
  const RC = S.race = { stopped: false, drv: null, stop() { this.stopped = true; if (this.drv) this.drv.stop(); } };
  const alive = () => !RC.stopped && S.race === RC && S.puzzle === pz;
  S.speed = 1; setSpeedUI(1);
  fDeck("run");
  const us = [lanesUI.A, lanesUI.B];
  us.forEach((u) => { u.lane = null; u.host.classList.remove("winner", "loser"); u.board.render({ path: [P.cps[0]], legal: [], animate: false }); laneSay(u, "warming up…", "thinking", true); });
  let results;
  try { results = await Promise.all(us.map((u) => solveFor(u.bot).catch((e) => ({ error: e })))); }
  catch { results = []; }
  if (!alive()) return;
  // countdown
  const cd = $("countdown");
  for (const k of ["3", "2", "1"]) {
    if (!alive()) return;
    cd.textContent = k; cd.className = "countdown g-count show"; void cd.offsetWidth; sound.count(false);
    await sleep(reducedMotion() ? 300 : 650);
    cd.className = "countdown g-count";
  }
  if (!alive()) return;
  cd.textContent = "Go!"; cd.className = "countdown g-count show go"; sound.count(true);
  setTimeout(() => { cd.className = "countdown g-count"; }, 500);
  const lanes = [];
  let winner = null, finishedAt = 0;
  us.forEach((u, i) => {
    const res = results[i];
    if (!res || res.error) {
      laneSay(u, "my circuits! 🔌", "sad", true);
      u.lane = new Lane({ board: u.board, P, bot: u.bot, res: { solved: false, path: [P.cps[0]] } });
      u.lane.broken = true;
    } else {
      u.lane = new Lane({ board: u.board, P, bot: u.bot, res, onEvent: (type, info) => onRaceEvent(u, type, info) });
    }
    lanes.push(u.lane);
  });
  // same tick rate for both: the robot needing fewer steps wins; ~7 s for the winner at 1x
  const solvers = lanes.filter((L) => L.solved).map((L) => L.total);
  const ref = solvers.length ? Math.min(...solvers) : Math.max(1, ...lanes.map((L) => L.total));
  const mpt = clamp(7000 / Math.max(1, ref), 3, 160);
  lanes.forEach((L) => { L.mpt = mpt; });
  us.forEach((u) => { if (!u.lane.broken) laneSay(u, pickLine(u.bot.say.go), "going"); });
  const onFrame = () => {
    us.forEach((u) => {
      const L = u.lane;
      u.tries.textContent = fmtInt(L.tries);
      u.tries.nextSibling.textContent = L.tries === 1 ? " try" : " tries";
      u.prog.style.width = `${(100 * (L.stack.length - 1)) / Math.max(1, P.n - 1)}%`;
      if (L.done && !u.fin) {
        u.fin = true;
        if (L.solved && !winner) {
          winner = u; finishedAt = L.tries;
          u.host.classList.add("winner");
          laneSay(u, u.bot.say.win, "win", true);
          sound.win();
          const r = u.board.svg && u.board.svg.getBoundingClientRect();
          if (r) fx.confetti(r, 110);
          // the other robot hurries up to show how its run ends
          us.forEach((o) => { if (o !== u && !o.lane.done) { const left = Math.max(1, o.lane.total - o.lane.i); o.lane.mpt = Math.min(o.lane.mpt, 1600 / left); laneSay(o, "…still going", "thinking", true); } });
        } else if (L.solved) laneSay(u, "done too! 😅", "going", true);
        else if (!L.broken) { laneSay(u, u.bot.say.stuck, "sad", true); u.host.classList.add("loser"); }
      }
    });
  };
  us.forEach((u) => { u.fin = false; });
  RC.drv = drive(lanes, { onFrame });
  const ok = await RC.drv.done;
  if (!alive() || !ok) return;
  onFrame();
  await sleep(700);
  if (!alive()) return;
  raceResult(us, winner);
}
function onRaceEvent(u, type) {
  const L = u.lane;
  if (type === "pop" && L.deadEnds % 3 === 1) laneSay(u, u.bot.say.back || "oops…", "confused");
  else if (type === "reset") laneSay(u, "start over! 🔄", "confused");
  else if ((type === "push" || type === "cp") && L.sincePop === 1 && L.deadEnds) laneSay(u, pickLine(u.bot.say.go), "going");
  else if (type === "cp" && S.speed < 5) sound.robot();
}
function raceResult(us, winner) {
  const [a, b] = us;
  const stats = [];
  const d = (u) => `${u.bot.emoji} ${fmtInt(u.lane.tries)} ${u.lane.tries === 1 ? "try" : "tries"}${u.lane.solved ? "" : u.lane.broken ? " (broken 🔌)" : " (stuck 😵)"}`;
  let title;
  if (winner) {
    const other = winner === a ? b : a;
    title = `${winner.bot.emoji} ${winner.bot.name} wins!`;
    if (other.lane.solved && other.lane.tries === winner.lane.tries) title = "It's a tie! 🤝";
    other.host.classList.add("loser");
    if (other.lane.solved && other.lane.tries !== winner.lane.tries) laneSay(other, other.bot.say.lose, "sad", true);
  } else title = "Nobody made it! 🤷";
  stats.push(d(a), d(b));
  $("fStats").innerHTML = `<div class="f-title">${title}</div>` + stats.map((s) => `<span class="pill-s">${s}</span>`).join("");
  if (winner) sound.levelUp(2);
  S.race = null;
  fDeck("done");
}

// ------------------------------------------------------------------ robot picker (face-off)
let pickSide = 0;
function openPicker(side) {
  pickSide = side;
  const list = $("pkList");
  list.innerHTML = "";
  CAST.forEach((b, i) => {
    if (b.off) return;
    const on = S.vs[side] === b.id;
    list.appendChild(h("button", { type: "button", class: `rival pk${on ? " on" : ""}`, style: `--i:${i}`, "aria-pressed": String(on), onclick: () => {
      S.vs[side] = b.id; closeSheet("picker"); buildLanes(); setHash(`#vs=${S.vs.join(",")}&${srcHash(S.puzzle.src)}`); solveFor(b).catch(() => {});
    } },
    h("span", { class: `rival-av ${avatarClass(b)}`, "aria-hidden": "true" }, b.emoji),
    h("span", { class: "rival-txt" }, h("b", {}, b.name), h("span", {}, b.line)),
    h("span", { class: "rival-go", "aria-hidden": "true" }, on ? "✓" : "›")));
  });
  $("pkTitle").textContent = side === 0 ? "Left robot" : "Right robot";
  openSheet("picker");
}
let lastFocus = null;
function openSheet(id) { lastFocus = document.activeElement; $(id).hidden = false; requestAnimationFrame(() => $(id).classList.add("show")); setTimeout(() => { const f = $(id).querySelector("button"); if (f) f.focus({ preventScroll: true }); }, 60); }
function closeSheet(id) { const s = $(id); if (s.hidden) return; s.classList.remove("show"); s.hidden = true; if (lastFocus && lastFocus.focus) lastFocus.focus({ preventScroll: true }); }

// ------------------------------------------------------------------ nerd mode
function paintNerd() {
  const body = $("nerdBody");
  body.innerHTML = "";
  const opt = (v, t, sel) => h("option", { value: v, selected: sel ? true : null }, t);
  const brain = h("select", { id: "nerdModel", onchange: async (e) => { NERD.model = e.target.value; saveNerd(); await loadCast(); toast("New brain plugged in 🧠"); } },
    opt("", `Auto (${S.dflt || "none found"})`, !NERD.model),
    ...S.models.map((m) => opt(m.name, m.name, NERD.model === m.name)));
  const time = h("select", { id: "nerdTime", onchange: (e) => { NERD.time = Number(e.target.value); saveNerd(); S.cache = new Map(); } },
    ...[5, 10, 20, 40].map((s) => opt(s, `${s} s`, NERD.time === s)));
  const budget = h("select", { id: "nerdBudget", onchange: (e) => { NERD.budget = Number(e.target.value); saveNerd(); S.cache = new Map(); } },
    ...[1000, 4000, 20000, 100000].map((s) => opt(s, `${fmtInt(s)} tries`, NERD.budget === s)));
  const odds = h("input", { type: "checkbox", id: "nerdOdds", checked: NERD.odds ? true : null, onchange: (e) => { NERD.odds = e.target.checked; saveNerd(); } });
  const row = (label, forId, ctl) => h("div", { class: "n-row" }, h("label", { for: forId }, label), ctl);
  body.append(
    row("Brain for 🐣 🦊 🦉", "nerdModel", brain),
    row("Thinking time limit", "nerdTime", time),
    row("Search budget (🦊 and challengers)", "nerdBudget", budget),
    h("div", { class: "n-row" }, h("label", { for: "nerdOdds" }, "Show move odds on the board"), odds),
  );
  const ch = CAST.filter((b) => b.challenger);
  if (ch.length) body.append(h("div", { class: "n-cast" }, h("div", { class: "n-k" }, "Challengers"),
    ...ch.map((b) => h("div", { class: "n-c" }, h("span", { "aria-hidden": "true" }, b.emoji), h("b", {}, b.name), h("code", {}, b.model)))));
  body.append(h("a", { class: "n-link", href: "/workbench" }, "Open the workbench: tables, compare, AI vision →"));
}

// ------------------------------------------------------------------ hero art
function heroArt() {
  let cells = "";
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) cells += `<rect x="${7 + c * 28}" y="${7 + r * 28}" width="22" height="22" rx="6"/>`;
  const glows = [[46, 46, 0], [102, 18, 0.35], [18, 74, 0.7], [74, 102, 1.05], [102, 74, 1.4]].map(([x, y, d]) => `<circle class="hg" cx="${x}" cy="${y}" r="17" style="animation-delay:${d}s"/>`).join("");
  const dots = [[18, 18, 1], [102, 46, 2], [18, 74, 3], [18, 102, 4]].map(([x, y, k]) => `<g transform="translate(${x} ${y})"><circle r="8.5"/><text y="3.6" text-anchor="middle">${k}</text></g>`).join("");
  $("heroArt").innerHTML = `<svg viewBox="0 0 120 120" class="hero-svg show-svg">
    <defs><radialGradient id="hgG"><stop offset="0" stop-color="#7ff3ff" stop-opacity=".95"/><stop offset=".6" stop-color="#5fd8ff" stop-opacity=".35"/><stop offset="1" stop-color="#5fd8ff" stop-opacity="0"/></radialGradient></defs>
    <g class="hero-cells">${cells}</g><g class="hglows">${glows}</g>
    <path class="h-wrong" d="M18 18 V46 H46 V18" pathLength="1"/>
    <path class="h-right" d="M18 18 H102 V46 H18 V74 H102 V102 H18" pathLength="1"/>
    <g class="hero-dots">${dots}</g>
    <g class="h-bot"><circle cx="100" cy="100" r="17"/><text x="100" y="107" text-anchor="middle">🦉</text></g></svg>`;
  $("vsArt").innerHTML = `<span class="vs-av av-rookie">🐣</span><span class="vs-x">VS</span><span class="vs-av av-grandmaster">🦉</span>`;
}

// ------------------------------------------------------------------ menu, sound
function paintSound() {
  $("soundBtn").innerHTML = icon(sound.muted ? "mute" : "sound");
  $("soundBtn").setAttribute("aria-label", sound.muted ? "Turn sound on" : "Turn sound off");
  $("miSoundIc").textContent = sound.muted ? "🔇" : "🔊";
  $("miSoundTxt").textContent = sound.muted ? "Sound off" : "Sound on";
}
function toggleSound() { sound.setMuted(!sound.muted); paintSound(); toast(sound.muted ? "Sound off 🔇" : "Sound on 🔊"); }
function openMenu() { $("menu").hidden = false; $("menuBtn").setAttribute("aria-expanded", "true"); requestAnimationFrame(() => $("menu").classList.add("show")); const f = $("menu").querySelector(".mi"); if (f) f.focus({ preventScroll: true }); }
function closeMenu() { if ($("menu").hidden) return; $("menu").classList.remove("show"); $("menu").hidden = true; $("menuBtn").setAttribute("aria-expanded", "false"); }
function openNerd() {
  if (S.screen !== "pick") goPick(true);
  const d = $("nerd");
  d.open = true;
  setTimeout(() => d.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" }), 50);
}

// ------------------------------------------------------------------ routing
async function route(hash, push = false) {
  const q = new URLSearchParams((hash || "").replace(/^#/, ""));
  const seed = q.get("seed") && /^\d+$/.test(q.get("seed")) ? Number(q.get("seed")) : null;
  let req = null;
  if (q.get("custom")) req = { type: "custom", id: q.get("custom") };
  else if (q.get("daily")) req = { type: "daily", difficulty: q.get("daily"), date: q.get("date") || "" };
  else if (q.get("play") && MODES[q.get("play")]) req = { type: "mode", mode: q.get("play"), diff: q.get("diff") || MODES[q.get("play")].dflt, seed };
  if (q.has("vs")) {
    const pair = (q.get("vs") || "").split(",").filter(Boolean);
    return openFaceoff(req || { type: "mode", mode: "islands", diff: "normal" }, { push, pair: pair.length === 2 ? pair : null });
  }
  if (req) return openStage(req, { push, bot: q.get("bot") });
  goPick(null);
}

// ------------------------------------------------------------------ keyboard
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest && ev.target.closest("input, select, textarea")) return;
  if (ev.key === "Escape") {
    if (!$("menu").hidden) { closeMenu(); $("menuBtn").focus(); ev.preventDefault(); return; }
    if (!$("picker").hidden) { closeSheet("picker"); ev.preventDefault(); return; }
    if (S.screen !== "pick") { goPick(true); ev.preventDefault(); }
    return;
  }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.key === "m" || ev.key === "M") { toggleSound(); ev.preventDefault(); }
  else if ((ev.key === " " || ev.key === "Enter") && S.screen === "stage" && !S.watch && !$("dPre").hidden && document.activeElement === document.body) { watch(); ev.preventDefault(); }
});

// ------------------------------------------------------------------ init
async function init() {
  fx = createFx($("fx"));
  $("menuBtn").innerHTML = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>';
  $("skipBtn").innerHTML = icon("skip"); $("fSkipBtn").innerHTML = icon("skip");
  paintBack();
  paintSound();
  const A = modeArt(ROBOTS);
  document.querySelectorAll("[data-art]").forEach((e) => { e.innerHTML = A[e.dataset.art] || ""; });
  heroArt();
  const cd = store.get("zip-classic-diff", "medium");
  document.querySelectorAll(".m-classic .diff").forEach((b) => b.classList.toggle("on", b.dataset.diff === cd));

  $("backBtn").addEventListener("click", (e) => { if (S.screen !== "pick") { e.preventDefault(); goPick(true); } });
  $("soundBtn").onclick = toggleSound;
  $("menuBtn").onclick = (e) => { e.stopPropagation(); if ($("menu").hidden) openMenu(); else closeMenu(); };
  document.addEventListener("click", (e) => { if (!$("menu").hidden && !e.target.closest("#menu")) closeMenu(); });
  $("menu").addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    const a = b.dataset.act;
    if (a === "sound") toggleSound();
    else if (a === "theme") window.zipTheme && window.zipTheme.toggle();
    else if (a === "nerd") { closeMenu(); openNerd(); }
  });
  document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", (e) => {
    e.stopPropagation();
    const m = b.dataset.pick;
    if (m === "surprise") { const [mm, dd] = randomPick(); return openStage({ type: "mode", mode: mm, diff: dd }, { push: true }); }
    let diff = b.dataset.diff || (m === "classic" ? store.get("zip-classic-diff", "medium") : MODES[m].dflt);
    if (m === "classic" && b.dataset.diff) { store.set("zip-classic-diff", diff); document.querySelectorAll(".m-classic .diff").forEach((x) => x.classList.toggle("on", x.dataset.diff === diff)); }
    openStage({ type: "mode", mode: m, diff }, { push: true });
  }));
  $("heroGo").onclick = () => { const [mm, dd] = randomPick(); openStage({ type: "mode", mode: mm, diff: dd }, { push: true, autostart: true }); };
  $("hero").addEventListener("click", (e) => { if (!e.target.closest("button")) $("heroGo").click(); });
  $("vsGo").onclick = () => { const [mm, dd] = randomPick(); openFaceoff({ type: "mode", mode: mm, diff: dd }, { push: true }); };

  $("watchBtn").onclick = () => watch();
  $("againBtn").onclick = () => { resetBoard(); paintBot(); setDeck("pre"); $("watchBtn").focus({ preventScroll: true }); };
  $("newBtn").onclick = () => { const s = S.puzzle.src; openStage(s.type === "mode" ? { type: "mode", mode: s.mode, diff: s.diff } : { type: "mode", mode: "surprise" }, { push: false, autostart: true }); };
  $("vsBtn").onclick = () => { const b = S.bot; S.vs = [b === "rookie" ? "grandmaster" : "rookie", b]; if (S.vs[0] === S.vs[1]) S.vs[0] = "tortoise"; openFaceoff(null, { push: true }); };
  $("raceBtn").onclick = () => race();
  $("rematchBtn").onclick = () => { const s = S.puzzle.src; openFaceoff(s.type === "mode" ? { type: "mode", mode: s.mode, diff: s.diff } : { type: "mode", mode: "surprise" }).then(() => race()); };
  $("swapBtn").onclick = () => {
    const ids = CAST.filter((b) => !b.off).map((b) => b.id);
    const pickTwo = () => { const a = ids[Math.floor(Math.random() * ids.length)]; let c = a; while (c === a && ids.length > 1) c = ids[Math.floor(Math.random() * ids.length)]; return [a, c]; };
    S.vs = pickTwo(); S.race = null; S.cache = new Map(); buildLanes(); fDeck("pre"); setHash(`#vs=${S.vs.join(",")}&${srcHash(S.puzzle.src)}`);
    for (const id of S.vs) solveFor(botById(id)).catch(() => {});
  };
  document.querySelectorAll(".speed [data-speed]").forEach((b) => b.addEventListener("click", () => {
    const v = b.dataset.speed;
    if (v === "skip") {
      if (S.watch && S.watch.drv) { S.speed = 50; S.watch.lane.holdBudget = 0; S.watch.drv.skip(); }
      if (S.race && S.race.drv) S.race.drv.skip();
      return;
    }
    S.speed = Number(v); setSpeedUI(S.speed);
  }));
  $("pkClose").onclick = () => closeSheet("picker");
  $("picker").addEventListener("click", (e) => { if (e.target === $("picker")) closeSheet("picker"); });

  window.addEventListener("zip-theme", () => { if (S.screen === "stage") layoutStage(); if (S.screen === "faceoff") layoutLanes(); });
  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (S.screen === "stage") layoutStage(); if (S.screen === "faceoff") layoutLanes(); }, 120); });
  window.addEventListener("popstate", () => route(location.hash));

  await loadCast();
  // /lab?model=<name> (dashboard "Play ▸"): that model becomes the brain
  const qm = new URLSearchParams(location.search).get("model");
  if (qm && S.models.some((m) => m.name === qm)) { NERD.model = qm; saveNerd(); await loadCast(); toast("Brain loaded from the dashboard 🧠"); }
  await route(location.hash);
}

window.__show = { S, watch, race, route, openStage, openFaceoff, cast: () => CAST, board: () => board, lanes: lanesUI };
init();
