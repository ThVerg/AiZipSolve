// Static API adapter: the online (GitHub Pages) build has no Python server, so the game and the AI show
// talk to this module instead of /api/* (util.js api() routes here when window.ZIP_STATIC is set).
// Everything comes from the precomputed puzzle bank under static/bank/:
//   index.json                      modes -> difficulties -> pool files
//   pools/<mode>-<diff>.json        [{id, mode, diff, label, score, size, puzzle (with its unique solution)}]
//   daily/<yyyy>-<mm>.json          {"YYYY-MM-DD": {number, easy, medium, hard, special}}
//   robots/<id>.json                {rookie, scout, grandmaster, tortoise, + detective, mcts, evolver, gambler,
//                                   sat where recorded}: /api/solve/* and /api/solve/robot runs
//   architect/weekly.json           {"YYYY-Www": entry}: the Architect's weekly challenge
// New puzzle kinds (portals, torus, hex, tri, oneway, overpass, keys, cubesurf) and co-op have their own pools
// ("<mode>-medium" / "-hard"); co-op entries keep their two unique paths under meta.coop.solution.
// Hints and "Show me" use the stored solution (the bank's puzzles have exactly one, so a path that leaves it
// can only be fixed by backing up to where it left; one-way arcs, keys and overpasses need nothing more).
// The Detective's teaching hints come from its recorded run; the Architect "designs" by picking from its
// pools. The AI's "vision" (/api/policy) and everything local-only (editor, custom puzzles, workbench,
// dashboard) answer 501.

const BANK = new URL("../bank/", import.meta.url);
const files = new Map();
function bank(rel) {
  if (!files.has(rel)) {
    const p = fetch(new URL(rel, BANK)).then((r) => {
      if (!r.ok) throw httpError(r.status, `${rel}: ${r.status}`);
      return r.json();
    });
    p.catch(() => files.delete(rel));
    files.set(rel, p);
  }
  return files.get(rel);
}
function httpError(status, msg) { const e = new Error(msg); e.status = status; return e; }
const clone = (x) => (typeof structuredClone === "function" ? structuredClone(x) : JSON.parse(JSON.stringify(x)));

// puzzles handed out on this page -> bank entry (for hints, solves and robot runs)
const known = new Map();
function fingerprint(d) {
  let h = 2166136261 >>> 0;
  const mix = (x) => { h = Math.imul(h ^ (x + 1), 16777619) >>> 0; };
  mix(d.coords.length); for (const c of d.checkpoints) mix(c);
  for (const [u, v] of d.edges) { mix(u); mix(v); }
  return `${d.coords.length}:${d.edges.length}:${h.toString(36)}`;
}
function remember(entry) {
  known.set(fingerprint(entry.puzzle), entry);
  known.set(`id:${entry.id}`, entry);
}
function entryFor(puzzle, id) {
  return (id && known.get(`id:${id}`)) || (puzzle && puzzle.coords && known.get(fingerprint(puzzle))) || null;
}
// a puzzle handed out on another page (the Architect's "Play it", a link): look through every pool once
let scanned = null;
async function lookup(puzzle, id) {
  const e = entryFor(puzzle, id);
  if (e || !puzzle || !puzzle.coords) return e;
  if (!scanned) {
    scanned = (async () => {
      const idx = await index();
      const files = new Set(Object.values(idx.modes || {}).flatMap((ds) => Object.values(ds).map((d) => d.file)));
      await Promise.all([...files].map(async (f) => { try { (await bank(f)).forEach(remember); } catch { /* skip */ } }));
      try { Object.values(await bank("architect/weekly.json")).forEach(remember); } catch { /* no weekly file */ }
    })();
  }
  await scanned;
  return entryFor(puzzle, id);
}
function publicPuzzle(entry) {
  const d = { ...entry.puzzle };
  delete d.solution;
  if (d.meta && d.meta.coop && d.meta.coop.solution) {
    const { solution: _drop, ...coop } = d.meta.coop;
    d.meta = { ...d.meta, coop };
  }
  return d;
}
const MODE_KIND = { classic: "grid2d", walls: "walls", islands: "islands", cube: "grid3d" };
const NEW_KINDS = ["portals", "torus", "hex", "tri", "oneway", "overpass", "keys", "cubesurf", "coop"];
function describe(entry, seed) {
  const d = entry.puzzle;
  return { id: entry.id, seed, kind: MODE_KIND[entry.mode] || d.kind, size: entry.size, unique: true,
    num_nodes: d.coords.length, seconds: 0, puzzle: publicPuzzle(entry),
    rating: { label: entry.label, score: entry.score, mode: entry.mode, diff: entry.diff }, bank_id: entry.id, source: "bank" };
}
// same as bank.seed_of on the server: a stable integer from the entry id's hex digits
function seedOf(entry) { const x = parseInt(String(entry.id).slice(1), 16); return Number.isFinite(x) ? x % 1e9 : 0; }

// ------------------------------------------------------------------ pools
async function index() { return bank("index.json"); }
async function pool(mode, diff) {
  const idx = await index();
  const m = idx.modes && idx.modes[mode];
  if (!m) throw httpError(404, `no ${mode} puzzles in the bank`);
  const d = m[diff] || m.medium || m[Object.keys(m)[0]];
  const list = await bank(d.file);
  if (!list.length) throw httpError(404, `no ${mode} puzzles in the bank`);
  return list;
}
// /api/generate body -> bank pool: its `bank` field ("<mode>-<diff>", see play/modes.js), else the pool that
// plays most like kind + size
function poolKey({ bank: b, kind, size }) {
  if (typeof b === "string" && b.includes("-")) { const i = b.indexOf("-"); return [b.slice(0, i), b.slice(i + 1)]; }
  const n = Array.isArray(size) ? Math.max(...size) : Number(size) || 7;
  if (kind === "grid2d" || kind === "grid" || kind === "mask") return ["classic", n <= 5 ? "easy" : n <= 7 ? "medium" : "hard"];
  if (kind === "walls") return ["walls", "medium"];
  if (kind === "islands") return ["islands", "medium"];
  if (kind === "grid3d" || kind === "grid4d") return ["cube", "hard"];
  if (NEW_KINDS.includes(kind)) return [kind, "medium"];
  throw httpError(400, `unknown kind ${kind}`);
}
async function generate(body = {}) {
  const [mode, diff] = poolKey(body);
  const list = await pool(mode, diff);
  const seed = Number.isInteger(body.seed) && body.seed >= 0 ? body.seed : Math.floor(Math.random() * 1e9);
  const entry = list[seed % list.length];
  remember(entry);
  return describe(entry, seed);
}

// ------------------------------------------------------------------ daily
const DIFF_SLOT = { easy: "easy", medium: "medium", hard: "hard", expert: "special", insane: "special", special: "special" };
// no bank record for the slot (e.g. "special" is set on Sundays only): a pool puzzle for the day instead
const DIFF_POOL = { easy: ["classic", "easy"], medium: ["classic", "medium"], hard: ["classic", "hard"],
  expert: ["islands", "medium"], insane: ["cube", "hard"], special: ["islands", "medium"] };
const DIFF_LABEL = { easy: "Easy", medium: "Medium", hard: "Hard", expert: "Expert", insane: "Insane", special: "Special" };
const EPOCH = Date.UTC(2026, 0, 1);
const dayNum = (iso) => { const [y, m, d] = iso.split("-").map(Number); return Math.round((Date.UTC(y, m - 1, d) - EPOCH) / 86400000); };
const isoOf = (k) => new Date(EPOCH + k * 86400000).toISOString().slice(0, 10);
function localISO(d = new Date()) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
async function daily(params) {
  const difficulty = params.get("difficulty") || "medium";
  if (!DIFF_SLOT[difficulty]) throw httpError(400, "difficulty must be one of easy, medium, hard, expert, insane, special");
  let date = params.get("date") || localISO();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(dayNum(date))) throw httpError(422, "date must be YYYY-MM-DD");
  const number = dayNum(date) + 1;
  // outside the bank's calendar: wrap around (same puzzle as the day that many periods earlier / later)
  const idx = await index();
  const lo = dayNum(idx.daily_start || "2026-01-01"), hi = dayNum(idx.daily_end || isoOf(lo + 364));
  let k = dayNum(date);
  if (k < lo || k > hi) k = lo + ((((k - lo) % (hi - lo + 1)) + (hi - lo + 1)) % (hi - lo + 1));
  const bankDate = isoOf(k);
  let entry = null;
  try {
    const month = await bank(`daily/${bankDate.slice(0, 7)}.json`);
    const day = month[bankDate];
    if (day) entry = day[DIFF_SLOT[difficulty]] || null;
  } catch { /* month missing: fall back to the pools below */ }
  if (!entry) {
    const list = await pool(...DIFF_POOL[difficulty]);
    entry = list[((k % list.length) + list.length) % list.length];
  }
  remember(entry);
  const out = describe(entry, seedOf(entry));
  out.daily = { date, number, difficulty, label: DIFF_LABEL[difficulty] };
  return out;
}

// ------------------------------------------------------------------ rules (mirror of engine.validate_prefix)
function graphOf(d) {
  const n = d.coords.length;
  const adj = Array.from({ length: n }, () => new Set());
  for (const [u, v] of d.edges) { adj[u].add(v); adj[v].add(u); }
  return { n, adj };
}
function validatePrefix(d, path) {
  const { n, adj } = graphOf(d);
  const blocked = new Set((d.arcs || []).map(([u, v]) => `${v}>${u}`));
  const prec = new Map();
  for (const [a, b] of d.precedence || []) { if (!prec.has(b)) prec.set(b, []); prec.get(b).push(a); }
  const cps = d.checkpoints, cpi = new Map(cps.map((c, i) => [c, i]));
  if (!path.length) return "empty path";
  if (path[0] !== cps[0]) return "path must start at checkpoint 1";
  const seen = new Set();
  let nxt = 0;
  for (let i = 0; i < path.length; i++) {
    const v = path[i];
    if (!Number.isInteger(v) || v < 0 || v >= n) return `invalid node id ${v}`;
    if (seen.has(v)) return `node ${v} visited twice`;
    seen.add(v);
    if (i > 0 && !adj[path[i - 1]].has(v)) return `nodes ${path[i - 1]} and ${v} are not adjacent`;
    if (i > 0 && blocked.has(`${path[i - 1]}>${v}`)) return `one-way: can't go from ${path[i - 1]} to ${v}`;
    for (const a of prec.get(v) || []) if (!seen.has(a)) return `node ${a} must come before node ${v}`;
    const k = cpi.get(v);
    if (k !== undefined) {
      if (k !== nxt) return `checkpoint ${k + 1} reached before checkpoint ${nxt + 1}`;
      nxt++;
    }
    if (v === cps[cps.length - 1] && i !== n - 1) return "the last checkpoint must be the final cell";
  }
  return null;
}
function checkSolution(d, path) {
  const bad = validatePrefix(d, path);
  if (bad) return bad;
  if (path.length !== d.coords.length) return `path covers ${path.length} of ${d.coords.length} cells`;
  return null;
}
function check({ puzzle, path = [] }) {
  const full = checkSolution(puzzle, path);
  const partial = path.length ? validatePrefix(puzzle, path) : "empty path";
  return { valid: full === null, reason: full, legal_prefix: partial === null, prefix_reason: partial,
    length: path.length, num_nodes: puzzle.coords.length };
}
async function solutionOf(puzzle, id) {
  const e = await lookup(puzzle, id);
  const sol = e && e.puzzle.solution;
  return sol && checkSolution(puzzle, sol) === null ? sol : null;
}
const lcp = (a, b) => { let k = 0; while (k < a.length && k < b.length && a[k] === b[k]) k++; return k; };

async function hint({ puzzle, path = [], id }) {
  path = path.length ? path : [puzzle.checkpoints[0]];
  const bad = validatePrefix(puzzle, path);
  if (bad) return { status: "invalid", message: bad };
  if (path.length === puzzle.coords.length) return { status: "done", message: "Already solved!" };
  const sol = await solutionOf(puzzle, id);
  if (!sol) return { status: "timeout", message: "No hint for this puzzle." };
  const k = lcp(path, sol);
  if (k === path.length) return { status: "next", next: sol[k], keep: k, source: "solution", nodes_expanded: 0, seconds: 0 };
  // the solution is unique: the longest completable prefix is the common prefix
  const keep = Math.max(1, k);
  return { status: "backtrack", keep, next: sol[keep], source: "solution", nodes_expanded: 0, seconds: 0,
    message: `Your path can't be completed - backtrack to step ${keep}.` };
}

// ------------------------------------------------------------------ co-op (two paths; mirror of zipsolve.coop rules)
function coopCps(d) { return (d.meta && d.meta.coop && d.meta.coop.checkpoints) || [d.checkpoints, []]; }
// null when legal so far, else the reason; paths may be partial ([] = not started)
function coopPrefix(d, paths) {
  const { n, adj } = graphOf(d), cps = coopCps(d), seen = new Set();
  const owner = new Map();
  cps.forEach((cs, i) => cs.forEach((c) => owner.set(c, i)));
  for (let i = 0; i < 2; i++) {
    const p = paths[i] || [], cs = cps[i], cpi = new Map(cs.map((c, k) => [c, k]));
    if (!p.length) continue;
    if (p[0] !== cs[0]) return `path ${i + 1} must start at its checkpoint 1`;
    let nxt = 0;
    for (let k = 0; k < p.length; k++) {
      const v = p[k];
      if (!Number.isInteger(v) || v < 0 || v >= n) return `invalid node id ${v}`;
      if (seen.has(v)) return `node ${v} visited twice`;
      seen.add(v);
      if (k > 0 && !adj[p[k - 1]].has(v)) return `nodes ${p[k - 1]} and ${v} are not adjacent`;
      if (owner.has(v) && owner.get(v) !== i) return `path ${i + 1} steps on the other path's checkpoint`;
      const c = cpi.get(v);
      if (c !== undefined) { if (c !== nxt) return `path ${i + 1}: checkpoint ${c + 1} before ${nxt + 1}`; nxt++; }
      if (v === cs[cs.length - 1] && k !== p.length - 1) return `path ${i + 1} must end at its last checkpoint`;
    }
  }
  return null;
}
function coopDone(d, paths) {
  const cps = coopCps(d);
  const bad = coopPrefix(d, paths);
  if (bad) return bad;
  for (let i = 0; i < 2; i++) { const p = paths[i] || [], cs = cps[i]; if (p[p.length - 1] !== cs[cs.length - 1]) return `path ${i + 1} is not finished`; }
  const tot = (paths[0] || []).length + (paths[1] || []).length;
  return tot === d.coords.length ? null : `paths cover ${tot} of ${d.coords.length} cells`;
}
async function coopSolutionOf(puzzle, id) {
  const e = await lookup(puzzle, id);
  const s = e && e.puzzle.meta && e.puzzle.meta.coop && e.puzzle.meta.coop.solution;
  const paths = s && s.paths;
  return paths && paths.length === 2 && coopDone(puzzle, paths) === null ? paths : null;
}
function coopCheck({ puzzle, path = [], paths }) {
  paths = paths || [path, []];
  const partial = coopPrefix(puzzle, paths), full = coopDone(puzzle, paths), cps = coopCps(puzzle);
  const covered = (paths[0] || []).length + (paths[1] || []).length;
  return { valid: full === null, reason: full, legal: partial === null, reason_partial: partial, covered, num_nodes: puzzle.coords.length,
    paths: [0, 1].map((i) => { const p = paths[i] || [], cs = cps[i]; return { legal: partial === null, reason: partial, length: p.length,
      complete: p.length > 0 && p[p.length - 1] === cs[cs.length - 1], next_checkpoint: null, moves: [] }; }) };
}
async function coopHint({ puzzle, path = [], paths, path_index: pi = 0, id }) {
  paths = (paths || [path, []]).map((p, i) => (p && p.length ? p : [coopCps(puzzle)[i][0]]));
  if (coopDone(puzzle, paths) === null) return { status: "done", message: "Already solved!" };
  const sol = await coopSolutionOf(puzzle, id);
  if (!sol) return { status: "timeout", message: "No hint for this puzzle." };
  const k = paths.map((p, i) => lcp(p, sol[i])), ok = paths.map((p, i) => k[i] === p.length);
  const open = (i) => paths[i].length < sol[i].length;
  let i = pi === 1 ? 1 : 0;
  if (ok[0] && ok[1]) {
    if (!open(i)) i = 1 - i;
    return { status: "next", path_index: i, which: i, next: sol[i][paths[i].length], keep: paths.map((p) => p.length), source: "solution", nodes_expanded: 0, seconds: 0 };
  }
  const keep = k.map((x) => Math.max(1, x));
  if (!(keep[i] < sol[i].length)) i = 1 - i;
  return { status: "backtrack", path_index: i, which: i, next: sol[i][keep[i]], keep, source: "solution", nodes_expanded: 0, seconds: 0,
    message: "Your paths can't be completed - backing up." };
}

// ------------------------------------------------------------------ robots (recorded runs)
const MODE_BOT = { greedy: "rookie", search: "scout", hybrid: "grandmaster", exact: "tortoise" };
// the strategy robots (zipsolve.robots) whose runs the bank recorded
const STRAT = [
  { id: "detective", name: "Detective", emoji: "🕵️", line: "Explains every move, guesses only when stuck" },
  { id: "mcts", name: "Sage", emoji: "🌳", line: "Grows a search tree of possible futures (AlphaZero-style)" },
  { id: "evolver", name: "Evolver", emoji: "🧬", line: "Breeds and mutates whole lines (genetic algorithm)" },
  { id: "gambler", name: "Gambler", emoji: "🎲", line: "Plays random futures and bets on the best odds" },
  { id: "sat", name: "Mathematician", emoji: "🧮", line: "Turns the puzzle into logic and calls a SAT solver" },
];
const ALIAS = { sage: "mcts", alphazero: "mcts", mathematician: "sat", genetic: "evolver" };
async function robotRun(puzzle, bot, id) {
  const e = await lookup(puzzle, id);
  if (!e) throw httpError(404, "no recorded robot runs for this puzzle");
  const runs = await bank(`robots/${e.id}.json`);
  const r = runs[bot];
  if (!r) throw httpError(404, `no ${bot} run for this puzzle`);
  return clone(r);
}
// compact bank form -> /api/solve/robot response (steps are stored as parallel lists: "sp" = p, "sh" = how)
function expandRun(r, meta) {
  const path = r.path || [], sp = r.sp || [], sh = r.sh || null;
  const steps = path.slice(1).map((node, i) => (sh ? { node, p: sp[i] ?? null, how: sh[i] ?? null } : { node, p: sp[i] ?? null }));
  delete r.sp; delete r.sh;
  return { robot: meta.id, robot_name: meta.name, emoji: meta.emoji, start_len: 1, trace: null, trace_truncated: false, ...r, steps };
}
async function solveRobot(body) {
  const key = ALIAS[String(body.robot || "").toLowerCase()] || String(body.robot || "").toLowerCase();
  const meta = STRAT.find((b) => b.id === key);
  if (!meta) throw httpError(400, `unknown robot ${body.robot}`);
  if (body.puzzle && body.puzzle.kind === "coop") return { robot: key, robot_name: meta.name, emoji: meta.emoji, status: "unsupported", solved: false, path: [], steps: [], trace: null, reason: "co-op puzzles are for humans" };
  let r;
  try { r = await robotRun(body.puzzle, key, body.id); }
  catch (e) { throw httpError(404, `${meta.name} hasn't studied this puzzle online - try another one (or the local app)`); }
  r = expandRun(r, meta);
  if (body.trace === false) r.trace = null;
  return r;
}
async function robotsList() {
  const idx = await index();
  const have = new Set(((idx.more_modes && idx.more_modes.robots) || []));
  return { robots: STRAT.map((b) => ({ ...b, available: have.has(b.id) })) };
}
// the Detective's teaching hint: its recorded notes along the unique solution, keyed by how far the line is
const reasons = new Map();
async function detectiveReasons(e) {
  if (reasons.has(e.id)) return reasons.get(e.id);
  const out = new Map();
  try {
    const run = (await bank(`robots/${e.id}.json`)).detective, sol = e.puzzle.solution;
    let stack = [];
    for (const ev of (run && run.trace) || []) {
      if (ev === "R") stack = [];
      else if (typeof ev === "number") { if (ev >= 0) stack.push(ev); else stack.splice(stack.length + ev); }
      else if (ev && ev.t === "path") stack = (ev.p || []).slice();
      else if (ev && ev.t === "note" && ev.move != null && (ev.kind === "deduce" || ev.kind === "forced")) {
        const k = stack.length;
        if (k < sol.length && sol[k] === ev.move && lcp(stack, sol) === k && !out.has(k)) out.set(k, ev);
      }
    }
  } catch { /* no recorded Detective run */ }
  reasons.set(e.id, out);
  return out;
}
const TECH = { rules: "Rules", L1: "Local dead end", L2: "Regions", LA1: "Short lookahead", L3: "Expert rule", LA2: "Deeper lookahead" };
async function explain({ puzzle, path = [], id }) {
  const e = await lookup(puzzle, id);
  if (!e || !e.puzzle.solution) throw httpError(404, "no explanation for this puzzle");
  path = path.length ? path : [puzzle.checkpoints[0]];
  const sol = e.puzzle.solution, k = path.length;
  if (lcp(path, sol) !== k || k >= sol.length) throw httpError(404, "no explanation here");
  const note = (await detectiveReasons(e)).get(k);
  if (!note) throw httpError(404, "no explanation here");
  let reason = String(note.msg || "").trim();
  if (reason && !/[.!?]$/.test(reason)) reason += ".";
  return { status: "next", move: sol[k], reason, technique: note.technique || "rules", technique_name: TECH[note.technique] || "Deduction",
    cells: note.cells || [sol[k]], seconds: 0, source: "recorded" };
}

// ------------------------------------------------------------------ the Architect (online: its certified designs)
async function architectDesign(body = {}) {
  const idx = await index();
  const pools = (idx.modes && idx.modes.architect) || {};
  const diff = body.target === "insane" && pools.insane ? "insane" : pools.expert ? "expert" : Object.keys(pools)[0];
  if (!diff) throw httpError(501, "no Architect designs in the bank");
  let list = await bank(pools[diff].file);
  if (body.kind) { const same = list.filter((x) => x.mode === body.kind); if (same.length) list = same; }
  const seed = Number.isInteger(body.seed) && body.seed >= 0 ? body.seed : Math.floor(Math.random() * 1e9);
  const e = list[seed % list.length];
  remember(e);
  await new Promise((res) => setTimeout(res, 2600 + (seed % 1400)));   // "designing…" (the real one takes ~20 s)
  const a = e.architect || {};
  const out = describe(e, seed);
  out.rating = { label: e.label, score: e.score, boring: false, guesses: a.guesses ?? 0, max_lookahead: a.max_lookahead ?? null,
    spread: a.spread ?? null, checkpoints: a.clues ?? e.puzzle.checkpoints.length, counts: a.counts || {} };
  out.stats = { evaluations: a.evaluations || 0, seconds: a.seconds || 0, recorded: true };
  out.log = [{ phase: "start", msg: `A ${e.size} ${e.mode} board, aiming for ${a.target || diff}.` },
    { phase: "anneal", msg: `Tried ${a.evaluations || "many"} designs, kept the meanest fair one.` },
    { phase: "verify", msg: "Exactly one solution, no guessing needed." }];
  return { ...out, ok: true, on_target: true, kind: e.mode, target: a.target || diff, source: "architect" };
}
function isoWeek(d = new Date()) {
  const t = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const day = t.getUTCDay() || 7;
  t.setUTCDate(t.getUTCDate() + 4 - day);
  const y = t.getUTCFullYear(), w = Math.ceil(((t - Date.UTC(y, 0, 1)) / 86400000 + 1) / 7);
  return `${y}-W${String(w).padStart(2, "0")}`;
}
async function weekly(params) {
  const week = params.get("week") || isoWeek();
  let wk;
  try { wk = await bank("architect/weekly.json"); } catch { throw httpError(404, "no weekly Architect challenge in the bank"); }
  const keys = Object.keys(wk).sort();
  let key = week;
  if (!wk[key]) {
    const m = /^(\d{4})-W(\d{2})$/.exec(week);
    if (!m || !keys.length) throw httpError(422, "week must be YYYY-Www");
    key = keys[(Number(m[1]) * 53 + Number(m[2])) % keys.length];   // same wrap-around as the server
  }
  const e = wk[key];
  remember(e);
  const out = describe(e, seedOf(e));
  out.weekly = { week, bank_week: key, number: e.number ?? null, label: e.label, clues: (e.architect || {}).clues ?? null };
  return out;
}

async function solveExact(body) {
  const { puzzle, start_path: start } = body;
  if (puzzle && puzzle.kind === "coop") {
    const paths = await coopSolutionOf(puzzle, body.id);
    return paths ? { status: "solved", paths: paths.map((p) => p.slice()), path: null, solved: true, nodes_expanded: 0, seconds: 0, kind: "coop" }
      : { status: "timeout", paths: null, path: null, solved: false, nodes_expanded: 0, seconds: 0, kind: "coop" };
  }
  if (body.trace && !(start && start.length > 1)) {
    try { return await robotRun(puzzle, "tortoise"); } catch { /* fall back to the stored solution */ }
  }
  const sol = await solutionOf(puzzle);
  if (!sol) return { status: "timeout", path: null, nodes_expanded: 0, seconds: 0, solved: false };
  if (start && start.length > 1) {
    const bad = validatePrefix(puzzle, start);
    if (bad) throw httpError(400, `start_path: ${bad}`);
    if (lcp(start, sol) < start.length) return { status: "unsat", path: null, nodes_expanded: 0, seconds: 0, solved: false };
  }
  return { status: "solved", path: sol.slice(), nodes_expanded: 0, seconds: 0, solved: true };
}
async function solveRL(body) {
  const bot = MODE_BOT[body.mode];
  if (!bot || bot === "tortoise") throw httpError(400, "mode must be one of greedy, search, hybrid");
  if (body.start_path && body.start_path.length > 1) throw httpError(501, "the online robots start from checkpoint 1");
  const r = await robotRun(body.puzzle, bot);
  r.model = "robot"; r.model_stage = null;
  return r;
}

// ------------------------------------------------------------------ dispatch
const PRESETS = [
  { id: "easy", label: "Easy", kind: "grid2d", size: 5, unique: true, options: {}, blurb: "5×5 grid" },
  { id: "medium", label: "Medium", kind: "grid2d", size: 7, unique: true, options: {}, blurb: "7×7 grid" },
  { id: "hard", label: "Hard", kind: "walls", size: 8, unique: true, options: { walls_frac: 0.35 }, blurb: "8×8 with walls" },
];
const LOCAL_ONLY = "This needs the local app (python -m zipsolve.app)";

export async function staticApi(url, body) {
  const u = new URL(url, location.href);
  const p = u.pathname.replace(/^.*?\/api\//, "/api/");
  switch (p) {
    case "/api/models":
      return { directory: "", models: [{ name: "robot", file: "robot", group: "", size: 0, mtime: 0, default: true }] };
    case "/api/presets": return { presets: PRESETS, daily_epoch: "2026-01-01" };
    case "/api/bank": {
      const idx = await index();
      return { available: true, modes: Object.fromEntries(Object.entries(idx.modes || {}).map(([m, ds]) => [m, Object.fromEntries(Object.entries(ds).map(([k, v]) => [k, v.count]))])) };
    }
    case "/api/generate": return generate(body);
    case "/api/daily": return daily(u.searchParams);
    case "/api/check": return body && body.puzzle && body.puzzle.kind === "coop" ? coopCheck(body) : check(body);
    case "/api/hint": return body && body.puzzle && body.puzzle.kind === "coop" ? coopHint(body) : hint(body);
    case "/api/hint/explain": return explain(body);
    case "/api/solve/exact": return solveExact(body);
    case "/api/solve/rl": return solveRL(body);
    case "/api/solve/robot": return solveRobot(body);
    case "/api/robots": return robotsList();
    case "/api/architect/design": return architectDesign(body);
    case "/api/architect/weekly": return weekly(u.searchParams);
    default: throw httpError(501, LOCAL_ONLY);
  }
}
