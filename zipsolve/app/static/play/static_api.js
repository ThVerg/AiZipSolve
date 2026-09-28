// Static API adapter: the online (GitHub Pages) build has no Python server, so the game and the AI show
// talk to this module instead of /api/* (util.js api() routes here when window.ZIP_STATIC is set).
// Everything comes from the precomputed puzzle bank under static/bank/:
//   index.json                      modes -> difficulties -> pool files
//   pools/<mode>-<diff>.json        [{id, mode, diff, label, score, size, puzzle (with its unique solution)}]
//   daily/<yyyy>-<mm>.json          {"YYYY-MM-DD": {number, easy, medium, hard, special}}
//   robots/<id>.json                {rookie, scout, grandmaster, tortoise}: recorded /api/solve/* runs
// Hints and "Show me" use the stored solution (the bank's puzzles have exactly one, so a path that leaves it
// can only be fixed by backing up to where it left). The AI's "vision" (/api/policy) and everything local-only
// (editor, custom puzzles, workbench, dashboard) answer 501.

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
function publicPuzzle(entry) {
  const d = { ...entry.puzzle };
  delete d.solution;
  return d;
}
const MODE_KIND = { classic: "grid2d", walls: "walls", islands: "islands", cube: "grid3d" };
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
function solutionOf(puzzle, id) {
  const e = entryFor(puzzle, id);
  const sol = e && e.puzzle.solution;
  return sol && checkSolution(puzzle, sol) === null ? sol : null;
}
const lcp = (a, b) => { let k = 0; while (k < a.length && k < b.length && a[k] === b[k]) k++; return k; };

function hint({ puzzle, path = [], id }) {
  path = path.length ? path : [puzzle.checkpoints[0]];
  const bad = validatePrefix(puzzle, path);
  if (bad) return { status: "invalid", message: bad };
  if (path.length === puzzle.coords.length) return { status: "done", message: "Already solved!" };
  const sol = solutionOf(puzzle, id);
  if (!sol) return { status: "timeout", message: "No hint for this puzzle." };
  const k = lcp(path, sol);
  if (k === path.length) return { status: "next", next: sol[k], keep: k, source: "solution", nodes_expanded: 0, seconds: 0 };
  // the solution is unique: the longest completable prefix is the common prefix
  const keep = Math.max(1, k);
  return { status: "backtrack", keep, next: sol[keep], source: "solution", nodes_expanded: 0, seconds: 0,
    message: `Your path can't be completed - backtrack to step ${keep}.` };
}

// ------------------------------------------------------------------ robots (recorded runs)
const MODE_BOT = { greedy: "rookie", search: "scout", hybrid: "grandmaster", exact: "tortoise" };
async function robotRun(puzzle, bot) {
  const e = entryFor(puzzle);
  if (!e) throw httpError(404, "no recorded robot runs for this puzzle");
  const runs = await bank(`robots/${e.id}.json`);
  const r = runs[bot];
  if (!r) throw httpError(404, `no ${bot} run for this puzzle`);
  return clone(r);
}
async function solveExact(body) {
  const { puzzle, start_path: start } = body;
  if (body.trace && !(start && start.length > 1)) {
    try { return await robotRun(puzzle, "tortoise"); } catch { /* fall back to the stored solution */ }
  }
  const sol = solutionOf(puzzle);
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
    case "/api/generate": return generate(body);
    case "/api/daily": return daily(u.searchParams);
    case "/api/check": return check(body);
    case "/api/hint": return hint(body);
    case "/api/solve/exact": return solveExact(body);
    case "/api/solve/rl": return solveRL(body);
    default: throw httpError(501, LOCAL_ONLY);
  }
}
