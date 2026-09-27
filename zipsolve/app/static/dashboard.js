/* Training dashboard: run cards, uPlot training curves, per-family small multiples,
   benchmark / eval / checkpoint tables, remote sync. Data: /api/dashboard/* (dashboard_api.py).
   Colour follows the run (fixed slot while it is selected), never its rank. */
"use strict";

const API = "/api/dashboard";
const MAX_SEL = 8;
const LS_KEY = "zip-dash";

const METRICS = [
  { key: "solve_rate", title: "Training solve rate", sub: "share of finished episodes solved", fmt: "pct", range: [0, 1], stageLabels: true },
  { key: "@val", title: "Validation solve rate", sub: "macro mean over families · greedy", fmt: "pct", range: [0, 1], points: true },
  { key: "ep_return", title: "Episode return" },
  { key: "ep_len", title: "Episode length", sub: "moves" },
  { key: "frac_visited", title: "Cells visited", sub: "share of the graph covered", fmt: "pct", range: [0, 1] },
  { key: "entropy", title: "Policy entropy" },
  { key: "kl", title: "Approx. KL", sub: "old → new policy" },
  { key: "v_loss", title: "Value loss" },
  { key: "explained_var", title: "Explained variance", sub: "value head" },
  { key: "grad_norm", title: "Gradient norm", sub: "before clipping" },
  { key: "clipfrac", title: "Clip fraction", fmt: "pct" },
  { key: "pg_loss", title: "Policy loss" },
];
const KNOWN = new Set(METRICS.map((m) => m.key));
const FAMILY_ORDER = ["grid2d", "walls", "mask", "islands", "islands_chain", "grid3d", "grid4d"];

const S = {
  runs: [], byId: {}, selected: [], slot: {}, data: {}, x: "iter", smooth: 0.8, raw: true, stages: true,
  extra: [], famSrc: "val_greedy", refreshSec: 30, charts: [], xRange: null, hover: null,
  bench: [], benchSig: "", evalSig: "", ckpts: [], ckptSig: "", metaQueue: new Set(), metaBusy: false,
  metaOpen: new Set(), sync: null, syncPoll: null, logRun: null, famSrcUser: false,
};

// ------------------------------------------------------------------ utils
const $ = (id) => document.getElementById(id);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}
function hexRgb(hex) {
  hex = (hex || "").trim();
  if (hex.startsWith("rgb")) { const m = hex.match(/[\d.]+/g).map(Number); return m.slice(0, 3); }
  hex = hex.replace("#", "");
  if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
  const n = parseInt(hex || "888888", 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const rgba = (hex, a) => { const [r, g, b] = hexRgb(hex); return `rgba(${r},${g},${b},${a})`; };
function mix(c1, c2, t) { const a = hexRgb(c1), b = hexRgb(c2); return a.map((v, i) => Math.round(v + (b[i] - v) * t)); }
function lum([r, g, b]) {
  const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
function cellColors(rgb) {
  const L = lum(rgb);
  const white = (1.05) / (L + 0.05), dark = (L + 0.05) / 0.05;
  return { bg: `rgb(${rgb.join(",")})`, fg: white >= 4.5 || white > dark ? "#fff" : "#141414" };
}
function fmtNum(v, digits = 3) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "K";
  if (a >= 100) return Math.round(v).toLocaleString();
  if (a === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  return Number(v.toPrecision(digits)).toString();
}
const fmtPct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%");
function fmtDur(s) {
  if (s == null || !isFinite(s)) return "–";
  s = Math.max(0, s);
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}
function fmtAgo(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  if (sec < 45) return "just now";
  if (sec < 3600) return `${Math.round(sec / 60)} min ago`;
  if (sec < 86400) return `${(sec / 3600).toFixed(sec < 36000 ? 1 : 0)} h ago`;
  return `${Math.round(sec / 86400)} d ago`;
}
const fmtBytes = (b) => (b >= 1 << 20 ? (b / (1 << 20)).toFixed(1) + " MB" : Math.round(b / 1024) + " KB");
function fmtX(v, x) { return x === "time" ? fmtDur(v) : fmtNum(v); }
function fmtVal(v, spec) { return spec && spec.fmt === "pct" ? fmtPct(v) : fmtNum(v, 4); }
function familyKey(f) {
  const [kind, size] = String(f).split(":");
  const i = FAMILY_ORDER.indexOf(kind);
  return [i < 0 ? 99 : i, kind, parseFloat(size) || 0];
}
const famSort = (a, b) => { const x = familyKey(a), y = familyKey(b); return x[0] - y[0] || (x[1] < y[1] ? -1 : x[1] > y[1] ? 1 : 0) || x[2] - y[2]; };
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

function save() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ selected: S.selected, slot: S.slot, x: S.x, smooth: S.smooth, raw: S.raw,
      stages: S.stages, extra: S.extra, refreshSec: S.refreshSec, famSrc: S.famSrcUser ? S.famSrc : null }));
  } catch { /* private mode */ }
}
function load() {
  try {
    const o = JSON.parse(localStorage.getItem(LS_KEY) || "null");
    if (!o) return false;
    Object.assign(S, { x: o.x || "iter", smooth: o.smooth ?? 0.8, raw: o.raw ?? true, stages: o.stages ?? true,
      extra: o.extra || [], refreshSec: o.refreshSec ?? 30 });
    if (o.famSrc) { S.famSrc = o.famSrc; S.famSrcUser = true; }
    S.selected = Array.isArray(o.selected) ? o.selected : [];
    if (o.slot && typeof o.slot === "object") for (const id of S.selected) if (Number.isInteger(o.slot[id]) && o.slot[id] >= 0 && o.slot[id] < MAX_SEL && !Object.values(S.slot).includes(o.slot[id])) S.slot[id] = o.slot[id];
    return true;
  } catch { return false; }
}

// ------------------------------------------------------------------ colours
function palette() {
  return {
    series: Array.from({ length: 8 }, (_, i) => cssVar(`--series-${i + 1}`)),
    surface: cssVar("--panel"), grid: cssVar("--grid"), axis: cssVar("--axis"), axisLine: cssVar("--axis-line"),
    ink: cssVar("--ink"), ink3: cssVar("--ink-3"), seqHi: cssVar("--seq-hi"),
    divPos: cssVar("--div-pos"), divNeg: cssVar("--div-neg"), divMid: cssVar("--div-mid"),
  };
}
let P = null;
function runColor(id) { const s = S.slot[id]; return s == null ? P.ink3 : P.series[s]; }
function assignSlot(id) {
  if (S.slot[id] != null) return;
  const used = new Set(S.selected.filter((x) => x !== id).map((x) => S.slot[x]));
  for (let i = 0; i < MAX_SEL; i++) if (!used.has(i)) { S.slot[id] = i; return; }
}

// ------------------------------------------------------------------ theme
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  try { t ? localStorage.setItem("zip-theme", t) : localStorage.removeItem("zip-theme"); } catch { /* ignore */ }
  rethemeAll();
}
function rethemeAll() {
  P = palette();
  renderCards();
  renderCharts();
  renderBench();
  renderEval();
}

// ------------------------------------------------------------------ runs
async function loadRuns() {
  const r = await getJSON(`${API}/runs`);
  S.runs = r.runs;
  S.byId = Object.fromEntries(r.runs.map((x) => [x.id, x]));
  S.serverNow = r.now;
  setSync(r.sync);
  S.selected = S.selected.filter((id) => S.byId[id]);
  if (!S.selected.length && !S.initDone) {
    const running = S.runs.filter((x) => x.status === "running").slice(0, 4);
    S.selected = (running.length ? running : S.runs.slice(0, 2)).map((x) => x.id);
  }
  S.initDone = true;
  for (const id of Object.keys(S.slot)) if (!S.selected.includes(id)) delete S.slot[id];
  S.selected.forEach(assignSlot);
  save();
}

async function loadSeries(force) {
  const jobs = S.selected.map(async (id) => {
    const run = S.byId[id];
    const cur = S.data[id];
    if (!force && cur && cur.mtime === run.mtime && cur.size === run.size) return false;
    const [src, name] = [run.source, run.name];
    const d = await getJSON(`${API}/runs/${encodeURIComponent(src)}/${encodeURIComponent(name)}?max_points=3000`);
    S.data[id] = { mtime: run.mtime, size: run.size, d };
    return true;
  });
  const res = await Promise.allSettled(jobs);
  return res.some((r) => r.status === "fulfilled" && r.value);
}

function stageInfo(run) {
  const planned = run.stages_planned && run.stages_planned.length ? run.stages_planned : run.stages_seen || [];
  return { planned, idx: run.stage_index };
}

function sparkSVG(vals) {
  const W = 120, H = 40, pad = 3;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("class", "run-spark");
  svg.setAttribute("aria-hidden", "true");
  if (!vals || vals.length < 2) return svg;
  const n = vals.length;
  const pts = vals.map((v, i) => [pad + (i / (n - 1)) * (W - 2 * pad), H - pad - Math.max(0, Math.min(1, v)) * (H - 2 * pad)]);
  const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join("");
  const mk = (tag, attrs) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  svg.append(mk("path", { d: d + `L${pts[n - 1][0]} ${H - pad}L${pts[0][0]} ${H - pad}Z`, class: "ar" }));
  svg.append(mk("path", { d, class: "ln", "vector-effect": "non-scaling-stroke" }));
  return svg;
}

const STATUS_LABEL = { running: "Running", finished: "Finished", idle: "Idle", crashed: "Crashed", error: "Error" };

function renderCards() {
  const box = $("runCards");
  box.replaceChildren();
  if (!S.runs.length) {
    box.append(h("div", { class: "empty" }, "No runs yet. Training writes runs/<name>.csv; press Sync to mirror the server's runs."));
    return;
  }
  for (const run of S.runs) {
    const sel = S.selected.includes(run.id);
    const color = sel ? runColor(run.id) : "";
    const { planned, idx } = stageInfo(run);
    const dots = h("span", { class: "stage-dots", "aria-hidden": "true" },
      planned.slice(0, 12).map((_, i) => h("i", { class: idx == null ? "" : i < idx ? "done" : i === idx ? "cur" : "" })));
    const stageTxt = run.stage ? `Stage ${idx != null ? idx + 1 : "?"}/${planned.length || "?"} · ${run.stage}` : "no stage info";
    const card = h("button", {
      class: "run-card" + (sel ? " sel" : ""), style: sel ? `--run-color:${color}` : null, "aria-pressed": sel ? "true" : "false",
      title: sel ? "Remove from charts" : "Add to charts", onclick: () => toggleRun(run.id),
    },
      h("div", { class: "rc-top" },
        h("span", { class: "rc-key" }),
        h("span", { class: "rc-name", title: run.name }, run.name),
        h("span", { class: "badge" }, run.source),
        h("span", { class: `pill ${run.status}` }, h("i"), STATUS_LABEL[run.status] || run.status)),
      run.error ? h("div", { class: "rc-err" }, run.error) : h("div", { class: "rc-stage", title: run.stage || "" }, planned.length ? dots : null, h("span", null, stageTxt)),
      h("div", { class: "rc-main" },
        h("div", { class: "rc-fig" }, h("b", null, fmtPct(run.solve_rate)), h("small", null, "solve rate (last 10)")),
        run.val_macro != null ? h("div", { class: "rc-fig sm" }, h("b", null, fmtPct(run.val_macro)), h("small", null, "validation")) : null,
        sparkSVG(run.spark)),
      h("div", { class: "rc-foot" },
        h("span", null, "iter ", h("b", null, fmtNum(run.iteration))),
        h("span", null, h("b", null, fmtNum(run.step)), " steps"),
        h("span", null, h("b", null, fmtDur(run.elapsed)), " elapsed"),
        h("span", { title: new Date(run.mtime * 1000).toLocaleString() }, "updated ", h("b", null, fmtAgo(run.age)))));
    box.append(card);
  }
}

async function toggleRun(id) {
  const i = S.selected.indexOf(id);
  if (i >= 0) { S.selected.splice(i, 1); delete S.slot[id]; }
  else {
    if (S.selected.length >= MAX_SEL) { flash(`At most ${MAX_SEL} runs can be compared at once.`); return; }
    S.selected.push(id); assignSlot(id);
  }
  save();
  renderCards();
  await loadSeries(false).catch(showErr);
  renderCharts();
  renderLogPicker();
  if ($("ckptSelected").checked) renderCkpts();
}

function flash(msg) {
  const el = $("refreshStatus");
  el.textContent = msg;
  setTimeout(() => updateRefreshStatus(), 3000);
}
function showErr(e) { console.error(e); flash("Error: " + (e.message || e)); }

// ------------------------------------------------------------------ series helpers
function runXY(id, key) {
  const d = S.data[id] && S.data[id].d;
  if (!d) return null;
  let xs, ys;
  if (key === "@val" || key.startsWith("@val:")) {
    if (!d.val) return null;
    const col = key === "@val" ? "global/_macro/greedy" : key.slice(5);
    xs = d.val.points[S.x]; ys = d.val.series[col];
  } else {
    xs = d.x[S.x]; ys = d.metrics[key];
  }
  if (!xs || !ys) return null;
  const m = new Map();
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] == null || ys[i] == null) continue;
    m.set(xs[i], ys[i]); // later rows win (resumed runs re-log iterations)
  }
  if (!m.size) return null;
  const x = [...m.keys()].sort((a, b) => a - b);
  return { x, y: x.map((k) => m.get(k)) };
}
function ema(y, a) {
  if (a <= 0) return y.slice();
  let last = 0, w = 0;
  return y.map((v) => { last = a * last + (1 - a) * v; w = a * w + (1 - a); return last / w; });
}
function bsearch(xs, v) {
  let lo = 0, hi = xs.length - 1;
  if (v <= xs[0]) return 0;
  if (v >= xs[hi]) return hi;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= v) lo = mid; else hi = mid; }
  return v - xs[lo] < xs[hi] - v ? lo : hi;
}
function stageAt(id, xv) {
  const d = S.data[id] && S.data[id].d;
  if (!d) return null;
  let st = null;
  for (const t of d.transitions) { if (t[S.x] != null && t[S.x] <= xv) st = t.stage; }
  return st;
}

// ------------------------------------------------------------------ charts
function destroyCharts() {
  for (const c of S.charts) { try { c.u.destroy(); } catch { /* ignore */ } c.ro && c.ro.disconnect(); }
  S.charts = [];
}

function yAxisSize(u, values) {
  if (!values) return 40;
  const w = Math.max(...values.map((v) => String(v).length));
  return Math.max(34, 8 + w * 6.6);
}

function makeChart(box, spec, tables, height) {
  const smoothOn = !spec.points && S.smooth > 0;
  const showRaw = smoothOn && S.raw;
  const cols = [];
  const series = [{}];
  const meta = []; // per uPlot series index: {t, raw}
  for (const t of tables) {
    const c = runColor(t.id);
    const cols_t = [t.x];
    if (showRaw) {
      cols_t.push(t.y);
      series.push({ stroke: rgba(c, 0.22), width: 1, spanGaps: true, points: { show: false } });
      meta.push({ t, raw: true });
    }
    cols_t.push(t.s);
    series.push({
      stroke: c, width: 2, spanGaps: true,
      points: { show: spec.points || t.x.length < 3, size: 7, width: 2, stroke: P.surface, fill: c },
    });
    meta.push({ t, raw: false });
    cols.push(cols_t);
  }
  const data = cols.length === 1 ? cols[0] : uPlot.join(cols);
  const font = `11px ${getComputedStyle(document.body).fontFamily}`;
  const axisBase = { stroke: P.axis, font, grid: { stroke: P.grid, width: 1 }, ticks: { show: false } };
  const opts = {
    width: Math.max(200, box.clientWidth), height,
    padding: [8, 10, 0, 2],
    legend: { show: false },
    scales: { x: { time: false }, y: spec.range ? { range: spec.range } : { range: (u, mn, mx) => autoRange(mn, mx) } },
    axes: [
      { ...axisBase, values: (u, vals) => vals.map((v) => fmtX(v, S.x)), space: 64, size: 28 },
      { ...axisBase, values: (u, vals) => vals.map((v) => spec.fmt === "pct" ? Math.round(v * 100) + "%" : fmtNum(v)), size: yAxisSize, space: 28 },
    ],
    series,
    cursor: {
      sync: { key: "dash-x", setSeries: false },
      drag: { x: true, y: false, setScale: false },
      points: {
        size: (u, si) => (meta[si - 1] && !meta[si - 1].raw ? 8 : 0),
        width: 2,
        stroke: () => P.surface,
        fill: (u, si) => (meta[si - 1] ? runColor(meta[si - 1].t.id) : "transparent"),
      },
      focus: { prox: -1 },
    },
    hooks: {
      draw: [(u) => drawStages(u, tables, spec)],
      setCursor: [(u) => tooltip(u, spec, tables)],
      setSelect: [(u) => {
        if (u.select.width < 4) return;
        const min = u.posToVal(u.select.left, "x"), max = u.posToVal(u.select.left + u.select.width, "x");
        u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false);
        setXRange([min, max]);
      }],
    },
  };
  const u = new uPlot(opts, data, box);
  u.over.addEventListener("mouseenter", () => { S.hover = u; });
  u.over.addEventListener("mouseleave", () => { if (S.hover === u) S.hover = null; hideTip(); });
  u.over.addEventListener("dblclick", () => setXRange(null));
  if (S.xRange) u.setScale("x", { min: S.xRange[0], max: S.xRange[1] });
  const ro = new ResizeObserver(() => {
    const w = Math.max(200, box.clientWidth);
    if (Math.abs(w - u.width) > 1) u.setSize({ width: w, height });
  });
  ro.observe(box);
  S.charts.push({ u, ro, data });
  return u;
}

function autoRange(mn, mx) {
  if (mn == null || mx == null) return [0, 1];
  if (mn === mx) { const d = Math.abs(mn) * 0.1 || 1; return [mn - d, mx + d]; }
  const pad = (mx - mn) * 0.06;
  let lo = mn - pad, hi = mx + pad;
  if (mn >= 0 && lo < 0) lo = 0;
  return [lo, hi];
}

function setXRange(r) {
  S.xRange = r;
  for (const c of S.charts) {
    const xs = c.u.data[0];
    if (!xs || !xs.length) continue;
    if (r) c.u.setScale("x", { min: r[0], max: r[1] });
    else c.u.setScale("x", { min: xs[0], max: xs[xs.length - 1] });
  }
}

function drawStages(u, tables, spec) {
  if (!S.stages) return;
  const ctx = u.ctx, { left, top, width, height } = u.bbox;
  const dpr = devicePixelRatio || 1;
  ctx.save();
  ctx.beginPath(); ctx.rect(left, top, width, height); ctx.clip();
  const single = tables.length === 1;
  let lastLabelEnd = -Infinity;
  for (const t of tables) {
    const d = S.data[t.id] && S.data[t.id].d;
    if (!d) continue;
    const c = runColor(t.id);
    d.transitions.forEach((tr, i) => {
      const xv = tr[S.x];
      if (xv == null) return;
      const px = Math.round(u.valToPos(xv, "x", true)) + 0.5;
      if (i > 0 && px >= left && px <= left + width) {
        ctx.strokeStyle = single ? rgba(P.ink3, 0.6) : rgba(c, 0.42);
        ctx.lineWidth = 1 * dpr;
        ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, top + height); ctx.stroke();
      }
      if (single && spec.stageLabels) {
        ctx.font = `${10.5 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
        const label = `${i + 1} · ${tr.stage}`;
        const tw = ctx.measureText(label).width;
        const lx = Math.max(left + 4 * dpr, px + 4 * dpr);
        const next = d.transitions[i + 1];
        const nextPx = next && next[S.x] != null ? u.valToPos(next[S.x], "x", true) : left + width;
        const room = Math.min(nextPx, left + width) - lx - 4 * dpr;
        if (lx > lastLabelEnd && tw <= room) {
          ctx.fillStyle = rgba(P.surface, 0.85);
          ctx.fillRect(lx - 2 * dpr, top + 2 * dpr, tw + 4 * dpr, 14 * dpr);
          ctx.fillStyle = P.ink3;
          ctx.textBaseline = "top";
          ctx.textAlign = "left";
          ctx.fillText(label, lx, top + 4 * dpr);
          lastLabelEnd = lx + tw + 6 * dpr;
        }
      }
    });
  }
  ctx.restore();
}

// one tooltip for the page
function hideTip() { $("tip").hidden = true; }
function tooltip(u, spec, tables) {
  if (S.hover !== u) return;
  const { left, top } = u.cursor;
  if (left == null || left < 0) { hideTip(); return; }
  const xv = u.posToVal(left, "x");
  const rows = [];
  for (const t of tables) {
    if (!t.x.length) continue;
    const i = bsearch(t.x, xv);
    const span = (t.x[t.x.length - 1] - t.x[0]) || 1;
    if (Math.abs(t.x[i] - xv) > span * 0.08 && (xv < t.x[0] || xv > t.x[t.x.length - 1])) continue;
    rows.push({ id: t.id, v: t.s[i], raw: t.y[i], x: t.x[i] });
  }
  if (!rows.length) { hideTip(); return; }
  const tip = $("tip");
  const head = h("div", { class: "tt-h" }, `${{ iter: "Iteration", step: "Env steps", time: "Wall time" }[S.x]} ${fmtX(rows.length === 1 ? rows[0].x : xv, S.x)}`);
  const body = rows.sort((a, b) => (b.v ?? -Infinity) - (a.v ?? -Infinity)).map((r) =>
    h("div", { class: "tt-r" }, h("i", { style: `background:${runColor(r.id)}` }), h("b", null, fmtVal(r.v, spec)),
      h("span", null, (S.byId[r.id] || { name: r.id }).name)));
  const extra = [];
  if (rows.length === 1) {
    const st = stageAt(rows[0].id, rows[0].x);
    if (st) extra.push(h("div", { class: "tt-s" }, "Stage " + st));
    if (!spec.points && S.smooth > 0) extra.push(h("div", { class: "tt-s" }, `raw ${fmtVal(rows[0].raw, spec)}`));
  }
  tip.replaceChildren(head, ...body, ...extra);
  tip.hidden = false;
  const rect = u.over.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = rect.left + left + 14, y = rect.top + top - th / 2;
  if (x + tw > innerWidth - 8) x = rect.left + left - tw - 14;
  x = Math.max(8, x);
  y = Math.max(8, Math.min(innerHeight - th - 8, y));
  tip.style.left = x + "px"; tip.style.top = y + "px";
}

function buildTables(key, smooth) {
  const out = [];
  for (const id of S.selected) {
    const xy = runXY(id, key);
    if (!xy) continue;
    out.push({ id, x: xy.x, y: xy.y, s: smooth ? ema(xy.y, S.smooth) : xy.y });
  }
  return out;
}

function chartCard(spec, tables, opts = {}) {
  const box = h("div", { class: "chart-box" });
  const card = h("div", { class: "card chart-card" + (opts.fam ? " fam" : "") },
    h("div", { class: "ch-head" }, h(opts.fam ? "h4" : "h4", null, spec.title), h("span", { class: "ch-sub" }, spec.sub || ""),
      opts.onClose ? h("button", { class: "ch-close", title: "Remove chart", "aria-label": "Remove chart", onclick: opts.onClose }, "×") : null),
    box);
  return { card, box };
}

function renderLegend() {
  const lg = $("legend");
  lg.replaceChildren();
  if (!S.selected.length) {
    lg.append(h("span", { class: "muted" }, "No runs selected: click a run card above."));
    return;
  }
  for (const id of S.selected) {
    const r = S.byId[id];
    if (!r) continue;
    lg.append(h("span", { class: "lg", style: `--run-color:${runColor(id)}` }, h("i"), r.name, h("small", null, r.source)));
  }
  if (S.stages) lg.append(h("span", { class: "lg lg-stage" }, h("i"), h("small", null, "curriculum stage change")));
}

function availableMetrics() {
  const set = new Set();
  for (const id of S.selected) {
    const d = S.data[id] && S.data[id].d;
    if (d) Object.keys(d.metrics).forEach((k) => set.add(k));
  }
  return set;
}

const HAS_UPLOT = () => typeof uPlot !== "undefined";
function renderCharts() {
  if (!P) P = palette();
  if (!HAS_UPLOT()) return;
  destroyCharts();
  hideTip();
  renderLegend();
  const grid = $("charts");
  grid.replaceChildren();
  const avail = availableMetrics();
  // "more metrics" picker: everything else that exists (flattened JSON columns included)
  const more = $("moreMetric");
  const others = [...avail].filter((k) => !KNOWN.has(k) && !S.extra.includes(k) && !k.startsWith("family_sr.")).sort();
  more.replaceChildren(h("option", { value: "" }, others.length ? "More metrics…" : "No other metrics"),
    ...others.map((k) => h("option", { value: k }, k)));
  const specs = [...METRICS, ...S.extra.map((k) => ({ key: k, title: k, sub: "custom", extra: true }))];
  let n = 0;
  for (const spec of specs) {
    const tables = buildTables(spec.key, !spec.points);
    if (!tables.length) continue;
    const { card, box } = chartCard(spec, tables, spec.extra ? { onClose: () => { S.extra = S.extra.filter((k) => k !== spec.key); save(); renderCharts(); } } : {});
    grid.append(card);
    makeChart(box, spec, tables, 210);
    n++;
  }
  if (!n) grid.append(h("div", { class: "empty" }, S.selected.length ? "Loading…" : "Select one or more runs to plot their curves."));
  renderFamilies();
}

function familySeries(id) {
  const d = S.data[id] && S.data[id].d;
  if (!d) return {};
  const out = {};
  if (S.famSrc === "train") {
    for (const k of Object.keys(d.metrics)) if (k.startsWith("family_sr.")) out[k.slice(10)] = k;
  } else if (d.val) {
    const m = S.famSrc === "val_sampled" ? "sampled" : "greedy";
    for (const k of Object.keys(d.val.series)) {
      const [scope, fam, metric] = k.split("/");
      if (scope === "global" && fam !== "_macro" && metric === m) out[fam] = "@val:" + k;
    }
  }
  return out;
}

function renderFamilies() {
  const grid = $("famCharts");
  grid.replaceChildren();
  // auto-pick a source with data unless the user chose one
  if (!S.famSrcUser && S.selected.length) {
    const hasVal = S.selected.some((id) => S.data[id] && S.data[id].d && S.data[id].d.val);
    S.famSrc = hasVal ? "val_greedy" : "train";
  }
  document.querySelectorAll("#famSource button").forEach((b) => b.classList.toggle("on", b.dataset.src === S.famSrc));
  const byRun = Object.fromEntries(S.selected.map((id) => [id, familySeries(id)]));
  const fams = [...new Set(Object.values(byRun).flatMap((o) => Object.keys(o)))].sort(famSort);
  if (!fams.length) {
    grid.append(h("div", { class: "empty" }, S.famSrc === "train"
      ? "No per-family training columns (family_sr) in the selected runs."
      : "No validation log (runs/<run>_val.csv) for the selected runs. Newer training code writes one; try “Training (rolling)”."));
    return;
  }
  const isVal = S.famSrc !== "train";
  for (const fam of fams) {
    const tables = [];
    for (const id of S.selected) {
      const key = byRun[id][fam];
      if (!key) continue;
      const xy = runXY(id, key);
      if (!xy) continue;
      tables.push({ id, x: xy.x, y: xy.y, s: isVal ? xy.y : ema(xy.y, S.smooth) });
    }
    if (!tables.length) continue;
    const spec = { key: fam, title: fam, fmt: "pct", range: [0, 1], points: isVal };
    const last = tables.length === 1 ? tables[0].s[tables[0].s.length - 1] : null;
    spec.sub = last != null ? `latest ${fmtPct(last, 0)}` : "";
    const { card, box } = chartCard(spec, tables, { fam: true });
    grid.append(card);
    makeChart(box, spec, tables, 150);
  }
}

// ------------------------------------------------------------------ log tail
function renderLogPicker() {
  const sel = $("logRun");
  const ids = S.selected.filter((id) => S.byId[id] && S.byId[id].log);
  if (!ids.includes(S.logRun)) S.logRun = ids[0] || null;
  sel.replaceChildren(...ids.map((id) => h("option", { value: id, selected: id === S.logRun }, `${S.byId[id].name} (${S.byId[id].source})`)));
  if ($("logCard").open) loadLog();
}
async function loadLog() {
  const id = S.logRun;
  if (!id) { $("logText").textContent = "No log for the selected runs."; $("logName").textContent = ""; return; }
  const r = S.byId[id];
  try {
    const d = await getJSON(`${API}/runs/${encodeURIComponent(r.source)}/${encodeURIComponent(r.name)}/log?lines=80`);
    $("logName").textContent = d.name ? `${r.source}/${d.name}` : "";
    const pre = $("logText");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
    pre.textContent = d.lines.join("\n");
    if (atBottom || !pre.dataset.init) { pre.scrollTop = pre.scrollHeight; pre.dataset.init = "1"; }
  } catch (e) { $("logText").textContent = String(e.message || e); }
}

// ------------------------------------------------------------------ benchmarks
const BENCH_COLS = [
  ["solve_rate", "Solve rate", "pct", "max"], ["median_ms", "Median ms", "ms", "min"], ["p90_ms", "p90 ms", "ms", "min"],
  ["mean_expansions", "Expansions", "num", "min"], ["mean_nn_calls", "NN calls", "num", "min"],
];
function fmtMetric(v, key) {
  if (v == null || !isFinite(v)) return "–";
  if (key === "solve_rate") return Math.round(100 * v) + "%";
  if (key.endsWith("_ms")) return v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(1);
  return fmtNum(v);
}
function seqCell(td, t) {
  // t in [0,1] -> sequential blue; near zero recedes toward the surface
  const rgb = mix(P.surface, P.seqHi, 0.06 + 0.94 * Math.max(0, Math.min(1, t)));
  const c = cellColors(rgb);
  td.style.background = c.bg; td.style.color = c.fg; td.classList.add("heat");
}
function divCell(td, d, span = 0.3) {
  const t = Math.max(-1, Math.min(1, d / span));
  const rgb = mix(P.divMid, t >= 0 ? P.divPos : P.divNeg, Math.abs(t));
  const c = cellColors(rgb);
  td.style.background = c.bg; td.style.color = c.fg; td.classList.add("heat");
}
function scaleKey(el, text, stops) {
  el.replaceChildren(h("span", null, text[0]), h("i", { style: `background:linear-gradient(90deg,${stops.join(",")})` }), h("span", null, text[1]));
}
function currentReport() {
  const id = $("benchReport").value;
  return S.bench.find((r) => r.id === id) || S.bench[0];
}
function methodLabel(k) { return k; }

function heatScale(values, key) {
  // map value -> [0,1]; solve rate absolute, others log-normalised within the table
  if (key === "solve_rate") return (v) => v;
  const vs = values.filter((v) => v != null && v > 0);
  if (!vs.length) return () => 0;
  const lo = Math.log10(Math.min(...vs)), hi = Math.log10(Math.max(...vs));
  return (v) => (v == null || v <= 0 || hi === lo ? 0 : (Math.log10(v) - lo) / (hi - lo));
}

function benchMatrix(rep, part, key) {
  const methods = Object.keys(rep.summary);
  const rows = [...new Set(methods.flatMap((m) => Object.keys(rep.summary[m][part] || {})))];
  if (part === "by_family") rows.sort(famSort);
  const all = [];
  for (const r of rows) for (const m of methods) all.push((rep.summary[m][part][r] || {})[key]);
  const scale = heatScale(all, key);
  const better = key === "solve_rate" ? "max" : "min";
  const table = h("table", { class: "dt" });
  table.append(h("thead", null, h("tr", null, h("th", { class: "l" }, part === "by_family" ? "Family" : "Density"),
    ...methods.map((m) => h("th", { class: "wrap", title: m }, methodLabel(m))))));
  const tb = h("tbody");
  const addRow = (label, getter, cls) => {
    const vals = methods.map(getter);
    const valid = vals.filter((v) => v != null);
    const best = valid.length ? (better === "max" ? Math.max(...valid) : Math.min(...valid)) : null;
    const tr = h("tr", { class: cls || null }, h("td", null, label));
    methods.forEach((m, i) => {
      const cell = rep.summary[m];
      const src = label === "Overall" ? cell.overall : (cell[part][label] || {});
      const v = vals[i];
      let txt = fmtMetric(v, key);
      if (key === "solve_rate" && src.solve_rate_std != null) txt += ` ±${Math.round(100 * src.solve_rate_std)}`;
      const td = h("td", { title: `${m} · ${label}: ${fmtMetric(v, key)}${src.count ? ` (n=${src.count})` : ""}` }, txt);
      if (v != null) seqCell(td, scale(v));
      if (v != null && v === best && valid.length > 1) td.classList.add("best");
      tr.append(td);
    });
    tb.append(tr);
  };
  for (const r of rows) addRow(r, (m) => (rep.summary[m][part][r] || {})[key]);
  addRow("Overall", (m) => (rep.summary[m].overall || {})[key], "total");
  table.append(tb);
  return table;
}

function renderBench() {
  if (!P) P = palette();
  const sel = $("benchReport");
  const prev = sel.value;
  sel.replaceChildren(...S.bench.map((r) => h("option", { value: r.id },
    `${r.name}${r.meta && r.meta.set ? " · " + r.meta.set : ""}${r.source !== "local" ? " · " + r.source : ""}`)));
  if (prev && S.bench.some((r) => r.id === prev)) sel.value = prev;
  const rep = currentReport();
  for (const id of ["benchOverall", "benchFamily", "benchDensity"]) $(id).replaceChildren();
  if (!rep || rep.error || !Object.keys(rep.summary || {}).length) {
    $("benchMeta").textContent = "";
    $("benchOverall").append(h("div", { class: "empty" }, rep && rep.error ? `Could not read ${rep.name}: ${rep.error}` :
      "No benchmark reports yet (runs/bench/*.json from python -m zipsolve.rl.benchmark run)."));
    renderCompare();
    return;
  }
  const m = rep.meta || {};
  const bits = [m.num_puzzles && `${m.num_puzzles} puzzles`, m.budget != null && `${m.budget}s budget`, m.k && `K=${m.k}`,
    m.date, m.fingerprint && `set ${m.fingerprint}`].filter(Boolean);
  $("benchMeta").textContent = bits.join(" · ");
  // overall table: methods x metrics
  const methods = Object.keys(rep.summary);
  const t = h("table", { class: "dt" });
  t.append(h("thead", null, h("tr", null, h("th", { class: "l" }, "Method"), h("th", null, "n"), ...BENCH_COLS.map((c) => h("th", null, c[1])))));
  const tb = h("tbody");
  const best = {};
  for (const [k, , , dir] of BENCH_COLS) {
    const vs = methods.map((mm) => rep.summary[mm].overall[k]).filter((v) => v != null);
    best[k] = vs.length ? (dir === "max" ? Math.max(...vs) : Math.min(...vs)) : null;
  }
  for (const mm of methods) {
    const o = rep.summary[mm].overall || {};
    const inst = rep.summary[mm].instances;
    const tr = h("tr", null, h("td", { title: (rep.summary[mm].ckpt_paths || []).join("\n") }, mm,
      inst > 1 ? h("span", { class: "sub" }, ` ×${inst}`) : null), h("td", null, fmtNum(o.count)));
    for (const [k] of BENCH_COLS) {
      let txt = fmtMetric(o[k], k);
      if (k === "solve_rate" && o.solve_rate_std != null) txt += ` ±${Math.round(100 * o.solve_rate_std)}`;
      const td = h("td", null, txt);
      if (k === "solve_rate" && o[k] != null) seqCell(td, o[k]);
      if (o[k] != null && o[k] === best[k] && methods.length > 1) td.classList.add("best");
      tr.append(td);
    }
    tb.append(tr);
  }
  t.append(tb);
  $("benchOverall").append(t);
  const key = $("benchMetric").value;
  $("benchFamily").append(benchMatrix(rep, "by_family", key));
  const hasDensity = methods.some((mm) => Object.keys(rep.summary[mm].by_density || {}).length);
  if (hasDensity) $("benchDensity").append(benchMatrix(rep, "by_density", key));
  else $("benchDensity").append(h("div", { class: "muted" }, "No density breakdown in this report."));
  const lo = mix(P.surface, P.seqHi, 0.06), hi = mix(P.surface, P.seqHi, 1);
  scaleKey($("benchKey"), key === "solve_rate" ? ["0%", "100%"] : ["lower", "higher (log)"], [`rgb(${lo})`, `rgb(${hi})`]);
  renderCompare();
}

function compareOptions() {
  const opts = [];
  for (const r of S.bench) for (const k of Object.keys(r.summary || {})) opts.push({ value: `${r.id}|${k}`, label: `${k} — ${r.name}${r.source !== "local" ? " (" + r.source + ")" : ""}`, rep: r, key: k });
  return opts;
}
function renderCompare() {
  const opts = compareOptions();
  const A = $("cmpA"), B = $("cmpB");
  const pa = A.value, pb = B.value;
  for (const sel of [A, B]) sel.replaceChildren(...opts.map((o) => h("option", { value: o.value }, o.label)));
  const has = (v) => opts.some((o) => o.value === v);
  if (has(pa)) A.value = pa; else {
    const newest = S.bench[0];
    const cands = opts.filter((o) => o.rep === newest);
    const model = cands.filter((o) => !o.key.startsWith("baseline/"));
    A.value = (cands.find((o) => o.key === "baseline/solver") || cands[0] || {}).value || "";
    if (!has(pb) && model.length) B.value = (model.find((o) => o.key.endsWith("/search")) || model[0]).value;
  }
  if (has(pb)) B.value = pb;
  const box = $("cmpTable");
  box.replaceChildren();
  const a = opts.find((o) => o.value === A.value), b = opts.find((o) => o.value === B.value);
  if (!a || !b) { box.append(h("div", { class: "empty" }, "Need at least one benchmark report to compare.")); return; }
  const sa = a.rep.summary[a.key], sb = b.rep.summary[b.key];
  const fams = [...new Set([...Object.keys(sa.by_family || {}), ...Object.keys(sb.by_family || {})])].sort(famSort);
  const t = h("table", { class: "dt" });
  t.append(h("thead", null, h("tr", null, h("th", { class: "l" }, "Family"), h("th", null, "A solve"), h("th", null, "B solve"),
    h("th", null, "Δ (B − A)"), h("th", null, "A median ms"), h("th", null, "B median ms"), h("th", null, "Latency ratio B/A"))));
  const tb = h("tbody");
  const row = (label, x, y, cls) => {
    x = x || {}; y = y || {};
    const d = x.solve_rate != null && y.solve_rate != null ? y.solve_rate - x.solve_rate : null;
    const tdA = h("td", null, fmtMetric(x.solve_rate, "solve_rate"));
    const tdB = h("td", null, fmtMetric(y.solve_rate, "solve_rate"));
    if (x.solve_rate != null) seqCell(tdA, x.solve_rate);
    if (y.solve_rate != null) seqCell(tdB, y.solve_rate);
    const tdD = h("td", null, d == null ? "–" : (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(Math.round(100 * d)) + " pp");
    if (d != null) divCell(tdD, d);
    const ratio = x.median_ms && y.median_ms ? y.median_ms / x.median_ms : null;
    tb.append(h("tr", { class: cls || null }, h("td", null, label), tdA, tdB, tdD, h("td", null, fmtMetric(x.median_ms, "median_ms")),
      h("td", null, fmtMetric(y.median_ms, "median_ms")), h("td", null, ratio == null ? "–" : ratio.toFixed(ratio < 10 ? 2 : 0) + "×")));
  };
  for (const f of fams) row(f, sa.by_family[f], sb.by_family[f]);
  row("Overall", sa.overall, sb.overall, "total");
  t.append(tb);
  box.append(t);
  scaleKey($("cmpKey"), ["A better", "B better"], [`rgb(${mix(P.divMid, P.divNeg, 1)})`, P.divMid, `rgb(${mix(P.divMid, P.divPos, 1)})`]);
}

// ------------------------------------------------------------------ eval
function renderEval() {
  const box = $("evalTable");
  box.replaceChildren();
  const d = S.eval;
  if (!d || !d.rows.length) { box.append(h("div", { class: "empty" }, "No evaluation results (runs/eval/*.json).")); return; }
  const cols = d.columns;
  const rate = cols.filter((c) => c.endsWith("solve_rate"));
  const lead = ["episodes", "avg_nodes"].filter((c) => cols.includes(c));
  const rest = cols.filter((c) => !rate.includes(c) && !lead.includes(c));
  const order = [...lead, ...rate, ...rest];
  const rows = d.rows.slice().sort((a, b) => famSort(a.spec, b.spec) || (a.source < b.source ? -1 : 1));
  const multiSrc = new Set(rows.map((r) => r.source)).size > 1;
  const t = h("table", { class: "dt" });
  t.append(h("thead", null, h("tr", null, h("th", { class: "l" }, "Spec"), multiSrc ? h("th", { class: "l" }, "Source") : null,
    ...order.map((c) => h("th", { class: "wrap", title: c }, c.replace(/_/g, " "))))));
  const tb = h("tbody");
  for (const r of rows) {
    const tr = h("tr", null, h("td", null, r.spec), multiSrc ? h("td", { class: "l" }, r.source) : null);
    for (const c of order) {
      const v = r[c];
      const txt = typeof v === "number" ? (rate.includes(c) ? fmtPct(v, 0) : c.includes("seconds") ? (v < 1 ? (v * 1000).toFixed(1) + " ms" : v.toFixed(2) + " s") : fmtNum(v)) : (v ?? "–");
      const td = h("td", null, txt);
      if (rate.includes(c) && typeof v === "number") seqCell(td, v);
      tr.append(td);
    }
    tb.append(tr);
  }
  t.append(tb);
  box.append(t);
}

// ------------------------------------------------------------------ checkpoints
function renderCkpts() {
  const box = $("ckptTable");
  box.replaceChildren();
  const q = $("ckptFilter").value.trim().toLowerCase();
  const onlySel = $("ckptSelected").checked;
  const selRuns = new Set(S.selected.map((id) => S.byId[id] && `${S.byId[id].source}|${S.byId[id].name}`));
  const items = S.ckpts.filter((c) => {
    if (onlySel && !selRuns.has(`${c.source}|${c.run}`)) return false;
    if (!q) return true;
    const m = c.meta || {};
    return [c.name, c.run, c.kind, c.source, m.stage].some((s) => s && String(s).toLowerCase().includes(q));
  });
  if (!items.length) { box.append(h("div", { class: "empty" }, S.ckpts.length ? "No checkpoints match the filter." : "No checkpoints found.")); return; }
  const t = h("table", { class: "dt" });
  t.append(h("thead", null, h("tr", null, ...["Checkpoint", "Source", "Kind", "Stage", "Iter", "Steps", "Model", "Val score", "Size", "Modified", ""].map((c, i) =>
    h("th", { class: [0, 1, 2, 3, 6].includes(i) ? "l" : null }, c)))));
  const tb = h("tbody");
  const now = Date.now() / 1000;
  for (const c of items) {
    const m = c.meta;
    const pending = !m && !c.meta_error;
    const cell = (v) => (pending ? h("span", { class: "sub loading-dots" }, "") : v);
    const model = m && m.hidden ? `${m.hidden}×${m.layers}${m.params ? " · " + fmtNum(m.params) : ""}` : "–";
    const open = S.metaOpen.has(c.name);
    const tr = h("tr", null,
      h("td", { title: c.name }, c.file),
      h("td", { class: "l" }, h("span", { class: "badge" }, c.source)),
      h("td", { class: "l" }, c.kind),
      h("td", { class: "l", title: m && m.stage || "" }, cell(m && m.stage ? m.stage : "–")),
      h("td", null, cell(fmtNum(m && m.iteration))),
      h("td", null, cell(fmtNum(m && m.global_step))),
      h("td", { class: "l" }, cell(model)),
      h("td", null, cell(m && m.val_score != null ? fmtPct(m.val_score, 0) : "–")),
      h("td", null, fmtBytes(c.size)),
      h("td", { title: new Date(c.mtime * 1000).toLocaleString() }, fmtAgo(now - c.mtime)),
      h("td", null, h("a", { class: "play-link", href: `/lab?model=${encodeURIComponent(c.name)}`, title: "Open the AI Lab with this model" }, "Play ▸"), " ",
        h("button", { class: "linkbtn", onclick: () => toggleMeta(c.name), "aria-expanded": open ? "true" : "false" }, open ? "Hide" : "Details")));
    tb.append(tr);
    if (c.meta_error) tr.title = c.meta_error;
    if (open) {
      const pre = h("pre", null, "Loading…");
      tb.append(h("tr", { class: "meta-row" }, h("td", { colspan: 11 }, pre)));
      getJSON(`${API}/checkpoints/meta?full=true&name=${encodeURIComponent(c.name)}`)
        .then((d) => { pre.textContent = JSON.stringify(d.meta, null, 1); })
        .catch((e) => { pre.textContent = String(e.message || e); });
    }
  }
  t.append(tb);
  box.append(t);
  queueMeta(items.filter((c) => !c.meta && !c.meta_error).map((c) => c.name));
}
function toggleMeta(name) { S.metaOpen.has(name) ? S.metaOpen.delete(name) : S.metaOpen.add(name); renderCkpts(); }
function queueMeta(names) {
  names.forEach((n) => S.metaQueue.add(n));
  if (!S.metaBusy) pumpMeta();
}
async function pumpMeta() {
  S.metaBusy = true;
  const rerender = debounce(renderCkpts, 250);
  while (S.metaQueue.size) {
    const name = S.metaQueue.values().next().value;
    S.metaQueue.delete(name);
    const c = S.ckpts.find((x) => x.name === name);
    if (!c || c.meta) continue;
    try {
      const d = await getJSON(`${API}/checkpoints/meta?name=${encodeURIComponent(name)}`);
      c.meta = d.summary || {};
    } catch (e) { c.meta_error = String(e.message || e); }
    rerender();
  }
  S.metaBusy = false;
}

// ------------------------------------------------------------------ sync
function setSync(st) {
  S.sync = st;
  const btn = $("syncBtn"), lab = $("syncLabel"), out = $("syncStatus");
  if (!st) return;
  lab.textContent = `Sync ${st.host}`;
  btn.classList.toggle("running", !!st.running);
  btn.disabled = !!st.running;
  if ($("autoSync").dataset.user !== "1") $("autoSync").value = String(st.auto_minutes || 0);
  out.replaceChildren();
  const now = Date.now() / 1000;
  if (st.running) { out.append(h("span", null, `Pulling runs and checkpoints from ${st.host}…`)); return; }
  const last = st.last;
  if (!last) { out.append(h("span", null, `${st.host} not synced yet`)); return; }
  if (last.ok) {
    const n = (last.changed_runs || []).length + (last.changed_checkpoints || []).length;
    out.append(h("span", null, `${st.host} synced `, h("b", null, fmtAgo(now - (last.finished || last.last_ok)))),
      h("span", { class: "muted" }, ` · ${last.method || "?"} · ${n} file${n === 1 ? "" : "s"} updated`));
  } else {
    out.append(h("span", { class: "err" }, "⚠ Sync failed: ", last.error || "unknown error"));
    if (last.last_ok) out.append(h("span", { class: "muted" }, ` · last good ${fmtAgo(now - last.last_ok)}`));
  }
}
async function startSync() {
  try {
    const st = await getJSON(`${API}/sync`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ force: true }) });
    setSync(st);
    pollSync();
  } catch (e) { showErr(e); }
}
function pollSync() {
  clearTimeout(S.syncPoll);
  S.syncPoll = setTimeout(async () => {
    try {
      const st = await getJSON(`${API}/sync`);
      setSync(st);
      if (st.running) pollSync(); else refreshAll(true);
    } catch (e) { showErr(e); }
  }, 1000);
}

// ------------------------------------------------------------------ refresh
function updateRefreshStatus() {
  $("refreshStatus").textContent = S.lastRefresh ? `Updated ${new Date(S.lastRefresh).toLocaleTimeString()}` : "";
}
async function refreshAll(forceRender) {
  try {
    await loadRuns();
    renderCards();
    const changed = await loadSeries(false);
    if (changed || forceRender || !S.charts.length) renderCharts();
    const [bench, ev, ck] = await Promise.all([getJSON(`${API}/bench`), getJSON(`${API}/eval`), getJSON(`${API}/checkpoints`)]);
    const bsig = JSON.stringify(bench.reports.map((r) => [r.id, r.mtime]));
    if (bsig !== S.benchSig) { S.benchSig = bsig; S.bench = bench.reports; renderBench(); }
    const esig = JSON.stringify(ev.rows.map((r) => [r.source, r.file, r.mtime]));
    if (esig !== S.evalSig) { S.evalSig = esig; S.eval = ev; renderEval(); }
    const csig = JSON.stringify(ck.checkpoints.map((c) => [c.name, c.mtime, c.size]));
    if (csig !== S.ckptSig) {
      const old = Object.fromEntries(S.ckpts.map((c) => [c.name, c]));
      for (const c of ck.checkpoints) {
        const o = old[c.name];
        if (!c.meta && o && o.meta && o.mtime === c.mtime && o.size === c.size) c.meta = o.meta;
      }
      S.ckptSig = csig; S.ckpts = ck.checkpoints; renderCkpts();
    }
    renderLogPicker();
    S.lastRefresh = Date.now();
    updateRefreshStatus();
  } catch (e) { showErr(e); }
}
function scheduleRefresh() {
  clearInterval(S.refreshTimer);
  if (S.refreshSec > 0) S.refreshTimer = setInterval(() => { if (!document.hidden) refreshAll(false); }, S.refreshSec * 1000);
}

// ------------------------------------------------------------------ wiring
function wire() {
  $("themeBtn").onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(dark ? "light" : "dark");
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => rethemeAll());
  document.querySelectorAll("#xAxis button").forEach((b) => {
    b.classList.toggle("on", b.dataset.x === S.x);
    b.onclick = () => {
      S.x = b.dataset.x; S.xRange = null; save();
      document.querySelectorAll("#xAxis button").forEach((o) => o.classList.toggle("on", o === b));
      renderCharts();
    };
  });
  const sm = $("smooth");
  sm.value = S.smooth; $("smoothVal").textContent = Number(S.smooth).toFixed(2);
  const rer = debounce(() => renderCharts(), 80);
  sm.oninput = () => { S.smooth = parseFloat(sm.value); $("smoothVal").textContent = S.smooth.toFixed(2); save(); rer(); };
  $("showRaw").checked = S.raw;
  $("showRaw").onchange = (e) => { S.raw = e.target.checked; save(); renderCharts(); };
  $("showStages").checked = S.stages;
  $("showStages").onchange = (e) => { S.stages = e.target.checked; save(); renderCharts(); };
  $("moreMetric").onchange = (e) => {
    const k = e.target.value;
    if (k && !S.extra.includes(k)) { S.extra.push(k); save(); renderCharts(); }
  };
  document.querySelectorAll("#famSource button").forEach((b) => {
    b.onclick = () => { S.famSrc = b.dataset.src; S.famSrcUser = true; save(); destroyCharts(); renderCharts(); };
  });
  $("autoRefresh").value = String(S.refreshSec);
  $("autoRefresh").onchange = (e) => { S.refreshSec = parseInt(e.target.value, 10) || 0; save(); scheduleRefresh(); };
  $("autoSync").onchange = async (e) => {
    e.target.dataset.user = "1";
    try {
      const st = await getJSON(`${API}/sync/auto`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ minutes: parseFloat(e.target.value) || 0 }) });
      e.target.dataset.user = "";
      setSync(st);
    } catch (err) { showErr(err); }
  };
  $("syncBtn").onclick = startSync;
  $("benchReport").onchange = renderBench;
  $("benchMetric").onchange = renderBench;
  $("cmpA").onchange = renderCompare;
  $("cmpB").onchange = renderCompare;
  $("ckptFilter").oninput = debounce(renderCkpts, 150);
  $("ckptSelected").onchange = renderCkpts;
  $("logCard").addEventListener("toggle", () => { if ($("logCard").open) loadLog(); });
  $("logRun").onchange = (e) => { S.logRun = e.target.value; $("logText").dataset.init = ""; loadLog(); };
  document.addEventListener("visibilitychange", () => { if (!document.hidden && S.refreshSec > 0) refreshAll(false); });
  // relative times ("2 min ago") tick without refetching
  setInterval(() => { if (S.sync) setSync(S.sync); }, 30000);
}

async function main() {
  load();
  P = palette();
  wire();
  if (typeof uPlot === "undefined") {
    $("charts").replaceChildren(h("div", { class: "empty" }, "Chart library (uPlot, cdn.jsdelivr.net) failed to load; tables still work."));
  }
  await refreshAll(true);
  if (S.sync && S.sync.running) pollSync();
  scheduleRefresh();
}
main();
