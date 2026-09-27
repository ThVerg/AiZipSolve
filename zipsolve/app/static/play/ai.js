// AI showcase for the play page: live policy "vision" overlay, solver/agent runs with a replay of the
// search ("watch it think"), You-vs-AI race, model comparison, and the last-run stats panel.
import { h, api, fmtTime, fmtSec, fmtInt, fmtPct, cssVar, icon, esc, svgEl as el, clamp, reducedMotion, store, sleep } from "./util.js";
import { Board } from "./board.js";

const MODE_LABEL = { greedy: "Greedy", search: "Policy DFS", hybrid: "Solver+GNN", exact: "Exact solver" };
const MODE_NOTE = {
  greedy: "The policy alone, no backtracking: shows what the network has learned. It can get stuck.",
  search: "Depth-first search ordered by the policy, basic pruning only, up to a node budget.",
  hybrid: "The exact solver's pruning with the GNN ordering its moves: complete (\"unsat\" is a proof). Compared with the plain solver.",
  exact: "The exact solver alone: degree / connectivity / parity pruning, restarts.",
};
const OPPONENTS = [
  { id: "rookie", label: "Rookie", mode: "greedy", pace: 800, desc: "Greedy policy · no take-backs · can get stuck" },
  { id: "challenger", label: "Challenger", mode: "search", pace: 480, desc: "Policy DFS · backtracks when it hits a wall" },
  { id: "grandmaster", label: "Grandmaster", mode: "hybrid", pace: 300, desc: "Solver + GNN · never fails" },
];

export function initAI(ctx) {
  const { S, $, sound } = ctx;
  const A = {
    vision: store.get("zip-vision", false), heat: null, policy: null, info: null, infoRev: -1, ctrl: null, timer: null,
    models: [], mode: "greedy", cmpMode: "greedy", race: null, opp: store.get("zip-opp", "rookie"), mini: null, cmpBoards: [],
  };
  const board = () => ctx.board();

  // ================================================================ models
  function hasModels() { return A.models.length > 0; }
  function selModel() { return $("model").value || null; }
  let wantModel = new URLSearchParams(location.search).get("model");   // /?model=<name> (dashboard links)
  function modelLabel(m) {
    const age = Math.round((Date.now() / 1000 - m.mtime) / 60);
    const file = (m.file || m.name).replace(/\.pt$/, "");
    return `${file}${m.stage ? " · " + m.stage : ""} (${age < 60 ? age + " min" : age < 2880 ? Math.round(age / 60) + " h" : Math.round(age / 1440) + " d"} ago)`;
  }
  async function loadModels() {
    const sel = $("model");
    let prev = sel.value;
    try {
      const res = await api("/api/models");
      A.models = res.models || [];
      A.dir = res.directory;
      sel.innerHTML = "";
      if (!A.models.length) sel.appendChild(h("option", { disabled: true, text: "No checkpoints found" }));
      if (wantModel) {
        if (A.models.some((m) => m.name === wantModel)) { prev = wantModel; ctx.toast(`Model ${wantModel} selected`); if (window.__zipApi) window.__zipApi.selectTab("ai"); }
        else ctx.toast(`Model ${wantModel} not found`);
        wantModel = null;
      }
      // local checkpoints first, then one <optgroup> per subfolder (e.g. remote/<host>)
      const groups = new Map();
      for (const m of A.models) { const g = m.group || ""; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(m); }
      const keys = [...groups.keys()].sort((a, b) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)));
      for (const g of keys) {
        const parent = g === "" ? sel : sel.appendChild(h("optgroup", { label: g }));
        for (const m of groups.get(g)) {
          const o = h("option", { value: m.name, text: modelLabel(m), title: m.name });
          if ((prev && prev === m.name) || (!prev && m.default)) o.selected = true;
          parent.appendChild(o);
        }
      }
      renderModelInfo(); renderCmpModels(); syncButtons();
    } catch (e) { ctx.banner(`Could not list models: ${esc(e.message)}`, "bad"); }
  }
  function renderModelInfo(extra) {
    const m = A.models.find((x) => x.name === selModel());
    const box = $("modelInfo");
    if (!m) { box.innerHTML = hasModels() ? "" : `<span class="muted">Train a model first: no <code>*.pt</code> in ${esc(A.dir || "checkpoints/")}. The exact solver still works.</span>`; return; }
    const kb = m.size ? `${Math.round(m.size / 1024)} KB` : "";
    box.innerHTML = `<span class="pill">${icon("brain")} loaded: <b>${esc(m.name)}</b></span>` +
      `<span class="muted">${[m.stage && `stage ${esc(m.stage)}`, m.iteration != null && `iter ${m.iteration}`, kb, m.default && "default"].filter(Boolean).join(" · ")}</span>` + (extra || "");
  }

  // ================================================================ vision (policy heatmap)
  function overlay() {
    if (!A.vision || S.anim || racing() || S.won) return {};
    return { heat: $("heatFull").checked ? A.heat : null, policy: A.infoRev === S.rev ? A.policy : null };
  }
  function setVision(on) {
    A.vision = !!on;
    store.set("zip-vision", A.vision);
    $("visionToggle").checked = A.vision;
    $("visionBtn").classList.toggle("on", A.vision);
    $("visionBtn").setAttribute("aria-pressed", String(A.vision));
    if (!A.vision) { A.heat = null; A.policy = null; A.info = null; $("insight").hidden = true; ctx.layoutBoard(); ctx.render(); }
    else { renderInsight(true); ctx.layoutBoard(); ctx.render(); schedule(0); }
  }
  function toggleVision() {
    if (racing()) { ctx.toast("AI vision is off during a race"); return; }
    if (!hasModels()) { ctx.toast("No trained model found in checkpoints/"); return; }
    setVision(!A.vision);
  }
  function schedule(delay = 110) {
    clearTimeout(A.timer);
    if (!A.vision || !S.data || racing()) return;
    A.timer = setTimeout(fetchPolicy, delay);
  }
  async function fetchPolicy() {
    if (!A.vision || !S.data || S.anim || racing()) return;
    if (S.path.length >= S.n) { A.policy = []; A.heat = null; A.infoRev = S.rev; A.info = { done: true }; renderInsight(); ctx.render(); return; }
    if (A.ctrl) A.ctrl.abort();
    const ctrl = A.ctrl = new AbortController();
    const tok = ctx.pathToken();
    renderInsight(true);
    try {
      const res = await api("/api/policy", { puzzle: S.data, path: S.path, model: selModel(), full: $("heatFull").checked, check: $("heatCheck").checked }, { signal: ctrl.signal });
      if (!ctx.tokenValid(tok) || !A.vision) return;
      A.info = res; A.policy = res.legal; A.heat = res.heat; A.infoRev = S.rev;
      renderInsight();
      ctx.render();
    } catch (e) {
      if (e.name === "AbortError") return;
      if (ctx.tokenValid(tok)) { A.info = { error: e.message }; renderInsight(); }
    }
  }
  function renderInsight(loading = false) {
    const box = $("insight");
    if (!A.vision || racing()) { box.hidden = true; return; }
    box.hidden = false;
    const r = A.info;
    if (!r || (loading && A.infoRev !== S.rev && !r.legal)) { box.innerHTML = `<div class="ins-row"><span class="spinner sm"></span><span class="muted">The network is looking at your position…</span></div>`; return; }
    if (r.error) { box.innerHTML = `<div class="ins-row"><span class="bad-txt">AI vision failed: ${esc(r.error)}</span></div>`; return; }
    if (r.done) { box.innerHTML = `<div class="ins-row muted">${icon("check")} Board complete.</div>`; return; }
    const stale = A.infoRev !== S.rev;
    const w = r.winnable;
    const pct = w == null ? null : Math.round(w * 100);
    const tone = w == null ? "" : w >= 0.66 ? "good" : w >= 0.33 ? "warn" : "bad";
    const top = r.legal && r.legal[0];
    const hd = S.path[S.path.length - 1];
    const dir = top ? board().dirLabel(hd, top[0]) : null;
    const verdict = { solved: [`${icon("check")} still solvable`, "good"], unsat: [`${icon("x")} dead end: undo!`, "bad"], timeout: ["? solver unsure (time)", "warn"] }[r.completable];
    const g = r.greedy;
    const greedy = g ? (g.status === "solved" ? [`greedy from here finishes (${g.moves} moves)`, "good"] : g.status === "timeout" ? ["greedy look-ahead timed out", "warn"] : [`greedy from here gets stuck after ${g.moves} moves`, "bad"]) : null;
    box.classList.toggle("stale", stale);
    box.innerHTML = `
      <div class="ins-head">${icon("brain")}<b>AI vision</b><span class="muted">${esc((r.model || "").replace(/\.pt$/, ""))}</span>${stale ? '<span class="spinner sm"></span>' : ""}</div>
      <div class="ins-grid">
        <div class="meter-block">
          ${pct == null
            ? `<div class="meter-label">Value head untrained <span class="muted">(imitation model: no win estimate)</span></div><div class="meter"><i style="width:0"></i></div>`
            : `<div class="meter-label">Agent thinks this position is <b class="${tone}-txt">${pct}% winnable</b></div><div class="meter ${tone}"><i style="width:${pct}%"></i></div>`}
          <div class="muted tiny">value ${r.value == null ? "-" : r.value.toFixed(3)} · policy entropy ${r.entropy == null ? "-" : r.entropy.toFixed(2)} nats · ${r.legal.length} legal</div>
        </div>
        <div class="ins-chips">
          ${top ? `<span class="ichip">Top move <b>${esc(dir)}</b> ${fmtPct(top[1])}</span>` : `<span class="ichip bad">no legal move</span>`}
          ${verdict ? `<span class="ichip ${verdict[1]}">Solver: ${verdict[0]}</span>` : ""}
          ${greedy ? `<span class="ichip ${greedy[1]}">${greedy[0]}</span>` : ""}
          ${top && !S.won ? `<button class="btn sm" type="button" id="aiMoveBtn" title="Play the network's top move (counts as assisted)">${icon("bolt")} Play it</button>` : ""}
        </div>
      </div>`;
    const b = $("aiMoveBtn");
    if (b) b.onclick = () => {
      if (A.infoRev !== S.rev || !top) return;
      S.assisted = true;
      window.__zipApi.push(top[0]);
    };
  }

  // ================================================================ last-run stats (comparison table + confidence)
  function updateCompare() {
    const a = S.agent;
    const hy = a && a.mode === "hybrid";
    const e = hy && a.exact ? a.exact : S.exact;
    const res = (r) => r ? `<span class="${r.solved ? "ok" : "fail"}">${r.solved ? "solved" : esc(r.status)}</span>` : "-";
    $("aHead").textContent = hy ? "Solver+GNN" : "Agent";
    $("eHead").textContent = hy ? "Solver" : "Exact";
    $("aRes").innerHTML = a ? res(a) + (hy ? "" : ` <span class="muted">${MODE_LABEL[a.mode] || a.mode}</span>`) : "-";
    $("eRes").innerHTML = res(e);
    $("aLen").textContent = a ? `${a.path.length}/${S.n}` : "-";
    $("eLen").textContent = e ? (e.path ? `${e.path.length}/${S.n}` : "-") : "-";
    $("aNodes").textContent = a ? fmtInt(a.nodes_expanded) : "-";
    $("eNodes").textContent = e ? fmtInt(e.nodes_expanded) : "-";
    $("aTime").textContent = a ? fmtSec(a.seconds) : "-";
    $("eTime").textContent = e ? fmtSec(e.seconds) : "-";
    $("aConf").textContent = a && a.mean_confidence != null ? `mean ${(a.mean_confidence * 100).toFixed(1)}% · min ${(a.min_confidence * 100).toFixed(1)}% (${a.model})` : "-";
    document.querySelectorAll("#cmpTable tr.hy").forEach((r) => { r.hidden = !hy; });
    $("hyNote").hidden = !hy;
    if (hy) {
      const hs = a.hybrid;
      $("aInf").textContent = fmtInt(hs.inference_calls);
      $("aCache").textContent = fmtInt(hs.cache_hits);
      $("aSplit").textContent = `${fmtSec(hs.inference_seconds + hs.obs_seconds)} / ${fmtSec(hs.search_seconds)}`;
      $("eSplit").textContent = e ? `0 / ${fmtSec(e.seconds)}` : "-";
      $("aAtt").textContent = fmtInt(hs.attempts);
    }
  }
  function updateConfPanel() {
    const a = S.agent, box = $("confBlock");
    if (!a || !a.steps || !a.steps.length) { box.hidden = true; return; }
    box.hidden = false;
    const st = S.curStep, list = $("candList");
    list.innerHTML = "";
    if (st) {
      const idx = a.steps.indexOf(st);
      $("confStep").textContent = `${a.start_len + idx}`;
      $("confLegal").textContent = `${st.n_legal} legal`;
      const from = S.path[S.path.indexOf(st.node) - 1];
      for (const [v, p] of st.top) {
        list.appendChild(h("div", { class: "cand", html: `<span>${from !== undefined ? esc(board().dirLabel(from, v)) : v}</span><span class="bar"><i class="${v === st.node ? "chosen" : ""}" style="width:${(p * 100).toFixed(1)}%"></i></span><span class="p">${(p * 100).toFixed(1)}%</span>` }));
      }
    }
    drawSpark();
  }
  function drawSpark() {
    const a = S.agent, sp = $("spark");
    sp.innerHTML = "";
    if (!a || !a.steps || !a.steps.length) return;
    const W = sp.clientWidth || 260, H = 56, pad = 4;
    sp.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const shown = S.curStep ? a.steps.indexOf(S.curStep) + 1 : a.steps.length;
    const N = a.steps.length;
    const X = (i) => pad + (N > 1 ? (i * (W - 2 * pad)) / (N - 1) : (W - 2 * pad) / 2);
    const Y = (p) => H - pad - p * (H - 2 * pad);
    for (const g of [0.5, 1]) el("line", { x1: pad, x2: W - pad, y1: Y(g), y2: Y(g), stroke: cssVar("--border"), "stroke-width": 1, "stroke-dasharray": g === 0.5 ? "3 3" : null }, sp);
    const pts = a.steps.slice(0, shown).map((s, i) => `${X(i).toFixed(1)},${Y(s.p).toFixed(1)}`).join(" ");
    el("polyline", { points: pts, fill: "none", stroke: cssVar("--accent"), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, sp);
    a.steps.slice(0, shown).forEach((s, i) => { if (s.p < 0.5) el("circle", { cx: X(i), cy: Y(s.p), r: 2.5, fill: cssVar("--warn") }, sp); });
    const hover = el("line", { y1: pad, y2: H - pad, stroke: cssVar("--ink-3"), "stroke-width": 1, opacity: 0 }, sp);
    const tip = $("sparkTip");
    sp.onpointermove = (ev) => {
      if (shown <= 0) return;
      const r = sp.getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * W;
      const i = clamp(Math.round(((x - pad) / (W - 2 * pad)) * (N - 1)), 0, shown - 1);
      hover.setAttribute("x1", X(i)); hover.setAttribute("x2", X(i)); hover.setAttribute("opacity", 1);
      tip.style.display = "block";
      tip.style.left = `${(X(i) / W) * r.width}px`; tip.style.top = `${(Y(a.steps[i].p) / H) * r.height}px`;
      tip.textContent = `step ${a.start_len + i}: ${(a.steps[i].p * 100).toFixed(1)}%  (${a.steps[i].n_legal} legal)`;
    };
    sp.onpointerleave = () => { hover.setAttribute("opacity", 0); tip.style.display = "none"; };
  }

  // ================================================================ watch it think: replay a search trace
  function traceRate() { const v = Number($("speed").value) / 100; return 4 * Math.pow(10, v * 3.1); }   // events / s
  function replayTrace(res, owner, startLen, onDone) {
    ctx.cancelAnim();
    const trace = res.trace || [];
    const final = res.solved ? res.path : null;
    const N = S.n;
    S.won = false; S.tEnd = null; S.hint = null; S.stuck = null; S.conf = new Map(); S.curStep = null;
    S.assisted = true;
    S.tried = new Float32Array(N);
    let i = 0, stack = [], pushes = 0, pops = 0, attempts = 0, maxDepth = 0, acc = 0, last = performance.now(), raf = 0, stopped = false;
    let sndT = 0;
    const hud = $("traceHud");
    hud.hidden = false;
    const label = MODE_LABEL[res.mode || (owner === "exact" ? "exact" : "search")] || "Search";
    const paint = () => {
      const pct = trace.length ? Math.round((100 * i) / trace.length) : 100;
      hud.innerHTML = `<span class="th-title">${icon("cpu")} ${label} thinking</span>` +
        `<span class="th-stat"><b class="mono">${fmtInt(pushes)}</b> moves tried</span><span class="th-stat"><b class="mono">${fmtInt(pops)}</b> backtracked</span>` +
        `<span class="th-stat">depth <b class="mono">${stack.length}</b>/${N} <span class="muted">(max ${maxDepth})</span></span>` +
        (attempts > 1 ? `<span class="th-stat">attempt <b class="mono">${attempts}</b></span>` : "") +
        `<span class="th-bar"><i style="width:${pct}%"></i></span>` +
        `<label class="th-speed"><span class="sr-only">Replay speed</span>${icon("bolt")}<input type="range" min="0" max="100" value="${$("speed").value}" id="thSpeed"></label>` +
        `<button class="btn sm" type="button" id="thSkip">${icon("skip")}Skip</button><button class="btn sm" type="button" id="thStop">${icon("stop")}Stop</button>`;
      $("thSkip").onclick = () => ctx.cancelAnim(true);
      $("thStop").onclick = () => ctx.cancelAnim(false);
      $("thSpeed").oninput = (e) => { $("speed").value = e.target.value; };
    };
    let lastPaint = 0;
    const apply = (ev) => {
      if (ev === "R") { stack = []; attempts++; return; }
      if (ev >= 0) { stack.push(ev); pushes++; if (stack.length > maxDepth) maxDepth = stack.length; return; }
      for (let k = 0; k < -ev && stack.length; k++) { const v = stack.pop(); S.tried[v] = 1; pops++; }
    };
    const show = (animate) => {
      ctx.setPathDirect(stack.length ? stack : [S.cps[0]], owner);
      S.animateNext = animate;
      ctx.render();
    };
    const finish = (jump) => {
      cancelAnimationFrame(raf);
      if (jump) for (; i < trace.length; i++) apply(trace[i]);
      if (final) ctx.setPathDirect(final, owner);
      else ctx.setPathDirect(res.path && res.path.length ? res.path : stack.length ? stack : [S.cps[0]], owner);
      hud.hidden = true;
      S.anim = null;
      // let the red "tried" trails fade out
      const fade = () => {
        if (!S.tried || S.anim) return;
        let any = false;
        for (let v = 0; v < N; v++) { S.tried[v] *= 0.85; if (S.tried[v] > 0.03) any = true; }
        if (!any) S.tried = null;
        S.animateNext = false; ctx.render();
        if (any) requestAnimationFrame(fade);
      };
      S.animateNext = false;
      ctx.render();
      if (!reducedMotion()) requestAnimationFrame(fade); else { S.tried = null; ctx.render(); }
      onDone && onDone({ truncated: res.trace_truncated, pushes, pops, attempts });
    };
    S.anim = { cancel(jump) { stopped = true; finish(jump); } };
    const frame = (t) => {
      if (stopped) return;
      const dt = Math.min(100, t - last); last = t;
      acc += (traceRate() * dt) / 1000;
      let steps = Math.floor(acc);
      acc -= steps;
      const many = steps > 3;
      while (steps-- > 0 && i < trace.length) apply(trace[i++]);
      for (let v = 0; v < N; v++) if (S.tried[v] > 0) S.tried[v] *= 0.94;
      show(!many);
      if (t - sndT > 90 && traceRate() < 60) { sndT = t; sound.search(); }
      if (t - lastPaint > 60) { paint(); lastPaint = t; }
      if (i >= trace.length) { stopped = true; finish(false); return; }
      raf = requestAnimationFrame(frame);
    };
    paint();
    ctx.banner("");
    raf = requestAnimationFrame(frame);
  }

  // ================================================================ solve (agent / solver)
  async function runSolve(modeOverride, { fromStart = false } = {}) {
    if (S.busy || !S.data) return;
    if (racing()) { ctx.toast("Finish the race first"); return; }
    ctx.cancelAnim();
    const mode = modeOverride || A.mode;
    if (mode !== "exact" && !hasModels()) { ctx.banner("No trained model in checkpoints/: only the exact solver is available.", "warn"); return; }
    const startPath = !fromStart && $("fromPath").checked && S.path.length > 1 ? S.path.slice() : null;
    const watch = $("watchThink").checked && mode !== "greedy";
    const tl = Number($("exactTL").value) || 10;
    ctx.setBusy(true, mode === "search" ? "Policy DFS searching…" : mode === "hybrid" ? "Solver+GNN searching (then the plain solver, for comparison)…" : mode === "exact" ? "Exact solver running…" : "Agent thinking…");
    const tok = ctx.pathToken();
    let res;
    try {
      if (mode === "exact") res = await api("/api/solve/exact", { puzzle: S.data, time_limit: tl, start_path: startPath, trace: watch });
      else res = await api("/api/solve/rl", { puzzle: S.data, model: selModel(), mode, budget: Number($("budget").value) || 5000, start_path: startPath,
        time_limit: mode === "hybrid" ? tl : 30, trace: watch });
    } catch (e) { ctx.setBusy(false); if (tok.data === S.data) ctx.banner(`${MODE_LABEL[mode]} failed: ${esc(e.message)}`, "bad"); return; }
    ctx.setBusy(false);
    if (tok.data !== S.data || tok.id !== S.id) return;   // puzzle changed meanwhile
    if (startPath && !ctx.tokenValid(tok)) { ctx.banner("Your path changed while the AI was thinking; its answer was discarded. Run it again.", "warn"); return; }
    const startLen = startPath ? startPath.length : 1;
    if (mode === "exact") {
      if (!startPath) S.exact = res;
      updateCompare();
      const after = () => {
        if (res.solved) ctx.banner(`Exact solver: solved in ${fmtSec(res.seconds)} after expanding ${fmtInt(res.nodes_expanded)} search nodes${res.attempts > 1 ? ` over ${res.attempts} restarts` : ""}.`, "good");
        else ctx.banner(res.status === "unsat" ? (startPath ? "Your current path cannot be completed (proved by the exact solver)." : "No solution exists.") : `Exact solver timed out after ${fmtSec(res.seconds)} (${fmtInt(res.nodes_expanded)} nodes). Raise the limit and retry.`, "warn");
      };
      if (watch && res.trace) replayTrace(res, "exact", startLen, (x) => { after(); if (x.truncated) ctx.toast("Long search: the replay shows its first 60,000 events"); });
      else if (res.solved) ctx.playPath(res.path, startLen, null, "exact", after);
      else after();
      return;
    }
    S.agent = res;
    updateCompare(); updateConfPanel(); ctx.renderLegend();
    if (!S.exact && res.mode !== "hybrid") backgroundExact();
    const done = () => agentDone(res, startPath);
    if (watch && res.trace && res.trace.length) replayTrace(res, "agent", res.start_len, () => { S.conf = new Map((res.steps || []).map((s) => [s.node, s.p])); S.curStep = res.steps && res.steps.length ? res.steps[res.steps.length - 1] : null; updateConfPanel(); ctx.render(); done(); });
    else ctx.playPath(res.path, res.start_len, res.steps, "agent", done);
  }
  async function backgroundExact() {
    const d = S.data;
    try {
      const r = await api("/api/solve/exact", { puzzle: d, time_limit: Number($("exactTL").value) || 10 });
      if (S.data === d) { S.exact = r; updateCompare(); }
    } catch { /* ignore */ }
  }
  function agentDone(res, startPath) {
    if (res.solved) {
      if (res.mode === "hybrid") {
        const hs = res.hybrid, e = res.exact;
        const vs = e ? ` Plain solver: ${e.solved ? "solved" : e.status} in ${fmtSec(e.seconds)}, ${fmtInt(e.nodes_expanded)} expansions.` : "";
        ctx.banner(`Solver+GNN solved it in ${fmtSec(res.seconds)}: ${fmtInt(res.nodes_expanded)} expansions, ${fmtInt(hs.inference_calls)} GNN calls (${fmtSec(hs.inference_seconds + hs.obs_seconds)}), ${fmtInt(hs.cache_hits)} cache hits.${vs}`, "good");
      } else ctx.banner(`The agent solved it (${res.mode === "search" ? `policy DFS, ${fmtInt(res.nodes_expanded)} nodes, ${res.backtracks} backtracks` : "greedy, no backtracking"}) in ${fmtSec(res.seconds)}. Mean confidence ${(res.mean_confidence * 100).toFixed(0)}%.`, "good");
      ctx.celebrate(60);
      sound.win();
      return;
    }
    const hd = res.path[res.path.length - 1];
    const deadNode = res.dead_from ? res.path[res.dead_from - 1] : undefined;
    S.stuck = { node: hd, deadNode };
    ctx.render();
    sound.lose();
    const where = res.dead_from ? ` The position became unsolvable at move <b>${res.dead_from - 1}</b> (orange ring).` : "";
    const why = res.status === "budget" ? `ran out of budget (${fmtInt(res.nodes_expanded)} nodes)` : res.status === "timeout" ? "ran out of time" : res.mode === "hybrid" ? (startPath ? "proved your path cannot be completed" : "proved there is no solution") : res.mode === "search" ? "exhausted its search" : `got stuck after ${res.path.length - 1} moves (red ring)`;
    const acts = [];
    if (res.mode !== "hybrid") acts.push(["Try Solver+GNN", () => runSolve("hybrid"), "agent"]);
    if (res.mode === "greedy") acts.push(["Try policy DFS", () => runSolve("search")]);
    const doneTok = ctx.pathToken();
    if (res.dead_from) acts.push([`Undo to move ${res.dead_from - 2}`, () => { if (!ctx.tokenValid(doneTok)) { ctx.banner("The path has changed since the agent's run.", "warn"); return; } ctx.truncate(Math.max(1, res.dead_from - 1)); S.stuck = null; ctx.render(); }]);
    acts.push(["Exact solver", () => runSolve("exact")]);
    ctx.banner(`${res.mode === "hybrid" ? "Solver+GNN" : "The agent"} ${why}.${where}`, "bad", acts);
  }

  // ================================================================ race
  function racing() { return !!(A.race && (A.race.phase === "countdown" || A.race.phase === "running")); }
  function opp() { return OPPONENTS.find((o) => o.id === A.opp) || OPPONENTS[0]; }
  function paceMs() { const f = Math.pow(2.6, (50 - Number($("racePace").value)) / 50); return opp().pace * f; }
  function renderOpps() {
    const box = $("oppList");
    box.innerHTML = "";
    for (const o of OPPONENTS) {
      const on = o.id === A.opp;
      box.appendChild(h("button", { class: `opp ${on ? "on" : ""}`, type: "button", role: "radio", "aria-checked": String(on),
        onclick: () => { A.opp = o.id; store.set("zip-opp", o.id); renderOpps(); paintPace(); } },
        h("span", { class: `opp-av av-${o.id}`, html: icon(o.id === "rookie" ? "bolt" : o.id === "challenger" ? "brain" : "trophy") }),
        h("span", { class: "opp-txt" }, h("b", { text: o.label }), h("span", { class: "muted", text: o.desc }))));
    }
  }
  function paintPace() { $("racePaceLbl").textContent = `${(1000 / paceMs()).toFixed(1)}/s`; }
  function renderRaceRecord() { const r = ctx.records().race; $("raceRecord").textContent = r.w || r.l ? `${r.w} W · ${r.l} L` : ""; }
  function layoutMini() {
    if ($("raceSide").hidden || !S.P) return;
    const host = $("raceBoard");
    if (!A.mini) A.mini = new Board(host, { mini: true, maxCell: 34, label: "AI opponent's board" });
    A.mini.setPuzzle(S.P);
    const w = Math.max(120, $("raceSide").clientWidth - 4);
    A.mini.build(w, Math.max(140, Math.min(w * 1.2, 360)));
    renderMini();
  }
  function renderMini() {
    if (!A.mini || !A.race) return;
    const R = A.race;
    A.mini.render({ path: R.aiPath, won: R.aiSolved, legal: [], stuck: R.aiStuck ? { node: R.aiPath[R.aiPath.length - 1] } : null, tried: R.tried, animate: true });
    $("raceProgress").style.width = `${(100 * (R.aiPath.length - 1)) / Math.max(1, S.n - 1)}%`;
  }
  function setRaceStatus(txt, tone = "") { const s = $("raceStatus"); s.textContent = txt; s.className = `race-status ${tone}`; }
  function endRace() {
    const R = A.race;
    if (!R) return;
    R.phase = "done";
    R.timers.forEach(clearTimeout); R.timers = [];
    cancelAnimationFrame(R.raf);
    syncButtons();
  }
  async function startRace() {
    if (S.busy) return;
    if (!hasModels()) { ctx.banner("Racing needs a trained model in checkpoints/.", "warn"); return; }
    if (A.race) { endRace(); }
    ctx.cancelAnim();
    const o = opp();
    if ($("raceFresh").checked) { const ok = await ctx.newPuzzle(); if (ok === false) return; } else ctx.replay();
    if (!S.data) return;
    setVision(A.vision);   // hides the overlay while racing (re-enabled after)
    const R = A.race = { phase: "countdown", opp: o, aiPath: [S.cps[0]], aiSolved: false, aiStuck: false, aiMs: null, userMs: null, timers: [], tried: null, data: S.data, raf: 0 };
    $("raceSide").hidden = false;
    $("raceName").innerHTML = `${icon(o.id === "rookie" ? "bolt" : o.id === "challenger" ? "brain" : "trophy")} ${o.label}`;
    $("raceMeta").textContent = `${MODE_LABEL[o.mode]} · ${(1000 / paceMs()).toFixed(1)} moves/s`;
    setRaceStatus("getting ready…");
    ctx.layoutBoard(); ctx.render();
    $("insight").hidden = true;
    syncButtons();
    // fetch the AI's plan during the countdown
    const plan = api("/api/solve/rl", { puzzle: S.data, model: selModel(), mode: o.mode, budget: 4000, time_limit: 20, compare: false, trace: o.mode === "search" })
      .catch((e) => ({ error: e.message }));
    const cd = $("countdown");
    for (const k of ["3", "2", "1"]) {
      if (A.race !== R) return;
      cd.textContent = k; cd.className = "countdown show"; sound.count(false);
      await sleep(650);
    }
    if (A.race !== R || S.data !== R.data) return;
    cd.textContent = "Go!"; cd.className = "countdown show go"; sound.count(true);
    setTimeout(() => { cd.className = "countdown"; }, 450);
    R.phase = "running";
    R.t0 = performance.now();
    S.t0 = R.t0; S.tEnd = null;
    setRaceStatus("thinking…");
    syncButtons();
    const res = await plan;
    if (A.race !== R || R.phase !== "running") return;
    if (res.error) { setRaceStatus("error", "bad"); ctx.banner(`The AI could not play: ${esc(res.error)}`, "bad"); endRace(); return; }
    R.res = res;
    // schedule the AI's moves
    const pace = paceMs();
    const events = [];
    if (o.mode === "search" && res.trace) {
      let stack = [];
      for (const ev of res.trace) {
        if (ev === "R") { stack = []; continue; }
        if (ev >= 0) { stack.push(ev); if (stack.length > 1) events.push({ path: stack.slice(), dt: pace }); }
        else { const popped = stack.splice(stack.length + ev); events.push({ path: stack.slice(), dt: pace * 0.6, popped }); }
      }
    } else {
      for (let i = 1; i < res.path.length; i++) events.push({ path: res.path.slice(0, i + 1), dt: pace });
    }
    // a long search (thousands of backtracks) is compressed to at most ~2.5x a straight run
    const total = events.reduce((a, e) => a + e.dt, 0), cap = 2.5 * S.n * pace;
    if (total > cap) { const f = cap / total; for (const e of events) e.dt *= f; }
    const think = o.mode === "hybrid" ? Math.max(600, res.seconds * 1000) : 250;
    let t = think, k = 0;
    const step = () => {
      if (A.race !== R || R.phase !== "running") return;
      const e = events[k++];
      if (e) {
        R.aiPath = e.path;
        if (e.popped) { R.tried = R.tried || new Float32Array(S.n); for (const v of e.popped) R.tried[v] = 1; setRaceStatus("backtracking", "warn"); }
        else setRaceStatus(`move ${R.aiPath.length - 1}`);
        if (R.tried) for (let v = 0; v < S.n; v++) R.tried[v] *= 0.8;
        renderMini();
        R.timers.push(setTimeout(step, e.dt));
        return;
      }
      // AI finished its plan
      R.aiPath = res.path; R.tried = null;
      if (res.solved) {
        R.aiSolved = true; R.aiMs = performance.now() - R.t0;
        renderMini(); setRaceStatus(`finished · ${fmtTime(R.aiMs)}`, "good");
        endRace();
        aiWins();
      } else {
        R.aiStuck = true;
        renderMini(); setRaceStatus(`stuck after ${res.path.length - 1} moves`, "bad");
        ctx.banner(`${o.label} got stuck! Finish the puzzle to win.`, "good");
        R.phase = "running";   // the user can still win
      }
    };
    R.timers.push(setTimeout(step, t));
  }
  function aiWins() {
    const R = A.race;
    sound.lose();
    const r = ctx.records(); r.race.l = (r.race.l || 0) + 1; ctx.saveRecords(r); renderRaceRecord();
    ctx.banner(`<b>${R.opp.label}</b> finished first in ${fmtTime(R.aiMs)}. Keep going to finish yours, or try a rematch.`, "bad",
      [["Rematch", () => startRace(), "agent"], ["Easier opponent", () => { const i = OPPONENTS.findIndex((x) => x.id === A.opp); A.opp = OPPONENTS[Math.max(0, i - 1)].id; renderOpps(); startRace(); }]]);
  }
  function raceUserStarted() {
    if (A.race && A.race.phase === "running" && S.t0 == null) S.t0 = A.race.t0;
  }
  function raceUserFinished(ms) {
    const R = A.race;
    if (!R || R.phase !== "running" || S.data !== R.data) return false;
    const userMs = performance.now() - R.t0;
    endRace();
    const r = ctx.records(); r.race.w = (r.race.w || 0) + 1; ctx.saveRecords(r); renderRaceRecord();
    sound.win(); ctx.celebrate();
    setRaceStatus(R.aiStuck ? "stuck" : "still going…", R.aiStuck ? "bad" : "warn");
    $("winBadge").innerHTML = icon("trophy"); $("winBadge").className = "win-badge gold";
    $("winKicker").textContent = `Race vs ${R.opp.label}`;
    $("winTitle").textContent = R.aiStuck ? "You win: the AI got stuck!" : "You beat the AI!";
    $("winTime").textContent = fmtTime(userMs);
    $("winPb").hidden = true;
    const aiLeft = R.aiStuck ? "stuck" : `${R.aiPath.length - 1}/${S.n - 1}`;
    $("winStats").innerHTML = [["You", fmtTime(userMs)], ["AI", aiLeft], ["Hints", S.hints], ["Record", `${r.race.w}–${r.race.l}`]].map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
    $("shareText").textContent = `Zip race · beat ${R.opp.label} (${MODE_LABEL[R.opp.mode]})\n⏱ ${fmtTime(userMs)} · AI ${R.aiStuck ? "got stuck" : `at ${aiLeft} moves`}`;
    ctx.openModal("winModal");
    ctx.banner(`You won the race in <b>${fmtTime(userMs)}</b>.`, "good", [["Rematch", () => startRace(), "agent"]]);
    return true;
  }
  function closeRace() {
    if (A.race) endRace();
    A.race = null;
    $("raceSide").hidden = true;
    ctx.layoutBoard(); ctx.render();
  }

  // ================================================================ compare models
  function renderCmpModels() {
    const box = $("cmpModels");
    const prev = new Set([...box.querySelectorAll("input:checked")].map((i) => i.value));
    box.innerHTML = "";
    if (!A.models.length) { box.innerHTML = `<span class="muted">No checkpoints.</span>`; return; }
    const dflt = [selModel(), ...A.models.filter((m) => m.default).map((m) => m.name), ...A.models.map((m) => m.name)]
      .filter((x, i, a) => x && a.indexOf(x) === i).slice(0, 2);
    A.models.forEach((m, i) => {
      const on = prev.size ? prev.has(m.name) : dflt.includes(m.name);
      box.appendChild(h("label", { class: "check cmp-model" }, h("input", { type: "checkbox", value: m.name, checked: on || null }),
        h("span", { text: m.name.replace(/\.pt$/, ""), title: m.name }), m.stage ? h("span", { class: "muted", text: m.stage }) : null));
    });
  }
  async function runCompare() {
    if (S.busy || !S.data) return;
    const names = [...$("cmpModels").querySelectorAll("input:checked")].map((i) => i.value);
    if (!names.length) { ctx.toast("Pick at least one model"); return; }
    const box = $("cmpResults");
    box.innerHTML = "";
    A.cmpBoards = [];
    const d = S.data, mode = A.cmpMode;
    const cards = names.map((name) => {
      const c = h("div", { class: "cmp-card loading" },
        h("div", { class: "cmp-card-head" }, h("b", { text: name.replace(/\.pt$/, "") }), h("span", { class: "cmp-st muted", html: '<span class="spinner sm"></span>' })),
        h("div", { class: "cmp-mini" }), h("div", { class: "cmp-kv muted tiny" }));
      box.appendChild(c);
      return c;
    });
    $("cmpBtn").disabled = true;
    const results = [];
    for (let i = 0; i < names.length; i++) {
      if (S.data !== d) break;
      const c = cards[i];
      let r;
      try { r = await api("/api/solve/rl", { puzzle: d, model: names[i], mode, budget: 3000, time_limit: 15, compare: false }); }
      catch (e) { r = { error: e.message }; }
      if (S.data !== d) break;
      results.push([names[i], r]);
      c.classList.remove("loading");
      const st = c.querySelector(".cmp-st");
      if (r.error) { st.innerHTML = `<span class="fail">error</span>`; c.querySelector(".cmp-kv").textContent = r.error; continue; }
      st.innerHTML = r.solved ? `<span class="ok">${icon("check")} solved</span>` : `<span class="fail">${esc(r.status)}</span>`;
      const b = new Board(c.querySelector(".cmp-mini"), { mini: true, maxCell: 22, label: `${names[i]} result` });
      b.setPuzzle(S.P);
      const w = Math.max(110, c.clientWidth - 16);
      b.build(w, w);
      b.render({ path: r.path, won: r.solved, legal: [], stuck: r.solved ? null : { node: r.path[r.path.length - 1], deadNode: r.dead_from ? r.path[r.dead_from - 1] : undefined }, animate: false });
      A.cmpBoards.push(b);
      c.querySelector(".cmp-kv").innerHTML =
        `<span>path <b>${r.path.length}/${S.n}</b></span><span>conf <b>${r.mean_confidence != null ? fmtPct(r.mean_confidence) : "-"}</b></span>` +
        `<span>min <b>${r.min_confidence != null ? fmtPct(r.min_confidence) : "-"}</b></span><span>time <b>${fmtSec(r.seconds)}</b></span>` +
        `<span>expanded <b>${fmtInt(r.nodes_expanded)}</b></span>`;
      c.onclick = () => { S.agent = r; updateCompare(); updateConfPanel(); ctx.playPath(r.path, r.start_len, r.steps, "agent", () => agentDone(r, null)); };
      c.title = "Click to replay this model's path on the main board";
    }
    // highlight the winner: solved first, then higher mean confidence, then faster
    const ok = results.filter(([, r]) => !r.error);
    if (ok.length > 1) {
      ok.sort((a, b) => (b[1].solved - a[1].solved) || (b[1].path.length - a[1].path.length) || ((b[1].mean_confidence || 0) - (a[1].mean_confidence || 0)));
      const best = names.indexOf(ok[0][0]);
      if (best >= 0) cards[best].classList.add("best");
    }
    $("cmpBtn").disabled = false;
  }

  // ================================================================ hooks + wiring
  function syncButtons() {
    const busy = S.busy, race = racing();
    const needModel = A.mode !== "exact" && !hasModels();
    $("runBtn").disabled = busy || needModel || race;
    $("runBtn").innerHTML = `${icon(A.mode === "exact" ? "cpu" : "play")}<span>Run ${MODE_LABEL[A.mode]}</span>`;
    $("raceBtn").disabled = busy || !hasModels();
    $("raceBtn").innerHTML = race ? `${icon("flag")}<span>Restart race</span>` : `${icon("race")}<span>Start race</span>`;
    $("cmpBtn").disabled = busy || !hasModels();
    $("visionBtn").disabled = !hasModels();
    $("hintBtn").disabled = busy;
  }
  function onPathChanged() {
    if (A.vision) { A.policy = null; schedule(); if (A.info) renderInsight(true); }
  }
  function onPuzzleWillChange() {
    if (A.race && A.race.phase !== "done") { endRace(); }
    if (A.race) closeRaceUI();
    A.heat = null; A.policy = null; A.info = null; A.infoRev = -1;
    $("cmpResults").innerHTML = "";
    $("traceHud").hidden = true;
  }
  function closeRaceUI() { A.race = null; $("raceSide").hidden = true; }
  function onPuzzleLoaded() {
    updateCompare(); updateConfPanel();
    if (A.vision) schedule(0);
    renderInsight();
  }
  function afterRender() { /* hook for future overlays */ }
  function onTab(name) { if (name === "race") { renderOpps(); paintPace(); renderRaceRecord(); } }

  $("visionToggle").checked = A.vision;
  $("visionToggle").onchange = () => { if ($("visionToggle").checked && !hasModels()) { $("visionToggle").checked = false; ctx.toast("No trained model found"); return; } setVision($("visionToggle").checked); };
  $("visionBtn").onclick = toggleVision;
  $("visionBtn").classList.toggle("on", A.vision);
  $("heatFull").onchange = () => { schedule(0); ctx.render(); };
  $("heatCheck").onchange = () => schedule(0);
  $("refreshModels").onclick = loadModels;
  $("model").addEventListener("change", () => { renderModelInfo(); if (A.vision) schedule(0); });
  $("runBtn").onclick = () => runSolve();
  $("raceBtn").onclick = () => startRace();
  $("racePace").oninput = paintPace;
  $("cmpBtn").onclick = runCompare;
  $("stopBtn").onclick = () => ctx.cancelAnim(false);
  document.querySelectorAll("#modeSeg button").forEach((b) => b.onclick = () => {
    A.mode = b.dataset.mode;
    document.querySelectorAll("#modeSeg button").forEach((x) => x.classList.toggle("on", x === b));
    $("budgetRow").hidden = A.mode !== "search";
    $("modeNote").textContent = MODE_NOTE[A.mode];
    syncButtons();
  });
  document.querySelectorAll("#cmpModeSeg button").forEach((b) => b.onclick = () => {
    A.cmpMode = b.dataset.mode;
    document.querySelectorAll("#cmpModeSeg button").forEach((x) => x.classList.toggle("on", x === b));
  });
  $("modeNote").textContent = MODE_NOTE[A.mode];
  renderOpps(); paintPace(); renderRaceRecord(); syncButtons();

  return {
    overlay, onPathChanged, onPuzzleLoaded, onPuzzleWillChange, afterRender, updateConfPanel, updateCompare, drawSpark,
    syncButtons, loadModels, runSolve, toggleVision, setVision, raceUserStarted, raceUserFinished, layoutMini, onTab,
    startRace, closeRace, runCompare, state: A,
  };
}
