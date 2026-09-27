// Orbitable three.js views of Zip puzzles on 3D grids (and, via view4d.js, 4D grids).
//
// createView3D(container, {coords, edges, cps, n, onClick, theme}) ->
//   { update({path, legal, hint, stuck, colorAt}), setHeat(map|null), refreshTheme(), projectNode(v), dispose() }
//
// The engine (createGraphView) is shared with view4d.js: rounded glossy cells, a smooth gradient tube for
// the path (rounded corners, animated growth), always-readable checkpoint badges, layer peeling with ghosting,
// spacing ("explode"), damped orbit, focus-follow, forgiving picking (legal moves win along the ray, then a
// screen-space fallback), hover highlight, heat glow (e.g. GNN move probabilities) and keyboard shortcuts.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const AXES = ["r", "c", "z", "w"];

// ------------------------------------------------------------------ small helpers
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
function col(c) { const x = new THREE.Color(); try { x.set(c); } catch { x.set("#888"); } return x; }
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected || document.getElementById("zv-styles")) { stylesInjected = true; return; }
  stylesInjected = true;
  const s = document.createElement("style");
  s.id = "zv-styles";
  s.textContent = `
.zv-host { position: relative; overflow: hidden; outline: none; }
.zv-host:focus-visible { box-shadow: inset 0 0 0 2px var(--accent, #6d4ae0); }
.zv-host > canvas { position: absolute; inset: 0; width: 100% !important; height: 100% !important; display: block; z-index: 1; touch-action: none; }
.zv-host > .zv-bg { position: absolute; inset: 0; z-index: 0; pointer-events: none;
  background: radial-gradient(120% 90% at 50% 38%, color-mix(in srgb, var(--panel, #fff) 92%, var(--accent, #6d4ae0) 8%) 0%,
    var(--panel-2, #faf8f5) 55%, color-mix(in srgb, var(--panel-2, #faf8f5) 86%, var(--ink, #000) 14%) 130%); }
.zv-host > .overlay-label { z-index: 3; }
.zv-ui { position: absolute; inset: 0; z-index: 4; pointer-events: none; font: 12px/1.3 Inter, ui-sans-serif, system-ui, sans-serif; color: var(--ink-2, #555); }
.zv-ui > * { pointer-events: auto; }
.zv-pill { background: color-mix(in srgb, var(--panel, #fff) 86%, transparent); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  border: 1px solid var(--border, #ddd); border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08); }
.zv-bar { position: absolute; left: 10px; bottom: 10px; display: flex; gap: 4px; padding: 4px; align-items: center; flex-wrap: wrap; max-width: calc(100% - 90px); }
.zv-btn { height: 28px; min-width: 28px; padding: 0 9px; border-radius: 8px; border: 1px solid transparent; background: transparent; cursor: pointer;
  color: var(--ink-2, #555); font: 600 12px Inter, ui-sans-serif, system-ui, sans-serif; display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.zv-btn:hover { background: var(--panel-2, #f4f4f4); color: var(--ink, #111); }
.zv-btn.on { background: var(--accent, #6d4ae0); color: var(--accent-ink, #fff); }
.zv-btn svg { width: 14px; height: 14px; }
.zv-sep { width: 1px; height: 18px; background: var(--border, #ddd); margin: 0 2px; }
.zv-range { display: inline-flex; align-items: center; gap: 6px; padding: 0 6px; color: var(--ink-3, #888); font-weight: 600; }
.zv-range input { width: 84px; accent-color: var(--accent, #6d4ae0); }
.zv-layers { position: absolute; right: 10px; top: 50%; transform: translateY(-50%); display: flex; flex-direction: column; gap: 3px; padding: 5px; align-items: stretch; max-height: calc(100% - 70px); overflow-y: auto; }
.zv-layers .zv-h { font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3, #888); text-align: center; padding: 2px 0 3px; }
.zv-chip { height: 26px; min-width: 42px; border-radius: 7px; border: 1px solid var(--border, #ddd); background: var(--panel-2, #f6f6f6); cursor: pointer;
  font: 700 11.5px Inter, ui-sans-serif, system-ui, sans-serif; color: var(--ink-2, #555); position: relative; padding: 0 6px; }
.zv-chip:hover { border-color: var(--ink-3, #888); color: var(--ink, #111); }
.zv-chip.on { background: var(--accent, #6d4ae0); border-color: var(--accent, #6d4ae0); color: var(--accent-ink, #fff); }
.zv-chip.dim { opacity: .55; }
.zv-chip .zv-dot { position: absolute; right: -3px; top: -3px; width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--panel, #fff); }
.zv-chip .zv-sw { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; vertical-align: 0; }
.zv-seg2 { display: flex; gap: 2px; margin-top: 3px; }
.zv-seg2 .zv-chip { min-width: 0; flex: 1; font-size: 11px; }
.zv-top { position: absolute; right: 10px; top: 10px; display: flex; gap: 4px; padding: 4px; align-items: center; }
.zv-heat { position: absolute; left: 50%; transform: translateX(-50%); top: 10px; padding: 5px 10px; display: none; align-items: center; gap: 8px; font-weight: 600; }
.zv-heat i { display: inline-block; width: 70px; height: 7px; border-radius: 4px; background: linear-gradient(90deg, color-mix(in srgb, var(--heat, #ff6a3d) 12%, transparent), var(--heat, #ff6a3d)); }
.zv-tip { position: absolute; pointer-events: none; padding: 4px 8px; border-radius: 7px; background: var(--ink, #111); color: var(--bg, #fff);
  font: 600 11.5px Inter, ui-sans-serif, system-ui, sans-serif; white-space: nowrap; transform: translate(12px, 12px); display: none; z-index: 6; }
.zv-help { position: absolute; left: 10px; bottom: 52px; padding: 9px 12px; display: none; max-width: min(340px, calc(100% - 20px)); line-height: 1.6; }
.zv-help kbd { display: inline-block; min-width: 18px; padding: 0 4px; height: 18px; line-height: 17px; border: 1px solid var(--border, #ddd); border-bottom-width: 2px;
  border-radius: 4px; text-align: center; font: 600 10.5px ui-monospace, Menlo, monospace; background: var(--panel-2, #f6f6f6); color: var(--ink-2, #555); }
@media (max-width: 560px) { .zv-range input { width: 60px; } .zv-bar { max-width: calc(100% - 70px); } .zv-chip { min-width: 34px; } }
.zv-compact .zv-btn span, .zv-compact .zv-range span { display: none; }
.zv-compact .zv-range input { width: 64px; }
.zv-compact .zv-bar { gap: 2px; padding: 3px; }
.zv-compact .zv-chip { min-width: 34px; height: 24px; font-size: 11px; }
.zv-compact .zv-layers { right: 6px; }
.zv-compact .zv-heat span { display: none; }
`;
  document.head.appendChild(s);
}

const ICONS = {
  reset: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
  follow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="3.2"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></svg>',
  spin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 1-15.5 6.2"/><path d="M3 12a9 9 0 0 1 15.5-6.2"/><path d="M18 2v4h-4M6 22v-4h4"/></svg>',
  help: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="9.5"/><path d="M9.5 9.3a2.6 2.6 0 0 1 5 .9c0 1.8-2.5 2.2-2.5 3.8"/><circle cx="12" cy="17.3" r=".6" fill="currentColor"/></svg>',
};

// ------------------------------------------------------------------ canvas textures
function badgeTexture(text, ring, fill, ink) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.shadowColor = "rgba(0,0,0,.35)"; g.shadowBlur = 8; g.shadowOffsetY = 2;
  g.fillStyle = fill; g.beginPath(); g.arc(64, 64, 52, 0, Math.PI * 2); g.fill();
  g.shadowColor = "transparent";
  g.strokeStyle = ring; g.lineWidth = 9; g.beginPath(); g.arc(64, 64, 52, 0, Math.PI * 2); g.stroke();
  g.fillStyle = ink;
  g.font = `800 ${text.length > 2 ? 40 : text.length > 1 ? 52 : 62}px Inter, system-ui, sans-serif`;
  g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, 64, 68);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
function pillTexture(text, bg, ink, h = 44) {
  const c = document.createElement("canvas");
  const g = c.getContext("2d");
  const font = `700 ${Math.round(h * 0.56)}px Inter, system-ui, sans-serif`;
  g.font = font;
  const w = Math.ceil(g.measureText(text).width + h * 0.9);
  c.width = w; c.height = h;
  g.font = font;
  g.fillStyle = bg;
  const r = h / 2;
  g.beginPath(); g.moveTo(r, 0); g.lineTo(w - r, 0); g.arc(w - r, r, r, -Math.PI / 2, Math.PI / 2); g.lineTo(r, h); g.arc(r, r, r, Math.PI / 2, -Math.PI / 2); g.fill();
  g.fillStyle = ink; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, w / 2, h / 2 + 1);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return { tex: t, aspect: w / h };
}
let glowTex = null;
function glowTexture() {
  if (glowTex) return glowTex;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.25, "rgba(255,255,255,.75)");
  gr.addColorStop(0.6, "rgba(255,255,255,.18)"); gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  glowTex = new THREE.CanvasTexture(c);
  return glowTex;
}
let ringTex = null;
function ringTexture() {
  if (ringTex) return ringTex;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.strokeStyle = "#fff"; g.lineWidth = 10; g.beginPath(); g.arc(64, 64, 50, 0, Math.PI * 2); g.stroke();
  ringTex = new THREE.CanvasTexture(c);
  return ringTex;
}
function roundedRectShape(w, d, r) {
  const s = new THREE.Shape(), x = -w / 2, y = -d / 2;
  r = Math.min(r, w / 2, d / 2);
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + d - r); s.quadraticCurveTo(x + w, y + d, x + w - r, y + d);
  s.lineTo(x + r, y + d); s.quadraticCurveTo(x, y + d, x, y + d - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

// ------------------------------------------------------------------ tube geometry (parallel-transport frames)
function buildTube(samples, radius, radial, colorFn) {
  const m = samples.length;
  if (m < 2) return null;
  const P = samples.map((s) => s.p);
  const T = P.map((p, i) => {
    const a = P[Math.max(0, i - 1)], b = P[Math.min(m - 1, i + 1)];
    return new THREE.Vector3().subVectors(b, a).normalize();
  });
  let N = new THREE.Vector3();
  const t0 = T[0];
  const ax = Math.abs(t0.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  N.crossVectors(t0, ax).normalize();
  const pos = new Float32Array(m * radial * 3), nor = new Float32Array(m * radial * 3), clr = new Float32Array(m * radial * 3);
  const B = new THREE.Vector3(), nn = new THREE.Vector3(), q = new THREE.Quaternion(), axis = new THREE.Vector3();
  for (let i = 0; i < m; i++) {
    if (i > 0) {
      axis.crossVectors(T[i - 1], T[i]);
      const l = axis.length();
      if (l > 1e-6) {
        axis.divideScalar(l);
        q.setFromAxisAngle(axis, Math.acos(Math.max(-1, Math.min(1, T[i - 1].dot(T[i])))));
        N.applyQuaternion(q);
      }
      N.addScaledVector(T[i], -N.dot(T[i])).normalize();
    }
    B.crossVectors(T[i], N);
    const c = colorFn(samples[i].idx);
    for (let j = 0; j < radial; j++) {
      const a = (j / radial) * Math.PI * 2;
      nn.copy(N).multiplyScalar(Math.cos(a)).addScaledVector(B, Math.sin(a));
      const k = (i * radial + j) * 3;
      pos[k] = P[i].x + nn.x * radius; pos[k + 1] = P[i].y + nn.y * radius; pos[k + 2] = P[i].z + nn.z * radius;
      nor[k] = nn.x; nor[k + 1] = nn.y; nor[k + 2] = nn.z;
      clr[k] = c.r; clr[k + 1] = c.g; clr[k + 2] = c.b;
    }
  }
  const idx = [];
  for (let i = 0; i < m - 1; i++) for (let j = 0; j < radial; j++) {
    const a = i * radial + j, b = (i + 1) * radial + j, c = (i + 1) * radial + ((j + 1) % radial), d = i * radial + ((j + 1) % radial);
    idx.push(a, b, d, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  g.setAttribute("color", new THREE.BufferAttribute(clr, 3));
  g.setIndex(idx);
  return g;
}

// ------------------------------------------------------------------ the engine
/**
 * spec: {
 *   axisName: "z" | "w",                    group (layer / slice) axis shown in the peeling strip
 *   groupOf: Int32Array(n), groupCount, groupValues: number[] (label per group index),
 *   groupTint: bool (colour cells per group),
 *   modes: [{id, label, title}]             optional layout modes (segmented control, top right)
 *   layout(mode, params) -> {pos: Vector3[], jump(u, v) -> bool, lift: Vector3, plates: [{g, x, y, z, w, d}],
 *                            labels: [{text, pos: Vector3, g}], dynamic: bool}
 *   params: {explode, angle, ...},  ranges: [{key, label, min, max, step, title, modes?}]
 *   tick(dt, params, mode) -> bool          optional; true if the layout must be recomputed this frame
 *   help: string (HTML)
 * }
 */
export async function createGraphView(container, opts, spec) {
  const { coords, edges, cps, n, onClick } = opts;
  const themeFn = typeof opts.theme === "function" ? opts.theme : () => ({});
  injectStyles();
  container.classList.add("zv-host");
  if (!container.hasAttribute("tabindex")) container.tabIndex = 0;
  if (getComputedStyle(container).position === "static") container.style.position = "relative";

  const bg = document.createElement("div");
  bg.className = "zv-bg";
  container.prepend(bg);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.NoToneMapping;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
  scene.environment = envTex;
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 1000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.85;
  controls.zoomSpeed = 0.9;
  controls.panSpeed = 0.8;
  controls.screenSpacePanning = true;
  controls.autoRotateSpeed = 1.2;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x6f675f, 0.7);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(6, 14, 9);
  scene.add(sun);
  const rim = new THREE.DirectionalLight(0xbfd0ff, 0.45);
  rim.position.set(-8, 4, -10);
  scene.add(rim);

  // ---- state
  const S = {
    mode: spec.modes ? spec.modes[0].id : "default",
    params: { explode: 1, ...(spec.params || {}) },
    peel: { mode: "all", k: 0, sub: "upto" },   // mode: all | auto | set;  sub: upto | only
    follow: false, spin: false,
    path: [], legal: [], legalSet: new Set(), hint: null, stuck: null, colorAt: () => "#d63a8f",
    heat: null, hover: null,
    drawIdx: 0, targetIdx: 0, growFrom: 0, growT0: 0, growDur: 0,
  };
  const cpIndex = new Map(cps.map((v, i) => [v, i]));
  const groupOf = spec.groupOf;
  const G = spec.groupCount;

  // ---- layout
  let L = spec.layout(S.mode, S.params);
  const cur = L.pos.map((p) => p.clone());          // current (possibly animating) positions
  let trans = null;                                   // {from, to, t0, dur}

  // ---- colours
  let C = {};
  function readColors() {
    const t = themeFn() || {};
    const dark = (() => { const c = col(cssVar("--panel", "#ffffff")); return c.r + c.g + c.b < 1.0; })();
    C = {
      dark,
      bg: t.bg || cssVar("--panel-2", "#faf8f5"),
      cell: (() => { const c = col(cssVar("--cell-edge", t.cell || "#dcd4c8")); return dark ? c : c.lerp(col(cssVar("--ink", "#1d1c1a")), 0.1); })(),
      plate: t.plate || cssVar("--cell", "#f2eee8"),
      edge: t.edge || cssVar("--border", "#e6e1d8"),
      ink: cssVar("--ink", "#1d1c1a"), ink3: cssVar("--ink-3", "#8a857b"),
      panel: cssVar("--panel", "#ffffff"),
      legal: t.legal || cssVar("--legal", "#6d4ae0"),
      accent: cssVar("--accent", "#6d4ae0"),
      good: t.good || cssVar("--good", "#1f8a4c"), bad: t.bad || cssVar("--bad", "#d23c3c"),
      warn: cssVar("--warn", "#c7771a"),
      cp: cssVar("--cp", "#141414"), cpInk: cssVar("--cp-ink", "#ffffff"), cpRing: cssVar("--cp-ring", "#141414"),
      bridge: cssVar("--bridge", "#8b909a"),
      heat: t.heat || cssVar("--heat", "#ff6a3d"),
      islands: Array.from({ length: 8 }, (_, i) => cssVar(`--island-${i}`, "#e3eefc")),
    };
  }
  readColors();
  const groupTints = () => Array.from({ length: G }, (_, g) => {
    const hue = (0.72 + g / Math.max(1, G)) % 1;
    return new THREE.Color().setHSL(hue, C.dark ? 0.35 : 0.55, C.dark ? 0.42 : 0.72);
  });
  let tints = groupTints();

  // ---- scene objects
  const cellGeo = new RoundedBoxGeometry(0.38, 0.38, 0.38, 3, 0.09);
  const cellMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.0, envMapIntensity: 0.45 });
  const ghostMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.13, depthWrite: false });
  const cells = new THREE.InstancedMesh(cellGeo, cellMat, n);
  const ghosts = new THREE.InstancedMesh(cellGeo, ghostMat, n);
  ghosts.renderOrder = 1;
  const hitGeo = new THREE.BoxGeometry(0.86, 0.86, 0.86);
  const hits = new THREE.InstancedMesh(hitGeo, new THREE.MeshBasicMaterial(), n);
  hits.visible = false;
  for (const m of [cells, ghosts]) { m.frustumCulled = false; for (let v = 0; v < n; v++) m.setColorAt(v, new THREE.Color(1, 1, 1)); }
  hits.frustumCulled = false;
  scene.add(cells, ghosts, hits);

  const plateGroup = new THREE.Group();
  const labelGroup = new THREE.Group();
  const edgeGroup = new THREE.Group();
  const pathGroup = new THREE.Group();
  const fxGroup = new THREE.Group();
  const heatGroup = new THREE.Group();
  scene.add(plateGroup, edgeGroup, pathGroup, fxGroup, heatGroup, labelGroup);

  // soft contact shadow under the structure
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: glowTexture(), transparent: true, depthWrite: false, opacity: 0.22, color: 0x000000 }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.renderOrder = -1;
  scene.add(shadow);

  const badges = cps.map((v, k) => {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, depthWrite: false, sizeAttenuation: false, transparent: true }));
    s.renderOrder = 20;
    s.userData.node = v;
    s.userData.key = "";
    scene.add(s);
    return s;
  });

  // ---- helpers over current positions
  const visibleGroup = (g) => {
    const pm = S.peel;
    if (pm.mode === "all") return true;
    let k = pm.k;
    if (pm.mode === "auto") {
      if (!S.path.length) return true;
      k = groupOf[S.path[S.path.length - 1]];
      return g <= k;
    }
    return pm.sub === "only" ? g === k : g <= k;
  };
  const visNode = (v) => visibleGroup(groupOf[v]);

  function stepSamples(u, v, i, out, skipFirst) {
    const a = cur[u], b = cur[v];
    if (L.jump(u, v)) {
      const len = a.distanceTo(b);
      const ctrl = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5).addScaledVector(L.lift, 0.28 * len + 0.35);
      const cnt = Math.max(10, Math.ceil(len / 0.15));
      const bez = new THREE.QuadraticBezierCurve3(a, ctrl, b);
      for (let k = skipFirst ? 1 : 0; k <= cnt; k++) out.push({ p: bez.getPoint(k / cnt), idx: i + k / cnt });
      return true;
    }
    return false;
  }
  function jumpPoints(u, v, cnt = 14) {
    const a = cur[u], b = cur[v], len = a.distanceTo(b);
    const ctrl = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5).addScaledVector(L.lift, 0.28 * len + 0.35);
    const bez = new THREE.QuadraticBezierCurve3(a, ctrl, b);
    return bez.getPoints(cnt);
  }

  // Dense samples along the path (rounded corners, arcs for jumps); idx = fractional path index.
  function pathSamples(path) {
    const out = [];
    if (path.length < 2) return out;
    const steps = path.length - 1, R = 0.3;
    const segs = [];
    for (let i = 0; i < steps; i++) {
      const u = path[i], v = path[i + 1], a = cur[u], b = cur[v];
      const d = new THREE.Vector3().subVectors(b, a), len = d.length() || 1e-6;
      segs.push({ u, v, a, b, dir: d.divideScalar(len), len, jump: L.jump(u, v) });
    }
    const corner = (i) => i > 0 && i < steps && !segs[i - 1].jump && !segs[i].jump && segs[i - 1].dir.dot(segs[i].dir) < 0.999;
    for (let i = 0; i < steps; i++) {
      const s = segs[i];
      if (s.jump) { stepSamples(s.u, s.v, i, out, out.length > 0); continue; }
      const r0 = corner(i) ? Math.min(R, s.len / 2) : 0, r1 = corner(i + 1) ? Math.min(R, s.len / 2) : 0;
      const L0 = r0, L1 = s.len - r1;
      const cnt = Math.max(1, Math.ceil((L1 - L0) / 0.3));
      for (let k = (out.length && !corner(i)) ? 1 : 0; k <= cnt; k++) {
        const l = L0 + (L1 - L0) * (k / cnt);
        out.push({ p: s.a.clone().addScaledVector(s.dir, l), idx: i + l / s.len });
      }
      if (corner(i + 1)) {
        const nx = segs[i + 1], r2 = Math.min(R, nx.len / 2);
        const from = s.b.clone().addScaledVector(s.dir, -r1), to = s.b.clone().addScaledVector(nx.dir, r2);
        const bez = new THREE.QuadraticBezierCurve3(from, s.b.clone(), to);
        for (let k = 1; k <= 6; k++) {
          const t = k / 6;
          out.push({ p: bez.getPoint(t), idx: i + 1 - r1 / s.len + t * (r1 / s.len + r2 / nx.len) });
        }
      }
    }
    return out;
  }

  // ---- static-ish geometry: plates, labels, edges (rebuilt on layout / visibility changes)
  function clearGroup(g) {
    while (g.children.length) {
      const c = g.children.pop();
      if (c.geometry && !c.userData.sharedGeo) c.geometry.dispose();
      if (c.material) { if (c.material.map && !c.userData.sharedTex) c.material.map.dispose(); c.material.dispose(); }
    }
  }
  function buildPlates() {
    clearGroup(plateGroup); clearGroup(labelGroup);
    for (const p of L.plates || []) {
      const vis = visibleGroup(p.g);
      const geo = new THREE.ShapeGeometry(roundedRectShape(p.w, p.d, 0.32), 6);
      const tint = spec.groupTint ? tints[p.g].clone().lerp(col(C.plate), 0.55) : col(C.plate).lerp(col(C.cell), C.dark ? 0.2 : 0.45);
      const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: tint, transparent: true, opacity: vis ? (C.dark ? 0.5 : 0.42) : 0.1, roughness: 0.9, depthWrite: false, side: THREE.DoubleSide }));
      m.rotation.x = -Math.PI / 2;
      m.position.set(p.x, p.y, p.z);
      m.renderOrder = -1;
      plateGroup.add(m);
      const edgeGeo = new THREE.BufferGeometry().setFromPoints(roundedRectShape(p.w, p.d, 0.32).getPoints(10).map((q) => new THREE.Vector3(q.x, 0, -q.y)));
      const ln = new THREE.LineLoop(edgeGeo, new THREE.LineBasicMaterial({ color: col(C.cell), transparent: true, opacity: vis ? 0.9 : 0.2 }));
      ln.position.set(p.x, p.y + 0.002, p.z);
      plateGroup.add(ln);
    }
    for (const lb of L.labels || []) {
      const vis = lb.g == null || visibleGroup(lb.g);
      const { tex, aspect } = pillTexture(lb.text, C.panel, C.ink3);
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, sizeAttenuation: false, opacity: vis ? 0.95 : 0.35 }));
      s.userData.px = 17; s.userData.aspect = aspect;
      s.position.copy(lb.pos);
      s.renderOrder = 15;
      labelGroup.add(s);
    }
    sizeSprites();
  }
  function buildEdges() {
    clearGroup(edgeGroup);
    const solid = [], ghost = [], jumpSolid = [], jumpGhost = [];
    for (const [u, v] of edges) {
      const vis = visNode(u) && visNode(v);
      if (L.jump(u, v)) {
        const pts = jumpPoints(u, v);
        const arr = vis ? jumpSolid : jumpGhost;
        for (let k = 0; k + 1 < pts.length; k++) arr.push(pts[k], pts[k + 1]);
      } else (vis ? solid : ghost).push(cur[u], cur[v]);
    }
    const mk = (pts, color, opacity, dashed) => {
      if (!pts.length) return;
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = dashed
        ? new THREE.LineDashedMaterial({ color: col(color), transparent: true, opacity, dashSize: 0.14, gapSize: 0.1, depthWrite: false })
        : new THREE.LineBasicMaterial({ color: col(color), transparent: true, opacity, depthWrite: false });
      const ls = new THREE.LineSegments(g, mat);
      if (dashed) ls.computeLineDistances();
      edgeGroup.add(ls);
    };
    const edgeCol = C.dark ? C.ink3 : C.cell;
    mk(solid, edgeCol, C.dark ? 0.55 : 0.95, false);
    mk(ghost, edgeCol, 0.12, false);
    mk(jumpSolid, C.bridge, spec.jumpOpacity ?? 0.5, true);
    mk(jumpGhost, C.bridge, 0.08, true);
  }

  // ---- dynamic: cells, badges, path tube, fx
  const mtx = new THREE.Matrix4(), qI = new THREE.Quaternion(), vS = new THREE.Vector3();
  let pulse = 0;
  function cellColor(v, onPath) {
    let c = spec.groupTint ? tints[groupOf[v]].clone() : col(C.cell);
    if (onPath != null) c = c.lerp(col(S.colorAt(onPath)), 0.6);
    const h = S.heat && S.heat.get(v);
    if (h) c.lerp(col(C.heat), Math.min(0.9, 0.2 + 0.75 * h));
    if (S.legalSet.has(v)) c = col(C.legal);
    if (S.hover === v) c.lerp(col(C.accent), 0.35).offsetHSL(0, 0, C.dark ? 0.12 : 0.04);
    return c;
  }
  let onPathMap = new Map();
  function applyCells() {
    const pulseS = 1 + 0.07 * Math.sin(pulse * 5);
    for (let v = 0; v < n; v++) {
      const vis = visNode(v);
      const op = onPathMap.get(v);
      let s = cpIndex.has(v) ? 0.001 : op != null ? 0.72 : 1;
      if (L.scale) s *= L.scale[v];
      if (S.legalSet.has(v)) s = 1.12 * pulseS;
      if (S.hover === v && !cpIndex.has(v)) s *= 1.28;
      if (S.heat && S.heat.get(v) && !cpIndex.has(v)) s = Math.max(s, 0.85 + 0.3 * S.heat.get(v));
      vS.setScalar(vis ? s : 0.0001);
      mtx.compose(cur[v], qI, vS);
      cells.setMatrixAt(v, mtx);
      vS.setScalar(vis ? 0.0001 : Math.max(0.001, s));
      mtx.compose(cur[v], qI, vS);
      ghosts.setMatrixAt(v, mtx);
      vS.setScalar(1);
      mtx.compose(cur[v], qI, vS);
      hits.setMatrixAt(v, mtx);
      const c = cellColor(v, op);
      cells.setColorAt(v, c);
      ghosts.setColorAt(v, c);
    }
    cells.instanceMatrix.needsUpdate = ghosts.instanceMatrix.needsUpdate = hits.instanceMatrix.needsUpdate = true;
    cells.instanceColor.needsUpdate = ghosts.instanceColor.needsUpdate = true;
    hits.computeBoundingSphere();
  }
  function applyBadges() {
    const next = (() => { let k = 0; for (const v of S.path) if (cpIndex.get(v) === k) k++; return k; })();
    badges.forEach((s, k) => {
      const v = cps[k];
      const op = onPathMap.get(v);
      const ring = op != null ? S.colorAt(op) : k === next && S.path.length ? C.accent : (C.dark ? C.cpRing : "#ffffff");
      const key = `${ring}|${C.cp}|${C.cpInk}`;
      if (s.userData.key !== key) {
        if (s.material.map) s.material.map.dispose();
        s.material.map = badgeTexture(String(k + 1), ring, C.cp, C.cpInk);
        s.material.needsUpdate = true;
        s.userData.key = key;
      }
      s.position.copy(cur[v]);
      const vis = visNode(v);
      s.material.opacity = vis ? 1 : 0.22;
      s.userData.px = (vis ? 30 : 20) * (S.hover === v ? 1.15 : 1) * (k === next && S.path.length ? 1 + 0.06 * Math.sin(pulse * 5) : 1);
    });
  }
  let pathMeshes = [];
  const tubeMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.35, metalness: 0.0, envMapIntensity: 0.35 });
  tubeMat.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace("#include <emissivemap_fragment>",
      "#include <emissivemap_fragment>\n#ifdef USE_COLOR\n totalEmissiveRadiance += vColor.rgb * 0.38;\n#endif");
  };
  const tubeGhostMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.22, depthWrite: false });
  const headGeo = new THREE.SphereGeometry(0.2, 24, 18);
  const capGeo = new THREE.SphereGeometry(0.105, 16, 12);
  function buildPath() {
    for (const m of pathMeshes) { pathGroup.remove(m); if (!m.userData.sharedGeo) m.geometry.dispose(); if (m.material !== tubeMat && m.material !== tubeGhostMat) m.material.dispose(); }
    pathMeshes = [];
    const P = S.path;
    const colorFn = (idx) => col(S.colorAt(idx));
    let endPoint = P.length ? cur[P[0]].clone() : null;
    if (P.length >= 2) {
      let sm = pathSamples(P);
      const lim = S.drawIdx;
      if (lim < P.length - 1 - 1e-6) {
        const cut = [];
        for (let i = 0; i < sm.length; i++) {
          if (sm[i].idx <= lim) cut.push(sm[i]);
          else {
            if (i > 0) {
              const a = sm[i - 1], b = sm[i], f = (lim - a.idx) / Math.max(1e-6, b.idx - a.idx);
              cut.push({ p: a.p.clone().lerp(b.p, f), idx: lim });
            }
            break;
          }
        }
        sm = cut;
      }
      // split into runs by visibility of the step
      const stepVis = (idx) => { const i = Math.min(P.length - 2, Math.max(0, Math.floor(idx - 1e-6))); return visNode(P[i]) && visNode(P[i + 1]); };
      let run = [], runVis = null;
      const flush = () => {
        if (run.length >= 2) {
          const g = buildTube(run, 0.105, 12, colorFn);
          if (g) { const m = new THREE.Mesh(g, runVis ? tubeMat : tubeGhostMat); pathGroup.add(m); pathMeshes.push(m); }
        }
      };
      for (const s of sm) {
        const vis = stepVis(s.idx);
        if (runVis !== null && vis !== runVis) { flush(); run = [run[run.length - 1]]; }
        runVis = vis;
        run.push(s);
      }
      flush();
      if (sm.length) endPoint = sm[sm.length - 1].p.clone();
      // round start cap
      const cap = new THREE.Mesh(capGeo, new THREE.MeshPhysicalMaterial({ color: col(S.colorAt(0)), roughness: 0.3, clearcoat: 0.8 }));
      cap.userData.sharedGeo = true;
      cap.position.copy(cur[P[0]]);
      pathGroup.add(cap); pathMeshes.push(cap);
    }
    if (P.length) {
      const head = P[P.length - 1];
      const hc = col(S.colorAt(Math.max(0, Math.min(S.drawIdx, P.length - 1))));
      const hm = new THREE.Mesh(headGeo, new THREE.MeshPhysicalMaterial({ color: hc, emissive: hc.clone().multiplyScalar(0.35), roughness: 0.25, clearcoat: 1 }));
      hm.userData.sharedGeo = true;
      hm.position.copy(endPoint);
      hm.userData.head = true;
      if (!cpIndex.has(head) || S.drawIdx < P.length - 1) { pathGroup.add(hm); pathMeshes.push(hm); }
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: hc, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
      glow.userData.sharedTex = true;
      glow.userData.glow = true;
      glow.position.copy(endPoint);
      glow.scale.setScalar(1.1);
      pathGroup.add(glow); pathMeshes.push(glow);
    }
  }
  function buildFx() {
    clearGroup(fxGroup);
    const ring = (v, color, scale, dashed) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTexture(), color: col(color), transparent: true, depthWrite: false, depthTest: false, opacity: dashed ? 0.7 : 0.95 }));
      s.userData.sharedTex = true;
      s.userData.pulse = true;
      s.userData.base = scale;
      s.position.copy(cur[v]);
      s.renderOrder = 18;
      fxGroup.add(s);
    };
    if (S.hint != null) ring(S.hint, C.good, 0.95);
    if (S.stuck != null) ring(S.stuck, C.bad, 0.95);
    // legal links from the head (arcs make w/bridge moves obvious)
    if (S.path.length && S.legal.length) {
      const h = S.path[S.path.length - 1];
      const pts = [];
      for (const v of S.legal) {
        if (!visNode(v)) continue;
        if (L.jump(h, v)) { const q = jumpPoints(h, v); for (let k = 0; k + 1 < q.length; k++) pts.push(q[k], q[k + 1]); }
        else pts.push(cur[h], cur[v]);
      }
      if (pts.length) {
        const ls = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: col(C.legal), transparent: true, opacity: 0.85, depthWrite: false }));
        fxGroup.add(ls);
      }
    }
    if (S.hover != null && !cpIndex.has(S.hover)) ring(S.hover, C.accent, 0.75);
  }
  const heatLabels = [];
  function buildHeat() {
    clearGroup(heatGroup);
    heatLabels.length = 0;
    if (!S.heat) { heatBadge.style.display = "none"; return; }
    heatBadge.style.display = "flex";
    // a probability distribution (e.g. the policy's move probabilities) gets % labels; any other heat map just glows
    let sum = 0;
    S.heat.forEach((h) => { sum += h; });
    const isProb = sum <= 1.05;
    heatBadge.querySelector("span").textContent = isProb ? "AI move probability" : "AI heat";
    const c = col(C.heat);
    const items = [...S.heat.entries()].filter(([v, h]) => h > 0.02 && v >= 0 && v < n).sort((a, b) => b[1] - a[1]);
    items.forEach(([v, h], rank) => {
      const g = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: c, transparent: true, opacity: (visNode(v) ? 0.18 : 0.05) + 0.45 * h, depthWrite: false, blending: C.dark ? THREE.AdditiveBlending : THREE.NormalBlending }));
      g.userData.sharedTex = true;
      g.position.copy(cur[v]);
      g.scale.setScalar(0.7 + (isProb ? 1.3 : 0.8) * h);
      g.userData.node = v;
      heatGroup.add(g);
      if (isProb && rank < 6 && h >= 0.03) {
        const { tex, aspect } = pillTexture(h >= 0.995 ? "99%" : `${Math.round(h * 100)}%`, `#${c.getHexString(THREE.SRGBColorSpace)}`, "#ffffff", 40);
        const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, sizeAttenuation: false, opacity: visNode(v) ? 1 : 0.35 }));
        s.userData.px = 15; s.userData.aspect = aspect; s.userData.node = v; s.userData.offset = true;
        s.renderOrder = 21;
        heatGroup.add(s);
        heatLabels.push(s);
      }
    });
    placeHeat();
    sizeSprites();
  }
  function placeHeat() {
    for (const s of heatGroup.children) {
      const v = s.userData.node;
      s.position.copy(cur[v]);
      if (s.userData.offset) s.position.y += 0.42;
    }
  }

  // ---- sizing of screen-constant sprites
  let viewH = 400;
  function sizeSprites() {
    const k = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / viewH;
    for (const s of badges) s.scale.set(s.userData.px * k, s.userData.px * k, 1);
    for (const s of [...labelGroup.children, ...heatLabels]) s.scale.set(s.userData.px * k * s.userData.aspect, s.userData.px * k, 1);
  }

  // ---- full refresh
  function refreshAll() {
    onPathMap = new Map(S.path.map((v, i) => [v, i]));
    applyCells(); applyBadges(); buildPath(); buildFx(); placeHeat(); sizeSprites();
  }
  function relayoutStatic() {
    buildPlates(); buildEdges();
    fitShadow();
  }
  function fitShadow() {
    const box = new THREE.Box3().setFromPoints(cur);
    const sz = box.getSize(new THREE.Vector3());
    shadow.position.set((box.min.x + box.max.x) / 2, box.min.y - 0.45, (box.min.z + box.max.z) / 2);
    shadow.scale.set(sz.x * 1.5 + 3, sz.z * 1.5 + 3, 1);
    shadow.material.opacity = C.dark ? 0.35 : 0.16;
  }

  // ---- camera
  let home = null;
  function computeHome() {
    // exact fit: smallest distance along the view direction that keeps every cell inside the frustum
    const box = new THREE.Box3().setFromPoints(L.pos);
    const center = box.getCenter(new THREE.Vector3());
    const cd = typeof spec.cameraDir === "function" ? spec.cameraDir(S.mode) : spec.cameraDir;
    const back = (cd || new THREE.Vector3(0.62, 0.62, 0.9)).clone().normalize();
    const right = new THREE.Vector3(0, 1, 0).cross(back).normalize();
    const up = back.clone().cross(right).normalize();
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2), tanH = tanV * (camera.aspect || 1);
    const pad = 0.55;   // cell half-size + badge room
    let dist = 2;
    const q = new THREE.Vector3();
    for (const p of L.pos) {
      q.subVectors(p, center);
      const x = Math.abs(q.dot(right)) + pad, y = Math.abs(q.dot(up)) + pad, z = q.dot(back) + pad;
      dist = Math.max(dist, z + x / (tanH * 0.92), z + y / (tanV * 0.88));
    }
    home = { target: center, pos: center.clone().addScaledVector(back, dist) };
  }
  let camAnim = null;
  function flyTo(target, pos, dur = 650) {
    camAnim = { t0: performance.now(), dur, fromT: controls.target.clone(), fromP: camera.position.clone(), toT: target.clone(), toP: pos.clone() };
  }
  function resetCamera(animate = true) {
    computeHome();
    if (animate) flyTo(home.target, home.pos);
    else { controls.target.copy(home.target); camera.position.copy(home.pos); controls.update(); }
  }

  // ---- overlay UI
  const ui = document.createElement("div");
  ui.className = "zv-ui";
  container.appendChild(ui);
  const tip = document.createElement("div");
  tip.className = "zv-tip";
  ui.appendChild(tip);
  const heatBadge = document.createElement("div");
  heatBadge.className = "zv-heat zv-pill";
  heatBadge.innerHTML = `<span>AI move probability</span><i></i>`;
  ui.appendChild(heatBadge);

  const bar = document.createElement("div");
  bar.className = "zv-bar zv-pill";
  ui.appendChild(bar);
  const mkBtn = (html, title, fn, parent = bar) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "zv-btn"; b.innerHTML = html; b.title = title;
    b.addEventListener("click", (e) => { e.stopPropagation(); fn(b); container.focus({ preventScroll: true }); });
    parent.appendChild(b);
    return b;
  };
  const resetBtn = mkBtn(`${ICONS.reset}<span>Reset</span>`, "Reset camera (C)", () => resetCamera());
  const followBtn = mkBtn(`${ICONS.follow}<span>Follow</span>`, "Keep the path head centred (F)", () => setFollow(!S.follow));
  const spinBtn = mkBtn(`${ICONS.spin}`, "Auto-rotate (O)", () => setSpin(!S.spin));
  void resetBtn;
  const sep = document.createElement("span"); sep.className = "zv-sep"; bar.appendChild(sep);
  const rangeInputs = {};
  for (const r of [{ key: "explode", label: "Spacing", min: 0.6, max: 2.6, step: 0.05, title: "Spread layers apart (+ / -)" }, ...(spec.ranges || [])]) {
    const w = document.createElement("label");
    w.className = "zv-range"; w.title = r.title || r.label;
    w.innerHTML = `<span>${r.label}</span>`;
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = r.min; inp.max = r.max; inp.step = r.step; inp.value = S.params[r.key];
    inp.addEventListener("input", () => { S.params[r.key] = Number(inp.value); relayout(false); });
    inp.addEventListener("keydown", (e) => e.stopPropagation());
    w.appendChild(inp);
    bar.appendChild(w);
    rangeInputs[r.key] = { el: w, input: inp, modes: r.modes };
  }
  const helpBtn = mkBtn(ICONS.help, "Controls", () => { help.style.display = help.style.display === "block" ? "none" : "block"; });
  void helpBtn;
  const help = document.createElement("div");
  help.className = "zv-help zv-pill";
  help.innerHTML = `<b>Mouse</b>: drag to orbit, right-drag / two fingers to pan, wheel to zoom, double-click a cell to centre it. Click a highlighted cell to move.<br>
    <kbd>[</kbd> <kbd>]</kbd> peel ${spec.axisName} layers &nbsp;<kbd>1</kbd>-<kbd>9</kbd> ${spec.axisName} = k &nbsp;<kbd>0</kbd> all &nbsp;<kbd>I</kbd> only / up to<br>
    <kbd>F</kbd> follow head &nbsp;<kbd>C</kbd> reset camera &nbsp;<kbd>O</kbd> spin &nbsp;<kbd>+</kbd> <kbd>-</kbd> spacing${spec.help ? "<br>" + spec.help : ""}`;
  ui.appendChild(help);

  // modes (top right)
  let modeBtns = [];
  if (spec.modes && spec.modes.length > 1) {
    const top = document.createElement("div");
    top.className = "zv-top zv-pill";
    ui.appendChild(top);
    modeBtns = spec.modes.map((m) => {
      const b = mkBtn(m.label, m.title || m.label, () => setMode(m.id), top);
      b.dataset.mode = m.id;
      return b;
    });
  }
  // layer strip (right)
  const strip = document.createElement("div");
  strip.className = "zv-layers zv-pill";
  ui.appendChild(strip);
  let chips = [];
  function buildStrip() {
    strip.innerHTML = "";
    if (G <= 1) { strip.style.display = "none"; return; }
    const h = document.createElement("div");
    h.className = "zv-h"; h.textContent = spec.axisName;
    strip.appendChild(h);
    const mkChip = (label, title, fn) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "zv-chip"; b.innerHTML = label; b.title = title;
      b.addEventListener("click", (e) => { e.stopPropagation(); fn(); container.focus({ preventScroll: true }); });
      strip.appendChild(b);
      return b;
    };
    chips = [];
    chips.push(mkChip("All", "Show every layer (0)", () => setPeel("all")));
    chips.push(mkChip("Auto", `Hide ${spec.axisName} layers above the path head`, () => setPeel("auto")));
    chips[0].dataset.p = "all"; chips[1].dataset.p = "auto";
    for (let g = G - 1; g >= 0; g--) {
      const sw = spec.groupTint ? `<span class="zv-sw" style="background:#${tints[g].getHexString(THREE.SRGBColorSpace)}"></span>` : "";
      const b = mkChip(`${sw}${spec.axisName}${spec.groupValues[g]}`, `Show ${spec.axisName} = ${spec.groupValues[g]} (${g + 1 <= 9 ? g + 1 : ""})`, () => {
        if (S.peel.mode === "set" && S.peel.k === g) setPeel("all"); else setPeel("set", g);
      });
      b.dataset.g = g;
      chips.push(b);
    }
    const seg = document.createElement("div");
    seg.className = "zv-seg2";
    strip.appendChild(seg);
    const subBtns = [["upto", "≤", "Show this layer and everything below it"], ["only", "=", "Show only this layer"]].map(([id, lbl, t]) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "zv-chip"; b.textContent = lbl; b.title = t + " (I)"; b.dataset.sub = id;
      b.addEventListener("click", (e) => { e.stopPropagation(); S.peel.sub = id; if (S.peel.mode !== "set") S.peel.mode = "set", S.peel.k = headGroup(); onPeel(); });
      seg.appendChild(b);
      return b;
    });
    chips.push(...subBtns);
    syncStrip();
  }
  const headGroup = () => (S.path.length ? groupOf[S.path[S.path.length - 1]] : G - 1);
  function syncStrip() {
    const hg = S.path.length ? groupOf[S.path[S.path.length - 1]] : -1;
    for (const b of chips) {
      if (b.dataset.p) b.classList.toggle("on", S.peel.mode === b.dataset.p);
      else if (b.dataset.sub) b.classList.toggle("on", S.peel.sub === b.dataset.sub && S.peel.mode === "set");
      else if (b.dataset.g != null) {
        const g = Number(b.dataset.g);
        b.classList.toggle("on", S.peel.mode === "set" && S.peel.k === g);
        b.classList.toggle("dim", !visibleGroup(g));
        let dot = b.querySelector(".zv-dot");
        if (g === hg) {
          if (!dot) { dot = document.createElement("span"); dot.className = "zv-dot"; b.appendChild(dot); }
          dot.style.background = S.colorAt(S.path.length - 1);
        } else if (dot) dot.remove();
      }
    }
    followBtn.classList.toggle("on", S.follow);
    spinBtn.classList.toggle("on", S.spin);
    for (const b of modeBtns) b.classList.toggle("on", b.dataset.mode === S.mode);
    for (const { el, modes } of Object.values(rangeInputs)) el.style.display = !modes || modes.includes(S.mode) ? "" : "none";
  }
  function setPeel(mode, k) {
    S.peel.mode = mode;
    if (k != null) S.peel.k = Math.max(0, Math.min(G - 1, k));
    onPeel();
  }
  function onPeel() { relayoutStatic(); refreshAll(); buildHeat(); syncStrip(); }
  function setFollow(on) { S.follow = on; syncStrip(); }
  function setSpin(on) { S.spin = on; controls.autoRotate = on && !(spec.spinParam && spec.spinParam(S.mode)); syncStrip(); }
  function setMode(m) {
    if (m === S.mode) return;
    S.mode = m;
    relayout(true);
    setTimeout(() => resetCamera(), 30);
    setSpin(S.spin);
  }
  buildStrip();

  // ---- layout transitions
  function relayout(animate) {
    const nl = spec.layout(S.mode, S.params);
    if (animate) {
      trans = { from: cur.map((p) => p.clone()), to: nl.pos, t0: performance.now(), dur: 520, next: nl };
      L = { ...nl, pos: nl.pos };
      L.jump = nl.jump;
    } else {
      L = nl;
      for (let v = 0; v < n; v++) cur[v].copy(nl.pos[v]);
      relayoutStatic(); refreshAll(); placeHeat();
    }
  }

  // ---- picking
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  function toNdc(e) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    return r;
  }
  function screenOf(v, r) {
    const p = cur[v].clone().project(camera);
    return { x: r.left + ((p.x + 1) / 2) * r.width, y: r.top + ((1 - p.y) / 2) * r.height, z: p.z };
  }
  function pick(e) {
    const r = toNdc(e);
    ray.setFromCamera(ndc, camera);
    const cand = [];
    for (const h of ray.intersectObject(hits, false)) if (h.instanceId != null && visNode(h.instanceId)) cand.push({ v: h.instanceId, d: h.distance });
    const legalHit = cand.find((c) => S.legalSet.has(c.v));
    if (legalHit) return legalHit.v;
    const sp = ray.intersectObjects(badges.filter((s) => visNode(s.userData.node)), false)[0];
    if (sp) return sp.object.userData.node;
    if (cand.length) return cand[0].v;
    // screen-space fallback: nearest legal (then any visible) node near the pointer
    let best = null, bestD = Infinity;
    for (const pass of [S.legal, null]) {
      const list = pass || [...Array(n).keys()];
      const lim = pass ? 34 : 16;
      for (const v of list) {
        if (!visNode(v)) continue;
        const s = screenOf(v, r);
        if (s.z > 1) continue;
        const d = Math.hypot(s.x - e.clientX, s.y - e.clientY);
        if (d < lim && d < bestD) { best = v; bestD = d; }
      }
      if (best != null) return best;
    }
    return null;
  }
  const el = renderer.domElement;
  let downAt = null, hoverRaf = 0, lastMove = null;
  const onDown = (e) => { downAt = [e.clientX, e.clientY, performance.now()]; };
  const onUp = (e) => {
    if (!downAt || e.button > 0) { downAt = null; return; }
    const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
    downAt = null;
    if (moved > 6) return;
    const v = pick(e);
    if (v != null && typeof onClick === "function") onClick(v);
  };
  const onMove = (e) => {
    lastMove = e;
    if (hoverRaf) return;
    hoverRaf = requestAnimationFrame(() => {
      hoverRaf = 0;
      const ev = lastMove;
      if (downAt && Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 6) { setHover(null); return; }
      setHover(pick(ev), ev);
    });
  };
  const onLeave = () => { setHover(null); };
  const onDbl = (e) => {
    const v = pick(e);
    if (v == null) return;
    const off = camera.position.clone().sub(controls.target);
    flyTo(cur[v], cur[v].clone().add(off.multiplyScalar(0.8)), 500);
  };
  el.addEventListener("pointerdown", onDown);
  el.addEventListener("pointerup", onUp);
  el.addEventListener("pointermove", onMove);
  el.addEventListener("pointerleave", onLeave);
  el.addEventListener("dblclick", onDbl);
  function setHover(v, ev) {
    if (v !== S.hover) {
      S.hover = v;
      applyCells(); applyBadges(); buildFx();
    }
    el.style.cursor = v != null && S.legalSet.has(v) ? "pointer" : v != null ? "default" : "grab";
    if (v == null || !ev) { tip.style.display = "none"; return; }
    const r = container.getBoundingClientRect();
    const parts = [coords[v].map((x, k) => `${AXES[k] || "d" + k}${Math.round(x)}`).join(" ")];
    if (cpIndex.has(v)) parts.push(`checkpoint ${cpIndex.get(v) + 1}`);
    if (S.legalSet.has(v)) parts.push("legal move");
    const pi = onPathMap.get(v);
    if (pi != null) parts.push(`step ${pi + 1}`);
    const h = S.heat && S.heat.get(v);
    if (h) parts.push(`p = ${(h * 100).toFixed(1)}%`);
    tip.textContent = parts.join(" · ");
    tip.style.display = "block";
    tip.style.left = `${Math.min(ev.clientX - r.left, r.width - 180)}px`;
    tip.style.top = `${ev.clientY - r.top}px`;
  }

  // ---- keyboard (only when the view has focus; keys the play page uses are left alone)
  const onKey = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.("input, select, textarea")) return;
    const k = e.key;
    let handled = true;
    if (k === "[") setPeel("set", S.peel.mode === "set" ? S.peel.k - 1 : headGroup() - 1);
    else if (k === "]") { if (S.peel.mode === "set" && S.peel.k >= G - 1) setPeel("all"); else setPeel("set", S.peel.mode === "set" ? S.peel.k + 1 : headGroup()); }
    else if (k === "0") setPeel("all");
    else if (/^[1-9]$/.test(k) && Number(k) <= G) setPeel("set", Number(k) - 1);
    else if (k === "i" || k === "I") { S.peel.sub = S.peel.sub === "only" ? "upto" : "only"; if (S.peel.mode !== "set") { S.peel.mode = "set"; S.peel.k = headGroup(); } onPeel(); }
    else if (k === "f" || k === "F") setFollow(!S.follow);
    else if (k === "c" || k === "C" || k === "Home") resetCamera();
    else if (k === "o" || k === "O") setSpin(!S.spin);
    else if (k === "+" || k === "=") { S.params.explode = Math.min(2.6, S.params.explode + 0.15); rangeInputs.explode.input.value = S.params.explode; relayout(false); }
    else if (k === "-" || k === "_") { S.params.explode = Math.max(0.6, S.params.explode - 0.15); rangeInputs.explode.input.value = S.params.explode; relayout(false); }
    else if ((k === "m" || k === "M") && spec.modes && spec.modes.length > 1) { const i = spec.modes.findIndex((m) => m.id === S.mode); setMode(spec.modes[(i + 1) % spec.modes.length].id); }
    else if (spec.onKey && spec.onKey(k, api)) { /* handled by the spec */ }
    else handled = false;
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  };
  container.addEventListener("keydown", onKey);

  // ---- resize
  function resize() {
    const w = container.clientWidth || 400, h = container.clientHeight || 340;
    container.classList.toggle("zv-compact", w < 600);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    viewH = h;
    sizeSprites();
  }
  const ro = new ResizeObserver(() => { resize(); });
  ro.observe(container);
  resize();
  relayoutStatic();
  refreshAll();
  resetCamera(false);

  // ---- loop
  let alive = true, lastT = performance.now();
  const followTmp = new THREE.Vector3();
  const loop = (now) => {
    if (!alive) return;
    const dt = Math.min(0.1, (now - lastT) / 1000);
    lastT = now;
    pulse += dt;
    let dirty = false;
    // layout transition
    if (trans) {
      const t = Math.min(1, (now - trans.t0) / trans.dur), e = easeInOut(t);
      for (let v = 0; v < n; v++) cur[v].lerpVectors(trans.from[v], trans.to[v], e);
      if (t >= 1) { trans = null; relayoutStatic(); } else { buildEdges(); clearGroup(plateGroup); clearGroup(labelGroup); }
      dirty = true;
    } else if (spec.tick && spec.tick(dt, S.params, S.mode, S.spin)) {
      const nl = spec.layout(S.mode, S.params);
      L = nl;
      for (let v = 0; v < n; v++) cur[v].copy(nl.pos[v]);
      buildEdges();
      dirty = true;
      if (rangeInputs.angle) rangeInputs.angle.input.value = S.params.angle;
    }
    // path growth animation
    if (S.drawIdx < S.targetIdx) {
      const t = Math.min(1, (now - S.growT0) / S.growDur);
      S.drawIdx = S.growFrom + (S.targetIdx - S.growFrom) * easeInOut(t);
      if (t >= 1) S.drawIdx = S.targetIdx;
      buildPath();
    }
    if (dirty) { applyCells(); applyBadges(); buildPath(); buildFx(); placeHeat(); }
    else if (S.legal.length || S.path.length) {
      // gentle pulse on legal moves / next checkpoint
      applyCellsPulse();
    }
    for (const c of fxGroup.children) if (c.userData.pulse) c.scale.setScalar(c.userData.base * (1 + 0.1 * Math.sin(pulse * 5)));
    for (const c of pathGroup.children) if (c.userData.glow) c.material.opacity = 0.4 + 0.2 * Math.sin(pulse * 4);
    // camera fly / follow
    if (camAnim) {
      const t = Math.min(1, (now - camAnim.t0) / camAnim.dur), e = easeInOut(t);
      controls.target.lerpVectors(camAnim.fromT, camAnim.toT, e);
      camera.position.lerpVectors(camAnim.fromP, camAnim.toP, e);
      if (t >= 1) camAnim = null;
    } else if (S.follow && S.path.length) {
      const head = cur[S.path[S.path.length - 1]];
      followTmp.subVectors(head, controls.target).multiplyScalar(Math.min(1, dt * 4));
      controls.target.add(followTmp);
      camera.position.add(followTmp);
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  };
  let pulseFrame = 0;
  function applyCellsPulse() {
    if (++pulseFrame % 2) return;           // every other frame is plenty
    if (S.legal.length) applyCells();
    applyBadges();
  }
  requestAnimationFrame(loop);

  // ---- public API
  function update({ path = [], legal = [], hint = null, stuck = null, colorAt } = {}) {
    const prev = S.path;
    if (typeof colorAt === "function") S.colorAt = colorAt;
    const isExt = prev.length && path.length > prev.length && prev.every((v, i) => path[i] === v);
    S.path = path.slice();
    S.legal = legal.slice();
    S.legalSet = new Set(legal);
    S.hint = hint; S.stuck = stuck;
    const target = Math.max(0, path.length - 1);
    if (isExt && path.length - prev.length <= 3) {
      S.growFrom = Math.min(S.drawIdx, prev.length - 1);
      S.targetIdx = target;
      S.growT0 = performance.now();
      S.growDur = 110 + 70 * (path.length - prev.length);
    } else { S.drawIdx = S.targetIdx = target; }
    if (S.peel.mode === "auto") { relayoutStatic(); }
    refreshAll();
    syncStrip();
  }
  function setHeat(map) {
    if (map == null) S.heat = null;
    else {
      const m = new Map();
      const add = (k, x) => { const v = Number(k), h = Number(x); if (Number.isInteger(v) && isFinite(h) && h > 0) m.set(v, Math.min(1, h)); };
      if (map instanceof Map) map.forEach((x, k) => add(k, x));
      else if (Array.isArray(map)) map.forEach((x, k) => (Array.isArray(x) ? add(x[0], x[1]) : add(k, x)));
      else Object.entries(map).forEach(([k, x]) => add(k, x));
      S.heat = m.size ? m : null;
    }
    applyCells(); buildHeat();
  }
  const api = {
    update,
    setHeat,
    refreshTheme() {
      readColors(); tints = groupTints();
      for (const s of badges) s.userData.key = "";
      buildStrip(); relayoutStatic(); refreshAll(); buildHeat();
    },
    projectNode(v) {
      const r = renderer.domElement.getBoundingClientRect();
      const s = screenOf(v, r);
      return { x: s.x, y: s.y };
    },
    // extra controls (used by tests / the editor preview)
    setLayer(k, sub) { if (sub) S.peel.sub = sub; if (k == null || k === "all") setPeel("all"); else if (k === "auto") setPeel("auto"); else setPeel("set", k); },
    setMode, setFollow, setSpin, resetCamera,
    setParam(key, value) { S.params[key] = value; if (rangeInputs[key]) rangeInputs[key].input.value = value; relayout(false); },
    get mode() { return S.mode; },
    get params() { return S.params; },
    orbit(azimuth = 0, polar = 0) {
      const off = camera.position.clone().sub(controls.target);
      const sph = new THREE.Spherical().setFromVector3(off);
      sph.theta += azimuth; sph.phi = Math.max(0.05, Math.min(Math.PI - 0.05, sph.phi + polar));
      camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(sph));
      controls.update();
    },
    zoom(f) { camera.position.sub(controls.target).multiplyScalar(f).add(controls.target); controls.update(); },
    dispose() {
      alive = false; ro.disconnect(); controls.dispose();
      container.removeEventListener("keydown", onKey);
      el.removeEventListener("pointerdown", onDown); el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointermove", onMove); el.removeEventListener("pointerleave", onLeave);
      el.removeEventListener("dblclick", onDbl);
      for (const g of [plateGroup, labelGroup, edgeGroup, fxGroup, heatGroup]) clearGroup(g);
      for (const m of pathMeshes) m.geometry && !m.userData.sharedGeo && m.geometry.dispose();
      for (const s of badges) { s.material.map && s.material.map.dispose(); s.material.dispose(); }
      cellGeo.dispose(); hitGeo.dispose(); headGeo.dispose(); capGeo.dispose();
      cellMat.dispose(); ghostMat.dispose(); tubeMat.dispose(); tubeGhostMat.dispose();
      envTex.dispose(); pmrem.dispose();
      renderer.dispose();
      renderer.domElement.remove(); ui.remove(); bg.remove();
      container.classList.remove("zv-host");
    },
  };
  return api;
}

// ------------------------------------------------------------------ 3D grids
export async function createView3D(container, opts) {
  const { coords, n } = opts;
  const dim = coords[0] ? coords[0].length : 3;
  const get = (c, k) => (k < c.length ? c[k] : 0);
  const zs = [...new Set(coords.map((c) => Math.round(get(c, 2))))].sort((a, b) => a - b);
  const zIndex = new Map(zs.map((z, i) => [z, i]));
  const groupOf = Int32Array.from(coords.map((c) => zIndex.get(Math.round(get(c, 2)))));
  const mins = [0, 1, 2].map((k) => Math.min(...coords.map((c) => get(c, k))));
  const maxs = [0, 1, 2].map((k) => Math.max(...coords.map((c) => get(c, k))));
  const mid = [0, 1, 2].map((k) => (mins[k] + maxs[k]) / 2);
  const unit = (u, v) => {
    const a = coords[u], b = coords[v];
    let s = 0;
    for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
    return Math.abs(s - 1) < 1e-6;
  };
  const spec = {
    axisName: "z",
    groupOf, groupCount: zs.length, groupValues: zs,
    params: { explode: 1 },
    layout(mode, params) {
      const floor = 1.55 * params.explode;
      const pos = coords.map((c) => new THREE.Vector3(get(c, 1) - mid[1], (get(c, 2) - mid[2]) * floor, get(c, 0) - mid[0]));
      const plates = [];
      const labels = [];
      zs.forEach((z, g) => {
        const vs = [];
        for (let v = 0; v < n; v++) if (groupOf[v] === g) vs.push(v);
        const xs = vs.map((v) => pos[v].x), zz = vs.map((v) => pos[v].z);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), z0 = Math.min(...zz), z1 = Math.max(...zz);
        const y = (z - mid[2]) * floor - 0.26;
        plates.push({ g, x: (x0 + x1) / 2, y, z: (z0 + z1) / 2, w: x1 - x0 + 0.95, d: z1 - z0 + 0.95 });
        if (zs.length > 1) labels.push({ text: `z = ${z}`, pos: new THREE.Vector3(x0 - 0.55, y + 0.05, z1 + 0.55), g });
      });
      return { pos, jump: (u, v) => !unit(u, v), lift: new THREE.Vector3(0, 1, 0), plates, labels };
    },
  };
  void dim;
  return createGraphView(container, opts, spec);
}
