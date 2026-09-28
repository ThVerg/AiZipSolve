// Shared helpers for the play page (DOM, SVG, formatting, colours, API, safe storage).
export const $ = (id) => document.getElementById(id);
export const SVGNS = "http://www.w3.org/2000/svg";
export const STOP_HEX = ["#ff9f1c", "#ff5a4e", "#d63a8f", "#7b3fd6", "#2f6fe0"];
const STOPS = STOP_HEX.map(hexRgb);
export const AXIS = ["y", "x", "z", "w"];

export function hexRgb(h) { const x = parseInt(h.slice(1), 16); return [(x >> 16) & 255, (x >> 8) & 255, x & 255]; }
export function pathColor(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(Math.floor(t), STOPS.length - 2), f = t - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
export function svgEl(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
export function fmtTime(ms) {
  if (ms == null || !isFinite(ms)) return "0:00";
  const s = Math.floor(ms / 1000);
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  return hh ? `${hh}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${mm}:${String(ss).padStart(2, "0")}`;
}
export function fmtTimeFine(ms) {
  if (ms == null) return "-";
  return ms < 60000 ? `${(ms / 1000).toFixed(1)} s` : fmtTime(ms);
}
export function fmtSec(s) { return s == null ? "-" : s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`; }
export function fmtInt(x) { return x == null ? "-" : Number(x).toLocaleString(); }
export function fmtPct(p, d = 0) { return p == null ? "-" : `${(p * 100).toFixed(d)}%`; }
export function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
export function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Online (static) build: set by scripts/build_site.py. No server there: api() answers from the puzzle bank
// (play/static_api.js) and page links point at the built .html files.
export const STATIC = typeof window !== "undefined" && !!window.ZIP_STATIC;
export const PAGES = STATIC ? { home: "./", lab: "lab.html" } : { home: "/", lab: "/lab" };
if (STATIC && typeof document !== "undefined") {
  const flip = () => {
    document.querySelectorAll("[data-local-only]").forEach((e) => { e.hidden = true; });
    document.querySelectorAll("[data-static-only]").forEach((e) => { e.hidden = false; });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", flip); else flip();
}

export async function api(url, body, { signal } = {}) {
  if (STATIC) return (await import("./static_api.js")).staticApi(url, body);
  const opt = body === undefined ? { signal } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal };
  const r = await fetch(url, opt);
  let j = null;
  try { j = await r.json(); } catch { /* ignore */ }
  if (!r.ok) {
    const err = new Error((j && (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail))) || `${r.status} ${r.statusText}`);
    err.status = r.status;
    throw err;
  }
  return j;
}

// localStorage wrapped: may be unavailable (private mode, blocked storage) - never required.
export const store = {
  get(key, dflt = null) {
    try { const v = localStorage.getItem(key); return v == null ? dflt : JSON.parse(v); } catch { return dflt; }
  },
  set(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch { /* ignore */ } },
  del(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
};

export function localDateISO(d = new Date()) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
export function addDays(iso, k) {
  const [y, m, d] = iso.split("-").map(Number);
  return localDateISO(new Date(y, m - 1, d + k));
}

// small inline icon set (stroke icons, 24x24 viewBox)
const ICONS = {
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
  bulb: '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V17h6v-.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2z"/>',
  reset: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  sound: '<path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a10 10 0 0 1 0 14"/>',
  mute: '<path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="m23 9-6 6M17 9l6 6"/>',
  help: '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  play: '<path d="m6 4 14 8-14 8V4z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  skip: '<path d="m5 4 10 8-10 8V4z"/><path d="M19 5v14"/>',
  bolt: '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
  cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
  flame: '<path d="M12 22c4 0 7-2.7 7-7 0-3-1.6-5.4-3.5-7.5-.4 2-1.6 3.3-3 3.8.5-3.3-1-6.6-3.5-9.3C9 5 5 8.5 5 15c0 4.3 3 7 7 7z"/>',
  trophy: '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z"/><path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3"/>',
  calendar: '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
  share: '<path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="m16 6-4-4-4 4M12 2v13"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  cube: '<path d="M21 16V8l-9-5-9 5v8l9 5 9-5z"/><path d="m3.3 7 8.7 5 8.7-5M12 22V12"/>',
  flag: '<path d="M4 22V4a1 1 0 0 1 1-1h13l-2 5 2 5H5"/>',
  brain: '<path d="M9.5 2A2.5 2.5 0 0 0 7 4.5v.3A3 3 0 0 0 4.5 8 3 3 0 0 0 3 10.6 3 3 0 0 0 4 15a3.5 3.5 0 0 0 5.5 4.3V2z"/><path d="M14.5 2A2.5 2.5 0 0 1 17 4.5v.3A3 3 0 0 1 19.5 8a3 3 0 0 1 1.5 2.6A3 3 0 0 1 20 15a3.5 3.5 0 0 1-5.5 4.3V2z"/>',
  keyboard: '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>',
  race: '<path d="M13 4v16M17 4v16M3 8h18M3 16h18"/>',
  layers: '<path d="m12 2 10 5-10 5L2 7l10-5z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
};
export function icon(name, cls = "") {
  return `<svg class="ic ${cls}" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;
}
