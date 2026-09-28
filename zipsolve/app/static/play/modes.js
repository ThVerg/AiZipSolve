// Shared content for the casual pages (the game at / and the AI show at /lab): puzzle modes, the robot
// characters and the little animated mode-card illustrations. Same seed + same mode = same puzzle on both pages.
import { outlineLoops, offsetLoop, roundedPath } from "./board.js";

export const MODES = {
  classic: { label: "Classic", dflt: "medium", diffs: {
    easy: { label: "Easy", kind: "grid2d", size: 5, unique: true, bank: "classic-easy" },
    medium: { label: "Medium", kind: "grid2d", size: 7, unique: true, bank: "classic-medium" },
    hard: { label: "Hard", kind: "grid2d", size: 9, unique: true, bank: "classic-hard" } } },
  walls: { label: "Walls", dflt: "normal", diffs: { normal: { label: "Walls", kind: "walls", size: 7, unique: true, options: { walls_frac: 0.35 }, bank: "walls-medium" } } },
  islands: { label: "Islands", dflt: "normal", diffs: { normal: { label: "Islands", kind: "islands", size: 4, unique: true, bank: "islands-medium" } } },
  cube: { label: "3D Cube", dflt: "normal", diffs: { normal: { label: "4×4×4", kind: "grid3d", size: 4, unique: false, bank: "cube-hard" } } },
};
// "More modes": the new puzzle types (contract kinds). `emoji` + `sub` show on the card and in the game title;
// `tip` is the one-line intro the first time you play one. Fog reuses the classic pool and switches meta.fog on.
export const MORE = {
  portals: { label: "Portals", emoji: "🌀", sub: "Step in, pop out", tip: "🌀 Step into a ring, pop out of its twin", kind: "portals", size: 6 },
  torus: { label: "Wraparound", emoji: "🍩", sub: "Edges connect", tip: "🍩 Off one edge, back on the other", kind: "torus", size: 5 },
  hex: { label: "Hex", emoji: "⬡", sub: "Six ways to go", tip: "⬡ Hexagons: six neighbours each", kind: "hex", size: 4 },
  tri: { label: "Triangles", emoji: "🔺", sub: "Zig and zag", tip: "🔺 Triangles: cross through the sides", kind: "tri", size: 3 },
  oneway: { label: "One-way", emoji: "➡️", sub: "Follow the arrows", tip: "➡️ Arrows only let you cross one way", kind: "oneway", size: 6 },
  overpass: { label: "Overpass", emoji: "🌉", sub: "Cross twice", tip: "🌉 Cross each bridge twice: over and under", kind: "overpass", size: 6 },
  keys: { label: "Keys & doors", emoji: "🔑", sub: "Key first", tip: "🔑 Grab the key before its door", kind: "keys", size: 6 },
  cubesurf: { label: "Cube", emoji: "🎲", sub: "Wrap the box", tip: "🎲 Draw around the cube · drag outside to spin", kind: "cubesurf", size: 3 },
  fog: { label: "Fog", emoji: "🌫️", sub: "Numbers hide", tip: "🌫️ Numbers show up when you get close", kind: "grid2d", size: 6, fog: true, bank: "classic-medium" },
  coop: { label: "Co-op", emoji: "👯", sub: "Two players", tip: "👯 🟠 and 🔵 fill the board together", kind: "coop", size: 6 },
};
for (const [id, m] of Object.entries(MORE)) {
  MODES[id] = { label: m.label, emoji: m.emoji, more: true, dflt: "normal",
    diffs: { normal: { label: m.label, kind: m.kind, size: m.size, unique: true, options: m.fog ? {} : {}, bank: m.bank || `${id}-medium`, fog: !!m.fog } } };
}
// which bank pool / kind each mode needs (for hiding modes the server or the static site can't serve yet)
export function modePool(id) { const b = MODES[id].diffs[MODES[id].dflt].bank; return b ? b.split("-") : null; }
// /api/generate body for a mode difficulty (the game and the show must agree, so "Can you beat it?" hands over the same puzzle).
// `bank` names the curated pool (static/bank/, "<mode>-<diff>"): the server (and the static site) serve pool[seed % count]
// from it, with its human difficulty rating; without a bank the server generates kind/size as before.
export function genParams(p, seed) { return { kind: p.kind, size: p.size, unique: !!p.unique, options: p.options || {}, seed: seed ?? null, time_limit: 20, bank: p.bank || null, ...(p.fog ? { fog: true } : {}) }; }

// the robots of "Race the robot" (game) and the AI show
export const ROBOTS = [
  { id: "rookie", name: "Rookie", emoji: "🐣", line: "Fast but reckless", mode: "greedy", pace: 430,
    say: { go: ["zoom!", "wheee!", "easy peasy", "go go go"], win: "Done! 🎉", lose: "no fair! 😤", stuck: "uh-oh… stuck 😵" } },
  { id: "scout", name: "Scout", emoji: "🦊", line: "Careful. Backs up when stuck", mode: "search", pace: 560,
    say: { go: ["sniff sniff", "this way…", "hmm, yes", "tracking…"], win: "Found it! 🦊", lose: "well played!", stuck: "lost the trail…", back: "backing up…" } },
  { id: "grandmaster", name: "Grandmaster", emoji: "🦉", line: "Never loses. Just slow", mode: "hybrid", pace: 980,
    say: { go: ["indeed.", "as foreseen", "patience…", "hoo hoo"], win: "Inevitable. 🦉", lose: "…impressive.", stuck: "curious…" } },
];

// ------------------------------------------------------------------ mode card art (SVG / CSS 3D, animated by game.css)
export function modeArt(robots = ROBOTS) {
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
    const isl = (cs, ox, oy, land) => {
      const loops = outlineLoops(cs);
      const sh = (off, rad) => loops.map((lp) => roundedPath(offsetLoop(lp, off).map(([r, c]) => [ox + c * S, oy + r * S]), rad * S)).join(" ");
      return `<path d="${sh(0.2, 0.3)}" fill="var(--water-deep)" opacity=".25" transform="translate(0 1.4)"/>
        <path d="${sh(0.34, 0.5)}" fill="none" stroke="var(--foam)" stroke-width="1" opacity=".55"/>
        <path d="${sh(0.2, 0.3)}" fill="var(--sand)" stroke="var(--sand-2)" stroke-width=".6"/><path d="${sh(0.03, 0.26)}" fill="var(${land})"/>
        ${cs.map(([r, c]) => `<rect x="${ox + c * S + 0.8}" y="${oy + r * S + 0.8}" width="${S - 1.6}" height="${S - 1.6}" rx="3" fill="var(--tile)" stroke="var(--tile-edge)" stroke-width=".5"/>`).join("")}`;
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
  // ---- more modes: tiny looping previews (one mechanic each)
  const G = (id, a = "#ff9f1c", b = "#d63a8f") => `<defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${a}"/><stop offset="1" stop-color="${b}"/></linearGradient></defs>`;
  const ring = (x, y, c, d = 0) => `<g class="a-portal" style="color:${c};animation-delay:${d}s" transform="translate(${x} ${y})"><circle r="10" fill="currentColor" opacity=".18"/><circle class="a-spin" r="9" fill="none" stroke="currentColor" stroke-width="2.6" stroke-dasharray="11 3.5" stroke-linecap="round"/></g>`;
  A.portals = `<svg viewBox="0 0 100 100">${G("mp1")}${cells(4, 6, 6, 22, 1.6)}
    ${ring(17, 83, "#19b8ff")}${ring(83, 17, "#19b8ff")}
    <path class="a-path a-p1" stroke="url(#mp1)" d="M17 17 H39 V83 H17" pathLength="1"/>
    <path class="a-path a-p2" stroke="url(#mp1)" d="M83 17 V83 H61 V17" pathLength="1"/>
    <circle class="a-zap" cx="83" cy="17" r="10" fill="none" stroke="#19b8ff" stroke-width="3"/>
    ${dot(17, 17, 1)}${dot(61, 17, 2)}</svg>`;
  A.torus = `<svg viewBox="-8 0 116 100">${G("mp2")}
    <g opacity=".35">${[17, 39, 61, 83].map((y) => `<rect class="a-cell" x="-6" y="${y - 10}" width="9" height="20" rx="4"/><rect class="a-cell" x="97" y="${y - 10}" width="9" height="20" rx="4"/>`).join("")}</g>
    ${cells(4, 6, 6, 22, 1.6)}
    <rect x="6" y="6" width="88" height="88" rx="8" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-dasharray="5 4" class="a-flow" opacity=".7"/>
    <path class="a-path a-p1" stroke="url(#mp2)" d="M39 39 H104" pathLength="1"/>
    <path class="a-path a-p2" stroke="url(#mp2)" d="M-4 39 H17 V61 H83" pathLength="1"/>
    ${dot(39, 39, 1)}${dot(83, 61, 2)}</svg>`;
  A.hex = (() => {
    const r = 12.5, w = r * Math.sqrt(3), hx = (q, rr) => [50 + w * (q + rr / 2), 50 + 1.5 * r * rr];
    const cellsH = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1], [2, -1], [-2, 1], [1, 1], [-1, -1], [2, -2], [-2, 2], [1, -2], [-1, 2], [2, 0], [-2, 0], [0, 2], [0, -2]];
    const hexP = (x, y) => Array.from({ length: 6 }, (_, i) => { const a = Math.PI / 6 + (i * Math.PI) / 3; return `${(x + (r - 1.3) * Math.cos(a)).toFixed(1)},${(y + (r - 1.3) * Math.sin(a)).toFixed(1)}`; }).join(" ");
    const route = [[-1, -1], [0, -1], [1, -1], [1, 0], [0, 0], [-1, 0], [-1, 1], [0, 1], [1, 1]].map(([q, rr]) => hx(q, rr));
    const [a, b] = [route[0], route[route.length - 1]];
    return `<svg viewBox="5 3 90 94">${G("mp3")}${cellsH.map(([q, rr]) => { const [x, y] = hx(q, rr); return `<polygon class="a-cell" points="${hexP(x, y)}"/>`; }).join("")}
      <path class="a-path" stroke="url(#mp3)" stroke-width="7" d="M${route.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L")}" pathLength="1"/>
      ${dot(a[0], a[1], 1, 6.5)}${dot(b[0], b[1], 2, 6.5)}</svg>`;
  })();
  A.tri = (() => {
    const a = 22, hh = a * 0.866, out = [], cen = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 7; c++) {
      const up = (r + c) % 2 === 0, x = 6 + c * a / 2 + a / 2, top = 18 + r * hh;
      const pts = up ? [[x, top + 1.8], [x - a / 2 + 2, top + hh - 1], [x + a / 2 - 2, top + hh - 1]] : [[x - a / 2 + 2, top + 1], [x + a / 2 - 2, top + 1], [x, top + hh - 1.8]];
      out.push(`<polygon class="a-cell" points="${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ")}"/>`);
      cen.push([x, top + (up ? 2 * hh / 3 : hh / 3)]);
    }
    const route = [0, 1, 2, 3, 4, 5, 6, 13, 12, 11, 10, 9, 8, 7].map((i) => cen[i]);
    return `<svg viewBox="4 10 92 64">${G("mp4")}${out.join("")}
      <path class="a-path" stroke="url(#mp4)" stroke-width="5.5" d="M${route.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L")}" pathLength="1"/>
      ${dot(cen[0][0], cen[0][1], 1, 5.5)}${dot(cen[7][0], cen[7][1], 2, 5.5)}</svg>`;
  })();
  const chev = (x, y, ang) => `<g transform="translate(${x} ${y}) rotate(${ang})" class="a-chev"><circle r="5.5" fill="var(--ink)"/><path d="M-1.8 -2.8 L1.6 0 L-1.8 2.8" stroke="var(--panel)" stroke-width="1.9" fill="none" stroke-linecap="round" stroke-linejoin="round"/></g>`;
  A.oneway = `<svg viewBox="0 0 100 100">${G("mp5")}${cells(4, 6, 6, 22, 1.6)}
    <path class="a-path" stroke="url(#mp5)" d="M17 17 H83 V39 H17 V61 H83" pathLength="1"/>
    ${chev(50, 17, 0)}${chev(83, 28, 90)}${chev(50, 39, 180)}${chev(17, 50, 90)}${chev(61, 83, 180)}
    ${dot(17, 17, 1)}${dot(83, 61, 2)}</svg>`;
  A.overpass = `<svg viewBox="0 0 100 100">${G("mp6")}${G("mp6b", "#d63a8f", "#6d4ae0")}${cells(3, 17, 17, 22, 1.6)}
    <path class="a-path a-p1" stroke="url(#mp6)" d="M28 50 H72" pathLength="1"/>
    <rect x="39.5" y="17" width="21" height="66" rx="5" fill="#000" opacity=".18" transform="translate(2 3)"/>
    <rect x="39.5" y="17" width="21" height="66" rx="5" fill="var(--panel)" stroke="var(--border)"/>
    <path d="M42.5 20 V80 M57.5 20 V80" stroke="var(--ink-3)" stroke-width="1.2" opacity=".8"/>
    <path class="a-path a-p2" stroke="url(#mp6b)" stroke-width="7" d="M50 20 V80" pathLength="1"/>
    ${dot(28, 50, 1)}${dot(50, 80, 2)}</svg>`;
  A.keys = `<svg viewBox="0 0 100 100">${G("mp7")}${cells(4, 6, 6, 22, 1.6)}
    <rect class="a-door" x="63" y="63" width="19.5" height="19.5" rx="5" fill="#f5b700" opacity=".3"/>
    <path class="a-path" stroke="url(#mp7)" d="M17 17 H83 V39 H17 V61 V83 H72" pathLength="1"/>
    <g class="a-key" transform="translate(50 39)"><circle r="8" fill="#f5b700" stroke="#fff" stroke-width="1.5"/><g transform="rotate(-45)" stroke="#fff" stroke-width="1.6" fill="none" stroke-linecap="round"><circle cx="-2.6" r="1.9"/><path d="M-.8 0 H4 M2.4 0 V1.8"/></g></g>
    <g class="a-lock" transform="translate(72.7 72.7)"><path class="a-shackle" d="M-3 0 V-4.5 a3 3 0 0 1 6 0 V0" stroke="#f5b700" stroke-width="2" fill="none" stroke-linecap="round"/><rect x="-5.5" y="-1.5" width="11" height="8.5" rx="2" fill="#f5b700" stroke="#fff" stroke-width="1"/></g>
    ${dot(17, 17, 1)}</svg>`;
  A.cubesurf = `<div class="cube-scene"><div class="cube3 wrapcube">${face("f1", "M10 10 H50 V50 H10 V30 H60")}${face("f2", "M0 30 H30 V10 H50 V50 H60")}${face("f3", "M0 50 H10 V10 H50 V50")}${face("f4", "M60 50 H10 V30")}${face("f5")}${face("f6")}</div></div>`;
  const cloud = (x, y, d, cls = "") => `<g transform="translate(${x} ${y})"><g class="a-cloud ${cls}" style="animation-delay:${d}s"><circle cx="-5.5" cy="1.5" r="5.5"/><circle cx="5.5" cy="1.5" r="5.5"/><circle cx="-1" cy="-3" r="7"/><circle cx="3" cy="3" r="6"/><text y="4.5" text-anchor="middle">?</text></g></g>`;
  A.fog = `<svg viewBox="0 0 100 100">${G("mp8")}${cells(4, 6, 6, 22, 1.6)}
    <path class="a-path" stroke="url(#mp8)" d="M17 17 H83 V39 H17 V61 H83" pathLength="1"/>
    ${dot(17, 17, 1)}<g class="a-reveal">${dot(17, 39, 2)}</g>${cloud(17, 39, 0, "a-poof")}${cloud(83, 61, -1.2)}${cloud(39, 83, -2.1)}</svg>`;
  A.coop = `<svg viewBox="0 0 100 100">${G("mp9", "#ffb000", "#d63a8f")}${G("mp9b", "#1fb7a6", "#6d4ae0")}${cells(4, 6, 6, 22, 1.6)}
    <path class="a-path" stroke="url(#mp9)" d="M17 17 H83 V39 H17" pathLength="1"/>
    <path class="a-path" stroke="url(#mp9b)" d="M17 83 H83 V61 H17" pathLength="1"/>
    <g class="a-dot"><circle cx="17" cy="17" r="8" style="fill:#f26a1b"/></g><g class="a-dot"><circle cx="17" cy="83" r="8" style="fill:#2466d8"/></g></svg>`;
  A.race = `<div class="race-art">${robots.map((b, i) => `<span class="ra-av" style="--i:${i}">${b.emoji}</span>`).join("")}<span class="ra-vs">VS</span><span class="ra-you">🙂</span></div>`;
  return A;
}
