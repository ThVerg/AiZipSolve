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
// /api/generate body for a mode difficulty (the game and the show must agree, so "Can you beat it?" hands over the same puzzle).
// `bank` names the curated pool (static/bank/, "<mode>-<diff>"): the server (and the static site) serve pool[seed % count]
// from it, with its human difficulty rating; without a bank the server generates kind/size as before.
export function genParams(p, seed) { return { kind: p.kind, size: p.size, unique: !!p.unique, options: p.options || {}, seed: seed ?? null, time_limit: 20, bank: p.bank || null }; }

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
  A.race = `<div class="race-art">${robots.map((b, i) => `<span class="ra-av" style="--i:${i}">${b.emoji}</span>`).join("")}<span class="ra-vs">VS</span><span class="ra-you">🙂</span></div>`;
  return A;
}
