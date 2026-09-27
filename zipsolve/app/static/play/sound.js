// Tiny WebAudio synth: no audio files. Default on but quiet; mute persisted (per viewer).
import { store } from "./util.js";

let ctx = null, master = null;
let muted = store.get("zip-muted", false);
const VOL = 0.14;
const PENTA = [0, 2, 4, 7, 9];

function ensure() {
  if (muted) return null;
  try {
    if (!ctx) {
      // browsers refuse audio before the first user gesture (and warn about it): stay silent until then
      if (navigator.userActivation && !navigator.userActivation.hasBeenActive) return null;
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
      master = ctx.createGain();
      master.gain.value = VOL;
      master.connect(ctx.destination);
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  } catch { return null; }
}
function tone(freq, { t = 0, dur = 0.09, type = "sine", gain = 0.6, attack = 0.004, slide = 0 } = {}) {
  const c = ensure();
  if (!c) return;
  const t0 = c.currentTime + t;
  const o = c.createOscillator(), g = c.createGain();
  o.type = type;
  o.frequency.setValueAtTime(freq, t0);
  if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(30, freq * slide), t0 + dur);
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(gain, t0 + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(master);
  o.start(t0);
  o.stop(t0 + dur + 0.02);
}
const midi = (m) => 440 * Math.pow(2, (m - 69) / 12);
function scaleNote(i, base = 67) { return midi(base + 12 * Math.floor(i / 5) + PENTA[((i % 5) + 5) % 5]); }

let lastTick = 0;
export const sound = {
  get muted() { return muted; },
  setMuted(m) { muted = !!m; store.set("zip-muted", muted); if (!muted) ensure(); },
  // a move: pitch climbs with progress (0..1) along a pentatonic scale
  move(progress) {
    const now = performance.now();
    if (now - lastTick < 28) return;   // fast drags: don't machine-gun
    lastTick = now;
    tone(scaleNote(Math.round(progress * 9)), { dur: 0.07, type: "triangle", gain: 0.35 });
  },
  undo() { tone(midi(60), { dur: 0.06, type: "sine", gain: 0.25, slide: 0.8 }); },
  checkpoint(k) {
    tone(scaleNote(k + 3, 72), { dur: 0.16, type: "sine", gain: 0.5 });
    tone(scaleNote(k + 5, 72), { t: 0.07, dur: 0.2, type: "sine", gain: 0.4 });
  },
  bad() { tone(150, { dur: 0.12, type: "square", gain: 0.12, slide: 0.7 }); },
  hint() { tone(midi(76), { dur: 0.12, gain: 0.3 }); tone(midi(83), { t: 0.08, dur: 0.16, gain: 0.25 }); },
  win() { [72, 76, 79, 84, 88].forEach((m, i) => tone(midi(m), { t: i * 0.085, dur: 0.32, type: "triangle", gain: 0.45 })); },
  lose() { [67, 63, 60].forEach((m, i) => tone(midi(m), { t: i * 0.14, dur: 0.25, type: "sine", gain: 0.35 })); },
  count(final = false) { tone(midi(final ? 84 : 72), { dur: final ? 0.3 : 0.12, type: "triangle", gain: 0.45 }); },
  search() { tone(midi(88 + Math.floor(Math.random() * 5)), { dur: 0.03, gain: 0.08 }); },
  // ---- casual game extras (index.html): a little per-puzzle melody, level-ups, robot blips
  // step i of the path plays note i of a motif picked by `seed`; a fast "flow" (combo) lifts it an octave
  note(i, { seed = 0, combo = 0 } = {}) {
    const now = performance.now();
    if (now - lastTick < 24) return;
    lastTick = now;
    const m = MOTIFS[Math.abs(seed) % MOTIFS.length];
    const phrase = Math.floor(i / m.length) % 4;
    const deg = m[i % m.length] + [0, 2, 1, 3][phrase] + (combo >= 6 ? 5 : 0);
    const f = scaleNote(deg, 67);
    tone(f, { dur: 0.11, type: "triangle", gain: 0.32 });
    tone(f * 2, { dur: 0.07, type: "sine", gain: 0.07 });
  },
  levelUp(k) {
    [0, 2, 4].forEach((d, j) => tone(scaleNote(k + d, 74), { t: j * 0.055, dur: 0.18, type: "triangle", gain: 0.34 }));
  },
  robot() { tone(midi(79), { dur: 0.05, type: "square", gain: 0.06 }); tone(midi(86), { t: 0.06, dur: 0.05, type: "square", gain: 0.05 }); },
  whoosh() { tone(220, { dur: 0.5, type: "sine", gain: 0.12, slide: 4, attack: 0.08 }); },
};
const MOTIFS = [[0, 2, 4, 3, 5, 4, 2, 3], [0, 4, 2, 5, 3, 6, 4, 7], [4, 2, 0, 2, 4, 4, 5, 3], [0, 1, 2, 4, 2, 5, 4, 6], [2, 4, 5, 4, 2, 0, 1, 3]];
