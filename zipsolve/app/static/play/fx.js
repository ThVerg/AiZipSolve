// Canvas particle effects for the casual game: head sparks, checkpoint bursts, confetti.
// One fixed, full-viewport canvas; the animation loop only runs while particles are alive.
import { reducedMotion } from "./util.js";

export function createFx(canvas) {
  const ctx = canvas.getContext("2d");
  let parts = [], raf = 0, W = 0, H = 0, dpr = 1;
  function size() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = innerWidth; H = innerHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  size();
  addEventListener("resize", size);

  function loop() {
    ctx.clearRect(0, 0, W, H);
    const next = [];
    for (const p of parts) {
      p.life -= 1;
      if (p.life <= 0) continue;
      p.vy += p.g; p.vx *= p.drag; p.vy *= p.drag; p.x += p.vx; p.y += p.vy; p.a += p.va;
      const t = p.life / p.max;
      ctx.save();
      ctx.globalAlpha = Math.min(1, t * 1.6);
      ctx.translate(p.x, p.y);
      if (p.kind === "conf") {
        ctx.rotate(p.a);
        ctx.fillStyle = p.c;
        ctx.fillRect(-p.s / 2, -p.s / 4, p.s, p.s / 2 * (0.4 + Math.abs(Math.sin(p.a * 2))));
      } else if (p.kind === "ring") {
        ctx.strokeStyle = p.c; ctx.lineWidth = 3 * t;
        ctx.beginPath(); ctx.arc(0, 0, p.s * (1 - t) * 1.6 + 4, 0, Math.PI * 2); ctx.stroke();
      } else {
        const r = p.s * (0.35 + 0.65 * t);
        ctx.globalCompositeOperation = "lighter";
        ctx.fillStyle = p.c;
        ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha *= 0.2;
        ctx.beginPath(); ctx.arc(0, 0, r * 1.9, 0, Math.PI * 2); ctx.fill();
      }
      ctx.restore();
      next.push(p);
    }
    parts = next;
    raf = parts.length ? requestAnimationFrame(loop) : 0;
  }
  function kick() { if (!raf) raf = requestAnimationFrame(loop); }
  function add(p) { parts.push(p); if (parts.length > 900) parts.splice(0, parts.length - 900); }

  return {
    // little sparks where the line head is
    sparks(x, y, color, n = 5, power = 1) {
      if (reducedMotion()) return;
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2, v = (0.5 + Math.random() * 1.8) * power;
        add({ kind: "spark", x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 0.3, g: 0.04, drag: 0.93, a: 0, va: 0,
          s: 1.1 + Math.random() * 1.5 * power, c: color, life: 14 + Math.random() * 12, max: 26 });
      }
      kick();
    },
    // checkpoint "level up": a ring + a fountain of sparks
    burst(x, y, color, n = 26) {
      if (reducedMotion()) return;
      add({ kind: "ring", x, y, vx: 0, vy: 0, g: 0, drag: 1, a: 0, va: 0, s: 34, c: color, life: 26, max: 26 });
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 + Math.random() * 0.2, v = 2.5 + Math.random() * 3.5;
        add({ kind: "spark", x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 1, g: 0.09, drag: 0.95, a: 0, va: 0,
          s: 1.5 + Math.random() * 1.8, c: color, life: 24 + Math.random() * 16, max: 40 });
      }
      kick();
    },
    confetti(rect, count = 180) {
      if (reducedMotion()) return;
      const cols = ["#ff9f1c", "#ff5a4e", "#d63a8f", "#7b3fd6", "#2f6fe0", "#20b27a", "#ffd23f"];
      const cx = rect ? rect.left + rect.width / 2 : W / 2, cy = rect ? rect.top + rect.height * 0.45 : H / 3;
      const spread = rect ? rect.width * 0.7 : W * 0.4;
      for (let i = 0; i < count; i++) {
        add({ kind: "conf", x: cx + (Math.random() - 0.5) * spread, y: cy, vx: (Math.random() - 0.5) * 15, vy: -Math.random() * 14 - 5,
          g: 0.33, drag: 0.985, a: Math.random() * 6, va: (Math.random() - 0.5) * 0.35, s: Math.random() * 7 + 5,
          c: cols[i % cols.length], life: 150 + Math.random() * 60, max: 210 });
      }
      kick();
    },
    clear() { parts = []; },
  };
}
