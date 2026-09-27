// 4D Zip puzzles in three.js. Same interface as view3d.js:
//   createView4D(container, {coords, edges, cps, n, onClick, theme})
//     -> { update, setHeat, refreshTheme, projectNode, dispose, ... }
// coords are (r, c, z, w). Two layouts, switchable (M) with a smooth morph:
//   * "Cubes":     every w-slice is a 3D cube (r, c, z), cubes side by side along the w axis; moves
//                  that change w are drawn as arcs jumping between cubes.
//   * "Tesseract": a rotatable perspective projection 4D -> 3D (nested cubes at angle 0, the classic
//                  tesseract). The w-rotation angle has a slider and can spin (O).
// Slices are colour-tinted; the side strip peels / isolates w-slices ("w = k only").
import * as THREE from "three";
import { createGraphView } from "./view3d.js";

export async function createView4D(container, opts) {
  const { coords, n } = opts;
  const get = (c, k) => (k < c.length ? c[k] : 0);
  const ws = [...new Set(coords.map((c) => Math.round(get(c, 3))))].sort((a, b) => a - b);
  const wIndex = new Map(ws.map((w, i) => [w, i]));
  const groupOf = Int32Array.from(coords.map((c) => wIndex.get(Math.round(get(c, 3)))));
  const mins = [0, 1, 2, 3].map((k) => Math.min(...coords.map((c) => get(c, k))));
  const maxs = [0, 1, 2, 3].map((k) => Math.max(...coords.map((c) => get(c, k))));
  const mid = mins.map((m, k) => (m + maxs[k]) / 2);
  const span = maxs.map((m, k) => m - mins[k]);
  const zs = [...new Set(coords.map((c) => Math.round(get(c, 2))))].sort((a, b) => a - b);
  const diff = (u, v) => {
    const a = coords[u], b = coords[v], d = [0, 0, 0, 0];
    for (let k = 0; k < 4; k++) d[k] = get(b, k) - get(a, k);
    return d;
  };
  const unitAxis = (u, v) => {
    const d = diff(u, v);
    let axis = -1;
    for (let k = 0; k < 4; k++) {
      if (Math.abs(d[k]) < 1e-6) continue;
      if (Math.abs(Math.abs(d[k]) - 1) > 1e-6 || axis >= 0) return -1;
      axis = k;
    }
    return axis;
  };

  function cubes(params) {
    const e = params.explode;
    const floor = 1.3 * e;
    const gap = 1.1 + 1.6 * e;
    const pitch = span[1] + 1 + gap;
    const pos = coords.map((c) => new THREE.Vector3(
      get(c, 1) - mid[1] + (get(c, 3) - mid[3]) * pitch,
      (get(c, 2) - mid[2]) * floor,
      get(c, 0) - mid[0],
    ));
    const plates = [], labels = [];
    ws.forEach((w, g) => {
      const cx = (w - mid[3]) * pitch;
      zs.forEach((z) => {
        plates.push({ g, x: cx, y: (z - mid[2]) * floor - 0.26, z: 0, w: span[1] + 0.95, d: span[0] + 0.95 });
      });
      labels.push({ text: `w = ${w}`, pos: new THREE.Vector3(cx, (maxs[2] - mid[2]) * floor + 0.75, -(span[0] / 2) - 0.4), g });
    });
    return { pos, jump: (u, v) => { const ax = unitAxis(u, v); return ax === 3 || ax < 0; }, lift: new THREE.Vector3(0, 1, 0), plates, labels };
  }

  function tesseract(params) {
    // normalise each axis to [-1, 1], rotate in the x-w and z-w planes, perspective-divide by w
    const s = (k) => (span[k] > 0 ? 2 / span[k] : 0);
    const a = params.angle, b = params.angle * 0.61 + 0.35;
    const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
    const D = 3.2 / Math.max(0.35, params.explode * 0.9);
    const size = 1.7 * (Math.max(span[0], span[1], span[2], 2) + 1.2);
    const scale = new Float32Array(coords.length);
    const pos = coords.map((c, v) => {
      let x = (get(c, 1) - mid[1]) * s(1), y = (get(c, 2) - mid[2]) * s(2), z = (get(c, 0) - mid[0]) * s(0), w = (get(c, 3) - mid[3]) * s(3);
      // x-w rotation
      [x, w] = [x * ca - w * sa, x * sa + w * ca];
      // z-w rotation (slower), plus a small fixed tilt so the nesting reads in 3D
      [z, w] = [z * cb - w * sb, z * sb + w * cb];
      const k = D / (D - w);
      scale[v] = Math.max(0.55, Math.min(1.35, k * 0.95));
      return new THREE.Vector3(x * k, y * k, z * k).multiplyScalar(size / 2.4);
    });
    return { pos, jump: (u, v) => unitAxis(u, v) < 0, lift: new THREE.Vector3(0, 1, 0), plates: [], labels: [], scale };
  }

  const spec = {
    axisName: "w",
    groupOf, groupCount: ws.length, groupValues: ws,
    groupTint: true,
    jumpOpacity: 0.28,
    modes: [
      { id: "cubes", label: "Cubes", title: "Each w-slice as a 3D cube, side by side; w-moves are arcs (M)" },
      { id: "tesseract", label: "Tesseract", title: "Rotatable 4D -> 3D perspective projection (M)" },
    ],
    params: { explode: 1, angle: 0 },
    ranges: [{ key: "angle", label: "w-rotation", min: 0, max: 6.283, step: 0.01, title: "Rotate through the 4th dimension (, / .)", modes: ["tesseract"] }],
    cameraDir: (mode) => (mode === "cubes" ? new THREE.Vector3(0.18, 0.62, 1) : new THREE.Vector3(0.62, 0.55, 0.9)),
    spinParam: (mode) => mode === "tesseract",
    layout(mode, params) { return mode === "tesseract" ? tesseract(params) : cubes(params); },
    tick(dt, params, mode, spin) {
      if (mode !== "tesseract" || !spin) return false;
      params.angle = (params.angle + dt * 0.45) % (Math.PI * 2);
      return true;
    },
    onKey(k, api) {
      if (k === "," || k === "<") { api.setParam("angle", ((api.params?.angle ?? 0) - 0.1 + 6.283) % 6.283); return true; }
      if (k === "." || k === ">") { api.setParam("angle", ((api.params?.angle ?? 0) + 0.1) % 6.283); return true; }
      return false;
    },
    help: "<kbd>M</kbd> cubes / tesseract &nbsp;<kbd>,</kbd> <kbd>.</kbd> rotate through w (tesseract)",
  };
  void n;
  return createGraphView(container, opts, spec);
}
