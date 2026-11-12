/*
 * Time-series lessons: a time-dependent signal, autoregressive lags and an interactive train/test split.
 * Mount point: [data-temporal-lesson="signal" | "autoregression" | "split"].
 * Requires spatial.js first (browser global SpatialLesson, or require("./spatial.js") in Node):
 * the player, the controls, the keyboard handling and the colors all come from there.
 * Browser: global TemporalLesson (auto-mounts). Node: CommonJS exports for tests.
 * All numbers drawn in the figures come from the same pure helpers the tests use.
 */
(function (root, factory) {
  const spatial =
    root && root.SpatialLesson ? root.SpatialLesson : typeof require === "function" ? require("./spatial.js") : null;
  if (!spatial) throw new Error("TemporalLesson needs spatial.js loaded first");
  const api = factory(spatial);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    root.TemporalLesson = api;
    const start = () => api.autoMount(root.document);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})(typeof window !== "undefined" ? window : null, function (S) {
  "use strict";

  const { COLORS, HOUSE } = S;
  // Roles shared with the validation lessons: train blue, validation amber, final test rose.
  const ROLE_COLORS = Object.freeze({
    train: COLORS.cyan,
    validation: "#B7791F",
    test: COLORS.rose,
    prediction: COLORS.green,
    observed: COLORS.ink,
    pending: "#A3ACB8",
    reference: "#8C96A3",
  });

  // ---------- Canonical synthetic signal (feature doc L4) ----------
  // y(t) = trend + 20-step cycle + seeded noise, t = 0..99. Synthetic: not produced by the AR model below.

  const SIGNAL_SEED = 7;
  const SIGNAL_LENGTH = 100;
  const PERIOD = 20;
  const NOISE_AMPLITUDE = 0.6;
  const AR_COEFFS = Object.freeze([0.6, 0.25, 0.1]); // newest lag first; illustrative, not trained
  const SPLIT_SEED = 7;
  const DEFAULT_SPLIT = Object.freeze({ mode: "random", testFraction: 0.2, seed: SPLIT_SEED });

  const round2 = (v) => Math.round(v * 100) / 100;
  const trendAt = (t) => 10 + 0.04 * t;
  const seasonalAt = (t) => 2 * Math.sin((2 * Math.PI * t) / PERIOD);

  function assertSeed(seed) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new RangeError("seed must be an integer between 0 and 2^32 - 1");
    }
  }

  // Small deterministic PRNG (mulberry32): same seed, same sequence in [0, 1).
  function mulberry32(seed) {
    assertSeed(seed);
    let a = seed >>> 0;
    return function next() {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Fisher–Yates on a copy; the input array is never modified.
  function seededShuffle(items, seed) {
    const rand = mulberry32(seed);
    const out = Array.from(items);
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rand() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  // Frozen array of frozen { t, y } points; y is rounded to the two decimals the figures print.
  function generateSignal(seed, length) {
    const n = length == null ? SIGNAL_LENGTH : length;
    const rand = mulberry32(seed == null ? SIGNAL_SEED : seed);
    return Object.freeze(
      Array.from({ length: n }, (_, t) => {
        const noise = (rand() * 2 - 1) * NOISE_AMPLITUDE;
        return Object.freeze({ t, y: round2(trendAt(t) + seasonalAt(t) + noise) });
      })
    );
  }

  const SIGNAL = generateSignal();
  // Plain number array indexed by t (values[t] = y(t)); a fresh copy every call.
  const signalValues = (signal) => (signal || SIGNAL).map((p) => p.y);

  // ---------- Train/test partition over time indices ----------

  /*
   * Returns { mode, testFraction, seed, train, test } with sorted time indices 0..n-1.
   * random: a seeded shuffle picks round(n * testFraction) test indices anywhere on the axis.
   * chronological: the test set is the last contiguous block; the seed plays no role (seed: null).
   * Both sets are always disjoint and together cover the whole sample.
   */
  function partition(options) {
    const o = Object.assign({}, DEFAULT_SPLIT, options);
    const n = o.n == null ? SIGNAL_LENGTH : o.n;
    if (o.mode !== "random" && o.mode !== "chronological") {
      throw new RangeError('mode must be "random" or "chronological"');
    }
    const testCount = Math.round(n * o.testFraction);
    if (!Number.isFinite(o.testFraction) || testCount < 1 || testCount > n - 1) {
      throw new RangeError("testFraction must leave at least one train and one test point");
    }
    assertSeed(o.seed);
    const indices = Array.from({ length: n }, (_, t) => t);
    const byTime = (a, b) => a - b;
    if (o.mode === "chronological") {
      return {
        mode: o.mode,
        testFraction: o.testFraction,
        seed: null,
        train: indices.slice(0, n - testCount),
        test: indices.slice(n - testCount),
      };
    }
    const shuffled = seededShuffle(indices, o.seed);
    return {
      mode: o.mode,
      testFraction: o.testFraction,
      seed: o.seed,
      train: shuffled.slice(testCount).sort(byTime),
      test: shuffled.slice(0, testCount).sort(byTime),
    };
  }

  // ---------- Autoregression: y(t) from strictly earlier values ----------

  function assertValues(values) {
    if (!Array.isArray(values) || values.length === 0 || !values.every(Number.isFinite)) {
      throw new TypeError("values must be a non-empty array of finite numbers");
    }
  }

  // ŷ(t) = Σ w_k · y(t − k), k = 1..p. Reads only values[target − p .. target − 1] for the forecast.
  function arForecast(values, target, coeffs) {
    const w = coeffs || AR_COEFFS;
    assertValues(values);
    const p = w.length;
    if (!Number.isInteger(target) || target < p || target > values.length - 1) {
      throw new RangeError(`target must be an integer between ${p} and ${values.length - 1}`);
    }
    const lags = w.map((weight, k) => {
      const t = target - 1 - k;
      return { lag: k + 1, t, y: values[t], weight, product: weight * values[t] };
    });
    const prediction = lags.reduce((s, l) => s + l.product, 0);
    const observed = values[target];
    return { target, lags, prediction, observed, error: observed - prediction };
  }

  // One chronological supervised row per target: lag features (newest first) and the target value.
  function lagTable(values, coeffs) {
    const p = (coeffs || AR_COEFFS).length;
    assertValues(values);
    return values.slice(p).map((y, i) => {
      const target = i + p;
      const featureTimes = Array.from({ length: p }, (_, k) => target - 1 - k);
      return { target, featureTimes, features: featureTimes.map((t) => values[t]), y };
    });
  }

  // [0,1,2,5,6] → [[0,1,2],[5,6]]: lines are drawn only inside runs, never across missing points.
  function contiguousRuns(indices) {
    const runs = [];
    indices.forEach((t, i) => {
      if (i > 0 && t === indices[i - 1] + 1) runs[runs.length - 1].push(t);
      else runs.push([t]);
    });
    return runs;
  }

  // ---------- SVG primitives (reusable by the validation lessons) ----------

  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
  const r1 = (v) => Math.round(v * 10) / 10;
  const fmt = (v) => v.toFixed(2);

  function text(x, y, str, attrs) {
    const a = Object.assign({ size: 13, fill: COLORS.ink, anchor: "start", weight: "normal" }, attrs);
    return (
      `<text x="${x}" y="${y}" font-size="${a.size}" fill="${a.fill}" text-anchor="${a.anchor}"` +
      ` font-weight="${a.weight}">${esc(str)}</text>`
    );
  }

  function line(x1, y1, x2, y2, color, width, extra) {
    return (
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}"` +
      ` stroke-width="${width || 1.5}"${extra ? ` ${extra}` : ""}/>`
    );
  }

  function arrow(x1, y1, x2, y2, color, width) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = (da) => `${r1(x2 - 9 * Math.cos(angle + da))},${r1(y2 - 9 * Math.sin(angle + da))}`;
    return (
      line(r1(x1), r1(y1), r1(x2 - 6 * Math.cos(angle)), r1(y2 - 6 * Math.sin(angle)), color, width || 1.8) +
      `<polygon points="${r1(x2)},${r1(y2)} ${head(0.45)} ${head(-0.45)}" fill="${color}"/>`
    );
  }

  function svg(title, body) {
    return (
      `<svg class="spatial-svg temporal-svg" viewBox="0 0 600 440" xmlns="http://www.w3.org/2000/svg"` +
      ` role="img" aria-label="${esc(title)}" preserveAspectRatio="xMidYMid meet">` +
      `<rect width="600" height="440" rx="12" fill="${COLORS.bg}"/>${body}</svg>`
    );
  }

  const Y_DOMAIN = Object.freeze([
    Math.floor(Math.min(...signalValues())),
    Math.ceil(Math.max(...signalValues())),
  ]);

  // Maps time t and value y into a plot box { x, y, width, height }. Same axis for every subset.
  function timeScale(o) {
    const box = Object.assign({ tMin: 0, tMax: SIGNAL_LENGTH - 1, yMin: Y_DOMAIN[0], yMax: Y_DOMAIN[1] }, o);
    const clampY = (v) => Math.min(box.yMax, Math.max(box.yMin, v));
    return Object.assign(box, {
      sx: (t) => r1(box.x + ((t - box.tMin) / (box.tMax - box.tMin)) * box.width),
      sy: (v) => r1(box.y + box.height - ((clampY(v) - box.yMin) / (box.yMax - box.yMin)) * box.height),
    });
  }

  // Time axis with sparse tick labels; optional value ticks on the left.
  function axes(sc, o) {
    const opts = Object.assign({ ticks: [0, 20, 40, 60, 80, 99], yTicks: [], size: 11, label: "t" }, o);
    const base = sc.y + sc.height;
    let b = `<g data-role="time-axis">`;
    b += line(sc.x, base, sc.x + sc.width, base, COLORS.muted, 1.2);
    b += line(sc.x, sc.y, sc.x, base, COLORS.line, 1);
    opts.ticks.forEach((t) => {
      b += line(sc.sx(t), base, sc.sx(t), base + 4, COLORS.muted, 1);
      b += text(sc.sx(t), base + 6 + opts.size, t, { size: opts.size, anchor: "middle", fill: COLORS.muted });
    });
    opts.yTicks.forEach((v) => {
      b += line(sc.x - 4, sc.sy(v), sc.x, sc.sy(v), COLORS.muted, 1);
      b += text(sc.x - 7, sc.sy(v) + 4, v, { size: opts.size, anchor: "end", fill: COLORS.muted });
    });
    if (opts.label) b += text(sc.x + sc.width + 6, base + 4, opts.label, { size: opts.size + 1, fill: COLORS.muted, weight: "bold" });
    return `${b}</g>`;
  }

  // One <path> per contiguous run of points { t, y }, so gaps are never bridged.
  function seriesPath(pts, sc, attrs) {
    const a = Object.assign({ color: COLORS.ink, width: 1.6, opacity: 1, role: "series" }, attrs);
    const byT = new Map(pts.map((p) => [p.t, p]));
    return contiguousRuns(pts.map((p) => p.t))
      .filter((run) => run.length > 1)
      .map(
        (run) =>
          `<path data-role="${a.role}" d="${run.map((t, i) => `${i ? "L" : "M"} ${sc.sx(t)} ${sc.sy(byT.get(t).y)}`).join(" ")}"` +
          ` fill="none" stroke="${a.color}" stroke-width="${a.width}" stroke-opacity="${a.opacity}"/>`
      )
      .join("");
  }

  // Points as circles carrying data-plot/data-set/data-t/data-y; tests read these back.
  function dots(pts, sc, attrs) {
    const a = Object.assign({ r: 2.6, fill: COLORS.ink, plot: "main", set: "observed", stroke: COLORS.paper }, attrs);
    return pts
      .map(
        (p) =>
          `<circle data-plot="${a.plot}" data-set="${typeof a.set === "function" ? a.set(p) : a.set}"` +
          ` data-t="${p.t}" data-y="${p.y}" cx="${sc.sx(p.t)}" cy="${sc.sy(p.y)}" r="${a.r}"` +
          ` fill="${typeof a.fill === "function" ? a.fill(p) : a.fill}" stroke="${a.stroke}" stroke-width="0.8"/>`
      )
      .join("");
  }

  // Shaded time interval [t0, t1] (inclusive, half a step of margin on each side).
  function band(sc, t0, t1, color, attrs) {
    const half = sc.width / (sc.tMax - sc.tMin) / 2;
    const x0 = Math.max(sc.x, sc.sx(t0) - half);
    const x1 = Math.min(sc.x + sc.width, sc.sx(t1) + half);
    return (
      `<rect ${attrs || ""} x="${r1(x0)}" y="${sc.y}" width="${r1(x1 - x0)}" height="${sc.height}"` +
      ` fill="${color}" fill-opacity="0.12"/>`
    );
  }

  function diamond(cx, cy, size, color, attrs, hollow) {
    const s = size;
    return (
      `<path ${attrs || ""} d="M ${cx} ${r1(cy - s)} L ${r1(cx + s)} ${cy} L ${cx} ${r1(cy + s)} L ${r1(cx - s)} ${cy} Z"` +
      ` fill="${hollow ? COLORS.paper : color}" stroke="${color}" stroke-width="1.6"/>`
    );
  }

  const centered = (lines, y, attrs) =>
    lines.map((s, i) => text(300, y + i * 18, s, Object.assign({ anchor: "middle", size: 12, fill: COLORS.muted }, attrs))).join("");

  // ---------- Lesson 1: from an image grid to an ordered time axis ----------

  const SIGNAL_INTRO = 2;
  const SIGNAL_STEPS = 20;
  const SIGNAL_STEP = SIGNAL_LENGTH / SIGNAL_STEPS;

  function clock(cx, cy, turn) {
    const angle = 2 * Math.PI * turn - Math.PI / 2;
    return (
      `<circle cx="${cx}" cy="${cy}" r="13" fill="${COLORS.paper}" stroke="${COLORS.muted}" stroke-width="1.5"/>` +
      line(cx, cy, r1(cx + 10 * Math.cos(angle)), r1(cy + 10 * Math.sin(angle)), COLORS.rose, 2)
    );
  }

  function houseGrid(x, y, size) {
    let b = `<g data-role="image-grid">`;
    HOUSE.forEach((row, r) =>
      row.forEach((v, c) => {
        b +=
          `<rect x="${x + c * size}" y="${y + r * size}" width="${size}" height="${size}"` +
          ` fill="${v ? COLORS.ink : COLORS.paper}" stroke="${COLORS.line}" stroke-width="1"/>`;
      })
    );
    return `${b}</g>`;
  }

  function renderSignal(frame) {
    const sc = timeScale({ x: 56, y: 82, width: 504, height: 220 });
    const yTicks = [Y_DOMAIN[0], Math.round((Y_DOMAIN[0] + Y_DOMAIN[1]) / 2), Y_DOMAIN[1]];
    let b = text(300, 30, "De la imagen a una señal en el tiempo", { size: 17, weight: "bold", anchor: "middle" });

    if (frame === 0) {
      b += houseGrid(70, 110, 34);
      ["Imagen: una grilla 2D.", "Cada píxel tiene vecinos", "arriba, abajo, izquierda", "y derecha.", "", "La vecindad es espacial."].forEach(
        (s, i) => (b += text(300, 140 + i * 24, s, { size: 15, fill: i === 5 ? COLORS.cyan : COLORS.ink, weight: i === 5 ? "bold" : "normal" }))
      );
      return {
        svg: svg("Una imagen es una grilla con vecindad espacial", b),
        label: "Imagen: una grilla 2D donde cada píxel tiene vecinos en el espacio.",
      };
    }
    if (frame === 1) {
      b += houseGrid(20, 46, 9);
      b += arrow(70, 70, 120, 70, COLORS.muted, 1.6);
      b += text(128, 75, "ahora: un valor por instante, en orden", { size: 13, fill: COLORS.muted });
      b += axes(sc, { yTicks });
      b += arrow(sc.x, sc.y + sc.height + 34, sc.x + sc.width, sc.y + sc.height + 34, COLORS.rose, 2);
      b += text(300, sc.y + sc.height + 54, "tiempo →  el vecino que importa es el pasado", { size: 13, anchor: "middle", fill: COLORS.rose });
      b += centered(["Serie de tiempo: cada valor tiene su instante t.", "El orden es parte del dato: no se puede barajar."], 400, { size: 13, fill: COLORS.ink });
      return {
        svg: svg("Una serie de tiempo ordena los valores sobre el eje del tiempo", b),
        label: "Serie de tiempo: cada valor tiene su instante t; el orden es parte del dato.",
      };
    }

    const final = frame >= SIGNAL_INTRO + SIGNAL_STEPS;
    const k = final ? SIGNAL_STEPS - 1 : frame - SIGNAL_INTRO;
    const end = final ? SIGNAL_LENGTH - 1 : SIGNAL_STEP * k + SIGNAL_STEP - 1;
    const shown = SIGNAL.slice(0, end + 1);
    b += clock(560, 50, (k + 1) / SIGNAL_STEPS);
    b += axes(sc, { yTicks });
    b += seriesPath(shown, sc, { color: COLORS.cyan, width: 1.6, opacity: 0.8 });
    b += dots(shown, sc, { plot: "signal", fill: COLORS.cyan, r: 2.4 });

    if (!final) {
      const p = SIGNAL[end];
      const cx = sc.sx(end);
      b += line(cx, sc.y, cx, sc.y + sc.height, COLORS.rose, 1.4, `data-cursor="${end}" stroke-dasharray="4 3"`);
      b += `<circle cx="${cx}" cy="${sc.sy(p.y)}" r="5" fill="none" stroke="${COLORS.rose}" stroke-width="2"/>`;
      const bx = Math.min(Math.max(cx - 70, 60), 396); // stays clear of the clock at x = 547..573
      b += `<rect x="${bx}" y="50" width="140" height="24" rx="6" fill="${COLORS.paper}" stroke="${COLORS.rose}"/>`;
      b += text(bx + 70, 67, `t = ${end} · y = ${fmt(p.y)}`, { size: 13, anchor: "middle", weight: "bold", fill: COLORS.rose });
      b += centered(["La historia importa: y(t) se parece a y(t−1), y(t−2)…", "Por eso el orden de los puntos no es intercambiable."], 360, { size: 13, fill: COLORS.ink });
    } else {
      b += line(sc.sx(0), sc.sy(trendAt(0)), sc.sx(99), sc.sy(trendAt(99)), COLORS.green, 2, `data-role="trend" stroke-dasharray="7 5"`);
      b += text(sc.sx(99) - 4, sc.sy(trendAt(99)) + 34, "tendencia", { size: 12, anchor: "end", fill: COLORS.green, weight: "bold" });
      // Bracket over one full cycle, from peak (t = 45) to peak (t = 65).
      const [x0, x1, yb] = [sc.sx(45), sc.sx(65), sc.y - 2];
      b += `<path data-role="period" d="M ${x0} ${yb + 8} L ${x0} ${yb} L ${x1} ${yb} L ${x1} ${yb + 8}" fill="none" stroke="${COLORS.rose}" stroke-width="1.8"/>`;
      b += text((x0 + x1) / 2, yb - 6, "un ciclo ≈ 20 pasos", { size: 12, anchor: "middle", fill: COLORS.rose, weight: "bold" });
      b += centered(["Tendencia creciente + ciclo que se repite + ruido.", "Señal sintética de ejemplo (semilla fija), no datos reales."], 360, { size: 13, fill: COLORS.ink });
    }
    b += centered(["Los valores salen de y(t) = tendencia + ciclo + ruido con semilla: siempre los mismos."], 420, { size: 11.5 });

    const label = final
      ? "Las 100 observaciones: tendencia creciente, ciclo de unos 20 pasos y ruido. Señal sintética."
      : `Reloj en t = ${end}: y(${end}) = ${fmt(SIGNAL[end].y)}. Van ${end + 1}/100 observaciones, en orden.`;
    return { svg: svg("Los puntos de la señal aparecen a medida que avanza el tiempo", b), label };
  }

  // ---------- Lesson 2: autoregressive forecast from three earlier lags ----------

  const AR_TARGETS = Object.freeze([30, 31, 32, 33, 34, 35, 36, 37]);
  const AR_WINDOW = Object.freeze([20, 40]);
  const LAG_ARC = [-34, -20, -8];

  function renderAutoregression(frame) {
    const values = signalValues();
    const i = Math.floor(frame / 3);
    const stage = ["origin", "forecast", "reveal"][frame % 3];
    const target = AR_TARGETS[i];
    const f = arForecast(values, target);
    const sc = timeScale({ x: 50, y: 60, width: 510, height: 170, tMin: AR_WINDOW[0], tMax: AR_WINDOW[1] });
    const known = SIGNAL.slice(AR_WINDOW[0], target);
    const origin = r1((sc.sx(target - 1) + sc.sx(target)) / 2);
    const tx = sc.sx(target);
    const ty = stage === "origin" ? sc.y + sc.height / 2 : sc.sy(f.prediction);
    let b = text(300, 26, "Autorregresión: predecir y(t) con su propio pasado", { size: 17, weight: "bold", anchor: "middle" });

    // Future region: nothing after the origin is drawn as data until the reveal.
    b += `<rect x="${origin}" y="${sc.y}" width="${r1(sc.x + sc.width - origin)}" height="${sc.height}" fill="${COLORS.empty}" fill-opacity="0.7"/>`;
    b += text(r1((origin + sc.x + sc.width) / 2) + 8, sc.y + sc.height - 10, "futuro: aún no observado", { size: 11, anchor: "middle", fill: COLORS.muted });
    b += axes(sc, { ticks: [20, 25, 30, 35, 40], yTicks: [] });
    b += line(origin, sc.y - 6, origin, sc.y + sc.height, COLORS.rose, 1.6, `stroke-dasharray="5 4"`);
    b += text(origin, sc.y - 10, "origen del pronóstico", { size: 11.5, anchor: "middle", fill: COLORS.rose, weight: "bold" });

    b += seriesPath(known, sc, { color: COLORS.ink, width: 1.4, opacity: 0.6 });
    b += dots(known, sc, { fill: COLORS.ink, r: 3 });
    AR_TARGETS.slice(0, i).forEach((t) => (b += diamond(sc.sx(t), sc.sy(arForecast(values, t).prediction), 4, COLORS.green, `data-role="past-prediction"`, true)));

    // Lags (inputs) → arrows with their weights → the target point.
    f.lags.forEach((lag, k) => {
      const lx = sc.sx(lag.t);
      const ly = sc.sy(lag.y);
      b += `<path d="M ${lx} ${ly} Q ${r1((lx + tx) / 2)} ${r1(Math.min(ly, ty) + LAG_ARC[k])} ${tx} ${ty}" fill="none" stroke="${COLORS.cyan}" stroke-width="${2.6 - k * 0.6}" stroke-opacity="0.75"/>`;
      b += text(lx, ly - 10, lag.weight, { size: 11.5, anchor: "middle", fill: COLORS.cyan, weight: "bold" });
    });
    b += dots(f.lags, sc, { set: "lag", fill: COLORS.cyan, r: 5.5, stroke: COLORS.paper });
    if (stage === "origin") {
      b += `<circle cx="${tx}" cy="${ty}" r="10" fill="${COLORS.paper}" stroke="${COLORS.green}" stroke-width="2" stroke-dasharray="3 2"/>`;
      b += text(tx, ty + 5, "?", { size: 14, anchor: "middle", weight: "bold", fill: COLORS.green });
    } else {
      b += diamond(tx, ty, 7, COLORS.green, `data-prediction="${fmt(f.prediction)}"`);
    }
    if (stage === "reveal") {
      const oy = sc.sy(f.observed);
      b += line(tx + 12, ty, tx + 12, oy, COLORS.rose, 1.6, `stroke-dasharray="3 2"`);
      b += dots([{ t: target, y: f.observed }], sc, { set: "target-observed", fill: COLORS.ink, r: 5 });
    }

    // Chronological lag row: three inputs, one target.
    const cols = [90, 190, 290, 430];
    b += text(40, 266, `Fila cronológica para t = ${target}`, { size: 13, weight: "bold" });
    b += text(190, 288, "rezagos = entradas", { size: 11.5, anchor: "middle", fill: COLORS.cyan, weight: "bold" });
    b += text(430, 288, "objetivo", { size: 11.5, anchor: "middle", fill: COLORS.green, weight: "bold" });
    f.lags.forEach((lag, k) => {
      b += `<rect x="${cols[k] - 44}" y="296" width="88" height="44" rx="6" fill="${COLORS.cyan}" fill-opacity="0.12" stroke="${COLORS.cyan}"/>`;
      b += text(cols[k], 313, `y(${lag.t})`, { size: 11.5, anchor: "middle", fill: COLORS.muted });
      b += text(cols[k], 332, fmt(lag.y), { size: 14, anchor: "middle", weight: "bold" });
    });
    b += line(350, 296, 350, 340, COLORS.line, 2);
    b += `<rect x="386" y="296" width="88" height="44" rx="6" fill="${COLORS.paper}" stroke="${COLORS.green}"/>`;
    b += text(430, 313, `y(${target})`, { size: 11.5, anchor: "middle", fill: COLORS.muted });
    b += text(430, 332, stage === "reveal" ? fmt(f.observed) : "?", { size: 14, anchor: "middle", weight: "bold" });
    if (stage === "reveal") {
      b += text(490, 313, "error y − ŷ", { size: 11.5, fill: COLORS.rose });
      b += text(490, 332, fmt(f.error), { size: 14, weight: "bold", fill: COLORS.rose });
    }

    const terms = f.lags.map((l) => `${l.weight}·${stage === "origin" ? `y(${l.t})` : fmt(l.y)}`).join(" + ");
    b += text(300, 368, `ŷ(${target}) = ${terms}${stage === "origin" ? "" : ` = ${fmt(f.prediction)}`}`, {
      size: 14,
      anchor: "middle",
      weight: "bold",
      fill: stage === "origin" ? COLORS.ink : COLORS.green,
    });
    b += centered(
      ["Solo se usan valores anteriores a t; y(t) se revela después del pronóstico.", "Coeficientes ilustrativos, no entrenados: no es un ajuste ni una medida de precisión."],
      400
    );

    const label =
      stage === "origin"
        ? `Origen en t = ${target}: se conocen los valores hasta y(${target - 1}); y(${target}) todavía no se observa.`
        : stage === "forecast"
        ? `Pronóstico: ŷ(${target}) = ${terms} = ${fmt(f.prediction)}.`
        : `Se revela y(${target}) = ${fmt(f.observed)}; error = ${fmt(f.error)}. El origen avanza un paso.`;
    return { svg: svg("Autorregresión de orden 3: tres valores pasados pronostican el siguiente", b), label };
  }

  // ---------- Lesson 3: interactive train/test split on the original time axis ----------

  const SPLIT_STEPS = 10;
  const SPLIT_FRAMES = SPLIT_STEPS + 2;

  function renderSplit(frame, config) {
    const cfg = Object.assign({}, DEFAULT_SPLIT, config);
    const part = partition(cfg);
    const testSet = new Set(part.test);
    const assigned = frame >= SPLIT_STEPS + 1 ? SIGNAL_LENGTH : Math.min(frame, SPLIT_STEPS) * (SIGNAL_LENGTH / SPLIT_STEPS);
    const roleOf = (p) => (p.t >= assigned ? "pending" : testSet.has(p.t) ? "test" : "train");
    const pct = Math.round(cfg.testFraction * 100);
    const random = part.mode === "random";
    const main = timeScale({ x: 50, y: 74, width: 520, height: 146 });
    const small = [
      { set: "train", title: "Solo train", x: 50, idx: part.train },
      { set: "test", title: "Solo test", x: 330, idx: part.test },
    ];
    let b = text(300, 26, "División train/test sobre el eje temporal original", { size: 16.5, weight: "bold", anchor: "middle" });
    b += text(300, 46, random ? `Modo aleatorio (IID) · test ${pct} % · semilla ${cfg.seed}` : `Modo cronológico · test ${pct} % = último bloque`, {
      size: 12.5,
      anchor: "middle",
      fill: COLORS.muted,
    });

    if (!random) {
      const from = part.test[0];
      b += band(main, from, SIGNAL_LENGTH - 1, ROLE_COLORS.test, `data-role="test-band" data-from="${from}" data-to="${SIGNAL_LENGTH - 1}"`);
      b += text(r1((main.sx(from) + main.sx(99)) / 2), main.y + 14, `test: t = ${from}…99`, { size: 11.5, anchor: "middle", fill: ROLE_COLORS.test, weight: "bold" });
    }
    b += axes(main, { yTicks: [Y_DOMAIN[0], Y_DOMAIN[1]] });
    b += seriesPath(SIGNAL, main, { color: ROLE_COLORS.reference, width: 1.2, opacity: 0.35, role: "reference" });
    b += dots(SIGNAL, main, { set: roleOf, fill: (p) => ROLE_COLORS[roleOf(p)], r: 3 });
    b += text(main.x + main.width, main.y - 6, "curva original de referencia (gris)", { size: 10.5, anchor: "end", fill: COLORS.muted });

    b += `<g data-count="train" data-value="${part.train.length}">${text(170, 254, `train: ${part.train.length} puntos`, { size: 13, anchor: "middle", weight: "bold", fill: ROLE_COLORS.train })}</g>`;
    b += `<g data-count="test" data-value="${part.test.length}">${text(430, 254, `test: ${part.test.length} puntos`, { size: 13, anchor: "middle", weight: "bold", fill: ROLE_COLORS.test })}</g>`;

    // Each subset alone, as dots on the same time axis: no line joins points across gaps.
    small.forEach((s) => {
      const sc = timeScale({ x: s.x, y: 282, width: 220, height: 78 });
      const pts = s.idx.filter((t) => t < assigned).map((t) => SIGNAL[t]);
      b += text(s.x, 276, `${s.title} (${s.idx.length})`, { size: 12, weight: "bold", fill: ROLE_COLORS[s.set] });
      b += axes(sc, { ticks: [0, 50, 99], size: 10, label: "" });
      b += dots(pts, sc, { plot: `${s.set}-only`, set: s.set, fill: ROLE_COLORS[s.set], r: 2.4 });
    });

    const notes = random
      ? ["Aleatoria: train y test se reparten por todo el eje y suelen seguir la misma forma.", "Es una línea base IID: no prueba que el modelo pronostique el futuro."]
      : ["Cronológica: se entrena con el pasado y se evalúa con el último bloque.", "El test queda siempre después de todo el train, como al pronosticar."];
    b += centered(notes.concat(["Partición retrospectiva de datos ya observados: el test no se usa para ajustar."]), 396, { size: 11.5 });

    const counts = `train ${part.train.length} · test ${part.test.length} (${pct} %)`;
    const label =
      assigned === 0
        ? `${random ? "Aleatoria (IID)" : "Cronológica"}: 100 observaciones en su orden, aún sin asignar · ${counts}.`
        : assigned < SIGNAL_LENGTH
        ? `Asignadas t = 0…${assigned - 1}: ${part.train.filter((t) => t < assigned).length} train, ${part.test.filter((t) => t < assigned).length} test.`
        : `${counts} · ${random ? "ambos cubren todo el eje (referencia IID, no prueba de pronóstico)" : `test = último bloque t = ${part.test[0]}…99`}.`;
    return { svg: svg("División train/test de la señal sobre su eje de tiempo", b), label };
  }

  const LESSONS = {
    signal: { frameCount: SIGNAL_INTRO + SIGNAL_STEPS + 1, intervalMs: 700, title: "Una señal que depende del tiempo", render: renderSignal },
    autoregression: { frameCount: AR_TARGETS.length * 3, intervalMs: 1500, title: "Autorregresión con tres rezagos", render: renderAutoregression },
    split: { frameCount: SPLIT_FRAMES, intervalMs: 900, title: "División train/test interactiva", render: (frame) => renderSplit(frame, DEFAULT_SPLIT) },
  };
  // Registering in the shared registry reuses SpatialLesson's player, controls and keyboard handling.
  Object.keys(LESSONS).forEach((kind) => (S.LESSONS[`temporal-${kind}`] = LESSONS[kind]));

  // ---------- DOM mounting ----------

  const SLIDER_KEYS = new Set([" ", "Spacebar", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);
  const instances = new WeakMap();
  let splitCounter = 0;

  // The split widget needs per-instance state, so it mounts a private copy of the lesson.
  function mountSplit(el) {
    let config = Object.assign({}, DEFAULT_SPLIT);
    const key = `temporal-split-instance-${(splitCounter += 1)}`;
    S.LESSONS[key] = Object.assign({}, LESSONS.split, { render: (frame) => renderSplit(frame, config) });
    let instance;
    try {
      instance = S.mount(el, key);
    } finally {
      delete S.LESSONS[key]; // S.mount keeps its own reference; the registry stays clean.
    }
    const doc = el.ownerDocument;
    const make = (tag, cls, txt) => {
      const node = doc.createElement(tag);
      if (cls) node.classList.add(cls);
      if (txt) node.textContent = txt;
      return node;
    };
    const options = make("div", "temporal-options");
    const modeLabel = make("span", "temporal-option-label", "División:");
    const modeButtons = { random: make("button", "spatial-button", "Aleatoria (IID)"), chronological: make("button", "spatial-button", "Cronológica") };
    const fractionLabel = make("label", "temporal-option-label", "Test:");
    const fractionInput = make("input", "temporal-range");
    const fractionValue = make("span", "temporal-option-value");
    const seedButton = make("button", "spatial-button");
    [["type", "range"], ["min", "10"], ["max", "40"], ["step", "10"], ["aria-label", "Porcentaje de test"]].forEach(([k, v]) =>
      fractionInput.setAttribute(k, v)
    );
    [modeButtons.random, modeButtons.chronological, seedButton].forEach((btn) => btn.setAttribute("type", "button"));
    fractionValue.setAttribute("aria-live", "polite");
    [modeLabel, modeButtons.random, modeButtons.chronological, fractionLabel, fractionInput, fractionValue, seedButton].forEach((n) =>
      options.appendChild(n)
    );
    el.appendChild(options);

    const syncControls = () => {
      const pct = Math.round(config.testFraction * 100);
      Object.keys(modeButtons).forEach((mode) => modeButtons[mode].setAttribute("aria-pressed", String(config.mode === mode)));
      fractionInput.value = String(pct);
      fractionValue.textContent = `${pct} %`;
      seedButton.textContent = `Nueva semilla (${config.seed})`;
      seedButton.disabled = config.mode !== "random";
    };
    // Every change goes through partition(), so an invalid value is rejected before it is drawn.
    const setConfig = (patch) => {
      const next = Object.assign({}, config, patch);
      partition(next);
      config = next;
      syncControls();
      instance.player.last(); // show the full new assignment; pauses like any manual step
    };
    const clickWith = (button, action) =>
      button.addEventListener("click", (event) => {
        action();
        if (event && event.detail > 0 && button.blur) button.blur();
      });
    clickWith(modeButtons.random, () => setConfig({ mode: "random" }));
    clickWith(modeButtons.chronological, () => setConfig({ mode: "chronological" }));
    clickWith(seedButton, () => setConfig({ seed: config.seed + 1 }));
    fractionInput.addEventListener("input", () => setConfig({ testFraction: Number(fractionInput.value) / 100 }));
    // The slider keeps its own arrow keys: they must not step the animation or move the deck.
    fractionInput.addEventListener("keydown", (event) => {
      if (SLIDER_KEYS.has(event.key)) event.stopPropagation();
    });
    syncControls();

    return Object.assign(instance, {
      options,
      modeButtons,
      fractionInput,
      fractionValue,
      seedButton,
      getConfig: () => Object.assign({}, config),
      getSplit: () => partition(config),
      setConfig,
    });
  }

  function mount(el, kind) {
    if (instances.has(el)) return instances.get(el);
    const key = kind || el.getAttribute("data-temporal-lesson");
    if (!Object.prototype.hasOwnProperty.call(LESSONS, key)) throw new Error(`Unknown temporal lesson: ${key}`);
    const instance = key === "split" ? mountSplit(el) : S.mount(el, `temporal-${key}`);
    instance.kind = key;
    instance.root.classList.add("temporal-lesson");
    instances.set(el, instance);
    return instance;
  }

  function autoMount(doc) {
    return Array.from(doc.querySelectorAll("[data-temporal-lesson]"))
      .filter((el) => Object.prototype.hasOwnProperty.call(LESSONS, el.getAttribute("data-temporal-lesson")))
      .map((el) => mount(el));
  }

  return {
    SIGNAL,
    SIGNAL_SEED,
    SIGNAL_LENGTH,
    PERIOD,
    NOISE_AMPLITUDE,
    Y_DOMAIN,
    AR_COEFFS,
    AR_TARGETS,
    SPLIT_SEED,
    DEFAULT_SPLIT,
    ROLE_COLORS,
    LESSONS,
    trendAt,
    seasonalAt,
    round2,
    mulberry32,
    seededShuffle,
    generateSignal,
    signalValues,
    partition,
    arForecast,
    lagTable,
    contiguousRuns,
    plot: { esc, text, line, arrow, svg, timeScale, axes, seriesPath, dots, band, diamond, centered },
    renderSignal,
    renderAutoregression,
    renderSplit,
    mount,
    autoMount,
  };
});
